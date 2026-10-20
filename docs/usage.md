# flash-train 使用指南

本库面向三类使用者，按接入成本从低到高排列。三类入口最终执行同一套引擎选择与内核调度，只是封装层次不同。

| 你是谁 | 用哪套接口 | 一次调用的成本 | 适合场景 |
|---|---|---|---|
| PyTorch 用户 | `flash_train.torch`（wheel） | 每次调用含参数组装与计划缓存查询 | 快速接入、原型验证 |
| C/C++ 调用方（图省事） | `flash_train.h` 便利 API | 同上 | 偶发调用、不建 Plan 的场景 |
| C/C++ 调用方（追求性能） | `common.h` 分阶段 API | Plan 复用后仅剩执行 | 训练热循环 |

---

## 1. PyTorch 用户

### 安装

在 HCU 环境（DTK + PyTorch）中从源码构建 wheel 并安装：

```bash
pip wheel . --no-deps -w dist
pip install --no-deps dist/flash_train-*.whl
```

### 计算 GEMM

```python
import torch, flash_train.torch

a = torch.randn(128, 64,  device="cuda")   # m×k
b = torch.randn(64, 96,   device="cuda")   # k×n
c = torch.randn(128, 96,  device="cuda")   # m×n，可选

d = flash_train.torch.gemm(a, b, c, alpha=1.5, beta=0.25)   # d = α·(a@b) + β·c
d = flash_train.torch.gemm(a, b)                           # c=None 时 β·c 项被跳过
```

要点：

- 参数约束：`a/b/c` 为 **FP32、二维、连续、设备**张量，`a.size(1) == b.size(0)`，`c` 形状为 `m×n`；不满足抛 `RuntimeError`。
- 输出 `d` 由库分配并返回；计算提交在 **torch 当前流** 上，与相邻算子自然保序。
- `alpha`/`beta` 是 Python 标量，按 FP32 传入内核。
- 热循环请改用分阶段 API 复用 Plan（见第 3 节）；本接口每次调用有微秒级的参数组装开销。

### 计算 KDA（Kimi Delta Attention）

```python
import torch, flash_train.torch as ft

T, H = 8192, 12                                         # 打包后的总 token 数、head 数
q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
g = torch.randn_like(q)                                 # Kimi K3：原始 gate
beta = torch.rand(T, H, device="cuda")                  # 已过 sigmoid 的 beta
A_log = torch.zeros(H, device="cuda"); dt_bias = torch.zeros(H * 128, device="cuda")
cu = torch.tensor([0, 3000, 8192], dtype=torch.int32, device="cuda")   # 两条序列

o = ft.kimi_delta_attention(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu)
o.sum().backward()                                      # 支持 autograd：q/k/v/g/beta/A_log/dt_bias 的梯度
```

要点：

- 布局为打包 THD：`q/k/v/o` 是 `[T, H, 128]` 的 BF16；`cu_seqlens`（int32 `[N+1]`）把 T 行切成互不相关的序列，每条序列从零状态开始，长度任意（不必是 64 的倍数，可以为 0）。省略时视为一条序列。
- 两种 gate 契约：
  - 传 `A_log`/`dt_bias` 时，g 是原始 gate，算子内部计算 `lower_bound·sigmoid(exp(A_log)·(g+dt_bias))`，用于 Kimi K3，`lower_bound` 取 [-5, 0)；
  - 不传时，g 是 FP32 的逐 token log 衰减（cuDNN-frontend PR #556 契约），要求 ≥ -5；`check_decay=True`（默认）会检查这一点。
- `use_qk_l2norm_in_kernel=True`（默认）表示在算子内部对 q/k 做 l2 归一化，eps 为 1e-6。
- 当前实现为 gfx936 手写内核（`G2KdaForward`/`G2KdaBackward`），只支持 K = V = 128，其他配置抛 `RuntimeError`（`FTRAIN_STATUS_UNSUPPORTED`）。确定性：只用固定顺序归约，不用原子操作。
- 精度：与 FP64 参考（`ft.kda_reference`）相比，相对 RMS 误差约 0.3–0.7%，判据为 o < 2e-2、dq/dk/dv < 4e-2、门控侧梯度 < 6e-2（与 PR #556 一致）。
- 暂不支持初始状态和最终状态（训练从零状态开始，K3 不使用）。

#### 前向为反向保存什么（全部逐位一致，只影响显存和速度）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `FLASH_TRAIN_KDA_SAVE_QKN` | `1` | 前向保存 l2 归一化后的 qn/kn，代替 q/k（保存的字节不变），反向不再重建 |
| `FLASH_TRAIN_KDA_SAVE_H` | `auto` | 前向保存递推状态 h（每 token·head 512 B），反向不再跑递推。`auto` 在 H ≤ 48 时开启（gfx936 实测更快），`1` 总是开启，`0` 关闭 |
| `FLASH_TRAIN_KDA_KEEP_STATE` | `auto` | 前向交给反向完整状态（h 加 v_new，每 token·head 1024 B），只在 SAVE_H 未开启时生效。`auto` 只在可重入激活重算（Megatron 的 recompute、`torch.utils.checkpoint(use_reentrant=True)`）的重算前向里开启 |

### 在 Megatron 等训练框架中接入（与 FLA 的 `chunk_kda` 对齐）

`flash_train.torch.chunk_kda` 的签名与 FLA 的 `fla.ops.kda.chunk_kda` 相同，输入为 `[B, T, H, 128]`，返回 `(o, None)`。在训练框架里只需要换 import：

```python
# 原来：from fla.ops.kda import chunk_kda
from flash_train.torch import chunk_kda

# Megatron-Bridge Kimi K3（kimi_k3_ops.kda）现有的调用，参数不变：
o, _ = chunk_kda(q, k, v, forget_gate, beta,              # q/k/v/forget_gate: bf16 [B, T, H, 128]; beta: fp32 [B, T, H]
                 A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                 initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True,
                 transpose_state_layout=True, cu_seqlens=cu_seqlens)
```

- 只做布局转换，与在打包 THD 张量上直接调用 `kimi_delta_attention` 逐位一致：`[B, T, H, D]` 展平成 `[B*T, H, D]`；`B > 1` 且不传 `cu_seqlens` 时按 B 条长度为 T 的序列处理；传 `cu_seqlens` 时要求 `B == 1`（与 FLA 相同）。
- gate：`use_gate_in_kernel=True` 时要求给出 `A_log`、`dt_bias` 和 `lower_bound`（FLA 的 safe_gate 形式，即 K3 的用法）；`use_gate_in_kernel=False` 时 g 是逐 token log 衰减。`use_beta_sigmoid_in_kernel=True` 时在宿主侧对 beta 做 sigmoid（`allow_neg_eigval` 时乘 2）。
- `disable_recompute`、`chunk_size`、`cu_seqlens_cpu` 是 FLA 的调优参数，这里接受但不起作用；保存策略见上表。
- 不支持的参数直接抛异常，不会悄悄回退：`initial_state`、`output_final_state=True`、`return_intermediate_states`、`cp_context`、没有 `lower_bound` 的 softplus gate 抛 `NotImplementedError`；未知关键字抛 `TypeError`。

### 用 C API 接入 KDA（与 PyTorch 入口同一套端口）

`flash_train.h` 的 `ftrainKdaForward` / `ftrainKdaBackward`（以及对应的 `...GetWorkspaceBytes`）接收 `FTrainKdaForwardTensors` / `FTrainKdaBackwardTensors`；`common.h` 的 `ftrainPatternAddKdaForward` / `ftrainPatternAddKdaBackward` 用同样的端口顺序建 Pattern，用于复用 Plan。

| 端口 | 前向 | 反向 | 说明 |
|---|---|---|---|
| q, k, v, g, beta, a_log, dt_bias, cu_seqlens | 输入 | 输入 | 不用的端口传空内存（a_log/dt_bias 不在算子内算 gate 时；cu_seqlens 只有一条序列时） |
| o / d_o | 输出 | 输入 | `[T, H, 128]` bf16 |
| q_rstd, k_rstd, aqk, akk | 输出 | 输入 | 必须保存给反向 |
| qn, kn | 可选输出 | 可选输入 | 只在 l2norm_qk 时；给出后反向的 q/k 可以传空内存 |
| h | 可选输出 | 可选输入 | `ftrainKdaSavedHElements(T, H, N, has_cu)` 个 bf16，布局不公开 |
| state | 可选输出 | 可选输入 | `ftrainKdaStateElements(T, H, N, has_cu)` 个 bf16，布局不公开；与 h 互斥 |
| dq, dk, dv, dg, dbeta, da_log, ddt_bias | — | 输出 | dq/dk/dg 在调用期间兼作临时空间，不得与任何输入重叠 |

可选端口都是“前向写了、反向就用；传空内存就重算”，结果逐位一致。执行是异步的：返回后，所有张量和 workspace 必须保持有效，直到流上的计算完成。

---

## 2. 便利 C API（flash_train.h）

一个函数覆盖"拼参数→选实现→执行"全过程，等价于第 3 节的分阶段流程写一遍样板：

```c
#include <flash_train/flash_train.h>

FTrainStatus status = ftrainGemm(a, b, c, d, alpha, beta,
                                 FTRAIN_NUMERIC_TYPE_FP32,
                                 workspace, workspace_bytes, stream);
```

| 参数 | 要求 |
|---|---|
| `a`/`b`/`c`/`d` | 秩 2 设备 Tensor 视图（`FTrainStorageView`）；`c` 内存可为 `NULL`（跳过 `β·c`） |
| `alpha`/`beta` | 宿主机标量 Tensor（秩 0） |
| `compute_type` | 计算精度 |
| `workspace`/`workspace_bytes` | 推荐实现所需 workspace（当前 FP32 内核为 0，传 `NULL, 0` 即可） |
| `stream` | 执行流；函数不同步，返回后设备内存须保持有效直至该流完成 |

状态码语义与分阶段 API 完全一致：`FTRAIN_STATUS_INVALID_ARGUMENT`（参数不一致/workspace 不足）、`FTRAIN_STATUS_UNSUPPORTED`（无实现满足参数）、`FTRAIN_STATUS_SUCCESS`。失败详情可用 `ftrainGetLastStatus()` 查询。

---

## 3. 通用 C API（common.h）

分阶段 API 把**一次性的描述**与**每次调用的数据**分开，Plan 建好后在循环外复用：

```c
/* ---- 一次性：拓扑描述与匹配 ---- */
FTrainPattern pattern;  ftrainPatternCreate(&pattern);
FTrainTensorId a,b,c,d,alpha,beta;  FTrainGemmOpId gemm;
ftrainPatternAddTensor(pattern, &a);      /* ×6，顺序任意 */
/* ... */
ftrainPatternAddGemm(pattern, &gemm, a, b, c, d, alpha, beta);

FTrainOps ops;  ftrainOpsCreate(&ops, pattern);
ftrainPatternDestroy(pattern);            /* Ops 不依赖 Pattern 存活 */

/* ---- 每个形状一次：参数与计划 ---- */
FTrainArgs args;  ftrainArgsCreate(&args, ops);
ftrainArgsSetTensor(args, a, view_a);     /* ×6 */
ftrainArgsSetGemm(args, gemm, FTRAIN_NUMERIC_TYPE_FP32);

FTrainPlan plan;  ftrainPlanCreate(&plan, args);

/* ---- 每次迭代：按预算挑一个并执行 ---- */
uint64_t n, bytes;
ftrainPlanGetNumPrimitives(plan, &n);
for (uint64_t i = 0; i < n; ++i) {
    const char* name;  ftrainPlanGetPrimitiveName(plan, i, &name);  /* 甄别/日志用 */
    ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, i, &bytes);
    if (bytes <= my_budget) { ftrainPlanExecutePrimitive(plan, i, ws, my_budget, stream); break; }
}

ftrainPlanDestroy(plan);  ftrainArgsDestroy(args);  ftrainOpsDestroy(ops);
```

### 需要知道的规则

- **匹配**：`ftrainOpsCreate` 把用户 Pattern 与库内受支持拓扑做结构同构匹配，忽略加入顺序；不匹配返回 `FTRAIN_STATUS_UNSUPPORTED`。
- **Plan 内容**：创建成功即保证至少一个可用 Primitive；primitive 0 是推荐默认。库里会返回全部可用实现（best-first），你按 workspace 预算遍历挑选。
- **生命周期**：Plan/Primitive 在创建后与源 Ops/Args 解耦；Tensor 内存不托管，异步执行期间须保持有效。
- **状态查询**：任何调用失败后 `ftrainGetLastStatus()` 返回线程局部的最近状态。

### 环境变量

进程级开关集中声明在 `env.h`：

| 变量 | 作用 |
|---|---|
| `FTRAIN_ENABLED_PRIMITIVES` / `FTRAIN_DISABLED_PRIMITIVES` | 按名字允许/禁止实现 |
| `FTRAIN_DISABLED_FINDERS` | 按名字禁用选择启发式 |
| `FTRAIN_DISABLE_CACHE` | 关闭选择缓存（调试用） |
