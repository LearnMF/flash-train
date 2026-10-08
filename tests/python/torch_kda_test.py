# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash_train.torch.kimi_delta_attention against its PyTorch FP64 reference.

Tolerances (relative RMS error vs the fp64 reference, the cuDNN-frontend PR #556 criteria):
o 2e-2; dq, dk, dv 4e-2; dg, dbeta, dA_log, ddt_bias 6e-2. The G2 kernels compute in bf16
operands with fp32 accumulation; observed errors are ~0.3-0.7%.
"""

import os

import pytest
import torch

import flash_train.torch as ft


def _gfx936():
    if not torch.cuda.is_available():
        return False
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith("gfx936")


pytestmark = pytest.mark.skipif(not _gfx936(), reason="the G2 KDA kernels target gfx936")

TOL = {"o": 2e-2, "q": 4e-2, "k": 4e-2, "v": 4e-2, "g": 6e-2, "beta": 6e-2, "A_log": 6e-2, "dt_bias": 6e-2}


def rel_rms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-6)).item()


def make_inputs(T, H, gate_in_kernel, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    x = {"q": (r(T, H, 128) * 0.3).bfloat16(), "k": (r(T, H, 128) * 0.3).bfloat16(),
         "v": (r(T, H, 128) * 0.3).bfloat16(), "beta": r(T, H).sigmoid()}
    if gate_in_kernel:
        x["g"] = (r(T, H, 128) * 0.3).bfloat16()
        x["A_log"] = torch.log(1.0 + 15.0 * torch.rand(H, device="cuda", generator=gen))
        x["dt_bias"] = r(H * 128) * 0.1
    else:
        x["g"] = -torch.rand(T, H, 128, device="cuda", generator=gen) * 0.5
    return x


def make_d_o(T, H, seed):
    # seeded: the default CUDA generator's seed differs per process, which made the
    # tolerance checks (e.g. the 2-element dA_log of H = 2) vary from run to run
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(T, H, 128, device="cuda", generator=gen).bfloat16()


def run(fn, x, d_o, **kw):
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
    o = fn(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], A_log=leaves.get("A_log"),
           dt_bias=leaves.get("dt_bias"), **kw)
    grads = torch.autograd.grad(o, list(leaves.values()), d_o.to(o.dtype))
    return o.detach(), dict(zip(leaves, grads))


CASES = [(64, 2, None), (100, 2, None), (257, 4, None), (512, 4, None),
         (333, 2, [0, 1, 64, 65, 200, 333]), (200, 3, [0, 0, 63, 127, 127, 200])]


@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_matches_fp64_reference(T, H, cu, gate_in_kernel):
    x = make_inputs(T, H, gate_in_kernel, seed=T + H)
    d_o = make_d_o(T, H, seed=T + H + 1)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    o, g = run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t)
    ro, rg = run(ft.kda_reference, x, d_o, cu_seqlens=cu_t)
    assert torch.isfinite(o.float()).all()
    assert rel_rms(o, ro) < TOL["o"]
    for name in x:
        assert torch.isfinite(g[name].float()).all(), name
        assert rel_rms(g[name], rg[name]) < TOL[name], (name, rel_rms(g[name], rg[name]))


def test_deterministic():
    x = make_inputs(512, 4, True, seed=3)
    d_o = make_d_o(512, 4, seed=4)
    a = run(ft.kimi_delta_attention, x, d_o)
    b = run(ft.kimi_delta_attention, x, d_o)
    assert torch.equal(a[0], b[0])
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def test_packed_equals_separate():
    """Sequences in one packed call are independent: bit-identical to separate varlen calls,
    and to the dense single-sequence path (no cu_seqlens, T % 64 == 0)."""
    lengths = [128, 64, 192]
    T, H = sum(lengths), 2
    x = make_inputs(T, H, True, seed=11)
    d_o = make_d_o(T, H, seed=12)
    cu = torch.tensor([0, 128, 192, 384], dtype=torch.int32, device="cuda")
    o, g = run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu)
    s = 0
    for n in lengths:
        part = {k: (t[s:s + n] if k not in ("A_log", "dt_bias") else t) for k, t in x.items()}
        one = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        op, gp = run(ft.kimi_delta_attention, part, d_o[s:s + n], cu_seqlens=one)
        assert torch.equal(op, o[s:s + n])
        for name in ("q", "k", "v", "g", "beta"):
            assert torch.equal(gp[name], g[name][s:s + n]), name
        od, gd = run(ft.kimi_delta_attention, part, d_o[s:s + n])
        assert torch.equal(od, op)
        for name in ("q", "k", "v", "g", "beta"):
            assert torch.equal(gd[name], gp[name]), name
        s += n


def test_rejects_unsupported_head_dim():
    x = make_inputs(64, 2, True)
    x = {k: (t[..., :64].contiguous() if k in ("q", "k", "v", "g") else t) for k, t in x.items()}
    x["dt_bias"] = x["dt_bias"][: 2 * 64]
    with pytest.raises(RuntimeError):
        ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"], x["beta"], A_log=x["A_log"], dt_bias=x["dt_bias"])


def test_rejects_decay_below_bound():
    x = make_inputs(64, 2, False)
    with pytest.raises(ValueError):
        ft.kimi_delta_attention(x["q"], x["k"], x["v"], x["g"] - 10.0, x["beta"])


def _with_keep(mode, fn):
    old = os.environ.get("FLASH_TRAIN_KDA_KEEP_STATE")
    os.environ["FLASH_TRAIN_KDA_KEEP_STATE"] = mode
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("FLASH_TRAIN_KDA_KEEP_STATE")
        else:
            os.environ["FLASH_TRAIN_KDA_KEEP_STATE"] = old


@pytest.mark.parametrize("gate_in_kernel", [True, False])
@pytest.mark.parametrize("T,H,cu", CASES)
def test_keep_state_is_bit_identical(T, H, cu, gate_in_kernel):
    """The forward handing its recurrent state to the backward (instead of the backward
    recomputing it) changes nothing: o and every gradient bit for bit."""
    x = make_inputs(T, H, gate_in_kernel, seed=2 * T + H)
    d_o = make_d_o(T, H, seed=2 * T + H + 1)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda") if cu is not None else None
    a = _with_keep("0", lambda: run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t))
    b = _with_keep("1", lambda: run(ft.kimi_delta_attention, x, d_o, cu_seqlens=cu_t))
    assert torch.equal(a[0], b[0])
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def _recurrence_launches(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(e.count for e in p.key_averages() if "g2_fwd_h_v8o" in e.key)


def test_checkpoint_recompute_keeps_state():
    """Under a reentrant activation checkpoint (Megatron's recompute form) the recompute forward
    keeps the state (auto): the backward runs no recurrence of its own -- 2 recurrence launches
    (no-grad forward + recompute) instead of 3 -- and the gradients equal a plain call bit for bit."""
    import torch.utils.checkpoint as tuc
    T, H = 512, 4
    x = make_inputs(T, H, True, seed=5)
    d_o = make_d_o(T, H, seed=6)
    plain = run(ft.kimi_delta_attention, x, d_o)

    def ckpt(mode):
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
        order = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
        f = lambda q, k, v, g, beta, a, dt: ft.kimi_delta_attention(q, k, v, g, beta, A_log=a, dt_bias=dt)

        def step():
            o = tuc.checkpoint(f, *(leaves[n] for n in order), use_reentrant=True)
            o.backward(d_o)
            return o
        launches = _with_keep(mode, lambda: _recurrence_launches(step))
        return launches, leaves

    for mode, want in (("auto", 2), ("0", 3)):
        launches, leaves = ckpt(mode)
        assert launches == want, (mode, launches)
        for name in x:
            assert torch.equal(leaves[name].grad, plain[1][name]), (mode, name)
