# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GDN-2 on gfx936. Vector erase/write gates, no FLA runtime dependency."""
import math

import torch
from torch.autograd.function import once_differentiable

from . import _flash_train_torch as _C


class _Gdn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, scale, norm):
        o, cache = _C.gdn2_forward(q, k, v, g, b, w, scale, norm)
        ctx.save_for_backward(q, k, v, g, b, w, cache)
        ctx.scale, ctx.norm = scale, norm
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        grads = _C.gdn2_backward(*ctx.saved_tensors, dout.contiguous(), ctx.scale, ctx.norm)
        return (*grads, None, None)


def chunk_gdn2(
    q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, cu_seqlens=None,
    cu_seqlens_cpu=None, safe_gate=False, lower_bound=None, disable_recompute=False,
    return_intermediate_states=False, state_v_first=False, **kwargs,
):
    """Compute GDN-2 and return ``(o, None)``.

    S[t] = (I - k[t] (b[t]*k[t])^T) diag(exp(g[t])) S[t-1]
           + k[t] (w[t]*v[t])^T; o[t] = scale q[t]^T S[t].

    q/k: [B,T,H,K], v/w: [B,T,HV,V], g/b: [B,T,HV,K].
    HV is a multiple of H (GVA); heads are local under TP, with no collectives.
    BF16 q/k/v, independently BF16/FP32 g/b/w; log decay in [-5, 0] (caller precondition).
    B>=1, any T including tails/zero, 1<=K<=128 and V>=1. L2 norm uses eps=1e-6.
    This first implementation always retains FP32 chunk intermediates until
    backward, regardless of disable_recompute. safe_gate is always used.
    V>128 uses independent native value tiles. Packed sequences use one native
    call per sequence (CPU boundary metadata, or one device-to-host sync).
    Nonzero initial/final states, raw gate activation,
    intermediate-state output and double backward are not supported.
    """
    if (initial_state is not None or output_final_state or use_gate_in_kernel
            or return_intermediate_states or state_v_first or lower_bound is not None):
        raise NotImplementedError("gdn2: only zero-state, post-activation gates are supported")
    if kwargs:
        raise TypeError(f"gdn2: unsupported keyword arguments: {', '.join(kwargs)}")
    if q.ndim != 4 or v.ndim != 4 or k.shape != q.shape:
        raise ValueError("gdn2: q/k/v must be BTHD, with q.shape == k.shape")
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    if (B < 1 or H < 1 or HV < 1 or HV % H or v.shape[:2] != (B, T)
            or not 1 <= K <= 128 or V < 1):
        raise ValueError("gdn2: B/H/HV must be positive, HV%H==0, 1<=K<=128 and V>=1")
    if g.shape != (B, T, HV, K) or b.shape != g.shape or w.shape != v.shape:
        raise ValueError("gdn2: g/b must be [B,T,HV,K], w must match v")
    tensors = (q, k, v, g, b, w)
    if any(x.device != q.device or not x.is_cuda for x in tensors):
        raise ValueError("gdn2: all inputs must be on the same GPU")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("gdn2: q/k/v must be BF16")
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (g, b, w)):
        raise TypeError("gdn2: g/b/w must be BF16 or FP32")
    scale = K ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or not 0 < scale <= 3.0e38:
        raise ValueError("gdn2: scale must be finite and positive")
    def aligned(x):
        x = x.contiguous()
        return x.clone() if x.numel() and x.data_ptr() % 16 else x

    def dense(xs):
        q_, k_, v_, g_, b_, w_ = xs
        if V <= 128:
            return _Gdn2.apply(*(aligned(x) for x in xs), scale, use_qk_l2norm_in_kernel)
        tiles = []
        for start in range(0, V, 128):
            tile = (q_, k_, v_[..., start:start + 128], g_, b_, w_[..., start:start + 128])
            tiles.append(_Gdn2.apply(*(aligned(x) for x in tile), scale, use_qk_l2norm_in_kernel))
        return torch.cat(tiles, dim=-1)

    if cu_seqlens is None:
        if cu_seqlens_cpu is not None:
            raise ValueError("gdn2: cu_seqlens_cpu requires cu_seqlens")
        return dense(tensors), None
    if (B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.device != q.device):
        raise ValueError("gdn2: packed inputs require B=1 and integer GPU cu_seqlens")
    boundaries = cu_seqlens.cpu() if cu_seqlens_cpu is None else cu_seqlens_cpu
    if (boundaries.device.type != "cpu" or boundaries.shape != cu_seqlens.shape
            or boundaries.dtype not in (torch.int32, torch.int64)):
        raise ValueError("gdn2: cu_seqlens_cpu must be matching integer CPU metadata")
    edges = boundaries.tolist()
    if edges[0] != 0 or edges[-1] != T or any(a > b for a, b in zip(edges, edges[1:])):
        raise ValueError("gdn2: boundaries must be monotone, start at zero and end at T")
    outputs = [dense(tuple(x[:, start:end] for x in tensors)) for start, end in zip(edges, edges[1:])]
    return torch.cat(outputs, dim=1), None
