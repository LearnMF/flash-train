# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gated Delta Net (scalar log decay), with an independent PyTorch reference."""

import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

_TILE = 128


def _is_native_device(x):
    return x.is_cuda


def _device_guard(device):
    return torch.cuda.device(device)


def _shape(q, k, v, g, beta):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("GDN: q/k must be [B,T,H,K], v must be [B,T,HV,V]")
    B, T, H, K = q.shape
    if min(B, H, K, v.shape[2], v.shape[3]) <= 0:
        raise ValueError("GDN: batch, heads and head dimensions must be positive")
    HV = v.shape[2]
    if v.shape[:2] != (B, T) or HV % H:
        raise ValueError("GDN: v must share B,T and HV must be a multiple of H")
    if g.shape != (B, T, HV) or beta.shape != g.shape:
        raise ValueError("GDN: g and beta must be [B,T,HV]")
    if any(x.device != q.device for x in (k, v, g, beta)):
        raise ValueError("GDN: inputs must be on the same device")
    return B, T, H, K, HV


def _expand_heads(x, heads):
    # Consecutive value heads share one q/k head. Autograd sums their dq/dk.
    ratio = heads // x.shape[2]
    return x if ratio == 1 else x.repeat_interleave(ratio, dim=2)


class _GdnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu, scale, l2norm, save_h, needs_bwd):
        from ._flash_train_torch import gdn_forward

        save_qkn = needs_bwd and l2norm
        o, qr, kr, aqk, akk, qn, kn, h = gdn_forward(
            q, k, v, g, beta, cu, scale, l2norm, save_qkn, needs_bwd and save_h)
        if needs_bwd:
            ctx.save_for_backward(None if save_qkn else q, None if save_qkn else k,
                                  v, g, beta, cu, qr, kr, aqk, akk, qn, kn, h)
        ctx.scale, ctx.l2norm = scale, l2norm
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        from ._flash_train_torch import gdn_backward

        q, k, v, g, beta, cu, qr, kr, aqk, akk, qn, kn, h = ctx.saved_tensors
        dq, dk, dv, dg, db = gdn_backward(
            do.contiguous(), q, k, v, g, beta, cu, qr, kr, aqk, akk, qn, kn, h,
            ctx.scale, ctx.l2norm)
        return dq, dk, dv, dg, db, None, None, None, None, None


def chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
        state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, *,
        use_gate_in_kernel=False, A_log=None, dt_bias=None, disable_recompute=None,
        check_decay=False):
    """Return ``(o, None)`` using native gfx936 HIP forward/backward kernels.

    q/k: bf16 [B,T,H,K<=128], v: bf16 [B,T,HV,V], g/beta: fp32 [B,T,HV].
    K<128 is zero-padded to 128 (L2 sums and the recurrence are unchanged);
    V is split into zero-padded 128-wide tiles, one native call per tile.
    HV must be divisible by H; local TP head counts 4/8/16/32 need no special
    global-head metadata. B>1, T tails and packed sequences are supported.
    g is a log decay in [-5,0], beta is post-activation unless sigmoid is
    requested. ``check_decay=True`` verifies the gate range (host sync).
    cu_seqlens must be monotone, start at 0 and end at T, with B=1. Contents
    are a caller precondition, like the C API; no host synchronization occurs.

    ``disable_recompute=True`` saves chunk states; False recomputes them in
    backward. None saves them at HV<=64 (inherited KDA policy, not GDN-tuned).
    Optional q/k L2 norm uses rsqrt(sum(x*x)+1e-6). This is a scalar-gate
    recurrence implemented in native HIP with FP32 intermediates/checkpoints,
    not a FLA or Python-reference fallback.
    Initial/final state, CP, raw-gate activation and K > 128 are unsupported.
    Only first-order gradients are supported. The FP32 accuracy path replaces
    the BF16-intermediate pipeline; performance is not yet optimized.
    """
    if state_v_first:
        raise NotImplementedError("GDN: state_v_first has no effect without state tensors")
    if initial_state is not None or output_final_state:
        raise NotImplementedError("GDN: initial/final state is not supported")
    if cp_context is not None:
        raise NotImplementedError("GDN: context parallelism is not supported")
    if use_gate_in_kernel or A_log is not None or dt_bias is not None:
        raise NotImplementedError("GDN: pass precomputed fp32 log decay g, not a raw gate")
    B, T, H, K, HV = _shape(q, k, v, g, beta)
    if K > _TILE:
        raise NotImplementedError("GDN: native gfx936 kernels currently require K<=128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("GDN: q/k/v must be bf16")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("GDN: g/beta must be fp32")
    if not _is_native_device(q):
        raise NotImplementedError("GDN: native kernels require a gfx936 device; use gdn_reference on CPU")
    if scale is None:
        scale = K ** -0.5
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("GDN: scale must be finite and positive")
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("GDN: cu_seqlens must be [N+1], N>=1, with B=1")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("GDN: cu_seqlens must be an integer tensor")
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    elif B > 1 and T:
        cu_seqlens = torch.arange(B + 1, device=q.device, dtype=torch.int32) * T
    if check_decay and not bool(torch.all(torch.isfinite(g) & (g >= -5) & (g <= 0))):
        raise ValueError("GDN: log decay must be finite and in [-5,0]")
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    if not T:
        # Preserve all five empty-input autograd edges without launching kernels.
        return v + (q.sum() + k.sum() + g.sum() + beta.sum()).to(v.dtype) * 0, None
    save_h = HV <= 64 if disable_recompute is None else bool(disable_recompute)
    # Guard the current device, not only allocation device (important under TP).
    with _device_guard(q.device):
        o = _native_tiles(q, k, v, g, beta, cu_seqlens, float(scale),
                          bool(use_qk_l2norm_in_kernel), save_h, _GdnFunction.apply)
    return o, None


def _native_tiles(q, k, v, g, beta, cu, scale, l2norm, save_h, op):
    """Map [B,T,H,K<=128] x [B,T,HV,V] onto native K=V=128 packed calls.

    V columns of the state evolve independently, so tiles are exact. Autograd
    slices padded gradients and sums dq/dk/dg/dbeta over tiles and head groups.
    """
    B, T, _, K, HV = _shape(q, k, v, g, beta)
    V = v.shape[-1]
    q, k = (_expand_heads(x, HV) for x in (q, k))
    if K < _TILE:
        q, k = (F.pad(x, (0, _TILE - K)) for x in (q, k))
    flat = lambda x: x.reshape(B * T, *x.shape[2:]).contiguous()  # noqa: E731
    q, k, g, beta = (flat(x) for x in (q, k, g, beta))
    tiles = []
    for start in range(0, V, _TILE):
        width = min(_TILE, V - start)
        tile = v[..., start:start + width]
        if width < _TILE:
            tile = F.pad(tile, (0, _TILE - width))
        tile = flat(tile)
        needs_bwd = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, tile, g, beta))
        tiles.append(op(q, k, tile, g, beta, cu, scale, l2norm, save_h, needs_bwd)[..., :width])
    o = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=-1)
    return o.reshape(B, T, HV, V)


def gdn_reference(q, k, v, g, beta, *, scale=None, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=None):
    """Differentiable FP64 recurrence for validation; arbitrary K/V, CPU or GPU.

    This is deliberately never called by the native operator. Returns o only.
    """
    B, T, H, K, HV = _shape(q, k, v, g, beta)
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = _expand_heads(q, HV), _expand_heads(k, HV)
    scale = K ** -0.5 if scale is None else scale
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("GDN reference: cu_seqlens requires B=1")
        offsets = cu_seqlens.tolist()
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != T or any(
                a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("GDN reference: invalid cu_seqlens")
        sequences = [(0, a, b) for a, b in zip(offsets, offsets[1:])]
    else:
        sequences = [(b, 0, T) for b in range(B)]
    outputs = []
    for batch, start, end in sequences:
        state = q.new_zeros(HV, K, v.shape[-1])
        for t in range(start, end):
            kt, qt = k[batch, t], q[batch, t]
            state = state * g[batch, t].exp()[:, None, None]
            residual = v[batch, t] - torch.einsum("hk,hkv->hv", kt, state)
            state = state + torch.einsum("hk,hv->hkv", kt, residual * beta[batch, t, :, None])
            outputs.append(torch.einsum("hk,hkv->hv", qt * scale, state))
    if not outputs:
        return v + (q.sum() + k.sum() + g.sum() + beta.sum()) * 0
    return torch.stack(outputs).reshape(B, T, HV, v.shape[-1])
