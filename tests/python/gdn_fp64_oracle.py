# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed FP64 block algebra, independent of the native recurrent kernels."""

import torch
from torch.utils.checkpoint import checkpoint


def chunk_reference(q, k, v, g, beta, *, norm=True, scale=None):
    q, k, v, g, beta = (a.double() for a in (q, k, v, g, beta))
    if norm:
        q, k = (a * torch.rsqrt(a.square().sum(-1, keepdim=True) + 1e-6) for a in (q, k))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    ratio = v.shape[2] // q.shape[2]
    q, k = (a.repeat_interleave(ratio, dim=2) for a in (q, k))
    q, k, v, g, beta = (a.transpose(1, 2) for a in (q, k, v, g, beta))
    state = q.new_zeros(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1])

    def block(state, q, k, v, g, beta):
        gate = g.cumsum(-1)
        decay = (gate[..., :, None] - gate[..., None, :]).clamp_max(0).exp().tril()
        gram = (k @ k.transpose(-1, -2)) * decay
        lhs = torch.eye(q.shape[-2], dtype=q.dtype, device=q.device) + gram.tril(-1) * beta[..., None]
        rhs = beta[..., None] * (v - (k * gate.exp()[..., None]) @ state)
        residual = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)
        out = (q * gate.exp()[..., None]) @ state + ((q @ k.transpose(-1, -2)) * decay) @ residual
        state = gate[..., -1, None, None].exp() * state + (
            k * (gate[..., -1, None] - gate).exp()[..., None]).transpose(-1, -2) @ residual
        return state, out * scale

    outputs = []
    for start in range(0, q.shape[2], 64):
        state, out = checkpoint(block, state, *(a[:, :, start:start + 64] for a in (q, k, v, g, beta)),
                                use_reentrant=False)
        outputs.append(out)
    return torch.cat(outputs, dim=2).transpose(1, 2)
