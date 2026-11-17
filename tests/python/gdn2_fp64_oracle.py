# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed FP64 GDN-2 block algebra with an independent triangular solve."""

import torch
from torch.utils.checkpoint import checkpoint


def chunk_reference(q, k, v, g, b, w, *, norm=True, scale=None):
    q, k, v, g, b, w = (x.double() for x in (q, k, v, g, b, w))
    if norm:
        q, k = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    ratio = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(ratio, dim=2) for x in (q, k))
    q, k, v, g, b, w = (x.transpose(1, 2) for x in (q, k, v, g, b, w))
    state = q.new_zeros(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1])

    def block(state, q, k, v, g, b, w):
        length = q.shape[-2]
        cumulative = g.cumsum(-2)
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        differences = cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)
        decay = torch.where(causal[..., None], differences, 0).exp()
        erase = ((b * k).unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).tril(-1)
        scores = (scale * q.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).tril()
        lhs = torch.eye(length, dtype=q.dtype, device=q.device) + erase
        rhs = w * v - (b * k * cumulative.exp()) @ state
        residual = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)
        out = (scale * q * cumulative.exp()) @ state + scores @ residual
        keys = k * (cumulative[..., -1:, :] - cumulative).exp()
        state = cumulative[..., -1, :].exp().unsqueeze(-1) * state + keys.transpose(-1, -2) @ residual
        return state, out

    outputs = []
    for start in range(0, q.shape[2], 64):
        state, out = checkpoint(block, state, *(x[:, :, start:start + 64] for x in (q, k, v, g, b, w)),
                                use_reentrant=False)
        outputs.append(out)
    return torch.cat(outputs, dim=2).transpose(1, 2)
