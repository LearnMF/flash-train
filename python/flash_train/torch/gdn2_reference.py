# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent differentiable recurrence for validation, not a runtime fallback."""
import torch


def gdn2_reference(q, k, v, g, b, w, scale=None, use_qk_l2norm_in_kernel=False):
    """Dense zero-state GDN-2; keep the caller's floating-point precision."""
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    q, k = (x.repeat_interleave(HV // H, dim=2) for x in (q, k))
    scale = K ** -0.5 if scale is None else scale
    state = q.new_zeros(B, HV, K, V)
    out = []
    for i in range(T):
        decayed = state * g[:, i].exp().unsqueeze(-1)
        erased = torch.einsum("bhk,bhkv->bhv", b[:, i] * k[:, i], decayed)
        state = decayed + k[:, i, :, :, None] * (w[:, i] * v[:, i] - erased).unsqueeze(-2)
        out.append(scale * torch.einsum("bhk,bhkv->bhv", q[:, i], state))
    if not out:
        # Preserve the zero derivative for all six inputs.
        return v * 0 + sum(x.sum() * 0 for x in (q, k, g, b, w))
    return torch.stack(out, dim=1)
