# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only independent recurrence/autograd vs chunk WY/analytic backward."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "gdn2_reference", Path(__file__).parents[2] / "python/flash_train/torch/gdn2_reference.py"
)
ref_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_module)
reference = ref_module.gdn2_reference
torch.set_num_threads(1)


def chunk_forward(inputs, scale, norm, chunk=64):
    q, k, v, g, b, w = inputs
    rawq, rawk = q, k
    qr, kr = (torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k))
    if norm:
        q, k = q * qr, k * kr
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    q, k = (x.repeat_interleave(HV // H, dim=2) for x in (q, k))
    q, k, v, g, b, w = (x.transpose(1, 2) for x in (q, k, v, g, b, w))
    state = q.new_zeros(B, HV, K, V)
    outputs, saved = [], []
    for start in range(0, T, chunk):
        length = min(chunk, T - start)
        qi, ki, vi, gi, bi, wi = (F.pad(x[:, :, start:start+length], (0, 0, 0, chunk-length))
                                 for x in (q, k, v, g, b, w))
        G = gi.cumsum(-2)
        causal = torch.ones(chunk, chunk, dtype=torch.bool).tril()
        e = torch.where(causal[..., None], G.unsqueeze(-2)-G.unsqueeze(-3), 0).exp()
        A = ((bi*ki).unsqueeze(-2) * ki.unsqueeze(-3) * e).sum(-1).tril(-1)
        P = (scale * qi.unsqueeze(-2) * ki.unsqueeze(-3) * e).sum(-1).tril()
        M = torch.linalg.solve_triangular(A + torch.eye(chunk), torch.eye(chunk).expand_as(A), upper=False)
        R, Q = bi*ki*G.exp(), scale*qi*G.exp()
        E = ki * (G[..., -1:, :] - G).exp()
        U, W = M @ (wi*vi), M @ R
        X = U - W @ state
        outputs.append((Q @ state + P @ X)[..., :length, :])
        saved.append((qi, ki, vi, bi, wi, G, e, M, Q, W, E, X, state, length))
        state = G[..., -1, :].exp().unsqueeze(-1)*state + E.transpose(-1, -2) @ X
    out = torch.cat(outputs, -2).transpose(1, 2)
    return out, (saved, rawq, rawk, qr, kr, norm, scale, H)


def chunk_backward(dout, ctx):
    saved, rawq, rawk, qr, kr, norm, scale, H = ctx
    B, T, HV, V = dout.shape
    K, chunk = rawq.shape[-1], saved[0][0].shape[-2]
    dh = dout.new_zeros(B, HV, K, V)
    grads = [[] for _ in range(6)]
    for c in range(len(saved)-1, -1, -1):
        q, k, v, b, w, G, e, M, Q, W, E, X, state, length = saved[c]
        do = F.pad(dout[:, c*chunk:c*chunk+length].transpose(1, 2), (0, 0, 0, chunk-length))
        P = (scale*q.unsqueeze(-2)*k.unsqueeze(-3)*e).sum(-1).tril()
        dx = E @ dh + P.transpose(-1, -2) @ do
        Y = M.transpose(-1, -2) @ dx
        dQ, dR = do @ state.transpose(-1, -2), -Y @ state.transpose(-1, -2)
        dE = X @ dh.transpose(-1, -2)
        dP, dA = (do @ X.transpose(-1, -2)).tril(), (-Y @ X.transpose(-1, -2)).tril(-1)
        dq = scale*G.exp()*dQ
        dk = b*G.exp()*dR + (G[..., -1:, :]-G).exp()*dE
        db = k*G.exp()*dR
        dG = Q*dQ + b*k*G.exp()*dR - E*dE
        dend = (state*dh).sum(-1)*G[..., -1, :].exp() + (E*dE).sum(-2)
        dq += scale * (dP[..., None]*k.unsqueeze(-3)*e).sum(-2)
        dk += b*(dA[..., None]*k.unsqueeze(-3)*e).sum(-2)
        db += k*(dA[..., None]*k.unsqueeze(-3)*e).sum(-2)
        column = (dA[..., None]*(b*k).unsqueeze(-2) + dP[..., None]*scale*q.unsqueeze(-2))*e
        dk += column.sum(-3)
        pairs = column * k.unsqueeze(-3)
        dG += pairs.sum(-2) - pairs.sum(-3)
        # Padded tail G entries all equal G[last real token]. The kernel adds dend there.
        dG[..., length-1, :] += dend
        dg = dG.flip(-2).cumsum(-2).flip(-2)
        for dest, value in zip(grads, (dq, dk, w*Y, dg, db, v*Y)):
            dest.append(value[..., :length, :])
        dh = G[..., -1, :].exp().unsqueeze(-1)*dh + Q.transpose(-1, -2)@do - W.transpose(-1, -2)@dx
    result = [torch.cat(x[::-1], -2).transpose(1, 2) for x in grads]
    for index, raw, rstd in ((0, rawq, qr), (1, rawk, kr)):
        value = result[index].reshape(B, T, H, HV//H, K).sum(3)
        if norm:
            unit = raw*rstd
            value = (value - unit*(value*unit).sum(-1, keepdim=True))*rstd
        result[index] = value
    return result


def make_inputs(shape, dtype=torch.float64, device="cpu"):
    B, T, H, HV, K, V = shape
    q, k = [torch.randn(B, T, H, K, dtype=dtype, device=device) for _ in range(2)]
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    g = -torch.rand(B, T, HV, K, dtype=dtype, device=device) * 0.35
    b = torch.rand_like(g) * 1.7
    w = torch.rand_like(v)
    return [q, k, v, g, b, w]


@pytest.mark.parametrize("shape", [(1,1,1,1,3,5), (2,7,2,4,5,9), (1,63,2,2,16,17),
                                 (1,64,1,1,32,32), (2,65,1,2,17,33), (1,129,1,1,128,128)])
@pytest.mark.parametrize("norm", [False, True])
def test_chunk_against_recurrence_and_autograd(shape, norm):
    torch.manual_seed(123)
    xs = make_inputs(shape)
    if not norm:
        xs[0], xs[1] = xs[0]/shape[-2]**0.5, xs[1]/shape[-2]**0.5
    xs = [x.requires_grad_() for x in xs]
    scale = 0.37
    expected = reference(*xs, scale=scale, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, xs, do)
    with torch.no_grad():
        actual, ctx = chunk_forward(xs, scale, norm)
        actual_grads = chunk_backward(do, ctx)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_scalar_gate_special_case_and_noncommutativity():
    torch.manual_seed(20)
    q,k,v,g,b,w = make_inputs((1,9,1,1,4,4))
    q,k = q/2,k/2
    beta = torch.rand(1,9,1,1,dtype=q.dtype,requires_grad=True)
    o = reference(q,k,v,g,beta.expand_as(b),beta.expand_as(w))
    state = q.new_zeros(1,1,4,4)
    wrong = state.clone()
    out, swapped = [], []
    for t in range(9):
        decayed = state*g[:,t].exp().unsqueeze(-1)
        erase = torch.einsum("bhk,bhkv->bhv",k[:,t],decayed)
        state = decayed + k[:,t,:,:,None]*(beta[:,t]*(v[:,t]-erase)).unsqueeze(-2)
        out.append(0.5*torch.einsum("bhk,bhkv->bhv",q[:,t],state))
        projection = torch.einsum("bhk,bhkv->bhv",beta[:,t]*k[:,t],wrong)
        wrong = (wrong-k[:,t,:,:,None]*projection.unsqueeze(-2))*g[:,t].exp().unsqueeze(-1)
        wrong = wrong+k[:,t,:,:,None]*(beta[:,t]*v[:,t]).unsqueeze(-2)
        swapped.append(0.5*torch.einsum("bhk,bhkv->bhv",q[:,t],wrong))
    torch.testing.assert_close(o,torch.stack(out,1))
    assert (o-torch.stack(swapped,1)).abs().max()>1e-4


@pytest.mark.parametrize("decay", [0.0, -5.0, -80.0])
def test_extreme_decay_and_zero_norm(decay):
    xs=make_inputs((1,65,1,1,7,11))
    xs[0].zero_()
    xs[1].zero_()
    xs[3].fill_(decay)
    xs=[x.requires_grad_() for x in xs]
    o=reference(*xs,use_qk_l2norm_in_kernel=True)
    do=torch.randn_like(o)
    expect=torch.autograd.grad(o,xs,do)
    actual,ctx=chunk_forward(xs,7**-0.5,True)
    result=chunk_backward(do,ctx)
    torch.testing.assert_close(actual,o)
    for a, b in zip(result, expect):
        torch.testing.assert_close(a, b)
