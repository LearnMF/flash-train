# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GPU acceptance tests; run on gfx936 after building the extension."""
import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires gfx936 and the compiled flash-train extension", allow_module_level=True)

from flash_train.torch import chunk_gdn2, gdn2_reference
from gdn2_math_test import make_inputs


# FLA v0.5.2 tests/ops/test_gdn2.py::test_chunk; retain the tighter old gate bound.
LIMITS = {"o": 0.005, "dq": 0.01, "dk": 0.01, "dv": 0.01, "dg": 0.015, "db": 0.015, "dw": 0.015}
GRADS = ("dq", "dk", "dv", "dg", "db", "dw")


def close(actual, expected, name="o"):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = actual - expected
    absolute = delta.abs().max().item()
    relative = (delta.square().mean().sqrt() / (expected.square().mean().sqrt() + 1e-8)).item()
    print(json.dumps(dict(tensor=name, relative_rms=relative, max_abs=absolute, limit=LIMITS[name])))
    assert absolute <= 1e-6 or relative < LIMITS[name], (name, relative)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("shape", [(1,1,1,1,1,1), (2,7,2,4,5,9), (1,63,2,2,16,17),
                                 (1,64,1,1,32,32), (2,65,1,2,17,33), (1,129,1,1,128,128), (1,65,1,2,17,256)])
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_six_gradients(shape, norm, gate_dtype):
    torch.manual_seed(12)
    inputs = make_inputs(shape, torch.float32)
    if not norm:
        inputs[0], inputs[1] = inputs[0]/shape[-2]**0.5, inputs[1]/shape[-2]**0.5
    inputs = [x.to(device="cuda", dtype=torch.bfloat16 if i<3 else gate_dtype).requires_grad_()
              for i,x in enumerate(inputs)]
    refs = [x.detach().double().cpu().requires_grad_() for x in inputs]
    expected = gdn2_reference(*refs, scale=0.37, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(inputs[2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out, state = chunk_gdn2(*inputs, scale=0.37, use_qk_l2norm_in_kernel=norm)
        grads = torch.autograd.grad(out, inputs, do)
    torch.cuda.current_stream().wait_stream(stream)
    expected_grads = torch.autograd.grad(expected, refs, do.double().cpu())
    assert state is None
    close(out, expected)
    for name, got, want in zip(GRADS, grads, expected_grads):
        close(got, want, name)


@pytest.mark.parametrize("tp", [1,2,4,8])
def test_head_sharding_forward_and_backward(tp):
    torch.manual_seed(21)
    inputs = [x.cuda().bfloat16().requires_grad_() for x in make_inputs((1,65,32,32,128,128),torch.float32)]
    out, _ = chunk_gdn2(*inputs, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(out)
    grads = torch.autograd.grad(out, inputs, do)
    for shard in range(tp):
        start,end=shard*32//tp,(shard+1)*32//tp
        local = [x.detach()[:,:,start:end].contiguous().requires_grad_() for x in inputs]
        actual, _ = chunk_gdn2(*local, use_qk_l2norm_in_kernel=True)
        local_grads = torch.autograd.grad(actual, local, do[:,:,start:end].contiguous())
        torch.testing.assert_close(actual, out[:,:,start:end], rtol=0, atol=0)
        for actual, expected in zip(local_grads,grads):
            torch.testing.assert_close(actual, expected[:,:,start:end], rtol=0, atol=0)


def test_rejects_unsupported_features():
    inputs = [x.cuda().bfloat16() for x in make_inputs((1,3,1,1,4,4),torch.float32)]
    for kw in ({"output_final_state": True}, {"use_gate_in_kernel": True},
               {"state_v_first": True}):
        with pytest.raises(NotImplementedError):
            chunk_gdn2(*inputs, **kw)
    with pytest.raises(ValueError):
        chunk_gdn2(*inputs[:4], inputs[4][...,:1], inputs[5])
    with pytest.raises(ValueError):
        chunk_gdn2(*inputs,scale=float("nan"))


def test_empty_sequence():
    xs=[x.cuda().bfloat16().requires_grad_() for x in make_inputs((2,0,1,2,17,33),torch.float32)]
    out,_=chunk_gdn2(*xs)
    assert out.shape==xs[2].shape
    for grad,x in zip(torch.autograd.grad(out,xs,torch.empty_like(out)),xs):
        assert grad.shape==x.shape


@pytest.mark.parametrize("pattern", ["zero", "strong", "alternating"])
def test_decay_extremes(pattern):
    torch.manual_seed(74)
    xs = [x.cuda().bfloat16() for x in make_inputs((1,65,1,2,37,33), torch.float32)]
    if pattern == "zero":
        xs[3].zero_()
    elif pattern == "strong":
        xs[3].fill_(-5)
    else:
        xs[3].zero_()
        xs[3][:, ::2, :, ::2] = -5
        xs[3][:, 1::2, :, 1::2] = -5
    xs = [x.requires_grad_() for x in xs]
    refs = [x.detach().double().cpu().requires_grad_() for x in xs]
    out, _ = chunk_gdn2(*xs, use_qk_l2norm_in_kernel=True)
    expected = gdn2_reference(*refs, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(out)
    close(out, expected)
    for name, a, b in zip(GRADS, torch.autograd.grad(out, xs, do),
                          torch.autograd.grad(expected, refs, do.double().cpu())):
        close(a, b, name)


@pytest.mark.parametrize("cpu_metadata", [True, False])
def test_packed_sequences(cpu_metadata):
    torch.manual_seed(72)
    xs = [x.cuda().bfloat16().requires_grad_() for x in make_inputs((1,71,1,2,17,256), torch.float32)]
    edges = [0, 5, 5, 70, 71]
    cu = torch.tensor(edges, device="cuda", dtype=torch.int32)
    out, _ = chunk_gdn2(*xs, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu() if cpu_metadata else None,
                       use_qk_l2norm_in_kernel=True)
    refs = [x.detach().double().cpu().requires_grad_() for x in xs]
    expected = torch.cat([gdn2_reference(*(x[:, a:b] for x in refs), use_qk_l2norm_in_kernel=True)
                          for a, b in zip(edges, edges[1:]) if a < b], dim=1)
    do = torch.randn_like(out)
    close(out, expected)
    for name, a, b in zip(GRADS, torch.autograd.grad(out, xs, do),
                          torch.autograd.grad(expected, refs, do.double().cpu())):
        close(a, b, name)


@pytest.mark.parametrize("K,V", [(37, 33), (64, 256), (128, 128), (128, 256)])
@pytest.mark.parametrize("decay", ["normal", "weak", "zero", "boundary"])
@pytest.mark.parametrize("distribution", ["positive", "normal"])
@pytest.mark.parametrize("norm", [False, True])
def test_fla_precision_stress(monkeypatch, K, V, decay, distribution, norm):
    from flash_train.torch import gdn2 as native

    torch.manual_seed(42)
    rand = torch.rand if distribution == "positive" else torch.randn
    q, k = [rand(2, 65, 1, K, device="cuda") for _ in range(2)]
    if not norm:
        q, k = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k)]
    v = rand(2, 65, 2, V, device="cuda") * 0.5
    g = -torch.empty(2, 65, 2, K, device="cuda").uniform_(0.1, 5.0)
    if decay == "weak":
        g *= 0.01
    elif decay == "zero":
        g.zero_()
    elif decay == "boundary":
        g.fill_(-5)
    b = torch.rand_like(g)
    w = torch.rand_like(v)
    xs = [x.bfloat16().requires_grad_() for x in (q, k, v, g, b, w)]
    refs = [x.detach().double().cpu().requires_grad_() for x in xs]
    calls = []
    apply = native._Gdn2.apply

    def recorded(*args):
        calls.append((args[0].shape[-1], args[2].shape[-1]))
        return apply(*args)

    monkeypatch.setattr(native._Gdn2, "apply", recorded)
    out, _ = chunk_gdn2(*xs, use_qk_l2norm_in_kernel=norm)
    assert calls == [(K, min(128, V - start)) for start in range(0, V, 128)]
    expected = gdn2_reference(*refs, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(out)
    close(out, expected)
    for name, got, want in zip(GRADS, torch.autograd.grad(out, xs, do),
                               torch.autograd.grad(expected, refs, do.double().cpu())):
        close(got, want, name)
