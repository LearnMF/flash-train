# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU algebra tests run without the extension; HIP tests require gfx936."""

import contextlib
import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "gdn_cpu", Path(__file__).parents[2] / "python/flash_train/torch/gdn.py")
gdn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gdn)


def inputs(B, T, H, HV, K, V, device="cpu", dtype=torch.float64):
    torch.manual_seed(290)
    q, k = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1 for _ in range(2))
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype) * 0.3
    g = -torch.rand(B, T, HV, device=device, dtype=torch.float64) * 0.3
    beta = torch.rand(B, T, HV, device=device, dtype=torch.float64)
    return [x.requires_grad_() for x in (q, k, v, g, beta)]


def vector_gate_recurrence(q, k, v, g, beta, norm):
    # Independent per-value-head matrix transition: (I - beta*k*k^T) exp(g) S.
    if norm:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    batches = []
    for b in range(B):
        heads = []
        for h in range(HV):
            state = q.new_zeros(K, V)
            rows = []
            for t in range(T):
                kt = k[b, t, h // (HV // H)]
                gt = g[b, t, h].expand(K)
                transition = torch.eye(K, dtype=q.dtype, device=q.device) - beta[b, t, h] * torch.outer(kt, kt)
                state = transition @ (gt.exp()[:, None] * state) + beta[b, t, h] * torch.outer(kt, v[b, t, h])
                rows.append((q[b, t, h // (HV // H)] @ state) * K**-0.5)
            heads.append(torch.stack(rows))
        batches.append(torch.stack(heads, dim=1))
    return torch.stack(batches)


@pytest.mark.parametrize("B,H,HV,K,V,T", [
    (1, 4, 4, 32, 33, 65), (2, 2, 8, 37, 64, 7), (1, 4, 8, 64, 128, 5), (1, 2, 4, 128, 256, 3)])
@pytest.mark.parametrize("norm", [False, True])
def test_scalar_gate_equals_vector_gate_and_gradients(B, H, HV, K, V, T, norm):
    torch.set_num_threads(1)
    x = inputs(B, T, H, HV, K, V)
    actual = gdn.gdn_reference(*x, use_qk_l2norm_in_kernel=norm)
    expected = vector_gate_recurrence(*x, norm)
    do = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, x, do, retain_graph=True)
    ge = torch.autograd.grad(expected, x, do)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-9)


def test_packed_batch_boundaries_and_empty_sequences():
    x = inputs(2, 7, 2, 4, 37, 33)
    packed = [t.reshape(1, 14, *t.shape[2:]) for t in x]
    expected = gdn.gdn_reference(*x)
    actual = gdn.gdn_reference(*packed, cu_seqlens=torch.tensor([0, 0, 7, 14, 14]))
    torch.testing.assert_close(actual.reshape_as(expected), expected)


def test_reduction_commutes_with_chunk_reverse_scan():
    x = torch.randn(65, 4, 128, dtype=torch.float64)
    for block in x.split(64):
        vector = block.flip(0).cumsum(0).flip(0).sum(-1)
        scalar = block.sum(-1).flip(0).cumsum(0).flip(0)
        torch.testing.assert_close(scalar, vector)


def test_reference_gradcheck():
    x = inputs(1, 3, 1, 2, 3, 4)
    assert torch.autograd.gradcheck(
        lambda *a: gdn.gdn_reference(*a, use_qk_l2norm_in_kernel=True), tuple(x), fast_mode=True)


ISSUE29_SHAPES = [(32, 32), (32, 64), (32, 256), (37, 33), (64, 32), (64, 64), (64, 256), (128, 33)]


class NativeMock:
    """Records native K=V=128 calls and evaluates them with the FP64 reference."""

    def __init__(self):
        self.calls = []
        self.reference = gdn.gdn_reference

    def __call__(self, q, k, v, g, beta, cu, scale, l2norm, save_h, needs_bwd):
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
        assert q.ndim == v.ndim == 3 and g.shape == beta.shape == v.shape[:2]
        self.calls.append((tuple(q.shape), tuple(v.shape), cu is not None, scale, l2norm))
        o = self.reference(*(x[None] for x in (q, k, v, g, beta)), scale=scale,
                           use_qk_l2norm_in_kernel=l2norm, cu_seqlens=cu)
        return o[0].to(v.dtype)


@pytest.mark.parametrize("K,V", ISSUE29_SHAPES)
@pytest.mark.parametrize("B,T,H,HV", [(1, 65, 4, 4), (2, 7, 2, 4)])
@pytest.mark.parametrize("norm", [False, True])
def test_non128_tiles_are_exact_and_use_native_calls(K, V, B, T, H, HV, norm):
    torch.set_num_threads(1)
    x = inputs(B, T, H, HV, K, V)
    native = NativeMock()
    cu = torch.arange(B + 1, dtype=torch.int32) * T if B > 1 else None
    actual = gdn._native_tiles(*x, cu, K**-0.5, norm, False, native)
    expected = gdn.gdn_reference(*x, use_qk_l2norm_in_kernel=norm)
    assert len(native.calls) == (V + 127) // 128
    assert all(c[:2] == ((B * T, HV, 128), (B * T, HV, 128)) and c[3] == K**-0.5 for c in native.calls)
    do = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for a, e in zip(torch.autograd.grad(actual, x, do), torch.autograd.grad(expected, x, do)):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-9)


def test_zero_padding_preserves_l2_eps():
    x = torch.randn(3, 37, dtype=torch.float64) * 1e-4
    padded = torch.nn.functional.pad(x, (0, 91))
    norm = lambda y: y * torch.rsqrt(y.square().sum(-1, keepdim=True) + 1e-6)  # noqa: E731
    torch.testing.assert_close(norm(padded)[:, :37], norm(x), atol=0, rtol=0)
    assert torch.count_nonzero(norm(padded)[:, 37:]) == 0


@pytest.mark.parametrize("K,V", [(37, 33), (32, 256)])
def test_public_entry_uses_native_tiles(monkeypatch, K, V):
    native = NativeMock()
    monkeypatch.setattr(gdn, "_is_native_device", lambda x: True)
    monkeypatch.setattr(gdn, "_device_guard", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(gdn._GdnFunction, "apply", native)
    monkeypatch.setattr(gdn, "gdn_reference", lambda *a, **k: pytest.fail("reference fallback"))
    raw = inputs(2, 5, 2, 4, K, V)
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32) for i, t in enumerate(raw)]
    o, final = gdn.chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=True)
    assert final is None and o.shape == (2, 5, 4, V) and len(native.calls) == (V + 127) // 128
    assert all(c[2] and c[3] == K**-0.5 and c[4] for c in native.calls)


@pytest.mark.parametrize("K,V", [(129, 128), (256, 33)])
def test_native_unsupported_dimensions(K, V):
    with pytest.raises(NotImplementedError, match="K<=128"):
        gdn.chunk_gated_delta_rule(*inputs(1, 2, 4, 4, K, V))


def test_invalid_head_mapping():
    with pytest.raises(ValueError, match="multiple"):
        gdn.chunk_gated_delta_rule(*inputs(1, 2, 3, 4, 128, 128))


def gfx936():
    return torch.cuda.is_available() and getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", "").startswith("gfx936")


def rel_rms(actual, expected):
    return ((actual.double() - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-6)).item()


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("B,T,H,HV", [(1, 128, 4, 4), (1, 65, 8, 8), (2, 65, 4, 16),
                                      (4, 65, 4, 8), (8, 65, 4, 4), (1, 128, 32, 32)])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("save_h", [False, True])
def test_native_against_fp64(B, T, H, HV, norm, save_h):
    from flash_train.torch import chunk_gated_delta_rule

    raw = inputs(B, T, H, HV, 128, 128, device="cuda")
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
         for i, t in enumerate(raw)]
    ref_x = [t.detach().double().requires_grad_() for t in x]
    o, final = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=norm, disable_recompute=save_h)
    assert final is None
    ref = gdn.gdn_reference(*ref_x, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(o)
    actual = torch.autograd.grad(o, x, do)
    expected = torch.autograd.grad(ref, ref_x, do.double())
    assert rel_rms(o, ref) < 0.02
    for name, a, e, tol in zip(("dq", "dk", "dv", "dg", "dbeta"), actual, expected,
                               (0.04, 0.04, 0.04, 0.06, 0.06)):
        assert rel_rms(a, e) < tol, (name, rel_rms(a, e))


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("save_h", [False, True])
def test_native_head_sharding_forward_and_backward(tp, save_h):
    from flash_train.torch import chunk_gated_delta_rule

    raw = inputs(1, 130, 32, 32, 128, 128, device="cuda")
    xs = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
          for i, t in enumerate(raw)]
    kwargs = dict(use_qk_l2norm_in_kernel=True, disable_recompute=save_h)
    full, _ = chunk_gated_delta_rule(*xs, **kwargs)
    do = torch.randn_like(full)
    full_grads = torch.autograd.grad(full, xs, do)
    outputs, gradients = [], [[] for _ in xs]
    for rank in range(tp):
        start, end = rank * 32 // tp, (rank + 1) * 32 // tp
        shard = [x[:, :, start:end].detach().contiguous().requires_grad_() for x in xs]
        out, _ = chunk_gated_delta_rule(*shard, **kwargs)
        grads = torch.autograd.grad(out, shard, do[:, :, start:end].contiguous())
        outputs.append(out.detach())
        for values, grad in zip(gradients, grads):
            values.append(grad)
    assert rel_rms(torch.cat(outputs, dim=2), full) < 0.02
    for values, expected, budget in zip(gradients, full_grads, (0.04, 0.04, 0.04, 0.06, 0.06)):
        assert rel_rms(torch.cat(values, dim=2), expected) < budget


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
def test_native_packed_and_empty():
    from flash_train.torch import chunk_gated_delta_rule

    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
         for i, t in enumerate(inputs(1, 130, 4, 8, 128, 128, "cuda"))]
    cu = torch.tensor([0, 0, 65, 130, 130], dtype=torch.int32, device="cuda")
    ref_x = [t.detach().double().requires_grad_() for t in x]
    o, _ = chunk_gated_delta_rule(*x, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    ref = gdn.gdn_reference(*ref_x, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(o)
    assert rel_rms(o, ref) < 0.02
    for a, e in zip(torch.autograd.grad(o, x, do), torch.autograd.grad(ref, ref_x, do.double())):
        assert rel_rms(a, e) < 0.06
    empty = [t[:, :0].detach().requires_grad_() for t in x]
    o, _ = chunk_gated_delta_rule(*empty)
    assert o.shape == (1, 0, 8, 128)
    assert all(t.numel() == 0 for t in torch.autograd.grad(o.sum(), empty))


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("K,V", ISSUE29_SHAPES)
@pytest.mark.parametrize("B,T,H,HV", [(1, 65, 4, 8), (2, 65, 8, 8)])
@pytest.mark.parametrize("norm", [False, True])
def test_native_non128_against_fp64(monkeypatch, K, V, B, T, H, HV, norm):
    from flash_train.torch import chunk_gated_delta_rule
    from flash_train.torch import gdn as native_gdn

    calls = []
    apply = native_gdn._GdnFunction.apply

    def recorded(*args):
        calls.append((tuple(args[0].shape), tuple(args[2].shape)))
        return apply(*args)

    monkeypatch.setattr(native_gdn._GdnFunction, "apply", recorded)
    raw = inputs(B, T, H, HV, K, V, device="cuda")
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
         for i, t in enumerate(raw)]
    ref_x = [t.detach().double().requires_grad_() for t in x]
    o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=norm)
    assert calls == [((B * T, HV, 128), (B * T, HV, 128))] * ((V + 127) // 128)
    ref = gdn.gdn_reference(*ref_x, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(o)
    assert rel_rms(o, ref) < 0.02
    for name, a, e, tol in zip(("dq", "dk", "dv", "dg", "dbeta"), torch.autograd.grad(o, x, do),
                               torch.autograd.grad(ref, ref_x, do.double()), (0.04, 0.04, 0.04, 0.06, 0.06)):
        assert rel_rms(a, e) < tol, (name, rel_rms(a, e))


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("case", ["dense", "tail", "packed"])
@pytest.mark.parametrize("gmin,gmax", [(5.0, 5.0), (0.0, 5.0), (4.0, 5.0)])
@pytest.mark.parametrize("save_h", [False, True])
def test_native_gram_negative_decay(case, gmin, gmax, save_h):
    from flash_train.torch import chunk_gated_delta_rule

    T = {"dense": 128, "tail": 37, "packed": 140}[case]
    raw = inputs(1, T, 1, 2, 128, 128, device="cuda")
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32) for i, t in enumerate(raw)]
    x[3] = -(gmin + torch.rand_like(x[3]) * (gmax - gmin))
    x = [t.requires_grad_() for t in x]
    ref_x = [t.detach().double().requires_grad_() for t in x]
    cu = torch.tensor([0, 0, 70, 140], device="cuda", dtype=torch.int32) if case == "packed" else None
    kw = dict(cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    out, _ = chunk_gated_delta_rule(*x, disable_recompute=save_h, **kw)
    expected = gdn.gdn_reference(*ref_x, **kw)
    do = torch.randn_like(out)
    actual = [out, *torch.autograd.grad(out, x, do)]
    reference = [expected, *torch.autograd.grad(expected, ref_x, do.double())]
    for name, got, want, budget in zip(
        ("o", "dq", "dk", "dv", "dg", "dbeta"), actual, reference,
        (0.02, 0.04, 0.04, 0.04, 0.06, 0.06),
    ):
        assert torch.isfinite(got).all(), name
        assert rel_rms(got, want) < budget, (name, rel_rms(got, want))
