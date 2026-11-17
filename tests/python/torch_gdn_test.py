# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU algebra tests run without the extension; HIP tests require gfx936."""

import contextlib
import importlib.util
import json
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


# FLA v0.5.2 tests/ops/test_gdn.py::test_chunk uses these relative RMS limits.
# Apply them strictly to BF16 here; upstream's corresponding matrix uses FP16.
FLA_TOL = {"o": 0.005, "dq": 0.008, "dk": 0.008, "dv": 0.008, "dg": 0.02, "dbeta": 0.02}


def assert_fla_close(name, actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = actual - expected
    abs_err = delta.abs().max().item()
    ratio = (delta.square().mean().sqrt() / (expected.square().mean().sqrt() + 1e-8)).item()
    print(json.dumps({"tensor": name, "shape": list(actual.shape), "max_abs": abs_err, "rel_rms": ratio}))
    assert abs_err <= 1e-6 or ratio < FLA_TOL[name], (name, abs_err, ratio, FLA_TOL[name])


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
    assert_fla_close("o", o, ref)
    for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta"), actual, expected):
        assert_fla_close(name, a, e)


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
    assert_fla_close("o", o, ref)
    for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta"), torch.autograd.grad(o, x, do),
                          torch.autograd.grad(ref, ref_x, do.double())):
        assert_fla_close(name, a, e)
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
    assert_fla_close("o", o, ref)
    for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta"), torch.autograd.grad(o, x, do),
                          torch.autograd.grad(ref, ref_x, do.double())):
        assert_fla_close(name, a, e)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("gate", ["weak", "zero"])
def test_same_inputs_fla_fp16_control(monkeypatch, norm, gate):
    _compare_native_and_fla(monkeypatch, 128, 128, norm, gate, torch.float16)


def _stress_inputs(K, V, norm, gate):
    torch.manual_seed(42)
    B, T, H, HV = 2, 65, 4, 8
    q, k = (torch.rand(B, T, H, K, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    if not norm:
        q, k = (torch.nn.functional.normalize(x.float(), dim=-1).bfloat16() for x in (q, k))
    v = torch.rand(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.rand(B, T, HV, device="cuda"))
    if gate == "weak":
        g = g / 10
    elif gate == "zero":
        g = torch.zeros_like(g)
    elif gate == "boundary":
        g = torch.full_like(g, -5)
    beta = torch.rand(B, T, HV, device="cuda").sigmoid()
    return q, k, v, g, beta


def _compare_native_and_fla(monkeypatch, K, V, norm, gate, fla_dtype):
    monkeypatch.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1")
    fla = pytest.importorskip("fla", minversion="0.5.2")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdn
    from flash_train.torch import chunk_gated_delta_rule
    from flash_train.torch import gdn as native_gdn

    assert fla.__version__ == "0.5.2", "the precision baseline must remain pinned to FLA 0.5.2"
    q, k, v, g, beta = _stress_inputs(K, V, norm, gate)
    if fla_dtype == torch.float16:
        q, k, v = (a.half().bfloat16() for a in (q, k, v))
    x = [a.requires_grad_() for a in (q, k, v, g, beta)]
    rx = [a.detach().double().requires_grad_() for a in x]
    # The control's inputs lie in the intersection of the two dtype ranges,
    # including tiny values that otherwise underflow on the cast to FP16.
    fx = [a.detach().to(fla_dtype if i < 3 else a.dtype).clone().requires_grad_() for i, a in enumerate(x)]
    assert all(torch.equal(a, b.to(a.dtype)) for a, b in zip(x, fx))
    calls, apply = [], native_gdn._GdnFunction.apply

    def recorded(*args):
        calls.append((args[0].shape[-1], args[2].shape[-1]))
        return apply(*args)

    monkeypatch.setattr(native_gdn._GdnFunction, "apply", recorded)
    monkeypatch.setattr(native_gdn, "gdn_reference", lambda *a, **kw: pytest.fail("reference fallback"))
    o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=norm, check_decay=True)
    assert calls == [(128, 128)] * ((V + 127) // 128)
    fo, _ = fla_gdn(*fx, use_qk_l2norm_in_kernel=norm)
    ref = gdn.gdn_reference(*rx, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(o)
    if fla_dtype == torch.float16:
        do = do.half().bfloat16()
    actual = (o, *torch.autograd.grad(o, x, do))
    assert torch.equal(do, do.to(fo.dtype).to(do.dtype))
    baseline = (fo, *torch.autograd.grad(fo, fx, do.to(fo.dtype)))
    expected = (ref, *torch.autograd.grad(ref, rx, do.double()))
    errors = []
    for label, values in (("native", actual), (f"fla_{fla_dtype}", baseline)):
        print(f"{label} vs FP64: K={K}, V={V}, norm={norm}, gate={gate}")
        for name, a, e in zip(FLA_TOL, values, expected):
            try:
                assert_fla_close(name, a, e)
            except AssertionError as error:
                errors.append(f"{label}/{name}: {error}")
    assert not errors, "\n".join(errors)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("decay", [-5.01, 0.01, float("nan"), float("inf")])
def test_native_rejects_unsupported_decay_when_checked(decay):
    from flash_train.torch import chunk_gated_delta_rule

    raw = inputs(1, 65, 4, 4, 128, 128, device="cuda")
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32) for i, t in enumerate(raw)]
    x[3].fill_(decay)
    with pytest.raises(ValueError, match=r"\[-5,0\]"):
        chunk_gated_delta_rule(*x, check_decay=True)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("K,V", ISSUE29_SHAPES + [(128, 128)])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("gate", ["normal", "weak", "zero", "boundary"])
def test_native_stress_against_fp64(monkeypatch, K, V, norm, gate):
    from flash_train.torch import chunk_gated_delta_rule
    from flash_train.torch import gdn as native_gdn

    x = [a.requires_grad_() for a in _stress_inputs(K, V, norm, gate)]
    rx = [a.detach().double().requires_grad_() for a in x]
    calls, apply = [], native_gdn._GdnFunction.apply

    def recorded(*args):
        calls.append((args[0].shape[-1], args[2].shape[-1]))
        return apply(*args)

    monkeypatch.setattr(native_gdn._GdnFunction, "apply", recorded)
    monkeypatch.setattr(native_gdn, "gdn_reference", lambda *a, **kw: pytest.fail("reference fallback"))
    o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=norm, check_decay=True)
    assert calls == [(128, 128)] * ((V + 127) // 128)
    ref = gdn.gdn_reference(*rx, use_qk_l2norm_in_kernel=norm)
    do = torch.randn_like(o)
    actual = (o, *torch.autograd.grad(o, x, do))
    expected = (ref, *torch.autograd.grad(ref, rx, do.double()))
    for name, a, e in zip(FLA_TOL, actual, expected):
        assert_fla_close(name, a, e)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("save_h", [False, True])
@pytest.mark.parametrize("save_qkn", [False, True])
def test_fp32_saved_tensors_and_recompute(norm, save_h, save_qkn):
    from flash_train.torch._flash_train_torch import gdn_forward, gdn_backward

    raw = inputs(1, 129, 4, 4, 128, 128, device="cuda")
    x = [t.detach().to(torch.bfloat16 if i < 3 else torch.float32)[0] for i, t in enumerate(raw)]
    q, k, v, g, beta = x
    g.zero_()
    beta.fill_(1)
    cu = torch.tensor([0, 0, 1, 65, 65, 129, 129], device="cuda", dtype=torch.int32)
    result = gdn_forward(*x, cu, 0.3, norm, save_qkn, save_h)
    o, qr, kr, aqk, akk, qn, kn, h = result
    assert aqk.numel() == akk.numel() == 0
    assert bool(h.numel()) == save_h
    if save_h:
        assert h.dtype == torch.float32
    assert bool(qn.numel()) == bool(kn.numel()) == (norm and save_qkn)
    if qn.numel():
        for normalized, original in ((qn, q), (kn, k)):
            assert normalized.dtype == torch.float32
            expected = original.double() * torch.rsqrt(original.double().square().sum(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(normalized.double(), expected, atol=1e-7, rtol=3e-7)
    do = torch.randn_like(o)
    grad = gdn_backward(do, None if qn.numel() else q, None if kn.numel() else k, v, g, beta,
                        cu, qr, kr, aqk, akk, qn, kn, h, 0.3, norm)
    rx = [a[None].double().requires_grad_() for a in x]
    ref = gdn.gdn_reference(
        *rx, scale=0.3, use_qk_l2norm_in_kernel=norm, cu_seqlens=cu)
    expected = (ref[0], *(a[0] for a in torch.autograd.grad(ref, rx, do[None].double())))
    for name, actual, value in zip(FLA_TOL, (o, *grad), expected):
        assert_fla_close(name, actual, value)
    assert all(a.dtype == torch.bfloat16 for a in grad[:3])


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("T", [1, 63, 64, 127, 129, 257])
@pytest.mark.parametrize("beta_value", [0.0, 1.0, 1.99])
def test_native_checkpoint_boundaries(T, beta_value):
    from flash_train.torch import chunk_gated_delta_rule

    raw = inputs(1, T, 4, 4, 37, 33, device="cuda")
    x = [a.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
         for i, a in enumerate(raw)]
    with torch.no_grad():
        x[3].zero_()
        x[4].fill_(beta_value)
    rx = [a.detach().double().requires_grad_() for a in x]
    do = torch.randn_like(x[2])
    ref = gdn.gdn_reference(*rx, use_qk_l2norm_in_kernel=True)
    expected = (ref, *torch.autograd.grad(ref, rx, do.double()))
    saved = None
    for save_h in (False, True):
        o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=True, disable_recompute=save_h)
        actual = (o, *torch.autograd.grad(o, x, do))
        for name, a, e in zip(FLA_TOL, actual, expected):
            assert_fla_close(name, a, e)
        if saved is not None:
            for a, b in zip(actual, saved):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        saved = actual


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("gate", [0.0, -0.03, -5.0])
def test_block_fp64_oracle_matches_token_recurrence(norm, gate):
    from gdn_fp64_oracle import chunk_reference

    x = inputs(2, 65, 2, 4, 7, 9)
    with torch.no_grad():
        x[3].fill_(gate)
    actual = chunk_reference(*x, norm=norm, scale=0.3)
    expected = gdn.gdn_reference(*x, use_qk_l2norm_in_kernel=norm, scale=0.3)
    do = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    for a, b in zip(torch.autograd.grad(actual, x, do), torch.autograd.grad(expected, x, do)):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("value_scale", [1e-8, 1e6])
def test_native_retains_bf16_exponent_range(value_scale):
    from flash_train.torch import chunk_gated_delta_rule

    raw = inputs(1, 65, 4, 4, 128, 128, device="cuda")
    x = [a.detach().to(torch.bfloat16 if i < 3 else torch.float32).requires_grad_()
         for i, a in enumerate(raw)]
    with torch.no_grad():
        x[2].mul_(value_scale)
        x[3].zero_()
    rx = [a.detach().double().requires_grad_() for a in x]
    o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=True)
    ref = gdn.gdn_reference(*rx, use_qk_l2norm_in_kernel=True)
    do = torch.randn_like(o)
    for name, a, e in zip(FLA_TOL, (o, *torch.autograd.grad(o, x, do)),
                          (ref, *torch.autograd.grad(ref, rx, do.double()))):
        assert_fla_close(name, a, e)


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("legacy", ["qn", "h"])
def test_rejects_legacy_bf16_saved_tensors(legacy):
    from flash_train.torch._flash_train_torch import gdn_forward, gdn_backward

    x = [a.detach().to(torch.bfloat16 if i < 3 else torch.float32)[0]
         for i, a in enumerate(inputs(1, 65, 4, 4, 128, 128, device="cuda"))]
    o, qr, kr, aqk, akk, qn, kn, h = gdn_forward(*x, None, 0.1, True, True, True)
    if legacy == "qn":
        qn, kn = qn.bfloat16(), kn.bfloat16()
    else:
        h = h.bfloat16()
    with pytest.raises(RuntimeError, match="(unsupported|Unsupported|applicable)"):
        gdn_backward(torch.randn_like(o), None, None, x[2], x[3], x[4], None,
                     qr, kr, aqk, akk, qn, kn, h, 0.1, True)


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
    assert_fla_close("o", torch.cat(outputs, dim=2), full)
    for name, values, expected in zip(("dq", "dk", "dv", "dg", "dbeta"), gradients, full_grads):
        assert_fla_close(name, torch.cat(values, dim=2), expected)
