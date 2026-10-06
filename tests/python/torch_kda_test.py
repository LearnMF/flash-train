# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""flash_train.torch.kimi_delta_attention against its PyTorch FP64 reference.

Tolerances (relative RMS error vs the fp64 reference, the cuDNN-frontend PR #556 criteria):
o 2e-2; dq, dk, dv 4e-2; dg, dbeta, dA_log, ddt_bias 6e-2. The G2 kernels compute in bf16
operands with fp32 accumulation; observed errors are ~0.3-0.7%.
"""

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
    d_o = torch.randn(T, H, 128, device="cuda").bfloat16()
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
    d_o = torch.randn(512, 4, 128, device="cuda").bfloat16()
    a = run(ft.kimi_delta_attention, x, d_o)
    b = run(ft.kimi_delta_attention, x, d_o)
    assert torch.equal(a[0], b[0])
    for name in x:
        assert torch.equal(a[1][name], b[1][name]), name


def test_packed_equals_separate():
    """Sequences in one packed call are independent: bit-identical to separate varlen calls.
    (The dense single-sequence path -- no cu_seqlens, T % 64 == 0 -- is a different kernel
    instantiation and may differ in the last bits; it is checked against the tolerance.)"""
    lengths = [128, 64, 192]
    T, H = sum(lengths), 2
    x = make_inputs(T, H, True, seed=11)
    d_o = torch.randn(T, H, 128, device="cuda").bfloat16()
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
        od, _ = run(ft.kimi_delta_attention, part, d_o[s:s + n])
        assert rel_rms(od, op) < 1e-3
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
