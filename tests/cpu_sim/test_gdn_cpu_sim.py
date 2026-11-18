"""GDN scalar-gate HIP kernels on the CPU stand-in runtime vs. the FP64 autograd reference.

Not collected by the default test run (needs clang++ >= 14 and minutes of CPU time):
    pytest tests/cpu_sim/test_gdn_cpu_sim.py
"""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gdn_ref", ROOT / "python/flash_train/torch/gdn.py")
gdn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gdn)

pytestmark = pytest.mark.skipif(shutil.which("clang++") is None, reason="needs clang++")


@pytest.fixture(scope="session")
def sim(tmp_path_factory):
    out = Path(os.environ.get("G2_SIM_BUILD") or tmp_path_factory.mktemp("g2sim"))
    if not (out / "sim").exists() or not os.environ.get("G2_SIM_REUSE"):
        subprocess.run([str(ROOT / "tests/cpu_sim/build.sh"), str(ROOT), str(out)], check=True)
    return out / "sim"


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    out = tmp_path_factory.mktemp("g2base")
    subprocess.run([str(ROOT / "tests/cpu_sim/build.sh"), os.environ["G2_SIM_BASELINE"], str(out)], check=True)
    return out / "sim"


def rel_rms(a, e):
    return ((a.double() - e).square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-9)).item()


def run(sim, d, T, H, cu, save_qn, save_h, seed=290, gmax=0.3, gmin=0.0):
    torch.manual_seed(seed)
    q, k = (torch.randn(1, T, H, 128) * 0.1 for _ in range(2))
    v = torch.randn(1, T, H, 128) * 0.3
    g = -(gmin + torch.rand(1, T, H) * (gmax - gmin))
    beta = torch.rand(1, T, H)
    do = torch.randn(1, T, H, 128)
    x = [t.to(torch.bfloat16) for t in (q, k, v)] + [g, beta]
    do = do.to(torch.bfloat16)
    for name, t in zip(("q", "k", "v", "g", "beta", "do"), x + [do]):
        t = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        t.numpy().tofile(d / f"{name}.bin")
    if cu is not None:
        np.asarray(cu, dtype=np.int32).tofile(d / "cu.bin")
    subprocess.run([str(sim), str(d), str(T), str(H), str(0 if cu is None else len(cu) - 1),
                    str(int(save_qn)), str(int(save_h))], check=True, timeout=3600)

    def load(name, bf16):
        a = torch.from_numpy(np.fromfile(d / f"{name}.bin", dtype=np.int16 if bf16 else np.float32))
        return (a.view(torch.bfloat16).reshape(1, T, H, 128) if bf16 else a.reshape(1, T, H)).double()

    got = {n: load(n, n in ("o", "dq", "dk", "dv")) for n in ("o", "dq", "dk", "dv", "dg", "dbeta")}
    rx = [t.double().requires_grad_() for t in x]
    ref = gdn.gdn_reference(*rx, use_qk_l2norm_in_kernel=True,
                            cu_seqlens=None if cu is None else torch.tensor(cu))
    grads = torch.autograd.grad(ref, rx, do.double())
    want = dict(zip(("dq", "dk", "dv", "dg", "dbeta"), grads), o=ref.detach())
    return {n: rel_rms(got[n], want[n]) for n in got}, got, want


# Dense 2 chunks, a single partial chunk (tail), multi-chunk + tail, packed with an empty sequence.
CASES = [
    ("dense", 128, 1, None),
    ("tail", 37, 2, None),
    ("dense_tail", 150, 1, None),
    ("packed", 140, 1, [0, 0, 70, 140]),
    ("heads", 70, 3, None),
]
TOL = dict(o=0.02, dq=0.04, dk=0.04, dv=0.04, dg=0.06, dbeta=0.06)


@pytest.mark.parametrize("name,T,H,cu", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("save", [(False, False), (True, True)], ids=["recompute", "saved"])
def test_sim_against_fp64(sim, tmp_path, name, T, H, cu, save):
    err, _, _ = run(sim, tmp_path, T, H, cu, *save)
    print(name, save, {k: f"{v:.2e}" for k, v in err.items()})
    for n, e in err.items():
        assert e < TOL[n], (n, e)


def test_scalar_dg_is_not_a_channel_slice(sim, tmp_path):
    """dg[t,h] must be the sum over all 128 key channels and the reverse scan over the chunk:
    a strong decay makes a dropped channel / lane or a wrong scan direction a first-order error."""
    err, got, want = run(sim, tmp_path, 128, 1, None, False, False, seed=7, gmax=2.0)
    assert err["dg"] < 0.06, err
    per_row = (got["dg"] - want["dg"]).abs().flatten() / want["dg"].abs().max()
    assert per_row.max() < 0.1, (per_row.argmax().item(), per_row.max().item())


@pytest.mark.skipif(not os.environ.get("G2_SIM_BASELINE"), reason="set G2_SIM_BASELINE=<baseline repo root>")
@pytest.mark.parametrize("name,T,H,cu", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("save", [(False, False), (True, True)], ids=["recompute", "saved"])
def test_same_bits_as_baseline(sim, base, tmp_path, name, T, H, cu, save):
    """A/B against another checkout: everything but dg must keep its bits (G2_SIM_DG_RTOL
    bounds dg, whose summation order a scalar-gate specialization may change)."""
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, new, _ = run(sim, tmp_path / "a", T, H, cu, *save)
    _, old, _ = run(base, tmp_path / "b", T, H, cu, *save)
    for n in ("o", "dq", "dk", "dv", "dbeta"):
        assert torch.equal(new[n], old[n]), n
    rtol = float(os.environ.get("G2_SIM_DG_RTOL", "2e-6"))
    assert (new["dg"] - old["dg"]).abs().max() <= rtol * old["dg"].abs().max()


# The safe-gate contract's extreme: every token decays by up to e^-5, so chunk-prefix factors span
# 2^-461 .. 1.  Any form that builds 2^(-prefix) (the reciprocal) overflows fp32 here; the outputs
# must stay finite and within the usual FP64 tolerances.
DECAY = [("all_-5", 5.0, 5.0), ("-5..0", 5.0, 0.0), ("-5..-4", 5.0, 4.0)]


@pytest.mark.parametrize("name,T,H,cu", [CASES[0], CASES[1], CASES[3]], ids=["dense", "tail", "packed"])
@pytest.mark.parametrize("dname,gmax,gmin", DECAY, ids=[d[0] for d in DECAY])
@pytest.mark.parametrize("save", [(False, False), (True, True)], ids=["recompute", "saved"])
def test_extreme_negative_decay(sim, tmp_path, name, T, H, cu, dname, gmax, gmin, save):
    err, got, _ = run(sim, tmp_path, T, H, cu, *save, seed=11, gmax=gmax, gmin=gmin)
    print(name, dname, save, {k: f"{v:.2e}" for k, v in err.items()})
    for n, x in got.items():
        assert torch.isfinite(x).all(), n
    for n, e in err.items():
        assert e < TOL[n], (n, e)
