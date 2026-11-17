# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Strict FP64 correctness for issue #29 and the complete TP/length matrix; no timing."""

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys

import torch

from gdn_fp64_oracle import chunk_reference as gdn_reference
from gdn2_fp64_oracle import chunk_reference as gdn2_reference
from torch_gdn_test import FLA_TOL
from torch_gdn2_test import LIMITS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from gdn_correctness_matrix import SHAPES


def run(op, shape, gate, distribution, save_h, seed):
    batch, length, heads, value_heads, keys, values = shape
    torch.manual_seed(seed)
    rand = torch.rand if distribution == "positive" else torch.randn
    q, k = [rand(batch, length, heads, keys, device="cuda").bfloat16() for _ in range(2)]
    v = (rand(batch, length, value_heads, values, device="cuda") * 0.5).bfloat16()
    gate_shape = (batch, length, value_heads) + ((keys,) if op == "gdn2" else ())
    g = -torch.empty(gate_shape, device="cuda").uniform_(0.1, 5.0)
    if gate == "weak":
        g *= 0.01
    elif gate == "zero":
        g.zero_()
    elif gate == "boundary":
        g.fill_(-5)
    elif gate == "mixed":
        g[:, :, 0::4] = 0
        g[:, :, 1::4] *= 0.01
        g[:, :, 2::4] = -5
    if op == "gdn":
        data = [q, k, v, g, torch.rand_like(g).sigmoid()]
        fn_name, ref, limits = "chunk_gated_delta_rule", gdn_reference, FLA_TOL
    else:
        data = [q, k, v, g.bfloat16(), torch.rand_like(g).bfloat16(), torch.rand_like(v)]
        fn_name, ref, limits = "chunk_gdn2", gdn2_reference, LIMITS
    fn = getattr(importlib.import_module("flash_train.torch"), fn_name)
    data = [x.requires_grad_() for x in data]
    do = torch.randn_like(v)
    out, _ = fn(*data, use_qk_l2norm_in_kernel=True, disable_recompute=save_h)
    actual = [x.detach() for x in (out, *torch.autograd.grad(out, data, do))]
    del out
    gc.collect()
    torch.cuda.empty_cache()
    sums = {name: dict(delta=0.0, reference=0.0, count=0, max_abs=0.0, finite=True) for name in limits}
    # The native invocation has the full shape; only the independent oracle is sharded.
    ratio = value_heads // heads
    width = max(1, 4 // ratio)
    for first in range(0, heads, width):
        last = min(heads, first + width)
        slices = [slice(first, last)] * 2 + [slice(first * ratio, last * ratio)] * (len(data) - 2)
        leaves = [x.detach()[:, :, sl].double().requires_grad_() for x, sl in zip(data, slices)]
        expected = ref(*leaves)
        dout = do[:, :, first * ratio:last * ratio].double()
        references = (expected, *torch.autograd.grad(expected, leaves, dout))
        for name, a, b, sl in zip(limits, actual, references, [slices[2], *slices]):
            a = a[:, :, sl].double()
            delta = a - b
            stats = sums[name]
            stats["finite"] &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            stats["delta"] += delta.square().sum().item()
            stats["reference"] += b.square().sum().item()
            stats["count"] += b.numel()
            stats["max_abs"] = max(stats["max_abs"], delta.abs().max().item())
        del leaves, expected, references, dout, a, b, delta
    metrics = {}
    for name, s in sums.items():
        relative = (s["delta"] / s["count"]) ** 0.5 / ((s["reference"] / s["count"]) ** 0.5 + 1e-8)
        passed = s["finite"] and (s["max_abs"] <= 1e-6 or relative < limits[name])
        metrics[name] = dict(relative_rms=relative, max_abs=s["max_abs"], finite=s["finite"], passed=passed)
    return dict(passed=all(m["passed"] for m in metrics.values()), metrics=metrics, limits=limits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=["gdn", "gdn2"], required=True)
    p.add_argument("--issue29", action="store_true")
    p.add_argument("--shapes", nargs="+", choices=list(SHAPES))
    p.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768, 65536])
    p.add_argument("--tp", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    p.add_argument("--gate", choices=["normal", "weak", "zero", "boundary", "mixed"], default="mixed")
    p.add_argument("--distribution", choices=["positive", "normal"], default="positive")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    assert "gfx936" in props.gcnArchName
    cases = [(name, SHAPES[name]) for name in args.shapes or SHAPES] if args.issue29 else [
        (f"tp{tp}-s{length}", (1, length, 32 // tp, 32 // tp, 128, 128))
        for length in args.lengths for tp in args.tp]
    failures = []
    with open(args.out, "x") as log:
        log.write(json.dumps(dict(meta=vars(args), gpu=str(props), uuid=str(props.uuid))) + "\n")
        for name, shape in cases:
            print(json.dumps(dict(start=name, shape=shape)), flush=True)
            try:
                row = run(args.op, shape, args.gate, args.distribution, not args.recompute, args.seed)
            except Exception as exc:
                row = dict(passed=False, error=str(exc))
            row.update(name=name, shape=shape, op=args.op, gate=args.gate, save_h=not args.recompute)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if not row["passed"]:
                failures.append(name)
            gc.collect()
            torch.cuda.empty_cache()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
