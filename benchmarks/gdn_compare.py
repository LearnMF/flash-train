"""GDN/GDN-2 correctness and same-device timing against FLA.

Run separate processes in ABBA order; aggregate process medians, not iterations.
TP means local head partitioning only, not communication or sequence partitioning.
"""

import argparse
import gc
import importlib
import json
import os
import socket
import statistics
import time

import torch


def operator(op, impl):
    if impl == "fla":
        module = "fla.ops.gated_delta_rule" if op == "gdn" else "fla.ops.gdn2"
    else:
        module = "flash_train.torch"
    return getattr(importlib.import_module(module),
                   "chunk_gated_delta_rule" if op == "gdn" else "chunk_gdn2")


def inputs(op, length, heads, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    shape = (1, length, heads, 128)
    q, k, v = [(rand(*shape) * 0.3).bfloat16() for _ in range(3)]
    if op == "gdn":
        g = torch.nn.functional.logsigmoid(rand(1, length, heads)) * 0.1
        gates = [g, rand(1, length, heads).sigmoid()]
    else:
        g = (torch.nn.functional.logsigmoid(rand(*shape)) * 0.1).bfloat16()
        gates = [g, rand(*shape).sigmoid().bfloat16(), rand(*shape).sigmoid().bfloat16()]
    return [q, k, v, *gates], rand(*shape).bfloat16()


def evaluate(fn, data, do):
    leaves = [x.detach().requires_grad_(True) for x in data]
    out, _ = fn(*leaves, use_qk_l2norm_in_kernel=True)
    grads = torch.autograd.grad(out, leaves, do)
    return [out.detach(), *grads]


def error(actual, reference):
    actual, reference = actual.double(), reference.double()
    diff = actual - reference
    return {"finite": bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
            "relative_rms": float(diff.square().mean().sqrt() /
                                  (reference.square().mean().sqrt() + 1e-8)),
            "max_abs": float(diff.abs().max())}


def tolerances(op):
    if op == "gdn":
        return dict(o=0.005, dq=0.008, dk=0.008, dv=0.008, dg=0.02, dbeta=0.02)
    return dict(o=0.005, dq=0.01, dk=0.01, dv=0.01, dg=0.015, db=0.015, dw=0.015)


def timed(fn, reps, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / reps)
    return {"median_ms": statistics.median(timings), "samples_ms": timings}


def kernel_profile(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    totals = {}
    for event in prof.events():
        if event.device_type.name == "CUDA":
            totals[event.name] = totals.get(event.name, 0.0) + event.device_time / 3000
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def run_case(args, length, tp):
    data, do = inputs(args.op, length, 32 // tp, args.seed)
    if args.check:
        actual = evaluate(operator(args.op, "flash_train"), data, do)
        reference = evaluate(operator(args.op, "fla"), data, do)
        names = ["o", "dq", "dk", "dv", "dg", "dbeta"] if args.op == "gdn" else [
            "o", "dq", "dk", "dv", "dg", "db", "dw"]
        metrics = {name: error(a, b) for name, a, b in zip(names, actual, reference)}
        # FLA BF16 comparison is a diagnostic, not an independent accuracy oracle.
        # Native acceptance is tests/python/gdn_precision_matrix.py against FP64.
        budgets = tolerances(args.op)
        passed = all(m["finite"] and (m["max_abs"] <= 1e-6 or m["relative_rms"] < budgets[n])
                     for n, m in metrics.items())
        return {"metrics": metrics, "passed": passed, "tolerances": budgets}
    fn = operator(args.op, args.impl)

    def fwd():
        leaves = [x.detach().requires_grad_(True) for x in data]
        fn(*leaves, use_qk_l2norm_in_kernel=True)

    def fwd_bwd():
        evaluate(fn, data, do)

    torch.cuda.reset_peak_memory_stats()
    result = {"fwd": timed(fwd, args.reps, args.samples),
              "fwd_bwd": timed(fwd_bwd, args.reps, args.samples)}
    result["bwd_estimate_ms"] = result["fwd_bwd"]["median_ms"] - result["fwd"]["median_ms"]
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if args.profile:
        result["fwd_kernels_ms"] = kernel_profile(fwd)
        result["fwd_bwd_kernels_ms"] = kernel_profile(fwd_bwd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["gdn", "gdn2"], required=True)
    parser.add_argument("--impl", choices=["fla", "flash_train"], default="flash_train")
    parser.add_argument("--tp", type=int, nargs="+", choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    parser.add_argument("--sequences", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check", action="store_true", help="strict same-input FLA diagnostic, not FP64 acceptance")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.samples < 1 or min(args.sequences) < 1:
        parser.error("sequence lengths, reps and samples must be positive")
    import fla
    import triton
    prop = torch.cuda.get_device_properties(0)
    meta = {"host": socket.gethostname(), "device": prop.name,
            "uuid": str(getattr(prop, "uuid", "")), "arch": prop.gcnArchName,
            "torch": torch.__version__, "hip": torch.version.hip,
            "fla": fla.__version__, "triton": triton.__version__,
            "build": os.getenv("FT_BUILD", "unrecorded"), "pid": os.getpid(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}
    failures = 0
    with open(args.out, "x") as stream:
        stream.write(json.dumps({"meta": meta}) + "\n")
        stream.flush()
        for tp in args.tp:
            for length in args.sequences:
                gc.collect()
                torch.cuda.empty_cache()
                row = {"op": args.op, "impl": args.impl, "tp": tp, "B": 1,
                       "S": length, "H": 32 // tp, "HV": 32 // tp, "K": 128, "V": 128}
                try:
                    row.update(run_case(args, length, tp))
                    failures += int(not row.get("passed", True))
                except Exception as exc:
                    row["error"] = repr(exc)
                    failures += 1
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
