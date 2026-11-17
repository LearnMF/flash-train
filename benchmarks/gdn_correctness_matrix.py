"""Issue #29 full shapes: native output and every gradient against same-device FLA."""
import argparse
import gc
import json

import torch

from gdn_compare import error, evaluate, operator

SHAPES = {
    "small": (1, 64, 4, 4, 32, 32), "small2": (1, 128, 8, 8, 64, 64),
    "tiny2": (2, 64, 4, 4, 64, 64), "tail": (1, 65, 4, 8, 37, 33),
    "medium": (1, 256, 16, 32, 64, 64), "medium2": (1, 512, 16, 16, 128, 128),
    "baseline": (1, 1024, 32, 32, 128, 128), "v256": (1, 1024, 16, 16, 128, 256),
    "narrowK": (1, 2048, 32, 32, 64, 128), "narrowV": (1, 2048, 32, 32, 128, 64),
    "gqa4": (1, 2048, 8, 32, 128, 128), "llm2k": (1, 2048, 32, 32, 128, 128),
    "wide": (2, 2048, 16, 32, 128, 128), "bigbatch": (8, 512, 32, 32, 128, 128),
    "batch4": (4, 1024, 32, 32, 128, 128), "large": (1, 4096, 32, 32, 128, 128),
    "llm4k": (1, 4096, 32, 32, 128, 128), "xlong": (1, 8192, 32, 32, 128, 128),
    "llm8k": (1, 8192, 32, 32, 128, 128), "llm16k": (1, 16384, 32, 32, 128, 128),
}


def run(op, shape):
    B, T, H, HV, K, V = shape
    torch.manual_seed(29)

    def rand(*dims):
        return torch.randn(*dims, device="cuda")

    q, k = [(rand(B, T, H, K) * 0.3).bfloat16() for _ in range(2)]
    v = (rand(B, T, HV, V) * 0.3).bfloat16()
    if op == "gdn":
        g = torch.nn.functional.logsigmoid(rand(B, T, HV)) * 0.1
        xs = [q, k, v, g, rand(B, T, HV).sigmoid()]
        names = ["o", "dq", "dk", "dv", "dg", "dbeta"]
    else:
        g = (torch.nn.functional.logsigmoid(rand(B, T, HV, K)) * 0.1).bfloat16()
        xs = [q, k, v, g, rand(B, T, HV, K).sigmoid().bfloat16(), rand(B, T, HV, V).sigmoid().bfloat16()]
        names = ["o", "dq", "dk", "dv", "dg", "db", "dw"]
    do = rand(B, T, HV, V).bfloat16()
    actual = evaluate(operator(op, "flash_train"), xs, do)
    fla = operator(op, "fla")

    def reference(q, k, *rest, **kwargs):
        if op == "gdn2" and H != HV:
            q, k = (x.repeat_interleave(HV // H, dim=2) for x in (q, k))
        return fla(q, k, *rest, **kwargs)

    expected = evaluate(reference, xs, do)
    metrics = {n: error(a, b) for n, a, b in zip(names, actual, expected)}
    budgets = dict(o=0.02, dq=0.04, dk=0.04, dv=0.04, dg=0.06, dbeta=0.06, db=0.06, dw=0.06)
    return {"metrics": metrics, "passed": all(x["finite"] and x["relative_rms"] < budgets[n]
                                              for n, x in metrics.items()), "tolerances": budgets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["gdn", "gdn2"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    args = p.parse_args()
    failures = []
    for name in args.shapes:
        shape = SHAPES[name]
        try:
            row = run(args.op, shape)
        except Exception as exc:
            row = {"passed": False, "error": str(exc)}
        row.update(op=args.op, name=name, shape=shape)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(name, row["passed"], row.get("error", ""), flush=True)
        if not row["passed"]:
            failures.append(name)
        gc.collect()
        torch.cuda.empty_cache()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
