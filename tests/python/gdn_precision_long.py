# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Strict long-sequence accuracy acceptance, not a performance benchmark.

Example: HIP_VISIBLE_DEVICES=1 python tests/python/gdn_precision_long.py --length 8192 --heads 32
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from gdn_fp64_oracle import chunk_reference
from torch_gdn_test import FLA_TOL, assert_fla_close


def run(length, heads, gate, seed, distribution, save_h):
    from flash_train.torch import chunk_gated_delta_rule

    torch.manual_seed(seed)
    rand = torch.rand if distribution == "positive" else torch.randn
    q, k, v = (rand(1, length, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    g = F.logsigmoid(torch.rand(1, length, heads, device="cuda"))
    if gate == "weak":
        g = g / 10
    elif gate == "zero":
        g = torch.zeros_like(g)
    elif gate == "mixed":
        g[:, :, 0::4] = 0
        g[:, :, 1::4] /= 10
        g[:, :, 2::4] = -5
    beta = torch.rand_like(g).sigmoid()
    x = [a.requires_grad_() for a in (q, k, v, g, beta)]
    do = torch.randn_like(v)
    start = time.monotonic()
    o, _ = chunk_gated_delta_rule(*x, use_qk_l2norm_in_kernel=True, disable_recompute=save_h)
    actual = [a.detach() for a in (o, *torch.autograd.grad(o, x, do))]
    torch.cuda.synchronize()
    print(json.dumps(dict(stage="native", seconds=time.monotonic() - start)), flush=True)
    del o
    rx = [a.detach().double().requires_grad_() for a in x]
    ref = chunk_reference(*rx)
    expected = (ref, *torch.autograd.grad(ref, rx, do.double()))
    torch.cuda.synchronize()
    print(json.dumps(dict(stage="fp64", seconds=time.monotonic() - start)), flush=True)
    errors = []
    for name, a, e in zip(FLA_TOL, actual, expected):
        try:
            assert_fla_close(name, a, e)
        except AssertionError as error:
            errors.append(str(error))
    print(json.dumps(dict(length=length, heads=heads, gate=gate, seed=seed, distribution=distribution,
                          save_h=save_h, passed=not errors, errors=errors)), flush=True)
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--gate", choices=["normal", "weak", "zero", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution", choices=["positive", "normal"], default="positive")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raise SystemExit(0 if run(args.length, args.heads, args.gate, args.seed, args.distribution,
                             not args.recompute) else 1)
