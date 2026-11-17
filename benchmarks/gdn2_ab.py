# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Same-device end-to-end forward+backward A/B; not a runtime fallback.

python benchmarks/gdn2_ab.py --operator both --lengths 8192 16384 32768 65536 --tp 1 2 4 8
Records measured timings only. Run each VALUE_TILE candidate in a fresh process.
"""
import argparse
import json
import os
import statistics

import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--operator",choices=("ft","fla","both"),default="both")
    parser.add_argument("--lengths",type=int,nargs="+",default=[8192,16384,32768,65536])
    parser.add_argument("--tp",type=int,nargs="+",default=[1,2,4,8])
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--repeat",type=int,default=20)
    parser.add_argument("--output")
    args=parser.parse_args()
    from flash_train.torch import chunk_gdn2
    ops={"ft":chunk_gdn2}
    if args.operator!="ft":
        from fla.ops.gdn2 import chunk_gdn2 as fla_gdn2
        ops["fla"]=fla_gdn2
    if args.operator == "fla":
        ops.pop("ft")
    results=[]
    for length in args.lengths:
        for tp in args.tp:
            if tp not in (1, 2, 4, 8):
                raise ValueError("TP must divide global H=32")
            h=32//tp
            torch.manual_seed(2026)
            q,k,v=[torch.randn(1,length,h,128,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
            g=(-torch.rand_like(q)*0.1).requires_grad_()
            b=(torch.rand_like(q)*1.5).requires_grad_()
            w=torch.rand_like(v).requires_grad_()
            xs=(q,k,v,g,b,w)
            do=torch.randn_like(v)
            for name,op in ops.items():
                def run():
                    o,_=op(*xs,use_qk_l2norm_in_kernel=True)
                    torch.autograd.grad(o,xs,do)
                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(args.repeat):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    run()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                row=dict(operator=name,T=length,TP=tp,H=h,median_ms=statistics.median(times),
                         min_ms=min(times),max_ms=max(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         value_tile=os.getenv("FTRAIN_GDN2_VALUE_TILE","16"))
                results.append(row)
                print(json.dumps(row), flush=True)
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,hip=torch.version.hip,results=results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
