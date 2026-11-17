"""Copy the G2 device sources into a build dir with the three AMDGPU asm statements replaced."""
import pathlib, re, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
(dst / "g2").mkdir(parents=True, exist_ok=True)
REPL = [
    ('asm volatile("s_waitcnt lgkmcnt(0)\\n\\ts_barrier" ::: "memory");', "g2_fake::cta_barrier();"),
    # Wave-scope LDS ordering: lanes of a wave run in lockstep on the GPU, threads here do not.
    ('asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");', "g2_fake::wave_barrier();"),
    ('asm volatile("v_mov_b32 %0, %1" : "=v"(y) : "v"(x));', "y = x;"),
    # x86 would use aligned SSE moves for the 16-byte vector typedefs; GPU loads need no alignment.
    ("__attribute__((ext_vector_type(", "__attribute__((aligned(2), ext_vector_type("),
]
n = [0] * len(REPL)
files = [src / "g2_kernels.hip", src / "g2_runner.cpp", *sorted((src / "g2").iterdir())]
for f in files:
    s = f.read_text()
    for i, (a, b) in enumerate(REPL):
        n[i] += s.count(a)
        s = s.replace(a, b)
    out = dst / f.relative_to(src)
    out = out.with_suffix(".cpp") if out.suffix == ".hip" else out
    out.write_text(s)
assert n[0] == 1 and n[1] == 4 and n[2] == 1, n
