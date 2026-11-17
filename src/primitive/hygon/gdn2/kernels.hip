// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "flash_train/error.hpp"
#include "../kda/g2/g2_fi2.cuh"
#include "flash_train/primitive/hygon/gdn2/runner.hpp"

namespace ftrain::gdn2 {
namespace {
using g2::bf16;
using g2::f2;
using g2::f4;
__device__ float read(const void* p, size_t i, bool fp32 = false) {
    return fp32 ? static_cast<const float*>(p)[i] : float(static_cast<const bf16*>(p)[i]);
}
__device__ void write(void* p, size_t i, float x, bool fp32 = false) {
    if (fp32)
        static_cast<float*>(p)[i] = x;
    else
        static_cast<bf16*>(p)[i] = __float2bfloat16(x);
}
struct Matrix {
    const float* p;
    int rs, cs;
    __device__ float operator()(int i, int j) const { return p[size_t(i) * rs + size_t(j) * cs]; }
};
// Same gfx936 lane mapping as KDA's fi2, with FP32 inputs/accumulators.
__device__ f4 dot16(Matrix a, Matrix b, int m, int n, int depth, int row, int col) {
    const int lane = threadIdx.x & 63, r = row + (lane & 15), c = col + (lane & 15), z = 2 * (lane >> 4);
    f4 acc = {0, 0, 0, 0};
    for (int k = 0; k < depth; k += 8) {
        f2 av = {0, 0}, bv = {0, 0};
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            if (r < m && k + z + e < depth) av[e] = a(r, k + z + e);
            if (c < n && k + z + e < depth) bv[e] = b(k + z + e, c);
        }
        acc = g2::mmac32(av, bv, acc);
    }
    return acc;
}
__device__ void gemm(Matrix a, Matrix b, float* dst, int m, int n, int depth, int ld, float alpha = 1, float beta = 0) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6, tiles_n = (n + 15) / 16;
    for (int tile = wave; tile < ((m + 15) / 16) * tiles_n; tile += 4) {
        const int row = 16 * (tile / tiles_n), col = 16 * (tile % tiles_n);
        const f4 acc = dot16(a, b, m, n, depth, row, col);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int r = row + (lane & 15), c = col + g2::c_col(lane, e);
            if (r < m && c < n) dst[r * ld + c] = alpha * acc[e] + (beta ? beta * dst[r * ld + c] : 0);
        }
    }
    __syncthreads();
}
__device__ size_t token(const Shape& s, int block, int i, bool qk = false) {
    const int c = block % s.chunks(), bh = block / s.chunks(), h = bh % s.HV, batch = bh / s.HV;
    return (size_t(batch) * s.T + c * C + i) * (qk ? s.H : s.HV) + (qk ? h / (s.HV / s.H) : h);
}
__device__ int length(const Shape& s, int block) { return min(C, s.T - (block % s.chunks()) * C); }

__global__ void normalize(Shape s, Args p, Layout l) {
    const int block = blockIdx.x, lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int len = length(s, block);
    float* base   = static_cast<float*>(p.cache) + size_t(block) * l.size;
    for (int i = wave; i < C; i += 4) {
        float q0 = 0, q1 = 0, k0 = 0, k1 = 0;
        const size_t off = token(s, block, i, true) * s.K;
        if (i < len && lane < s.K) {
            q0 = read(p.q, off + lane);
            k0 = read(p.k, off + lane);
        }
        if (i < len && lane + 64 < s.K) {
            q1 = read(p.q, off + lane + 64);
            k1 = read(p.k, off + lane + 64);
        }
        float qq = q0 * q0 + q1 * q1, kk = k0 * k0 + k1 * k1;
#pragma unroll
        for (int off = 32; off; off /= 2) {
            qq += __shfl_xor(qq, off, 64);
            kk += __shfl_xor(kk, off, 64);
        }
        const float qr = p.norm ? rsqrtf(qq + 1e-6f) : 1.f;
        const float kr = p.norm ? rsqrtf(kk + 1e-6f) : 1.f;
        if (lane == 0) {
            base[l.norm + 2 * i]     = qr;
            base[l.norm + 2 * i + 1] = kr;
        }
        if (lane < s.K) {
            base[l.q + i * s.K + lane] = q0 * qr;
            base[l.k + i * s.K + lane] = k0 * kr;
        }
        if (lane + 64 < s.K) {
            base[l.q + i * s.K + lane + 64] = q1 * qr;
            base[l.k + i * s.K + lane + 64] = k1 * kr;
        }
    }
}
__global__ void prepare(Shape s, Args p, Layout l) {
    const int block = blockIdx.x, tid = threadIdx.x, len = length(s, block);
    float* base = static_cast<float*>(p.cache) + size_t(block) * l.size;
    for (int d = tid; d < s.K; d += 256) {
        float sum = 0;
        for (int i = 0; i < C; ++i) {
            if (i < len) sum += read(p.g, token(s, block, i) * s.K + d, p.gf32);
            base[l.g + i * s.K + d] = sum;
        }
        for (int i = 0; i < C; ++i) {
            const int ix   = i * s.K + d;
            float eg       = __expf(base[l.g + ix]);
            base[l.Q + ix] = p.scale * base[l.q + ix] * eg;
            base[l.W + ix] = i < len ? read(p.b, token(s, block, i) * s.K + d, p.bf32) * base[l.k + ix] * eg : 0;
            base[l.E + ix] = base[l.k + ix] * __expf(sum - base[l.g + ix]);
        }
    }
    for (int ix = tid; ix < C * s.V; ix += 256) {
        int i = ix / s.V, d = ix % s.V;
        base[l.U + ix] =
            i < len ? read(p.v, token(s, block, i) * s.V + d) * read(p.w, token(s, block, i) * s.V + d, p.wf32) : 0;
    }
}
// Off-diagonal factors are nonpositive. Within a diagonal tile, a midpoint
// reference bounds each exponent by 40 for log gates in [-5, 0].
__global__ void intra(Shape s, Args p, Layout l) {
    const int block = blockIdx.x, ib = blockIdx.y, jb = threadIdx.x >> 6, lane = threadIdx.x & 63;
    float* base = static_cast<float*>(p.cache) + size_t(block) * l.size;
    if (jb <= ib) {
        const int i = 16 * ib + (lane & 15), j = 16 * jb + (lane & 15), z = 2 * (lane >> 4);
        f4 a = {0, 0, 0, 0}, o = {0, 0, 0, 0};
        for (int k = 0; k < s.K; k += 8) {
            f2 av = {0, 0}, qv = {0, 0}, bv = {0, 0};
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                int d = k + z + e;
                if (d < s.K) {
                    float ref = base[l.g + (16 * ib + (jb == ib ? 7 : -1)) * s.K + d];
                    float ea = __expf(base[l.g + i * s.K + d] - ref), eb = __expf(ref - base[l.g + j * s.K + d]);
                    qv[e] = p.scale * base[l.q + i * s.K + d] * ea;
                    av[e] = i < length(s, block)
                                ? read(p.b, token(s, block, i) * s.K + d, p.bf32) * base[l.k + i * s.K + d] * ea
                                : 0;
                    bv[e] = base[l.k + j * s.K + d] * eb;
                }
            }
            a = g2::mmac32(av, bv, a);
            o = g2::mmac32(qv, bv, o);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int ix         = i * C + 16 * jb + g2::c_col(lane, e);
            const int jcol = 16 * jb + g2::c_col(lane, e);
            base[l.M + ix] = i > jcol ? a[e] : 0;
            base[l.P + ix] = i >= jcol ? o[e] : 0;
        }
    } else {
        for (int pair = lane; pair < 256; pair += 64) {
            int i = 16 * ib + pair / 16, j = 16 * jb + pair % 16;
            float a = 0, o = 0;
            if (jb == ib && i < length(s, block) && j <= i)
                for (int d = 0; d < s.K; ++d) {
                    float kj  = base[l.k + j * s.K + d] * __expf(base[l.g + i * s.K + d] - base[l.g + j * s.K + d]);
                    o        += p.scale * base[l.q + i * s.K + d] * kj;
                    if (j < i) a += read(p.b, token(s, block, i) * s.K + d, p.bf32) * base[l.k + i * s.K + d] * kj;
                }
            base[l.P + i * C + j] = o;
            base[l.M + i * C + j] = a;
        }
    }
}
// Block forward substitution avoids cancellation between large alternating powers.
__global__ void wy(Shape s, Args p, Layout l) {
    const int block = blockIdx.x, tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row = lane & 15, quarter = lane >> 4;
    float* base = static_cast<float*>(p.cache) + size_t(block) * l.size;
    __shared__ float inverse[C * 68], tmp[C * 32];
    for (int ix = tid; ix < C * 68; ix += 256) inverse[ix] = 0;
    f4 raw = {}, diagonal = {};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int col = g2::c_col(lane, e);
        raw[e]        = col < row ? -base[l.M + (16 * wave + row) * C + 16 * wave + col] : 0;
    }
    diagonal = raw;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float coefficient = __shfl(raw[j >> 2], 16 * (j & 3) + row, 64);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float pivot = __shfl(diagonal[e], 16 * quarter + j, 64);
            if (row > j) diagonal[e] = fmaf(coefficient, pivot, diagonal[e]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int col                                     = g2::c_col(lane, e);
        inverse[(16 * wave + row) * 68 + 16 * wave + col] = diagonal[e] + float(col == row);
    }
    __syncthreads();
    for (int distance = 1; distance < 4; ++distance) {
        const int j = wave - distance;
        f4 sum      = {};
        if (j >= 0) {
            for (int k = j; k < wave; ++k)
                sum += dot16({base + l.M + 16 * wave * C + 16 * k, C, 1}, {inverse + 16 * k * 68 + 16 * j, 68, 1}, 16,
                             16, 16, 0, 0);
#pragma unroll
            for (int e = 0; e < 4; ++e) tmp[wave * 320 + row * 20 + g2::c_col(lane, e)] = sum[e];
        }
        __syncthreads();
        if (j >= 0) {
            const f4 result =
                dot16({inverse + 16 * wave * 68 + 16 * wave, 68, 1}, {tmp + wave * 320, 20, 1}, 16, 16, 16, 0, 0);
#pragma unroll
            for (int e = 0; e < 4; ++e) inverse[(16 * wave + row) * 68 + 16 * j + g2::c_col(lane, e)] = -result[e];
        }
        __syncthreads();
    }
    for (int ix = tid; ix < C * C; ix += 256) base[l.M + ix] = inverse[(ix / C) * 68 + ix % C];
    for (int kind = 0; kind < 2; ++kind) {
        const int width = kind ? s.V : s.K;
        float* rhs      = base + (kind ? l.U : l.W);
        for (int col = 0; col < width; col += 32) {
            const int n = min(32, width - col);
            gemm({inverse, 68, 1}, {rhs + col, width, 1}, tmp, C, n, C, 32);
            for (int ix = tid; ix < C * 32; ix += 256)
                if (ix % 32 < n) rhs[(ix / 32) * width + col + ix % 32] = tmp[ix];
            __syncthreads();
        }
    }
}
__device__ void stage(float* dst, const float* src, int rows, int cols) {
    for (int ix = threadIdx.x; ix < rows * cols; ix += 256) dst[(ix / cols) * 132 + ix % cols] = src[ix];
    __syncthreads();
}
template<int VS, bool CACHE>
__global__ void recurrence(Shape s, Args p, Layout l) {
    const int bh = blockIdx.x, vb = blockIdx.y * VS, tid = threadIdx.x;
    __shared__ float state[128 * VS], x[C * VS], out[C * VS], operand[CACHE ? C * 132 : 1];
    for (int ix = tid; ix < s.K * VS; ix += 256) state[ix] = 0;
    __syncthreads();
    for (int c = 0; c < s.chunks(); ++c) {
        int block   = bh * s.chunks() + c;
        float* base = static_cast<float*>(p.cache) + size_t(block) * l.size;
        for (int ix = tid; ix < s.K * VS; ix += 256)
            if (vb + ix % VS < s.V) base[l.S + (ix / VS) * s.V + vb + ix % VS] = state[ix];
        for (int ix = tid; ix < C * VS; ix += 256)
            x[ix] = vb + ix % VS < s.V ? base[l.U + (ix / VS) * s.V + vb + ix % VS] : 0;
        __syncthreads();
        if constexpr (CACHE) stage(operand, base + l.W, C, s.K);
        gemm({CACHE ? operand : base + l.W, CACHE ? 132 : s.K, 1}, {state, VS, 1}, x, C, VS, s.K, VS, -1, 1);
        for (int ix = tid; ix < C * VS; ix += 256)
            if (vb + ix % VS < s.V) base[l.X + (ix / VS) * s.V + vb + ix % VS] = x[ix];
        if constexpr (CACHE) stage(operand, base + l.Q, C, s.K);
        gemm({CACHE ? operand : base + l.Q, CACHE ? 132 : s.K, 1}, {state, VS, 1}, out, C, VS, s.K, VS);
        if constexpr (CACHE) stage(operand, base + l.P, C, C);
        gemm({CACHE ? operand : base + l.P, CACHE ? 132 : C, 1}, {x, VS, 1}, out, C, VS, C, VS, 1, 1);
        for (int ix = tid; ix < C * VS; ix += 256)
            if (ix / VS < length(s, block) && vb + ix % VS < s.V)
                write(p.out, token(s, block, ix / VS) * s.V + vb + ix % VS, out[ix]);
        for (int ix = tid; ix < s.K * VS; ix += 256) state[ix] *= __expf(base[l.g + (C - 1) * s.K + ix / VS]);
        __syncthreads();
        if constexpr (CACHE) stage(operand, base + l.E, C, s.K);
        gemm({CACHE ? operand : base + l.E, 1, CACHE ? 132 : s.K}, {x, VS, 1}, state, s.K, VS, C, VS, 1, 1);
    }
}
template<int VS, bool CACHE>
__global__ void reverse(Shape s, Args p, Layout l, GradLayout gl, float* grad) {
    const int bh = blockIdx.x, vb = blockIdx.y * VS, tid = threadIdx.x;
    __shared__ float dh[128 * VS], dx[C * VS], dout[C * VS], y[C * VS], operand[CACHE ? C * 132 : 1];
    for (int ix = tid; ix < s.K * VS; ix += 256) dh[ix] = 0;
    __syncthreads();
    for (int c = s.chunks() - 1; c >= 0; --c) {
        int block         = bh * s.chunks() + c;
        const float* base = static_cast<const float*>(p.cache) + size_t(block) * l.size;
        float* dst        = grad + size_t(block) * gl.size;
        for (int ix = tid; ix < s.K * VS; ix += 256)
            if (vb + ix % VS < s.V) dst[gl.dS + (ix / VS) * s.V + vb + ix % VS] = dh[ix];
        for (int ix = tid; ix < C * VS; ix += 256)
            dout[ix] = (ix / VS < length(s, block) && vb + ix % VS < s.V)
                           ? read(p.dout, token(s, block, ix / VS) * s.V + vb + ix % VS)
                           : 0;
        __syncthreads();
        if constexpr (CACHE) stage(operand, base + l.E, C, s.K);
        gemm({CACHE ? operand : base + l.E, CACHE ? 132 : s.K, 1}, {dh, VS, 1}, dx, C, VS, s.K, VS);
        if constexpr (CACHE) stage(operand, base + l.P, C, C);
        gemm({CACHE ? operand : base + l.P, 1, CACHE ? 132 : C}, {dout, VS, 1}, dx, C, VS, C, VS, 1, 1);
        if constexpr (CACHE) stage(operand, base + l.M, C, C);
        gemm({CACHE ? operand : base + l.M, 1, CACHE ? 132 : C}, {dx, VS, 1}, y, C, VS, C, VS);
        for (int ix = tid; ix < C * VS; ix += 256)
            if (vb + ix % VS < s.V) {
                int i = ix / VS, v = vb + ix % VS;
                dst[gl.Y + i * s.V + v] = y[ix];
                if (i < length(s, block)) {
                    size_t o = token(s, block, i) * s.V + v;
                    write(p.dv, o, y[ix] * read(p.w, o, p.wf32));
                    write(p.dw, o, y[ix] * read(p.v, o), p.wf32);
                }
            }
        for (int ix = tid; ix < s.K * VS; ix += 256) dh[ix] *= __expf(base[l.g + (C - 1) * s.K + ix / VS]);
        __syncthreads();
        if constexpr (CACHE) stage(operand, base + l.Q, C, s.K);
        gemm({CACHE ? operand : base + l.Q, 1, CACHE ? 132 : s.K}, {dout, VS, 1}, dh, s.K, VS, C, VS, 1, 1);
        if constexpr (CACHE) stage(operand, base + l.W, C, s.K);
        gemm({CACHE ? operand : base + l.W, 1, CACHE ? 132 : s.K}, {dx, VS, 1}, dh, s.K, VS, C, VS, -1, 1);
    }
}
// Each CTA owns one 32x32 output tile; both operands are loaded once.
__global__ void products(Shape s, Args p, Layout l, GradLayout gl, float* grad) {
    const int block = blockIdx.x, tid = threadIdx.x, kind = blockIdx.y;
    const int width = kind < 3 ? s.K : C, nt = (width + 31) / 32;
    const int row = (blockIdx.z / nt) * 32, col = (blockIdx.z % nt) * 32;
    if (row >= C) return;
    const float* base = static_cast<const float*>(p.cache) + size_t(block) * l.size;
    float* dst        = grad + size_t(block) * gl.size;
    __shared__ float a[32 * 128], b[32 * 128];
    for (int ix = tid; ix < 32 * s.V; ix += 256) {
        const int r = ix / s.V, v = ix % s.V, i = row + r, j = col + r;
        float av = 0, bv = 0;
        if (kind == 0 || kind == 3)
            av = i < length(s, block) ? read(p.dout, token(s, block, i) * s.V + v) : 0;
        else
            av = kind == 2 ? base[l.X + i * s.V + v] : dst[gl.Y + i * s.V + v];
        if (j < width) {
            if (kind < 2)
                bv = base[l.S + j * s.V + v];
            else if (kind == 2)
                bv = dst[gl.dS + j * s.V + v];
            else
                bv = base[l.X + j * s.V + v];
        }
        const int index = r * 128 + (v ^ ((r & 15) << 2));
        a[index]        = av;
        b[index]        = bv;
    }
    __syncthreads();
    const int lane = tid & 63, wave = tid >> 6;
    const int r = 16 * (wave / 2) + (lane & 15), c = 16 * (wave % 2) + (lane & 15);
    const int z = 2 * (lane >> 4);
    f4 acc      = {0, 0, 0, 0};
    for (int v = 0; v < s.V; v += 8) {
        f2 av = {0, 0}, bv = {0, 0};
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            if (v + z + e < s.V) {
                av[e] = a[r * 128 + ((v + z + e) ^ ((r & 15) << 2))];
                if (col + c < width) bv[e] = b[c * 128 + ((v + z + e) ^ ((c & 15) << 2))];
            }
        }
        acc = g2::mmac32(av, bv, acc);
    }
    const uint64_t offset = kind == 0 ? gl.dQ : kind == 1 ? gl.dR : kind == 2 ? gl.dE : kind == 3 ? gl.dP : gl.dA;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int j = col + 16 * (wave % 2) + g2::c_col(lane, e);
        if (j < width) dst[offset + (row + r) * width + j] = (kind == 1 || kind == 4 ? -1.f : 1.f) * acc[e];
    }
}
__global__ void gates(Shape s, Args p, Layout l, GradLayout gl, float* grad) {
    const int block = blockIdx.x, ib = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 63, wave = tid >> 6, dbeg = blockIdx.z * 64;
    const int row = 16 * ib + (lane & 15), len = length(s, block);
    const float* base = static_cast<const float*>(p.cache) + size_t(block) * l.size;
    float* dst        = grad + size_t(block) * gl.size;
    __shared__ float a[16 * 20], rhs[16 * 68];
    f4 dq = {}, dk = {}, db = {}, dg = {};
    f4 q = {}, k = {}, beta = {}, grow = {};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int d = dbeg + 16 * wave + g2::c_col(lane, e), gi = row * s.K + d;
        if (d < s.K) {
            grow[e]        = base[l.g + gi];
            const float eg = __expf(grow[e]);
            q[e]           = p.scale * base[l.q + gi];
            k[e]           = base[l.k + gi];
            beta[e]        = row < len ? read(p.b, token(s, block, row) * s.K + d, p.bf32) : 0;
            dq[e]          = p.scale * eg * dst[gl.dQ + gi];
            dk[e] = beta[e] * eg * dst[gl.dR + gi] + __expf(base[l.g + (C - 1) * s.K + d] - grow[e]) * dst[gl.dE + gi];
            db[e] = k[e] * eg * dst[gl.dR + gi];
            dg[e] = base[l.Q + gi] * dst[gl.dQ + gi] + beta[e] * k[e] * eg * dst[gl.dR + gi] -
                    dst[gl.dE + gi] * base[l.E + gi];
        }
    }
    for (int jb = 0; jb <= ib; ++jb) {
        const int ref = 16 * ib + (jb == ib ? 7 : -1);
        for (int ix = tid; ix < 16 * 64; ix += 256) {
            const int j = 16 * jb + ix / 64, d = dbeg + ix % 64;
            rhs[(ix / 64) * 68 + ix % 64] =
                d < s.K ? base[l.k + j * s.K + d] * __expf(base[l.g + ref * s.K + d] - base[l.g + j * s.K + d]) : 0;
        }
        for (int kind = 0; kind < 2; ++kind) {
            const int i = 16 * ib + tid / 16, j = 16 * jb + tid % 16;
            a[(tid / 16) * 20 + tid % 16] = (kind ? i > j : i >= j) ? dst[(kind ? gl.dA : gl.dP) + i * C + j] : 0;
            __syncthreads();
            const f4 result = dot16({a, 20, 1}, {rhs, 68, 1}, 16, 64, 16, 0, 16 * wave);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int d = dbeg + 16 * wave + g2::c_col(lane, e);
                if (d < s.K) {
                    const float term = result[e] * __expf(grow[e] - base[l.g + ref * s.K + d]);
                    if (kind == 0) {
                        dq[e] += p.scale * term;
                        dg[e] += q[e] * term;
                    } else {
                        dk[e] += beta[e] * term;
                        db[e] += k[e] * term;
                        dg[e] += beta[e] * k[e] * term;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int jb = ib; jb < 4; ++jb) {
        const int ref = jb == ib ? 16 * ib + 7 : 16 * jb - 1;
        for (int kind = 0; kind < 2; ++kind) {
            const int i = 16 * ib + tid / 16, j = 16 * jb + tid % 16;
            a[(tid / 16) * 20 + tid % 16] = (kind ? j > i : j >= i) ? dst[(kind ? gl.dA : gl.dP) + j * C + i] : 0;
            for (int ix = tid; ix < 16 * 64; ix += 256) {
                const int r = 16 * jb + ix / 64, d = dbeg + ix % 64, gi = r * s.K + d;
                float value = 0;
                if (d < s.K) {
                    const float bi = r < len ? read(p.b, token(s, block, r) * s.K + d, p.bf32) : 0;
                    value          = (kind ? bi * base[l.k + gi] : p.scale * base[l.q + gi]) *
                            __expf(base[l.g + gi] - base[l.g + ref * s.K + d]);
                }
                rhs[(ix / 64) * 68 + ix % 64] = value;
            }
            __syncthreads();
            const f4 result = dot16({a, 20, 1}, {rhs, 68, 1}, 16, 64, 16, 0, 16 * wave);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int d = dbeg + 16 * wave + g2::c_col(lane, e);
                if (d < s.K) {
                    const float term  = result[e] * __expf(base[l.g + ref * s.K + d] - grow[e]);
                    dk[e]            += term;
                    dg[e]            -= k[e] * term;
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int d = dbeg + 16 * wave + g2::c_col(lane, e), gi = row * s.K + d;
        if (d < s.K) {
            dst[gl.dq + gi] = dq[e];
            dst[gl.dk + gi] = dk[e];
            dst[gl.dQ + gi] = dg[e];
            if (row < len) write(p.db, token(s, block, row) * s.K + d, db[e], p.bf32);
        }
    }
}
__global__ void gate_scan(Shape s, Args p, Layout l, GradLayout gl, const float* grad) {
    const int block = blockIdx.x, d = threadIdx.x, len = length(s, block);
    if (d >= s.K) return;
    const float* base = static_cast<const float*>(p.cache) + size_t(block) * l.size;
    const float* dst  = grad + size_t(block) * gl.size;
    float carry       = 0;
    for (int v = 0; v < s.V; ++v) carry += base[l.S + d * s.V + v] * dst[gl.dS + d * s.V + v];
    carry *= __expf(base[l.g + (C - 1) * s.K + d]);
    for (int i = 0; i < len; ++i) carry += dst[gl.dE + i * s.K + d] * base[l.E + i * s.K + d];
    for (int i = len - 1; i >= 0; --i) {
        carry += dst[gl.dQ + i * s.K + d];
        write(p.dg, token(s, block, i) * s.K + d, carry, p.gf32);
    }
}
__global__ void qk_grad(Shape s, Args p, Layout l, GradLayout gl, const float* grad) {
    const int t = blockIdx.x % s.T, batch = blockIdx.x / s.T, h = blockIdx.y, d = threadIdx.x;
    const int c = t / C, i = t % C, group = s.HV / s.H, first = (batch * s.HV + h * group) * s.chunks() + c;
    const float* base = static_cast<const float*>(p.cache) + size_t(first) * l.size;
    float dq = 0, dk = 0;
    if (d < s.K)
        for (int j = 0; j < group; ++j) {
            const float* src  = grad + size_t(first + j * s.chunks()) * gl.size;
            dq               += src[gl.dq + i * s.K + d];
            dk               += src[gl.dk + i * s.K + d];
        }
    __shared__ float qs[128], ks[128];
    const float q = d < s.K ? base[l.q + i * s.K + d] : 0, k = d < s.K ? base[l.k + i * s.K + d] : 0;
    qs[d] = dq * q;
    ks[d] = dk * k;
    __syncthreads();
    for (int off = 64; off; off /= 2) {
        if (d < off) {
            qs[d] += qs[d + off];
            ks[d] += ks[d + off];
        }
        __syncthreads();
    }
    if (d < s.K) {
        if (p.norm) {
            dq = (dq - q * qs[0]) * base[l.norm + 2 * i];
            dk = (dk - k * ks[0]) * base[l.norm + 2 * i + 1];
        }
        const size_t ix = ((size_t(batch) * s.T + t) * s.H + h) * s.K + d;
        write(p.dq, ix, dq);
        write(p.dk, ix, dk);
    }
}
int valueTile() {
    const char* value = std::getenv("FTRAIN_GDN2_VALUE_TILE");
    if (!value || !*value || std::strcmp(value, "16") == 0) return 16;
    if (std::strcmp(value, "8") == 0) return 8;
    if (std::strcmp(value, "4") == 0) return 4;
    throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "FTRAIN_GDN2_VALUE_TILE must be 4, 8 or 16");
}
void check(const char* name) {
    auto e = hipGetLastError();
    if (e != hipSuccess) throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN2 %s: %s", name, hipGetErrorString(e));
}
}  // namespace
bool deviceSupported(FTrainDeviceId device) noexcept {
    hipDeviceProp_t prop{};
    return hipGetDeviceProperties(&prop, device) == hipSuccess && std::strncmp(prop.gcnArchName, "gfx936", 6) == 0;
}
void forward(const Shape& s, const Args& p, FTrainStream stream) {
    if (!s.T) return;
    const int vs = valueTile();
    const Layout l(s);
    const int nb = static_cast<int>(s.blocks());
    hipLaunchKernelGGL(normalize, dim3(nb), dim3(256), 0, stream, s, p, l);
    check("normalize");
    hipLaunchKernelGGL(prepare, dim3(nb), dim3(256), 0, stream, s, p, l);
    check("prepare");
    hipLaunchKernelGGL(intra, dim3(nb, 4), dim3(256), 0, stream, s, p, l);
    check("intra");
    hipLaunchKernelGGL(wy, dim3(nb), dim3(256), 0, stream, s, p, l);
    check("wy");
#define LAUNCH_F(VS)                                                                                                   \
    do {                                                                                                               \
        if (s.B * s.HV <= 8) {                                                                                         \
            hipLaunchKernelGGL((recurrence<VS, true>), dim3(s.B * s.HV, (s.V + VS - 1) / VS), dim3(256), 0, stream, s, \
                               p, l);                                                                                  \
        } else {                                                                                                       \
            hipLaunchKernelGGL((recurrence<VS, false>), dim3(s.B * s.HV, (s.V + VS - 1) / VS), dim3(256), 0, stream,   \
                               s, p, l);                                                                               \
        }                                                                                                              \
    } while (false)
    if (vs == 4) {
        LAUNCH_F(4);
    } else if (vs == 8) {
        LAUNCH_F(8);
    } else {
        LAUNCH_F(16);
    }
#undef LAUNCH_F
    check("recurrence");
}
void backward(const Shape& s, const Args& p, void* workspace, FTrainStream stream) {
    if (!s.T) return;
    const int vs = valueTile();
    const Layout l(s);
    const GradLayout gl(s);
    float* grad = static_cast<float*>(workspace);
#define LAUNCH_B(VS)                                                                                                   \
    do {                                                                                                               \
        if (s.B * s.HV <= 8) {                                                                                         \
            hipLaunchKernelGGL((reverse<VS, true>), dim3(s.B * s.HV, (s.V + VS - 1) / VS), dim3(256), 0, stream, s, p, \
                               l, gl, grad);                                                                           \
        } else {                                                                                                       \
            hipLaunchKernelGGL((reverse<VS, false>), dim3(s.B * s.HV, (s.V + VS - 1) / VS), dim3(256), 0, stream, s,   \
                               p, l, gl, grad);                                                                        \
        }                                                                                                              \
    } while (false)
    if (vs == 4) {
        LAUNCH_B(4);
    } else if (vs == 8) {
        LAUNCH_B(8);
    } else {
        LAUNCH_B(16);
    }
#undef LAUNCH_B
    check("reverse");
    hipLaunchKernelGGL(products, dim3(s.blocks(), 5, 2 * ((std::max(s.K, C) + 31) / 32)), dim3(256), 0, stream, s, p, l,
                       gl, grad);
    check("products");
    hipLaunchKernelGGL(gates, dim3(s.blocks(), 4, (s.K + 63) / 64), dim3(256), 0, stream, s, p, l, gl, grad);
    check("gates");
    hipLaunchKernelGGL(gate_scan, dim3(s.blocks()), dim3(128), 0, stream, s, p, l, gl, grad);
    check("gate_scan");
    hipLaunchKernelGGL(qk_grad, dim3(s.B * s.T, s.H), dim3(128), 0, stream, s, p, l, gl, grad);
    check("qk_grad");
}
}  // namespace ftrain::gdn2
