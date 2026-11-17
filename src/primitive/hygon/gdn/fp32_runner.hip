// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/gdn/fp32_runner.hpp"

namespace ftrain::gdn_fp32 {
namespace {
using bf16      = __hip_bfloat16;
constexpr int D = 128, BT = 64, BV = 32, NV = D / BV, R = BV / 2, THREADS = 256;
constexpr int MAX_BLOCKS = 128;

struct Arena {
    char* base;
    std::uint64_t used = 0;
    explicit Arena(void* p) : base(static_cast<char*>(p)) {}
    template<class T>
    T* take(std::uint64_t n) {
        auto* p  = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used    += (n * sizeof(T) + 255) / 256 * 256;
        return p;
    }
};

int blocks(const Shape& s) { return static_cast<int>(std::min<std::int64_t>(s.seqs * s.heads * NV, MAX_BLOCKS)); }

struct Layout {
    int* seg;
    float *qn, *kn, *h, *scratch, *dq, *dk, *dg, *db;
};

Layout layout(const Shape& s, Arena& a, bool backward) {
    Layout l{};
    const std::uint64_t rows = s.tokens * s.heads;
    l.seg                    = a.take<int>(s.seqs + 1);
    if (s.l2norm_qk && !s.has_qkn) {
        l.qn = a.take<float>(rows * D);
        l.kn = a.take<float>(rows * D);
    }
    if (backward) {
        if (!s.has_saved_h) l.h = a.take<float>(s.stateElements());
        // One chunk's replay per resident CTA, independent of sequence length/count.
        l.scratch = a.take<float>(static_cast<std::uint64_t>(blocks(s)) * BT * D * BV);
        l.dq      = a.take<float>(rows * NV * D);
        l.dk      = a.take<float>(rows * NV * D);
        l.dg      = a.take<float>(rows * NV);
        l.db      = a.take<float>(rows * NV);
    }
    return l;
}

void check(hipError_t e) {
    if (e != hipSuccess) throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN FP32: %s", hipGetErrorString(e));
}

__device__ float wave_sum(float x) {
#pragma unroll
    for (int d = 32; d; d >>= 1) x += __shfl_xor(x, d, 64);
    return x;
}

__global__ void sequence_offsets(const int* cu, int n, int t, int* seg) {
    __shared__ int scan[256];
    __shared__ int carry;
    const int tid = threadIdx.x;
    if (tid == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += 256) {
        const int i     = base + tid;
        const int len   = i < n ? (cu ? cu[i + 1] - cu[i] : t) : 0;
        const int count = (len + BT - 1) / BT;
        scan[tid]       = count;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            const int v = tid >= d ? scan[tid - d] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        if (i < n) seg[i] = carry + scan[tid] - count;
        __syncthreads();
        if (tid == 255) carry += scan[255];
        __syncthreads();
    }
    if (tid == 0) seg[n] = carry;
}

__global__ void normalize(const bf16* q, const bf16* k, float* qn, float* kn, float* qr, float* kr) {
    __shared__ float sums[4];
    const int d           = threadIdx.x;
    const std::size_t row = blockIdx.x, idx = row * D + d;
    const float qv = static_cast<float>(q[idx]), kv = static_cast<float>(k[idx]);
    const float qs = wave_sum(qv * qv), ks = wave_sum(kv * kv);
    if ((d & 63) == 0) {
        sums[d >> 6]       = qs;
        sums[2 + (d >> 6)] = ks;
    }
    __syncthreads();
    const float rq = rsqrtf(sums[0] + sums[1] + 1e-6f), rk = rsqrtf(sums[2] + sums[3] + 1e-6f);
    qn[idx] = qv * rq;
    kn[idx] = kv * rk;
    if (d == 0) {
        if (qr) qr[row] = rq;
        if (kr) kr[row] = rk;
    }
}

struct Params {
    const bf16 *q, *k, *v, *d_o;
    const float *qn, *kn, *g, *beta;
    const int *cu, *seg;
    bf16 *o, *dv;
    float *h, *scratch, *dq, *dk, *dg, *db;
    int T, H, N;
    float scale;
};

__device__ float load_qk(const bf16* raw, const float* normalized, std::size_t idx) {
    return normalized ? normalized[idx] : static_cast<float>(raw[idx]);
}

// Each lane owns one key coordinate and R value columns. Two waves sum K=128.
__device__ void key_sum(float (&x)[R], float* shared) {
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
#pragma unroll
    for (int j = 0; j < R; ++j) {
        x[j] = wave_sum(x[j]);
        if (lane == 0) shared[wave * R + j] = x[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < R; ++j) x[j] += shared[(wave ^ 1) * R + j];
    __syncthreads();
}

__device__ std::size_t state_offset(int chunk, int head, int tile, int heads) {
    return ((static_cast<std::size_t>(chunk) * heads + head) * NV + tile) * D * BV;
}

__global__ __launch_bounds__(256) void forward_kernel(Params p) {
    __shared__ float shared[4 * R];
    const int tid = threadIdx.x, key = tid & 127, group = tid >> 7;
    for (int job = blockIdx.x; job < p.N * p.H * NV; job += gridDim.x) {
        const int tile = job % NV, head = (job / NV) % p.H, seq = job / (NV * p.H);
        const int start = p.cu ? p.cu[seq] : 0, end = p.cu ? p.cu[seq + 1] : p.T;
        float h[R] = {};
        for (int t = start; t < end; ++t) {
            const std::size_t row = static_cast<std::size_t>(t) * p.H + head;
            const float q = load_qk(p.q, p.qn, row * D + key), k = load_qk(p.k, p.kn, row * D + key);
            const float alpha = expf(p.g[row]), beta = p.beta[row];
            if (p.h && (t - start) % BT == 0) {
                const auto off = state_offset(p.seg[seq] + (t - start) / BT, head, tile, p.H);
#pragma unroll
                for (int j = 0; j < R; ++j) p.h[off + j * THREADS + tid] = h[j];
            }
            float pred[R];
#pragma unroll
            for (int j = 0; j < R; ++j) pred[j] = k * h[j];
            key_sum(pred, shared);
            float out[R];
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const int value      = tile * BV + group * R + j;
                const float residual = static_cast<float>(p.v[row * D + value]) - alpha * pred[j];
                h[j]                 = fmaf(k * beta, residual, alpha * h[j]);
                out[j]               = q * h[j];
            }
            if (p.o) {
                key_sum(out, shared);
                if (key == 0) {
#pragma unroll
                    for (int j = 0; j < R; ++j) p.o[row * D + tile * BV + group * R + j] = bf16(p.scale * out[j]);
                }
            }
        }
    }
}

__global__ __launch_bounds__(256) void backward_kernel(Params p) {
    __shared__ float shared[520];
    const int tid = threadIdx.x, key = tid & 127, group = tid >> 7;
    float* replay = p.scratch + static_cast<std::size_t>(blockIdx.x) * BT * D * BV;
    for (int job = blockIdx.x; job < p.N * p.H * NV; job += gridDim.x) {
        const int tile = job % NV, head = (job / NV) % p.H, seq = job / (NV * p.H);
        const int start = p.cu ? p.cu[seq] : 0, end = p.cu ? p.cu[seq + 1] : p.T;
        float dh[R] = {};
        for (int chunk = (end - start + BT - 1) / BT - 1; chunk >= 0; --chunk) {
            const int first = start + chunk * BT, len = min(BT, end - first);
            const auto off = state_offset(p.seg[seq] + chunk, head, tile, p.H);
            float h[R];
#pragma unroll
            for (int j = 0; j < R; ++j) h[j] = p.h[off + j * THREADS + tid];
            // Replay from an FP32 checkpoint; never invert a nearly singular update.
            for (int r = 0; r < len; ++r) {
                const std::size_t row = static_cast<std::size_t>(first + r) * p.H + head;
                const float k = load_qk(p.k, p.kn, row * D + key), alpha = expf(p.g[row]), beta = p.beta[row];
                float pred[R];
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    replay[(r * R + j) * THREADS + tid] = h[j];
                    pred[j]                             = k * h[j];
                }
                key_sum(pred, shared);
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    const int value      = tile * BV + group * R + j;
                    const float residual = static_cast<float>(p.v[row * D + value]) - alpha * pred[j];
                    h[j]                 = fmaf(k * beta, residual, alpha * h[j]);
                }
            }
            for (int r = len - 1; r >= 0; --r) {
                const std::size_t row = static_cast<std::size_t>(first + r) * p.H + head;
                const float q = load_qk(p.q, p.qn, row * D + key), k = load_qk(p.k, p.kn, row * D + key);
                const float alpha = expf(p.g[row]), beta = p.beta[row];
                float pred[R], de[R], pre[R], dout[R];
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    pre[j]  = alpha * replay[(r * R + j) * THREADS + tid];
                    pred[j] = k * pre[j];
                    dout[j] = static_cast<float>(p.d_o[row * D + tile * BV + group * R + j]) * p.scale;
                    dh[j]   = fmaf(q, dout[j], dh[j]);
                    de[j]   = k * dh[j];
                }
                key_sum(pred, shared);
                key_sum(de, shared);
                float dq = 0, dk = 0, dg = 0, db = 0;
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    const int value       = tile * BV + group * R + j;
                    const float residual  = static_cast<float>(p.v[row * D + value]) - pred[j];
                    const float dr        = beta * de[j];
                    const float state     = fmaf(k * beta, residual, pre[j]);
                    dq                    = fmaf(dout[j], state, dq);
                    dk                   += beta * residual * dh[j] - pre[j] * dr;
                    const float ds        = fmaf(-k, dr, dh[j]);
                    dg                    = fmaf(pre[j], ds, dg);
                    dh[j]                 = alpha * ds;
                    if (key == 0) {
                        p.dv[row * D + value] = bf16(dr);
                        db                    = fmaf(residual, de[j], db);
                    }
                }
                shared[tid]       = dq;
                shared[256 + tid] = dk;
                const float gs = wave_sum(dg), bs = wave_sum(db);
                if ((tid & 63) == 0) {
                    shared[512 + (tid >> 6)] = gs;
                    shared[516 + (tid >> 6)] = bs;
                }
                __syncthreads();
                if (tid < 128) {
                    const auto idx = (row * NV + tile) * D + key;
                    p.dq[idx]      = shared[tid] + shared[128 + tid];
                    p.dk[idx]      = shared[256 + tid] + shared[384 + tid];
                }
                if (tid == 0) {
                    p.dg[row * NV + tile] = (shared[512] + shared[513]) + (shared[514] + shared[515]);
                    p.db[row * NV + tile] = (shared[516] + shared[517]) + (shared[518] + shared[519]);
                }
                __syncthreads();
            }
        }
    }
}

__global__ void finish_backward(Params p, const float* qr, const float* kr, bf16* dq, bf16* dk, float* dg, float* db) {
    __shared__ float sums[4];
    const int d           = threadIdx.x;
    const std::size_t row = blockIdx.x, idx = row * D + d;
    float qg = 0, kg = 0;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        qg += p.dq[(row * NV + i) * D + d];
        kg += p.dk[(row * NV + i) * D + d];
    }
    if (p.qn) {
        const float qs = wave_sum(qg * p.qn[idx]), ks = wave_sum(kg * p.kn[idx]);
        if ((d & 63) == 0) {
            sums[d >> 6]       = qs;
            sums[2 + (d >> 6)] = ks;
        }
        __syncthreads();
        qg = qr[row] * (qg - p.qn[idx] * (sums[0] + sums[1]));
        kg = kr[row] * (kg - p.kn[idx] * (sums[2] + sums[3]));
    }
    dq[idx] = bf16(qg);
    dk[idx] = bf16(kg);
    if (d == 0) {
        float gs = 0, bs = 0;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            gs += p.dg[row * NV + i];
            bs += p.db[row * NV + i];
        }
        dg[row] = gs;
        db[row] = bs;
    }
}

Params params(const Shape& s, const void* q, const void* k, const void* v, const void* g, const void* b, const void* qn,
              const void* kn, const int* cu, Layout l, float scale) {
    Params p{};
    p.q       = static_cast<const bf16*>(q);
    p.k       = static_cast<const bf16*>(k);
    p.v       = static_cast<const bf16*>(v);
    p.g       = static_cast<const float*>(g);
    p.beta    = static_cast<const float*>(b);
    p.qn      = static_cast<const float*>(qn);
    p.kn      = static_cast<const float*>(kn);
    p.cu      = cu;
    p.seg     = l.seg;
    p.T       = s.tokens;
    p.H       = s.heads;
    p.N       = s.seqs;
    p.scale   = scale;
    p.scratch = l.scratch;
    p.dq      = l.dq;
    p.dk      = l.dk;
    p.dg      = l.dg;
    p.db      = l.db;
    return p;
}

}  // namespace

std::uint64_t forwardWorkspaceBytes(const Shape& s) noexcept {
    Arena a(nullptr);
    layout(s, a, false);
    return a.used;
}
std::uint64_t backwardWorkspaceBytes(const Shape& s) noexcept {
    Arena a(nullptr);
    layout(s, a, true);
    return a.used;
}

void forward(const Shape& s, const ForwardArgs& args, void* workspace, FTrainStream stream) {
    const auto st = static_cast<hipStream_t>(stream);
    Arena a(workspace);
    const auto l = layout(s, a, false);
    auto* qn     = args.qn ? static_cast<float*>(args.qn) : l.qn;
    auto* kn     = args.kn ? static_cast<float*>(args.kn) : l.kn;
    hipLaunchKernelGGL(sequence_offsets, dim3(1), dim3(256), 0, st, args.cu_seqlens, s.seqs, s.tokens, l.seg);
    check(hipGetLastError());
    if (s.l2norm_qk) {
        hipLaunchKernelGGL(normalize, dim3(s.tokens * s.heads), dim3(128), 0, st, static_cast<const bf16*>(args.q),
                           static_cast<const bf16*>(args.k), qn, kn, static_cast<float*>(args.q_rstd),
                           static_cast<float*>(args.k_rstd));
        check(hipGetLastError());
    }
    auto p = params(s, args.q, args.k, args.v, args.g, args.beta, qn, kn, args.cu_seqlens, l, args.scale);
    p.o    = static_cast<bf16*>(args.o);
    p.h    = static_cast<float*>(args.saved_h);
    hipLaunchKernelGGL(forward_kernel, dim3(blocks(s)), dim3(THREADS), 0, st, p);
    check(hipGetLastError());
}

void backward(const Shape& s, const BackwardArgs& args, void* workspace, FTrainStream stream) {
    const auto st = static_cast<hipStream_t>(stream);
    Arena a(workspace);
    const auto l   = layout(s, a, true);
    const auto* qn = args.qn ? static_cast<const float*>(args.qn) : l.qn;
    const auto* kn = args.kn ? static_cast<const float*>(args.kn) : l.kn;
    hipLaunchKernelGGL(sequence_offsets, dim3(1), dim3(256), 0, st, args.cu_seqlens, s.seqs, s.tokens, l.seg);
    check(hipGetLastError());
    if (s.l2norm_qk && !s.has_qkn) {
        hipLaunchKernelGGL(normalize, dim3(s.tokens * s.heads), dim3(128), 0, st, static_cast<const bf16*>(args.q),
                           static_cast<const bf16*>(args.k), l.qn, l.kn, nullptr, nullptr);
        check(hipGetLastError());
    }
    auto p = params(s, args.q, args.k, args.v, args.g, args.beta, qn, kn, args.cu_seqlens, l, args.scale);
    p.h    = args.saved_h ? const_cast<float*>(static_cast<const float*>(args.saved_h)) : l.h;
    if (!s.has_saved_h) {
        hipLaunchKernelGGL(forward_kernel, dim3(blocks(s)), dim3(THREADS), 0, st, p);
        check(hipGetLastError());
    }
    p.d_o = static_cast<const bf16*>(args.d_o);
    p.dv  = static_cast<bf16*>(args.dv);
    hipLaunchKernelGGL(backward_kernel, dim3(blocks(s)), dim3(THREADS), 0, st, p);
    check(hipGetLastError());
    hipLaunchKernelGGL(finish_backward, dim3(s.tokens * s.heads), dim3(128), 0, st, p,
                       static_cast<const float*>(args.q_rstd), static_cast<const float*>(args.k_rstd),
                       static_cast<bf16*>(args.dq), static_cast<bf16*>(args.dk), static_cast<float*>(args.dg),
                       static_cast<float*>(args.dbeta));
    check(hipGetLastError());
}

}  // namespace ftrain::gdn_fp32
