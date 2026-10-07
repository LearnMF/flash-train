// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// The G2 KDA device translation unit: the handwritten gfx936 kernels (g2/) plus the two
// device-side helpers the host runner needs so that no step synchronizes with the host:
// the varlen chunk tables built from cu_seqlens, and the chunk-end gate gather (Gn).
#include "g2/g2_common.h"
#include "g2/g2_head.cuh"
#include "g2/g2_fwd_intra.cuh"
#include "g2/g2_fwd_intra2.cuh"
#include "g2/g2_fwd_prep.cuh"
#include "g2/g2_fwd_v8.cuh"
#include "g2/g2_engine.cuh"
#include "g2/g2_wu.cuh"
#include "g2/g2_dav.cuh"
#include "g2/g2_wy.cuh"
#include "g2/g2_bwd_tail.cuh"

namespace g2 {
namespace {

// Chunk tables of the packed stream (DESIGN: every sequence starts on a 64-row chunk).
// seg[s] = first chunk of sequence s (seg[N] = used chunks), ctok[c] / clen[c] = first token
// and valid rows of chunk c.  Chunks past seg[N] (the NT = ceil(T/64) + N capacity slack)
// get clen = 0 and ctok = T: every kernel masks all their rows, and their clamped g row
// (ctok + clen - 1 = T - 1) stays in bounds.  cu == null: one sequence of T rows.
__global__ void __launch_bounds__(256)
    kda_tables_kernel(const int* __restrict__ cu, int N, int T, int NT, int* __restrict__ ctok, int* __restrict__ clen,
                      int* __restrict__ seg) {
    __shared__ int scan[256];
    __shared__ int carry;
    const int tid = threadIdx.x;
    if (tid == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < N; base += 256) {
        const int s = base + tid;
        int start = 0, len = 0;
        if (s < N) {
            start = cu ? cu[s] : 0;
            len   = cu ? cu[s + 1] - start : T;
        }
        const int nc = (len + 63) / 64;
        scan[tid]    = nc;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {  // inclusive Hillis-Steele scan
            const int v = tid >= off ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const int first = carry + scan[tid] - nc;
        if (s < N) {
            seg[s] = first;
            for (int j = 0; j < nc; ++j) {
                ctok[first + j] = start + 64 * j;
                clen[first + j] = min(64, len - 64 * j);
            }
        }
        __syncthreads();
        if (tid == 255) carry += scan[255];
        __syncthreads();
    }
    if (tid == 0) seg[N] = carry;
    for (int c = carry + tid; c < NT; c += 256) {
        ctok[c] = T;
        clen[c] = 0;
    }
}

// Gn = the chunk-end g row per (head, chunk) in the v8 native order (H, NT, 8, 4, 4) with
// the last two axes swapped: out[((h*NT + i)*8 + a)*16 + c*4 + b] = g[row, h, 16a + 4b + c].
// Dense (ctok == null): row = 64i + 63; varlen: the chunk's last real token.
__global__ void __launch_bounds__(128)
    kda_to_gn_kernel(const float* __restrict__ g, const int* __restrict__ ctok, const int* __restrict__ clen, int H,
                     int NT, float* __restrict__ gn) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long row = ctok ? (long)ctok[i] + clen[i] - 1 : (long)i * 64 + 63;
    const int a = d >> 4, b = (d >> 2) & 3, c = d & 3;
    gn[(((size_t)h * NT + i) * 8 + a) * 16 + c * 4 + b] = g[((size_t)row * H + h) * 128 + d];
}

}  // namespace
}  // namespace g2

extern "C" int g2_kda_tables(const int* cu, int N, int T, int NT, int* ctok, int* clen, int* seg, hipStream_t s) {
    hipLaunchKernelGGL(g2::kda_tables_kernel, dim3(1), dim3(256), 0, s, cu, N, T, NT, ctok, clen, seg);
    return (int)hipGetLastError();
}

extern "C" int g2_kda_to_gn(const float* g, const int* ctok, const int* clen, int H, int NT, float* gn, hipStream_t s) {
    if (NT <= 0 || H <= 0) return 0;
    hipLaunchKernelGGL(g2::kda_to_gn_kernel, dim3(NT, H), dim3(128), 0, s, g, ctok, clen, H, NT, gn);
    return (int)hipGetLastError();
}
