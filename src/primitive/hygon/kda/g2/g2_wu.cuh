// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// S3-9/S3-11: recompute_w_u as one HIP kernel emitting ALL P1/P2 native operands.
//
// Math (FLA wy_fast.recompute_w_u_fwd, dense, safe-gate tier):
//   kb = bf16((k * beta) * 2^g)        w = A @ kb      (A = saved Akk, the UT inverse)
//   vb = bf16(v * beta)                u = A @ vb
//   kg = bf16(k * 2^(gn - g))  (gn = chunk-last g row)  -> A form (KgA) + B form (Kn)
//   qg = bf16(q * 2^g)                                       -> B form (Qn)
// The MMAC tier matches FLA (bf16 operands, fp32 accumulate); outputs verified
// bitwise against the FLA + torch-conversion reference chain.
//
// S3-11 structure: ONE 17.4KB LDS slab reused by the MMAC phases (kbT / vbT) plus
// a 16.9KB row-major tile for the gated emits -- ~34KB total but only ONE of the
// two slabs is live per phase... (both are declared, 34.6KB, 1 CTA/CU was the S3-9
// trap) -- instead the tile reuses the SAME storage via a union, keeping peak LDS
// at 17.4KB so 2+ CTAs fit per CU.
//
// T1 (GLM): the phase-1/2 fill loops additionally emit P3's two B5 operands
// straight from the k/g and v loads they already perform:
//   kgB = bf16(k * 2^g)   absolute gate -- NOT the 2^(gn-g) of phases 3/4, no beta
//   vB  = v               raw bf16, no beta
// both in the vB layout = A form with the row low nibble pi16-permuted and the
// P3 block order (i*H + hh).  The source quad is contiguous on BOTH sides, so the
// permutation only moves the quad base -- one extra 8B store per quad, no LDS
// round-trip, and the two layout2 convert2 calls (which re-read k/g/v, ~400 MB
// at 8K/48) disappear from B5.
//
// T2 instrumentation: G2_WU_STOP=N (env, read at launch) makes every CTA return
// right after stage N -- 1 sBeta/sGl/sA+fill1(kbT+kgB+KgA/Kn), 2 w-MMAC,
// 3 Wn/Wb emit, 4 fill2, 5 u-MMAC, 6 Un emit, 7 Qn direct pass; 0 = full.
// Stage costs come from consecutive differences (tests/g2/probe_wu_stages.py).
// Uniform branch, no data effect at stop=0.
//
// T2 round 2: (a) C-pi -- kbT staged with pi16-permuted rows so the w-MMAC C
// fragment holds lane-contiguous native Wn columns: the Wn emit is ONE aligned
// 8B quad per lane (was 4 stride-64 scalars) and Wb lands at stride 4 (was 16);
// vbT/u-MMAC keep identity staging (Un's C form IS the fragment layout).
// (b) A cached in LDS (both MMAC phases re-read it, 50 MB each).  (c) Qn
// emitted by a light standalone pass instead of the slab-tile round-trip.
//
// Emitting from the MMAC C fragments is free of layout translation except the
// known swaps: lane l holds (row = l&15, col = (l>>4)+4e), so the A form needs
// stride-64 scalars, the B form stride-16, the C form l*4 vecs.
#include "g2_common.h"

namespace g2 {

struct WuParams {
    const short* k; const short* v; const short* q; const float* beta; const short* A;
    const float* g;
    short* Wn; short* Wb; short* Un; short* KgA; short* Kn; short* Qn;
    short* kgB; short* vB;
    int H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    int stop;      // T2 staged timing (G2_WU_STOP): 0 = full kernel; 1..9 return
                   // right after the named stage (uniform branch, no data effect)
};

// vB-layout quad base for source quad (row j, column vq..vq+3): A form with the
// row low nibble pi16-permuted and the P3 block order (g2_layout2 L2_KGB/L2_VB).
__device__ __forceinline__ size_t wu_vb_quad(int j, int vq, size_t nblk3) {
    return nblk3 + (size_t)(4 * pi16(j & 15) + 64 * ((vq & 15) >> 2)
                            + 256 * (vq >> 4) + 2048 * (j >> 4));
}

constexpr int kPadJ2 = 68;

// VL (varlen phase 2): k/v/q/beta/A rows r >= clen read as 0 (the zero padding), g rows
// are read from the CLAMPED row (address only, the value math is the dense code), incl.
// the chunk-end row sGl = g[clen-1].  No row-major outputs.
template <bool VL>
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p) {
    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;
    __shared__ __align__(16) short slab[128 * kPadJ2];
    __shared__ float sBeta[kBT];
    __shared__ __align__(16) float sGl[kK];   // chunk-last g row: kg = k * 2^(gl-g)
    __shared__ __align__(16) short sA[kBT * kBT];
    const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const size_t rstride = (size_t)p.H * kK;           // [B,T,H,128] row stride
        const size_t sbase = ((size_t)g2_row_tok<VL>(p.ctok, i) * p.H + hh) * kK;
        const size_t nblk = (size_t)(hh * p.NT + i) * 8192;
        const size_t nblk3 = g2_nat_blk(i, hh, p.H, p.NT, true, 8192);   // P3's (i*H+hh)
        // 17.4KB union: transposed MMAC staging (kbT/vbT) OR the row-major gated tile
        // A (UT inverse, 64x64 bf16): read by BOTH MMAC phases -- cache it in LDS
        // (8KB; occupancy is VGPR-bound at 2 CTAs/CU, so the LDS is free).

        if (tid < kBT)
            sBeta[tid] = g2_row_ok<M>(tid, cl)
                ? p.beta[(size_t)(g2_row_tok<VL>(p.ctok, i) + tid) * p.H + hh] : 0.f;
        if (tid < 32)                  // 128 floats = 32 f4 quads
            reinterpret_cast<f4*>(sGl)[tid] =
                ld16f(p.g + sbase + g2_row_g<M>(63, cl) * rstride + 4 * tid);
        for (int t = tid; t < 1024; t += 256) {          // sA: 64x64 bf16 row-major
            const int r = t >> 4, c4 = (t & 15) << 2;
            *reinterpret_cast<v4bh*>(sA + r * kBT + c4) = g2_row_ok<M>(r, cl)
                ? *reinterpret_cast<const v4bh*>(
                      p.A + g2_rm_tok(g2_row_tok<VL>(p.ctok, i) + r, hh, p.H, kBT) + c4)
                : v4bh{0, 0, 0, 0};
        }
        __syncthreads();   // sBeta/sGl/sA must land before the phase-1 fill reads them

        // ---- MMAC phase: A @ (B-side staged transposed in slab) -> emit from C frags.
        // A comes from sA (cached).  EMIT CONVENTION: the w phase stages kbT with
        // pi16-permuted rows (see fill1), so its C fragment holds lane-contiguous
        // native columns: Wn (A form) is ONE aligned 8B quad store per lane and Wb
        // (B form) lands at native stride 4.  The u phase stages vbT with identity
        // rows and only uses the C-form emit (the C form IS the fragment layout).
        auto mmac_phase = [&](short* outA, short* outB, short* outC, int stop_acc) {
            __syncthreads();
            f4 acc[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
            for (int js = 0; js < 4; ++js) {
                const v4bh af = *reinterpret_cast<const v4bh*>(
                    sA + (size_t)(16 * wv + (l & 15)) * kBT + 16 * js + (l >> 4) * 4);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const size_t lo = (size_t)(16 * vt + (l & 15)) * kPadJ2 + 16 * js + (l >> 4) * 4;
                    acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(slab + lo), acc[vt]);
                }
            }
            if (p.stop == stop_acc) return;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                short we[4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) we[e] = bf_rne(acc[vt][e]);
                if (outC)   // C form is the C-fragment layout itself: one 8B vec store
                    *reinterpret_cast<v4bh*>(outC + nblk + (size_t)(vt + 8 * wv) * 256 + l * 4) =
                        *reinterpret_cast<const v4bh*>(we);
                if (outA) { // A form, pi-staged: lane quad is native-contiguous
                    *reinterpret_cast<v4bh*>(outA + nblk + (size_t)(4 * (l & 15) + 64 * (l >> 4)
                        + 256 * vt + 2048 * wv)) = *reinterpret_cast<const v4bh*>(we);
                }
                if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e
                    const int ib = ((l & 15) & 3) + 16 * (l >> 4) + 64 * ((l & 15) >> 2)
                                   + 256 * wv + 1024 * vt;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) outB[nblk + ib + 4 * e] = we[e];
                }
            }
        };

        // ---- phase 1: kbT = bf16((k*beta)*2^g)^T ; w = A @ kb -> Wn (A) + Wb (B)
        // T1: the same k/g loads also emit kgB = bf16(k * 2^g) (absolute gate, no beta)
        // into the vB layout -- one extra 8B store per quad, no extra traffic reads.
        // T2: the SAME pass also computes kg = bf16(k * 2^(gl-g)) (gl = chunk-last g
        // row, sGl) and emits KgA (A form, aligned 8B quads) + Kn (B form, stride-4
        // scalars) directly -- the old kg phases re-read k+g (290 MB @8K/48) purely
        // redundantly.  Value expressions are character-identical to the retired
        // stage_gated(last_ref)/emit_AB path (bitwise gate holds).
        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0};
            const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
            const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
            const float b = sBeta[j];
            // T7: one exp2f(gf) per element serves both kb and kgB (identical
            // value; the multiply associations below are unchanged).
            f4 eg;
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) eg[dv] = exp2f(gf[dv]);
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                // kbT staged with pi16-permuted rows (C-pi): the w-MMAC C
                // fragment then holds lane-contiguous native Wn columns.
                const int vv = vq + dv;
                slab[(size_t)(16 * (vv >> 4) + pi16(vv & 15)) * kPadJ2 + j] =
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);
            }
            short gv[4], gkv[4];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                gv[dv] = bf_rne(bf2f(kv[dv]) * eg[dv]);
                gkv[dv] = bf_rne(bf2f(kv[dv]) * exp2f(gl[dv] - gf[dv]));
            }
            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                *reinterpret_cast<const v4bh*>(gv);
            // KgA: A form, plain row nibble, P1/P2 block order
            *reinterpret_cast<v4bh*>(p.KgA + nblk + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) =
                *reinterpret_cast<const v4bh*>(gkv);
            // Kn: B form -- the quad is 4 consecutive shorts at an e-dependent
            // (generally unaligned) offset, so scalar stride-1 stores
            const int ibn = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                            + 256 * (j >> 4) + 1024 * (vq >> 4);
            if (p.Kn != nullptr) {                          // null: keep-state backward
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) p.Kn[nblk + ibn + 4 * dv] = gkv[dv];
            }
        }
        if (p.stop == 1) return;
        mmac_phase(p.Wn, p.Wb, nullptr, 2);
        if (p.stop == 2 || p.stop == 3) return;

        // ---- phase 2: vbT = bf16(v*beta)^T ; u = A @ vb -> Un (C)
        // T1: vB = raw v (no beta) into the same vB layout, from the same load.
        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};
            const float b = sBeta[j];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv)
                slab[(size_t)(vq + dv) * kPadJ2 + j] = bf_rne(bf2f(vv[dv]) * b);
            *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        if (p.stop == 4) return;
        mmac_phase(nullptr, nullptr, p.Un, 5);
        if (p.stop == 5 || p.stop == 6) return;

        // ---- phase 3 (T7): qg = bf16(q * 2^g) to Qn (B form) through a B-form
        // tile in the now-FREE slab: the stride-4 global scatter (the measured
        // 465 GB/s tax, Kn/Qn class) moves to the LDS side as scalar stores and
        // the global emit becomes linear 8B quads.  tile = this CTA's whole
        // 8192-element native block, aliasing the slab union (16 KB <= 17.4 KB).
        {
            short* tile = slab;                             // 64x128 in native order
            for (int t = tid; t < 2048; t += 256) {
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                const v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                                     : v4bh{0, 0, 0, 0};
                const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
                short qgv[4];
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) qgv[dv] = bf_rne(bf2f(qv[dv]) * exp2f(gf[dv]));
                const int ibn = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                                + 256 * (j >> 4) + 1024 * (vq >> 4);
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) tile[ibn + 4 * dv] = qgv[dv];
            }
            __syncthreads();
            for (int t = tid; t < 2048; t += 256)            // linear 8B global emit
                *reinterpret_cast<v4bh*>(p.Qn + nblk + 4 * t) =
                    *reinterpret_cast<const v4bh*>(tile + 4 * t);
        }
        if (p.stop == 7) return;
    });
}

}  // namespace g2

extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    int stop = 0;
    if (const char* e = getenv("G2_WU_STOP")) stop = atoi(e);
    g2::WuParams p{(const short*)k, (const short*)v, (const short*)q, (const float*)beta,
                   (const short*)A, (const float*)g,
                   (short*)Wn, (short*)Wb, (short*)Un, (short*)KgA, (short*)Kn, (short*)Qn,
                   (short*)kgB, (short*)vB,
                   H, T / 64, ctok, clen, stop};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
