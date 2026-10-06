// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// S3-11: HIP chunk_kda_bwd_dAv -- reads v_new in its NATIVE C-form layout
// (P1's Vn output, killing from_vn) and emits dv straight into the C-form
// native dVn (killing the dVn conversion).
//
// Math (fla chunk_bwd.chunk_kda_bwd_kernel_dAv, dense):
//   A    = Aqk row-major bf16 [B,T,H,64], causal-masked col <= row (FLA leaves
//          intermittent garbage above the diagonal, so the mask is required)
//   dA   = (do @ v_new^T) * scale, fp32 row-major, kept where row >= col else 0
//   dv   = A^T @ do, emitted here directly in C-form native
// Two MMACs per (chunk, head).  v_new arrives as C-form native: the quad at
// sb*256 + c*4 holds v_new[t = 16*d0 + (c&15)][v = 16*d1 + (c>>4) + 4e] with
// sb = d1 + 8*d0, so contiguous 8B reads scatter into the [v][t] LDS staging.
#include "g2_common.h"

namespace g2 {

struct DavParams {
    const short* do_; const short* Vn; const short* A;
    float* dA; short* dVn;
    short* dOT; short* aNdo;      // T3: P2's dOT (B form) + P3's aN_do (A form, p3 order)
    float scale; int H, NT;
    const int* ctok; const int* clen;        // varlen phase 2 (null = dense)
};

constexpr int kPadJ3 = 68;

// VL (varlen phase 2): do / Aqk rows r >= clen read as 0 (= the zero padding; they only
// feed MMAC operands), dA rows r >= clen are not stored.  Natives keep NT chunks.
template <bool VL>
__global__ void __launch_bounds__(256) g2_dav_kernel(DavParams p) {
    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;
    const long t0 = g2_row_tok<VL>(p.ctok, i);
    __shared__ __align__(16) short slab[128 * kPadJ3];       // 8704 shorts:
    const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const size_t rstride = (size_t)p.H * kV;                  // do_ rows ([B,T,H,128])
        const size_t vbase = ((size_t)t0 * p.H + hh) * kV;
        const size_t nblk = (size_t)(hh * p.NT + i) * 8192;       // Vn / dVn native
        const size_t nblk3 = g2_nat_blk(i, hh, p.H, p.NT, true, 8192);   // aN_do's (i*H+hh)
        // pass 1 stages the B operand as [j][v] (n = j, k = v -- 132-stride, v spans
        // 128); pass 2 stages [v][t] (n = v, k = t -- 68-stride).  Both fit 17.4KB.
                                                                  // pass1 [j][132] = 8448,
                                                                  // pass2 [v][68] = 8704

        // ---- pass 1: dA = do @ vT, reduction over v (8 tiles), 4 j-tile outputs
        // (dA == nullptr skips the pass -- bisection hook, bindings never pass null)
        if (p.dA != nullptr)
        for (int t = tid; t < 2048; t += 256) {
            const int sb = t >> 6, c = t & 63;                    // 32 sub-blocks x 64 quads
            const v4bh vq = *reinterpret_cast<const v4bh*>(p.Vn + nblk + (size_t)sb * 256 + c * 4);
            const int d0 = sb >> 3, d1 = sb & 7, m = c & 15, q = c >> 4;
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                slab[(size_t)(16 * d0 + m) * (kK + 4) + 16 * d1 + q + 4 * e] = vq[e];
        }
        __syncthreads();
        if (p.dA != nullptr)
        {
            f4 acc[4];
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) acc[jt] = f4{0.f, 0.f, 0.f, 0.f};
            const int rdo = 16 * wv + (l & 15);
            const bool dok = g2_row_ok<M>(rdo, cl);
            for (int vs = 0; vs < 8; ++vs) {                      // reduction over v
                const v4bh df = dok ? *reinterpret_cast<const v4bh*>(
                    p.do_ + g2_rm_tok(t0 + rdo, hh, p.H, kV)
                    + 16 * vs + (l >> 4) * 4) : v4bh{0, 0, 0, 0};
                #pragma unroll
                for (int jt = 0; jt < 4; ++jt) {
                    const size_t lo = (size_t)(16 * jt + (l & 15)) * (kK + 4) + 16 * vs + (l >> 4) * 4;
                    acc[jt] = mmac(df, *reinterpret_cast<const v4bh*>(slab + lo), acc[jt]);
                }
            }
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const int t = 16 * wv + (l & 15), j0 = 16 * jt + (l >> 4);
                if (!g2_row_ok<M>(t, cl)) continue;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {           // C-frag col = (l>>4) + 4e
                    const float val = (t >= j0 + 4 * e) ? acc[jt][e] * p.scale : 0.f;
                    p.dA[g2_rm_tok(t0 + t, hh, p.H, kBT) + j0 + 4 * e] = val;
                }
            }
        }

        // ---- pass 2: dv = A^T @ do, reduction over t (4 tiles), 8 v-tile outputs.
        // doT[v][t] staged transposed; A^T fragments are gathered straight from the
        // row-major Aqk (4 scalar loads per lane, L1-hot) with the causal zeroing --
        // FLA's upper-triangle garbage must not leak in.
        __syncthreads();
        if (p.dVn != nullptr)
        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const v4bh dv_ = g2_row_ok<M>(j, cl)
                ? *reinterpret_cast<const v4bh*>(p.do_ + vbase + (size_t)j * rstride + vq)
                : v4bh{0, 0, 0, 0};
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                slab[(size_t)(vq + d) * kPadJ3 + j] = dv_[d];
            // T3: the SAME do loads also emit P2's dOT (B form, stride-4 quads, std
            // block order) and P3's aN_do (A form, plain row nibble, P3 block order)
            // -- the two B3.5 convert2 calls (which re-read do, ~200 MB @8K/48)
            // disappear.  Index math identical to g2_wu's verified Kn/KgA emits.
            *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;
            const int ibd = (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3)
                            + 256 * (j >> 4) + 1024 * (vq >> 4);
            #pragma unroll
            for (int d = 0; d < 4; ++d) p.dOT[nblk + ibd + 4 * d] = dv_[d];
        }
        __syncthreads();
        if (p.dVn != nullptr)
        {
            f4 acc[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
            const int tp = 16 * wv + (l & 15);                    // this lane's t' row
            for (int js = 0; js < 4; ++js) {                      // reduction over t
                v4bh af;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = 16 * js + 4 * (l >> 4) + e;
                    af[e] = (t >= tp && g2_row_ok<M>(t, cl)) ?           // causal keep
                        *reinterpret_cast<const short*>(
                            p.A + g2_rm_tok(t0 + t, hh, p.H, kBT) + tp) : 0;
                }
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const size_t lo = (size_t)(16 * vt + (l & 15)) * kPadJ3 + 16 * js + (l >> 4) * 4;
                    acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(slab + lo), acc[vt]);
                }
            }
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {                       // C-form: vec store
                v4bh x;
                #pragma unroll
                for (int e = 0; e < 4; ++e) x[e] = bf_rne(acc[vt][e]);
                *reinterpret_cast<v4bh*>(p.dVn + nblk + (size_t)(vt + 8 * wv) * 256 + l * 4) = x;
            }
        }
    });
}

}  // namespace g2

extern "C" int g2_dav(const void* do_, const void* Vn, const void* A,
                      void* dA, void* dVn, void* dOT, void* aNdo, double scale,
                      int B, int T, int H, const int* ctok, const int* clen,
                      hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen do/A/dA rows are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    g2::DavParams p{(const short*)do_, (const short*)Vn, (const short*)A,
                    (float*)dA, (short*)dVn, (short*)dOT, (short*)aNdo,
                    (float)scale, H, T / 64, ctok, clen};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
