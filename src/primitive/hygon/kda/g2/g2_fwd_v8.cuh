// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 P1 v8: fused output + optional bN(h) emission (REVIEW S3-6 step 2 item 1).
//
// Extends g2_fwd_h_v7_kernel<8> (same recurrence, same numerics for h/v_new):
//   * o fused: each wave owns the [16x16] o block (t-tile r1, v-tile v1a) of its
//     CTA's 32-column slice; per chunk step
//         o_block = (qg @ h_i) * scale + Aqk @ v_new_i
//     with qg / Aqk supplied A-fragment-native (same layout family as Wn; Aqk is the
//     64-wide variant).  h is read from the shared hB exactly where GEMM1 reads it
//     (B operand, col=v d=k) and v_new from vB (col=v d=t) -- zero extra staging.
//     The forward pass no longer writes h and no longer calls FLA's o kernel.
//   * bN emit (backward recompute path): bN slot (kt, vt, l, e) holds
//     h[16*kt + pi(m)][16*vt + 4q + e] (DESIGN section 4; verified in
//     scripts/verify_ref.py).  After the barrier where hB holds h_i complete, wave
//     wv emits the kt=wv blocks for its CTA's two global v-tiles straight from hB,
//     replacing from_native_h + to_bN (the non-coalesced 8-D permutes that dominated
//     the S3-5 wall-clock breakdown).
//
// Register note: qg/Aqk fragment loads use dedicated arrays; the recurrence's wf[]
// (W operands) stay resident as in v7.
#include "g2_common.h"

namespace g2 {

struct FwdOParams {
    const bf16 *Wn, *Kn, *Un; const float* Gn;
    const bf16* QgA;                                   // bf16(q * 2^g), A-native (128-wide)
    const bf16* AqkA;                                  // FLA Aqk (scale in), A-native (64-wide)
    float* O;                                          // [B,T,H,128] fp32, null = skip
    bf16* Ob;                                          // T9a: [B,T,H,128] bf16, null = skip
    bf16* Vn;
    bf16* HbN;                                         // bN(h), null = skip
    bf16* ANv;                                         // aN(v_new), null = skip
    float scale; int H, NT;
    // Varlen/batch (DESIGN_VARLEN_BATCH.md): chunk-aligned segments.  seg[0..nseg] are
    // chunk offsets; null = one segment [0, NT).  grid.y = nseg * H.
    const int* seg; int nseg;
    // Varlen phase 2: packed o rows (chunk i -> tokens ctok[i] .. +clen[i]); null = dense.
    const int* ctok; const int* clen;
};

// T9a: wave-scope LDS ordering -- wave-private oStage regions need the compiler
// fence + lgkmcnt drain only, NOT a CTA barrier (S3-16 lesson: a barrier-carrying
// wave sync deadlocks when called from wave-divergent code; here all lanes of a
// wave participate, but s_barrier would still serialize the 8 waves needlessly).
__device__ __forceinline__ void o_wave_order() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

template <bool VL>
__global__ void __launch_bounds__(512) g2_fwd_h_v8o_kernel(FwdOParams p) {
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    __shared__ __align__(16) bf16 oStage[8][256];   // T9a: wave-private 16x16 bf16
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;                              // empty segment (whole CTA exits)
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv & 3, v1a = wv >> 2;
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;               // token stride of [T,H,128]
    const size_t tok0 = chunk * kBT * rowT;             // chunk row 0, whole [H,128] block

    f4 ht[2];
    ht[0] = f4{0.f,0.f,0.f,0.f}; ht[1] = ht[0];
    v4bh wf[8], kf[4], uf, qgf[8], aqf[4];
    f4 gl;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) wf[kt] = ld8(src + kt * 256);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) qgf[kt] = ld8(src + kt * 256);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 4;
        #pragma unroll
        for (int st = 0; st < 4; ++st) aqf[st] = ld8(src + st * 256);
    };
    auto loadU = [&](int i) {
        uf = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4);
    };
    auto loadG = [&](int i) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + wv) * 16 + q4 * 4);
    };
    auto loadK = [&](int i) {
        #pragma unroll
        for (int rt = 0; rt < 4; ++rt)
            kf[rt] = ld8(p.Kn + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
    };
    v4bh hx[2];
    auto write_h = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(ht[vt][e]);
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };

    loadW(c0); loadU(c0); loadG(c0); loadK(c0);
    write_h();
    for (int i = c0; i < c1; ++i) {                    // h = 0 at the segment start
        // varlen: issue the chunk's row-map scalar loads up front (latency hidden by the
        // GEMMs; issued at the o store they sat on the per-step critical path)
        const long otok = g2_row_tok<VL>(p.ctok, i);
        const int olen = g2_row_len<VL>(p.clen, i);
        const int nxt = i + 1 < c1 ? i + 1 : i;
        lds_barrier();                                   // hB (h_i) ready
        // ---- optional bN(h_i) emit: wave wv writes blocks kt=wv, vt in {2vs, 2vs+1}
        if (p.HbN != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                // slot (l, e) = h[16*wv + pi(m)][16*(2vs+vt) + 4*q4 + e]
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * wv + pi16(m)];
                // bN block order is P3's (B,NT,H): block = i*H + bh (B=1)
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;
            }
        }
        // ---- GEMM1: v_new = u - W @ h_i   (identical to v7)
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(wf[kt], ld8(hB + (16 * v1a + m) * kPadK + 16 * kt + q4 * 4), acc);
        loadW(nxt);
        {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float uv = __uint_as_float(((uint32_t)(unsigned short)uf[e]) << 16);
                x[e] = bf_rne(uv - acc[e]);
                reinterpret_cast<short*>(vB)[(16 * v1a + c_col(l, e)) * kPadT + 16 * r1 + m] = x[e];
            }
            *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4) = x;
            if (p.ANv != nullptr)   // aN(v_new): A form, P3 block order; the C-frag
                #pragma unroll      // col (l>>4)+4e swaps into 64*e + (l>>4)
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1a) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        loadU(nxt);
        {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[e]);
            ht[0] *= d; ht[1] *= d;
        }
        loadG(nxt);
        // ---- fused output, part 1: oh = qg @ h_i  BEFORE the second barrier.  hB is
        // written only by write_h at loop end, so the second barrier itself protects
        // these reads exactly the way it protects GEMM1's; after the barrier a fast
        // wave's write_h would race a slow wave's hB reads (h_1 leaking into o_i -- the
        // single-chunk test catches it: h_0 = 0 yet o came out 0.017).
        f4 oh = f4{0.f,0.f,0.f,0.f};
        if (p.O != nullptr || p.Ob != nullptr) {
            loadQg(i); loadAq(i);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                oh = mmac(qgf[kt], ld8(hB + (16 * v1a + m) * kPadK + 16 * kt + q4 * 4), oh);
        }
        lds_barrier();                                   // vB (v_new_i) ready
        // ---- fused output, part 2: oa = Aqk @ v_new_i (vB reads only: the next vB
        // writer is the next iteration's scatter, which sits behind the next barrier)
        if (p.O != nullptr || p.Ob != nullptr) {
            f4 oa = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);
            // o = (qg @ h) * scale + Aqk @ v_new (FLA chunk_gla_fwd_kernel_o:
            // b_o *= scale BEFORE += dot(A, v); Aqk carries scale from fwd_intra).
            // T9a: the bf16 path consumes the SAME fp32 value (fusion status of
            // this mul+add must not flip -- R-bitwise gate arbitrates).
            f4 o;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = oh[e] * p.scale + oa[e];
            // C fragment: row = 16*r1 + m (token), col = 16*v1a + q4 + 4e (within slice)
            const int orow = 16 * r1 + m;                 // row within the chunk
            const bool ook = g2_row_ok<VL ? 2 : 0>(orow, olen);   // masked mode
            const size_t tok = (size_t)(otok + orow);
            const size_t obase = tok * rowT + (size_t)bh * kK + 32 * vs + 16 * v1a;
            if (p.O != nullptr) {
                if (ook) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        p.O[obase + q4 + 4 * e] = o[e];
                }
            } else {
                // T9a: bf16 o via the wave-private LDS transpose.  Lane (m, q4)
                // scatters its 4 strided C-fragment columns into oStage[wv]; after
                // the wave-scope ordering, lane (m', g') reads the 4 CONSECUTIVE
                // columns as one 8B quad and stores one 8B vector to global
                // (4 scalar 2B stores are forbidden by the task).
                bf16* st = oStage[wv];
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(st)[m * 16 + q4 + 4 * e] = bf_rne(o[e]);
                o_wave_order();
                const int m2 = l & 15, g2_ = l >> 4;
                const v4bh ov = *reinterpret_cast<const v4bh*>(
                        reinterpret_cast<const short*>(st) + m2 * 16 + 4 * g2_);
                if (ook) *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * g2_) = ov;
            }
        }
        // ---- GEMM2: h update (identical to v7)
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf[rt], upd);
            }
            ht[vt] += upd;
        }
        loadK(nxt);
        write_h();
    }
}

extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                            const void* QgA, const void* AqkA, void* O, void* Ob, void* Vn,
                            void* HbN, void* ANv,
                            float scale, int B, int T, int H, const int* seg, int nseg,
                            const int* ctok, const int* clen, hipStream_t s) {
    // T is the chunk-padded row count (NT = T / 64); with ctok/clen the o rows are packed.
    if (T % kBT || B != 1) return -1;                  // batch = segments (seg/nseg)
    if (seg == nullptr) nseg = 1;
    if ((ctok == nullptr) != (clen == nullptr)) return -2;
    FwdOParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (const bf16*)QgA, (const bf16*)AqkA, (float*)O, (bf16*)Ob, (bf16*)Vn,
                 (bf16*)HbN, (bf16*)ANv,
                 scale, H, T / kBT, seg, nseg, ctok, clen};
    if (ctok != nullptr)
        hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
    else
        hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

}  // namespace g2
