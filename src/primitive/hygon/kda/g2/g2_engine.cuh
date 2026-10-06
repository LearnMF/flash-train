// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 engine P1 (forward state recurrence) + P2 (reverse state recurrence), gfx936.
//
// Verbatim extraction of the two verified prototypes:
//   deepseek/handoff_20260925_G2_integration/src/prototypes/g2_engine.hip
//     P1 = g2_fwd_h_v7_kernel<8> / g2_fwd_h_v7w8
//     P2 = g2_dhu_w8b_kernel    / g2_dhu_w8b
// sha256(ca9f6fa3...) -- kernel bodies are copied without semantic change; only the
// surrounding declarations moved into g2_common.h.
//
// Layout contract (native fragments, see DESIGN.md section 4):
//   Wn  [BH][NT][r4][kt8][lane][4]   A(m=t,d=k)     Kn [BH][NT][kt8][rt4][lane][4]  B(d=t,n=k)
//   Un  [BH][NT][r4][VT8][lane][4]   C(m=t,n=v)     Gn [BH][NT][kt8][q4][e4] fp32  (k=16kt+q+4e)
//   Hn  [BH][NT][VT8][KT8][lane][4]  C(m=v,n=k)     Vn like Un
// Block order is (B, H, NT): head-major, chunk-minor.  P3 in g2_wy.cuh uses (NT, H).
#include "g2_common.h"

namespace g2 {

struct FwdNParams {
    const bf16* Wn; const bf16* Kn; const bf16* Un; const float* Gn;
    bf16* Hn; bf16* Vn; int H, NT;
};

// ---------------------------------------------------------------- P1
template <int NW>
__global__ void __launch_bounds__(64 * NW) g2_fwd_h_v7_kernel(FwdNParams p) {
    constexpr int KPW = 8 / NW;                // state k-tiles per wave: 2 (NW=4) or 1 (NW=8)
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x, bh = blockIdx.y;
    const size_t chunk = (size_t)bh * p.NT;
    // GEMM1 ownership: t-tile r1, v-tiles [v1a, v1a + NV1)
    constexpr int NV1 = NW == 4 ? 2 : 1;
    const int r1 = NW == 4 ? wv : (wv & 3);
    const int v1a = NW == 4 ? 0 : (wv >> 2);
    f4 ht[2][KPW];
    #pragma unroll
    for (int a = 0; a < 2; ++a) for (int b = 0; b < KPW; ++b) ht[a][b] = f4{0.f, 0.f, 0.f, 0.f};
    v4bh wf[8], kf[KPW][4], uf[NV1]; f4 gl[KPW];
    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) wf[kt] = ld8(src + kt * 256);
    };
    auto loadU = [&](int i) {
        #pragma unroll
        for (int j = 0; j < NV1; ++j)
            uf[j] = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4);
    };
    auto loadG = [&](int i) {
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl)
            gl[kl] = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + KPW * wv + kl) * 16 + (l >> 4) * 4);
    };
    auto loadK = [&](int i) {
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl)
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt)
                kf[kl][rt] = ld8(p.Kn + (((chunk + i) * 8 + KPW * wv + kl) * 4 + rt) * 256 + l * 4);
    };
    v4bh hx[2][KPW];
    auto write_h = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int kl = 0; kl < KPW; ++kl)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    hx[vt][kl][e] = bf_rne(ht[vt][kl][e]);
                    reinterpret_cast<short*>(hB)[(16 * vt + c_row(l)) * kPadK + 16 * (KPW * wv + kl) + c_col(l, e)] = hx[vt][kl][e];
                }
    };
    loadW(0); loadU(0); loadG(0); loadK(0);
    write_h();
    for (int i = 0; i < p.NT; ++i) {
        const int nxt = i + 1 < p.NT ? i + 1 : i;
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int kl = 0; kl < KPW; ++kl)
                *reinterpret_cast<v4bh*>(p.Hn + (((chunk + i) * 8 + 2 * vs + vt) * 8 + KPW * wv + kl) * 256 + l * 4) = hx[vt][kl];
        lds_barrier();
        f4 acc[NV1];
        #pragma unroll
        for (int j = 0; j < NV1; ++j) acc[j] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int j = 0; j < NV1; ++j)
                acc[j] = mmac(wf[kt], ld8(hB + (16 * (v1a + j) + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc[j]);
        loadW(nxt);
        #pragma unroll
        for (int j = 0; j < NV1; ++j) {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float uv = __uint_as_float(((uint32_t)(unsigned short)uf[j][e]) << 16);
                x[e] = bf_rne(uv - acc[j][e]);
                reinterpret_cast<short*>(vB)[(16 * (v1a + j) + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4) = x;
        }
        loadU(nxt);
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl) {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[kl][e]);
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) ht[vt][kl] *= d;
        }
        loadG(nxt);
        lds_barrier();
        #pragma unroll
        for (int rt = 0; rt < 4; ++rt)
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                const v4bh va = ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4);
                #pragma unroll
                for (int kl = 0; kl < KPW; ++kl) ht[vt][kl] = mmac(va, kf[kl][rt], ht[vt][kl]);
            }
        loadK(nxt);
        write_h();
    }
}
extern "C" int g2_fwd_h_v7(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                           void* Hn, void* Vn, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    FwdNParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (bf16*)Hn, (bf16*)Vn, H, T / kBT};
    hipLaunchKernelGGL(g2_fwd_h_v7_kernel<4>, dim3(kK / kBV, B * H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
extern "C" int g2_fwd_h_v7w8(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                             void* Hn, void* Vn, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    FwdNParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (bf16*)Hn, (bf16*)Vn, H, T / kBT};
    hipLaunchKernelGGL(g2_fwd_h_v7_kernel<8>, dim3(kK / kBV, B * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

// ================================================================ P2: reverse engine (FLA dhu role)
// per chunk i = NT-1 .. 0, per (seq, head, V-slice of 32), 8 waves:
//   dh_out[i] = dh                                                  (bf16)
//   dv2       = Kg dh + dv                                          (stored bf16)
//   dh        = diag(2^{gk_last}) dh + scale * Qg^T dO - W^T dv2
// Fragment-native operands (our own producers choose them):
//   KgA [BH][NT][r4][kt8][lane][4]  A(m=t,d=k)    Qg,Wb [BH][NT][kt8][rt4][lane][4]  B(d=t,n=k)
//   dOT [BH][NT][VT8][rt4][lane][4] A(m=v,d=t)    dv, dv2 [BH][NT][r4][VT8][lane][4]  C(m=t,n=v)
//   Gn  [BH][NT][kt8][q4][e4]                     dh_out [BH][NT][VT8][KT8][lane][4]  C(m=v,n=k)
struct BwdNParams {
    const bf16* KgA; const bf16* Qg; const bf16* Wb; const bf16* dOT; const bf16* dv; const float* Gn;
    bf16* dh; bf16* dv2; float scale; int H, NT;
    const bf16* HbN;                                  // REVIEW S3-6: bN(dh) direct emit (null = off)
    bf16* ANv2;                                       // S3-10: aN(dv2) direct emit (null = off)
    // Varlen/batch (DESIGN_VARLEN_BATCH.md): chunk-aligned segments, as in FwdOParams
    // (dhu_v8b only; the w8b variant ignores them and keeps a single segment).
    const int* seg; int nseg;
};
extern "C" __global__ void __launch_bounds__(512) g2_dhu_w8b_kernel(BwdNParams p) {
    __shared__ __align__(16) bf16 dhB[kBV * kPadK];      // dh as [v][k]   (GEMM-A B operand)
    __shared__ __align__(16) bf16 vB[kBV * kPadT];       // dv2 as [v][t]  (GEMM-B A operand)
    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x, bh = blockIdx.y;
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv & 3, v1 = wv >> 2;                 // GEMM-A tile: t-tile r1, v-tile v1
    f4 st[2];                                            // DH^T C-frags [vt]: v=16vt+c_row, k=16wv+c_col
    st[0] = f4{0.f, 0.f, 0.f, 0.f}; st[1] = st[0];
    v4bh ka[8], qb[4], wb[4], dof, dvf; f4 gl;       // dof: this wave's 1/8 of dO^T
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) ka[kt] = ld8(src + kt * 256);
    };
    auto loadDV = [&](int i) { dvf = ld8(p.dv + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4); };
    auto loadG = [&](int i) { gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + wv) * 16 + (l >> 4) * 4); };
    auto loadQW = [&](int i) {
        #pragma unroll
        for (int rt = 0; rt < 4; ++rt) {
            qb[rt] = ld8(p.Qg + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
            wb[rt] = ld8(p.Wb + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
        }
    };
    auto loadDO = [&](int i) {
        dof = ld8(p.dOT + (((chunk + i) * 8 + 2 * vs + (wv >> 2)) * 4 + (wv & 3)) * 256 + l * 4);
    };
    v4bh hx[2];
    auto write_dh = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };
    const int last = p.NT - 1;
    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
    for (int s = 0; s < p.NT; ++s) {
        const int i = last - s, nxt = i > 0 ? i - 1 : 0;
        if (p.dh != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + wv) * 256 + l * 4) = hx[vt];
        }
        dOs[s & 1][wv * 64 + l] = dof;                   // publish chunk i's dO^T slice
        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (p.HbN != nullptr) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
        loadKA(nxt);
        {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = acc[e] + __uint_as_float(((uint32_t)(unsigned short)dvf[e]) << 16);
                x[e] = bf_rne(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4) = x;
        }
        loadDV(nxt);
        {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[e]);
            st[0] *= d; st[1] *= d;
        }
        loadG(nxt);
        lds_barrier();                                   // vB ready, dhB reads done
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb[rt], wd);
            }
            st[vt] += qd * p.scale - wd;
        }
        loadQW(nxt);
        write_dh();
    }
}

extern "C" __global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {
    __shared__ __align__(16) bf16 dhB[kBV * kPadK];      // dh as [v][k]   (GEMM-A B operand)
    __shared__ __align__(16) bf16 vB[kBV * kPadT];       // dv2 as [v][t]  (GEMM-B A operand)
    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;                                // empty segment
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv & 3, v1 = wv >> 2;                 // GEMM-A tile: t-tile r1, v-tile v1
    f4 st[2];                                            // DH^T C-frags [vt]: v=16vt+c_row, k=16wv+c_col
    st[0] = f4{0.f, 0.f, 0.f, 0.f}; st[1] = st[0];
    v4bh ka[8], qb[4], wb[4], dof, dvf; f4 gl;       // dof: this wave's 1/8 of dO^T
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 4;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) ka[kt] = ld8(src + kt * 256);
    };
    auto loadDV = [&](int i) { dvf = ld8(p.dv + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4); };
    auto loadG = [&](int i) { gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + wv) * 16 + (l >> 4) * 4); };
    auto loadQW = [&](int i) {
        #pragma unroll
        for (int rt = 0; rt < 4; ++rt) {
            qb[rt] = ld8(p.Qg + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
            wb[rt] = ld8(p.Wb + (((chunk + i) * 8 + wv) * 4 + rt) * 256 + l * 4);
        }
    };
    auto loadDO = [&](int i) {
        dof = ld8(p.dOT + (((chunk + i) * 8 + 2 * vs + (wv >> 2)) * 4 + (wv & 3)) * 256 + l * 4);
    };
    v4bh hx[2];
    auto write_dh = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };
    const int last = c1 - 1;                             // dh = 0 at the segment end
    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
    {
        const int i = last;
        lds_barrier();
        // bN(dh) emit: dhB holds this wave's k-tile after write_dh; slot (kt=wv, vt,
        // l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (p.HbN != nullptr) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
    }
    for (int s = 0; s < c1 - c0; ++s) {
        const int i = last - s, nxt = i > c0 ? i - 1 : c0;
        if (p.dh != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + wv) * 256 + l * 4) = hx[vt];
        }
        dOs[s & 1][wv * 64 + l] = dof;                   // publish chunk i's dO^T slice
        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (p.HbN != nullptr) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
        loadKA(nxt);
        {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = acc[e] + __uint_as_float(((uint32_t)(unsigned short)dvf[e]) << 16);
                x[e] = bf_rne(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            if (p.dv2 != nullptr)  // T9b: the global dV2n write is optional; the
                *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4) = x;
            if (p.ANv2 != nullptr)  // aN(dv2): same C-frag swap as the engines' aN emits
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv2)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        loadDV(nxt);
        {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = exp2f(gl[e]);
            st[0] *= d; st[1] *= d;
        }
        loadG(nxt);
        lds_barrier();                                   // vB ready, dhB reads done
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb[rt], wd);
            }
            st[vt] += qd * p.scale - wd;
        }
        loadQW(nxt);
        write_dh();
    }
}

extern "C" int g2_dhu_w8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,
                          const void* Gn, void* dh, void* dv2, float scale, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    BwdNParams p{(const bf16*)KgA, (const bf16*)Qg, (const bf16*)Wb, (const bf16*)dOT, (const bf16*)dv,
                 (const float*)Gn, (bf16*)dh, (bf16*)dv2, scale, H, T / kBT};
    hipLaunchKernelGGL(g2_dhu_w8b_kernel, dim3(kK / kBV, B * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,
                          const void* Gn, void* dv2, void* HbN, void* ANv2, float scale, int B, int T, int H,
                          const int* seg, int nseg, hipStream_t s) {
    if (T % kBT || B != 1) return -1;                    // batch = segments (seg/nseg)
    if (seg == nullptr) nseg = 1;
    BwdNParams p{(const bf16*)KgA, (const bf16*)Qg, (const bf16*)Wb, (const bf16*)dOT, (const bf16*)dv,
                 (const float*)Gn, nullptr, (bf16*)dv2, scale, H, T / kBT, (const bf16*)HbN,
                 (bf16*)ANv2, seg, nseg};
    hipLaunchKernelGGL(g2_dhu_v8b_kernel, dim3(kK / kBV, nseg * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

}  // namespace g2
