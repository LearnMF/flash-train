// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_GDN2_RUNNER_HPP_
#define FTRAIN_GDN2_RUNNER_HPP_
#include <cstdint>
#include "flash_train/common.h"
namespace ftrain::gdn2 {
constexpr int C = 64;
struct Shape {
    int B, T, H, HV, K, V;
    __host__ __device__ int chunks() const { return (T + C - 1) / C; }
    std::uint64_t blocks() const { return std::uint64_t(B) * HV * chunks(); }
};
// Each chunk/head owns a contiguous FP32 slab. No inverse decay is formed.
struct Layout {
    std::uint64_t q, k, g, Q, W, E, P, M, U, X, S, norm, size;
    explicit Layout(const Shape& s) {
        const std::uint64_t ck = C * s.K, cv = C * s.V, cc = C * C;
        q    = 0;
        k    = q + ck;
        g    = k + ck;
        Q    = g + ck;
        W    = Q + ck;
        E    = W + ck;
        P    = E + ck;
        M    = P + cc;
        U    = M + cc;
        X    = U + cv;
        S    = X + cv;
        norm = S + s.K * s.V;
        size = norm + 2 * C;
    }
};
struct GradLayout {
    std::uint64_t dS, Y, dQ, dR, dE, dP, dA, dq, dk, size;
    explicit GradLayout(const Shape& s) {
        const std::uint64_t ck = C * s.K, cc = C * C;
        dS   = 0;
        Y    = s.K * s.V;
        dQ   = Y + C * s.V;
        dR   = dQ + ck;
        dE   = dR + ck;
        dP   = dE + ck;
        dA   = dP + cc;
        dq   = dA + cc;
        dk   = dq + ck;
        size = dk + ck;
    }
};
struct Args {
    const void *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr, *b = nullptr, *w = nullptr, *dout = nullptr;
    void *out = nullptr, *cache = nullptr, *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr,
         *dw  = nullptr;
    bool gf32 = false, bf32 = false, wf32 = false, norm = false;
    float scale = 1;
};
inline std::uint64_t savedElements(const Shape& s) { return s.blocks() * Layout(s).size; }
inline std::uint64_t backwardBytes(const Shape& s) { return s.blocks() * GradLayout(s).size * sizeof(float); }
bool deviceSupported(FTrainDeviceId device) noexcept;
void forward(const Shape&, const Args&, FTrainStream);
void backward(const Shape&, const Args&, void*, FTrainStream);
}  // namespace ftrain::gdn2
#endif
