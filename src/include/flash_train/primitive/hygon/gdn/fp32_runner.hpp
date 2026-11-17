// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN_FP32_RUNNER_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN_FP32_RUNNER_HPP_

#include <cstdint>
#include "flash_train/common.h"
#include "flash_train/gdn.h"

namespace ftrain::gdn_fp32 {

struct Shape {
    std::int64_t tokens = 0, heads = 0, seqs = 1;
    bool has_cu_seqlens = false, l2norm_qk = false;
    bool has_saved_h = false, has_qkn = false;
    std::uint64_t stateElements() const noexcept {
        return ftrainGdnSavedHElements(tokens, heads, seqs, has_cu_seqlens);
    }
};

struct ForwardArgs {
    const void *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr;
    const int* cu_seqlens = nullptr;
    void *o = nullptr, *q_rstd = nullptr, *k_rstd = nullptr;
    void *qn = nullptr, *kn = nullptr, *saved_h = nullptr;
    float scale = 1;
};

struct BackwardArgs {
    const void *d_o = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr;
    const int* cu_seqlens = nullptr;
    const void *q_rstd = nullptr, *k_rstd = nullptr, *qn = nullptr, *kn = nullptr, *saved_h = nullptr;
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *dbeta = nullptr;
    float scale = 1;
};

std::uint64_t forwardWorkspaceBytes(const Shape&) noexcept;
std::uint64_t backwardWorkspaceBytes(const Shape&) noexcept;
void forward(const Shape&, const ForwardArgs&, void*, FTrainStream);
void backward(const Shape&, const BackwardArgs&, void*, FTrainStream);

}  // namespace ftrain::gdn_fp32
#endif
