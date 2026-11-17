// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/primitive/hygon/gdn2/primitive.hpp"
namespace ftrain {
namespace {
Result applicable(const Gdn2Problem& p, const Constraints& c) {
    if (!gdn2::deviceSupported(c.getDeviceId())) return {FTRAIN_STATUS_UNSUPPORTED, "GDN2 targets gfx936 only"};
    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto& v = p.view(i);
        if (v.getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS || reinterpret_cast<std::uintptr_t>(v.getMemory()) % 16)
            return {FTRAIN_STATUS_UNSUPPORTED, "GDN2 requires contiguous, 16-byte aligned ports"};
        auto type = v.getNumericType();
        if (i == 3 || i == 4 || i == 5) {
            if (type != FTRAIN_NUMERIC_TYPE_BF16 && type != FTRAIN_NUMERIC_TYPE_FP32)
                return {FTRAIN_STATUS_UNSUPPORTED, "GDN2 gates must be BF16 or FP32"};
        } else {
            FTrainNumericType expected =
                (i == std::size_t(p.backward ? 6 : 7)) ? FTRAIN_NUMERIC_TYPE_FP32 : FTRAIN_NUMERIC_TYPE_BF16;
            if (p.backward && i >= 11) expected = p.view(i - 8).getNumericType();
            if (type != expected) return {FTRAIN_STATUS_UNSUPPORTED, "GDN2 unsupported port dtype"};
        }
    }
    return {};
}
gdn2::Args arguments(const Gdn2Problem& p) {
    gdn2::Args a;
    a.q     = p.view(0).getMemory();
    a.k     = p.view(1).getMemory();
    a.v     = p.view(2).getMemory();
    a.g     = p.view(3).getMemory();
    a.b     = p.view(4).getMemory();
    a.w     = p.view(5).getMemory();
    a.gf32  = p.view(3).getNumericType() == FTRAIN_NUMERIC_TYPE_FP32;
    a.bf32  = p.view(4).getNumericType() == FTRAIN_NUMERIC_TYPE_FP32;
    a.wf32  = p.view(5).getNumericType() == FTRAIN_NUMERIC_TYPE_FP32;
    a.norm  = p.attributes.value.l2norm_qk != 0;
    a.scale = p.attributes.value.scale;
    a.cache = p.view(p.backward ? 6 : 7).getMemory();
    if (p.backward) {
        a.dout = p.view(7).getMemory();
        a.dq   = p.view(8).getMemory();
        a.dk   = p.view(9).getMemory();
        a.dv   = p.view(10).getMemory();
        a.dg   = p.view(11).getMemory();
        a.db   = p.view(12).getMemory();
        a.dw   = p.view(13).getMemory();
    } else
        a.out = p.view(6).getMemory();
    return a;
}
}  // namespace
Result Gdn2Forward::isApplicable(const Gdn2ForwardProblem& p, const Constraints& c) const { return applicable(p, c); }
void Gdn2Forward::configure(const Gdn2ForwardProblem& p) {
    shape_ = p.shape;
    args_  = arguments(p);
}
Result Gdn2Backward::isApplicable(const Gdn2BackwardProblem& p, const Constraints& c) const { return applicable(p, c); }
void Gdn2Backward::configure(const Gdn2BackwardProblem& p) {
    shape_ = p.shape;
    args_  = arguments(p);
}
}  // namespace ftrain
