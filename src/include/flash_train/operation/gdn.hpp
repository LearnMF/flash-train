// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_GDN_HPP_
#define FTRAIN_OPERATION_GDN_HPP_

#include <vector>

#include "flash_train/gdn.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

// Gated Delta Net parameters shared by the forward and backward operations. Stores
// already-validated values and performs no validation.
class GdnAttributes final {
  public:
    explicit GdnAttributes(const FTrainGdnAttributes& attributes) noexcept : attributes_(attributes) {}

    float getScale() const noexcept { return attributes_.scale; }
    bool isL2NormQk() const noexcept { return attributes_.l2norm_qk != 0; }

  private:
    FTrainGdnAttributes attributes_;
};

namespace detail {
inline std::vector<PatternOperandId> gdnTensors(std::initializer_list<FTrainTensorId> ids) {
    std::vector<PatternOperandId> out;
    out.reserve(ids.size());
    for (const FTrainTensorId id : ids) {
        out.push_back(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id));
    }
    return out;
}
}  // namespace detail

template<>
class OperationTraits<OperationKind::kGdnForward> {
  public:
    using Id   = FTrainGdnForwardOpId;
    using Type = GdnAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId v,
                                                           FTrainTensorId g, FTrainTensorId beta,
                                                           FTrainTensorId cu_seqlens, FTrainTensorId o,
                                                           FTrainTensorId q_rstd, FTrainTensorId k_rstd,
                                                           FTrainTensorId aqk, FTrainTensorId akk, FTrainTensorId qn,
                                                           FTrainTensorId kn, FTrainTensorId h) {
        return PatternOperationNode{OperationKind::kGdnForward, detail::gdnTensors({q, k, v, g, beta, cu_seqlens}),
                                    detail::gdnTensors({o, q_rstd, k_rstd, aqk, akk, qn, kn, h})};
    }
};

template<>
class OperationTraits<OperationKind::kGdnBackward> {
  public:
    using Id   = FTrainGdnBackwardOpId;
    using Type = GdnAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId d_o, FTrainTensorId q, FTrainTensorId k,
                                                           FTrainTensorId v, FTrainTensorId g, FTrainTensorId beta,
                                                           FTrainTensorId cu_seqlens, FTrainTensorId q_rstd,
                                                           FTrainTensorId k_rstd, FTrainTensorId aqk,
                                                           FTrainTensorId akk, FTrainTensorId qn, FTrainTensorId kn,
                                                           FTrainTensorId dq, FTrainTensorId dk, FTrainTensorId dv,
                                                           FTrainTensorId dg, FTrainTensorId dbeta, FTrainTensorId h) {
        return PatternOperationNode{
            OperationKind::kGdnBackward,
            detail::gdnTensors({d_o, q, k, v, g, beta, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, h}),
            detail::gdnTensors({dq, dk, dv, dg, dbeta})};
    }
};

}  // namespace ftrain

#endif
