// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_OPERATION_KDA_HPP_
#define FTRAIN_OPERATION_KDA_HPP_

#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

// Kimi Delta Attention parameters shared by the forward and backward operations. Stores
// already-validated values and performs no validation.
class KdaAttributes final {
  public:
    explicit KdaAttributes(const FTrainKdaAttributes& attributes) noexcept : attributes_(attributes) {}

    float getScale() const noexcept { return attributes_.scale; }
    float getLowerBound() const noexcept { return attributes_.lower_bound; }
    bool isGateInKernel() const noexcept { return attributes_.gate_in_kernel != 0; }
    bool isL2NormQk() const noexcept { return attributes_.l2norm_qk != 0; }

  private:
    FTrainKdaAttributes attributes_;
};

namespace detail {
inline std::vector<PatternOperandId> kdaTensors(std::initializer_list<FTrainTensorId> ids) {
    std::vector<PatternOperandId> out;
    out.reserve(ids.size());
    for (const FTrainTensorId id : ids) {
        out.push_back(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id));
    }
    return out;
}
}  // namespace detail

template<>
class OperationTraits<OperationKind::kKdaForward> {
  public:
    using Id   = FTrainKdaForwardOpId;
    using Type = KdaAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    // Inputs q, k, v, g, beta, a_log, dt_bias, cu_seqlens; outputs o and the saved
    // tensors q_rstd, k_rstd, aqk, akk the backward consumes.
    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId v,
                                                           FTrainTensorId g, FTrainTensorId beta, FTrainTensorId a_log,
                                                           FTrainTensorId dt_bias, FTrainTensorId cu_seqlens,
                                                           FTrainTensorId o, FTrainTensorId q_rstd,
                                                           FTrainTensorId k_rstd, FTrainTensorId aqk,
                                                           FTrainTensorId akk) {
        return PatternOperationNode{OperationKind::kKdaForward,
                                    detail::kdaTensors({q, k, v, g, beta, a_log, dt_bias, cu_seqlens}),
                                    detail::kdaTensors({o, q_rstd, k_rstd, aqk, akk})};
    }
};

template<>
class OperationTraits<OperationKind::kKdaBackward> {
  public:
    using Id   = FTrainKdaBackwardOpId;
    using Type = KdaAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    // Inputs d_o, the forward's inputs, and its saved tensors; outputs the gradients of
    // q, k, v, g, beta, a_log, dt_bias.
    static PatternOperationNode createPatternOperationNode(
        FTrainTensorId d_o, FTrainTensorId q, FTrainTensorId k, FTrainTensorId v, FTrainTensorId g, FTrainTensorId beta,
        FTrainTensorId a_log, FTrainTensorId dt_bias, FTrainTensorId cu_seqlens, FTrainTensorId q_rstd,
        FTrainTensorId k_rstd, FTrainTensorId aqk, FTrainTensorId akk, FTrainTensorId dq, FTrainTensorId dk,
        FTrainTensorId dv, FTrainTensorId dg, FTrainTensorId dbeta, FTrainTensorId da_log, FTrainTensorId ddt_bias) {
        return PatternOperationNode{
            OperationKind::kKdaBackward,
            detail::kdaTensors({d_o, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk}),
            detail::kdaTensors({dq, dk, dv, dg, dbeta, da_log, ddt_bias})};
    }
};

}  // namespace ftrain

#endif
