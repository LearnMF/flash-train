// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_GDN2_HPP_
#define FTRAIN_OPERATION_GDN2_HPP_
#include <vector>
#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"
namespace ftrain {
struct Gdn2Attributes {
    explicit Gdn2Attributes(FTrainGdn2Attributes a) : value(a) {}
    FTrainGdn2Attributes value;
};
namespace detail {
inline std::vector<PatternOperandId> gdn2Tensors(std::initializer_list<FTrainTensorId> ids) {
    std::vector<PatternOperandId> result;
    for (auto id : ids) result.push_back(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id));
    return result;
}
}  // namespace detail
template<>
class OperationTraits<OperationKind::kGdn2Forward> {
  public:
    using Id   = FTrainGdn2ForwardOpId;
    using Type = Gdn2Attributes;
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }
    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId v,
                                                           FTrainTensorId g, FTrainTensorId b, FTrainTensorId w,
                                                           FTrainTensorId o, FTrainTensorId cache) {
        return {OperationKind::kGdn2Forward, detail::gdn2Tensors({q, k, v, g, b, w}), detail::gdn2Tensors({o, cache})};
    }
};
template<>
class OperationTraits<OperationKind::kGdn2Backward> {
  public:
    using Id   = FTrainGdn2BackwardOpId;
    using Type = Gdn2Attributes;
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }
    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId v,
                                                           FTrainTensorId g, FTrainTensorId b, FTrainTensorId w,
                                                           FTrainTensorId cache, FTrainTensorId d_o, FTrainTensorId dq,
                                                           FTrainTensorId dk, FTrainTensorId dv, FTrainTensorId dg,
                                                           FTrainTensorId db, FTrainTensorId dw) {
        return {OperationKind::kGdn2Backward, detail::gdn2Tensors({q, k, v, g, b, w, cache, d_o}),
                detail::gdn2Tensors({dq, dk, dv, dg, db, dw})};
    }
};
}  // namespace ftrain
#endif
