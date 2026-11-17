// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_GDN_FAMILY_HPP_
#define FTRAIN_FAMILY_GDN_FAMILY_HPP_

#include <array>
#include <memory>
#include <vector>

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

namespace detail {
inline Tensor gdnPort(const Args& args, FTrainTensorId id) {
    return std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id)));
}
}  // namespace detail

// Gated Delta Net forward: o = GDN(q, k, v, g, beta) plus the tensors its backward
// needs (see GdnForwardProblem for the port contract).
struct GdnForwardFamily {
    using Problem = GdnForwardProblem;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, GdnForwardProblem::kCount> ports;
        FTrainGdnForwardOpId op;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, GdnForwardProblem::kCount> ports{};
            for (FTrainTensorId& id : ports) { id = pattern.addOperand<OperandKind::kTensor>(); }
            using P                       = GdnForwardProblem;
            const FTrainGdnForwardOpId op = pattern.addOperation<OperationKind::kGdnForward>(
                ports[P::kQ], ports[P::kK], ports[P::kV], ports[P::kG], ports[P::kBeta], ports[P::kCuSeqlens],
                ports[P::kO], ports[P::kQRstd], ports[P::kKRstd], ports[P::kAqk], ports[P::kAkk], ports[P::kQn],
                ports[P::kKn], ports[P::kH]);
            return Roles{pattern.buildPattern(), ports, op};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<GdnForwardProblem>>> makeRecords();

    static GdnForwardProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        std::vector<Tensor> ports;
        ports.reserve(GdnForwardProblem::kCount);
        for (const FTrainTensorId id : roles.ports) { ports.push_back(detail::gdnPort(args, id)); }
        return GdnForwardProblem{std::move(ports),
                                 std::get<GdnAttributes>(*args.getOpArgument(
                                     OperationTraits<OperationKind::kGdnForward>::createPatternOperationId(roles.op)))};
    }
};

// GDN backward from d_o and saved tensors.
struct GdnBackwardFamily {
    using Problem = GdnBackwardProblem;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, GdnBackwardProblem::kCount> ports;
        FTrainGdnBackwardOpId op;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, GdnBackwardProblem::kCount> ports{};
            for (FTrainTensorId& id : ports) { id = pattern.addOperand<OperandKind::kTensor>(); }
            using P                        = GdnBackwardProblem;
            const FTrainGdnBackwardOpId op = pattern.addOperation<OperationKind::kGdnBackward>(
                ports[P::kDO], ports[P::kQ], ports[P::kK], ports[P::kV], ports[P::kG], ports[P::kBeta],
                ports[P::kCuSeqlens], ports[P::kQRstd], ports[P::kKRstd], ports[P::kAqk], ports[P::kAkk], ports[P::kQn],
                ports[P::kKn], ports[P::kDq], ports[P::kDk], ports[P::kDv], ports[P::kDg], ports[P::kDBeta],
                ports[P::kH]);
            return Roles{pattern.buildPattern(), ports, op};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<GdnBackwardProblem>>> makeRecords();

    static GdnBackwardProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        std::vector<Tensor> ports;
        ports.reserve(GdnBackwardProblem::kCount);
        for (const FTrainTensorId id : roles.ports) { ports.push_back(detail::gdnPort(args, id)); }
        return GdnBackwardProblem{
            std::move(ports), std::get<GdnAttributes>(*args.getOpArgument(
                                  OperationTraits<OperationKind::kGdnBackward>::createPatternOperationId(roles.op)))};
    }
};

}  // namespace ftrain

#endif
