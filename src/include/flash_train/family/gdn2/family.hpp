// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_GDN2_FAMILY_HPP_
#define FTRAIN_GDN2_FAMILY_HPP_
#include <array>
#include <memory>
#include "flash_train/family/gdn2/problem.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"
namespace ftrain {
struct Gdn2RegistrationOrderFinder {
    static const char* getName() { return "Gdn2RegistrationOrder"; }
    template<class P>
    static std::vector<std::string> findCandidates(const P&, const Constraints&) {
        return {};
    }
};
struct Gdn2ForwardFamily {
    using Problem = Gdn2ForwardProblem;
    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, 8> ports;
        FTrainGdn2ForwardOpId op;
    };
    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, 8> p{};
            for (auto& id : p) id = pattern.addOperand<OperandKind::kTensor>();
            auto op = pattern.addOperation<OperationKind::kGdn2Forward>(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
            return Roles{pattern.buildPattern(), p, op};
        }();
        return roles;
    }
    static Pattern makePattern() { return makeRoles().pattern; }
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
    static Problem makeProblem(const Args& args) {
        const auto& r = makeRoles();
        std::vector<Tensor> ports;
        for (auto id : r.ports)
            ports.push_back(
                std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id))));
        return Problem{std::move(ports),
                       std::get<Gdn2Attributes>(*args.getOpArgument(
                           OperationTraits<OperationKind::kGdn2Forward>::createPatternOperationId(r.op)))};
    }
};
struct Gdn2BackwardFamily {
    using Problem = Gdn2BackwardProblem;
    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, 14> ports;
        FTrainGdn2BackwardOpId op;
    };
    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, 14> p{};
            for (auto& id : p) id = pattern.addOperand<OperandKind::kTensor>();
            auto op = pattern.addOperation<OperationKind::kGdn2Backward>(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                                                                         p[8], p[9], p[10], p[11], p[12], p[13]);
            return Roles{pattern.buildPattern(), p, op};
        }();
        return roles;
    }
    static Pattern makePattern() { return makeRoles().pattern; }
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
    static Problem makeProblem(const Args& args) {
        const auto& r = makeRoles();
        std::vector<Tensor> ports;
        for (auto id : r.ports)
            ports.push_back(
                std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id))));
        return Problem{std::move(ports),
                       std::get<Gdn2Attributes>(*args.getOpArgument(
                           OperationTraits<OperationKind::kGdn2Backward>::createPatternOperationId(r.op)))};
    }
};
}  // namespace ftrain
#endif
