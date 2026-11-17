// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include "flash_train/handle.hpp"
#include <cstdint>
#include <utility>

#include "flash_train/flash_train.h"

#include "flash_train/trace.hpp"
#include "flash_train/api.hpp"

namespace ftrain {
namespace {

// The GDN convenience path's immutable pieces, built once per process through the staged C
// API (as for ftrainGemm): one Ops per direction plus the user-side role IDs.
template<std::size_t N, typename OpId>
struct GdnEntry {
    FTrainOps ops{nullptr};
    std::array<FTrainTensorId, N> ports{};
    OpId op{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

using ForwardEntry  = GdnEntry<14, FTrainGdnForwardOpId>;
using BackwardEntry = GdnEntry<19, FTrainGdnBackwardOpId>;

template<typename Entry, typename AddOp>
Entry buildEntry(AddOp&& add_op) {
    Entry entry;
    FTrainPattern pattern = nullptr;
    FTrainStatus status   = ftrainPatternCreate(&pattern);
    for (FTrainTensorId& id : entry.ports) {
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &id); }
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = add_op(pattern, entry); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainOpsCreate(&entry.ops, pattern); }
    if (pattern != nullptr) { ftrainPatternDestroy(pattern); }
    entry.status = status;
    return entry;
}

const ForwardEntry& forwardEntry() {
    static const ForwardEntry entry = buildEntry<ForwardEntry>([](FTrainPattern pattern, ForwardEntry& e) {
        const auto& p = e.ports;
        return ftrainPatternAddGdnForward(pattern, &e.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                          p[10], p[11], p[12], p[13]);
    });
    return entry;
}

const BackwardEntry& backwardEntry() {
    static const BackwardEntry entry = buildEntry<BackwardEntry>([](FTrainPattern pattern, BackwardEntry& e) {
        const auto& p = e.ports;
        return ftrainPatternAddGdnBackward(pattern, &e.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                           p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18]);
    });
    return entry;
}

FTrainStatus setOp(FTrainArgs args, FTrainGdnForwardOpId op, FTrainGdnAttributes a) {
    return ftrainArgsSetGdnForward(args, op, a);
}
FTrainStatus setOp(FTrainArgs args, FTrainGdnBackwardOpId op, FTrainGdnAttributes a) {
    return ftrainArgsSetGdnBackward(args, op, a);
}

// Binds the ports (the tensors struct is N consecutive FTrainStorageViews in port order),
// creates the Plan, then either reports primitive 0's workspace or executes it.
template<typename Entry, typename Tensors>
FTrainStatus run(const Entry& entry, const Tensors* tensors, FTrainGdnAttributes attributes, uint64_t* query_bytes,
                 void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    static_assert(sizeof(Tensors) == sizeof(FTrainStorageView) * std::tuple_size_v<decltype(entry.ports)>);
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    if (tensors == nullptr) {
        setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: tensors must not be null");
        return FTRAIN_STATUS_INVALID_ARGUMENT;
    }
    const FTrainStorageView* views = reinterpret_cast<const FTrainStorageView*>(tensors);
    FTrainArgs args                = nullptr;
    FTrainPlan plan                = nullptr;
    FTrainStatus status            = ftrainArgsCreate(&args, entry.ops);
    for (std::size_t i = 0; i < entry.ports.size() && status == FTRAIN_STATUS_SUCCESS; ++i) {
        status = ftrainArgsSetTensor(args, entry.ports[i], views[i]);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = setOp(args, entry.op, attributes); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPlanCreate(&plan, args); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = query_bytes != nullptr ? ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, query_bytes)
                                        : ftrainPlanExecutePrimitive(plan, 0, workspace, workspace_bytes, stream);
    }
    const Result failure = status != FTRAIN_STATUS_SUCCESS ? getLastResult() : Result{};
    if (plan != nullptr) { ftrainPlanDestroy(plan); }
    if (args != nullptr) { ftrainArgsDestroy(args); }
    if (status != FTRAIN_STATUS_SUCCESS) { setLastResult(failure.getStatus(), failure.getMessage()); }
    return status;
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainGdnForwardGetWorkspaceBytes(const FTrainGdnForwardTensors* tensors,
                                                          FTrainGdnAttributes attributes, uint64_t* workspace_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnForwardGetWorkspaceBytes"};
    if (workspace_bytes == nullptr) {
        ftrain::setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: workspace_bytes must not be null");
        return FTRAIN_STATUS_INVALID_ARGUMENT;
    }
    return ftrain::run(ftrain::forwardEntry(), tensors, attributes, workspace_bytes, nullptr, 0, nullptr);
}

extern "C" FTrainStatus ftrainGdnForward(const FTrainGdnForwardTensors* tensors, FTrainGdnAttributes attributes,
                                         void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnForward"};
    return ftrain::run(ftrain::forwardEntry(), tensors, attributes, nullptr, workspace, workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainGdnBackwardGetWorkspaceBytes(const FTrainGdnBackwardTensors* tensors,
                                                           FTrainGdnAttributes attributes, uint64_t* workspace_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnBackwardGetWorkspaceBytes"};
    if (workspace_bytes == nullptr) {
        ftrain::setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: workspace_bytes must not be null");
        return FTRAIN_STATUS_INVALID_ARGUMENT;
    }
    return ftrain::run(ftrain::backwardEntry(), tensors, attributes, workspace_bytes, nullptr, 0, nullptr);
}

extern "C" FTrainStatus ftrainGdnBackward(const FTrainGdnBackwardTensors* tensors, FTrainGdnAttributes attributes,
                                          void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnBackward"};
    return ftrain::run(ftrain::backwardEntry(), tensors, attributes, nullptr, workspace, workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainPatternAddGdnForward(FTrainPattern pattern, FTrainGdnForwardOpId* op, FTrainTensorId q,
                                                   FTrainTensorId k, FTrainTensorId v, FTrainTensorId g,
                                                   FTrainTensorId beta, FTrainTensorId cu_seqlens, FTrainTensorId o,
                                                   FTrainTensorId q_rstd, FTrainTensorId k_rstd, FTrainTensorId aqk,
                                                   FTrainTensorId akk, FTrainTensorId qn, FTrainTensorId kn,
                                                   FTrainTensorId h) {
    return ftrain::invokeApi([&] {
        if (!pattern || !op) throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: null pattern/op");
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGdnForward>(q, k, v, g, beta, cu_seqlens, o, q_rstd,
                                                                                k_rstd, aqk, akk, qn, kn, h);
    });
}
extern "C" FTrainStatus ftrainArgsSetGdnForward(FTrainArgs args, FTrainGdnForwardOpId op,
                                                FTrainGdnAttributes attributes) {
    return ftrain::invokeApi([&] {
        if (!args || !std::isfinite(attributes.scale) || attributes.scale <= 0)
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: null args or invalid scale");
        args->args.setOperation<ftrain::OperationKind::kGdnForward>(ftrain::PatternOperationId{op.opaque},
                                                                    ftrain::GdnAttributes(attributes));
    });
}

extern "C" FTrainStatus ftrainPatternAddGdnBackward(FTrainPattern pattern, FTrainGdnBackwardOpId* op,
                                                    FTrainTensorId d_o, FTrainTensorId q, FTrainTensorId k,
                                                    FTrainTensorId v, FTrainTensorId g, FTrainTensorId beta,
                                                    FTrainTensorId cu_seqlens, FTrainTensorId q_rstd,
                                                    FTrainTensorId k_rstd, FTrainTensorId aqk, FTrainTensorId akk,
                                                    FTrainTensorId qn, FTrainTensorId kn, FTrainTensorId dq,
                                                    FTrainTensorId dk, FTrainTensorId dv, FTrainTensorId dg,
                                                    FTrainTensorId dbeta, FTrainTensorId h) {
    return ftrain::invokeApi([&] {
        if (!pattern || !op) throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: null pattern/op");
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGdnBackward>(
            d_o, q, k, v, g, beta, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, dq, dk, dv, dg, dbeta, h);
    });
}
extern "C" FTrainStatus ftrainArgsSetGdnBackward(FTrainArgs args, FTrainGdnBackwardOpId op,
                                                 FTrainGdnAttributes attributes) {
    return ftrain::invokeApi([&] {
        if (!args || !std::isfinite(attributes.scale) || attributes.scale <= 0)
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN: null args or invalid scale");
        args->args.setOperation<ftrain::OperationKind::kGdnBackward>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::GdnAttributes(attributes));
    });
}
