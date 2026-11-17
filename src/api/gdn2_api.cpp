// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>

#include "flash_train/flash_train.h"

#include "flash_train/trace.hpp"
#include "flash_train/api.hpp"
#include "flash_train/primitive/hygon/gdn2/runner.hpp"
#include <limits>

namespace ftrain {
namespace {

// The GDN2 convenience path's immutable pieces, built once per process through the staged C
// API (as for ftrainGemm): one Ops per direction plus the user-side role IDs.
template<std::size_t N, typename OpId>
struct Gdn2Entry {
    FTrainOps ops{nullptr};
    std::array<FTrainTensorId, N> ports{};
    OpId op{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

using ForwardEntry  = Gdn2Entry<8, FTrainGdn2ForwardOpId>;
using BackwardEntry = Gdn2Entry<14, FTrainGdn2BackwardOpId>;

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
        return ftrainPatternAddGdn2Forward(pattern, &e.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    });
    return entry;
}

const BackwardEntry& backwardEntry() {
    static const BackwardEntry entry = buildEntry<BackwardEntry>([](FTrainPattern pattern, BackwardEntry& e) {
        const auto& p = e.ports;
        return ftrainPatternAddGdn2Backward(pattern, &e.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                            p[10], p[11], p[12], p[13]);
    });
    return entry;
}

FTrainStatus setOp(FTrainArgs args, FTrainGdn2ForwardOpId op, FTrainGdn2Attributes a) {
    return ftrainArgsSetGdn2Forward(args, op, a);
}
FTrainStatus setOp(FTrainArgs args, FTrainGdn2BackwardOpId op, FTrainGdn2Attributes a) {
    return ftrainArgsSetGdn2Backward(args, op, a);
}

// Binds the ports (the tensors struct is N consecutive FTrainStorageViews in port order),
// creates the Plan, then either reports primitive 0's workspace or executes it.
template<typename Entry, typename Tensors>
FTrainStatus run(const Entry& entry, const Tensors* tensors, FTrainGdn2Attributes attributes, uint64_t* query_bytes,
                 void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    static_assert(sizeof(Tensors) == sizeof(FTrainStorageView) * std::tuple_size_v<decltype(entry.ports)>);
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    if (tensors == nullptr) {
        setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2: tensors must not be null");
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

extern "C" FTrainStatus ftrainGdn2ForwardGetWorkspaceBytes(const FTrainGdn2ForwardTensors* tensors,
                                                           FTrainGdn2Attributes attributes, uint64_t* workspace_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2ForwardGetWorkspaceBytes"};
    if (workspace_bytes == nullptr) {
        ftrain::setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2: workspace_bytes must not be null");
        return FTRAIN_STATUS_INVALID_ARGUMENT;
    }
    return ftrain::run(ftrain::forwardEntry(), tensors, attributes, workspace_bytes, nullptr, 0, nullptr);
}

extern "C" FTrainStatus ftrainGdn2Forward(const FTrainGdn2ForwardTensors* tensors, FTrainGdn2Attributes attributes,
                                          void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2Forward"};
    return ftrain::run(ftrain::forwardEntry(), tensors, attributes, nullptr, workspace, workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainGdn2BackwardGetWorkspaceBytes(const FTrainGdn2BackwardTensors* tensors,
                                                            FTrainGdn2Attributes attributes,
                                                            uint64_t* workspace_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2BackwardGetWorkspaceBytes"};
    if (workspace_bytes == nullptr) {
        ftrain::setLastResult(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2: workspace_bytes must not be null");
        return FTRAIN_STATUS_INVALID_ARGUMENT;
    }
    return ftrain::run(ftrain::backwardEntry(), tensors, attributes, workspace_bytes, nullptr, 0, nullptr);
}

extern "C" FTrainStatus ftrainGdn2Backward(const FTrainGdn2BackwardTensors* tensors, FTrainGdn2Attributes attributes,
                                           void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2Backward"};
    return ftrain::run(ftrain::backwardEntry(), tensors, attributes, nullptr, workspace, workspace_bytes, stream);
}

extern "C" uint64_t ftrainGdn2SavedElements(int64_t B, int64_t T, int64_t H, int64_t HV, int64_t K, int64_t V) {
    constexpr int64_t limit = std::numeric_limits<int>::max();
    if (B < 1 || T < 0 || H < 1 || HV < H || HV % H || K < 1 || K > 128 || V < 1 || V > 128 || B > limit ||
        T > limit - 63 || H > limit || HV > limit || B > limit / HV || (T && B > limit / T))
        return 0;
    const int64_t chunks = (T + 63) / 64;
    if (chunks && B * HV > limit / chunks) return 0;
    return ftrain::gdn2::savedElements({int(B), int(T), int(H), int(HV), int(K), int(V)});
}
