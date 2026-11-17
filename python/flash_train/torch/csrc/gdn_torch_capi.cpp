// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <flash_train/flash_train.h>

#include "gdn_bridge.h"

namespace flash_train_torch {
namespace {

FTrainStorageView makeGdnView(const GdnBridgeView& v) {
    static const std::int64_t kEmpty[1] = {0};
    const FTrainNumericType type        = v.type == kGdnBf16   ? FTRAIN_NUMERIC_TYPE_BF16
                                          : v.type == kGdnFp32 ? FTRAIN_NUMERIC_TYPE_FP32
                                                               : FTRAIN_NUMERIC_TYPE_INT32;
    if (v.memory == nullptr) {  // unused optional port
        return FTrainStorageView{nullptr, kEmpty, nullptr, 1, type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    }
    return FTrainStorageView{const_cast<void*>(v.memory),  v.dims, nullptr, static_cast<std::uint8_t>(v.num_dims), type,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

FTrainGdnAttributes makeGdnAttributes(const GdnBridgeAttributes& a) {
    return FTrainGdnAttributes{a.scale, a.l2norm_qk};
}

template<typename Tensors, std::size_t N, typename Query, typename Exec>
unsigned char callGdn(const GdnBridgeView* views, GdnBridgeAttributes attributes, void* workspace,
                      std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream, const char** message,
                      Query query, Exec exec) {
    static_assert(sizeof(Tensors) == N * sizeof(FTrainStorageView));
    FTrainStorageView ports[N];
    for (std::size_t i = 0; i < N; ++i) { ports[i] = makeGdnView(views[i]); }
    const Tensors* tensors    = reinterpret_cast<const Tensors*>(ports);
    const FTrainStatus status = query_bytes != nullptr ? query(tensors, makeGdnAttributes(attributes), query_bytes)
                                                       : exec(tensors, makeGdnAttributes(attributes), workspace,
                                                              workspace_bytes, static_cast<FTrainStream>(stream));
    if (status != FTRAIN_STATUS_SUCCESS && message != nullptr) { *message = ftrainGetLastMessage(); }
    return static_cast<unsigned char>(status);
}

}  // namespace

std::uint64_t gdnSavedHElements(std::int64_t tokens, std::int64_t heads, std::int64_t seqs, bool has_cu_seqlens) {
    return ftrainGdnSavedHElements(tokens, heads, seqs, has_cu_seqlens ? 1 : 0);
}

unsigned char callGdnForward(const GdnBridgeView* views, GdnBridgeAttributes attributes, void* workspace,
                             std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                             const char** message) {
    return callGdn<FTrainGdnForwardTensors, 14>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                message, ftrainGdnForwardGetWorkspaceBytes, ftrainGdnForward);
}

unsigned char callGdnBackward(const GdnBridgeView* views, GdnBridgeAttributes attributes, void* workspace,
                              std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                              const char** message) {
    return callGdn<FTrainGdnBackwardTensors, 19>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                 message, ftrainGdnBackwardGetWorkspaceBytes, ftrainGdnBackward);
}

}  // namespace flash_train_torch
