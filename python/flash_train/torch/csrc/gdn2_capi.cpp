// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <flash_train/flash_train.h>
#include "gdn2_bridge.h"
namespace flash_train_torch {
std::uint64_t gdn2SavedElements(std::int64_t B, std::int64_t T, std::int64_t H, std::int64_t HV, std::int64_t K,
                                std::int64_t V) {
    return ftrainGdn2SavedElements(B, T, H, HV, K, V);
}
unsigned char callGdn2(bool bw, const Gdn2View* views, float scale, bool norm, void* workspace, std::uint64_t bytes,
                       std::uint64_t* query, void* stream, const char** message) {
    FTrainStorageView ports[14];
    for (int i = 0; i < (bw ? 14 : 8); ++i) {
        const auto& v = views[i];
        ports[i]      = {v.memory,
                         v.dims,
                         nullptr,
                         static_cast<uint8_t>(v.rank),
                         static_cast<FTrainNumericType>(v.fp32 ? FTRAIN_NUMERIC_TYPE_FP32 : FTRAIN_NUMERIC_TYPE_BF16),
                         FTRAIN_INDEX_TYPE_CONTINUOUS,
                         false};
    }
    FTrainGdn2Attributes a{scale, norm ? 1 : 0};
    FTrainStatus status;
    if (bw) {
        FTrainGdn2BackwardTensors p{ports[0], ports[1], ports[2], ports[3],  ports[4],  ports[5],  ports[6],
                                    ports[7], ports[8], ports[9], ports[10], ports[11], ports[12], ports[13]};
        status = query ? ftrainGdn2BackwardGetWorkspaceBytes(&p, a, query)
                       : ftrainGdn2Backward(&p, a, workspace, bytes, static_cast<FTrainStream>(stream));
    } else {
        FTrainGdn2ForwardTensors p{ports[0], ports[1], ports[2], ports[3], ports[4], ports[5], ports[6], ports[7]};
        status = query ? ftrainGdn2ForwardGetWorkspaceBytes(&p, a, query)
                       : ftrainGdn2Forward(&p, a, workspace, bytes, static_cast<FTrainStream>(stream));
    }
    if (status && message) *message = ftrainGetLastMessage();
    return status;
}
}  // namespace flash_train_torch
