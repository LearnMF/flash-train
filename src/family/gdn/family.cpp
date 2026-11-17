// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/primitive/hygon/gdn/g2_gdn.hpp"

#include "flash_train/family/gdn/family.hpp"

namespace ftrain {

// The FP32-intermediate gfx936 implementation is the accuracy-safe default.
std::vector<std::shared_ptr<const Primitive<GdnForwardProblem>>> GdnForwardFamily::makeRecords() {
    return {std::make_shared<G2GdnForward>()};
}

std::vector<std::shared_ptr<const Primitive<GdnBackwardProblem>>> GdnBackwardFamily::makeRecords() {
    return {std::make_shared<G2GdnBackward>()};
}

}  // namespace ftrain
