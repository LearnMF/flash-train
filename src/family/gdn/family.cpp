// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/primitive/hygon/gdn/g2_gdn.hpp"

#include "flash_train/family/gdn/family.hpp"

namespace ftrain {

// Registration order is the no-Finder preference: the G2 gfx936 implementation first.
std::vector<std::shared_ptr<const Primitive<GdnForwardProblem>>> GdnForwardFamily::makeRecords() {
    return {std::make_shared<G2GdnForward>()};
}

std::vector<std::shared_ptr<const Primitive<GdnBackwardProblem>>> GdnBackwardFamily::makeRecords() {
    return {std::make_shared<G2GdnBackward>()};
}

}  // namespace ftrain
