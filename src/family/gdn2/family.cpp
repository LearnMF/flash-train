// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/family/gdn2/family.hpp"
#include "flash_train/primitive/hygon/gdn2/primitive.hpp"
namespace ftrain {
std::vector<std::shared_ptr<const Primitive<Gdn2ForwardProblem>>> Gdn2ForwardFamily::makeRecords() {
    return {std::make_shared<Gdn2Forward>()};
}
std::vector<std::shared_ptr<const Primitive<Gdn2BackwardProblem>>> Gdn2BackwardFamily::makeRecords() {
    return {std::make_shared<Gdn2Backward>()};
}
}  // namespace ftrain
