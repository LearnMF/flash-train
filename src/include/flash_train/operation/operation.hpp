// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_OPERATION_HPP_
#define FTRAIN_OPERATION_OPERATION_HPP_

#include <variant>

#include "flash_train/operation/gemm.hpp"
#include "flash_train/operation/kda.hpp"
#include "flash_train/operation/gdn.hpp"
#include "flash_train/operation/gdn2.hpp"

namespace ftrain {

using OperationValue = std::variant<GemmAttributes, GroupedABCDGemmAttributes, GroupedBCDGemmAttributes,
                                    GroupedABGemmAttributes, KdaAttributes, GdnAttributes, Gdn2Attributes>;

}  // namespace ftrain

#endif
