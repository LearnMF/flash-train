// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "flash_train/flash_train.h"
TEST(Gdn2Api, RejectsNullHandlesAndInvalidScale) {
    FTrainGdn2ForwardOpId op{};
    EXPECT_EQ(ftrainPatternAddGdn2Forward(nullptr, &op, {}, {}, {}, {}, {}, {}, {}, {}),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGdn2Forward(nullptr, op, {1, 0}), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainGdn2ForwardGetWorkspaceBytes(nullptr, {1, 0}, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    FTrainTensorId p[8]{};
    for (auto& id : p) ASSERT_EQ(ftrainPatternAddTensor(pattern, &id), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddGdn2Forward(pattern, &op, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]),
              FTRAIN_STATUS_SUCCESS);
    FTrainOps ops   = nullptr;
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGdn2Forward(args, op, {-1, 0}), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGdn2Forward(args, op, {1, 2}), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGdn2Forward(args, op, {1, 1}), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}
TEST(Gdn2Api, SavedCountAndInvalidShapes) {
    EXPECT_EQ(ftrainGdn2SavedElements(1, 64, 1, 1, 128, 128), 90240u);
    EXPECT_EQ(ftrainGdn2SavedElements(2, 65, 2, 4, 128, 128), 16 * 90240u);
    EXPECT_EQ(ftrainGdn2SavedElements(1, 0, 1, 1, 128, 128), 0u);
    EXPECT_EQ(ftrainGdn2SavedElements(1, 64, 3, 4, 128, 128), 0u);
    EXPECT_EQ(ftrainGdn2SavedElements(1, 64, 1, 1, 129, 128), 0u);
    EXPECT_EQ(ftrainGdn2SavedElements(1, -1, 1, 1, 128, 128), 0u);
}
