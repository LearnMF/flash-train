// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "flash_train/family/gdn2/problem.hpp"
#include "flash_train/error.hpp"
namespace ftrain {
namespace {
Tensor tensor(int slot, std::vector<int64_t> dims, FTrainNumericType type = FTRAIN_NUMERIC_TYPE_BF16) {
    FTrainStorageView view{reinterpret_cast<void*>(0x10000u * (slot + 1)),
                           dims.data(),
                           nullptr,
                           static_cast<uint8_t>(dims.size()),
                           type,
                           FTRAIN_INDEX_TYPE_CONTINUOUS,
                           false};
    return Tensor(StorageView(view));
}
std::vector<Tensor> ports(gdn2::Shape s) {
    std::vector<int64_t> q{s.B, s.T, s.H, s.K}, v{s.B, s.T, s.HV, s.V}, g{s.B, s.T, s.HV, s.K};
    return {tensor(0, q), tensor(1, q),
            tensor(2, v), tensor(3, g),
            tensor(4, g), tensor(5, v),
            tensor(6, v), tensor(7, {static_cast<int64_t>(gdn2::savedElements(s))}, FTRAIN_NUMERIC_TYPE_FP32)};
}
TEST(Gdn2Problem, DenseTailsGvaAndNon128) {
    for (int t : {0, 1, 63, 64, 65, 129}) {
        gdn2::Shape s{2, t, 2, 4, 17, 33};
        Gdn2ForwardProblem p(ports(s), Gdn2Attributes({0.25f, 1}));
        EXPECT_EQ(p.shape.chunks(), (t + 63) / 64);
        EXPECT_EQ(p.shape.blocks(), 8u * ((t + 63) / 64));
    }
}
TEST(Gdn2Problem, RejectsScalarEraseAndWrongCache) {
    gdn2::Shape s{1, 64, 1, 2, 128, 128};
    auto p = ports(s);
    p[4]   = tensor(4, {1, 64, 2});
    EXPECT_THROW(Gdn2ForwardProblem(std::move(p), Gdn2Attributes({1, 0})), Exception);
    p    = ports(s);
    p[7] = tensor(7, {1}, FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_THROW(Gdn2ForwardProblem(std::move(p), Gdn2Attributes({1, 0})), Exception);
    p    = ports(s);
    p[2] = tensor(2, {1, 64, 3, 128});
    EXPECT_THROW(Gdn2ForwardProblem(std::move(p), Gdn2Attributes({1, 0})), Exception);
}
TEST(Gdn2Problem, KeyIncludesNormAndGateTypes) {
    gdn2::Shape s{1, 64, 1, 1, 128, 128};
    Gdn2ForwardProblem a(ports(s), Gdn2Attributes({1, 0})), b(ports(s), Gdn2Attributes({1, 1}));
    EXPECT_NE(a.getProblemKey(), b.getProblemKey());
    auto p = ports(s);
    p[3]   = tensor(3, {1, 64, 1, 128}, FTRAIN_NUMERIC_TYPE_FP32);
    Gdn2ForwardProblem c(std::move(p), Gdn2Attributes({1, 0}));
    EXPECT_NE(a.getProblemKey(), c.getProblemKey());
}
TEST(Gdn2Layout, BuffersAreDisjointAndSized) {
    for (int k : {1, 17, 128})
        for (int v : {1, 33, 128}) {
            gdn2::Shape s{1, 65, 1, 4, k, v};
            gdn2::Layout l(s);
            gdn2::GradLayout g(s);
            EXPECT_EQ(l.size, 6 * 64 * k + 2 * 64 * 64 + 2 * 64 * v + k * v + 128);
            EXPECT_EQ(g.size, 5 * 64 * k + 2 * 64 * 64 + 64 * v + k * v);
            EXPECT_EQ(gdn2::backwardBytes(s), s.blocks() * g.size * 4);
        }
}
}  // namespace
}  // namespace ftrain
