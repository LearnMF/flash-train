// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/engine.hpp"
#include "flash_train/error.hpp"
#include "flash_train/family/kda/family.hpp"
#include "flash_train/family/kda/finder.hpp"
#include "flash_train/family/kda/problem.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

// Fake, 256-byte aligned device addresses: Problems never dereference memory.
void* fakeAddress(std::uintptr_t slot) { return reinterpret_cast<void*>((slot + 1) * 0x10000); }

Tensor makeTensor(std::uintptr_t slot, std::vector<std::int64_t> dims, FTrainNumericType type, bool present = true) {
    const FTrainStorageView view{
        present ? fakeAddress(slot) : nullptr, dims.data(), nullptr, static_cast<std::uint8_t>(dims.size()), type,
        FTRAIN_INDEX_TYPE_CONTINUOUS,          false};
    return Tensor(StorageView(view));
}

constexpr FTrainNumericType kBf16 = FTRAIN_NUMERIC_TYPE_BF16;
constexpr FTrainNumericType kFp32 = FTRAIN_NUMERIC_TYPE_FP32;

std::vector<Tensor> forwardPorts(std::int64_t T, std::int64_t H, std::int64_t K, bool gate, bool cu = false) {
    std::vector<Tensor> p;
    p.push_back(makeTensor(0, {T, H, K}, kBf16));                    // q
    p.push_back(makeTensor(1, {T, H, K}, kBf16));                    // k
    p.push_back(makeTensor(2, {T, H, K}, kBf16));                    // v
    p.push_back(makeTensor(3, {T, H, K}, gate ? kBf16 : kFp32));     // g
    p.push_back(makeTensor(4, {T, H}, kFp32));                       // beta
    p.push_back(makeTensor(5, {H}, kFp32, gate));                    // a_log
    p.push_back(makeTensor(6, {H, K}, kFp32, gate));                 // dt_bias
    p.push_back(makeTensor(7, {3}, FTRAIN_NUMERIC_TYPE_INT32, cu));  // cu_seqlens
    p.push_back(makeTensor(8, {T, H, K}, kBf16));                    // o
    p.push_back(makeTensor(9, {T, H}, kFp32));                       // q_rstd
    p.push_back(makeTensor(10, {T, H}, kFp32));                      // k_rstd
    p.push_back(makeTensor(11, {T, H, 64}, kBf16));                  // aqk
    p.push_back(makeTensor(12, {T, H, 64}, kBf16));                  // akk
    p.push_back(makeTensor(13, {1}, kBf16, false));                  // state (optional, absent)
    return p;
}

KdaAttributes attributes(bool gate, bool l2 = true) {
    return KdaAttributes(FTrainKdaAttributes{0.0883883F, -5.0F, gate ? 1 : 0, l2 ? 1 : 0});
}

TEST(KdaFamilyTest, EnginesDefineTheSupportedPatterns) {
    const OpsEngine<KdaForwardFamily, KdaRegistrationOrderFinder> forward;
    EXPECT_EQ(forward.getPattern().getNumOperands(), 14U);
    ASSERT_EQ(forward.getPattern().getNumOps(), 1U);
    EXPECT_EQ(forward.getPattern().getOpNode(PatternOperationId{0}).getKind(), OperationKind::kKdaForward);
    const OpsEngine<KdaBackwardFamily, KdaRegistrationOrderFinder> backward;
    EXPECT_EQ(backward.getPattern().getNumOperands(), 21U);
    ASSERT_EQ(backward.getPattern().getNumOps(), 1U);
    EXPECT_EQ(backward.getPattern().getOpNode(PatternOperationId{0}).getKind(), OperationKind::kKdaBackward);
}

TEST(KdaFamilyTest, BuiltinEnginesAreRegistered) {
    EXPECT_NE(getGlobalHandle().findOpsEngine(KdaForwardFamily::makePattern().getKey()), nullptr);
    EXPECT_NE(getGlobalHandle().findOpsEngine(KdaBackwardFamily::makePattern().getKey()), nullptr);
}

TEST(KdaProblemTest, AcceptsBothGateContracts) {
    const KdaForwardProblem raw(forwardPorts(256, 4, 128, true), attributes(true));
    EXPECT_EQ(raw.getShape().tokens, 256);
    EXPECT_EQ(raw.getShape().heads, 4);
    EXPECT_EQ(raw.getShape().seqs, 1);
    const KdaForwardProblem pre(forwardPorts(256, 4, 128, false, true), attributes(false));
    EXPECT_EQ(pre.getShape().seqs, 2);
    EXPECT_NE(raw.getProblemKey(), pre.getProblemKey());
}

TEST(KdaProblemTest, RejectsStructuralViolations) {
    auto expectInvalid = [](std::vector<Tensor> ports, KdaAttributes a) {
        try {
            KdaForwardProblem problem(std::move(ports), a);
            ADD_FAILURE() << "expected an exception";
        }
        catch (const Exception& e) {
            EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        }
    };
    std::vector<Tensor> bad_rank = forwardPorts(64, 2, 128, true);
    bad_rank[0]                  = makeTensor(0, {64, 256}, kBf16);
    expectInvalid(std::move(bad_rank), attributes(true));

    std::vector<Tensor> bad_beta = forwardPorts(64, 2, 128, true);
    bad_beta[4]                  = makeTensor(4, {64, 3}, kFp32);
    expectInvalid(std::move(bad_beta), attributes(true));

    expectInvalid(forwardPorts(64, 2, 128, false), attributes(true));  // gate without a_log/dt_bias

    std::vector<Tensor> no_rstd = forwardPorts(64, 2, 128, true);
    no_rstd[9]                  = makeTensor(9, {64, 2}, kFp32, false);
    expectInvalid(std::move(no_rstd), attributes(true, true));

    std::vector<Tensor> bad_state        = forwardPorts(256, 4, 128, true);  // needs 4 * 4 * 32768 elements
    bad_state[KdaForwardProblem::kState] = makeTensor(13, {4 * 4 * 32768 - 1}, kBf16);
    expectInvalid(std::move(bad_state), attributes(true));
}

TEST(KdaProblemTest, AcceptsTheOptionalState) {
    // dense T % 64 == 0: H * T/64 chunks; packed (cu_seqlens, N = 2): H * (ceil(T/64) + N)
    std::vector<Tensor> dense        = forwardPorts(256, 4, 128, true);
    dense[KdaForwardProblem::kState] = makeTensor(13, {4 * 4 * 32768}, kBf16);
    const KdaForwardProblem a(std::move(dense), attributes(true));
    EXPECT_TRUE(a.hasMemory(KdaForwardProblem::kState));
    std::vector<Tensor> packed        = forwardPorts(256, 4, 128, true, true);
    packed[KdaForwardProblem::kState] = makeTensor(13, {4 * (4 + 2) * 32768}, kBf16);
    const KdaForwardProblem b(std::move(packed), attributes(true));
    EXPECT_EQ(kdaStateElements(b.getShape(), true), 4 * 6 * 32768);
    EXPECT_NE(a.getProblemKey(), KdaForwardProblem(forwardPorts(256, 4, 128, true), attributes(true)).getProblemKey());
}

}  // namespace
}  // namespace ftrain
