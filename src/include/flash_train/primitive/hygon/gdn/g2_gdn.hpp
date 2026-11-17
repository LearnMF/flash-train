// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN_G2_GDN_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN_G2_GDN_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/primitive/hygon/kda/g2_runner.hpp"

namespace ftrain {

// Scalar-gate specializations of the gfx936 MFMA chunk pipeline, K = V = 128.
class G2GdnForward final : public Primitive<GdnForwardProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const GdnForwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    kda_g2::Shape shape_{};
    kda_g2::ForwardArgs args_{};
};

class G2GdnBackward final : public Primitive<GdnBackwardProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const GdnBackwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const GdnBackwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    kda_g2::Shape shape_{};
    kda_g2::BackwardArgs args_{};
};

}  // namespace ftrain

#endif
