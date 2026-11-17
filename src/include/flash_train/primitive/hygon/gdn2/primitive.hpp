// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_GDN2_PRIMITIVE_HPP_
#define FTRAIN_GDN2_PRIMITIVE_HPP_
#include "flash_train/primitive.hpp"
#include "flash_train/family/gdn2/problem.hpp"
namespace ftrain {
class Gdn2Forward final : public Primitive<Gdn2ForwardProblem> {
  public:
    const char* getName() const noexcept override { return "Gdn2ForwardGfx936"; }
    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<Gdn2Forward>(*this); }
    Result isApplicable(const Gdn2ForwardProblem&, const Constraints&) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return 0; }
    void configure(const Gdn2ForwardProblem&) override;

  private:
    void executeImpl(const Resources& r) override { gdn2::forward(shape_, args_, r.getStream()); }
    gdn2::Shape shape_{};
    gdn2::Args args_{};
};
class Gdn2Backward final : public Primitive<Gdn2BackwardProblem> {
  public:
    const char* getName() const noexcept override { return "Gdn2BackwardGfx936"; }
    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<Gdn2Backward>(*this); }
    Result isApplicable(const Gdn2BackwardProblem&, const Constraints&) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return gdn2::backwardBytes(shape_); }
    void configure(const Gdn2BackwardProblem&) override;

  private:
    void executeImpl(const Resources& r) override { gdn2::backward(shape_, args_, r.getWorkspace(), r.getStream()); }
    gdn2::Shape shape_{};
    gdn2::Args args_{};
};
}  // namespace ftrain
#endif
