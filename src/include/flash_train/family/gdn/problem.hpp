// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_GDN_PROBLEM_HPP_
#define FTRAIN_FAMILY_GDN_PROBLEM_HPP_
#include "flash_train/family/kda/problem.hpp"
#include "flash_train/operation/gdn.hpp"
namespace ftrain {
using GdnShape = KdaShape;
class GdnForwardProblem final : public KdaPortSet {
  public:
    enum Port : std::size_t { kQ, kK, kV, kG, kBeta, kCuSeqlens, kO, kQRstd, kKRstd, kAqk, kAkk, kQn, kKn, kH, kCount };
    GdnForwardProblem(std::vector<Tensor> ports, GdnAttributes attributes);
    const GdnShape& getShape() const noexcept { return shape_; }
    const GdnAttributes& getAttributes() const noexcept { return attributes_; }
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    GdnShape shape_{};
    GdnAttributes attributes_;
};
class GdnBackwardProblem final : public KdaPortSet {
  public:
    enum Port : std::size_t {
        kDO,
        kQ,
        kK,
        kV,
        kG,
        kBeta,
        kCuSeqlens,
        kQRstd,
        kKRstd,
        kAqk,
        kAkk,
        kQn,
        kKn,
        kDq,
        kDk,
        kDv,
        kDg,
        kDBeta,
        kH,
        kCount
    };
    GdnBackwardProblem(std::vector<Tensor> ports, GdnAttributes attributes);
    const GdnShape& getShape() const noexcept { return shape_; }
    const GdnAttributes& getAttributes() const noexcept { return attributes_; }
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    GdnShape shape_{};
    GdnAttributes attributes_;
};
}  // namespace ftrain
#endif
