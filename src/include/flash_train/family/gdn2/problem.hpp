// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_GDN2_PROBLEM_HPP_
#define FTRAIN_GDN2_PROBLEM_HPP_
#include <vector>
#include "flash_train/tensor.hpp"
#include "flash_train/operation/gdn2.hpp"
#include "flash_train/primitive/hygon/gdn2/runner.hpp"
namespace ftrain {
class Gdn2Problem {
  public:
    Gdn2Problem(std::vector<Tensor> ports, Gdn2Attributes attributes, bool backward);
    std::vector<std::uint64_t> getProblemKey() const;
    const StorageView& view(std::size_t i) const { return tensors_[i].getStorageView(); }
    gdn2::Shape shape;
    Gdn2Attributes attributes;
    bool backward;
    std::size_t size() const { return tensors_.size(); }

  private:
    std::vector<Tensor> tensors_;
};
struct Gdn2ForwardProblem final : Gdn2Problem {
    static constexpr std::size_t kCount = 8;
    Gdn2ForwardProblem(std::vector<Tensor> ports, Gdn2Attributes a) : Gdn2Problem(std::move(ports), a, false) {}
};
struct Gdn2BackwardProblem final : Gdn2Problem {
    static constexpr std::size_t kCount = 14;
    Gdn2BackwardProblem(std::vector<Tensor> ports, Gdn2Attributes a) : Gdn2Problem(std::move(ports), a, true) {}
};
}  // namespace ftrain
#endif
