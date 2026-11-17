// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_GDN_FINDER_HPP_
#define FTRAIN_FAMILY_GDN_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/constraints.hpp"
#include "flash_train/family/gdn/problem.hpp"

namespace ftrain {

// The GDN families' selection policy: no preference beyond the registration order (the
// recommended implementation is registered first).
struct GdnRegistrationOrderFinder final {
    static const char* getName() { return "GdnRegistrationOrder"; }

    static std::vector<std::string> findCandidates(const GdnForwardProblem&, const Constraints&) { return {}; }
    static std::vector<std::string> findCandidates(const GdnBackwardProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
