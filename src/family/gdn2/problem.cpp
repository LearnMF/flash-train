// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include "flash_train/family/gdn2/problem.hpp"
#include "flash_train/error.hpp"
namespace ftrain {
Gdn2Problem::Gdn2Problem(std::vector<Tensor> ports, Gdn2Attributes a, bool bw)
    : attributes(a), backward(bw), tensors_(std::move(ports)) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2: %s", message);
    };
    require(size() == (bw ? 14 : 8), "wrong port count");
    auto q = view(0).getDims(), v = view(2).getDims();
    require(q.size() == 4 && v.size() == 4, "q/k/v must be BTHD rank four");
    for (auto d : q) require(d >= 0 && d <= std::numeric_limits<int>::max() - 63, "dimension exceeds int32");
    for (auto d : v) require(d >= 0 && d <= std::numeric_limits<int>::max() - 63, "dimension exceeds int32");
    require(q[0] > 0 && q[2] > 0 && q[3] > 0 && v[2] > 0 && v[3] > 0, "B/H/HV/K/V must be positive");
    require(v[0] == q[0] && v[1] == q[1] && v[2] % q[2] == 0, "incompatible B/T or HV not divisible by H");
    require(q[0] * v[2] <= std::numeric_limits<int>::max() && q[0] * q[1] <= std::numeric_limits<int>::max(),
            "grid exceeds int32");
    shape = {int(q[0]), int(q[1]), int(q[2]), int(v[2]), int(q[3]), int(v[3])};
    require(shape.blocks() <= std::numeric_limits<int>::max(), "chunk grid exceeds int32");
    // Bound dimensions before computing opaque cache sizes (the kernels handle at most 128).
    if (shape.K > 128 || shape.V > 128) throw Exception(FTRAIN_STATUS_UNSUPPORTED, "GDN2: K,V must be <=128");
    const std::vector<int64_t> gate = {q[0], q[1], v[2], q[3]};
    auto dims                       = [&](int p, const std::vector<int64_t>& expected) {
        require(view(p).getDims() == expected, "wrong tensor shape");
    };
    dims(1, q);
    dims(3, gate);
    dims(4, gate);
    dims(5, v);
    dims(bw ? 6 : 7, {static_cast<int64_t>(gdn2::savedElements(shape))});
    if (bw) {
        dims(7, v);
        dims(8, q);
        dims(9, q);
        dims(10, v);
        dims(11, gate);
        dims(12, gate);
        dims(13, v);
    } else
        dims(6, v);
    for (std::size_t i = 0; i < size(); ++i) {
        require(!view(i).isHostMemory(), "ports must be device memory");
        require(shape.T == 0 || view(i).getMemory() != nullptr, "missing port");
    }
}
std::vector<std::uint64_t> Gdn2Problem::getProblemKey() const {
    std::vector<std::uint64_t> key{std::uint64_t(backward), std::uint64_t(attributes.value.l2norm_qk)};
    for (std::size_t i = 0; i < size(); ++i) {
        const auto& v = view(i);
        key.push_back(v.getNumericType());
        key.push_back(v.getIndexType());
        key.push_back(reinterpret_cast<std::uintptr_t>(v.getMemory()) % 16);
        key.push_back(v.getDims().size());
        for (auto d : v.getDims()) key.push_back(d);
    }
    return key;
}
}  // namespace ftrain
