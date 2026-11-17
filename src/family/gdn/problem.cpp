// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/gdn/problem.hpp"

namespace ftrain {
namespace {

using Dims = std::vector<std::int64_t>;

const Dims& dimsOf(const std::vector<Tensor>& ports, std::size_t index) {
    return ports[index].getStorageView().getDims();
}

bool portHasMemory(const std::vector<Tensor>& ports, std::size_t index) {
    return ports[index].getStorageView().getMemory() != nullptr;
}

void requireDims(const std::vector<Tensor>& ports, std::size_t index, const Dims& want, const char* name) {
    if (dimsOf(ports, index) != want) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN port %s has the wrong shape", name);
    }
}

void requireDevice(const std::vector<Tensor>& ports, std::initializer_list<std::size_t> indices) {
    for (const std::size_t index : indices) {
        if (ports[index].getStorageView().isHostMemory()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN requires every port in device memory");
        }
    }
}

void requireState(const std::vector<Tensor>& ports, std::size_t h, const GdnShape& shape, bool has_cu) {
    if (portHasMemory(ports, h)) {
        requireDims(ports, h,
                    {static_cast<std::int64_t>(ftrainGdnSavedHElements(shape.tokens, shape.heads, shape.seqs, has_cu))},
                    "h");
    }
}

void requireQkn(const std::vector<Tensor>& ports, std::size_t qn, std::size_t kn, const Dims& qd,
                const GdnAttributes& attributes) {
    const bool has_qn = portHasMemory(ports, qn);
    if (has_qn != portHasMemory(ports, kn)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN qn and kn are given together or not at all");
    }
    if (!has_qn) { return; }
    if (!attributes.isL2NormQk()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN qn and kn require l2norm_qk");
    }
    requireDims(ports, qn, qd, "qn");
    requireDims(ports, kn, qd, "kn");
}

// The checks the forward and backward share: the THD q/k/v/g/beta block, the gate
// parameters, cu_seqlens, and the saved tensors. q / k name the ports that carry the [T, H, K]
// queries and keys (the backward passes qn / kn when it takes no q / k). Returns the shape.
GdnShape validateCommon(const std::vector<Tensor>& ports, std::size_t q, std::size_t k, std::size_t v, std::size_t g,
                        std::size_t beta, std::size_t cu, std::size_t q_rstd, std::size_t k_rstd, std::size_t aqk,
                        std::size_t akk, const GdnAttributes& attributes) {
    const Dims& qd = dimsOf(ports, q);
    if (qd.size() != 3) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN q must be rank three [T, H, K]"); }
    const Dims& vd = dimsOf(ports, v);
    if (vd.size() != 3 || vd[0] != qd[0] || vd[1] != qd[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN v must be [T, H, V] with q's T and H");
    }
    GdnShape shape{qd[0], qd[1], qd[2], vd[2], 1};
    requireDims(ports, k, qd, "k");
    requireDims(ports, g, {qd[0], qd[1]}, "g");
    requireDims(ports, beta, Dims{shape.tokens, shape.heads}, "beta");
    if (portHasMemory(ports, aqk)) requireDims(ports, aqk, Dims{shape.tokens, shape.heads, 64}, "aqk");
    if (portHasMemory(ports, akk)) requireDims(ports, akk, Dims{shape.tokens, shape.heads, 64}, "akk");

    if (portHasMemory(ports, cu)) {
        const Dims& cd = dimsOf(ports, cu);
        if (cd.size() != 1 || cd[0] < 2) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN cu_seqlens must be rank one [N + 1] with N >= 1");
        }
        shape.seqs = cd[0] - 1;
    }
    if (attributes.isL2NormQk()) {
        if (!portHasMemory(ports, q_rstd) || !portHasMemory(ports, k_rstd)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN l2norm_qk requires q_rstd and k_rstd");
        }
        requireDims(ports, q_rstd, Dims{shape.tokens, shape.heads}, "q_rstd");
        requireDims(ports, k_rstd, Dims{shape.tokens, shape.heads}, "k_rstd");
    }
    return shape;
}

void appendKey(std::vector<std::uint64_t>& key, std::uint64_t& field, const Tensor& tensor) {
    const StorageView& view = tensor.getStorageView();
    const bool has_memory   = view.getMemory() != nullptr;
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(has_memory));
    key.push_back(++field);  // 16-byte alignment class: the kernels use 16 B vector accesses
    key.push_back(has_memory ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(view.getMemory()) % 16) : 0);
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(view.getDims().size()));
    for (const std::int64_t dim : view.getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
}

std::vector<std::uint64_t> makeKey(const std::vector<Tensor>& ports, const GdnAttributes& attributes) {
    std::vector<std::uint64_t> key;
    key.reserve(ports.size() * 12 + 8);
    std::uint64_t field = 0;
    for (const Tensor& tensor : ports) { appendKey(key, field, tensor); }
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(attributes.isL2NormQk()));
    return key;
}

}  // namespace

GdnForwardProblem::GdnForwardProblem(std::vector<Tensor> ports, GdnAttributes attributes) : attributes_(attributes) {
    if (ports.size() != kCount) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN forward needs 14 ports"); }
    shape_ = validateCommon(ports, kQ, kK, kV, kG, kBeta, kCuSeqlens, kQRstd, kKRstd, kAqk, kAkk, attributes_);
    requireDims(ports, kO, dimsOf(ports, kV), "o");
    requireQkn(ports, kQn, kKn, dimsOf(ports, kQ), attributes_);
    requireState(ports, kH, shape_, portHasMemory(ports, kCuSeqlens));
    requireDevice(ports, {kQ, kK, kV, kG, kBeta, kCuSeqlens, kO, kQRstd, kKRstd, kAqk, kAkk, kQn, kKn, kH});
    tensors_ = std::move(ports);
}

std::vector<std::uint64_t> GdnForwardProblem::getProblemKey() const { return makeKey(tensors_, attributes_); }

GdnBackwardProblem::GdnBackwardProblem(std::vector<Tensor> ports, GdnAttributes attributes) : attributes_(attributes) {
    if (ports.size() != kCount) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN backward needs 19 ports"); }
    // q / k may be left out when the forward saved qn / kn: the queries and keys are then qn / kn.
    const bool has_qk = portHasMemory(ports, kQ) || portHasMemory(ports, kK);
    if (!has_qk && !portHasMemory(ports, kQn)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN backward needs q and k, or qn and kn");
    }
    const std::size_t q = has_qk ? kQ : kQn, k = has_qk ? kK : kKn;
    shape_ = validateCommon(ports, q, k, kV, kG, kBeta, kCuSeqlens, kQRstd, kKRstd, kAqk, kAkk, attributes_);
    requireQkn(ports, kQn, kKn, dimsOf(ports, q), attributes_);
    requireDims(ports, kDO, dimsOf(ports, kV), "d_o");
    requireDims(ports, kDq, dimsOf(ports, q), "dq");
    requireDims(ports, kDk, dimsOf(ports, k), "dk");
    requireDims(ports, kDv, dimsOf(ports, kV), "dv");
    requireDims(ports, kDg, dimsOf(ports, kG), "dg");
    requireDims(ports, kDBeta, dimsOf(ports, kBeta), "dbeta");

    requireState(ports, kH, shape_, portHasMemory(ports, kCuSeqlens));
    requireDevice(ports, {kDO, kQ, kK, kV, kG, kBeta, kCuSeqlens, kQRstd, kKRstd, kAqk, kAkk, kQn, kKn, kDq, kDk, kDv,
                          kDg, kDBeta, kH});
    tensors_ = std::move(ports);
}

std::vector<std::uint64_t> GdnBackwardProblem::getProblemKey() const { return makeKey(tensors_, attributes_); }

}  // namespace ftrain
