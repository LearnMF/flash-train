// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FLASH_TRAIN_TORCH_KDA_BRIDGE_H_
#define FLASH_TRAIN_TORCH_KDA_BRIDGE_H_

#include <cstdint>

// Plain-type bridge between the torch-facing and the HIP-facing binding translation units
// (the HCU torch headers and the HIP runtime headers cannot share one TU).
namespace flash_train_torch {

enum KdaBridgeType : std::int32_t { kKdaBf16 = 0, kKdaFp32 = 1, kKdaInt32 = 2 };

// One port: null memory marks an unused optional port.
struct KdaBridgeView {
    const void* memory;
    std::int64_t dims[3];
    std::int32_t num_dims;
    std::int32_t type;
};

struct KdaBridgeAttributes {
    float scale;
    float lower_bound;
    std::int32_t gate_in_kernel;
    std::int32_t l2norm_qk;
};

// views: 14 (forward) / 21 (backward) ports in the C API's port order. With query_bytes
// non-null only the recommended Primitive's workspace size is reported. Returns the
// FTrainStatus (0 = success); message receives the library's last message on failure.
// Elements of the optional state port (ftrainKdaStateElements).
std::uint64_t kdaStateElements(std::int64_t tokens, std::int64_t heads, std::int64_t seqs, bool has_cu_seqlens);

unsigned char callKdaForward(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                             std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                             const char** message);
unsigned char callKdaBackward(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                              std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                              const char** message);

}  // namespace flash_train_torch

#endif
