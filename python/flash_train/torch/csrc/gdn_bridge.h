// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FLASH_TRAIN_TORCH_GDN_BRIDGE_H_
#define FLASH_TRAIN_TORCH_GDN_BRIDGE_H_

#include <cstdint>

// Plain-type bridge between the torch-facing and the HIP-facing binding translation units
// (the HCU torch headers and the HIP runtime headers cannot share one TU).
namespace flash_train_torch {

enum GdnBridgeType : std::int32_t { kGdnBf16 = 0, kGdnFp32 = 1, kGdnInt32 = 2 };

// One port: null memory marks an unused optional port.
struct GdnBridgeView {
    const void* memory;
    std::int64_t dims[3];
    std::int32_t num_dims;
    std::int32_t type;
};

struct GdnBridgeAttributes {
    float scale;
    std::int32_t l2norm_qk;
};

// Elements of the optional h port (ftrainGdnSavedHElements).
std::uint64_t gdnSavedHElements(std::int64_t tokens, std::int64_t heads, std::int64_t seqs, bool has_cu_seqlens);

// views: 14 (forward) / 19 (backward) ports in the C API's port order. With query_bytes
// non-null only the recommended Primitive's workspace size is reported. Returns the
// FTrainStatus (0 = success); message receives the library's last message on failure.
unsigned char callGdnForward(const GdnBridgeView* views, GdnBridgeAttributes attributes, void* workspace,
                             std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                             const char** message);
unsigned char callGdnBackward(const GdnBridgeView* views, GdnBridgeAttributes attributes, void* workspace,
                              std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                              const char** message);

}  // namespace flash_train_torch

#endif
