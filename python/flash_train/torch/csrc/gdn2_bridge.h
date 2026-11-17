// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FLASH_TRAIN_GDN2_BRIDGE_H_
#define FLASH_TRAIN_GDN2_BRIDGE_H_
#include <cstdint>
namespace flash_train_torch {
struct Gdn2View {
    void* memory;
    std::int64_t dims[4];
    int rank;
    bool fp32;
};
std::uint64_t gdn2SavedElements(std::int64_t B, std::int64_t T, std::int64_t H, std::int64_t HV, std::int64_t K,
                                std::int64_t V);
unsigned char callGdn2(bool backward, const Gdn2View* views, float scale, bool norm, void* workspace,
                       std::uint64_t bytes, std::uint64_t* query, void* stream, const char** message);
}  // namespace flash_train_torch
#endif
