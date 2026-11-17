#pragma once
#include <cstdint>
#include <cstring>
struct __hip_bfloat16 { unsigned short x; };
inline __hip_bfloat16 __float2bfloat16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __hip_bfloat16{(unsigned short)(u >> 16)};
}
inline unsigned short __bfloat16_as_ushort(__hip_bfloat16 b) { return b.x; }
