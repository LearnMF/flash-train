// CPU stand-in for the HIP runtime: one std::thread per GPU thread, CTAs run one at a time
// (so `static` models __shared__), barriers for s_barrier / wave shuffles, and the gfx936 MMAC
// instructions emulated with the lane mapping documented in g2_common.h / g2_intra_pi.cuh.
#pragma once
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__ inline
#define __forceinline__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define amdgpu_waves_per_eu(...)
#define __builtin_amdgcn_sched_barrier(x) ((void)0)
#define __builtin_amdgcn_exp2f(x) ::exp2f(x)
#define __builtin_amdgcn_readfirstlane(x) (x)
#define __builtin_amdgcn_readlane(x, l) g2_fake::shfl((x), (l))
#define __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c) g2_fake::mmac16((a), (b), (c))
#define __builtin_amdgcn_mmac_f32_16x16x8f32(a, b, c) g2_fake::mmac8((a), (b), (c))

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct float2 { float x, y; };
struct short2 { short x, y; };
struct int4 { int x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
typedef void* hipStream_t;
enum { hipSuccess = 0 };
struct hipFuncAttributes { size_t sharedSizeBytes; };
struct hipDeviceProp_t { char gcnArchName[64]; };
inline int hipGetLastError() { return 0; }
inline int hipFuncGetAttributes(hipFuncAttributes*, const void*) { return 1; }
inline int hipGetDeviceProperties(hipDeviceProp_t* p, int) { strcpy(p->gcnArchName, "gfx936"); return 0; }
template <typename T> inline T min(T a, T b) { return a < b ? a : b; }
template <typename T> inline T max(T a, T b) { return a > b ? a : b; }
inline uint32_t __float_as_uint(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
inline float __uint_as_float(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
inline int __float_as_int(float x) { return (int)__float_as_uint(x); }
inline float __int_as_float(int u) { return __uint_as_float((uint32_t)u); }

namespace g2_fake {
struct Wave { std::barrier<> b{64}; uint32_t slot[64][4]; };
struct Cta {
    explicit Cta(int n) : bar(n), waves((n + 63) / 64) {}
    std::barrier<> bar;
    std::vector<Wave> waves;
    std::mutex atom;
};
inline thread_local Cta* cta = nullptr;
inline thread_local Wave* wave = nullptr;
inline thread_local int lane = 0;
inline long n_kernels = 0;

inline void cta_barrier() { cta->bar.arrive_and_wait(); }
inline void wave_barrier() { wave->b.arrive_and_wait(); }
template <typename T> inline T shfl(T x, int src) {
    static_assert(sizeof(T) == 4);
    memcpy(&wave->slot[lane][0], &x, 4);
    wave->b.arrive_and_wait();
    T r;
    memcpy(&r, &wave->slot[src & 63][0], 4);
    wave->b.arrive_and_wait();
    return r;
}
// C[m][n] += sum_d A[m][d] * B[d][n]; a lane holds row/col (l & 15) and the 4 (2) entries
// (l >> 4) + 4e of the other index (16x16x8 fp32: 2 * (l >> 4) + i for A / B, C as above).
template <typename V4h, typename F4> inline F4 mmac16(V4h a, V4h b, F4 c) {
    uint32_t pk[4];
    for (int e = 0; e < 4; ++e) pk[e] = (uint32_t)(uint16_t)a[e] | ((uint32_t)(uint16_t)b[e] << 16);
    memcpy(wave->slot[lane], pk, 16);
    wave->b.arrive_and_wait();
    const int m = lane & 15;
    for (int e = 0; e < 4; ++e) {
        const int n = (lane >> 4) + 4 * e;
        float acc = c[e];
        for (int d = 0; d < 16; ++d) {
            const uint32_t pa = wave->slot[16 * (d & 3) + m][d >> 2];   // A[m][d]
            const uint32_t pb = wave->slot[16 * (d & 3) + n][d >> 2];   // B[d][n]
            acc += __uint_as_float((pa & 0xffffu) << 16) * __uint_as_float(pb & 0xffff0000u);
        }
        c[e] = acc;
    }
    wave->b.arrive_and_wait();
    return c;
}
template <typename F2, typename F4> inline F4 mmac8(F2 a, F2 b, F4 c) {
    float pk[4] = {a[0], a[1], b[0], b[1]};
    memcpy(wave->slot[lane], pk, 16);
    wave->b.arrive_and_wait();
    const int m = lane & 15;
    for (int e = 0; e < 4; ++e) {
        const int n = (lane >> 4) + 4 * e;
        float acc = c[e];
        for (int d = 0; d < 8; ++d) {
            float pa, pb;
            memcpy(&pa, &wave->slot[16 * (d >> 1) + m][d & 1], 4);       // A[m][d]
            memcpy(&pb, &wave->slot[16 * (d >> 1) + n][2 + (d & 1)], 4); // B[d][n]
            acc += pa * pb;
        }
        c[e] = acc;
    }
    wave->b.arrive_and_wait();
    return c;
}
}  // namespace g2_fake

struct g2_fake_idx { unsigned x, y, z; };
inline thread_local g2_fake_idx threadIdx, blockIdx;
inline dim3 gridDim, blockDim;

template <typename T> inline T __shfl(T x, int src, int = 64) { return g2_fake::shfl(x, src); }
template <typename T> inline T __shfl_xor(T x, int m, int = 64) { return g2_fake::shfl(x, g2_fake::lane ^ m); }
template <typename T> inline T __shfl_down(T x, int d, int = 64) {
    return g2_fake::shfl(x, g2_fake::lane + d < 64 ? g2_fake::lane + d : g2_fake::lane);
}
inline void __syncthreads() { g2_fake::cta_barrier(); }
inline void __syncwarp() { g2_fake::wave_barrier(); }
inline float atomicAdd(float* p, float v) {
    std::lock_guard<std::mutex> g(g2_fake::cta->atom);
    const float o = *p; *p = o + v; return o;
}

namespace g2_fake {
template <typename F, typename... A> inline void launch(F f, dim3 grid, dim3 block, A... args) {
    ++n_kernels;
    gridDim = grid; blockDim = block;
    const int n = block.x;
    if (n % 64) { fprintf(stderr, "block size %d not a multiple of 64\n", n); abort(); }
    for (unsigned by = 0; by < grid.y; ++by)
        for (unsigned bx = 0; bx < grid.x; ++bx) {
            Cta c(n);
            std::vector<std::thread> th;
            for (int t = 0; t < n; ++t)
                th.emplace_back([&, t] {
                    threadIdx = {(unsigned)t, 0, 0}; blockIdx = {bx, by, 0};
                    cta = &c; wave = &c.waves[t >> 6]; lane = t & 63;
                    f(args...);
                    wave->b.arrive_and_drop();
                    c.bar.arrive_and_drop();
                });
            for (auto& x : th) x.join();
        }
}
}  // namespace g2_fake
#define hipLaunchKernelGGL(k, g, b, sh, st, ...) g2_fake::launch(k, g, b, __VA_ARGS__)
