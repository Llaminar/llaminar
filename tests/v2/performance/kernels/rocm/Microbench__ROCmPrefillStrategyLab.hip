#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "fort.hpp"

#include <omp.h>

extern "C"
{
void *rocmQuantGemm_createKernelContext(int rocm_device_id);
void rocmQuantGemm_destroyKernelContext(void *kernel_ctx, int rocm_device_id);
bool rocmQuantGemm_executeNoScale(
    const int8_t *d_A_int8,
    const int8_t *d_B_int8,
    int32_t *d_C_int32,
    int M, int N, int K,
    int rocm_device_id, void *stream,
    void *kernel_ctx);

bool rocmGemm_int8_int8_int32_vnni_prefill(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int tile_variant,
    int cpt,
    int device_id, void* stream);

bool rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int split_k_slices,
    int tile_variant,
    int cpt,
    int device_id, void* stream);
}

namespace
{
struct Shape
{
    std::string name;
    int M;
    int N;
    int K;
};

struct Options
{
    int device = 0;
    bool all_devices = false;
    std::vector<int> devices;
    int warmup = 2;
    int iters = 8;
    std::vector<Shape> shapes;
    std::vector<std::string> strategies;
    bool user_strategy_override = false;
    std::vector<int> splitk_values;
    bool check_correctness = true;
    bool full_lm_head = false;
    bool prefer_ck_wide = true;
    bool lm_head_chunked = true;
    int lm_head_chunk_n = 16384;
    bool int8_vnni_prepacked = true;
    bool persist_leaderboard = true;
    std::string leaderboard_dir = "artifacts/rocm_prefill_strategy_lab";
};

struct Result
{
    std::string shape;
    std::string strategy;
    int splitk = 1;
    double ms = 0.0;
    double repack_ms = 0.0;
    double e2e_ms = 0.0;
    double tflops = 0.0;
    double speedup_vs_ck = 0.0;
    double speedup_vs_ck_e2e = 0.0;
    double speedup_vs_ck_canonical_k = 0.0;
    double speedup_vs_ck_canonical_e2e = 0.0;
    int max_abs_diff = 0;
    bool ok = true;
};

struct CanonicalCKBaseline
{
    double mean_ms = 0.0;
    double mean_tflops = 0.0;
};

const std::unordered_map<std::string, CanonicalCKBaseline> &canonicalCKBaselines()
{
    static const std::unordered_map<std::string, CanonicalCKBaseline> kBaselines = {
        {"Qwen2.5-0.5B_AttnOut", {0.762022, 0.269705}},
        {"Qwen2.5-0.5B_FFN_Up", {1.924783, 0.579642}},
        {"Qwen2.5-0.5B_FFN_Gate", {1.932187, 0.577421}},
        {"Qwen2.5-0.5B_FFN_Down", {2.286271, 0.487993}},
        {"Qwen2.5-0.5B_LM_Head", {45.368305, 0.768168}},
        {"Qwen2.5-3B_AttnOut", {1.664761, 0.644983}},
        {"Qwen2.5-3B_FFN_Up", {4.252353, 1.357216}},
        {"Qwen2.5-3B_FFN_Gate", {4.263356, 1.353713}},
        {"Qwen2.5-3B_FFN_Down", {5.012889, 1.151305}},
        {"Qwen2.5-3B_LM_Head", {47.199634, 1.687687}},
    };
    return kBaselines;
}

const CanonicalCKBaseline *findCanonicalCKBaseline(const std::string &shape_name)
{
    const auto &baselines = canonicalCKBaselines();
    const auto it = baselines.find(shape_name);
    if (it == baselines.end())
    {
        return nullptr;
    }
    return &it->second;
}

double shapeCost(const Shape &shape)
{
    return 2.0 * static_cast<double>(shape.M) * static_cast<double>(shape.N) * static_cast<double>(shape.K);
}

__device__ __forceinline__ void store_i32x4(uint32_t *dst, int32_t v0, int32_t v1, int32_t v2, int32_t v3)
{
    *reinterpret_cast<uint4 *>(dst) = make_uint4(
        static_cast<uint32_t>(v0),
        static_cast<uint32_t>(v1),
        static_cast<uint32_t>(v2),
        static_cast<uint32_t>(v3));
}

__device__ __constant__ uint16_t k_q4_pair_decode_lut[256];

std::array<uint16_t, 256> buildQ4PairDecodeLutHost()
{
    std::array<uint16_t, 256> lut{};
    for (int b = 0; b < 256; ++b)
    {
        const int low_nib = b & 0x0F;
        const int high_nib = (b >> 4) & 0x0F;
        const int8_t low_i8 = static_cast<int8_t>(low_nib < 8 ? low_nib : (low_nib - 16));
        const int8_t high_i8 = static_cast<int8_t>(high_nib < 8 ? high_nib : (high_nib - 16));
        const uint16_t packed = static_cast<uint16_t>(
            static_cast<uint8_t>(low_i8) |
            (static_cast<uint16_t>(static_cast<uint8_t>(high_i8)) << 8));
        lut[static_cast<size_t>(b)] = packed;
    }
    return lut;
}

bool ensureQ4PairDecodeLutOnDevice(int device)
{
    static std::mutex init_mutex;
    static bool initialized[16] = {false};
    static const std::array<uint16_t, 256> host_lut = buildQ4PairDecodeLutHost();

    const int did = (device >= 0 && device < 16) ? device : 0;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized[did])
    {
        return true;
    }

    const hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(k_q4_pair_decode_lut), host_lut.data(), host_lut.size() * sizeof(uint16_t));
    if (err != hipSuccess)
    {
        return false;
    }
    initialized[did] = true;
    return true;
}

__global__ void gemm_scalar_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 16;
    constexpr int BY = 16;
    const int n = blockIdx.x * BX + threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);
        const int32_t b_packed = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    d_C[static_cast<int64_t>(m) * N + n] = acc;
}

template <int BX, int BY, int CPT>
__global__ void gemm_cpt_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BX * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int2 b2 = *reinterpret_cast<const int2 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b2.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b2.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            if ((n_base + 3) < N)
            {
                const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b4.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b4.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b4.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b4.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            d_C[static_cast<int64_t>(m) * N + n] = acc[c];
        }
    }
}

template <int BX, int BY, int CPT>
__global__ void gemm_cpt_splitk_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K,
    int splitk)
{
    const int n_base = blockIdx.x * (BX * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    const int kb = blockIdx.z;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int k_groups_total = K / 4;
    const int groups_per = (k_groups_total + splitk - 1) / splitk;
    const int g0 = kb * groups_per;
    const int g1 = min(k_groups_total, g0 + groups_per);
    if (g0 >= g1)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int kg = g0; kg < g1; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);
        if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int2 b2 = *reinterpret_cast<const int2 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b2.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b2.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}

template <int BX, int BY>
__global__ void gemm_cpt2_shared_b_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int CPT = 2;
    const int n_base = blockIdx.x * (BX * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    __shared__ int2 shared_b[BX];

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        if (threadIdx.y == 0)
        {
            if ((n_base + 1) < N)
            {
                shared_b[threadIdx.x] = *reinterpret_cast<const int2 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            }
            else
            {
                int2 tmp{};
                if (n_base < N)
                {
                    tmp.x = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                }
                tmp.y = 0;
                shared_b[threadIdx.x] = tmp;
            }
        }
        __syncthreads();

        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);
        const int2 b2 = shared_b[threadIdx.x];
        acc0 = __builtin_amdgcn_sdot4(a_packed, b2.x, acc0, false);
        if ((n_base + 1) < N)
        {
            acc1 = __builtin_amdgcn_sdot4(a_packed, b2.y, acc1, false);
        }

        __syncthreads();
    }

    d_C[static_cast<int64_t>(m) * N + n_base] = acc0;
    if ((n_base + 1) < N)
    {
        d_C[static_cast<int64_t>(m) * N + n_base + 1] = acc1;
    }
}

template <int BX, int BY>
__global__ void gemm_cpt2_shared_b_splitk_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K,
    int splitk)
{
    constexpr int CPT = 2;
    const int n_base = blockIdx.x * (BX * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    const int kb = blockIdx.z;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int k_groups_total = K / 4;
    const int groups_per = (k_groups_total + splitk - 1) / splitk;
    const int g0 = kb * groups_per;
    const int g1 = min(k_groups_total, g0 + groups_per);
    if (g0 >= g1)
    {
        return;
    }

    __shared__ int2 shared_b[BX];

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    for (int kg = g0; kg < g1; ++kg)
    {
        if (threadIdx.y == 0)
        {
            if ((n_base + 1) < N)
            {
                shared_b[threadIdx.x] = *reinterpret_cast<const int2 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            }
            else
            {
                int2 tmp{};
                if (n_base < N)
                {
                    tmp.x = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                }
                tmp.y = 0;
                shared_b[threadIdx.x] = tmp;
            }
        }
        __syncthreads();

        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);
        const int2 b2 = shared_b[threadIdx.x];
        acc0 = __builtin_amdgcn_sdot4(a_packed, b2.x, acc0, false);
        if ((n_base + 1) < N)
        {
            acc1 = __builtin_amdgcn_sdot4(a_packed, b2.y, acc1, false);
        }

        __syncthreads();
    }

    atomicAdd(&d_C[static_cast<int64_t>(m) * N + n_base], acc0);
    if ((n_base + 1) < N)
    {
        atomicAdd(&d_C[static_cast<int64_t>(m) * N + n_base + 1], acc1);
    }
}

template <int BX, int BY, int KG_TILE>
__global__ void gemm_cpt2_shared_ab_ktile_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int CPT = 2;
    const int n_base = blockIdx.x * (BX * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    __shared__ int2 shared_b[KG_TILE][BX];
    __shared__ int32_t shared_a[BY][KG_TILE];

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups_total = K / 4;
    for (int kg0 = 0; kg0 < k_groups_total; kg0 += KG_TILE)
    {
        const int tile_groups = min(KG_TILE, k_groups_total - kg0);

        if (threadIdx.y == 0 && threadIdx.x < BX)
        {
            #pragma unroll
            for (int t = 0; t < KG_TILE; ++t)
            {
                if (t < tile_groups)
                {
                    const int kg = kg0 + t;
                    if ((n_base + 1) < N)
                    {
                        shared_b[t][threadIdx.x] = *reinterpret_cast<const int2 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                    }
                    else
                    {
                        int2 tmp{};
                        if (n_base < N)
                        {
                            tmp.x = *reinterpret_cast<const int32_t *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                        }
                        tmp.y = 0;
                        shared_b[t][threadIdx.x] = tmp;
                    }
                }
            }
        }

        if (threadIdx.x < KG_TILE)
        {
            if (threadIdx.x < tile_groups)
            {
                const int kg = kg0 + threadIdx.x;
                shared_a[threadIdx.y][threadIdx.x] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + kg * 4);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < KG_TILE; ++t)
        {
            if (t < tile_groups)
            {
                const int32_t a_packed = shared_a[threadIdx.y][t];
                const int2 b2 = shared_b[t][threadIdx.x];
                acc0 = __builtin_amdgcn_sdot4(a_packed, b2.x, acc0, false);
                if ((n_base + 1) < N)
                {
                    acc1 = __builtin_amdgcn_sdot4(a_packed, b2.y, acc1, false);
                }
            }
        }
        __syncthreads();
    }

    d_C[static_cast<int64_t>(m) * N + n_base] = acc0;
    if ((n_base + 1) < N)
    {
        d_C[static_cast<int64_t>(m) * N + n_base + 1] = acc1;
    }
}

template <int BY>
__global__ void gemm_wave64_cpt4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
}

template <int BY>
__global__ void gemm_wave64_cpt4_gridstride_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M)
    {
        return;
    }

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; (n_base + 3) < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__global__ void gemm_wave64_cpt4_gridstride_aligned_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__global__ void gemm_wave64_cpt4_gridstride_aligned_unroll2_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;

        int kg = 0;
        int32_t a_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        int4 b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

        for (kg = 1; kg < k_groups; ++kg)
        {
            acc0 = __builtin_amdgcn_sdot4(a_cur, b_cur.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_cur, b_cur.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_cur, b_cur.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_cur, b_cur.w, acc3, false);

            a_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        }

        acc0 = __builtin_amdgcn_sdot4(a_cur, b_cur.x, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_cur, b_cur.y, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_cur, b_cur.z, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_cur, b_cur.w, acc3, false);

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_unroll2_lb_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;

        int kg = 0;
        int32_t a_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        int4 b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

        for (kg = 1; kg < k_groups; ++kg)
        {
            acc0 = __builtin_amdgcn_sdot4(a_cur, b_cur.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_cur, b_cur.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_cur, b_cur.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_cur, b_cur.w, acc3, false);

            a_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        }

        acc0 = __builtin_amdgcn_sdot4(a_cur, b_cur.x, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_cur, b_cur.y, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_cur, b_cur.z, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_cur, b_cur.w, acc3, false);

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_mr2_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    constexpr int MR = 2;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR;
    const int m1 = m0 + 1;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;

        int kg = 0;
        int32_t a0_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + static_cast<int64_t>(kg) * 4);
        int32_t a1_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + static_cast<int64_t>(kg) * 4);
        int4 b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

        for (kg = 1; kg < k_groups; ++kg)
        {
            acc00 = __builtin_amdgcn_sdot4(a0_cur, b_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b_cur.w, acc03, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b_cur.w, acc13, false);

            a0_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + static_cast<int64_t>(kg) * 4);
            a1_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + static_cast<int64_t>(kg) * 4);
            b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        }

        acc00 = __builtin_amdgcn_sdot4(a0_cur, b_cur.x, acc00, false);
        acc01 = __builtin_amdgcn_sdot4(a0_cur, b_cur.y, acc01, false);
        acc02 = __builtin_amdgcn_sdot4(a0_cur, b_cur.z, acc02, false);
        acc03 = __builtin_amdgcn_sdot4(a0_cur, b_cur.w, acc03, false);

        acc10 = __builtin_amdgcn_sdot4(a1_cur, b_cur.x, acc10, false);
        acc11 = __builtin_amdgcn_sdot4(a1_cur, b_cur.y, acc11, false);
        acc12 = __builtin_amdgcn_sdot4(a1_cur, b_cur.z, acc12, false);
        acc13 = __builtin_amdgcn_sdot4(a1_cur, b_cur.w, acc13, false);

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base), acc10, acc11, acc12, acc13);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_mr4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

        int kg = 0;
        int32_t a0_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + static_cast<int64_t>(kg) * 4);
        int32_t a1_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + static_cast<int64_t>(kg) * 4);
        int32_t a2_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K + static_cast<int64_t>(kg) * 4);
        int32_t a3_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K + static_cast<int64_t>(kg) * 4);
        int4 b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

        for (kg = 1; kg < k_groups; ++kg)
        {
            acc00 = __builtin_amdgcn_sdot4(a0_cur, b_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b_cur.w, acc03, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b_cur.w, acc13, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b_cur.w, acc23, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b_cur.w, acc33, false);

            a0_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + static_cast<int64_t>(kg) * 4);
            a1_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + static_cast<int64_t>(kg) * 4);
            a2_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K + static_cast<int64_t>(kg) * 4);
            a3_cur = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K + static_cast<int64_t>(kg) * 4);
            b_cur = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        }

        acc00 = __builtin_amdgcn_sdot4(a0_cur, b_cur.x, acc00, false);
        acc01 = __builtin_amdgcn_sdot4(a0_cur, b_cur.y, acc01, false);
        acc02 = __builtin_amdgcn_sdot4(a0_cur, b_cur.z, acc02, false);
        acc03 = __builtin_amdgcn_sdot4(a0_cur, b_cur.w, acc03, false);

        acc10 = __builtin_amdgcn_sdot4(a1_cur, b_cur.x, acc10, false);
        acc11 = __builtin_amdgcn_sdot4(a1_cur, b_cur.y, acc11, false);
        acc12 = __builtin_amdgcn_sdot4(a1_cur, b_cur.z, acc12, false);
        acc13 = __builtin_amdgcn_sdot4(a1_cur, b_cur.w, acc13, false);

        acc20 = __builtin_amdgcn_sdot4(a2_cur, b_cur.x, acc20, false);
        acc21 = __builtin_amdgcn_sdot4(a2_cur, b_cur.y, acc21, false);
        acc22 = __builtin_amdgcn_sdot4(a2_cur, b_cur.z, acc22, false);
        acc23 = __builtin_amdgcn_sdot4(a2_cur, b_cur.w, acc23, false);

        acc30 = __builtin_amdgcn_sdot4(a3_cur, b_cur.x, acc30, false);
        acc31 = __builtin_amdgcn_sdot4(a3_cur, b_cur.y, acc31, false);
        acc32 = __builtin_amdgcn_sdot4(a3_cur, b_cur.z, acc32, false);
        acc33 = __builtin_amdgcn_sdot4(a3_cur, b_cur.w, acc33, false);

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base), acc30, acc31, acc32, acc33);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_mr8_abcast_lane0_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    constexpr int MR = 8;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;

    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;
    const int m4 = row_group * MR + 4;
    const int m5 = row_group * MR + 5;
    const int m6 = row_group * MR + 6;
    const int m7 = row_group * MR + 7;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;
        int32_t acc40 = 0, acc41 = 0, acc42 = 0, acc43 = 0;
        int32_t acc50 = 0, acc51 = 0, acc52 = 0, acc53 = 0;
        int32_t acc60 = 0, acc61 = 0, acc62 = 0, acc63 = 0;
        int32_t acc70 = 0, acc71 = 0, acc72 = 0, acc73 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *a4_ptr = d_A + static_cast<int64_t>(m4) * K;
        const int8_t *a5_ptr = d_A + static_cast<int64_t>(m5) * K;
        const int8_t *a6_ptr = d_A + static_cast<int64_t>(m6) * K;
        const int8_t *a7_ptr = d_A + static_cast<int64_t>(m7) * K;

        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0, a1_lane = 0, a2_lane = 0, a3_lane = 0;
            int32_t a4_lane = 0, a5_lane = 0, a6_lane = 0, a7_lane = 0;

            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
                a4_lane = *reinterpret_cast<const int32_t *>(a4_ptr);
                a5_lane = *reinterpret_cast<const int32_t *>(a5_ptr);
                a6_lane = *reinterpret_cast<const int32_t *>(a6_ptr);
                a7_lane = *reinterpret_cast<const int32_t *>(a7_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);
            const int32_t a4_cur = __builtin_amdgcn_readlane(a4_lane, 0);
            const int32_t a5_cur = __builtin_amdgcn_readlane(a5_lane, 0);
            const int32_t a6_cur = __builtin_amdgcn_readlane(a6_lane, 0);
            const int32_t a7_cur = __builtin_amdgcn_readlane(a7_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);

            acc40 = __builtin_amdgcn_sdot4(a4_cur, b0_cur.x, acc40, false);
            acc41 = __builtin_amdgcn_sdot4(a4_cur, b0_cur.y, acc41, false);
            acc42 = __builtin_amdgcn_sdot4(a4_cur, b0_cur.z, acc42, false);
            acc43 = __builtin_amdgcn_sdot4(a4_cur, b0_cur.w, acc43, false);

            acc50 = __builtin_amdgcn_sdot4(a5_cur, b0_cur.x, acc50, false);
            acc51 = __builtin_amdgcn_sdot4(a5_cur, b0_cur.y, acc51, false);
            acc52 = __builtin_amdgcn_sdot4(a5_cur, b0_cur.z, acc52, false);
            acc53 = __builtin_amdgcn_sdot4(a5_cur, b0_cur.w, acc53, false);

            acc60 = __builtin_amdgcn_sdot4(a6_cur, b0_cur.x, acc60, false);
            acc61 = __builtin_amdgcn_sdot4(a6_cur, b0_cur.y, acc61, false);
            acc62 = __builtin_amdgcn_sdot4(a6_cur, b0_cur.z, acc62, false);
            acc63 = __builtin_amdgcn_sdot4(a6_cur, b0_cur.w, acc63, false);

            acc70 = __builtin_amdgcn_sdot4(a7_cur, b0_cur.x, acc70, false);
            acc71 = __builtin_amdgcn_sdot4(a7_cur, b0_cur.y, acc71, false);
            acc72 = __builtin_amdgcn_sdot4(a7_cur, b0_cur.z, acc72, false);
            acc73 = __builtin_amdgcn_sdot4(a7_cur, b0_cur.w, acc73, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            a4_ptr += 4;
            a5_ptr += 4;
            a6_ptr += 4;
            a7_ptr += 4;
            b0_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        const int64_t c4_base = static_cast<int64_t>(m4) * N + n_base;
        const int64_t c5_base = static_cast<int64_t>(m5) * N + n_base;
        const int64_t c6_base = static_cast<int64_t>(m6) * N + n_base;
        const int64_t c7_base = static_cast<int64_t>(m7) * N + n_base;

        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c4_base), acc40, acc41, acc42, acc43);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c5_base), acc50, acc51, acc52, acc53);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c6_base), acc60, acc61, acc62, acc63);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c7_base), acc70, acc71, acc72, acc73);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        int32_t a0_cur = *reinterpret_cast<const int32_t *>(a0_ptr);
        int32_t a1_cur = *reinterpret_cast<const int32_t *>(a1_ptr);
        int32_t a2_cur = *reinterpret_cast<const int32_t *>(a2_ptr);
        int32_t a3_cur = *reinterpret_cast<const int32_t *>(a3_ptr);
        int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
        int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

        #pragma unroll 2
        for (int kg = 1; kg < k_groups; ++kg)
        {
            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;

            a0_cur = *reinterpret_cast<const int32_t *>(a0_ptr);
            a1_cur = *reinterpret_cast<const int32_t *>(a1_ptr);
            a2_cur = *reinterpret_cast<const int32_t *>(a2_ptr);
            a3_cur = *reinterpret_cast<const int32_t *>(a3_ptr);
            b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);
        }

        acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
        acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
        acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
        acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
        acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
        acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
        acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
        acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

        acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
        acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
        acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
        acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
        acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
        acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
        acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
        acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

        acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
        acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
        acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
        acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
        acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
        acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
        acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
        acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

        acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
        acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
        acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
        acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
        acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
        acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
        acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
        acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * 4;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + BX * 4) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + (BX * 4)), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + (BX * 4)), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + (BX * 4)), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + (BX * 4)), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_i32addr_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * 4;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    const int32_t *a0_row = reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K);
    const int32_t *a1_row = reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K);
    const int32_t *a2_row = reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K);
    const int32_t *a3_row = reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K);

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int32_t *a0_ptr = a0_row;
        const int32_t *a1_ptr = a1_row;
        const int32_t *a2_ptr = a2_row;
        const int32_t *a3_ptr = a3_row;

        const int32_t *b0_i32 = reinterpret_cast<const int32_t *>(d_B_vnni + static_cast<int64_t>(n_base) * 4);
        const int32_t *b1_i32 = b0_i32 + BX * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *a0_ptr;
                a1_lane = *a1_ptr;
                a2_lane = *a2_ptr;
                a3_lane = *a3_ptr;
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_i32);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_i32);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            ++a0_ptr;
            ++a1_ptr;
            ++a2_ptr;
            ++a3_ptr;
            b0_i32 += N;
            b1_i32 += N;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + (BX * 4)), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + (BX * 4)), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + (BX * 4)), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + (BX * 4)), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__device__ __forceinline__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_body(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * 4;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        const int8_t *a0_row = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_row = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_row = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_row = d_A + static_cast<int64_t>(m3) * K;

        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

        const int8_t *a0_ptr = a0_row;
        const int8_t *a1_ptr = a1_row;
        const int8_t *a2_ptr = a2_row;
        const int8_t *a3_ptr = a3_row;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);
            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
        }

        int32_t acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        a0_ptr = a0_row;
        a1_ptr = a1_row;
        a2_ptr = a2_row;
        a3_ptr = a3_row;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + BX * 4) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + (BX * 4)), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + (BX * 4)), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + (BX * 4)), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + (BX * 4)), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_body<BY>(d_A, d_B_vnni, d_C, M, N, K);
}

template <int BY>
__launch_bounds__(64 * BY, 3) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_lb3_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_body<BY>(d_A, d_B_vnni, d_C, M, N, K);
}

template <int BY>
__launch_bounds__(64 * BY, 4) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_lb4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_body<BY>(d_A, d_B_vnni, d_C, M, N, K);
}

template <int BY, int KGT>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_ldsB_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    __shared__ int4 s_b0[KGT][BX];
    __shared__ int4 s_b1[KGT][BX];

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_row = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_row = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_row = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_row = d_A + static_cast<int64_t>(m3) * K;

        for (int kg0 = 0; kg0 < k_groups; kg0 += KGT)
        {
            const int tile = min(KGT, k_groups - kg0);

            if (ty == 0)
            {
                const int8_t *b0_tile = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4 + static_cast<int64_t>(kg0) * b_group_stride;
                const int8_t *b1_tile = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4 + static_cast<int64_t>(kg0) * b_group_stride;
                #pragma unroll
                for (int t = 0; t < KGT; ++t)
                {
                    if (t < tile)
                    {
                        s_b0[t][lane] = *reinterpret_cast<const int4 *>(b0_tile + static_cast<int64_t>(t) * b_group_stride);
                        s_b1[t][lane] = *reinterpret_cast<const int4 *>(b1_tile + static_cast<int64_t>(t) * b_group_stride);
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int t = 0; t < KGT; ++t)
            {
                if (t >= tile)
                {
                    continue;
                }

                const int kg = kg0 + t;
                int32_t a0_lane = 0;
                int32_t a1_lane = 0;
                int32_t a2_lane = 0;
                int32_t a3_lane = 0;
                if (lane == 0)
                {
                    a0_lane = *reinterpret_cast<const int32_t *>(a0_row + static_cast<int64_t>(kg) * 4);
                    a1_lane = *reinterpret_cast<const int32_t *>(a1_row + static_cast<int64_t>(kg) * 4);
                    a2_lane = *reinterpret_cast<const int32_t *>(a2_row + static_cast<int64_t>(kg) * 4);
                    a3_lane = *reinterpret_cast<const int32_t *>(a3_row + static_cast<int64_t>(kg) * 4);
                }

                const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
                const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
                const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
                const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);
                const int4 b0_cur = s_b0[t][lane];
                const int4 b1_cur = s_b1[t][lane];

                acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
                acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
                acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
                acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
                acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
                acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
                acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
                acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

                acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
                acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
                acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
                acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
                acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
                acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
                acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
                acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

                acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
                acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
                acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
                acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
                acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
                acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
                acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
                acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

                acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
                acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
                acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
                acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
                acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
                acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
                acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
                acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);
            }
            __syncthreads();
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_avec4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        for (int kg = 0; kg < k_groups; kg += 4)
        {
            int4 a0_pack = make_int4(0, 0, 0, 0);
            int4 a1_pack = make_int4(0, 0, 0, 0);
            int4 a2_pack = make_int4(0, 0, 0, 0);
            int4 a3_pack = make_int4(0, 0, 0, 0);
            if (lane == 0)
            {
                a0_pack = *reinterpret_cast<const int4 *>(a0_ptr);
                a1_pack = *reinterpret_cast<const int4 *>(a1_ptr);
                a2_pack = *reinterpret_cast<const int4 *>(a2_ptr);
                a3_pack = *reinterpret_cast<const int4 *>(a3_ptr);
            }

            #pragma unroll
            for (int s = 0; s < 4; ++s)
            {
                if (kg + s >= k_groups)
                {
                    continue;
                }

                int32_t a0_lane = 0;
                int32_t a1_lane = 0;
                int32_t a2_lane = 0;
                int32_t a3_lane = 0;
                if (s == 0)
                {
                    a0_lane = a0_pack.x;
                    a1_lane = a1_pack.x;
                    a2_lane = a2_pack.x;
                    a3_lane = a3_pack.x;
                }
                else if (s == 1)
                {
                    a0_lane = a0_pack.y;
                    a1_lane = a1_pack.y;
                    a2_lane = a2_pack.y;
                    a3_lane = a3_pack.y;
                }
                else if (s == 2)
                {
                    a0_lane = a0_pack.z;
                    a1_lane = a1_pack.z;
                    a2_lane = a2_pack.z;
                    a3_lane = a3_pack.z;
                }
                else
                {
                    a0_lane = a0_pack.w;
                    a1_lane = a1_pack.w;
                    a2_lane = a2_pack.w;
                    a3_lane = a3_pack.w;
                }

                const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
                const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
                const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
                const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

                const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr + static_cast<int64_t>(s) * b_group_stride);
                const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr + static_cast<int64_t>(s) * b_group_stride);

                acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
                acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
                acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
                acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
                acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
                acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
                acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
                acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

                acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
                acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
                acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
                acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
                acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
                acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
                acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
                acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

                acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
                acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
                acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
                acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
                acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
                acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
                acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
                acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

                acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
                acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
                acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
                acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
                acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
                acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
                acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
                acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);
            }

            a0_ptr += 16;
            a1_ptr += 16;
            a2_ptr += 16;
            a3_ptr += 16;
            b0_ptr += static_cast<int64_t>(4) * b_group_stride;
            b1_ptr += static_cast<int64_t>(4) * b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 1) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_lb1_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coal_pipe_u4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * 4;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + BX * 4) * 4;

        int32_t a0_lane_next = 0;
        int32_t a1_lane_next = 0;
        int32_t a2_lane_next = 0;
        int32_t a3_lane_next = 0;
        if (lane == 0)
        {
            a0_lane_next = *reinterpret_cast<const int32_t *>(a0_ptr);
            a1_lane_next = *reinterpret_cast<const int32_t *>(a1_ptr);
            a2_lane_next = *reinterpret_cast<const int32_t *>(a2_ptr);
            a3_lane_next = *reinterpret_cast<const int32_t *>(a3_ptr);
        }
        int4 b0_next = *reinterpret_cast<const int4 *>(b0_ptr);
        int4 b1_next = *reinterpret_cast<const int4 *>(b1_ptr);

        #pragma unroll 4
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane_next, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane_next, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane_next, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane_next, 0);
            const int4 b0_cur = b0_next;
            const int4 b1_cur = b1_next;

            if (kg + 1 < k_groups)
            {
                if (lane == 0)
                {
                    a0_lane_next = *reinterpret_cast<const int32_t *>(a0_ptr + 4);
                    a1_lane_next = *reinterpret_cast<const int32_t *>(a1_ptr + 4);
                    a2_lane_next = *reinterpret_cast<const int32_t *>(a2_ptr + 4);
                    a3_lane_next = *reinterpret_cast<const int32_t *>(a3_ptr + 4);
                }
                b0_next = *reinterpret_cast<const int4 *>(b0_ptr + b_group_stride);
                b1_next = *reinterpret_cast<const int4 *>(b1_ptr + b_group_stride);
            }

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + (BX * 4)), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + (BX * 4)), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + (BX * 4)), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + (BX * 4)), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_pf1_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        int32_t a0_lane = 0;
        int32_t a1_lane = 0;
        int32_t a2_lane = 0;
        int32_t a3_lane = 0;
        if (lane == 0)
        {
            a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
            a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
            a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
            a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
        }

        int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
        int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
        int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
        int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);
        int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
        int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

        #pragma unroll 2
        for (int kg = 1; kg < k_groups; ++kg)
        {
            const int8_t *a0_next_ptr = a0_ptr + 4;
            const int8_t *a1_next_ptr = a1_ptr + 4;
            const int8_t *a2_next_ptr = a2_ptr + 4;
            const int8_t *a3_next_ptr = a3_ptr + 4;
            const int8_t *b0_next_ptr = b0_ptr + b_group_stride;
            const int8_t *b1_next_ptr = b1_ptr + b_group_stride;

            __builtin_prefetch(b0_next_ptr + b_group_stride, 0, 1);
            __builtin_prefetch(b1_next_ptr + b_group_stride, 0, 1);

            int32_t a0_next_lane = 0;
            int32_t a1_next_lane = 0;
            int32_t a2_next_lane = 0;
            int32_t a3_next_lane = 0;
            if (lane == 0)
            {
                a0_next_lane = *reinterpret_cast<const int32_t *>(a0_next_ptr);
                a1_next_lane = *reinterpret_cast<const int32_t *>(a1_next_ptr);
                a2_next_lane = *reinterpret_cast<const int32_t *>(a2_next_ptr);
                a3_next_lane = *reinterpret_cast<const int32_t *>(a3_next_ptr);
            }
            const int4 b0_next = *reinterpret_cast<const int4 *>(b0_next_ptr);
            const int4 b1_next = *reinterpret_cast<const int4 *>(b1_next_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr = a0_next_ptr;
            a1_ptr = a1_next_ptr;
            a2_ptr = a2_next_ptr;
            a3_ptr = a3_next_ptr;
            b0_ptr = b0_next_ptr;
            b1_ptr = b1_next_ptr;

            a0_cur = __builtin_amdgcn_readlane(a0_next_lane, 0);
            a1_cur = __builtin_amdgcn_readlane(a1_next_lane, 0);
            a2_cur = __builtin_amdgcn_readlane(a2_next_lane, 0);
            a3_cur = __builtin_amdgcn_readlane(a3_next_lane, 0);
            b0_cur = b0_next;
            b1_cur = b1_next;
        }

        acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
        acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
        acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
        acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
        acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
        acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
        acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
        acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

        acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
        acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
        acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
        acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
        acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
        acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
        acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
        acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

        acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
        acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
        acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
        acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
        acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
        acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
        acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
        acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

        acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
        acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
        acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
        acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
        acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
        acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
        acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
        acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_k2_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        const int k_groups2 = (k_groups / 2) * 2;
        #pragma unroll 2
        for (int kg = 0; kg < k_groups2; kg += 2)
        {
            int2 a0_lane2 = make_int2(0, 0);
            int2 a1_lane2 = make_int2(0, 0);
            int2 a2_lane2 = make_int2(0, 0);
            int2 a3_lane2 = make_int2(0, 0);
            if (lane == 0)
            {
                a0_lane2 = *reinterpret_cast<const int2 *>(a0_ptr);
                a1_lane2 = *reinterpret_cast<const int2 *>(a1_ptr);
                a2_lane2 = *reinterpret_cast<const int2 *>(a2_ptr);
                a3_lane2 = *reinterpret_cast<const int2 *>(a3_ptr);
            }

            const int32_t a00 = __builtin_amdgcn_readlane(a0_lane2.x, 0);
            const int32_t a01 = __builtin_amdgcn_readlane(a0_lane2.y, 0);
            const int32_t a10 = __builtin_amdgcn_readlane(a1_lane2.x, 0);
            const int32_t a11 = __builtin_amdgcn_readlane(a1_lane2.y, 0);
            const int32_t a20 = __builtin_amdgcn_readlane(a2_lane2.x, 0);
            const int32_t a21 = __builtin_amdgcn_readlane(a2_lane2.y, 0);
            const int32_t a30 = __builtin_amdgcn_readlane(a3_lane2.x, 0);
            const int32_t a31 = __builtin_amdgcn_readlane(a3_lane2.y, 0);

            const int4 b0_0 = *reinterpret_cast<const int4 *>(b0_ptr + 0 * b_group_stride);
            const int4 b1_0 = *reinterpret_cast<const int4 *>(b1_ptr + 0 * b_group_stride);
            const int4 b0_1 = *reinterpret_cast<const int4 *>(b0_ptr + 1 * b_group_stride);
            const int4 b1_1 = *reinterpret_cast<const int4 *>(b1_ptr + 1 * b_group_stride);

            acc00 = __builtin_amdgcn_sdot4(a00, b0_0.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a00, b0_0.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a00, b0_0.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a00, b0_0.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a00, b1_0.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a00, b1_0.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a00, b1_0.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a00, b1_0.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a10, b0_0.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a10, b0_0.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a10, b0_0.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a10, b0_0.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a10, b1_0.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a10, b1_0.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a10, b1_0.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a10, b1_0.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a20, b0_0.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a20, b0_0.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a20, b0_0.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a20, b0_0.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a20, b1_0.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a20, b1_0.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a20, b1_0.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a20, b1_0.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a30, b0_0.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a30, b0_0.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a30, b0_0.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a30, b0_0.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a30, b1_0.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a30, b1_0.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a30, b1_0.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a30, b1_0.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a01, b0_1.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a01, b0_1.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a01, b0_1.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a01, b0_1.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a01, b1_1.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a01, b1_1.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a01, b1_1.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a01, b1_1.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a11, b0_1.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a11, b0_1.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a11, b0_1.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a11, b0_1.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a11, b1_1.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a11, b1_1.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a11, b1_1.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a11, b1_1.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a21, b0_1.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a21, b0_1.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a21, b0_1.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a21, b0_1.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a21, b1_1.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a21, b1_1.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a21, b1_1.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a21, b1_1.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a31, b0_1.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a31, b0_1.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a31, b0_1.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a31, b0_1.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a31, b1_1.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a31, b1_1.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a31, b1_1.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a31, b1_1.w, acc37, false);

            a0_ptr += 8;
            a1_ptr += 8;
            a2_ptr += 8;
            a3_ptr += 8;
            b0_ptr += 2 * b_group_stride;
            b1_ptr += 2 * b_group_stride;
        }

        for (int kg = k_groups2; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_cpt16split_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 16;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc08 = 0, acc09 = 0, acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0;
        int32_t acc16 = 0, acc17 = 0, acc18 = 0, acc19 = 0, acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0, acc28 = 0, acc29 = 0, acc30 = 0, acc31 = 0;
        int32_t acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0, acc38 = 0, acc39 = 0;
        int32_t acc40 = 0, acc41 = 0, acc42 = 0, acc43 = 0, acc44 = 0, acc45 = 0, acc46 = 0, acc47 = 0;
        int32_t acc48 = 0, acc49 = 0, acc50 = 0, acc51 = 0, acc52 = 0, acc53 = 0, acc54 = 0, acc55 = 0;
        int32_t acc56 = 0, acc57 = 0, acc58 = 0, acc59 = 0, acc60 = 0, acc61 = 0, acc62 = 0, acc63 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;
        const int8_t *b2_ptr = d_B_vnni + static_cast<int64_t>(n_base + 8) * 4;
        const int8_t *b3_ptr = d_B_vnni + static_cast<int64_t>(n_base + 12) * 4;

        #pragma unroll 2
        for (int kg = 0; kg < k_groups; ++kg)
        {
            int32_t a0_lane = 0;
            int32_t a1_lane = 0;
            int32_t a2_lane = 0;
            int32_t a3_lane = 0;
            if (lane == 0)
            {
                a0_lane = *reinterpret_cast<const int32_t *>(a0_ptr);
                a1_lane = *reinterpret_cast<const int32_t *>(a1_ptr);
                a2_lane = *reinterpret_cast<const int32_t *>(a2_ptr);
                a3_lane = *reinterpret_cast<const int32_t *>(a3_ptr);
            }

            const int32_t a0_cur = __builtin_amdgcn_readlane(a0_lane, 0);
            const int32_t a1_cur = __builtin_amdgcn_readlane(a1_lane, 0);
            const int32_t a2_cur = __builtin_amdgcn_readlane(a2_lane, 0);
            const int32_t a3_cur = __builtin_amdgcn_readlane(a3_lane, 0);

            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);
            const int4 b2_cur = *reinterpret_cast<const int4 *>(b2_ptr);
            const int4 b3_cur = *reinterpret_cast<const int4 *>(b3_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);
            acc08 = __builtin_amdgcn_sdot4(a0_cur, b2_cur.x, acc08, false); acc09 = __builtin_amdgcn_sdot4(a0_cur, b2_cur.y, acc09, false); acc10 = __builtin_amdgcn_sdot4(a0_cur, b2_cur.z, acc10, false); acc11 = __builtin_amdgcn_sdot4(a0_cur, b2_cur.w, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a0_cur, b3_cur.x, acc12, false); acc13 = __builtin_amdgcn_sdot4(a0_cur, b3_cur.y, acc13, false); acc14 = __builtin_amdgcn_sdot4(a0_cur, b3_cur.z, acc14, false); acc15 = __builtin_amdgcn_sdot4(a0_cur, b3_cur.w, acc15, false);

            acc16 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc16, false); acc17 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc17, false); acc18 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc18, false); acc19 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc19, false);
            acc20 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a1_cur, b2_cur.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a1_cur, b2_cur.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a1_cur, b2_cur.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a1_cur, b2_cur.w, acc27, false);
            acc28 = __builtin_amdgcn_sdot4(a1_cur, b3_cur.x, acc28, false); acc29 = __builtin_amdgcn_sdot4(a1_cur, b3_cur.y, acc29, false); acc30 = __builtin_amdgcn_sdot4(a1_cur, b3_cur.z, acc30, false); acc31 = __builtin_amdgcn_sdot4(a1_cur, b3_cur.w, acc31, false);

            acc32 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc32, false); acc33 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc33, false); acc34 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc34, false); acc35 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc36, false); acc37 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc37, false); acc38 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc38, false); acc39 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc39, false);
            acc40 = __builtin_amdgcn_sdot4(a2_cur, b2_cur.x, acc40, false); acc41 = __builtin_amdgcn_sdot4(a2_cur, b2_cur.y, acc41, false); acc42 = __builtin_amdgcn_sdot4(a2_cur, b2_cur.z, acc42, false); acc43 = __builtin_amdgcn_sdot4(a2_cur, b2_cur.w, acc43, false);
            acc44 = __builtin_amdgcn_sdot4(a2_cur, b3_cur.x, acc44, false); acc45 = __builtin_amdgcn_sdot4(a2_cur, b3_cur.y, acc45, false); acc46 = __builtin_amdgcn_sdot4(a2_cur, b3_cur.z, acc46, false); acc47 = __builtin_amdgcn_sdot4(a2_cur, b3_cur.w, acc47, false);

            acc48 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc48, false); acc49 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc49, false); acc50 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc50, false); acc51 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc51, false);
            acc52 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc52, false); acc53 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc53, false); acc54 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc54, false); acc55 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc55, false);
            acc56 = __builtin_amdgcn_sdot4(a3_cur, b2_cur.x, acc56, false); acc57 = __builtin_amdgcn_sdot4(a3_cur, b2_cur.y, acc57, false); acc58 = __builtin_amdgcn_sdot4(a3_cur, b2_cur.z, acc58, false); acc59 = __builtin_amdgcn_sdot4(a3_cur, b2_cur.w, acc59, false);
            acc60 = __builtin_amdgcn_sdot4(a3_cur, b3_cur.x, acc60, false); acc61 = __builtin_amdgcn_sdot4(a3_cur, b3_cur.y, acc61, false); acc62 = __builtin_amdgcn_sdot4(a3_cur, b3_cur.z, acc62, false); acc63 = __builtin_amdgcn_sdot4(a3_cur, b3_cur.w, acc63, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
            b2_ptr += b_group_stride;
            b3_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 8), acc08, acc09, acc10, acc11);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 12), acc12, acc13, acc14, acc15);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc16, acc17, acc18, acc19);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 8), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 12), acc28, acc29, acc30, acc31);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc32, acc33, acc34, acc35);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc36, acc37, acc38, acc39);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 8), acc40, acc41, acc42, acc43);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 12), acc44, acc45, acc46, acc47);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc48, acc49, acc50, acc51);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc52, acc53, acc54, acc55);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 8), acc56, acc57, acc58, acc59);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 12), acc60, acc61, acc62, acc63);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_veca2_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        const int k_groups2 = (k_groups / 2) * 2;
        #pragma unroll 2
        for (int kg = 0; kg < k_groups2; kg += 2)
        {
            const int2 a0_vec = *reinterpret_cast<const int2 *>(a0_ptr);
            const int2 a1_vec = *reinterpret_cast<const int2 *>(a1_ptr);
            const int2 a2_vec = *reinterpret_cast<const int2 *>(a2_ptr);
            const int2 a3_vec = *reinterpret_cast<const int2 *>(a3_ptr);

            const int4 b0_0 = *reinterpret_cast<const int4 *>(b0_ptr + 0 * b_group_stride);
            const int4 b1_0 = *reinterpret_cast<const int4 *>(b1_ptr + 0 * b_group_stride);

            acc00 = __builtin_amdgcn_sdot4(a0_vec.x, b0_0.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_vec.x, b0_0.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_vec.x, b0_0.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_vec.x, b0_0.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_vec.x, b1_0.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_vec.x, b1_0.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_vec.x, b1_0.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_vec.x, b1_0.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_vec.x, b0_0.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_vec.x, b0_0.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_vec.x, b0_0.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_vec.x, b0_0.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_vec.x, b1_0.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_vec.x, b1_0.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_vec.x, b1_0.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_vec.x, b1_0.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_vec.x, b0_0.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_vec.x, b0_0.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_vec.x, b0_0.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_vec.x, b0_0.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_vec.x, b1_0.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_vec.x, b1_0.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_vec.x, b1_0.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_vec.x, b1_0.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_vec.x, b0_0.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_vec.x, b0_0.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_vec.x, b0_0.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_vec.x, b0_0.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_vec.x, b1_0.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_vec.x, b1_0.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_vec.x, b1_0.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_vec.x, b1_0.w, acc37, false);

            const int4 b0_1 = *reinterpret_cast<const int4 *>(b0_ptr + 1 * b_group_stride);
            const int4 b1_1 = *reinterpret_cast<const int4 *>(b1_ptr + 1 * b_group_stride);

            acc00 = __builtin_amdgcn_sdot4(a0_vec.y, b0_1.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_vec.y, b0_1.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_vec.y, b0_1.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_vec.y, b0_1.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_vec.y, b1_1.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_vec.y, b1_1.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_vec.y, b1_1.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_vec.y, b1_1.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_vec.y, b0_1.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_vec.y, b0_1.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_vec.y, b0_1.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_vec.y, b0_1.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_vec.y, b1_1.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_vec.y, b1_1.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_vec.y, b1_1.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_vec.y, b1_1.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_vec.y, b0_1.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_vec.y, b0_1.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_vec.y, b0_1.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_vec.y, b0_1.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_vec.y, b1_1.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_vec.y, b1_1.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_vec.y, b1_1.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_vec.y, b1_1.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_vec.y, b0_1.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_vec.y, b0_1.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_vec.y, b0_1.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_vec.y, b0_1.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_vec.y, b1_1.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_vec.y, b1_1.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_vec.y, b1_1.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_vec.y, b1_1.w, acc37, false);

            a0_ptr += 8;
            a1_ptr += 8;
            a2_ptr += 8;
            a3_ptr += 8;
            b0_ptr += 2 * b_group_stride;
            b1_ptr += 2 * b_group_stride;
        }

        for (int kg = k_groups2; kg < k_groups; ++kg)
        {
            const int32_t a0_cur = *reinterpret_cast<const int32_t *>(a0_ptr);
            const int32_t a1_cur = *reinterpret_cast<const int32_t *>(a1_ptr);
            const int32_t a2_cur = *reinterpret_cast<const int32_t *>(a2_ptr);
            const int32_t a3_cur = *reinterpret_cast<const int32_t *>(a3_ptr);
            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_veca4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        const int k_groups4 = (k_groups / 4) * 4;
        for (int kg = 0; kg < k_groups4; kg += 4)
        {
            const int4 a0_vec = *reinterpret_cast<const int4 *>(a0_ptr);
            const int4 a1_vec = *reinterpret_cast<const int4 *>(a1_ptr);
            const int4 a2_vec = *reinterpret_cast<const int4 *>(a2_ptr);
            const int4 a3_vec = *reinterpret_cast<const int4 *>(a3_ptr);

            const int4 b0_0 = *reinterpret_cast<const int4 *>(b0_ptr + 0 * b_group_stride);
            const int4 b1_0 = *reinterpret_cast<const int4 *>(b1_ptr + 0 * b_group_stride);
            const int4 b0_1 = *reinterpret_cast<const int4 *>(b0_ptr + 1 * b_group_stride);
            const int4 b1_1 = *reinterpret_cast<const int4 *>(b1_ptr + 1 * b_group_stride);
            const int4 b0_2 = *reinterpret_cast<const int4 *>(b0_ptr + 2 * b_group_stride);
            const int4 b1_2 = *reinterpret_cast<const int4 *>(b1_ptr + 2 * b_group_stride);
            const int4 b0_3 = *reinterpret_cast<const int4 *>(b0_ptr + 3 * b_group_stride);
            const int4 b1_3 = *reinterpret_cast<const int4 *>(b1_ptr + 3 * b_group_stride);

            const int32_t a00 = a0_vec.x;
            const int32_t a01 = a0_vec.y;
            const int32_t a02 = a0_vec.z;
            const int32_t a03 = a0_vec.w;
            const int32_t a10 = a1_vec.x;
            const int32_t a11 = a1_vec.y;
            const int32_t a12 = a1_vec.z;
            const int32_t a13 = a1_vec.w;
            const int32_t a20 = a2_vec.x;
            const int32_t a21 = a2_vec.y;
            const int32_t a22 = a2_vec.z;
            const int32_t a23 = a2_vec.w;
            const int32_t a30 = a3_vec.x;
            const int32_t a31 = a3_vec.y;
            const int32_t a32 = a3_vec.z;
            const int32_t a33 = a3_vec.w;

            acc00 = __builtin_amdgcn_sdot4(a00, b0_0.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a00, b0_0.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a00, b0_0.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a00, b0_0.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a00, b1_0.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a00, b1_0.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a00, b1_0.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a00, b1_0.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a10, b0_0.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a10, b0_0.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a10, b0_0.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a10, b0_0.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a10, b1_0.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a10, b1_0.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a10, b1_0.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a10, b1_0.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a20, b0_0.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a20, b0_0.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a20, b0_0.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a20, b0_0.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a20, b1_0.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a20, b1_0.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a20, b1_0.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a20, b1_0.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a30, b0_0.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a30, b0_0.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a30, b0_0.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a30, b0_0.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a30, b1_0.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a30, b1_0.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a30, b1_0.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a30, b1_0.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a01, b0_1.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a01, b0_1.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a01, b0_1.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a01, b0_1.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a01, b1_1.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a01, b1_1.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a01, b1_1.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a01, b1_1.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a11, b0_1.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a11, b0_1.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a11, b0_1.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a11, b0_1.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a11, b1_1.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a11, b1_1.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a11, b1_1.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a11, b1_1.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a21, b0_1.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a21, b0_1.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a21, b0_1.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a21, b0_1.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a21, b1_1.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a21, b1_1.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a21, b1_1.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a21, b1_1.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a31, b0_1.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a31, b0_1.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a31, b0_1.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a31, b0_1.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a31, b1_1.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a31, b1_1.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a31, b1_1.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a31, b1_1.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a02, b0_2.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a02, b0_2.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a02, b0_2.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a02, b0_2.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a02, b1_2.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a02, b1_2.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a02, b1_2.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a02, b1_2.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a12, b0_2.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a12, b0_2.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a12, b0_2.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a12, b0_2.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a12, b1_2.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a12, b1_2.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a12, b1_2.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a12, b1_2.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a22, b0_2.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a22, b0_2.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a22, b0_2.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a22, b0_2.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a22, b1_2.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a22, b1_2.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a22, b1_2.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a22, b1_2.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a32, b0_2.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a32, b0_2.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a32, b0_2.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a32, b0_2.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a32, b1_2.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a32, b1_2.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a32, b1_2.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a32, b1_2.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a03, b0_3.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a03, b0_3.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a03, b0_3.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a03, b0_3.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a03, b1_3.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a03, b1_3.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a03, b1_3.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a03, b1_3.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a13, b0_3.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a13, b0_3.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a13, b0_3.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a13, b0_3.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a13, b1_3.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a13, b1_3.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a13, b1_3.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a13, b1_3.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a23, b0_3.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a23, b0_3.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a23, b0_3.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a23, b0_3.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a23, b1_3.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a23, b1_3.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a23, b1_3.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a23, b1_3.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a33, b0_3.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a33, b0_3.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a33, b0_3.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a33, b0_3.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a33, b1_3.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a33, b1_3.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a33, b1_3.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a33, b1_3.w, acc37, false);

            a0_ptr += 16;
            a1_ptr += 16;
            a2_ptr += 16;
            a3_ptr += 16;
            b0_ptr += 4 * b_group_stride;
            b1_ptr += 4 * b_group_stride;
        }

        for (int kg = k_groups4; kg < k_groups; ++kg)
        {
            const int32_t a0_cur = *reinterpret_cast<const int32_t *>(a0_ptr);
            const int32_t a1_cur = *reinterpret_cast<const int32_t *>(a1_ptr);
            const int32_t a2_cur = *reinterpret_cast<const int32_t *>(a2_ptr);
            const int32_t a3_cur = *reinterpret_cast<const int32_t *>(a3_ptr);
            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_veca4_rfl_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int64_t b_group_stride = static_cast<int64_t>(N) * 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int8_t *a0_ptr = d_A + static_cast<int64_t>(m0) * K;
        const int8_t *a1_ptr = d_A + static_cast<int64_t>(m1) * K;
        const int8_t *a2_ptr = d_A + static_cast<int64_t>(m2) * K;
        const int8_t *a3_ptr = d_A + static_cast<int64_t>(m3) * K;
        const int8_t *b0_ptr = d_B_vnni + static_cast<int64_t>(n_base + 0) * 4;
        const int8_t *b1_ptr = d_B_vnni + static_cast<int64_t>(n_base + 4) * 4;

        const int k_groups4 = (k_groups / 4) * 4;
        for (int kg = 0; kg < k_groups4; kg += 4)
        {
            const int4 a0_vec = *reinterpret_cast<const int4 *>(a0_ptr);
            const int4 a1_vec = *reinterpret_cast<const int4 *>(a1_ptr);
            const int4 a2_vec = *reinterpret_cast<const int4 *>(a2_ptr);
            const int4 a3_vec = *reinterpret_cast<const int4 *>(a3_ptr);

            const int32_t a00 = __builtin_amdgcn_readfirstlane(a0_vec.x);
            const int32_t a01 = __builtin_amdgcn_readfirstlane(a0_vec.y);
            const int32_t a02 = __builtin_amdgcn_readfirstlane(a0_vec.z);
            const int32_t a03 = __builtin_amdgcn_readfirstlane(a0_vec.w);
            const int32_t a10 = __builtin_amdgcn_readfirstlane(a1_vec.x);
            const int32_t a11 = __builtin_amdgcn_readfirstlane(a1_vec.y);
            const int32_t a12 = __builtin_amdgcn_readfirstlane(a1_vec.z);
            const int32_t a13 = __builtin_amdgcn_readfirstlane(a1_vec.w);
            const int32_t a20 = __builtin_amdgcn_readfirstlane(a2_vec.x);
            const int32_t a21 = __builtin_amdgcn_readfirstlane(a2_vec.y);
            const int32_t a22 = __builtin_amdgcn_readfirstlane(a2_vec.z);
            const int32_t a23 = __builtin_amdgcn_readfirstlane(a2_vec.w);
            const int32_t a30 = __builtin_amdgcn_readfirstlane(a3_vec.x);
            const int32_t a31 = __builtin_amdgcn_readfirstlane(a3_vec.y);
            const int32_t a32 = __builtin_amdgcn_readfirstlane(a3_vec.z);
            const int32_t a33 = __builtin_amdgcn_readfirstlane(a3_vec.w);

            const int4 b0_0 = *reinterpret_cast<const int4 *>(b0_ptr + 0 * b_group_stride);
            const int4 b1_0 = *reinterpret_cast<const int4 *>(b1_ptr + 0 * b_group_stride);
            const int4 b0_1 = *reinterpret_cast<const int4 *>(b0_ptr + 1 * b_group_stride);
            const int4 b1_1 = *reinterpret_cast<const int4 *>(b1_ptr + 1 * b_group_stride);
            const int4 b0_2 = *reinterpret_cast<const int4 *>(b0_ptr + 2 * b_group_stride);
            const int4 b1_2 = *reinterpret_cast<const int4 *>(b1_ptr + 2 * b_group_stride);
            const int4 b0_3 = *reinterpret_cast<const int4 *>(b0_ptr + 3 * b_group_stride);
            const int4 b1_3 = *reinterpret_cast<const int4 *>(b1_ptr + 3 * b_group_stride);

            acc00 = __builtin_amdgcn_sdot4(a00, b0_0.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a00, b0_0.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a00, b0_0.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a00, b0_0.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a00, b1_0.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a00, b1_0.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a00, b1_0.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a00, b1_0.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a10, b0_0.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a10, b0_0.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a10, b0_0.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a10, b0_0.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a10, b1_0.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a10, b1_0.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a10, b1_0.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a10, b1_0.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a20, b0_0.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a20, b0_0.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a20, b0_0.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a20, b0_0.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a20, b1_0.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a20, b1_0.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a20, b1_0.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a20, b1_0.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a30, b0_0.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a30, b0_0.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a30, b0_0.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a30, b0_0.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a30, b1_0.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a30, b1_0.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a30, b1_0.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a30, b1_0.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a01, b0_1.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a01, b0_1.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a01, b0_1.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a01, b0_1.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a01, b1_1.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a01, b1_1.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a01, b1_1.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a01, b1_1.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a11, b0_1.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a11, b0_1.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a11, b0_1.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a11, b0_1.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a11, b1_1.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a11, b1_1.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a11, b1_1.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a11, b1_1.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a21, b0_1.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a21, b0_1.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a21, b0_1.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a21, b0_1.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a21, b1_1.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a21, b1_1.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a21, b1_1.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a21, b1_1.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a31, b0_1.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a31, b0_1.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a31, b0_1.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a31, b0_1.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a31, b1_1.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a31, b1_1.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a31, b1_1.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a31, b1_1.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a02, b0_2.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a02, b0_2.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a02, b0_2.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a02, b0_2.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a02, b1_2.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a02, b1_2.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a02, b1_2.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a02, b1_2.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a12, b0_2.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a12, b0_2.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a12, b0_2.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a12, b0_2.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a12, b1_2.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a12, b1_2.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a12, b1_2.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a12, b1_2.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a22, b0_2.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a22, b0_2.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a22, b0_2.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a22, b0_2.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a22, b1_2.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a22, b1_2.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a22, b1_2.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a22, b1_2.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a32, b0_2.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a32, b0_2.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a32, b0_2.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a32, b0_2.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a32, b1_2.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a32, b1_2.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a32, b1_2.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a32, b1_2.w, acc37, false);

            acc00 = __builtin_amdgcn_sdot4(a03, b0_3.x, acc00, false); acc01 = __builtin_amdgcn_sdot4(a03, b0_3.y, acc01, false); acc02 = __builtin_amdgcn_sdot4(a03, b0_3.z, acc02, false); acc03 = __builtin_amdgcn_sdot4(a03, b0_3.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a03, b1_3.x, acc04, false); acc05 = __builtin_amdgcn_sdot4(a03, b1_3.y, acc05, false); acc06 = __builtin_amdgcn_sdot4(a03, b1_3.z, acc06, false); acc07 = __builtin_amdgcn_sdot4(a03, b1_3.w, acc07, false);
            acc10 = __builtin_amdgcn_sdot4(a13, b0_3.x, acc10, false); acc11 = __builtin_amdgcn_sdot4(a13, b0_3.y, acc11, false); acc12 = __builtin_amdgcn_sdot4(a13, b0_3.z, acc12, false); acc13 = __builtin_amdgcn_sdot4(a13, b0_3.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a13, b1_3.x, acc14, false); acc15 = __builtin_amdgcn_sdot4(a13, b1_3.y, acc15, false); acc16 = __builtin_amdgcn_sdot4(a13, b1_3.z, acc16, false); acc17 = __builtin_amdgcn_sdot4(a13, b1_3.w, acc17, false);
            acc20 = __builtin_amdgcn_sdot4(a23, b0_3.x, acc20, false); acc21 = __builtin_amdgcn_sdot4(a23, b0_3.y, acc21, false); acc22 = __builtin_amdgcn_sdot4(a23, b0_3.z, acc22, false); acc23 = __builtin_amdgcn_sdot4(a23, b0_3.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a23, b1_3.x, acc24, false); acc25 = __builtin_amdgcn_sdot4(a23, b1_3.y, acc25, false); acc26 = __builtin_amdgcn_sdot4(a23, b1_3.z, acc26, false); acc27 = __builtin_amdgcn_sdot4(a23, b1_3.w, acc27, false);
            acc30 = __builtin_amdgcn_sdot4(a33, b0_3.x, acc30, false); acc31 = __builtin_amdgcn_sdot4(a33, b0_3.y, acc31, false); acc32 = __builtin_amdgcn_sdot4(a33, b0_3.z, acc32, false); acc33 = __builtin_amdgcn_sdot4(a33, b0_3.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a33, b1_3.x, acc34, false); acc35 = __builtin_amdgcn_sdot4(a33, b1_3.y, acc35, false); acc36 = __builtin_amdgcn_sdot4(a33, b1_3.z, acc36, false); acc37 = __builtin_amdgcn_sdot4(a33, b1_3.w, acc37, false);

            a0_ptr += 16;
            a1_ptr += 16;
            a2_ptr += 16;
            a3_ptr += 16;
            b0_ptr += 4 * b_group_stride;
            b1_ptr += 4 * b_group_stride;
        }

        for (int kg = k_groups4; kg < k_groups; ++kg)
        {
            const int32_t a0_cur = __builtin_amdgcn_readfirstlane(*reinterpret_cast<const int32_t *>(a0_ptr));
            const int32_t a1_cur = __builtin_amdgcn_readfirstlane(*reinterpret_cast<const int32_t *>(a1_ptr));
            const int32_t a2_cur = __builtin_amdgcn_readfirstlane(*reinterpret_cast<const int32_t *>(a2_ptr));
            const int32_t a3_cur = __builtin_amdgcn_readfirstlane(*reinterpret_cast<const int32_t *>(a3_ptr));
            const int4 b0_cur = *reinterpret_cast<const int4 *>(b0_ptr);
            const int4 b1_cur = *reinterpret_cast<const int4 *>(b1_ptr);

            acc00 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0_cur, b0_cur.w, acc03, false);
            acc04 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.x, acc04, false);
            acc05 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.y, acc05, false);
            acc06 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.z, acc06, false);
            acc07 = __builtin_amdgcn_sdot4(a0_cur, b1_cur.w, acc07, false);

            acc10 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1_cur, b0_cur.w, acc13, false);
            acc14 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.x, acc14, false);
            acc15 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.y, acc15, false);
            acc16 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.z, acc16, false);
            acc17 = __builtin_amdgcn_sdot4(a1_cur, b1_cur.w, acc17, false);

            acc20 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2_cur, b0_cur.w, acc23, false);
            acc24 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.x, acc24, false);
            acc25 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.y, acc25, false);
            acc26 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.z, acc26, false);
            acc27 = __builtin_amdgcn_sdot4(a2_cur, b1_cur.w, acc27, false);

            acc30 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3_cur, b0_cur.w, acc33, false);
            acc34 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.x, acc34, false);
            acc35 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.y, acc35, false);
            acc36 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.z, acc36, false);
            acc37 = __builtin_amdgcn_sdot4(a3_cur, b1_cur.w, acc37, false);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            b0_ptr += b_group_stride;
            b1_ptr += b_group_stride;
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY, int KG_TILE>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_ldsdb_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    __shared__ int4 sB0[2][KG_TILE][BX];
    __shared__ int4 sB1[2][KG_TILE][BX];

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    const int n_tile_base = blockIdx.x * (BX * CPT);

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        const int64_t a0_row = static_cast<int64_t>(m0) * K;
        const int64_t a1_row = static_cast<int64_t>(m1) * K;
        const int64_t a2_row = static_cast<int64_t>(m2) * K;
        const int64_t a3_row = static_cast<int64_t>(m3) * K;

        int ping = 0;
        int pong = 1;
        const int total_threads = BX * BY;
        const int linear_tid = ty * BX + lane;

        {
            const int tile_kg = min(KG_TILE, k_groups);
            const int loads = tile_kg * BX;
            for (int idx = linear_tid; idx < loads; idx += total_threads)
            {
                const int kt = idx / BX;
                const int ln = idx % BX;
                const int64_t b_off = (static_cast<int64_t>(kt) * N + (n_tile_base + ln * CPT)) * 4;
                sB0[ping][kt][ln] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off);
                sB1[ping][kt][ln] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off + 16);
            }
            __syncthreads();
        }

        for (int kg_base = 0; kg_base < k_groups; kg_base += KG_TILE)
        {
            const int tile_kg = min(KG_TILE, k_groups - kg_base);

            for (int kt = 0; kt < tile_kg; ++kt)
            {
                const int64_t kg = static_cast<int64_t>(kg_base + kt) * 4;
                const int32_t a0 = *reinterpret_cast<const int32_t *>(d_A + a0_row + kg);
                const int32_t a1 = *reinterpret_cast<const int32_t *>(d_A + a1_row + kg);
                const int32_t a2 = *reinterpret_cast<const int32_t *>(d_A + a2_row + kg);
                const int32_t a3 = *reinterpret_cast<const int32_t *>(d_A + a3_row + kg);
                const int4 b0 = sB0[ping][kt][lane];
                const int4 b1 = sB1[ping][kt][lane];

                acc00 = __builtin_amdgcn_sdot4(a0, b0.x, acc00, false);
                acc01 = __builtin_amdgcn_sdot4(a0, b0.y, acc01, false);
                acc02 = __builtin_amdgcn_sdot4(a0, b0.z, acc02, false);
                acc03 = __builtin_amdgcn_sdot4(a0, b0.w, acc03, false);
                acc04 = __builtin_amdgcn_sdot4(a0, b1.x, acc04, false);
                acc05 = __builtin_amdgcn_sdot4(a0, b1.y, acc05, false);
                acc06 = __builtin_amdgcn_sdot4(a0, b1.z, acc06, false);
                acc07 = __builtin_amdgcn_sdot4(a0, b1.w, acc07, false);

                acc10 = __builtin_amdgcn_sdot4(a1, b0.x, acc10, false);
                acc11 = __builtin_amdgcn_sdot4(a1, b0.y, acc11, false);
                acc12 = __builtin_amdgcn_sdot4(a1, b0.z, acc12, false);
                acc13 = __builtin_amdgcn_sdot4(a1, b0.w, acc13, false);
                acc14 = __builtin_amdgcn_sdot4(a1, b1.x, acc14, false);
                acc15 = __builtin_amdgcn_sdot4(a1, b1.y, acc15, false);
                acc16 = __builtin_amdgcn_sdot4(a1, b1.z, acc16, false);
                acc17 = __builtin_amdgcn_sdot4(a1, b1.w, acc17, false);

                acc20 = __builtin_amdgcn_sdot4(a2, b0.x, acc20, false);
                acc21 = __builtin_amdgcn_sdot4(a2, b0.y, acc21, false);
                acc22 = __builtin_amdgcn_sdot4(a2, b0.z, acc22, false);
                acc23 = __builtin_amdgcn_sdot4(a2, b0.w, acc23, false);
                acc24 = __builtin_amdgcn_sdot4(a2, b1.x, acc24, false);
                acc25 = __builtin_amdgcn_sdot4(a2, b1.y, acc25, false);
                acc26 = __builtin_amdgcn_sdot4(a2, b1.z, acc26, false);
                acc27 = __builtin_amdgcn_sdot4(a2, b1.w, acc27, false);

                acc30 = __builtin_amdgcn_sdot4(a3, b0.x, acc30, false);
                acc31 = __builtin_amdgcn_sdot4(a3, b0.y, acc31, false);
                acc32 = __builtin_amdgcn_sdot4(a3, b0.z, acc32, false);
                acc33 = __builtin_amdgcn_sdot4(a3, b0.w, acc33, false);
                acc34 = __builtin_amdgcn_sdot4(a3, b1.x, acc34, false);
                acc35 = __builtin_amdgcn_sdot4(a3, b1.y, acc35, false);
                acc36 = __builtin_amdgcn_sdot4(a3, b1.z, acc36, false);
                acc37 = __builtin_amdgcn_sdot4(a3, b1.w, acc37, false);
            }

            const int next_kg_base = kg_base + KG_TILE;
            if (next_kg_base < k_groups)
            {
                const int next_tile_kg = min(KG_TILE, k_groups - next_kg_base);
                const int loads = next_tile_kg * BX;
                for (int idx = linear_tid; idx < loads; idx += total_threads)
                {
                    const int kt = idx / BX;
                    const int ln = idx % BX;
                    const int64_t b_off = (static_cast<int64_t>(next_kg_base + kt) * N + (n_tile_base + ln * CPT)) * 4;
                    sB0[pong][kt][ln] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off);
                    sB1[pong][kt][ln] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off + 16);
                }
                __syncthreads();
                const int tmp = ping;
                ping = pong;
                pong = tmp;
            }
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}

template <int BY, int KG_TILE>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt8_gridstride_aligned_mr4_lds_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 8;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    __shared__ int4 sB0[KG_TILE][BX];
    __shared__ int4 sB1[KG_TILE][BX];
    __shared__ int32_t sA[BY][MR][KG_TILE];

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0, acc04 = 0, acc05 = 0, acc06 = 0, acc07 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0, acc16 = 0, acc17 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0, acc24 = 0, acc25 = 0, acc26 = 0, acc27 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0, acc34 = 0, acc35 = 0, acc36 = 0, acc37 = 0;

        for (int kg_base = 0; kg_base < k_groups; kg_base += KG_TILE)
        {
            const int tile_kg = min(KG_TILE, k_groups - kg_base);

            if (ty == 0)
            {
                for (int kt = 0; kt < tile_kg; ++kt)
                {
                    const int64_t b_off = (static_cast<int64_t>(kg_base + kt) * N + n_base) * 4;
                    sB0[kt][lane] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off);
                    sB1[kt][lane] = *reinterpret_cast<const int4 *>(d_B_vnni + b_off + 16);
                }
            }

            if (lane < tile_kg)
            {
                const int64_t kg = static_cast<int64_t>(kg_base + lane) * 4;
                sA[ty][0][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + kg);
                sA[ty][1][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + kg);
                sA[ty][2][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K + kg);
                sA[ty][3][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K + kg);
            }

            __syncthreads();

            for (int kt = 0; kt < tile_kg; ++kt)
            {
                const int4 b0 = sB0[kt][lane];
                const int4 b1 = sB1[kt][lane];
                const int32_t a0 = sA[ty][0][kt];
                const int32_t a1 = sA[ty][1][kt];
                const int32_t a2 = sA[ty][2][kt];
                const int32_t a3 = sA[ty][3][kt];

                acc00 = __builtin_amdgcn_sdot4(a0, b0.x, acc00, false);
                acc01 = __builtin_amdgcn_sdot4(a0, b0.y, acc01, false);
                acc02 = __builtin_amdgcn_sdot4(a0, b0.z, acc02, false);
                acc03 = __builtin_amdgcn_sdot4(a0, b0.w, acc03, false);
                acc04 = __builtin_amdgcn_sdot4(a0, b1.x, acc04, false);
                acc05 = __builtin_amdgcn_sdot4(a0, b1.y, acc05, false);
                acc06 = __builtin_amdgcn_sdot4(a0, b1.z, acc06, false);
                acc07 = __builtin_amdgcn_sdot4(a0, b1.w, acc07, false);

                acc10 = __builtin_amdgcn_sdot4(a1, b0.x, acc10, false);
                acc11 = __builtin_amdgcn_sdot4(a1, b0.y, acc11, false);
                acc12 = __builtin_amdgcn_sdot4(a1, b0.z, acc12, false);
                acc13 = __builtin_amdgcn_sdot4(a1, b0.w, acc13, false);
                acc14 = __builtin_amdgcn_sdot4(a1, b1.x, acc14, false);
                acc15 = __builtin_amdgcn_sdot4(a1, b1.y, acc15, false);
                acc16 = __builtin_amdgcn_sdot4(a1, b1.z, acc16, false);
                acc17 = __builtin_amdgcn_sdot4(a1, b1.w, acc17, false);

                acc20 = __builtin_amdgcn_sdot4(a2, b0.x, acc20, false);
                acc21 = __builtin_amdgcn_sdot4(a2, b0.y, acc21, false);
                acc22 = __builtin_amdgcn_sdot4(a2, b0.z, acc22, false);
                acc23 = __builtin_amdgcn_sdot4(a2, b0.w, acc23, false);
                acc24 = __builtin_amdgcn_sdot4(a2, b1.x, acc24, false);
                acc25 = __builtin_amdgcn_sdot4(a2, b1.y, acc25, false);
                acc26 = __builtin_amdgcn_sdot4(a2, b1.z, acc26, false);
                acc27 = __builtin_amdgcn_sdot4(a2, b1.w, acc27, false);

                acc30 = __builtin_amdgcn_sdot4(a3, b0.x, acc30, false);
                acc31 = __builtin_amdgcn_sdot4(a3, b0.y, acc31, false);
                acc32 = __builtin_amdgcn_sdot4(a3, b0.z, acc32, false);
                acc33 = __builtin_amdgcn_sdot4(a3, b0.w, acc33, false);
                acc34 = __builtin_amdgcn_sdot4(a3, b1.x, acc34, false);
                acc35 = __builtin_amdgcn_sdot4(a3, b1.y, acc35, false);
                acc36 = __builtin_amdgcn_sdot4(a3, b1.z, acc36, false);
                acc37 = __builtin_amdgcn_sdot4(a3, b1.w, acc37, false);
            }

            __syncthreads();
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 0), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base + 4), acc04, acc05, acc06, acc07);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 0), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base + 4), acc14, acc15, acc16, acc17);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 0), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base + 4), acc24, acc25, acc26, acc27);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 0), acc30, acc31, acc32, acc33);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base + 4), acc34, acc35, acc36, acc37);
    }
}


template <int BY, int KG_TILE>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_mr4_lds_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    __shared__ int4 sB[KG_TILE][BX];
    __shared__ int32_t sA[BY][MR][KG_TILE];

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

        for (int kg_base = 0; kg_base < k_groups; kg_base += KG_TILE)
        {
            const int tile_kg = min(KG_TILE, k_groups - kg_base);

            if (ty == 0)
            {
                for (int kt = 0; kt < tile_kg; ++kt)
                {
                    sB[kt][lane] = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg_base + kt) * N + n_base) * 4);
                }
            }

            if (lane < tile_kg)
            {
                const int64_t kg = static_cast<int64_t>(kg_base + lane) * 4;
                sA[ty][0][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + kg);
                sA[ty][1][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + kg);
                sA[ty][2][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K + kg);
                sA[ty][3][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K + kg);
            }

            __syncthreads();

            for (int kt = 0; kt < tile_kg; ++kt)
            {
                const int4 b = sB[kt][lane];
                const int32_t a0 = sA[ty][0][kt];
                const int32_t a1 = sA[ty][1][kt];
                const int32_t a2 = sA[ty][2][kt];
                const int32_t a3 = sA[ty][3][kt];

                acc00 = __builtin_amdgcn_sdot4(a0, b.x, acc00, false);
                acc01 = __builtin_amdgcn_sdot4(a0, b.y, acc01, false);
                acc02 = __builtin_amdgcn_sdot4(a0, b.z, acc02, false);
                acc03 = __builtin_amdgcn_sdot4(a0, b.w, acc03, false);

                acc10 = __builtin_amdgcn_sdot4(a1, b.x, acc10, false);
                acc11 = __builtin_amdgcn_sdot4(a1, b.y, acc11, false);
                acc12 = __builtin_amdgcn_sdot4(a1, b.z, acc12, false);
                acc13 = __builtin_amdgcn_sdot4(a1, b.w, acc13, false);

                acc20 = __builtin_amdgcn_sdot4(a2, b.x, acc20, false);
                acc21 = __builtin_amdgcn_sdot4(a2, b.y, acc21, false);
                acc22 = __builtin_amdgcn_sdot4(a2, b.z, acc22, false);
                acc23 = __builtin_amdgcn_sdot4(a2, b.w, acc23, false);

                acc30 = __builtin_amdgcn_sdot4(a3, b.x, acc30, false);
                acc31 = __builtin_amdgcn_sdot4(a3, b.y, acc31, false);
                acc32 = __builtin_amdgcn_sdot4(a3, b.z, acc32, false);
                acc33 = __builtin_amdgcn_sdot4(a3, b.w, acc33, false);
            }

            __syncthreads();
        }

        const int64_t c0_base = static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c0_base), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c1_base), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c2_base), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c3_base), acc30, acc31, acc32, acc33);
    }
}

template <int BY, int ROWS_PER_GROUP>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt2_gridstride_aligned_ckclone128_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 2;
    constexpr int ROW_VECTOR_WIDTH = 4;
    static_assert(BY * ROWS_PER_GROUP == 128, "CK-clone tile expects 128 rows per block");
    static_assert((ROWS_PER_GROUP % ROW_VECTOR_WIDTH) == 0, "CK-clone vector row packet expects ROWS_PER_GROUP divisible by 4");

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m_base = row_group * ROWS_PER_GROUP;

    __shared__ int32_t sA[BY][ROWS_PER_GROUP];

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int2 acc[ROWS_PER_GROUP];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r)
        {
            acc[r] = make_int2(0, 0);
        }

        for (int kg = 0; kg < k_groups; ++kg)
        {
            if (lane < ROWS_PER_GROUP)
            {
                sA[ty][lane] = *reinterpret_cast<const int32_t *>(
                    d_A + static_cast<int64_t>(m_base + lane) * K + static_cast<int64_t>(kg) * 4);
            }

            __syncthreads();

            const int2 b = *reinterpret_cast<const int2 *>(
                d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

            const int4 *sA_row_packets = reinterpret_cast<const int4 *>(&sA[ty][0]);

            #pragma unroll
            for (int rv = 0; rv < (ROWS_PER_GROUP / ROW_VECTOR_WIDTH); ++rv)
            {
                const int4 a4 = sA_row_packets[rv];
                int2 &acc_r0 = acc[rv * ROW_VECTOR_WIDTH + 0];
                int2 &acc_r1 = acc[rv * ROW_VECTOR_WIDTH + 1];
                int2 &acc_r2 = acc[rv * ROW_VECTOR_WIDTH + 2];
                int2 &acc_r3 = acc[rv * ROW_VECTOR_WIDTH + 3];

                acc_r0.x = __builtin_amdgcn_sdot4(a4.x, b.x, acc_r0.x, false);
                acc_r0.y = __builtin_amdgcn_sdot4(a4.x, b.y, acc_r0.y, false);

                acc_r1.x = __builtin_amdgcn_sdot4(a4.y, b.x, acc_r1.x, false);
                acc_r1.y = __builtin_amdgcn_sdot4(a4.y, b.y, acc_r1.y, false);

                acc_r2.x = __builtin_amdgcn_sdot4(a4.z, b.x, acc_r2.x, false);
                acc_r2.y = __builtin_amdgcn_sdot4(a4.z, b.y, acc_r2.y, false);

                acc_r3.x = __builtin_amdgcn_sdot4(a4.w, b.x, acc_r3.x, false);
                acc_r3.y = __builtin_amdgcn_sdot4(a4.w, b.y, acc_r3.y, false);
            }

            __syncthreads();
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r)
        {
            const int64_t c_base = static_cast<int64_t>(m_base + r) * N + n_base;
            *reinterpret_cast<int2 *>(d_C + c_base) = acc[r];
        }
    }
}

template <int BY, int ROWS_PER_GROUP, int KG_TILE>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt2_gridstride_aligned_ckmirror128_lds_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 2;
    constexpr int ROW_VECTOR_WIDTH = 4;
    static_assert(BY * ROWS_PER_GROUP == 128, "CK-mirror tile expects 128 rows per block");
    static_assert((ROWS_PER_GROUP % ROW_VECTOR_WIDTH) == 0, "ROW_VECTOR_WIDTH must divide ROWS_PER_GROUP");

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_group = blockIdx.y * BY + ty;
    const int m_base = row_group * ROWS_PER_GROUP;

    __shared__ int32_t sA[BY][KG_TILE][ROWS_PER_GROUP];
    __shared__ int2 sB[KG_TILE][BX];

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;

    for (; n_base < N; n_base += n_stride)
    {
        int2 acc[ROWS_PER_GROUP];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r)
        {
            acc[r] = make_int2(0, 0);
        }

        for (int kg_base = 0; kg_base < k_groups; kg_base += KG_TILE)
        {
            const int tile_kg = min(KG_TILE, k_groups - kg_base);

            if (ty == 0)
            {
                for (int kt = 0; kt < tile_kg; ++kt)
                {
                    sB[kt][lane] = *reinterpret_cast<const int2 *>(
                        d_B_vnni + (static_cast<int64_t>(kg_base + kt) * N + n_base) * 4);
                }
            }

            if (lane < ROWS_PER_GROUP)
            {
                const int row = m_base + lane;
                for (int kt = 0; kt < tile_kg; ++kt)
                {
                    sA[ty][kt][lane] = *reinterpret_cast<const int32_t *>(
                        d_A + static_cast<int64_t>(row) * K + static_cast<int64_t>(kg_base + kt) * 4);
                }
            }

            __syncthreads();

            for (int kt = 0; kt < tile_kg; ++kt)
            {
                const int2 b = sB[kt][lane];
                const int4 *sA_row_packets = reinterpret_cast<const int4 *>(&sA[ty][kt][0]);

                #pragma unroll
                for (int rv = 0; rv < (ROWS_PER_GROUP / ROW_VECTOR_WIDTH); ++rv)
                {
                    const int4 a4 = sA_row_packets[rv];
                    int2 &acc_r0 = acc[rv * ROW_VECTOR_WIDTH + 0];
                    int2 &acc_r1 = acc[rv * ROW_VECTOR_WIDTH + 1];
                    int2 &acc_r2 = acc[rv * ROW_VECTOR_WIDTH + 2];
                    int2 &acc_r3 = acc[rv * ROW_VECTOR_WIDTH + 3];

                    acc_r0.x = __builtin_amdgcn_sdot4(a4.x, b.x, acc_r0.x, false);
                    acc_r0.y = __builtin_amdgcn_sdot4(a4.x, b.y, acc_r0.y, false);

                    acc_r1.x = __builtin_amdgcn_sdot4(a4.y, b.x, acc_r1.x, false);
                    acc_r1.y = __builtin_amdgcn_sdot4(a4.y, b.y, acc_r1.y, false);

                    acc_r2.x = __builtin_amdgcn_sdot4(a4.z, b.x, acc_r2.x, false);
                    acc_r2.y = __builtin_amdgcn_sdot4(a4.z, b.y, acc_r2.y, false);

                    acc_r3.x = __builtin_amdgcn_sdot4(a4.w, b.x, acc_r3.x, false);
                    acc_r3.y = __builtin_amdgcn_sdot4(a4.w, b.y, acc_r3.y, false);
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r)
        {
            const int64_t c_base = static_cast<int64_t>(m_base + r) * N + n_base;
            *reinterpret_cast<int2 *>(d_C + c_base) = acc[r];
        }
    }
}

template <int BY>
__global__ void gemm_wave64_cpt4_gridstride_aligned_dualacc_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; n_base < N; n_base += n_stride)
    {
        int32_t even0 = 0;
        int32_t even1 = 0;
        int32_t even2 = 0;
        int32_t even3 = 0;
        int32_t odd0 = 0;
        int32_t odd1 = 0;
        int32_t odd2 = 0;
        int32_t odd3 = 0;

        int kg = 0;
        for (; kg + 1 < k_groups; kg += 2)
        {
            const int32_t a0 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 0) * 4);
            const int32_t a1 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 1) * 4);
            const int4 b0 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 0) * N + n_base) * 4);
            const int4 b1 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 1) * N + n_base) * 4);

            even0 = __builtin_amdgcn_sdot4(a0, b0.x, even0, false);
            even1 = __builtin_amdgcn_sdot4(a0, b0.y, even1, false);
            even2 = __builtin_amdgcn_sdot4(a0, b0.z, even2, false);
            even3 = __builtin_amdgcn_sdot4(a0, b0.w, even3, false);

            odd0 = __builtin_amdgcn_sdot4(a1, b1.x, odd0, false);
            odd1 = __builtin_amdgcn_sdot4(a1, b1.y, odd1, false);
            odd2 = __builtin_amdgcn_sdot4(a1, b1.z, odd2, false);
            odd3 = __builtin_amdgcn_sdot4(a1, b1.w, odd3, false);
        }

        if (kg < k_groups)
        {
            const int32_t a = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const int4 b = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            even0 = __builtin_amdgcn_sdot4(a, b.x, even0, false);
            even1 = __builtin_amdgcn_sdot4(a, b.y, even1, false);
            even2 = __builtin_amdgcn_sdot4(a, b.z, even2, false);
            even3 = __builtin_amdgcn_sdot4(a, b.w, even3, false);
        }

        const int32_t acc0 = even0 + odd0;
        const int32_t acc1 = even1 + odd1;
        const int32_t acc2 = even2 + odd2;
        const int32_t acc3 = even3 + odd3;

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_gridstride_aligned_mr4_splitk_stage1_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_partial,
    int M,
    int N,
    int K,
    int splitk)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;
    constexpr int MR = 4;

    const int lane = threadIdx.x;
    const int row_group = blockIdx.y * BY + threadIdx.y;
    const int kb = static_cast<int>(blockIdx.z);

    const int m0 = row_group * MR + 0;
    const int m1 = row_group * MR + 1;
    const int m2 = row_group * MR + 2;
    const int m3 = row_group * MR + 3;

    const int k_groups_total = K / 4;
    const int groups_per = (k_groups_total + splitk - 1) / splitk;
    const int g0 = kb * groups_per;
    const int g1 = min(k_groups_total, g0 + groups_per);

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
        int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
        int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
        int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

        for (int kg = g0; kg < g1; ++kg)
        {
            const int32_t a0 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m0) * K + static_cast<int64_t>(kg) * 4);
            const int32_t a1 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m1) * K + static_cast<int64_t>(kg) * 4);
            const int32_t a2 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m2) * K + static_cast<int64_t>(kg) * 4);
            const int32_t a3 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m3) * K + static_cast<int64_t>(kg) * 4);
            const int4 b = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);

            acc00 = __builtin_amdgcn_sdot4(a0, b.x, acc00, false);
            acc01 = __builtin_amdgcn_sdot4(a0, b.y, acc01, false);
            acc02 = __builtin_amdgcn_sdot4(a0, b.z, acc02, false);
            acc03 = __builtin_amdgcn_sdot4(a0, b.w, acc03, false);

            acc10 = __builtin_amdgcn_sdot4(a1, b.x, acc10, false);
            acc11 = __builtin_amdgcn_sdot4(a1, b.y, acc11, false);
            acc12 = __builtin_amdgcn_sdot4(a1, b.z, acc12, false);
            acc13 = __builtin_amdgcn_sdot4(a1, b.w, acc13, false);

            acc20 = __builtin_amdgcn_sdot4(a2, b.x, acc20, false);
            acc21 = __builtin_amdgcn_sdot4(a2, b.y, acc21, false);
            acc22 = __builtin_amdgcn_sdot4(a2, b.z, acc22, false);
            acc23 = __builtin_amdgcn_sdot4(a2, b.w, acc23, false);

            acc30 = __builtin_amdgcn_sdot4(a3, b.x, acc30, false);
            acc31 = __builtin_amdgcn_sdot4(a3, b.y, acc31, false);
            acc32 = __builtin_amdgcn_sdot4(a3, b.z, acc32, false);
            acc33 = __builtin_amdgcn_sdot4(a3, b.w, acc33, false);
        }

        const int64_t kb_base = static_cast<int64_t>(kb) * static_cast<int64_t>(M) * static_cast<int64_t>(N);
        const int64_t c0_base = kb_base + static_cast<int64_t>(m0) * N + n_base;
        const int64_t c1_base = kb_base + static_cast<int64_t>(m1) * N + n_base;
        const int64_t c2_base = kb_base + static_cast<int64_t>(m2) * N + n_base;
        const int64_t c3_base = kb_base + static_cast<int64_t>(m3) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_partial + c0_base), acc00, acc01, acc02, acc03);
        store_i32x4(reinterpret_cast<uint32_t *>(d_partial + c1_base), acc10, acc11, acc12, acc13);
        store_i32x4(reinterpret_cast<uint32_t *>(d_partial + c2_base), acc20, acc21, acc22, acc23);
        store_i32x4(reinterpret_cast<uint32_t *>(d_partial + c3_base), acc30, acc31, acc32, acc33);
    }
}

template <int BY>
__launch_bounds__(64 * BY, 2) __global__ void gemm_wave64_cpt4_splitk_reduce_kernel(
    const int32_t *__restrict__ d_partial,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int splitk)
{
    (void)M;
    constexpr int BX = 64;
    constexpr int CPT = 4;

    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;

    for (; n_base < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;

        for (int kb = 0; kb < splitk; ++kb)
        {
            const int64_t kb_base = static_cast<int64_t>(kb) * static_cast<int64_t>(M) * static_cast<int64_t>(N);
            const int64_t p_base = kb_base + static_cast<int64_t>(m) * N + n_base;
            const int4 p = *reinterpret_cast<const int4 *>(d_partial + p_base);
            acc0 += p.x;
            acc1 += p.y;
            acc2 += p.z;
            acc3 += p.w;
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__global__ void gemm_wave64_cpt4_gridstride_unroll4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M)
    {
        return;
    }

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; (n_base + 3) < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;

        int kg = 0;
        for (; kg + 3 < k_groups; kg += 4)
        {
            const int32_t a0 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 0) * 4);
            const int32_t a1 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 1) * 4);
            const int32_t a2 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 2) * 4);
            const int32_t a3 = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg + 3) * 4);

            const int4 b0 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 0) * N + n_base) * 4);
            const int4 b1 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 1) * N + n_base) * 4);
            const int4 b2 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 2) * N + n_base) * 4);
            const int4 b3 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg + 3) * N + n_base) * 4);

            acc0 = __builtin_amdgcn_sdot4(a0, b0.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a0, b0.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a0, b0.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a0, b0.w, acc3, false);

            acc0 = __builtin_amdgcn_sdot4(a1, b1.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a1, b1.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a1, b1.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a1, b1.w, acc3, false);

            acc0 = __builtin_amdgcn_sdot4(a2, b2.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a2, b2.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a2, b2.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a2, b2.w, acc3, false);

            acc0 = __builtin_amdgcn_sdot4(a3, b3.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a3, b3.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a3, b3.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a3, b3.w, acc3, false);
        }

        for (; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY, int KG_TILE>
__global__ void gemm_wave64_cpt4_shared_b_ktile_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    __shared__ int4 shared_b[KG_TILE][BX];

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;

    const int k_groups_total = K / 4;
    for (int kg0 = 0; kg0 < k_groups_total; kg0 += KG_TILE)
    {
        const int tile_groups = min(KG_TILE, k_groups_total - kg0);

        if (threadIdx.y == 0)
        {
            #pragma unroll
            for (int t = 0; t < KG_TILE; ++t)
            {
                if (t < tile_groups)
                {
                    const int kg = kg0 + t;
                    shared_b[t][lane] = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < KG_TILE; ++t)
        {
            if (t < tile_groups)
            {
                const int kg = kg0 + t;
                const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
                const int4 b4 = shared_b[t][lane];
                acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
                acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
                acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
                acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
            }
        }
        __syncthreads();
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
}

template <int BY>
__global__ void gemm_wave64_cpt4_splitk_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K,
    int splitk)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    const int kb = blockIdx.z;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    const int k_groups_total = K / 4;
    const int groups_per = (k_groups_total + splitk - 1) / splitk;
    const int g0 = kb * groups_per;
    const int g1 = min(k_groups_total, g0 + groups_per);
    if (g0 >= g1)
    {
        return;
    }

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    for (int kg = g0; kg < g1; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        const int4 b4 = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
        acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    atomicAdd(&d_C[c_base + 0], acc0);
    atomicAdd(&d_C[c_base + 1], acc1);
    atomicAdd(&d_C[c_base + 2], acc2);
    atomicAdd(&d_C[c_base + 3], acc3);
}

template <int BY, int KG_TILE>
__global__ void gemm_wave64_cpt4_shared_ab_ktile_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const int8_t *__restrict__ d_B_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    __shared__ int4 shared_b[KG_TILE][BX];
    __shared__ int32_t shared_a[BY][KG_TILE];

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;

    const int k_groups_total = K / 4;
    for (int kg0 = 0; kg0 < k_groups_total; kg0 += KG_TILE)
    {
        const int tile_groups = min(KG_TILE, k_groups_total - kg0);

        if (threadIdx.y == 0)
        {
            #pragma unroll
            for (int t = 0; t < KG_TILE; ++t)
            {
                if (t < tile_groups)
                {
                    const int kg = kg0 + t;
                    shared_b[t][lane] = *reinterpret_cast<const int4 *>(d_B_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                }
            }
        }

        if (lane < KG_TILE && lane < tile_groups)
        {
            const int kg = kg0 + lane;
            shared_a[threadIdx.y][lane] = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < KG_TILE; ++t)
        {
            if (t < tile_groups)
            {
                const int32_t a_packed = shared_a[threadIdx.y][t];
                const int4 b4 = shared_b[t][lane];
                acc0 = __builtin_amdgcn_sdot4(a_packed, b4.x, acc0, false);
                acc1 = __builtin_amdgcn_sdot4(a_packed, b4.y, acc1, false);
                acc2 = __builtin_amdgcn_sdot4(a_packed, b4.z, acc2, false);
                acc3 = __builtin_amdgcn_sdot4(a_packed, b4.w, acc3, false);
            }
        }
        __syncthreads();
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
}

__device__ __forceinline__ int32_t decode_q4_nibbles_to_i8pack(uint16_t packed_nibbles)
{
    const int32_t n0 = (static_cast<int32_t>(packed_nibbles & 0x000Fu) << 28) >> 28;
    const int32_t n1 = (static_cast<int32_t>((packed_nibbles >> 4) & 0x000Fu) << 28) >> 28;
    const int32_t n2 = (static_cast<int32_t>((packed_nibbles >> 8) & 0x000Fu) << 28) >> 28;
    const int32_t n3 = (static_cast<int32_t>((packed_nibbles >> 12) & 0x000Fu) << 28) >> 28;

    uint32_t out = 0;
    out |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(n0)));
    out |= (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(n1))) << 8);
    out |= (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(n2))) << 16);
    out |= (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(n3))) << 24);
    return static_cast<int32_t>(out);
}

__device__ __forceinline__ int32_t decode_q4_nibbles_to_i8pack_lut(uint16_t packed_nibbles)
{
    const uint16_t lo = k_q4_pair_decode_lut[packed_nibbles & 0x00FFu];
    const uint16_t hi = k_q4_pair_decode_lut[(packed_nibbles >> 8) & 0x00FFu];
    const uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    return static_cast<int32_t>(packed);
}

__device__ __forceinline__ uint16_t decode_q4_byte_to_two_i8_perm(uint8_t packed_byte)
{
    constexpr uint32_t lut0 = 0x03020100u;
    constexpr uint32_t lut1 = 0x07060504u;
    constexpr uint32_t lut2 = 0xFBFAF9F8u;
    constexpr uint32_t lut3 = 0xFFFEFDFCu;

    const uint32_t sel =
        (static_cast<uint32_t>(packed_byte & 0x07u)) |
        ((static_cast<uint32_t>((packed_byte >> 4) & 0x07u)) << 8);

    const uint32_t low_vals = __builtin_amdgcn_perm(lut1, lut0, sel);
    const uint32_t high_vals = __builtin_amdgcn_perm(lut3, lut2, sel);

    uint32_t sign =
        (static_cast<uint32_t>((packed_byte >> 3) & 0x01u)) |
        ((static_cast<uint32_t>((packed_byte >> 7) & 0x01u)) << 8);
    sign |= (sign << 1);
    sign |= (sign << 2);
    sign |= (sign << 4);

    const uint32_t out = (high_vals & sign) | (low_vals & ~sign);
    return static_cast<uint16_t>(out & 0xFFFFu);
}

__device__ __forceinline__ int32_t decode_q4_nibbles_to_i8pack_perm(uint16_t packed_nibbles)
{
    const uint8_t b0 = static_cast<uint8_t>(packed_nibbles & 0x00FFu);
    const uint8_t b1 = static_cast<uint8_t>((packed_nibbles >> 8) & 0x00FFu);
    const uint16_t lo = decode_q4_byte_to_two_i8_perm(b0);
    const uint16_t hi = decode_q4_byte_to_two_i8_perm(b1);
    const uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    return static_cast<int32_t>(packed);
}

template <int BY>
__global__ void gemm_wave64_q4nib_cpt4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

        const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
        const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
        const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
        const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

        const int32_t b0 = decode_q4_nibbles_to_i8pack(p0);
        const int32_t b1 = decode_q4_nibbles_to_i8pack(p1);
        const int32_t b2 = decode_q4_nibbles_to_i8pack(p2);
        const int32_t b3 = decode_q4_nibbles_to_i8pack(p3);

        acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    d_C[c_base + 0] = acc0;
    d_C[c_base + 1] = acc1;
    d_C[c_base + 2] = acc2;
    d_C[c_base + 3] = acc3;
}

template <int BY>
__global__ void gemm_wave64_q4nib_cpt4_gridstride_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M)
    {
        return;
    }

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; (n_base + 3) < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

            const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
            const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
            const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
            const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

            const int32_t b0 = decode_q4_nibbles_to_i8pack(p0);
            const int32_t b1 = decode_q4_nibbles_to_i8pack(p1);
            const int32_t b2 = decode_q4_nibbles_to_i8pack(p2);
            const int32_t b3 = decode_q4_nibbles_to_i8pack(p3);

            acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__global__ void gemm_wave64_q4nib_lut_cpt4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

        const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
        const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
        const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
        const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

        const int32_t b0 = decode_q4_nibbles_to_i8pack_lut(p0);
        const int32_t b1 = decode_q4_nibbles_to_i8pack_lut(p1);
        const int32_t b2 = decode_q4_nibbles_to_i8pack_lut(p2);
        const int32_t b3 = decode_q4_nibbles_to_i8pack_lut(p3);

        acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
}

template <int BY>
__global__ void gemm_wave64_q4nib_lut_cpt4_gridstride_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M)
    {
        return;
    }

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; (n_base + 3) < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

            const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
            const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
            const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
            const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

            const int32_t b0 = decode_q4_nibbles_to_i8pack_lut(p0);
            const int32_t b1 = decode_q4_nibbles_to_i8pack_lut(p1);
            const int32_t b2 = decode_q4_nibbles_to_i8pack_lut(p2);
            const int32_t b3 = decode_q4_nibbles_to_i8pack_lut(p3);

            acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        store_i32x4(reinterpret_cast<uint32_t *>(d_C + c_base), acc0, acc1, acc2, acc3);
    }
}

template <int BY>
__global__ void gemm_wave64_q4nib_perm_cpt4_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || (n_base + 3) >= N)
    {
        return;
    }

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    int32_t acc2 = 0;
    int32_t acc3 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
        const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

        const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
        const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
        const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
        const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

        const int32_t b0 = decode_q4_nibbles_to_i8pack_perm(p0);
        const int32_t b1 = decode_q4_nibbles_to_i8pack_perm(p1);
        const int32_t b2 = decode_q4_nibbles_to_i8pack_perm(p2);
        const int32_t b3 = decode_q4_nibbles_to_i8pack_perm(p3);

        acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
        acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
        acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
    }

    const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
    d_C[c_base + 0] = acc0;
    d_C[c_base + 1] = acc1;
    d_C[c_base + 2] = acc2;
    d_C[c_base + 3] = acc3;
}

template <int BY>
__global__ void gemm_wave64_q4nib_perm_cpt4_gridstride_prefill_kernel(
    const int8_t *__restrict__ d_A,
    const uint8_t *__restrict__ d_B_q4_vnni,
    int32_t *__restrict__ d_C,
    int M,
    int N,
    int K)
{
    constexpr int BX = 64;
    constexpr int CPT = 4;
    const int lane = threadIdx.x;
    const int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M)
    {
        return;
    }

    int n_base = blockIdx.x * (BX * CPT) + lane * CPT;
    const int n_stride = static_cast<int>(gridDim.x) * BX * CPT;
    const int k_groups = K / 4;
    for (; (n_base + 3) < N; n_base += n_stride)
    {
        int32_t acc0 = 0;
        int32_t acc1 = 0;
        int32_t acc2 = 0;
        int32_t acc3 = 0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t *>(d_A + static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4);
            const uint2 nibs = *reinterpret_cast<const uint2 *>(d_B_q4_vnni + (static_cast<int64_t>(kg) * N + n_base) * 2);

            const uint16_t p0 = static_cast<uint16_t>(nibs.x & 0xFFFFu);
            const uint16_t p1 = static_cast<uint16_t>((nibs.x >> 16) & 0xFFFFu);
            const uint16_t p2 = static_cast<uint16_t>(nibs.y & 0xFFFFu);
            const uint16_t p3 = static_cast<uint16_t>((nibs.y >> 16) & 0xFFFFu);

            const int32_t b0 = decode_q4_nibbles_to_i8pack_perm(p0);
            const int32_t b1 = decode_q4_nibbles_to_i8pack_perm(p1);
            const int32_t b2 = decode_q4_nibbles_to_i8pack_perm(p2);
            const int32_t b3 = decode_q4_nibbles_to_i8pack_perm(p3);

            acc0 = __builtin_amdgcn_sdot4(a_packed, b0, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(a_packed, b1, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(a_packed, b2, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(a_packed, b3, acc3, false);
        }

        const int64_t c_base = static_cast<int64_t>(m) * N + n_base;
        d_C[c_base + 0] = acc0;
        d_C[c_base + 1] = acc1;
        d_C[c_base + 2] = acc2;
        d_C[c_base + 3] = acc3;
    }
}

bool parseIntArg(const std::string &arg, const std::string &key, int &out)
{
    const std::string prefix = key + "=";
    if (arg.rfind(prefix, 0) != 0)
    {
        return false;
    }
    out = std::stoi(arg.substr(prefix.size()));
    return true;
}

std::vector<std::string> splitCSV(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<int> splitCSVInt(const std::string &s)
{
    std::vector<int> out;
    for (const auto &item : splitCSV(s))
    {
        out.push_back(std::stoi(item));
    }
    return out;
}

std::vector<Shape> defaultShapes()
{
    return {
        {"Qwen2.5-0.5B_AttnOut", 128, 896, 896},
        {"Qwen2.5-0.5B_FFN_Up", 128, 4864, 896},
        {"Qwen2.5-0.5B_FFN_Gate", 128, 4864, 896},
        {"Qwen2.5-0.5B_FFN_Down", 128, 896, 4864},
        {"Qwen2.5-0.5B_LM_Head", 128, 151936, 896},
        {"Qwen2.5-3B_AttnOut", 128, 2048, 2048},
        {"Qwen2.5-3B_FFN_Up", 128, 11008, 2048},
        {"Qwen2.5-3B_FFN_Gate", 128, 11008, 2048},
        {"Qwen2.5-3B_FFN_Down", 128, 2048, 11008},
        {"Qwen2.5-3B_LM_Head", 128, 151936, 2048}};
}

bool isHelpArg(const std::string &arg)
{
    return arg == "--help" || arg == "-h";
}

void printHelp(const char *argv0)
{
    std::cout
        << "ROCm Prefill Strategy Lab\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Device selection:\n"
        << "  --device=<id>               Run on one device (default: 0)\n"
        << "  --all-devices               Run on all visible ROCm devices\n"
        << "  --devices=<csv>             Run on explicit device list (e.g. 0,1,2)\n\n"
        << "Timing controls:\n"
        << "  --warmup=<n>                Warmup iterations per measured loop (default: 2)\n"
        << "  --iters=<n>                 Timed iterations per measured loop (default: 8)\n\n"
        << "Shape controls:\n"
        << "  --m=<rows> --n=<cols> --k=<depth>\n"
        << "                              Use one custom shape instead of built-in defaults\n\n"
        << "Strategy controls:\n"
        << "  --strategies=<csv>          Override strategy shortlist\n"
        << "  --splitk=<csv>              Split-K values for split-K strategies (default: 2,4,8)\n"
        << "  --full-lm-head              Use exhaustive LM-head strategy list\n"
        << "  --no-prefer-ck-wide         Benchmark custom strategies on wide shapes\n"
        << "                              (default policy prefers CK baseline for wide N/K)\n"
        << "  --no-lm-head-chunked        Disable LM-head chunked prototype row\n"
        << "  --lm-head-chunk-n=<n>       Chunk width for LM-head chunked prototype (default: 16384)\n\n"
        << "Leaderboard persistence:\n"
        << "  --leaderboard-dir=<path>    Directory for on-disk leaderboard outputs\n"
        << "                              (default: artifacts/rocm_prefill_strategy_lab)\n"
        << "  --no-disk-leaderboard       Disable on-disk leaderboard writes\n\n"
        << "Correctness / speed:\n"
        << "  --no-check                  Disable correctness checks for faster sweeps\n"
        << "                              (skips CPU reference GEMM and CK layout-validation path)\n"
        << "  --int8-vnni-repack          Deprecated (ignored): CK baseline repack is always accounted\n"
        << "                              from INT8-VNNI->CK format; non-CK repack remains excluded\n\n"
        << "Help:\n"
        << "  -h, --help                  Show this help and exit\n";
}

Options parseOptions(int argc, char **argv)
{
    Options opt;
    bool saw_deprecated_int8_vnni_repack = false;
    opt.shapes = defaultShapes();
    opt.strategies = {
        "cpt2",
        "cpt4",
        "cpt2_splitk",
        "cpt2_sharedb",
        "cpt2_sharedb_splitk",
        "cpt2_shared_ab_kg8",
        "cpt2_shared_ab_kg16",
        "wave64_cpt4",
        "wave64_cpt4_by8",
        "wave64_cpt4_grid",
        "wave64_cpt4_grid_by8",
        "wave64_cpt4_grid_by8_cap24",
        "wave64_cpt4_grid_by8_cap48",
        "wave64_cpt4_grid_by8_cap64",
        "wave64_cpt4_grid_by8_cap128",
        "wave64_cpt4_grid_by8_cap192",
        "wave64_cpt4_grid_cap64",
        "wave64_cpt4_aligned_cap128",
        "wave64_cpt4_aligned_by4_cap128",
        "wave64_cpt4_aligned_unroll2_cap128",
        "wave64_cpt4_aligned_unroll2_lb_cap128",
        "wave64_cpt4_aligned_dualacc_cap128",
        "wave64_cpt8_aligned_mr4_cap128",
        "wave64_cpt4_auto_tuned",
        "wave64_cpt4_unroll4_grid_by8",
        "wave64_cpt4_splitk",
        "wave64_cpt4_splitk_by8",
        "wave64_cpt4_shared_ab_kg8",
        "wave64_cpt4_shared_ab_kg16",
        "wave64_cpt4_shared_ab_kg8_by8",
        "wave64_cpt4_shared_ab_kg16_by8",
        "wave64_cpt4_sharedb_kg8_by8",
        "wave64_cpt4_sharedb_kg16_by8",
        "q4nib_wave64_cpt4",
        "q4nib_wave64_cpt4_grid",
        "q4nib_lut_wave64_cpt4",
        "q4nib_lut_wave64_cpt4_grid",
        "q4nib_perm_wave64_cpt4",
        "q4nib_perm_wave64_cpt4_grid"};
    opt.splitk_values = {2, 4, 8};

    int m = -1;
    int n = -1;
    int k = -1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (parseIntArg(arg, "--device", opt.device) || parseIntArg(arg, "--warmup", opt.warmup) || parseIntArg(arg, "--iters", opt.iters) || parseIntArg(arg, "--m", m) || parseIntArg(arg, "--n", n) || parseIntArg(arg, "--k", k) || parseIntArg(arg, "--lm-head-chunk-n", opt.lm_head_chunk_n))
        {
            continue;
        }
        if (arg == "--all-devices")
        {
            opt.all_devices = true;
            continue;
        }
        if (arg == "--full-lm-head")
        {
            opt.full_lm_head = true;
            continue;
        }
        if (arg == "--no-prefer-ck-wide")
        {
            opt.prefer_ck_wide = false;
            continue;
        }
        if (arg == "--no-lm-head-chunked")
        {
            opt.lm_head_chunked = false;
            continue;
        }
        if (arg.rfind("--devices=", 0) == 0)
        {
            opt.devices = splitCSVInt(arg.substr(std::strlen("--devices=")));
            continue;
        }
        if (arg.rfind("--strategies=", 0) == 0)
        {
            opt.strategies = splitCSV(arg.substr(std::strlen("--strategies=")));
            opt.user_strategy_override = true;
            continue;
        }
        if (arg.rfind("--splitk=", 0) == 0)
        {
            opt.splitk_values = splitCSVInt(arg.substr(std::strlen("--splitk=")));
            continue;
        }
        if (arg == "--no-check")
        {
            opt.check_correctness = false;
            continue;
        }
        if (arg == "--int8-vnni-repack")
        {
            saw_deprecated_int8_vnni_repack = true;
            continue;
        }
        if (arg == "--no-disk-leaderboard")
        {
            opt.persist_leaderboard = false;
            continue;
        }
        if (arg.rfind("--leaderboard-dir=", 0) == 0)
        {
            opt.leaderboard_dir = arg.substr(std::strlen("--leaderboard-dir="));
            continue;
        }
    }

    if (m > 0 && n > 0 && k > 0)
    {
        opt.shapes = {{"custom", m, n, k}};
    }

    if (saw_deprecated_int8_vnni_repack)
    {
        std::cout << "[prefill-lab] note: --int8-vnni-repack is deprecated and ignored; "
                  << "CK baseline repack is always included; non-CK weight repack remains excluded" << std::endl;
    }

    return opt;
}

void fillRandomInt8(std::vector<int8_t> &buffer, uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &v : buffer)
    {
        v = static_cast<int8_t>(dist(rng));
    }
}

std::vector<int8_t> packRowMajorToVNNI(const std::vector<int8_t> &b_rowmajor, int N, int K)
{
    const int k_groups = K / 4;
    std::vector<int8_t> b_vnni(static_cast<size_t>(k_groups) * static_cast<size_t>(N) * 4u);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kg = 0; kg < k_groups; ++kg)
    {
        for (int n = 0; n < N; ++n)
        {
            const size_t base = (static_cast<size_t>(kg) * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4u;
            b_vnni[base + 0] = b_rowmajor[static_cast<size_t>(kg * 4 + 0) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 1] = b_rowmajor[static_cast<size_t>(kg * 4 + 1) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 2] = b_rowmajor[static_cast<size_t>(kg * 4 + 2) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 3] = b_rowmajor[static_cast<size_t>(kg * 4 + 3) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        }
    }
    return b_vnni;
}

void packRowMajorToVNNIInplace(const std::vector<int8_t> &b_rowmajor, int N, int K, std::vector<int8_t> &b_vnni)
{
    const int k_groups = K / 4;
    const size_t expected = static_cast<size_t>(k_groups) * static_cast<size_t>(N) * 4u;
    if (b_vnni.size() != expected)
    {
        b_vnni.resize(expected);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kg = 0; kg < k_groups; ++kg)
    {
        for (int n = 0; n < N; ++n)
        {
            const size_t base = (static_cast<size_t>(kg) * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4u;
            b_vnni[base + 0] = b_rowmajor[static_cast<size_t>(kg * 4 + 0) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 1] = b_rowmajor[static_cast<size_t>(kg * 4 + 1) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 2] = b_rowmajor[static_cast<size_t>(kg * 4 + 2) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            b_vnni[base + 3] = b_rowmajor[static_cast<size_t>(kg * 4 + 3) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        }
    }
}

void unpackVNNIToRowMajorInplace(const std::vector<int8_t> &b_vnni, int N, int K, std::vector<int8_t> &b_rowmajor)
{
    const int k_groups = K / 4;
    const size_t expected = static_cast<size_t>(K) * static_cast<size_t>(N);
    if (b_rowmajor.size() != expected)
    {
        b_rowmajor.resize(expected);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kg = 0; kg < k_groups; ++kg)
    {
        for (int n = 0; n < N; ++n)
        {
            const size_t base = (static_cast<size_t>(kg) * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4u;
            b_rowmajor[static_cast<size_t>(kg * 4 + 0) * static_cast<size_t>(N) + static_cast<size_t>(n)] = b_vnni[base + 0];
            b_rowmajor[static_cast<size_t>(kg * 4 + 1) * static_cast<size_t>(N) + static_cast<size_t>(n)] = b_vnni[base + 1];
            b_rowmajor[static_cast<size_t>(kg * 4 + 2) * static_cast<size_t>(N) + static_cast<size_t>(n)] = b_vnni[base + 2];
            b_rowmajor[static_cast<size_t>(kg * 4 + 3) * static_cast<size_t>(N) + static_cast<size_t>(n)] = b_vnni[base + 3];
        }
    }
}

std::vector<int8_t> clipToQ4Signed(const std::vector<int8_t> &in)
{
    std::vector<int8_t> out(in.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < in.size(); ++i)
    {
        const int v = static_cast<int>(in[i]);
        out[i] = static_cast<int8_t>(max(-8, min(7, v)));
    }
    return out;
}

std::vector<uint8_t> packRowMajorToQ4VNNI(const std::vector<int8_t> &b_rowmajor_q4, int N, int K)
{
    const int k_groups = K / 4;
    std::vector<uint8_t> out(static_cast<size_t>(k_groups) * static_cast<size_t>(N) * 2u);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kg = 0; kg < k_groups; ++kg)
    {
        for (int n = 0; n < N; ++n)
        {
            const int8_t w0 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 0) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w1 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 1) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w2 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 2) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w3 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 3) * static_cast<size_t>(N) + static_cast<size_t>(n)];

            const uint8_t n0 = static_cast<uint8_t>(w0) & 0x0Fu;
            const uint8_t n1 = static_cast<uint8_t>(w1) & 0x0Fu;
            const uint8_t n2 = static_cast<uint8_t>(w2) & 0x0Fu;
            const uint8_t n3 = static_cast<uint8_t>(w3) & 0x0Fu;

            const size_t base = (static_cast<size_t>(kg) * static_cast<size_t>(N) + static_cast<size_t>(n)) * 2u;
            out[base + 0] = static_cast<uint8_t>(n0 | (n1 << 4));
            out[base + 1] = static_cast<uint8_t>(n2 | (n3 << 4));
        }
    }
    return out;
}

void packRowMajorToQ4VNNIInplace(const std::vector<int8_t> &b_rowmajor_q4, int N, int K, std::vector<uint8_t> &out)
{
    const int k_groups = K / 4;
    const size_t expected = static_cast<size_t>(k_groups) * static_cast<size_t>(N) * 2u;
    if (out.size() != expected)
    {
        out.resize(expected);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kg = 0; kg < k_groups; ++kg)
    {
        for (int n = 0; n < N; ++n)
        {
            const int8_t w0 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 0) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w1 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 1) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w2 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 2) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            const int8_t w3 = b_rowmajor_q4[static_cast<size_t>(kg * 4 + 3) * static_cast<size_t>(N) + static_cast<size_t>(n)];

            const uint8_t n0 = static_cast<uint8_t>(w0) & 0x0Fu;
            const uint8_t n1 = static_cast<uint8_t>(w1) & 0x0Fu;
            const uint8_t n2 = static_cast<uint8_t>(w2) & 0x0Fu;
            const uint8_t n3 = static_cast<uint8_t>(w3) & 0x0Fu;

            const size_t base = (static_cast<size_t>(kg) * static_cast<size_t>(N) + static_cast<size_t>(n)) * 2u;
            out[base + 0] = static_cast<uint8_t>(n0 | (n1 << 4));
            out[base + 1] = static_cast<uint8_t>(n2 | (n3 << 4));
        }
    }
}

std::vector<int8_t> transposeKxNToNxK(const std::vector<int8_t> &b_rowmajor, int N, int K)
{
    std::vector<int8_t> out(static_cast<size_t>(N) * static_cast<size_t>(K));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            out[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                b_rowmajor[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        }
    }
    return out;
}

void transposeKxNToNxKInplace(const std::vector<int8_t> &b_rowmajor, int N, int K, std::vector<int8_t> &out)
{
    const size_t expected = static_cast<size_t>(N) * static_cast<size_t>(K);
    if (out.size() != expected)
    {
        out.resize(expected);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            out[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                b_rowmajor[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
        }
    }
}

std::vector<int32_t> computeCpuReference(const std::vector<int8_t> &a, const std::vector<int8_t> &b_rowmajor, int M, int N, int K)
{
    std::vector<int32_t> out(static_cast<size_t>(M) * static_cast<size_t>(N), 0);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
            {
                const int8_t av = a[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)];
                const int8_t bv = b_rowmajor[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
                acc += static_cast<int32_t>(av) * static_cast<int32_t>(bv);
            }
            out[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = acc;
        }
    }
    return out;
}

double runTimedLoop(
    int warmup,
    int iters,
    const std::function<void()> &launcher,
    hipStream_t stream)
{
    for (int i = 0; i < warmup; ++i)
    {
        launcher();
    }
    hipStreamSynchronize(stream);

    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, stream);
    for (int i = 0; i < iters; ++i)
    {
        launcher();
    }
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, start, stop);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

double runTimedHostPrepLoop(
    int warmup,
    int iters,
    const std::function<void()> &prep)
{
    for (int i = 0; i < warmup; ++i)
    {
        prep();
    }

    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i)
    {
        prep();
    }
    const auto t1 = clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / static_cast<double>(iters);
}

double tflopsFor(int M, int N, int K, double ms)
{
    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return (ops / (ms * 1e-3)) / 1e12;
}

int roundUpToMultiple(int value, int multiple)
{
    if (multiple <= 1)
    {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

int chooseNoAtomicSplitK(int K)
{
    const int k_groups = std::max(1, K / 4);
    constexpr int target_groups_per_slice = 64;
    const int slices_raw = (k_groups + target_groups_per_slice - 1) / target_groups_per_slice;
    if (slices_raw <= 2)
    {
        return 2;
    }
    if (slices_raw <= 4)
    {
        return 4;
    }
    return 8;
}

int computeProdPrefillGridSlices(int M, int N, int K, int grid_variant, int cpt)
{
    const int k_groups = std::max(1, K / 4);
    constexpr int target_groups_per_slice = 64;
    constexpr int target_min_waves_per_cu = 8;
    constexpr int num_cus = 60;

    int auto_slices = std::max(1, k_groups / target_groups_per_slice);

    int tile_x = 16;
    if (grid_variant == 1)
    {
        tile_x = 32;
    }
    else if (grid_variant == 2 || grid_variant == 3)
    {
        tile_x = 8;
    }

    const int logical_tile_n = std::max(1, tile_x * std::max(1, cpt));
    const int grid_n = std::max(1, (N + logical_tile_n - 1) / logical_tile_n);
    const int min_slices_for_waves = std::max(1, (target_min_waves_per_cu * num_cus + grid_n - 1) / grid_n);
    auto_slices = std::max(auto_slices, min_slices_for_waves);

    if (M < 8)
    {
        auto_slices = std::min(auto_slices, 2);
    }
    else if (M >= 24 && K >= 1024)
    {
        auto_slices = std::max(auto_slices, 4);
    }

    const int n_tiles = std::max(1, (N + 127) / 128);
    const int hard_cap = std::min(8, std::max(1, n_tiles * 2));
    return std::clamp(auto_slices, 1, hard_cap);
}

int maxAbsDiff(const std::vector<int32_t> &a, const std::vector<int32_t> &b)
{
    int max_diff = 0;
    #pragma omp parallel for reduction(max:max_diff) schedule(static)
    for (size_t i = 0; i < a.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

Result benchmarkStrategy(
    const Shape &shape,
    const std::string &strategy,
    int splitk,
    const Options &opt,
    const int8_t *d_A,
    const int8_t *d_B_rowmajor,
    const int8_t *d_B_vnni,
    const uint8_t *d_B_q4_vnni,
    int32_t *d_C,
    hipStream_t stream,
    const std::vector<int32_t> &cpu_ref,
    const std::vector<int32_t> &cpu_ref_q4)
{
    const int M = shape.M;
    const int N = shape.N;
    const int K = shape.K;
    bool launch_failed = false;

    auto launch = [&]() {
        hipMemsetAsync(d_C, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t), stream);

        if (strategy == "scalar")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX - 1) / BX),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL(gemm_scalar_prefill_kernel, grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "cpt2")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_cpt_prefill_kernel<BX, BY, CPT>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "cpt4")
        {
            constexpr int BX = 16;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_cpt_prefill_kernel<BX, BY, CPT>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "cpt2_splitk")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                static_cast<unsigned>(splitk));
            hipLaunchKernelGGL((gemm_cpt_splitk_prefill_kernel<BX, BY, CPT>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K, splitk);
        }
        else if (strategy == "cpt2_sharedb")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_cpt2_shared_b_prefill_kernel<BX, BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "cpt2_sharedb_splitk")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                static_cast<unsigned>(splitk));
            hipLaunchKernelGGL((gemm_cpt2_shared_b_splitk_prefill_kernel<BX, BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K, splitk);
        }
        else if (strategy == "cpt2_shared_ab_kg8")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_cpt2_shared_ab_ktile_prefill_kernel<BX, BY, 8>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "cpt2_shared_ab_kg16")
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            constexpr int CPT = 2;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_cpt2_shared_ab_ktile_prefill_kernel<BX, BY, 16>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8_cap24")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 24))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8_cap48")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 48))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8_cap64")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 64))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_by8_cap192")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 192))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_grid_cap64")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 64))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_aligned_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_aligned_by4_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_aligned_unroll2_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_unroll2_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_aligned_dualacc_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_dualacc_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_aligned_unroll2_lb_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_unroll2_lb_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_auto_tuned")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % BY) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);

            int cap = 96;
            bool use_aligned_unroll2_lb = false;

            if (aligned && N <= 4096 && K <= 4096)
            {
                cap = 128;
                use_aligned_unroll2_lb = true;
            }
            else if (N >= 16384 && K <= 4096)
            {
                cap = 128;
            }
            else if (K >= 16384)
            {
                cap = 96;
            }
            else
            {
                cap = 128;
            }

            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);

            if (use_aligned_unroll2_lb)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_unroll2_lb_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
        }
        else if (strategy == "wave64_cpt4_aligned_mr2_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 2)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 2) - 1) / (BY * 2)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_mr2_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt4_aligned_mr4_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_mr4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt4_aligned_mr8_abcast_lane0_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            constexpr int MR = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * MR)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * MR) - 1) / (BY * MR)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_mr8_abcast_lane0_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_cap64" ||
                 strategy == "wave64_cpt8_aligned_mr4_cap96" ||
                 strategy == "wave64_cpt8_aligned_mr4_cap192" ||
                 strategy == "wave64_cpt8_aligned_mr4_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);

            int cap = 128;
            if (strategy == "wave64_cpt8_aligned_mr4_cap64")
            {
                cap = 64;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_cap96")
            {
                cap = 96;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_cap192")
            {
                cap = 192;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_cap256")
            {
                cap = 256;
            }

            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap64" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap96" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap128" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap192" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);

            int cap = 256;
            if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap64")
            {
                cap = 64;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap96")
            {
                cap = 96;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap128")
            {
                cap = 128;
            }
            else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cap192")
            {
                cap = 192;
            }

            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coal_pipe_u4_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coal_pipe_u4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_i32addr_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_i32addr_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_lb3_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_lb3_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_lb4_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_coalonly_splitacc_lb4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_ldsB_kgt8_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_ldsB_prefill_kernel<BY, 8>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_avec4_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 16) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_avec4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_cap256" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const int cap = (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_cap128") ? 128 : 256;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_lb1_cap256" ||
                 strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_lb1_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const int cap = (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_by8_lb1_cap128") ? 128 : 256;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_lb1_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_pf1_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_pf1_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_k2_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_k2_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_k2_by8_cap256")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 256))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_k2_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_abcast_lane0_cpt16split_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 16;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_abcast_lane0_cpt16split_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_veca2_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_veca2_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_veca4_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_veca4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_veca4_rfl_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_veca4_rfl_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_ldsbdb_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            constexpr int KG_TILE = 16;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_ldsdb_prefill_kernel<BY, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_ldsbdb_kgt8_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            constexpr int KG_TILE = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            int cap = 128;
            if (const char *cap_env = std::getenv("LLAMINAR_PREFILL_CPT8_CAP"))
            {
                const int parsed = std::atoi(cap_env);
                if (parsed > 0)
                {
                    cap = parsed;
                }
            }
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, cap))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_ldsdb_prefill_kernel<BY, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_lds_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            constexpr int KG_TILE = 8;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_lds_prefill_kernel<BY, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt8_aligned_mr4_lds_kgt16_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 8;
            constexpr int KG_TILE = 16;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt8_gridstride_aligned_mr4_lds_prefill_kernel<BY, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt4_aligned_mr4_lds_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            constexpr int KG_TILE = 16;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * 4)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * 4) - 1) / (BY * 4)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_aligned_mr4_lds_prefill_kernel<BY, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt2_aligned_ckmirror128_lds_cap128")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 2;
            constexpr int ROWS_PER_GROUP = 32;
            constexpr int KG_TILE = 16;
            const bool aligned = ((N % (BX * CPT)) == 0) && ((K % 4) == 0) && ((M % (BY * ROWS_PER_GROUP)) == 0);
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1); // 256 threads/block
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 128))),
                static_cast<unsigned>((M + (BY * ROWS_PER_GROUP) - 1) / (BY * ROWS_PER_GROUP)),
                1);
            if (aligned)
            {
                hipLaunchKernelGGL((gemm_wave64_cpt2_gridstride_aligned_ckmirror128_lds_prefill_kernel<BY, ROWS_PER_GROUP, KG_TILE>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
            }
            else
            {
                launch_failed = true;
            }
        }
        else if (strategy == "wave64_cpt4_unroll4_grid_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_gridstride_unroll4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_splitk")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                static_cast<unsigned>(splitk));
            hipLaunchKernelGGL((gemm_wave64_cpt4_splitk_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K, splitk);
        }
        else if (strategy == "wave64_cpt4_splitk_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                static_cast<unsigned>(splitk));
            hipLaunchKernelGGL((gemm_wave64_cpt4_splitk_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K, splitk);
        }
        else if (strategy == "wave64_cpt4_shared_ab_kg8")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_ab_ktile_prefill_kernel<BY, 8>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_shared_ab_kg16")
        {
            constexpr int BX = 64;
            constexpr int BY = 4;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_ab_ktile_prefill_kernel<BY, 16>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_shared_ab_kg8_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_ab_ktile_prefill_kernel<BY, 8>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_shared_ab_kg16_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_ab_ktile_prefill_kernel<BY, 16>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_sharedb_kg8_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_b_ktile_prefill_kernel<BY, 8>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "wave64_cpt4_sharedb_kg16_by8")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_cpt4_shared_b_ktile_prefill_kernel<BY, 16>), grid, block, 0, stream, d_A, d_B_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_wave64_cpt4")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_cpt4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_wave64_cpt4_grid")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_lut_wave64_cpt4")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_lut_cpt4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_lut_wave64_cpt4_grid")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_lut_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_perm_wave64_cpt4")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>((N + BX * CPT - 1) / (BX * CPT)),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_perm_cpt4_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "q4nib_perm_wave64_cpt4_grid")
        {
            constexpr int BX = 64;
            constexpr int BY = 8;
            constexpr int CPT = 4;
            const int full_tiles = (N + BX * CPT - 1) / (BX * CPT);
            const dim3 block(BX, BY, 1);
            const dim3 grid(
                static_cast<unsigned>(max(1, min(full_tiles, 96))),
                static_cast<unsigned>((M + BY - 1) / BY),
                1);
            hipLaunchKernelGGL((gemm_wave64_q4nib_perm_cpt4_gridstride_prefill_kernel<BY>), grid, block, 0, stream, d_A, d_B_q4_vnni, d_C, M, N, K);
        }
        else if (strategy == "native_prefill_auto")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill(d_A, d_B_vnni, d_C, M, N, K,
                                                       -1,
                                                       4,
                                                       device_id,
                                                       stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_auto_cpt1")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill(d_A, d_B_vnni, d_C, M, N, K,
                                                       -1,
                                                       1,
                                                       device_id,
                                                       stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_32x8_cpt4")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill(d_A, d_B_vnni, d_C, M, N, K,
                                                       1,
                                                       4,
                                                       device_id,
                                                       stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_32x8_cpt1")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill(d_A, d_B_vnni, d_C, M, N, K,
                                                       1,
                                                       1,
                                                       device_id,
                                                       stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s8")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  8,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s4")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  4,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s6")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  6,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s5")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  5,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s7")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  7,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s6_gv2")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  6,
                                                                  2,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s6_gv3")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  6,
                                                                  3,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_s10")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  10,
                                                                  1,
                                                                  4,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_auto_cpt1")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            const int cpt = 1;
            const int grid_variant = -1;
            const int slices = computeProdPrefillGridSlices(M, N, K, grid_variant, cpt);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  slices,
                                                                  grid_variant,
                                                                  cpt,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_auto_cpt4")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            const int cpt = 4;
            const int grid_variant = -1;
            const int slices = computeProdPrefillGridSlices(M, N, K, grid_variant, cpt);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  slices,
                                                                  grid_variant,
                                                                  cpt,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_auto_cpt4_gv1")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            const int cpt = 4;
            const int grid_variant = 1;
            const int slices = computeProdPrefillGridSlices(M, N, K, grid_variant, cpt);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  slices,
                                                                  grid_variant,
                                                                  cpt,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_auto_cpt4_gv2")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            const int cpt = 4;
            const int grid_variant = 2;
            const int slices = computeProdPrefillGridSlices(M, N, K, grid_variant, cpt);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  slices,
                                                                  grid_variant,
                                                                  cpt,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else if (strategy == "native_prefill_gridkpar_auto_cpt4_gv3")
        {
            int device_id = opt.device;
            hipGetDevice(&device_id);
            const int cpt = 4;
            const int grid_variant = 3;
            const int slices = computeProdPrefillGridSlices(M, N, K, grid_variant, cpt);
            if (!rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(d_A, d_B_vnni, d_C, M, N, K,
                                                                  slices,
                                                                  grid_variant,
                                                                  cpt,
                                                                  device_id,
                                                                  stream))
            {
                launch_failed = true;
            }
        }
        else
        {
            launch_failed = true;
        }
    };

    Result out;
    out.shape = shape.name;
    out.strategy = strategy;
    out.splitk = splitk;

    out.ms = runTimedLoop(opt.warmup, opt.iters, launch, stream);
    out.tflops = tflopsFor(M, N, K, out.ms);

    if (launch_failed)
    {
        out.max_abs_diff = std::numeric_limits<int>::max();
        out.ok = false;
        return out;
    }

    if (opt.check_correctness)
    {
        std::vector<int32_t> host(static_cast<size_t>(M) * static_cast<size_t>(N));
        hipMemcpy(host.data(), d_C, host.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
        const bool is_q4nib =
            (strategy == "q4nib_wave64_cpt4" || strategy == "q4nib_wave64_cpt4_grid" ||
             strategy == "q4nib_lut_wave64_cpt4" || strategy == "q4nib_lut_wave64_cpt4_grid" ||
             strategy == "q4nib_perm_wave64_cpt4" || strategy == "q4nib_perm_wave64_cpt4_grid");
        out.max_abs_diff = maxAbsDiff(host, is_q4nib ? cpu_ref_q4 : cpu_ref);
        out.ok = (out.max_abs_diff == 0);
    }

    return out;
}

void printTable(const std::vector<Result> &results)
{
    std::cout << "[prefill-lab] table note: RepackMS reports host-side repack overhead. "
              << "vsCK(E2E) uses harvested canonical CK E2E when available (otherwise measured CK E2E). "
              << "Only CK baseline reports simulated INT8-VNNI->CK repack; non-CK strategies keep repack excluded. "
              << "Activation quantization stays inside KernelMS."
              << std::endl;

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Shape" << "Strategy" << "SplitK" << "KernelMS" << "RepackMS" << "E2EMS" << "TFLOPS" << "vsCK(K)" << "vsCK(E2E)" << "vsCanon(K)" << "vsCanon(E2E)" << "MaxAbsDiff" << "OK" << fort::endr;

    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::left);
    table.column(2).set_cell_text_align(fort::text_align::right);
    table.column(3).set_cell_text_align(fort::text_align::right);
    table.column(4).set_cell_text_align(fort::text_align::right);
    table.column(5).set_cell_text_align(fort::text_align::right);
    table.column(6).set_cell_text_align(fort::text_align::right);
    table.column(7).set_cell_text_align(fort::text_align::right);
    table.column(8).set_cell_text_align(fort::text_align::right);
    table.column(9).set_cell_text_align(fort::text_align::right);
    table.column(10).set_cell_text_align(fort::text_align::right);
    table.column(11).set_cell_text_align(fort::text_align::right);
    table.column(12).set_cell_text_align(fort::text_align::right);
    table.column(13).set_cell_text_align(fort::text_align::center);

    for (const auto &r : results)
    {
        std::ostringstream ms;
        std::ostringstream tf;
        std::ostringstream vs;
        std::ostringstream repack;
        std::ostringstream e2e;
        std::ostringstream vse2e;
        std::ostringstream vcanonk;
        std::ostringstream vcanone2e;
        ms << std::fixed << std::setprecision(3) << r.ms;
        repack << std::fixed << std::setprecision(3) << r.repack_ms;
        e2e << std::fixed << std::setprecision(3) << r.e2e_ms;
        tf << std::fixed << std::setprecision(3) << r.tflops;
        vs << std::fixed << std::setprecision(3) << r.speedup_vs_ck;
        vse2e << std::fixed << std::setprecision(3) << r.speedup_vs_ck_e2e;
        vcanonk << std::fixed << std::setprecision(3) << r.speedup_vs_ck_canonical_k;
        vcanone2e << std::fixed << std::setprecision(3) << r.speedup_vs_ck_canonical_e2e;

        table << r.shape
              << r.strategy
              << r.splitk
              << ms.str()
              << repack.str()
              << e2e.str()
              << tf.str()
              << vs.str()
              << vse2e.str()
              << vcanonk.str()
              << vcanone2e.str()
              << r.max_abs_diff
              << (r.ok ? "Y" : "N")
              << fort::endr;
    }

    std::cout << table.to_string() << std::endl;
}

bool isInt8VnniGemmSubtype(const std::string &strategy)
{
    if (strategy == "ck_baseline")
    {
        return false;
    }
    if (strategy.rfind("q4nib_", 0) == 0)
    {
        return false;
    }
    return true;
}

struct ShapeWinner
{
    std::string shape;
    std::string strategy;
    int splitk = 1;
    double kernel_ms = 0.0;
    double repack_ms = 0.0;
    double e2e_ms = 0.0;
    double vs_ck_e2e = 0.0;
};

struct LeaderEntry
{
    std::string subtype;
    int wins = 0;
    double sum_vs_ck_e2e = 0.0;
    double best_vs_ck_e2e = 0.0;
    std::string best_shape;
};

struct LeaderboardSnapshot
{
    std::vector<ShapeWinner> winners;
    std::vector<LeaderEntry> leaderboard;
};

std::vector<ShapeWinner> collectShapeWinnersVsCkE2E(const std::vector<Result> &results)
{
    std::unordered_map<std::string, ShapeWinner> winners_by_shape;
    for (const auto &r : results)
    {
        if (!r.ok || !isInt8VnniGemmSubtype(r.strategy) || !(r.speedup_vs_ck_e2e > 1.0))
        {
            continue;
        }

        auto it = winners_by_shape.find(r.shape);
        if (it == winners_by_shape.end() || r.speedup_vs_ck_e2e > it->second.vs_ck_e2e)
        {
            winners_by_shape[r.shape] = ShapeWinner{r.shape, r.strategy, r.splitk, r.ms, r.repack_ms, r.e2e_ms, r.speedup_vs_ck_e2e};
        }
    }

    std::vector<ShapeWinner> winners;
    winners.reserve(winners_by_shape.size());
    for (const auto &kv : winners_by_shape)
    {
        winners.push_back(kv.second);
    }

    std::sort(winners.begin(), winners.end(), [](const ShapeWinner &a, const ShapeWinner &b) {
        if (a.vs_ck_e2e != b.vs_ck_e2e)
        {
            return a.vs_ck_e2e > b.vs_ck_e2e;
        }
        return a.shape < b.shape;
    });

    return winners;
}

std::vector<LeaderEntry> buildLeaderboardEntries(const std::vector<ShapeWinner> &winners)
{
    std::unordered_map<std::string, LeaderEntry> board;
    for (const auto &w : winners)
    {
        const std::string subtype_key = w.strategy + "|splitk=" + std::to_string(w.splitk);
        auto &entry = board[subtype_key];
        if (entry.subtype.empty())
        {
            entry.subtype = subtype_key;
        }
        entry.wins += 1;
        entry.sum_vs_ck_e2e += w.vs_ck_e2e;
        if (w.vs_ck_e2e > entry.best_vs_ck_e2e)
        {
            entry.best_vs_ck_e2e = w.vs_ck_e2e;
            entry.best_shape = w.shape;
        }
    }

    std::vector<LeaderEntry> leaderboard;
    leaderboard.reserve(board.size());
    for (const auto &kv : board)
    {
        leaderboard.push_back(kv.second);
    }

    std::sort(leaderboard.begin(), leaderboard.end(), [](const LeaderEntry &a, const LeaderEntry &b) {
        if (a.wins != b.wins)
        {
            return a.wins > b.wins;
        }
        if (a.best_vs_ck_e2e != b.best_vs_ck_e2e)
        {
            return a.best_vs_ck_e2e > b.best_vs_ck_e2e;
        }
        return a.subtype < b.subtype;
    });

    return leaderboard;
}

LeaderboardSnapshot printE2EWinnersAndLeaderboard(const std::vector<Result> &results)
{
    const std::vector<ShapeWinner> winners = collectShapeWinnersVsCkE2E(results);
    const std::vector<LeaderEntry> leaderboard = buildLeaderboardEntries(winners);

    fort::utf8_table winners_table;
    winners_table.set_border_style(FT_DOUBLE2_STYLE);
    winners_table << fort::header
                  << "Shape"
                  << "Winning Subtype"
                  << "SplitK"
                  << "KernelMS"
                  << "RepackMS"
                  << "E2EMS"
                  << "vsCK(E2E)"
                  << fort::endr;

    winners_table.column(0).set_cell_text_align(fort::text_align::left);
    winners_table.column(1).set_cell_text_align(fort::text_align::left);
    winners_table.column(2).set_cell_text_align(fort::text_align::right);
    winners_table.column(3).set_cell_text_align(fort::text_align::right);
    winners_table.column(4).set_cell_text_align(fort::text_align::right);
    winners_table.column(5).set_cell_text_align(fort::text_align::right);
    winners_table.column(6).set_cell_text_align(fort::text_align::right);

    if (winners.empty())
    {
        winners_table << "-" << "No INT8 VNNI GEMM subtype beats CK end-to-end yet" << "-" << "-" << "-" << "-" << "-" << fort::endr;
    }
    else
    {
        for (const auto &w : winners)
        {
            std::ostringstream kernel_ms;
            std::ostringstream repack_ms;
            std::ostringstream e2e_ms;
            std::ostringstream vs_e2e;
            kernel_ms << std::fixed << std::setprecision(3) << w.kernel_ms;
            repack_ms << std::fixed << std::setprecision(3) << w.repack_ms;
            e2e_ms << std::fixed << std::setprecision(3) << w.e2e_ms;
            vs_e2e << std::fixed << std::setprecision(3) << w.vs_ck_e2e;

            winners_table << w.shape
                          << w.strategy
                          << w.splitk
                          << kernel_ms.str()
                          << repack_ms.str()
                          << e2e_ms.str()
                          << vs_e2e.str()
                          << fort::endr;
        }
    }

    std::cout << "\nINT8 VNNI GEMM Winners vs CK (E2E)\n";
    std::cout << winners_table.to_string() << std::endl;

    fort::utf8_table leaderboard_table;
    leaderboard_table.set_border_style(FT_DOUBLE2_STYLE);
    leaderboard_table << fort::header
                      << "Rank"
                      << "INT8 VNNI GEMM Subtype"
                      << "Shape Wins"
                      << "Avg vsCK(E2E)"
                      << "Best vsCK(E2E)"
                      << "Best Shape"
                      << fort::endr;

    leaderboard_table.column(0).set_cell_text_align(fort::text_align::right);
    leaderboard_table.column(1).set_cell_text_align(fort::text_align::left);
    leaderboard_table.column(2).set_cell_text_align(fort::text_align::right);
    leaderboard_table.column(3).set_cell_text_align(fort::text_align::right);
    leaderboard_table.column(4).set_cell_text_align(fort::text_align::right);
    leaderboard_table.column(5).set_cell_text_align(fort::text_align::left);

    if (leaderboard.empty())
    {
        leaderboard_table << "-" << "No entries yet" << 0 << "-" << "-" << "-" << fort::endr;
    }
    else
    {
        int rank = 1;
        for (const auto &entry : leaderboard)
        {
            std::ostringstream avg_vs;
            std::ostringstream best_vs;
            const double avg = entry.wins > 0 ? (entry.sum_vs_ck_e2e / static_cast<double>(entry.wins)) : 0.0;
            avg_vs << std::fixed << std::setprecision(3) << avg;
            best_vs << std::fixed << std::setprecision(3) << entry.best_vs_ck_e2e;

            leaderboard_table << rank
                              << entry.subtype
                              << entry.wins
                              << avg_vs.str()
                              << best_vs.str()
                              << (entry.best_shape.empty() ? "-" : entry.best_shape)
                              << fort::endr;
            ++rank;
        }
    }

    std::cout << "INT8 VNNI GEMM Leaderboard (CK E2E Winners)\n";
    std::cout << leaderboard_table.to_string() << std::endl;

    return LeaderboardSnapshot{winners, leaderboard};
}

std::string makeUtcTimestampString()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string joinDevicesCSV(const std::vector<int> &devices)
{
    std::ostringstream oss;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (i > 0)
        {
            oss << ",";
        }
        oss << devices[i];
    }
    return oss.str();
}

void persistLeaderboardOnDisk(
    const Options &opt,
    const std::vector<int> &devices,
    const LeaderboardSnapshot &snapshot)
{
    if (!opt.persist_leaderboard)
    {
        return;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(opt.leaderboard_dir, ec);
    if (ec)
    {
        std::cerr << "[prefill-lab] warning: failed to create leaderboard dir '"
                  << opt.leaderboard_dir << "': " << ec.message() << std::endl;
        return;
    }

    const std::string ts = makeUtcTimestampString();
    const std::string devices_csv = joinDevicesCSV(devices);

    const fs::path latest_winners_path = fs::path(opt.leaderboard_dir) / "int8_vnni_gemm_winners_latest.csv";
    const fs::path latest_board_path = fs::path(opt.leaderboard_dir) / "int8_vnni_gemm_leaderboard_latest.csv";
    const fs::path winners_hist_path = fs::path(opt.leaderboard_dir) / "int8_vnni_gemm_winners_history.csv";
    const fs::path board_hist_path = fs::path(opt.leaderboard_dir) / "int8_vnni_gemm_leaderboard_history.csv";

    {
        std::ofstream out(latest_winners_path);
        if (out)
        {
            out << "timestamp_utc,devices,shape,strategy,splitk,kernel_ms,repack_ms,e2e_ms,vs_ck_e2e\n";
            for (const auto &w : snapshot.winners)
            {
                out << ts << ","
                    << '"' << devices_csv << '"' << ","
                    << '"' << w.shape << '"' << ","
                    << '"' << w.strategy << '"' << ","
                    << w.splitk << ","
                    << std::fixed << std::setprecision(6) << w.kernel_ms << ","
                    << std::fixed << std::setprecision(6) << w.repack_ms << ","
                    << std::fixed << std::setprecision(6) << w.e2e_ms << ","
                    << std::fixed << std::setprecision(6) << w.vs_ck_e2e
                    << "\n";
            }
        }
    }

    {
        std::ofstream out(latest_board_path);
        if (out)
        {
            out << "timestamp_utc,devices,rank,subtype,shape_wins,avg_vs_ck_e2e,best_vs_ck_e2e,best_shape\n";
            int rank = 1;
            for (const auto &entry : snapshot.leaderboard)
            {
                const double avg = entry.wins > 0 ? (entry.sum_vs_ck_e2e / static_cast<double>(entry.wins)) : 0.0;
                out << ts << ","
                    << '"' << devices_csv << '"' << ","
                    << rank << ","
                    << '"' << entry.subtype << '"' << ","
                    << entry.wins << ","
                    << std::fixed << std::setprecision(6) << avg << ","
                    << std::fixed << std::setprecision(6) << entry.best_vs_ck_e2e << ","
                    << '"' << entry.best_shape << '"'
                    << "\n";
                ++rank;
            }
        }
    }

    const bool winners_hist_exists = fs::exists(winners_hist_path, ec) && !ec;
    {
        std::ofstream out(winners_hist_path, std::ios::app);
        if (out)
        {
            if (!winners_hist_exists)
            {
                out << "timestamp_utc,devices,shape,strategy,splitk,kernel_ms,repack_ms,e2e_ms,vs_ck_e2e\n";
            }
            for (const auto &w : snapshot.winners)
            {
                out << ts << ","
                    << '"' << devices_csv << '"' << ","
                    << '"' << w.shape << '"' << ","
                    << '"' << w.strategy << '"' << ","
                    << w.splitk << ","
                    << std::fixed << std::setprecision(6) << w.kernel_ms << ","
                    << std::fixed << std::setprecision(6) << w.repack_ms << ","
                    << std::fixed << std::setprecision(6) << w.e2e_ms << ","
                    << std::fixed << std::setprecision(6) << w.vs_ck_e2e
                    << "\n";
            }
        }
    }

    const bool board_hist_exists = fs::exists(board_hist_path, ec) && !ec;
    {
        std::ofstream out(board_hist_path, std::ios::app);
        if (out)
        {
            if (!board_hist_exists)
            {
                out << "timestamp_utc,devices,rank,subtype,shape_wins,avg_vs_ck_e2e,best_vs_ck_e2e,best_shape\n";
            }
            int rank = 1;
            for (const auto &entry : snapshot.leaderboard)
            {
                const double avg = entry.wins > 0 ? (entry.sum_vs_ck_e2e / static_cast<double>(entry.wins)) : 0.0;
                out << ts << ","
                    << '"' << devices_csv << '"' << ","
                    << rank << ","
                    << '"' << entry.subtype << '"' << ","
                    << entry.wins << ","
                    << std::fixed << std::setprecision(6) << avg << ","
                    << std::fixed << std::setprecision(6) << entry.best_vs_ck_e2e << ","
                    << '"' << entry.best_shape << '"'
                    << "\n";
                ++rank;
            }
        }
    }

    std::cout << "[prefill-lab] persisted leaderboard files in " << opt.leaderboard_dir << std::endl;
}

bool runShapesOnDevice(const Options &opt, int device, const std::vector<Shape> &shapes, std::vector<Result> &results_out)
{
    if (shapes.empty())
    {
        return true;
    }

    hipSetDevice(device);
    hipStream_t stream = nullptr;
    hipStreamCreate(&stream);

    if (!ensureQ4PairDecodeLutOnDevice(device))
    {
        std::cerr << "Failed to initialize Q4 pair decode LUT on device " << device << std::endl;
        hipStreamDestroy(stream);
        return false;
    }

    std::cout << "[prefill-lab] device=" << device
              << " accounting policy: vsCK(E2E) uses canonical harvested CK E2E when available (fallback: measured CK E2E); "
              << "CK row still reports simulated INT8-VNNI->CK repack, non-CK weight repack remains excluded; "
              << "activation-quantization remains in kernel timing"
              << std::endl;

    for (const auto &shape : shapes)
    {
        const CanonicalCKBaseline *canonical_ck = findCanonicalCKBaseline(shape.name);
        if (canonical_ck)
        {
            std::cout << "[prefill-lab] device=" << device
                      << " canonical_ck shape=" << shape.name
                      << " ms=" << std::fixed << std::setprecision(6) << canonical_ck->mean_ms
                      << " tflops=" << std::fixed << std::setprecision(6) << canonical_ck->mean_tflops
                      << std::endl;
        }

        std::cout << "[prefill-lab] device=" << device
                  << " start shape=" << shape.name
                  << " (M=" << shape.M << ",N=" << shape.N << ",K=" << shape.K << ")"
                  << std::endl;

        if ((shape.K % 4) != 0)
        {
            std::cerr << "Skipping shape " << shape.name << " on device " << device << ": K must be multiple of 4" << std::endl;
            continue;
        }

        const size_t a_elems = static_cast<size_t>(shape.M) * static_cast<size_t>(shape.K);
        const size_t b_elems = static_cast<size_t>(shape.K) * static_cast<size_t>(shape.N);
        const size_t c_elems = static_cast<size_t>(shape.M) * static_cast<size_t>(shape.N);

        std::vector<int8_t> h_A(a_elems);
        std::vector<int8_t> h_B_rowmajor(b_elems);
        fillRandomInt8(h_A, 1234 + static_cast<uint64_t>(shape.M + shape.N + shape.K));
        fillRandomInt8(h_B_rowmajor, 5678 + static_cast<uint64_t>(shape.M + shape.N + shape.K));
        std::vector<int8_t> h_B_rowmajor_q4 = clipToQ4Signed(h_B_rowmajor);
        std::vector<int8_t> h_B_nxk;
        if (opt.check_correctness)
        {
            h_B_nxk = transposeKxNToNxK(h_B_rowmajor, shape.N, shape.K);
        }
        std::vector<int8_t> h_B_vnni = packRowMajorToVNNI(h_B_rowmajor, shape.N, shape.K);
        std::vector<uint8_t> h_B_q4_vnni = packRowMajorToQ4VNNI(h_B_rowmajor_q4, shape.N, shape.K);
        std::vector<int32_t> cpu_ref;
        std::vector<int32_t> cpu_ref_q4;
        if (opt.check_correctness)
        {
            cpu_ref = computeCpuReference(h_A, h_B_rowmajor, shape.M, shape.N, shape.K);
            cpu_ref_q4 = computeCpuReference(h_A, h_B_rowmajor_q4, shape.M, shape.N, shape.K);
        }

        int8_t *d_A = nullptr;
        int8_t *d_B_rowmajor = nullptr;
        int8_t *d_B_nxk = nullptr;
        int8_t *d_B_vnni = nullptr;
        uint8_t *d_B_q4_vnni = nullptr;
        int32_t *d_C = nullptr;

        hipMalloc(&d_A, a_elems * sizeof(int8_t));
        hipMalloc(&d_B_rowmajor, b_elems * sizeof(int8_t));
        if (opt.check_correctness)
        {
            hipMalloc(&d_B_nxk, b_elems * sizeof(int8_t));
        }
        hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
        hipMalloc(&d_B_q4_vnni, h_B_q4_vnni.size() * sizeof(uint8_t));
        hipMalloc(&d_C, c_elems * sizeof(int32_t));

        hipMemcpyAsync(d_A, h_A.data(), a_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_B_rowmajor, h_B_rowmajor.data(), b_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
        if (opt.check_correctness)
        {
            hipMemcpyAsync(d_B_nxk, h_B_nxk.data(), b_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
        }
        hipMemcpyAsync(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_B_q4_vnni, h_B_q4_vnni.data(), h_B_q4_vnni.size() * sizeof(uint8_t), hipMemcpyHostToDevice, stream);
        hipStreamSynchronize(stream);

        void *ck_ctx = rocmQuantGemm_createKernelContext(device);
        if (!ck_ctx)
        {
            std::cerr << "Failed to create CK context on device " << device << std::endl;
            hipFree(d_A);
            hipFree(d_B_rowmajor);
            if (d_B_nxk)
            {
                hipFree(d_B_nxk);
            }
            hipFree(d_B_vnni);
            hipFree(d_B_q4_vnni);
            hipFree(d_C);
            hipStreamDestroy(stream);
            return false;
        }

        auto run_ck = [&](const int8_t *d_B_for_ck) {
            rocmQuantGemm_executeNoScale(
                d_A,
                d_B_for_ck,
                d_C,
                shape.M,
                shape.N,
                shape.K,
                device,
                stream,
                ck_ctx);
        };

                const int8_t *d_ck_weights = d_B_rowmajor;
                int ck_best_diff = 0;
                std::vector<int8_t> h_B_ck_rowmajor;
                const double ck_repack_ms = runTimedHostPrepLoop(opt.warmup, opt.iters, [&]() {
                    unpackVNNIToRowMajorInplace(h_B_vnni, shape.N, shape.K, h_B_ck_rowmajor);
                });
                if (opt.check_correctness)
                {
                        hipMemsetAsync(d_C, 0, c_elems * sizeof(int32_t), stream);
                        run_ck(d_B_rowmajor);
                        hipStreamSynchronize(stream);
                        std::vector<int32_t> ck_rowmajor_ref(c_elems);
                        hipMemcpy(ck_rowmajor_ref.data(), d_C, c_elems * sizeof(int32_t), hipMemcpyDeviceToHost);
                        const int diff_rowmajor = maxAbsDiff(ck_rowmajor_ref, cpu_ref);

                        hipMemsetAsync(d_C, 0, c_elems * sizeof(int32_t), stream);
                        run_ck(d_B_nxk);
                        hipStreamSynchronize(stream);
                        std::vector<int32_t> ck_nxk_ref(c_elems);
                        hipMemcpy(ck_nxk_ref.data(), d_C, c_elems * sizeof(int32_t), hipMemcpyDeviceToHost);
                        const int diff_nxk = maxAbsDiff(ck_nxk_ref, cpu_ref);

                        d_ck_weights = (diff_nxk < diff_rowmajor) ? d_B_nxk : d_B_rowmajor;
                        ck_best_diff = std::min(diff_rowmajor, diff_nxk);

                        (void)d_ck_weights;
                }

        const double ck_ms = runTimedLoop(opt.warmup, opt.iters, [&]() {
            run_ck(d_ck_weights);
        }, stream);
        const double ck_e2e_ms = ck_ms + ck_repack_ms;
        const double ck_e2e_compare_ms = canonical_ck ? canonical_ck->mean_ms : ck_e2e_ms;

        const double ck_tflops = tflopsFor(shape.M, shape.N, shape.K, ck_ms);
        Result ck_row;
        ck_row.shape = shape.name;
        ck_row.strategy = "ck_baseline";
        ck_row.splitk = 1;
        ck_row.ms = ck_ms;
        ck_row.repack_ms = ck_repack_ms;
        ck_row.e2e_ms = ck_e2e_ms;
        ck_row.tflops = ck_tflops;
        ck_row.speedup_vs_ck = 1.0;
        ck_row.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(ck_row.e2e_ms, 1e-9);
        if (canonical_ck)
        {
            ck_row.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(ck_row.ms, 1e-9);
            ck_row.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(ck_row.e2e_ms, 1e-9);
        }
        ck_row.max_abs_diff = ck_best_diff;
        ck_row.ok = (ck_best_diff == 0);
        results_out.push_back(ck_row);

        const double custom_int8_repack_ms = 0.0;
        const double custom_q4_repack_ms = 0.0;

        std::vector<std::string> shape_strategies = opt.strategies;
        const bool user_strategy_override = opt.user_strategy_override && !shape_strategies.empty();
        const double n_over_k = static_cast<double>(shape.N) / static_cast<double>(std::max(shape.K, 1));
        const bool is_lm_head = (shape.N >= 65536) && (n_over_k >= 32.0);
        const bool is_wide_or_extrawide = (n_over_k >= 2.0);

        if (is_wide_or_extrawide && opt.prefer_ck_wide && !user_strategy_override)
        {
            shape_strategies.clear();
            std::cout << "[prefill-lab] device=" << device
                      << " dispatch policy: CK preferred for wide shape"
                      << " (N/K=" << std::fixed << std::setprecision(2) << n_over_k << ")"
                      << "; pass --no-prefer-ck-wide to benchmark custom variants"
                      << std::endl;
        }
        else if (is_wide_or_extrawide && !shape_strategies.empty() && !user_strategy_override)
        {
            // S5 promotion: CK-style skinny/wide shortlist for wide/extrawide classes.
            // Focuses on rectangular/grid-cap variants that have shown better behavior
            // on FFN-up/gate and LM-head-like aspect ratios.
            shape_strategies = {
                "lmhead_vec_mr4_pad_cpt8",
                "lmhead_vec_mr4_pad_cpt4",
                "native_prefill_auto_cpt1",
                "native_prefill_32x8_cpt1",
                "native_prefill_gridkpar_auto_cpt1",
                "native_prefill_gridkpar_s8",
                "native_prefill_32x8_cpt4",
                "native_prefill_auto",
                "wave64_cpt4_grid_by8_cap24",
                "wave64_cpt4_grid_by8_cap48",
                "wave64_cpt4_grid_by8_cap64",
                "wave64_cpt4_grid_cap64",
                "wave64_cpt4_aligned_unroll2_lb_cap128",
                "wave64_cpt4_sharedb_kg16_by8",
                "wave64_cpt4_auto_tuned"};
            std::cout << "[prefill-lab] device=" << device
                      << " using S5 skinny/wide shortlist (" << shape_strategies.size() << " strategies)"
                      << " for wide shape N/K=" << std::fixed << std::setprecision(2) << n_over_k
                      << std::endl;
        }

        if (is_lm_head && !opt.full_lm_head && !user_strategy_override)
        {
            if (!shape_strategies.empty())
            {
                shape_strategies = {
                    "lmhead_vec_mr4_pad_cpt8",
                    "lmhead_vec_mr4_pad_cpt4",
                    "native_prefill_auto_cpt1",
                    "native_prefill_32x8_cpt1",
                    "native_prefill_gridkpar_auto_cpt1",
                    "native_prefill_gridkpar_s8",
                    "native_prefill_32x8_cpt4",
                    "native_prefill_auto",
                    "wave64_cpt4_grid_by8_cap64",
                    "wave64_cpt4_grid_by8_cap128",
                    "wave64_cpt4_aligned_unroll2_lb_cap128",
                    "wave64_cpt4_auto_tuned",
                    "wave64_cpt4_sharedb_kg16_by8"};
                std::cout << "[prefill-lab] device=" << device
                          << " using fast LM-head subset (" << shape_strategies.size() << " strategies); pass --full-lm-head for exhaustive" << std::endl;
            }
        }

        if (user_strategy_override)
        {
            std::cout << "[prefill-lab] device=" << device
                      << " honoring explicit --strategies override (" << shape_strategies.size()
                      << " strategies)" << std::endl;
        }

        for (const auto &strategy : shape_strategies)
        {
            if (strategy == "ck_baseline")
            {
                continue;
            }

            if (strategy == "lmhead_vec_mr4_pad_cpt4")
            {
                constexpr int kVectorCols = 256; // 64 lanes * CPT4
                constexpr int kVectorRows = 32;  // BY(8) * MR(4)

                const int n_padded = roundUpToMultiple(shape.N, kVectorCols);
                const int m_padded = roundUpToMultiple(shape.M, kVectorRows);

                const size_t a_pad_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(shape.K);
                const size_t b_pad_elems = static_cast<size_t>(shape.K) * static_cast<size_t>(n_padded);
                const size_t c_pad_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(n_padded);

                std::vector<int8_t> h_A_padded;
                int8_t *d_A_padded = d_A;
                if (m_padded != shape.M)
                {
                    h_A_padded.assign(a_pad_elems, 0);
                    for (int mm = 0; mm < shape.M; ++mm)
                    {
                        const int8_t *src = h_A.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                        int8_t *dst = h_A_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                        std::memcpy(dst, src, static_cast<size_t>(shape.K) * sizeof(int8_t));
                    }
                    hipMalloc(&d_A_padded, a_pad_elems * sizeof(int8_t));
                    hipMemcpyAsync(d_A_padded, h_A_padded.data(), a_pad_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                }

                std::vector<int8_t> h_B_rowmajor_padded(b_pad_elems, 0);
                for (int kk = 0; kk < shape.K; ++kk)
                {
                    const int8_t *src = h_B_rowmajor.data() + static_cast<size_t>(kk) * static_cast<size_t>(shape.N);
                    int8_t *dst = h_B_rowmajor_padded.data() + static_cast<size_t>(kk) * static_cast<size_t>(n_padded);
                    std::memcpy(dst, src, static_cast<size_t>(shape.N) * sizeof(int8_t));
                }

                std::vector<int8_t> h_B_rowmajor_padded_q4 = clipToQ4Signed(h_B_rowmajor_padded);
                std::vector<int8_t> h_B_vnni_padded = packRowMajorToVNNI(h_B_rowmajor_padded, n_padded, shape.K);
                std::vector<uint8_t> h_B_q4_vnni_padded = packRowMajorToQ4VNNI(h_B_rowmajor_padded_q4, n_padded, shape.K);

                std::vector<int32_t> cpu_ref_padded;
                std::vector<int32_t> cpu_ref_q4_padded;
                if (opt.check_correctness)
                {
                    cpu_ref_padded.assign(c_pad_elems, 0);
                    cpu_ref_q4_padded.assign(c_pad_elems, 0);
                    for (int mm = 0; mm < shape.M; ++mm)
                    {
                        const int32_t *src_ref = cpu_ref.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N);
                        int32_t *dst_ref = cpu_ref_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(n_padded);
                        std::memcpy(dst_ref, src_ref, static_cast<size_t>(shape.N) * sizeof(int32_t));

                        const int32_t *src_ref_q4 = cpu_ref_q4.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N);
                        int32_t *dst_ref_q4 = cpu_ref_q4_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(n_padded);
                        std::memcpy(dst_ref_q4, src_ref_q4, static_cast<size_t>(shape.N) * sizeof(int32_t));
                    }
                }

                int8_t *d_B_rowmajor_padded = nullptr;
                int8_t *d_B_vnni_padded = nullptr;
                uint8_t *d_B_q4_vnni_padded = nullptr;
                int32_t *d_C_padded = nullptr;
                hipMalloc(&d_B_rowmajor_padded, b_pad_elems * sizeof(int8_t));
                hipMalloc(&d_B_vnni_padded, h_B_vnni_padded.size() * sizeof(int8_t));
                hipMalloc(&d_B_q4_vnni_padded, h_B_q4_vnni_padded.size() * sizeof(uint8_t));
                hipMalloc(&d_C_padded, c_pad_elems * sizeof(int32_t));

                hipMemcpyAsync(d_B_rowmajor_padded, h_B_rowmajor_padded.data(), b_pad_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                hipMemcpyAsync(d_B_vnni_padded, h_B_vnni_padded.data(), h_B_vnni_padded.size() * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                hipMemcpyAsync(d_B_q4_vnni_padded, h_B_q4_vnni_padded.data(), h_B_q4_vnni_padded.size() * sizeof(uint8_t), hipMemcpyHostToDevice, stream);
                hipStreamSynchronize(stream);

                const double vecpad_repack_ms = 0.0;

                Shape padded_shape{shape.name + "_vecpad_mr4", m_padded, n_padded, shape.K};
                Result r = benchmarkStrategy(
                    padded_shape,
                    "wave64_cpt4_aligned_mr4_cap128",
                    1,
                    opt,
                    d_A_padded,
                    d_B_rowmajor_padded,
                    d_B_vnni_padded,
                    d_B_q4_vnni_padded,
                    d_C_padded,
                    stream,
                    cpu_ref_padded,
                    cpu_ref_q4_padded);

                r.shape = shape.name;
                r.strategy = "lmhead_vec_mr4_pad_cpt4";
                r.splitk = 1;
                r.tflops = tflopsFor(shape.M, shape.N, shape.K, r.ms);
                r.speedup_vs_ck = ck_ms / std::max(r.ms, 1e-9);
                r.repack_ms = vecpad_repack_ms;
                r.e2e_ms = r.ms + r.repack_ms;
                r.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(r.e2e_ms, 1e-9);
                if (canonical_ck)
                {
                    r.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(r.ms, 1e-9);
                    r.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(r.e2e_ms, 1e-9);
                }
                results_out.push_back(r);

                if (d_A_padded != d_A)
                {
                    hipFree(d_A_padded);
                }
                hipFree(d_B_rowmajor_padded);
                hipFree(d_B_vnni_padded);
                hipFree(d_B_q4_vnni_padded);
                hipFree(d_C_padded);
                continue;
            }

            if (strategy == "lmhead_vec_mr4_pad_cpt8" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_cap64" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_cap96" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_cap192" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_cap256" ||
                strategy == "lmhead_vec_mr8_pad_cpt4_abcast_lane0_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap64" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap96" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap128" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap192" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_i32addr_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_lb3_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_lb4_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_ldsB_kgt8_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_avec4_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coal_pipe_u4_cap256" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_pf1" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_cap128" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_lb1" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_lb1_cap128" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2_by8" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cpt16split" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_veca2" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_veca4" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_veca4_rfl" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_ldsbdb" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_lds_kgt16" ||
                strategy == "lmhead_vec_mr4_pad_cpt8_ldsbdb_kgt8")
            {
                const int kVectorRows =
                    (strategy == "lmhead_vec_mr8_pad_cpt4_abcast_lane0_cap256") ? 32 : 16; // BY(4) * MR(8|4)

                int kVectorCols =
                    (strategy == "lmhead_vec_mr8_pad_cpt4_abcast_lane0_cap256") ? 256 : 512; // 64 lanes * CPT(4|8)

                std::string padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap256";
                std::string padded_suffix = "_vecpad_mr4cpt8";
                if (strategy == "lmhead_vec_mr4_pad_cpt8_veca2")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_veca2_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_veca2";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_cap64")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_cap64";
                    padded_suffix = "_vecpad_mr4cpt8_cap64";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_cap96")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_cap96";
                    padded_suffix = "_vecpad_mr4cpt8_cap96";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_cap192")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_cap192";
                    padded_suffix = "_vecpad_mr4cpt8_cap192";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap64")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap64";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_cap64";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap96")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap96";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_cap96";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap128")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_cap128";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap192")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap192";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_cap192";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_cap256";
                }
                else if (strategy == "lmhead_vec_mr8_pad_cpt4_abcast_lane0_cap256")
                {
                    padded_strategy = "wave64_cpt4_aligned_mr8_abcast_lane0_cap256";
                    padded_suffix = "_vecpad_mr8cpt4_abcast_lane0_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coalonly_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_i32addr_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_i32addr_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coalonly_i32addr_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coalonly_splitacc_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_lb3_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_lb3_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coalonly_splitacc_lb3_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coalonly_splitacc_lb4_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coalonly_splitacc_lb4_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coalonly_splitacc_lb4_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_ldsB_kgt8_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_ldsB_kgt8_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_ldsB_kgt8_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_avec4_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_avec4_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_avec4_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_pf1")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_pf1_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_pf1";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_coal_pipe_u4_cap256")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_coal_pipe_u4_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_coal_pipe_u4_cap256";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_by8_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_by8";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_cap128")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_by8_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_by8_cap128";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_lb1")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_by8_lb1_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_by8_lb1";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_by8_lb1_cap128")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_by8_lb1_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_by8_lb1_cap128";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_k2_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_k2";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2_by8")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_k2_by8_cap256";
                    padded_suffix = "_vecpad_mr4cpt8_abcast_lane0_k2_by8";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_abcast_lane0_cpt16split")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_abcast_lane0_cpt16split_cap128";
                    padded_suffix = "_vecpad_mr4cpt16_abcast_lane0_split";
                    kVectorCols = 1024; // 64 lanes * CPT16
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_veca4")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_veca4_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_veca4";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_veca4_rfl")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_veca4_rfl_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_veca4rfl";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_ldsbdb")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_ldsbdb_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_ldsbdb";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_lds_kgt16")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_lds_kgt16_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_lds_kgt16";
                }
                else if (strategy == "lmhead_vec_mr4_pad_cpt8_ldsbdb_kgt8")
                {
                    padded_strategy = "wave64_cpt8_aligned_mr4_ldsbdb_kgt8_cap128";
                    padded_suffix = "_vecpad_mr4cpt8_ldsbdb_kgt8";
                }

                const int n_padded = roundUpToMultiple(shape.N, kVectorCols);
                const int m_padded = roundUpToMultiple(shape.M, kVectorRows);

                const size_t a_pad_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(shape.K);
                const size_t b_pad_elems = static_cast<size_t>(shape.K) * static_cast<size_t>(n_padded);
                const size_t c_pad_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(n_padded);

                std::vector<int8_t> h_A_padded;
                int8_t *d_A_padded = d_A;
                if (m_padded != shape.M)
                {
                    h_A_padded.assign(a_pad_elems, 0);
                    for (int mm = 0; mm < shape.M; ++mm)
                    {
                        const int8_t *src = h_A.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                        int8_t *dst = h_A_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                        std::memcpy(dst, src, static_cast<size_t>(shape.K) * sizeof(int8_t));
                    }
                    hipMalloc(&d_A_padded, a_pad_elems * sizeof(int8_t));
                    hipMemcpyAsync(d_A_padded, h_A_padded.data(), a_pad_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                }

                std::vector<int8_t> h_B_rowmajor_padded(b_pad_elems, 0);
                for (int kk = 0; kk < shape.K; ++kk)
                {
                    const int8_t *src = h_B_rowmajor.data() + static_cast<size_t>(kk) * static_cast<size_t>(shape.N);
                    int8_t *dst = h_B_rowmajor_padded.data() + static_cast<size_t>(kk) * static_cast<size_t>(n_padded);
                    std::memcpy(dst, src, static_cast<size_t>(shape.N) * sizeof(int8_t));
                }

                std::vector<int8_t> h_B_rowmajor_padded_q4 = clipToQ4Signed(h_B_rowmajor_padded);
                std::vector<int8_t> h_B_vnni_padded = packRowMajorToVNNI(h_B_rowmajor_padded, n_padded, shape.K);
                std::vector<uint8_t> h_B_q4_vnni_padded = packRowMajorToQ4VNNI(h_B_rowmajor_padded_q4, n_padded, shape.K);

                std::vector<int32_t> cpu_ref_padded;
                std::vector<int32_t> cpu_ref_q4_padded;
                if (opt.check_correctness)
                {
                    cpu_ref_padded.assign(c_pad_elems, 0);
                    cpu_ref_q4_padded.assign(c_pad_elems, 0);
                    for (int mm = 0; mm < shape.M; ++mm)
                    {
                        const int32_t *src_ref = cpu_ref.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N);
                        int32_t *dst_ref = cpu_ref_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(n_padded);
                        std::memcpy(dst_ref, src_ref, static_cast<size_t>(shape.N) * sizeof(int32_t));

                        const int32_t *src_ref_q4 = cpu_ref_q4.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N);
                        int32_t *dst_ref_q4 = cpu_ref_q4_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(n_padded);
                        std::memcpy(dst_ref_q4, src_ref_q4, static_cast<size_t>(shape.N) * sizeof(int32_t));
                    }
                }

                int8_t *d_B_rowmajor_padded = nullptr;
                int8_t *d_B_vnni_padded = nullptr;
                uint8_t *d_B_q4_vnni_padded = nullptr;
                int32_t *d_C_padded = nullptr;
                hipMalloc(&d_B_rowmajor_padded, b_pad_elems * sizeof(int8_t));
                hipMalloc(&d_B_vnni_padded, h_B_vnni_padded.size() * sizeof(int8_t));
                hipMalloc(&d_B_q4_vnni_padded, h_B_q4_vnni_padded.size() * sizeof(uint8_t));
                hipMalloc(&d_C_padded, c_pad_elems * sizeof(int32_t));

                hipMemcpyAsync(d_B_rowmajor_padded, h_B_rowmajor_padded.data(), b_pad_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                hipMemcpyAsync(d_B_vnni_padded, h_B_vnni_padded.data(), h_B_vnni_padded.size() * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                hipMemcpyAsync(d_B_q4_vnni_padded, h_B_q4_vnni_padded.data(), h_B_q4_vnni_padded.size() * sizeof(uint8_t), hipMemcpyHostToDevice, stream);
                hipStreamSynchronize(stream);

                const double vecpad_repack_ms = 0.0;

                Shape padded_shape{shape.name + padded_suffix, m_padded, n_padded, shape.K};
                Result r = benchmarkStrategy(
                    padded_shape,
                    padded_strategy,
                    1,
                    opt,
                    d_A_padded,
                    d_B_rowmajor_padded,
                    d_B_vnni_padded,
                    d_B_q4_vnni_padded,
                    d_C_padded,
                    stream,
                    cpu_ref_padded,
                    cpu_ref_q4_padded);

                r.shape = shape.name;
                r.strategy = strategy;
                r.splitk = 1;
                r.tflops = tflopsFor(shape.M, shape.N, shape.K, r.ms);
                r.speedup_vs_ck = ck_ms / std::max(r.ms, 1e-9);
                r.repack_ms = vecpad_repack_ms;
                r.e2e_ms = r.ms + r.repack_ms;
                r.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(r.e2e_ms, 1e-9);
                if (canonical_ck)
                {
                    r.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(r.ms, 1e-9);
                    r.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(r.e2e_ms, 1e-9);
                }
                results_out.push_back(r);

                if (d_A_padded != d_A)
                {
                    hipFree(d_A_padded);
                }
                hipFree(d_B_rowmajor_padded);
                hipFree(d_B_vnni_padded);
                hipFree(d_B_q4_vnni_padded);
                hipFree(d_C_padded);
                continue;
            }


            if (strategy == "cpt2_splitk" || strategy == "cpt2_sharedb_splitk" || strategy == "wave64_cpt4_splitk" || strategy == "wave64_cpt4_splitk_by8")
            {
                for (const int splitk : opt.splitk_values)
                {
                    if (splitk <= 1)
                    {
                        continue;
                    }
                    Result r = benchmarkStrategy(shape, strategy, splitk, opt, d_A, d_B_rowmajor, d_B_vnni, d_B_q4_vnni, d_C, stream, cpu_ref, cpu_ref_q4);
                    r.speedup_vs_ck = ck_ms / r.ms;
                    const bool is_q4nib =
                        (strategy == "q4nib_wave64_cpt4" || strategy == "q4nib_wave64_cpt4_grid" ||
                         strategy == "q4nib_lut_wave64_cpt4" || strategy == "q4nib_lut_wave64_cpt4_grid" ||
                         strategy == "q4nib_perm_wave64_cpt4" || strategy == "q4nib_perm_wave64_cpt4_grid");
                    r.repack_ms = is_q4nib ? custom_q4_repack_ms : custom_int8_repack_ms;
                    r.e2e_ms = r.ms + r.repack_ms;
                    r.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(r.e2e_ms, 1e-9);
                    if (canonical_ck)
                    {
                        r.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(r.ms, 1e-9);
                        r.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(r.e2e_ms, 1e-9);
                    }
                    results_out.push_back(r);
                }
            }
            else
            {
                Result r = benchmarkStrategy(shape, strategy, 1, opt, d_A, d_B_rowmajor, d_B_vnni, d_B_q4_vnni, d_C, stream, cpu_ref, cpu_ref_q4);
                r.speedup_vs_ck = ck_ms / r.ms;
                const bool is_q4nib =
                    (strategy == "q4nib_wave64_cpt4" || strategy == "q4nib_wave64_cpt4_grid" ||
                     strategy == "q4nib_lut_wave64_cpt4" || strategy == "q4nib_lut_wave64_cpt4_grid" ||
                     strategy == "q4nib_perm_wave64_cpt4" || strategy == "q4nib_perm_wave64_cpt4_grid");
                r.repack_ms = is_q4nib ? custom_q4_repack_ms : custom_int8_repack_ms;
                r.e2e_ms = r.ms + r.repack_ms;
                r.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(r.e2e_ms, 1e-9);
                if (canonical_ck)
                {
                    r.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(r.ms, 1e-9);
                    r.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(r.e2e_ms, 1e-9);
                }
                results_out.push_back(r);
            }
        }

        if (is_lm_head && !opt.prefer_ck_wide && opt.lm_head_chunked)
        {
            constexpr int kVectorCols = 256; // 64 lanes * CPT4
            constexpr int kVectorRows = 8;   // BY for aligned_unroll2_lb path
            const int chunk_n_raw = std::max(kVectorCols, std::min(opt.lm_head_chunk_n, shape.N));
            const int chunk_n = ((chunk_n_raw + kVectorCols - 1) / kVectorCols) * kVectorCols;
            if (chunk_n < shape.N)
            {
                const std::string chunk_base_strategy = "wave64_cpt4_aligned_unroll2_lb_cap128";
                if (shape.K % 4 != 0)
                {
                    std::cout << "[prefill-lab] device=" << device
                              << " LM-head chunked prototype skipped: vectorized alignment preconditions not met"
                              << " (K=" << shape.K << ")"
                              << std::endl;
                }
                else
                {
                    const int m_padded = ((shape.M + kVectorRows - 1) / kVectorRows) * kVectorRows;
                    int8_t *d_A_chunk = d_A;
                    if (m_padded != shape.M)
                    {
                        const size_t a_chunk_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(shape.K);
                        std::vector<int8_t> h_A_padded(a_chunk_elems, 0);
                        for (int mm = 0; mm < shape.M; ++mm)
                        {
                            const int8_t *src = h_A.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                            int8_t *dst = h_A_padded.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.K);
                            std::memcpy(dst, src, static_cast<size_t>(shape.K) * sizeof(int8_t));
                        }
                        hipMalloc(&d_A_chunk, a_chunk_elems * sizeof(int8_t));
                        hipMemcpyAsync(d_A_chunk, h_A_padded.data(), a_chunk_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                        hipStreamSynchronize(stream);
                    }

                    std::cout << "[prefill-lab] device=" << device
                              << " LM-head chunked prototype enforcing vectorized strategy=" << chunk_base_strategy
                              << " chunk_n=" << chunk_n
                              << " (padded to " << kVectorRows << "-row and " << kVectorCols << "-col alignment)"
                              << std::endl;

                    double chunk_kernel_ms = 0.0;
                    const double chunk_repack_ms = 0.0;
                    int chunk_max_abs_diff = 0;
                    bool chunk_ok = true;

                    for (int n0 = 0; n0 < shape.N; n0 += chunk_n)
                    {
                        const int chunk_cols_valid = std::min(chunk_n, shape.N - n0);
                        const int chunk_cols_padded = ((chunk_cols_valid + kVectorCols - 1) / kVectorCols) * kVectorCols;
                        const size_t b_chunk_elems = static_cast<size_t>(shape.K) * static_cast<size_t>(chunk_cols_padded);
                        const size_t c_chunk_elems = static_cast<size_t>(m_padded) * static_cast<size_t>(chunk_cols_padded);

                        std::vector<int8_t> h_B_rowmajor_chunk(b_chunk_elems, 0);
                        for (int kk = 0; kk < shape.K; ++kk)
                        {
                            const int8_t *src = h_B_rowmajor.data() + static_cast<size_t>(kk) * static_cast<size_t>(shape.N) + static_cast<size_t>(n0);
                            int8_t *dst = h_B_rowmajor_chunk.data() + static_cast<size_t>(kk) * static_cast<size_t>(chunk_cols_padded);
                            std::memcpy(dst, src, static_cast<size_t>(chunk_cols_valid) * sizeof(int8_t));
                        }

                        std::vector<int8_t> h_B_rowmajor_chunk_q4 = clipToQ4Signed(h_B_rowmajor_chunk);
                        std::vector<int8_t> h_B_vnni_chunk = packRowMajorToVNNI(h_B_rowmajor_chunk, chunk_cols_padded, shape.K);
                        std::vector<uint8_t> h_B_q4_vnni_chunk = packRowMajorToQ4VNNI(h_B_rowmajor_chunk_q4, chunk_cols_padded, shape.K);

                        std::vector<int32_t> cpu_ref_chunk(opt.check_correctness ? c_chunk_elems : 0, 0);
                        std::vector<int32_t> cpu_ref_q4_chunk(opt.check_correctness ? c_chunk_elems : 0, 0);
                        if (opt.check_correctness)
                        {
                            for (int mm = 0; mm < shape.M; ++mm)
                            {
                                const int32_t *src_ref = cpu_ref.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N) + static_cast<size_t>(n0);
                                int32_t *dst_ref = cpu_ref_chunk.data() + static_cast<size_t>(mm) * static_cast<size_t>(chunk_cols_padded);
                                std::memcpy(dst_ref, src_ref, static_cast<size_t>(chunk_cols_valid) * sizeof(int32_t));

                                const int32_t *src_ref_q4 = cpu_ref_q4.data() + static_cast<size_t>(mm) * static_cast<size_t>(shape.N) + static_cast<size_t>(n0);
                                int32_t *dst_ref_q4 = cpu_ref_q4_chunk.data() + static_cast<size_t>(mm) * static_cast<size_t>(chunk_cols_padded);
                                std::memcpy(dst_ref_q4, src_ref_q4, static_cast<size_t>(chunk_cols_valid) * sizeof(int32_t));
                            }
                        }

                        int8_t *d_B_rowmajor_chunk = nullptr;
                        int8_t *d_B_vnni_chunk = nullptr;
                        uint8_t *d_B_q4_vnni_chunk = nullptr;
                        int32_t *d_C_chunk = nullptr;
                        hipMalloc(&d_B_rowmajor_chunk, b_chunk_elems * sizeof(int8_t));
                        hipMalloc(&d_B_vnni_chunk, h_B_vnni_chunk.size() * sizeof(int8_t));
                        hipMalloc(&d_B_q4_vnni_chunk, h_B_q4_vnni_chunk.size() * sizeof(uint8_t));
                        hipMalloc(&d_C_chunk, c_chunk_elems * sizeof(int32_t));

                        hipMemcpyAsync(d_B_rowmajor_chunk, h_B_rowmajor_chunk.data(), b_chunk_elems * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                        hipMemcpyAsync(d_B_vnni_chunk, h_B_vnni_chunk.data(), h_B_vnni_chunk.size() * sizeof(int8_t), hipMemcpyHostToDevice, stream);
                        hipMemcpyAsync(d_B_q4_vnni_chunk, h_B_q4_vnni_chunk.data(), h_B_q4_vnni_chunk.size() * sizeof(uint8_t), hipMemcpyHostToDevice, stream);
                        hipStreamSynchronize(stream);

                        Shape chunk_shape{shape.name + "_chunk", m_padded, chunk_cols_padded, shape.K};
                        Result chunk_result = benchmarkStrategy(
                            chunk_shape,
                            chunk_base_strategy,
                            1,
                            opt,
                            d_A_chunk,
                            d_B_rowmajor_chunk,
                            d_B_vnni_chunk,
                            d_B_q4_vnni_chunk,
                            d_C_chunk,
                            stream,
                            cpu_ref_chunk,
                            cpu_ref_q4_chunk);

                        chunk_kernel_ms += chunk_result.ms;
                        chunk_max_abs_diff = std::max(chunk_max_abs_diff, chunk_result.max_abs_diff);
                        chunk_ok = chunk_ok && chunk_result.ok;

                        hipFree(d_B_rowmajor_chunk);
                        hipFree(d_B_vnni_chunk);
                        hipFree(d_B_q4_vnni_chunk);
                        hipFree(d_C_chunk);
                    }

                    if (d_A_chunk != d_A)
                    {
                        hipFree(d_A_chunk);
                    }

                    Result chunk_row;
                    chunk_row.shape = shape.name;
                    const int m_pad = m_padded - shape.M;
                    chunk_row.strategy = chunk_base_strategy + "_chunked_n" + std::to_string(chunk_n) + "_mpad" + std::to_string(m_pad);
                    chunk_row.splitk = 1;
                    chunk_row.ms = chunk_kernel_ms;
                    chunk_row.repack_ms = chunk_repack_ms;
                    chunk_row.e2e_ms = chunk_row.ms + chunk_row.repack_ms;
                    chunk_row.tflops = tflopsFor(shape.M, shape.N, shape.K, std::max(chunk_row.ms, 1e-9));
                    chunk_row.speedup_vs_ck = ck_ms / std::max(chunk_row.ms, 1e-9);
                    chunk_row.speedup_vs_ck_e2e = ck_e2e_compare_ms / std::max(chunk_row.e2e_ms, 1e-9);
                    if (canonical_ck)
                    {
                        chunk_row.speedup_vs_ck_canonical_k = canonical_ck->mean_ms / std::max(chunk_row.ms, 1e-9);
                        chunk_row.speedup_vs_ck_canonical_e2e = canonical_ck->mean_ms / std::max(chunk_row.e2e_ms, 1e-9);
                    }
                    chunk_row.max_abs_diff = chunk_max_abs_diff;
                    chunk_row.ok = chunk_ok;
                    results_out.push_back(chunk_row);

                    std::cout << "[prefill-lab] device=" << device
                              << " LM-head chunked prototype base=" << chunk_base_strategy
                              << " chunk_n=" << chunk_n
                              << " vsCK(E2E)=" << std::fixed << std::setprecision(3) << chunk_row.speedup_vs_ck_e2e
                              << std::endl;
                }
            }
        }

        const bool use_e2e_gate = is_wide_or_extrawide;
        double best_vs_ck = 0.0;
        std::string best_strategy = "ck_baseline";
        for (const auto &r : results_out)
        {
            if (r.shape != shape.name)
            {
                continue;
            }

            const double candidate_speedup = use_e2e_gate ? r.speedup_vs_ck_e2e : r.speedup_vs_ck;
            if (candidate_speedup >= best_vs_ck)
            {
                best_vs_ck = candidate_speedup;
                best_strategy = r.strategy;
            }
        }
        std::cout << "[prefill-lab] device=" << device
                  << " done shape=" << shape.name
                  << " best=" << best_strategy
                  << " " << (use_e2e_gate ? "vsCK(E2E)=" : "vsCK(K)=")
                  << std::fixed << std::setprecision(3) << best_vs_ck
                  << std::endl;

        rocmQuantGemm_destroyKernelContext(ck_ctx, device);
        hipFree(d_A);
        hipFree(d_B_rowmajor);
        if (d_B_nxk)
        {
            hipFree(d_B_nxk);
        }
        hipFree(d_B_vnni);
        hipFree(d_B_q4_vnni);
        hipFree(d_C);
    }

    hipStreamDestroy(stream);
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (isHelpArg(std::string(argv[i])))
        {
            printHelp(argv[0]);
            return 0;
        }
    }

    Options opt = parseOptions(argc, argv);

    if (opt.shapes.empty() || opt.strategies.empty())
    {
        std::cerr << "No shapes/strategies configured." << std::endl;
        return 1;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0)
    {
        std::cerr << "No ROCm device found." << std::endl;
        return 1;
    }
    std::vector<int> selected_devices;
    if (!opt.devices.empty())
    {
        selected_devices = opt.devices;
    }
    else if (opt.all_devices)
    {
        selected_devices.resize(static_cast<size_t>(device_count));
        std::iota(selected_devices.begin(), selected_devices.end(), 0);
    }
    else
    {
        selected_devices.push_back(opt.device);
    }

    for (const int d : selected_devices)
    {
        if (d < 0 || d >= device_count)
        {
            std::cerr << "Invalid device id " << d << ", available devices: " << device_count << std::endl;
            return 1;
        }
    }

    std::cout << "ROCm Prefill Strategy Lab"
              << " | devices=";
    for (size_t i = 0; i < selected_devices.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ",";
        }
        std::cout << selected_devices[i];
    }
    std::cout
              << " | warmup=" << opt.warmup
              << " | iters=" << opt.iters
              << " | correctness=" << (opt.check_correctness ? "on" : "off")
              << " | weight_repack_accounting=ck_only"
              << " | openmp=on"
              << " | omp_max_threads=" << omp_get_max_threads()
              << std::endl;

    std::vector<Result> all_results;

    std::vector<std::vector<Shape>> shards(selected_devices.size());
    std::vector<double> shard_load(selected_devices.size(), 0.0);
    std::vector<Shape> sorted_shapes = opt.shapes;
    std::sort(sorted_shapes.begin(), sorted_shapes.end(), [](const Shape &a, const Shape &b) {
        return shapeCost(a) > shapeCost(b);
    });
    for (const auto &shape : sorted_shapes)
    {
        const size_t idx = static_cast<size_t>(std::distance(shard_load.begin(), std::min_element(shard_load.begin(), shard_load.end())));
        shards[idx].push_back(shape);
        shard_load[idx] += shapeCost(shape);
    }

    std::vector<std::thread> workers;
    std::vector<std::vector<Result>> shard_results(selected_devices.size());
    std::vector<bool> shard_ok(selected_devices.size(), true);

    for (size_t i = 0; i < selected_devices.size(); ++i)
    {
        workers.emplace_back([&, i]() {
            shard_ok[i] = runShapesOnDevice(opt, selected_devices[i], shards[i], shard_results[i]);
        });
    }

    for (auto &t : workers)
    {
        t.join();
    }

    const bool worker_failed = std::any_of(shard_ok.begin(), shard_ok.end(), [](bool ok) {
        return !ok;
    });
    if (worker_failed)
    {
        return 1;
    }

    for (auto &v : shard_results)
    {
        all_results.insert(all_results.end(), v.begin(), v.end());
    }

    std::sort(all_results.begin(), all_results.end(), [](const Result &a, const Result &b) {
        if (a.shape != b.shape)
        {
            return a.shape < b.shape;
        }
        if (a.strategy != b.strategy)
        {
            return a.strategy < b.strategy;
        }
        return a.splitk < b.splitk;
    });

    printTable(all_results);
    const LeaderboardSnapshot snapshot = printE2EWinnersAndLeaderboard(all_results);
    persistLeaderboardOnDisk(opt, selected_devices, snapshot);

    const bool any_bad = std::any_of(all_results.begin(), all_results.end(), [](const Result &r) {
        return !r.ok;
    });

    return any_bad ? 2 : 0;
}
