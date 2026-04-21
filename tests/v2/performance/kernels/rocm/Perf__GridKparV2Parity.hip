/**
 * @file Perf__GridKparV2Parity.hip
 * @brief Bit-exact parity test: legacy grid_kpar kernel vs modular v2 kernel
 *
 * Verifies that gemv_grid_kpar_v2_t produces identical INT32 output to the
 * legacy gemv_grid_kpar_t for all parameter combinations where the v2 kernel
 * should be equivalent:
 *   - WAVES_PER_BLOCK = 1 (single wavefront, no intra-block K-split)
 *   - PERSISTENT_N = 1 (one N-tile per block, same grid geometry)
 *   - REDUCE = Atomic (same atomicAdd path)
 *   - Any UNROLL value (integer accumulation is associative)
 *   - KB = 0 (runtime kblocks, same as legacy)
 *
 * All tests compare INT32 accumulators — since sdot4 is exact integer math
 * and atomicAdd on INT32 is order-independent, the output must be bit-identical.
 *
 * Shapes tested: all Qwen2.5-7B Q8_0 decode projections (M=1).
 */

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "fort.hpp"

// ============================================================================
// Legacy kernel (verbatim copy from ROCmGemvKernel.hip)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_legacy_grid_kpar_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int32_t b_packed = *(b_row + kg * b_stride + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// V2 modular kernel (verbatim copy from ROCmGemvKernel.hip)
// ============================================================================
enum class GridKparReduce : int {
    Atomic = 0,
    BlockReduce = 1
};

namespace grid_kpar_detail {

template <int CPT>
__device__ inline void zero_acc(int32_t (&acc)[CPT])
{
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;
}

template <int CPT>
__device__ inline void atomic_store(int32_t* d_C, int n_base, int32_t (&acc)[CPT], int kblocks)
{
    #pragma unroll
    for (int c = 0; c < CPT; ++c) {
        if (kblocks == 1)
            d_C[n_base + c] = acc[c];
        else
            atomicAdd(&d_C[n_base + c], acc[c]);
    }
}

template <int CPT>
__device__ inline void reduce_block(
    int32_t* d_C,
    int n_base,
    int32_t (&acc)[CPT],
    int kblocks,
    int wave_id,
    int lane_n,
    int waves_per_block,
    int32_t* shared)
{
    const int base = ((wave_id * blockDim.x) + lane_n) * CPT;
    #pragma unroll
    for (int c = 0; c < CPT; ++c) shared[base + c] = acc[c];
    __syncthreads();

    if (wave_id == 0) {
        #pragma unroll
        for (int c = 0; c < CPT; ++c) {
            int32_t sum = 0;
            for (int w = 0; w < waves_per_block; ++w) {
                sum += shared[((w * blockDim.x) + lane_n) * CPT + c];
            }
            if (kblocks == 1)
                d_C[n_base + c] = sum;
            else
                atomicAdd(&d_C[n_base + c], sum);
        }
    }
}

} // namespace grid_kpar_detail

template <
    int TILE_N,
    int CPT,
    int UNROLL,
    int KB,
    int WAVES_PER_BLOCK,
    int PERSISTENT_N,
    GridKparReduce REDUCE>
__global__ void gemv_v2_grid_kpar_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks_runtime)
{
    static_assert(TILE_N % CPT == 0, "TILE_N must be divisible by CPT");
    static_assert(UNROLL >= 1, "UNROLL must be >= 1");
    static_assert(WAVES_PER_BLOCK >= 1, "WAVES_PER_BLOCK must be >= 1");
    static_assert(PERSISTENT_N >= 1, "PERSISTENT_N must be >= 1");

    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE <= 64, "THREADS_PER_WAVE must be <= 64");

    const int lane_n = threadIdx.x;
    const int wave_id = (WAVES_PER_BLOCK == 1) ? 0 : threadIdx.y;
    const int kblocks = (KB > 0) ? KB : kblocks_runtime;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_block = blockIdx.y;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if constexpr (WAVES_PER_BLOCK == 1 && PERSISTENT_N == 1 && REDUCE == GridKparReduce::Atomic) {
        const int n_base = blockIdx.x * TILE_N + lane_n * CPT;
        const bool active = (n_base < N);

        int32_t acc[CPT];
        grid_kpar_detail::zero_acc(acc);

        if (active) {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;
            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);

            if constexpr (UNROLL == 4) {
                #pragma unroll 4
                for (int kg = kg_start; kg < kg_end; ++kg) {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
                    if constexpr (CPT == 2) {
                        const int2 b_pair = *reinterpret_cast<const int2*>(
                            b_row + kg * b_stride);
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                    } else {
                        #pragma unroll
                        for (int c = 0; c < CPT; ++c) {
                            const int32_t b_packed = *(b_row + kg * b_stride + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                        }
                    }
                }
            } else {
                for (int kg = kg_start; kg < kg_end; kg += UNROLL) {
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        const int kg_u = kg + u;
                        if (kg_u >= kg_end) break;
                        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg_u * 4);
                        if constexpr (CPT == 2) {
                            const int2 b_pair = *reinterpret_cast<const int2*>(
                                b_row + kg_u * b_stride);
                            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                            acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                        } else {
                            #pragma unroll
                            for (int c = 0; c < CPT; ++c) {
                                const int32_t b_packed = *(b_row + kg_u * b_stride + c);
                                acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                            }
                        }
                    }
                }
            }

            grid_kpar_detail::atomic_store(d_C_int32, n_base, acc, kblocks);
        }
        return;
    }

    const int tile_group = blockIdx.x * PERSISTENT_N;

    extern __shared__ int32_t s_reduce[];

    if constexpr (PERSISTENT_N == 1) {
        const int tile_n = tile_group;
        const int n_base = tile_n * TILE_N + lane_n * CPT;
        const bool active = (n_base < N);

        int32_t acc[CPT];
        grid_kpar_detail::zero_acc(acc);

        if (active && k_start < K) {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;
            const int kg_total = kg_end - kg_start;
            const int kg_per_wave = (kg_total + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
            const int kg_wave_start = kg_start + wave_id * kg_per_wave;
            const int kg_wave_end = min(kg_wave_start + kg_per_wave, kg_end);

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);

            for (int kg = kg_wave_start; kg < kg_wave_end; kg += UNROLL) {
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    const int kg_u = kg + u;
                    if (kg_u >= kg_wave_end) break;
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg_u * 4);

                    if constexpr (CPT == 2) {
                        const int2 b_pair = *reinterpret_cast<const int2*>(
                            b_row + kg_u * b_stride);
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                    } else {
                        #pragma unroll
                        for (int c = 0; c < CPT; ++c) {
                            const int32_t b_packed = *(b_row + kg_u * b_stride + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                        }
                    }
                }
            }
        }

        if (active) {
            if constexpr (REDUCE == GridKparReduce::BlockReduce) {
                grid_kpar_detail::reduce_block(d_C_int32, n_base, acc, kblocks,
                    wave_id, lane_n, WAVES_PER_BLOCK, s_reduce);
            } else {
                grid_kpar_detail::atomic_store(d_C_int32, n_base, acc, kblocks);
            }
        }
        return;
    }

    for (int tile_offset = 0; tile_offset < PERSISTENT_N; ++tile_offset) {
        const int tile_n = tile_group + tile_offset;
        const int n_base = tile_n * TILE_N + lane_n * CPT;
        const bool active = (n_base < N);

        int32_t acc[CPT];
        grid_kpar_detail::zero_acc(acc);

        if (active && k_start < K) {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;
            const int kg_total = kg_end - kg_start;
            const int kg_per_wave = (kg_total + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
            const int kg_wave_start = kg_start + wave_id * kg_per_wave;
            const int kg_wave_end = min(kg_wave_start + kg_per_wave, kg_end);

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);

            for (int kg = kg_wave_start; kg < kg_wave_end; kg += UNROLL) {
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    const int kg_u = kg + u;
                    if (kg_u >= kg_wave_end) break;
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg_u * 4);

                    if constexpr (CPT == 2) {
                        const int2 b_pair = *reinterpret_cast<const int2*>(
                            b_row + kg_u * b_stride);
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                    } else {
                        #pragma unroll
                        for (int c = 0; c < CPT; ++c) {
                            const int32_t b_packed = *(b_row + kg_u * b_stride + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                        }
                    }
                }
            }
        }

        if (active) {
            if constexpr (REDUCE == GridKparReduce::BlockReduce) {
                grid_kpar_detail::reduce_block(d_C_int32, n_base, acc, kblocks,
                    wave_id, lane_n, WAVES_PER_BLOCK, s_reduce);
            } else {
                grid_kpar_detail::atomic_store(d_C_int32, n_base, acc, kblocks);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Test infrastructure
// ============================================================================

struct Shape {
    const char* name;
    int N;
    int K;
};

static const Shape kQwen7BShapes[] = {
    {"Q proj",     3584,  3584},
    {"K proj",      512,  3584},
    {"V proj",      512,  3584},
    {"Wo proj",    3584,  3584},
    {"FFN Gate",  18944,  3584},
    {"FFN Up",    18944,  3584},
    {"FFN Down",   3584, 18944},
    {"LM Head",  152064,  3584},
};

static const Shape kQwen05BShapes[] = {
    {"Q proj 0.5B",   896,  896},
    {"K proj 0.5B",   128,  896},
    {"V proj 0.5B",   128,  896},
    {"Wo proj 0.5B",  896,  896},
    {"FFN Gate 0.5B", 4864, 896},
    {"FFN Up 0.5B",   4864, 896},
    {"FFN Down 0.5B",  896, 4864},
};

static void packVnniWeights(const std::vector<int8_t>& row_major,
                            int N, int K,
                            std::vector<int8_t>& vnni_out)
{
    vnni_out.resize(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; k += 4) {
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < 4; ++i) {
                vnni_out[static_cast<size_t>((k / 4) * N + n) * 4 + i] =
                    row_major[static_cast<size_t>(k + i) * N + n];
            }
        }
    }
}

// ============================================================================
// Test fixture
// ============================================================================
class GridKparV2Parity : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No HIP device available";
        }
        hipSetDevice(0);
    }

    // Run legacy kernel and return output
    template <int TN, int CPT>
    std::vector<int32_t> runLegacy(
        const int8_t* d_A, const int8_t* d_B_vnni,
        int N, int K, int kb)
    {
        int32_t* d_C = nullptr;
        hipMalloc(&d_C, N * sizeof(int32_t));
        hipMemset(d_C, 0, N * sizeof(int32_t));

        const int grid_n = (N + TN - 1) / TN;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_legacy_grid_kpar_t<TN, CPT>),
            dim3(grid_n, kb), dim3(TN / CPT), 0, nullptr,
            d_A, d_B_vnni, d_C, N, K, kb);
        hipDeviceSynchronize();

        std::vector<int32_t> result(N);
        hipMemcpy(result.data(), d_C, N * sizeof(int32_t), hipMemcpyDeviceToHost);
        hipFree(d_C);
        return result;
    }

    // Run v2 kernel and return output
    template <int TN, int CPT, int UNROLL, int KB_TMPL, int WAVES, int PERSISTENT,
              GridKparReduce REDUCE>
    std::vector<int32_t> runV2(
        const int8_t* d_A, const int8_t* d_B_vnni,
        int N, int K, int kb_runtime)
    {
        int32_t* d_C = nullptr;
        hipMalloc(&d_C, N * sizeof(int32_t));
        hipMemset(d_C, 0, N * sizeof(int32_t));

        constexpr int threads_x = TN / CPT;
        const int grid_n = (N + (TN * PERSISTENT) - 1) / (TN * PERSISTENT);
        const int kblocks = (KB_TMPL > 0) ? KB_TMPL : kb_runtime;

        const size_t shared_bytes = (REDUCE == GridKparReduce::BlockReduce)
            ? static_cast<size_t>(threads_x) * WAVES * CPT * sizeof(int32_t)
            : 0;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME((gemv_v2_grid_kpar_t<TN, CPT, UNROLL, KB_TMPL, WAVES, PERSISTENT, REDUCE>)),
            dim3(grid_n, kblocks), dim3(threads_x, WAVES, 1),
            shared_bytes, nullptr,
            d_A, d_B_vnni, d_C, N, K, kb_runtime);
        hipDeviceSynchronize();

        std::vector<int32_t> result(N);
        hipMemcpy(result.data(), d_C, N * sizeof(int32_t), hipMemcpyDeviceToHost);
        hipFree(d_C);
        return result;
    }

    // Compare two output vectors, return number of mismatches
    int compareOutputs(const std::vector<int32_t>& legacy,
                       const std::vector<int32_t>& v2,
                       const char* shape_name,
                       const char* config_name)
    {
        EXPECT_EQ(legacy.size(), v2.size());
        int mismatches = 0;
        for (size_t i = 0; i < legacy.size(); ++i) {
            if (legacy[i] != v2[i]) {
                if (mismatches < 5) {
                    fprintf(stderr, "  MISMATCH [%s / %s] index %zu: legacy=%d v2=%d (delta=%d)\n",
                            shape_name, config_name,
                            i, legacy[i], v2[i], legacy[i] - v2[i]);
                }
                ++mismatches;
            }
        }
        return mismatches;
    }

    // Allocate device memory and pack VNNI weights for a shape
    struct DeviceBuffers {
        int8_t* d_A = nullptr;
        int8_t* d_B_vnni = nullptr;
        int N, K;

        ~DeviceBuffers() {
            if (d_A) hipFree(d_A);
            if (d_B_vnni) hipFree(d_B_vnni);
        }
    };

    std::unique_ptr<DeviceBuffers> allocateForShape(const Shape& shape, uint32_t seed = 42)
    {
        auto buf = std::make_unique<DeviceBuffers>();
        buf->N = shape.N;
        buf->K = shape.K;

        std::mt19937 rng(seed);
        std::vector<int8_t> h_A(shape.K);
        for (auto& v : h_A) v = static_cast<int8_t>(rng() % 255 - 127);

        std::vector<int8_t> h_B_row(static_cast<size_t>(shape.K) * shape.N);
        for (auto& v : h_B_row) v = static_cast<int8_t>(rng() % 255 - 127);

        std::vector<int8_t> h_B_vnni;
        packVnniWeights(h_B_row, shape.N, shape.K, h_B_vnni);

        hipMalloc(&buf->d_A, shape.K * sizeof(int8_t));
        hipMalloc(&buf->d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
        hipMemcpy(buf->d_A, h_A.data(), shape.K * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(buf->d_B_vnni, h_B_vnni.data(), h_B_vnni.size(), hipMemcpyHostToDevice);

        return buf;
    }
};

// ============================================================================
// Test: UNROLL sweep parity (WAVES=1, PERSISTENT_N=1, REDUCE=Atomic)
//
// The v2 kernel with any UNROLL value should produce bit-identical output
// to the legacy kernel when WAVES=1, PERSISTENT_N=1, REDUCE=Atomic.
// Integer accumulation is associative, so unroll factor doesn't affect results.
// ============================================================================

TEST_F(GridKparV2Parity, UnrollSweep_TN128_CPT2_AllShapes)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    for (const auto& shape : kQwen7BShapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        // Compute production KB
        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        // Legacy baseline
        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        // V2 with UNROLL=1
        {
            auto v2 = runV2<TN, CPT, 1, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            int m = compareOutputs(legacy, v2, shape.name, "UNROLL=1");
            EXPECT_EQ(m, 0) << shape.name << " UNROLL=1: " << m << " mismatches";
        }

        // V2 with UNROLL=2
        {
            auto v2 = runV2<TN, CPT, 2, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            int m = compareOutputs(legacy, v2, shape.name, "UNROLL=2");
            EXPECT_EQ(m, 0) << shape.name << " UNROLL=2: " << m << " mismatches";
        }

        // V2 with UNROLL=4 (matches legacy's #pragma unroll 4)
        {
            auto v2 = runV2<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            int m = compareOutputs(legacy, v2, shape.name, "UNROLL=4");
            EXPECT_EQ(m, 0) << shape.name << " UNROLL=4: " << m << " mismatches";
        }

        // V2 with UNROLL=8
        {
            auto v2 = runV2<TN, CPT, 8, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            int m = compareOutputs(legacy, v2, shape.name, "UNROLL=8");
            EXPECT_EQ(m, 0) << shape.name << " UNROLL=8: " << m << " mismatches";
        }

        // V2 with UNROLL=16
        {
            auto v2 = runV2<TN, CPT, 16, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            int m = compareOutputs(legacy, v2, shape.name, "UNROLL=16");
            EXPECT_EQ(m, 0) << shape.name << " UNROLL=16: " << m << " mismatches";
        }
    }
}

// ============================================================================
// Test: CPT=4 parity (covers the non-vectorized B load path)
// ============================================================================

TEST_F(GridKparV2Parity, UnrollSweep_TN128_CPT4_AllShapes)
{
    constexpr int TN = 128;
    constexpr int CPT = 4;

    for (const auto& shape : kQwen7BShapes) {
        if (shape.N % CPT != 0) continue;
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        for (int unroll : {1, 2, 4, 8}) {
            char tag[32];
            snprintf(tag, sizeof(tag), "UNROLL=%d", unroll);
            SCOPED_TRACE(tag);

            std::vector<int32_t> v2;
            if (unroll == 1)
                v2 = runV2<TN, CPT, 1, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else if (unroll == 2)
                v2 = runV2<TN, CPT, 2, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else if (unroll == 4)
                v2 = runV2<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else
                v2 = runV2<TN, CPT, 8, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

            int m = compareOutputs(legacy, v2, shape.name, tag);
            EXPECT_EQ(m, 0) << shape.name << " " << tag << ": " << m << " mismatches";
        }
    }
}

// ============================================================================
// Test: KB sweep parity — verify various KB values all produce same output
// ============================================================================

TEST_F(GridKparV2Parity, KBSweep_TN128_CPT2)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    // Test a subset of shapes with various KB values
    const Shape shapes[] = {
        {"Q proj",    3584,  3584},
        {"FFN Down",  3584, 18944},
        {"FFN Gate", 18944,  3584},
    };

    const int kb_values[] = {2, 4, 8, 16, 32, 56};

    for (const auto& shape : shapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        for (int kb : kb_values) {
            // Skip KB values where k_per < 4, which means < 1 VNNI group per block
            int k_per = (shape.K + kb - 1) / kb;
            if (k_per < 4) continue;

            char tag[32];
            snprintf(tag, sizeof(tag), "KB=%d", kb);
            SCOPED_TRACE(tag);

            auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            auto v2 = runV2<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>(
                buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

            int m = compareOutputs(legacy, v2, shape.name, tag);
            EXPECT_EQ(m, 0) << shape.name << " " << tag << ": " << m << " mismatches";
        }
    }
}

// ============================================================================
// Test: PERSISTENT_N parity — verify persistent tiling produces same output
//
// With PERSISTENT_N > 1, each block loops over multiple N-tiles. The output
// should be identical since each tile's computation is independent.
// ============================================================================

TEST_F(GridKparV2Parity, PersistentN_TN128_CPT2)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    for (const auto& shape : kQwen7BShapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        for (int persistent : {2, 4}) {
            char tag[32];
            snprintf(tag, sizeof(tag), "PERSISTENT_N=%d", persistent);
            SCOPED_TRACE(tag);

            std::vector<int32_t> v2;
            if (persistent == 2)
                v2 = runV2<TN, CPT, 4, 0, 1, 2, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else
                v2 = runV2<TN, CPT, 4, 0, 1, 4, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

            int m = compareOutputs(legacy, v2, shape.name, tag);
            EXPECT_EQ(m, 0) << shape.name << " " << tag << ": " << m << " mismatches";
        }
    }
}

// ============================================================================
// Test: WAVES_PER_BLOCK + BlockReduce parity
//
// With WAVES=2, the K-segment is split across 2 wavefronts within the block
// and reduced via shared memory. The final atomicAdd accumulates the same
// total, so output should be identical.
// ============================================================================

TEST_F(GridKparV2Parity, WavesBlockReduce_TN128_CPT2)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    for (const auto& shape : kQwen7BShapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        // Ensure each wave has at least 1 k-group to process
        int k_per = ((shape.K + kb - 1) / kb + 3) / 4 * 4;
        int kg_per_block = k_per / 4;
        if (kg_per_block < 2) continue;  // Need at least 2 kgroups for 2 waves

        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        // V2 with WAVES=2, BlockReduce
        auto v2 = runV2<TN, CPT, 4, 0, 2, 1, GridKparReduce::BlockReduce>(
            buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        int m = compareOutputs(legacy, v2, shape.name, "WAVES=2/BlockReduce");
        EXPECT_EQ(m, 0) << shape.name << " WAVES=2/BlockReduce: " << m << " mismatches";
    }
}

// ============================================================================
// Test: Qwen 0.5B shapes (smaller dimensions, edge cases)
// ============================================================================

TEST_F(GridKparV2Parity, Qwen05B_UnrollSweep_TN128_CPT2)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    for (const auto& shape : kQwen05BShapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        for (int unroll : {1, 4, 8}) {
            char tag[32];
            snprintf(tag, sizeof(tag), "UNROLL=%d", unroll);
            SCOPED_TRACE(tag);

            std::vector<int32_t> v2;
            if (unroll == 1)
                v2 = runV2<TN, CPT, 1, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else if (unroll == 4)
                v2 = runV2<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);
            else
                v2 = runV2<TN, CPT, 8, 0, 1, 1, GridKparReduce::Atomic>(
                    buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

            int m = compareOutputs(legacy, v2, shape.name, tag);
            EXPECT_EQ(m, 0) << shape.name << " " << tag << ": " << m << " mismatches";
        }
    }
}

// ============================================================================
// Test: Combined levers — all five together
//
// Test with UNROLL=8, WAVES=2, PERSISTENT_N=2, BlockReduce simultaneously.
// This exercises the most "different" v2 configuration against the legacy kernel.
// ============================================================================

TEST_F(GridKparV2Parity, CombinedLevers_AllFive)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;

    // Use a shape subset with enough N and K for PERSISTENT_N=2 + WAVES=2
    const Shape shapes[] = {
        {"Q proj",    3584,  3584},
        {"FFN Gate", 18944,  3584},
        {"FFN Down",  3584, 18944},
    };

    for (const auto& shape : shapes) {
        SCOPED_TRACE(shape.name);
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        auto legacy = runLegacy<TN, CPT>(buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        // UNROLL=8, WAVES=2, PERSISTENT_N=2, BlockReduce
        auto v2 = runV2<TN, CPT, 8, 0, 2, 2, GridKparReduce::BlockReduce>(
            buf->d_A, buf->d_B_vnni, shape.N, shape.K, kb);

        int m = compareOutputs(legacy, v2, shape.name, "UNROLL=8/WAVES=2/PERSIST=2/BLOCK");
        EXPECT_EQ(m, 0) << shape.name
                        << " UNROLL=8/WAVES=2/PERSIST=2/BlockReduce: "
                        << m << " mismatches";
    }
}

// ============================================================================
// Performance: Legacy vs V2 throughput comparison
//
// Benchmarks both kernels with identical parameters (UNROLL=4, WAVES=1,
// PERSISTENT_N=1, REDUCE=Atomic) to verify no performance regression.
// Prints GB/s and relative overhead for each Qwen 7B shape.
// ============================================================================

TEST_F(GridKparV2Parity, PerformanceComparison_LegacyVsV2)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;
    constexpr int WARMUP = 50;
    constexpr int ITERS  = 500;

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Shape" << "N" << "K" << "Legacy GB/s" << "V2 GB/s" << "Delta %"
          << fort::endr;
    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::right);
    table.column(2).set_cell_text_align(fort::text_align::right);
    table.column(3).set_cell_text_align(fort::text_align::right);
    table.column(4).set_cell_text_align(fort::text_align::right);
    table.column(5).set_cell_text_align(fort::text_align::right);

    auto fmt1 = [](double v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1f", v);
        return std::string(buf);
    };
    auto fmtDelta = [](double v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%+.2f%%", v);
        return std::string(buf);
    };

    for (const auto& shape : kQwen7BShapes) {
        auto buf = allocateForShape(shape);

        const int k_groups = shape.K / 4;
        int kb;
        if (shape.K > 8192) {
            kb = std::max(4, std::min(32, shape.K / 512));
        } else {
            const int grid_n = (shape.N + TN - 1) / TN;
            int kb_for_kgroups = std::max(2, k_groups / 64);
            int kb_for_occupancy = std::max(2, (600 + grid_n - 1) / grid_n);
            int kb_max = std::max(2, k_groups / 16);
            kb = std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        // Pre-allocate persistent output buffers
        int32_t* d_C_legacy = nullptr;
        int32_t* d_C_v2 = nullptr;
        hipMalloc(&d_C_legacy, shape.N * sizeof(int32_t));
        hipMalloc(&d_C_v2, shape.N * sizeof(int32_t));

        const int grid_n = (shape.N + TN - 1) / TN;
        constexpr int threads_x = TN / CPT;

        // --- Legacy benchmark ---
        for (int i = 0; i < WARMUP; ++i) {
            hipMemset(d_C_legacy, 0, shape.N * sizeof(int32_t));
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_legacy_grid_kpar_t<TN, CPT>),
                dim3(grid_n, kb), dim3(threads_x), 0, nullptr,
                buf->d_A, buf->d_B_vnni, d_C_legacy, shape.N, shape.K, kb);
        }
        hipDeviceSynchronize();

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start);
        for (int i = 0; i < ITERS; ++i) {
            hipMemset(d_C_legacy, 0, shape.N * sizeof(int32_t));
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_legacy_grid_kpar_t<TN, CPT>),
                dim3(grid_n, kb), dim3(threads_x), 0, nullptr,
                buf->d_A, buf->d_B_vnni, d_C_legacy, shape.N, shape.K, kb);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float legacy_ms = 0.0f;
        hipEventElapsedTime(&legacy_ms, start, stop);
        double legacy_avg_us = (legacy_ms * 1000.0) / ITERS;

        // --- V2 benchmark (UNROLL=4, same as legacy's #pragma unroll 4) ---
        for (int i = 0; i < WARMUP; ++i) {
            hipMemset(d_C_v2, 0, shape.N * sizeof(int32_t));
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gemv_v2_grid_kpar_t<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>)),
                dim3(grid_n, kb), dim3(threads_x, 1, 1), 0, nullptr,
                buf->d_A, buf->d_B_vnni, d_C_v2, shape.N, shape.K, kb);
        }
        hipDeviceSynchronize();

        hipEventRecord(start);
        for (int i = 0; i < ITERS; ++i) {
            hipMemset(d_C_v2, 0, shape.N * sizeof(int32_t));
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME((gemv_v2_grid_kpar_t<TN, CPT, 4, 0, 1, 1, GridKparReduce::Atomic>)),
                dim3(grid_n, kb), dim3(threads_x, 1, 1), 0, nullptr,
                buf->d_A, buf->d_B_vnni, d_C_v2, shape.N, shape.K, kb);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float v2_ms = 0.0f;
        hipEventElapsedTime(&v2_ms, start, stop);
        double v2_avg_us = (v2_ms * 1000.0) / ITERS;

        // Compute GB/s (weight matrix dominates: N*K bytes read per call)
        double bytes = static_cast<double>(shape.N) * shape.K;
        double legacy_gbs = (bytes / (legacy_avg_us * 1e-6)) / 1e9;
        double v2_gbs = (bytes / (v2_avg_us * 1e-6)) / 1e9;
        double delta_pct = ((v2_gbs - legacy_gbs) / legacy_gbs) * 100.0;

          table << shape.name
              << shape.N
              << shape.K
              << fmt1(legacy_gbs)
              << fmt1(v2_gbs)
              << fmtDelta(delta_pct)
              << fort::endr;

        // Allow up to 5% regression (compiler differences in template expansion)
        EXPECT_GT(v2_gbs, legacy_gbs * 0.95)
            << shape.name << ": v2 more than 5% slower ("
            << v2_gbs << " vs " << legacy_gbs << " GB/s)";

        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipFree(d_C_legacy);
        hipFree(d_C_v2);
    }

    std::cout << "\n" << table.to_string();
    std::cout << "\n  UNROLL=4, WAVES=1, PERSISTENT_N=1, REDUCE=Atomic (identical to legacy path)\n";
    std::cout << "  " << WARMUP << " warmup + " << ITERS << " timed iterations per shape\n\n";
}
