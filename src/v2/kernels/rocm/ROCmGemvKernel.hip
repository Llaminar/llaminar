/**
 * @file ROCmGemvKernel.hip
 * @brief Bandwidth-optimized INT8 GEMV kernel for decode (M=1) on MI50/MI60 (gfx906)
 *
 * ## Problem
 *
 * CK INT8 GEMM at M=1 achieves only 2.4 TFLOPS (~8% of MI50's 29.5 TOPS INT8),
 * because 32×32 tiles waste 31 of 32 rows and can't fill the GPU. Decode is
 * memory-bandwidth-bound (reading all 7B weights per token), not compute-bound.
 *
 * ## Solution: Direct FP32×INT8 GEMV
 *
 * Skip activation quantization entirely. Read INT8 weights, multiply with FP32
 * activations directly, scale by per-column weight scale only.
 *
 * For M=1:  output[n] = scale_B[n] * sum_k( A_fp32[k] * B_int8[k * N + n] )
 *
 * ## Memory Layout
 *
 * Weights are [K × N] RowMajor INT8 (same as CK's packed format).
 * For GEMV we want to stream through K (outer dim), accessing N elements per row.
 *
 * ## Kernel Design (gfx906-optimized)
 *
 * - Each workgroup handles a tile of output columns [n_start, n_start + TILE_N)
 * - Each thread accumulates one output element across all K
 * - K-reduction done in registers (no shared memory needed for M=1)
 * - Uses vectorized loads (int4 = 16 bytes = 16 INT8 values) for coalesced access
 * - MI50 has 64 CUs, 1000 GB/s HBM2 bandwidth
 *
 * ## Expected Performance
 *
 * Weights: 7 GB INT8. At 480 GB/s → ~14.6ms theoretical minimum.
 * With activation reads + scale reads + output writes → ~16-18ms realistic.
 * Current CK: ~113ms. Target: ~15-20ms (6-7× speedup).
 *
 * @author Llaminar V2
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <mutex>
#include <atomic>

#include "../../backends/rocm/HipDeviceGuard.h"

// ============================================================================
// Kernel: gemv_int8_fp32
//
// Each thread computes one output element: output[n] = scale_B[n] * dot(A, B[:, n])
//
// Grid: (ceil(N / TILE_N), 1, 1)
// Block: (TILE_N, 1, 1)
//
// Uses K-reduction in registers with vectorized INT8 loads.
// ============================================================================

// Tile width: number of output columns per workgroup
// 256 threads per block — each thread does one output column
constexpr int GEMV_TILE_N = 256;

// VNNI kernel tile parameters
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_WIDE2_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_SQUARE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_KPAR_TILE_K = 256;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_CPT = 2;  // 64 threads = 1 full wavefront (was 4 = 32 threads = half wavefront)

constexpr uint8_t RATIO_VNNI_CODEBOOK_LINEAR = 0;
constexpr uint8_t RATIO_VNNI_CODEBOOK_IQ4 = 4;

static std::atomic<int> g_ratio_linear_tn_override{-1};
static std::atomic<int> g_ratio_linear_kb_override{-1};
static std::atomic<int> g_ratio_iq4_tn_override{-1};
static std::atomic<int> g_ratio_iq4_kb_override{-1};

__device__ __constant__ int8_t k_ratio_vnni_iq4_lut_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

__device__ __constant__ uint32_t k_ratio_iq4_perm_lut_words[1024];
__device__ __constant__ uint32_t k_ratio_linear_perm_lut_words[1024];

static std::array<uint32_t, 1024> build_ratio_iq4_perm_lut_table_host()
{
    std::array<uint32_t, 1024> table{};
    for (int rb = 0; rb < 256; ++rb)
    {
        const int8_t ratio = static_cast<int8_t>(rb);
        const int base = rb * 4;

        uint32_t lut0 = 0;
        uint32_t lut1 = 0;
        uint32_t lut2 = 0;
        uint32_t lut3 = 0;

        for (int i = 0; i < 4; ++i)
        {
            const int8_t wi_lo_0 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8[i]) + 64) >> 7);
            const int8_t wi_lo_1 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8[i + 4]) + 64) >> 7);
            const int8_t wi_hi_0 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8[i + 8]) + 64) >> 7);
            const int8_t wi_hi_1 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8[i + 12]) + 64) >> 7);

            lut0 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_lo_0)) << (i * 8));
            lut1 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_lo_1)) << (i * 8));
            lut2 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_hi_0)) << (i * 8));
            lut3 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_hi_1)) << (i * 8));
        }

        table[base + 0] = lut0;
        table[base + 1] = lut1;
        table[base + 2] = lut2;
        table[base + 3] = lut3;
    }
    return table;
}

static std::array<uint32_t, 1024> build_ratio_linear_perm_lut_table_host()
{
    std::array<uint32_t, 1024> table{};
    for (int rb = 0; rb < 256; ++rb)
    {
        const int8_t ratio = static_cast<int8_t>(rb);
        const int base = rb * 4;

        uint32_t lut0 = 0;
        uint32_t lut1 = 0;
        uint32_t lut2 = 0;
        uint32_t lut3 = 0;

        for (int i = 0; i < 4; ++i)
        {
            const int8_t wi_lo_0 = static_cast<int8_t>((static_cast<int>(ratio) * (i - 8) + 64) >> 7);
            const int8_t wi_lo_1 = static_cast<int8_t>((static_cast<int>(ratio) * (i - 4) + 64) >> 7);
            const int8_t wi_hi_0 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 0) + 64) >> 7);
            const int8_t wi_hi_1 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 4) + 64) >> 7);

            lut0 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_lo_0)) << (i * 8));
            lut1 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_lo_1)) << (i * 8));
            lut2 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_hi_0)) << (i * 8));
            lut3 |= (static_cast<uint32_t>(static_cast<uint8_t>(wi_hi_1)) << (i * 8));
        }

        table[base + 0] = lut0;
        table[base + 1] = lut1;
        table[base + 2] = lut2;
        table[base + 3] = lut3;
    }
    return table;
}

static bool ensure_ratio_iq4_perm_lut_table_on_device(int device_id)
{
    static std::mutex init_mutex;
    static bool initialized[16] = {false};
    static const std::array<uint32_t, 1024> host_table = build_ratio_iq4_perm_lut_table_host();

    const int did = (device_id >= 0 && device_id < 16) ? device_id : 0;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized[did])
    {
        return true;
    }

    hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(k_ratio_iq4_perm_lut_words),
                                       host_table.data(),
                                       host_table.size() * sizeof(uint32_t),
                                       0,
                                       hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Failed to init mode3 LUT table on device %d: %s\n",
                did, hipGetErrorString(err));
        return false;
    }

    initialized[did] = true;
    return true;
}

static bool ensure_ratio_linear_perm_lut_table_on_device(int device_id)
{
    static std::mutex init_mutex;
    static bool initialized[16] = {false};
    static const std::array<uint32_t, 1024> host_table = build_ratio_linear_perm_lut_table_host();

    const int did = (device_id >= 0 && device_id < 16) ? device_id : 0;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized[did])
    {
        return true;
    }

    hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(k_ratio_linear_perm_lut_words),
                                       host_table.data(),
                                       host_table.size() * sizeof(uint32_t),
                                       0,
                                       hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Failed to init linear mode3 LUT table on device %d: %s\n",
                did, hipGetErrorString(err));
        return false;
    }

    initialized[did] = true;
    return true;
}

template <bool IQ4>
__device__ __forceinline__ int32_t decode_q4_pack4_low(
    uint8_t q0,
    uint8_t q1,
    uint8_t q2,
    uint8_t q3,
    int8_t ratio)
{
    const uint8_t i0 = static_cast<uint8_t>(q0 & 0x0F);
    const uint8_t i1 = static_cast<uint8_t>(q1 & 0x0F);
    const uint8_t i2 = static_cast<uint8_t>(q2 & 0x0F);
    const uint8_t i3 = static_cast<uint8_t>(q3 & 0x0F);

    const int d0 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i0]) : (static_cast<int>(i0) - 8);
    const int d1 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i1]) : (static_cast<int>(i1) - 8);
    const int d2 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i2]) : (static_cast<int>(i2) - 8);
    const int d3 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i3]) : (static_cast<int>(i3) - 8);

    const int8_t w0 = static_cast<int8_t>((static_cast<int>(ratio) * d0 + 64) >> 7);
    const int8_t w1 = static_cast<int8_t>((static_cast<int>(ratio) * d1 + 64) >> 7);
    const int8_t w2 = static_cast<int8_t>((static_cast<int>(ratio) * d2 + 64) >> 7);
    const int8_t w3 = static_cast<int8_t>((static_cast<int>(ratio) * d3 + 64) >> 7);

    uint32_t packed = 0;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(w0));
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w1)) << 8);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w2)) << 16);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w3)) << 24);
    return static_cast<int32_t>(packed);
}

template <bool IQ4>
__device__ __forceinline__ int32_t decode_q4_pack4_high(
    uint8_t q0,
    uint8_t q1,
    uint8_t q2,
    uint8_t q3,
    int8_t ratio)
{
    const uint8_t i0 = static_cast<uint8_t>(q0 >> 4);
    const uint8_t i1 = static_cast<uint8_t>(q1 >> 4);
    const uint8_t i2 = static_cast<uint8_t>(q2 >> 4);
    const uint8_t i3 = static_cast<uint8_t>(q3 >> 4);

    const int d0 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i0]) : (static_cast<int>(i0) - 8);
    const int d1 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i1]) : (static_cast<int>(i1) - 8);
    const int d2 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i2]) : (static_cast<int>(i2) - 8);
    const int d3 = IQ4 ? static_cast<int>(k_ratio_vnni_iq4_lut_i8[i3]) : (static_cast<int>(i3) - 8);

    const int8_t w0 = static_cast<int8_t>((static_cast<int>(ratio) * d0 + 64) >> 7);
    const int8_t w1 = static_cast<int8_t>((static_cast<int>(ratio) * d1 + 64) >> 7);
    const int8_t w2 = static_cast<int8_t>((static_cast<int>(ratio) * d2 + 64) >> 7);
    const int8_t w3 = static_cast<int8_t>((static_cast<int>(ratio) * d3 + 64) >> 7);

    uint32_t packed = 0;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(w0));
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w1)) << 8);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w2)) << 16);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w3)) << 24);
    return static_cast<int32_t>(packed);
}

template <bool IQ4>
__device__ __forceinline__ int32_t decode_q4_group_to_packed_i8(
    const uint8_t* payload16,
    int group,
    int8_t ratio)
{
    const int byte_base = (group & 3) * 4;
    const bool high = (group >= 4);

    const uint8_t q0 = payload16[byte_base + 0];
    const uint8_t q1 = payload16[byte_base + 1];
    const uint8_t q2 = payload16[byte_base + 2];
    const uint8_t q3 = payload16[byte_base + 3];

    if (high)
    {
        return decode_q4_pack4_high<IQ4>(q0, q1, q2, q3, ratio);
    }
    return decode_q4_pack4_low<IQ4>(q0, q1, q2, q3, ratio);
}

template <bool IQ4>
__device__ __forceinline__ void decode_q4_block8_to_packed_i8(
    const uint8_t* payload16,
    int8_t ratio,
    int32_t out[8])
{
    const uint4 packed = *reinterpret_cast<const uint4*>(payload16);
    const uint32_t p0 = packed.x;
    const uint32_t p1 = packed.y;
    const uint32_t p2 = packed.z;
    const uint32_t p3 = packed.w;

    const uint8_t q0 = static_cast<uint8_t>(p0 & 0xFFu);
    const uint8_t q1 = static_cast<uint8_t>((p0 >> 8) & 0xFFu);
    const uint8_t q2 = static_cast<uint8_t>((p0 >> 16) & 0xFFu);
    const uint8_t q3 = static_cast<uint8_t>((p0 >> 24) & 0xFFu);
    const uint8_t q4 = static_cast<uint8_t>(p1 & 0xFFu);
    const uint8_t q5 = static_cast<uint8_t>((p1 >> 8) & 0xFFu);
    const uint8_t q6 = static_cast<uint8_t>((p1 >> 16) & 0xFFu);
    const uint8_t q7 = static_cast<uint8_t>((p1 >> 24) & 0xFFu);
    const uint8_t q8 = static_cast<uint8_t>(p2 & 0xFFu);
    const uint8_t q9 = static_cast<uint8_t>((p2 >> 8) & 0xFFu);
    const uint8_t q10 = static_cast<uint8_t>((p2 >> 16) & 0xFFu);
    const uint8_t q11 = static_cast<uint8_t>((p2 >> 24) & 0xFFu);
    const uint8_t q12 = static_cast<uint8_t>(p3 & 0xFFu);
    const uint8_t q13 = static_cast<uint8_t>((p3 >> 8) & 0xFFu);
    const uint8_t q14 = static_cast<uint8_t>((p3 >> 16) & 0xFFu);
    const uint8_t q15 = static_cast<uint8_t>((p3 >> 24) & 0xFFu);

    out[0] = decode_q4_pack4_low<IQ4>(q0, q1, q2, q3, ratio);
    out[1] = decode_q4_pack4_low<IQ4>(q4, q5, q6, q7, ratio);
    out[2] = decode_q4_pack4_low<IQ4>(q8, q9, q10, q11, ratio);
    out[3] = decode_q4_pack4_low<IQ4>(q12, q13, q14, q15, ratio);
    out[4] = decode_q4_pack4_high<IQ4>(q0, q1, q2, q3, ratio);
    out[5] = decode_q4_pack4_high<IQ4>(q4, q5, q6, q7, ratio);
    out[6] = decode_q4_pack4_high<IQ4>(q8, q9, q10, q11, ratio);
    out[7] = decode_q4_pack4_high<IQ4>(q12, q13, q14, q15, ratio);
}


__device__ __forceinline__ void decode_q4_low_high_pair_to_packed_i8_with_perm_lut16(
    uint32_t p,
    uint32_t lut0,
    uint32_t lut1,
    uint32_t lut2,
    uint32_t lut3,
    int32_t& out_low,
    int32_t& out_high)
{
    const uint32_t q_even = p;
    const uint32_t q_odd = p >> 4;

    const uint32_t sel_even = q_even & 0x07070707u;
    const uint32_t sel_odd = q_odd & 0x07070707u;

    const uint32_t v_even_low = __builtin_amdgcn_perm(lut1, lut0, sel_even);
    const uint32_t v_odd_low = __builtin_amdgcn_perm(lut1, lut0, sel_odd);
    const uint32_t v_even_high = __builtin_amdgcn_perm(lut3, lut2, sel_even);
    const uint32_t v_odd_high = __builtin_amdgcn_perm(lut3, lut2, sel_odd);

    uint32_t b3e = (q_even >> 3) & 0x01010101u;
    uint32_t me = b3e;
    me |= me << 1;
    me |= me << 2;
    me |= me << 4;

    uint32_t b3o = (q_odd >> 3) & 0x01010101u;
    uint32_t mo = b3o;
    mo |= mo << 1;
    mo |= mo << 2;
    mo |= mo << 4;

    out_low = static_cast<int32_t>((v_even_high & me) | (v_even_low & ~me));
    out_high = static_cast<int32_t>((v_odd_high & mo) | (v_odd_low & ~mo));
}

__device__ __forceinline__ void decode_q4_two_pairs_to_packed_i8_with_perm_lut16(
    uint32_t p0,
    uint32_t p1,
    uint32_t lut0,
    uint32_t lut1,
    uint32_t lut2,
    uint32_t lut3,
    int32_t& out0_low,
    int32_t& out0_high,
    int32_t& out1_low,
    int32_t& out1_high)
{
    decode_q4_low_high_pair_to_packed_i8_with_perm_lut16(p0, lut0, lut1, lut2, lut3, out0_low, out0_high);
    decode_q4_low_high_pair_to_packed_i8_with_perm_lut16(p1, lut0, lut1, lut2, lut3, out1_low, out1_high);
}


__device__ __forceinline__ void fetch_iq4_perm_lut_words_from_ratio_table(
    int8_t ratio,
    uint32_t& lut0,
    uint32_t& lut1,
    uint32_t& lut2,
    uint32_t& lut3)
{
    const uint32_t idx = static_cast<uint32_t>(static_cast<uint8_t>(ratio)) * 4u;
    lut0 = k_ratio_iq4_perm_lut_words[idx + 0];
    lut1 = k_ratio_iq4_perm_lut_words[idx + 1];
    lut2 = k_ratio_iq4_perm_lut_words[idx + 2];
    lut3 = k_ratio_iq4_perm_lut_words[idx + 3];
}

__device__ __forceinline__ void fetch_linear_perm_lut_words_from_ratio_table(
    int8_t ratio,
    uint32_t& lut0,
    uint32_t& lut1,
    uint32_t& lut2,
    uint32_t& lut3)
{
    const uint32_t idx = static_cast<uint32_t>(static_cast<uint8_t>(ratio)) * 4u;
    lut0 = k_ratio_linear_perm_lut_words[idx + 0];
    lut1 = k_ratio_linear_perm_lut_words[idx + 1];
    lut2 = k_ratio_linear_perm_lut_words[idx + 2];
    lut3 = k_ratio_linear_perm_lut_words[idx + 3];
}

__global__ void gemv_int8_int8_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    d_C_int32[n] = acc;
}

// Vectorized variant: each thread handles 4 columns (N must be divisible by 4)
__global__ void gemv_int8_int8_wide_vec4_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_INT8_VNNI_WIDE_TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        // Load 4 INT8 weights for each of 4 columns (VNNI = already packed)
        const int32_t p0 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 0)) * 4);
        const int32_t p1 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 1)) * 4);
        const int32_t p2 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 2)) * 4);
        const int32_t p3 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 3)) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc.x = __builtin_amdgcn_sdot4(a_packed, p0, acc.x, false);
        acc.y = __builtin_amdgcn_sdot4(a_packed, p1, acc.y, false);
        acc.z = __builtin_amdgcn_sdot4(a_packed, p2, acc.z, false);
        acc.w = __builtin_amdgcn_sdot4(a_packed, p3, acc.w, false);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// Wide-N kernel: 2 columns per thread
__global__ void gemv_int8_int8_wide2_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_INT8_VNNI_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        // Load 4 INT8 weights for each of 2 columns (VNNI = already packed)
        const int32_t p0 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const int32_t p1 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc0 = __builtin_amdgcn_sdot4(a_packed, p0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, p1, acc1, false);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
__global__ void gemv_int8_int8_square_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = 0; kg < tile_groups; ++kg)
        {
            // Load 4 INT8 activations from shared memory as packed int32
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);

            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);

            // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
            acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

// ============================================================================
// Grid-level K-parallel VNNI kernel with atomicAdd reduction
//
// Each block processes a TILE_N/CPT-thread tile × one K-segment.
// Grid: (ceil(N/TILE_N), KB) where KB splits K across blockIdx.y.
// Output MUST be zeroed before launch (atomicAdd accumulation).
//
// Key insight: Grid-level K-splitting creates 28×48=1344 independent blocks
// (vs 56 with square_vnni), providing enough wavefronts to fully hide
// HBM2 latency on MI50 (60 CUs × 40 max concurrent waves = 2400).
//
// Benchmark results (MI50, gfx906):
//   Q/Wo (3584×3584):    22.0μs / 58.5% peak BW (9.6× vs square_vnni)
//   FFN Down (3584×18944): 95.7μs / 70.9% peak BW (2.0× vs smem kpar)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes its K-segment: [k_start, k_end)
    // CRITICAL: k_per must be a multiple of 4 to ensure k_start aligns with
    // VNNI 4-byte groups. If k_start is not a multiple of 4, kg_offset would
    // misalign with the weight K-groups, producing garbage output.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;  // Round up to multiple of 4
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // Accumulate partial dot product for CPT output columns
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        // Process VNNI k-groups within this block's K-segment
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        // Precompute B pointer: b_row[kg] points to the start of this thread's
        // columns for k-group kg. b_row = B_int32_ptr + n_base, stride = N (int32 units).
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Partial unroll: expose 4 k-groups per iteration for better
        // memory-level parallelism and instruction scheduling
        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // Vectorized 8-byte load: two adjacent VNNI columns in one
                // memory transaction. n_base is always even (threadIdx.x * 2),
                // so int2 alignment is guaranteed for N divisible by 2.
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

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

template <int TILE_N, int CPT, bool IQ4>
__global__ void gemv_ratio_vnni_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    const int8_t* __restrict__ d_ratio,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks,
    uint8_t)
{
    static_assert(CPT == 1, "Canonical ratio-VNNI path requires CPT==1");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);

            if constexpr (CPT == 1)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n_base);
                const uint8_t* payload16 = d_payload + linear * 16;
                const int8_t ratio = d_ratio[linear];

                if constexpr (IQ4)
                {
                    const uint4 packed = *reinterpret_cast<const uint4*>(payload16);
                    uint32_t lut0 = 0;
                    uint32_t lut1 = 0;
                    uint32_t lut2 = 0;
                    uint32_t lut3 = 0;
                    fetch_iq4_perm_lut_words_from_ratio_table(ratio, lut0, lut1, lut2, lut3);

                    const int32_t a0 = a4[0];
                    const int32_t a1 = a4[1];
                    const int32_t a2 = a4[2];
                    const int32_t a3 = a4[3];
                    const int32_t a4v = a4[4];
                    const int32_t a5 = a4[5];
                    const int32_t a6 = a4[6];
                    const int32_t a7 = a4[7];

                    int32_t acc0 = acc[0];
                    int32_t acc1 = 0;

                    int32_t w0_low, w0_high, w1_low, w1_high;
                    decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.x, packed.y, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
                    acc0 = __builtin_amdgcn_sdot4(a0, w0_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a4v, w0_high, acc1, false);
                    acc0 = __builtin_amdgcn_sdot4(a1, w1_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a5, w1_high, acc1, false);

                    decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.z, packed.w, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
                    acc0 = __builtin_amdgcn_sdot4(a2, w0_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a6, w0_high, acc1, false);
                    acc0 = __builtin_amdgcn_sdot4(a3, w1_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a7, w1_high, acc1, false);

                    acc[0] = acc0 + acc1;
                }
                else
                {
                    const uint4 packed = *reinterpret_cast<const uint4*>(payload16);
                    uint32_t lut0 = 0;
                    uint32_t lut1 = 0;
                    uint32_t lut2 = 0;
                    uint32_t lut3 = 0;
                    fetch_linear_perm_lut_words_from_ratio_table(ratio, lut0, lut1, lut2, lut3);

                    const int32_t a0 = a4[0];
                    const int32_t a1 = a4[1];
                    const int32_t a2 = a4[2];
                    const int32_t a3 = a4[3];
                    const int32_t a4v = a4[4];
                    const int32_t a5 = a4[5];
                    const int32_t a6 = a4[6];
                    const int32_t a7 = a4[7];

                    int32_t acc0 = acc[0];
                    int32_t acc1 = 0;

                    int32_t w0_low, w0_high, w1_low, w1_high;
                    decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.x, packed.y, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
                    acc0 = __builtin_amdgcn_sdot4(a0, w0_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a4v, w0_high, acc1, false);
                    acc0 = __builtin_amdgcn_sdot4(a1, w1_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a5, w1_high, acc1, false);

                    decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.z, packed.w, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
                    acc0 = __builtin_amdgcn_sdot4(a2, w0_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a6, w0_high, acc1, false);
                    acc0 = __builtin_amdgcn_sdot4(a3, w1_low, acc0, false);
                    acc1 = __builtin_amdgcn_sdot4(a7, w1_high, acc1, false);

                    acc[0] = acc0 + acc1;
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            if (n_base + c < N)
            {
                if (kblocks == 1)
                {
                    d_C_int32[n_base + c] = acc[c];
                }
                else
                {
                    atomicAdd(&d_C_int32[n_base + c], acc[c]);
                }
            }
        }
    }
}

__global__ void expand_ratio_vnni_to_rowmajor_q4_kernel(
    const uint8_t* __restrict__ d_payload,
    const int8_t* __restrict__ d_ratio,
    int8_t* __restrict__ d_B_rowmajor,
    int N, int K,
    uint8_t codebook_id)
{
    const int blocks_per_row = K / 32;
    const int total = N * blocks_per_row;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int n = tid % N;
    const int b = tid / N;
    const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);

    const uint8_t* payload16 = d_payload + linear * 16;
    const int8_t ratio = d_ratio[linear];
    int8_t* row_ptr = d_B_rowmajor + static_cast<int64_t>(n) * K + b * 32;

    int32_t packed_groups[8];
    if (codebook_id == RATIO_VNNI_CODEBOOK_IQ4)
    {
        decode_q4_block8_to_packed_i8<true>(payload16, ratio, packed_groups);
    }
    else
    {
        decode_q4_block8_to_packed_i8<false>(payload16, ratio, packed_groups);
    }

    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        *reinterpret_cast<int32_t*>(row_ptr + g * 4) = packed_groups[g];
    }
}


// ============================================================================
// Fused FP32→INT8 GEMV kernel: Quantize + GEMV + Scale in one launch
//
// Eliminates the 3-kernel pipeline (quantize → GEMV → scale) by:
//   1. Loading FP32 activations to shared memory per K-block
//   2. Computing block-local absmax for INT8 quantization
//   3. Quantizing in-register and using v_dot4_i32_i8 for the dot product
//   4. Converting INT32 accumulator to FP32 with (block_scale * scale_B[n])
//   5. atomicAdd FP32 partial results across K-blocks
//   6. Optional fused bias addition
//
// Memory traffic: Same weight reads as INT8 GEMV, but no separate
// activation INT8 buffer, no INT32 output buffer, no scale_A buffer.
// Activations (14KB for K=3584) stay in L2 after first projection.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N/CPT)
// Same grid structure as grid_kpar, but FP32 in → FP32 out.
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_fp32_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,                // [N] FP32 output (zeroed before launch)
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    // Process K in chunks of QUANT_TILE_K elements at a time for quantization.
    // Each chunk: load FP32 → find absmax → quantize → dot product.
    // 256 elements = 1KB FP32 in shared memory, fits easily.
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    // Each block processes its K-segment: [k_start, k_end)
    // Round k_per UP to a multiple of 4 so every k_start aligns with
    // VNNI K-groups (4 consecutive K values packed together).
    // Without this, k_start for intermediate blocks can land mid-group,
    // causing activation/weight misalignment in sdot4.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // After rounding k_per up, the last block(s) may have k_start >= K.
    // Those blocks have nothing to do — skip entirely.
    if (k_start >= K) return;

    // Shared memory: FP32 activations tile + reduction scratch
    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;        // Block-wide absmax for this tile
    __shared__ int8_t s_A_int8[QUANT_TILE_K]; // Quantized activations

    // FP32 accumulators for partial results (will be atomicAdd'd)
    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    // Process K-segment in QUANT_TILE_K-sized chunks
    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // --- Phase 1: Cooperative load of FP32 activations to shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        }
        __syncthreads();

        // --- Phase 2: Find absmax via parallel reduction ---
        // Each thread finds local max over its portion
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        // Warp reduction (wavefront = 64 threads on gfx906)
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Thread 0 writes the block absmax
        if (tid == 0)
        {
            s_absmax = local_max;
        }
        __syncthreads();

        const float absmax = s_absmax;
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // --- Phase 3: Cooperative quantize FP32 → INT8 in shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // --- Phase 4: INT8 dot product using v_dot4_i32_i8 ---
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
            }
        }

        __syncthreads();  // Ensure shared memory is safe for next tile
    }

    // --- Phase 5: Apply weight scales, write to output ---
    if (active)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            float result = f_acc[c] * d_scales_B[n_base + c];
            if (kblocks == 1)
                d_C_fp32[n_base + c] = result;      // Plain store — no contention
            else
                atomicAdd(&d_C_fp32[n_base + c], result);
        }
    }

    // Bias: only apply once across all K-blocks (last block does it)
    // We use k_block == kblocks - 1 as the "last" block.
    // Note: this has a data race if atomicAdds from other blocks haven't
    // completed yet. Instead, bias is applied as a separate follow-up.
    // See dispatch function for bias handling.
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {

void rocmGemv_ratio_vnni_set_tuning_overrides(
    int linear_tn,
    int linear_kb,
    int iq4_tn,
    int iq4_kb)
{
    g_ratio_linear_tn_override.store(linear_tn, std::memory_order_relaxed);
    g_ratio_linear_kb_override.store(linear_kb, std::memory_order_relaxed);
    g_ratio_iq4_tn_override.store(iq4_tn, std::memory_order_relaxed);
    g_ratio_iq4_kb_override.store(iq4_kb, std::memory_order_relaxed);
}

void rocmGemv_ratio_vnni_reset_tuning_overrides()
{
    rocmGemv_ratio_vnni_set_tuning_overrides(-1, -1, -1, -1);
}

bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const bool use_wide_vec4 = (N >= 8 * K) && use_vec4 && (N >= GEMV_INT8_VNNI_WIDE_TILE_N);
    // Grid-level K-parallel: handles ALL non-wide_vec4 shapes with N%4==0.
    // Previous wide2 path (N>=3K, N<8K) only generated N/256 blocks → 148 waves
    // for FFN Gate/Up, achieving just 17% peak BW. Grid_kpar K-splits via
    // blockIdx.y to generate thousands of blocks, fully saturating the GPU.
    // Profiling: grid_kpar achieves 488-649 GB/s (48-63% peak) vs wide2's 176 GB/s.
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (use_wide_vec4)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_WIDE_TILE_N - 1) / GEMV_INT8_VNNI_WIDE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide_vec4_vnni_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        //
        // KB heuristic: adapt based on N-tile count and K size.
        // Key insight from PMC profiling: inner loop length (k-groups/wave) is the
        // dominant factor for BW efficiency. Longer loops amortize per-block overhead
        // (index calc, atomicAdd, wave scheduling). At ≥10 waves/CU, adding more
        // waves doesn't help — it just shortens loops and adds overhead.
        //
        // Empirical targets (MI50 gfx906):
        //   FFN Down: 148 kgrps/wave, 14.9 w/CU → 62% peak BW (best)
        //   FFN Gate/Up: 32 kgrps/wave, 69.1 w/CU → 57% peak (too many waves)
        //   Q/Wo: 16 kgrps/wave, 26.1 w/CU → 50% peak (inner loop too short)
        //
        // Strategy: target ≥ MIN_KGROUPS_PER_BLOCK kgrps/wave while keeping
        // waves/CU above TARGET_MIN_WAVES_PER_CU for latency hiding.
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
        constexpr int NUM_CUS = 60;  // MI50

        const int grid_n = (N + TN - 1) / TN;
        const int k_groups = K / 4;

        int kb;
        if (K > 8192) {
            // Large-K (FFN Down): inherently long inner loops, tune for occupancy
            kb = max(4, min(32, K / 512));
        } else {
            // All other shapes: target ≥64 k-groups per wave for efficiency,
            // while keeping ≥10 waves/CU for memory latency hiding.
            // Cap KB so inner loop never drops below 16 k-groups (proven floor
            // from Q/Wo baseline — going shorter always hurts).
            constexpr int TARGET_KGROUPS = 64;
            constexpr int TARGET_MIN_WAVES_PER_CU = 10;
            constexpr int MIN_KGROUPS_FLOOR = 16;
            
            // Start from ideal KB for target k-groups
            int kb_for_kgroups = max(2, k_groups / TARGET_KGROUPS);
            
            // But ensure minimum occupancy
            int min_blocks = TARGET_MIN_WAVES_PER_CU * NUM_CUS;
            int kb_for_occupancy = max(2, (min_blocks + grid_n - 1) / grid_n);
            
            // Take the larger of the two constraints, but cap to preserve
            // minimum inner loop length (protects small-N shapes like K/V proj)
            int kb_max = max(2, k_groups / MIN_KGROUPS_FLOOR);
            kb = min(max(kb_for_kgroups, kb_for_occupancy), kb_max);
        }

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));  // NOLINT(clang-diagnostic-unused-result)

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

bool rocmGemv_ratio_vnni_int8_int32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const int8_t* d_ratio,
    int32_t* d_C_int32,
    int N, int K,
    uint8_t bitwidth,
    uint8_t codebook_id,
    uint8_t has_min,
    uint8_t block_size,
    uint16_t payload_bytes,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (bitwidth != 4 || has_min != 0 || block_size != 32 || payload_bytes != 16)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Unsupported header (bitwidth=%u has_min=%u block=%u payload=%u)\n",
                static_cast<unsigned>(bitwidth), static_cast<unsigned>(has_min),
                static_cast<unsigned>(block_size), static_cast<unsigned>(payload_bytes));
        return false;
    }
    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }
    if (codebook_id != RATIO_VNNI_CODEBOOK_IQ4 && codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Unsupported codebook_id=%u\n", static_cast<unsigned>(codebook_id));
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int TN_IQ4_WIDE = 256;
    const int linear_tn_override = g_ratio_linear_tn_override.load(std::memory_order_relaxed);
    const int linear_kb_override = g_ratio_linear_kb_override.load(std::memory_order_relaxed);
    const int iq4_tn_override = g_ratio_iq4_tn_override.load(std::memory_order_relaxed);
    const int iq4_kb_override = g_ratio_iq4_kb_override.load(std::memory_order_relaxed);

    const int blocks_per_row = K / 32;
    const int min_dim = min(N, K);
    const int max_dim = max(N, K);
    const bool is_squareish_shape = (min_dim >= 2048) && (max_dim * 100 <= min_dim * 140);

    int kb;
    if (codebook_id == RATIO_VNNI_CODEBOOK_IQ4)
    {
        if (blocks_per_row > 256)
            kb = min(16, max(4, blocks_per_row / 16));
        else
            kb = min(12, max(2, blocks_per_row / 8));

        if (is_squareish_shape)
        {
            kb = 16;
        }

        if (iq4_kb_override > 0)
        {
            kb = iq4_kb_override;
        }

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));

        bool iq4_use_wide_tile = (blocks_per_row > 256);
        if (is_squareish_shape)
        {
            iq4_use_wide_tile = false;
        }
        if (iq4_tn_override == TN)
        {
            iq4_use_wide_tile = false;
        }
        else if (iq4_tn_override == TN_IQ4_WIDE)
        {
            iq4_use_wide_tile = true;
        }
        const int tn_iq4 = iq4_use_wide_tile ? TN_IQ4_WIDE : TN;
        const int grid_n_iq4 = (N + tn_iq4 - 1) / tn_iq4;

        if (!ensure_ratio_iq4_perm_lut_table_on_device(device_id))
        {
            return false;
        }

        if (iq4_use_wide_tile)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE, 1, true>),
                dim3(grid_n_iq4, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_payload, d_ratio, d_C_int32,
                N, K, kb, codebook_id);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN, 1, true>),
                dim3(grid_n_iq4, kb), dim3(TN), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_payload, d_ratio, d_C_int32,
                N, K, kb, codebook_id);
        }
    }
    else
    {
        if (blocks_per_row > 256)
            kb = min(20, max(4, blocks_per_row / 16));
        else
            kb = min(12, max(2, blocks_per_row / 8));

        if (is_squareish_shape)
        {
            kb = 16;
        }

        if (linear_kb_override > 0)
        {
            kb = linear_kb_override;
        }

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));

        bool linear_use_wide_tile = (blocks_per_row > 256);
        if (is_squareish_shape)
        {
            linear_use_wide_tile = false;
        }
        if (linear_tn_override == TN)
        {
            linear_use_wide_tile = false;
        }
        else if (linear_tn_override == TN_IQ4_WIDE)
        {
            linear_use_wide_tile = true;
        }
        const int tn_linear = linear_use_wide_tile ? TN_IQ4_WIDE : TN;
        const int grid_n_linear = (N + tn_linear - 1) / tn_linear;

        if (!ensure_ratio_linear_perm_lut_table_on_device(device_id))
        {
            return false;
        }

        if (linear_use_wide_tile)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE, 1, false>),
                dim3(grid_n_linear, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_payload, d_ratio, d_C_int32,
                N, K, kb, codebook_id);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN, 1, false>),
                dim3(grid_n_linear, kb), dim3(TN), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_payload, d_ratio, d_C_int32,
                N, K, kb, codebook_id);
        }
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

bool rocmGemv_expandRatioVNNI_to_rowmajor(
    const uint8_t* d_payload,
    const int8_t* d_ratio,
    int8_t* d_B_rowmajor,
    int N, int K,
    uint8_t bitwidth,
    uint8_t codebook_id,
    uint8_t has_min,
    uint8_t block_size,
    uint16_t payload_bytes,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (bitwidth != 4 || has_min != 0 || block_size != 32 || payload_bytes != 16)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] Unsupported header (bitwidth=%u has_min=%u block=%u payload=%u)\n",
                static_cast<unsigned>(bitwidth), static_cast<unsigned>(has_min),
                static_cast<unsigned>(block_size), static_cast<unsigned>(payload_bytes));
        return false;
    }
    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    const int blocks_per_row = K / 32;
    const int total = N * blocks_per_row;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        expand_ratio_vnni_to_rowmajor_q4_kernel,
        dim3(grid), dim3(BLOCK), 0, static_cast<hipStream_t>(stream),
        d_payload, d_ratio, d_B_rowmajor, N, K, codebook_id);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

// =============================================================================
// Fused FP32→FP32 GEMV: Quantize + GEMV + Scale in one kernel
// =============================================================================
//
// Replaces the 3-kernel decode pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: rocmGemv_fused_fp32_int8_vnni (single launch, directly FP32→FP32)
//
// Eliminates:
//   - d_A_int8 buffer (activation quantization done in shared memory)
//   - d_scales_A buffer (scale folded into per-tile FP32 conversion)
//   - d_CK_int32 buffer (output is FP32 directly)
//   - 2 kernel launches per projection (quantize + scale)
//
bool rocmGemv_fused_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output (will be zeroed)
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic for FUSED kernel: very different from INT8 grid_kpar!
    //
    // The fused kernel uses FP32 atomicAdd for cross-block K-accumulation.
    // On gfx906, FP32 atomicAdd is emulated via CAS loop (not hardware native),
    // so ANY K-splitting causes massive contention overhead.
    //
    // Force kb=1: each block processes the entire K dimension sequentially
    // via QUANT_TILE_K=256 tiles in a loop, then does a single plain store.
    // No atomicAdd, no hipMemsetAsync needed.
    const int kb = 1;

    // Zero output only if using atomicAdd accumulation across K-blocks
    if (kb > 1)
        hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
        dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, d_bias, N, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    // Apply bias separately if provided and using K-splitting.
    // With kb==1, bias could be fused but we keep the separate path for simplicity.
    if (d_bias)
    {
        // Reuse existing biasAdd from ROCmQuantisedGemmKernel_CK.hip
        extern bool rocmQuantGemm_biasAdd(
            float *d_output, const float *d_bias,
            int M, int N, int rocm_device_id, void *stream);

        if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
        {
            fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] biasAdd failed (N=%d)\n", N);
            return false;
        }
    }

    return true;
}

// =============================================================================
// VNNI → Row-Major Repack Kernel
// =============================================================================
//
// Repacks weights from flat VNNI layout [K/4][N][4] to row-major [N×K]
// (equivalently [K×N] column-major for CK).
//
// VNNI layout:  vnni[(kg*N + n)*4 + i]  where kg = k/4, i = k%4
// Row-major:    rowmajor[n*K + k]       where k = kg*4 + i
//
// Mapping:  rowmajor[n*K + kg*4 + i] = vnni[(kg*N + n)*4 + i]
//
// Each thread handles one (n, kg) pair = 4 elements, packed as int32_t.
// Grid: ceil(N * K/4 / blockDim.x) blocks, blockDim.x = 256 threads.
//
// Access pattern:
//   - VNNI reads:  consecutive tids (consecutive n) → stride 4 → coalesced
//   - Row-major writes: consecutive tids (consecutive n) → stride K → scattered
//     but each write is a 4-byte int32_t, so individual writes are efficient.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    const int k_groups = K / 4;
    const int total = N * k_groups;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total)
        return;

    const int n  = tid % N;
    const int kg = tid / N;

    // Read 4 bytes from VNNI as a single int32_t (coalesced for consecutive n)
    const int32_t packed = *reinterpret_cast<const int32_t*>(
        &d_B_vnni[(static_cast<int64_t>(kg) * N + n) * 4]);

    // Write 4 bytes to row-major (scattered for consecutive n, but 4B atomic)
    *reinterpret_cast<int32_t*>(
        &d_B_rowmajor[static_cast<int64_t>(n) * K + kg * 4]) = packed;
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id, void* stream)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int k_groups = K / 4;
    const int total = N * k_groups;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid), dim3(BLOCK), 0, static_cast<hipStream_t>(stream),
        d_B_vnni, d_B_rowmajor, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

} // extern "C"
