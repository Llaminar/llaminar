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
#include "ROCmRatioVNNIAbi.h"

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
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_WIDE_DEFAULT_CPT = 2;
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
static std::atomic<int> g_int8_vnni_tn_override{-1};
static std::atomic<int> g_int8_vnni_kb_override{-1};
static std::atomic<int> g_int8_vnni_wide_tn_override{-1};
static std::atomic<int> g_int8_vnni_wide_cpt_override{-1};
static std::atomic<int> g_int8_vnni_wide_vec_load_override{-1};

__device__ __constant__ int8_t k_ratio_vnni_iq4_lut_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

__device__ __constant__ uint32_t k_ratio_iq4_perm_lut_words[1024];
__device__ __constant__ uint32_t k_ratio_linear_perm_lut_words[1024];
__device__ __constant__ uint32_t k_ratio_linear_unsigned_perm_lut_words[1024];

static std::array<uint32_t, 1024> build_ratio_iq4_perm_lut_table_host()
{
    constexpr int8_t k_ratio_vnni_iq4_lut_i8_host[16] = {
        -127, -104, -83, -65,
        -49, -35, -22, -10,
        1, 13, 25, 38,
        53, 69, 89, 113};

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
            const int8_t wi_lo_0 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8_host[i]) + 64) >> 7);
            const int8_t wi_lo_1 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8_host[i + 4]) + 64) >> 7);
            const int8_t wi_hi_0 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8_host[i + 8]) + 64) >> 7);
            const int8_t wi_hi_1 = static_cast<int8_t>((static_cast<int>(ratio) * static_cast<int>(k_ratio_vnni_iq4_lut_i8_host[i + 12]) + 64) >> 7);

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

static std::array<uint32_t, 1024> build_ratio_linear_unsigned_perm_lut_table_host()
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
            const int8_t wi_lo_0 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 0) + 64) >> 7);
            const int8_t wi_lo_1 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 4) + 64) >> 7);
            const int8_t wi_hi_0 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 8) + 64) >> 7);
            const int8_t wi_hi_1 = static_cast<int8_t>((static_cast<int>(ratio) * (i + 12) + 64) >> 7);

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

static bool ensure_ratio_linear_unsigned_perm_lut_table_on_device(int device_id)
{
    static std::mutex init_mutex;
    static bool initialized[16] = {false};
    static const std::array<uint32_t, 1024> host_table = build_ratio_linear_unsigned_perm_lut_table_host();

    const int did = (device_id >= 0 && device_id < 16) ? device_id : 0;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized[did])
    {
        return true;
    }

    hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(k_ratio_linear_unsigned_perm_lut_words),
                                       host_table.data(),
                                       host_table.size() * sizeof(uint32_t),
                                       0,
                                       hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_ratio_vnni_int8_int32] Failed to init linear unsigned mode3 LUT table on device %d: %s\n",
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

template <bool HAS_MIN>
__device__ __forceinline__ int32_t decode_q5_group_to_packed_i8_linear(
    const uint8_t* payload20,
    int group,
    int8_t ratio,
    int min_q)
{
    const uint8_t* qs = payload20;
    const uint8_t* qh_bytes = payload20 + 16;
    const uint32_t qh =
        static_cast<uint32_t>(qh_bytes[0]) |
        (static_cast<uint32_t>(qh_bytes[1]) << 8) |
        (static_cast<uint32_t>(qh_bytes[2]) << 16) |
        (static_cast<uint32_t>(qh_bytes[3]) << 24);

    int8_t decoded[4];
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane)
    {
        const int idx = group * 4 + lane;
        const int j = idx & 0x0F;
        const bool upper = (idx >= 16);
        const uint8_t nibble = upper
            ? static_cast<uint8_t>(qs[j] >> 4)
            : static_cast<uint8_t>(qs[j] & 0x0F);
        const uint8_t xh = upper
            ? static_cast<uint8_t>((qh >> (j + 12)) & 0x10)
            : static_cast<uint8_t>(((qh >> j) << 4) & 0x10);

        int d = static_cast<int>(nibble | xh);
        if constexpr (!HAS_MIN)
        {
            d -= 16;
        }

        int w = ((static_cast<int>(ratio) * d + 64) >> 7);
        if constexpr (HAS_MIN)
        {
            w += min_q;
        }

        w = max(-128, min(127, w));
        decoded[lane] = static_cast<int8_t>(w);
    }

    uint32_t packed = 0;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(decoded[0]));
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(decoded[1])) << 8);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(decoded[2])) << 16);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(decoded[3])) << 24);
    return static_cast<int32_t>(packed);
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

__device__ __forceinline__ int32_t add_min_clamp_to_packed_i8x4(
    int32_t packed_i8x4,
    int min_q)
{
    const int v0 = max(-128, min(127, static_cast<int>(static_cast<int8_t>(packed_i8x4 & 0xFF)) + min_q));
    const int v1 = max(-128, min(127, static_cast<int>(static_cast<int8_t>((packed_i8x4 >> 8) & 0xFF)) + min_q));
    const int v2 = max(-128, min(127, static_cast<int>(static_cast<int8_t>((packed_i8x4 >> 16) & 0xFF)) + min_q));
    const int v3 = max(-128, min(127, static_cast<int>(static_cast<int8_t>((packed_i8x4 >> 24) & 0xFF)) + min_q));

    uint32_t packed = 0;
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(v0));
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16);
    packed |= (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
    return static_cast<int32_t>(packed);
}

__device__ __forceinline__ void decode_q4_block8_to_packed_i8_linear_with_min(
    const uint8_t* payload16,
    int8_t ratio,
    int min_q,
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

    out[0] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_low<false>(q0, q1, q2, q3, ratio), min_q);
    out[1] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_low<false>(q4, q5, q6, q7, ratio), min_q);
    out[2] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_low<false>(q8, q9, q10, q11, ratio), min_q);
    out[3] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_low<false>(q12, q13, q14, q15, ratio), min_q);
    out[4] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_high<false>(q0, q1, q2, q3, ratio), min_q);
    out[5] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_high<false>(q4, q5, q6, q7, ratio), min_q);
    out[6] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_high<false>(q8, q9, q10, q11, ratio), min_q);
    out[7] = add_min_clamp_to_packed_i8x4(decode_q4_pack4_high<false>(q12, q13, q14, q15, ratio), min_q);
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

__device__ __forceinline__ void fetch_linear_unsigned_perm_lut_words_from_ratio_table(
    int8_t ratio,
    uint32_t& lut0,
    uint32_t& lut1,
    uint32_t& lut2,
    uint32_t& lut3)
{
    const uint32_t idx = static_cast<uint32_t>(static_cast<uint8_t>(ratio)) * 4u;
    lut0 = k_ratio_linear_unsigned_perm_lut_words[idx + 0];
    lut1 = k_ratio_linear_unsigned_perm_lut_words[idx + 1];
    lut2 = k_ratio_linear_unsigned_perm_lut_words[idx + 2];
    lut3 = k_ratio_linear_unsigned_perm_lut_words[idx + 3];
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

/**
 * @brief 2D INT8 VNNI prefill kernel (M>1): computes C[M,N] in one launch.
 *
 * Thread mapping:
 * - threadIdx.y / blockIdx.y choose output row `m`
 * - threadIdx.x / blockIdx.x choose output column `n`
 *
 * For each output element C[m,n], we iterate over K in groups of 4 bytes and
 * use `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) for efficient INT8 MAC.
 *
 * Layout reminder for junior developers:
 * - A is row-major [M x K]
 * - B is VNNI-packed [K/4 x N x 4]
 * - C is row-major [M x N]
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
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

    // Compute one output element by reducing across K.
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

/**
 * @brief 2D INT8 VNNI prefill kernel with grid-kpar split-K accumulation.
 *
 * grid.z partitions K into slices. Each z-slice computes a partial sum for
 * C[m,n] and atomically accumulates into INT32 output.
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = split_idx * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int kg = k_begin; kg < k_end; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N && acc[c] != 0)
        {
            atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}

/**
 * @brief 2D ratio-VNNI prefill kernel (M>1): computes C[M,N] in one launch.
 *
 * This kernel mirrors the INT8-VNNI prefill shape traversal but decodes the
 * compact ratio-VNNI payload per (block, column) on the fly, avoiding a full
 * row-major expansion buffer.
 *
 * Layout reminder:
 * - A is row-major INT8 [M x K]
 * - Payload is [K/32 x N x 16 bytes]
 * - Ratio side metadata is [K/32 x N]
 * - C is row-major INT32 [M x N]
 */
template <bool IQ4, bool BITWIDTH5>
__global__ void gemm_ratio_vnni_prefill_kernel(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    const int8_t* __restrict__ d_ratio,
    const int8_t* __restrict__ d_min,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    uint8_t has_min,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
        constexpr size_t payload_stride = BITWIDTH5 ? 20u : 16u;
        const uint8_t* payload_ptr = d_payload + linear * payload_stride;
        const int8_t ratio = d_ratio[linear];

        int32_t packed_groups[8];
        if constexpr (BITWIDTH5)
        {
            if (has_min != 0)
            {
                const int min_q = static_cast<int>(d_min[linear]);
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    packed_groups[g] = decode_q5_group_to_packed_i8_linear<true>(payload_ptr, g, ratio, min_q);
                }
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    packed_groups[g] = decode_q5_group_to_packed_i8_linear<false>(payload_ptr, g, ratio, 0);
                }
            }
        }
        else if (has_min != 0)
        {
            const int min_q = static_cast<int>(d_min[linear]);
            const uint4 packed = *reinterpret_cast<const uint4*>(payload_ptr);
            uint32_t lut0 = 0;
            uint32_t lut1 = 0;
            uint32_t lut2 = 0;
            uint32_t lut3 = 0;
            fetch_linear_unsigned_perm_lut_words_from_ratio_table(ratio, lut0, lut1, lut2, lut3);

            int32_t w0_low, w0_high, w1_low, w1_high;
            decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.x, packed.y, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
            packed_groups[0] = add_min_clamp_to_packed_i8x4(w0_low, min_q);
            packed_groups[4] = add_min_clamp_to_packed_i8x4(w0_high, min_q);
            packed_groups[1] = add_min_clamp_to_packed_i8x4(w1_low, min_q);
            packed_groups[5] = add_min_clamp_to_packed_i8x4(w1_high, min_q);

            decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.z, packed.w, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
            packed_groups[2] = add_min_clamp_to_packed_i8x4(w0_low, min_q);
            packed_groups[6] = add_min_clamp_to_packed_i8x4(w0_high, min_q);
            packed_groups[3] = add_min_clamp_to_packed_i8x4(w1_low, min_q);
            packed_groups[7] = add_min_clamp_to_packed_i8x4(w1_high, min_q);
        }
        else
        {
            const uint4 packed = *reinterpret_cast<const uint4*>(payload_ptr);
            uint32_t lut0 = 0;
            uint32_t lut1 = 0;
            uint32_t lut2 = 0;
            uint32_t lut3 = 0;

            if constexpr (IQ4)
            {
                fetch_iq4_perm_lut_words_from_ratio_table(ratio, lut0, lut1, lut2, lut3);
            }
            else
            {
                fetch_linear_perm_lut_words_from_ratio_table(ratio, lut0, lut1, lut2, lut3);
            }

            int32_t w0_low, w0_high, w1_low, w1_high;
            decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.x, packed.y, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
            packed_groups[0] = w0_low;
            packed_groups[4] = w0_high;
            packed_groups[1] = w1_low;
            packed_groups[5] = w1_high;

            decode_q4_two_pairs_to_packed_i8_with_perm_lut16(packed.z, packed.w, lut0, lut1, lut2, lut3, w0_low, w0_high, w1_low, w1_high);
            packed_groups[2] = w0_low;
            packed_groups[6] = w0_high;
            packed_groups[3] = w1_low;
            packed_groups[7] = w1_high;
        }

        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc, false);
        }
    }

    const int64_t out_idx = static_cast<int64_t>(m) * N + n;
    if (kb == 1)
    {
        d_C_int32[out_idx] = acc;
    }
    else
    {
        atomicAdd(&d_C_int32[out_idx], acc);
    }
}

/**
 * @brief 2D payload-v2 bitwidth-8 prefill kernel (M>1): computes C[M,N].
 *
 * Layout:
 * - A is row-major INT8 [M x K]
 * - Payload is [K/32 x N x 32 bytes] where each record stores 32 signed INT8
 *   values for one (block, output-column)
 * - C is row-major INT32 [M x N]
 */
__global__ void gemm_payload_v2_i8_prefill_kernel(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const bool valid_n = (n < N);
    const bool valid_m = (m < M);

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;
    extern __shared__ int32_t shared_payload[]; // [blockDim.x * 8]

    for (int b = b0; b < b1; ++b)
    {
        if (threadIdx.y == 0)
        {
            int32_t* payload_dst = shared_payload + static_cast<int>(threadIdx.x) * 8;
            if (valid_n)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
                const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);
                payload_dst[0] = static_cast<int32_t>(packed0.x);
                payload_dst[1] = static_cast<int32_t>(packed0.y);
                payload_dst[2] = static_cast<int32_t>(packed0.z);
                payload_dst[3] = static_cast<int32_t>(packed0.w);
                payload_dst[4] = static_cast<int32_t>(packed1.x);
                payload_dst[5] = static_cast<int32_t>(packed1.y);
                payload_dst[6] = static_cast<int32_t>(packed1.z);
                payload_dst[7] = static_cast<int32_t>(packed1.w);
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    payload_dst[g] = 0;
                }
            }
        }

        __syncthreads();

        if (valid_m && valid_n)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                d_A_int8 + static_cast<int64_t>(m) * K + b * 32);
            const int32_t* payload_i32 = shared_payload + static_cast<int>(threadIdx.x) * 8;

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc, false);
            }
        }

        __syncthreads();
    }

    if (valid_m && valid_n)
    {
        const int64_t out_idx = static_cast<int64_t>(m) * N + n;
        if (kb == 1)
        {
            d_C_int32[out_idx] = acc;
        }
        else
        {
            atomicAdd(&d_C_int32[out_idx], acc);
        }
    }
}

__global__ void gemm_payload_v2_i8_prefill_kernel_direct(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
        const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
        const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);

        acc = __builtin_amdgcn_sdot4(a4[0], static_cast<int32_t>(packed0.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[1], static_cast<int32_t>(packed0.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[2], static_cast<int32_t>(packed0.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[3], static_cast<int32_t>(packed0.w), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[4], static_cast<int32_t>(packed1.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[5], static_cast<int32_t>(packed1.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[6], static_cast<int32_t>(packed1.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[7], static_cast<int32_t>(packed1.w), acc, false);
    }

    const int64_t out_idx = static_cast<int64_t>(m) * N + n;
    if (kb == 1)
    {
        d_C_int32[out_idx] = acc;
    }
    else
    {
        atomicAdd(&d_C_int32[out_idx], acc);
    }
}

template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_payload_v2_i8_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear0 = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n_base);
        const int32_t* payload_i32_0 = reinterpret_cast<const int32_t*>(d_payload + linear0 * 32u);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                const int32_t* payload_i32_2 = payload_i32_1 + 8;
                const int32_t* payload_i32_3 = payload_i32_2 + 8;

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a, payload_i32_2[g], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a, payload_i32_3[g], acc[3], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc[0] = __builtin_amdgcn_sdot4(a4[g], payload_i32_0[g], acc[0], false);
            }
        }
    }

    const int64_t out_base = static_cast<int64_t>(m) * N + n_base;
    if (kb == 1)
    {
        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                *reinterpret_cast<uint4*>(d_C_int32 + out_base) = make_uint4(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]),
                    static_cast<uint32_t>(acc[2]),
                    static_cast<uint32_t>(acc[3]));
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        d_C_int32[out_base + c] = acc[c];
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                *reinterpret_cast<uint2*>(d_C_int32 + out_base) = make_uint2(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]));
            }
            else
            {
                d_C_int32[out_base] = acc[0];
                if ((n_base + 1) < N)
                {
                    d_C_int32[out_base + 1] = acc[1];
                }
            }
        }
        else
        {
            d_C_int32[out_base] = acc[0];
        }
    }
    else
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                atomicAdd(&d_C_int32[out_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_base + c < N)
        {
            d_C_int32[n_base + c] = acc[c];
        }
    }
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

__global__ void gemv_int8_int8_vnni_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    float result = static_cast<float>(acc) * d_scale_A[0] * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                }
            }
        }
    }

    const float scale_a = d_scale_A[0];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float out = static_cast<float>(acc[c]) * scale_a * d_scale_B[n];
            out *= alpha;
            if (beta != 0.0f)
            {
                out += beta * d_C_existing[n];
            }
            if (d_bias)
            {
                out += d_bias[n];
            }
            d_C_fp32[n] = out;
        }
    }
}

__global__ void gemv_int8_int8_square_vnni_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
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
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);
            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);
            acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
        }
        __syncthreads();
    }

    float result = static_cast<float>(acc) * d_scale_A[0] * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
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
    static_assert(CPT == 1 || CPT == 2, "ratio-VNNI grid-kpar supports CPT in {1,2}");

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
            else if constexpr (CPT == 2)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const uint8_t* payload16 = d_payload + linear * 16;
                        const int8_t ratio = d_ratio[linear];

                        int32_t packed_groups[8];
                        if constexpr (IQ4)
                        {
                            decode_q4_block8_to_packed_i8<true>(payload16, ratio, packed_groups);
                        }
                        else
                        {
                            decode_q4_block8_to_packed_i8<false>(payload16, ratio, packed_groups);
                        }

                        int32_t lane_acc = acc[c];
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            lane_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], lane_acc, false);
                        }
                        acc[c] = lane_acc;
                    }
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
    const int8_t* __restrict__ d_min,
    int8_t* __restrict__ d_B_rowmajor,
    int N, int K,
    uint8_t codebook_id,
    uint8_t has_min)
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

    if (has_min != 0)
    {
        const int min_q = static_cast<int>(d_min[linear]);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            const uint8_t q = payload16[i];
            const int low = ((static_cast<int>(ratio) * static_cast<int>(q & 0x0Fu) + 64) >> 7) + min_q;
            const int high = ((static_cast<int>(ratio) * static_cast<int>(q >> 4) + 64) >> 7) + min_q;

            const int low_clamped = max(-128, min(127, low));
            const int high_clamped = max(-128, min(127, high));
            row_ptr[i] = static_cast<int8_t>(low_clamped);
            row_ptr[i + 16] = static_cast<int8_t>(high_clamped);
        }
        return;
    }

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

__global__ void expand_ratio_vnni_to_rowmajor_q5_kernel(
    const uint8_t* __restrict__ d_payload,
    const int8_t* __restrict__ d_ratio,
    const int8_t* __restrict__ d_min,
    int8_t* __restrict__ d_B_rowmajor,
    int N, int K,
    uint8_t has_min)
{
    const int blocks_per_row = K / 32;
    const int total = N * blocks_per_row;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int n = tid % N;
    const int b = tid / N;
    const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);

    const uint8_t* payload20 = d_payload + linear * 20;
    const int8_t ratio = d_ratio[linear];
    const int min_q = (has_min != 0) ? static_cast<int>(d_min[linear]) : 0;
    int8_t* row_ptr = d_B_rowmajor + static_cast<int64_t>(n) * K + b * 32;

    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        const int32_t packed = (has_min != 0)
            ? decode_q5_group_to_packed_i8_linear<true>(payload20, g, ratio, min_q)
            : decode_q5_group_to_packed_i8_linear<false>(payload20, g, ratio, 0);
        *reinterpret_cast<int32_t*>(row_ptr + g * 4) = packed;
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

void rocmGemv_int8_vnni_set_tuning_overrides(
    int tn,
    int kb)
{
    g_int8_vnni_tn_override.store(tn, std::memory_order_relaxed);
    g_int8_vnni_kb_override.store(kb, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_tuning_overrides()
{
    rocmGemv_int8_vnni_set_tuning_overrides(-1, -1);
}

void rocmGemv_int8_vnni_set_wide_tuning_overrides(
    int wide_tn,
    int wide_cpt,
    int wide_vec_load)
{
    g_int8_vnni_wide_tn_override.store(wide_tn, std::memory_order_relaxed);
    g_int8_vnni_wide_cpt_override.store(wide_cpt, std::memory_order_relaxed);
    g_int8_vnni_wide_vec_load_override.store(wide_vec_load, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_wide_tuning_overrides()
{
    rocmGemv_int8_vnni_set_wide_tuning_overrides(-1, -1, -1);
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
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    // Grid-level K-parallel: handles ALL non-wide_vec4 shapes with N%4==0.
    // Previous wide2 path (N>=3K, N<8K) only generated N/256 blocks → 148 waves
    // for FFN Gate/Up, achieving just 17% peak BW. Grid_kpar K-splits via
    // blockIdx.y to generate thousands of blocks, fully saturating the GPU.
    // Profiling: grid_kpar achieves 488-649 GB/s (48-63% peak) vs wide2's 176 GB/s.
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);
    const int min_dim = min(N, K);
    const int max_dim = max(N, K);
    const bool is_squareish_shape = (min_dim >= 2048) && (max_dim * 100 <= min_dim * 140);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
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
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE_TILE_N;  // 256
        constexpr int NUM_CUS = 60;  // MI50

        int tn = TN;
        if (is_squareish_shape)
        {
            tn = TN_WIDE;
        }
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
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

        if (is_squareish_shape)
        {
            kb = 16;
        }

        if (int8_kb_override > 0)
        {
            kb = int8_kb_override;
        }

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));  // NOLINT(clang-diagnostic-unused-result)

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
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

bool rocmGemv_int8_int8_fp32_vnni_scaled(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scale_A,
    const float* d_scale_B,
    int N, int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* d_bias,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (beta != 0.0f && d_C_existing == nullptr)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] d_C_existing required when beta != 0 (N=%d K=%d)\n", N, K);
        return false;
    }

    if (use_grid_kpar)
    {
        return false;
    }

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_scaled_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_scaled_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] Kernel launch failed: %s (N=%d K=%d)\n",
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
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
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
        const bool iq4_use_cpt2 = (CPT == 2) && (N >= (tn_iq4 * CPT));
        const int logical_tn_iq4 = tn_iq4 * (iq4_use_cpt2 ? CPT : 1);
        const int grid_n_iq4 = (N + logical_tn_iq4 - 1) / logical_tn_iq4;

        if (!ensure_ratio_iq4_perm_lut_table_on_device(device_id))
        {
            return false;
        }

        if (iq4_use_wide_tile)
        {
            if (iq4_use_cpt2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE * CPT, CPT, true>),
                    dim3(grid_n_iq4, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_payload, d_ratio, d_C_int32,
                    N, K, kb, codebook_id);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE, 1, true>),
                    dim3(grid_n_iq4, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_payload, d_ratio, d_C_int32,
                    N, K, kb, codebook_id);
            }
        }
        else
        {
            if (iq4_use_cpt2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN * CPT, CPT, true>),
                    dim3(grid_n_iq4, kb), dim3(TN), 0, static_cast<hipStream_t>(stream),
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
        const bool linear_use_cpt2 = (CPT == 2) && (N >= (tn_linear * CPT));
        const int logical_tn_linear = tn_linear * (linear_use_cpt2 ? CPT : 1);
        const int grid_n_linear = (N + logical_tn_linear - 1) / logical_tn_linear;

        if (!ensure_ratio_linear_perm_lut_table_on_device(device_id))
        {
            return false;
        }

        if (linear_use_wide_tile)
        {
            if (linear_use_cpt2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE * CPT, CPT, false>),
                    dim3(grid_n_linear, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_payload, d_ratio, d_C_int32,
                    N, K, kb, codebook_id);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN_IQ4_WIDE, 1, false>),
                    dim3(grid_n_linear, kb), dim3(TN_IQ4_WIDE), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_payload, d_ratio, d_C_int32,
                    N, K, kb, codebook_id);
            }
        }
        else
        {
            if (linear_use_cpt2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_ratio_vnni_grid_kpar_kernel_t<TN * CPT, CPT, false>),
                    dim3(grid_n_linear, kb), dim3(TN), 0, static_cast<hipStream_t>(stream),
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
    const int8_t* d_min,
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

    const bool is_q4 = (bitwidth == 4 && block_size == 32 && payload_bytes == 16);
    const bool is_q5 = (bitwidth == 5 && block_size == 32 && payload_bytes == 20);
    if (!is_q4 && !is_q5)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] Unsupported header (bitwidth=%u has_min=%u block=%u payload=%u)\n",
                static_cast<unsigned>(bitwidth), static_cast<unsigned>(has_min),
                static_cast<unsigned>(block_size), static_cast<unsigned>(payload_bytes));
        return false;
    }

    if (has_min != 0 && !d_min)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] has_min set but min side-channel pointer is null\n");
        return false;
    }

    if (has_min != 0 && codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] has_min currently supports only linear codebook (codebook_id=%u)\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    if (is_q5 && codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
    {
        fprintf(stderr, "[rocmGemv_expandRatioVNNI_to_rowmajor] bitwidth=5 currently supports only linear codebook (codebook_id=%u)\n",
                static_cast<unsigned>(codebook_id));
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

    if (is_q4)
    {
        hipLaunchKernelGGL(
            expand_ratio_vnni_to_rowmajor_q4_kernel,
            dim3(grid), dim3(BLOCK), 0, static_cast<hipStream_t>(stream),
            d_payload, d_ratio, d_min, d_B_rowmajor, N, K, codebook_id, has_min);
    }
    else
    {
        hipLaunchKernelGGL(
            expand_ratio_vnni_to_rowmajor_q5_kernel,
            dim3(grid), dim3(BLOCK), 0, static_cast<hipStream_t>(stream),
            d_payload, d_ratio, d_min, d_B_rowmajor, N, K, has_min);
    }

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
// Treat each 4-byte pack as one int32 element in matrix A[kg, n] and transpose
// to B[n, kg] using a tiled shared-memory transpose to keep both reads and
// writes coalesced.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int k_groups = K / 4; // number of int32 packs per output row

    const int x_in = blockIdx.x * TILE_DIM + threadIdx.x; // n index in source A[kg, n]
    const int y_in = blockIdx.y * TILE_DIM + threadIdx.y; // kg index in source A[kg, n]

    const int32_t* __restrict__ src32 = reinterpret_cast<const int32_t*>(d_B_vnni);
    int32_t* __restrict__ dst32 = reinterpret_cast<int32_t*>(d_B_rowmajor);

    __shared__ int32_t tile[TILE_DIM][TILE_DIM + 1];

    // Load tile from source (A = [k_groups, N], row-major in n)
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_in + j;
        if (x_in < N && y < k_groups)
        {
            tile[threadIdx.y + j][threadIdx.x] = src32[static_cast<int64_t>(y) * N + x_in];
        }
    }

    __syncthreads();

    // Store transposed tile to destination (B = [N, k_groups], row-major in kg)
    const int x_out = blockIdx.y * TILE_DIM + threadIdx.x; // kg
    const int y_out = blockIdx.x * TILE_DIM + threadIdx.y; // n

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_out + j;
        if (x_out < k_groups && y < N)
        {
            dst32[static_cast<int64_t>(y) * k_groups + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
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
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int grid_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_y = (k_groups + TILE_DIM - 1) / TILE_DIM;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid_x, grid_y), dim3(TILE_DIM, BLOCK_ROWS), 0, static_cast<hipStream_t>(stream),
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

/**
 * @brief Experimental scaffold entrypoint for INT8 VNNI prefill GEMM.
 *
 * This function exists to stabilize API wiring while the real M>1 kernel is
 * still under active implementation. Returning false here is intentional: it
 * instructs the caller to keep using the proven CK fallback path.
 *
 * Junior-dev note:
 * - The decode path (M=1) already has specialized GEMV kernels.
 * - Prefill path (M>1) will eventually land here with tiled GEMM kernels.
 * - Until then, this entrypoint provides a safe hook for incremental rollout.
 */
enum PrefillInt8TileVariant : int
{
    PREFILL_TILE_16x16 = 0,
    PREFILL_TILE_32x8 = 1,
    PREFILL_TILE_8x32 = 2,
    PREFILL_TILE_8x8 = 3,
};

static const char *prefillTileVariantName(int variant)
{
    switch (variant)
    {
    case PREFILL_TILE_16x16:
        return "16x16";
    case PREFILL_TILE_32x8:
        return "32x8";
    case PREFILL_TILE_8x32:
        return "8x32";
    case PREFILL_TILE_8x8:
        return "8x8";
    default:
        return "unknown";
    }
}

static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;
    if (M <= 12 && N <= 512)
    {
        return PREFILL_TILE_8x8;
    }

    if (N >= 4 * M && N >= 1024)
    {
        return PREFILL_TILE_32x8;
    }

    if (M >= 4 * N && M >= 128)
    {
        return PREFILL_TILE_8x32;
    }

    if (grid_kpar && N >= 2 * M && N >= 2048)
    {
        return PREFILL_TILE_32x8;
    }

    return PREFILL_TILE_16x16;
}

static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

bool rocmGemm_int8_int8_int32_vnni_prefill(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                hipGetErrorString(err), M, N, K);
        return false;
    }

    return true;
}

/**
 * @brief Experimental INT8 VNNI prefill grid-kpar split-K variant.
 *
 * This variant uses grid.z to split K reduction into slices with INT32
 * atomic accumulation. It is opt-in from higher-level dispatch.
 */
bool rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int split_k_slices,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const int total_k_groups = K / 4;
    const int slices = max(1, min(split_k_slices, total_k_groups));
    const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
    hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
    if (memset_err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
        return false;
    }

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                hipGetErrorString(err), M, N, K, slices);
        return false;
    }

    return true;
}

/**
 * @brief Native ratio-VNNI prefill GEMM entrypoint (M>1), ABI v2.
 *
 * ABI v2 takes a versioned descriptor to allow future payload families
 * (different block sizes / side metadata) without changing this function
 * signature again.
 */
bool rocmGemm_ratio_vnni_int8_int32_prefill_v2(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const int8_t* d_ratio,
    const int8_t* d_min,
    int32_t* d_C_int32,
    int M, int N, int K,
    const llaminar2::rocm::ROCmRatioVNNIPrefillAbiV2Desc* abi,
    int prefill_variant,
    int prefill_kb,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_payload || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if (!abi)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Null ABI descriptor\n");
        return false;
    }

    if (abi->abi_version != llaminar2::rocm::ROCM_RATIO_VNNI_PREFILL_ABI_V2)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported abi_version=%u\n",
                static_cast<unsigned>(abi->abi_version));
        return false;
    }

    const bool is_q4_family = (abi->bitwidth == 4 && abi->block_size == 32 && abi->payload_bytes == 16);
    const bool is_q5_family = (abi->bitwidth == 5 && abi->block_size == 32 && abi->payload_bytes == 20);
    const bool is_i8_payload_family = (abi->bitwidth == 8 && abi->block_size == 32 && abi->payload_bytes == 32);
    if (!is_q4_family && !is_q5_family && !is_i8_payload_family)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported descriptor (bitwidth=%u has_min=%u block=%u payload=%u)\n",
                static_cast<unsigned>(abi->bitwidth), static_cast<unsigned>(abi->has_min),
                static_cast<unsigned>(abi->block_size), static_cast<unsigned>(abi->payload_bytes));
        return false;
    }

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const uint32_t expected_blocks_per_row = static_cast<uint32_t>(K / 32);
    if (abi->blocks_per_row != 0 && abi->blocks_per_row != expected_blocks_per_row)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] blocks_per_row mismatch (abi=%u expected=%u K=%d)\n",
                static_cast<unsigned>(abi->blocks_per_row), static_cast<unsigned>(expected_blocks_per_row), K);
        return false;
    }

    if (abi->payload_stride_bytes != abi->payload_bytes)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported payload stride (payload_stride=%u payload_bytes=%u)\n",
                static_cast<unsigned>(abi->payload_stride_bytes),
                static_cast<unsigned>(abi->payload_bytes));
        return false;
    }

    if (!is_i8_payload_family)
    {
        if (abi->ratio_bytes != 1 || abi->ratio_stride_bytes != 1 || !d_ratio)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported or missing ratio side-channel for bitwidth=%u (ratio_bytes=%u ratio_stride=%u d_ratio=%p)\n",
                    static_cast<unsigned>(abi->bitwidth),
                    static_cast<unsigned>(abi->ratio_bytes),
                    static_cast<unsigned>(abi->ratio_stride_bytes),
                    static_cast<const void*>(d_ratio));
            return false;
        }
    }
    else
    {
        if (abi->has_min != 0 || abi->codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] bitwidth=8 payload family requires has_min=0 and linear codebook (has_min=%u codebook_id=%u)\n",
                    static_cast<unsigned>(abi->has_min),
                    static_cast<unsigned>(abi->codebook_id));
            return false;
        }
    }

    if (abi->has_min != 0)
    {
        if (!d_min)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] has_min set but min side-channel pointer is null\n");
            return false;
        }
        if (abi->codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] has_min currently supports only linear codebook (codebook_id=%u)\n",
                    static_cast<unsigned>(abi->codebook_id));
            return false;
        }
        if (abi->min_bytes != 1 || abi->min_stride_bytes != 1)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported min side-channel config (min_bytes=%u min_stride=%u)\n",
                    static_cast<unsigned>(abi->min_bytes),
                    static_cast<unsigned>(abi->min_stride_bytes));
            return false;
        }
    }

    if (abi->codebook_id != RATIO_VNNI_CODEBOOK_IQ4 && abi->codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Unsupported codebook_id=%u\n", static_cast<unsigned>(abi->codebook_id));
        return false;
    }

    if (is_q5_family && abi->codebook_id != RATIO_VNNI_CODEBOOK_LINEAR)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] bitwidth=5 currently supports only linear codebook (codebook_id=%u)\n",
                static_cast<unsigned>(abi->codebook_id));
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (is_q4_family)
    {
        bool lut_ok = false;
        if (abi->has_min != 0)
        {
            lut_ok = ensure_ratio_linear_unsigned_perm_lut_table_on_device(device_id);
        }
        else if (abi->codebook_id == RATIO_VNNI_CODEBOOK_IQ4)
        {
            lut_ok = ensure_ratio_iq4_perm_lut_table_on_device(device_id);
        }
        else
        {
            lut_ok = ensure_ratio_linear_perm_lut_table_on_device(device_id);
        }

        if (!lut_ok)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Failed to initialize ratio perm LUT table on device\n");
            return false;
        }
    }

    int block_x = 16;
    int block_y = 16;
    if (prefill_variant >= 0)
    {
        switch (prefill_variant)
        {
        case 0:
            block_x = 16;
            block_y = 16;
            break;
        case 1:
            block_x = 32;
            block_y = 8;
            break;
        case 2:
            block_x = 8;
            block_y = 32;
            break;
        case 3:
            block_x = 8;
            block_y = 8;
            break;
        default:
            break;
        }
    }
    else
    {
        if (is_i8_payload_family)
        {
            // Auto-tuned default for payload-v2 bitwidth-8 families (Q*_K / IQ*).
            // Wider Y tile improves reuse of B payload rows across output rows.
            block_x = 8;
            block_y = 32;
        }
        else if (N >= 12288 && M <= 256)
        {
            block_x = 32;
            block_y = 8;
        }
        else if (M >= 384 && N <= 4096)
        {
            block_x = 8;
            block_y = 32;
        }
    }

    int kb = prefill_kb;
    if (kb <= 0)
    {
        if (is_i8_payload_family)
        {
            // Auto-tuned split-K for payload-v2 bitwidth=8 route.
            kb = 2;
        }
        else if (K >= 16384)
        {
            kb = 8;
        }
        else if (K >= 8192)
        {
            kb = 4;
        }
        else if (K >= 4096)
        {
            kb = 2;
        }
        else
        {
            kb = 1;
        }
    }
    kb = max(1, min(32, kb));

    if (kb > 1)
    {
        const hipError_t memset_err = hipMemsetAsync(
            d_C_int32,
            0,
            static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t),
            static_cast<hipStream_t>(stream));
        if (memset_err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] hipMemsetAsync failed: %s\n",
                    hipGetErrorString(memset_err));
            return false;
        }
    }

    const int payload_cpt = 1;

    const dim3 block(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y), 1);
    const int logical_tile_n = block_x * payload_cpt;
    const dim3 grid(
        static_cast<unsigned>((N + logical_tile_n - 1) / logical_tile_n),
        static_cast<unsigned>((M + block_y - 1) / block_y),
        static_cast<unsigned>(kb));

    if (is_q4_family && abi->codebook_id == RATIO_VNNI_CODEBOOK_IQ4)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_ratio_vnni_prefill_kernel<true, false>),
            grid,
            block,
            0,
            static_cast<hipStream_t>(stream),
            d_A_int8,
            d_payload,
            d_ratio,
                d_min,
            d_C_int32,
            M,
            N,
            K,
            abi->has_min,
            kb);
    }
    else if (is_q4_family)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_ratio_vnni_prefill_kernel<false, false>),
            grid,
            block,
            0,
            static_cast<hipStream_t>(stream),
            d_A_int8,
            d_payload,
            d_ratio,
                d_min,
            d_C_int32,
            M,
            N,
            K,
            abi->has_min,
            kb);
    }
    else if (is_q5_family)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_ratio_vnni_prefill_kernel<false, true>),
            grid,
            block,
            0,
            static_cast<hipStream_t>(stream),
            d_A_int8,
            d_payload,
            d_ratio,
            d_min,
            d_C_int32,
            M,
            N,
            K,
            abi->has_min,
            kb);
    }
    else
    {
        if (payload_cpt >= 4)
        {
            if (block_x == 32 && block_y == 8)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<32, 8, 4>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
            else if (block_x == 8 && block_y == 32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<8, 32, 4>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<16, 16, 4>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
        }
        else if (payload_cpt == 2)
        {
            if (block_x == 32 && block_y == 8)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<32, 8, 2>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
            else if (block_x == 8 && block_y == 32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<8, 32, 2>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemm_payload_v2_i8_prefill_kernel_t<16, 16, 2>),
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
        }
        else
        {
            const bool use_shared_payload = (N <= 8192);
            if (use_shared_payload)
            {
                hipLaunchKernelGGL(
                    gemm_payload_v2_i8_prefill_kernel,
                    grid,
                    block,
                    static_cast<size_t>(block_x) * 8u * sizeof(int32_t),
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
            else
            {
                hipLaunchKernelGGL(
                    gemm_payload_v2_i8_prefill_kernel_direct,
                    grid,
                    block,
                    0,
                    static_cast<hipStream_t>(stream),
                    d_A_int8,
                    d_payload,
                    d_C_int32,
                    M,
                    N,
                    K,
                    kb);
            }
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_ratio_vnni_int8_int32_prefill_v2] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                hipGetErrorString(err), M, N, K);
        return false;
    }

    return true;
}

/**
 * @brief Legacy ratio-VNNI prefill ABI wrapper.
 *
 * Maintains existing callsites while forwarding to descriptor-based ABI v2.
 */
bool rocmGemm_ratio_vnni_int8_int32_prefill(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const int8_t* d_ratio,
    const int8_t* d_min,
    int32_t* d_C_int32,
    int M, int N, int K,
    uint8_t bitwidth,
    uint8_t codebook_id,
    uint8_t has_min,
    uint8_t block_size,
    uint16_t payload_bytes,
    int prefill_variant,
    int prefill_kb,
    int device_id, void* stream)
{
    llaminar2::rocm::ROCmRatioVNNIPrefillAbiV2Desc abi{};
    abi.abi_version = llaminar2::rocm::ROCM_RATIO_VNNI_PREFILL_ABI_V2;
    abi.bitwidth = bitwidth;
    abi.codebook_id = codebook_id;
    abi.has_min = has_min;
    abi.block_size = block_size;
    abi.payload_bytes = payload_bytes;
    abi.ratio_bytes = 1;
    abi.min_bytes = (has_min != 0) ? 1 : 0;
    abi.flags = 0;
    abi.blocks_per_row = (block_size > 0) ? static_cast<uint32_t>(K / static_cast<int>(block_size)) : 0;
    abi.payload_stride_bytes = payload_bytes;
    abi.ratio_stride_bytes = 1;
    abi.min_stride_bytes = (has_min != 0) ? 1 : 0;

    return rocmGemm_ratio_vnni_int8_int32_prefill_v2(
        d_A_int8,
        d_payload,
        d_ratio,
        d_min,
        d_C_int32,
        M, N, K,
        &abi,
        prefill_variant,
        prefill_kb,
        device_id,
        stream);
}

} // extern "C"
