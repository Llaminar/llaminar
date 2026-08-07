/**
 * @file ROCmMoEGroupedPrefillKernels.hip
 * @brief Grouped MoE prefill kernels for AMD MI50/MI60 (gfx906)
 *
 * The pipeline is entirely device-owned and graph-capturable. It consumes the
 * route groups and descriptor tables already resident on the GPU; no launch
 * reads route counts on the host and no stream synchronization is required.
 *
 * Small verifier buckets (M <= 8) use direct route-owned projections because
 * their latency is dominated by launch and directory overhead. Longer prefill
 * uses a compact device-built tile directory and a generated policy trained
 * across every execution codebook, reciprocal aspect ratio, and M anchor:
 *
 *   1. Reuse router-owned Q8 rows, or quantize each original hidden row once.
 *   2. Build a packed expert/tile directory entirely on the device.
 *   3. Decode each gate/up weight block once for a trained 4/8/12/16-row tile
 *      while retaining serial M=1 partition boundaries and reduction order.
 *   4. Quantize exact SwiGLU rows for the down projection.
 *   5. Decode each down weight block once per independently trained row tile
 *      into route partials.
 *   6. Publish those partials with one writer per token/output column, adding
 *      routes in original top-k order.
 *
 * The long-prefill projection kernels expose 64/128/256 output-column lanes and
 * 4/8/12/16-row tiles. Production dispatch comes exclusively from the checked-in
 * all-format trainer output; a missing policy is a hard failure. No production
 * path uses floating-point atomics, row replay, replay-time host-authored work
 * metadata, or scheduler-dependent reduction order, so grouped verifier
 * publication remains byte-identical to serial row decode.
 *
 * @see ROCmGemvKernel_native_VNNI.hip for decode grouped kernels (M=1)
 * @see ROCmNativeVNNIGemmShardImpl.hip.inc for non-grouped prefill GEMM
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../common/NativeVNNIGroupedDecodePolicy.h"
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"
#include "ROCmNativeVNNIDecodeEquivalentMath.h"
#include "ROCmMoEGroupedPrefillDispatchGenerated.inc"
#include "ROCmMoEProductionPrefillOverlayGenerated.inc"

// Local copy of DeviceNativeVNNIMatrixDesc to avoid pulling in x86-only headers
// Must stay in sync with src/v2/tensors/TensorKernels.h
struct DeviceNativeVNNIMatrixDesc
{
    const uint8_t *payload;
    const void *scales;
    const void *mins;
    const void *emins;
    int n;
    int k;
    uint32_t blocks_per_row;
    uint8_t codebook_id;
    uint8_t reserved[3];
};

__device__ __forceinline__ void moe_prefill_fail_fast_invalid_grouped_index()
{
    __builtin_trap();
}

// NativeVNNIFormat enum (must match ROCmGemvKernel_native_VNNI.hip)
enum NativeVNNIFormatPrefill : uint8_t {
    NVNNI_PF_Q4_0    = 0,
    NVNNI_PF_IQ4_NL  = 4,
    NVNNI_PF_Q4_1    = 5,
    NVNNI_PF_Q5_0    = 6,
    NVNNI_PF_Q5_1    = 7,
    NVNNI_PF_Q6_K    = 8,
    NVNNI_PF_Q3_K    = 9,
    NVNNI_PF_Q2_K    = 10,
    NVNNI_PF_IQ3_S   = 11,
    NVNNI_PF_IQ3_XXS = 12,
    NVNNI_PF_IQ2_S   = 13,
    NVNNI_PF_IQ2_XS  = 14,
    NVNNI_PF_IQ2_XXS = 15,
    NVNNI_PF_IQ1_S   = 16,
    NVNNI_PF_IQ1_M   = 17,
    NVNNI_PF_Q8_0    = 19,
};

/**
 * @brief Capture-time serial-M1 split-K policy carried into grouped prefill.
 *
 * A NativeVNNI codebook and matrix geometry select a generated serial-decode
 * K-partition count. That count changes FP32 arithmetic because each partition
 * starts from zero and an ordered reducer then adds the independently rounded
 * partials. Grouped prefill therefore carries the complete codebook-indexed
 * policy by value in its graph-captured kernel arguments. Mixed-codebook down
 * tables can select the correct entry on device without consulting the host.
 */
struct MoEPrefillSerialM1KPartitionPolicy
{
    uint8_t by_codebook[32] = {};

    /** Return the captured partition count, or zero for an invalid codebook. */
    __host__ __device__ int forCodebook(uint8_t codebook_id) const
    {
        return codebook_id < 32 ? static_cast<int>(by_codebook[codebook_id]) : 0;
    }
};

extern "C" bool rocmGemv_native_vnni_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu);

// Compile-time format properties for prefill (mirrors NVNNITraits in GEMV)
template <NativeVNNIFormatPrefill FMT>
struct PrefillNVNNITraits {
    static constexpr bool is_asymmetric = (FMT == NVNNI_PF_Q4_1 || FMT == NVNNI_PF_Q5_1 || FMT == NVNNI_PF_IQ1_S);
    static constexpr bool is_iq3_grid = (FMT == NVNNI_PF_IQ3_S || FMT == NVNNI_PF_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NVNNI_PF_IQ2_S || FMT == NVNNI_PF_IQ2_XS || FMT == NVNNI_PF_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NVNNI_PF_IQ1_S || FMT == NVNNI_PF_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid);
    static constexpr int qh_bits_per_lookup =
        (FMT == NVNNI_PF_IQ2_S) ? 2 : (FMT == NVNNI_PF_IQ3_S || FMT == NVNNI_PF_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    static constexpr bool is_dual_scale = (FMT == NVNNI_PF_Q6_K || FMT == NVNNI_PF_Q3_K || FMT == NVNNI_PF_Q2_K
                                        || FMT == NVNNI_PF_IQ2_S || FMT == NVNNI_PF_IQ2_XS
                                        || FMT == NVNNI_PF_IQ1_M);
    static constexpr bool is_dual_scale_asym = (FMT == NVNNI_PF_Q2_K);
    static constexpr bool needs_emin_array = (FMT == NVNNI_PF_Q2_K);
    static constexpr bool is_5bit = (FMT == NVNNI_PF_Q5_0 || FMT == NVNNI_PF_Q5_1);
    static constexpr bool is_6bit = (FMT == NVNNI_PF_Q6_K);
    static constexpr bool is_3bit = (FMT == NVNNI_PF_Q3_K);
    static constexpr bool is_2bit = (FMT == NVNNI_PF_Q2_K);
    static constexpr bool is_8bit = (FMT == NVNNI_PF_Q8_0);
    static constexpr bool uses_lut = (FMT == NVNNI_PF_IQ4_NL);
    static constexpr int payload_bytes =
        is_8bit ? 32 :
        is_iq1_grid ? 6 :
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    static constexpr int center_4bit = (FMT == NVNNI_PF_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NVNNI_PF_Q5_0) ? 16 : 0;
};

template <NativeVNNIFormatPrefill FMT>
__device__ __forceinline__ bool prefill_native_vnni_desc_shape_ok(
    const DeviceNativeVNNIMatrixDesc& desc,
    int N,
    int K)
{
    using Traits = PrefillNVNNITraits<FMT>;
    return desc.payload &&
           desc.scales &&
           desc.n == N &&
           desc.k == K &&
           desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
           desc.codebook_id == static_cast<uint8_t>(FMT) &&
           (!(Traits::is_asymmetric || Traits::is_dual_scale) || desc.mins) &&
           (!Traits::is_dual_scale_asym || desc.emins);
}

// IQ grid lookup tables (separate copies for this TU — initialized via
// rocmInitIQGridTables_moe_prefill, same data as GEMV/GEMM TUs)
__device__ __constant__ uint32_t d_pf_iq3s_grid[512];
__device__ __constant__ uint32_t d_pf_iq3xxs_grid[256];
__device__ __constant__ uint64_t d_pf_iq2s_grid[1024];
__device__ __constant__ uint64_t d_pf_iq2xs_grid[512];
__device__ __constant__ uint64_t d_pf_iq2xxs_grid[256];
__device__ __constant__ uint64_t d_pf_iq1s_grid[2048];

// IQ3 grid lookup (compile-time format dispatch)
template <NativeVNNIFormatPrefill FMT>
__device__ inline uint32_t pf_iq3_grid_lookup(int idx)
{
    if constexpr (FMT == NVNNI_PF_IQ3_S) return d_pf_iq3s_grid[idx];
    else                                  return d_pf_iq3xxs_grid[idx];
}

// IQ2 grid lookup (compile-time format dispatch)
template <NativeVNNIFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ inline uint64_t pf_iq2_grid_lookup(
    int idx,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    if constexpr (FMT == NVNNI_PF_IQ2_S)
    {
        if constexpr (USE_STAGED_IQ2_S)
            return staged_iq2s_grid[idx];
        else
            return d_pf_iq2s_grid[idx];
    }
    else if constexpr (FMT == NVNNI_PF_IQ2_XS) return d_pf_iq2xs_grid[idx];
    else                                        return d_pf_iq2xxs_grid[idx];
}

// IQ1 grid lookup (shared for IQ1_S and IQ1_M)
__device__ inline uint64_t pf_iq1_grid_lookup(int idx)
{
    return d_pf_iq1s_grid[idx];
}

// Conditionally negate 4 unsigned bytes packed in uint32_t using 4 sign bits.
// sign_lo4 bit j → element j: 0=positive, 1=negative.
__device__ inline uint32_t pf_iq_apply_signs_4(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t spread = (static_cast<uint32_t>(sign_lo4 & 0xFu) * 0x00204081u) & 0x01010101u;
    uint32_t mask;
    asm("v_lshlrev_b32 %0, 8, %1\n\t"
        "v_sub_u32 %0, %0, %1"
        : "=v"(mask) : "v"(spread));
    return (grid4 ^ mask) + spread;
}


// ============================================================================
// Kernel 1: Original-Row Quantization (all tokens, single launch)
//
// Each routed expert consumes the same hidden row. Quantizing `[seq_len, K]`
// once avoids writing `top_k` byte-identical copies before expert grouping.
//
// Grid: dim3(blocks_per_row, seq_len)
//   blockIdx.x = quantization block within the row
//   blockIdx.y = original token row
// Block: 32 threads (warp-cooperative quantization)
// ============================================================================

__global__ __launch_bounds__(32)
void moe_prefill_hidden_quantize_rows_kernel(
    const float* __restrict__ hidden,            // [seq_len, d_model]
    int8_t*      __restrict__ A_int8_out,        // [seq_len, K] quantized original rows
    float*       __restrict__ scales_out,        // [seq_len, blocks_per_row] per-block scales
    int seq_len,
    int K)                                       // d_model (columns)
{
    const int block_in_row = blockIdx.x;   // which 32-element block in the row
    const int original_token = blockIdx.y;

    const int lane = threadIdx.x;  // 0..31
    const int k = block_in_row * 32 + lane;
    if (original_token >= seq_len || k >= K)
        return;

    // Load one element from the hidden state
    const float val = hidden[static_cast<size_t>(original_token) * K + k];

    // Warp-cooperative max reduction for quantization scale
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);
    const float inv_scale = 1.0f / scale;

    // Write scale (one per block, lane 0 writes)
    const int blocks_per_row = K / 32;
    if (lane == 0)
        scales_out[static_cast<size_t>(original_token) * blocks_per_row + block_in_row] = scale;

    // Quantize and write
    A_int8_out[static_cast<size_t>(original_token) * K + k] =
        static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8(
                val, inv_scale));
}


// ============================================================================
// Kernel 2: Grouped Gate+Up GEMM (all experts, single launch)
//
// Uses the GEMV-style approach (1 thread per N-column, iterate over K blocks)
// extended to M>1 via blockIdx.y for token indexing.
//
// Grid: dim3((N + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts)
//   blockIdx.x = N-tile (64 threads per tile)
//   blockIdx.y = token within expert
//   blockIdx.z = expert index
// Block: 64 threads
//
// Each thread computes one output element: output[slot, n] = dot(A_row, B_col_n)
// ============================================================================

template <NativeVNNIFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ void moe_prefill_decode_groups(
    const uint8_t* __restrict__ payload,
    int32_t* packed_groups,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNVNNITraits<FMT>;

    if constexpr (Traits::is_8bit) {
        // Q8_0: 32 raw INT8 values — direct load, no decode needed
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            packed_groups[g] = *reinterpret_cast<const int32_t*>(payload + g * 4);
    }
    else if constexpr (Traits::is_iq1_grid) {
        // IQ1_S / IQ1_M: 4 grid lookups × 8 signed elements {-1,0,+1} = 32 elements
        if constexpr (FMT == NVNNI_PF_IQ1_S) {
            const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                                   | (static_cast<uint16_t>(payload[5]) << 8);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
                const uint64_t grid8 = pf_iq1_grid_lookup(idx);
                packed_groups[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            }
        } else { // NVNNI_PF_IQ1_M
            const uint8_t qh0 = payload[4], qh1 = payload[5];
            const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
            const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
            const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
            const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);
            uint64_t g;
            g = pf_iq1_grid_lookup(idx0);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx1);
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx2);
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx3);
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
        }
    }
    else if constexpr (Traits::is_iq_grid) {
        // IQ3 and IQ2 grid formats — LUT lookup + sign application
        if constexpr (Traits::is_iq3_grid && !Traits::has_qh) {
            // IQ3_XXS: 8-byte qs + 4-byte signs = 12 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const uint32_t word = (g < 4) ? w0 : w1;
                const int idx = (word >> ((g & 3) * 8)) & 0xFF;
                const uint32_t grid4 = pf_iq3_grid_lookup<FMT>(idx);
                const uint8_t sb = static_cast<uint8_t>(w2 >> ((g / 2) * 8));
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(pf_iq_apply_signs_4(grid4, sn));
            }
        }
        else if constexpr (Traits::is_iq3_grid) {
            // IQ3_S: 8-byte qs + 1-byte qh + 4-byte signs = 13 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
            const uint8_t signs3 = payload[12];
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const uint32_t word = (g < 4) ? w0 : w1;
                int idx = (word >> ((g & 3) * 8)) & 0xFF;
                idx |= (static_cast<int>((w2 >> g) & 1) << 8);
                const uint32_t grid4 = pf_iq3_grid_lookup<FMT>(idx);
                const uint8_t sb = (g < 6)
                    ? static_cast<uint8_t>(w2 >> ((g / 2 + 1) * 8))
                    : signs3;
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(pf_iq_apply_signs_4(grid4, sn));
            }
        }
        else if constexpr (Traits::is_iq2_grid && !Traits::has_qh) {
            // IQ2_XXS: 4-byte qs + 4-byte signs = 8 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w_qs = pw[0], w_signs = pw[1];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = (w_qs >> (l * 8)) & 0xFF;
                const uint64_t grid8 =
                    pf_iq2_grid_lookup<FMT, USE_STAGED_IQ2_S>(
                        idx, staged_iq2s_grid);
                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t signs = static_cast<uint8_t>(w_signs >> (l * 8));
                packed_groups[l * 2]     = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_lo, signs & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_hi, signs >> 4));
            }
        }
        else if constexpr (Traits::is_iq2_grid) {
            // IQ2_S / IQ2_XS: 4-byte qs + 4-byte qh_signs + 1-byte signs3 = 9 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w_qs = pw[0], w_qh_signs = pw[1];
            const uint8_t signs3 = payload[8];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int idx = (w_qs >> (l * 8)) & 0xFF;
                if constexpr (Traits::qh_bits_per_lookup == 1)
                    idx |= (static_cast<int>((w_qh_signs >> l) & 1) << 8);
                else if constexpr (Traits::qh_bits_per_lookup == 2)
                    idx |= (static_cast<int>((w_qh_signs >> (2 * l)) & 3) << 8);
                const uint64_t grid8 =
                    pf_iq2_grid_lookup<FMT, USE_STAGED_IQ2_S>(
                        idx, staged_iq2s_grid);
                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t signs_byte = (l < 3)
                    ? static_cast<uint8_t>(w_qh_signs >> ((l + 1) * 8))
                    : signs3;
                packed_groups[l * 2]     = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_lo, signs_byte & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_hi, signs_byte >> 4));
            }
        }
    }
    else if constexpr (Traits::is_3bit) {
        // Q3_K: 12-byte payload [8B pre-transposed 2-bit][4B grouped hbits]
        constexpr uint32_t BIT_SPREAD = 0x00204081u;
        const uint32_t hbit_word = *reinterpret_cast<const uint32_t*>(payload + 8);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint32_t r = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t low2_4 = (r >> (g * 2)) & 0x03030303u;
                const uint32_t hb4 = (hbit_word >> ((half * 4 + g) * 4)) & 0xFu;
                const uint32_t hb = ((hb4 * BIT_SPREAD) & 0x01010101u) << 2;
                uint32_t val = low2_4 | hb;
                // Center: [0..7] → [-4..+3]
                val = (val ^ 0x80808080u) - 0x04040404u;
                val ^= 0x80808080u;
                packed_groups[half * 4 + g] = static_cast<int32_t>(val);
            }
        }
    }
    else if constexpr (Traits::is_2bit) {
        // Q2_K: 8-byte pre-transposed payload
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            const int bg = half * 4;
            packed_groups[bg]     = static_cast<int32_t>(raw & 0x03030303u);
            packed_groups[bg + 1] = static_cast<int32_t>((raw >> 2) & 0x03030303u);
            packed_groups[bg + 2] = static_cast<int32_t>((raw >> 4) & 0x03030303u);
            packed_groups[bg + 3] = static_cast<int32_t>((raw >> 6) & 0x03030303u);
        }
    }
    else if constexpr (Traits::is_5bit) {
        // Q5_0 / Q5_1: 20-byte payload [16B nibbles + 4B high bits]
        const uint32_t qh_bits = *reinterpret_cast<const uint32_t*>(payload + 16);
        if constexpr (Traits::center_5bit == 0) {
            // Q5_1: unsigned [0,31]
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                packed_groups[g] = static_cast<int32_t>(lo_nibs | hb_lo);
                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                     | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                packed_groups[g + 4] = static_cast<int32_t>(hi_nibs | hb_hi);
            }
        } else {
            // Q5_0: centered [-16, +15]
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                val_lo -= 0x10101010u;
                val_lo ^= 0x80808080u;
                packed_groups[g] = static_cast<int32_t>(val_lo);
                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                     | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                uint32_t val_hi = (hi_nibs | hb_hi) ^ 0x80808080u;
                val_hi -= 0x10101010u;
                val_hi ^= 0x80808080u;
                packed_groups[g + 4] = static_cast<int32_t>(val_hi);
            }
        }
    }
    else if constexpr (Traits::is_6bit) {
        // Q6_K: 24-byte payload [16B lo nibbles + 8B hi 2-bit fields]
        const uint8_t* qh_p = payload + 16;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            const uint32_t h_lo = qh_p[g];
            const uint32_t h_hi = qh_p[g + 4];
            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t hb_lo = ((h_lo & 0x03u) << 4)  | ((h_lo & 0x0Cu) << 10)
                                 | ((h_lo & 0x30u) << 16) | ((h_lo & 0xC0u) << 22);
            uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
            val_lo -= 0x20202020u;
            val_lo ^= 0x80808080u;
            packed_groups[g] = static_cast<int32_t>(val_lo);
            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            const uint32_t hb_hi = ((h_hi & 0x03u) << 4)  | ((h_hi & 0x0Cu) << 10)
                                 | ((h_hi & 0x30u) << 16) | ((h_hi & 0xC0u) << 22);
            uint32_t val_hi = (hi_nibs | hb_hi) ^ 0x80808080u;
            val_hi -= 0x20202020u;
            val_hi ^= 0x80808080u;
            packed_groups[g + 4] = static_cast<int32_t>(val_hi);
        }
    }
    else if constexpr (FMT == NVNNI_PF_IQ4_NL) {
        /*
         * IQ4_NL / IQ4_XS verifier prefill must follow the same v_perm decode
         * expression as the M=1 MoE row-decode helper.  The removed pair-LUT
         * produced the intended int8 values, but it gave this translation unit
         * a different expression tree and shifted FP32 verifier rows by ULPs.
         */
        constexpr uint32_t CB_LO_A = 0xBFAD9881u;
        constexpr uint32_t CB_LO_B = 0xF6EADDCFu;
        constexpr uint32_t CB_HI_A = 0x26190D01u;
        constexpr uint32_t CB_HI_B = 0x71594535u;
        constexpr uint32_t MUX_IDENTITY = 0x03020100u;
        constexpr uint32_t MUX_BIT3_MASK = 0x04040404u;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t lo_sel = lo_nibs & 0x07070707u;
            const uint32_t lo_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, lo_sel);
            const uint32_t lo_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, lo_sel);
            const uint32_t lo_mux = MUX_IDENTITY | ((lo_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g] =
                static_cast<int32_t>(__builtin_amdgcn_perm(lo_r1, lo_r0, lo_mux));

            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            const uint32_t hi_sel = hi_nibs & 0x07070707u;
            const uint32_t hi_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, hi_sel);
            const uint32_t hi_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, hi_sel);
            const uint32_t hi_mux = MUX_IDENTITY | ((hi_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g + 4] =
                static_cast<int32_t>(__builtin_amdgcn_perm(hi_r1, hi_r0, hi_mux));
        }
    }
    else if constexpr (Traits::center_4bit == 8) {
        // Q4_0: nibble - 8 → [-8, +7]
        constexpr uint32_t SWAR_ADD = 0x78787878u;
        constexpr uint32_t SWAR_XOR = 0x80808080u;
        constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g]     = static_cast<int32_t>(((raw & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
            packed_groups[g + 4] = static_cast<int32_t>((((raw >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
        }
    }
    else {
        // Q4_1: unsigned nibbles [0,15]
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
            packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
        }
    }
}

/**
 * @brief Accumulate one decoded NativeVNNI K block into an FP32 row result.
 *
 * Serial decode and grouped prefill must execute the same floating-point
 * expression for every row.  Keeping the expression in one helper is more than
 * code reuse: it prevents a row-tiled optimization from accidentally changing
 * the order of a dual-scale base term, an asymmetric minimum correction, or an
 * IQ1_M delta correction.  The caller may share the decoded weight payload
 * across several independent rows. Each serial partition starts from zero and
 * invokes this helper once per K block before its independently rounded result
 * enters the ordered partition reducer.
 *
 * @tparam FMT NativeVNNI codebook represented by @p desc.
 * @param desc Prepared device descriptor for one expert matrix.
 * @param linear Block-major descriptor index, `k_block * N + output_column`.
 * @param payload Pointer to the encoded weight payload for @p linear.
 * @param packed_groups Eight packed INT8 groups decoded from @p payload.
 * @param a4 Eight packed INT8 activation groups for the current row/K block.
 * @param scale_ab Per-row activation scale for the current K block.
 * @param acc Running FP32 accumulator for this row and output column.
 * @return The accumulator after exactly one serial-shaped K-block update.
 */
template <NativeVNNIFormatPrefill FMT>
__device__ __forceinline__ float moe_prefill_accumulate_decoded_block(
    const DeviceNativeVNNIMatrixDesc& desc,
    size_t linear,
    const uint8_t* __restrict__ payload,
    const int32_t* __restrict__ packed_groups,
    const int32_t* __restrict__ a4,
    float scale_ab,
    float acc)
{
    using Traits = PrefillNVNNITraits<FMT>;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);

    if constexpr (Traits::is_dual_scale)
    {
        int32_t acc_lo = 0;
        int32_t acc_hi = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
#pragma unroll
        for (int g = 4; g < 8; ++g)
            acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

        const uint16_t scale_lo_bits = scale_base[linear];
        const uint16_t scale_hi_bits = min_base[linear];
        const float scale_lo =
            __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
        const float scale_hi =
            __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
        acc = rocm_native_vnni_commit_dual_scale_block(
            acc,
            acc_lo,
            acc_hi,
            scale_lo,
            scale_hi,
            scale_ab);

        if constexpr (Traits::is_dual_scale_asym)
        {
            constexpr int32_t ones = 0x01010101;
            int32_t sum_a_lo = 0;
            int32_t sum_a_hi = 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                sum_a_lo = __builtin_amdgcn_sdot4(
                    a4[g], ones, sum_a_lo, false);
#pragma unroll
            for (int g = 4; g < 8; ++g)
                sum_a_hi = __builtin_amdgcn_sdot4(
                    a4[g], ones, sum_a_hi, false);

            const uint32_t emin_packed = emin_base[linear];
            const uint16_t emin_lo_bits =
                static_cast<uint16_t>(emin_packed);
            const uint16_t emin_hi_bits =
                static_cast<uint16_t>(emin_packed >> 16);
            const float min_lo = __half2float(
                *reinterpret_cast<const __half*>(&emin_lo_bits));
            const float min_hi = __half2float(
                *reinterpret_cast<const __half*>(&emin_hi_bits));
            acc = rocm_native_vnni_commit_dual_scale_correction(
                acc,
                sum_a_lo,
                sum_a_hi,
                min_lo,
                min_hi,
                scale_ab);
        }

        if constexpr (FMT == NVNNI_PF_IQ1_M)
        {
            constexpr float IQ1S_DELTA = 0.125f;
            constexpr int32_t ones = 0x01010101;
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];

            const int32_t sum_a_g0 = __builtin_amdgcn_sdot4(
                a4[1], ones,
                __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
            const int32_t sum_a_g1 = __builtin_amdgcn_sdot4(
                a4[3], ones,
                __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
            const int32_t sum_a_g2 = __builtin_amdgcn_sdot4(
                a4[5], ones,
                __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
            const int32_t sum_a_g3 = __builtin_amdgcn_sdot4(
                a4[7], ones,
                __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

            const float delta0 =
                (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta1 =
                (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta2 =
                (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta3 =
                (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            acc = rocm_native_vnni_commit_iq1_m_delta_correction(
                acc,
                sum_a_g0,
                sum_a_g1,
                sum_a_g2,
                sum_a_g3,
                delta0,
                delta1,
                delta2,
                delta3,
                scale_lo,
                scale_hi,
                scale_ab);
        }
    }
    else
    {
        int32_t dot = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g)
            dot = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], dot, false);

        const uint16_t scale_bits = scale_base[linear];
        const float block_d =
            __half2float(*reinterpret_cast<const __half*>(&scale_bits));
        acc = rocm_native_vnni_commit_single_scale_block(
            acc, dot, block_d, scale_ab);

        if constexpr (Traits::is_asymmetric)
        {
            constexpr int32_t ones = 0x01010101;
            int32_t sum_a = 0;
#pragma unroll
            for (int g = 0; g < 8; ++g)
                sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

            const uint16_t min_bits = min_base[linear];
            const float block_m =
                __half2float(*reinterpret_cast<const __half*>(&min_bits));
            acc = rocm_native_vnni_commit_single_scale_correction(
                acc, sum_a, block_m, scale_ab);
        }
    }

    return acc;
}

/**
 * @brief Compute one NativeVNNI dot product over a bounded K-block range.
 *
 * This is the canonical single-row expression used by grouped prefill and its
 * serial-row oracle.  Long-prefill kernels below share decoded weights across
 * rows but deliberately call the same per-block accumulator so optimization
 * cannot relax byte equality.
 */
template <NativeVNNIFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ float moe_prefill_dot_desc_range(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int b_start,
    int b_end,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNVNNITraits<FMT>;
    if (!slot_A ||
        !slot_scales ||
        n < 0 ||
        n >= N ||
        desc.n != N ||
        desc.blocks_per_row == 0 ||
        b_start < 0 ||
        b_end > static_cast<int>(desc.blocks_per_row) ||
        !prefill_native_vnni_desc_shape_ok<FMT>(
            desc, N, static_cast<int>(desc.blocks_per_row) * 32))
    {
        return 0.0f;
    }

    const uint8_t* payload_base = desc.payload;
    float acc = 0.0f;

    for (int b = b_start; b < b_end; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) +
                              static_cast<size_t>(n);
        int32_t packed_groups[8];
        constexpr int PB = Traits::payload_bytes;
        const uint8_t* payload = payload_base + linear * PB;
        moe_prefill_decode_groups<FMT, USE_STAGED_IQ2_S>(
            payload, packed_groups, staged_iq2s_grid);

        const int32_t* a4 = reinterpret_cast<const int32_t*>(slot_A + b * 32);
        acc = moe_prefill_accumulate_decoded_block<FMT>(
            desc, linear, payload, packed_groups, a4, slot_scales[b], acc);
    }

    return acc;
}

/**
 * @brief Reproduce serial M=1 split-K partials and their ordered reduction.
 *
 * This function keeps all partials local to the owning output lane. It does not
 * replay rows or materialize a global split-K workspace: each partition runs
 * the canonical block accumulator from zero, and the lane adds completed
 * partials in increasing partition order with the same explicit RN operation
 * used by serial decode's reducer kernel.
 *
 * @tparam FMT NativeVNNI codebook represented by @p desc.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use an LDS copy.
 * @param desc Prepared matrix descriptor.
 * @param n Output column owned by this lane.
 * @param slot_A Quantized activation row.
 * @param slot_scales Per-block activation scales.
 * @param N Matrix output width.
 * @param k_partitions Exact effective serial-M1 partition count.
 * @param staged_iq2s_grid Optional LDS-resident IQ2_S codebook.
 * @return Byte-equivalent serial-M1 result for this output element.
 */
template <NativeVNNIFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ float moe_prefill_dot_desc_serial_m1(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int k_partitions,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    const int blocks_per_row = static_cast<int>(desc.blocks_per_row);
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }

    if (k_partitions == 1)
    {
        return moe_prefill_dot_desc_range<FMT, USE_STAGED_IQ2_S>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            0,
            blocks_per_row,
            staged_iq2s_grid);
    }

    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    float total = 0.0f;
#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        const float partial = begin < end
                                  ? moe_prefill_dot_desc_range<
                                        FMT, USE_STAGED_IQ2_S>(
                                        desc,
                                        n,
                                        slot_A,
                                        slot_scales,
                                        N,
                                        begin,
                                        end,
                                        staged_iq2s_grid)
                                  : 0.0f;
        total = rocm_native_vnni_accumulate_rn(total, partial);
    }
    return total;
}

/**
 * @brief Integer and scale terms for one exact dual-scale K-block contribution.
 *
 * These are the operands consumed by
 * @ref rocm_native_vnni_commit_dual_scale_block. Keeping the FP32 accumulator
 * out of this structure lets several lanes prepare independent K blocks while
 * one owner lane still commits them in serial-decode order.
 */
struct MoEPrefillDualScaleBlockTerms
{
    int32_t low_dot = 0;       ///< INT32 dot for K elements 0..15.
    int32_t high_dot = 0;      ///< INT32 dot for K elements 16..31.
    float low_scale = 0.0f;    ///< FP16-derived low-half weight scale.
    float high_scale = 0.0f;   ///< FP16-derived high-half weight scale.
    float activation_scale = 0.0f; ///< Blockwise activation scale.
};

/**
 * @brief Prepare one correction-free dual-scale block without changing FP32 state.
 *
 * The payload decode and both `sdot4` chains are identical to
 * @ref moe_prefill_accumulate_decoded_block. Only the independent integer work
 * is separated from the serial FP32 commit so neighboring lanes can expose
 * enough wavefronts to occupy gfx906 for M=1..4 verifier buckets.
 *
 * @tparam FMT Correction-free dual-scale NativeVNNI format.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use the LDS copy.
 * @param desc Prepared matrix descriptor.
 * @param n Output column owned by the cooperating lane subgroup.
 * @param slot_A Quantized activation row, optionally staged in LDS.
 * @param slot_scales Activation block scales, optionally staged in LDS.
 * @param N Matrix output width.
 * @param block_index Increasing-K block index to prepare.
 * @return Exact integer dots and scale operands for the block.
 */
template <NativeVNNIFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ MoEPrefillDualScaleBlockTerms
moe_prefill_prepare_dual_scale_block_terms(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int block_index,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNVNNITraits<FMT>;
    static_assert(
        Traits::is_dual_scale &&
            !Traits::is_dual_scale_asym &&
            FMT != NVNNI_PF_IQ1_M,
        "subgroup preparation requires a correction-free dual-scale format");
    static_assert(!USE_STAGED_IQ2_S || FMT == NVNNI_PF_IQ2_S,
                  "only IQ2_S may consume the staged IQ2_S codebook");

    const size_t linear =
        static_cast<size_t>(block_index) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const uint8_t* payload =
        desc.payload + linear * static_cast<size_t>(Traits::payload_bytes);
    int32_t packed_groups[8];
    moe_prefill_decode_groups<FMT, USE_STAGED_IQ2_S>(
        payload, packed_groups, staged_iq2s_grid);

    const int32_t* a4 = reinterpret_cast<const int32_t*>(
        slot_A + static_cast<size_t>(block_index) * 32u);
    MoEPrefillDualScaleBlockTerms terms{};
#pragma unroll
    for (int group = 0; group < 4; ++group)
    {
        terms.low_dot = __builtin_amdgcn_sdot4(
            a4[group], packed_groups[group], terms.low_dot, false);
    }
#pragma unroll
    for (int group = 4; group < 8; ++group)
    {
        terms.high_dot = __builtin_amdgcn_sdot4(
            a4[group], packed_groups[group], terms.high_dot, false);
    }

    const uint16_t* low_scales =
        static_cast<const uint16_t*>(desc.scales);
    const uint16_t* high_scales =
        static_cast<const uint16_t*>(desc.mins);
    const uint16_t low_bits = low_scales[linear];
    const uint16_t high_bits = high_scales[linear];
    terms.low_scale =
        __half2float(*reinterpret_cast<const __half*>(&low_bits));
    terms.high_scale =
        __half2float(*reinterpret_cast<const __half*>(&high_bits));
    terms.activation_scale = slot_scales[block_index];
    return terms;
}

/**
 * @brief Compute one dual-scale output with parallel preparation and serial commits.
 *
 * `K_LANES` adjacent lanes own one output column. During each iteration they
 * prepare consecutive K blocks independently. Every lane then participates in
 * bit-preserving shuffles, while subgroup lane zero calls the canonical RN
 * commit helper for source lanes 0..K_LANES-1. Consequently the observable
 * FP32 expression remains block 0, block 1, ... exactly as in serial decode;
 * this is parallel work preparation, not a floating-point reduction.
 *
 * @tparam FMT Correction-free dual-scale NativeVNNI format.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use the LDS copy.
 * @tparam K_LANES Number of adjacent lanes cooperating on one output.
 * @param desc Prepared matrix descriptor.
 * @param n Output column shared by this subgroup.
 * @param slot_A Quantized activation row, normally staged in LDS.
 * @param slot_scales Activation block scales, normally staged in LDS.
 * @param N Matrix output width.
 * @param blocks_per_row Number of 32-element K blocks.
 * @param k_partitions Exact effective serial-M1 partition count.
 * @return Exact output on subgroup lane zero; other lanes return an unused zero.
 */
template <
    NativeVNNIFormatPrefill FMT,
    bool USE_STAGED_IQ2_S,
    int K_LANES>
__device__ __forceinline__ float moe_prefill_dual_scale_subgroup_dot(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int blocks_per_row,
    int k_partitions,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    static_assert(K_LANES == 2 || K_LANES == 4,
                  "dual-scale verifier subgroup width must be two or four lanes");
    using Traits = PrefillNVNNITraits<FMT>;
    static_assert(
        Traits::is_dual_scale &&
            !Traits::is_dual_scale_asym &&
            FMT != NVNNI_PF_IQ1_M,
        "subgroup dot requires a correction-free dual-scale format");
    const int subgroup_lane = static_cast<int>(threadIdx.x) & (K_LANES - 1);
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }

    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    float total = 0.0f;
#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        float partial = 0.0f;

        for (int first_block = begin;
             first_block < end;
             first_block += K_LANES)
        {
            const int block_index = first_block + subgroup_lane;
            MoEPrefillDualScaleBlockTerms local{};
            if (block_index < end)
            {
                local = moe_prefill_prepare_dual_scale_block_terms<
                    FMT, USE_STAGED_IQ2_S>(
                    desc,
                    n,
                    slot_A,
                    slot_scales,
                    N,
                    block_index,
                    staged_iq2s_grid);
            }

#pragma unroll
            for (int source_lane = 0; source_lane < K_LANES; ++source_lane)
            {
                const int32_t low_dot =
                    __shfl(local.low_dot, source_lane, K_LANES);
                const int32_t high_dot =
                    __shfl(local.high_dot, source_lane, K_LANES);
                const float low_scale =
                    __shfl(local.low_scale, source_lane, K_LANES);
                const float high_scale =
                    __shfl(local.high_scale, source_lane, K_LANES);
                const float activation_scale =
                    __shfl(local.activation_scale, source_lane, K_LANES);

                if (subgroup_lane == 0 &&
                    first_block + source_lane < end)
                {
                    partial = rocm_native_vnni_commit_dual_scale_block(
                        partial,
                        low_dot,
                        high_dot,
                        low_scale,
                        high_scale,
                        activation_scale);
                }
            }
        }

        if (subgroup_lane == 0)
        {
            total = k_partitions == 1
                        ? partial
                        : rocm_native_vnni_accumulate_rn(total, partial);
        }
    }

    return subgroup_lane == 0 ? total : 0.0f;
}

namespace
{
    /** Number of low bits reserved for an expert id in a packed work item. */
    constexpr uint32_t kMoEPrefillExpertBits = 9u;
    /**
     * @brief Maximum descriptor-table width represented by the work directory.
     *
     * Qwen 3.6 has 256 routed experts and one separate shared expert.  A
     * verifier may economically publish those nine selected routes through one
     * descriptor table, so the directory must admit expert id 256 rather than
     * treating 256 as a hard table-size limit.  Nine id bits represent ids
     * 0..511, hence 512 table entries without enlarging each packed work item.
     */
    constexpr int kMoEPrefillMaxExperts =
        static_cast<int>(uint32_t{1} << kMoEPrefillExpertBits);
    /** Maximum expert id representable by the compact device work directory. */
    constexpr uint32_t kMoEPrefillExpertMask =
        (uint32_t{1} << kMoEPrefillExpertBits) - 1u;
    /** Trainer/runtime tile-row marker for device-adaptive TM12/TM16 work. */
    constexpr int kMoEPrefillAdaptiveTileRows = 20;
    /** Number of count words preceding adaptive TM12 and TM16 spans. */
    constexpr int kMoEPrefillAdaptiveDirectoryHeaderWords = 2;
    /** Header word publishing the live compact TM12 span length. */
    constexpr int kMoEPrefillAdaptiveTile12CountWord = 0;
    /** Header word publishing the live compact TM16 span length. */
    constexpr int kMoEPrefillAdaptiveTile16CountWord = 1;
    /** Sentinel written to unused graph-captured work-directory entries. */
    constexpr uint32_t kMoEPrefillUnusedTile = UINT32_MAX;
}

/**
 * @brief Build a compact device-owned directory of populated expert row tiles.
 *
 * A graph-captured prefill launch cannot read the current maximum expert count
 * on the host.  Launching a rectangular
 * `num_experts * ceil(seq_len / TILE_M)` grid would preserve device ownership,
 * but most workgroups would immediately exit and long-prefill throughput would
 * collapse. This one-block planner performs a parallel inclusive scan over the
 * already device-resident grouping counts and writes only populated
 * `(expert, first_row)` tiles into @p directory.
 * The following projection kernels launch against a conservative fixed capacity
 * and skip the `UINT32_MAX` tail.
 *
 * Each directory entry uses nine low bits for the expert id and the remaining
 * bits for the first row local to that expert. Nine bits support descriptor
 * tables with up to 512 entries, including a 256-routed-plus-one-shared fused
 * verifier table, while still supporting millions of grouped rows. Planning is
 * deterministic, contains no host synchronization, and is safe to capture and
 * replay as part of the full GPU graph.
 *
 * @tparam TILE_M Number of consecutive grouped rows consumed by one projection
 *                 workgroup.
 */
template <int TILE_M>
__global__ __launch_bounds__(kMoEPrefillMaxExperts, 1)
void moe_grouped_build_prefill_tile_directory_kernel(
    const int* __restrict__ group_counts,
    uint32_t* __restrict__ directory,
    int num_experts,
    int max_directory_tiles)
{
    static_assert(TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill sweep supports row tiles 4/8/12/16");
    const int tid = threadIdx.x;

    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        max_directory_tiles <= 0)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    for (int tile = tid; tile < max_directory_tiles; tile += blockDim.x)
        directory[tile] = kMoEPrefillUnusedTile;

    __shared__ int inclusive_tile_counts[kMoEPrefillMaxExperts];
    int count = tid < num_experts ? group_counts[tid] : 0;
    if (count < 0)
    {
        if (tid < num_experts)
            moe_prefill_fail_fast_invalid_grouped_index();
        count = 0;
    }
    inclusive_tile_counts[tid] = (count + TILE_M - 1) / TILE_M;
    __syncthreads();

    // Hillis-Steele is deliberately used for this tiny 256-element directory:
    // its regular barriers are graph-capture friendly and much cheaper than a
    // lane-zero serial scan on gfx906.
    for (int stride = 1; stride < kMoEPrefillMaxExperts; stride <<= 1)
    {
        const int addend = tid >= stride
                               ? inclusive_tile_counts[tid - stride]
                               : 0;
        __syncthreads();
        inclusive_tile_counts[tid] += addend;
        __syncthreads();
    }

    const int total_tiles = inclusive_tile_counts[kMoEPrefillMaxExperts - 1];
    if (total_tiles > max_directory_tiles)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    if (tid < num_experts)
    {
        const int first_tile = tid == 0 ? 0 : inclusive_tile_counts[tid - 1];
        const int expert_tiles = (count + TILE_M - 1) / TILE_M;
        for (int tile = 0; tile < expert_tiles; ++tile)
        {
            const uint32_t first_row = static_cast<uint32_t>(tile * TILE_M);
            directory[first_tile + tile] =
                (first_row << kMoEPrefillExpertBits) |
                static_cast<uint32_t>(tid);
        }
    }
}

/** Projection role used by the device-owned adaptive tile planner. */
enum class MoEPrefillAdaptiveProjectionRole : int
{
    GateUp = 0,
    Down = 1,
};

/**
 * @brief Choose TM12 or TM16 from one expert's live grouped-row count.
 *
 * Gate/up makes one projection-wide choice. TM16 is used only when every active
 * expert contains exactly 16 rows, where it removes one third of the workgroups
 * without mixing a lower-occupancy specialization into the replay. Every other
 * live distribution uses TM12. This distinction matters on gfx906: the TM12
 * gate/up kernel occupies four blocks per CU at 64 VGPRs, whereas TM16 occupies
 * three at 72 VGPRs. Launching both for one skewed distribution costs more than
 * the handful of TM16 tiles save.
 *
 * Down projection is less register-bound, so it chooses the tile with fewer
 * padded row evaluations; ties favor TM16's greater weight reuse. These choices
 * change only grouping of independent rows. Each row still traverses K and
 * reduces partials in the serial-M1 order implemented by the fixed
 * specializations.
 *
 * @param count Number of live rows owned by one expert.
 * @param gateup_all_active_counts_are_16 Device-reduced projection-wide gate/up
 *        predicate. It is ignored for the down projection.
 * @return The exact fixed specialization that owns this expert's rows.
 */
template <MoEPrefillAdaptiveProjectionRole ROLE>
__device__ __forceinline__ int moe_prefill_adaptive_tile_rows(
    int count,
    bool gateup_all_active_counts_are_16)
{
    if constexpr (ROLE == MoEPrefillAdaptiveProjectionRole::GateUp)
        return gateup_all_active_counts_are_16 ? 16 : 12;

    const int padded_12 = ((count + 11) / 12) * 12;
    const int padded_16 = ((count + 15) / 16) * 16;
    return padded_12 < padded_16 ? 12 : 16;
}

/**
 * @brief Build compact TM12 and TM16 spans entirely from device row counts.
 *
 * One planner emits a two-word device header followed by independently compact
 * TM12 and TM16 spans. Each projection specialization receives its own span and
 * live count, so it never launches the other specialization's conservative
 * capacity merely to reject tagged entries. Both launches remain permanently
 * present in the captured graph; replay therefore needs neither a conditional
 * graph node nor host-authored route state.
 *
 * @tparam ROLE Gate/up or down projection selection policy.
 */
template <MoEPrefillAdaptiveProjectionRole ROLE>
__global__ __launch_bounds__(kMoEPrefillMaxExperts, 1)
void moe_grouped_build_adaptive_prefill_tile_directory_kernel(
    const int* __restrict__ group_counts,
    uint32_t* __restrict__ directory,
    int num_experts,
    int max_tile12_tiles,
    int max_tile16_tiles)
{
    const int tid = threadIdx.x;
    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        max_tile12_tiles <= 0 || max_tile16_tiles <= 0)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    __shared__ int inclusive_tile12_counts[kMoEPrefillMaxExperts];
    __shared__ int inclusive_tile16_counts[kMoEPrefillMaxExperts];
    int count = tid < num_experts ? group_counts[tid] : 0;
    if (count < 0)
    {
        if (tid < num_experts)
            moe_prefill_fail_fast_invalid_grouped_index();
        count = 0;
    }

    bool gateup_all_active_counts_are_16 = false;
    if constexpr (ROLE == MoEPrefillAdaptiveProjectionRole::GateUp)
    {
        /*
         * Reuse the TM12 prefix array for an exact projection-wide predicate
         * before populating the real directory counts. The scan is integer-only
         * and has a fixed synchronization tree, so no route-dependent atomic or
         * host observation enters the captured transaction.
         */
        inclusive_tile12_counts[tid] = count > 0 && count != 16 ? 1 : 0;
        __syncthreads();
        for (int stride = 1; stride < kMoEPrefillMaxExperts; stride <<= 1)
        {
            const int addend = tid >= stride
                                   ? inclusive_tile12_counts[tid - stride]
                                   : 0;
            __syncthreads();
            inclusive_tile12_counts[tid] += addend;
            __syncthreads();
        }
        gateup_all_active_counts_are_16 =
            inclusive_tile12_counts[kMoEPrefillMaxExperts - 1] == 0;
    }

    const int tile_rows = count > 0
                              ? moe_prefill_adaptive_tile_rows<ROLE>(
                                    count,
                                    gateup_all_active_counts_are_16)
                              : 12;
    const int expert_tiles =
        count > 0 ? (count + tile_rows - 1) / tile_rows : 0;
    inclusive_tile12_counts[tid] = tile_rows == 12 ? expert_tiles : 0;
    inclusive_tile16_counts[tid] = tile_rows == 16 ? expert_tiles : 0;
    __syncthreads();

    for (int stride = 1; stride < kMoEPrefillMaxExperts; stride <<= 1)
    {
        const int tile12_addend = tid >= stride
                                      ? inclusive_tile12_counts[tid - stride]
                                      : 0;
        const int tile16_addend = tid >= stride
                                      ? inclusive_tile16_counts[tid - stride]
                                      : 0;
        __syncthreads();
        inclusive_tile12_counts[tid] += tile12_addend;
        inclusive_tile16_counts[tid] += tile16_addend;
        __syncthreads();
    }

    const int total_tile12 =
        inclusive_tile12_counts[kMoEPrefillMaxExperts - 1];
    const int total_tile16 =
        inclusive_tile16_counts[kMoEPrefillMaxExperts - 1];
    if (total_tile12 > max_tile12_tiles || total_tile16 > max_tile16_tiles)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    if (tid == 0)
    {
        directory[kMoEPrefillAdaptiveTile12CountWord] =
            static_cast<uint32_t>(total_tile12);
        directory[kMoEPrefillAdaptiveTile16CountWord] =
            static_cast<uint32_t>(total_tile16);
    }

    if (tid < num_experts && count > 0)
    {
        const bool owns_tile16 = tile_rows == 16;
        const int first_tile =
            tid == 0
                ? 0
                : (owns_tile16
                       ? inclusive_tile16_counts[tid - 1]
                       : inclusive_tile12_counts[tid - 1]);
        uint32_t* const owned_directory =
            directory + kMoEPrefillAdaptiveDirectoryHeaderWords +
            (owns_tile16 ? max_tile12_tiles : 0);
        for (int tile = 0; tile < expert_tiles; ++tile)
        {
            const uint32_t first_row =
                static_cast<uint32_t>(tile * tile_rows);
            owned_directory[first_tile + tile] =
                (first_row << kMoEPrefillExpertBits) |
                static_cast<uint32_t>(tid);
        }
    }
}

/**
 * @brief Batch-invariant expert-tiled gate/up projection for long prefill.
 *
 * One workgroup owns one populated expert tile, one 64-column output tile, and
 * one projection role (`gate` or `up`).  The expert weight payload is decoded
 * once per K block and reused across `TILE_M` independent routed rows.  Each row
 * still visits K blocks in increasing order and updates its own FP32 accumulator
 * through @ref moe_prefill_accumulate_decoded_block, so the optimization changes
 * data reuse without changing a row's serial-decode arithmetic.
 *
 * The activation block is staged in a very small LDS tile.  This prevents all 64
 * output-column lanes from redundantly loading the same routed activation while
 * retaining four-wave occupancy on gfx906.  Gate and up use separate workgroups
 * in one launch (`blockIdx.z`) to keep their accumulator lifetimes disjoint and
 * avoid register pressure from halving MI50 occupancy.
 *
 * @tparam FMT NativeVNNI codebook for experts handled by this launch.
 * @tparam TILE_M Trained expert-row tile (4, 8, 12, or 16 rows).
 * @tparam TILE_N Trained output-column tile (64, 128, or 256 lanes).
 */
template <NativeVNNIFormatPrefill FMT, int TILE_M, int TILE_N>
__global__ __launch_bounds__(TILE_N, 2)
void moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const int* __restrict__ group_token_indices,
    const int* __restrict__ group_counts,
    const int* __restrict__ group_offsets,
    const uint32_t* __restrict__ work_directory,
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    int max_directory_tiles,
    int seq_len,
    int top_k,
    int grouped_indices_are_route_slots,
    int input_is_original_rows,
    int N,
    int K,
    int k_partitions,
    int num_experts,
    const uint32_t* __restrict__ adaptive_tile_count)
{
    static_assert(TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill gate/up supports row tiles 4/8/12/16");
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "grouped-prefill gate/up supports column tiles 64/128/256");
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int directory_index = static_cast<int>(blockIdx.y);
    const int projection_role = static_cast<int>(blockIdx.z);
    if (directory_index >= max_directory_tiles || n >= N || projection_role > 1)
        return;
    if (adaptive_tile_count &&
        directory_index >= static_cast<int>(*adaptive_tile_count))
    {
        return;
    }

    uint32_t packed_tile = work_directory[directory_index];
    if (packed_tile == kMoEPrefillUnusedTile)
        return;

    const int expert_id =
        static_cast<int>(packed_tile & kMoEPrefillExpertMask);
    const int first_row =
        static_cast<int>(packed_tile >> kMoEPrefillExpertBits);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    const int count = group_counts[expert_id];
    const int grouped_offset = group_offsets[expert_id];
    if (first_row < 0 || first_row >= count || grouped_offset < 0)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const int rows_in_tile = min(TILE_M, count - first_row);
    const int first_grouped_slot = grouped_offset + first_row;

    const DeviceNativeVNNIMatrixDesc desc =
        projection_role == 0 ? gate_descs[expert_id] : up_descs[expert_id];
    if (!prefill_native_vnni_desc_shape_ok<FMT>(desc, N, K))
        return;

    float* projection_out = projection_role == 0 ? gate_out : up_out;
    const int blocks_per_row = K / 32;
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const uint8_t* payload_base = desc.payload;

    __shared__ __align__(16) int32_t staged_A[TILE_M * 8];
    __shared__ float staged_scales[TILE_M];
    float totals[TILE_M] = {};
    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;

#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        float partials[TILE_M] = {};
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        for (int b = begin; b < end; ++b)
        {
            const size_t linear =
                static_cast<size_t>(b) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            constexpr int payload_bytes = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* payload = payload_base + linear * payload_bytes;
            int32_t packed_groups[8];
            moe_prefill_decode_groups<FMT>(payload, packed_groups);

            const int staged_words = rows_in_tile * 8;
            for (int word = threadIdx.x; word < staged_words; word += blockDim.x)
            {
                const int local_row = word >> 3;
                const int group = word & 7;
                const int grouped_slot = first_grouped_slot + local_row;
                int input_row = grouped_slot;
                if (input_is_original_rows)
                {
                    const int source_index = group_token_indices[grouped_slot];
                    input_row = grouped_indices_are_route_slots
                                    ? (source_index / top_k)
                                    : source_index;
                }
                if (input_row < 0 ||
                    input_row >= (input_is_original_rows ? seq_len : INT32_MAX))
                {
                    moe_prefill_fail_fast_invalid_grouped_index();
                    return;
                }
                const int32_t* input_words = reinterpret_cast<const int32_t*>(
                    A_int8 + static_cast<size_t>(input_row) * static_cast<size_t>(K) +
                    static_cast<size_t>(b) * 32u);
                staged_A[word] = input_words[group];
            }
            for (int local_row = threadIdx.x;
                 local_row < rows_in_tile;
                 local_row += blockDim.x)
            {
                const int grouped_slot = first_grouped_slot + local_row;
                int input_row = grouped_slot;
                if (input_is_original_rows)
                {
                    const int source_index = group_token_indices[grouped_slot];
                    input_row = grouped_indices_are_route_slots
                                    ? (source_index / top_k)
                                    : source_index;
                }
                if (input_row < 0 ||
                    input_row >= (input_is_original_rows ? seq_len : INT32_MAX))
                {
                    moe_prefill_fail_fast_invalid_grouped_index();
                    return;
                }
                staged_scales[local_row] = scales_A_blockwise[
                    static_cast<size_t>(input_row) * static_cast<size_t>(blocks_per_row) +
                    static_cast<size_t>(b)];
            }
            __syncthreads();

#pragma unroll
            for (int local_row = 0; local_row < TILE_M; ++local_row)
            {
                if (local_row >= rows_in_tile)
                    break;
                partials[local_row] =
                    moe_prefill_accumulate_decoded_block<FMT>(
                        desc,
                        linear,
                        payload,
                        packed_groups,
                        staged_A + local_row * 8,
                        staged_scales[local_row],
                        partials[local_row]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int local_row = 0; local_row < TILE_M; ++local_row)
        {
            if (local_row >= rows_in_tile)
                break;
            totals[local_row] = k_partitions == 1
                                    ? partials[local_row]
                                    : rocm_native_vnni_accumulate_rn(
                                          totals[local_row], partials[local_row]);
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < TILE_M; ++local_row)
    {
        if (local_row >= rows_in_tile)
            break;
        const int grouped_slot = first_grouped_slot + local_row;
        projection_out[
            static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
            static_cast<size_t>(n)] = totals[local_row];
    }
}


/**
 * @brief Decode-equivalent grouped gate/up projection for routed prefill rows.
 *
 * Grouped publication must match rowwise serial decode byte-for-byte for every
 * prefill bucket, not only MTP verifier rows. One block owns one original route
 * slot, one trained output-column tile, and one projection role.
 * `blockIdx.z == 0` computes gate while `blockIdx.z == 1` computes up. The two
 * projections have no arithmetic dependency, so exposing them as independent
 * workgroups improves occupancy without changing either projection's serial-M1
 * partition tree or adding another graph launch. The route-to-grouped inverse
 * map skips non-local routes and directs valid results to compact expert
 * scratch, so the launch has exactly `2 * seq_len * top_k` projection-row work
 * rather than a sparse expert-by-max-M rectangle. FP32 gate/up scratch is written exactly
 * like the M=1 decode-table path before one grouped SwiGLU quantization pass
 * consumes it.
 */
template <
    NativeVNNIFormatPrefill FMT,
    int TILE_N = 64,
    bool STAGE_IQ2_S_GRID = false,
    int K_LANES = 1>
__global__ __launch_bounds__(TILE_N * K_LANES)
void moe_grouped_gate_up_decode_equivalent_prefill_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, K]
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, K / 32]
    const int*     __restrict__ original_to_grouped, // [total_slots] original route -> grouped slot
    const int*     __restrict__ original_expert_ids, // [total_slots], null only for singleton expert
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs, // [num_experts]
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,   // [num_experts]
    float*         __restrict__ gate_out,            // [total_slots, N]
    float*         __restrict__ up_out,              // [total_slots, N]
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    int k_partitions,
    int num_experts)
{
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "route-owned grouped gate/up supports measured gfx906 column tiles");
    static_assert(!STAGE_IQ2_S_GRID || FMT == NVNNI_PF_IQ2_S,
                  "only IQ2_S may consume the staged IQ2_S codebook");
    static_assert(K_LANES == 1 || K_LANES == 2 || K_LANES == 4,
                  "route-owned grouped gate/up supports one, two, or four K lanes");
    static_assert(
        K_LANES == 1 ||
            FMT == NVNNI_PF_Q6_K ||
            FMT == NVNNI_PF_IQ2_S,
        "only tuned correction-free dual-scale formats may use subgroup preparation");
    static_assert(TILE_N * K_LANES <= 256,
                  "route-owned grouped gate/up exceeds the tuned gfx906 block size");

    /*
     * IQ2_S performs four data-dependent 64-bit codebook lookups for every
     * 32-element K block. With one-wave workgroups those vector-memory reads
     * were repeated by every route/projection wave even though the complete
     * codebook is only 8 KiB. The IQ2_S specialization uses four waves per
     * workgroup and cooperatively copies the immutable table into LDS once.
     * Every output lane then reads the identical uint64_t entries from LDS;
     * its dot products, dual-scale commits, corrections, partition boundaries,
     * and ordered partition reduction are unchanged.
     *
     * Dynamic shared storage is zero bytes for every other codebook, so this
     * specialization cannot silently tax their occupancy or alter their
     * established launch geometry.
     */
    constexpr size_t kIQ2SGridBytes = 1024u * sizeof(uint64_t);
    constexpr size_t kStagedCodebookBytes =
        STAGE_IQ2_S_GRID ? kIQ2SGridBytes : 0u;
    extern __shared__ uint8_t staged_bytes[];
    uint64_t* staged_iq2s_grid =
        reinterpret_cast<uint64_t*>(staged_bytes);
    if constexpr (STAGE_IQ2_S_GRID)
    {
        constexpr int kIQ2SGridEntries = 1024;
        for (int i = threadIdx.x; i < kIQ2SGridEntries; i += blockDim.x)
            staged_iq2s_grid[i] = d_pf_iq2s_grid[i];
    }

    const int output_lane = static_cast<int>(threadIdx.x) / K_LANES;
    const int n = blockIdx.x * TILE_N + output_lane;
    const int original_slot = blockIdx.y;
    const int projection_role = static_cast<int>(blockIdx.z);
    if (original_slot >= total_slots || projection_role > 1)
        return;

    const int grouped_slot = original_to_grouped[original_slot];
    if (grouped_slot < 0 || grouped_slot >= total_slots)
        return;

    const int expert_id = original_expert_ids
                              ? original_expert_ids[original_slot]
                              : ((num_experts == 1) ? 0 : -1);
    if (expert_id < 0 || expert_id >= num_experts)
        return;

    const DeviceNativeVNNIMatrixDesc desc =
        projection_role == 0 ? gate_descs[expert_id] : up_descs[expert_id];
    if (!prefill_native_vnni_desc_shape_ok<FMT>(desc, N, K))
        return;

    const int blocks_per_row = K / 32;
    int input_row = grouped_slot;
    if (input_is_original_rows)
    {
        input_row = original_slot / top_k;
        if (input_row < 0 || input_row >= seq_len)
            return;
    }
    const int8_t* slot_A = A_int8 + static_cast<size_t>(input_row) * K;
    const float* slot_scales =
        scales_A_blockwise + static_cast<size_t>(input_row) * blocks_per_row;

    /*
     * The subgroup geometry makes activation addresses lane-varying, so relying
     * on scalar-cache broadcast would turn four unique block reads into
     * redundant vector traffic. One 256-thread workgroup instead copies the
     * activation row and its scale row into launch LDS before any lane exits for
     * a partial N tile. IQ2_S places these rows immediately after its immutable
     * codebook. Both publications share one barrier, avoiding an extra
     * workgroup-wide rendezvous while retaining an explicit producer/consumer
     * edge for every staged byte.
     */
    if constexpr (K_LANES > 1)
    {
        int8_t* staged_A = reinterpret_cast<int8_t*>(
            staged_bytes + kStagedCodebookBytes);
        float* staged_scales = reinterpret_cast<float*>(
            staged_bytes + kStagedCodebookBytes + static_cast<size_t>(K));
        for (int index = static_cast<int>(threadIdx.x);
             index < K;
             index += static_cast<int>(blockDim.x))
        {
            staged_A[index] = slot_A[index];
        }
        for (int index = static_cast<int>(threadIdx.x);
             index < blocks_per_row;
             index += static_cast<int>(blockDim.x))
        {
            staged_scales[index] = slot_scales[index];
        }
        slot_A = staged_A;
        slot_scales = staged_scales;
    }

    if constexpr (STAGE_IQ2_S_GRID || K_LANES > 1)
        __syncthreads();

    if (n >= N)
        return;

    const size_t out_index =
        static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
        static_cast<size_t>(n);

    float* projection_out = projection_role == 0 ? gate_out : up_out;
    if constexpr (K_LANES > 1)
    {
        const float value = moe_prefill_dual_scale_subgroup_dot<
            FMT, STAGE_IQ2_S_GRID, K_LANES>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            blocks_per_row,
            k_partitions,
            staged_iq2s_grid);
        if ((static_cast<int>(threadIdx.x) & (K_LANES - 1)) == 0)
            projection_out[out_index] = value;
    }
    else
    {
        projection_out[out_index] =
            moe_prefill_dot_desc_serial_m1<FMT, STAGE_IQ2_S_GRID>(
                desc,
                n,
                slot_A,
                slot_scales,
                N,
                k_partitions,
                staged_iq2s_grid);
    }
}

/**
 * @brief Launch one compile-time route-owned gate/up geometry.
 *
 * This helper is the single host boundary for verifier gate/up geometry.  It
 * computes persistent LDS capacity before graph capture and launches on the
 * caller's exact stream; graph replay performs no host policy lookup.  Every
 * specialization invokes the same byte-exact kernel body above.
 *
 * @tparam FMT NativeVNNI execution codebook.
 * @tparam TILE_N Output columns owned by one workgroup before K subdivision.
 * @tparam K_LANES Adjacent lanes preparing blocks within one serial partition.
 * @param stream Non-null graph-capture stream that owns this launch.
 * @param A_int8 Quantized input rows.
 * @param scales_A_blockwise Per-row activation scales.
 * @param original_to_grouped Original-route to compact-route mapping.
 * @param original_expert_ids Expert id for every original route.
 * @param gate_descs Prepared gate descriptors.
 * @param up_descs Prepared up descriptors.
 * @param gate_out Compact gate output rows.
 * @param up_out Compact up output rows.
 * @param total_slots Number of original route slots.
 * @param seq_len Number of original token rows.
 * @param top_k Routed experts per original token.
 * @param input_is_original_rows Whether activation rows are token-owned.
 * @param N Projection output width.
 * @param K Projection reduction width.
 * @param k_partitions Exact effective serial-M1 partition count.
 * @param num_experts Number of descriptors in each table.
 */
template <
    NativeVNNIFormatPrefill FMT,
    int TILE_N,
    int K_LANES>
void launch_moe_grouped_gate_up_route_owned_geometry(
    hipStream_t stream,
    const int8_t* A_int8,
    const float* scales_A_blockwise,
    const int* original_to_grouped,
    const int* original_expert_ids,
    const DeviceNativeVNNIMatrixDesc* gate_descs,
    const DeviceNativeVNNIMatrixDesc* up_descs,
    float* gate_out,
    float* up_out,
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    int k_partitions,
    int num_experts)
{
    constexpr bool kStageIQ2SGrid = FMT == NVNNI_PF_IQ2_S;
    constexpr bool kStageActivation = K_LANES > 1;
    const size_t lds_bytes =
        (kStageIQ2SGrid ? 1024u * sizeof(uint64_t) : 0u) +
        (kStageActivation
             ? static_cast<size_t>(K) +
                   static_cast<size_t>(K / 32) * sizeof(float)
             : 0u);
    const dim3 grid((N + TILE_N - 1) / TILE_N, total_slots, 2);
    const dim3 block(TILE_N * K_LANES);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_gate_up_decode_equivalent_prefill_kernel<
            FMT, TILE_N, kStageIQ2SGrid, K_LANES>),
        grid,
        block,
        lds_bytes,
        stream,
        A_int8,
        scales_A_blockwise,
        original_to_grouped,
        original_expert_ids,
        gate_descs,
        up_descs,
        gate_out,
        up_out,
        total_slots,
        seq_len,
        top_k,
        input_is_original_rows,
        N,
        K,
        k_partitions,
        num_experts);
}

/**
 * @brief Select the measured route-owned geometry once, during graph capture.
 *
 * IQ2_S is codebook-stage limited at tiny route counts and K-loop limited once
 * the route grid broadens.  Measurements on gfx906 select four K lanes for at
 * most sixteen route slots and two K lanes above that boundary. Q6_K has no
 * 8-KiB codebook publication and retains its four-lane winner. Other formats
 * preserve one-lane serial-K execution until an exact profiled specialization
 * proves a different geometry.
 */
template <NativeVNNIFormatPrefill FMT>
void launch_moe_grouped_gate_up_route_owned(
    hipStream_t stream,
    const int8_t* A_int8,
    const float* scales_A_blockwise,
    const int* original_to_grouped,
    const int* original_expert_ids,
    const DeviceNativeVNNIMatrixDesc* gate_descs,
    const DeviceNativeVNNIMatrixDesc* up_descs,
    float* gate_out,
    float* up_out,
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    int k_partitions,
    int num_experts)
{
    if constexpr (FMT == NVNNI_PF_IQ2_S)
    {
        if (total_slots <= 16)
        {
            launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 4>(
                stream, A_int8, scales_A_blockwise, original_to_grouped,
                original_expert_ids, gate_descs, up_descs, gate_out, up_out,
                total_slots, seq_len, top_k, input_is_original_rows,
                N, K, k_partitions, num_experts);
        }
        else
        {
            launch_moe_grouped_gate_up_route_owned_geometry<FMT, 128, 2>(
                stream, A_int8, scales_A_blockwise, original_to_grouped,
                original_expert_ids, gate_descs, up_descs, gate_out, up_out,
                total_slots, seq_len, top_k, input_is_original_rows,
                N, K, k_partitions, num_experts);
        }
    }
    else if constexpr (FMT == NVNNI_PF_Q6_K)
    {
        launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 4>(
            stream, A_int8, scales_A_blockwise, original_to_grouped,
            original_expert_ids, gate_descs, up_descs, gate_out, up_out,
            total_slots, seq_len, top_k, input_is_original_rows,
            N, K, k_partitions, num_experts);
    }
    else
    {
        launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 1>(
            stream, A_int8, scales_A_blockwise, original_to_grouped,
            original_expert_ids, gate_descs, up_descs, gate_out, up_out,
            total_slots, seq_len, top_k, input_is_original_rows,
            N, K, k_partitions, num_experts);
    }
}


// ============================================================================
// Kernel 3: Grouped SwiGLU + Quantize (all experts, single launch)
//
// FLAT GRID: grid(blocks_per_intermediate, total_slots) — no empty blocks.
// Each block handles one 32-element quantization block of one slot.
//
// Grid: dim3(blocks_per_intermediate_row, total_slots)
// Block: 32 threads
// ============================================================================

__global__ __launch_bounds__(32)
void moe_grouped_swiglu_quantize_prefill_kernel(
    const float* __restrict__ gate_data,         // [total_slots, intermediate]
    const float* __restrict__ up_data,           // [total_slots, intermediate]
    int8_t*      __restrict__ A_int8_out,        // [total_slots, intermediate] quantized SwiGLU output
    float*       __restrict__ scales_out,        // [total_slots, blocks_per_intermediate]
    int intermediate)                            // expert intermediate dimension
{
    const int block_in_row = blockIdx.x;
    const int global_slot = blockIdx.y;

    const int lane = threadIdx.x;
    const int k = block_in_row * 32 + lane;
    if (k >= intermediate) return;

    // Use the exact serial-decode exponential and reciprocal instruction tree.
    const size_t idx = static_cast<size_t>(global_slot) * intermediate + k;
    const float g = gate_data[idx];
    const float u = up_data[idx];
    const float val =
        llaminar2::rocm::decode_equivalent::swigluValue(g, u);

    // Warp-cooperative max reduction for quantization
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);
    const float inv_scale = 1.0f / scale;

    const int blocks_per_row = intermediate / 32;
    if (lane == 0)
        scales_out[static_cast<size_t>(global_slot) * blocks_per_row + block_in_row] = scale;

    A_int8_out[static_cast<size_t>(global_slot) * intermediate + k] =
        static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8(
                val, inv_scale));
}

/**
 * @brief Batch-invariant expert-tiled down projection for long prefill.
 *
 * The kernel mirrors the gate/up row tile: one decoded expert weight block is
 * reused across several grouped SwiGLU rows, while every row owns an independent
 * accumulator whose K-block order matches serial decode.  Results are written to
 * per-route FP32 rows in @p down_partials.  A following ordered publisher reads
 * those rows in original router order; separating projection from publication is
 * what permits weight reuse without reintroducing nondeterministic FP32 atomics.
 *
 * @tparam FMT NativeVNNI codebook for experts handled by this launch.
 * @tparam TILE_M Trained expert-row tile (4, 8, 12, or 16 rows).
 * @tparam TILE_N Trained output-column tile (64, 128, or 256 lanes).
 */
template <NativeVNNIFormatPrefill FMT, int TILE_M, int TILE_N>
__global__ __launch_bounds__(TILE_N, 2)
void moe_grouped_down_batch_invariant_tiled_prefill_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ down_descs,
    const int* __restrict__ group_counts,
    const int* __restrict__ group_offsets,
    const uint32_t* __restrict__ work_directory,
    float* __restrict__ down_partials,
    int max_directory_tiles,
    int N,
    int K,
    int k_partitions,
    int num_experts,
    const uint32_t* __restrict__ adaptive_tile_count)
{
    static_assert(TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill down supports row tiles 4/8/12/16");
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "grouped-prefill down supports column tiles 64/128/256");
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int directory_index = static_cast<int>(blockIdx.y);
    if (directory_index >= max_directory_tiles || n >= N)
        return;
    if (adaptive_tile_count &&
        directory_index >= static_cast<int>(*adaptive_tile_count))
    {
        return;
    }

    uint32_t packed_tile = work_directory[directory_index];
    if (packed_tile == kMoEPrefillUnusedTile)
        return;

    const int expert_id =
        static_cast<int>(packed_tile & kMoEPrefillExpertMask);
    const int first_row =
        static_cast<int>(packed_tile >> kMoEPrefillExpertBits);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    const int count = group_counts[expert_id];
    const int grouped_offset = group_offsets[expert_id];
    if (first_row < 0 || first_row >= count || grouped_offset < 0)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const int rows_in_tile = min(TILE_M, count - first_row);
    const int first_grouped_slot = grouped_offset + first_row;

    const DeviceNativeVNNIMatrixDesc desc = down_descs[expert_id];
    if (!prefill_native_vnni_desc_shape_ok<FMT>(desc, N, K))
        return;

    const int blocks_per_row = K / 32;
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const uint8_t* payload_base = desc.payload;
    __shared__ __align__(16) int32_t staged_A[TILE_M * 8];
    __shared__ float staged_scales[TILE_M];
    float totals[TILE_M] = {};
    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;

#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        float partials[TILE_M] = {};
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        for (int b = begin; b < end; ++b)
        {
            const size_t linear =
                static_cast<size_t>(b) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            constexpr int payload_bytes = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* payload = payload_base + linear * payload_bytes;
            int32_t packed_groups[8];
            moe_prefill_decode_groups<FMT>(payload, packed_groups);

            const int staged_words = rows_in_tile * 8;
            for (int word = threadIdx.x; word < staged_words; word += blockDim.x)
            {
                const int local_row = word >> 3;
                const int group = word & 7;
                const int grouped_slot = first_grouped_slot + local_row;
                const int32_t* input_words = reinterpret_cast<const int32_t*>(
                    A_int8 + static_cast<size_t>(grouped_slot) * static_cast<size_t>(K) +
                    static_cast<size_t>(b) * 32u);
                staged_A[word] = input_words[group];
            }
            for (int local_row = threadIdx.x;
                 local_row < rows_in_tile;
                 local_row += blockDim.x)
            {
                const int grouped_slot = first_grouped_slot + local_row;
                staged_scales[local_row] = scales_A_blockwise[
                    static_cast<size_t>(grouped_slot) * static_cast<size_t>(blocks_per_row) +
                    static_cast<size_t>(b)];
            }
            __syncthreads();

#pragma unroll
            for (int local_row = 0; local_row < TILE_M; ++local_row)
            {
                if (local_row >= rows_in_tile)
                    break;
                partials[local_row] =
                    moe_prefill_accumulate_decoded_block<FMT>(
                        desc,
                        linear,
                        payload,
                        packed_groups,
                        staged_A + local_row * 8,
                        staged_scales[local_row],
                        partials[local_row]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int local_row = 0; local_row < TILE_M; ++local_row)
        {
            if (local_row >= rows_in_tile)
                break;
            totals[local_row] = k_partitions == 1
                                    ? partials[local_row]
                                    : rocm_native_vnni_accumulate_rn(
                                          totals[local_row], partials[local_row]);
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < TILE_M; ++local_row)
    {
        if (local_row >= rows_in_tile)
            break;
        const int grouped_slot = first_grouped_slot + local_row;
        down_partials[
            static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
            static_cast<size_t>(n)] = totals[local_row];
    }
}

/**
 * @brief Publish precomputed down rows in original top-k order.
 *
 * Each lane owns exactly one token/output-column result.  It walks the router's
 * original route slots, resolves each slot to its grouped partial row, and
 * executes `total += weight * partial` in the same order as serial decode.  The
 * kernel has one writer per output value, so publication is deterministic and
 * contains no floating-point atomics.
 */
__global__ __launch_bounds__(64, 4)
void moe_grouped_down_partials_ordered_publish_prefill_kernel(
    const float* __restrict__ down_partials,
    const int* __restrict__ original_to_grouped,
    const float* __restrict__ grouped_weights,
    float* __restrict__ output,
    int seq_len,
    int top_k,
    int d_model)
{
    constexpr int TILE_N = 64;
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int token = static_cast<int>(blockIdx.y);
    if (token >= seq_len || n >= d_model)
        return;

    float total = 0.0f;
#pragma unroll 1
    for (int route = 0; route < top_k; ++route)
    {
        const int original_slot = token * top_k + route;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0)
            continue;
        const float weighted_route = rocm_native_vnni_weight_route_rn(
            grouped_weights[grouped_slot],
            down_partials[
                static_cast<size_t>(grouped_slot) *
                    static_cast<size_t>(d_model) +
                static_cast<size_t>(n)]);
        total = rocm_native_vnni_accumulate_rn(total, weighted_route);
    }

    output[
        static_cast<size_t>(token) * static_cast<size_t>(d_model) +
        static_cast<size_t>(n)] = total;
}


/**
 * @brief Dispatch one NativeVNNI down dot by the descriptor's codebook id.
 *
 * The verifier publication kernel below must walk original route slots in a
 * single thread so it can preserve serial decode's `route 0, route 1, ...`
 * accumulation order.  Production descriptor tables are allowed to contain
 * mixed down codebooks across experts, so launching one templated kernel per
 * codebook would either overwrite earlier partial sums or reintroduce atomic
 * ordering.  This small device-side switch keeps the ordered publication kernel
 * generic while still reusing the exact same per-codebook dot helper as the
 * grouped gate/up kernels and serial decode.
 */
__device__ __forceinline__ float moe_prefill_dot_desc_serial_m1_dispatch(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int k_partitions)
{
    switch (static_cast<NativeVNNIFormatPrefill>(desc.codebook_id))
    {
    case NVNNI_PF_Q4_0:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q4_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ4_NL:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ4_NL>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q4_1:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q4_1>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q5_0:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q5_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q5_1:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q5_1>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q6_K:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q6_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q3_K:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q3_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q2_K:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q2_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ3_S:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ3_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ3_XXS:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ3_XXS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ2_S:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ2_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ2_XS:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ2_XS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ2_XXS:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ2_XXS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ1_S:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ1_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_IQ1_M:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_IQ1_M>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NVNNI_PF_Q8_0:
        return moe_prefill_dot_desc_serial_m1<NVNNI_PF_Q8_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    default:
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }
}

/** Sentinel selecting the first-class heterogeneous descriptor-table kernel. */
constexpr int kMoEPrefillMixedCodebookTag = -1;

/**
 * @brief Resolve one down-projection dot under a capture-time codebook policy.
 *
 * Homogeneous prepared tables instantiate the exact NativeVNNI format, letting
 * HIP inline only that decoder and discard the per-output format switch. A
 * genuinely heterogeneous table instantiates @ref kMoEPrefillMixedCodebookTag
 * and performs the descriptor-owned dispatch intentionally.
 *
 * @tparam FORMAT_TAG NativeVNNI format value, or the mixed-table sentinel.
 */
template <int FORMAT_TAG>
__device__ __forceinline__ float moe_prefill_down_dot(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int blocks_per_row,
    const MoEPrefillSerialM1KPartitionPolicy& k_partition_policy)
{
    if constexpr (FORMAT_TAG == kMoEPrefillMixedCodebookTag)
    {
        return moe_prefill_dot_desc_serial_m1_dispatch(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            k_partition_policy.forCodebook(desc.codebook_id));
    }
    else
    {
        constexpr auto kFormat =
            static_cast<NativeVNNIFormatPrefill>(FORMAT_TAG);
        if (!prefill_native_vnni_desc_shape_ok<kFormat>(
                desc, N, blocks_per_row * 32))
        {
            /*
             * A homogeneous mask and a mismatched descriptor cannot describe a
             * valid captured graph. Trap on device instead of publishing a
             * plausible zero row and allowing inference to limp onward.
             */
            moe_prefill_fail_fast_invalid_grouped_index();
            return 0.0f;
        }
        return moe_prefill_dot_desc_serial_m1<kFormat>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            k_partition_policy.forCodebook(
                static_cast<uint8_t>(kFormat)));
    }
}

/**
 * @brief Batch-invariant ordered down projection and publication.
 *
 * This is the canonical ROCm grouped-prefill down contract for every M bucket.
 * For each token/output column, one thread walks the original route slots in
 * top-k order, computes the down dot for that route's expert, applies the route
 * weight, and accumulates into a single FP32 total. No per-route down rows are
 * materialized and no atomic add participates in the result, so full prefill,
 * split prefill, and row-by-row decode all retain identical arithmetic order.
 *
 * The kernel is still a true grouped implementation: all request tokens, all
 * route slots, and all active experts are published by one graph-capturable
 * device dispatch. The only serial loop is the tiny top-k route loop inside
 * each token/column lane, exactly where serial decode defines the mathematical
 * accumulation order.
 *
 * @tparam FORMAT_TAG NativeVNNI format value, or the mixed-table sentinel.
 */
template <int FORMAT_TAG>
__global__ __launch_bounds__(64)
void moe_grouped_down_publish_original_token_ordered_prefill_kernel(
    const int8_t* __restrict__ swiglu_int8,          // [total_slots, intermediate]
    const float* __restrict__ swiglu_scales,         // [total_slots, intermediate / 32]
    const DeviceNativeVNNIMatrixDesc* __restrict__ down_descs, // [num_experts]
    const int* __restrict__ original_to_grouped,     // [seq_len * top_k]
    const int* __restrict__ original_expert_ids,     // [seq_len * top_k], null when num_experts == 1
    const float* __restrict__ grouped_weights,       // [total_slots]
    float* __restrict__ output,                      // [seq_len, d_model], overwritten
    int seq_len,
    int top_k,
    int N,
    int K,
    int num_experts,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= seq_len || n >= N)
        return;

    const int blocks_per_row = K / 32;
    float total = 0.0f;
#pragma unroll 1
    for (int route = 0; route < top_k; ++route)
    {
        const int original_slot = token * top_k + route;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0)
            continue;

        const int expert_id = original_expert_ids
                                  ? original_expert_ids[original_slot]
                                  : ((num_experts == 1) ? 0 : -1);
        if (expert_id < 0 || expert_id >= num_experts)
            continue;

        const DeviceNativeVNNIMatrixDesc desc = down_descs[expert_id];
        const int8_t* slot_A = swiglu_int8 + static_cast<size_t>(grouped_slot) * K;
        const float* slot_scales =
            swiglu_scales + static_cast<size_t>(grouped_slot) * blocks_per_row;
        const float expert_acc = moe_prefill_down_dot<FORMAT_TAG>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            blocks_per_row,
            k_partition_policy);
        const float weighted_route = rocm_native_vnni_weight_route_rn(
            grouped_weights[grouped_slot], expert_acc);
        total = rocm_native_vnni_accumulate_rn(total, weighted_route);
    }

    output[static_cast<size_t>(token) * static_cast<size_t>(N) + static_cast<size_t>(n)] = total;
}

/**
 * @brief Publish one weighted grouped down row per original router slot.
 *
 * Invalid or participant-remote slots explicitly write zero. The route axis is
 * preserved through the LocalTP collective and is reduced only afterward by
 * the canonical router-order epilogue.
 *
 * @tparam FORMAT_TAG NativeVNNI format value, or the mixed-table sentinel.
 */
template <int FORMAT_TAG>
__global__ __launch_bounds__(64)
void moe_grouped_down_publish_canonical_routes_prefill_kernel(
    const int8_t* __restrict__ swiglu_int8,
    const float* __restrict__ swiglu_scales,
    const DeviceNativeVNNIMatrixDesc* __restrict__ down_descs,
    const int* __restrict__ original_to_grouped,
    const int* __restrict__ original_expert_ids,
    const float* __restrict__ grouped_weights,
    float* __restrict__ route_output,
    int total_slots,
    int N,
    int K,
    int num_experts,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int original_slot = blockIdx.y;
    if (original_slot >= total_slots || n >= N)
        return;

    const size_t output_index =
        static_cast<size_t>(original_slot) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int grouped_slot = original_to_grouped[original_slot];
    const int expert_id = original_expert_ids
                              ? original_expert_ids[original_slot]
                              : ((num_experts == 1) ? 0 : -1);
    if (grouped_slot < 0 || expert_id < 0 || expert_id >= num_experts)
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const DeviceNativeVNNIMatrixDesc desc = down_descs[expert_id];
    const int8_t* slot_A =
        swiglu_int8 + static_cast<size_t>(grouped_slot) * K;
    const float* slot_scales =
        swiglu_scales + static_cast<size_t>(grouped_slot) * blocks_per_row;
    const float expert_acc = moe_prefill_down_dot<FORMAT_TAG>(
        desc,
        n,
        slot_A,
        slot_scales,
        N,
        blocks_per_row,
        k_partition_policy);
    route_output[output_index] = rocm_native_vnni_weight_route_rn(
        grouped_weights[grouped_slot], expert_acc);
}

/** @brief Scatter expert-tiled down partials into original route slots. */
__global__ __launch_bounds__(64, 4)
void moe_grouped_down_partials_canonical_publish_prefill_kernel(
    const float* __restrict__ down_partials,
    const int* __restrict__ original_to_grouped,
    const float* __restrict__ grouped_weights,
    float* __restrict__ route_output,
    int total_slots,
    int d_model)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int original_slot = blockIdx.y;
    if (original_slot >= total_slots || n >= d_model)
        return;

    const int grouped_slot = original_to_grouped[original_slot];
    const size_t output_index =
        static_cast<size_t>(original_slot) *
            static_cast<size_t>(d_model) +
        static_cast<size_t>(n);
    route_output[output_index] =
        grouped_slot < 0
            ? 0.0f
            : rocm_native_vnni_weight_route_rn(
                  grouped_weights[grouped_slot],
                  down_partials[
                      static_cast<size_t>(grouped_slot) *
                          static_cast<size_t>(d_model) +
                      static_cast<size_t>(n)]);
}

/** @brief Sum allreduced route slots in serial router order. */
__global__ __launch_bounds__(256)
void moe_reduce_canonical_route_contributions_kernel(
    const float* __restrict__ route_contributions,
    float* __restrict__ output,
    int seq_len,
    int top_k,
    int d_model)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= seq_len || n >= d_model)
        return;

    float sum = 0.0f;
#pragma unroll 1
    for (int route = 0; route < top_k; ++route)
    {
        const size_t index =
            (static_cast<size_t>(token) * static_cast<size_t>(top_k) +
             static_cast<size_t>(route)) *
                static_cast<size_t>(d_model) +
            static_cast<size_t>(n);
        sum = rocm_native_vnni_accumulate_rn(
            sum, route_contributions[index]);
    }
    output[
        static_cast<size_t>(token) * static_cast<size_t>(d_model) +
        static_cast<size_t>(n)] = sum;
}

// ============================================================================
// C API: Full grouped MoE prefill pipeline (graph-capturable)
//
// Launches one fixed, fully grouped pipeline on the caller's stream. Tiny MTP
// buckets use direct route-owned projection. Long prefill builds a compact
// device work directory, reuses decoded expert weights across row tiles, and
// publishes materialized down rows in original top-k order. No path uses FP32
// atomics, host synchronization, or replay-time host-authored work metadata.
// ============================================================================

namespace
{
    /** Projection role with an independently trained grouped-prefill policy. */
    enum class MoEPrefillProjectionRole : uint8_t
    {
        GateUp,
        Down,
    };

    /** Compile-time candidate selected for one graph-captured projection. */
    struct MoEPrefillTileConfig
    {
        int rows = 0;
        int columns = 0;
    };

    /** Joint gate/up and down policy chosen once for one captured MoE graph. */
    struct MoEPrefillPairConfig
    {
        MoEPrefillTileConfig gateup{};
        MoEPrefillTileConfig down{};
        bool exact_overlay = false;
    };

    bool moe_prefill_valid_tile_rows(int rows)
    {
        return rows == 4 || rows == 8 || rows == 12 || rows == 16 ||
               rows == kMoEPrefillAdaptiveTileRows;
    }

    bool moe_prefill_valid_tile_columns(int columns)
    {
        return columns == 64 || columns == 128 || columns == 256;
    }

    /** Return the sole execution codebook in a descriptor mask, if any. */
    int moe_prefill_single_codebook(uint32_t codebook_mask)
    {
        if (codebook_mask == 0 ||
            (codebook_mask & (codebook_mask - 1u)) != 0u)
        {
            return -1;
        }

        int codebook = 0;
        while ((codebook_mask >>= 1u) != 0u)
            ++codebook;
        return codebook;
    }

    /**
     * @brief Resolve every codebook's exact serial-M1 split-K policy for capture.
     *
     * The generated decode selector is queried only while the graph is being
     * constructed. The resulting fixed-size value becomes an ordinary captured
     * kernel argument, so graph replay performs no host lookup or transfer. A
     * missing or out-of-range policy is fatal to graph construction rather than
     * silently changing the arithmetic tree.
     */
    bool moe_prefill_resolve_serial_m1_k_partitions(
        uint32_t codebook_mask,
        int n,
        int k,
        MoEPrefillSerialM1KPartitionPolicy& policy)
    {
        if (codebook_mask == 0 || n <= 0 || k <= 0 || (k % 32) != 0)
            return false;

        bool resolved_any = false;
        for (uint8_t codebook_id = 0; codebook_id < 32; ++codebook_id)
        {
            if ((codebook_mask &
                 (uint32_t{1} << static_cast<uint32_t>(codebook_id))) == 0)
            {
                continue;
            }

            int k_partitions = 0;
            if (!rocmGemv_native_vnni_query_serial_m1_config(
                    codebook_id,
                    n,
                    k,
                    &k_partitions,
                    nullptr) ||
                k_partitions <= 0 ||
                k_partitions > k / 32 ||
                k_partitions >
                    llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
            {
                return false;
            }

            policy.by_codebook[codebook_id] =
                static_cast<uint8_t>(k_partitions);
            resolved_any = true;
        }
        return resolved_any;
    }

    /** Apply explicit trainer controls after generated production selection. */
    void moe_prefill_apply_tile_override(
        MoEPrefillProjectionRole role,
        MoEPrefillTileConfig &config)
    {
        const auto &env = llaminar2::debugEnv().rocm;
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            if (env.moe_prefill_gateup_tile_m > 0)
                config.rows = env.moe_prefill_gateup_tile_m;
            if (env.moe_prefill_gateup_tile_n > 0)
                config.columns = env.moe_prefill_gateup_tile_n;
            return;
        }

        if (env.moe_prefill_down_tile_m > 0)
            config.rows = env.moe_prefill_down_tile_m;
        if (env.moe_prefill_down_tile_n > 0)
            config.columns = env.moe_prefill_down_tile_n;
    }

    /**
     * @brief Select a grouped projection candidate for capture or trainer use.
     *
     * Production is selected exclusively by the complete generated
     * codebook/M/aspect-ratio table. Role-specific environment overrides are
     * intentionally explicit trainer controls: they let the Perf__ sweep hold
     * down projection fixed while training gate/up, then reverse the experiment,
     * without introducing a second benchmark-only implementation. A generated
     * policy miss returns an invalid zero tile and is rejected by the caller;
     * there is no silent generic production fallback.
     */
    MoEPrefillTileConfig moe_prefill_select_tile_config(
        MoEPrefillProjectionRole role,
        NativeVNNIFormatPrefill format,
        int m,
        int n,
        int k)
    {
        MoEPrefillTileConfig config{};
        llaminar2::rocm::generated::ROCmMoEGroupedPrefillConfig generated{};
        const uint8_t role_id =
            role == MoEPrefillProjectionRole::GateUp ? 0u : 1u;
        if (!llaminar2::rocm::generated::selectROCmMoEGroupedPrefillGenerated(
                static_cast<uint8_t>(format), role_id, m, n, k, generated))
        {
            return config;
        }
        config.rows = generated.tile_m;
        config.columns = generated.tile_n;
        moe_prefill_apply_tile_override(role, config);
        return config;
    }

    /**
     * @brief Select one joint production policy for a uniform-codebook pair.
     *
     * Exact production overlays have precedence over the complementary generic
     * role tables. A key miss is expected for an unseen geometry or positive M
     * and therefore retains the total generic policy. By contrast, a malformed
     * exact entry remains selected and invalid so the launch fails hard instead
     * of silently hiding a corrupt generated artifact.
     */
    MoEPrefillPairConfig moe_prefill_select_pair_config(
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int m,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k)
    {
        MoEPrefillPairConfig pair{};
        const int gateup_codebook =
            moe_prefill_single_codebook(gateup_codebook_mask);
        const int down_codebook =
            moe_prefill_single_codebook(down_codebook_mask);
        if (gateup_codebook < 0 || down_codebook < 0)
            return pair;

        pair.gateup = moe_prefill_select_tile_config(
            MoEPrefillProjectionRole::GateUp,
            static_cast<NativeVNNIFormatPrefill>(gateup_codebook),
            m,
            expert_width,
            hidden_size);
        pair.down = moe_prefill_select_tile_config(
            MoEPrefillProjectionRole::Down,
            static_cast<NativeVNNIFormatPrefill>(down_codebook),
            m,
            hidden_size,
            expert_width);

        llaminar2::rocm::generated::ROCmMoEProductionPrefillOverlayConfig
            exact{};
        if (!llaminar2::rocm::generated::selectROCmMoEProductionPrefillOverlay(
                static_cast<uint8_t>(gateup_codebook),
                static_cast<uint8_t>(down_codebook),
                hidden_size,
                expert_width,
                expert_count,
                top_k,
                m,
                exact))
        {
            return pair;
        }

        pair.gateup = {
            .rows = exact.gateup_tile_m,
            .columns = exact.gateup_tile_n,
        };
        pair.down = {
            .rows = exact.down_tile_m,
            .columns = exact.down_tile_n,
        };
        pair.exact_overlay = true;
        moe_prefill_apply_tile_override(
            MoEPrefillProjectionRole::GateUp, pair.gateup);
        moe_prefill_apply_tile_override(
            MoEPrefillProjectionRole::Down, pair.down);
        return pair;
    }

    int moe_prefill_directory_capacity(
        int total_slots,
        int num_experts,
        int tile_rows)
    {
        const int active_expert_bound =
            total_slots < num_experts ? total_slots : num_experts;
        const int conservative_bound =
            active_expert_bound + (total_slots + tile_rows - 1) / tile_rows;
        return conservative_bound < total_slots
                   ? conservative_bound
                   : total_slots;
    }

    /**
     * @brief Read immutable compiler resources for one HIP kernel function.
     *
     * Candidate training calls this before the first launch.  In particular,
     * `localSizeBytes` is the runtime's authoritative static indication that
     * the compiler assigned per-thread scratch/local memory.  Such a candidate
     * is not eligible for timing because register spilling is a launch-policy
     * defect, not a performance feature for the learner to rediscover.
     */
    bool moe_prefill_query_function_resources(
        const void* function,
        int block_threads,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        if (!function || block_threads <= 0 || !registers_per_thread ||
            !local_memory_bytes_per_thread || !static_shared_memory_bytes ||
            !max_threads_per_block || !max_active_blocks_per_sm)
        {
            return false;
        }

        hipFuncAttributes attributes{};
        if (hipFuncGetAttributes(&attributes, function) != hipSuccess)
            return false;

        int active_blocks = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                function,
                block_threads,
                /*dynamicSMemSize=*/0) != hipSuccess ||
            active_blocks <= 0)
        {
            return false;
        }

        *registers_per_thread = attributes.numRegs;
        *local_memory_bytes_per_thread = attributes.localSizeBytes;
        *static_shared_memory_bytes = attributes.sharedSizeBytes;
        *max_threads_per_block = attributes.maxThreadsPerBlock;
        *max_active_blocks_per_sm = active_blocks;
        return attributes.numRegs > 0 &&
               attributes.maxThreadsPerBlock >= block_threads;
    }

    /**
     * @brief Resolve one role-specific template specialization for inspection.
     */
    template <NativeVNNIFormatPrefill FMT, int TILE_M, int TILE_N>
    bool moe_prefill_query_candidate_resources(
        MoEPrefillProjectionRole role,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        const void* function = nullptr;
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<
                    FMT, TILE_M, TILE_N>));
        }
        else
        {
            function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                moe_grouped_down_batch_invariant_tiled_prefill_kernel<
                    FMT, TILE_M, TILE_N>));
        }
        return moe_prefill_query_function_resources(
            function,
            TILE_N,
            registers_per_thread,
            local_memory_bytes_per_thread,
            static_shared_memory_bytes,
            max_threads_per_block,
            max_active_blocks_per_sm);
    }

    /**
     * @brief Inspect every kernel embedded by one adaptive projection choice.
     *
     * Adaptive dispatch is a single capture-time policy choice, but its replay
     * consists of the device directory planner followed by the TM12 and TM16
     * projection specializations. The trainer must reject the choice when any
     * member spills or cannot sustain useful occupancy, so this query reports
     * the maximum pressure and minimum execution ceiling across all three
     * kernels. Reporting only one projection specialization would let an
     * uninspected kernel enter the production graph through a misleadingly
     * healthy candidate record.
     *
     * @tparam FMT NativeVNNI codebook used by both projection kernels.
     * @tparam TILE_N Output-column width and projection block size.
     * @param role Gate/up or down projection role.
     * @param registers_per_thread Receives the largest VGPR allocation.
     * @param local_memory_bytes_per_thread Receives the largest spill footprint.
     * @param static_shared_memory_bytes Receives the largest LDS allocation.
     * @param max_threads_per_block Receives the smallest launch ceiling.
     * @param max_active_blocks_per_sm Receives the smallest occupancy ceiling.
     * @return true only when the planner and both projections can be inspected.
     */
    template <NativeVNNIFormatPrefill FMT, int TILE_N>
    bool moe_prefill_query_adaptive_candidate_resources(
        MoEPrefillProjectionRole role,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        int registers[3] = {};
        size_t local_memory[3] = {};
        size_t shared_memory[3] = {};
        int max_threads[3] = {};
        int active_blocks[3] = {};

        if (!moe_prefill_query_candidate_resources<FMT, 12, TILE_N>(
                role,
                &registers[0],
                &local_memory[0],
                &shared_memory[0],
                &max_threads[0],
                &active_blocks[0]) ||
            !moe_prefill_query_candidate_resources<FMT, 16, TILE_N>(
                role,
                &registers[1],
                &local_memory[1],
                &shared_memory[1],
                &max_threads[1],
                &active_blocks[1]))
        {
            return false;
        }

        const void* planner =
            role == MoEPrefillProjectionRole::GateUp
                ? reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                      moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                          MoEPrefillAdaptiveProjectionRole::GateUp>))
                : reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                      moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                          MoEPrefillAdaptiveProjectionRole::Down>));
        if (!moe_prefill_query_function_resources(
                planner,
                kMoEPrefillMaxExperts,
                &registers[2],
                &local_memory[2],
                &shared_memory[2],
                &max_threads[2],
                &active_blocks[2]))
        {
            return false;
        }

        *registers_per_thread = registers[0];
        *local_memory_bytes_per_thread = local_memory[0];
        *static_shared_memory_bytes = shared_memory[0];
        *max_threads_per_block = max_threads[0];
        *max_active_blocks_per_sm = active_blocks[0];
        for (int member = 1; member < 3; ++member)
        {
            if (registers[member] > *registers_per_thread)
                *registers_per_thread = registers[member];
            if (local_memory[member] > *local_memory_bytes_per_thread)
                *local_memory_bytes_per_thread = local_memory[member];
            if (shared_memory[member] > *static_shared_memory_bytes)
                *static_shared_memory_bytes = shared_memory[member];
            if (max_threads[member] < *max_threads_per_block)
                *max_threads_per_block = max_threads[member];
            if (active_blocks[member] < *max_active_blocks_per_sm)
                *max_active_blocks_per_sm = active_blocks[member];
        }
        return true;
    }
}

extern "C" {

/**
 * @brief Query the generated tile selected for one long-prefill projection.
 *
 * This host-only introspection entry point shares the production selector,
 * including explicit trainer overrides. It performs no GPU work and is used by
 * PerfStats and integration tests to prove that the policy advertised for a
 * graph is the policy its capture launcher selected. M<=8 direct execution is
 * intentionally handled by the caller because it bypasses this generated
 * long-prefill surface.
 *
 * @param codebook_id NativeVNNI execution codebook stored in the descriptor.
 * @param projection_role Zero for gate/up, one for down.
 * @param m Number of original token rows.
 * @param n Projection output width.
 * @param k Projection reduction width.
 * @param tile_m Receives the trained expert-row tile.
 * @param tile_n Receives the trained output-column tile.
 * @return true when the generated table contains a valid compiled candidate;
 *         false for invalid arguments, unsupported codebooks, or a table miss.
 */
bool rocmMoE_grouped_prefill_query_tile_config(
    uint8_t codebook_id,
    int projection_role,
    int m,
    int n,
    int k,
    int* tile_m,
    int* tile_n)
{
    if (!tile_m || !tile_n || projection_role < 0 || projection_role > 1)
        return false;
    const auto role = projection_role == 0
                          ? MoEPrefillProjectionRole::GateUp
                          : MoEPrefillProjectionRole::Down;
    const MoEPrefillTileConfig config = moe_prefill_select_tile_config(
        role,
        static_cast<NativeVNNIFormatPrefill>(codebook_id),
        m,
        n,
        k);
    if (!moe_prefill_valid_tile_rows(config.rows) ||
        !moe_prefill_valid_tile_columns(config.columns))
    {
        return false;
    }
    *tile_m = config.rows;
    *tile_n = config.columns;
    return true;
}

/**
 * @brief Query the complete capture-time policy for one production MoE pair.
 *
 * Unlike the role-only trainer query, this API has every runtime key used by
 * the exact overlay. It therefore reports the same joint gate/up and down
 * geometry that the production launcher will embed in a captured graph. An
 * unseen exact key remains valid through the complementary generic tables and
 * reports @p exact_overlay as zero.
 *
 * @param gateup_codebook_id Uniform gate/up execution codebook.
 * @param down_codebook_id Uniform down execution codebook.
 * @param m Number of original token rows.
 * @param hidden_size Model hidden width.
 * @param expert_width Routed expert intermediate width.
 * @param expert_count Number of routed experts.
 * @param top_k Routes selected per token.
 * @param gateup_tile_m Receives the gate/up expert-row tile.
 * @param gateup_tile_n Receives the gate/up output-column tile.
 * @param down_tile_m Receives the down expert-row tile.
 * @param down_tile_n Receives the down output-column tile.
 * @param exact_overlay Receives one for an exact overlay and zero for generic.
 * @return true when both selected configurations name compiled candidates.
 */
bool rocmMoE_grouped_prefill_query_production_pair_config(
    uint8_t gateup_codebook_id,
    uint8_t down_codebook_id,
    int m,
    int hidden_size,
    int expert_width,
    int expert_count,
    int top_k,
    int *gateup_tile_m,
    int *gateup_tile_n,
    int *down_tile_m,
    int *down_tile_n,
    int *exact_overlay)
{
    if (gateup_codebook_id >= 32 || down_codebook_id >= 32 ||
        m <= 0 || hidden_size <= 0 || expert_width <= 0 ||
        expert_count <= 0 || top_k <= 0 || top_k > expert_count ||
        !gateup_tile_m || !gateup_tile_n || !down_tile_m || !down_tile_n ||
        !exact_overlay)
    {
        return false;
    }

    const MoEPrefillPairConfig pair = moe_prefill_select_pair_config(
        uint32_t{1} << gateup_codebook_id,
        uint32_t{1} << down_codebook_id,
        m,
        hidden_size,
        expert_width,
        expert_count,
        top_k);
    if (!moe_prefill_valid_tile_rows(pair.gateup.rows) ||
        !moe_prefill_valid_tile_columns(pair.gateup.columns) ||
        !moe_prefill_valid_tile_rows(pair.down.rows) ||
        !moe_prefill_valid_tile_columns(pair.down.columns))
    {
        return false;
    }

    *gateup_tile_m = pair.gateup.rows;
    *gateup_tile_n = pair.gateup.columns;
    *down_tile_m = pair.down.rows;
    *down_tile_n = pair.down.columns;
    *exact_overlay = pair.exact_overlay ? 1 : 0;
    return true;
}

/**
 * @brief Query static resources for one compiled grouped-prefill candidate.
 *
 * This inspection performs no allocation, launch, transfer, or synchronization.
 * It is the pre-timing eligibility gate used by the exhaustive trainer.  A
 * successful query with non-zero `local_memory_bytes_per_thread` identifies a
 * register-spilling specialization and therefore excludes that specialization
 * from the launch tournament.
 *
 * @param codebook_id NativeVNNI execution codebook compiled into the decoder.
 * @param projection_role Zero for gate/up, one for down.
 * @param tile_m Candidate expert-row tile.
 * @param tile_n Candidate output-column tile and physical block width.
 * @param registers_per_thread Receives compiler-assigned VGPR usage.
 * @param local_memory_bytes_per_thread Receives compiler-owned scratch bytes.
 * @param static_shared_memory_bytes Receives compiler-owned LDS bytes.
 * @param max_threads_per_block Receives the kernel's launch ceiling.
 * @param max_active_blocks_per_sm Receives occupancy at this block geometry.
 * @return true only when the exact compiled specialization can be inspected.
 */
bool rocmMoE_grouped_prefill_query_kernel_resources(
    uint8_t codebook_id,
    int projection_role,
    int tile_m,
    int tile_n,
    int* registers_per_thread,
    size_t* local_memory_bytes_per_thread,
    size_t* static_shared_memory_bytes,
    int* max_threads_per_block,
    int* max_active_blocks_per_sm)
{
    if (projection_role < 0 || projection_role > 1 ||
        !moe_prefill_valid_tile_rows(tile_m) ||
        !moe_prefill_valid_tile_columns(tile_n))
    {
        return false;
    }

    const auto role = projection_role == 0
                          ? MoEPrefillProjectionRole::GateUp
                          : MoEPrefillProjectionRole::Down;

    #define QUERY_TILE_N(FMT, TM) \
        do { \
            switch (tile_n) { \
            case 64: return moe_prefill_query_candidate_resources<FMT, TM, 64>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            case 128: return moe_prefill_query_candidate_resources<FMT, TM, 128>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            case 256: return moe_prefill_query_candidate_resources<FMT, TM, 256>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            default: return false; \
            } \
        } while (0)

    #define QUERY_ADAPTIVE_TILE_N(FMT) \
        do { \
            switch (tile_n) { \
            case 64: return moe_prefill_query_adaptive_candidate_resources<FMT, 64>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            case 128: return moe_prefill_query_adaptive_candidate_resources<FMT, 128>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            case 256: return moe_prefill_query_adaptive_candidate_resources<FMT, 256>( \
                role, registers_per_thread, local_memory_bytes_per_thread, \
                static_shared_memory_bytes, max_threads_per_block, \
                max_active_blocks_per_sm); \
            default: return false; \
            } \
        } while (0)

    #define QUERY_TILE_M(FMT) \
        do { \
            switch (tile_m) { \
            case 4: QUERY_TILE_N(FMT, 4); \
            case 8: QUERY_TILE_N(FMT, 8); \
            case 12: QUERY_TILE_N(FMT, 12); \
            case 16: QUERY_TILE_N(FMT, 16); \
            case kMoEPrefillAdaptiveTileRows: QUERY_ADAPTIVE_TILE_N(FMT); \
            default: return false; \
            } \
        } while (0)

    switch (static_cast<NativeVNNIFormatPrefill>(codebook_id))
    {
    case NVNNI_PF_Q4_0: QUERY_TILE_M(NVNNI_PF_Q4_0);
    case NVNNI_PF_IQ4_NL: QUERY_TILE_M(NVNNI_PF_IQ4_NL);
    case NVNNI_PF_Q4_1: QUERY_TILE_M(NVNNI_PF_Q4_1);
    case NVNNI_PF_Q5_0: QUERY_TILE_M(NVNNI_PF_Q5_0);
    case NVNNI_PF_Q5_1: QUERY_TILE_M(NVNNI_PF_Q5_1);
    case NVNNI_PF_Q6_K: QUERY_TILE_M(NVNNI_PF_Q6_K);
    case NVNNI_PF_Q3_K: QUERY_TILE_M(NVNNI_PF_Q3_K);
    case NVNNI_PF_Q2_K: QUERY_TILE_M(NVNNI_PF_Q2_K);
    case NVNNI_PF_IQ3_S: QUERY_TILE_M(NVNNI_PF_IQ3_S);
    case NVNNI_PF_IQ3_XXS: QUERY_TILE_M(NVNNI_PF_IQ3_XXS);
    case NVNNI_PF_IQ2_S: QUERY_TILE_M(NVNNI_PF_IQ2_S);
    case NVNNI_PF_IQ2_XS: QUERY_TILE_M(NVNNI_PF_IQ2_XS);
    case NVNNI_PF_IQ2_XXS: QUERY_TILE_M(NVNNI_PF_IQ2_XXS);
    case NVNNI_PF_IQ1_S: QUERY_TILE_M(NVNNI_PF_IQ1_S);
    case NVNNI_PF_IQ1_M: QUERY_TILE_M(NVNNI_PF_IQ1_M);
    case NVNNI_PF_Q8_0: QUERY_TILE_M(NVNNI_PF_Q8_0);
    default: break;
    }

    #undef QUERY_TILE_M
    #undef QUERY_ADAPTIVE_TILE_N
    #undef QUERY_TILE_N
    return false;
}

bool rocmMoE_grouped_prefill_pipeline(
    // Input
    const float*  d_hidden,              // [seq_len, d_model] FP32 hidden states
    const int8_t* d_prequantized_hidden, // optional [seq_len, d_model] router-owned Q8 rows
    const float*  d_prequantized_hidden_scales, // optional [seq_len, d_model / 32]
    // Weight descriptors (already on device)
    const void*   d_gate_desc_table,     // DeviceNativeVNNIMatrixDesc[num_experts]
    const void*   d_up_desc_table,       // DeviceNativeVNNIMatrixDesc[num_experts]
    const void*   d_down_desc_table,     // DeviceNativeVNNIMatrixDesc[num_experts]
    // Grouping data (from prepareExpertGroupsAsync)
    const int*    d_group_counts,        // [num_experts] per-expert token count
    const int*    d_group_offsets,       // [num_experts] prefix-sum offsets
    const int*    d_group_token_indices, // [total_slots] original token indices
    const int*    d_original_to_grouped,  // [seq_len * top_k] required route slot -> grouped slot
    const int*    d_original_expert_ids,  // [seq_len * top_k] required unless num_experts == 1
    const float*  d_group_weights,       // [total_slots] routing weights
    int*          d_group_work_directory,// dedicated header + TM12/TM16 device spans
    // Scratch buffers (caller provides, sized for total_slots)
    int8_t*       d_scratch_A_int8,      // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_scales,      // [total_slots, max_blocks_per_row]
    float*        d_scratch_gate,        // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_up,          // [total_slots, intermediate]
    int8_t*       d_scratch_swiglu_int8, // [total_slots, intermediate]
    float*        d_scratch_swiglu_scales,// [total_slots, blocks_per_intermediate]
    // Output
    float*        d_output,              // [seq_len, d_model] overwritten in original top-k order
    float*        d_canonical_route_contributions, // optional [seq_len, top_k, d_model]
    // Dimensions
    int num_experts,
    int d_model,
    int intermediate,
    int total_slots,                     // seq_len * top_k
    int top_k,                           // routes per token, needed by ordered publication
    int grouped_indices_are_route_slots, // d_group_token_indices stores route slots, not token rows
    // Format masks launch only the compiled NativeVNNI specializations present
    // in each descriptor table. Mixed-codebook expert tables remain supported.
    uint32_t gateup_codebook_mask,
    uint32_t down_codebook_mask,
    int device_id,
    void* stream)
{
    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        d_model <= 0 || intermediate <= 0 ||
        total_slots <= 0 || top_k <= 0 || (total_slots % top_k) != 0 ||
        (d_model % 32) != 0 || (intermediate % 32) != 0 ||
        gateup_codebook_mask == 0 || down_codebook_mask == 0 || !stream)
        return false;
    if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_down_desc_table ||
        !d_group_counts || !d_group_offsets || !d_group_token_indices ||
        !d_original_to_grouped || !d_group_weights || !d_group_work_directory ||
        !d_scratch_A_int8 || !d_scratch_scales || !d_scratch_gate ||
        !d_scratch_up || !d_scratch_swiglu_int8 || !d_scratch_swiglu_scales ||
        (!d_output && !d_canonical_route_contributions) ||
        (!d_original_expert_ids && num_experts != 1))
        return false;
    const int seq_len = total_slots / top_k;
    const bool canonical_publication =
        d_canonical_route_contributions != nullptr;
    const bool use_expert_row_tiles = seq_len > 8;
    const bool has_uniform_codebook_pair =
        moe_prefill_single_codebook(gateup_codebook_mask) >= 0 &&
        moe_prefill_single_codebook(down_codebook_mask) >= 0;
    const MoEPrefillPairConfig pair_config =
        use_expert_row_tiles && has_uniform_codebook_pair
            ? moe_prefill_select_pair_config(
                  gateup_codebook_mask,
                  down_codebook_mask,
                  seq_len,
                  d_model,
                  intermediate,
                  num_experts,
                  top_k)
            : MoEPrefillPairConfig{};
    if (use_expert_row_tiles && has_uniform_codebook_pair &&
        (!moe_prefill_valid_tile_rows(pair_config.gateup.rows) ||
         !moe_prefill_valid_tile_columns(pair_config.gateup.columns) ||
         !moe_prefill_valid_tile_rows(pair_config.down.rows) ||
         !moe_prefill_valid_tile_columns(pair_config.down.columns)))
    {
        return false;
    }

    MoEPrefillSerialM1KPartitionPolicy gateup_k_partition_policy{};
    MoEPrefillSerialM1KPartitionPolicy down_k_partition_policy{};
    if (!moe_prefill_resolve_serial_m1_k_partitions(
            gateup_codebook_mask,
            intermediate,
            d_model,
            gateup_k_partition_policy) ||
        !moe_prefill_resolve_serial_m1_k_partitions(
            down_codebook_mask,
            d_model,
            intermediate,
            down_k_partition_policy))
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError(); // clear prior errors
    const bool has_prequantized_hidden =
        d_prequantized_hidden && d_prequantized_hidden_scales;
    if ((d_prequantized_hidden == nullptr) !=
        (d_prequantized_hidden_scales == nullptr))
        return false;
    const bool reuse_prequantized_hidden = has_prequantized_hidden;

    int directory_tile_rows = 0;
    int current_directory_capacity = 0;
    const auto ensure_work_directory = [&](int tile_rows) -> int
    {
        if (!moe_prefill_valid_tile_rows(tile_rows))
            return 0;

        const int capacity = moe_prefill_directory_capacity(
            total_slots, num_experts, tile_rows);
        if (directory_tile_rows == tile_rows)
            return current_directory_capacity;

        /*
         * One thread owns one descriptor-table entry in the inclusive scan.
         * Launch the full representable width so inactive tail entries are
         * explicitly zero and the final scan element always contains the
         * complete tile count, including expert id 256 and above.
         */
        dim3 directory_block(kMoEPrefillMaxExperts);
        switch (tile_rows)
        {
        case 4:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<4>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity);
            break;
        case 8:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<8>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity);
            break;
        case 12:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<12>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity);
            break;
        case 16:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<16>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity);
            break;
        default:
            return 0;
        }

        directory_tile_rows = tile_rows;
        current_directory_capacity = capacity;
        return capacity;
    };

    /** Capture-time addresses and conservative bounds for two device spans. */
    struct AdaptiveDirectoryView
    {
        uint32_t* header = nullptr;
        uint32_t* tile12 = nullptr;
        uint32_t* tile16 = nullptr;
        int tile12_capacity = 0;
        int tile16_capacity = 0;

        /** @return true when every captured pointer and bound is usable. */
        [[nodiscard]] bool valid() const noexcept
        {
            return header && tile12 && tile16 &&
                   tile12_capacity > 0 && tile16_capacity > 0;
        }
    };

    const auto build_adaptive_work_directory = [&] (
        MoEPrefillProjectionRole role) -> AdaptiveDirectoryView
    {
        const int tile12_capacity = moe_prefill_directory_capacity(
            total_slots, num_experts, /*minimum adaptive tile=*/12);
        const int tile16_capacity =
            role == MoEPrefillProjectionRole::GateUp
                ? (total_slots < num_experts ? total_slots : num_experts)
                : moe_prefill_directory_capacity(
                      total_slots,
                      num_experts,
                      /*maximum adaptive tile=*/16);
        auto* const header =
            reinterpret_cast<uint32_t*>(d_group_work_directory);
        AdaptiveDirectoryView view{
            .header = header,
            .tile12 = header + kMoEPrefillAdaptiveDirectoryHeaderWords,
            .tile16 = header + kMoEPrefillAdaptiveDirectoryHeaderWords +
                      tile12_capacity,
            .tile12_capacity = tile12_capacity,
            .tile16_capacity = tile16_capacity,
        };
        if (!view.valid())
            return {};

        dim3 directory_block(kMoEPrefillMaxExperts);
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                        MoEPrefillAdaptiveProjectionRole::GateUp>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                header,
                num_experts,
                tile12_capacity,
                tile16_capacity);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                        MoEPrefillAdaptiveProjectionRole::Down>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                header,
                num_experts,
                tile12_capacity,
                tile16_capacity);
        }
        directory_tile_rows = 0;
        current_directory_capacity = 0;
        return view;
    };

    const int blocks_per_d_model = d_model / 32;

    // =========================================================================
    // Kernel 1: Quantize each original hidden row once.
    //
    // A verifier router already quantizes every original row for its Q8 logits
    // GEMV.  When those device rows are published, gate/up addresses them
    // through d_group_token_indices and this launch is unnecessary.  Ordinary
    // prefill and non-Q8 routing use the same row-owned Q8 contract here. The
    // grouped projection resolves every expert slot back to this original row.
    // =========================================================================
    if (!reuse_prequantized_hidden)
    {
        dim3 grid(blocks_per_d_model, seq_len);
        dim3 block(32);
        hipLaunchKernelGGL(moe_prefill_hidden_quantize_rows_kernel,
            grid, block, 0, hip_stream,
            d_hidden, d_scratch_A_int8, d_scratch_scales, seq_len, d_model);
    }

    // =========================================================================
    // Kernel 2: Batch-invariant gate/up projection.
    //
    // Verifier buckets retain the low-latency route-owned geometry. Long prefill
    // consumes the compact device directory and amortizes weight decode across
    // rows. Trained expert-row tiles preserve every row's serial-M1 partition
    // boundaries and ordered partial reduction.
    // =========================================================================
    {
        const auto* gate_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_gate_desc_table);
        const auto* up_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_up_desc_table);

        #define LAUNCH_GATEUP_ROUTE_OWNED(FMT) \
            do { \
                launch_moe_grouped_gate_up_route_owned<FMT>( \
                    hip_stream, \
                    reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8, \
                    reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales, \
                    d_original_to_grouped, d_original_expert_ids, gate_descs, up_descs, \
                    d_scratch_gate, d_scratch_up, total_slots, seq_len, top_k, \
                    1, intermediate, d_model, \
                    gateup_k_partition_policy.forCodebook( \
                        static_cast<uint8_t>(FMT)), num_experts); \
            } while (0)
        #define LAUNCH_GATEUP_ADAPTIVE(FMT, TN) \
            do { \
                const AdaptiveDirectoryView adaptive_directory = \
                    build_adaptive_work_directory( \
                        MoEPrefillProjectionRole::GateUp); \
                if (!adaptive_directory.valid()) \
                    return false; \
                dim3 tile12_grid((intermediate + TN - 1) / TN, \
                                 adaptive_directory.tile12_capacity, 2); \
                dim3 tile16_grid((intermediate + TN - 1) / TN, \
                                 adaptive_directory.tile16_capacity, 2); \
                dim3 expert_block(TN); \
                hipLaunchKernelGGL( \
                    HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, 12, TN>), \
                    tile12_grid, expert_block, 0, hip_stream, \
                    reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8, \
                    reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales, \
                    d_group_token_indices, d_group_counts, d_group_offsets, \
                    adaptive_directory.tile12, \
                    gate_descs, up_descs, d_scratch_gate, d_scratch_up, \
                    adaptive_directory.tile12_capacity, seq_len, top_k, grouped_indices_are_route_slots, \
                    1, intermediate, d_model, \
                    gateup_k_partition_policy.forCodebook( \
                        static_cast<uint8_t>(FMT)), num_experts, \
                    adaptive_directory.header + \
                        kMoEPrefillAdaptiveTile12CountWord); \
                hipLaunchKernelGGL( \
                    HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, 16, TN>), \
                    tile16_grid, expert_block, 0, hip_stream, \
                    reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8, \
                    reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales, \
                    d_group_token_indices, d_group_counts, d_group_offsets, \
                    adaptive_directory.tile16, \
                    gate_descs, up_descs, d_scratch_gate, d_scratch_up, \
                    adaptive_directory.tile16_capacity, seq_len, top_k, grouped_indices_are_route_slots, \
                    1, intermediate, d_model, \
                    gateup_k_partition_policy.forCodebook( \
                        static_cast<uint8_t>(FMT)), num_experts, \
                    adaptive_directory.header + \
                        kMoEPrefillAdaptiveTile16CountWord); \
            } while (0)
        #define LAUNCH_GATEUP_EXPERT_TILED(FMT, TM, TN) \
            do { \
                const int candidate_directory_capacity = ensure_work_directory(TM); \
                if (candidate_directory_capacity <= 0) \
                    return false; \
                dim3 grid((intermediate + TN - 1) / TN, candidate_directory_capacity, 2); \
                dim3 expert_block(TN); \
                hipLaunchKernelGGL( \
                    HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, TM, TN>), \
                    grid, expert_block, 0, hip_stream, \
                    reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8, \
                    reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales, \
                    d_group_token_indices, d_group_counts, d_group_offsets, \
                    reinterpret_cast<const uint32_t*>(d_group_work_directory), \
                    gate_descs, up_descs, d_scratch_gate, d_scratch_up, \
                    candidate_directory_capacity, seq_len, top_k, grouped_indices_are_route_slots, \
                    1, intermediate, d_model, \
                    gateup_k_partition_policy.forCodebook( \
                        static_cast<uint8_t>(FMT)), num_experts, nullptr); \
            } while (0)

        #define DISPATCH_GATEUP_TILE_N(FMT, TM, CONFIG) \
            do { \
                switch ((CONFIG).columns) { \
                case 64:  LAUNCH_GATEUP_EXPERT_TILED(FMT, TM, 64); break; \
                case 128: LAUNCH_GATEUP_EXPERT_TILED(FMT, TM, 128); break; \
                case 256: LAUNCH_GATEUP_EXPERT_TILED(FMT, TM, 256); break; \
                default: return false; \
                } \
            } while (0)

        #define DISPATCH_GATEUP_CONFIG(FMT, CONFIG) \
            do { \
                switch ((CONFIG).rows) { \
                case 4:  DISPATCH_GATEUP_TILE_N(FMT, 4, CONFIG); break; \
                case 8:  DISPATCH_GATEUP_TILE_N(FMT, 8, CONFIG); break; \
                case 12: DISPATCH_GATEUP_TILE_N(FMT, 12, CONFIG); break; \
                case 16: DISPATCH_GATEUP_TILE_N(FMT, 16, CONFIG); break; \
                case kMoEPrefillAdaptiveTileRows: \
                    switch ((CONFIG).columns) { \
                    case 64:  LAUNCH_GATEUP_ADAPTIVE(FMT, 64); break; \
                    case 128: LAUNCH_GATEUP_ADAPTIVE(FMT, 128); break; \
                    case 256: LAUNCH_GATEUP_ADAPTIVE(FMT, 256); break; \
                    default: return false; \
                    } \
                    break; \
                default: return false; \
                } \
            } while (0)

        bool launched_gateup = false;
        #define LAUNCH_GATEUP_IF_PRESENT(FMT) \
            do { \
                if (gateup_codebook_mask & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
                    if (!use_expert_row_tiles) { \
                        LAUNCH_GATEUP_ROUTE_OWNED(FMT); \
                    } else { \
                        const MoEPrefillTileConfig candidate = has_uniform_codebook_pair \
                            ? pair_config.gateup \
                            : moe_prefill_select_tile_config( \
                                  MoEPrefillProjectionRole::GateUp, FMT, seq_len, intermediate, d_model); \
                        if (!moe_prefill_valid_tile_rows(candidate.rows) || \
                            !moe_prefill_valid_tile_columns(candidate.columns)) \
                            return false; \
                        DISPATCH_GATEUP_CONFIG(FMT, candidate); \
                    } \
                    launched_gateup = true; \
                } \
            } while (0)

        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q4_0);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ4_NL);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q4_1);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q5_0);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q5_1);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q6_K);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q3_K);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q2_K);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ3_S);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ3_XXS);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ2_S);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ2_XS);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ2_XXS);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ1_S);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_IQ1_M);
        LAUNCH_GATEUP_IF_PRESENT(NVNNI_PF_Q8_0);
        if (!launched_gateup)
        {
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported gate/up codebook mask=0x%x\n",
                        gateup_codebook_mask);
                return false;
        }
        #undef LAUNCH_GATEUP_IF_PRESENT
        #undef DISPATCH_GATEUP_CONFIG
        #undef DISPATCH_GATEUP_TILE_N
        #undef LAUNCH_GATEUP_ADAPTIVE
        #undef LAUNCH_GATEUP_EXPERT_TILED
        #undef LAUNCH_GATEUP_ROUTE_OWNED

        constexpr int TILE_N = 32;
        dim3 quant_grid((intermediate + TILE_N - 1) / TILE_N, total_slots);
        dim3 quant_block(TILE_N);
        hipLaunchKernelGGL(
            moe_grouped_swiglu_quantize_prefill_kernel,
            quant_grid, quant_block, 0, hip_stream,
            d_scratch_gate, d_scratch_up,
            d_scratch_swiglu_int8, d_scratch_swiglu_scales,
            intermediate);
    }

    // =========================================================================
    // Kernels 4-5: down projection and fixed-order publication.
    //
    // Tiny verifier buckets compute and publish directly. Long prefill writes
    // row-tiled down partials into the now-dead gate scratch, then one owner per
    // token/column reduces those rows in original top-k order.
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 block(TILE_N);
        const auto* down_descs =
            static_cast<const DeviceNativeVNNIMatrixDesc*>(d_down_desc_table);

        if (!use_expert_row_tiles)
        {
            /*
             * The codebook mask is immutable graph identity. A one-bit mask
             * therefore selects one fully inlined down decoder during capture;
             * only a table that deliberately contains multiple formats enters
             * the mixed-table specialization. Both routes execute the same
             * ordered publication kernels and differ solely in decoder dispatch.
             */
            #define LAUNCH_ROUTE_OWNED_DOWN(FORMAT_TAG) \
                do { \
                    if (canonical_publication) { \
                        dim3 grid((d_model + TILE_N - 1) / TILE_N, total_slots); \
                        hipLaunchKernelGGL( \
                            HIP_KERNEL_NAME(moe_grouped_down_publish_canonical_routes_prefill_kernel<FORMAT_TAG>), \
                            grid, block, 0, hip_stream, \
                            d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                            d_original_to_grouped, d_original_expert_ids, d_group_weights, \
                            d_canonical_route_contributions, total_slots, d_model, \
                            intermediate, num_experts, down_k_partition_policy); \
                    } else { \
                        dim3 grid((d_model + TILE_N - 1) / TILE_N, seq_len); \
                        hipLaunchKernelGGL( \
                            HIP_KERNEL_NAME(moe_grouped_down_publish_original_token_ordered_prefill_kernel<FORMAT_TAG>), \
                            grid, block, 0, hip_stream, \
                            d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                            d_original_to_grouped, d_original_expert_ids, d_group_weights, \
                            d_output, seq_len, top_k, d_model, intermediate, num_experts, \
                            down_k_partition_policy); \
                    } \
                } while (0)

            #define LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(FMT) \
                case (uint32_t{1} << static_cast<uint32_t>(FMT)): \
                    LAUNCH_ROUTE_OWNED_DOWN(static_cast<int>(FMT)); \
                    break

            switch (down_codebook_mask)
            {
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q4_0);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ4_NL);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q4_1);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q5_0);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q5_1);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q6_K);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q3_K);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q2_K);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ3_S);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ3_XXS);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ2_S);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ2_XS);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ2_XXS);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ1_S);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_IQ1_M);
            LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT(NVNNI_PF_Q8_0);
            default:
                if (down_codebook_mask == 0 ||
                    (down_codebook_mask & (down_codebook_mask - 1u)) == 0)
                {
                    fprintf(stderr,
                            "[rocmMoE_grouped_prefill_pipeline] invalid route-owned down codebook mask=0x%x\n",
                            down_codebook_mask);
                    return false;
                }
                LAUNCH_ROUTE_OWNED_DOWN(kMoEPrefillMixedCodebookTag);
                break;
            }

            #undef LAUNCH_ROUTE_OWNED_DOWN_IF_EXACT
            #undef LAUNCH_ROUTE_OWNED_DOWN
        }
        else
        {
            bool launched_down = false;
            #define LAUNCH_DOWN_EXPERT_TILED(FMT, TM, TN) \
                do { \
                    const int candidate_directory_capacity = ensure_work_directory(TM); \
                    if (candidate_directory_capacity <= 0) \
                        return false; \
                    dim3 grid((d_model + TN - 1) / TN, candidate_directory_capacity); \
                    dim3 expert_block(TN); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, TM, TN>), \
                        grid, expert_block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, \
                            reinterpret_cast<const uint32_t*>(d_group_work_directory), \
                            d_scratch_gate, candidate_directory_capacity, d_model, intermediate, \
                            down_k_partition_policy.forCodebook( \
                            static_cast<uint8_t>(FMT)), num_experts, nullptr); \
                } while (0)
            #define LAUNCH_DOWN_ADAPTIVE(FMT, TN) \
                do { \
                    const AdaptiveDirectoryView adaptive_directory = \
                        build_adaptive_work_directory( \
                            MoEPrefillProjectionRole::Down); \
                    if (!adaptive_directory.valid()) \
                        return false; \
                    dim3 tile12_grid((d_model + TN - 1) / TN, \
                                     adaptive_directory.tile12_capacity); \
                    dim3 tile16_grid((d_model + TN - 1) / TN, \
                                     adaptive_directory.tile16_capacity); \
                    dim3 expert_block(TN); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, 12, TN>), \
                        tile12_grid, expert_block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, \
                        adaptive_directory.tile12, \
                        d_scratch_gate, adaptive_directory.tile12_capacity, d_model, intermediate, \
                        down_k_partition_policy.forCodebook( \
                            static_cast<uint8_t>(FMT)), num_experts, \
                        adaptive_directory.header + \
                            kMoEPrefillAdaptiveTile12CountWord); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, 16, TN>), \
                        tile16_grid, expert_block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, \
                        adaptive_directory.tile16, \
                        d_scratch_gate, adaptive_directory.tile16_capacity, d_model, intermediate, \
                        down_k_partition_policy.forCodebook( \
                            static_cast<uint8_t>(FMT)), num_experts, \
                        adaptive_directory.header + \
                            kMoEPrefillAdaptiveTile16CountWord); \
                } while (0)

            #define DISPATCH_DOWN_TILE_N(FMT, TM, CONFIG) \
                do { \
                    switch ((CONFIG).columns) { \
                    case 64:  LAUNCH_DOWN_EXPERT_TILED(FMT, TM, 64); break; \
                    case 128: LAUNCH_DOWN_EXPERT_TILED(FMT, TM, 128); break; \
                    case 256: LAUNCH_DOWN_EXPERT_TILED(FMT, TM, 256); break; \
                    default: return false; \
                    } \
                } while (0)

            #define DISPATCH_DOWN_CONFIG(FMT, CONFIG) \
                do { \
                    switch ((CONFIG).rows) { \
                    case 4:  DISPATCH_DOWN_TILE_N(FMT, 4, CONFIG); break; \
                    case 8:  DISPATCH_DOWN_TILE_N(FMT, 8, CONFIG); break; \
                    case 12: DISPATCH_DOWN_TILE_N(FMT, 12, CONFIG); break; \
                    case 16: DISPATCH_DOWN_TILE_N(FMT, 16, CONFIG); break; \
                    case kMoEPrefillAdaptiveTileRows: \
                        switch ((CONFIG).columns) { \
                        case 64:  LAUNCH_DOWN_ADAPTIVE(FMT, 64); break; \
                        case 128: LAUNCH_DOWN_ADAPTIVE(FMT, 128); break; \
                        case 256: LAUNCH_DOWN_ADAPTIVE(FMT, 256); break; \
                        default: return false; \
                        } \
                        break; \
                    default: return false; \
                    } \
                } while (0)
            #define LAUNCH_DOWN_IF_PRESENT(FMT) \
                do { \
                    if (down_codebook_mask & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
                        const MoEPrefillTileConfig candidate = has_uniform_codebook_pair \
                            ? pair_config.down \
                            : moe_prefill_select_tile_config( \
                                  MoEPrefillProjectionRole::Down, FMT, seq_len, d_model, intermediate); \
                        if (!moe_prefill_valid_tile_rows(candidate.rows) || \
                            !moe_prefill_valid_tile_columns(candidate.columns)) \
                            return false; \
                        DISPATCH_DOWN_CONFIG(FMT, candidate); \
                        launched_down = true; \
                    } \
                } while (0)

            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q4_0);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ4_NL);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q4_1);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q5_0);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q5_1);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q6_K);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q3_K);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q2_K);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ3_S);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ3_XXS);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ2_S);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ2_XS);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ2_XXS);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ1_S);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_IQ1_M);
            LAUNCH_DOWN_IF_PRESENT(NVNNI_PF_Q8_0);
            #undef LAUNCH_DOWN_IF_PRESENT
            #undef DISPATCH_DOWN_CONFIG
            #undef DISPATCH_DOWN_TILE_N
            #undef LAUNCH_DOWN_ADAPTIVE
            #undef LAUNCH_DOWN_EXPERT_TILED

            if (!launched_down)
            {
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported down codebook mask=0x%x\n",
                        down_codebook_mask);
                return false;
            }

            if (canonical_publication)
            {
                dim3 publish_grid(
                    (d_model + TILE_N - 1) / TILE_N,
                    total_slots);
                hipLaunchKernelGGL(
                    moe_grouped_down_partials_canonical_publish_prefill_kernel,
                    publish_grid, block, 0, hip_stream,
                    d_scratch_gate,
                    d_original_to_grouped,
                    d_group_weights,
                    d_canonical_route_contributions,
                    total_slots,
                    d_model);
            }
            else
            {
                dim3 publish_grid(
                    (d_model + TILE_N - 1) / TILE_N,
                    seq_len);
                hipLaunchKernelGGL(
                    moe_grouped_down_partials_ordered_publish_prefill_kernel,
                    publish_grid, block, 0, hip_stream,
                    d_scratch_gate, d_original_to_grouped, d_group_weights,
                    d_output, seq_len, top_k, d_model);
            }
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] kernel launch error: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool rocmMoE_reduce_canonical_route_contributions(
    const float* d_route_contributions,
    float* d_output,
    int seq_len,
    int top_k,
    int d_model,
    int device_id,
    void* stream)
{
    if (!d_route_contributions || !d_output || !stream ||
        seq_len <= 0 || top_k <= 0 || d_model <= 0)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);
    constexpr int kThreads = 256;
    dim3 grid((d_model + kThreads - 1) / kThreads, seq_len);
    hipLaunchKernelGGL(
        moe_reduce_canonical_route_contributions_kernel,
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        d_route_contributions,
        d_output,
        seq_len,
        top_k,
        d_model);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_reduce_canonical_route_contributions] launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// IQ grid table initialization for this TU.
// Must be called once per device before launching any IQ grid format kernel.
// ============================================================================
bool rocmInitIQGridTables_moe_prefill(
    int device_id,
    const void* h_iq3s_grid,
    const void* h_iq3xxs_grid,
    const void* h_iq2s_grid,
    const void* h_iq2xs_grid,
    const void* h_iq2xxs_grid,
    const void* h_iq1s_grid)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipError_t e;
    e = hipMemcpyToSymbol(d_pf_iq3s_grid, h_iq3s_grid, sizeof(d_pf_iq3s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq3s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq3xxs_grid, h_iq3xxs_grid, sizeof(d_pf_iq3xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq3xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2s_grid, h_iq2s_grid, sizeof(d_pf_iq2s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2xs_grid, h_iq2xs_grid, sizeof(d_pf_iq2xs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2xs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2xxs_grid, h_iq2xxs_grid, sizeof(d_pf_iq2xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq1s_grid, h_iq1s_grid, sizeof(d_pf_iq1s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq1s failed: %s\n", hipGetErrorString(e)); return false; }
    return true;
}

} // extern "C"
