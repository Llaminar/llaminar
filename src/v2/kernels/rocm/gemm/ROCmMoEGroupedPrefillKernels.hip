/**
 * @file ROCmMoEGroupedPrefillKernels.hip
 * @brief Grouped MoE prefill kernels for AMD MI50/MI60 (gfx906)
 *
 * These kernels process ALL experts in a single launch, eliminating the
 * per-expert host dispatch loop and hipStreamSynchronize that caused 99.9%
 * host overhead in prefill. Fully graph-capturable (no host-device sync).
 *
 * Pipeline (5 launches per MoE layer, replaces ~800):
 *   1. grouped_gather_quantize_prefill - gather + INT8 quantize for all experts
 *   2. grouped_gate_up_prefill_gemm    - gate+up GEMM for all experts
 *   3. grouped_swiglu_quantize_prefill - SwiGLU + INT8 quantize for all experts
 *   4. grouped_down_prefill_gemm       - down projection for all experts
 *   5. grouped_weighted_scatter_prefill - scatter-add results to output
 *
 * Key design: all kernels read expert counts/offsets from DEVICE memory
 * (d_group_counts, d_group_offsets) — no host reads required.
 *
 * Grid strategy: dim3(N_tiles, max_M_tiles, num_experts)
 *   - blockIdx.z = expert index
 *   - Blocks where blockIdx.y * TILE_M >= count[expert] early-exit
 *   - This over-provisions M-tiles but avoids host sync entirely
 *
 * @see ROCmGemvKernel_native_VNNI.hip for decode grouped kernels (M=1)
 * @see ROCmQuantisedGemmKernel_native_VNNI.hip for non-grouped prefill GEMM
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "../../../backends/rocm/HipDeviceGuard.h"

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

// IQ4_NL codebook (must match GEMV file)
__device__ __constant__ int8_t k_iq4_pf_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

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
template <NativeVNNIFormatPrefill FMT>
__device__ inline uint64_t pf_iq2_grid_lookup(int idx)
{
    if constexpr (FMT == NVNNI_PF_IQ2_S)       return d_pf_iq2s_grid[idx];
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
// Kernel 1: Grouped Gather + Quantize (all experts, single launch)
//
// FLAT GRID: grid(blocks_per_row, total_slots) — no empty blocks.
// Each block handles one 32-element quantization block of one token slot.
//
// Grid: dim3(blocks_per_row, total_slots)
//   blockIdx.x = quantization block within the row
//   blockIdx.y = global slot index (0..total_slots-1)
// Block: 32 threads (warp-cooperative quantization)
// ============================================================================

__global__ __launch_bounds__(32)
void moe_grouped_gather_quantize_prefill_kernel(
    const float* __restrict__ hidden,            // [seq_len, d_model]
    int8_t*      __restrict__ A_int8_out,        // [total_slots, K] gathered + quantized
    float*       __restrict__ scales_out,        // [total_slots, blocks_per_row] per-block scales
    const int*   __restrict__ d_group_token_indices, // [total_slots] original token indices
    int K)                                       // d_model (columns)
{
    const int block_in_row = blockIdx.x;   // which 32-element block in the row
    const int global_slot = blockIdx.y;    // which slot (flat across all experts)

    const int original_token = d_group_token_indices[global_slot];

    const int lane = threadIdx.x;  // 0..31
    const int k = block_in_row * 32 + lane;
    if (k >= K) return;

    // Load one element from the hidden state
    const float val = hidden[static_cast<size_t>(original_token) * K + k];

    // Warp-cooperative max reduction for quantization scale
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;

    // Write scale (one per block, lane 0 writes)
    const int blocks_per_row = K / 32;
    if (lane == 0)
        scales_out[static_cast<size_t>(global_slot) * blocks_per_row + block_in_row] = scale;

    // Quantize and write
    const float qval = val * inv_scale;
    A_int8_out[static_cast<size_t>(global_slot) * K + k] =
        static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
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

template <NativeVNNIFormatPrefill FMT>
__device__ __forceinline__ void moe_prefill_decode_groups(
    const uint8_t* __restrict__ payload,
    int32_t* packed_groups)
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
                const uint64_t grid8 = pf_iq2_grid_lookup<FMT>(idx);
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
                const uint64_t grid8 = pf_iq2_grid_lookup<FMT>(idx);
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
        // IQ4_NL: 4-bit LUT decode
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            packed_groups[g] = static_cast<int32_t>(
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[lo_nibs & 0xF]))) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(lo_nibs >> 8) & 0xF])) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(lo_nibs >> 16) & 0xF])) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(lo_nibs >> 24) & 0xF])) << 24));
            packed_groups[g + 4] = static_cast<int32_t>(
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[hi_nibs & 0xF]))) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(hi_nibs >> 8) & 0xF])) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(hi_nibs >> 16) & 0xF])) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(k_iq4_pf_codebook_i8[(hi_nibs >> 24) & 0xF])) << 24));
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

template <NativeVNNIFormatPrefill FMT>
__global__ __launch_bounds__(128)
void moe_grouped_gate_up_prefill_gemm_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, K] quantized activations
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, blocks_per_row]
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs, // [num_experts]
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,   // [num_experts]
    const int*     __restrict__ d_group_counts,      // [num_experts]
    const int*     __restrict__ d_group_offsets,     // [num_experts]
    const int*     __restrict__ d_active_expert_ids, // [active_expert_slots] or null
    float*         __restrict__ gate_output,         // [total_slots, N]
    float*         __restrict__ up_output,           // [total_slots, N]
    int N, int K)
{
    constexpr int TILE_N = 128;
    constexpr int TILE_M = 8;  // Process 8 tokens per block, amortizing weight decode
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_group = blockIdx.y;  // group of TILE_M tokens
    const int expert_id = d_active_expert_ids ? d_active_expert_ids[blockIdx.z] : static_cast<int>(blockIdx.z);
    if (expert_id < 0) return;

    const int count = d_group_counts[expert_id];
    const int first_token = token_group * TILE_M;
    if (first_token >= count || n >= N) return;

    // How many tokens in this group (1..TILE_M)
    const int tokens_in_group = min(TILE_M, count - first_token);
    const int offset = d_group_offsets[expert_id];
    const int blocks_per_row = K / 32;

    // Get weight descriptors for this expert
    const DeviceNativeVNNIMatrixDesc& gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc& up_desc = up_descs[expert_id];

    const uint8_t* gate_payload_base = gate_desc.payload;
    const uint16_t* gate_scale_base = static_cast<const uint16_t*>(gate_desc.scales);
    const uint16_t* gate_min_base = static_cast<const uint16_t*>(gate_desc.mins);
    const uint32_t* gate_emin_base = static_cast<const uint32_t*>(gate_desc.emins);

    const uint8_t* up_payload_base = up_desc.payload;
    const uint16_t* up_scale_base = static_cast<const uint16_t*>(up_desc.scales);
    const uint16_t* up_min_base = static_cast<const uint16_t*>(up_desc.mins);
    const uint32_t* up_emin_base = static_cast<const uint32_t*>(up_desc.emins);

    // Accumulators for TILE_M tokens
    float gate_acc[TILE_M] = {};
    float up_acc[TILE_M] = {};

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);

        // Decode weights ONCE per K-block (shared across all TILE_M tokens)
        int32_t gate_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* gate_payload = gate_payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(gate_payload, gate_groups);
        }

        int32_t up_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* up_payload = up_payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(up_payload, up_groups);
        }

        // Process each of TILE_M tokens with the same decoded weights
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m)
        {
            if (m >= tokens_in_group) break;

            const int global_slot = offset + first_token + m;
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                A_int8 + static_cast<size_t>(global_slot) * K + b * 32);
            const float scale_ab = scales_A_blockwise[
                static_cast<size_t>(global_slot) * blocks_per_row + b];

            if constexpr (PrefillNVNNITraits<FMT>::is_dual_scale) {
                int32_t gate_acc_lo = 0, gate_acc_hi = 0;
                int32_t up_acc_lo = 0, up_acc_hi = 0;
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    gate_acc_lo = __builtin_amdgcn_sdot4(a4[g], gate_groups[g], gate_acc_lo, false);
                    up_acc_lo = __builtin_amdgcn_sdot4(a4[g], up_groups[g], up_acc_lo, false);
                }
                #pragma unroll
                for (int g = 4; g < 8; ++g) {
                    gate_acc_hi = __builtin_amdgcn_sdot4(a4[g], gate_groups[g], gate_acc_hi, false);
                    up_acc_hi = __builtin_amdgcn_sdot4(a4[g], up_groups[g], up_acc_hi, false);
                }
                const uint16_t gate_scale_lo_bits = gate_scale_base[linear];
                const uint16_t gate_scale_hi_bits = gate_min_base[linear];
                const float gate_s_lo = __half2float(*reinterpret_cast<const __half*>(&gate_scale_lo_bits));
                const float gate_s_hi = __half2float(*reinterpret_cast<const __half*>(&gate_scale_hi_bits));
                gate_acc[m] += (static_cast<float>(gate_acc_lo) * gate_s_lo +
                                static_cast<float>(gate_acc_hi) * gate_s_hi) * scale_ab;

                const uint16_t up_scale_lo_bits = up_scale_base[linear];
                const uint16_t up_scale_hi_bits = up_min_base[linear];
                const float up_s_lo = __half2float(*reinterpret_cast<const __half*>(&up_scale_lo_bits));
                const float up_s_hi = __half2float(*reinterpret_cast<const __half*>(&up_scale_hi_bits));
                up_acc[m] += (static_cast<float>(up_acc_lo) * up_s_lo +
                              static_cast<float>(up_acc_hi) * up_s_hi) * scale_ab;

                // Q2_K: per-sub-block emin correction (asymmetric dual-scale)
                if constexpr (PrefillNVNNITraits<FMT>::is_dual_scale_asym) {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a_lo = 0, sum_a_hi = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        sum_a_lo = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_lo, false);
                    #pragma unroll
                    for (int g = 4; g < 8; ++g)
                        sum_a_hi = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_hi, false);

                    const uint32_t gate_emin_packed = gate_emin_base[linear];
                    const uint16_t gate_emin_lo_bits = static_cast<uint16_t>(gate_emin_packed);
                    const uint16_t gate_emin_hi_bits = static_cast<uint16_t>(gate_emin_packed >> 16);
                    const float gate_emin_lo = __half2float(*reinterpret_cast<const __half*>(&gate_emin_lo_bits));
                    const float gate_emin_hi = __half2float(*reinterpret_cast<const __half*>(&gate_emin_hi_bits));
                    gate_acc[m] += (static_cast<float>(sum_a_lo) * gate_emin_lo
                                 + static_cast<float>(sum_a_hi) * gate_emin_hi) * scale_ab;

                    const uint32_t up_emin_packed = up_emin_base[linear];
                    const uint16_t up_emin_lo_bits = static_cast<uint16_t>(up_emin_packed);
                    const uint16_t up_emin_hi_bits = static_cast<uint16_t>(up_emin_packed >> 16);
                    const float up_emin_lo = __half2float(*reinterpret_cast<const __half*>(&up_emin_lo_bits));
                    const float up_emin_hi = __half2float(*reinterpret_cast<const __half*>(&up_emin_hi_bits));
                    up_acc[m] += (static_cast<float>(sum_a_lo) * up_emin_lo
                               + static_cast<float>(sum_a_hi) * up_emin_hi) * scale_ab;
                }

                // IQ1_M: ±IQ1S_DELTA per-group correction
                if constexpr (FMT == NVNNI_PF_IQ1_M) {
                    constexpr float IQ1S_DELTA = 0.125f;
                    constexpr int32_t ones = 0x01010101;
                    constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
                    const uint8_t* gate_pay = gate_payload_base + linear * PB;
                    const uint8_t gate_qh0 = gate_pay[4], gate_qh1 = gate_pay[5];

                    int32_t sa_g0 = __builtin_amdgcn_sdot4(a4[1], ones,
                                    __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
                    int32_t sa_g1 = __builtin_amdgcn_sdot4(a4[3], ones,
                                    __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
                    int32_t sa_g2 = __builtin_amdgcn_sdot4(a4[5], ones,
                                    __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
                    int32_t sa_g3 = __builtin_amdgcn_sdot4(a4[7], ones,
                                    __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

                    float gate_delta_corr = 0.0f;
                    gate_delta_corr += static_cast<float>(sa_g0) * ((gate_qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    gate_delta_corr += static_cast<float>(sa_g1) * ((gate_qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    gate_delta_corr += static_cast<float>(sa_g2) * ((gate_qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    gate_delta_corr += static_cast<float>(sa_g3) * ((gate_qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    gate_acc[m] += gate_delta_corr * gate_s_lo * scale_ab;

                    const uint8_t* up_pay = up_payload_base + linear * PB;
                    const uint8_t up_qh0 = up_pay[4], up_qh1 = up_pay[5];
                    float up_delta_corr = 0.0f;
                    up_delta_corr += static_cast<float>(sa_g0) * ((up_qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    up_delta_corr += static_cast<float>(sa_g1) * ((up_qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    up_delta_corr += static_cast<float>(sa_g2) * ((up_qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    up_delta_corr += static_cast<float>(sa_g3) * ((up_qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    up_acc[m] += up_delta_corr * up_s_lo * scale_ab;
                }
            }
            else {
                int32_t gate_dot = 0, up_dot = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g) {
                    gate_dot = __builtin_amdgcn_sdot4(a4[g], gate_groups[g], gate_dot, false);
                    up_dot = __builtin_amdgcn_sdot4(a4[g], up_groups[g], up_dot, false);
                }

                const uint16_t gate_scale_bits = gate_scale_base[linear];
                const float gate_d = __half2float(*reinterpret_cast<const __half*>(&gate_scale_bits));
                gate_acc[m] += static_cast<float>(gate_dot) * gate_d * scale_ab;

                const uint16_t up_scale_bits = up_scale_base[linear];
                const float up_d = __half2float(*reinterpret_cast<const __half*>(&up_scale_bits));
                up_acc[m] += static_cast<float>(up_dot) * up_d * scale_ab;

                if constexpr (PrefillNVNNITraits<FMT>::is_asymmetric) {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                    const uint16_t gate_min_bits = gate_min_base[linear];
                    const float gate_m = __half2float(*reinterpret_cast<const __half*>(&gate_min_bits));
                    gate_acc[m] += static_cast<float>(sum_a) * gate_m * scale_ab;

                    const uint16_t up_min_bits = up_min_base[linear];
                    const float up_m = __half2float(*reinterpret_cast<const __half*>(&up_min_bits));
                    up_acc[m] += static_cast<float>(sum_a) * up_m * scale_ab;
                }
            }
        }
    }

    // Write outputs for all tokens in this group
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m)
    {
        if (m >= tokens_in_group) break;
        const int global_slot = offset + first_token + m;
        gate_output[static_cast<size_t>(global_slot) * N + n] = gate_acc[m];
        up_output[static_cast<size_t>(global_slot) * N + n] = up_acc[m];
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

    // Compute SwiGLU: silu(gate) * up = (gate / (1 + exp(-gate))) * up
    const size_t idx = static_cast<size_t>(global_slot) * intermediate + k;
    const float g = gate_data[idx];
    const float u = up_data[idx];
    const float val = (g / (1.0f + expf(-g))) * u;

    // Warp-cooperative max reduction for quantization
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;

    const int blocks_per_row = intermediate / 32;
    if (lane == 0)
        scales_out[static_cast<size_t>(global_slot) * blocks_per_row + block_in_row] = scale;

    const float qval = val * inv_scale;
    A_int8_out[static_cast<size_t>(global_slot) * intermediate + k] =
        static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
}


// ============================================================================
// Kernel 4: Grouped Down GEMM (all experts, single launch)
//
// Same structure as gate+up but for the down projection.
// Output is FP32 (not quantized) since it goes directly to scatter.
// M-tiled: each block processes TILE_M tokens, amortizing weight decode.
//
// Grid: dim3((d_model + TILE_N - 1) / TILE_N, ceil(max_tokens/TILE_M), num_experts)
// Block: 64 threads
// ============================================================================

template <NativeVNNIFormatPrefill FMT>
__global__ __launch_bounds__(128)
void moe_grouped_down_prefill_gemm_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, intermediate] (from SwiGLU output)
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, blocks_per_intermediate]
    const DeviceNativeVNNIMatrixDesc* __restrict__ down_descs, // [num_experts]
    const int*     __restrict__ d_group_counts,      // [num_experts]
    const int*     __restrict__ d_group_offsets,     // [num_experts]
    const int*     __restrict__ d_active_expert_ids, // [active_expert_slots] or null
    float*         __restrict__ output,              // [total_slots, d_model]
    int N,   // d_model (output columns)
    int K)   // intermediate (input columns / reduction dim)
{
    constexpr int TILE_N = 128;
    constexpr int TILE_M = 8;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_group = blockIdx.y;
    const int expert_id = d_active_expert_ids ? d_active_expert_ids[blockIdx.z] : static_cast<int>(blockIdx.z);
    if (expert_id < 0) return;

    const int count = d_group_counts[expert_id];
    const int first_token = token_group * TILE_M;
    if (first_token >= count || n >= N) return;

    const int tokens_in_group = min(TILE_M, count - first_token);
    const int offset = d_group_offsets[expert_id];
    const int blocks_per_row = K / 32;

    const DeviceNativeVNNIMatrixDesc& desc = down_descs[expert_id];
    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);

    float acc[TILE_M] = {};

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);

        // Decode weights ONCE per K-block (shared across TILE_M tokens)
        int32_t packed_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* payload = payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(payload, packed_groups);
        }

        // Process TILE_M tokens with decoded weights
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m)
        {
            if (m >= tokens_in_group) break;

            const int global_slot = offset + first_token + m;
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                A_int8 + static_cast<size_t>(global_slot) * K + b * 32);
            const float scale_ab = scales_A_blockwise[
                static_cast<size_t>(global_slot) * blocks_per_row + b];

            if constexpr (PrefillNVNNITraits<FMT>::is_dual_scale) {
                int32_t acc_lo = 0, acc_hi = 0;
                #pragma unroll
                for (int g = 0; g < 4; ++g)
                    acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
                #pragma unroll
                for (int g = 4; g < 8; ++g)
                    acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

                const uint16_t scale_lo_bits = scale_base[linear];
                const uint16_t scale_hi_bits = min_base[linear];
                const float s_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
                const float s_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
                acc[m] += (static_cast<float>(acc_lo) * s_lo +
                           static_cast<float>(acc_hi) * s_hi) * scale_ab;

                // Q2_K: per-sub-block emin correction (asymmetric dual-scale)
                if constexpr (PrefillNVNNITraits<FMT>::is_dual_scale_asym) {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a_lo = 0, sum_a_hi = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        sum_a_lo = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_lo, false);
                    #pragma unroll
                    for (int g = 4; g < 8; ++g)
                        sum_a_hi = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_hi, false);

                    const uint32_t emin_packed = emin_base[linear];
                    const uint16_t emin_lo_bits = static_cast<uint16_t>(emin_packed);
                    const uint16_t emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
                    const float emin_lo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
                    const float emin_hi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
                    acc[m] += (static_cast<float>(sum_a_lo) * emin_lo
                             + static_cast<float>(sum_a_hi) * emin_hi) * scale_ab;
                }

                // IQ1_M: ±IQ1S_DELTA per-group correction
                if constexpr (FMT == NVNNI_PF_IQ1_M) {
                    constexpr float IQ1S_DELTA = 0.125f;
                    constexpr int32_t ones = 0x01010101;
                    constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
                    const uint8_t* pay = payload_base + linear * PB;
                    const uint8_t qh0 = pay[4], qh1 = pay[5];

                    int32_t sa_g0 = __builtin_amdgcn_sdot4(a4[1], ones,
                                    __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
                    int32_t sa_g1 = __builtin_amdgcn_sdot4(a4[3], ones,
                                    __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
                    int32_t sa_g2 = __builtin_amdgcn_sdot4(a4[5], ones,
                                    __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
                    int32_t sa_g3 = __builtin_amdgcn_sdot4(a4[7], ones,
                                    __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

                    float delta_corr = 0.0f;
                    delta_corr += static_cast<float>(sa_g0) * ((qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    delta_corr += static_cast<float>(sa_g1) * ((qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    delta_corr += static_cast<float>(sa_g2) * ((qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA);
                    delta_corr += static_cast<float>(sa_g3) * ((qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA);
                    acc[m] += delta_corr * s_lo * scale_ab;
                }
            }
            else {
                int32_t dot = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    dot = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], dot, false);

                const uint16_t scale_bits = scale_base[linear];
                const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                acc[m] += static_cast<float>(dot) * block_d * scale_ab;

                if constexpr (PrefillNVNNITraits<FMT>::is_asymmetric) {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                    const uint16_t min_bits = min_base[linear];
                    const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                    acc[m] += static_cast<float>(sum_a) * block_m * scale_ab;
                }
            }
        }
    }

    // Write outputs
    #pragma unroll
    for (int m = 0; m < TILE_M; ++m)
    {
        if (m >= tokens_in_group) break;
        const int global_slot = offset + first_token + m;
        output[static_cast<size_t>(global_slot) * N + n] = acc[m];
    }
}


// ============================================================================
// Kernel 5: Grouped Weighted Scatter-Add (all experts, single launch)
//
// FLAT GRID: grid(tiles_per_row, total_slots) — no empty blocks.
// For each slot: output[original_token, :] += weight * expert_result[slot, :]
//
// Grid: dim3((d_model + 63) / 64, total_slots)
// Block: 64 threads
//
// Uses atomicAdd since multiple experts can contribute to the same output token.
// ============================================================================

__global__ __launch_bounds__(64)
void moe_grouped_weighted_scatter_prefill_kernel(
    float*       __restrict__ output,                // [seq_len, d_model]
    const float* __restrict__ expert_results,        // [total_slots, d_model]
    const int*   __restrict__ d_group_token_indices, // [total_slots] original token index
    const float* __restrict__ d_group_weights,       // [total_slots] routing weights
    int d_model)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int global_slot = blockIdx.y;

    if (n >= d_model) return;

    const int original_token = d_group_token_indices[global_slot];
    const float weight = d_group_weights[global_slot];

    const float val = expert_results[static_cast<size_t>(global_slot) * d_model + n];
    atomicAdd(output + static_cast<size_t>(original_token) * d_model + n, weight * val);
}


// ============================================================================
// C API: Full grouped MoE prefill pipeline (graph-capturable)
//
// Launches all 5 kernels in sequence on the same stream.
// NO host-device synchronization — all data flow is device-to-device.
// ============================================================================

extern "C" {

bool rocmMoE_grouped_prefill_pipeline(
    // Input
    const float*  d_hidden,              // [seq_len, d_model] FP32 hidden states
    // Weight descriptors (already on device)
    const void*   d_gate_desc_table,     // DeviceNativeVNNIMatrixDesc[num_experts]
    const void*   d_up_desc_table,       // DeviceNativeVNNIMatrixDesc[num_experts]
    const void*   d_down_desc_table,     // DeviceNativeVNNIMatrixDesc[num_experts]
    // Grouping data (from prepareExpertGroupsAsync)
    const int*    d_group_counts,        // [num_experts] per-expert token count
    const int*    d_group_offsets,       // [num_experts] prefix-sum offsets
    const int*    d_group_token_indices, // [total_slots] original token indices
    const float*  d_group_weights,       // [total_slots] routing weights
    const int*    d_active_expert_ids,   // [active_expert_slots] compact active experts, optional
    // Scratch buffers (caller provides, sized for total_slots)
    int8_t*       d_scratch_A_int8,      // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_scales,      // [total_slots, max_blocks_per_row]
    float*        d_scratch_gate,        // [total_slots, intermediate]
    float*        d_scratch_up,          // [total_slots, intermediate]
    int8_t*       d_scratch_swiglu_int8, // [total_slots, intermediate]
    float*        d_scratch_swiglu_scales,// [total_slots, blocks_per_intermediate]
    float*        d_scratch_down_out,    // [total_slots, d_model]
    // Output
    float*        d_output,              // [seq_len, d_model] accumulated output (must be pre-zeroed)
    // Dimensions
    int num_experts,
    int d_model,
    int intermediate,
    int max_tokens_per_expert,           // upper bound on tokens any expert gets
    int total_slots,                     // seq_len * top_k
    int active_expert_slots,             // fixed graph-capturable expert grid, 0 => all experts
    // Formats and device. Gate/up and down can differ on Qwen35 MoE
    // (for example Q4_K gate/up with Q5_K down), so keep them separate.
    uint8_t gateup_codebook_id,
    uint8_t down_codebook_id,
    int device_id,
    void* stream)
{
    if (num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
        max_tokens_per_expert <= 0 || total_slots <= 0 || !stream)
        return false;
    const bool use_active_expert_grid = d_active_expert_ids && active_expert_slots > 0;
    if (use_active_expert_grid &&
        (active_expert_slots > total_slots || active_expert_slots > num_experts))
        return false;
    const int expert_grid = use_active_expert_grid ? active_expert_slots : num_experts;

    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError(); // clear prior errors

    const int blocks_per_d_model = d_model / 32;
    const int blocks_per_intermediate = intermediate / 32;

    // =========================================================================
    // Kernel 1: Gather + Quantize hidden → A_int8 (FLAT grid over total_slots)
    // =========================================================================
    {
        dim3 grid(blocks_per_d_model, total_slots);
        dim3 block(32);
        hipLaunchKernelGGL(moe_grouped_gather_quantize_prefill_kernel,
            grid, block, 0, hip_stream,
            d_hidden, d_scratch_A_int8, d_scratch_scales,
            d_group_token_indices, d_model);
    }

    // =========================================================================
    // Kernel 2: Gate+Up GEMM (M-tiled: each block handles TILE_M=8 tokens)
    // =========================================================================
    {
        constexpr int TILE_N = 128;
        constexpr int TILE_M = 8;
        dim3 grid((intermediate + TILE_N - 1) / TILE_N,
                  (max_tokens_per_expert + TILE_M - 1) / TILE_M,
                  expert_grid);
        dim3 block(TILE_N);

        const auto* gate_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_gate_desc_table);
        const auto* up_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_up_desc_table);

        #define LAUNCH_GATEUP_PREFILL(FMT) \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(moe_grouped_gate_up_prefill_gemm_kernel<FMT>), \
                grid, block, 0, hip_stream, \
                d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                d_group_counts, d_group_offsets, d_active_expert_ids, \
                d_scratch_gate, d_scratch_up, intermediate, d_model)

        switch (static_cast<NativeVNNIFormatPrefill>(gateup_codebook_id)) {
            case NVNNI_PF_Q4_0:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q4_0); break;
            case NVNNI_PF_IQ4_NL:  LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ4_NL); break;
            case NVNNI_PF_Q4_1:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q4_1); break;
            case NVNNI_PF_Q5_0:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q5_0); break;
            case NVNNI_PF_Q5_1:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q5_1); break;
            case NVNNI_PF_Q6_K:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q6_K); break;
            case NVNNI_PF_Q3_K:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q3_K); break;
            case NVNNI_PF_Q2_K:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q2_K); break;
            case NVNNI_PF_IQ3_S:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ3_S); break;
            case NVNNI_PF_IQ3_XXS: LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ3_XXS); break;
            case NVNNI_PF_IQ2_S:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ2_S); break;
            case NVNNI_PF_IQ2_XS:  LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ2_XS); break;
            case NVNNI_PF_IQ2_XXS: LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ2_XXS); break;
            case NVNNI_PF_IQ1_S:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ1_S); break;
            case NVNNI_PF_IQ1_M:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ1_M); break;
            case NVNNI_PF_Q8_0:    LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q8_0); break;
            default:
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported gate/up codebook_id=%d\n", gateup_codebook_id);
                return false;
        }
        #undef LAUNCH_GATEUP_PREFILL
    }

    // =========================================================================
    // Kernel 3: SwiGLU + Quantize (FLAT grid over total_slots)
    // =========================================================================
    {
        dim3 grid(blocks_per_intermediate, total_slots);
        dim3 block(32);
        hipLaunchKernelGGL(moe_grouped_swiglu_quantize_prefill_kernel,
            grid, block, 0, hip_stream,
            d_scratch_gate, d_scratch_up,
            d_scratch_swiglu_int8, d_scratch_swiglu_scales,
            intermediate);
    }

    // =========================================================================
    // Kernel 4: Down GEMM (M-tiled: each block handles TILE_M=8 tokens)
    // =========================================================================
    {
        constexpr int TILE_N = 128;
        constexpr int TILE_M = 8;
        dim3 grid((d_model + TILE_N - 1) / TILE_N,
                  (max_tokens_per_expert + TILE_M - 1) / TILE_M,
                  expert_grid);
        dim3 block(TILE_N);

        const auto* down_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_down_desc_table);

        #define LAUNCH_DOWN_PREFILL(FMT) \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT>), \
                grid, block, 0, hip_stream, \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                d_group_counts, d_group_offsets, d_active_expert_ids, \
                d_scratch_down_out, d_model, intermediate)

        switch (static_cast<NativeVNNIFormatPrefill>(down_codebook_id)) {
            case NVNNI_PF_Q4_0:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q4_0); break;
            case NVNNI_PF_IQ4_NL:  LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ4_NL); break;
            case NVNNI_PF_Q4_1:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q4_1); break;
            case NVNNI_PF_Q5_0:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q5_0); break;
            case NVNNI_PF_Q5_1:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q5_1); break;
            case NVNNI_PF_Q6_K:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q6_K); break;
            case NVNNI_PF_Q3_K:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q3_K); break;
            case NVNNI_PF_Q2_K:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q2_K); break;
            case NVNNI_PF_IQ3_S:   LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ3_S); break;
            case NVNNI_PF_IQ3_XXS: LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ3_XXS); break;
            case NVNNI_PF_IQ2_S:   LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ2_S); break;
            case NVNNI_PF_IQ2_XS:  LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ2_XS); break;
            case NVNNI_PF_IQ2_XXS: LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ2_XXS); break;
            case NVNNI_PF_IQ1_S:   LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ1_S); break;
            case NVNNI_PF_IQ1_M:   LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ1_M); break;
            case NVNNI_PF_Q8_0:    LAUNCH_DOWN_PREFILL(NVNNI_PF_Q8_0); break;
            default:
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported down codebook_id=%d\n", down_codebook_id);
                return false;
        }
        #undef LAUNCH_DOWN_PREFILL
    }

    // =========================================================================
    // Kernel 5: Weighted Scatter-Add (FLAT grid over total_slots)
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 grid((d_model + TILE_N - 1) / TILE_N, total_slots);
        dim3 block(TILE_N);
        hipLaunchKernelGGL(moe_grouped_weighted_scatter_prefill_kernel,
            grid, block, 0, hip_stream,
            d_output, d_scratch_down_out,
            d_group_token_indices, d_group_weights,
            d_model);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] kernel launch error: %s\n",
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
