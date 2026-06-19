/**
 * @file ROCmMoEGroupedPrefillKernels.hip
 * @brief Grouped MoE prefill kernels for AMD MI50/MI60 (gfx906)
 *
 * These kernels process ALL experts in a single launch, eliminating the
 * per-expert host dispatch loop and hipStreamSynchronize that caused 99.9%
 * host overhead in prefill. Fully graph-capturable (no host-device sync).
 *
 * Pipeline (4 launches per MoE layer, replaces ~800):
 *   1. grouped_gather_quantize_prefill - gather + INT8 quantize for all experts
 *   2. grouped_gate_up_swiglu_prefill  - gate+up GEMM fused with SwiGLU quant
 *   3. grouped_down_prefill_gemm       - down projection for all experts
 *   4. grouped_weighted_scatter_prefill - scatter-add results to output
 *
 * Key design: all kernels read expert counts/offsets from DEVICE memory
 * (d_group_counts, d_group_offsets) — no host reads required.
 *
 * Grid strategy: dim3(N_tiles, max_M_tiles, num_experts)
 *   - blockIdx.z = expert index
 *   - Blocks where blockIdx.y * TILE_M >= count[expert] early-exit
 *   - This over-provisions M-tiles but avoids host sync entirely
 *
 * Verifier-sized MTP forwards usually have M=2..4.  Keep those rows on
 * dedicated small-M instantiations so each thread owns fewer accumulators and
 * the HIP compiler does not burn verifier time carrying TILE_M=8 register
 * state for rows that do not exist.
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

__host__ __device__ inline int rocm_moe_select_grouped_prefill_tile_m(
    int requested_tile_m,
    int max_tokens_per_expert)
{
    switch (requested_tile_m)
    {
    case 2:
    case 4:
    case 8:
        return requested_tile_m;
    default:
        break;
    }

    if (max_tokens_per_expert <= 2) return 2;
    if (max_tokens_per_expert <= 4) return 2;
    // The tile-4 instantiation is kept for explicit sweeps, but the
    // production Qwen3.6 MoE verifier gate favors tile-2 for 3/4-row buckets.
    // Keep the selector shape-based and generic rather than special-casing
    // individual codebooks.
    return 8;
}

// IQ4_NL codebook (must match GEMV file)
__device__ __constant__ int8_t k_iq4_pf_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

// Byte-pair LUT for IQ4_NL: low nibble -> low byte, high nibble -> high byte.
// This preserves the exact codebook math while halving variable codebook
// lookups in the verifier-sized grouped prefill kernels.
__device__ __constant__ uint16_t k_iq4_pf_pair_i16[256] = {
    0x8181, 0x8198, 0x81ad, 0x81bf, 0x81cf, 0x81dd, 0x81ea, 0x81f6,
    0x8101, 0x810d, 0x8119, 0x8126, 0x8135, 0x8145, 0x8159, 0x8171,
    0x9881, 0x9898, 0x98ad, 0x98bf, 0x98cf, 0x98dd, 0x98ea, 0x98f6,
    0x9801, 0x980d, 0x9819, 0x9826, 0x9835, 0x9845, 0x9859, 0x9871,
    0xad81, 0xad98, 0xadad, 0xadbf, 0xadcf, 0xaddd, 0xadea, 0xadf6,
    0xad01, 0xad0d, 0xad19, 0xad26, 0xad35, 0xad45, 0xad59, 0xad71,
    0xbf81, 0xbf98, 0xbfad, 0xbfbf, 0xbfcf, 0xbfdd, 0xbfea, 0xbff6,
    0xbf01, 0xbf0d, 0xbf19, 0xbf26, 0xbf35, 0xbf45, 0xbf59, 0xbf71,
    0xcf81, 0xcf98, 0xcfad, 0xcfbf, 0xcfcf, 0xcfdd, 0xcfea, 0xcff6,
    0xcf01, 0xcf0d, 0xcf19, 0xcf26, 0xcf35, 0xcf45, 0xcf59, 0xcf71,
    0xdd81, 0xdd98, 0xddad, 0xddbf, 0xddcf, 0xdddd, 0xddea, 0xddf6,
    0xdd01, 0xdd0d, 0xdd19, 0xdd26, 0xdd35, 0xdd45, 0xdd59, 0xdd71,
    0xea81, 0xea98, 0xeaad, 0xeabf, 0xeacf, 0xeadd, 0xeaea, 0xeaf6,
    0xea01, 0xea0d, 0xea19, 0xea26, 0xea35, 0xea45, 0xea59, 0xea71,
    0xf681, 0xf698, 0xf6ad, 0xf6bf, 0xf6cf, 0xf6dd, 0xf6ea, 0xf6f6,
    0xf601, 0xf60d, 0xf619, 0xf626, 0xf635, 0xf645, 0xf659, 0xf671,
    0x0181, 0x0198, 0x01ad, 0x01bf, 0x01cf, 0x01dd, 0x01ea, 0x01f6,
    0x0101, 0x010d, 0x0119, 0x0126, 0x0135, 0x0145, 0x0159, 0x0171,
    0x0d81, 0x0d98, 0x0dad, 0x0dbf, 0x0dcf, 0x0ddd, 0x0dea, 0x0df6,
    0x0d01, 0x0d0d, 0x0d19, 0x0d26, 0x0d35, 0x0d45, 0x0d59, 0x0d71,
    0x1981, 0x1998, 0x19ad, 0x19bf, 0x19cf, 0x19dd, 0x19ea, 0x19f6,
    0x1901, 0x190d, 0x1919, 0x1926, 0x1935, 0x1945, 0x1959, 0x1971,
    0x2681, 0x2698, 0x26ad, 0x26bf, 0x26cf, 0x26dd, 0x26ea, 0x26f6,
    0x2601, 0x260d, 0x2619, 0x2626, 0x2635, 0x2645, 0x2659, 0x2671,
    0x3581, 0x3598, 0x35ad, 0x35bf, 0x35cf, 0x35dd, 0x35ea, 0x35f6,
    0x3501, 0x350d, 0x3519, 0x3526, 0x3535, 0x3545, 0x3559, 0x3571,
    0x4581, 0x4598, 0x45ad, 0x45bf, 0x45cf, 0x45dd, 0x45ea, 0x45f6,
    0x4501, 0x450d, 0x4519, 0x4526, 0x4535, 0x4545, 0x4559, 0x4571,
    0x5981, 0x5998, 0x59ad, 0x59bf, 0x59cf, 0x59dd, 0x59ea, 0x59f6,
    0x5901, 0x590d, 0x5919, 0x5926, 0x5935, 0x5945, 0x5959, 0x5971,
    0x7181, 0x7198, 0x71ad, 0x71bf, 0x71cf, 0x71dd, 0x71ea, 0x71f6,
    0x7101, 0x710d, 0x7119, 0x7126, 0x7135, 0x7145, 0x7159, 0x7171
};

__device__ __forceinline__ void pf_iq4_decode_raw_pair(uint32_t raw, int32_t &lo, int32_t &hi)
{
    const uint32_t p0 = k_iq4_pf_pair_i16[raw & 0xffu];
    const uint32_t p1 = k_iq4_pf_pair_i16[(raw >> 8) & 0xffu];
    const uint32_t p2 = k_iq4_pf_pair_i16[(raw >> 16) & 0xffu];
    const uint32_t p3 = k_iq4_pf_pair_i16[(raw >> 24) & 0xffu];
    lo = static_cast<int32_t>(
        (p0 & 0x00ffu) |
        ((p1 & 0x00ffu) << 8) |
        ((p2 & 0x00ffu) << 16) |
        ((p3 & 0x00ffu) << 24));
    hi = static_cast<int32_t>(
        ((p0 >> 8) & 0x00ffu) |
        (p1 & 0xff00u) |
        ((p2 & 0xff00u) << 8) |
        ((p3 & 0xff00u) << 16));
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
        // IQ4_NL: byte-pair LUT decode. Each payload byte contains two
        // nibbles; the pair table returns both codebook values at once.
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            pf_iq4_decode_raw_pair(raw, packed_groups[g], packed_groups[g + 4]);
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
 * @brief Compute one NativeVNNI dot product over a bounded K-block range.
 *
 * Tiny MTP verifier batches need more parallelism across the reduction
 * dimension than a single full-K block provides.  This helper keeps the format
 * math shared by the split-K prefill path: every Q/K/IQ codebook follows the
 * same dequantization, dual-scale, min-correction, and IQ1_M delta rules as
 * the fused gate/up prefill kernel above.
 */
template <NativeVNNIFormatPrefill FMT>
__device__ __forceinline__ float moe_prefill_dot_desc_range(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int b_start,
    int b_end)
{
    using Traits = PrefillNVNNITraits<FMT>;

    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
    float acc = 0.0f;

#pragma unroll 1
    for (int b = b_start; b < b_end; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) +
                              static_cast<size_t>(n);
        int32_t packed_groups[8];
        constexpr int PB = Traits::payload_bytes;
        const uint8_t* payload = payload_base + linear * PB;
        moe_prefill_decode_groups<FMT>(payload, packed_groups);

        const int32_t* a4 = reinterpret_cast<const int32_t*>(slot_A + b * 32);
        const float scale_ab = slot_scales[b];

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
            const float s_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
            const float s_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
            acc += (static_cast<float>(acc_lo) * s_lo +
                    static_cast<float>(acc_hi) * s_hi) * scale_ab;

            if constexpr (Traits::is_dual_scale_asym)
            {
                constexpr int32_t ones = 0x01010101;
                int32_t sum_a_lo = 0;
                int32_t sum_a_hi = 0;
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
                acc += (static_cast<float>(sum_a_lo) * emin_lo +
                        static_cast<float>(sum_a_hi) * emin_hi) * scale_ab;
            }

            if constexpr (FMT == NVNNI_PF_IQ1_M)
            {
                constexpr float IQ1S_DELTA = 0.125f;
                constexpr int32_t ones = 0x01010101;
                const uint8_t qh0 = payload[4];
                const uint8_t qh1 = payload[5];

                const int32_t sa_g0 = __builtin_amdgcn_sdot4(
                    a4[1], ones, __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
                const int32_t sa_g1 = __builtin_amdgcn_sdot4(
                    a4[3], ones, __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
                const int32_t sa_g2 = __builtin_amdgcn_sdot4(
                    a4[5], ones, __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
                const int32_t sa_g3 = __builtin_amdgcn_sdot4(
                    a4[7], ones, __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

                const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                // IQ1_M has two dequant scales. Groups 0/1 use the low scale,
                // groups 2/3 use the high scale. This matches ROCm GEMV and
                // CUDA MoE; using s_lo for all groups drifts verifier rows.
                acc += ((d0 * static_cast<float>(sa_g0) + d1 * static_cast<float>(sa_g1)) * s_lo +
                        (d2 * static_cast<float>(sa_g2) + d3 * static_cast<float>(sa_g3)) * s_hi) *
                       scale_ab;
            }
        }
        else
        {
            int32_t dot = 0;
#pragma unroll
            for (int g = 0; g < 8; ++g)
                dot = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], dot, false);

            const uint16_t scale_bits = scale_base[linear];
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            acc += static_cast<float>(dot) * block_d * scale_ab;

            if constexpr (Traits::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                int32_t sum_a = 0;
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                const uint16_t min_bits = min_base[linear];
                const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                acc += static_cast<float>(sum_a) * block_m * scale_ab;
            }
        }
    }

    return acc;
}

/**
 * @brief Split-K gate/up prefill kernel for tiny verifier batches.
 *
 * Each CTA owns one output-column tile, one token group, one active expert,
 * and one K partition.  It writes FP32 partials into
 * `[grouped_slot][k_partition][intermediate]`.  A small reduce kernel then
 * restores the normal contiguous gate/up scratch layout consumed by the
 * existing SwiGLU quantizer.
 */
template <NativeVNNIFormatPrefill FMT, int TILE_M>
__global__ __launch_bounds__(64)
void moe_grouped_gate_up_prefill_kpart_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ d_group_counts,
    const int* __restrict__ d_group_offsets,
    const int* __restrict__ d_active_expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int N,
    int K,
    int k_partitions)
{
    static_assert(TILE_M == 2 || TILE_M == 4 || TILE_M == 8,
                  "ROCm grouped MoE split-K prefill only instantiates verifier-sized TILE_M buckets");
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_group = blockIdx.y;
    const int expert_grid_index = blockIdx.z / k_partitions;
    const int k_part = blockIdx.z - expert_grid_index * k_partitions;
    if (n >= N || k_part >= k_partitions)
        return;

    const int expert_id = d_active_expert_ids ? d_active_expert_ids[expert_grid_index]
                                              : expert_grid_index;
    if (expert_id < 0)
        return;

    const int count = d_group_counts[expert_id];
    const int first_token = token_group * TILE_M;
    if (first_token >= count)
        return;

    const int tokens_in_group = min(TILE_M, count - first_token);
    const int first_slot = d_group_offsets[expert_id] + first_token;
    const int blocks_per_row = K / 32;
    const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    const DeviceNativeVNNIMatrixDesc& gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc& up_desc = up_descs[expert_id];
    if (gate_desc.codebook_id != static_cast<uint8_t>(FMT) ||
        up_desc.codebook_id != static_cast<uint8_t>(FMT))
        return;

#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
    {
        if (m >= tokens_in_group)
            break;
        const int slot = first_slot + m;
        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);
        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            continue;
        }

        const int8_t* slot_A = A_int8 + static_cast<size_t>(slot) * static_cast<size_t>(K);
        const float* slot_scales =
            scales_A_blockwise + static_cast<size_t>(slot) * static_cast<size_t>(blocks_per_row);
        gate_partials[partial_index] =
            moe_prefill_dot_desc_range<FMT>(gate_desc, n, slot_A, slot_scales, N, b_start, b_end);
        up_partials[partial_index] =
            moe_prefill_dot_desc_range<FMT>(up_desc, n, slot_A, slot_scales, N, b_start, b_end);
    }
}

/**
 * @brief Sum split-K gate/up partials into the standard grouped-prefill scratch.
 */
__global__ __launch_bounds__(64)
void moe_grouped_gate_up_prefill_kpart_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    float* __restrict__ gate_output,
    float* __restrict__ up_output,
    int total_slots,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= total_slots || n >= N)
        return;

    const size_t slot_base =
        static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t idx =
            slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
        gate_sum += gate_partials[idx];
        up_sum += up_partials[idx];
    }

    gate_output[static_cast<size_t>(slot) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
        gate_sum;
    up_output[static_cast<size_t>(slot) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
        up_sum;
}

template <NativeVNNIFormatPrefill FMT, int TILE_M>
__global__ __launch_bounds__(128)
void moe_grouped_gate_up_swiglu_prefill_gemm_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, K] quantized activations
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, blocks_per_row]
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs, // [num_experts]
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,   // [num_experts]
    const int*     __restrict__ d_group_counts,      // [num_experts]
    const int*     __restrict__ d_group_offsets,     // [num_experts]
    const int*     __restrict__ d_active_expert_ids, // [active_expert_slots] or null
    int8_t*        __restrict__ swiglu_int8,         // [total_slots, N]
    float*         __restrict__ swiglu_scales,       // [total_slots, N / 32]
    int N, int K)
{
    static_assert(TILE_M == 2 || TILE_M == 4 || TILE_M == 8,
                  "ROCm grouped MoE prefill only instantiates verifier-sized TILE_M buckets");
    constexpr int TILE_N = 128;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_group = blockIdx.y;  // group of TILE_M tokens
    const int expert_id = d_active_expert_ids ? d_active_expert_ids[blockIdx.z] : static_cast<int>(blockIdx.z);
    if (expert_id < 0) return;

    const int count = d_group_counts[expert_id];
    const int first_token = token_group * TILE_M;
    if (first_token >= count) return;

    // How many tokens in this group (1..TILE_M)
    const int tokens_in_group = min(TILE_M, count - first_token);
    const int offset = d_group_offsets[expert_id];
    const int blocks_per_row = K / 32;
    const bool active = n < N;

    // Get weight descriptors for this expert
    const DeviceNativeVNNIMatrixDesc& gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc& up_desc = up_descs[expert_id];
    if (gate_desc.codebook_id != static_cast<uint8_t>(FMT) ||
        up_desc.codebook_id != static_cast<uint8_t>(FMT))
        return;

    const uint8_t* gate_payload_base = gate_desc.payload;
    const uint16_t* gate_scale_base = static_cast<const uint16_t*>(gate_desc.scales);
    const uint16_t* gate_min_base = static_cast<const uint16_t*>(gate_desc.mins);
    const uint32_t* gate_emin_base = static_cast<const uint32_t*>(gate_desc.emins);

    const uint8_t* up_payload_base = up_desc.payload;
    const uint16_t* up_scale_base = static_cast<const uint16_t*>(up_desc.scales);
    const uint16_t* up_min_base = static_cast<const uint16_t*>(up_desc.mins);
    const uint32_t* up_emin_base = static_cast<const uint32_t*>(up_desc.emins);

    /*
     * Stage the verifier-sized activation tile once per output-column block.
     * Without this, every one of the 128 N-lane threads reloads the same
     * TILE_Mx32 activation values from global memory for each K block.  CUDA's
     * MoE verifier path already uses this pattern; keeping ROCm aligned avoids
     * backend-specific performance drift while preserving graph capturability.
     */
    __shared__ __align__(16) int32_t s_a[TILE_M * 8];
    __shared__ float s_scale[TILE_M];

    // Accumulators for TILE_M tokens
    float gate_acc[TILE_M] = {};
    float up_acc[TILE_M] = {};

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);

        // Decode weights ONCE per K-block (shared across all TILE_M tokens)
        int32_t gate_groups[8];
        int32_t up_groups[8];
        if (active)
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* gate_payload = gate_payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(gate_payload, gate_groups);

            const uint8_t* up_payload = up_payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(up_payload, up_groups);
        }

        const int staged_values = tokens_in_group * 8;
        for (int idx = threadIdx.x; idx < staged_values; idx += TILE_N)
        {
            const int m = idx >> 3;
            const int g = idx & 7;
            const int global_slot = offset + first_token + m;
            s_a[idx] = reinterpret_cast<const int32_t*>(
                A_int8 + static_cast<size_t>(global_slot) * K + b * 32)[g];
        }
        for (int m = threadIdx.x; m < tokens_in_group; m += TILE_N)
        {
            const int global_slot = offset + first_token + m;
            s_scale[m] = scales_A_blockwise[
                static_cast<size_t>(global_slot) * blocks_per_row + b];
        }
        __syncthreads();

        // Process each of TILE_M tokens with the same decoded weights
        if (active)
        {
            #pragma unroll
            for (int m = 0; m < TILE_M; ++m)
            {
                if (m >= tokens_in_group) break;

                const int32_t* a4 = s_a + m * 8;
                const float scale_ab = s_scale[m];

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

                        const float gate_d0 = (gate_qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float gate_d1 = (gate_qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float gate_d2 = (gate_qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float gate_d3 = (gate_qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        gate_acc[m] += ((gate_d0 * static_cast<float>(sa_g0) +
                                         gate_d1 * static_cast<float>(sa_g1)) * gate_s_lo +
                                        (gate_d2 * static_cast<float>(sa_g2) +
                                         gate_d3 * static_cast<float>(sa_g3)) * gate_s_hi) *
                                       scale_ab;

                        const uint8_t* up_pay = up_payload_base + linear * PB;
                        const uint8_t up_qh0 = up_pay[4], up_qh1 = up_pay[5];
                        const float up_d0 = (up_qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float up_d1 = (up_qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float up_d2 = (up_qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float up_d3 = (up_qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        up_acc[m] += ((up_d0 * static_cast<float>(sa_g0) +
                                       up_d1 * static_cast<float>(sa_g1)) * up_s_lo +
                                      (up_d2 * static_cast<float>(sa_g2) +
                                       up_d3 * static_cast<float>(sa_g3)) * up_s_hi) *
                                     scale_ab;
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
        __syncthreads();
    }

    /*
     * Fused SwiGLU + blockwise INT8 quantization.
     *
     * CUDA already uses this contract for verifier-prefill.  Keeping ROCm on
     * the same graph-native path removes the FP32 gate/up global round trip and
     * the extra SwiGLU launch while preserving the same row/block layout consumed
     * by the down-projection kernel.  Each warp owns one 32-column quant block.
     */
    const int lane = threadIdx.x & 31;
    const int blocks_per_row_out = N / 32;
    const int quant_block = n >> 5;
#pragma unroll
    for (int m = 0; m < TILE_M; ++m)
    {
        if (m >= tokens_in_group) break;

        const int global_slot = offset + first_token + m;
        const float gate_value = gate_acc[m];
        const float value = active ? ((gate_value / (1.0f + expf(-gate_value))) * up_acc[m]) : 0.0f;

        float abs_val = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

        const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        if (active)
        {
            if (lane == 0)
                swiglu_scales[static_cast<size_t>(global_slot) * blocks_per_row_out + quant_block] = scale;
            const float qval = value / scale;
            swiglu_int8[static_cast<size_t>(global_slot) * N + n] =
                static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
        }
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

template <NativeVNNIFormatPrefill FMT, int TILE_M, int TILE_N>
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
    static_assert(TILE_M == 2 || TILE_M == 4 || TILE_M == 8,
                  "ROCm grouped MoE prefill only instantiates verifier-sized TILE_M buckets");
    static_assert(TILE_N == 64 || TILE_N == 128,
                  "ROCm grouped MoE down prefill only supports 64- or 128-column tiles");
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_group = blockIdx.y;
    const int expert_id = d_active_expert_ids ? d_active_expert_ids[blockIdx.z] : static_cast<int>(blockIdx.z);
    if (expert_id < 0) return;

    const int count = d_group_counts[expert_id];
    const int first_token = token_group * TILE_M;
    if (first_token >= count) return;

    const int tokens_in_group = min(TILE_M, count - first_token);
    const int offset = d_group_offsets[expert_id];
    const int blocks_per_row = K / 32;
    const bool active = n < N;

    const DeviceNativeVNNIMatrixDesc& desc = down_descs[expert_id];
    if (desc.codebook_id != static_cast<uint8_t>(FMT))
        return;

    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);

    /*
     * The down projection consumes the quantized SwiGLU rows produced by the
     * previous kernel. Stage those TILE_Mx32 rows once per block before all
     * output-column lanes reuse them. Tiny verifier batches use TILE_N=64 to
     * match CUDA's active-verifier geometry and reduce the per-block lane
     * footprint when the grouped path has only M=2..4 rows to amortize work.
     */
    __shared__ __align__(16) int32_t s_a[TILE_M * 8];
    __shared__ float s_scale[TILE_M];

    float acc[TILE_M] = {};

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);

        // Decode weights ONCE per K-block (shared across TILE_M tokens)
        int32_t packed_groups[8];
        if (active)
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* payload = payload_base + linear * PB;
            moe_prefill_decode_groups<FMT>(payload, packed_groups);
        }

        const int staged_values = tokens_in_group * 8;
        for (int idx = threadIdx.x; idx < staged_values; idx += TILE_N)
        {
            const int m = idx >> 3;
            const int g = idx & 7;
            const int global_slot = offset + first_token + m;
            s_a[idx] = reinterpret_cast<const int32_t*>(
                A_int8 + static_cast<size_t>(global_slot) * K + b * 32)[g];
        }
        for (int m = threadIdx.x; m < tokens_in_group; m += TILE_N)
        {
            const int global_slot = offset + first_token + m;
            s_scale[m] = scales_A_blockwise[
                static_cast<size_t>(global_slot) * blocks_per_row + b];
        }
        __syncthreads();

        // Process TILE_M tokens with decoded weights
        if (active)
        {
            #pragma unroll
            for (int m = 0; m < TILE_M; ++m)
            {
                if (m >= tokens_in_group) break;

                const int32_t* a4 = s_a + m * 8;
                const float scale_ab = s_scale[m];

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

                        const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        acc[m] += ((d0 * static_cast<float>(sa_g0) +
                                    d1 * static_cast<float>(sa_g1)) * s_lo +
                                   (d2 * static_cast<float>(sa_g2) +
                                    d3 * static_cast<float>(sa_g3)) * s_hi) *
                                  scale_ab;
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
        __syncthreads();
    }

    // Write outputs
    if (active)
    {
        #pragma unroll
        for (int m = 0; m < TILE_M; ++m)
        {
            if (m >= tokens_in_group) break;
            const int global_slot = offset + first_token + m;
            output[static_cast<size_t>(global_slot) * N + n] = acc[m];
        }
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

/**
 * @brief Ordered verifier scatter for small grouped-prefill batches.
 *
 * The flat scatter above is a good generic fallback, but combined MTP verifier
 * rows have a fixed route order for each token: routed top-k plus one shared
 * expert.  CUDA uses this map to avoid atomics and to sum each token's routes
 * deterministically in one thread.  ROCm must follow the same contract so the
 * vLLM-style combined verifier path gets the intended speedup on both GPU
 * backends.
 */
__global__ __launch_bounds__(64)
void moe_grouped_weighted_scatter_ordered_prefill_kernel(
    float*       __restrict__ output,              // [seq_len, d_model]
    const float* __restrict__ expert_results,      // [total_slots, d_model]
    const int*   __restrict__ original_to_grouped, // [seq_len * top_k]
    const float* __restrict__ grouped_weights,     // [total_slots]
    int seq_len,
    int top_k,
    int d_model)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= seq_len || n >= d_model) return;

    float sum = 0.0f;
#pragma unroll 1
    for (int k = 0; k < top_k; ++k)
    {
        const int original_slot = token * top_k + k;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0) continue;
        const float weight = grouped_weights[grouped_slot];
        const float value = expert_results[static_cast<size_t>(grouped_slot) * d_model + n];
        sum += weight * value;
    }
    output[static_cast<size_t>(token) * d_model + n] = sum;
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
    const int*    d_original_to_grouped,  // [seq_len * top_k] route slot -> grouped slot, optional
    const float*  d_group_weights,       // [total_slots] routing weights
    const int*    d_active_expert_ids,   // [active_expert_slots] compact active experts, optional
    const int*    d_single_expert_ids,   // [1] optional appended singleton active expert
    // Scratch buffers (caller provides, sized for total_slots)
    int8_t*       d_scratch_A_int8,      // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_scales,      // [total_slots, max_blocks_per_row]
    float*        d_scratch_gate,        // [total_slots, intermediate]
    float*        d_scratch_up,          // [total_slots, intermediate]
    float*        d_gate_partials,       // [total_slots, k_partitions, intermediate], optional
    float*        d_up_partials,         // [total_slots, k_partitions, intermediate], optional
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
    int top_k,                           // routes per token, needed by ordered scatter
    int active_expert_slots,             // fixed graph-capturable expert grid, 0 => all experts
    // Formats and device. Gate/up and down can differ on Qwen35 MoE
    // (for example Q4_K gate/up with Q5_K down), so keep them separate.
    uint8_t gateup_codebook_id,
    uint8_t down_codebook_id,
    uint32_t gateup_codebook_mask,
    uint32_t down_codebook_mask,
    uint8_t single_gateup_codebook_id,
    uint8_t single_down_codebook_id,
    int requested_tile_m,
    int gateup_k_partitions,
    int device_id,
    void* stream)
{
    if (num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
        max_tokens_per_expert <= 0 || total_slots <= 0 || top_k <= 0 || !stream)
        return false;
    const bool use_active_expert_grid = d_active_expert_ids && active_expert_slots > 0;
    if (use_active_expert_grid &&
        (active_expert_slots > total_slots || active_expert_slots > num_experts))
        return false;
    const int expert_grid = use_active_expert_grid ? active_expert_slots : num_experts;

    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError(); // clear prior errors
    const bool use_gateup_kpart =
        use_active_expert_grid &&
        max_tokens_per_expert <= 4 &&
        gateup_k_partitions > 1 &&
        d_scratch_gate &&
        d_scratch_up &&
        d_gate_partials &&
        d_up_partials;

    const int blocks_per_d_model = d_model / 32;

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

    /*
     * A shared-expert prefill group has exactly one expert and one routed slot
     * per verifier row.  TILE_M=2 keeps the verifier live register set smaller
     * for M=3/4, while routed expert groups continue to use the measured
     * default selector.  This is role-based, not codebook-specific.
     */
    const bool shared_expert_group = num_experts == 1;
    const int tile_m = shared_expert_group
                           ? 2
                           : rocm_moe_select_grouped_prefill_tile_m(requested_tile_m,
                                                                    max_tokens_per_expert);
    // =========================================================================
    // Kernel 2: Gate+Up GEMM.  Ordinary prefill keeps the fused SwiGLU path.
    // Tiny verifier batches can instead split K to expose more parallelism,
    // then reduce into FP32 gate/up scratch for the existing SwiGLU quantizer.
    // =========================================================================
    {
        constexpr int kFusedTileN = 128;
        constexpr int kKPartTileN = 64;
        const int tile_n = use_gateup_kpart ? kKPartTileN : kFusedTileN;
        dim3 block(tile_n);
        const int n_tiles = (intermediate + tile_n - 1) / tile_n;
        const int m_tiles = (max_tokens_per_expert + tile_m - 1) / tile_m;

        const auto* gate_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_gate_desc_table);
        const auto* up_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_up_desc_table);

        #define LAUNCH_GATEUP_PREFILL_TM(FMT, TM) \
            do { \
                dim3 grid(n_tiles, m_tiles, expert_grid); \
                hipLaunchKernelGGL( \
                    HIP_KERNEL_NAME(moe_grouped_gate_up_swiglu_prefill_gemm_kernel<FMT, TM>), \
                    grid, block, 0, hip_stream, \
                    d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                    d_group_counts, d_group_offsets, d_active_expert_ids, \
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales, intermediate, d_model); \
            } while (0)
        #define LAUNCH_GATEUP_KPART_PREFILL_TM(FMT, TM) \
            do { \
                dim3 grid(n_tiles, m_tiles, expert_grid * gateup_k_partitions); \
                hipLaunchKernelGGL( \
                    HIP_KERNEL_NAME(moe_grouped_gate_up_prefill_kpart_kernel<FMT, TM>), \
                    grid, block, 0, hip_stream, \
                    d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                    d_group_counts, d_group_offsets, d_active_expert_ids, \
                    d_gate_partials, d_up_partials, intermediate, d_model, gateup_k_partitions); \
            } while (0)
        #define LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, TM) \
            do { \
                if (use_gateup_kpart) { \
                    dim3 single_grid(n_tiles, m_tiles, gateup_k_partitions); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_gate_up_prefill_kpart_kernel<FMT, TM>), \
                        single_grid, block, 0, hip_stream, \
                        d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                        d_group_counts, d_group_offsets, d_single_expert_ids, \
                        d_gate_partials, d_up_partials, intermediate, d_model, gateup_k_partitions); \
                } else { \
                    dim3 single_grid(n_tiles, m_tiles, 1); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_gate_up_swiglu_prefill_gemm_kernel<FMT, TM>), \
                        single_grid, block, 0, hip_stream, \
                        d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                        d_group_counts, d_group_offsets, d_single_expert_ids, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, intermediate, d_model); \
                } \
            } while (0)
        #define LAUNCH_GATEUP_SINGLE_PREFILL(FMT) \
            do { \
                if (use_gateup_kpart) { \
                    if (tile_m <= 2)      LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 2); \
                    else if (tile_m <= 4) LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 4); \
                    else                  LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 8); \
                } else { \
                    if (tile_m <= 2)      LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 2); \
                    else if (tile_m <= 4) LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 4); \
                    else                  LAUNCH_GATEUP_SINGLE_PREFILL_TM(FMT, 8); \
                } \
            } while (0)
        #define LAUNCH_GATEUP_PREFILL(FMT) \
            do { \
                if (use_gateup_kpart) { \
                    if (tile_m <= 2)      LAUNCH_GATEUP_KPART_PREFILL_TM(FMT, 2); \
                    else if (tile_m <= 4) LAUNCH_GATEUP_KPART_PREFILL_TM(FMT, 4); \
                    else                  LAUNCH_GATEUP_KPART_PREFILL_TM(FMT, 8); \
                } else { \
                    if (tile_m <= 2)      LAUNCH_GATEUP_PREFILL_TM(FMT, 2); \
                    else if (tile_m <= 4) LAUNCH_GATEUP_PREFILL_TM(FMT, 4); \
                    else                  LAUNCH_GATEUP_PREFILL_TM(FMT, 8); \
                } \
            } while (0)

        bool launched_gateup = false;
        #define LAUNCH_GATEUP_IF_PRESENT(FMT) \
            do { \
                if (gateup_codebook_mask & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
                    LAUNCH_GATEUP_PREFILL(FMT); \
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
        if (d_single_expert_ids)
        {
            #define LAUNCH_GATEUP_SINGLE_IF(FMT) \
                do { \
                    if (static_cast<uint8_t>(FMT) == single_gateup_codebook_id) { \
                        LAUNCH_GATEUP_SINGLE_PREFILL(FMT); \
                        launched_gateup = true; \
                    } \
                } while (0)
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q4_0);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ4_NL);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q4_1);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q5_0);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q5_1);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q6_K);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q3_K);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q2_K);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ3_S);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ3_XXS);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ2_S);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ2_XS);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ2_XXS);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ1_S);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_IQ1_M);
            LAUNCH_GATEUP_SINGLE_IF(NVNNI_PF_Q8_0);
            #undef LAUNCH_GATEUP_SINGLE_IF
        }
        if (!launched_gateup)
        {
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported gate/up codebook_id=%d mask=0x%x\n",
                        gateup_codebook_id, gateup_codebook_mask);
                return false;
        }
        #undef LAUNCH_GATEUP_IF_PRESENT
        #undef LAUNCH_GATEUP_SINGLE_PREFILL
        #undef LAUNCH_GATEUP_SINGLE_PREFILL_TM
        #undef LAUNCH_GATEUP_PREFILL
        #undef LAUNCH_GATEUP_KPART_PREFILL_TM
        #undef LAUNCH_GATEUP_PREFILL_TM

        if (use_gateup_kpart)
        {
            constexpr int TILE_N = 64;
            dim3 reduce_grid((intermediate + TILE_N - 1) / TILE_N, total_slots);
            dim3 reduce_block(TILE_N);
            hipLaunchKernelGGL(
                moe_grouped_gate_up_prefill_kpart_reduce_kernel,
                reduce_grid, reduce_block, 0, hip_stream,
                d_gate_partials, d_up_partials,
                d_scratch_gate, d_scratch_up,
                total_slots, intermediate, gateup_k_partitions);

            dim3 swiglu_grid(intermediate / 32, total_slots);
            dim3 swiglu_block(32);
            hipLaunchKernelGGL(
                moe_grouped_swiglu_quantize_prefill_kernel,
                swiglu_grid, swiglu_block, 0, hip_stream,
                d_scratch_gate, d_scratch_up,
                d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                intermediate);
        }
    }

    // =========================================================================
    // Kernel 3: Down GEMM (same TILE_M bucket as gate/up)
    // =========================================================================
    {
        /*
         * CUDA's verifier path uses narrower down tiles for tiny active-expert
         * batches. Mirror that geometry on ROCm so M=2..4 rows do not pay a
         * full 128-lane column tile when there are too few rows to amortize it.
         */
        const bool tiny_active_verifier =
            use_active_expert_grid && max_tokens_per_expert <= 4;
        const int down_tile_n = tiny_active_verifier ? 64 : 128;
        dim3 block(down_tile_n);
        const int n_tiles = (d_model + down_tile_n - 1) / down_tile_n;
        const int m_tiles = (max_tokens_per_expert + tile_m - 1) / tile_m;

        const auto* down_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_down_desc_table);

        #define LAUNCH_DOWN_PREFILL_TM(FMT, TM) \
            do { \
                dim3 grid(n_tiles, m_tiles, expert_grid); \
                if (down_tile_n == 64) { \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT, TM, 64>), \
                        grid, block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, d_active_expert_ids, \
                        d_scratch_down_out, d_model, intermediate); \
                } else { \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT, TM, 128>), \
                        grid, block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, d_active_expert_ids, \
                        d_scratch_down_out, d_model, intermediate); \
                } \
            } while (0)
        #define LAUNCH_DOWN_SINGLE_PREFILL_TM(FMT, TM) \
            do { \
                dim3 single_grid(n_tiles, m_tiles, 1); \
                if (down_tile_n == 64) { \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT, TM, 64>), \
                        single_grid, block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, d_single_expert_ids, \
                        d_scratch_down_out, d_model, intermediate); \
                } else { \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT, TM, 128>), \
                        single_grid, block, 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_group_counts, d_group_offsets, d_single_expert_ids, \
                        d_scratch_down_out, d_model, intermediate); \
                } \
            } while (0)
        #define LAUNCH_DOWN_PREFILL(FMT) \
            do { \
                if (tile_m <= 2)      LAUNCH_DOWN_PREFILL_TM(FMT, 2); \
                else if (tile_m <= 4) LAUNCH_DOWN_PREFILL_TM(FMT, 4); \
                else                  LAUNCH_DOWN_PREFILL_TM(FMT, 8); \
            } while (0)

        bool launched_down = false;
        #define LAUNCH_DOWN_IF_PRESENT(FMT) \
            do { \
                if (down_codebook_mask & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
                    LAUNCH_DOWN_PREFILL(FMT); \
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
        if (d_single_expert_ids)
        {
            #define LAUNCH_DOWN_SINGLE_IF(FMT) \
                do { \
                    if (static_cast<uint8_t>(FMT) == single_down_codebook_id) { \
                        if (tile_m <= 2)      LAUNCH_DOWN_SINGLE_PREFILL_TM(FMT, 2); \
                        else if (tile_m <= 4) LAUNCH_DOWN_SINGLE_PREFILL_TM(FMT, 4); \
                        else                  LAUNCH_DOWN_SINGLE_PREFILL_TM(FMT, 8); \
                        launched_down = true; \
                    } \
                } while (0)
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q4_0);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ4_NL);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q4_1);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q5_0);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q5_1);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q6_K);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q3_K);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q2_K);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ3_S);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ3_XXS);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ2_S);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ2_XS);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ2_XXS);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ1_S);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_IQ1_M);
            LAUNCH_DOWN_SINGLE_IF(NVNNI_PF_Q8_0);
            #undef LAUNCH_DOWN_SINGLE_IF
        }
        if (!launched_down)
        {
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported down codebook_id=%d mask=0x%x\n",
                        down_codebook_id, down_codebook_mask);
                return false;
        }
        #undef LAUNCH_DOWN_IF_PRESENT
        #undef LAUNCH_DOWN_PREFILL
        #undef LAUNCH_DOWN_SINGLE_PREFILL_TM
        #undef LAUNCH_DOWN_PREFILL_TM
    }

    // =========================================================================
    // Kernel 4: Weighted Scatter-Add (FLAT grid over total_slots)
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 block(TILE_N);
        if (d_original_to_grouped)
        {
            const int seq_len = total_slots / top_k;
            dim3 grid((d_model + TILE_N - 1) / TILE_N, seq_len);
            hipLaunchKernelGGL(moe_grouped_weighted_scatter_ordered_prefill_kernel,
                grid, block, 0, hip_stream,
                d_output, d_scratch_down_out,
                d_original_to_grouped, d_group_weights,
                seq_len, top_k, d_model);
        }
        else
        {
            dim3 grid((d_model + TILE_N - 1) / TILE_N, total_slots);
            hipLaunchKernelGGL(moe_grouped_weighted_scatter_prefill_kernel,
                grid, block, 0, hip_stream,
                d_output, d_scratch_down_out,
                d_group_token_indices, d_group_weights,
                d_model);
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
