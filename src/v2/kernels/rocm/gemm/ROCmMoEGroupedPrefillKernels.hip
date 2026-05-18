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
    NVNNI_PF_Q4_0   = 0,
    NVNNI_PF_IQ4_NL = 4,
    NVNNI_PF_Q4_1   = 5,
    NVNNI_PF_Q5_0   = 6,
    NVNNI_PF_Q5_1   = 7,
    NVNNI_PF_Q6_K   = 8,
};

// Compile-time format properties for prefill
template <NativeVNNIFormatPrefill FMT>
struct PrefillNVNNITraits {
    static constexpr bool is_asymmetric = (FMT == NVNNI_PF_Q4_1 || FMT == NVNNI_PF_Q5_1);
    static constexpr bool is_dual_scale = (FMT == NVNNI_PF_Q6_K);
    static constexpr int payload_bytes =
        (FMT == NVNNI_PF_Q4_0 || FMT == NVNNI_PF_IQ4_NL || FMT == NVNNI_PF_Q4_1) ? 16 :
        (FMT == NVNNI_PF_Q5_0 || FMT == NVNNI_PF_Q5_1) ? 20 :
        (FMT == NVNNI_PF_Q6_K) ? 24 : 16;
};

// IQ4_NL codebook (must match GEMV file)
__device__ __constant__ int8_t k_iq4_pf_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};


// ============================================================================
// Kernel 1: Grouped Gather + Quantize (all experts, single launch)
//
// For each expert e:
//   - Reads d_group_counts[e] tokens from hidden at indices d_group_token_indices[offset..offset+count]
//   - Quantizes each row to INT8 with per-block (32-element) scales
//   - Writes to contiguous output: A_int8[expert_base + token * K + ...]
//
// Grid: dim3(blocks_per_row, max_tokens_per_expert, num_experts)
//   blockIdx.x = quantization block within the row
//   blockIdx.y = token within this expert's batch
//   blockIdx.z = expert index
// Block: 32 threads (warp-cooperative quantization)
// ============================================================================

__global__ __launch_bounds__(32)
void moe_grouped_gather_quantize_prefill_kernel(
    const float* __restrict__ hidden,            // [seq_len, d_model]
    int8_t*      __restrict__ A_int8_out,        // [total_slots, K] gathered + quantized
    float*       __restrict__ scales_out,        // [total_slots, blocks_per_row] per-block scales
    const int*   __restrict__ d_group_counts,    // [num_experts] tokens per expert
    const int*   __restrict__ d_group_offsets,   // [num_experts] prefix sum offsets
    const int*   __restrict__ d_group_token_indices, // [total_slots] original token indices
    int K)                                       // d_model (columns)
{
    const int block_in_row = blockIdx.x;   // which 32-element block in the row
    const int token_in_expert = blockIdx.y; // which token within this expert
    const int expert_id = blockIdx.z;       // which expert

    const int count = d_group_counts[expert_id];
    if (token_in_expert >= count) return;   // early-exit for over-provisioned M-tiles

    const int offset = d_group_offsets[expert_id];
    const int global_slot = offset + token_in_expert;
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
__device__ __forceinline__ void moe_prefill_decode_q4like_groups(
    const uint8_t* __restrict__ payload,
    int32_t* packed_groups)
{
    if constexpr (FMT == NVNNI_PF_Q4_0) {
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g] = static_cast<int32_t>((raw & 0x0F0F0F0Fu) - 0x08080808u);
            packed_groups[g + 4] = static_cast<int32_t>(((raw >> 4) & 0x0F0F0F0Fu) - 0x08080808u);
        }
    }
    else if constexpr (FMT == NVNNI_PF_IQ4_NL) {
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
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
    else if constexpr (FMT == NVNNI_PF_Q4_1) {
        // Asymmetric: q values are unsigned (0..15), no bias subtraction
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
            packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
        }
    }
    else if constexpr (FMT == NVNNI_PF_Q6_K) {
        // Dual-scale: lo 4 bits + hi 2 bits → 6-bit signed value - 32
        const uint32_t* raw4 = reinterpret_cast<const uint32_t*>(payload);
        const uint32_t* hi_raw = reinterpret_cast<const uint32_t*>(payload + 16);
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo_nibbles = raw4[g];
            const uint32_t lo4 = lo_nibbles & 0x0F0F0F0Fu;
            const uint32_t hi4 = (lo_nibbles >> 4) & 0x0F0F0F0Fu;
            // hi_raw encodes 2 bits per element (8 elements per uint32)
            const uint32_t hi_word = hi_raw[g / 2];
            const int shift_lo = (g % 2) * 16;
            const int shift_hi = shift_lo + 8;

            // Reconstruct 6-bit values for first 4 elements (lo nibbles)
            uint32_t val_lo = 0;
            for (int i = 0; i < 4; ++i) {
                int lo_val = (lo4 >> (i * 8)) & 0xF;
                int hi_val = (hi_word >> (shift_lo + i * 2)) & 0x3;
                int full = (hi_val << 4) | lo_val;
                val_lo |= static_cast<uint32_t>(static_cast<uint8_t>(full - 32)) << (i * 8);
            }
            packed_groups[g] = static_cast<int32_t>(val_lo);

            // Reconstruct for hi nibbles
            uint32_t val_hi = 0;
            for (int i = 0; i < 4; ++i) {
                int lo_val = (hi4 >> (i * 8)) & 0xF;
                int hi_val = (hi_word >> (shift_hi + i * 2)) & 0x3;
                int full = (hi_val << 4) | lo_val;
                val_hi |= static_cast<uint32_t>(static_cast<uint8_t>(full - 32)) << (i * 8);
            }
            packed_groups[g + 4] = static_cast<int32_t>(val_hi);
        }
    }
}

template <NativeVNNIFormatPrefill FMT>
__global__ __launch_bounds__(64)
void moe_grouped_gate_up_prefill_gemm_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, K] quantized activations
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, blocks_per_row]
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs, // [num_experts]
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,   // [num_experts]
    const int*     __restrict__ d_group_counts,      // [num_experts]
    const int*     __restrict__ d_group_offsets,     // [num_experts]
    float*         __restrict__ gate_output,         // [total_slots, N]
    float*         __restrict__ up_output,           // [total_slots, N]
    int N, int K)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_in_expert = blockIdx.y;
    const int expert_id = blockIdx.z;

    const int count = d_group_counts[expert_id];
    if (token_in_expert >= count || n >= N) return;

    const int offset = d_group_offsets[expert_id];
    const int global_slot = offset + token_in_expert;
    const int blocks_per_row = K / 32;

    // Get weight descriptors for this expert
    const DeviceNativeVNNIMatrixDesc& gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc& up_desc = up_descs[expert_id];

    const uint8_t* gate_payload_base = gate_desc.payload;
    const uint16_t* gate_scale_base = static_cast<const uint16_t*>(gate_desc.scales);
    const uint16_t* gate_min_base = static_cast<const uint16_t*>(gate_desc.mins);

    const uint8_t* up_payload_base = up_desc.payload;
    const uint16_t* up_scale_base = static_cast<const uint16_t*>(up_desc.scales);
    const uint16_t* up_min_base = static_cast<const uint16_t*>(up_desc.mins);

    // Input row for this token
    const int8_t* a_row = A_int8 + static_cast<size_t>(global_slot) * K;
    const float* a_scales = scales_A_blockwise + static_cast<size_t>(global_slot) * blocks_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(a_row + b * 32);
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
        const float scale_ab = a_scales[b];

        // Decode gate weights
        int32_t gate_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* gate_payload = gate_payload_base + linear * PB;
            moe_prefill_decode_q4like_groups<FMT>(gate_payload, gate_groups);
        }

        // Decode up weights
        int32_t up_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* up_payload = up_payload_base + linear * PB;
            moe_prefill_decode_q4like_groups<FMT>(up_payload, up_groups);
        }

        if constexpr (PrefillNVNNITraits<FMT>::is_dual_scale) {
            // Q6_K: dual-scale accumulation
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
            gate_acc += (static_cast<float>(gate_acc_lo) * gate_s_lo +
                         static_cast<float>(gate_acc_hi) * gate_s_hi) * scale_ab;

            const uint16_t up_scale_lo_bits = up_scale_base[linear];
            const uint16_t up_scale_hi_bits = up_min_base[linear];
            const float up_s_lo = __half2float(*reinterpret_cast<const __half*>(&up_scale_lo_bits));
            const float up_s_hi = __half2float(*reinterpret_cast<const __half*>(&up_scale_hi_bits));
            up_acc += (static_cast<float>(up_acc_lo) * up_s_lo +
                       static_cast<float>(up_acc_hi) * up_s_hi) * scale_ab;
        }
        else {
            // Symmetric / asymmetric: single-scale accumulation
            int32_t gate_dot = 0, up_dot = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                gate_dot = __builtin_amdgcn_sdot4(a4[g], gate_groups[g], gate_dot, false);
                up_dot = __builtin_amdgcn_sdot4(a4[g], up_groups[g], up_dot, false);
            }

            const uint16_t gate_scale_bits = gate_scale_base[linear];
            const float gate_d = __half2float(*reinterpret_cast<const __half*>(&gate_scale_bits));
            gate_acc += static_cast<float>(gate_dot) * gate_d * scale_ab;

            const uint16_t up_scale_bits = up_scale_base[linear];
            const float up_d = __half2float(*reinterpret_cast<const __half*>(&up_scale_bits));
            up_acc += static_cast<float>(up_dot) * up_d * scale_ab;

            if constexpr (PrefillNVNNITraits<FMT>::is_asymmetric) {
                // min correction: sum_a * block_min * scale_ab
                constexpr int32_t ones = 0x01010101;
                int32_t sum_a = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                const uint16_t gate_min_bits = gate_min_base[linear];
                const float gate_m = __half2float(*reinterpret_cast<const __half*>(&gate_min_bits));
                gate_acc += static_cast<float>(sum_a) * gate_m * scale_ab;

                const uint16_t up_min_bits = up_min_base[linear];
                const float up_m = __half2float(*reinterpret_cast<const __half*>(&up_min_bits));
                up_acc += static_cast<float>(sum_a) * up_m * scale_ab;
            }
        }
    }

    gate_output[static_cast<size_t>(global_slot) * N + n] = gate_acc;
    up_output[static_cast<size_t>(global_slot) * N + n] = up_acc;
}


// ============================================================================
// Kernel 3: Grouped SwiGLU + Quantize (all experts, single launch)
//
// Grid: dim3(blocks_per_intermediate_row, max_tokens_per_expert, num_experts)
//   blockIdx.x = quantization block within the intermediate dim
//   blockIdx.y = token within expert
//   blockIdx.z = expert index
// Block: 32 threads
// ============================================================================

__global__ __launch_bounds__(32)
void moe_grouped_swiglu_quantize_prefill_kernel(
    const float* __restrict__ gate_data,         // [total_slots, intermediate]
    const float* __restrict__ up_data,           // [total_slots, intermediate]
    int8_t*      __restrict__ A_int8_out,        // [total_slots, intermediate] quantized SwiGLU output
    float*       __restrict__ scales_out,        // [total_slots, blocks_per_intermediate]
    const int*   __restrict__ d_group_counts,    // [num_experts]
    const int*   __restrict__ d_group_offsets,   // [num_experts]
    int intermediate)                            // expert intermediate dimension
{
    const int block_in_row = blockIdx.x;
    const int token_in_expert = blockIdx.y;
    const int expert_id = blockIdx.z;

    const int count = d_group_counts[expert_id];
    if (token_in_expert >= count) return;

    const int offset = d_group_offsets[expert_id];
    const int global_slot = offset + token_in_expert;

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
//
// Grid: dim3((d_model + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts)
// Block: 64 threads
// ============================================================================

template <NativeVNNIFormatPrefill FMT>
__global__ __launch_bounds__(64)
void moe_grouped_down_prefill_gemm_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, intermediate] (from SwiGLU output)
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, blocks_per_intermediate]
    const DeviceNativeVNNIMatrixDesc* __restrict__ down_descs, // [num_experts]
    const int*     __restrict__ d_group_counts,      // [num_experts]
    const int*     __restrict__ d_group_offsets,     // [num_experts]
    float*         __restrict__ output,              // [total_slots, d_model]
    int N,   // d_model (output columns)
    int K)   // intermediate (input columns / reduction dim)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_in_expert = blockIdx.y;
    const int expert_id = blockIdx.z;

    const int count = d_group_counts[expert_id];
    if (token_in_expert >= count || n >= N) return;

    const int offset = d_group_offsets[expert_id];
    const int global_slot = offset + token_in_expert;
    const int blocks_per_row = K / 32;

    const DeviceNativeVNNIMatrixDesc& desc = down_descs[expert_id];
    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);

    const int8_t* a_row = A_int8 + static_cast<size_t>(global_slot) * K;
    const float* a_scales = scales_A_blockwise + static_cast<size_t>(global_slot) * blocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(a_row + b * 32);
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
        const float scale_ab = a_scales[b];

        int32_t packed_groups[8];
        {
            constexpr int PB = PrefillNVNNITraits<FMT>::payload_bytes;
            const uint8_t* payload = payload_base + linear * PB;
            moe_prefill_decode_q4like_groups<FMT>(payload, packed_groups);
        }

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
            acc += (static_cast<float>(acc_lo) * s_lo +
                    static_cast<float>(acc_hi) * s_hi) * scale_ab;
        }
        else {
            int32_t dot = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                dot = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], dot, false);

            const uint16_t scale_bits = scale_base[linear];
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            acc += static_cast<float>(dot) * block_d * scale_ab;

            if constexpr (PrefillNVNNITraits<FMT>::is_asymmetric) {
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

    output[static_cast<size_t>(global_slot) * N + n] = acc;
}


// ============================================================================
// Kernel 5: Grouped Weighted Scatter-Add (all experts, single launch)
//
// For each expert e, for each token assigned to it:
//   output[original_token, :] += weight * expert_result[slot, :]
//
// Grid: dim3((d_model + 63) / 64, max_tokens_per_expert, num_experts)
// Block: 64 threads
//
// Uses atomicAdd since multiple experts can contribute to the same output token.
// ============================================================================

__global__ __launch_bounds__(64)
void moe_grouped_weighted_scatter_prefill_kernel(
    float*       __restrict__ output,                // [seq_len, d_model]
    const float* __restrict__ expert_results,        // [total_slots, d_model]
    const int*   __restrict__ d_group_counts,        // [num_experts]
    const int*   __restrict__ d_group_offsets,       // [num_experts]
    const int*   __restrict__ d_group_token_indices, // [total_slots] original token index
    const float* __restrict__ d_group_weights,       // [total_slots] routing weights
    int d_model)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int token_in_expert = blockIdx.y;
    const int expert_id = blockIdx.z;

    const int count = d_group_counts[expert_id];
    if (token_in_expert >= count || n >= d_model) return;

    const int offset = d_group_offsets[expert_id];
    const int global_slot = offset + token_in_expert;
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
    // Format and device
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    if (num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
        max_tokens_per_expert <= 0 || total_slots <= 0)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError(); // clear prior errors

    const int blocks_per_d_model = d_model / 32;
    const int blocks_per_intermediate = intermediate / 32;

    // =========================================================================
    // Kernel 1: Gather + Quantize hidden → A_int8
    // =========================================================================
    {
        dim3 grid(blocks_per_d_model, max_tokens_per_expert, num_experts);
        dim3 block(32);
        hipLaunchKernelGGL(moe_grouped_gather_quantize_prefill_kernel,
            grid, block, 0, hip_stream,
            d_hidden, d_scratch_A_int8, d_scratch_scales,
            d_group_counts, d_group_offsets, d_group_token_indices, d_model);
    }

    // =========================================================================
    // Kernel 2: Gate+Up GEMM
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 grid((intermediate + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts);
        dim3 block(TILE_N);

        const auto* gate_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_gate_desc_table);
        const auto* up_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_up_desc_table);

        #define LAUNCH_GATEUP_PREFILL(FMT) \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(moe_grouped_gate_up_prefill_gemm_kernel<FMT>), \
                grid, block, 0, hip_stream, \
                d_scratch_A_int8, d_scratch_scales, gate_descs, up_descs, \
                d_group_counts, d_group_offsets, \
                d_scratch_gate, d_scratch_up, intermediate, d_model)

        switch (static_cast<NativeVNNIFormatPrefill>(codebook_id)) {
            case NVNNI_PF_Q4_0:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q4_0); break;
            case NVNNI_PF_IQ4_NL: LAUNCH_GATEUP_PREFILL(NVNNI_PF_IQ4_NL); break;
            case NVNNI_PF_Q4_1:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q4_1); break;
            case NVNNI_PF_Q5_0:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q5_0); break;
            case NVNNI_PF_Q5_1:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q5_1); break;
            case NVNNI_PF_Q6_K:   LAUNCH_GATEUP_PREFILL(NVNNI_PF_Q6_K); break;
            default:
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported codebook_id=%d for gate+up\n", codebook_id);
                return false;
        }
        #undef LAUNCH_GATEUP_PREFILL
    }

    // =========================================================================
    // Kernel 3: SwiGLU + Quantize
    // =========================================================================
    {
        dim3 grid(blocks_per_intermediate, max_tokens_per_expert, num_experts);
        dim3 block(32);
        hipLaunchKernelGGL(moe_grouped_swiglu_quantize_prefill_kernel,
            grid, block, 0, hip_stream,
            d_scratch_gate, d_scratch_up,
            d_scratch_swiglu_int8, d_scratch_swiglu_scales,
            d_group_counts, d_group_offsets, intermediate);
    }

    // =========================================================================
    // Kernel 4: Down GEMM
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 grid((d_model + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts);
        dim3 block(TILE_N);

        const auto* down_descs = static_cast<const DeviceNativeVNNIMatrixDesc*>(d_down_desc_table);

        #define LAUNCH_DOWN_PREFILL(FMT) \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(moe_grouped_down_prefill_gemm_kernel<FMT>), \
                grid, block, 0, hip_stream, \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                d_group_counts, d_group_offsets, \
                d_scratch_down_out, d_model, intermediate)

        switch (static_cast<NativeVNNIFormatPrefill>(codebook_id)) {
            case NVNNI_PF_Q4_0:   LAUNCH_DOWN_PREFILL(NVNNI_PF_Q4_0); break;
            case NVNNI_PF_IQ4_NL: LAUNCH_DOWN_PREFILL(NVNNI_PF_IQ4_NL); break;
            case NVNNI_PF_Q4_1:   LAUNCH_DOWN_PREFILL(NVNNI_PF_Q4_1); break;
            case NVNNI_PF_Q5_0:   LAUNCH_DOWN_PREFILL(NVNNI_PF_Q5_0); break;
            case NVNNI_PF_Q5_1:   LAUNCH_DOWN_PREFILL(NVNNI_PF_Q5_1); break;
            case NVNNI_PF_Q6_K:   LAUNCH_DOWN_PREFILL(NVNNI_PF_Q6_K); break;
            default:
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported codebook_id=%d for down\n", codebook_id);
                return false;
        }
        #undef LAUNCH_DOWN_PREFILL
    }

    // =========================================================================
    // Kernel 5: Weighted Scatter-Add
    // =========================================================================
    {
        constexpr int TILE_N = 64;
        dim3 grid((d_model + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts);
        dim3 block(TILE_N);
        hipLaunchKernelGGL(moe_grouped_weighted_scatter_prefill_kernel,
            grid, block, 0, hip_stream,
            d_output, d_scratch_down_out,
            d_group_counts, d_group_offsets, d_group_token_indices, d_group_weights,
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

} // extern "C"
