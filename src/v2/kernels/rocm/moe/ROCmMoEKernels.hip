/**
 * @file ROCmMoEKernels.hip
 * @brief HIP __global__ kernels for MoE operations
 *
 * Contains HIP kernels for device-native MoE execution:
 * 1. gate_logits_kernel    — GEMV: hidden × gate_weights^T → logits
 * 2. softmax_topk_kernel   — Row-wise softmax + top-k selection
 * 3. gather_tokens_kernel  — Gather token rows into contiguous batch
 * 4. scatter_add_kernel    — Weighted scatter-add expert outputs
 * 5. shared_expert_gate_kernel — sigmoid(dot) * scale per token
 * 6. swiglu_kernel         — silu(gate) * up activation
 *
 * Compiled by hipcc. Only includes HIP headers + HipDeviceGuard.
 * Linkage to host C++ is via extern "C" wrapper functions.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace
{
    constexpr int kDeviceMoEMaxExperts = 256;
    constexpr int kDeviceMoEMaxTopK = 16;
    constexpr int kGateLogitsExpertsPerBlock = 4;
    constexpr int kGateLogitsGroupedThreads = 256;

    __device__ inline float route_weight_to_float(__half value)
    {
        return __half2float(value);
    }

    __device__ inline float route_weight_to_float(hip_bfloat16 value)
    {
        return static_cast<float>(value);
    }

    struct DeviceNativeVNNIMatrixDescView
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

    struct DeviceMoEExpertDescriptorView
    {
        DeviceNativeVNNIMatrixDescView gate;
        DeviceNativeVNNIMatrixDescView up;
        DeviceNativeVNNIMatrixDescView down;
        int32_t logical_expert_id;
        int32_t owner_participant;
        int32_t local_slot;
        uint32_t flags;
    };

    struct DeviceMoEPlacementBankView
    {
        DeviceMoEExpertDescriptorView experts[kDeviceMoEMaxExperts];
        uint8_t local_compute_mask[kDeviceMoEMaxExperts];
        uint8_t replica_role[kDeviceMoEMaxExperts];
        uint32_t epoch;
        uint32_t expert_count;
        uint32_t reserved[2];
    };

    struct DeviceMoELayerRuntimeView
    {
        uint32_t active_bank;
        uint32_t active_epoch;
        uint32_t expert_count;
        uint32_t top_k;
        DeviceMoEPlacementBankView banks[2];
        int32_t topk_expert_ids[kDeviceMoEMaxTopK];
        float topk_weights[kDeviceMoEMaxTopK];
        uint64_t decode_histogram[kDeviceMoEMaxExperts];
        int32_t *route_expert_ids;
        float *route_weights;
        int32_t *expert_counts;
        int32_t *expert_offsets;
        int32_t *grouped_token_ids;
        float *grouped_route_weights;
        float *grouped_gate_scratch;
        float *grouped_up_scratch;
        float *grouped_output_partials;
        void *decode_scratch;
        void *reserved_ptrs[3];
        uint64_t reserved_u64[4];
        uint32_t prefill_token_capacity;
        uint32_t prefill_route_capacity;
        uint32_t reserved_u32[2];
    };

    constexpr int kMoEPointerStageMax = 16;

    struct MoEPointerStagePayload
    {
        const float *const_ptrs_a[kMoEPointerStageMax];
        const float *const_ptrs_b[kMoEPointerStageMax];
        float *mutable_ptrs_a[kMoEPointerStageMax];
        float *mutable_ptrs_b[kMoEPointerStageMax];
    };

    __global__ void moe_stage_const_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        const float **dst_a,
        const float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.const_ptrs_a[i];
        dst_b[i] = payload.const_ptrs_b[i];
    }

    __global__ void moe_stage_mutable_pointer_arrays_kernel(
        MoEPointerStagePayload payload,
        float **dst_a,
        float **dst_b,
        int count)
    {
        const int i = static_cast<int>(threadIdx.x);
        if (i >= count)
            return;
        dst_a[i] = payload.mutable_ptrs_a[i];
        dst_b[i] = payload.mutable_ptrs_b[i];
    }

    __device__ __forceinline__ bool moe_topk_pair_better(
        float candidate_value,
        int candidate_id,
        float best_value,
        int best_id)
    {
        return candidate_value > best_value ||
               (candidate_value == best_value && candidate_id < best_id);
    }

    __device__ void moe_select_topk_probabilities_block(
        float *__restrict__ expert_probs,
        int num_experts,
        int top_k,
        int *__restrict__ selected_ids,
        float *__restrict__ selected_weights)
    {
        __shared__ float warp_best_values[8];
        __shared__ int warp_best_ids[8];

        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = -1.0f;
            int best_id = kDeviceMoEMaxExperts;

            if (threadIdx.x < num_experts)
            {
                best_value = expert_probs[threadIdx.x];
                best_id = threadIdx.x;
            }

            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            {
                const float other_value = __shfl_down(best_value, offset);
                const int other_id = __shfl_down(best_id, offset);
                if (moe_topk_pair_better(other_value, other_id, best_value, best_id))
                {
                    best_value = other_value;
                    best_id = other_id;
                }
            }

            if (lane == 0)
            {
                warp_best_values[warp_id] = best_value;
                warp_best_ids[warp_id] = best_id;
            }
            __syncthreads();

            if (warp_id == 0)
            {
                best_value = (lane < num_warps) ? warp_best_values[lane] : -1.0f;
                best_id = (lane < num_warps) ? warp_best_ids[lane] : kDeviceMoEMaxExperts;
                for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                {
                    const float other_value = __shfl_down(best_value, offset);
                    const int other_id = __shfl_down(best_id, offset);
                    if (moe_topk_pair_better(other_value, other_id, best_value, best_id))
                    {
                        best_value = other_value;
                        best_id = other_id;
                    }
                }

                if (lane == 0)
                {
                    selected_ids[slot] = best_id;
                    selected_weights[slot] = best_value;
                    if (best_id >= 0 && best_id < num_experts)
                    {
                        expert_probs[best_id] = -1.0f;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// =========================================================================
// Kernel 1: Single-token Gate Logits
//
// Decode routes one token at a time. hipBLAS is efficient for prefill GEMM but
// pays too much fixed overhead for a 1 x d_model by num_experts router GEMV.
// This path computes one expert logit per block and feeds the existing
// softmax/top-k kernel.
// =========================================================================

__global__ void rocm_moe_gate_logits_single_token_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 3) == 0)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_small_m_kernel(
    const float *__restrict__ hidden,       // [seq_len, d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [seq_len, num_experts]
    int seq_len, int d_model, int num_experts,
    bool use_vec4)
{
    extern __shared__ float4 hidden_shared_vec4[];
    float *hidden_shared = reinterpret_cast<float *>(hidden_shared_vec4);

    const int row = blockIdx.y;
    if (row >= seq_len) return;

    const float *h = hidden + static_cast<size_t>(row) * d_model;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(h);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            hidden_shared_vec4[i] = hidden4[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
            hidden_shared[i] = h[i];
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const float *w = gate_weights + static_cast<size_t>(expert) * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden_shared);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = lane; i < vec_count; i += warpSize)
        {
            const float4 hh = hidden4[i];
            const float4 ww = w4[i];
            sum += hh.x * ww.x + hh.y * ww.y + hh.z * ww.z + hh.w * ww.w;
        }
    }
    else
    {
        for (int i = lane; i < d_model; i += warpSize)
            sum += hidden_shared[i] * w[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (lane == 0)
        logits[static_cast<size_t>(row) * num_experts + expert] = sum;
}

template <typename GateWeightT>
__global__ void rocm_moe_gate_logits_small_m_mixed_weights_kernel(
    const float *__restrict__ hidden,              // [seq_len, d_model]
    const GateWeightT *__restrict__ gate_weights,  // [num_experts, d_model]
    float *__restrict__ logits,                    // [seq_len, num_experts]
    int seq_len, int d_model, int num_experts)
{
    extern __shared__ float hidden_shared[];

    const int row = blockIdx.y;
    if (row >= seq_len) return;

    const float *h = hidden + static_cast<size_t>(row) * d_model;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        hidden_shared[i] = h[i];
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const GateWeightT *w =
        gate_weights + static_cast<size_t>(expert) * d_model;
    float sum = 0.0f;
    for (int i = lane; i < d_model; i += warpSize)
        sum += hidden_shared[i] * route_weight_to_float(w[i]);

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (lane == 0)
        logits[static_cast<size_t>(row) * num_experts + expert] = sum;
}

__global__ void rocm_moe_gate_logits_single_token_kpart_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ partials,           // [num_experts, k_partitions]
    int d_model, int num_experts, int k_partitions,
    bool use_vec4)
{
    const int expert = blockIdx.x;
    const int partition = blockIdx.y;
    if (expert >= num_experts || partition >= k_partitions) return;

    const int block_count = (d_model + 31) / 32;
    const int blocks_per_partition = (block_count + k_partitions - 1) / k_partitions;
    const int block_begin = partition * blocks_per_partition;
    const int block_end_unclamped = block_begin + blocks_per_partition;
    const int block_end = (block_end_unclamped < block_count) ? block_end_unclamped : block_count;
    const int element_begin = block_begin * 32;
    const int element_end_unclamped = block_end * 32;
    const int element_end = (element_end_unclamped < d_model) ? element_end_unclamped : d_model;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_begin = element_begin >> 2;
        const int vec_end = element_end >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = vec_begin + threadIdx.x; i < vec_end; i += blockDim.x)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = element_begin + threadIdx.x; i < element_end; i += blockDim.x)
        {
            sum += hidden[i] * w[i];
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) partials[expert * k_partitions + partition] = sum;
    }
}

__global__ void rocm_moe_gate_logits_kpart_reduce_kernel(
    const float *__restrict__ partials, // [num_experts, k_partitions]
    float *__restrict__ logits,         // [num_experts]
    int num_experts, int k_partitions)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const float *expert_partials = partials + expert * k_partitions;
    float sum = 0.0f;
    for (int partition = threadIdx.x; partition < k_partitions; partition += blockDim.x)
    {
        sum += expert_partials[partition];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_fp32_to_fp16_kernel(
    const float *__restrict__ input,
    __half *__restrict__ output,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    output[idx] = __float2half(input[idx]);
}

__global__ void rocm_moe_router_gate_quantize_q8_kernel(
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    int8_t *__restrict__ gate_weights_q8,   // [num_experts, blocks_per_row, 32]
    float *__restrict__ gate_scales,        // [num_experts, blocks_per_row]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    const int block_idx = blockIdx.y;
    const int lane = threadIdx.x;
    if (expert >= num_experts || lane >= BLOCK_SIZE) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    const int k = block_idx * BLOCK_SIZE + lane;
    const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
    const float value = (k < d_model) ? gate_weights[static_cast<size_t>(expert) * d_model + k] : 0.0f;

    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    if (lane == 0)
    {
        gate_scales[scale_idx] = scale;
    }

    const float q_value = value / scale;
    gate_weights_q8[scale_idx * BLOCK_SIZE + lane] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, q_value))));
}

__global__ void rocm_moe_router_hidden_quantize_q8_kernel(
    const float *__restrict__ hidden,       // [d_model]
    int8_t *__restrict__ hidden_q8,         // [d_model]
    float *__restrict__ hidden_scales,      // [blocks_per_row]
    int d_model)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (lane >= BLOCK_SIZE || k >= d_model) return;

    const float value = hidden[k];
    float max_abs = fabsf(value);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask));
    }

    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    if (lane == 0)
    {
        hidden_scales[block_idx] = scale;
    }

    const float q_value = value / scale;
    hidden_q8[k] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, q_value))));
}

__global__ void rocm_moe_gate_logits_single_token_q8_weights_kernel(
    const int8_t *__restrict__ hidden_q8,       // [d_model]
    const float *__restrict__ hidden_scales,    // [blocks_per_row]
    const int8_t *__restrict__ gate_weights_q8, // [num_experts, blocks_per_row, 32]
    const float *__restrict__ gate_scales,      // [num_experts, blocks_per_row]
    float *__restrict__ logits,                 // [num_experts]
    int d_model, int num_experts)
{
    constexpr int BLOCK_SIZE = 32;
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const int blocks_per_row = d_model / BLOCK_SIZE;
    float sum = 0.0f;
    for (int block_idx = threadIdx.x; block_idx < blocks_per_row; block_idx += blockDim.x)
    {
        const int32_t *h4 = reinterpret_cast<const int32_t *>(hidden_q8 + block_idx * BLOCK_SIZE);
        const size_t scale_idx = static_cast<size_t>(expert) * blocks_per_row + block_idx;
        const int32_t *w4 = reinterpret_cast<const int32_t *>(gate_weights_q8 + scale_idx * BLOCK_SIZE);

        int32_t block_acc = 0;
        #pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            block_acc = __builtin_amdgcn_sdot4(h4[group], w4[group], block_acc, false);
        }
        sum += static_cast<float>(block_acc) * hidden_scales[block_idx] * gate_scales[scale_idx];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_single_token_fp16_weights_kernel(
    const float *__restrict__ hidden,          // [d_model]
    const __half *__restrict__ gate_weights,   // [num_experts, d_model]
    float *__restrict__ logits,                // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const __half *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if ((d_model & 1) == 0)
    {
        const int pair_count = d_model >> 1;
        const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
        for (int i = threadIdx.x; i < pair_count; i += blockDim.x)
        {
            const float2 ww = __half22float2(w2[i]);
            const int h = i << 1;
            sum += hidden[h] * ww.x + hidden[h + 1] * ww.y;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            sum += hidden[i] * __half2float(w[i]);
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) logits[expert] = sum;
    }
}

__global__ void rocm_moe_gate_logits_single_token_bf16_weights_kernel(
    const float *__restrict__ hidden,              // [d_model]
    const hip_bfloat16 *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,                    // [num_experts]
    int d_model, int num_experts)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;

    const hip_bfloat16 *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += hidden[i] * static_cast<float>(w[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0) logits[expert] = sum;
    }
}

// =========================================================================
// Kernel 1b: Single-token Gate Logits, Grouped Experts
//
// Each CTA handles a contiguous group of experts and loads the decode hidden
// vector into LDS once. On gfx906 this maps four wave64 waves to four experts,
// avoiding cross-wave reductions while preserving the original logits layout.
// =========================================================================

__global__ void rocm_moe_gate_logits_single_token_grouped_kernel(
    const float *__restrict__ hidden,       // [d_model]
    const float *__restrict__ gate_weights, // [num_experts, d_model]
    float *__restrict__ logits,             // [num_experts]
    int d_model, int num_experts,
    bool use_vec4)
{
    extern __shared__ float4 hidden_shared_vec4[];
    float *hidden_shared = reinterpret_cast<float *>(hidden_shared_vec4);

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
        for (int i = threadIdx.x; i < vec_count; i += blockDim.x)
        {
            hidden_shared_vec4[i] = hidden4[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        {
            hidden_shared[i] = hidden[i];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (warp_id >= kGateLogitsExpertsPerBlock)
        return;

    const int expert = blockIdx.x * kGateLogitsExpertsPerBlock + warp_id;
    if (expert >= num_experts)
        return;

    const float *w = gate_weights + expert * d_model;
    float sum = 0.0f;

    if (use_vec4)
    {
        const int vec_count = d_model >> 2;
        const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden_shared);
        const float4 *w4 = reinterpret_cast<const float4 *>(w);
        for (int i = lane; i < vec_count; i += warpSize)
        {
            const float4 h = hidden4[i];
            const float4 ww = w4[i];
            sum += h.x * ww.x + h.y * ww.y + h.z * ww.z + h.w * ww.w;
        }
    }
    else
    {
        for (int i = lane; i < d_model; i += warpSize)
        {
            sum += hidden_shared[i] * w[i];
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (lane == 0)
        logits[expert] = sum;
}

// =========================================================================
// Kernel 2: Row-wise Softmax + Top-K Selection
//
// For each token:
//   1. Find max logit (numerical stability)
//   2. Compute exp(logit - max) and sum
//   3. Normalize to probabilities
//   4. Select top-k experts by probability
//   5. Optionally renormalize top-k weights
//
// One block per token. Uses shared memory for the logits array.
// =========================================================================

__global__ void rocm_moe_softmax_topk_kernel(
    float *__restrict__ logits,             // [seq_len, num_experts] IN/OUT (probabilities)
    int *__restrict__ expert_indices,       // [seq_len * top_k] OUT
    float *__restrict__ expert_weights,     // [seq_len * top_k] OUT
    int seq_len, int num_experts, int top_k,
    bool normalize_weights)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    float *row = logits + token * num_experts;
    __shared__ float values[kDeviceMoEMaxExperts];
    __shared__ float reductions[256];
    __shared__ int reduction_indices[256];
    __shared__ int selected_indices[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];

    // Step 1: Find max (thread-parallel)
    float local_max = -FLT_MAX;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        local_max = fmaxf(local_max, row[e]);
    }
    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    __shared__ float shared_max[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < nw) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down(local_max, offset));
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float max_val = shared_max[0];

    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        float val = expf(row[e] - max_val);
        row[e] = val;
        local_sum += val;
    }
    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down(local_sum, offset);

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < nw) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            local_sum += __shfl_down(local_sum, offset);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    float sum_val = shared_sum[0];
    float inv_sum = (sum_val > 0.0f) ? 1.0f / sum_val : 0.0f;

    // Step 3: Normalize probabilities
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        const float prob = row[e] * inv_sum;
        row[e] = prob;
        values[e] = prob;
    }
    __syncthreads();

    /**
     * Select top-k with a block-wide argmax reduction.
     *
     * The old ROCm path did the whole top_k * num_experts scan on lane 0 after
     * softmax. MTP verifier rows execute this kernel for every MoE layer, so
     * leaving 255 lanes idle made routing dominate ROCm verifier time. This
     * mirrors the CUDA contract: each round finds one winner in parallel, masks
     * it from the shared-memory copy only, and leaves row[] intact for callers
     * that inspect router probabilities. Ties intentionally choose the lowest
     * expert id, matching the previous ascending serial scan.
     */
    float topk_sum = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        float best_value = -1.0f;
        int best_idx = 0;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            const float value = values[e];
            if (value > best_value)
            {
                best_value = value;
                best_idx = e;
            }
        }
        reductions[threadIdx.x] = best_value;
        reduction_indices[threadIdx.x] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                const float other = reductions[threadIdx.x + stride];
                const int other_idx = reduction_indices[threadIdx.x + stride];
                const float current = reductions[threadIdx.x];
                const int current_idx = reduction_indices[threadIdx.x];
                if (other > current || (other == current && other_idx < current_idx))
                {
                    reductions[threadIdx.x] = other;
                    reduction_indices[threadIdx.x] = other_idx;
                }
            }
            __syncthreads();
        }

        const int winner = reduction_indices[0];
        const float winner_value = reductions[0];
        topk_sum += winner_value;
        if (threadIdx.x == 0)
        {
            selected_indices[k] = winner;
            selected_weights[k] = winner_value;
            values[winner] = -1.0f;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        int *out_idx = expert_indices + token * top_k;
        float *out_wt = expert_weights + token * top_k;
        const float inv_topk_sum = (normalize_weights && topk_sum > 0.0f) ? 1.0f / topk_sum : 1.0f;
        for (int k = 0; k < top_k; ++k)
        {
            out_idx[k] = selected_indices[k];
            out_wt[k] = selected_weights[k] * inv_topk_sum;
        }
    }
}

// =========================================================================
// Kernel 2b: Decode Softmax + Top-K + Runtime Write
//
// Single-token decode variant that keeps routing entirely device-side: it
// converts logits to probabilities, selects top-k, writes the stable runtime
// table, optionally mirrors legacy FP32 routing tensors, and updates the
// decode histogram in one launch.
// =========================================================================

__global__ void rocm_moe_softmax_topk_decode_runtime_kernel(
    float *__restrict__ logits,                       // [num_experts] IN/OUT (probabilities)
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        local_max = fmaxf(local_max, logits[expert]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    __shared__ float shared_max[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            local_max = fmaxf(local_max, __shfl_down(local_max, offset));
        }
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        logits[expert] = value;
        local_sum += value;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        local_sum += __shfl_down(local_sum, offset);
    }

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            local_sum += __shfl_down(local_sum, offset);
        }
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        logits[expert] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int selected_ids[kDeviceMoEMaxTopK];
        float selected_weights[kDeviceMoEMaxTopK];

        for (int slot = 0; slot < top_k; ++slot)
        {
            selected_ids[slot] = -1;
            selected_weights[slot] = 0.0f;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            float best_value = -1.0f;
            int best_expert = 0;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (logits[expert] > best_value)
                {
                    bool already_selected = false;
                    for (int previous = 0; previous < slot; ++previous)
                    {
                        if (selected_ids[previous] == expert)
                        {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected)
                    {
                        best_value = logits[expert];
                        best_expert = expert;
                    }
                }
            }
            selected_ids[slot] = best_expert;
            selected_weights[slot] = best_value;
        }

        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        const bool runtime_shape_ok =
            runtime->active_bank <= 1u &&
            runtime->active_epoch != 0u &&
            runtime->expert_count == static_cast<uint32_t>(num_experts) &&
            runtime->top_k == static_cast<uint32_t>(top_k);

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_ids[slot];
            const float weight = selected_weights[slot];

            if (runtime_shape_ok)
            {
                runtime->topk_expert_ids[slot] = expert_id;
                runtime->topk_weights[slot] = weight;
                if (update_runtime_histogram && expert_id >= 0 && expert_id < num_experts)
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert_id]), 1ULL);
                }
            }

            if (write_legacy_outputs && legacy_indices && legacy_weights)
            {
                legacy_indices[slot] = static_cast<float>(expert_id);
                legacy_weights[slot] = weight;
            }
        }
    }
}

__global__ void rocm_moe_softmax_topk_decode_runtime_wave64_kernel(
    const float *__restrict__ logits,                 // [num_experts] input logits, left unmodified
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,               // optional [top_k]
    float *__restrict__ legacy_weights,               // optional [top_k]
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram)
{
    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    __shared__ float expert_probs[kDeviceMoEMaxExperts];
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    float max_val = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        max_val = fmaxf(max_val, logits[expert]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
    }
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        }
        if (lane == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    float sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(logits[expert] - max_val);
        expert_probs[expert] = value;
        sum += value;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            sum += __shfl_down(sum, offset);
        }
        if (lane == 0) shared_sum[0] = sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        expert_probs[expert] *= inv_sum;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        const bool runtime_shape_ok =
            runtime->active_bank <= 1u &&
            runtime->active_epoch != 0u &&
            runtime->expert_count == static_cast<uint32_t>(num_experts) &&
            runtime->top_k == static_cast<uint32_t>(top_k);

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_ids[slot];
            const float weight = selected_weights[slot];

            if (runtime_shape_ok)
            {
                runtime->topk_expert_ids[slot] = expert_id;
                runtime->topk_weights[slot] = weight;
                if (update_runtime_histogram && expert_id >= 0 && expert_id < num_experts)
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert_id]), 1ULL);
                }
            }

            if (write_legacy_outputs && legacy_indices && legacy_weights)
            {
                legacy_indices[slot] = static_cast<float>(expert_id);
                legacy_weights[slot] = weight;
            }
        }
    }
}

__global__ void rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel(
    const float *__restrict__ partials,                // [num_experts, k_partitions]
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,                // optional [top_k]
    float *__restrict__ legacy_weights,                // optional [top_k]
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram)
{
    if (!partials || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return;
    }

    __shared__ float expert_probs[kDeviceMoEMaxExperts];

    float local_max = -FLT_MAX;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float *expert_partials = partials + expert * k_partitions;
        float sum = 0.0f;
        for (int partition = 0; partition < k_partitions; ++partition)
        {
            sum += expert_partials[partition];
        }
        expert_probs[expert] = sum;
        local_max = fmaxf(local_max, sum);
    }

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    __shared__ float shared_max[8];
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_max = (lane < num_warps) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            local_max = fmaxf(local_max, __shfl_down(local_max, offset));
        }
        if (lane == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    const float max_val = shared_max[0];

    float local_sum = 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        const float value = expf(expert_probs[expert] - max_val);
        expert_probs[expert] = value;
        local_sum += value;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        local_sum += __shfl_down(local_sum, offset);
    }

    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            local_sum += __shfl_down(local_sum, offset);
        }
        if (lane == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    const float inv_sum = (shared_sum[0] > 0.0f) ? 1.0f / shared_sum[0] : 0.0f;
    for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x)
    {
        expert_probs[expert] *= inv_sum;
    }
    __syncthreads();

    __shared__ int selected_ids[kDeviceMoEMaxTopK];
    __shared__ float selected_weights[kDeviceMoEMaxTopK];
    moe_select_topk_probabilities_block(expert_probs, num_experts, top_k, selected_ids, selected_weights);

    if (threadIdx.x == 0)
    {
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int slot = 0; slot < top_k; ++slot)
            {
                topk_sum += selected_weights[slot];
            }
            if (topk_sum > 0.0f)
            {
                const float inv_topk_sum = 1.0f / topk_sum;
                for (int slot = 0; slot < top_k; ++slot)
                {
                    selected_weights[slot] *= inv_topk_sum;
                }
            }
        }

        const bool runtime_shape_ok =
            runtime->active_bank <= 1u &&
            runtime->active_epoch != 0u &&
            runtime->expert_count == static_cast<uint32_t>(num_experts) &&
            runtime->top_k == static_cast<uint32_t>(top_k);

        for (int slot = 0; slot < top_k; ++slot)
        {
            const int expert_id = selected_ids[slot];
            const float weight = selected_weights[slot];

            if (runtime_shape_ok)
            {
                runtime->topk_expert_ids[slot] = expert_id;
                runtime->topk_weights[slot] = weight;
                if (update_runtime_histogram && expert_id >= 0 && expert_id < num_experts)
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert_id]), 1ULL);
                }
            }

            if (write_legacy_outputs && legacy_indices && legacy_weights)
            {
                legacy_indices[slot] = static_cast<float>(expert_id);
                legacy_weights[slot] = weight;
            }
        }
    }
}

// =========================================================================
// Kernel 2c: Decode Route Select Runtime Copy
//
// Copies single-token top-k ids/weights into the stable device runtime table,
// optionally mirrors them to legacy routing tensors, and records a device-side
// decode histogram. The routing tensors remain necessary until the expert
// decode kernels consume runtime top-k directly.
// =========================================================================

__global__ void rocm_moe_decode_route_select_runtime_kernel(
    const int *__restrict__ expert_indices,       // [top_k]
    const float *__restrict__ expert_weights,     // [top_k]
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    float *__restrict__ legacy_indices,           // optional [top_k]
    float *__restrict__ legacy_weights,           // optional [top_k]
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram)
{
    const int k = threadIdx.x;
    if (k >= top_k || k >= kDeviceMoEMaxTopK || !runtime) return;

    const int expert_id = expert_indices[k];
    const float weight = expert_weights[k];

    const bool runtime_shape_ok =
        runtime->active_bank <= 1u &&
        runtime->active_epoch != 0u &&
        runtime->expert_count == static_cast<uint32_t>(num_experts) &&
        runtime->top_k == static_cast<uint32_t>(top_k);

    if (runtime_shape_ok)
    {
        runtime->topk_expert_ids[k] = expert_id;
        runtime->topk_weights[k] = weight;
        if (update_runtime_histogram && expert_id >= 0 && expert_id < num_experts)
        {
            atomicAdd(reinterpret_cast<unsigned long long *>(&runtime->decode_histogram[expert_id]), 1ULL);
        }
    }

    if (write_legacy_outputs)
    {
        legacy_indices[k] = static_cast<float>(expert_id);
        legacy_weights[k] = weight;
    }
}

// =========================================================================
// Kernel 3: Gather Tokens
//
// Copy selected token rows into a contiguous batch buffer.
// Each block handles one token copy; threads cover the d_model dimension.
// =========================================================================

__global__ void rocm_moe_gather_tokens_kernel(
    const float *__restrict__ hidden,        // [seq_len, d_model]
    float *__restrict__ batch_buffer,        // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int src_row = token_indices[token];
    const float *src = hidden + src_row * d_model;
    float *dst = batch_buffer + token * d_model;

    // Vectorized copy: try float4 first
    const int d_model4 = d_model >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);

    for (int i = threadIdx.x; i < d_model4; i += blockDim.x)
    {
        dst4[i] = src4[i];
    }

    // Handle tail elements
    const int tail_start = d_model4 << 2;
    for (int i = tail_start + threadIdx.x; i < d_model; i += blockDim.x)
    {
        dst[i] = src[i];
    }
}

/**
 * @brief Copy one row using a by-value row index.
 *
 * Verifier replay uses this instead of the general gather kernel so graph and
 * stream replay do not depend on an async H2D copy from a stack-owned host row
 * index.  The row width is normally a multiple of four, but the scalar tail
 * keeps the helper valid for routing rows such as top_k=8 as well.
 */
__global__ void rocm_moe_copy_row_kernel(
    const float *__restrict__ source,
    float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    const float *src = source + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float4 *dst4 = reinterpret_cast<float4 *>(row_buffer);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        row_buffer[i] = src[i];
}

// =========================================================================
// Kernel 4: Scatter-Add Weighted
//
// output[token_indices[i]] += weights[i] * expert_output[i]
// Each block handles one token; threads cover d_model.
// =========================================================================

__global__ void rocm_moe_scatter_add_kernel(
    float *__restrict__ output,             // [seq_len, d_model]
    const float *__restrict__ expert_output, // [num_tokens, d_model]
    const int *__restrict__ token_indices,   // [num_tokens]
    const float *__restrict__ weights,       // [num_tokens]
    int num_tokens, int d_model)
{
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int dst_row = token_indices[token];
    const float w = weights[token];
    const float *src = expert_output + token * d_model;
    float *dst = output + dst_row * d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        atomicAdd(&dst[i], w * src[i]);
    }
}

/**
 * @brief Store one scratch row into a destination row selected by value.
 *
 * The operation is an overwrite, not an add.  Decode-equivalent verifier replay
 * owns each destination row exactly once, so direct store avoids atomics and
 * avoids staging a `{row, weight}` pair through host memory.
 */
__global__ void rocm_moe_write_row_kernel(
    float *__restrict__ destination,
    const float *__restrict__ row_buffer,
    int row_index,
    int row_width)
{
    if (row_index < 0 || row_width <= 0)
        return;

    float *dst = destination + static_cast<size_t>(row_index) * row_width;
    const int row_width4 = row_width >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(row_buffer);
    float4 *dst4 = reinterpret_cast<float4 *>(dst);
    for (int i = threadIdx.x; i < row_width4; i += blockDim.x)
        dst4[i] = src4[i];

    const int tail_start = row_width4 << 2;
    for (int i = tail_start + threadIdx.x; i < row_width; i += blockDim.x)
        dst[i] = row_buffer[i];
}

// =========================================================================
// Kernel 5: Shared Expert Gate
//
// For each token:
//   gate = sigmoid(dot(gate_inp, input[t]))
//   shared_output[t] *= gate
//
// Two-phase: first compute dot products (reduction), then scale.
// =========================================================================

__global__ void rocm_moe_shared_expert_gate_dot_kernel(
    const float *__restrict__ input,        // [seq_len, d_model]
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ gate_values,        // [seq_len] scratch
    int seq_len, int d_model)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    const float *x = input + token * d_model;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += x[i] * gate_inp[i];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
        if (lane == 0)
        {
            // Apply sigmoid
            gate_values[token] = 1.0f / (1.0f + expf(-sum));
        }
    }
}

__global__ void rocm_moe_shared_expert_gate_scale_kernel(
    float *__restrict__ shared_output,      // [seq_len, d_model]
    const float *__restrict__ gate_values,  // [seq_len]
    int seq_len, int d_model)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * d_model;
    if (idx >= total) return;

    const int token = idx / d_model;
    shared_output[idx] *= gate_values[token];
}

__global__ void rocm_moe_shared_expert_gate_decode_fused_kernel(
    const float *__restrict__ input,        // [d_model], token 0
    const float *__restrict__ gate_inp,     // [d_model]
    float *__restrict__ shared_output,      // [d_model], token 0
    int d_model)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += input[i] * gate_inp[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            block_sum += __shfl_down(block_sum, offset);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    const float gate = warp_sums[0];
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        shared_output[i] *= gate;
    }
}

__global__ void rocm_moe_shared_expert_gate_add_kernel(
    const float *__restrict__ input,            // [seq_len, d_model]
    const float *__restrict__ gate_inp,         // [d_model]
    float *__restrict__ shared_output,          // [seq_len, d_model]
    const float *__restrict__ routed_residual,  // [seq_len, d_model]
    float *__restrict__ combined_output,        // [seq_len, d_model]
    int seq_len, int d_model)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    const size_t row_offset = static_cast<size_t>(token) * d_model;
    const float *x = input + row_offset;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        sum += x[i] * gate_inp[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float warp_sums[16];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0)
    {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            block_sum += __shfl_down(block_sum, offset);
        if (lane == 0)
            warp_sums[0] = 1.0f / (1.0f + expf(-block_sum));
    }
    __syncthreads();

    const float gate = warp_sums[0];
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        const float gated_shared = gate * shared_output[row_offset + i];
        shared_output[row_offset + i] = gated_shared;
        combined_output[row_offset + i] = routed_residual[row_offset + i] + gated_shared;
    }
}

// =========================================================================
// Kernel 6: SwiGLU — silu(gate) * up
//
// gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// Vectorized with float4.
// =========================================================================

__global__ void rocm_moe_swiglu_kernel(
    float *__restrict__ gate,
    const float *__restrict__ up,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    gate[idx] = silu * up[idx];
}

// =========================================================================
// Kernel 6b: Weighted Add — output += weight * input
//
// Simple element-wise weighted accumulation for expert output aggregation.
// =========================================================================

__global__ void rocm_moe_weighted_add_kernel(
    float *__restrict__ output,
    const float *__restrict__ input,
    float weight,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    output[idx] += weight * input[idx];
}

/**
 * @brief Launch the small-M router for FP16/BF16 gate weights.
 *
 * Request-batched MTP verification produces M=2..4 router rows. Those rows need
 * the same dtype support as decode's single-token router, but the C bridge must
 * stay non-templated so host code can link it predictably.
 */
template <typename GateWeightT>
bool launch_gate_logits_small_m_mixed_weights(
    const float *hidden, const void *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream,
    const char *label)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !hidden || !gate_weights || !logits ||
        seq_len < 2 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE %s invalid arguments\n", label);
        return false;
    }

    constexpr size_t kSmallMGroupedSharedCapBytes = 48 * 1024;
    const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes > kSmallMGroupedSharedCapBytes)
    {
        fprintf(stderr, "HIP MoE %s shared memory too large: requested=%zu cap=%zu\n",
                label, shared_mem_bytes, kSmallMGroupedSharedCapBytes);
        return false;
    }

    const int blocks =
        (num_experts + kGateLogitsExpertsPerBlock - 1) /
        kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_small_m_mixed_weights_kernel<GateWeightT>,
        dim3(blocks, seq_len), dim3(kGateLogitsGroupedThreads), shared_mem_bytes,
        static_cast<hipStream_t>(stream),
        hidden, static_cast<const GateWeightT *>(gate_weights), logits,
        seq_len, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE %s kernel launch failed: %s\n",
                label, hipGetErrorString(err));
        return false;
    }
    return true;
}

// =========================================================================
// Extern "C" bridge functions — called from ROCmMoEKernel.cpp
// =========================================================================

extern "C"
{

bool hipMoE_gate_logits_single_token(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_small_m(
    const float *hidden, const float *gate_weights, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !hidden || !gate_weights || !logits ||
        seq_len < 2 || seq_len > 4 || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m invalid arguments\n");
        return false;
    }

    constexpr size_t kSmallMGroupedSharedCapBytes = 48 * 1024;
    const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes > kSmallMGroupedSharedCapBytes)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m shared memory too large: requested=%zu cap=%zu\n",
                shared_mem_bytes, kSmallMGroupedSharedCapBytes);
        return false;
    }

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int blocks = (num_experts + kGateLogitsExpertsPerBlock - 1) / kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_small_m_kernel,
        dim3(blocks, seq_len), dim3(kGateLogitsGroupedThreads), shared_mem_bytes, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, seq_len, d_model, num_experts, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_small_m kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_small_m_fp16_weights(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    return launch_gate_logits_small_m_mixed_weights<__half>(
        hidden, gate_weights_fp16, logits, seq_len, d_model, num_experts,
        device_idx, stream, "gate_logits_small_m_fp16_weights");
}

bool hipMoE_gate_logits_small_m_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int seq_len, int d_model, int num_experts,
    int device_idx, void *stream)
{
    return launch_gate_logits_small_m_mixed_weights<hip_bfloat16>(
        hidden, gate_weights_bf16, logits, seq_len, d_model, num_experts,
        device_idx, stream, "gate_logits_small_m_bf16_weights");
}

bool hipMoE_gate_logits_single_token_kpart(
    const float *hidden, const float *gate_weights, float *logits,
    float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int reduce_threads = 64;
    hipLaunchKernelGGL(rocm_moe_gate_logits_kpart_reduce_kernel,
        dim3(num_experts), dim3(reduce_threads), 0, static_cast<hipStream_t>(stream),
        partials, logits, num_experts, k_partitions);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_kpart_reduce kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_kpart_partials(
    const float *hidden, const float *gate_weights, float *partials,
    int d_model, int num_experts, int k_partitions,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !partials ||
        d_model <= 0 || num_experts <= 0 || k_partitions <= 0)
    {
        fprintf(stderr, "HIP MoE K-part gate logits partials invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int partial_threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_kpart_kernel,
        dim3(num_experts, k_partitions), dim3(partial_threads), 0, static_cast<hipStream_t>(stream),
        hidden, gate_weights, partials, d_model, num_experts, k_partitions, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_kpart_partials kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_fp32_to_fp16(
    const float *input, void *output_fp16, int count,
    int device_idx, void *stream)
{
    if (!input || !output_fp16 || count < 0)
    {
        fprintf(stderr, "HIP MoE fp32_to_fp16 invalid args\n");
        return false;
    }
    if (count == 0)
        return true;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_fp32_to_fp16_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, static_cast<__half *>(output_fp16), count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE fp32_to_fp16 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_fp16_weights(
    const float *hidden, const void *gate_weights_fp16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_fp16 || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE fp16 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_fp16_weights_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const __half *>(gate_weights_fp16), logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_fp16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_bf16_weights(
    const float *hidden, const void *gate_weights_bf16, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights_bf16 || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE bf16 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_bf16_weights_kernel,
        dim3(num_experts), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, static_cast<const hip_bfloat16 *>(gate_weights_bf16), logits, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gate_logits_single_token_bf16_weights kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_quantize_router_gate_q8(
    const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!gate_weights || !gate_weights_q8 || !gate_scales ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_gate_quantize_q8_kernel,
        dim3(num_experts, blocks_per_row), dim3(32), 0, static_cast<hipStream_t>(stream),
        gate_weights, gate_weights_q8, gate_scales, d_model, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 router gate quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_q8_weights(
    const float *hidden, int8_t *hidden_q8, float *hidden_scales,
    const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
    int d_model, int num_experts,
    int device_idx, void *stream)
{
    if (!hidden || !hidden_q8 || !hidden_scales || !gate_weights_q8 || !gate_scales || !logits ||
        d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();

    const int blocks_per_row = d_model / 32;
    hipLaunchKernelGGL(rocm_moe_router_hidden_quantize_q8_kernel,
        dim3(blocks_per_row), dim3(32), 0, hip_stream,
        hidden, hidden_q8, hidden_scales, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 hidden quantize kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int threads = 128;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_q8_weights_kernel,
        dim3(num_experts), dim3(threads), 0, hip_stream,
        hidden_q8, hidden_scales, gate_weights_q8, gate_scales, logits,
        d_model, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE Q8 gate logits kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gate_logits_single_token_grouped(
    const float *hidden, const float *gate_weights, float *logits,
    int d_model, int num_experts, size_t shared_mem_bytes,
    int device_idx, void *stream)
{
    if (!hidden || !gate_weights || !logits || d_model <= 0 || num_experts <= 0)
    {
        fprintf(stderr, "HIP MoE grouped gate logits invalid args\n");
        return false;
    }

    const size_t required_shared = static_cast<size_t>(d_model) * sizeof(float);
    if (shared_mem_bytes < required_shared)
    {
        fprintf(stderr, "HIP MoE grouped gate logits shared memory too small: requested=%zu required=%zu\n",
                shared_mem_bytes, required_shared);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const bool use_vec4 = ((d_model & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0) &&
                          ((reinterpret_cast<uintptr_t>(gate_weights) & 0xfu) == 0);
    const int blocks = (num_experts + kGateLogitsExpertsPerBlock - 1) / kGateLogitsExpertsPerBlock;
    hipLaunchKernelGGL(rocm_moe_gate_logits_single_token_grouped_kernel,
        dim3(blocks), dim3(kGateLogitsGroupedThreads), shared_mem_bytes, static_cast<hipStream_t>(stream),
        hidden, gate_weights, logits, d_model, num_experts, use_vec4);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE grouped gate_logits_single_token kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !logits || !expert_indices || !expert_weights ||
        seq_len <= 0 || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK || top_k > num_experts)
    {
        fprintf(stderr, "HIP MoE softmax_topk invalid arguments\n");
        return false;
    }

    // One block per token, 256 threads for parallel softmax
    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_softmax_topk_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits, expert_indices, expert_weights,
        seq_len, num_experts, top_k, normalize_weights);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime(
    float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_softmax_topk_decode_runtime_wave64(
    const float *logits,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!logits || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_softmax_topk_decode_runtime_wave64_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE softmax_topk_decode_runtime_wave64 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
    const float *partials,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int k_partitions, int top_k,
    bool normalize_weights,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!partials || !runtime || num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        k_partitions <= 0 || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
    {
        return false;
    }
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
    {
        return false;
    }

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_router_kpart_reduce_softmax_topk_decode_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        partials,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, k_partitions, top_k,
        normalize_weights,
        write_legacy_outputs,
        update_runtime_histogram);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE router_kpart_reduce_softmax_topk_decode_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_decode_route_select_runtime(
    const int *expert_indices,
    const float *expert_weights,
    void *runtime,
    float *legacy_indices,
    float *legacy_weights,
    int num_experts, int top_k,
    bool write_legacy_outputs,
    bool update_runtime_histogram,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!expert_indices || !expert_weights || !runtime || top_k <= 0 || top_k > kDeviceMoEMaxTopK)
        return false;
    if (write_legacy_outputs && (!legacy_indices || !legacy_weights))
        return false;

    const int threads = 32;
    hipLaunchKernelGGL(rocm_moe_decode_route_select_runtime_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_indices, expert_weights,
        static_cast<DeviceMoELayerRuntimeView *>(runtime),
        legacy_indices, legacy_weights,
        num_experts, top_k,
        write_legacy_outputs,
        update_runtime_histogram);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE decode_route_select_runtime kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_gather_tokens(
    const float *hidden, float *batch_buffer,
    const int *token_indices,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_gather_tokens_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        hidden, batch_buffer, token_indices, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE gather_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_copy_token_row(
    const float *source, float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!source || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_copy_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        source, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE copy_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_scatter_add(
    float *output, const float *expert_output,
    const int *token_indices, const float *weights,
    int num_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_scatter_add_kernel,
        dim3(num_tokens), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output, token_indices, weights, num_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_write_token_row(
    float *destination, const float *row_buffer,
    int row_index, int row_width,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!destination || !row_buffer || row_index < 0 || row_width <= 0 || !stream)
        return false;

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_write_row_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        destination, row_buffer, row_index, row_width);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE write_token_row kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_shared_expert_gate(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    int device_idx, void *stream)
{
    // TIMING: skip device guard and error clear to test if they cause sync
    // llaminar2::HipDeviceGuard::setDevice(device_idx);
    // (void)hipGetLastError();

    // Phase 1: Compute dot products → sigmoid → gate_scratch[seq_len]
    {
        const int threads = 256;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_dot_kernel,
            dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
            input, gate_inp, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate dot kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    // Phase 2: Scale shared_output by gate values
    {
        const int total = seq_len * d_model;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_moe_shared_expert_gate_scale_kernel,
            dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
            shared_output, gate_scratch, seq_len, d_model);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP MoE shared_expert_gate scale kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    return true;
}

bool hipMoE_shared_expert_gate_decode_fused(
    const float *input, const float *gate_inp,
    float *shared_output, int d_model,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_decode_fused_kernel,
        dim3(1), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_decode_fused kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_shared_expert_gate_add(
    const float *input, const float *gate_inp,
    float *shared_output, const float *routed_residual,
    float *combined_output, int seq_len, int d_model,
    int device_idx, void *stream)
{
    if (!input || !gate_inp || !shared_output || !routed_residual ||
        !combined_output || seq_len <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add invalid args\n");
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    hipLaunchKernelGGL(rocm_moe_shared_expert_gate_add_kernel,
        dim3(seq_len), dim3(threads), 0, static_cast<hipStream_t>(stream),
        input, gate_inp, shared_output, routed_residual, combined_output,
        seq_len, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared_expert_gate_add kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_swiglu(
    float *gate, const float *up,
    int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    if (blocks == 0 && count > 0) return true; // No-op for empty

    hipLaunchKernelGGL(rocm_moe_swiglu_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        gate, up, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE swiglu kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_weighted_add(
    float *output, const float *input,
    float weight, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (count <= 0) return true;

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_weighted_add_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, input, weight, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE weighted_add kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"

// =========================================================================
// Phase 2 Kernels: Histogram + Expert Mask
// =========================================================================

// =========================================================================
// Kernel 7: Histogram Record
//
// Atomically increments per-expert counters based on routing indices.
// One block per token; threads cover the top_k dimension.
// =========================================================================

__global__ void rocm_moe_histogram_record_kernel(
    const int *__restrict__ routing_indices, // [seq_len * top_k]
    unsigned long long *__restrict__ histogram, // [max_layers * num_experts]
    int seq_len, int num_experts, int top_k,
    int layer_offset) // = layer_idx * num_experts
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    for (int k = threadIdx.x; k < top_k; k += blockDim.x)
    {
        int expert_idx = routing_indices[token * top_k + k];
        if (expert_idx >= 0 && expert_idx < num_experts)
        {
            atomicAdd(&histogram[layer_offset + expert_idx], 1ULL);
        }
    }
}

// =========================================================================
// Kernel 8: Apply Expert Mask
//
// Zero out routing weights for masked-off experts.
// One block per token; threads cover the top_k dimension.
// =========================================================================

__global__ void rocm_moe_apply_expert_mask_kernel(
    float *__restrict__ routing_weights,       // [seq_len * top_k]
    const int *__restrict__ routing_indices,   // [seq_len * top_k]
    const bool *__restrict__ expert_mask,      // [num_experts]
    int seq_len, int top_k)
{
    const int token = blockIdx.x;
    if (token >= seq_len) return;

    for (int k = threadIdx.x; k < top_k; k += blockDim.x)
    {
        int idx = token * top_k + k;
        int expert_idx = routing_indices[idx];
        if (!expert_mask[expert_idx])
        {
            routing_weights[idx] = 0.0f;
        }
    }
}

// =========================================================================
// Kernel 9: Histogram Reset
//
// Zero a layer's slice of the histogram buffer.
// =========================================================================

__global__ void rocm_moe_histogram_reset_kernel(
    unsigned long long *__restrict__ histogram, // [max_layers * num_experts]
    int layer_offset, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts)
    {
        histogram[layer_offset + idx] = 0ULL;
    }
}

// =========================================================================
// Phase 2 extern "C" bridge functions
// =========================================================================

extern "C"
{

bool hipMoE_histogram_record(
    const int *routing_indices, unsigned long long *histogram,
    int seq_len, int num_experts, int top_k, int layer_idx,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = (top_k <= 256) ? ((top_k + 31) / 32 * 32) : 256;
    const int actual_threads = (threads < 32) ? 32 : threads;
    const int layer_offset = layer_idx * num_experts;

    hipLaunchKernelGGL(rocm_moe_histogram_record_kernel,
        dim3(seq_len), dim3(actual_threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, histogram,
        seq_len, num_experts, top_k, layer_offset);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE histogram_record kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_apply_expert_mask(
    float *routing_weights, const int *routing_indices,
    const bool *expert_mask,
    int seq_len, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = (top_k <= 256) ? ((top_k + 31) / 32 * 32) : 256;
    const int actual_threads = (threads < 32) ? 32 : threads;

    hipLaunchKernelGGL(rocm_moe_apply_expert_mask_kernel,
        dim3(seq_len), dim3(actual_threads), 0, static_cast<hipStream_t>(stream),
        routing_weights, routing_indices, expert_mask,
        seq_len, top_k);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE apply_expert_mask kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_histogram_reset(
    unsigned long long *histogram, int layer_idx, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int layer_offset = layer_idx * num_experts;
    const int threads = 256;
    const int blocks = (num_experts + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_histogram_reset_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        histogram, layer_offset, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE histogram_reset kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 2)

// =========================================================================
// Phase 3 Kernels: Device-Side Token Grouping
// =========================================================================

// =========================================================================
// Kernel 10: Count Per Expert
//
// Counts how many token-slots go to each expert via atomicAdd.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

__global__ void rocm_moe_count_per_expert_kernel(
    const int *__restrict__ routing_indices, // [total_slots]
    int *__restrict__ expert_counts,         // [num_experts]
    int total_slots, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;

    int expert_id = routing_indices[idx];
    if (expert_id >= 0 && expert_id < num_experts)
    {
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// =========================================================================
// Kernel 11: Exclusive Scan
//
// Computes exclusive prefix sum on expert_counts → expert_offsets.
// Single block, single thread (num_experts ≤ 256; serial cost negligible).
// Also preserves expert_counts (the original counts are needed by caller).
// =========================================================================

__global__ void rocm_moe_exclusive_scan_kernel(
    int *__restrict__ expert_counts,   // [num_experts] IN (read, then overwritten with original counts)
    int *__restrict__ expert_offsets,   // [num_experts] OUT (exclusive prefix sum)
    int num_experts)
{
    if (threadIdx.x != 0) return;

    int running = 0;
    for (int e = 0; e < num_experts; ++e)
    {
        int count = expert_counts[e];
        expert_offsets[e] = running;
        running += count;
        // expert_counts[e] already contains the correct count — no modification needed
    }
}

// =========================================================================
// Kernel 11b: Max Expert Count (device-side reduction)
//
// Computes max(expert_counts[0..num_experts-1]) entirely on device.
// Single wavefront (64 threads), warp-shuffle reduction.
// Grid: 1 block, Block: 64 threads
// =========================================================================

__global__ void rocm_moe_max_expert_count_kernel(
    const int *__restrict__ expert_counts,  // [num_experts]
    int *__restrict__ d_max_out,            // [1] output scalar
    int num_experts)
{
    const int tid = threadIdx.x;
    int local_max = 0;

    // Each thread processes multiple experts (stride 64)
    for (int e = tid; e < num_experts; e += 64)
        local_max = max(local_max, expert_counts[e]);

    // Warp-shuffle reduction within a single wavefront (64 threads)
    for (int offset = 32; offset > 0; offset >>= 1)
        local_max = max(local_max, __shfl_down(local_max, offset));

    if (tid == 0)
        *d_max_out = local_max;
}

// =========================================================================
// Kernel 12: Scatter Tokens
//
// Scatters token indices and weights into expert-grouped arrays.
// Each thread handles one (token, slot) pair, uses atomicAdd on write_heads.
// Grid: ceil(total_slots / 256), Block: 256
// =========================================================================

__global__ void rocm_moe_scatter_tokens_kernel(
    const int *__restrict__ routing_indices,     // [total_slots]
    const float *__restrict__ routing_weights,   // [total_slots]
    const int *__restrict__ expert_offsets,       // [num_experts]
    int *__restrict__ write_heads,               // [num_experts] (init to 0)
    int *__restrict__ grouped_token_indices,     // [total_slots] OUT
    float *__restrict__ grouped_weights,         // [total_slots] OUT
    int total_slots, int num_experts, int top_k)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;

    int token_idx = slot / top_k;
    int expert_id = routing_indices[slot];

    if (expert_id >= 0 && expert_id < num_experts)
    {
        int write_pos = atomicAdd(&write_heads[expert_id], 1);
        int dest = expert_offsets[expert_id] + write_pos;
        grouped_token_indices[dest] = token_idx;
        grouped_weights[dest] = routing_weights[slot];
    }
}

__global__ void rocm_moe_group_tokens_small_float_kernel(
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    int *__restrict__ expert_counts,
    int *__restrict__ expert_offsets,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
    float *__restrict__ grouped_weights,
    int *__restrict__ active_expert_ids,
    int total_slots,
    int num_experts,
    int top_k,
    int max_active_experts)
{
    __shared__ int counts[kDeviceMoEMaxExperts];
    __shared__ int offsets[kDeviceMoEMaxExperts];
    __shared__ int write_heads[kDeviceMoEMaxExperts];

    const int tid = threadIdx.x;
    if (tid < num_experts)
    {
        counts[tid] = 0;
        offsets[tid] = 0;
        write_heads[tid] = 0;
    }
    __syncthreads();

    if (tid < total_slots)
    {
        original_to_grouped[tid] = -1;
        const int expert_id = static_cast<int>(routing_indices[tid]);
        if (expert_id >= 0 && expert_id < num_experts)
            atomicAdd(&counts[expert_id], 1);
    }
    __syncthreads();

    if (tid == 0)
    {
        int running = 0;
        int active = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            offsets[expert] = running;
            const int count = counts[expert];
            if (count > 0 && active < max_active_experts)
                active_expert_ids[active++] = expert;
            running += count;
        }
        for (int slot = active; slot < max_active_experts; ++slot)
            active_expert_ids[slot] = -1;
    }
    __syncthreads();

    if (tid < num_experts)
    {
        expert_counts[tid] = counts[tid];
        expert_offsets[tid] = offsets[tid];
    }

    if (tid < total_slots)
    {
        const int expert_id = static_cast<int>(routing_indices[tid]);
        if (expert_id >= 0 && expert_id < num_experts)
        {
            const int local = atomicAdd(&write_heads[expert_id], 1);
            const int dest = offsets[expert_id] + local;
            grouped_token_indices[dest] = tid / top_k;
            original_to_grouped[tid] = dest;
            grouped_weights[dest] = routing_weights[tid];
        }
    }
}

__global__ void rocm_moe_prepare_shared_expert_group_kernel(
    int *__restrict__ expert_offsets,
    int *__restrict__ expert_counts,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
    float *__restrict__ grouped_weights,
    int *__restrict__ active_expert_ids,
    int seq_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        expert_offsets[0] = 0;
        expert_counts[0] = seq_len;
        active_expert_ids[0] = 0;
    }
    if (idx < seq_len)
    {
        grouped_token_indices[idx] = idx;
        original_to_grouped[idx] = idx;
        grouped_weights[idx] = 1.0f;
    }
}

/**
 * @brief Build combined routed+shared verifier grouping metadata.
 *
 * The shared expert is appended as logical expert `num_routed_experts`.
 * Its route weight is the Qwen shared-expert sigmoid gate.  The grouped
 * prefill pipeline then scatters routed and gated shared outputs in one pass.
 */
__global__ void rocm_moe_group_tokens_with_shared_gate_kernel(
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    const float *__restrict__ hidden,
    const float *__restrict__ shared_gate_inp,
    int *__restrict__ expert_counts,
    int *__restrict__ expert_offsets,
    int *__restrict__ grouped_token_indices,
    int *__restrict__ original_to_grouped,
    int *__restrict__ single_expert_ids,
    int *__restrict__ active_expert_ids,
    float *__restrict__ grouped_weights,
    int seq_len,
    int d_model,
    int num_routed_experts,
    int routed_top_k,
    int max_active_experts)
{
    const int tid = threadIdx.x;
    const int row_block = blockIdx.x;
    const int shared_expert = num_routed_experts;
    const int combined_experts = num_routed_experts + 1;
    const int routed_slots = seq_len * routed_top_k;
    const int combined_top_k = routed_top_k + 1;
    const int combined_slots = seq_len * combined_top_k;

    __shared__ int counts[kDeviceMoEMaxExperts + 1];
    __shared__ int offsets[kDeviceMoEMaxExperts + 1];
    __shared__ float partial[256];

    /*
     * Block 0 owns routed metadata and count publication.  The remaining
     * blocks only compute the per-row shared-expert sigmoid weight below.  We
     * deliberately keep the shared expert out of the routed active-expert list
     * so mixed routed/shared codebooks cannot be decoded through one lane.
     */
    if (row_block == 0)
    {
        for (int expert = tid; expert < combined_experts; expert += blockDim.x)
        {
            counts[expert] = 0;
            offsets[expert] = 0;
        }
        if (tid < max_active_experts)
            active_expert_ids[tid] = -1;
        if (tid == 0)
            single_expert_ids[0] = shared_expert;
        for (int slot = tid; slot < routed_slots; slot += blockDim.x)
        {
            const int row = slot / routed_top_k;
            const int route = slot - row * routed_top_k;
            original_to_grouped[row * combined_top_k + route] = -1;
        }
        __syncthreads();

        if (tid < routed_slots)
        {
            const int expert = static_cast<int>(routing_indices[tid]);
            if (expert >= 0 && expert < num_routed_experts)
                atomicAdd(&counts[expert], 1);
        }
        if (tid == 0)
            counts[shared_expert] = seq_len;
        __syncthreads();

        if (tid == 0)
        {
            int running = 0;
            int active_count = 0;
            for (int expert = 0; expert < combined_experts; ++expert)
            {
                const int count = counts[expert];
                if (expert == shared_expert)
                {
                    /*
                     * The shared rows are written by per-row blocks at the
                     * fixed tail position.  Valid router output makes this
                     * equal to the routed prefix length and avoids any
                     * inter-block dependency on block 0's prefix scan.
                     */
                    offsets[expert] = routed_slots;
                }
                else
                {
                    offsets[expert] = running;
                    if (count > 0 && active_count < max_active_experts)
                        active_expert_ids[active_count++] = expert;
                    running += count;
                }
            }
        }
        __syncthreads();

        /*
         * Qwen3.6 MoE has 256 routed experts plus one logical shared expert.
         * A 256-thread block therefore cannot use a single `tid < combined_experts`
         * copy without dropping expert 256.  Use the same strided form as the
         * initialization so the shared expert count/offset is always published.
         */
        for (int expert = tid; expert < combined_experts; expert += blockDim.x)
        {
            expert_counts[expert] = counts[expert];
            expert_offsets[expert] = offsets[expert];
        }

        if (tid < routed_slots)
        {
            const int expert = static_cast<int>(routing_indices[tid]);
            if (expert >= 0 && expert < num_routed_experts)
            {
                int local = 0;
                for (int prev = 0; prev < tid; ++prev)
                {
                    if (static_cast<int>(routing_indices[prev]) == expert)
                        ++local;
                }
                const int row = tid / routed_top_k;
                const int route = tid - row * routed_top_k;
                const int grouped = offsets[expert] + local;
                const int original_slot = row * combined_top_k + route;
                grouped_token_indices[grouped] = row;
                original_to_grouped[original_slot] = grouped;
                grouped_weights[grouped] = routing_weights[tid];
            }
        }
    }

    const int row = row_block;
    if (row < seq_len)
    {
        float dot = 0.0f;
        const float *x = hidden + static_cast<size_t>(row) * d_model;
        for (int col = tid; col < d_model; col += blockDim.x)
            dot += x[col] * shared_gate_inp[col];

        partial[tid] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
                partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
        {
            const int grouped = routed_slots + row;
            const int original_slot = row * combined_top_k + routed_top_k;
            grouped_token_indices[grouped] = row;
            original_to_grouped[original_slot] = grouped;
            grouped_weights[grouped] = 1.0f / (1.0f + expf(-partial[0]));
        }
    }
}

__global__ void rocm_moe_prefill_group_clear_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int max_slots, int num_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime) return;

    if (idx < num_experts && runtime->expert_counts && runtime->expert_offsets)
    {
        runtime->expert_counts[idx] = 0;
        runtime->expert_offsets[idx] = 0;
    }

    if (idx < max_slots)
    {
        if (runtime->route_expert_ids) runtime->route_expert_ids[idx] = -1;
        if (runtime->route_weights) runtime->route_weights[idx] = 0.0f;
        if (runtime->grouped_token_ids) runtime->grouped_token_ids[idx] = -1;
        if (runtime->grouped_route_weights) runtime->grouped_route_weights[idx] = 0.0f;
    }
}

__global__ void rocm_moe_prefill_group_cast_count_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ routing_indices,
    const float *__restrict__ routing_weights,
    int current_slots, int max_slots, int num_experts)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (!runtime || slot >= max_slots) return;
    if (!runtime->route_expert_ids || !runtime->route_weights || !runtime->expert_counts) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    int expert_id = -1;
    float weight = 0.0f;
    if (slot < current_slots)
    {
        expert_id = static_cast<int>(routing_indices[slot]);
        weight = routing_weights[slot];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            expert_id = -1;
            weight = 0.0f;
        }
    }

    runtime->route_expert_ids[slot] = expert_id;
    runtime->route_weights[slot] = weight;
    if (expert_id >= 0)
        atomicAdd(&runtime->expert_counts[expert_id], 1);
}

__global__ void rocm_moe_prefill_group_exclusive_scan_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int num_experts)
{
    if (!runtime || threadIdx.x != 0) return;
    if (!runtime->expert_counts || !runtime->expert_offsets) return;

    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int count = runtime->expert_counts[expert];
        runtime->expert_offsets[expert] = running;
        running += count;
    }
}

__global__ void rocm_moe_prefill_group_scatter_deterministic_kernel(
    DeviceMoELayerRuntimeView *__restrict__ runtime,
    int current_slots, int max_slots, int num_experts, int top_k)
{
    const int expert = blockIdx.x;
    if (!runtime || expert >= num_experts || threadIdx.x != 0) return;
    if (!runtime->route_expert_ids || !runtime->route_weights ||
        !runtime->expert_offsets || !runtime->expert_counts ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (runtime->prefill_route_capacity < static_cast<uint32_t>(max_slots)) return;

    int write = runtime->expert_offsets[expert];
    const int end = runtime->expert_offsets[expert] + runtime->expert_counts[expert];
    for (int slot = 0; slot < max_slots; ++slot)
    {
        if (slot >= current_slots) continue;
        if (runtime->route_expert_ids[slot] != expert) continue;
        if (write < end)
        {
            runtime->grouped_token_ids[write] = slot / top_k;
            runtime->grouped_route_weights[write] = runtime->route_weights[slot];
            ++write;
        }
    }
}

__global__ void rocm_moe_prefill_gather_expert_runtime_kernel(
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    const float *__restrict__ hidden,
    float *__restrict__ batch_buffer,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !hidden || !batch_buffer || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets || !runtime->grouped_token_ids) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

    const int row = linear / d_model;
    const int col = linear - row * d_model;
    const int count = runtime->expert_counts[expert_id];
    const int offset = runtime->expert_offsets[expert_id];

    float value = 0.0f;
    if (row < count)
    {
        const int token_id = runtime->grouped_token_ids[offset + row];
        if (token_id >= 0 && token_id < max_tokens)
            value = hidden[token_id * d_model + col];
    }

    batch_buffer[linear] = value;
}

__global__ void rocm_moe_prefill_scatter_expert_runtime_kernel(
    float *__restrict__ output,
    const float *__restrict__ expert_output,
    const DeviceMoELayerRuntimeView *__restrict__ runtime,
    int expert_id, int max_tokens, int d_model)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = max_tokens * d_model;
    if (!runtime || !output || !expert_output || linear >= total) return;
    if (!runtime->expert_counts || !runtime->expert_offsets ||
        !runtime->grouped_token_ids || !runtime->grouped_route_weights) return;
    if (expert_id < 0 || expert_id >= static_cast<int>(runtime->expert_count)) return;
    if (runtime->prefill_token_capacity < static_cast<uint32_t>(max_tokens)) return;

    const int row = linear / d_model;
    const int col = linear - row * d_model;
    const int count = runtime->expert_counts[expert_id];
    if (row >= count) return;

    const int offset = runtime->expert_offsets[expert_id];
    const int token_id = runtime->grouped_token_ids[offset + row];
    if (token_id < 0 || token_id >= max_tokens) return;

    const float weight = runtime->grouped_route_weights[offset + row];
    atomicAdd(&output[token_id * d_model + col], weight * expert_output[linear]);
}

// =========================================================================
// Phase 3 extern "C" bridge functions
// =========================================================================

extern "C"
{

bool hipMoE_count_per_expert(
    const int *routing_indices, int *expert_counts,
    int total_slots, int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_count_per_expert_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, expert_counts, total_slots, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE count_per_expert kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_exclusive_scan(
    int *expert_counts, int *expert_offsets,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_exclusive_scan_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        expert_counts, expert_offsets, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE exclusive_scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_max_expert_count(
    const int *expert_counts, int *d_max_out,
    int num_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    hipLaunchKernelGGL(rocm_moe_max_expert_count_kernel,
        dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream),
        expert_counts, d_max_out, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE max_expert_count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_scatter_tokens(
    const int *routing_indices, const float *routing_weights,
    const int *expert_offsets, int *write_heads,
    int *grouped_token_indices, float *grouped_weights,
    int total_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (total_slots + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_scatter_tokens_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights, expert_offsets, write_heads,
        grouped_token_indices, grouped_weights,
        total_slots, num_experts, top_k);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE scatter_tokens kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_tokens_small_float(
    const float *routing_indices, const float *routing_weights,
    int *expert_counts, int *expert_offsets,
    int *grouped_token_indices, int *original_to_grouped,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots, int num_experts, int top_k,
    int max_active_experts,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream ||
        !routing_indices || !routing_weights || !expert_counts || !expert_offsets ||
        !grouped_token_indices || !original_to_grouped ||
        !grouped_weights || !active_expert_ids ||
        total_slots <= 0 || total_slots > 64 ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts ||
        top_k <= 0 || top_k > kDeviceMoEMaxTopK ||
        max_active_experts <= 0 || max_active_experts > total_slots ||
        max_active_experts > num_experts)
    {
        fprintf(stderr, "HIP MoE small float grouping invalid arguments\n");
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_group_tokens_small_float_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights,
        expert_counts, expert_offsets,
        grouped_token_indices, original_to_grouped, grouped_weights,
        active_expert_ids,
        total_slots, num_experts, top_k,
        max_active_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE small float grouping kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_tokens_small_float_with_shared_gate(
    const float *routing_indices,
    const float *routing_weights,
    const float *hidden,
    const float *shared_gate_inp,
    int *expert_counts,
    int *expert_offsets,
    int *grouped_token_indices,
    int *original_to_grouped,
    int *single_expert_ids,
    int *active_expert_ids,
    float *grouped_weights,
    int seq_len,
    int d_model,
    int num_routed_experts,
    int routed_top_k,
    int max_active_experts,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int combined_experts = num_routed_experts + 1;
    const int combined_slots = seq_len * (routed_top_k + 1);
    if (!stream || !routing_indices || !routing_weights || !hidden ||
        !shared_gate_inp || !expert_counts || !expert_offsets ||
        !grouped_token_indices || !original_to_grouped || !single_expert_ids ||
        !active_expert_ids || !grouped_weights ||
        seq_len <= 0 || seq_len > 4 || d_model <= 0 ||
        num_routed_experts <= 0 || num_routed_experts > kDeviceMoEMaxExperts ||
        routed_top_k <= 0 || routed_top_k > kDeviceMoEMaxTopK ||
        max_active_experts <= 0 || max_active_experts > combined_slots ||
        max_active_experts > combined_experts)
    {
        fprintf(stderr, "HIP MoE combined shared grouping invalid arguments\n");
        return false;
    }

    /*
     * Keep four wavefronts for the verifier-sized shared-gate dot. A two-wave
     * launch is numerically correct, but measured slower on gfx906 once graph
     * replay overhead and the metadata block are included.
     */
    constexpr int kSharedGateThreads = 256;
    hipLaunchKernelGGL(rocm_moe_group_tokens_with_shared_gate_kernel,
        dim3(seq_len), dim3(kSharedGateThreads), 0, static_cast<hipStream_t>(stream),
        routing_indices, routing_weights, hidden, shared_gate_inp,
        expert_counts, expert_offsets, grouped_token_indices, original_to_grouped,
        single_expert_ids, active_expert_ids, grouped_weights,
        seq_len, d_model, num_routed_experts, routed_top_k, max_active_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE combined shared grouping kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prepare_shared_expert_group(
    int *expert_offsets,
    int *expert_counts,
    int *grouped_token_indices,
    int *original_to_grouped,
    float *grouped_weights,
    int *active_expert_ids,
    int seq_len,
    int device_idx,
    void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!stream || !expert_offsets || !expert_counts ||
        !grouped_token_indices || !original_to_grouped ||
        !grouped_weights || !active_expert_ids ||
        seq_len <= 0)
    {
        fprintf(stderr, "HIP MoE shared expert grouping invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int blocks = (seq_len + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prepare_shared_expert_group_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        expert_offsets, expert_counts,
        grouped_token_indices, original_to_grouped, grouped_weights,
        active_expert_ids, seq_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE shared expert grouping kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_group_prefill_routes_runtime(
    const float *routing_indices, const float *routing_weights,
    void *runtime,
    int current_slots, int max_slots, int num_experts, int top_k,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !routing_indices || !routing_weights ||
        current_slots < 0 || max_slots <= 0 || current_slots > max_slots ||
        num_experts <= 0 || num_experts > kDeviceMoEMaxExperts || top_k <= 0)
    {
        fprintf(stderr, "HIP MoE group_prefill_routes invalid arguments\n");
        return false;
    }

    auto *runtime_view = static_cast<DeviceMoELayerRuntimeView *>(runtime);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int threads = 256;
    const int clear_items = (max_slots > num_experts) ? max_slots : num_experts;
    const int clear_blocks = (clear_items + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_prefill_group_clear_kernel,
        dim3(clear_blocks), dim3(threads), 0, hip_stream,
        runtime_view, max_slots, num_experts);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group clear kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    const int slot_blocks = (max_slots + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_group_cast_count_kernel,
        dim3(slot_blocks), dim3(threads), 0, hip_stream,
        runtime_view, routing_indices, routing_weights,
        current_slots, max_slots, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group cast/count kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_exclusive_scan_kernel,
        dim3(1), dim3(1), 0, hip_stream,
        runtime_view, num_experts);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group scan kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(rocm_moe_prefill_group_scatter_deterministic_kernel,
        dim3(num_experts), dim3(1), 0, hip_stream,
        runtime_view, current_slots, max_slots, num_experts, top_k);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill group scatter kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    return true;
}

bool hipMoE_prefill_gather_expert_runtime(
    const void *runtime,
    const float *hidden,
    float *batch_buffer,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!runtime || !hidden || !batch_buffer || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_gather_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        hidden, batch_buffer, expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill gather runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_prefill_scatter_expert_runtime(
    float *output,
    const float *expert_output,
    const void *runtime,
    int expert_id, int max_tokens, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    if (!output || !expert_output || !runtime || expert_id < 0 || max_tokens <= 0 || d_model <= 0)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime invalid arguments\n");
        return false;
    }

    const int threads = 256;
    const int total = max_tokens * d_model;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_moe_prefill_scatter_expert_runtime_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        output, expert_output,
        static_cast<const DeviceMoELayerRuntimeView *>(runtime),
        expert_id, max_tokens, d_model);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE prefill scatter runtime launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 3)

// =========================================================================
// Phase 4: Tensor-aware utility kernels
// =========================================================================

/// Convert int array to float array on device (for routing index tensors)
__global__ void rocm_moe_int_to_float_kernel(
    const int *__restrict__ input, float *__restrict__ output, int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count)
        output[idx] = static_cast<float>(input[idx]);
}

/// Convert float array to int array on device (reverse: for GPU-side token grouping)
__global__ void rocm_moe_float_to_int_kernel(
    const float *__restrict__ input, int *__restrict__ output, int count)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count)
        output[idx] = static_cast<int>(input[idx]);
}

extern "C" {

bool hipMoE_stage_mutable_pointer_arrays(
    float **d_a,
    float **d_b,
    float *const *h_a,
    float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE mutable pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.mutable_ptrs_a[i] = h_a[i];
        payload.mutable_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_mutable_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE mutable pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_stage_const_pointer_arrays(
    const float **d_a,
    const float **d_b,
    const float *const *h_a,
    const float *const *h_b,
    int count,
    int device_idx,
    void *stream)
{
    if (!d_a || !d_b || !h_a || !h_b || count <= 0 ||
        count > kMoEPointerStageMax || !stream)
    {
        fprintf(stderr, "HIP MoE const pointer staging invalid arguments: count=%d stream=%p\n", count, stream);
        return false;
    }

    MoEPointerStagePayload payload{};
    for (int i = 0; i < count; ++i)
    {
        if (!h_a[i] || !h_b[i])
        {
            fprintf(stderr, "HIP MoE const pointer staging null pointer at slot %d\n", i);
            return false;
        }
        payload.const_ptrs_a[i] = h_a[i];
        payload.const_ptrs_b[i] = h_b[i];
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();
    hipLaunchKernelGGL(moe_stage_const_pointer_arrays_kernel,
        dim3(1), dim3(kMoEPointerStageMax), 0, static_cast<hipStream_t>(stream),
        payload, d_a, d_b, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE const pointer staging launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_int_to_float(
    const int *d_input, float *d_output, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_moe_int_to_float_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE int_to_float kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool hipMoE_float_to_int(
    const float *d_input, int *d_output, int count,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    // Reuse int_to_float kernel pattern with reversed types —
    // a trivial __device__ lambda would be ideal but HIP doesn't support
    // device lambdas in extern "C".  Instead, add a dedicated kernel below.
    // For now, use a trivial inline kernel via hipLaunchKernelGGL.
    // The kernel is defined right after this bridge.
    hipLaunchKernelGGL(rocm_moe_float_to_int_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        d_input, d_output, count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP MoE float_to_int kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (Phase 4)
