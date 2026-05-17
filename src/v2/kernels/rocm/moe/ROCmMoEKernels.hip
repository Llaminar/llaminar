/**
 * @file ROCmMoEKernels.hip
 * @brief HIP __global__ kernels for MoE operations
 *
 * Contains 6 HIP kernels for device-native MoE execution:
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
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "../../../backends/rocm/HipDeviceGuard.h"

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
    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += hidden[i] * w[i];
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
        row[e] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Top-k selection (single thread — k is small, typically 8)
    // NOTE: We must NOT modify row[] here — it contains post-softmax probabilities
    // that will be copied to result.router_logits by the host.
    if (threadIdx.x == 0)
    {
        int *out_idx = expert_indices + token * top_k;
        float *out_wt = expert_weights + token * top_k;

        // Initialize selected indices to -1 for the "already picked" check
        for (int k = 0; k < top_k; ++k)
            out_idx[k] = -1;

        for (int k = 0; k < top_k; ++k)
        {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int e = 0; e < num_experts; ++e)
            {
                if (row[e] > best_val)
                {
                    // Check not already selected
                    bool already_picked = false;
                    for (int j = 0; j < k; ++j)
                    {
                        if (out_idx[j] == e) { already_picked = true; break; }
                    }
                    if (!already_picked)
                    {
                        best_val = row[e];
                        best_idx = e;
                    }
                }
            }
            out_idx[k] = best_idx;
            out_wt[k] = best_val;
        }

        // Step 5: Optionally renormalize top-k weights
        if (normalize_weights)
        {
            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                topk_sum += out_wt[k];
            if (topk_sum > 0.0f)
            {
                float inv = 1.0f / topk_sum;
                for (int k = 0; k < top_k; ++k)
                    out_wt[k] *= inv;
            }
        }
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

    const int threads = 256;
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

bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices, float *expert_weights,
    int seq_len, int num_experts, int top_k,
    bool normalize_weights,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

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

bool hipMoE_shared_expert_gate(
    const float *input, const float *gate_inp,
    float *shared_output, float *gate_scratch,
    int seq_len, int d_model,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    (void)hipGetLastError();

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
