/**
 * @file CUDASamplingKernels.cu
 * @brief CUDA GPU-side sampling kernels for argmax and top-k
 *
 * Provides GPU-side argmax and top-k selection over FP32 logits.
 * Mirrors the ROCm implementations in ROCmArgmaxKernels.hip and
 * ROCmSamplingKernels.hip for cross-backend parity.
 *
 * Kernels:
 *   - Argmax: Single-block 1024-thread shared memory reduction (~5µs for 76K)
 *   - Top-K: Single-block 32-thread insertion sort + merge (~15-25µs for 76K, k=40)
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

// Maximum k supported by the top-k kernel
constexpr int TOPK_MAX_K = 256;
constexpr int TOPK_THREADS = 32;

// ============================================================================
// Argmax Reduction Kernel — Single-block shared memory reduction
// ============================================================================

__global__ void cuda_argmax_f32_kernel(
    const float *__restrict__ data,
    int n,
    float *__restrict__ out_value,
    int *__restrict__ out_index)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;

    // Phase 1: Each thread finds the max in its strided portion
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < n; i += blockDim.x)
    {
        float val = data[i];
        if (val > local_max)
        {
            local_max = val;
            local_idx = i;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Phase 2: Tree reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (smax[tid + stride] > smax[tid])
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Phase 3: Thread 0 writes result
    if (tid == 0)
    {
        *out_value = smax[0];
        *out_index = sidx[0];
    }
}

// ============================================================================
// Top-K Selection Kernel — Single-block, warp-level reduction
// ============================================================================

__global__ void cuda_topk_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float *__restrict__ out_values,
    int *__restrict__ out_indices)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Phase 1: Per-thread strided scan with insertion sort
    float local_vals[TOPK_MAX_K];
    int local_idxs[TOPK_MAX_K];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < n; i += num_threads)
    {
        float val = data[i];

        if (local_count >= k && val <= local_vals[k - 1])
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (val > local_vals[j])
            {
                local_vals[j + 1] = local_vals[j];
                local_idxs[j + 1] = local_idxs[j];
                pos = j;
            }
            else
            {
                break;
            }
        }
        local_vals[pos] = val;
        local_idxs[pos] = i;
        if (local_count < k)
            local_count++;
    }

    // Phase 2: Write candidates to shared memory
    extern __shared__ char shared_mem[];
    float *s_vals = reinterpret_cast<float *>(shared_mem);
    int *s_idxs = reinterpret_cast<int *>(shared_mem + num_threads * k * sizeof(float));

    int base = tid * k;
    for (int i = 0; i < k; ++i)
    {
        s_vals[base + i] = local_vals[i];
        s_idxs[base + i] = local_idxs[i];
    }
    __syncthreads();

    // Phase 3: Thread 0 merges all candidates → global top-k
    if (tid == 0)
    {
        int ptrs[TOPK_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_thread = -1;

            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    float v = s_vals[t * k + ptrs[t]];
                    if (v > best_val)
                    {
                        best_val = v;
                        best_thread = t;
                    }
                }
            }

            if (best_thread >= 0)
            {
                out_values[out_i] = best_val;
                out_indices[out_i] = s_idxs[best_thread * k + ptrs[best_thread]];
                ptrs[best_thread]++;
            }
            else
            {
                out_values[out_i] = -FLT_MAX;
                out_indices[out_i] = -1;
            }
        }
    }
}

// ============================================================================
// Logit Penalty Application Kernel — Subtract sparse penalties from logits
// ============================================================================

__global__ void cuda_apply_logit_penalties_f32_kernel(
    float *__restrict__ logits,
    const int *__restrict__ token_ids,
    const float *__restrict__ penalties,
    int num_penalties,
    int vocab_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_penalties)
        return;

    int token_id = token_ids[idx];
    if (token_id >= 0 && token_id < vocab_size)
    {
        logits[token_id] -= penalties[idx];
    }
}

// ============================================================================
// Extern "C" Wrappers
// ============================================================================

extern "C"
{

    bool cudaOps_argmax_f32(
        const float *data,
        int n,
        float *out_value,
        int *out_index,
        int device_idx,
        void *stream)
    {
        if (n <= 0 || !data || !out_value || !out_index)
            return false;

        cudaSetDevice(device_idx);

        const int threads = 1024;
        const size_t smem_size = threads * (sizeof(float) + sizeof(int));

        cuda_argmax_f32_kernel<<<1, threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
            data, n, out_value, out_index);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Argmax FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_topk_f32(
        const float *data,
        int n,
        int k,
        float *out_values,
        int *out_indices,
        int device_idx,
        void *stream)
    {
        if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_values || !out_indices)
            return false;

        if (k > n)
            k = n;

        cudaSetDevice(device_idx);

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

        cuda_topk_f32_kernel<<<1, threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
            data, n, k, out_values, out_indices);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Top-K FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // ============================================================================
    // Logit Penalty Application Kernel
    // ============================================================================

    bool cudaOps_apply_logit_penalties_f32(
        float *logits,
        const int *token_ids,
        const float *penalties,
        int num_penalties,
        int vocab_size,
        int device_idx,
        void *stream)
    {
        if (num_penalties <= 0 || !logits || !token_ids || !penalties)
            return false;

        cudaSetDevice(device_idx);

        // Simple 1D grid: one thread per penalty entry
        const int threads = 256;
        const int blocks = (num_penalties + threads - 1) / threads;

        cuda_apply_logit_penalties_f32_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, token_ids, penalties, num_penalties, vocab_size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Apply Logit Penalties kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
