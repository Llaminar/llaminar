/**
 * @file ROCmSamplingKernels.hip
 * @brief ROCm/HIP GPU-side sampling kernels for top-k/top-p
 *
 * Provides a GPU-side top-k selection over FP32 logits, returning the top-k
 * (value, index) pairs. Used in combination with host-side softmax + top-p
 * filtering for full top-k/top-p sampling without D2H of full logits.
 *
 * Design:
 *   - Single-block kernel with 32 threads (one warp/half-wavefront)
 *   - Each thread scans n/32 elements, maintaining local top-k via insertion sort
 *   - Shared memory merge step combines all thread-local results
 *   - Output: top-k (value, index) pairs in descending order of value
 *
 * Performance: ~15-25µs for 76K elements, k=40 on MI50
 *   (vs ~600KB D2H that takes ~300µs)
 */

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../common/SamplingMath.h"

// Maximum k supported by the kernel (limited by shared memory and registers)
constexpr int TOPK_MAX_K = 256;
static_assert(TOPK_MAX_K == llaminar2::sampling_math::kMaxTopK,
              "ROCm sampling TOPK_MAX_K must match shared sampling math");
constexpr int TOPK_SMALL_K_CAP = 32;

// Number of threads per block — one warp on NVIDIA, half wavefront on AMD
// Using 32 threads keeps register pressure manageable for large k
constexpr int TOPK_THREADS = 32;

// ============================================================================
// Top-K Selection Kernel — Single-block, warp-level reduction
// ============================================================================
//
// Algorithm:
//   1. Each of 32 threads scans n/32 elements with insertion sort into local
//      top-k buffer (register arrays)
//   2. All threads write their local top-k candidates to shared memory
//   3. Thread 0 performs a final merge of 32*k candidates → global top-k
//   4. Results written to output arrays in descending order (highest value first)
//
// Correctness guarantee: Each thread keeps all top-k candidates from its chunk,
//   so no globally-top-k element can be missed.

__global__ void rocm_topk_f32_kernel(
    const float* __restrict__ data,
    int n,
    int k,
    float* __restrict__ out_values,
    int* __restrict__ out_indices)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // ========================================================================
    // Phase 1: Per-thread strided scan with insertion sort
    // ========================================================================
    // Each thread maintains a sorted (desc) buffer of its local top-k.
    // Register arrays: max 256 floats + 256 ints = 2KB per thread.
    // With 32 threads = 64KB total — fits in register file on MI50 (256KB/CU).

    float local_vals[TOPK_MAX_K];
    int   local_idxs[TOPK_MAX_K];
    int   local_count = 0;

    // Initialize with -INF so insertion sort works correctly
    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    // Strided scan: thread tid handles elements tid, tid+32, tid+64, ...
    for (int i = tid; i < n; i += num_threads)
    {
        float val = data[i];

        // Quick reject: skip if below current minimum (last element in sorted array)
        if (local_count >= k && val <= local_vals[k - 1])
            continue;

        // Insertion sort: find position and shift
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

    // ========================================================================
    // Phase 2: Write candidates to shared memory
    // ========================================================================
    // Total candidates = num_threads * k (worst case 32 * 256 = 8192)
    // Shared memory layout: [values: num_threads*k floats][indices: num_threads*k ints]

    extern __shared__ char shared_mem[];
    float* s_vals = reinterpret_cast<float*>(shared_mem);
    int*   s_idxs = reinterpret_cast<int*>(shared_mem + num_threads * k * sizeof(float));

    int base = tid * k;
    for (int i = 0; i < k; ++i)
    {
        s_vals[base + i] = local_vals[i];
        s_idxs[base + i] = local_idxs[i];
    }
    __syncthreads();

    // ========================================================================
    // Phase 3: Thread 0 merges all candidates → global top-k
    // ========================================================================
    // We have num_threads * k candidates, each thread's portion is sorted desc.
    // Use a multi-way merge: maintain a pointer into each thread's sorted list,
    // pick the maximum across all pointers, advance that pointer.
    // O(k * num_threads) total work for k output elements.

    if (tid == 0)
    {
        // Pointers into each thread's sorted list
        int ptrs[TOPK_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int   best_thread = -1;

            // Find the thread with the highest unmerged candidate
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
                // Shouldn't happen unless n < k
                out_values[out_i] = -FLT_MAX;
                out_indices[out_i] = -1;
            }
        }
    }
}

// ============================================================================
// Top-K / Top-P / Temperature Sampling Kernel
// ============================================================================

template <int K_CAP>
__global__ void rocm_topk_topp_sample_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float top_p,
    float temperature,
    unsigned long long rng_seed,
    unsigned long long rng_offset,
    int *__restrict__ out_token)
{
    if (k > K_CAP)
        return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < n; i += num_threads)
    {
        const float val = data[i];
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
            ++local_count;
    }

    extern __shared__ char shared_mem[];
    float *s_vals = reinterpret_cast<float *>(shared_mem);
    int *s_idxs = reinterpret_cast<int *>(shared_mem + num_threads * k * sizeof(float));

    const int base = tid * k;
    for (int i = 0; i < k; ++i)
    {
        s_vals[base + i] = local_vals[i];
        s_idxs[base + i] = local_idxs[i];
    }
    __syncthreads();

    if (tid == 0)
    {
        float merged_vals[K_CAP];
        int merged_idxs[K_CAP];
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
                    const float v = s_vals[t * k + ptrs[t]];
                    if (v > best_val)
                    {
                        best_val = v;
                        best_thread = t;
                    }
                }
            }

            if (best_thread >= 0)
            {
                merged_vals[out_i] = best_val;
                merged_idxs[out_i] = s_idxs[best_thread * k + ptrs[best_thread]];
                ++ptrs[best_thread];
            }
            else
            {
                merged_vals[out_i] = -FLT_MAX;
                merged_idxs[out_i] = -1;
            }
        }

        float weights[K_CAP];
        *out_token = llaminar2::sampling_math::sample_topk_topp_from_sorted_with_threshold(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            llaminar2::sampling_math::uniform01(rng_seed, rng_offset),
            weights);
    }
}

template <int K_CAP>
__global__ void rocm_topk_topp_distribution_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float top_p,
    float temperature,
    int *__restrict__ out_token_ids,
    float *__restrict__ out_probs)
{
    if (k > K_CAP)
        return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < n; i += num_threads)
    {
        const float val = data[i];
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
            ++local_count;
    }

    extern __shared__ char shared_mem[];
    float *s_vals = reinterpret_cast<float *>(shared_mem);
    int *s_idxs = reinterpret_cast<int *>(shared_mem + num_threads * k * sizeof(float));

    const int base = tid * k;
    for (int i = 0; i < k; ++i)
    {
        s_vals[base + i] = local_vals[i];
        s_idxs[base + i] = local_idxs[i];
    }
    __syncthreads();

    if (tid == 0)
    {
        float merged_vals[K_CAP];
        int merged_idxs[K_CAP];
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
                    const float v = s_vals[t * k + ptrs[t]];
                    if (v > best_val)
                    {
                        best_val = v;
                        best_thread = t;
                    }
                }
            }

            if (best_thread >= 0)
            {
                merged_vals[out_i] = best_val;
                merged_idxs[out_i] = s_idxs[best_thread * k + ptrs[best_thread]];
                ++ptrs[best_thread];
            }
            else
            {
                merged_vals[out_i] = -FLT_MAX;
                merged_idxs[out_i] = -1;
            }
        }

        float weights[K_CAP];
        llaminar2::sampling_math::build_topk_topp_distribution_from_sorted(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            out_token_ids,
            out_probs,
            weights);
    }
}

__global__ void rocm_sample_distribution_f32_kernel(
    const int *__restrict__ token_ids,
    const float *__restrict__ probs,
    int k,
    float threshold,
    int *__restrict__ out_token)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    *out_token = llaminar2::sampling_math::sample_distribution_with_threshold(
        token_ids, probs, k, threshold);
}

__global__ void rocm_speculative_verify_distribution_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int draft_token,
    unsigned long long accept_seed,
    unsigned long long accept_offset,
    unsigned long long residual_seed,
    unsigned long long residual_offset,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        llaminar2::sampling_math::uniform01(accept_seed, accept_offset),
        llaminar2::sampling_math::uniform01(residual_seed, residual_offset),
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);
}

__global__ void rocm_speculative_verify_distribution_threshold_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int draft_token,
    float accept_threshold,
    float residual_threshold,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_threshold,
        residual_threshold,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);
}

__global__ void rocm_speculative_verify_distribution_thresholds_batch_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int distribution_stride,
    int draft_token0,
    int draft_token1,
    int draft_token2,
    int draft_token3,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    float residual_threshold0,
    float residual_threshold1,
    float residual_threshold2,
    float residual_threshold3,
    int row_count,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count)
        return;

    int draft_token = draft_token0;
    float accept_threshold = accept_threshold0;
    float residual_threshold = residual_threshold0;
    if (row == 1)
    {
        draft_token = draft_token1;
        accept_threshold = accept_threshold1;
        residual_threshold = residual_threshold1;
    }
    else if (row == 2)
    {
        draft_token = draft_token2;
        accept_threshold = accept_threshold2;
        residual_threshold = residual_threshold2;
    }
    else if (row == 3)
    {
        draft_token = draft_token3;
        accept_threshold = accept_threshold3;
        residual_threshold = residual_threshold3;
    }

    const int offset = row * distribution_stride;
    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids + offset,
        target_probs + offset,
        draft_token_ids + offset,
        draft_probs + offset,
        k,
        draft_token,
        accept_threshold,
        residual_threshold,
        out_token + row,
        out_accepted + row,
        out_accept_probability ? out_accept_probability + row : nullptr,
        out_accept_threshold ? out_accept_threshold + row : nullptr);
}

// ============================================================================
// Extern "C" Wrapper
// ============================================================================

extern "C"
{

/**
 * @brief Launch top-k selection kernel on device
 *
 * Finds the k largest elements (value and index) in a float array.
 * Results are in descending order of value (highest first).
 *
 * @param data       Device pointer to FP32 data
 * @param n          Number of elements
 * @param k          Number of top elements to select (1..256)
 * @param out_values Device pointer to store top-k values (k floats)
 * @param out_indices Device pointer to store top-k indices (k ints)
 * @param device_idx GPU device index
 * @param stream     HIP stream (nullptr for default)
 * @return true on success
 */
bool rocmOps_topk_f32(
    const float* data,
    int n,
    int k,
    float* out_values,
    int* out_indices,
    int device_idx,
    void* stream)
{
    if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_values || !out_indices)
        return false;

    // Clamp k to n (can't return more elements than exist)
    if (k > n)
        k = n;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    // Single block, 32 threads — each thread handles n/32 elements
    // Shared memory: (32 * k) floats + (32 * k) ints for candidate staging
    const int threads = TOPK_THREADS;
    const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

    hipLaunchKernelGGL(
        rocm_topk_f32_kernel,
        dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
        data, n, k, out_values, out_indices);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Top-K FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_sample_topk_topp_f32(
    const float *data,
    int n,
    int k,
    float top_p,
    float temperature,
    unsigned long long rng_seed,
    unsigned long long rng_offset,
    int *out_token,
    int device_idx,
    void *stream)
{
    if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_token || !stream)
        return false;

    if (k > n)
        k = n;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int threads = TOPK_THREADS;
    const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

    if (k <= TOPK_SMALL_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_sample_f32_kernel<TOPK_SMALL_K_CAP>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, rng_seed, rng_offset, out_token);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_sample_f32_kernel<TOPK_MAX_K>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, rng_seed, rng_offset, out_token);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Top-K/Top-P Sample FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_topk_topp_distribution_f32(
    const float *data,
    int n,
    int k,
    float top_p,
    float temperature,
    int *out_token_ids,
    float *out_probs,
    int device_idx,
    void *stream)
{
    if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_token_ids || !out_probs || !stream)
        return false;

    if (k > n)
        k = n;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int threads = TOPK_THREADS;
    const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

    if (k <= TOPK_SMALL_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_distribution_f32_kernel<TOPK_SMALL_K_CAP>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, out_token_ids, out_probs);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_distribution_f32_kernel<TOPK_MAX_K>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, out_token_ids, out_probs);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int draft_token,
    unsigned long long accept_seed,
    unsigned long long accept_offset,
    unsigned long long residual_seed,
    unsigned long long residual_offset,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K || !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs || !out_token || !out_accepted || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_speculative_verify_distribution_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_seed,
        accept_offset,
        residual_seed,
        residual_offset,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Distribution FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_sample_distribution_f32(
    const int *token_ids,
    const float *probs,
    int k,
    float threshold,
    int *out_token,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K || !token_ids || !probs || !out_token || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_sample_distribution_f32_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        token_ids,
        probs,
        k,
        threshold,
        out_token);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Sample Distribution FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_threshold_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int draft_token,
    float accept_threshold,
    float residual_threshold,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K || !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs || !out_token || !out_accepted || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_speculative_verify_distribution_threshold_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_threshold,
        residual_threshold,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Distribution Threshold FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_thresholds_batch_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int distribution_stride,
    int draft_token0,
    int draft_token1,
    int draft_token2,
    int draft_token3,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    float residual_threshold0,
    float residual_threshold1,
    float residual_threshold2,
    float residual_threshold3,
    int row_count,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K ||
        distribution_stride < k ||
        row_count <= 0 || row_count > 4 ||
        !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs ||
        !out_token || !out_accepted || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_speculative_verify_distribution_thresholds_batch_kernel,
        dim3(1), dim3(4), 0, static_cast<hipStream_t>(stream),
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        distribution_stride,
        draft_token0,
        draft_token1,
        draft_token2,
        draft_token3,
        accept_threshold0,
        accept_threshold1,
        accept_threshold2,
        accept_threshold3,
        residual_threshold0,
        residual_threshold1,
        residual_threshold2,
        residual_threshold3,
        row_count,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Batch FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// Logit Penalty Application Kernel — Subtract sparse penalties from logits
// ============================================================================

__global__ void rocm_apply_logit_penalties_f32_kernel(
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

bool rocmOps_apply_logit_penalties_f32(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int threads = 256;
    const int blocks = (num_penalties + threads - 1) / threads;

    hipLaunchKernelGGL(
        rocm_apply_logit_penalties_f32_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits, token_ids, penalties, num_penalties, vocab_size);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Apply Logit Penalties kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
