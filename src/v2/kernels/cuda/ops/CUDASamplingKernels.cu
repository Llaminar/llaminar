/**
 * @file CUDASamplingKernels.cu
 * @brief CUDA GPU-side sampling kernels for argmax and top-k
 *
 * Provides GPU-side argmax and top-k selection over FP32 logits.
 * Mirrors the ROCm implementations in ROCmArgmaxKernels.hip and
 * ROCmSamplingKernels.hip for cross-backend parity.
 *
 * Kernels:
 *   - Argmax: Two-pass multi-block reduction (pass 1 spreads the vocab across
 *     many blocks/SMs producing one partial per block; pass 2 reduces the
 *     partials in a single small block). Falls back to a single-block reduction
 *     when no partial scratch is supplied.
 *   - Top-K: Single-block 32-thread insertion sort + merge (~15-25µs for 76K, k=40)
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include "../../common/SamplingMath.h"

// Maximum k supported by the top-k kernel
constexpr int TOPK_MAX_K = 256;
constexpr int TOPK_THREADS = 32;
static_assert(TOPK_MAX_K == llaminar2::sampling_math::kMaxTopK,
              "CUDA sampling TOPK_MAX_K must match shared sampling math");

// ── Argmax multi-block reduction tuning ─────────────────────────────────────
// Threads per block for both reduction passes. Must be a power of two because
// the in-block tree reduction halves the active range each step.
constexpr int ARGMAX_REDUCE_THREADS = 256;
constexpr int ARGMAX_FINALIZE_THREADS = 256;
// Target elements processed per thread in pass 1. Chosen so each thread reads a
// handful of elements (good memory coalescing) while still spreading the vocab
// across enough blocks to occupy most SMs. 8 → ~74 blocks for a 152K vocab,
// which keeps all 82 SMs of an RTX 3090 busy without oversubscription.
constexpr int ARGMAX_ELEMS_PER_THREAD = 8;

// ============================================================================
// Argmax Pass 1 — Multi-block partial reduction
// ============================================================================
//
// Each block performs a grid-strided scan over its slice of the input and
// reduces it (in shared memory) to a single (value, index) partial, written to
// partial_vals[blockIdx.x] / partial_idxs[blockIdx.x]. Spreading the work over
// gridDim.x blocks lets the reduction use many SMs instead of a single one.
__global__ void cuda_argmax_partial_f32_kernel(
    const float *__restrict__ data,
    int n,
    float *__restrict__ partial_vals,
    int *__restrict__ partial_idxs)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;

    // Phase 1: Grid-strided scan — each thread reduces its share to a local max.
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = gid; i < n; i += grid_stride)
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

    // Phase 2: In-block tree reduction (blockDim.x is a power of two).
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

    // Phase 3: Thread 0 emits this block's partial result.
    if (tid == 0)
    {
        partial_vals[blockIdx.x] = smax[0];
        partial_idxs[blockIdx.x] = sidx[0];
    }
}

// ============================================================================
// Argmax Pass 2 — Finalize over per-block partials (single block)
// ============================================================================
//
// Reduces the `num_partials` partial results from pass 1 into the final
// (value, index). Runs as a single small block; num_partials is bounded by the
// pass-1 grid size (a few hundred at most), so this is cheap.
__global__ void cuda_argmax_finalize_f32_kernel(
    const float *__restrict__ partial_vals,
    const int *__restrict__ partial_idxs,
    int num_partials,
    float *__restrict__ out_value,
    int *__restrict__ out_index)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;

    // Phase 1: Strided scan of the partials into a per-thread local max.
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = tid; i < num_partials; i += blockDim.x)
    {
        float val = partial_vals[i];
        if (val > local_max)
        {
            local_max = val;
            local_idx = partial_idxs[i];
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Phase 2: In-block tree reduction (blockDim.x is a power of two).
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

    // Phase 3: Thread 0 writes the global argmax.
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
// Top-K / Top-P / Temperature Sampling Kernel
// ============================================================================

__global__ void cuda_topk_topp_sample_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float top_p,
    float temperature,
    unsigned long long rng_seed,
    unsigned long long rng_offset,
    int *__restrict__ out_token)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

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
        float merged_vals[TOPK_MAX_K];
        int merged_idxs[TOPK_MAX_K];
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

        float weights[TOPK_MAX_K];
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

__global__ void cuda_topk_topp_distribution_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float top_p,
    float temperature,
    int *__restrict__ out_token_ids,
    float *__restrict__ out_probs)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

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
        float merged_vals[TOPK_MAX_K];
        int merged_idxs[TOPK_MAX_K];
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

        float weights[TOPK_MAX_K];
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

__global__ void cuda_sample_distribution_f32_kernel(
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

__global__ void cuda_speculative_verify_distribution_kernel(
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

__global__ void cuda_speculative_verify_distribution_threshold_kernel(
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
        float *partial_vals,
        int *partial_idxs,
        int partial_capacity,
        int device_idx,
        void *stream)
    {
        if (n <= 0 || !data || !out_value || !out_index)
            return false;

        // The partial-reduction scratch is mandatory: every production caller
        // supplies arena-owned scratch (single-device orchestrator and the
        // multi-device sampler). There is no single-block fallback — a missing
        // or undersized scratch buffer is a wiring bug, so fail loud.
        if (!partial_vals || !partial_idxs || partial_capacity < 1)
        {
            fprintf(stderr,
                    "CUDA Argmax FP32: missing partial-reduction scratch "
                    "(partial_vals=%p partial_idxs=%p capacity=%d)\n",
                    (void *)partial_vals, (void *)partial_idxs, partial_capacity);
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        // Two-pass multi-block reduction.
        // Size the pass-1 grid so each thread processes ~ARGMAX_ELEMS_PER_THREAD
        // elements, then clamp to the partial-buffer capacity. This spreads the
        // vocab across many blocks (and thus SMs) instead of a single block.
        const int threads = ARGMAX_REDUCE_THREADS;
        const long per_block = static_cast<long>(threads) * ARGMAX_ELEMS_PER_THREAD;
        long blocks = (static_cast<long>(n) + per_block - 1) / per_block;
        if (blocks < 1)
            blocks = 1;
        if (blocks > partial_capacity)
            blocks = partial_capacity;
        const int num_blocks = static_cast<int>(blocks);

        // Pass 1: each block reduces its slice to one partial.
        const size_t smem1 = threads * (sizeof(float) + sizeof(int));
        cuda_argmax_partial_f32_kernel<<<num_blocks, threads, smem1, s>>>(
            data, n, partial_vals, partial_idxs);

        // Pass 2: single small block reduces the partials to the final result.
        const int fthreads = ARGMAX_FINALIZE_THREADS;
        const size_t smem2 = fthreads * (sizeof(float) + sizeof(int));
        cuda_argmax_finalize_f32_kernel<<<1, fthreads, smem2, s>>>(
            partial_vals, partial_idxs, num_blocks, out_value, out_index);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Argmax FP32 (multi-block) launch failed: %s\n",
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

    bool cudaOps_sample_topk_topp_f32(
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

        cudaSetDevice(device_idx);

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

        cuda_topk_topp_sample_f32_kernel<<<1, threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
            data, n, k, top_p, temperature, rng_seed, rng_offset, out_token);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Top-K/Top-P Sample FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_topk_topp_distribution_f32(
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

        cudaSetDevice(device_idx);

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

        cuda_topk_topp_distribution_f32_kernel<<<1, threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
            data, n, k, top_p, temperature, out_token_ids, out_probs);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_distribution_f32(
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

        cudaSetDevice(device_idx);

        cuda_speculative_verify_distribution_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
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

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_sample_distribution_f32(
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

        cudaSetDevice(device_idx);

        cuda_sample_distribution_f32_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            token_ids,
            probs,
            k,
            threshold,
            out_token);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Sample Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_distribution_threshold_f32(
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

        cudaSetDevice(device_idx);

        cuda_speculative_verify_distribution_threshold_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
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

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Distribution Threshold FP32 kernel launch failed: %s\n",
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
