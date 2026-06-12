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
#include <climits>
#include <cmath>
#include <cstdio>
#include "../../common/SamplingMath.h"

// Maximum k supported by the top-k kernel
constexpr int TOPK_MAX_K = 256;
constexpr int TOPK_THREADS = 32;
// Qwen chat defaults use top_k=40. A 40-entry specialization keeps register
// pressure aligned with the actual compact table size used by stochastic MTP.
constexpr int TOPK_MEDIUM_K_CAP = 40;
constexpr int TOPK_SMALL_K_CAP = 64;
constexpr int TOPK_SMALL_K_PARTIAL_BLOCKS = 128;
// Qwen-style top_k=40/64 distributions still benefit from wider partial
// reductions; 32 blocks regressed the ROCm fixed-depth-1 MTP lane.
constexpr int TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS = 64;
constexpr int TOPK_SMALL_K_THREADS = 64;
static_assert(TOPK_MAX_K == llaminar2::sampling_math::kMaxTopK,
              "CUDA sampling TOPK_MAX_K must match shared sampling math");

constexpr int topkSmallKPartialBlockCap(int k)
{
    return k > 32 ? TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS : TOPK_SMALL_K_PARTIAL_BLOCKS;
}

/**
 * @brief Deterministic ordering for Top-K candidates.
 *
 * A parallel reduction should not let equal logits depend on thread traversal
 * order.  We therefore sort by logit descending and token id ascending.  This
 * is the same contract as the ROCm sampler and keeps seeded stochastic decode
 * reproducible when quantized logits contain ties.
 */
__device__ __forceinline__ bool topkCandidateBetter(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index)
{
    if (candidate_index < 0)
        return false;
    if (current_index < 0)
        return true;
    return candidate_value > current_value ||
           (candidate_value == current_value &&
            candidate_index < current_index);
}

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

__device__ __forceinline__ bool argmax_better(float candidate_value,
                                              int candidate_index,
                                              float current_value,
                                              int current_index)
{
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

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
    int local_idx = INT_MAX;
    for (int i = gid; i < n; i += grid_stride)
    {
        float val = data[i];
        if (argmax_better(val, i, local_max, local_idx))
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
            if (argmax_better(smax[tid + stride], sidx[tid + stride],
                              smax[tid], sidx[tid]))
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
    int local_idx = INT_MAX;
    for (int i = tid; i < num_partials; i += blockDim.x)
    {
        float val = partial_vals[i];
        const int idx = partial_idxs[i];
        if (argmax_better(val, idx, local_max, local_idx))
        {
            local_max = val;
            local_idx = idx;
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
            if (argmax_better(smax[tid + stride], sidx[tid + stride],
                              smax[tid], sidx[tid]))
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

__global__ void cuda_argmax_partial_f32_batched_rows_kernel(
    const float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    float *__restrict__ partial_vals,
    int *__restrict__ partial_idxs,
    int row_partial_capacity)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int row = blockIdx.y;
    if (row >= rows)
        return;

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    const float *row_data = data + static_cast<size_t>(row) * static_cast<size_t>(row_stride);

    float local_max = -FLT_MAX;
    int local_idx = INT_MAX;
    for (int i = gid; i < cols; i += grid_stride)
    {
        const float val = row_data[i];
        if (argmax_better(val, i, local_max, local_idx))
        {
            local_max = val;
            local_idx = i;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            argmax_better(smax[tid + stride], sidx[tid + stride],
                          smax[tid], sidx[tid]))
        {
            smax[tid] = smax[tid + stride];
            sidx[tid] = sidx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        const int offset = row * row_partial_capacity + blockIdx.x;
        partial_vals[offset] = smax[0];
        partial_idxs[offset] = sidx[0];
    }
}

__global__ void cuda_argmax_finalize_f32_batched_rows_kernel(
    const float *__restrict__ partial_vals,
    const int *__restrict__ partial_idxs,
    int num_partials,
    int row_partial_capacity,
    float *__restrict__ out_values,
    int *__restrict__ out_indices)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * row_partial_capacity;

    float local_max = -FLT_MAX;
    int local_idx = INT_MAX;
    for (int i = tid; i < num_partials; i += blockDim.x)
    {
        const float val = partial_vals[base + i];
        const int idx = partial_idxs[base + i];
        if (argmax_better(val, idx, local_max, local_idx))
        {
            local_max = val;
            local_idx = idx;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            argmax_better(smax[tid + stride], sidx[tid + stride],
                          smax[tid], sidx[tid]))
        {
            smax[tid] = smax[tid + stride];
            sidx[tid] = sidx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out_values[row] = smax[0];
        out_indices[row] = sidx[0];
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

        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, i, local_vals[j], local_idxs[j]))
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
            int best_idx = -1;
            int best_thread = -1;

            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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
        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, i, local_vals[j], local_idxs[j]))
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
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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
        const float threshold =
            llaminar2::sampling_math::uniform01(rng_seed, rng_offset);
        *out_token = llaminar2::sampling_math::sample_topk_topp_from_sorted_with_threshold(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            threshold,
            weights);
    }
}

template <int K_CAP>
__global__ void cuda_topk_smallk_partials_f32_kernel(
    const float *__restrict__ data,
    int n,
    int k,
    float *__restrict__ partial_values,
    int *__restrict__ partial_indices)
{
    if (k > K_CAP)
        return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int block_start =
        static_cast<int>((static_cast<long long>(blockIdx.x) * n) / gridDim.x);
    const int block_end =
        static_cast<int>((static_cast<long long>(blockIdx.x + 1) * n) / gridDim.x);

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = block_start + tid; i < block_end; i += num_threads)
    {
        const float val = data[i];
        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, i, local_vals[j], local_idxs[j]))
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
        int ptrs[TOPK_SMALL_K_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        const int out_base = blockIdx.x * k;
        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
                        best_thread = t;
                    }
                }
            }

            if (best_thread >= 0)
            {
                partial_values[out_base + out_i] = best_val;
                partial_indices[out_base + out_i] =
                    s_idxs[best_thread * k + ptrs[best_thread]];
                ++ptrs[best_thread];
            }
            else
            {
                partial_values[out_base + out_i] = -FLT_MAX;
                partial_indices[out_base + out_i] = -1;
            }
        }
    }
}

template <int K_CAP>
__global__ void cuda_topk_topp_distribution_from_partials_f32_kernel(
    const float *__restrict__ partial_values,
    const int *__restrict__ partial_indices,
    int partial_blocks,
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
    const int candidate_count = partial_blocks * k;

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < candidate_count; i += num_threads)
    {
        const int idx = partial_indices[i];
        if (idx < 0)
            continue;
        const float val = partial_values[i];
        if (local_count >= k &&
            !topkCandidateBetter(val, idx, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, idx, local_vals[j], local_idxs[j]))
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
        local_idxs[pos] = idx;
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
        int ptrs[TOPK_SMALL_K_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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

template <int K_CAP>
__global__ void cuda_topk_smallk_partials_batched_f32_kernel(
    const float *__restrict__ data,
    int n,
    int row_stride,
    int k,
    int partial_blocks,
    float *__restrict__ partial_values,
    int *__restrict__ partial_indices)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.y;
    const int partial = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const float *row_data = data + static_cast<size_t>(row) * row_stride;
    const int block_start =
        static_cast<int>((static_cast<long long>(partial) * n) / partial_blocks);
    const int block_end =
        static_cast<int>((static_cast<long long>(partial + 1) * n) / partial_blocks);

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = block_start + tid; i < block_end; i += num_threads)
    {
        const float val = row_data[i];
        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, i, local_vals[j], local_idxs[j]))
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
        int ptrs[TOPK_SMALL_K_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        const int out_base = (row * partial_blocks + partial) * k;
        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
                        best_thread = t;
                    }
                }
            }

            if (best_thread >= 0)
            {
                partial_values[out_base + out_i] = best_val;
                partial_indices[out_base + out_i] =
                    s_idxs[best_thread * k + ptrs[best_thread]];
                ++ptrs[best_thread];
            }
            else
            {
                partial_values[out_base + out_i] = -FLT_MAX;
                partial_indices[out_base + out_i] = -1;
            }
        }
    }
}

template <int K_CAP>
__global__ void cuda_topk_topp_distribution_batched_from_partials_f32_kernel(
    const float *__restrict__ partial_values,
    const int *__restrict__ partial_indices,
    int partial_blocks,
    int k,
    float top_p,
    float temperature,
    int *__restrict__ out_token_ids,
    int out_stride,
    float *__restrict__ out_probs)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int candidate_count = partial_blocks * k;
    const int partial_base = row * candidate_count;

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < candidate_count; i += num_threads)
    {
        const int idx = partial_indices[partial_base + i];
        if (idx < 0)
            continue;
        const float val = partial_values[partial_base + i];
        if (local_count >= k &&
            !topkCandidateBetter(val, idx, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, idx, local_vals[j], local_idxs[j]))
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
        local_idxs[pos] = idx;
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
        int ptrs[TOPK_SMALL_K_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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
        const int out_base = row * out_stride;
        llaminar2::sampling_math::build_topk_topp_distribution_from_sorted(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            out_token_ids + out_base,
            out_probs + out_base,
            weights);
    }
}

template <int K_CAP>
__device__ void cuda_write_active_topk_topp_logits_from_sorted(
    const float *__restrict__ sorted_logits,
    const int *__restrict__ sorted_token_ids,
    int k,
    float top_p,
    float temperature,
    float *__restrict__ out_logits,
    int vocab_size)
{
    if (threadIdx.x != 0)
        return;

    const float temp = temperature > 0.0f ? temperature : 1.0f;
    const float max_logit = sorted_logits[0];
    float weights[K_CAP];
    float total = 0.0f;
    for (int i = 0; i < k; ++i)
    {
        if (sorted_token_ids[i] < 0)
        {
            weights[i] = 0.0f;
            continue;
        }
        const float w = expf((sorted_logits[i] - max_logit) / temp);
        weights[i] = w;
        total += w;
    }

    int nucleus = k;
    if (total > 0.0f && top_p > 0.0f && top_p < 1.0f)
    {
        float cumulative = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            cumulative += weights[i] / total;
            if (cumulative >= top_p)
            {
                nucleus = i + 1;
                break;
            }
        }
    }

    for (int i = 0; i < nucleus; ++i)
    {
        const int token = sorted_token_ids[i];
        if (token >= 0 && token < vocab_size && weights[i] > 0.0f)
            out_logits[token] = sorted_logits[i] / temp;
    }
}

template <int K_CAP>
__global__ void cuda_topk_topp_processed_logits_batched_from_partials_f32_kernel(
    const float *__restrict__ partial_values,
    const int *__restrict__ partial_indices,
    int partial_blocks,
    int k,
    float top_p,
    float temperature,
    float *__restrict__ out_logits,
    int out_row_stride,
    int vocab_size)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int candidate_count = partial_blocks * k;
    const int partial_base = row * candidate_count;

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;
    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int i = tid; i < candidate_count; i += num_threads)
    {
        const int idx = partial_indices[partial_base + i];
        if (idx < 0)
            continue;
        const float val = partial_values[partial_base + i];
        if (local_count >= k &&
            !topkCandidateBetter(val, idx, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, idx, local_vals[j], local_idxs[j]))
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
        local_idxs[pos] = idx;
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

    float *row_out = out_logits + static_cast<size_t>(row) * out_row_stride;
    for (int token = tid; token < vocab_size; token += blockDim.x)
        row_out[token] = -INFINITY;
    __syncthreads();

    if (tid == 0)
    {
        float merged_vals[K_CAP];
        int merged_idxs[K_CAP];
        int ptrs[TOPK_SMALL_K_THREADS];
        for (int t = 0; t < num_threads; ++t)
            ptrs[t] = 0;

        for (int out_i = 0; out_i < k; ++out_i)
        {
            float best_val = -FLT_MAX;
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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

        cuda_write_active_topk_topp_logits_from_sorted<K_CAP>(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            row_out,
            vocab_size);
    }
}

template <int K_CAP>
__global__ void cuda_topk_topp_processed_logits_batched_f32_kernel(
    const float *__restrict__ data,
    int row_count,
    int vocab_size,
    int row_stride,
    int k,
    float top_p,
    float temperature,
    float *__restrict__ out_logits,
    int out_row_stride)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.x;
    if (row >= row_count)
        return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const float *row_data =
        data + static_cast<size_t>(row) * static_cast<size_t>(row_stride);
    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;
    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    for (int token = tid; token < vocab_size; token += num_threads)
    {
        const float val = row_data[token];
        if (local_count >= k &&
            !topkCandidateBetter(val, token, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, token, local_vals[j], local_idxs[j]))
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
        local_idxs[pos] = token;
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

    float *row_out = out_logits + static_cast<size_t>(row) * out_row_stride;
    for (int token = tid; token < vocab_size; token += blockDim.x)
        row_out[token] = -INFINITY;
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
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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

        cuda_write_active_topk_topp_logits_from_sorted<K_CAP>(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            row_out,
            vocab_size);
    }
}

template <int K_CAP>
__global__ void cuda_topk_topp_distribution_f32_kernel(
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
        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        int pos = (local_count < k) ? local_count : k - 1;
        for (int j = pos - 1; j >= 0; --j)
        {
            if (topkCandidateBetter(val, i, local_vals[j], local_idxs[j]))
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
            int best_idx = -1;
            int best_thread = -1;
            for (int t = 0; t < num_threads; ++t)
            {
                if (ptrs[t] < k)
                {
                    const float v = s_vals[t * k + ptrs[t]];
                    const int idx = s_idxs[t * k + ptrs[t]];
                    if (topkCandidateBetter(v, idx, best_val, best_idx))
                    {
                        best_val = v;
                        best_idx = idx;
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

__global__ void cuda_sample_distribution_f32_kernel(
    const int *__restrict__ token_ids,
    const float *__restrict__ probs,
    int k,
    float threshold,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    *out_token =
        llaminar2::sampling_math::sample_distribution_with_threshold_and_probability(
            token_ids, probs, k, threshold, out_probability);
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

__global__ void cuda_speculative_verify_distribution_thresholds_batch_kernel(
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

/**
 * @brief Batched verifier variant whose draft tokens are already on device.
 *
 * This is the first device-resident-token step toward vLLM-style speculative
 * sampling. Each row still receives host-generated thresholds as kernel
 * arguments, but the accepted draft-token sequence is read from arena scratch
 * produced by the draft sampler instead of from host scalar arguments.
 */
__global__ void cuda_speculative_verify_distribution_thresholds_batch_device_tokens_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int distribution_stride,
    const int *__restrict__ sampled_draft_tokens,
    const float *__restrict__ sampled_draft_probabilities,
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

    float accept_threshold = accept_threshold0;
    float residual_threshold = residual_threshold0;
    if (row == 1)
    {
        accept_threshold = accept_threshold1;
        residual_threshold = residual_threshold1;
    }
    else if (row == 2)
    {
        accept_threshold = accept_threshold2;
        residual_threshold = residual_threshold2;
    }
    else if (row == 3)
    {
        accept_threshold = accept_threshold3;
        residual_threshold = residual_threshold3;
    }

    const int offset = row * distribution_stride;
    llaminar2::sampling_math::speculative_verify_with_thresholds_and_draft_probability(
        target_token_ids + offset,
        target_probs + offset,
        draft_token_ids + offset,
        draft_probs + offset,
        k,
        sampled_draft_tokens[row],
        sampled_draft_probabilities ? sampled_draft_probabilities[row] : 0.0f,
        sampled_draft_probabilities != nullptr,
        accept_threshold,
        residual_threshold,
        out_token + row,
        out_accepted + row,
        out_accept_probability ? out_accept_probability + row : nullptr,
        out_accept_threshold ? out_accept_threshold + row : nullptr);
}

constexpr int PROCESSED_LOGIT_VERIFY_THREADS = 256;

__device__ __forceinline__ bool processedLogitActive(float value)
{
    return isfinite(value);
}

__device__ __forceinline__ bool processedLogitBetter(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index)
{
    if (candidate_index < 0)
        return false;
    if (current_index < 0)
        return true;
    return candidate_value > current_value ||
           (candidate_value == current_value &&
            candidate_index < current_index);
}

template <int THREADS>
__device__ void cuda_processed_logit_row_stats_block(
    const float *__restrict__ logits,
    int vocab_size,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    float *__restrict__ out_max,
    float *__restrict__ out_sum,
    int *__restrict__ out_argmax)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value = logits[token];
        if (processedLogitActive(value) &&
            processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const int argmax_token = scratch_idxs[0];
    float local_sum = 0.0f;
    if (argmax_token >= 0)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value = logits[token];
            if (processedLogitActive(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
    {
        *out_max = max_logit;
        *out_sum = scratch_vals[0];
        *out_argmax = argmax_token;
    }
    __syncthreads();
}

__device__ __forceinline__ float cuda_processed_logit_probability(
    const float *__restrict__ logits,
    int vocab_size,
    float max_logit,
    float exp_sum,
    int token)
{
    if (token < 0 || token >= vocab_size || !(exp_sum > 0.0f))
        return 0.0f;
    const float value = logits[token];
    if (!processedLogitActive(value))
        return 0.0f;
    return expf(value - max_logit) / exp_sum;
}

__device__ __forceinline__ float cuda_temperature_scaled_logit(
    const float *__restrict__ logits,
    int token,
    float inv_temperature)
{
    const float value = logits[token];
    return isfinite(value) ? value * inv_temperature : -FLT_MAX;
}

/**
 * @brief Build draft proposal probabilities and sample from them.
 *
 * This is the vLLM-style draft-side primitive: only temperature affects the
 * proposal distribution. Target rows remain responsible for top-k/top-p,
 * penalties, and residual correction, so this kernel intentionally avoids
 * compact top-k/top-p table construction.
 */
template <int THREADS>
__device__ void cuda_softmax_sample_temperature_logits_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float inv_temperature,
    float threshold,
    float *__restrict__ out_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    int *__restrict__ selected_token,
    float *__restrict__ selected_probability)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value =
            cuda_temperature_scaled_logit(logits, token, inv_temperature);
        if (processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const bool has_active = scratch_idxs[0] >= 0;
    float local_sum = 0.0f;
    if (has_active)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value =
                cuda_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    const float exp_sum = scratch_vals[0];
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);
    float chunk_mass = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        float probability = 0.0f;
        if (has_active && exp_sum > 0.0f)
        {
            const float value =
                cuda_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                probability = expf(value - max_logit) / exp_sum;
        }
        out_probabilities[token] = probability;
        chunk_mass += probability;
    }
    scratch_vals[tid] = chunk_mass;
    __syncthreads();

    if (tid == 0)
    {
        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = has_active ? scratch_idxs[0] : 0;
        float selected_prob = has_active ? out_probabilities[selected] : 0.0f;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + scratch_vals[i])
            {
                prefix += scratch_vals[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float probability = out_probabilities[token];
                if (!(probability > 0.0f))
                    continue;
                prefix += probability;
                if (target_mass <= prefix)
                {
                    selected = token;
                    selected_prob = probability;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
        *selected_probability = selected_prob;
    }
    __syncthreads();
}

/**
 * @brief Store temperature-scaled draft logits and sample from their softmax.
 *
 * This is the lower-memory vLLM-shaped draft primitive. It persists logits so
 * the verifier can recover q(token) from row-local logsumexp instead of reading
 * a full proposal probability matrix.
 */
template <int THREADS>
__device__ void cuda_scale_sample_temperature_logits_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float inv_temperature,
    float threshold,
    float *__restrict__ out_logits,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    int *__restrict__ selected_token,
    float *__restrict__ selected_probability)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value =
            cuda_temperature_scaled_logit(logits, token, inv_temperature);
        if (processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const bool has_active = scratch_idxs[0] >= 0;
    float local_sum = 0.0f;
    if (has_active)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value =
                cuda_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    const float exp_sum = scratch_vals[0];
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);
    float chunk_mass = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float value =
            cuda_temperature_scaled_logit(logits, token, inv_temperature);
        out_logits[token] = value;
        if (has_active && exp_sum > 0.0f && isfinite(value))
            chunk_mass += expf(value - max_logit) / exp_sum;
    }
    scratch_vals[tid] = chunk_mass;
    __syncthreads();

    if (tid == 0)
    {
        const float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = has_active ? scratch_idxs[0] : 0;
        float selected_prob = 0.0f;
        if (has_active && exp_sum > 0.0f)
            selected_prob = expf(out_logits[selected] - max_logit) / exp_sum;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + scratch_vals[i])
            {
                prefix += scratch_vals[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float value = out_logits[token];
                if (!isfinite(value) || !(exp_sum > 0.0f))
                    continue;
                const float probability = expf(value - max_logit) / exp_sum;
                if (!(probability > 0.0f))
                    continue;
                prefix += probability;
                if (target_mass <= prefix)
                {
                    selected = token;
                    selected_prob = probability;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
        *selected_probability = selected_prob;
    }
    __syncthreads();
}

template <int THREADS>
__device__ int cuda_sample_processed_logit_residual_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int vocab_size,
    float target_max,
    float target_sum,
    int target_argmax,
    float draft_max,
    float draft_sum,
    float threshold,
    float *__restrict__ chunk_sums,
    int *__restrict__ selected_token)
{
    const int tid = threadIdx.x;
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);

    float local_residual = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float p = cuda_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        if (p > 0.0f)
        {
            const float q = cuda_processed_logit_probability(
                draft_logits, vocab_size, draft_max, draft_sum, token);
            local_residual += fmaxf(0.0f, p - q);
        }
    }
    chunk_sums[tid] = local_residual;
    __syncthreads();

    if (tid == 0)
    {
        float total = 0.0f;
        for (int i = 0; i < THREADS; ++i)
            total += chunk_sums[i];
        const bool use_residual = total > 0.0f;
        if (!use_residual)
        {
            total = target_sum;
            for (int i = 0; i < THREADS; ++i)
                chunk_sums[i] = 0.0f;
        }

        if (!use_residual)
        {
            for (int i = 0; i < THREADS; ++i)
            {
                const int b = i * chunk;
                const int e = min(vocab_size, b + chunk);
                float sum = 0.0f;
                for (int token = b; token < e; ++token)
                {
                    const float value = target_logits[token];
                    if (processedLogitActive(value))
                        sum += expf(value - target_max);
                }
                chunk_sums[i] = sum;
            }
        }

        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold) * total;
        float prefix = 0.0f;
        int selected = target_argmax;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + chunk_sums[i])
            {
                prefix += chunk_sums[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                float weight = 0.0f;
                if (use_residual)
                {
                    const float p = cuda_processed_logit_probability(
                        target_logits, vocab_size, target_max, target_sum, token);
                    if (p > 0.0f)
                    {
                        const float q = cuda_processed_logit_probability(
                            draft_logits, vocab_size, draft_max, draft_sum, token);
                        weight = fmaxf(0.0f, p - q);
                    }
                }
                else
                {
                    const float value = target_logits[token];
                    if (processedLogitActive(value))
                        weight = expf(value - target_max);
                }
                if (!(weight > 0.0f))
                    continue;
                prefix += weight;
                if (target_mass <= prefix)
                {
                    selected = token;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
    }
    __syncthreads();
    return *selected_token;
}

template <int THREADS>
__device__ int cuda_sample_processed_logit_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float max_logit,
    float exp_sum,
    int argmax_token,
    float threshold,
    float *__restrict__ chunk_sums,
    int *__restrict__ selected_token)
{
    const int tid = threadIdx.x;
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);

    float local_sum = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float value = logits[token];
        if (processedLogitActive(value))
            local_sum += expf(value - max_logit);
    }
    chunk_sums[tid] = local_sum;
    __syncthreads();

    if (tid == 0)
    {
        const float total = exp_sum;
        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold) * total;
        float prefix = 0.0f;
        int selected = argmax_token;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + chunk_sums[i])
            {
                prefix += chunk_sums[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float value = logits[token];
                if (!processedLogitActive(value))
                    continue;
                prefix += expf(value - max_logit);
                if (target_mass <= prefix)
                {
                    selected = token;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
    }
    __syncthreads();
    return *selected_token;
}

template <int THREADS>
__device__ int cuda_sample_recovered_probability_row_block(
    const float *__restrict__ target_probs,
    const float *__restrict__ draft_probs,
    const float *__restrict__ inverse_rejection_samples,
    int vocab_size,
    int draft_token,
    bool no_draft_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    const int tid = threadIdx.x;
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        float probability = target_probs[token];
        if (!isfinite(probability) || !(probability > 0.0f))
            probability = 0.0f;

        if (no_draft_probabilities)
        {
            if (token == draft_token)
                probability = 0.0f;
        }
        else
        {
            float draft_probability = draft_probs[token];
            if (!isfinite(draft_probability) || !(draft_probability > 0.0f))
                draft_probability = 0.0f;
            probability = fmaxf(0.0f, probability - draft_probability);
        }

        const float inverse_sample = inverse_rejection_samples[token];
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

/**
 * @brief Sample one processed full-logit row on an explicit stream.
 *
 * This is used for the vLLM-style all-accepted bonus token without creating a
 * compact top-k/top-p table.
 */
__global__ void cuda_sample_processed_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float threshold,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;
    __shared__ int selected_token;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &max_logit,
        &exp_sum,
        &argmax_token);

    cuda_sample_processed_logit_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        max_logit,
        exp_sum,
        argmax_token,
        threshold,
        scratch_vals,
        &selected_token);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
        {
            *out_probability = cuda_processed_logit_probability(
                logits,
                vocab_size,
                max_logit,
                exp_sum,
                selected_token);
        }
    }
}

__global__ void cuda_softmax_sample_temperature_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *__restrict__ out_probabilities,
    int out_row_stride,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    (void)out_row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int selected_token;
    __shared__ float selected_probability;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const float safe_temperature =
        (isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
    cuda_softmax_sample_temperature_logits_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        1.0f / safe_temperature,
        threshold,
        out_probabilities,
        scratch_vals,
        scratch_idxs,
        &selected_token,
        &selected_probability);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
            *out_probability = selected_probability;
    }
}

__global__ void cuda_scale_sample_temperature_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *__restrict__ out_logits,
    int out_row_stride,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    (void)out_row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int selected_token;
    __shared__ float selected_probability;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const float safe_temperature =
        (isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
    cuda_scale_sample_temperature_logits_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        1.0f / safe_temperature,
        threshold,
        out_logits,
        scratch_vals,
        scratch_idxs,
        &selected_token,
        &selected_probability);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
            *out_probability = selected_probability;
    }
}

__global__ void cuda_softmax_processed_logits_f32_kernel(
    const float *__restrict__ logits,
    int row_count,
    int vocab_size,
    int row_stride,
    float *__restrict__ out_probabilities,
    int out_row_stride)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;

    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int tid = threadIdx.x;
    const float *row_logits =
        logits + static_cast<size_t>(blockIdx.x) * row_stride;
    float *row_out =
        out_probabilities + static_cast<size_t>(blockIdx.x) * out_row_stride;

    cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        row_logits,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &max_logit,
        &exp_sum,
        &argmax_token);

    for (int token = tid; token < vocab_size; token += blockDim.x)
    {
        row_out[token] =
            cuda_processed_logit_probability(
                row_logits,
                vocab_size,
                max_logit,
                exp_sum,
            token);
    }
}

/**
 * @brief Materialize vLLM-style inverse-exponential rejection samples.
 *
 * The output matrix is owned by BufferArena. This kernel only fills it on the
 * caller's explicit stream, using a deterministic seed domain so stochastic
 * verifier replay is independent of capture timing.
 */
__global__ void cuda_fill_inverse_exponential_samples_f32_kernel(
    float *__restrict__ out_samples,
    int row_count,
    int vocab_size,
    int row_stride,
    unsigned long long seed,
    int first_logical_position)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int row = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || token >= vocab_size)
        return;

    const int signed_position = first_logical_position + row;
    const unsigned long long logical_position =
        static_cast<unsigned long long>(signed_position > 0 ? signed_position : 0);
    const unsigned long long offset =
        logical_position * static_cast<unsigned long long>(vocab_size) +
        static_cast<unsigned long long>(token);
    const float uniform =
        llaminar2::sampling_math::uniform01(seed ^ kInverseSampleDomain, offset);
    out_samples[static_cast<size_t>(row) * row_stride + token] =
        llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
}

/**
 * @brief Verify stochastic MTP rows directly from processed full logits.
 *
 * This is the backend kernel counterpart to MTPRejectionSampler's CPU
 * processed-logit reference. The target production design uses row/block
 * softmax stats instead of compact top-k tables, then samples a residual row
 * only when the draft token fails the p/q acceptance test.
 */
__global__ void cuda_speculative_verify_processed_logits_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    float residual_threshold0,
    float residual_threshold1,
    float residual_threshold2,
    float residual_threshold3,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold,
    const float *__restrict__ draft_token_probabilities)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ float draft_max;
    __shared__ float draft_sum;
    __shared__ int target_argmax;
    __shared__ int draft_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        draft_logits + static_cast<size_t>(row) * draft_row_stride;
    cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);
    const bool has_draft_token_probability =
        draft_token_probabilities != nullptr;
    if (!has_draft_token_probability)
    {
        cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            draft_row,
            vocab_size,
            scratch_vals,
            scratch_idxs,
            &draft_max,
            &draft_sum,
            &draft_argmax);
    }

    float accept_threshold = accept_threshold0;
    float residual_threshold = residual_threshold0;
    if (row == 1)
    {
        accept_threshold = accept_threshold1;
        residual_threshold = residual_threshold1;
    }
    else if (row == 2)
    {
        accept_threshold = accept_threshold2;
        residual_threshold = residual_threshold2;
    }
    else if (row == 3)
    {
        accept_threshold = accept_threshold3;
        residual_threshold = residual_threshold3;
    }

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
            const int draft_token = sampled_draft_tokens[row];
            const float p = cuda_processed_logit_probability(
                target_row, vocab_size, target_max, target_sum, draft_token);
            const float q =
                has_draft_token_probability
                    ? draft_token_probabilities[row]
                    : cuda_processed_logit_probability(
                          draft_row, vocab_size, draft_max, draft_sum, draft_token);
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold < accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        if (has_draft_token_probability)
        {
            cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                draft_row,
                vocab_size,
                scratch_vals,
                scratch_idxs,
                &draft_max,
                &draft_sum,
                &draft_argmax);
        }
        cuda_sample_processed_logit_residual_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            target_row,
            draft_row,
            vocab_size,
            target_max,
            target_sum,
            target_argmax,
            draft_max,
            draft_sum,
            residual_threshold,
            scratch_vals,
            &row_token);
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Verify rows from processed target logits and draft proposal probabilities.
 *
 * This is the lower-memory vLLM-shaped stochastic verifier. Target
 * probabilities are computed from processed logits inside the row block, while
 * draft proposal probabilities are read from the proposal rows captured during
 * MTP draft sampling. Recovered-token sampling uses the same inverse
 * exponential noise as the full-probability verifier, generated on the fly so
 * no full inverse-random matrix is materialized.
 */
template <int THREADS>
__device__ int cuda_sample_recovered_processed_target_draft_probability_row_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_probs,
    int vocab_size,
    float target_max,
    float target_sum,
    int draft_token,
    unsigned long long inverse_sample_seed,
    int logical_position,
    bool no_draft_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int tid = threadIdx.x;
    const unsigned long long safe_position =
        static_cast<unsigned long long>(logical_position > 0 ? logical_position : 0);
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float p = cuda_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        float q = no_draft_probabilities
                      ? (token == draft_token ? 1.0f : 0.0f)
                      : draft_probs[token];
        if (!isfinite(q) || !(q > 0.0f))
            q = 0.0f;
        float probability = fmaxf(0.0f, p - q);

        const unsigned long long offset =
            safe_position * static_cast<unsigned long long>(vocab_size) +
            static_cast<unsigned long long>(token);
        const float uniform =
            llaminar2::sampling_math::uniform01(
                inverse_sample_seed ^ kInverseSampleDomain,
                offset);
        const float inverse_sample =
            llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

template <int THREADS>
__device__ int cuda_sample_recovered_processed_target_draft_logit_row_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int vocab_size,
    float target_max,
    float target_sum,
    float draft_max,
    float draft_sum,
    int draft_token,
    unsigned long long inverse_sample_seed,
    int logical_position,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int tid = threadIdx.x;
    const unsigned long long safe_position =
        static_cast<unsigned long long>(logical_position > 0 ? logical_position : 0);
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float p = cuda_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        const float q = cuda_processed_logit_probability(
            draft_logits, vocab_size, draft_max, draft_sum, token);
        const float probability = fmaxf(0.0f, p - q);

        const unsigned long long offset =
            safe_position * static_cast<unsigned long long>(vocab_size) +
            static_cast<unsigned long long>(token);
        const float uniform =
            llaminar2::sampling_math::uniform01(
                inverse_sample_seed ^ kInverseSampleDomain,
                offset);
        const float inverse_sample =
            llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

__global__ void cuda_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_probabilities,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold,
    int no_draft_probabilities)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ int target_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        no_draft_probabilities
            ? nullptr
            : draft_probabilities + static_cast<size_t>(row) * draft_row_stride;

    cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);

    float accept_threshold = accept_threshold0;
    if (row == 1)
        accept_threshold = accept_threshold1;
    else if (row == 2)
        accept_threshold = accept_threshold2;
    else if (row == 3)
        accept_threshold = accept_threshold3;

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        const float p = cuda_processed_logit_probability(
            target_row, vocab_size, target_max, target_sum, draft_token);
        float q = no_draft_probabilities ? 1.0f : 0.0f;
        if (draft_token >= 0 && draft_token < vocab_size)
        {
            if (!no_draft_probabilities)
            {
                q = draft_row[draft_token];
                if (!isfinite(q) || !(q > 0.0f))
                    q = 0.0f;
            }
        }
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            cuda_sample_recovered_processed_target_draft_probability_row_block<
                PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                vocab_size,
                target_max,
                target_sum,
                draft_token,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row,
                no_draft_probabilities != 0,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

__global__ void cuda_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    const float *__restrict__ sampled_draft_probabilities,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ float draft_max;
    __shared__ float draft_sum;
    __shared__ int target_argmax;
    __shared__ int draft_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        draft_logits + static_cast<size_t>(row) * draft_row_stride;
    const bool has_sampled_draft_probability =
        sampled_draft_probabilities != nullptr;

    cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);

    if (!has_sampled_draft_probability)
    {
        cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            draft_row,
            vocab_size,
            scratch_vals,
            scratch_idxs,
            &draft_max,
            &draft_sum,
            &draft_argmax);
    }

    float accept_threshold = accept_threshold0;
    if (row == 1)
        accept_threshold = accept_threshold1;
    else if (row == 2)
        accept_threshold = accept_threshold2;
    else if (row == 3)
        accept_threshold = accept_threshold3;

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        const float p = cuda_processed_logit_probability(
            target_row, vocab_size, target_max, target_sum, draft_token);
        float q = 0.0f;
        if (has_sampled_draft_probability)
        {
            q = sampled_draft_probabilities[row];
            if (!isfinite(q) || !(q > 0.0f))
                q = 0.0f;
        }
        else
        {
            q = cuda_processed_logit_probability(
                draft_row, vocab_size, draft_max, draft_sum, draft_token);
        }
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        if (has_sampled_draft_probability)
        {
            cuda_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                draft_row,
                vocab_size,
                scratch_vals,
                scratch_idxs,
                &draft_max,
                &draft_sum,
                &draft_argmax);
        }
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            cuda_sample_recovered_processed_target_draft_logit_row_block<
                PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                vocab_size,
                target_max,
                target_sum,
                draft_max,
                draft_sum,
                draft_token,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Verify stochastic MTP rows from full target/draft probability rows.
 *
 * This is the direct vLLM-style rejection primitive. Draft acceptance reads
 * only `p(draft)` and `q(draft)`. On rejection, the recovered token is selected
 * with a full-vocab parallel reduction over `max(p - q, 0) * inv_q[token]`.
 */
__global__ void cuda_speculative_verify_probabilities_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_probabilities,
    const float *__restrict__ draft_probabilities,
    const float *__restrict__ inverse_rejection_samples,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    int inverse_sample_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    float accept_threshold0,
    float accept_threshold1,
    float accept_threshold2,
    float accept_threshold3,
    int no_draft_probabilities,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_probabilities + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        no_draft_probabilities
            ? nullptr
            : draft_probabilities + static_cast<size_t>(row) * draft_row_stride;
    const float *inverse_row =
        inverse_rejection_samples +
        static_cast<size_t>(row) * inverse_sample_row_stride;

    float accept_threshold = accept_threshold0;
    if (row == 1)
        accept_threshold = accept_threshold1;
    else if (row == 2)
        accept_threshold = accept_threshold2;
    else if (row == 3)
        accept_threshold = accept_threshold3;

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        float p = 0.0f;
        float q = no_draft_probabilities ? 1.0f : 0.0f;
        if (draft_token >= 0 && draft_token < vocab_size)
        {
            p = target_row[draft_token];
            if (!isfinite(p) || !(p > 0.0f))
                p = 0.0f;
            if (!no_draft_probabilities)
            {
                q = draft_row[draft_token];
                if (!isfinite(q) || !(q > 0.0f))
                    q = 0.0f;
            }
        }

        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            cuda_sample_recovered_probability_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                inverse_row,
                vocab_size,
                draft_token,
                no_draft_probabilities != 0,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Reduce row-wise stochastic verifier results into one speculative commit plan.
 *
 * The kernel is deliberately one thread: MTP currently verifies at most four
 * rows, and making the reduction serial keeps the stop/rejection semantics
 * easy to audit while remaining graph-capturable.
 */
__global__ void cuda_summarize_speculative_verify_batch_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    int *__restrict__ out_tokens,
    int *__restrict__ out_meta)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    const int ready_token =
        has_bonus_token && bonus_token ? *bonus_token : -1;
    llaminar2::sampling_math::summarize_speculative_verify_batch(
        first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        out_tokens,
        out_meta);
}

/**
 * @brief Device-first-token variant of the speculative batch reducer.
 *
 * The first sampled main-model token is produced by an earlier sampler kernel.
 * Reading it here avoids a CPU round trip before MTP verifier summarization.
 * The actual accept/reject semantics remain in SamplingMath so CUDA, ROCm, and
 * CPU tests share one source of truth.
 */
__global__ void cuda_summarize_speculative_verify_batch_device_first_token_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    const int *__restrict__ first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    int *__restrict__ out_tokens,
    int *__restrict__ out_meta)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    const int ready_token =
        has_bonus_token && bonus_token ? *bonus_token : -1;
    const int sampled_first_token = first_token ? *first_token : -1;
    llaminar2::sampling_math::summarize_speculative_verify_batch(
        sampled_first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        out_tokens,
        out_meta);
}

/**
 * @brief Reduce greedy verifier argmax rows into one speculative commit plan.
 *
 * The verifier argmax rows and compact verifier input tokens are both
 * device-resident. This keeps greedy MTP on the same graph-capturable compact
 * summary path as stochastic MTP: compare rows on GPU, then copy only the
 * small summary buffers back to the host.
 */
__global__ void cuda_summarize_greedy_speculative_verify_batch_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ draft_tokens,
    int compare_row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *__restrict__ out_tokens,
    int *__restrict__ out_meta)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    llaminar2::sampling_math::summarize_greedy_speculative_verify_batch(
        first_token,
        verify_tokens,
        draft_tokens,
        compare_row_count,
        stop_tokens,
        stop_token_count,
        out_tokens,
        out_meta);
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

    bool cudaOps_argmax_f32_batched_rows(
        const float *data,
        int rows,
        int cols,
        int row_stride,
        float *out_values,
        int *out_indices,
        float *partial_vals,
        int *partial_idxs,
        int partial_capacity,
        int device_idx,
        void *stream)
    {
        if (rows <= 0 || cols <= 0 || row_stride < cols ||
            !data || !out_values || !out_indices ||
            !partial_vals || !partial_idxs || partial_capacity < rows)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        const int row_partial_capacity = partial_capacity / rows;
        if (row_partial_capacity < 1)
            return false;

        const int threads = ARGMAX_REDUCE_THREADS;
        const long per_block = static_cast<long>(threads) * ARGMAX_ELEMS_PER_THREAD;
        long blocks = (static_cast<long>(cols) + per_block - 1) / per_block;
        if (blocks < 1)
            blocks = 1;
        if (blocks > row_partial_capacity)
            blocks = row_partial_capacity;
        const int num_blocks = static_cast<int>(blocks);

        const size_t smem1 = threads * (sizeof(float) + sizeof(int));
        cuda_argmax_partial_f32_batched_rows_kernel<<<dim3(num_blocks, rows), threads, smem1, s>>>(
            data,
            rows,
            cols,
            row_stride,
            partial_vals,
            partial_idxs,
            row_partial_capacity);

        const int fthreads = ARGMAX_FINALIZE_THREADS;
        const size_t smem2 = fthreads * (sizeof(float) + sizeof(int));
        cuda_argmax_finalize_f32_batched_rows_kernel<<<rows, fthreads, smem2, s>>>(
            partial_vals,
            partial_idxs,
            num_blocks,
            row_partial_capacity,
            out_values,
            out_indices);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Argmax FP32 batched rows launch failed: %s\n",
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
            data,
            n,
            k,
            top_p,
            temperature,
            rng_seed,
            rng_offset,
            out_token);

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
        float *scratch_values,
        int *scratch_indices,
        int scratch_capacity,
        int device_idx,
        void *stream)
    {
        if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_token_ids || !out_probs || !stream)
            return false;

        if (k > n)
            k = n;

        cudaSetDevice(device_idx);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        if (k <= TOPK_SMALL_K_CAP &&
            scratch_values &&
            scratch_indices)
        {
            const int threads = TOPK_SMALL_K_THREADS;
            int partial_blocks = (n + threads - 1) / threads;
            if (partial_blocks < 1)
                partial_blocks = 1;
            const int partial_block_cap = topkSmallKPartialBlockCap(k);
            if (partial_blocks > partial_block_cap)
                partial_blocks = partial_block_cap;
            const int required_scratch = partial_blocks * k;
            if (scratch_capacity >= required_scratch)
            {
                const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));
                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_smallk_partials_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<partial_blocks, threads, smem_size, s>>>(
                            data, n, k, scratch_values, scratch_indices);
                }
                else
                {
                    cuda_topk_smallk_partials_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_blocks, threads, smem_size, s>>>(
                            data, n, k, scratch_values, scratch_indices);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA small-k Top-K partial FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }

                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_topp_distribution_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<1, threads, smem_size, s>>>(
                            scratch_values,
                            scratch_indices,
                            partial_blocks,
                            k,
                            top_p,
                            temperature,
                            out_token_ids,
                            out_probs);
                }
                else
                {
                    cuda_topk_topp_distribution_from_partials_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<1, threads, smem_size, s>>>(
                            scratch_values,
                            scratch_indices,
                            partial_blocks,
                            k,
                            top_p,
                            temperature,
                            out_token_ids,
                            out_probs);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA small-k Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }
                return true;
            }
        }

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

        if (k <= TOPK_MEDIUM_K_CAP)
        {
            cuda_topk_topp_distribution_f32_kernel<TOPK_MEDIUM_K_CAP><<<1, threads, smem_size, s>>>(
                data, n, k, top_p, temperature, out_token_ids, out_probs);
        }
        else if (k <= TOPK_SMALL_K_CAP)
        {
            cuda_topk_topp_distribution_f32_kernel<TOPK_SMALL_K_CAP><<<1, threads, smem_size, s>>>(
                data, n, k, top_p, temperature, out_token_ids, out_probs);
        }
        else
        {
            cuda_topk_topp_distribution_f32_kernel<TOPK_MAX_K><<<1, threads, smem_size, s>>>(
                data, n, k, top_p, temperature, out_token_ids, out_probs);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_topk_topp_distributions_f32(
        const float *data,
        int row_count,
        int n,
        int row_stride,
        int k,
        float top_p,
        float temperature,
        int *out_token_ids,
        int out_stride,
        float *out_probs,
        float *scratch_values,
        int *scratch_indices,
        int scratch_capacity,
        int device_idx,
        void *stream)
    {
        if (row_count <= 0 || n <= 0 || row_stride < n ||
            k <= 0 || k > TOPK_MAX_K || out_stride < k ||
            !data || !out_token_ids || !out_probs || !stream)
        {
            return false;
        }

        if (k > n)
            k = n;

        cudaSetDevice(device_idx);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        if (k <= TOPK_SMALL_K_CAP && scratch_values && scratch_indices)
        {
            const int threads = TOPK_SMALL_K_THREADS;
            int partial_blocks = (n + threads - 1) / threads;
            if (partial_blocks < 1)
                partial_blocks = 1;
            const int partial_block_cap = topkSmallKPartialBlockCap(k);
            if (partial_blocks > partial_block_cap)
                partial_blocks = partial_block_cap;
            const int required_scratch = row_count * partial_blocks * k;
            if (scratch_capacity >= required_scratch)
            {
                const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));
                const dim3 partial_grid(partial_blocks, row_count);
                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices);
                }
                else
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA batched small-k Top-K partial FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }

                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride, out_probs);
                }
                else
                {
                    cuda_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride, out_probs);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA batched small-k Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }
                return true;
            }
        }

        return false;
    }

    bool cudaOps_topk_topp_processed_logits_f32(
        const float *data,
        int row_count,
        int n,
        int row_stride,
        int k,
        float top_p,
        float temperature,
        float *out_logits,
        int out_row_stride,
        float *scratch_values,
        int *scratch_indices,
        int scratch_capacity,
        int device_idx,
        void *stream)
    {
        if (row_count <= 0 || n <= 0 || row_stride < n ||
            k <= 0 || k > TOPK_MAX_K || out_row_stride < n ||
            !data || !out_logits || !stream)
        {
            return false;
        }

        if (k > n)
            k = n;

        cudaSetDevice(device_idx);
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        if (k <= TOPK_SMALL_K_CAP && scratch_values && scratch_indices)
        {
            const int threads = TOPK_SMALL_K_THREADS;
            int partial_blocks = (n + threads - 1) / threads;
            if (partial_blocks < 1)
                partial_blocks = 1;
            const int partial_block_cap = topkSmallKPartialBlockCap(k);
            if (partial_blocks > partial_block_cap)
                partial_blocks = partial_block_cap;
            const int required_scratch = row_count * partial_blocks * k;
            if (scratch_capacity >= required_scratch)
            {
                const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));
                const dim3 partial_grid(partial_blocks, row_count);
                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices);
                }
                else
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA processed-logit Top-K partial FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }

                if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_topp_processed_logits_batched_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_logits, out_row_stride, n);
                }
                else
                {
                    cuda_topk_topp_processed_logits_batched_from_partials_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_logits, out_row_stride, n);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA processed-logit Top-K/Top-P FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }
                return true;
            }
        }

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));
        if (k <= TOPK_MEDIUM_K_CAP)
        {
            cuda_topk_topp_processed_logits_batched_f32_kernel<TOPK_MEDIUM_K_CAP>
                <<<row_count, threads, smem_size, s>>>(
                    data, row_count, n, row_stride, k, top_p, temperature,
                    out_logits, out_row_stride);
        }
        else if (k <= TOPK_SMALL_K_CAP)
        {
            cuda_topk_topp_processed_logits_batched_f32_kernel<TOPK_SMALL_K_CAP>
                <<<row_count, threads, smem_size, s>>>(
                    data, row_count, n, row_stride, k, top_p, temperature,
                    out_logits, out_row_stride);
        }
        else
        {
            cuda_topk_topp_processed_logits_batched_f32_kernel<TOPK_MAX_K>
                <<<row_count, threads, smem_size, s>>>(
                    data, row_count, n, row_stride, k, top_p, temperature,
                    out_logits, out_row_stride);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA direct processed-logit Top-K/Top-P FP32 kernel launch failed: %s\n",
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
        float *out_probability,
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
            out_token,
            out_probability);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Sample Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_sample_processed_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream)
    {
        if (!logits || !out_token || !stream ||
            vocab_size <= 0 || row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_sample_processed_logits_f32_kernel<<<
            1,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            logits,
            vocab_size,
            row_stride,
            threshold,
            out_token,
            out_probability);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Processed-Logit Sample kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_softmax_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_probabilities,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream)
    {
        if (!logits || !out_probabilities || !out_token || !stream ||
            vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_softmax_sample_temperature_logits_f32_kernel<<<
            1,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            logits,
            vocab_size,
            row_stride,
            temperature,
            threshold,
            out_probabilities,
            out_row_stride,
            out_token,
            out_probability);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Temperature Draft Proposal kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_scale_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_logits,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream)
    {
        if (!logits || !out_logits || !out_token || !stream ||
            vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_scale_sample_temperature_logits_f32_kernel<<<
            1,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            logits,
            vocab_size,
            row_stride,
            temperature,
            threshold,
            out_logits,
            out_row_stride,
            out_token,
            out_probability);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Temperature Draft Logit Proposal kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_softmax_processed_logits_f32(
        const float *logits,
        int row_count,
        int vocab_size,
        int row_stride,
        float *out_probabilities,
        int out_row_stride,
        int device_idx,
        void *stream)
    {
        if (!logits || !out_probabilities || !stream ||
            row_count <= 0 || vocab_size <= 0 ||
            row_stride < vocab_size || out_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_softmax_processed_logits_f32_kernel<<<
            row_count,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            logits,
            row_count,
            vocab_size,
            row_stride,
            out_probabilities,
            out_row_stride);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Processed-Logit Softmax kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_fill_inverse_exponential_samples_f32(
        float *out_samples,
        int row_count,
        int vocab_size,
        int row_stride,
        unsigned long long seed,
        int first_logical_position,
        int device_idx,
        void *stream)
    {
        if (!out_samples || !stream ||
            row_count <= 0 || row_count > 4 ||
            vocab_size <= 0 || row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        const dim3 grid((vocab_size + threads - 1) / threads, row_count);
        cuda_fill_inverse_exponential_samples_f32_kernel<<<
            grid,
            threads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            out_samples,
            row_count,
            vocab_size,
            row_stride,
            seed,
            first_logical_position);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA inverse-exponential sample fill launch failed: %s\n",
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

    bool cudaOps_speculative_verify_distribution_thresholds_batch_f32(
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

        cudaSetDevice(device_idx);

        cuda_speculative_verify_distribution_thresholds_batch_kernel<<<1, 4, 0, static_cast<cudaStream_t>(stream)>>>(
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

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Batch FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
        const int *target_token_ids,
        const float *target_probs,
        const int *draft_token_ids,
        const float *draft_probs,
        int k,
        int distribution_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
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
            !sampled_draft_tokens ||
            !out_token || !out_accepted || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);

        cuda_speculative_verify_distribution_thresholds_batch_device_tokens_kernel<<<1, 4, 0, static_cast<cudaStream_t>(stream)>>>(
            target_token_ids,
            target_probs,
            draft_token_ids,
            draft_probs,
            k,
            distribution_stride,
            sampled_draft_tokens,
            sampled_draft_probabilities,
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

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Batch Device Tokens FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        float accept_threshold0,
        float accept_threshold1,
        float accept_threshold2,
        float accept_threshold3,
        float residual_threshold0,
        float residual_threshold1,
        float residual_threshold2,
        float residual_threshold3,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        const float *draft_token_probabilities,
        int device_idx,
        void *stream)
    {
        if (!target_logits || !draft_logits || !sampled_draft_tokens ||
            !out_token || !out_accepted || !stream ||
            row_count <= 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_speculative_verify_processed_logits_thresholds_batch_device_tokens_kernel<<<
            row_count,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            target_logits,
            draft_logits,
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens,
            accept_threshold0,
            accept_threshold1,
            accept_threshold2,
            accept_threshold3,
            residual_threshold0,
            residual_threshold1,
            residual_threshold2,
            residual_threshold3,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold,
            draft_token_probabilities);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Processed-Logit Stochastic Speculative Verify kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_probabilities,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        float accept_threshold0,
        float accept_threshold1,
        float accept_threshold2,
        float accept_threshold3,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int no_draft_probabilities,
        int device_idx,
        void *stream)
    {
        if (!target_logits ||
            (!no_draft_probabilities && !draft_probabilities) ||
            !sampled_draft_tokens ||
            !out_token || !out_accepted || !stream ||
            row_count <= 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_kernel<<<
            row_count,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            target_logits,
            draft_probabilities,
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens,
            accept_threshold0,
            accept_threshold1,
            accept_threshold2,
            accept_threshold3,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold,
            no_draft_probabilities);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Processed-Target/Draft-Probability Stochastic Verify kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
        float accept_threshold0,
        float accept_threshold1,
        float accept_threshold2,
        float accept_threshold3,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream)
    {
        if (!target_logits || !draft_logits || !sampled_draft_tokens ||
            !out_token || !out_accepted || !stream ||
            row_count <= 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_kernel<<<
            row_count,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            target_logits,
            draft_logits,
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens,
            sampled_draft_probabilities,
            accept_threshold0,
            accept_threshold1,
            accept_threshold2,
            accept_threshold3,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Processed-Target/Draft-Logit Stochastic Verify kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_probabilities,
        const float *draft_probabilities,
        const float *inverse_rejection_samples,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        int inverse_sample_row_stride,
        const int *sampled_draft_tokens,
        float accept_threshold0,
        float accept_threshold1,
        float accept_threshold2,
        float accept_threshold3,
        int no_draft_probabilities,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream)
    {
        if (!target_probabilities || !inverse_rejection_samples ||
            (!no_draft_probabilities && !draft_probabilities) ||
            !sampled_draft_tokens || !out_token || !out_accepted || !stream ||
            row_count <= 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            inverse_sample_row_stride < vocab_size)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_speculative_verify_probabilities_thresholds_batch_device_tokens_kernel<<<
            row_count,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            target_probabilities,
            draft_probabilities,
            inverse_rejection_samples,
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            inverse_sample_row_stride,
            sampled_draft_tokens,
            accept_threshold0,
            accept_threshold1,
            accept_threshold2,
            accept_threshold3,
            no_draft_probabilities,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Full-Probability Stochastic Speculative Verify kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_summarize_speculative_verify_batch(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int *out_meta,
        int device_idx,
        void *stream)
    {
        if (row_count < 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count >
                llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
            !verify_tokens || !verify_accepted ||
            (has_bonus_token && !bonus_token) ||
            !out_tokens || !out_meta || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_summarize_speculative_verify_batch_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            verify_tokens,
            verify_accepted,
            row_count,
            first_token,
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7,
            stop_token_count,
            bonus_token,
            has_bonus_token,
            out_tokens,
            out_meta);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Batch Summary kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_summarize_speculative_verify_batch_device_first_token(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        const int *first_token,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int *out_meta,
        int device_idx,
        void *stream)
    {
        if (row_count < 0 ||
            row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count >
                llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
            !verify_tokens || !verify_accepted || !first_token ||
            (has_bonus_token && !bonus_token) ||
            !out_tokens || !out_meta || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_summarize_speculative_verify_batch_device_first_token_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            verify_tokens,
            verify_accepted,
            row_count,
            first_token,
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7,
            stop_token_count,
            bonus_token,
            has_bonus_token,
            out_tokens,
            out_meta);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Batch Device-First Summary kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_summarize_greedy_speculative_verify_batch(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        int first_token,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        int *out_tokens,
        int *out_meta,
        int device_idx,
        void *stream)
    {
        if (compare_row_count < 0 ||
            compare_row_count > llaminar2::sampling_math::kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count >
                llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
            !verify_tokens || !draft_tokens ||
            !out_tokens || !out_meta || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_summarize_greedy_speculative_verify_batch_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            verify_tokens,
            draft_tokens,
            compare_row_count,
            first_token,
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7,
            stop_token_count,
            out_tokens,
            out_meta);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Greedy Speculative Verify Batch Summary kernel launch failed: %s\n",
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
