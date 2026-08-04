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
 *   - Top-K: deterministic value/index ordering. Qwen-sized batched rows use
 *     a two-stage cooperative register selector; larger geometries retain a
 *     total generic partial-list implementation.
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include "../../common/SamplingMath.h"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"

// Maximum k supported by the top-k kernel
constexpr int TOPK_MAX_K = 256;
constexpr int TOPK_THREADS = 32;
// Qwen chat defaults use top_k=40. A 40-entry specialization keeps register
// pressure aligned with the actual compact table size used by stochastic MTP.
constexpr int TOPK_MEDIUM_K_CAP = 40;
constexpr int TOPK_SMALL_K_CAP = 64;
constexpr int TOPK_SMALL_K_PARTIAL_BLOCKS = 128;
// Qwen-style top_k=40/64 distributions expose enough independent partials to
// cover one full GA102 while keeping each segment within the cooperative
// selector's fixed register capacity.
constexpr int TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS = 64;
constexpr int TOPK_SMALL_K_THREADS = 64;
constexpr int TOPK_BATCHED_COOPERATIVE_THREADS = 256;
constexpr int TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD = 16;
constexpr int TOPK_BATCHED_COOPERATIVE_CAPACITY =
    TOPK_BATCHED_COOPERATIVE_THREADS *
    TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
constexpr int TOPK_BATCHED_COOPERATIVE_WARPS =
    TOPK_BATCHED_COOPERATIVE_THREADS / 32;
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
// Threads per block for both reduction passes. Every production geometry is a
// power-of-two multiple of one CUDA warp so the deterministic warp reduction
// below can remove repeated block-wide barriers.
constexpr int ARGMAX_REDUCE_THREADS = 256;
constexpr int ARGMAX_FINALIZE_THREADS = 256;
// Target elements processed per thread in pass 1.  The production Qwen 3.6
// vocabulary has 248,320 columns, so four elements per thread launches 243
// blocks at the 256-thread geometry.  A repeated full-matrix Perf__ sweep on an
// RTX 3090 selected this geometry over the lower-wave 8-element launch while
// preserving the serial lower-token tie break exactly.
constexpr int ARGMAX_ELEMS_PER_THREAD = 4;

/**
 * @brief Publish one request-policy INT32 scalar into persistent device state.
 *
 * Kernel launch arguments are copied by the CUDA runtime before enqueue
 * returns, unlike an asynchronous H2D transfer sourced from a caller's stack
 * scalar. This one-thread kernel is therefore the correct ownership boundary
 * for host-selected control tokens such as a bounded-thinking stop sequence.
 */
__global__ void cuda_publish_int32_control_scalar_kernel(
    int32_t value,
    int32_t *out_value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *out_value = value;
}

__device__ __forceinline__ bool argmax_better(float candidate_value,
                                              int candidate_index,
                                              float current_value,
                                              int current_index)
{
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

/**
 * @brief Value/index pair ordered by serial argmax semantics.
 *
 * The index tie break is part of correctness, not merely determinism: serial
 * decode retains the lowest token id among equal logits.  Carrying both words
 * through every shuffle makes the parallel grouping mathematically identical
 * to that order for all finite FP32 logits.
 */
struct CudaArgmaxPair
{
    float value;
    int index;
};

/**
 * @brief Reduce one full CUDA warp without shared memory or barriers.
 */
__device__ __forceinline__ CudaArgmaxPair cuda_argmax_warp_reduce(
    CudaArgmaxPair pair)
{
    constexpr unsigned kFullWarpMask = 0xffffffffU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        const CudaArgmaxPair candidate{
            __shfl_down_sync(kFullWarpMask, pair.value, offset),
            __shfl_down_sync(kFullWarpMask, pair.index, offset)};
        if (argmax_better(
                candidate.value,
                candidate.index,
                pair.value,
                pair.index))
        {
            pair = candidate;
        }
    }
    return pair;
}

/**
 * @brief Reduce one block with one barrier and one final warp reduction.
 *
 * All supported launch geometries contain complete warps.  Each warp first
 * reduces independently, lane zero publishes one pair to shared memory, and
 * warp zero performs the final reduction.  Only thread zero consumes the
 * returned pair, so no trailing block barrier is necessary.
 */
__device__ __forceinline__ CudaArgmaxPair cuda_argmax_block_reduce(
    CudaArgmaxPair pair,
    float *warp_values,
    int *warp_indices)
{
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int warp_count = static_cast<int>(blockDim.x) / warpSize;

    pair = cuda_argmax_warp_reduce(pair);
    if (lane == 0)
    {
        warp_values[warp] = pair.value;
        warp_indices[warp] = pair.index;
    }
    __syncthreads();

    if (warp == 0)
    {
        pair = lane < warp_count
                   ? CudaArgmaxPair{
                         warp_values[lane],
                         warp_indices[lane]}
                   : CudaArgmaxPair{-FLT_MAX, INT_MAX};
        pair = cuda_argmax_warp_reduce(pair);
    }
    return pair;
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
    const int warp_count = static_cast<int>(blockDim.x) / warpSize;
    float *warp_values = reinterpret_cast<float *>(shared_mem);
    int *warp_indices = reinterpret_cast<int *>(
        shared_mem + warp_count * sizeof(float));

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

    const CudaArgmaxPair block_max = cuda_argmax_block_reduce(
        CudaArgmaxPair{local_max, local_idx},
        warp_values,
        warp_indices);

    if (tid == 0)
    {
        const int offset = row * row_partial_capacity + blockIdx.x;
        partial_vals[offset] = block_max.value;
        partial_idxs[offset] = block_max.index;
    }
}

/**
 * @brief Grouped verifier argmax pass with serial-row penalty semantics.
 *
 * Row `r` sees the durable generated-token histogram plus verifier input
 * tokens `[prefix_begin, r]`.  That is exactly the history visible to `r`
 * serial decode calls, but it avoids materializing one sparse host map per row
 * and avoids mutating the logits matrix before reduction.
 */
__global__ void cuda_argmax_partial_f32_batched_rows_mtp_penalty_kernel(
    const float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ active_rows_device,
    float *__restrict__ partial_vals,
    int *__restrict__ partial_idxs,
    int row_partial_capacity)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(
        shared_mem + blockDim.x * sizeof(float));

    const int row = blockIdx.y;
    const int active_rows = *active_rows_device;
    if (active_rows <= 0 || active_rows > rows || row >= active_rows)
        return;

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    const float *row_data =
        data + static_cast<size_t>(row) * static_cast<size_t>(row_stride);
    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    const int prefix_begin =
        request_policy.first_token_already_in_history != 0 ? 1 : 0;

    float local_max = -FLT_MAX;
    int local_idx = INT_MAX;
    for (int token = gid; token < cols; token += grid_stride)
    {
        float value = row_data[token];
        if (request_policy.enabled != 0)
        {
            int count = generated_token_counts[token];
            for (int history_index = prefix_begin;
                 history_index <= row;
                 ++history_index)
            {
                count += verifier_input_tokens[history_index] == token ? 1 : 0;
            }

            if (count > 0)
            {
                float penalty = 0.0f;
                if (request_policy.presence_penalty != 0.0f)
                    penalty += request_policy.presence_penalty;
                if (request_policy.frequency_penalty != 0.0f)
                {
                    const float frequency_component =
                        request_policy.frequency_penalty *
                        static_cast<float>(count);
                    penalty += frequency_component;
                }
                value -= penalty;
            }
        }
        if (argmax_better(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            argmax_better(
                smax[tid + stride],
                sidx[tid + stride],
                smax[tid],
                sidx[tid]))
        {
            smax[tid] = smax[tid + stride];
            sidx[tid] = sidx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        const int offset =
            row * row_partial_capacity + static_cast<int>(blockIdx.x);
        partial_vals[offset] = smax[0];
        partial_idxs[offset] = sidx[0];
    }
}

template <bool PUBLISH_MTP_CHAIN, bool DEVICE_ACTIVE_ROWS = false>
__global__ void cuda_argmax_finalize_f32_batched_rows_kernel(
    const float *__restrict__ partial_vals,
    const int *__restrict__ partial_idxs,
    int num_partials,
    int row_partial_capacity,
    float *__restrict__ out_values,
    int *__restrict__ out_indices,
    int output_stride,
    int *__restrict__ chain_condition_tokens,
    int *__restrict__ chain_position_ids,
    int chain_position_increment,
    const int *__restrict__ active_rows_device)
{
    extern __shared__ char shared_mem[];
    const int warp_count = static_cast<int>(blockDim.x) / warpSize;
    float *warp_values = reinterpret_cast<float *>(shared_mem);
    int *warp_indices = reinterpret_cast<int *>(
        shared_mem + warp_count * sizeof(float));

    const int row = blockIdx.x;
    if constexpr (DEVICE_ACTIVE_ROWS)
    {
        const int active_rows = *active_rows_device;
        if (active_rows <= 0 || active_rows > static_cast<int>(gridDim.x) ||
            row >= active_rows)
        {
            return;
        }
    }
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

    const CudaArgmaxPair block_max = cuda_argmax_block_reduce(
        CudaArgmaxPair{local_max, local_idx},
        warp_values,
        warp_indices);

    if (tid == 0)
    {
        const int out_row = row * output_stride;
        out_values[out_row] = block_max.value;
        out_indices[out_row] = block_max.index;
        if constexpr (PUBLISH_MTP_CHAIN)
        {
            /*
             * The chained sidecar consumes this exact winner and the next
             * absolute position. Publishing both from the reduction owner makes
             * the producer boundary one graph node: no later copy kernel can be
             * omitted, reordered, or accidentally pointed at host-authored state.
             */
            chain_condition_tokens[row] = block_max.index;
            chain_position_ids[row] += chain_position_increment;
        }
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
struct CUDABatchedTopKCooperativeStorage
{
    float warp_values[TOPK_BATCHED_COOPERATIVE_WARPS];
    int warp_indices[TOPK_BATCHED_COOPERATIVE_WARPS];
    int winner_index;
    float selected_values[K_CAP];
    int selected_indices[K_CAP];
};

struct CUDATopKCandidate
{
    float value;
    int index;
};

/**
 * @brief Reduce one deterministic candidate across a full CUDA warp.
 *
 * Every batched selector launches a whole number of 32-lane warps. The same
 * value-descending/index-ascending comparator used by the serial-row oracle is
 * applied at every shuffle edge, so work partitioning cannot change ties.
 */
__device__ __forceinline__ CUDATopKCandidate cudaTopKReduceWarpBest(
    CUDATopKCandidate candidate)
{
    constexpr unsigned kFullWarp = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float other_value =
            __shfl_down_sync(kFullWarp, candidate.value, offset);
        const int other_index =
            __shfl_down_sync(kFullWarp, candidate.index, offset);
        if (lane + offset < 32 &&
            topkCandidateBetter(
                other_value,
                other_index,
                candidate.value,
                candidate.index))
        {
            candidate.value = other_value;
            candidate.index = other_index;
        }
    }
    return candidate;
}

/**
 * @brief Select a block-wide sorted Top-K from register-resident candidates.
 *
 * Each thread owns exactly sixteen candidate slots. Forty cooperative maxima
 * therefore replace the former per-thread insertion sorts and thread-zero
 * k-way merge. The winner index is invalidated only in its owning thread after
 * every rank, which preserves one deterministic global order without atomics,
 * temporary allocation, or cross-block reduction.
 */
template <int K_CAP>
__device__ __forceinline__ void cudaSelectBatchedTopKCooperative(
    float (&local_values)[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD],
    int (&local_indices)[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD],
    int k,
    CUDABatchedTopKCooperativeStorage<K_CAP> &storage)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int rank = 0; rank < k; ++rank)
    {
        CUDATopKCandidate local_best{-FLT_MAX, -1};
#pragma unroll
        for (int item = 0;
             item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
             ++item)
        {
            if (topkCandidateBetter(
                    local_values[item],
                    local_indices[item],
                    local_best.value,
                    local_best.index))
            {
                local_best.value = local_values[item];
                local_best.index = local_indices[item];
            }
        }

        const CUDATopKCandidate warp_best =
            cudaTopKReduceWarpBest(local_best);
        if (lane == 0)
        {
            storage.warp_values[warp] = warp_best.value;
            storage.warp_indices[warp] = warp_best.index;
        }
        __syncthreads();

        if (warp == 0)
        {
            CUDATopKCandidate block_best{-FLT_MAX, -1};
            if (lane < TOPK_BATCHED_COOPERATIVE_WARPS)
            {
                block_best.value = storage.warp_values[lane];
                block_best.index = storage.warp_indices[lane];
            }
            block_best = cudaTopKReduceWarpBest(block_best);
            if (lane == 0)
            {
                storage.winner_index = block_best.index;
                storage.selected_values[rank] = block_best.value;
                storage.selected_indices[rank] = block_best.index;
            }
        }
        __syncthreads();

        const int winner_index = storage.winner_index;
#pragma unroll
        for (int item = 0;
             item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
             ++item)
        {
            if (local_indices[item] == winner_index)
                local_indices[item] = -1;
        }
    }
}

/**
 * @brief Produce exact sorted Top-K partials for Qwen-sized batched rows.
 */
template <int K_CAP>
__global__ void cuda_topk_cooperative_partials_batched_f32_kernel(
    const float *__restrict__ data,
    int n,
    int row_stride,
    int k,
    int partial_blocks,
    float *__restrict__ partial_values,
    int *__restrict__ partial_indices,
    const int *__restrict__ active_rows_device)
{
    const int row = blockIdx.y;
    const int active_rows =
        active_rows_device ? *active_rows_device : static_cast<int>(gridDim.y);
    if (k > K_CAP || active_rows <= 0 ||
        active_rows > static_cast<int>(gridDim.y) || row >= active_rows)
    {
        return;
    }

    const int partial = blockIdx.x;
    const int block_start =
        static_cast<int>((static_cast<long long>(partial) * n) / partial_blocks);
    const int block_end = static_cast<int>(
        (static_cast<long long>(partial + 1) * n) / partial_blocks);
    const float *row_data = data + static_cast<size_t>(row) * row_stride;

    float local_values[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD];
    int local_indices[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD];
#pragma unroll
    for (int item = 0;
         item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
         ++item)
    {
        const int offset =
            threadIdx.x + item * TOPK_BATCHED_COOPERATIVE_THREADS;
        const int index = block_start + offset;
        const bool valid = index < block_end;
        local_values[item] = valid ? row_data[index] : -FLT_MAX;
        local_indices[item] = valid ? index : -1;
    }

    __shared__ CUDABatchedTopKCooperativeStorage<K_CAP> storage;
    cudaSelectBatchedTopKCooperative(
        local_values, local_indices, k, storage);

    const int out_base = (row * partial_blocks + partial) * k;
    for (int rank = threadIdx.x; rank < k; rank += blockDim.x)
    {
        partial_values[out_base + rank] = storage.selected_values[rank];
        partial_indices[out_base + rank] = storage.selected_indices[rank];
    }
}

/**
 * @brief Merge cooperative partials and build one compact Top-K/Top-P row.
 */
template <int K_CAP>
__global__ void cuda_topk_topp_distribution_cooperative_batched_f32_kernel(
    const float *__restrict__ partial_values,
    const int *__restrict__ partial_indices,
    int partial_blocks,
    int k,
    float top_p,
    float temperature,
    int *__restrict__ out_token_ids,
    int out_stride,
    float *__restrict__ out_probs,
    const int *__restrict__ active_rows_device)
{
    const int row = blockIdx.x;
    const int active_rows =
        active_rows_device ? *active_rows_device : static_cast<int>(gridDim.x);
    if (k > K_CAP || active_rows <= 0 ||
        active_rows > static_cast<int>(gridDim.x) || row >= active_rows)
    {
        return;
    }

    const int candidate_count = partial_blocks * k;
    const int partial_base = row * candidate_count;
    float local_values[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD];
    int local_indices[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD];
#pragma unroll
    for (int item = 0;
         item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
         ++item)
    {
        const int offset =
            threadIdx.x + item * TOPK_BATCHED_COOPERATIVE_THREADS;
        const bool valid = offset < candidate_count;
        local_values[item] = valid
                                 ? partial_values[partial_base + offset]
                                 : -FLT_MAX;
        local_indices[item] = valid
                                  ? partial_indices[partial_base + offset]
                                  : -1;
    }

    __shared__ CUDABatchedTopKCooperativeStorage<K_CAP> storage;
    __shared__ float distribution_weights[K_CAP];
    cudaSelectBatchedTopKCooperative(
        local_values, local_indices, k, storage);

    if (threadIdx.x == 0)
    {
        const int out_base = row * out_stride;
        llaminar2::sampling_math::build_topk_topp_distribution_from_sorted(
            storage.selected_values,
            storage.selected_indices,
            k,
            top_p,
            temperature,
            out_token_ids + out_base,
            out_probs + out_base,
            distribution_weights);
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
    int *__restrict__ partial_indices,
    const int *__restrict__ active_rows_device)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.y;
    const int active_rows =
        active_rows_device ? *active_rows_device : static_cast<int>(gridDim.y);
    if (active_rows <= 0 || active_rows > static_cast<int>(gridDim.y) ||
        row >= active_rows)
    {
        return;
    }
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
    float *__restrict__ out_probs,
    const int *__restrict__ active_rows_device)
{
    if (k > K_CAP)
        return;

    const int row = blockIdx.x;
    const int active_rows =
        active_rows_device ? *active_rows_device : static_cast<int>(gridDim.x);
    if (active_rows <= 0 || active_rows > static_cast<int>(gridDim.x) ||
        row >= active_rows)
    {
        return;
    }
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
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    /*
     * Request-batched GPU MTP keeps the mutable logical position in its
     * publication mailbox. Reading that scalar here makes the random draw and
     * the state it describes part of one ordered device stream. The ordinary
     * scalar-threshold API remains available for nonresident callers.
     */
    const float effective_threshold =
        threshold_position
            ? llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                  threshold_seed,
                  *threshold_position + threshold_position_offset,
                  0 /* MTPSpecStochasticDrawPurpose::Sample */)
            : threshold;
    *out_token =
        llaminar2::sampling_math::sample_distribution_with_threshold_and_probability(
            token_ids, probs, k, effective_threshold, out_probability);
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
    llaminar2::sampling_math::SpeculativeBatchHostParameters parameters,
    int row_count,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count)
        return;

    const int offset = row * distribution_stride;
    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids + offset,
        target_probs + offset,
        draft_token_ids + offset,
        draft_probs + offset,
        k,
        parameters.draft_tokens[row],
        parameters.thresholds.accept[row],
        parameters.thresholds.residual[row],
        out_token + row,
        out_accepted + row,
        out_accept_probability ? out_accept_probability + row : nullptr,
        out_accept_threshold ? out_accept_threshold + row : nullptr);
}

/**
 * @brief Batched verifier variant whose draft tokens are already on device.
 *
 * This is the first device-resident-token step toward vLLM-style speculative
 * sampling. Each row may receive explicit thresholds, derive seeded thresholds
 * from a compatibility host position, or read the mutable base position from
 * the device-resident publication mailbox. The accepted draft-token sequence
 * is read from arena scratch produced by the draft sampler instead of from host
 * scalar arguments.
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
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    int row_count,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int inverse_sample_vocab_size,
    unsigned long long threshold_seed,
    int threshold_first_logical_position,
    int thresholds_from_seed,
    const int *__restrict__ threshold_base_position,
    int threshold_position_offset,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count)
        return;

    float accept_threshold = 0.0f;
    float residual_threshold = 0.0f;

    const int threshold_logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : threshold_first_logical_position + row;
    const int inverse_sample_logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : inverse_sample_first_logical_position + row;

    if (thresholds_from_seed)
    {
        accept_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                threshold_seed,
                threshold_logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */);
        residual_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                threshold_seed,
                threshold_logical_position,
                2 /* MTPSpecStochasticDrawPurpose::Residual */);
    }
    else
    {
        /*
         * Explicit host thresholds are packed into bounded value-owned kernel
         * parameters by the diagnostic wrapper. Seeded production rows never
         * index this bounded payload, so their runtime depth is unrestricted.
         */
        accept_threshold = thresholds.accept[row];
        residual_threshold = thresholds.residual[row];
    }

    const int offset = row * distribution_stride;
    if (!draft_token_ids && !draft_probs)
    {
        if (inverse_sample_vocab_size > 0)
        {
            llaminar2::sampling_math::speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
                target_token_ids + offset,
                target_probs + offset,
                k,
                inverse_sample_vocab_size,
                sampled_draft_tokens[row],
                accept_threshold,
                inverse_sample_seed,
                inverse_sample_logical_position,
                out_token + row,
                out_accepted + row,
                out_accept_probability ? out_accept_probability + row : nullptr,
                out_accept_threshold ? out_accept_threshold + row : nullptr);
        }
        else
        {
            llaminar2::sampling_math::speculative_verify_with_thresholds_one_hot_draft(
                target_token_ids + offset,
                target_probs + offset,
                k,
                sampled_draft_tokens[row],
                accept_threshold,
                residual_threshold,
                out_token + row,
                out_accepted + row,
                out_accept_probability ? out_accept_probability + row : nullptr,
                out_accept_threshold ? out_accept_threshold + row : nullptr);
        }
    }
    else
    {
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

/**
 * @brief Lazily sample a processed bonus row only if the verifier needs it.
 *
 * The row verifier has already written compact per-row accept/reject outputs.
 * If the first token stops or any verifier row rejects/stops, the bonus ready
 * token will not be consumed, so this kernel writes `-1` and avoids scanning the
 * full vocabulary row. When every row accepts, it falls through to the same
 * processed-logit sampler used by the eager bonus path.
 */
__global__ void cuda_sample_processed_logits_if_speculative_batch_needs_bonus_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float threshold,
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    int first_token,
    const int *__restrict__ first_token_device,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    __shared__ int should_sample_bonus;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;
    __shared__ int selected_token;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    if (threadIdx.x == 0)
    {
        int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7};
        const int sampled_first_token =
            first_token_device ? *first_token_device : first_token;
        should_sample_bonus =
            llaminar2::sampling_math::speculative_batch_needs_bonus_ready_token(
                sampled_first_token,
                verify_tokens,
                verify_accepted,
                row_count,
                stop_tokens,
                stop_token_count)
                ? 1
                : 0;
        if (!should_sample_bonus)
        {
            *out_token = -1;
            if (out_probability)
                *out_probability = 0.0f;
        }
    }
    __syncthreads();
    if (!should_sample_bonus)
        return;

    const float effective_threshold =
        threshold_position
            ? llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                  threshold_seed,
                  *threshold_position + threshold_position_offset,
                  0 /* MTPSpecStochasticDrawPurpose::Sample */)
            : threshold;

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
        effective_threshold,
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
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
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

    const float accept_threshold = thresholds.accept[row];
    const float residual_threshold = thresholds.residual[row];

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
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int thresholds_from_seed,
    const int *__restrict__ threshold_base_position,
    int threshold_position_offset,
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

    float accept_threshold = 0.0f;

    const int logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : inverse_sample_first_logical_position + row;
    if (thresholds_from_seed)
    {
        accept_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */);
    }
    else
    {
        accept_threshold = thresholds.accept[row];
    }

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
                logical_position,
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
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
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

    const float accept_threshold = thresholds.accept[row];

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
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
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

    const float accept_threshold = thresholds.accept[row];

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
 * The kernel is deliberately one thread: the bounded verifier transaction has
 * only a handful of rows, and making the reduction serial keeps stop/rejection
 * semantics easy to audit while remaining graph-capturable.
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
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
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
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : row_count + 1;
    llaminar2::sampling_math::summarize_speculative_verify_batch_at_commit_boundary(
        first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        /*greedy_draft_tokens=*/nullptr,
        leading_committed_output_count);
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
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
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
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : row_count + 1;
    llaminar2::sampling_math::summarize_speculative_verify_batch_at_commit_boundary(
        sampled_first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        /*greedy_draft_tokens=*/nullptr,
        leading_committed_output_count);
}

/**
 * @brief Reduce a generation transaction using only resident mutable controls.
 *
 * Request admission publishes the fixed stop-token row once.  The generation
 * controller is then the sole authority for both the current serial-visible
 * commit budget and the pending-correction carry bit.  Keeping those reads in
 * this kernel gives captured graph replay a stable parameter block and prevents
 * a host transaction shadow from selecting a different compact-row boundary.
 *
 * Exactly one row-comparison source is present.  Probability rejection supplies
 * @p verify_accepted; seeded serial-equivalent generation supplies
 * @p greedy_draft_tokens and compares the sampled target rows byte-for-byte.
 */
__global__ void
cuda_summarize_speculative_verify_batch_device_generation_controls_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    const int *__restrict__ greedy_draft_tokens,
    int row_count,
    const int *__restrict__ first_token,
    const int *__restrict__ stop_tokens,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    const int *__restrict__ generation_control,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const int ready_token =
        has_bonus_token && bonus_token ? *bonus_token : -1;
    const int commit_budget =
        generation_control
            ? generation_control[llaminar2::sampling_math::
                                     kDeviceGenerationControlTransactionCommitBudget]
            : 0;
    const int leading_committed_output_count =
        generation_control
            ? generation_control[llaminar2::sampling_math::
                                     kDeviceGenerationControlNextLeadingCommittedOutputCount]
            : -1;
    llaminar2::sampling_math::summarize_speculative_verify_batch_at_commit_boundary(
        first_token ? *first_token : -1,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
        ready_token,
        has_bonus_token,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        greedy_draft_tokens,
        leading_committed_output_count);
}

/**
 * @brief Fused seeded sampling and serial-equivalent speculative reduction.
 *
 * MTP admits at most fifteen speculative comparison rows, plus one bonus row.
 * One CUDA lane owns each compact row and executes the canonical scalar scan in
 * SamplingMath, preserving exactly the floating-point addition order used by
 * serial decode. A block barrier then makes the sampled row vector visible to
 * lane zero, which performs the ordinary compact transaction reduction.
 *
 * This is intentionally one warp and one block. The operation is launch- and
 * dependency-latency bound; increasing the grid would require global
 * coordination, atomics, or a second kernel. No local-memory scratch is used.
 */
template <bool RetainFirstTransactionDiagnostic>
__global__ __launch_bounds__(32, 1) void
cuda_sample_and_summarize_serial_equivalent_speculative_batch_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    int target_row_stride,
    int top_k,
    int row_count,
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ stop_tokens,
    const int *__restrict__ generation_control,
    int *__restrict__ sampled_target_tokens,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord
        *__restrict__ first_transaction_diagnostic)
{
    constexpr int kMaxSampleRows = 16;
    __shared__ int sampled_rows[kMaxSampleRows];
    __shared__ int active_comparison_rows;

    const int sample_row = static_cast<int>(threadIdx.x);
    if (sample_row == 0)
    {
        const int controller_depth =
            generation_control[
                llaminar2::sampling_math::
                    kDeviceGenerationControlCurrentDraftDepth];
        const int controller_verifier_rows =
            generation_control[
                llaminar2::sampling_math::
                    kDeviceGenerationControlActiveVerifierRowCount];
        const bool valid_depth =
            generation_control[
                llaminar2::sampling_math::kDeviceGenerationControlOk] != 0 &&
            controller_depth >= 1 && controller_depth <= row_count &&
            controller_verifier_rows == controller_depth + 1;
        active_comparison_rows = valid_depth ? controller_depth : 0;
        if (!valid_depth)
        {
            llaminar2::sampling_math::fail_device_generation_control(
                const_cast<int *>(generation_control),
                llaminar2::sampling_math::
                    DeviceGenerationError::InvalidDepthSelector);
        }
    }
    __syncthreads();

    if (active_comparison_rows == 0)
    {
        if (sample_row < kMaxSampleRows)
            sampled_target_tokens[sample_row] = -1;
        return;
    }

    const int sample_row_count = active_comparison_rows + 1;
    const int transaction_count =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlTransactionCount];
    const bool retain_first_transaction =
        RetainFirstTransactionDiagnostic &&
        first_transaction_diagnostic &&
        transaction_count == 0;
    if (sample_row < kMaxSampleRows)
    {
        const bool active = sample_row < sample_row_count;
        float threshold = 0.0f;
        int sampled = -1;
        if (active)
        {
            threshold =
                llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                    threshold_seed,
                    *threshold_position + threshold_position_offset +
                        sample_row,
                    0 /* MTPSpecStochasticDrawPurpose::Sample */);
            sampled =
                llaminar2::sampling_math::
                    sample_distribution_with_threshold(
                        target_token_ids +
                            sample_row * target_row_stride,
                        target_probs + sample_row * target_row_stride,
                        top_k,
                        threshold);
            sampled_rows[sample_row] = sampled;
            sampled_target_tokens[sample_row] = sampled;
        }
        else
        {
            sampled_rows[sample_row] = -1;
            sampled_target_tokens[sample_row] = -1;
        }
        if constexpr (RetainFirstTransactionDiagnostic)
        {
            if (retain_first_transaction)
            {
                llaminar2::sampling_math::
                    record_mtp_first_transaction_sample_row(
                        first_transaction_diagnostic,
                        sample_row,
                        active,
                        active
                            ? target_token_ids +
                                  sample_row * target_row_stride
                            : nullptr,
                        active
                            ? target_probs +
                                  sample_row * target_row_stride
                            : nullptr,
                        top_k,
                        verifier_input_tokens,
                        active_comparison_rows,
                        threshold,
                        sampled);
            }
        }
    }
    __syncthreads();

    if (sample_row != 0)
        return;

    const int commit_budget =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlTransactionCommitBudget];
    const int leading_committed_output_count =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlNextLeadingCommittedOutputCount];
    llaminar2::sampling_math::
        summarize_speculative_verify_batch_at_commit_boundary(
            verifier_input_tokens[0],
            sampled_rows,
            /*row_accepted=*/nullptr,
            active_comparison_rows,
            stop_tokens,
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
            sampled_rows[active_comparison_rows],
            /*has_bonus_ready_token=*/1,
            commit_budget,
            out_tokens,
            out_token_capacity,
            out_meta,
            verifier_input_tokens,
            leading_committed_output_count);

    if constexpr (RetainFirstTransactionDiagnostic)
    {
        if (retain_first_transaction)
        {
            llaminar2::sampling_math::
                finalize_mtp_first_transaction_diagnostic(
                    first_transaction_diagnostic,
                    threshold_seed,
                    *threshold_position,
                    threshold_position_offset,
                    active_comparison_rows,
                    top_k,
                    transaction_count,
                    commit_budget,
                    leading_committed_output_count,
                    out_tokens,
                    out_token_capacity,
                    out_meta);
        }
    }
}

/**
 * @brief Retain one sidecar boundary before its shared workspace is reused.
 *
 * One block hashes strided 32-bit words in parallel.  Lane zero folds the
 * lane-local hashes in a fixed order, giving identical bytes an identical
 * digest independent of scheduling.  This kernel is present only in an
 * explicitly diagnostic graph; the production publication specialization has
 * no corresponding launch or branch.
 */
__global__ __launch_bounds__(256, 1) void
cuda_retain_mtp_first_transaction_draft_boundary_kernel(
    const uint32_t *__restrict__ data_words,
    int word_count,
    int boundary,
    int draft_slot,
    const int *__restrict__ condition_token,
    const int *__restrict__ position_id,
    const int *__restrict__ generation_control,
    int generation_control_stride,
    llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord
        *__restrict__ diagnostic)
{
    using namespace llaminar2::sampling_math;
    constexpr int kThreads = 256;
    __shared__ unsigned long long lane_hashes[kThreads];

    if (!generation_control || !diagnostic || generation_control_stride <
            kDeviceGenerationControlCount ||
        generation_control[kDeviceGenerationControlTransactionCount] != 0)
    {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    uint64_t hash = 1469598103934665603ULL ^
                    static_cast<uint64_t>(lane + 1);
    for (int index = lane; index < word_count; index += kThreads)
    {
        hash = append_diagnostic_u32_fnv1a(
            hash,
            static_cast<uint32_t>(index));
        hash = append_diagnostic_u32_fnv1a(hash, data_words[index]);
    }
    lane_hashes[lane] = hash;
    __syncthreads();

    if (lane != 0)
        return;

    if (draft_slot == 0 && boundary == 0)
    {
        diagnostic->valid = 0;
        diagnostic->version = kMTPFirstTransactionDiagnosticVersion;
        diagnostic->draft_diagnostic_depth = 0;
        for (int slot = 0; slot < kSpeculativeBatchMaxRows; ++slot)
        {
            diagnostic->draft_condition_tokens[slot] = -1;
            diagnostic->draft_position_ids[slot] = -1;
            for (int retained_boundary = 0;
                 retained_boundary < kMTPFirstTransactionDraftBoundaryCount;
                 ++retained_boundary)
            {
                diagnostic->draft_boundary_word_counts[slot]
                                                       [retained_boundary] = 0;
                diagnostic->draft_boundary_hashes[slot]
                                                   [retained_boundary] = 0;
            }
        }
    }

    if (boundary == 0)
    {
        diagnostic->draft_condition_tokens[draft_slot] = *condition_token;
        diagnostic->draft_position_ids[draft_slot] = *position_id;
        diagnostic->draft_diagnostic_depth = draft_slot + 1;
    }

    uint64_t folded = 1469598103934665603ULL;
    if (word_count > 0)
    {
        for (int source_lane = 0; source_lane < kThreads; ++source_lane)
        {
            const uint64_t lane_hash = lane_hashes[source_lane];
            folded = append_diagnostic_u32_fnv1a(
                folded,
                static_cast<uint32_t>(lane_hash));
            folded = append_diagnostic_u32_fnv1a(
                folded,
                static_cast<uint32_t>(lane_hash >> 32U));
        }
    }
    else
    {
        folded = 0;
    }
    diagnostic->draft_boundary_word_counts[draft_slot][boundary] =
        static_cast<uint32_t>(word_count);
    diagnostic->draft_boundary_hashes[draft_slot][boundary] = folded;
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
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
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
    const int sampled_first_token = draft_tokens ? draft_tokens[0] : first_token;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(compare_row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : compare_row_count + 1;
    llaminar2::sampling_math::summarize_greedy_speculative_verify_batch_at_commit_boundary(
        sampled_first_token,
        verify_tokens,
        draft_tokens,
        compare_row_count,
        stop_tokens,
        stop_token_count,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        leading_committed_output_count);
}

/**
 * @brief Graph-replayable greedy reducer with all mutable controls on device.
 *
 * The verifier input row owns the first target token and the fixed stop-token
 * row owns eight entries with `-1` padding.  Reading both in the kernel keeps a
 * cached CUDA graph valid when request values change between launches.
 */
__global__ void
cuda_summarize_greedy_speculative_verify_batch_device_controls_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ draft_tokens,
    int compare_row_count,
    const int *__restrict__ active_verifier_row_count,
    const int *__restrict__ stop_tokens,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ penalty_policy)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const int active_rows = *active_verifier_row_count;
    if (active_rows <= 0 || active_rows > compare_row_count + 1)
    {
        for (int token = 0; token < out_token_capacity; ++token)
            out_tokens[token] = -1;
        for (int field = 0;
             field < llaminar2::sampling_math::kSpeculativeBatchMetaCount;
             ++field)
        {
            out_meta[field] = 0;
        }
        return;
    }

    /*
     * compare_row_count is the immutable physical bucket geometry captured in
     * the graph.  The request-owned scalar supplies the logical verifier width
     * for this replay, so padded rows never participate in acceptance or bonus
     * token selection.
     */
    const int active_compare_row_count = active_rows - 1;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(active_rows)
            ? static_cast<int>(*max_state_commit_rows)
            : active_rows;
    const int leading_committed_output_count =
        penalty_policy && penalty_policy->first_token_already_in_history != 0
            ? 1
            : 0;
    llaminar2::sampling_math::summarize_greedy_speculative_verify_batch_at_commit_boundary(
        draft_tokens[0],
        verify_tokens,
        draft_tokens,
        active_compare_row_count,
        stop_tokens,
        llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        leading_committed_output_count);
}

/**
 * @brief Advance one shared decode-round boundary from compact request rows.
 *
 * Request summaries are produced sequentially on the verifier stream and all
 * read the same pre-transaction budget. The largest committed-state count is
 * the number of lockstep serial decode rounds represented by the batch.
 */
__global__ void cuda_advance_speculative_commit_boundary_kernel(
    const int *__restrict__ meta,
    int request_count,
    int meta_stride,
    uint32_t *__restrict__ decode_rounds_committed,
    uint32_t *__restrict__ decode_rounds_until_maintenance,
    uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (!meta || request_count <= 0 ||
        meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
        !decode_rounds_committed ||
        !decode_rounds_until_maintenance ||
        !maintenance_due || !decode_boundary_advanced ||
        *maintenance_due != 0u || *decode_boundary_advanced != 0u)
    {
        if (maintenance_due)
            *maintenance_due = 2u;
        return;
    }

    uint32_t committed_rounds = 0u;
    for (int request = 0; request < request_count; ++request)
    {
        const int *request_meta =
            meta + static_cast<size_t>(request) * meta_stride;
        if (request_meta[llaminar2::sampling_math::kSpecBatchMetaOk] == 0)
        {
            *maintenance_due = 2u;
            return;
        }
        const int request_commits =
            llaminar2::sampling_math::speculative_new_commit_count(
                request_meta[llaminar2::sampling_math::kSpecBatchMetaOutputCount],
                request_meta[llaminar2::sampling_math::kSpecBatchMetaLeadingCommittedOutputCount]);
        if (request_commits < 0)
        {
            *maintenance_due = 2u;
            return;
        }
        committed_rounds = max(
            committed_rounds,
            static_cast<uint32_t>(request_commits));
    }

    const uint32_t remaining = *decode_rounds_until_maintenance;
    if (committed_rounds > remaining)
    {
        *maintenance_due = 2u;
        return;
    }
    *decode_rounds_committed += committed_rounds;
    *decode_rounds_until_maintenance = remaining - committed_rounds;
    if (committed_rounds == remaining)
        *maintenance_due = 1u;
    *decode_boundary_advanced = 1u;
}

/**
 * @brief Publish one request's immutable grouped-greedy controls on device.
 */
__global__ void cuda_configure_mtp_greedy_penalty_policy_kernel(
    llaminar2::MTPGreedyPenaltyPolicy *__restrict__ controls,
    float presence_penalty,
    float frequency_penalty,
    int first_token_already_in_history)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    controls->presence_penalty = presence_penalty;
    controls->frequency_penalty = frequency_penalty;
    controls->first_token_already_in_history =
        first_token_already_in_history != 0 ? 1 : 0;
    controls->enabled =
        presence_penalty != 0.0f || frequency_penalty != 0.0f ? 1 : 0;
}

/**
 * @brief Apply durable-history penalties when no verifier prefix is present.
 *
 * This is the latency-critical M=1 specialization. Adjacent threads own
 * adjacent vocabulary tokens and reuse one computed penalty across every row.
 * Keeping verifier-prefix control flow out of this kernel preserves the small
 * register footprint and launch latency of ordinary serial decode.
 */
__global__ void cuda_apply_mtp_durable_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ active_rows_device)
{
    const int active_rows = active_rows_device ? *active_rows_device : rows;
    if (active_rows <= 0 || active_rows > rows)
        return;

    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    for (int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         token < cols;
         token += static_cast<int>(blockDim.x * gridDim.x))
    {
        const int count = generated_token_counts[token];
        if (count <= 0)
            continue;

        float penalty = 0.0f;
        if (request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        for (int row = 0; row < active_rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] -= penalty;
        }
    }
}

/**
 * @brief Apply device-owned serial-decode penalties to verifier logit rows.
 *
 * Threads traverse adjacent vocabulary tokens, yielding coalesced histogram
 * reads and in-place logit writes. A token that occurs anywhere in the short
 * verifier branch is deliberately left for sparse row owners in block zero.
 * All other tokens have the same durable-history count in every row, so one
 * thread can apply their penalties to all rows after scanning the branch
 * exactly once.
 *
 * This ownership split reduces the verifier work from O(rows^2 * vocabulary)
 * token comparisons to O(rows * vocabulary), while retaining one writer for
 * every output element. The sparse owners and dense owners touch disjoint
 * token addresses, so they can share one kernel launch without a grid-wide
 * barrier. The operation needs no atomics or temporary storage.
 */
__global__ void cuda_apply_mtp_verifier_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ active_rows_device)
{
    const int active_rows = active_rows_device ? *active_rows_device : rows;
    if (active_rows <= 0 || active_rows > rows)
        return;

    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    const int prefix_begin =
        request_policy.first_token_already_in_history != 0 ? 1 : 0;

    // A verifier launch reserves block zero for sparse branch-token ownership.
    // It executes concurrently with all dense blocks, and then exits because
    // dense blocks deliberately skip every address this block can modify.
    if (blockIdx.x == 0)
    {
        for (int row = static_cast<int>(threadIdx.x);
             row < active_rows;
             row += static_cast<int>(blockDim.x))
        {
            for (int candidate_index = prefix_begin;
                 candidate_index < active_rows;
                 ++candidate_index)
            {
                const int token = verifier_input_tokens[candidate_index];
                if (token < 0 || token >= cols)
                    continue;

                bool already_processed = false;
                for (int earlier = prefix_begin;
                     earlier < candidate_index;
                     ++earlier)
                {
                    already_processed |=
                        verifier_input_tokens[earlier] == token;
                }
                if (already_processed)
                    continue;

                int count = generated_token_counts[token];
                for (int visible = prefix_begin; visible <= row; ++visible)
                    count += verifier_input_tokens[visible] == token ? 1 : 0;
                if (count <= 0)
                    continue;

                float penalty = 0.0f;
                if (request_policy.presence_penalty != 0.0f)
                    penalty += request_policy.presence_penalty;
                if (request_policy.frequency_penalty != 0.0f)
                {
                    penalty += request_policy.frequency_penalty *
                               static_cast<float>(count);
                }
                data[static_cast<size_t>(row) *
                         static_cast<size_t>(row_stride) +
                     static_cast<size_t>(token)] -= penalty;
            }
        }
        return;
    }

    const int dense_block = static_cast<int>(blockIdx.x) - 1;
    const int dense_blocks = static_cast<int>(gridDim.x) - 1;
    for (int token = dense_block * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
         token < cols;
         token += static_cast<int>(blockDim.x) * dense_blocks)
    {
        bool verifier_branch_token = false;
        for (int history_index = prefix_begin;
             history_index < active_rows;
             ++history_index)
        {
            verifier_branch_token |=
                verifier_input_tokens[history_index] == token;
        }
        if (verifier_branch_token)
            continue;

        const int count = generated_token_counts[token];
        if (count <= 0)
            continue;

        float penalty = 0.0f;
        if (request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        for (int row = 0; row < active_rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] -= penalty;
        }
    }

}

/**
 * @brief Return one token from the scalar MTP proposal branch.
 *
 * The first condition token lives in the full-sidecar capture slot while all
 * later branch tokens live in contiguous draft sample slots.  Treating those
 * two persistent owners as one logical row avoids staging a temporary verifier
 * prefix merely to score the next proposal.
 */
__device__ __forceinline__ int cuda_mtp_branch_token_at(
    int branch_index,
    int include_first_condition,
    const int *__restrict__ first_condition_token,
    const int *__restrict__ prior_draft_tokens)
{
    if (include_first_condition != 0 && branch_index == 0)
        return first_condition_token[0];
    return prior_draft_tokens[
        branch_index - (include_first_condition != 0 ? 1 : 0)];
}

/**
 * @brief Apply durable plus speculative-branch penalties to one proposal row.
 *
 * Block zero owns every vocabulary address mentioned by the short branch and
 * deduplicates repeated token ids.  All remaining blocks own the dense durable
 * histogram addresses and skip branch tokens.  The two ownership sets are
 * disjoint, so they may execute concurrently without atomics or a grid barrier.
 */
__global__ void cuda_apply_mtp_branch_penalties_f32_row_kernel(
    float *__restrict__ data,
    int cols,
    const int *__restrict__ first_condition_token,
    const int *__restrict__ prior_draft_tokens,
    int prior_draft_count,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy)
{
    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    const int include_first_condition =
        request_policy.first_token_already_in_history == 0 ? 1 : 0;
    const int branch_count = include_first_condition + prior_draft_count;

    if (blockIdx.x == 0)
    {
        for (int branch_index = static_cast<int>(threadIdx.x);
             branch_index < branch_count;
             branch_index += static_cast<int>(blockDim.x))
        {
            const int token = cuda_mtp_branch_token_at(
                branch_index,
                include_first_condition,
                first_condition_token,
                prior_draft_tokens);
            if (token < 0 || token >= cols)
                continue;

            bool already_processed = false;
            for (int earlier = 0; earlier < branch_index; ++earlier)
            {
                already_processed |=
                    cuda_mtp_branch_token_at(
                        earlier,
                        include_first_condition,
                        first_condition_token,
                        prior_draft_tokens) == token;
            }
            if (already_processed)
                continue;

            int count = generated_token_counts[token];
            for (int visible = 0; visible < branch_count; ++visible)
            {
                count += cuda_mtp_branch_token_at(
                             visible,
                             include_first_condition,
                             first_condition_token,
                             prior_draft_tokens) == token
                             ? 1
                             : 0;
            }
            if (count <= 0)
                continue;

            float penalty = 0.0f;
            if (request_policy.presence_penalty != 0.0f)
                penalty += request_policy.presence_penalty;
            if (request_policy.frequency_penalty != 0.0f)
            {
                penalty += request_policy.frequency_penalty *
                           static_cast<float>(count);
            }
            data[token] -= penalty;
        }
        return;
    }

    const int dense_block = static_cast<int>(blockIdx.x) - 1;
    const int dense_blocks = static_cast<int>(gridDim.x) - 1;
    for (int token = dense_block * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
         token < cols;
         token += dense_blocks * static_cast<int>(blockDim.x))
    {
        bool branch_owned = false;
        for (int branch_index = 0;
             branch_index < branch_count;
             ++branch_index)
        {
            branch_owned |=
                cuda_mtp_branch_token_at(
                    branch_index,
                    include_first_condition,
                    first_condition_token,
                    prior_draft_tokens) == token;
        }
        if (branch_owned)
            continue;

        const int count = generated_token_counts[token];
        if (count <= 0)
            continue;

        float penalty = 0.0f;
        if (request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        data[token] -= penalty;
    }
}

/**
 * @brief Advance the generated-token histogram from the compact outcome.
 *
 * A single thread owns the short row, so duplicate tokens are updated in
 * deterministic program order without atomics.  Pending-condition row zero was
 * emitted by the previous transaction and is deliberately skipped.
 */
__global__ void cuda_commit_mtp_greedy_penalty_history_kernel(
    const int *__restrict__ output_tokens,
    const int *__restrict__ output_meta,
    llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ accepted_state_counts,
    const int *__restrict__ stopped_flags,
    int output_token_capacity,
    int vocab_size,
    int *__restrict__ generated_token_counts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (output_meta[llaminar2::sampling_math::kSpecBatchMetaOk] == 0)
        return;

    if (policy->enabled == 0)
    {
        policy->first_token_already_in_history = 0;
        return;
    }

    int output_count =
        output_meta[llaminar2::sampling_math::kSpecBatchMetaOutputCount];
    if (output_count < 0)
        output_count = 0;
    if (output_count > output_token_capacity)
        output_count = output_token_capacity;
    const int accepted_state_count = accepted_state_counts[0];
    if (accepted_state_count < 0 || accepted_state_count > output_count)
        return;
    const int begin =
        policy->first_token_already_in_history != 0 ? 1 : 0;
    for (int i = begin; i < output_count; ++i)
    {
        const int token = output_tokens[i];
        if (token >= 0 && token < vocab_size)
            ++generated_token_counts[token];
    }

    policy->first_token_already_in_history =
        stopped_flags[0] == 0 && output_count > accepted_state_count ? 1 : 0;
}

/**
 * @brief Initialize one persistent device-generation controller per request.
 *
 * The response payload is intentionally left untouched.  Its authoritative
 * length starts at zero, so clearing a potentially long context-sized token row
 * would add request-admission bandwidth without changing any observable state.
 */
__global__ void cuda_initialize_device_generation_kernel(
    int request_count,
    int max_new_tokens,
    llaminar2::sampling_math::DeviceGenerationDepthPolicy depth_policy,
    int response_token_stride,
    int control_stride,
    int *__restrict__ control)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    llaminar2::sampling_math::initialize_device_generation_control(
        max_new_tokens,
        response_token_stride,
        depth_policy,
        request_control);
}

/**
 * @brief Acknowledge an ordinary prior boundary and publish the next budget.
 *
 * A speculative outcome leaves `decode_boundary_advanced` set so the due
 * maintenance transaction can consume that exact edge without advancing the
 * clock twice. When the outcome did not make maintenance due, the next verifier
 * admission is the structural consumer of that acknowledgement. A due,
 * poisoned, or malformed boundary is rejected before any new transaction can
 * observe a budget.
 */
__global__ void cuda_prepare_device_generation_transaction_budget_kernel(
    int *__restrict__ control,
    int control_stride,
    int request_count,
    int verifier_row_capacity,
    const uint32_t *__restrict__ maintenance_rows_remaining,
    const uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    const bool maintenance_boundary_bound =
        maintenance_rows_remaining || maintenance_due ||
        decode_boundary_advanced;
    const bool maintenance_boundary_complete =
        maintenance_rows_remaining && maintenance_due &&
        decode_boundary_advanced;
    const uint32_t due =
        maintenance_boundary_complete ? *maintenance_due : 0u;
    const uint32_t boundary_advanced =
        maintenance_boundary_complete ? *decode_boundary_advanced : 0u;
    const bool maintenance_boundary_valid =
        !maintenance_boundary_bound ||
        (maintenance_boundary_complete && due == 0u &&
         boundary_advanced <= 1u);
    const int maintenance_budget =
        maintenance_boundary_valid && maintenance_rows_remaining
            ? static_cast<int>(*maintenance_rows_remaining)
            : (maintenance_boundary_valid ? verifier_row_capacity : 0);
    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    llaminar2::sampling_math::prepare_device_generation_transaction_budget(
        verifier_row_capacity,
        maintenance_budget,
        request_control);

    /*
     * Exactly one lane retires the shared acknowledgement. Other lanes may
     * observe either one or zero, both of which are valid ordinary states; no
     * request depends on the value after validating it.
     */
    if (request_index == 0 && maintenance_boundary_valid &&
        maintenance_boundary_complete && boundary_advanced == 1u)
    {
        *decode_boundary_advanced = 0u;
    }
}

/**
 * @brief Commit one compact outcome and derive its accepted publication row.
 *
 * One thread owns one request transaction.  Combining response-ledger commit
 * with publication derivation removes a launch from every MTP transaction and
 * makes the serial-visible response/state boundary indivisible.
 */
__global__ void
cuda_commit_device_generation_and_derive_speculative_publication_metadata_kernel(
    const int32_t *__restrict__ compact_tokens,
    int output_token_stride,
    int *__restrict__ compact_meta,
    int meta_stride,
    const int *__restrict__ base_cached_tokens,
    int request_count,
    int padded_state_rows_per_request,
    int32_t *__restrict__ response_tokens,
    int response_token_stride,
    int *__restrict__ control,
    int control_stride,
    int *__restrict__ out_restore_rows,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok,
    int32_t *__restrict__ out_next_condition_tokens,
    int *__restrict__ out_all_drafts_accepted_flags,
    int *__restrict__ out_stopped_flags,
    int32_t *__restrict__ out_next_sidecar_condition_tokens,
    int32_t *__restrict__ out_next_sidecar_position_ids,
    int32_t *__restrict__ out_next_verifier_condition_tokens)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int32_t *request_response_tokens =
        response_tokens + static_cast<size_t>(request_index) *
                              static_cast<size_t>(response_token_stride);
    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    const int base_cached_tokens_for_request =
        base_cached_tokens ? base_cached_tokens[request_index] : -1;
    llaminar2::sampling_math::
        commit_device_generation_and_derive_speculative_publication_metadata(
        compact_tokens,
        output_token_stride,
        compact_meta,
        meta_stride,
        request_index,
        padded_state_rows_per_request,
        base_cached_tokens_for_request,
        request_response_tokens,
        response_token_stride,
        request_control,
        out_restore_rows ? out_restore_rows + request_index : nullptr,
        out_target_cached_tokens
            ? out_target_cached_tokens + request_index
            : nullptr,
        out_accepted_state_counts
            ? out_accepted_state_counts + request_index
            : nullptr,
        out_ok ? out_ok + request_index : nullptr,
        out_next_condition_tokens
            ? out_next_condition_tokens + request_index
            : nullptr,
        out_all_drafts_accepted_flags
            ? out_all_drafts_accepted_flags + request_index
            : nullptr,
        out_stopped_flags ? out_stopped_flags + request_index : nullptr);

    /*
     * These mirrors are the direct inputs of the next loop iteration's full
     * sidecar. Keeping them in the response/state commit kernel guarantees that
     * the controller, logical mailbox, and sidecar mailbox cross one atomic
     * device publication boundary.
     */
    if (out_next_sidecar_condition_tokens &&
        out_next_sidecar_position_ids)
    {
        out_next_sidecar_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
        out_next_sidecar_position_ids[request_index] =
            out_target_cached_tokens[request_index];
    }
    if (out_next_verifier_condition_tokens)
    {
        out_next_verifier_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
    }
}

/**
 * @brief Derive publication rows/counts from compact speculative metadata.
 *
 * Each request is independent, so one CUDA thread maps one compact metadata row
 * to the state row and cache count that later publication stages consume. The
 * helper is deliberately shared with CPU tests to keep the tricky
 * "accepted draft prefix" versus "committed verifier state rows" distinction in
 * one place.
 */
__global__ void cuda_derive_speculative_publication_metadata_kernel(
    const int *__restrict__ meta,
    int meta_stride,
    const int *__restrict__ base_cached_tokens,
    int request_count,
    int padded_state_rows_per_request,
    int max_state_commit_rows,
    int *__restrict__ out_restore_rows,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok,
    int32_t *__restrict__ out_next_condition_tokens,
    const int32_t *__restrict__ output_tokens,
    int output_token_stride,
    int *__restrict__ out_all_drafts_accepted_flags,
    int *__restrict__ out_stopped_flags,
    int32_t *__restrict__ out_next_verifier_condition_tokens)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    const int base_cached_tokens_for_request =
        base_cached_tokens ? base_cached_tokens[request_index] : -1;
    llaminar2::sampling_math::derive_speculative_publication_metadata(
        meta,
        meta_stride,
        request_index,
        padded_state_rows_per_request,
        base_cached_tokens_for_request,
        max_state_commit_rows,
        out_restore_rows ? out_restore_rows + request_index : nullptr,
        out_target_cached_tokens ? out_target_cached_tokens + request_index : nullptr,
        out_accepted_state_counts ? out_accepted_state_counts + request_index : nullptr,
        out_ok ? out_ok + request_index : nullptr,
        output_tokens,
        output_token_stride,
        out_next_condition_tokens ? out_next_condition_tokens + request_index : nullptr,
        out_all_drafts_accepted_flags ? out_all_drafts_accepted_flags + request_index : nullptr,
        out_stopped_flags ? out_stopped_flags + request_index : nullptr);
    if (out_next_verifier_condition_tokens)
    {
        out_next_verifier_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
    }
}

/**
 * @brief Derive shifted MTP KV counts from canonical primary publication.
 *
 * One CUDA thread handles one request. The shared SamplingMath helper owns the
 * depth-dependent off-by-one rule.  The primary rows already carry the exact
 * device-owned transaction boundary, so this kernel cannot diverge by
 * re-reading compact metadata with a host scalar.
 */
__global__ void
cuda_derive_shifted_speculative_publication_metadata_from_primary_kernel(
    const int *__restrict__ base_cached_tokens,
    const int *__restrict__ main_target_cached_tokens,
    const int *__restrict__ main_publication_ok,
    int request_count,
    int mtp_depth,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    llaminar2::sampling_math::
        derive_shifted_speculative_publication_metadata_from_primary(
        base_cached_tokens[request_index],
        main_target_cached_tokens[request_index],
        main_publication_ok[request_index],
        mtp_depth,
        out_target_cached_tokens ? out_target_cached_tokens + request_index : nullptr,
        out_accepted_state_counts ? out_accepted_state_counts + request_index : nullptr,
        out_ok ? out_ok + request_index : nullptr);
}

/**
 * @brief Prepare shifted-MTP suffix tokens and absolute positions from compact metadata.
 *
 * A single CUDA thread is enough for the four-row MTP verifier bound.  Keeping
 * this as a kernel still matters because the accepted-state count stays
 * device-resident; the CPU never decides how many suffix rows are publishable.
 * The same lane expands the request's canonical verifier-base position into the
 * contiguous position row consumed by the captured KV-only sidecar. Fusing
 * these stores avoids a second launch and keeps both dynamic graph inputs under
 * one stream-ordered device owner.
 */
__global__ void cuda_prepare_speculative_shifted_kv_tokens_kernel(
    const int *__restrict__ meta,
    int meta_stride,
    const int32_t *__restrict__ output_tokens,
    int output_token_stride,
    int request_index,
    int first_output_token_index,
    int row_count,
    int32_t filler_token,
    int32_t *__restrict__ out_tokens,
    const int32_t *__restrict__ base_positions,
    int position_offset,
    int32_t *__restrict__ out_position_ids)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::prepare_speculative_shifted_kv_tokens(
        meta,
        meta_stride,
        output_tokens,
        output_token_stride,
        request_index,
        first_output_token_index,
        row_count,
        filler_token,
        out_tokens);

    const int32_t first_position =
        base_positions[request_index] + position_offset;
    for (int row = 0; row < row_count; ++row)
        out_position_ids[row] = first_position + row;
}

/**
 * @brief Gather strided MTP draft slots and derive grouped sidecar positions.
 *
 * Request-batched drafting lays proposal tokens out request-major so the final
 * verifier can consume each request row contiguously. At an intermediate depth,
 * adjacent requests are therefore separated by the configured draft depth. One
 * thread per request gathers that token and adds the current depth to the
 * device-resident logical position. The output arrays are stable arena buffers
 * captured by the next MTP sidecar graph.
 */
__global__ void cuda_prepare_mtp_batched_sidecar_inputs_kernel(
    const int32_t *__restrict__ condition_tokens,
    int condition_token_stride,
    const int32_t *__restrict__ base_positions,
    int position_offset,
    int request_count,
    int32_t *__restrict__ out_condition_tokens,
    int32_t *__restrict__ out_position_ids)
{
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= request_count)
        return;

    out_condition_tokens[request] =
        condition_tokens[request * condition_token_stride];
    out_position_ids[request] = base_positions[request] + position_offset;
}

/**
 * @brief Publish all dynamic grouped-verifier geometry from device-owned rows.
 *
 * Each request owns one device-resident next-position scalar in its KV cache.
 * Grouped verification needs one absolute position per physical graph row. A
 * flattened thread index covers the request-major matrix and derives both
 * coordinates without any host-visible cursor:
 *
 * `position[request, token] = live_kv_count[request] + token`.
 *
 * Padded columns are populated as well. The first @p request_count threads also
 * publish each request's valid width. Rectangular graphs need no row-index
 * input; ragged graphs count their already-published physical verifier rows.
 * Thus position and recurrent-state masks become visible in one ordered launch,
 * with no host length mirror or second metadata transfer.
 */
__global__ void cuda_prepare_mtp_verifier_geometry_kernel(
    const int32_t *__restrict__ base_positions,
    const int32_t *__restrict__ valid_graph_rows,
    int valid_graph_row_count,
    int *__restrict__ generation_control,
    int generation_control_stride,
    int request_count,
    int padded_seq_len,
    int32_t *__restrict__ out_position_ids,
    int32_t *__restrict__ out_request_lengths)
{
    const int flat_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = request_count * padded_seq_len;
    if (flat_row < total_rows)
    {
        const int request = flat_row / padded_seq_len;
        const int token = flat_row - request * padded_seq_len;
        out_position_ids[flat_row] = base_positions[request] + token;
    }

    if (out_request_lengths && flat_row < request_count)
    {
        int valid_tokens = valid_graph_row_count / request_count;
        if (generation_control)
        {
            int *const control =
                generation_control +
                static_cast<size_t>(flat_row) *
                    static_cast<size_t>(generation_control_stride);
            const int depth =
                control[llaminar2::sampling_math::
                            kDeviceGenerationControlCurrentDraftDepth];
            valid_tokens =
                control[llaminar2::sampling_math::
                            kDeviceGenerationControlActiveVerifierRowCount];
            if (depth <= 0 || valid_tokens != depth + 1 ||
                valid_tokens > padded_seq_len)
            {
                llaminar2::sampling_math::fail_device_generation_control(
                    control,
                    llaminar2::sampling_math::
                        DeviceGenerationError::InvalidDepthSelector);
                valid_tokens = 0;
            }
        }
        else if (valid_graph_rows)
        {
            valid_tokens = 0;
            for (int row = 0; row < valid_graph_row_count; ++row)
            {
                const int physical_row = valid_graph_rows[row];
                valid_tokens +=
                    physical_row / padded_seq_len == flat_row ? 1 : 0;
            }
        }
        out_request_lengths[flat_row] = valid_tokens;
    }
}

/**
 * @brief Fuse one controller-selected verifier row into graph-stable buffers.
 *
 * One warp owns the complete scalar-request transaction. Lane zero validates
 * the controller and publishes its scalar outputs; every lane then writes one
 * or more token/position columns. The physical launch is invariant while the
 * active prefix changes from M=2 through M=16 behind the controller pointer.
 */
__global__ void cuda_prepare_mtp_verifier_controlled_row_kernel(
    const int32_t *__restrict__ first_token,
    const int32_t *__restrict__ draft_tokens,
    const int32_t *__restrict__ base_position,
    int *__restrict__ generation_control_row,
    int padded_seq_len,
    int32_t *__restrict__ out_tokens,
    int32_t *__restrict__ out_position_ids,
    int32_t *__restrict__ out_request_length,
    int32_t *__restrict__ out_base_position_snapshot)
{
    __shared__ int active_rows;
    if (threadIdx.x == 0)
    {
        const bool active_request =
            generation_control_row[
                llaminar2::sampling_math::kDeviceGenerationControlOk] != 0 &&
            generation_control_row[
                llaminar2::sampling_math::
                    kDeviceGenerationControlRequestComplete] == 0;
        const int depth = generation_control_row[
            llaminar2::sampling_math::
                kDeviceGenerationControlCurrentDraftDepth];
        const int verifier_rows = generation_control_row[
            llaminar2::sampling_math::
                kDeviceGenerationControlActiveVerifierRowCount];
        if (!active_request || depth <= 0 || verifier_rows != depth + 1 ||
            verifier_rows > padded_seq_len)
        {
            llaminar2::sampling_math::fail_device_generation_control(
                generation_control_row,
                llaminar2::sampling_math::
                    DeviceGenerationError::InvalidDepthSelector);
            active_rows = 0;
        }
        else
        {
            active_rows = verifier_rows;
        }
        *out_request_length = active_rows;
        *out_base_position_snapshot = *base_position;
    }
    __syncthreads();

    const int base = *base_position;
    for (int row = static_cast<int>(threadIdx.x); row < padded_seq_len;
         row += static_cast<int>(blockDim.x))
    {
        out_position_ids[row] = base + row;
        out_tokens[row] =
            row == 0 ? *first_token
                     : row < active_rows ? draft_tokens[row - 1] : 0;
    }
}

/**
 * @brief Publish the first device-owned logical rows from terminal prefill samples.
 *
 * Request admission has already copied prompt positions into persistent device
 * metadata, and the sampler has written token identities into device slots. One
 * thread loads both device-owned values and publishes every mailbox field for one
 * request, making the initialized row self-consistent at the event recorded after
 * this launch. `target_positions` may alias `out_target_positions`; consequently
 * those two parameters intentionally do not use `__restrict__`.
 */
__global__ void cuda_initialize_mtp_device_logical_state_kernel(
    const int32_t *__restrict__ sampled_tokens,
    const int32_t *target_positions,
    int request_count,
    int32_t *__restrict__ out_base_cached_tokens,
    int32_t *out_target_positions,
    int32_t *__restrict__ out_accepted_state_counts,
    int32_t *__restrict__ out_next_condition_tokens,
    int32_t *__restrict__ out_all_drafts_accepted_flags,
    int32_t *__restrict__ out_stopped_flags,
    int32_t *__restrict__ out_publication_ok_flags)
{
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= request_count)
        return;

    const int position = target_positions[request];
    out_base_cached_tokens[request] = position;
    out_target_positions[request] = position;
    out_accepted_state_counts[request] = 0;
    out_next_condition_tokens[request] = sampled_tokens[request];
    out_all_drafts_accepted_flags[request] = 0;
    out_stopped_flags[request] = 0;
    out_publication_ok_flags[request] = 1;
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

/**
 * @brief Copy optional explicit threshold rows into a graph-owned launch value.
 *
 * Seeded production verification passes null pointers because the kernel reads
 * device-resident logical positions and derives every draw itself. Integration
 * probes may provide explicit host values; copying them into the kernel
 * parameter block avoids retaining any host pointer across graph capture.
 */
static llaminar2::sampling_math::SpeculativeBatchThresholdParameters
packCudaSpeculativeBatchThresholds(
    const float *accept_thresholds,
    const float *residual_thresholds,
    int row_count)
{
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters packed{};
    for (int row = 0; row < row_count; ++row)
    {
        if (accept_thresholds)
            packed.accept[row] = accept_thresholds[row];
        if (residual_thresholds)
            packed.residual[row] = residual_thresholds[row];
    }
    return packed;
}

/**
 * @brief Validate a launch width accepted by the warp-synchronous reducer.
 */
static bool validCudaArgmaxThreadCount(int threads)
{
    return threads >= 32 && threads <= 1024 &&
           (threads & (threads - 1)) == 0 &&
           threads % 32 == 0;
}

/**
 * @brief Launch one explicit two-pass argmax geometry.
 *
 * This function is shared by the stable production entry point and the
 * focused Perf__ geometry sweep.  Keeping one launcher prevents the benchmark
 * from accidentally timing a test-only kernel.  The benchmark varies only
 * immutable graph-capture geometry; production calls it with compile-time
 * selected constants and performs no runtime autotuning.
 */
static bool launchCudaArgmaxF32BatchedRowsGeometry(
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
    void *stream,
    int output_stride,
    int reduce_threads,
    int elements_per_thread,
    int finalize_threads,
    int *chain_condition_tokens,
    int *chain_position_ids,
    int chain_position_increment)
{
    const bool publish_mtp_chain =
        chain_condition_tokens != nullptr || chain_position_ids != nullptr ||
        chain_position_increment != 0;
    if (rows <= 0 || cols <= 0 || row_stride < cols ||
        !data || !out_values || !out_indices ||
        !partial_vals || !partial_idxs || partial_capacity < rows ||
        !stream || output_stride <= 0 || elements_per_thread <= 0 ||
        !validCudaArgmaxThreadCount(reduce_threads) ||
        !validCudaArgmaxThreadCount(finalize_threads) ||
        (publish_mtp_chain &&
         (!chain_condition_tokens || !chain_position_ids ||
          chain_position_increment <= 0)))
    {
        return false;
    }

    cudaSetDevice(device_idx);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int row_partial_capacity = partial_capacity / rows;
    if (row_partial_capacity < 1)
        return false;

    const long per_block =
        static_cast<long>(reduce_threads) * elements_per_thread;
    long blocks = (static_cast<long>(cols) + per_block - 1) / per_block;
    if (blocks < 1)
        blocks = 1;
    if (blocks > row_partial_capacity)
        blocks = row_partial_capacity;
    const int num_blocks = static_cast<int>(blocks);

    const size_t partial_shared_bytes =
        static_cast<size_t>(reduce_threads / 32) *
        (sizeof(float) + sizeof(int));
    cuda_argmax_partial_f32_batched_rows_kernel<<<
        dim3(num_blocks, rows),
        reduce_threads,
        partial_shared_bytes,
        s>>>(
        data,
        rows,
        cols,
        row_stride,
        partial_vals,
        partial_idxs,
        row_partial_capacity);

    const size_t finalize_shared_bytes =
        static_cast<size_t>(finalize_threads / 32) *
        (sizeof(float) + sizeof(int));
    if (publish_mtp_chain)
    {
        cuda_argmax_finalize_f32_batched_rows_kernel<true><<<
            rows,
            finalize_threads,
            finalize_shared_bytes,
            s>>>(
            partial_vals,
            partial_idxs,
            num_blocks,
            row_partial_capacity,
            out_values,
            out_indices,
            output_stride,
            chain_condition_tokens,
            chain_position_ids,
            chain_position_increment,
            nullptr);
    }
    else
    {
        cuda_argmax_finalize_f32_batched_rows_kernel<false><<<
            rows,
            finalize_threads,
            finalize_shared_bytes,
            s>>>(
            partial_vals,
            partial_idxs,
            num_blocks,
            row_partial_capacity,
            out_values,
            out_indices,
            output_stride,
            nullptr,
            nullptr,
            0,
            nullptr);
    }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Argmax FP32 batched rows launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

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

        const cudaError_t err = cudaGetLastError();
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
        void *stream,
        int output_stride)
    {
        return launchCudaArgmaxF32BatchedRowsGeometry(
            data,
            rows,
            cols,
            row_stride,
            out_values,
            out_indices,
            partial_vals,
            partial_idxs,
            partial_capacity,
            device_idx,
            stream,
            output_stride,
            ARGMAX_REDUCE_THREADS,
            ARGMAX_ELEMS_PER_THREAD,
            ARGMAX_FINALIZE_THREADS,
            nullptr,
            nullptr,
            0);
    }

    bool cudaOps_argmax_f32_batched_rows_publish_mtp_chain(
        const float *data,
        int rows,
        int cols,
        int row_stride,
        float *out_values,
        int *out_indices,
        int *chain_condition_tokens,
        int *chain_position_ids,
        int chain_position_increment,
        float *partial_vals,
        int *partial_idxs,
        int partial_capacity,
        int device_idx,
        void *stream,
        int output_stride)
    {
        return launchCudaArgmaxF32BatchedRowsGeometry(
            data,
            rows,
            cols,
            row_stride,
            out_values,
            out_indices,
            partial_vals,
            partial_idxs,
            partial_capacity,
            device_idx,
            stream,
            output_stride,
            ARGMAX_REDUCE_THREADS,
            ARGMAX_ELEMS_PER_THREAD,
            ARGMAX_FINALIZE_THREADS,
            chain_condition_tokens,
            chain_position_ids,
            chain_position_increment);
    }

    /**
     * @brief Perf__ entry point for explicit argmax launch geometry.
     *
     * This is intentionally not surfaced through IBackend: inference always
     * uses the stable winner above.  The performance suite keeps this symbol
     * live and verifies every candidate against the same byte-exact output.
     */
    bool cudaOps_argmax_f32_batched_rows_geometry(
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
        void *stream,
        int output_stride,
        int reduce_threads,
        int elements_per_thread,
        int finalize_threads)
    {
        return launchCudaArgmaxF32BatchedRowsGeometry(
            data,
            rows,
            cols,
            row_stride,
            out_values,
            out_indices,
            partial_vals,
            partial_idxs,
            partial_capacity,
            device_idx,
            stream,
            output_stride,
            reduce_threads,
            elements_per_thread,
            finalize_threads,
            nullptr,
            nullptr,
            0);
    }

    /**
     * @brief Perf__ entry point for producer-owned MTP-chain publication.
     *
     * The candidate geometry reaches the exact production specialization that
     * writes both the selected draft token and the next chained-sidecar
     * position. Keeping this entry point out of IBackend prevents runtime
     * autotuning while still allowing the stable constants above to be selected
     * from isolated, byte-checked measurements.
     */
    bool cudaOps_argmax_f32_batched_rows_publish_mtp_chain_geometry(
        const float *data,
        int rows,
        int cols,
        int row_stride,
        float *out_values,
        int *out_indices,
        int *chain_condition_tokens,
        int *chain_position_ids,
        int chain_position_increment,
        float *partial_vals,
        int *partial_idxs,
        int partial_capacity,
        int device_idx,
        void *stream,
        int output_stride,
        int reduce_threads,
        int elements_per_thread,
        int finalize_threads)
    {
        return launchCudaArgmaxF32BatchedRowsGeometry(
            data,
            rows,
            cols,
            row_stride,
            out_values,
            out_indices,
            partial_vals,
            partial_idxs,
            partial_capacity,
            device_idx,
            stream,
            output_stride,
            reduce_threads,
            elements_per_thread,
            finalize_threads,
            chain_condition_tokens,
            chain_position_ids,
            chain_position_increment);
    }

    bool cudaOps_configure_mtp_greedy_penalty_policy(
        llaminar2::MTPGreedyPenaltyPolicy *controls,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_idx,
        void *stream)
    {
        if (!controls || !stream)
            return false;

        cudaSetDevice(device_idx);
        cuda_configure_mtp_greedy_penalty_policy_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            controls,
            presence_penalty,
            frequency_penalty,
            first_token_already_in_history ? 1 : 0);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP greedy penalty policy launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_argmax_f32_batched_rows_mtp_penalties(
        const float *data,
        int rows,
        int cols,
        int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        float *out_values,
        int *out_indices,
        float *partial_vals,
        int *partial_idxs,
        int partial_capacity,
        int device_idx,
        void *stream,
        int output_stride)
    {
        if (rows <= 0 || cols <= 0 || row_stride < cols ||
            !data || !verifier_input_tokens || !generated_token_counts ||
            !policy || !active_rows || !out_values || !out_indices ||
            !partial_vals || !partial_idxs || partial_capacity < rows ||
            !stream || output_stride <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const int row_partial_capacity = partial_capacity / rows;
        if (row_partial_capacity < 1)
            return false;

        const int threads = ARGMAX_REDUCE_THREADS;
        const long per_block =
            static_cast<long>(threads) * ARGMAX_ELEMS_PER_THREAD;
        long blocks =
            (static_cast<long>(cols) + per_block - 1) / per_block;
        if (blocks < 1)
            blocks = 1;
        if (blocks > row_partial_capacity)
            blocks = row_partial_capacity;
        const int num_blocks = static_cast<int>(blocks);
        const size_t shared_bytes =
            static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
        cuda_argmax_partial_f32_batched_rows_mtp_penalty_kernel<<<
            dim3(num_blocks, rows),
            dim3(threads),
            shared_bytes,
            static_cast<cudaStream_t>(stream)>>>(
            data,
            rows,
            cols,
            row_stride,
            verifier_input_tokens,
            generated_token_counts,
            policy,
            active_rows,
            partial_vals,
            partial_idxs,
            row_partial_capacity);
        cuda_argmax_finalize_f32_batched_rows_kernel<false, true><<<
            rows,
            ARGMAX_FINALIZE_THREADS,
            static_cast<size_t>(ARGMAX_FINALIZE_THREADS) *
                (sizeof(float) + sizeof(int)),
            static_cast<cudaStream_t>(stream)>>>(
            partial_vals,
            partial_idxs,
            num_blocks,
            row_partial_capacity,
            out_values,
            out_indices,
            output_stride,
            nullptr,
            nullptr,
            0,
            active_rows);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP penalty-aware batched argmax launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_apply_mtp_penalties_f32_rows(
        float *data,
        int rows,
        int cols,
        int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        int device_idx,
        void *stream)
    {
        if (!data || rows <= 0 || cols <= 0 || row_stride < cols ||
            !generated_token_counts || !policy || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int blocks = (cols + threads - 1) / threads;
        blocks = blocks > 1024 ? 1024 : blocks;
        if (verifier_input_tokens)
        {
            cuda_apply_mtp_verifier_penalties_f32_rows_kernel<<<
                dim3(blocks + 1),
                dim3(threads),
                0,
                static_cast<cudaStream_t>(stream)>>>(
                data,
                rows,
                cols,
                row_stride,
                verifier_input_tokens,
                generated_token_counts,
                policy,
                active_rows);
        }
        else
        {
            cuda_apply_mtp_durable_penalties_f32_rows_kernel<<<
                dim3(blocks),
                dim3(threads),
                0,
                static_cast<cudaStream_t>(stream)>>>(
                data,
                rows,
                cols,
                row_stride,
                generated_token_counts,
                policy,
                active_rows);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP penalty row transform launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool cudaOps_apply_mtp_branch_penalties_f32_row(
        float *data,
        int cols,
        const int *first_condition_token,
        const int *prior_draft_tokens,
        int prior_draft_count,
        const int *generated_token_counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        int device_idx,
        void *stream)
    {
        if (!data || cols <= 0 || !first_condition_token ||
            prior_draft_count < 0 ||
            (prior_draft_count > 0 && !prior_draft_tokens) ||
            !generated_token_counts || !policy || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int dense_blocks = (cols + threads - 1) / threads;
        dense_blocks = dense_blocks > 1024 ? 1024 : dense_blocks;
        cuda_apply_mtp_branch_penalties_f32_row_kernel<<<
            dim3(dense_blocks + 1),
            dim3(threads),
            0,
            static_cast<cudaStream_t>(stream)>>>(
            data,
            cols,
            first_condition_token,
            prior_draft_tokens,
            prior_draft_count,
            generated_token_counts,
            policy);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP branch penalty row transform launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_commit_mtp_greedy_penalty_history(
        const int *output_tokens,
        const int *output_meta,
        llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *accepted_state_counts,
        const int *stopped_flags,
        int output_token_capacity,
        int vocab_size,
        int *generated_token_counts,
        int device_idx,
        void *stream)
    {
        if (!output_tokens || !output_meta || !policy ||
            !accepted_state_counts || !stopped_flags ||
            output_token_capacity <= 0 || vocab_size <= 0 ||
            !generated_token_counts || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_commit_mtp_greedy_penalty_history_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            output_tokens,
            output_meta,
            policy,
            accepted_state_counts,
            stopped_flags,
            output_token_capacity,
            vocab_size,
            generated_token_counts);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP greedy penalty history commit launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_publish_int32_control_scalar(
        int32_t value,
        int32_t *out_value,
        int device_idx,
        void *stream)
    {
        if (value < 0 || !out_value || !stream)
            return false;

        cudaSetDevice(device_idx);
        cuda_publish_int32_control_scalar_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(value, out_value);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA INT32 control-scalar publication launch failed: %s\n",
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
        const int *active_rows,
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
                const int maximum_partial_span =
                    (n + partial_blocks - 1) / partial_blocks;
                const bool use_cooperative_selector =
                    maximum_partial_span <= TOPK_BATCHED_COOPERATIVE_CAPACITY &&
                    partial_blocks * k <= TOPK_BATCHED_COOPERATIVE_CAPACITY;
                if (use_cooperative_selector && k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_cooperative_partials_batched_f32_kernel<
                        TOPK_MEDIUM_K_CAP>
                        <<<partial_grid,
                           TOPK_BATCHED_COOPERATIVE_THREADS,
                           0,
                           s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, active_rows);
                }
                else if (use_cooperative_selector)
                {
                    cuda_topk_cooperative_partials_batched_f32_kernel<
                        TOPK_SMALL_K_CAP>
                        <<<partial_grid,
                           TOPK_BATCHED_COOPERATIVE_THREADS,
                           0,
                           s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, active_rows);
                }
                else if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, active_rows);
                }
                else
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, active_rows);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "CUDA batched small-k Top-K partial FP32 kernel launch failed: %s\n",
                            cudaGetErrorString(err));
                    return false;
                }

                if (use_cooperative_selector && k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_topp_distribution_cooperative_batched_f32_kernel<
                        TOPK_MEDIUM_K_CAP>
                        <<<row_count,
                           TOPK_BATCHED_COOPERATIVE_THREADS,
                           0,
                           s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride,
                            out_probs, active_rows);
                }
                else if (use_cooperative_selector)
                {
                    cuda_topk_topp_distribution_cooperative_batched_f32_kernel<
                        TOPK_SMALL_K_CAP>
                        <<<row_count,
                           TOPK_BATCHED_COOPERATIVE_THREADS,
                           0,
                           s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride,
                            out_probs, active_rows);
                }
                else if (k <= TOPK_MEDIUM_K_CAP)
                {
                    cuda_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride, out_probs,
                            active_rows);
                }
                else
                {
                    cuda_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<row_count, threads, smem_size, s>>>(
                            scratch_values, scratch_indices, partial_blocks, k,
                            top_p, temperature, out_token_ids, out_stride, out_probs,
                            active_rows);
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
                            scratch_values, scratch_indices, nullptr);
                }
                else
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, nullptr);
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
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx,
        void *stream)
    {
        if (k <= 0 || k > TOPK_MAX_K || !token_ids || !probs || !out_token ||
            !stream || (threshold_position && threshold_seed == 0))
            return false;

        cudaSetDevice(device_idx);

        cuda_sample_distribution_f32_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            token_ids,
            probs,
            k,
            threshold,
            threshold_seed,
            threshold_position,
            threshold_position_offset,
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

    bool cudaOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        const int *first_token_device,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        int *out_token,
        float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx,
        void *stream)
    {
        if (!logits || !verify_tokens || !verify_accepted || !out_token ||
            !stream || vocab_size <= 0 || row_stride < vocab_size ||
            row_count < 0 ||
            (first_token < 0 && !first_token_device) ||
            stop_token_count < 0 ||
            stop_token_count >
                llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
            (threshold_position && threshold_seed == 0))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_sample_processed_logits_if_speculative_batch_needs_bonus_f32_kernel<<<
            1,
            PROCESSED_LOGIT_VERIFY_THREADS,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            logits,
            vocab_size,
            row_stride,
            threshold,
            threshold_seed,
            threshold_position,
            threshold_position_offset,
            verify_tokens,
            verify_accepted,
            row_count,
            first_token,
            first_token_device,
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7,
            stop_token_count,
            out_token,
            out_probability);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Lazy Processed-Logit Bonus Sample kernel launch failed: %s\n",
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
            row_count <= 0 ||
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
        const int *draft_tokens_host,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
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
            row_count <= 0 ||
            !target_token_ids || !target_probs ||
            !draft_token_ids || !draft_probs ||
            !draft_tokens_host || !accept_thresholds_host ||
            !residual_thresholds_host ||
            !out_token || !out_accepted || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);

        constexpr int kRowsPerLaunch =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        for (int row_base = 0; row_base < row_count;
             row_base += kRowsPerLaunch)
        {
            const int launch_rows =
                std::min(kRowsPerLaunch, row_count - row_base);
            llaminar2::sampling_math::SpeculativeBatchHostParameters
                parameters{};
            for (int row = 0; row < launch_rows; ++row)
            {
                parameters.draft_tokens[row] =
                    draft_tokens_host[row_base + row];
                parameters.thresholds.accept[row] =
                    accept_thresholds_host[row_base + row];
                parameters.thresholds.residual[row] =
                    residual_thresholds_host[row_base + row];
            }

            const size_t row_offset =
                static_cast<size_t>(row_base) *
                static_cast<size_t>(distribution_stride);
            cuda_speculative_verify_distribution_thresholds_batch_kernel<<<
                1,
                kRowsPerLaunch,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_token_ids + row_offset,
                target_probs + row_offset,
                draft_token_ids + row_offset,
                draft_probs + row_offset,
                k,
                distribution_stride,
                parameters,
                launch_rows,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Speculative Verify Batch FP32 kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int inverse_sample_vocab_size,
        unsigned long long threshold_seed,
        int threshold_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream)
    {
        const bool has_draft_distribution =
            draft_token_ids != nullptr && draft_probs != nullptr;
        const bool has_one_hot_draft_distribution =
            draft_token_ids == nullptr && draft_probs == nullptr;
        if (k <= 0 || k > TOPK_MAX_K ||
            distribution_stride < k ||
            row_count <= 0 ||
            !target_token_ids || !target_probs ||
            (!has_draft_distribution && !has_one_hot_draft_distribution) ||
            !sampled_draft_tokens ||
            (threshold_base_position && !thresholds_from_seed) ||
            !out_token || !out_accepted || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);

        constexpr int kRuntimeVerifierThreads = 128;
        constexpr int kExplicitThresholdRows =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        const bool use_seeded_thresholds = thresholds_from_seed != 0;
        if (!use_seeded_thresholds &&
            (!accept_thresholds_host || !residual_thresholds_host))
        {
            return false;
        }

        const int launch_count = use_seeded_thresholds
                                     ? 1
                                     : (row_count + kExplicitThresholdRows - 1) /
                                           kExplicitThresholdRows;
        for (int launch = 0; launch < launch_count; ++launch)
        {
            const int row_base = use_seeded_thresholds
                                     ? 0
                                     : launch * kExplicitThresholdRows;
            const int launch_rows = use_seeded_thresholds
                                        ? row_count
                                        : std::min(kExplicitThresholdRows,
                                                   row_count - row_base);
            llaminar2::sampling_math::SpeculativeBatchThresholdParameters
                thresholds{};
            if (!use_seeded_thresholds)
            {
                for (int row = 0; row < launch_rows; ++row)
                {
                    thresholds.accept[row] =
                        accept_thresholds_host[row_base + row];
                    thresholds.residual[row] =
                        residual_thresholds_host[row_base + row];
                }
            }

            const int threads = use_seeded_thresholds
                                    ? kRuntimeVerifierThreads
                                    : kExplicitThresholdRows;
            const int blocks = (launch_rows + threads - 1) / threads;
            const size_t row_offset =
                static_cast<size_t>(row_base) *
                static_cast<size_t>(distribution_stride);
            cuda_speculative_verify_distribution_thresholds_batch_device_tokens_kernel<<<
                blocks,
                threads,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_token_ids + row_offset,
                target_probs + row_offset,
                draft_token_ids ? draft_token_ids + row_offset : nullptr,
                draft_probs ? draft_probs + row_offset : nullptr,
                k,
                distribution_stride,
                sampled_draft_tokens + row_base,
                sampled_draft_probabilities
                    ? sampled_draft_probabilities + row_base
                    : nullptr,
                thresholds,
                launch_rows,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row_base,
                inverse_sample_vocab_size,
                threshold_seed,
                threshold_first_logical_position + row_base,
                thresholds_from_seed,
                threshold_base_position,
                threshold_position_offset + row_base,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Speculative Verify Batch Device Tokens FP32 kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
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
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host || !residual_thresholds_host)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerLaunch =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        for (int row_base = 0; row_base < row_count;
             row_base += kRowsPerLaunch)
        {
            const int launch_rows =
                std::min(kRowsPerLaunch, row_count - row_base);
            const auto thresholds = packCudaSpeculativeBatchThresholds(
                accept_thresholds_host + row_base,
                residual_thresholds_host + row_base,
                launch_rows);
            cuda_speculative_verify_processed_logits_thresholds_batch_device_tokens_kernel<<<
                launch_rows,
                PROCESSED_LOGIT_VERIFY_THREADS,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_logits +
                    static_cast<size_t>(row_base) * target_row_stride,
                draft_logits +
                    static_cast<size_t>(row_base) * draft_row_stride,
                launch_rows,
                vocab_size,
                target_row_stride,
                draft_row_stride,
                sampled_draft_tokens + row_base,
                thresholds,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr,
                draft_token_probabilities
                    ? draft_token_probabilities + row_base
                    : nullptr);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Processed-Logit Stochastic Speculative Verify kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        const float *accept_thresholds_host,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
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
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            (threshold_base_position && !thresholds_from_seed))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerLaunch =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        const bool use_seeded_thresholds = thresholds_from_seed != 0;
        if (!use_seeded_thresholds && !accept_thresholds_host)
            return false;

        const int launch_count = use_seeded_thresholds
                                     ? 1
                                     : (row_count + kRowsPerLaunch - 1) /
                                           kRowsPerLaunch;
        for (int launch = 0; launch < launch_count; ++launch)
        {
            const int row_base = use_seeded_thresholds
                                     ? 0
                                     : launch * kRowsPerLaunch;
            const int launch_rows = use_seeded_thresholds
                                        ? row_count
                                        : std::min(kRowsPerLaunch,
                                                   row_count - row_base);
            const auto thresholds = packCudaSpeculativeBatchThresholds(
                use_seeded_thresholds
                    ? nullptr
                    : accept_thresholds_host + row_base,
                /*residual_thresholds=*/nullptr,
                use_seeded_thresholds ? 0 : launch_rows);
            cuda_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_kernel<<<
                launch_rows,
                PROCESSED_LOGIT_VERIFY_THREADS,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_logits +
                    static_cast<size_t>(row_base) * target_row_stride,
                no_draft_probabilities
                    ? nullptr
                    : draft_probabilities +
                          static_cast<size_t>(row_base) * draft_row_stride,
                launch_rows,
                vocab_size,
                target_row_stride,
                draft_row_stride,
                sampled_draft_tokens + row_base,
                thresholds,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row_base,
                thresholds_from_seed,
                threshold_base_position,
                threshold_position_offset + row_base,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr,
                no_draft_probabilities);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Processed-Target/Draft-Probability Stochastic Verify kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        const float *accept_thresholds_host,
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
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerLaunch =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        for (int row_base = 0; row_base < row_count;
             row_base += kRowsPerLaunch)
        {
            const int launch_rows =
                std::min(kRowsPerLaunch, row_count - row_base);
            const auto thresholds = packCudaSpeculativeBatchThresholds(
                accept_thresholds_host + row_base,
                /*residual_thresholds=*/nullptr,
                launch_rows);
            cuda_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_kernel<<<
                launch_rows,
                PROCESSED_LOGIT_VERIFY_THREADS,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_logits +
                    static_cast<size_t>(row_base) * target_row_stride,
                draft_logits +
                    static_cast<size_t>(row_base) * draft_row_stride,
                launch_rows,
                vocab_size,
                target_row_stride,
                draft_row_stride,
                sampled_draft_tokens + row_base,
                sampled_draft_probabilities
                    ? sampled_draft_probabilities + row_base
                    : nullptr,
                thresholds,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row_base,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Processed-Target/Draft-Logit Stochastic Verify kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        const float *accept_thresholds_host,
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
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            inverse_sample_row_stride < vocab_size ||
            !accept_thresholds_host)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kRowsPerLaunch =
            llaminar2::sampling_math::kSpeculativeBatchMaxRows;
        for (int row_base = 0; row_base < row_count;
             row_base += kRowsPerLaunch)
        {
            const int launch_rows =
                std::min(kRowsPerLaunch, row_count - row_base);
            const auto thresholds = packCudaSpeculativeBatchThresholds(
                accept_thresholds_host + row_base,
                /*residual_thresholds=*/nullptr,
                launch_rows);
            cuda_speculative_verify_probabilities_thresholds_batch_device_tokens_kernel<<<
                launch_rows,
                PROCESSED_LOGIT_VERIFY_THREADS,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                target_probabilities +
                    static_cast<size_t>(row_base) * target_row_stride,
                no_draft_probabilities
                    ? nullptr
                    : draft_probabilities +
                          static_cast<size_t>(row_base) * draft_row_stride,
                inverse_rejection_samples +
                    static_cast<size_t>(row_base) * inverse_sample_row_stride,
                launch_rows,
                vocab_size,
                target_row_stride,
                draft_row_stride,
                inverse_sample_row_stride,
                sampled_draft_tokens + row_base,
                thresholds,
                no_draft_probabilities,
                out_token + row_base,
                out_accepted + row_base,
                out_accept_probability
                    ? out_accept_probability + row_base
                    : nullptr,
                out_accept_threshold
                    ? out_accept_threshold + row_base
                    : nullptr);

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA Full-Probability Stochastic Speculative Verify kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
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
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream)
    {
        if (row_count < 0 ||
            out_token_capacity < row_count + 1 ||
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
            out_token_capacity,
            out_meta,
            max_state_commit_rows,
            leading_committed_output_count);

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
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream)
    {
        if (row_count < 0 ||
            out_token_capacity < row_count + 1 ||
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
            out_token_capacity,
            out_meta,
            max_state_commit_rows,
            leading_committed_output_count);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Verify Batch Device-First Summary kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_summarize_speculative_verify_batch_device_generation_controls(
        const int *verify_tokens,
        const int *verify_accepted,
        const int *greedy_draft_tokens,
        int row_count,
        const int *first_token,
        const int *stop_tokens,
        const int *bonus_token,
        int has_bonus_token,
        const int *generation_control,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        int device_idx,
        void *stream)
    {
        const bool has_acceptance_rows = verify_accepted != nullptr;
        const bool has_greedy_rows = greedy_draft_tokens != nullptr;
        if (!verify_tokens || has_acceptance_rows == has_greedy_rows ||
            row_count < 0 || !first_token || !stop_tokens ||
            (has_bonus_token && !bonus_token) || !generation_control ||
            out_token_capacity < row_count + 1 || !out_tokens || !out_meta ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_summarize_speculative_verify_batch_device_generation_controls_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            verify_tokens,
            verify_accepted,
            greedy_draft_tokens,
            row_count,
            first_token,
            stop_tokens,
            bonus_token,
            has_bonus_token,
            generation_control,
            out_tokens,
            out_token_capacity,
            out_meta);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA device-generation speculative summary launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool
    cudaOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
        const int *target_token_ids,
        const float *target_probs,
        int target_row_stride,
        int top_k,
        int row_count,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        const int *verifier_input_tokens,
        const int *stop_tokens,
        const int *generation_control,
        int *sampled_target_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        void *first_transaction_diagnostic,
        int device_idx,
        void *stream)
    {
        constexpr int kMaxComparisonRows = 15;
        if (!target_token_ids || !target_probs ||
            target_row_stride <= 0 || top_k <= 0 ||
            top_k > target_row_stride || row_count <= 0 ||
            row_count > kMaxComparisonRows || threshold_seed == 0 ||
            !threshold_position || !verifier_input_tokens || !stop_tokens ||
            !generation_control || !sampled_target_tokens || !out_tokens ||
            out_token_capacity < row_count + 1 || !out_meta || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        using DiagnosticRecord =
            llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord;
        auto *diagnostic =
            static_cast<DiagnosticRecord *>(first_transaction_diagnostic);
        if (diagnostic)
        {
            cuda_sample_and_summarize_serial_equivalent_speculative_batch_kernel<
                true><<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
                target_token_ids,
                target_probs,
                target_row_stride,
                top_k,
                row_count,
                threshold_seed,
                threshold_position,
                threshold_position_offset,
                verifier_input_tokens,
                stop_tokens,
                generation_control,
                sampled_target_tokens,
                out_tokens,
                out_token_capacity,
                out_meta,
                diagnostic);
        }
        else
        {
            cuda_sample_and_summarize_serial_equivalent_speculative_batch_kernel<
                false><<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
                target_token_ids,
                target_probs,
                target_row_stride,
                top_k,
                row_count,
                threshold_seed,
                threshold_position,
                threshold_position_offset,
                verifier_input_tokens,
                stop_tokens,
                generation_control,
                sampled_target_tokens,
                out_tokens,
                out_token_capacity,
                out_meta,
                nullptr);
        }

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA fused serial-equivalent speculative outcome launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_retain_mtp_first_transaction_draft_boundary(
        const uint32_t *data_words,
        int word_count,
        int boundary,
        int draft_slot,
        const int *condition_token,
        const int *position_id,
        const int *generation_control,
        int generation_control_stride,
        void *diagnostic_record,
        int device_idx,
        void *stream)
    {
        using namespace llaminar2::sampling_math;
        if (word_count < 0 || (word_count > 0 && !data_words) ||
            boundary < 0 ||
            boundary >= kMTPFirstTransactionDraftBoundaryCount ||
            draft_slot < 0 || draft_slot >= kSpeculativeBatchMaxRows ||
            !condition_token || !position_id || !generation_control ||
            generation_control_stride < kDeviceGenerationControlCount ||
            !diagnostic_record || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_retain_mtp_first_transaction_draft_boundary_kernel<<<
            1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            data_words,
            word_count,
            boundary,
            draft_slot,
            condition_token,
            position_id,
            generation_control,
            generation_control_stride,
            static_cast<MTPFirstTransactionDiagnosticRecord *>(
                diagnostic_record));

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA first-transaction MTP draft-boundary diagnostic launch failed: %s\n",
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
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream)
    {
        if (compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
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
            out_token_capacity,
            out_meta,
            max_state_commit_rows,
            leading_committed_output_count);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Greedy Speculative Verify Batch Summary kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_summarize_greedy_speculative_verify_batch_device_controls(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *active_verifier_row_count,
        const int *stop_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        const llaminar2::MTPGreedyPenaltyPolicy *penalty_policy,
        int device_idx,
        void *stream)
    {
        if (compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
            !verify_tokens || !draft_tokens || !active_verifier_row_count ||
            !stop_tokens || !penalty_policy ||
            !out_tokens || !out_meta || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_summarize_greedy_speculative_verify_batch_device_controls_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            verify_tokens,
            draft_tokens,
            compare_row_count,
            active_verifier_row_count,
            stop_tokens,
            out_tokens,
            out_token_capacity,
            out_meta,
            max_state_commit_rows,
            penalty_policy);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA device-control greedy speculative summary launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_advance_speculative_commit_boundary(
        const int *meta,
        int request_count,
        int meta_stride,
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        if (!meta || request_count <= 0 ||
            meta_stride <
                llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            !decode_rounds_committed ||
            !decode_rounds_until_maintenance ||
            !maintenance_due || !decode_boundary_advanced || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_advance_speculative_commit_boundary_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            request_count,
            meta_stride,
            decode_rounds_committed,
            decode_rounds_until_maintenance,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA speculative commit-boundary advance launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_initialize_device_generation(
        int request_count,
        int max_new_tokens,
        const llaminar2::sampling_math::DeviceGenerationDepthPolicy &depth_policy,
        int response_token_stride,
        int control_stride,
        int *control,
        int device_idx,
        void *stream)
    {
        if (request_count <= 0 || max_new_tokens <= 0 ||
            !depth_policy.valid() ||
            response_token_stride < max_new_tokens ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            !control || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // One request owns one scalar controller row. Production generation is
        // normally batch one, so one warp avoids launching three immediately
        // retired warps while retaining grid totality for larger request sets.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_initialize_device_generation_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            request_count,
            max_new_tokens,
            depth_policy,
            response_token_stride,
            control_stride,
            control);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA device-generation initialization launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_prepare_device_generation_transaction_budget(
        int *control,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const uint32_t *maintenance_rows_remaining,
        const uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        const bool has_maintenance_boundary =
            maintenance_rows_remaining || maintenance_due ||
            decode_boundary_advanced;
        const bool has_complete_maintenance_boundary =
            maintenance_rows_remaining && maintenance_due &&
            decode_boundary_advanced;
        if (!control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || verifier_row_capacity <= 0 || !stream ||
            (has_maintenance_boundary &&
             !has_complete_maintenance_boundary))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // Keep the tiny row transition to one physical CUDA warp per block.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_prepare_device_generation_transaction_budget_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            control,
            control_stride,
            request_count,
            verifier_row_capacity,
            maintenance_rows_remaining,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA device-generation budget launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool
    cudaOps_commit_device_generation_and_derive_speculative_publication_metadata(
        const int32_t *compact_tokens,
        int output_token_stride,
        int *compact_meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int32_t *response_tokens,
        int response_token_stride,
        int *control,
        int control_stride,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_sidecar_condition_tokens,
        int *out_next_sidecar_position_ids,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream)
    {
        if (!compact_tokens || output_token_stride <= 0 || !compact_meta ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            !base_cached_tokens || request_count <= 0 ||
            padded_state_rows_per_request <= 0 ||
            !response_tokens || response_token_stride <= 0 ||
            !control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            !out_restore_rows || !out_target_cached_tokens ||
            !out_accepted_state_counts || !out_ok ||
            ((out_next_sidecar_condition_tokens ||
              out_next_sidecar_position_ids) &&
             (!out_next_sidecar_condition_tokens ||
              !out_next_sidecar_position_ids ||
              !out_next_condition_tokens)) ||
            (out_next_verifier_condition_tokens &&
             !out_next_condition_tokens) ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // The fused transaction is request-independent. One physical warp is
        // the minimum economical CUDA launch for the production batch-one case;
        // larger batches retain grid totality without divergent cooperation.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_commit_device_generation_and_derive_speculative_publication_metadata_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            compact_tokens,
            output_token_stride,
            compact_meta,
            meta_stride,
            base_cached_tokens,
            request_count,
            padded_state_rows_per_request,
            response_tokens,
            response_token_stride,
            control,
            control_stride,
            out_restore_rows,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok,
            reinterpret_cast<int32_t *>(out_next_condition_tokens),
            out_all_drafts_accepted_flags,
            out_stopped_flags,
            reinterpret_cast<int32_t *>(
                out_next_sidecar_condition_tokens),
            reinterpret_cast<int32_t *>(out_next_sidecar_position_ids),
            reinterpret_cast<int32_t *>(
                out_next_verifier_condition_tokens));
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA fused device-generation publication launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int max_state_commit_rows,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        const int32_t *output_tokens,
        int output_token_stride,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream)
    {
        if (!meta || !base_cached_tokens ||
            !out_restore_rows || !out_target_cached_tokens ||
            !out_accepted_state_counts || !out_ok || !stream ||
            ((out_next_condition_tokens || output_tokens) &&
             (!out_next_condition_tokens || !output_tokens || output_token_stride <= 0)) ||
            (out_next_verifier_condition_tokens &&
             !out_next_condition_tokens) ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            request_count <= 0 ||
            padded_state_rows_per_request <= 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_derive_speculative_publication_metadata_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            meta_stride,
            base_cached_tokens,
            request_count,
            padded_state_rows_per_request,
            max_state_commit_rows,
            out_restore_rows,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok,
            reinterpret_cast<int32_t *>(out_next_condition_tokens),
            output_tokens,
            output_token_stride,
            out_all_drafts_accepted_flags,
            out_stopped_flags,
            reinterpret_cast<int32_t *>(
                out_next_verifier_condition_tokens));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Publication Metadata kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_derive_shifted_speculative_publication_metadata_from_primary(
        const int *base_cached_tokens,
        const int *main_target_cached_tokens,
        const int *main_publication_ok,
        int request_count,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int device_idx,
        void *stream)
    {
        if (!base_cached_tokens ||
            !main_target_cached_tokens ||
            !main_publication_ok ||
            !out_target_cached_tokens ||
            !out_accepted_state_counts ||
            !out_ok ||
            !stream ||
            request_count <= 0 ||
            mtp_depth < 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_derive_shifted_speculative_publication_metadata_from_primary_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_cached_tokens,
            main_target_cached_tokens,
            main_publication_ok,
            request_count,
            mtp_depth,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Shifted Speculative Publication Metadata kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_prepare_speculative_shifted_kv_tokens(
        const int *meta,
        int meta_stride,
        const int32_t *output_tokens,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int32_t *out_tokens,
        const int32_t *base_positions,
        int position_offset,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!meta ||
            !output_tokens ||
            !out_tokens ||
            !base_positions ||
            !out_position_ids ||
            !stream ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            output_token_stride <= first_output_token_index ||
            request_index < 0 ||
            first_output_token_index < 0 ||
            row_count <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_prepare_speculative_shifted_kv_tokens_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            meta_stride,
            output_tokens,
            output_token_stride,
            request_index,
            first_output_token_index,
            row_count,
            filler_token,
            out_tokens,
            base_positions,
            position_offset,
            out_position_ids);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Shifted KV Token Prep kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Enqueue device-resident preparation for one grouped MTP sidecar.
     */
    bool cudaOps_prepare_mtp_batched_sidecar_inputs(
        const int32_t *condition_tokens,
        int condition_token_stride,
        const int32_t *base_positions,
        int position_offset,
        int request_count,
        int32_t *out_condition_tokens,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!condition_tokens ||
            condition_token_stride <= 0 ||
            !base_positions ||
            request_count <= 0 ||
            !out_condition_tokens ||
            !out_position_ids ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        cuda_prepare_mtp_batched_sidecar_inputs_kernel<<<
            1,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            condition_tokens,
            condition_token_stride,
            base_positions,
            position_offset,
            request_count,
            out_condition_tokens,
            out_position_ids);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA batched MTP sidecar input preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue the position-only scalar/rectangular compatibility form. */
    bool cudaOps_prepare_mtp_verifier_position_ids(
        const int32_t *base_positions,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!base_positions ||
            request_count <= 0 ||
            padded_seq_len <= 0 ||
            !out_position_ids ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int total_rows = request_count * padded_seq_len;
        const int blocks =
            (total_rows + threads_per_block - 1) / threads_per_block;
        cuda_prepare_mtp_verifier_geometry_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_positions,
            /*valid_graph_rows=*/nullptr,
            /*valid_graph_row_count=*/0,
            /*generation_control=*/nullptr,
            /*generation_control_stride=*/0,
            request_count,
            padded_seq_len,
            out_position_ids,
            /*out_request_lengths=*/nullptr);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA grouped MTP verifier position preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Enqueue one device-resident grouped-verifier geometry publication.
     */
    bool cudaOps_prepare_mtp_verifier_geometry(
        const int32_t *base_positions,
        const int32_t *valid_graph_rows,
        int valid_graph_row_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int32_t *out_request_lengths,
        int device_idx,
        void *stream)
    {
        const int total_rows = request_count * padded_seq_len;
        const bool rectangular = valid_graph_rows == nullptr;
        const bool device_controlled = generation_control != nullptr;
        if (!base_positions || request_count <= 0 || padded_seq_len <= 0 ||
            !out_position_ids || !out_request_lengths || !stream ||
            (device_controlled != (generation_control_stride > 0)) ||
            (device_controlled &&
             generation_control_stride <
                 llaminar2::sampling_math::kDeviceGenerationControlCount) ||
            (rectangular &&
             (valid_graph_row_count <= 0 ||
              valid_graph_row_count > total_rows ||
              valid_graph_row_count % request_count != 0)) ||
            (!rectangular &&
             (valid_graph_row_count <= 0 || valid_graph_row_count > total_rows)))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int blocks =
            (total_rows + threads_per_block - 1) / threads_per_block;
        cuda_prepare_mtp_verifier_geometry_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_positions,
            valid_graph_rows,
            valid_graph_row_count,
            generation_control,
            generation_control_stride,
            request_count,
            padded_seq_len,
            out_position_ids,
            out_request_lengths);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA grouped MTP verifier geometry preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue one fused controller-owned verifier row publication. */
    bool cudaOps_prepare_mtp_verifier_controlled_row(
        const int32_t *first_token,
        const int32_t *draft_tokens,
        const int32_t *base_position,
        int *generation_control_row,
        int generation_control_stride,
        int padded_seq_len,
        int32_t *out_tokens,
        int32_t *out_position_ids,
        int32_t *out_request_length,
        int32_t *out_base_position_snapshot,
        int device_idx,
        void *stream)
    {
        if (!first_token || !draft_tokens || !base_position ||
            !generation_control_row ||
            generation_control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            padded_seq_len <= 1 || !out_tokens || !out_position_ids ||
            !out_request_length || !out_base_position_snapshot || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        cuda_prepare_mtp_verifier_controlled_row_kernel<<<
            1,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            first_token,
            draft_tokens,
            base_position,
            generation_control_row,
            padded_seq_len,
            out_tokens,
            out_position_ids,
            out_request_length,
            out_base_position_snapshot);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA controlled MTP verifier row preparation failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Enqueue first-publication metadata from request-batched prefill.
     */
    bool cudaOps_initialize_mtp_device_logical_state(
        const int32_t *sampled_tokens,
        const int32_t *target_positions,
        int request_count,
        int32_t *out_base_cached_tokens,
        int32_t *out_target_positions,
        int32_t *out_accepted_state_counts,
        int32_t *out_next_condition_tokens,
        int32_t *out_all_drafts_accepted_flags,
        int32_t *out_stopped_flags,
        int32_t *out_publication_ok_flags,
        int device_idx,
        void *stream)
    {
        if (!sampled_tokens ||
            !target_positions ||
            request_count <= 0 ||
            !out_base_cached_tokens ||
            !out_target_positions ||
            !out_accepted_state_counts ||
            !out_next_condition_tokens ||
            !out_all_drafts_accepted_flags ||
            !out_stopped_flags ||
            !out_publication_ok_flags ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_initialize_mtp_device_logical_state_kernel<<<
            1,
            32,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            sampled_tokens,
            target_positions,
            request_count,
            out_base_cached_tokens,
            out_target_positions,
            out_accepted_state_counts,
            out_next_condition_tokens,
            out_all_drafts_accepted_flags,
            out_stopped_flags,
            out_publication_ok_flags);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA MTP device logical-state initialization failed: %s\n",
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
