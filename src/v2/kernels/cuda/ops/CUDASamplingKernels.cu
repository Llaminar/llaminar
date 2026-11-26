/**
 * @file CUDASamplingKernels.cu
 * @brief CUDA argmax, top-k, top-p distribution, and logit penalty kernels.
 */
#include "CUDASamplingCommon.cuh"

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
 * @brief Fused history-aware argmax with serial-row penalty semantics.
 *
 * Row `r` sees the durable generated-token histogram plus verifier input
 * tokens `[prefix_begin, r]`.  That is exactly the history visible to `r`
 * serial decode calls, but it avoids materializing one sparse host map per row
 * and avoids mutating the logits matrix before reduction.
 * The committed-only specialization reads the durable histogram alone, sharing
 * that arithmetic and reduction order without verifier storage or extra work.
 */
template <bool SPECULATIVE_HISTORY>
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
    const int active_rows = SPECULATIVE_HISTORY ? *active_rows_device : rows;
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
            const int history = generated_token_counts[token];
        int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
            if constexpr (SPECULATIVE_HISTORY)
            {
                // Ordinary sampling consumes only the committed histogram.
                // Compile out branch reads instead of fabricating a verifier
                // row or adding a separate logit transformation pass.
                for (int history_index = prefix_begin;
                     history_index <= row;
                     ++history_index)
                    count += verifier_input_tokens[history_index] == token ? 1 : 0;
            }

            if (count > 0 || (request_policy.repetition_penalty != 1.0f &&
                llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
            {
                float penalty = 0.0f;
                if (count > 0 && request_policy.presence_penalty != 0.0f)
                    penalty += request_policy.presence_penalty;
                if (request_policy.frequency_penalty != 0.0f)
                {
                    const float frequency_component =
                        request_policy.frequency_penalty *
                        static_cast<float>(count);
                    penalty += frequency_component;
                }
                value = llaminar2::sampling_math::penalizedLogit(
                    value, request_policy.repetition_penalty, penalty);
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
template <int K_CAP, bool WithHistory = false>
__global__ void cuda_topk_cooperative_partials_batched_f32_kernel(
    const float *__restrict__ data,
    int n,
    int row_stride,
    int k,
    int partial_blocks,
    float *__restrict__ partial_values,
    int *__restrict__ partial_indices,
    const int *__restrict__ active_rows_device,
    llaminar2::sampling_math::PenaltyLogitRead history)
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
        local_values[item] = valid
            ? llaminar2::sampling_math::historyLogit<WithHistory>(row_data[index], index, row, history)
            : -FLT_MAX;
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

template <int K_CAP, bool WithHistory = false>
__global__ void cuda_topk_smallk_partials_batched_f32_kernel(
    const float *__restrict__ data,
    int n,
    int row_stride,
    int k,
    int partial_blocks,
    float *__restrict__ partial_values,
    int *__restrict__ partial_indices,
    const int *__restrict__ active_rows_device,
    llaminar2::sampling_math::PenaltyLogitRead history)
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
        const float val = llaminar2::sampling_math::historyLogit<WithHistory>(
            row_data[i], i, row, history);
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

/**
 * @brief Draw one token from a bounded compact table on its owning stream.
 *
 * A captured scalar draw, a captured seed with resident position, or an
 * admitted resident seed/position pair selects the same sampling arithmetic.
 * The launch bridge rejects ambiguous sources. Only the compact support (at
 * most 256 entries) is serial here; vocabulary scoring remains parallel.
 * Invalid resident seed data publishes -1 for the generation owner to reject.
 */
__global__ void cuda_sample_distribution_f32_kernel(
    const int *__restrict__ token_ids,
    const float *__restrict__ probs,
    int k,
    float threshold,
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    int *__restrict__ out_token,
    float *__restrict__ out_probability,
    const uint64_t *__restrict__ threshold_seed_device)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Seed bytes are admission data, not mutable host state or graph identity.
    // Invalid admission poisons the sample so ordinary publication fails closed.
    const uint64_t seed = threshold_seed_device ? *threshold_seed_device : threshold_seed;
    if (threshold_seed_device && seed == 0)
    {
        *out_token = -1;
        if (out_probability) *out_probability = 0.0F;
        return;
    }

    /*
     * Request-batched GPU MTP keeps the mutable logical position in its
     * publication mailbox. Reading that scalar here makes the random draw and
     * the state it describes part of one ordered device stream. The ordinary
     * scalar-threshold API remains available for nonresident callers.
     */
    const float effective_threshold =
        threshold_position
            ? llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                  seed,
                  *threshold_position + threshold_position_offset,
                  0 /* MTPSpecStochasticDrawPurpose::Sample */)
            : threshold;
    *out_token =
        llaminar2::sampling_math::sample_distribution_with_threshold_and_probability(
            token_ids, probs, k, effective_threshold, out_probability);
}


// ============================================================================
// Logit Penalty Application Kernel — Subtract sparse penalties from logits
// ============================================================================

/**
 * @brief Mark sorted unique prompt tokens in the existing history row.
 * @param counts Persistent packed history; each live token has exactly one writer.
 * @param tokens Live unique request token IDs, never padded allocation capacity.
 * @param token_count Live unique count.
 * @param vocab_size Logical bounds; request admission already validated the IDs.
 */
__global__ void cuda_initialize_prompt_repetition_history_kernel(
    int32_t *counts, const int32_t *tokens, int token_count, int vocab_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < token_count)
    {
        const int token = tokens[i];
        if (token >= 0 && token < vocab_size)
            counts[token] = llaminar2::sampling_math::TokenPenaltyHistory::withPrompt(counts[token]);
    }
}

__global__ void cuda_apply_logit_penalties_f32_kernel(
    float *__restrict__ logits,
    const int *__restrict__ token_ids,
    const float *__restrict__ penalties,
    int num_penalties,
    int vocab_size, float repetition_penalty)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_penalties)
        return;

    int token_id = token_ids[idx];
    if (token_id >= 0 && token_id < vocab_size)
    {
        logits[token_id] = llaminar2::sampling_math::penalizedLogit(
            logits[token_id], repetition_penalty, penalties[idx]);
    }
}


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
            !data || !generated_token_counts ||
            !policy || !out_values || !out_indices ||
            ((verifier_input_tokens != nullptr) != (active_rows != nullptr)) ||
            (!verifier_input_tokens && rows != 1) ||
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
        // The typed public binding selects one exact compiled history mode.
        // Both retain the same reduction tree and scratch geometry.
        const auto launch = [&]<bool SpeculativeHistory>()
        {
            cuda_argmax_partial_f32_batched_rows_mtp_penalty_kernel<SpeculativeHistory><<<
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
            cuda_argmax_finalize_f32_batched_rows_kernel<false, SpeculativeHistory><<<
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
        };
        if (verifier_input_tokens)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
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
        if (n <= 0 || k <= 0 || k > TOPK_MAX_K || !data || !out_values || !out_indices || !stream)
            return false;

        if (k > n)
            k = n;

        cudaSetDevice(device_idx);

        const int threads = TOPK_THREADS;
        const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

        if (!prepareTopKSharedMemory(cuda_topk_f32_kernel, smem_size))
            return false;
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

        if (!prepareTopKSharedMemory(cuda_topk_topp_sample_f32_kernel, smem_size))
            return false;
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
            if (!prepareTopKSharedMemory(cuda_topk_topp_distribution_f32_kernel<TOPK_MAX_K>, smem_size))
                return false;
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

} // extern "C"

/** @brief Shared selection/normalization dispatch; history changes only candidate reads. */
template <bool WithHistory>
static bool cudaTopKDistributionsImpl(
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
    void *stream,
    llaminar2::sampling_math::PenaltyLogitRead history)
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

    if (k > TOPK_SMALL_K_CAP)
    {
        const auto geometry = llaminar2::sampling_wide::geometry(n, row_count, k, scratch_capacity);
        if (!geometry || !scratch_values || !scratch_indices)
            return false;
        // Wide K reuses the admitted scratch pair. The complete warp
        // merges bounded shared lists; no K-entry automatic arrays spill.
        llaminar2::sampling_wide::partials<32, WithHistory>
            <<<dim3(geometry->partial_blocks, row_count), 32, geometry->shared_bytes, s>>>(
                data, n, row_stride, k, geometry->partial_blocks,
                scratch_values, scratch_indices, active_rows, history);
        if (cudaGetLastError() != cudaSuccess)
            return false;
        llaminar2::sampling_wide::distributions<32><<<row_count, 32, 0, s>>>(
            scratch_values, scratch_indices, geometry->partial_blocks, k,
            top_p, temperature, out_token_ids, out_stride, out_probs, active_rows);
        return cudaGetLastError() == cudaSuccess;
    }

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
                cuda_topk_cooperative_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP, WithHistory>
                    <<<partial_grid,
                       TOPK_BATCHED_COOPERATIVE_THREADS,
                       0,
                       s>>>(
                        data, n, row_stride, k, partial_blocks,
                        scratch_values, scratch_indices, active_rows, history);
            }
            else if (use_cooperative_selector)
            {
                cuda_topk_cooperative_partials_batched_f32_kernel<TOPK_SMALL_K_CAP, WithHistory>
                    <<<partial_grid,
                       TOPK_BATCHED_COOPERATIVE_THREADS,
                       0,
                       s>>>(
                        data, n, row_stride, k, partial_blocks,
                        scratch_values, scratch_indices, active_rows, history);
            }
            else if (k <= TOPK_MEDIUM_K_CAP)
            {
                cuda_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP, WithHistory>
                    <<<partial_grid, threads, smem_size, s>>>(
                        data, n, row_stride, k, partial_blocks,
                        scratch_values, scratch_indices, active_rows, history);
            }
            else
            {
                cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP, WithHistory>
                    <<<partial_grid, threads, smem_size, s>>>(
                        data, n, row_stride, k, partial_blocks,
                        scratch_values, scratch_indices, active_rows, history);
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

extern "C"
{
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
        return cudaTopKDistributionsImpl<false>(
            data, row_count, n, row_stride, k, top_p, temperature,
            out_token_ids, out_stride, out_probs, scratch_values, scratch_indices,
            scratch_capacity, active_rows, device_idx, stream, {});
    }

/** @brief Read-only row selection with original durable/speculative history. */
    bool cudaOps_topk_topp_distributions_f32_with_history(
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
        const int *counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *tokens,
        int device_idx,
        void *stream)
    {
        if (!counts || !policy)
            return false;
        return cudaTopKDistributionsImpl<true>(
            data, row_count, n, row_stride, k, top_p, temperature,
            out_token_ids, out_stride, out_probs, scratch_values, scratch_indices,
            scratch_capacity, active_rows, device_idx, stream, {counts, policy, tokens});
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
                            scratch_values, scratch_indices, nullptr, llaminar2::sampling_math::PenaltyLogitRead{});
                }
                else
                {
                    cuda_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>
                        <<<partial_grid, threads, smem_size, s>>>(
                            data, n, row_stride, k, partial_blocks,
                            scratch_values, scratch_indices, nullptr, llaminar2::sampling_math::PenaltyLogitRead{});
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
            if (!prepareTopKSharedMemory(cuda_topk_topp_processed_logits_batched_f32_kernel<TOPK_MAX_K>, smem_size))
                return false;
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


    /** @brief Validate one explicit draw source and enqueue without allocation or observation. */
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
        void *stream,
        const uint64_t *threshold_seed_device)
    {
        if (k <= 0 || k > TOPK_MAX_K || !token_ids || !probs || !out_token ||
            !stream || (threshold_position && threshold_seed == 0 && !threshold_seed_device) ||
            (threshold_seed_device && (!threshold_position || threshold_seed != 0)))
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
            out_probability,
            threshold_seed_device);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Sample Distribution FP32 kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }


    // ============================================================================
    // Logit Penalty Application Kernel
    // ============================================================================

        bool cudaOps_initialize_prompt_repetition_history(
        int32_t *counts, const int32_t *tokens, int token_count, int vocab_size, void *stream)
    {
        if (!counts || !tokens || !stream || token_count <= 0 || token_count > vocab_size)
            return false;
        constexpr int threads = 256;
        const int blocks = (token_count + threads - 1) / threads;
    cuda_initialize_prompt_repetition_history_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        counts, tokens, token_count, vocab_size);
    return cudaGetLastError() == cudaSuccess;
    }

bool cudaOps_apply_logit_penalties_f32(
        float *logits,
        const int *token_ids,
        const float *penalties,
        int num_penalties,
        int vocab_size,
        int device_idx,
        void *stream, float repetition_penalty)
    {
        if (num_penalties <= 0 || !logits || !token_ids || !penalties)
            return false;

        cudaSetDevice(device_idx);

        // Simple 1D grid: one thread per penalty entry
        const int threads = 256;
        const int blocks = (num_penalties + threads - 1) / threads;

        cuda_apply_logit_penalties_f32_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, token_ids, penalties, num_penalties, vocab_size, repetition_penalty);

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
