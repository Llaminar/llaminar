/**
 * @file ROCmSamplingKernels.hip
 * @brief ROCm top-k, top-p distribution, and logit penalty kernels.
 */
#include "ROCmSamplingCommon.hip.inc"

/**
 * @brief Publish one request-policy INT32 scalar into persistent device state.
 *
 * The scalar travels as a HIP kernel argument, whose lifetime is owned by the
 * launch, rather than through an asynchronous H2D copy from stack storage.
 * Later consumers are ordered by the target-slot readiness event published by
 * the orchestrator on the same explicit stream.
 */
__global__ void rocm_publish_int32_control_scalar_kernel(
    int32_t value,
    int32_t *out_value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *out_value = value;
}

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
        if (local_count >= k &&
            !topkCandidateBetter(val, i, local_vals[k - 1], local_idxs[k - 1]))
            continue;

        // Insertion sort: find position and shift
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
            int best_idx = -1;
            int best_thread = -1;

            // Find the thread with the highest unmerged candidate
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

template <int K_CAP>
__global__ void rocm_topk_smallk_partials_f32_kernel(
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

    float local_vals[K_CAP];
    int local_idxs[K_CAP];
    int local_count = 0;

    for (int i = 0; i < k; ++i)
    {
        local_vals[i] = -FLT_MAX;
        local_idxs[i] = -1;
    }

    /*
     * Split the vocabulary into contiguous block-owned ranges, matching the
     * CUDA kernel. A previous ROCm version grid-strided every block across the
     * full vocab; that is correct, but it makes all partial blocks compete for
     * the same cache lines. Contiguous ranges keep each block's scan local and
     * make the partial top-k path easier to reason about when profiling.
     */
    const int block_start =
        static_cast<int>((static_cast<long long>(blockIdx.x) * n) / gridDim.x);
    const int block_end =
        static_cast<int>((static_cast<long long>(blockIdx.x + 1) * n) / gridDim.x);

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
__global__ void rocm_topk_topp_distribution_from_partials_f32_kernel(
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

/**
 * @brief Produce exact sorted Top-K partials with cooperative wave64 selection.
 *
 * Every workgroup owns one contiguous vocabulary span of at most
 * TOPK_BATCHED_COOPERATIVE_CAPACITY entries.  The fixed sixteen entries per
 * lane make vocabulary reads coalesced while keeping all candidates in VGPRs
 * for the repeated deterministic maximum selections.
 */
template <int K_CAP, bool WithHistory = false>
__global__ __launch_bounds__(TOPK_BATCHED_COOPERATIVE_THREADS, 2)
void rocm_topk_cooperative_partials_batched_f32_kernel(
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
            static_cast<int>(threadIdx.x) +
            item * TOPK_BATCHED_COOPERATIVE_THREADS;
        const int index = block_start + offset;
        const bool valid = index < block_end;
        local_values[item] = valid
            ? llaminar2::sampling_math::historyLogit<WithHistory>(row_data[index], index, row, history)
            : -FLT_MAX;
        local_indices[item] = valid ? index : -1;
    }

    __shared__ ROCmBatchedTopKCooperativeStorage<K_CAP> storage;
    rocmSelectBatchedTopKCooperative(
        local_values,
        local_indices,
        k,
        storage);

    const int out_base = (row * partial_blocks + partial) * k;
    for (int rank = static_cast<int>(threadIdx.x);
         rank < k;
         rank += static_cast<int>(blockDim.x))
    {
        partial_values[out_base + rank] = storage.selected_values[rank];
        partial_indices[out_base + rank] = storage.selected_indices[rank];
    }
}

/**
 * @brief Merge cooperative partials and build one compact Top-K/Top-P row.
 *
 * The same block selector is used at both levels, so partial construction and
 * final ranking share one exact comparator and one arithmetic-free ordering
 * proof.  Top-P normalization remains the existing canonical serial routine in
 * lane zero after the sorted Top-K is complete.
 */
template <int K_CAP>
__global__ __launch_bounds__(TOPK_BATCHED_COOPERATIVE_THREADS, 2)
void rocm_topk_topp_distribution_cooperative_batched_f32_kernel(
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
            static_cast<int>(threadIdx.x) +
            item * TOPK_BATCHED_COOPERATIVE_THREADS;
        const bool valid = offset < candidate_count;
        local_values[item] = valid
                                 ? partial_values[partial_base + offset]
                                 : -FLT_MAX;
        local_indices[item] = valid
                                  ? partial_indices[partial_base + offset]
                                  : -1;
    }

    __shared__ ROCmBatchedTopKCooperativeStorage<K_CAP> storage;
    __shared__ float distribution_weights[K_CAP];
    rocmSelectBatchedTopKCooperative(
        local_values,
        local_indices,
        k,
        storage);

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

/**
 * @brief Merge at most 64 already-sorted partial lists with one wave64.
 *
 * Partial selection publishes one descending K-entry list per vocabulary
 * partition.  Re-running a generic Top-K selector over all
 * `partial_blocks * k` entries discards that useful ordering.  Here each lane
 * owns one list cursor and keeps its current head in registers.  A canonical
 * wave reduction chooses the next global rank; only the unique lane owning
 * that token advances.  The complete merge therefore reads 64 initial heads
 * plus at most K successors instead of repeatedly examining every partial
 * candidate, while retaining exactly the same total-order result.
 */
template <int K_CAP>
__global__ __launch_bounds__(64, 4)
void rocm_topk_topp_distribution_sorted_partials_batched_f32_kernel(
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
    if (k > K_CAP || partial_blocks > 64 || active_rows <= 0 ||
        active_rows > static_cast<int>(gridDim.x) || row >= active_rows)
    {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    const int row_base = row * partial_blocks * k;
    const int list_base = row_base + lane * k;
    int cursor = 0;
    ROCmTopKCandidate head{-FLT_MAX, -1};
    if (lane < partial_blocks)
    {
        head.value = partial_values[list_base];
        head.index = partial_indices[list_base];
    }

    __shared__ float selected_values[K_CAP];
    __shared__ int selected_indices[K_CAP];
    __shared__ float distribution_weights[K_CAP];
    for (int rank = 0; rank < k; ++rank)
    {
        const ROCmTopKCandidate winner = rocmTopKReduceWaveBest(head);
        const float winner_value = __shfl(winner.value, 0, 64);
        const int winner_index = __shfl(winner.index, 0, 64);
        if (lane == 0)
        {
            selected_values[rank] = winner_value;
            selected_indices[rank] = winner_index;
        }

        /*
         * Token ids are globally unique within a row, so exactly one partial
         * list can own the selected head.  That lane alone reads its successor;
         * all other lanes retain their register-resident head for the next rank.
         */
        if (head.index == winner_index && winner_index >= 0)
        {
            ++cursor;
            if (cursor < k)
            {
                head.value = partial_values[list_base + cursor];
                head.index = partial_indices[list_base + cursor];
            }
            else
            {
                head.value = -FLT_MAX;
                head.index = -1;
            }
        }
    }

    if (lane == 0)
    {
        const int out_base = row * out_stride;
        llaminar2::sampling_math::build_topk_topp_distribution_from_sorted(
            selected_values,
            selected_indices,
            k,
            top_p,
            temperature,
            out_token_ids + out_base,
            out_probs + out_base,
            distribution_weights);
    }
}

template <int K_CAP, bool WithHistory = false>
__global__ void rocm_topk_smallk_partials_batched_f32_kernel(
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

    /*
     * The lane-private sorted list lives in LDS rather than an automatic array.
     * HIP otherwise spills both K-entry arrays to scratch memory on gfx906. No
     * synchronization is needed while scanning because each lane owns a
     * disjoint contiguous slice; the barrier below publishes all slices to the
     * deterministic single-lane merge.
     */
    extern __shared__ char shared_mem[];
    float *s_vals = reinterpret_cast<float *>(shared_mem);
    int *s_idxs = reinterpret_cast<int *>(
        shared_mem + num_threads * k * sizeof(float));
    const int base = tid * k;
    float *local_vals = s_vals + base;
    int *local_idxs = s_idxs + base;
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

    __syncthreads();

    const int out_base = (row * partial_blocks + partial) * k;
    mergeTopkWave64LaneLists(
        local_vals,
        local_idxs,
        local_count,
        k,
        partial_values + out_base,
        partial_indices + out_base);
}

template <int K_CAP>
__global__ void rocm_topk_topp_distribution_batched_from_partials_f32_kernel(
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

    /*
     * This is the same lane-list layout as the partial-selection kernel. The
     * final merge also uses the trailing LDS region for cursors, merged
     * candidates, and weights, leaving no top-k-sized automatic array for the
     * compiler to spill into per-thread scratch.
     */
    extern __shared__ char shared_mem[];
    float *s_vals = reinterpret_cast<float *>(shared_mem);
    int *s_idxs = reinterpret_cast<int *>(
        shared_mem + num_threads * k * sizeof(float));
    const int base = tid * k;
    float *local_vals = s_vals + base;
    int *local_idxs = s_idxs + base;
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

    __syncthreads();

    std::byte *merge_storage =
        reinterpret_cast<std::byte *>(shared_mem) +
        num_threads * k * (sizeof(float) + sizeof(int));
    float *merged_vals = reinterpret_cast<float *>(merge_storage);
    int *merged_idxs = reinterpret_cast<int *>(merged_vals + k);
    float *weights = reinterpret_cast<float *>(merged_idxs + k);
    mergeTopkWave64LaneLists(
        local_vals,
        local_idxs,
        local_count,
        k,
        merged_vals,
        merged_idxs);

    if (tid == 0)
    {
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
__device__ void rocm_write_active_topk_topp_logits_from_sorted(
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
__global__ void rocm_topk_topp_processed_logits_batched_from_partials_f32_kernel(
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

        rocm_write_active_topk_topp_logits_from_sorted<K_CAP>(
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
__global__ void rocm_topk_topp_processed_logits_batched_f32_kernel(
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

        rocm_write_active_topk_topp_logits_from_sorted<K_CAP>(
            merged_vals,
            merged_idxs,
            k,
            top_p,
            temperature,
            row_out,
            vocab_size);
    }
}

/**
 * @brief Draw from one compact support while keeping seed and position device-owned.
 *
 * Request admission may replace seed bytes behind the retained pointer. This
 * kernel preserves the scalar seed/offset sampling law and rejects zero seed
 * data with an invalid token. Only the bounded compact support is serial;
 * full-vocabulary preparation is performed by the preceding parallel kernels.
 */
__global__ void rocm_sample_distribution_f32_kernel(
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

    // Admission publishes seed data once; retained graphs read it on their
    // exact sampling stream. A malformed seed must not become another RNG law.
    const uint64_t seed = threshold_seed_device ? *threshold_seed_device : threshold_seed;
    if (threshold_seed_device && seed == 0)
    {
        *out_token = -1;
        if (out_probability) *out_probability = 0.0F;
        return;
    }

    /*
     * The request-batched resident lane reads its mutable logical position
     * directly from the publication mailbox. This keeps the draw and the state
     * it describes ordered on one HIP stream without adopting a host shadow.
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
bool rocmOps_publish_int32_control_scalar(
    int32_t value,
    int32_t *out_value,
    int device_idx,
    void *stream)
{
    if (value < 0 || !out_value || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_publish_int32_control_scalar_kernel,
        dim3(1),
        dim3(1),
        0,
        static_cast<hipStream_t>(stream),
        value,
        out_value);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm INT32 control-scalar publication launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

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

    if (k <= TOPK_MEDIUM_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_sample_f32_kernel<TOPK_MEDIUM_K_CAP>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, rng_seed, rng_offset, out_token);
    }
    else if (k <= TOPK_SMALL_K_CAP)
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (k <= TOPK_SMALL_K_CAP &&
        scratch_values &&
        scratch_indices)
    {
        const int threads = TOPK_SMALL_K_THREADS;
        int partial_blocks = (n + threads - 1) / threads;
        if (partial_blocks < 1)
            partial_blocks = 1;
        const int partial_block_cap = runtimeTopkSmallKPartialBlockCap(k);
        if (partial_blocks > partial_block_cap)
            partial_blocks = partial_block_cap;
        const int required_scratch = partial_blocks * k;
        if (scratch_capacity >= required_scratch)
        {
            const size_t smem_size =
                batchedTopkSmallKSharedBytes(threads, k);
            if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(partial_blocks), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, k, scratch_values, scratch_indices);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(partial_blocks), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, k, scratch_values, scratch_indices);
            }

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm small-k Top-K partial FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }

            if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k, top_p, temperature,
                    out_token_ids, out_probs);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_from_partials_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k, top_p, temperature,
                    out_token_ids, out_probs);
            }

            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm small-k Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }
            return true;
        }
    }

    const int threads = TOPK_THREADS;
    const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));

    if (k <= TOPK_MEDIUM_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_distribution_f32_kernel<TOPK_MEDIUM_K_CAP>),
            dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, n, k, top_p, temperature, out_token_ids, out_probs);
    }
    else if (k <= TOPK_SMALL_K_CAP)
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

} // extern "C"

/** @brief Shared selection/normalization dispatch; history changes only candidate reads. */
template <bool WithHistory>
static bool rocmTopKDistributionsImpl(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (k > TOPK_SMALL_K_CAP)
    {
        const auto geometry = llaminar2::sampling_wide::geometry(n, row_count, k, scratch_capacity);
        if (!geometry || !scratch_values || !scratch_indices)
            return false;
        // All 64 lanes participate even though sixteen own private LDS lists.
        // That preserves valid wave shuffles with a bounded 32-KiB list bank.
        hipLaunchKernelGGL(HIP_KERNEL_NAME(llaminar2::sampling_wide::partials<64, WithHistory>),
            dim3(geometry->partial_blocks, row_count), dim3(64), geometry->shared_bytes,
            static_cast<hipStream_t>(stream), data, n, row_stride, k,
            geometry->partial_blocks, scratch_values, scratch_indices, active_rows, history);
        if (hipGetLastError() != hipSuccess)
            return false;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(llaminar2::sampling_wide::distributions<64>),
            dim3(row_count), dim3(64), 0, static_cast<hipStream_t>(stream),
            scratch_values, scratch_indices, geometry->partial_blocks, k,
            top_p, temperature, out_token_ids, out_stride, out_probs, active_rows);
        return hipGetLastError() == hipSuccess;
    }

    if (k <= TOPK_SMALL_K_CAP &&
        scratch_values &&
        scratch_indices)
    {
        const int threads = TOPK_SMALL_K_THREADS;
        int partial_blocks = (n + threads - 1) / threads;
        if (partial_blocks < 1)
            partial_blocks = 1;
        const int partial_block_cap = runtimeTopkSmallKPartialBlockCap(k, row_count);
        if (partial_blocks > partial_block_cap)
            partial_blocks = partial_block_cap;
        const int required_scratch = row_count * partial_blocks * k;
        if (scratch_capacity >= required_scratch)
        {
            const size_t smem_size =
                batchedTopkSmallKSharedBytes(threads, k);
            const dim3 partial_grid(partial_blocks, row_count);
            const int maximum_partial_span =
                (n + partial_blocks - 1) / partial_blocks;
            const bool use_cooperative_selector =
                maximum_partial_span <=
                    TOPK_BATCHED_COOPERATIVE_CAPACITY &&
                partial_blocks * k <=
                    TOPK_BATCHED_COOPERATIVE_CAPACITY;
            if (use_cooperative_selector && k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_cooperative_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP, WithHistory>),
                    partial_grid,
                    dim3(TOPK_BATCHED_COOPERATIVE_THREADS),
                    0,
                    static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks,
                    scratch_values, scratch_indices, active_rows, history);
            }
            else if (use_cooperative_selector)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_cooperative_partials_batched_f32_kernel<TOPK_SMALL_K_CAP, WithHistory>),
                    partial_grid,
                    dim3(TOPK_BATCHED_COOPERATIVE_THREADS),
                    0,
                    static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks,
                    scratch_values, scratch_indices, active_rows, history);
            }
            else if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP, WithHistory>),
                    partial_grid, dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks, scratch_values, scratch_indices,
                    active_rows, history);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP, WithHistory>),
                    partial_grid, dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks, scratch_values, scratch_indices,
                    active_rows, history);
            }

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm batched small-k Top-K partial FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }

            const bool use_sorted_partial_wave_merge =
                partial_blocks <= 64;
            if (use_sorted_partial_wave_merge &&
                k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_sorted_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(row_count),
                    dim3(64),
                    0,
                    static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }
            else if (use_sorted_partial_wave_merge)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_sorted_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(row_count),
                    dim3(64),
                    0,
                    static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }
            else if (use_cooperative_selector && k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_cooperative_batched_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(row_count),
                    dim3(TOPK_BATCHED_COOPERATIVE_THREADS),
                    0,
                    static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }
            else if (use_cooperative_selector)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_cooperative_batched_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(row_count),
                    dim3(TOPK_BATCHED_COOPERATIVE_THREADS),
                    0,
                    static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }
            else if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_distribution_batched_from_partials_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_token_ids, out_stride, out_probs,
                    active_rows);
            }

            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm batched small-k Top-K/Top-P Distribution FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }
            return true;
        }
    }

    return false;
}

extern "C"
{
bool rocmOps_topk_topp_distributions_f32(
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
    return rocmTopKDistributionsImpl<false>(
        data, row_count, n, row_stride, k, top_p, temperature,
        out_token_ids, out_stride, out_probs, scratch_values, scratch_indices,
        scratch_capacity, active_rows, device_idx, stream, {});
}

/** @brief Read-only row selection with original durable/speculative history. */
bool rocmOps_topk_topp_distributions_f32_with_history(
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
    return rocmTopKDistributionsImpl<true>(
        data, row_count, n, row_stride, k, top_p, temperature,
        out_token_ids, out_stride, out_probs, scratch_values, scratch_indices,
        scratch_capacity, active_rows, device_idx, stream, {counts, policy, tokens});
}

bool rocmOps_topk_topp_processed_logits_f32(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (k <= TOPK_SMALL_K_CAP &&
        scratch_values &&
        scratch_indices)
    {
        const int threads = TOPK_SMALL_K_THREADS;
        int partial_blocks = (n + threads - 1) / threads;
        if (partial_blocks < 1)
            partial_blocks = 1;
        const int partial_block_cap = runtimeTopkSmallKPartialBlockCap(k, row_count);
        if (partial_blocks > partial_block_cap)
            partial_blocks = partial_block_cap;
        const int required_scratch = row_count * partial_blocks * k;
        if (scratch_capacity >= required_scratch)
        {
            /*
             * The shared partial selector and every consumer must launch with
             * the same complete LDS contract.  In particular, the selector's
             * merge cursors occupy the trailing region after the lane lists;
             * allocating only the historical lane-list footprint corrupts
             * adjacent state before the processed-logit stage can consume it.
             */
            const size_t smem_size =
                batchedTopkSmallKSharedBytes(threads, k);
            const dim3 partial_grid(partial_blocks, row_count);
            if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_batched_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    partial_grid, dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks, scratch_values, scratch_indices,
                    nullptr, llaminar2::sampling_math::PenaltyLogitRead{});
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_smallk_partials_batched_f32_kernel<TOPK_SMALL_K_CAP>),
                    partial_grid, dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    data, n, row_stride, k, partial_blocks, scratch_values, scratch_indices,
                    nullptr, llaminar2::sampling_math::PenaltyLogitRead{});
            }

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm processed-logit Top-K partial FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }

            if (k <= TOPK_MEDIUM_K_CAP)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_processed_logits_batched_from_partials_f32_kernel<TOPK_MEDIUM_K_CAP>),
                    dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_logits, out_row_stride, n);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(rocm_topk_topp_processed_logits_batched_from_partials_f32_kernel<TOPK_SMALL_K_CAP>),
                    dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
                    scratch_values, scratch_indices, partial_blocks, k,
                    top_p, temperature, out_logits, out_row_stride, n);
            }

            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "ROCm processed-logit Top-K/Top-P FP32 kernel launch failed: %s\n",
                        hipGetErrorString(err));
                return false;
            }
            return true;
        }
    }

    const int threads = TOPK_THREADS;
    const size_t smem_size = threads * k * (sizeof(float) + sizeof(int));
    if (k <= TOPK_MEDIUM_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_processed_logits_batched_f32_kernel<TOPK_MEDIUM_K_CAP>),
            dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, row_count, n, row_stride, k, top_p, temperature,
            out_logits, out_row_stride);
    }
    else if (k <= TOPK_SMALL_K_CAP)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_processed_logits_batched_f32_kernel<TOPK_SMALL_K_CAP>),
            dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, row_count, n, row_stride, k, top_p, temperature,
            out_logits, out_row_stride);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_topk_topp_processed_logits_batched_f32_kernel<TOPK_MAX_K>),
            dim3(row_count), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
            data, row_count, n, row_stride, k, top_p, temperature,
            out_logits, out_row_stride);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm direct processed-logit Top-K/Top-P FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}


/** @brief Reject ambiguous draw sources and enqueue on the exact non-null HIP stream. */
bool rocmOps_sample_distribution_f32(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_sample_distribution_f32_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
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

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Sample Distribution FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
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
__global__ void rocm_initialize_prompt_repetition_history_kernel(
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

__global__ void rocm_apply_logit_penalties_f32_kernel(
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

    bool rocmOps_initialize_prompt_repetition_history(
        int32_t *counts, const int32_t *tokens, int token_count, int vocab_size, void *stream)
    {
        if (!counts || !tokens || !stream || token_count <= 0 || token_count > vocab_size)
            return false;
        constexpr int threads = 256;
        const int blocks = (token_count + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_initialize_prompt_repetition_history_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        counts, tokens, token_count, vocab_size);
    return hipGetLastError() == hipSuccess;
    }

bool rocmOps_apply_logit_penalties_f32(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int threads = 256;
    const int blocks = (num_penalties + threads - 1) / threads;

    hipLaunchKernelGGL(
        rocm_apply_logit_penalties_f32_kernel,
        dim3(blocks), dim3(threads), 0, static_cast<hipStream_t>(stream),
        logits, token_ids, penalties, num_penalties, vocab_size, repetition_penalty);

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
