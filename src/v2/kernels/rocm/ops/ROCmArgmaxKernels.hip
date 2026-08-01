/**
 * @file ROCmArgmaxKernels.hip
 * @brief ROCm/HIP Argmax kernel for greedy sampling
 *
 * Provides a GPU-side argmax over FP32 logits, returning the index and value
 * of the maximum element. Used for greedy decode sampling to avoid D2H of
 * the entire logits tensor (~600 KB for 152K vocab).
 *
 * Mirrors the CUDA implementation in CUDASamplingKernels.cu for cross-backend
 * parity: a two-pass multi-block reduction (pass 1 spreads the vocab across
 * many blocks/CUs producing one partial per block; pass 2 reduces the partials
 * in a single small block). The caller supplies the per-block partial scratch
 * (owned by the orchestrator's BufferArena); there is no single-block fallback.
 *
 * Performance: ~5µs for 76K elements on MI50 vs ~170µs for CPU std::max_element
 */

#include <hip/hip_runtime.h>
#include <cfloat>
#include <climits>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../common/SamplingMath.h"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"

// ── Argmax multi-block reduction tuning ─────────────────────────────────────
// Threads per block for both reduction passes. Must be a power of two because
// the in-block tree reduction halves the active range each step.
constexpr int ARGMAX_REDUCE_THREADS = 256;
constexpr int ARGMAX_FINALIZE_THREADS = 256;
// Target elements processed per thread in pass 1. Chosen so each thread reads a
// handful of elements (good memory coalescing) while still spreading the vocab
// across enough blocks to occupy most CUs. 8 → ~74 blocks for a 152K vocab.
constexpr int ARGMAX_ELEMS_PER_THREAD = 8;

__device__ __forceinline__ bool rocm_argmax_better(float candidate_value,
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
// gridDim.x blocks lets the reduction use many CUs instead of a single one.
__global__ void rocm_argmax_partial_f32_kernel(
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
        if (rocm_argmax_better(val, i, local_max, local_idx))
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
            if (rocm_argmax_better(smax[tid + stride], sidx[tid + stride],
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
__global__ void rocm_argmax_finalize_f32_kernel(
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
        int idx = partial_idxs[i];
        if (rocm_argmax_better(val, idx, local_max, local_idx))
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
            if (rocm_argmax_better(smax[tid + stride], sidx[tid + stride],
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

__global__ void rocm_argmax_partial_f32_batched_rows_kernel(
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
        if (rocm_argmax_better(val, i, local_max, local_idx))
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
        if (tid < stride)
        {
            if (rocm_argmax_better(smax[tid + stride], sidx[tid + stride],
                                   smax[tid], sidx[tid]))
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
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

/**
 * @brief HIP grouped verifier argmax with serial-row penalty semantics.
 *
 * The persistent histogram owns prior emitted tokens.  Verifier input tokens
 * up to the current row complete that row's speculative branch history, so one
 * grouped launch observes exactly the same counts as serial decode.
 */
__global__ void rocm_argmax_partial_f32_batched_rows_mtp_penalty_kernel(
    const float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    float *__restrict__ partial_vals,
    int *__restrict__ partial_idxs,
    int row_partial_capacity)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(
        shared_mem + blockDim.x * sizeof(float));

    const int row = blockIdx.y;
    if (row >= rows)
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
        if (rocm_argmax_better(value, token, local_max, local_idx))
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
            rocm_argmax_better(
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

/**
 * @brief Publish request policy into graph-stable device storage.
 */
__global__ void rocm_configure_mtp_greedy_penalty_policy_kernel(
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
 * @brief Apply durable-history penalties without a verifier prefix.
 *
 * This M=1 specialization gives adjacent work-items adjacent vocabulary
 * entries and excludes every verifier-only branch. Ordinary serial decode
 * therefore retains a compact VGPR/SGPR footprint and low launch latency.
 */
__global__ void rocm_apply_mtp_durable_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy)
{
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
        for (int row = 0; row < rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] -= penalty;
        }
    }
}

/**
 * @brief Apply device-owned serial-decode penalties to verifier logit rows.
 *
 * Adjacent work-items own adjacent vocabulary entries, preserving coalesced
 * accesses on AMD wavefronts. Tokens named anywhere in the verifier branch are
 * left to sparse row owners in workgroup zero. Every other token has one
 * durable count shared by all rows, allowing this pass to scan the short
 * branch once per vocabulary token instead of once per row.
 *
 * The two-kernel ownership split removes O(rows^2 * vocabulary) redundant
 * comparisons while retaining one captured launch. Dense and sparse owners
 * touch disjoint token addresses, requiring no grid-wide barrier, atomics,
 * temporary buffers, transfers, or synchronization points.
 */
__global__ void rocm_apply_mtp_verifier_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy)
{
    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    const int prefix_begin =
        request_policy.first_token_already_in_history != 0 ? 1 : 0;

    // Verifier launches reserve workgroup zero for sparse branch-token rows.
    // The sparse workgroup runs concurrently with dense workgroups and exits;
    // dense workgroups skip every address the sparse owner can modify.
    if (blockIdx.x == 0)
    {
        for (int row = static_cast<int>(threadIdx.x);
             row < rows;
             row += static_cast<int>(blockDim.x))
        {
            for (int candidate_index = prefix_begin;
                 candidate_index < rows;
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
             history_index < rows;
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
        for (int row = 0; row < rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] -= penalty;
        }
    }

}

/** @brief Resolve one token from persistent condition and draft branch rows. */
__device__ __forceinline__ int rocm_mtp_branch_token_at(
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
 * @brief Apply one proposal row's durable and speculative branch penalties.
 *
 * Sparse branch tokens belong to block zero; dense durable-only tokens belong
 * to every later block.  This disjoint ownership preserves serial arithmetic
 * without atomics, scratch, or a grid-wide synchronization point.
 */
__global__ void rocm_apply_mtp_branch_penalties_f32_row_kernel(
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
            const int token = rocm_mtp_branch_token_at(
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
                    rocm_mtp_branch_token_at(
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
                count += rocm_mtp_branch_token_at(
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
                rocm_mtp_branch_token_at(
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
 * @brief Advance the mirrored generated-token histogram after publication.
 */
__global__ void rocm_commit_mtp_greedy_penalty_history_kernel(
    const int *__restrict__ output_tokens,
    const int *__restrict__ output_meta,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    int output_token_capacity,
    int vocab_size,
    int *__restrict__ generated_token_counts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (output_meta[llaminar2::sampling_math::kSpecBatchMetaOk] == 0)
        return;

    int output_count =
        output_meta[llaminar2::sampling_math::kSpecBatchMetaOutputCount];
    if (output_count < 0)
        output_count = 0;
    if (output_count > output_token_capacity)
        output_count = output_token_capacity;
    const int begin =
        policy->first_token_already_in_history != 0 ? 1 : 0;
    for (int i = begin; i < output_count; ++i)
    {
        const int token = output_tokens[i];
        if (token >= 0 && token < vocab_size)
            ++generated_token_counts[token];
    }
}

__global__ void rocm_argmax_finalize_f32_batched_rows_kernel(
    const float *__restrict__ partial_vals,
    const int *__restrict__ partial_idxs,
    int num_partials,
    int row_partial_capacity,
    float *__restrict__ out_values,
    int *__restrict__ out_indices,
    int output_stride)
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
        if (rocm_argmax_better(val, idx, local_max, local_idx))
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
        if (tid < stride)
        {
            if (rocm_argmax_better(smax[tid + stride], sidx[tid + stride],
                                   smax[tid], sidx[tid]))
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        const int out_row = row * output_stride;
        out_values[out_row] = smax[0];
        out_indices[out_row] = sidx[0];
    }
}

// ============================================================================
// Extern "C" Wrapper
// ============================================================================

extern "C"
{

/**
 * @brief Launch argmax kernel on device (two-pass multi-block reduction)
 *
 * Finds the index and value of the maximum element in a float array.
 *
 * @param data             Device pointer to FP32 data
 * @param n                Number of elements
 * @param out_value        Device pointer to store max value (1 float)
 * @param out_index        Device pointer to store max index (1 int)
 * @param partial_vals     Device scratch for pass-1 per-block partial values
 * @param partial_idxs     Device scratch for pass-1 per-block partial indices
 * @param partial_capacity Number of entries in the partial scratch (bounds grid)
 * @param device_idx       GPU device index
 * @param stream           HIP stream (nullptr for default)
 * @return true on success
 */
bool rocmOps_argmax_f32(
    const float* data,
    int n,
    float* out_value,
    int* out_index,
    float* partial_vals,
    int* partial_idxs,
    int partial_capacity,
    int device_idx,
    void* stream)
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
                "ROCm Argmax FP32: missing partial-reduction scratch "
                "(partial_vals=%p partial_idxs=%p capacity=%d)\n",
                (void *)partial_vals, (void *)partial_idxs, partial_capacity);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t s = static_cast<hipStream_t>(stream);

    // Two-pass multi-block reduction.
    // Size the pass-1 grid so each thread processes ~ARGMAX_ELEMS_PER_THREAD
    // elements, then clamp to the partial-buffer capacity. This spreads the
    // vocab across many blocks (and thus CUs) instead of a single block.
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
    hipLaunchKernelGGL(
        rocm_argmax_partial_f32_kernel,
        dim3(num_blocks), dim3(threads), smem1, s,
        data, n, partial_vals, partial_idxs);

    // Pass 2: single small block reduces the partials to the final result.
    const int fthreads = ARGMAX_FINALIZE_THREADS;
    const size_t smem2 = fthreads * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(
        rocm_argmax_finalize_f32_kernel,
        dim3(1), dim3(fthreads), smem2, s,
        partial_vals, partial_idxs, num_blocks, out_value, out_index);

    (void)hipGetLastError();  // Clear stale errors
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Argmax FP32 (multi-block) launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_argmax_f32_batched_rows(
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
    if (rows <= 0 || cols <= 0 || row_stride < cols ||
        !data || !out_values || !out_indices ||
        !partial_vals || !partial_idxs || partial_capacity < rows ||
        output_stride <= 0)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t s = static_cast<hipStream_t>(stream);

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
    hipLaunchKernelGGL(
        rocm_argmax_partial_f32_batched_rows_kernel,
        dim3(num_blocks, rows), dim3(threads), smem1, s,
        data, rows, cols, row_stride, partial_vals, partial_idxs,
        row_partial_capacity);

    const int fthreads = ARGMAX_FINALIZE_THREADS;
    const size_t smem2 = fthreads * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(
        rocm_argmax_finalize_f32_batched_rows_kernel,
        dim3(rows), dim3(fthreads), smem2, s,
        partial_vals, partial_idxs, num_blocks, row_partial_capacity,
        out_values, out_indices, output_stride);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Argmax FP32 batched rows launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_configure_mtp_greedy_penalty_policy(
    llaminar2::MTPGreedyPenaltyPolicy *controls,
    float presence_penalty,
    float frequency_penalty,
    bool first_token_already_in_history,
    int device_idx,
    void *stream)
{
    if (!controls || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_configure_mtp_greedy_penalty_policy_kernel,
        dim3(1),
        dim3(1),
        0,
        static_cast<hipStream_t>(stream),
        controls,
        presence_penalty,
        frequency_penalty,
        first_token_already_in_history ? 1 : 0);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm MTP greedy penalty policy launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_argmax_f32_batched_rows_mtp_penalties(
    const float *data,
    int rows,
    int cols,
    int row_stride,
    const int *verifier_input_tokens,
    const int *generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *policy,
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
        !policy || !out_values || !out_indices ||
        !partial_vals || !partial_idxs || partial_capacity < rows ||
        !stream || output_stride <= 0)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
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
    hipLaunchKernelGGL(
        rocm_argmax_partial_f32_batched_rows_mtp_penalty_kernel,
        dim3(num_blocks, rows),
        dim3(threads),
        shared_bytes,
        static_cast<hipStream_t>(stream),
        data,
        rows,
        cols,
        row_stride,
        verifier_input_tokens,
        generated_token_counts,
        policy,
        partial_vals,
        partial_idxs,
        row_partial_capacity);
    hipLaunchKernelGGL(
        rocm_argmax_finalize_f32_batched_rows_kernel,
        dim3(rows),
        dim3(ARGMAX_FINALIZE_THREADS),
        static_cast<size_t>(ARGMAX_FINALIZE_THREADS) *
            (sizeof(float) + sizeof(int)),
        static_cast<hipStream_t>(stream),
        partial_vals,
        partial_idxs,
        num_blocks,
        row_partial_capacity,
        out_values,
        out_indices,
        output_stride);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm MTP penalty-aware batched argmax launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_apply_mtp_penalties_f32_rows(
    float *data,
    int rows,
    int cols,
    int row_stride,
    const int *verifier_input_tokens,
    const int *generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *policy,
    int device_idx,
    void *stream)
{
    if (!data || rows <= 0 || cols <= 0 || row_stride < cols ||
        !generated_token_counts || !policy || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    blocks = blocks > 1024 ? 1024 : blocks;
    if (verifier_input_tokens)
    {
        hipLaunchKernelGGL(
            rocm_apply_mtp_verifier_penalties_f32_rows_kernel,
            dim3(blocks + 1),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            data,
            rows,
            cols,
            row_stride,
            verifier_input_tokens,
            generated_token_counts,
            policy);
    }
    else
    {
        hipLaunchKernelGGL(
            rocm_apply_mtp_durable_penalties_f32_rows_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            data,
            rows,
            cols,
            row_stride,
            generated_token_counts,
            policy);
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm MTP penalty row transform launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    return true;
}

bool rocmOps_apply_mtp_branch_penalties_f32_row(
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

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    int dense_blocks = (cols + threads - 1) / threads;
    dense_blocks = dense_blocks > 1024 ? 1024 : dense_blocks;
    hipLaunchKernelGGL(
        rocm_apply_mtp_branch_penalties_f32_row_kernel,
        dim3(dense_blocks + 1),
        dim3(threads),
        0,
        static_cast<hipStream_t>(stream),
        data,
        cols,
        first_condition_token,
        prior_draft_tokens,
        prior_draft_count,
        generated_token_counts,
        policy);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm MTP branch penalty row transform launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_commit_mtp_greedy_penalty_history(
    const int *output_tokens,
    const int *output_meta,
    const llaminar2::MTPGreedyPenaltyPolicy *policy,
    int output_token_capacity,
    int vocab_size,
    int *generated_token_counts,
    int device_idx,
    void *stream)
{
    if (!output_tokens || !output_meta || !policy ||
        output_token_capacity <= 0 || vocab_size <= 0 ||
        !generated_token_counts || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_commit_mtp_greedy_penalty_history_kernel,
        dim3(1),
        dim3(1),
        0,
        static_cast<hipStream_t>(stream),
        output_tokens,
        output_meta,
        policy,
        output_token_capacity,
        vocab_size,
        generated_token_counts);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm MTP greedy penalty history commit launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
