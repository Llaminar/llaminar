/**
 * @file CUDAGatedDeltaNetKernels.cu
 * @brief CUDA kernels for Gated Delta Net (GDN) recurrence and short convolution
 *
 * Implements the delta rule linear attention recurrence on GPU:
 *   S_t = exp(g_t) * S_{t-1}
 *   kv = S^T * k
 *   delta = (v - kv) * beta
 *   S += outer(k, delta)
 *   o = S^T * q
 *
 * Kernel owns ALL preprocessing (same as CPU):
 * - L2 normalization of Q and K
 * - Query scaling by 1/sqrt(d_k)
 * - Gate computation: g = A_log * softplus(alpha + dt_bias)
 * - Beta sigmoid: beta_sig = sigmoid(beta_raw)
 *
 * Design: One thread block per head. State matrix S[d_k, d_v] lives in global
 * memory (persistent between decode steps). Shared memory used for Q/K scratch.
 */

#include "../ops/CUDAHelpers.cuh"
#include "../../../utils/DebugEnv.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace
{ // anonymous namespace to avoid symbol conflicts with ROCm kernels

    // =========================================================================
    // Decode Recurrence Kernel (seq_len=1, one block per head)
    // =========================================================================
    // Decode Recurrence Kernel (seq_len=1, multi-block per head)
    //
    // Optimizations vs V1:
    //   1. Fused decay into per-column loops — eliminates separate decay pass
    //      and one full traversal of the d_k×d_v state matrix
    //   2. Two-pass column processing: Pass 1 reads state (decay+kv), Pass 2
    //      reads+writes (decay+update+output). Saves 50% memory traffic.
    //   3. 2D grid: (n_heads, col_blocks) for better SM utilization
    //      (was 16 blocks / 108 SMs, now 32+ blocks)
    // =========================================================================

    constexpr int kGdnRecurrentThreads = 256;
    constexpr int kGdnRecurrentRowSplit = 8;
    constexpr int kGdnRecurrentColumnsPerBlock =
        kGdnRecurrentThreads / kGdnRecurrentRowSplit;

    /**
     * @brief Return the byte-stable query scale shared by every CUDA GDN regime.
     *
     * CUDA emits different FP32 values for runtime `rsqrtf(128)` and the same
     * expression constant-folded in a `D_K=128` specialization. Query scaling
     * affects outputs without affecting recurrent state, which allowed long
     * prefill to differ from serial decode by several ULPs. Supported GDN key
     * widths therefore use explicit correctly rounded FP32 constants. The host
     * route rejects every other width before launch; the trap makes violating
     * that contract fatal even if a future caller bypasses host validation.
     *
     * @param d_k Runtime key width selected by model topology.
     * @return Exact FP32 value of `1 / sqrt(d_k)` for a supported width.
     */
    __device__ __forceinline__ float cuda_gdn_query_scale(int d_k)
    {
        if (d_k == 64)
            return 0x1p-3f;
        if (d_k == 128)
            return 0x1.6a09e6p-4f;
        asm volatile("trap;");
        return 0.0f;
    }

    /**
     * @brief Start one vectorized Ampere global-to-shared Q/K transfer group.
     *
     * Long-prefill recurrence consumes one complete Q row and one complete K
     * row at every causal step.  Each participating lane moves four adjacent
     * FP32 values directly into shared memory with `cp.async`, avoiding the
     * register-staged load/store dependency that otherwise dominates sampled
     * long-scoreboard stalls.  The caller owns the double-buffer lifetime and
     * must wait for this group before exposing the destination stage.
     *
     * The CUDA build requires an SM80-or-newer target, so this is the sole
     * production implementation rather than an architecture-dependent eager
     * alternative.
     *
     * @tparam D_K Compile-time Q/K row width, divisible by four floats.
     * @param q_stage Shared-memory destination for the Q row.
     * @param k_stage Shared-memory destination for the K row.
     * @param q_src Global-memory source for the Q row.
     * @param k_src Global-memory source for the K row.
     */
    template <int D_K>
    __device__ __forceinline__ void cuda_gdn_stage_qk_async(
        float *q_stage,
        float *k_stage,
        const float *q_src,
        const float *k_src)
    {
        static_assert(D_K % 4 == 0);
        const int vector_index = static_cast<int>(threadIdx.x);
        if (vector_index < D_K / 4)
        {
            const int element = vector_index * 4;
            const uint32_t q_shared = static_cast<uint32_t>(
                __cvta_generic_to_shared(q_stage + element));
            const uint32_t k_shared = static_cast<uint32_t>(
                __cvta_generic_to_shared(k_stage + element));
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                :
                : "r"(q_shared), "l"(q_src + element), "r"(16)
                : "memory");
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                :
                : "r"(k_shared), "l"(k_src + element), "r"(16)
                : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    /**
     * @brief Wait until every Q/K transfer group issued by this lane is done.
     *
     * A following CTA barrier publishes the completed stage to all consumers.
     * Keeping the wait and barrier separate makes the producer/consumer edge
     * explicit and lets recurrence work overlap the in-flight next-row group.
     */
    __device__ __forceinline__ void cuda_gdn_wait_qk_async()
    {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }

    /**
     * @brief Advance request-local GDN rows with one fixed, batch-invariant tree.
     *
     * Temporal rows remain causal and execute in increasing order inside one
     * resident block. Within a row, eight lanes own contiguous, disjoint K
     * partitions for each value column. Both M=1 decode and runtime-sized MTP
     * verification launch this exact specialization, so the partial-sum order
     * is independent of M and grouped outputs remain byte-identical to serial
     * calls of the same production kernel.
     *
     * A lane keeps its state partition in registers across every row. For the
     * Qwen GDN geometry (`D_K=128`) this is sixteen floats per lane: state is
     * loaded once and committed once, while optional post-row snapshots are
     * still published for every verifier row. This removes the old second
     * global state read and exposes eight times as many warps without atomics,
     * launch-level row replay, or a host-owned recurrence.
     *
     * Q/K normalization deliberately retains the former one-warp arithmetic
     * tree. Preserving that preprocessing order limits numerical change to the
     * newly explicit K-part reduction itself and keeps capture identity stable.
     *
     * @tparam D_K Compile-time key width. Supported graph geometries select a
     *              specialization before capture; no runtime fallback exists.
     */
    template <int D_K, bool OutOfPlaceState>
    __global__ __launch_bounds__(kGdnRecurrentThreads, 2)
    void cuda_gdn_recurrent_step_kernel(
        const float *__restrict__ q,        // [n_heads * d_k]
        const float *__restrict__ k,        // [n_heads * d_k]
        const float *__restrict__ v,        // [n_heads * d_v]
        const float *__restrict__ alpha,    // [n_heads]
        const float *__restrict__ beta_raw, // [n_heads]
        const float *__restrict__ A_log,    // [n_heads]
        const float *__restrict__ dt_bias,  // [n_heads]
        float *__restrict__ output,         // [n_heads * d_v]
        const float *initial_state,
        float *updated_state,
        int request_count, int request_seq_len,
        int n_heads, int d_v,
        bool use_qk_l2norm,
        const int *__restrict__ effective_seq_len_ptr,
        int effective_row_idx,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        const int request_head = blockIdx.x;
        const int request = request_head / n_heads;
        const int h = request_head - request * n_heads;
        if (request >= request_count || h >= n_heads)
            return;

        static_assert(D_K % kGdnRecurrentRowSplit == 0);
        constexpr int kRowsPerSplit = D_K / kGdnRecurrentRowSplit;

        const int tid = threadIdx.x;
        const int split_id = tid / kGdnRecurrentColumnsPerBlock;
        const int column_in_block = tid % kGdnRecurrentColumnsPerBlock;
        const int vi =
            blockIdx.y * kGdnRecurrentColumnsPerBlock + column_in_block;

        // Request-local state persists across every row processed by this block.
        const size_t request_qk_stride =
            static_cast<size_t>(n_heads) * static_cast<size_t>(D_K);
        const size_t request_v_stride =
            static_cast<size_t>(n_heads) * static_cast<size_t>(d_v);
        const size_t request_state_stride =
            static_cast<size_t>(n_heads) * static_cast<size_t>(D_K) *
            static_cast<size_t>(d_v);
        const float *state_source =
            OutOfPlaceState ? initial_state : updated_state;
        const float *S_initial =
            state_source + static_cast<size_t>(request) * request_state_stride +
            static_cast<size_t>(h) * D_K * d_v;
        float *S_updated =
            updated_state + static_cast<size_t>(request) * request_state_stride +
            static_cast<size_t>(h) * D_K * d_v;

        // Shared memory for preprocessed Q and K
        extern __shared__ float smem[];
        float *q_local = smem;                       // [D_K]
        float *k_local = smem + D_K;                 // [D_K]
        float *reduce_kv = smem + 2 * D_K;           // [blockDim.x]
        float *reduce_out = reduce_kv + blockDim.x;  // [blockDim.x]

        const float scale = cuda_gdn_query_scale(D_K);
        __shared__ float warp_sums[1];
        __shared__ float decay_shared;
        __shared__ float beta_shared;
        const int lane_id = tid % 32;
        /*
         * Each lane owns one contiguous state partition. The compile-time
         * extent is essential: nvcc scalarizes this array into registers, so
         * grouped rows reuse live state without local-memory spills.
         */
        float state_rows[kRowsPerSplit];
        const int row_begin = split_id * kRowsPerSplit;
        if (vi < d_v)
        {
#pragma unroll
            for (int local_row = 0; local_row < kRowsPerSplit; ++local_row)
            {
                const int state_row = row_begin + local_row;
                state_rows[local_row] = S_initial[state_row * d_v + vi];
            }
        }
        else
        {
#pragma unroll
            for (int local_row = 0; local_row < kRowsPerSplit; ++local_row)
                state_rows[local_row] = 0.0f;
        }

        for (int request_row = 0; request_row < request_seq_len; ++request_row)
        {
            const int flat_row = request * request_seq_len + request_row;
            const float *q_head =
                q + static_cast<size_t>(flat_row) * request_qk_stride +
                static_cast<size_t>(h) * D_K;
            const float *k_head =
                k + static_cast<size_t>(flat_row) * request_qk_stride +
                static_cast<size_t>(h) * D_K;
            const float *v_head =
                v + static_cast<size_t>(flat_row) * request_v_stride +
                static_cast<size_t>(h) * d_v;
            float *o_head =
                output + static_cast<size_t>(flat_row) * request_v_stride +
                static_cast<size_t>(h) * d_v;

            const int logical_row = effective_row_idx + request_row;
            if (effective_seq_len_ptr &&
                logical_row >= effective_seq_len_ptr[request])
            {
                if (vi < d_v && split_id == 0)
                    o_head[vi] = 0.0f;
                __syncthreads();
                continue;
            }

            // Load and preprocess Q/K exactly as the M=1 scalar route does.
            for (int i = tid; i < D_K; i += blockDim.x)
            {
                q_local[i] = q_head[i];
                k_local[i] = k_head[i];
            }
            __syncthreads();

            if (use_qk_l2norm)
            {
                if (tid < 32)
                {
                    float q_sum = 0.0f;
                    for (int i = tid; i < D_K; i += 32)
                        q_sum += q_local[i] * q_local[i];
                    for (int offset = 16; offset > 0; offset >>= 1)
                    {
                        q_sum +=
                            __shfl_xor_sync(0xFFFFFFFF, q_sum, offset);
                    }
                    if (lane_id == 0)
                        warp_sums[0] = q_sum;
                }
                __syncthreads();
                const float q_inv =
                    scale / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
                for (int i = tid; i < D_K; i += blockDim.x)
                    q_local[i] *= q_inv;
                __syncthreads();

                if (tid < 32)
                {
                    float k_sum = 0.0f;
                    for (int i = tid; i < D_K; i += 32)
                        k_sum += k_local[i] * k_local[i];
                    for (int offset = 16; offset > 0; offset >>= 1)
                    {
                        k_sum +=
                            __shfl_xor_sync(0xFFFFFFFF, k_sum, offset);
                    }
                    if (lane_id == 0)
                        warp_sums[0] = k_sum;
                }
                __syncthreads();
                const float k_inv =
                    1.0f / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
                for (int i = tid; i < D_K; i += blockDim.x)
                    k_local[i] *= k_inv;
                __syncthreads();
            }
            else
            {
                for (int i = tid; i < D_K; i += blockDim.x)
                    q_local[i] *= scale;
                __syncthreads();
            }

            if (tid == 0)
            {
                const size_t gate_index =
                    static_cast<size_t>(flat_row) *
                        static_cast<size_t>(n_heads) +
                    h;
                const float x = alpha[gate_index] + dt_bias[h];
                const float sp = (x > 20.0f) ? x : log1pf(expf(x));
                decay_shared = expf(A_log[h] * sp);
                beta_shared =
                    1.0f / (1.0f + expf(-beta_raw[gate_index]));
            }
            __syncthreads();
            const float decay = decay_shared;
            const float beta_h = beta_shared;

            float partial_kv = 0.0f;
            if (vi < d_v)
            {
#pragma unroll
                for (int local_row = 0;
                     local_row < kRowsPerSplit;
                     ++local_row)
                {
                    const int state_row = row_begin + local_row;
                    state_rows[local_row] *= decay;
                    partial_kv += state_rows[local_row] * k_local[state_row];
                }
            }
            reduce_kv[tid] = partial_kv;
            __syncthreads();

            float delta = 0.0f;
            if (vi < d_v)
            {
                float kv = 0.0f;
#pragma unroll
                for (int split = 0;
                     split < kGdnRecurrentRowSplit;
                     ++split)
                {
                    kv += reduce_kv[
                        column_in_block +
                        split * kGdnRecurrentColumnsPerBlock];
                }
                delta = (v_head[vi] - kv) * beta_h;
            }

            float partial_out = 0.0f;
            float *snapshot =
                state_snapshots && flat_row < max_snapshot_rows
                    ? state_snapshots +
                          static_cast<size_t>(flat_row) *
                              static_cast<size_t>(snapshot_stride_floats) +
                          static_cast<size_t>(h) * D_K * d_v
                    : nullptr;
            if (vi < d_v)
            {
#pragma unroll
                for (int local_row = 0;
                     local_row < kRowsPerSplit;
                     ++local_row)
                {
                    const int state_row = row_begin + local_row;
                    const float s_new =
                        state_rows[local_row] +
                        k_local[state_row] * delta;
                    state_rows[local_row] = s_new;
                    if (snapshot)
                        snapshot[state_row * d_v + vi] = s_new;
                    partial_out += s_new * q_local[state_row];
                }
            }
            reduce_out[tid] = partial_out;
            __syncthreads();

            if (vi < d_v && split_id == 0)
            {
                float out_vi = 0.0f;
#pragma unroll
                for (int split = 0;
                     split < kGdnRecurrentRowSplit;
                     ++split)
                {
                    out_vi += reduce_out[
                        column_in_block +
                        split * kGdnRecurrentColumnsPerBlock];
                }
                o_head[vi] = out_vi;
            }

            // Shared Q/K and reduction slots belong to exactly one causal row.
            __syncthreads();
        }

        const bool terminal_state_is_snapshotted =
            state_snapshots != nullptr &&
            max_snapshot_rows >= request_count * request_seq_len;
        if (vi < d_v && !terminal_state_is_snapshotted)
        {
#pragma unroll
            for (int local_row = 0; local_row < kRowsPerSplit; ++local_row)
            {
                const int state_row = row_begin + local_row;
                S_updated[state_row * d_v + vi] = state_rows[local_row];
            }
        }
    }

    /**
     * @brief Launch the fixed CUDA GDN recurrence specialization for a graph.
     *
     * GDN key width is model topology, not a hot-path tuning input. Selecting
     * the concrete specialization here makes register ownership and launch
     * geometry part of capture construction. Unknown widths fail immediately;
     * there is no scalar or eager alternate path.
     */
    bool launch_cuda_gdn_recurrent_step(
        const char *caller,
        const float *q,
        const float *k,
        const float *v,
        const float *alpha,
        const float *beta_raw,
        const float *A_log,
        const float *dt_bias,
        float *output,
        const float *initial_state,
        float *updated_state,
        int request_count,
        int request_seq_len,
        int n_heads,
        int d_k,
        int d_v,
        bool use_qk_l2norm,
        const int *effective_seq_len_ptr,
        int effective_row_idx,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        cudaStream_t stream)
    {
        const dim3 grid(
            request_count * n_heads,
            (d_v + kGdnRecurrentColumnsPerBlock - 1) /
                kGdnRecurrentColumnsPerBlock);
        const size_t shared_bytes =
            static_cast<size_t>(2 * d_k + 2 * kGdnRecurrentThreads) *
            sizeof(float);

#define LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT(D_K, OUT_OF_PLACE)                  \
    cuda_gdn_recurrent_step_kernel<D_K, OUT_OF_PLACE>                          \
        <<<grid, kGdnRecurrentThreads, shared_bytes, stream>>>(                 \
            q, k, v, alpha, beta_raw, A_log, dt_bias, output,                  \
            initial_state, updated_state,                                      \
            request_count, request_seq_len, n_heads, d_v, use_qk_l2norm,       \
            effective_seq_len_ptr, effective_row_idx, state_snapshots,          \
            snapshot_stride_floats, max_snapshot_rows)

        switch (d_k)
        {
        case 64:
            if (initial_state == updated_state)
                LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT(64, false);
            else
                LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT(64, true);
            break;
        case 128:
            if (initial_state == updated_state)
                LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT(128, false);
            else
                LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT(128, true);
            break;
        default:
            fprintf(
                stderr,
                "[%s] unsupported CUDA GDN d_k=%d; capture supports d_k in "
                "{64,128}\n",
                caller,
                d_k);
            return false;
        }

#undef LLAMINAR_LAUNCH_CUDA_GDN_RECURRENT

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[%s] %s\n", caller, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // Prefill Recurrence Kernel (seq_len>1, multi-block per head)
    //
    // Optimizations vs V1:
    //   1. 2D grid: dim3(n_heads, col_blocks) — each column block processes
    //      a subset of the d_v columns. Columns are fully independent in the
    //      delta-rule recurrence, so no cross-block sync is needed.
    //   2. Fused recurrence: decay each resident state row exactly once while
    //      accumulating K*S, then update that same resident row and accumulate
    //      Q*S. This is the serial-decode arithmetic schedule.
    //   3. A fixed eight-lane reduction tree owns each output column. D_K is a
    //      template parameter so both supported widths have complete row
    //      coverage and no runtime-width indexing can silently truncate state.
    // =========================================================================

    template <int D_K>
    __global__ __launch_bounds__(64, 10) void cuda_gdn_chunk_forward_kernel(
        const float *__restrict__ Q,        // [seq_len, n_heads * d_k]
        const float *__restrict__ K,        // [seq_len, n_heads * d_k]
        const float *__restrict__ V,        // [seq_len, n_heads * d_v]
        const float *__restrict__ decay,    // preprocessed [seq_len, n_heads]
        const float *__restrict__ beta,     // preprocessed [seq_len, n_heads]
        float *__restrict__ output,         // [seq_len, n_heads * d_v]
        const float *initial_state,
        float *updated_state,
        const int *__restrict__ effective_seq_len_ptr,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int request_count, int request_seq_len,
        int n_heads, int d_v)
    {
        static_assert(D_K % kGdnRecurrentRowSplit == 0);
        const int request = blockIdx.z;
        const int h = blockIdx.x;
        if (request >= request_count || h >= n_heads)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        // Row-split parallelism: eight lanes collaborate on each column and
        // own disjoint contiguous D_K/8 state rows.  Keep all eight lanes in
        // one warp so their partial sums can be exchanged without shared
        // reduction buffers or CTA-wide barriers.  Four columns fit in each
        // warp; a 64-thread block owns eight columns and exposes enough
        // independent CTAs to hide the remaining scalar and value-load latency.
        constexpr int ROW_SPLIT = kGdnRecurrentRowSplit;
        constexpr int ROWS_PER_SPLIT = D_K / ROW_SPLIT;
        constexpr int COLUMNS_PER_WARP = 32 / ROW_SPLIT;
        const int cols_per_block = block_size / ROW_SPLIT;
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;
        const int split_id = lane_id / COLUMNS_PER_WARP;
        const int col_in_warp = lane_id % COLUMNS_PER_WARP;
        const int col_in_block =
            warp_id * COLUMNS_PER_WARP + col_in_warp;
        const int vi = blockIdx.y * cols_per_block + col_in_block;

        const int qk_stride = n_heads * D_K;
        const int v_stride = n_heads * d_v;

        const int request_row_base = request * request_seq_len;
        const size_t request_state_stride =
            static_cast<size_t>(n_heads) * D_K * d_v;
        const float *S_initial = initial_state +
                                 static_cast<size_t>(request) * request_state_stride +
                                 static_cast<size_t>(h) * D_K * d_v;
        float *S_updated = updated_state +
                           static_cast<size_t>(request) * request_state_stride +
                           static_cast<size_t>(h) * D_K * d_v;

        extern __shared__ float smem[];
        float *q_stages = smem;             // [2][D_K]
        float *k_stages = smem + 2 * D_K;   // [2][D_K]

        int effective_seq_len = request_seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = effective_seq_len_ptr[request];
            effective_seq_len = raw_effective < 0
                                    ? 0
                                    : (raw_effective > request_seq_len
                                           ? request_seq_len
                                           : raw_effective);
        }

        // Keep this thread's recurrence-state slice resident across the entire
        // prefill. The recurrence is sequential in time, but each (head, column,
        // row-split) lane owns the same 32 state rows for every token; loading
        // them once removes hundreds of global-memory passes at bucket sizes.
        const int j_start = split_id * ROWS_PER_SPLIT;
        float sc[ROWS_PER_SPLIT];

        if (vi < d_v)
        {
#pragma unroll
            for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                sc[j] = S_initial[(j_start + j) * d_v + vi];
        }
        else
        {
#pragma unroll
            for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                sc[j] = 0.0f;
        }

        // Seed the first Q/K stage once.  Subsequent rows are loaded into the
        // opposite stage while the current causal recurrence is executing.
        int active_qk_stage = 0;
        if (effective_seq_len > 0)
        {
            const int first_row = request_row_base;
            cuda_gdn_stage_qk_async<D_K>(
                q_stages,
                k_stages,
                Q + first_row * qk_stride + h * D_K,
                K + first_row * qk_stride + h * D_K);
            cuda_gdn_wait_qk_async();
            __syncthreads();
        }

        // Process each timestep sequentially (inherent to recurrence).
        for (int t = 0; t < request_seq_len; t++)
        {
            const int row = request_row_base + t;
            const float *v_src = V + row * v_stride + h * d_v;
            float *o_dst = output + row * v_stride + h * d_v;

            if (t >= effective_seq_len)
            {
                if (vi < d_v && split_id == 0)
                    o_dst[vi] = 0.0f;
                continue;
            }

            const int next_qk_stage = active_qk_stage ^ 1;
            const bool has_next_row = t + 1 < effective_seq_len;
            if (has_next_row)
            {
                const int next_row = row + 1;
                cuda_gdn_stage_qk_async<D_K>(
                    q_stages + next_qk_stage * D_K,
                    k_stages + next_qk_stage * D_K,
                    Q + next_row * qk_stride + h * D_K,
                    K + next_row * qk_stride + h * D_K);
            }

            const float *q_local = q_stages + active_qk_stage * D_K;
            const float *k_local = k_stages + active_qk_stage * D_K;

            // The mandatory captured preprocessor publishes canonical Q/K and
            // gate values before this kernel. Long recurrence has no second
            // preprocessing regime or hidden scalar path.
            const int gate_idx = row * n_heads + h;
            const float decay_h = decay[gate_idx];
            const float beta_h = beta[gate_idx];

            float partial_kv = 0.0f;
            if (vi < d_v)
            {
#pragma unroll
                for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                {
                    sc[j] *= decay_h;
                    partial_kv += sc[j] * k_local[j_start + j];
                }
            }
            float kv = 0.0f;
            // Every lane named by the full-warp mask participates, including
            // lanes assigned to a padded tail column.  Their partial is zero,
            // which keeps the exchange valid without changing a real column.
#pragma unroll
            for (int s = 0; s < ROW_SPLIT; ++s)
            {
                // Preserve the historical arithmetic exactly: partition zero
                // is added first and partition seven last.  A tree reduction
                // would be faster-looking but would change rounding and break
                // serial-decode byte equivalence.
                const int source_lane =
                    s * COLUMNS_PER_WARP + col_in_warp;
                kv += __shfl_sync(
                    0xFFFFFFFF,
                    partial_kv,
                    source_lane);
            }

            float delta = 0.0f;
            if (vi < d_v)
            {
                delta = (v_src[vi] - kv) * beta_h;
            }

            float partial_out = 0.0f;
            if (vi < d_v)
            {
#pragma unroll
                for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                {
                    sc[j] += k_local[j_start + j] * delta;
                    partial_out += sc[j] * q_local[j_start + j];
                }
            }
            float out_vi = 0.0f;
#pragma unroll
            for (int s = 0; s < ROW_SPLIT; ++s)
            {
                const int source_lane =
                    s * COLUMNS_PER_WARP + col_in_warp;
                out_vi += __shfl_sync(
                    0xFFFFFFFF,
                    partial_out,
                    source_lane);
            }
            if (vi < d_v && split_id == 0)
            {
                o_dst[vi] = out_vi;
            }

            if (state_snapshots &&
                t < effective_seq_len &&
                row < max_snapshot_rows &&
                vi < d_v)
            {
                float *snapshot =
                    state_snapshots +
                    static_cast<size_t>(row) * static_cast<size_t>(snapshot_stride_floats) +
                    static_cast<size_t>(h) * static_cast<size_t>(D_K) * static_cast<size_t>(d_v);
#pragma unroll
                for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                    snapshot[(j_start + j) * d_v + vi] = sc[j];
            }
            if (has_next_row)
            {
                // Every thread finishes consuming the active stage before any
                // thread may recycle it two iterations later.  The same edge
                // also publishes the completed asynchronous next-row stage.
                cuda_gdn_wait_qk_async();
                __syncthreads();
                active_qk_stage = next_qk_stage;
            }
        }

        if (vi < d_v)
        {
#pragma unroll
            for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                S_updated[(j_start + j) * d_v + vi] = sc[j];
        }
    }

    __global__ void cuda_gdn_prefill_preprocess_kernel(
        float *__restrict__ Q,
        float *__restrict__ K,
        float *__restrict__ alpha,
        float *__restrict__ beta_raw,
        const float *__restrict__ A_log,
        const float *__restrict__ dt_bias,
        const int *__restrict__ effective_seq_lens,
        int request_count, int request_seq_len,
        int n_heads, int d_k,
        bool use_qk_l2norm)
    {
        const int work = blockIdx.x;
        const int h = work % n_heads;
        const int row = work / n_heads;
        const int request = row / request_seq_len;
        const int request_row = row - request * request_seq_len;
        if (request >= request_count || h >= n_heads)
            return;
        if (effective_seq_lens)
        {
            const int raw_effective = effective_seq_lens[request];
            const int effective = raw_effective < 0
                                      ? 0
                                      : (raw_effective > request_seq_len
                                             ? request_seq_len
                                             : raw_effective);
            if (request_row >= effective)
                return;
        }

        if (row >= request_count * request_seq_len)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;
        const int qk_stride = n_heads * d_k;
        const float scale = cuda_gdn_query_scale(d_k);

        float *q_head = Q + row * qk_stride + h * d_k;
        float *k_head = K + row * qk_stride + h * d_k;

        // Decode uses one fixed warp to reduce D_K. Long prefill must use that
        // same tree so preprocessing does not introduce a batch-size-dependent
        // rounding regime before the recurrent kernel begins.
        __shared__ float norm_sum;
        const int lane_id = tid % 32;

        if (use_qk_l2norm)
        {
            if (tid < 32)
            {
                float q_sum = 0.0f;
                for (int i = lane_id; i < d_k; i += 32)
                {
                    const float qv = q_head[i];
                    q_sum += qv * qv;
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    q_sum +=
                        __shfl_xor_sync(0xFFFFFFFF, q_sum, offset);
                if (lane_id == 0)
                    norm_sum = q_sum;
            }
            __syncthreads();
            const float q_inv =
                scale / fmaxf(sqrtf(norm_sum), 1e-6f);
            for (int i = tid; i < d_k; i += block_size)
                q_head[i] *= q_inv;
            __syncthreads();

            if (tid < 32)
            {
                float k_sum = 0.0f;
                for (int i = lane_id; i < d_k; i += 32)
                {
                    const float kv = k_head[i];
                    k_sum += kv * kv;
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    k_sum +=
                        __shfl_xor_sync(0xFFFFFFFF, k_sum, offset);
                if (lane_id == 0)
                    norm_sum = k_sum;
            }
            __syncthreads();
            const float k_inv =
                1.0f / fmaxf(sqrtf(norm_sum), 1e-6f);
            for (int i = tid; i < d_k; i += block_size)
                k_head[i] *= k_inv;
        }
        else
        {
            for (int i = tid; i < d_k; i += block_size)
                q_head[i] *= scale;
        }

        if (tid == 0)
        {
            const int gate_idx = row * n_heads + h;
            const float x = alpha[gate_idx] + dt_bias[h];
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            alpha[gate_idx] = expf(A_log[h] * sp);
            beta_raw[gate_idx] = 1.0f / (1.0f + expf(-beta_raw[gate_idx]));
        }
    }

    // =========================================================================
    // Short Conv1d Kernel - Decode (seq_len=1)
    //
    // One thread per channel, depthwise 1d convolution with SiLU.
    // conv_state[ch, kernel_size-1] stores history for causal convolution.
    // =========================================================================

    __global__ void cuda_short_conv1d_decode_kernel(
        const float *__restrict__ input,  // [channels]
        const float *__restrict__ weight, // [channels, kernel_size]
        const float *__restrict__ bias,   // [channels] or nullptr
        float *__restrict__ output,       // [channels]
        const float *initial_conv_state,
        float *updated_conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels)
            return;

        const int ks_minus1 = kernel_size - 1;

        // Compute convolution FIRST using OLD state, then shift
        // (must compute before shift to avoid overwriting state values)
        float sum = 0.0f;
        for (int k = 0; k < ks_minus1; k++)
            sum += initial_conv_state[ch * ks_minus1 + k] * weight[ch * kernel_size + k];
        sum += input[ch] * weight[ch * kernel_size + ks_minus1];

        if (bias)
            sum += bias[ch];

        // Apply SiLU activation
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));

        output[ch] = sum;

        // Now shift conv_state left by 1, insert new input at the end
        for (int k = 0; k < ks_minus1 - 1; k++)
            updated_conv_state[ch * ks_minus1 + k] =
                initial_conv_state[ch * ks_minus1 + k + 1];
        updated_conv_state[ch * ks_minus1 + ks_minus1 - 1] = input[ch];
    }

    // =========================================================================
    // Short Conv1d Kernel - Prefill (seq_len>1)
    //
    // One thread per (timestep, channel) pair.
    // Performs full causal 1d convolution over sequence.
    // =========================================================================

    __global__ void cuda_short_conv1d_prefill_kernel(
        const float *__restrict__ input,  // [seq_len, channels]
        const float *__restrict__ weight, // [channels, kernel_size]
        const float *__restrict__ bias,   // [channels] or nullptr
        float *__restrict__ output,       // [seq_len, channels]
        const float *__restrict__ initial_conv_state,
        const int *__restrict__ effective_seq_len_ptr,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int ks_minus1 = kernel_size - 1;
        const int total = seq_len * channels;
        if (idx >= total)
            return;

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
        }

        int t = idx / channels;
        int ch = idx % channels;

        float sum = 0.0f;
        if (t < effective_seq_len)
        {
            for (int k = 0; k < kernel_size; k++)
            {
                int src_t = t - ks_minus1 + k;
                float val = 0.0f;
                if (src_t >= 0)
                    val = input[src_t * channels + ch];
                else if (initial_conv_state)
                    val = initial_conv_state[ch * ks_minus1 + ks_minus1 + src_t];
                sum += val * weight[ch * kernel_size + k];
            }

            if (bias)
                sum += bias[ch];
            if (apply_silu)
                sum = sum / (1.0f + expf(-sum));
        }
        output[t * channels + ch] = sum;

        if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows)
        {
            float *snapshot =
                state_snapshots +
                static_cast<size_t>(t) * static_cast<size_t>(snapshot_stride_floats) +
                static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = t - ks_minus1 + 1 + state_idx;
                snapshot[state_idx] =
                    (src_t >= 0) ? input[src_t * channels + ch]
                                 : (initial_conv_state ? initial_conv_state[ch * ks_minus1 + ks_minus1 + src_t] : 0.0f);
            }
        }
    }

    /**
     * @brief Decode-equivalent grouped short-conv kernel for MTP verifier rows.
     *
     * The long-prefill kernel uses a second launch to update live state safely.
     * A verifier already needs every post-row snapshot, so this kernel keeps one
     * lane responsible for a channel, walks the runtime-sized row group in
     * causal order, writes every snapshot, and only then commits the channel's
     * final speculative state. There is one launch for the complete matrix and
     * no host or launch-level row replay.
     */
    __global__ void cuda_short_conv1d_small_m_kernel(
        const float *__restrict__ input,
        const float *__restrict__ weight,
        const float *__restrict__ bias,
        float *__restrict__ output,
        const float *initial_conv_state,
        float *updated_conv_state,
        const int *__restrict__ effective_seq_len_ptr,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        const int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels)
            return;

        const int ks_minus1 = kernel_size - 1;
        if (ks_minus1 <= 0)
            return;

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            effective_seq_len =
                raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
        }

        const float *channel_weight =
            weight + static_cast<size_t>(ch) * static_cast<size_t>(kernel_size);
        const float *channel_initial_state =
            initial_conv_state
                ? initial_conv_state + static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1)
                : nullptr;

        for (int t = 0; t < seq_len; ++t)
        {
            float sum = 0.0f;
            if (t < effective_seq_len)
            {
                for (int k = 0; k < kernel_size; ++k)
                {
                    const int src_t = t - ks_minus1 + k;
                    const float val =
                        (src_t >= 0)
                            ? input[static_cast<size_t>(src_t) * channels + ch]
                            : (channel_initial_state
                                   ? channel_initial_state[ks_minus1 + src_t]
                                   : 0.0f);
                    sum += val * channel_weight[k];
                }

                if (bias)
                    sum += bias[ch];
                if (apply_silu)
                    sum = sum / (1.0f + expf(-sum));
            }
            output[static_cast<size_t>(t) * channels + ch] = sum;

            if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows)
            {
                float *snapshot =
                    state_snapshots +
                    static_cast<size_t>(t) * static_cast<size_t>(snapshot_stride_floats) +
                    static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
                for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
                {
                    const int src_t = t - ks_minus1 + 1 + state_idx;
                    snapshot[state_idx] =
                        (src_t >= 0)
                            ? input[static_cast<size_t>(src_t) * channels + ch]
                            : (channel_initial_state
                                   ? channel_initial_state[ks_minus1 + src_t]
                                   : 0.0f);
                }
            }
        }

        const bool terminal_state_is_snapshotted =
            state_snapshots != nullptr && max_snapshot_rows >= seq_len;
        if (updated_conv_state && !terminal_state_is_snapshotted)
        {
            float *state =
                updated_conv_state + static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = effective_seq_len - ks_minus1 + state_idx;
                state[state_idx] =
                    (src_t >= 0 && src_t < effective_seq_len)
                        ? input[static_cast<size_t>(src_t) * channels + ch]
                    : channel_initial_state[ks_minus1 + src_t];
            }
        }
    }

    __global__ void cuda_short_conv1d_state_update_kernel(
        const float *__restrict__ input,
        const float *initial_conv_state,
        float *updated_conv_state,
        const int *__restrict__ effective_seq_len_ptr,
        int seq_len, int channels, int kernel_size)
    {
        int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels || !initial_conv_state || !updated_conv_state)
            return;

        const int ks_minus1 = kernel_size - 1;
        if (ks_minus1 <= 0)
            return;

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
        }

        const float *initial_state = initial_conv_state + ch * ks_minus1;
        float *updated_state = updated_conv_state + ch * ks_minus1;
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = effective_seq_len - ks_minus1 + state_idx;
            updated_state[state_idx] =
                (src_t >= 0 && src_t < effective_seq_len) ? input[src_t * channels + ch]
                                                           : initial_state[ks_minus1 + src_t];
        }
    }

    /**
     * @brief Grouped short-conv for small request-local row groups.
     *
     * One lane owns one `(request, channel)` pair and advances that channel in
     * row order. This preserves the scalar decode accumulation order while a
     * single launch covers the entire request matrix. Padded rows are zeroed,
     * and each request commits state after its own device-resident real length.
     */
    __global__ void cuda_short_conv1d_batched_small_m_kernel(
        const float *__restrict__ input,
        const float *__restrict__ weight,
        const float *__restrict__ bias,
        float *__restrict__ output,
        const float *initial_request_states,
        float *updated_request_states,
        const int *__restrict__ request_seq_lens,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int request_count,
        int request_seq_len,
        int channels,
        int kernel_size,
        bool apply_silu)
    {
        const int ch = blockIdx.x * blockDim.x + threadIdx.x;
        const int request = blockIdx.y;
        if (ch >= channels || request >= request_count)
            return;

        const int ks_minus1 = kernel_size - 1;
        if (ks_minus1 <= 0)
            return;

        const int raw_real_len = request_seq_lens[request];
        const int real_len =
            raw_real_len < 0 ? 0 :
            (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
        const size_t row_base =
            static_cast<size_t>(request) *
            static_cast<size_t>(request_seq_len);
        const size_t element_base = row_base * static_cast<size_t>(channels);
        const size_t state_base =
            static_cast<size_t>(request) *
            static_cast<size_t>(channels) *
            static_cast<size_t>(ks_minus1);
        const float *channel_initial_state =
            initial_request_states + state_base +
            static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
        float *channel_updated_state =
            updated_request_states + state_base +
            static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
        const float *channel_weight =
            weight + static_cast<size_t>(ch) * static_cast<size_t>(kernel_size);

        for (int t = 0; t < request_seq_len; ++t)
        {
            float sum = 0.0f;
            if (t < real_len)
            {
                for (int k = 0; k < kernel_size; ++k)
                {
                    const int src_t = t - ks_minus1 + k;
                    const float value =
                        src_t >= 0
                            ? input[element_base +
                                    static_cast<size_t>(src_t) * channels + ch]
                            : channel_initial_state[ks_minus1 + src_t];
                    sum += value * channel_weight[k];
                }
                if (bias)
                    sum += bias[ch];
                if (apply_silu)
                    sum = sum / (1.0f + expf(-sum));
            }
            output[element_base + static_cast<size_t>(t) * channels + ch] = sum;

            const int snapshot_row =
                request * request_seq_len + t;
            if (state_snapshots &&
                t < real_len &&
                snapshot_row < max_snapshot_rows)
            {
                float *snapshot =
                    state_snapshots +
                    static_cast<size_t>(snapshot_row) *
                        static_cast<size_t>(snapshot_stride_floats) +
                    static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
                for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
                {
                    const int src_t = t - ks_minus1 + 1 + state_idx;
                    snapshot[state_idx] =
                        src_t >= 0
                            ? input[element_base +
                                    static_cast<size_t>(src_t) * channels + ch]
                            : channel_initial_state[ks_minus1 + src_t];
                }
            }
        }

        const bool terminal_state_is_snapshotted =
            state_snapshots != nullptr &&
            max_snapshot_rows >= request_count * request_seq_len;
        if (!terminal_state_is_snapshotted)
        {
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = real_len - ks_minus1 + state_idx;
                channel_updated_state[state_idx] =
                    src_t >= 0 && src_t < real_len
                        ? input[element_base +
                                static_cast<size_t>(src_t) * channels + ch]
                        : channel_initial_state[ks_minus1 + src_t];
            }
        }
    }

    /**
     * @brief Grouped long-prefill convolution over flattened request rows.
     *
     * Every thread computes one `(request, row, channel)` output. Request-local
     * indexing prevents padding or a neighboring request from entering the
     * causal window. State publication is performed by the companion grouped
     * commit kernel after all outputs have stopped reading initial state.
     */
    __global__ void cuda_short_conv1d_batched_prefill_kernel(
        const float *__restrict__ input,
        const float *__restrict__ weight,
        const float *__restrict__ bias,
        float *__restrict__ output,
        const float *__restrict__ request_states,
        const int *__restrict__ request_seq_lens,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int request_count,
        int request_seq_len,
        int channels,
        int kernel_size,
        bool apply_silu)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int request_elements = request_seq_len * channels;
        const int total = request_count * request_elements;
        if (idx >= total)
            return;

        const int request = idx / request_elements;
        const int request_element = idx - request * request_elements;
        const int t = request_element / channels;
        const int ch = request_element - t * channels;
        const int ks_minus1 = kernel_size - 1;
        const int raw_real_len = request_seq_lens[request];
        const int real_len =
            raw_real_len < 0 ? 0 :
            (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
        const size_t row_base =
            static_cast<size_t>(request) *
            static_cast<size_t>(request_seq_len);
        const size_t element_base = row_base * static_cast<size_t>(channels);
        const float *channel_state =
            request_states +
            static_cast<size_t>(request) * channels * ks_minus1 +
            static_cast<size_t>(ch) * ks_minus1;

        float sum = 0.0f;
        if (t < real_len)
        {
            for (int k = 0; k < kernel_size; ++k)
            {
                const int src_t = t - ks_minus1 + k;
                const float value =
                    src_t >= 0
                        ? input[element_base +
                                static_cast<size_t>(src_t) * channels + ch]
                        : channel_state[ks_minus1 + src_t];
                sum += value * weight[ch * kernel_size + k];
            }
            if (bias)
                sum += bias[ch];
            if (apply_silu)
                sum = sum / (1.0f + expf(-sum));
        }
        output[element_base + static_cast<size_t>(t) * channels + ch] = sum;

        const int snapshot_row = request * request_seq_len + t;
        if (state_snapshots &&
            t < real_len &&
            snapshot_row < max_snapshot_rows)
        {
            float *snapshot =
                state_snapshots +
                static_cast<size_t>(snapshot_row) *
                    static_cast<size_t>(snapshot_stride_floats) +
                static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = t - ks_minus1 + 1 + state_idx;
                snapshot[state_idx] =
                    src_t >= 0
                        ? input[element_base +
                                static_cast<size_t>(src_t) * channels + ch]
                        : channel_state[ks_minus1 + src_t];
            }
        }
    }

    /** @brief Commit each request's terminal real-row convolution state. */
    __global__ void cuda_short_conv1d_batched_state_update_kernel(
        const float *__restrict__ input,
        const float *initial_request_states,
        float *updated_request_states,
        const int *__restrict__ request_seq_lens,
        int request_count,
        int request_seq_len,
        int channels,
        int kernel_size)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = request_count * channels;
        if (idx >= total)
            return;

        const int request = idx / channels;
        const int ch = idx - request * channels;
        const int ks_minus1 = kernel_size - 1;
        if (ks_minus1 <= 0)
            return;
        const int raw_real_len = request_seq_lens[request];
        const int real_len =
            raw_real_len < 0 ? 0 :
            (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
        const size_t element_base =
            static_cast<size_t>(request) *
            static_cast<size_t>(request_seq_len) *
            static_cast<size_t>(channels);
        const float *channel_initial_state =
            initial_request_states +
            static_cast<size_t>(request) * channels * ks_minus1 +
            static_cast<size_t>(ch) * ks_minus1;
        float *channel_updated_state =
            updated_request_states +
            static_cast<size_t>(request) * channels * ks_minus1 +
            static_cast<size_t>(ch) * ks_minus1;

        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = real_len - ks_minus1 + state_idx;
            channel_updated_state[state_idx] =
                src_t >= 0 && src_t < real_len
                    ? input[element_base +
                            static_cast<size_t>(src_t) * channels + ch]
                    : channel_initial_state[ks_minus1 + src_t];
        }
    }

    // =========================================================================
    // GatedRMSNorm Kernel
    //
    // output[i] = RMSNorm(input, gamma)[i] * gate_act[i]
    // where gate_act = SiLU(gate) when gate_silu=true, else gate_act = gate
    //
    // V2: Cooperative block design — one block per (seq, group) pair.
    // Threads within a block cooperate on the norm_dim reduction via warp shuffle.
    // Eliminates the V1 problem of 1 thread per (seq, group) which only used
    // 20 threads at decode time.
    // =========================================================================

    __global__ void cuda_gated_rmsnorm_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate,
        const float *__restrict__ gamma,
        float *__restrict__ output,
        int total_work,   // seq_len * n_groups
        int n_groups,     // number of norm groups per row (d_model / norm_dim)
        int norm_dim,     // normalization dimension
        int d_model,      // full model dimension
        int gamma_period, // gamma cycling period
        float eps,
        bool subtract_one,
        bool gate_silu)
    {
        int work_idx = blockIdx.x;
        if (work_idx >= total_work)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        int t = work_idx / n_groups;
        int g = work_idx % n_groups;
        int offset = t * d_model + g * norm_dim;

        // ── Pass 1: Cooperative sum-of-squares via register caching + warp shuffle ──
        constexpr int kMaxPerThread = 16;
        float local_vals[kMaxPerThread];
        int elems_per_thread = (norm_dim + block_size - 1) / block_size;

        float sum_sq = 0.0f;
        for (int e = 0; e < elems_per_thread; e++)
        {
            int j = tid + e * block_size;
            float v = 0.0f;
            if (j < norm_dim)
            {
                v = input[offset + j];
                sum_sq += v * v;
            }
            if (e < kMaxPerThread)
                local_vals[e] = v;
        }

        // Warp-level reduction (32-wide warps)
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

        // Cross-warp reduction via shared memory
        __shared__ float warp_sums[32]; // max 1024 threads = 32 warps
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int num_warps = (block_size + 31) / 32;
        if (lane_id == 0)
            warp_sums[warp_id] = sum_sq;
        __syncthreads();
        if (tid == 0)
        {
            float total = 0.0f;
            for (int w = 0; w < num_warps; w++)
                total += warp_sums[w];
            warp_sums[0] = total;
        }
        __syncthreads();
        float inv_rms = rsqrtf(warp_sums[0] / (float)norm_dim + eps);

        // ── Pass 2: Normalize, apply gamma, multiply by gate (from registers) ──
        for (int e = 0; e < elems_per_thread; e++)
        {
            int j = tid + e * block_size;
            if (j < norm_dim)
            {
                float normalized = local_vals[e] * inv_rms;
                int gamma_idx = j % gamma_period;
                float gamma_eff = subtract_one ? (1.0f + gamma[gamma_idx]) : gamma[gamma_idx];
                float gate_val = gate[offset + j];
                float gate_act;
                if (gate_silu)
                    gate_act = gate_val / (1.0f + expf(-gate_val)); // SiLU
                else
                    gate_act = gate_val;
                output[offset + j] = normalized * gamma_eff * gate_act;
            }
        }
    }

    // =========================================================================
    // AttentionOutputGate Kernel
    //
    // output[i] = sigmoid(gate[i]) * input[i]
    // =========================================================================

    __global__ void cuda_attention_output_gate_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate,
        float *__restrict__ output,
        int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            float sig = 1.0f / (1.0f + expf(-gate[idx]));
            output[idx] = sig * input[idx];
        }
    }

} // anonymous namespace (kernel functions)

// =========================================================================
// QKV Deinterleave Kernel (for merged QKV buffer on device)
// =========================================================================

namespace
{

    /**
     * Deinterleave merged QKV buffer into separate contiguous Q, K, V arrays.
     *
     * Merged layout per row t: [Q0..Q_{q_src_dim-1} | K0..K_{k_src_dim-1} | V0..V_{v_dim-1}]
     * stride = q_src_dim + k_src_dim + v_dim
     *
     * Output Q/K: [seq_len, n_v_heads * d_k], V: [seq_len, n_v_heads * d_v]
     * Head mapping:
     *   k_head_for_v_head_j = (j + global_v_offset) % n_k_heads
     */
    __global__ void cuda_gdn_deinterleave_qkv_kernel(
        const float *__restrict__ merged,
        float *__restrict__ out_q,
        float *__restrict__ out_k,
        float *__restrict__ out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset)
    {
        if (n_k_heads <= 0 || n_v_heads <= 0 || d_k <= 0 || d_v <= 0)
            return;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Layout sizes
        int q_src_dim = n_k_heads * d_k;
        int k_src_dim = n_k_heads * d_k;
        int v_dim = n_v_heads * d_v;
        int stride = q_src_dim + k_src_dim + v_dim;

        int q_dst_dim = n_v_heads * d_k;
        int k_dst_dim = n_v_heads * d_k;

        // Total elements: Q + K + V = seq_len * (q_dst_dim + k_dst_dim + v_dim)
        int q_total = seq_len * q_dst_dim;
        int k_total = seq_len * k_dst_dim;
        int v_total = seq_len * v_dim;
        int total = q_total + k_total + v_total;

        if (idx >= total)
            return;

        if (idx < q_total)
        {
            // Q region
            int t = idx / q_dst_dim;
            int rem = idx % q_dst_dim;
            int j = rem / d_k; // output head index
            int d = rem % d_k; // element within head
            int k_idx = (j + global_v_offset) % n_k_heads;
            if (k_idx < 0)
                k_idx += n_k_heads;
            out_q[idx] = merged[t * stride + k_idx * d_k + d];
        }
        else if (idx < q_total + k_total)
        {
            // K region
            int local = idx - q_total;
            int t = local / k_dst_dim;
            int rem = local % k_dst_dim;
            int j = rem / d_k;
            int d = rem % d_k;
            int k_idx = (j + global_v_offset) % n_k_heads;
            if (k_idx < 0)
                k_idx += n_k_heads;
            out_k[local] = merged[t * stride + q_src_dim + k_idx * d_k + d];
        }
        else
        {
            // V region: straight copy (already n_v_heads wide)
            int local = idx - q_total - k_total;
            int t = local / v_dim;
            int d = local % v_dim;
            out_v[local] = merged[t * stride + q_src_dim + k_src_dim + d];
        }
    }

    /**
     * Copy one captured verifier state row selected by a device scalar.
     *
     * This is intentionally tiny and graph-capturable: the row selector is read
     * on the same stream as verifier outcome publication, so no D2H row-index
     * synchronization is needed before restoring GDN/short-conv live state.
     */
    __global__ void cuda_gdn_copy_capture_row_from_device_index_kernel(
        float *__restrict__ dst,
        const float *__restrict__ capture,
        const int *__restrict__ row_index,
        int rows,
        int state_size)
    {
        if (!dst || !capture || !row_index || rows <= 0 || state_size <= 0)
            return;
        const int row = *row_index;
        if (row < 0 || row >= rows)
            return;

        const float *src =
            capture + static_cast<size_t>(row) * static_cast<size_t>(state_size);
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < state_size;
             i += stride)
        {
            dst[i] = src[i];
        }
    }

    /**
     * @brief Copy one accepted verifier state row into each request live slot.
     *
     * Row indices are already device-resident speculative metadata. Negative
     * row indices mean "no accepted row for this request", so the live slot is
     * left untouched. The capture buffer uses the flat verifier-row namespace
     * shared by the transaction metadata.
     */
    __global__ void cuda_gdn_copy_capture_rows_from_device_indices_kernel(
        float *__restrict__ dst,
        const float *__restrict__ capture,
        const int *__restrict__ row_indices,
        int request_count,
        int row_index_stride,
        int request_row_width,
        int rows,
        int state_size)
    {
        if (!dst || !capture || !row_indices ||
            request_count <= 0 || row_index_stride <= 0 ||
            rows <= 0 || state_size <= 0)
        {
            return;
        }

        const int total = request_count * state_size;
        const int stride = blockDim.x * gridDim.x;
        for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
             linear < total;
             linear += stride)
        {
            const int request = linear / state_size;
            const int state_offset = linear - request * state_size;
            const int metadata_value = row_indices[request * row_index_stride];
            const int row =
                request_row_width > 0
                    ? request * request_row_width + metadata_value - 1
                    : metadata_value;
            if (row < 0 || row >= rows)
                continue;
            dst[linear] =
                capture[static_cast<size_t>(row) * static_cast<size_t>(state_size) +
                        static_cast<size_t>(state_offset)];
        }
    }

    /**
     * @brief Convert rank-major allgather output to modulo-linked group order.
     *
     * Prefix groups describe Q and K when present. Value groups are one group
     * per modulo repeat. Every group is uniformly partitioned by key-head
     * ownership, so each output element resolves to exactly one participant
     * and one participant-local offset without a lookup table.
     */
    __global__ void cuda_gdn_reassemble_modulo_linked_state_kernel(
        const float *__restrict__ gathered,
        float *__restrict__ full,
        int degree,
        int global_key_heads,
        int repeat_factor,
        int key_elements_per_head,
        int value_elements_per_head,
        int prefix_group_count)
    {
        if (!gathered || !full ||
            degree <= 0 || global_key_heads <= 0 ||
            global_key_heads % degree != 0 || repeat_factor <= 0 ||
            key_elements_per_head < 0 || value_elements_per_head <= 0 ||
            prefix_group_count < 0)
        {
            return;
        }

        const int local_key_heads = global_key_heads / degree;
        const int local_key_group =
            local_key_heads * key_elements_per_head;
        const int full_key_group =
            global_key_heads * key_elements_per_head;
        const int local_value_group =
            local_key_heads * value_elements_per_head;
        const int full_value_group =
            global_key_heads * value_elements_per_head;
        const int local_prefix = prefix_group_count * local_key_group;
        const int full_prefix = prefix_group_count * full_key_group;
        const int local_state_size =
            local_prefix + repeat_factor * local_value_group;
        const int full_state_size =
            full_prefix + repeat_factor * full_value_group;
        const int stride = blockDim.x * gridDim.x;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < full_state_size;
             idx += stride)
        {
            int participant = 0;
            int participant_offset = 0;
            if (idx < full_prefix)
            {
                const int group = idx / full_key_group;
                const int within_group = idx - group * full_key_group;
                participant = within_group / local_key_group;
                participant_offset =
                    group * local_key_group + within_group % local_key_group;
            }
            else
            {
                const int value_offset = idx - full_prefix;
                const int repeat = value_offset / full_value_group;
                const int within_group =
                    value_offset - repeat * full_value_group;
                participant = within_group / local_value_group;
                participant_offset =
                    local_prefix + repeat * local_value_group +
                    within_group % local_value_group;
            }
            full[idx] = gathered[
                static_cast<size_t>(participant) *
                    static_cast<size_t>(local_state_size) +
                static_cast<size_t>(participant_offset)];
        }
    }

} // anonymous namespace (deinterleave kernel)

// =========================================================================
// GPU transfer helpers (called from headers via extern "C")
// =========================================================================

extern "C"
{

    void cudaGDN_gpu_memset_zero(float *ptr, size_t count)
    {
        if (ptr && count > 0)
            cudaMemset(ptr, 0, count * sizeof(float));
    }

    void cudaGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream)
    {
        if (ptr && count > 0)
            cudaMemsetAsync(ptr, 0, count * sizeof(float), (cudaStream_t)stream);
    }

    void cudaGDN_gpu_set_device(int ordinal)
    {
        cudaSetDevice(ordinal);
    }

    void cudaGDN_gpu_memcpy(float *dst, const float *src, size_t count)
    {
        cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDefault);
    }

    void cudaGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream)
    {
        cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDefault, (cudaStream_t)stream);
    }

    void cudaGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count)
    {
        cudaMemcpy(host_dst, device_src, count * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void cudaGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream)
    {
        cudaMemcpyAsync(host_dst, device_src, count * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    }

    void cudaGDN_stream_synchronize(void *stream)
    {
        cudaStreamSynchronize((cudaStream_t)stream);
    }

    bool cudaGDN_gpu_copy_capture_row_from_device_index(
        float *dst,
        const float *capture,
        const int *device_row_index,
        int rows,
        int state_size,
        int device_idx,
        void *stream)
    {
        if (!dst || !capture || !device_row_index ||
            rows <= 0 || state_size <= 0 || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int blocks = (state_size + threads - 1) / threads;
        if (blocks < 1)
            blocks = 1;
        if (blocks > 1024)
            blocks = 1024;
        cuda_gdn_copy_capture_row_from_device_index_kernel<<<
            blocks,
            threads,
            0,
            (cudaStream_t)stream>>>(
            dst,
            capture,
            device_row_index,
            rows,
            state_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_gpu_copy_capture_row_from_device_index] %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_gpu_copy_capture_rows_from_device_indices(
        float *dst,
        const float *capture,
        const int *device_row_indices,
        int request_count,
        int row_index_stride,
        int rows,
        int state_size,
        int device_idx,
        void *stream)
    {
        if (!dst || !capture || !device_row_indices ||
            request_count <= 0 || row_index_stride <= 0 ||
            rows <= 0 || state_size <= 0 || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int blocks = ((request_count * state_size) + threads - 1) / threads;
        if (blocks < 1)
            blocks = 1;
        if (blocks > 1024)
            blocks = 1024;
        cuda_gdn_copy_capture_rows_from_device_indices_kernel<<<
            blocks,
            threads,
            0,
            (cudaStream_t)stream>>>(
            dst,
            capture,
            device_row_indices,
            request_count,
            row_index_stride,
            /*request_row_width=*/0,
            rows,
            state_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_gpu_copy_capture_rows_from_device_indices] %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_gpu_copy_capture_terminal_rows_from_device_lengths(
        float *dst,
        const float *capture,
        const int *device_request_seq_lens,
        int request_count,
        int request_row_width,
        int rows,
        int state_size,
        int device_idx,
        void *stream)
    {
        if (!dst || !capture || !device_request_seq_lens ||
            request_count <= 0 || request_row_width <= 0 ||
            rows < request_count * request_row_width ||
            state_size <= 0 || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int blocks = ((request_count * state_size) + threads - 1) / threads;
        blocks = std::max(1, std::min(blocks, 1024));
        cuda_gdn_copy_capture_rows_from_device_indices_kernel<<<
            blocks,
            threads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            dst,
            capture,
            device_request_seq_lens,
            request_count,
            /*row_index_stride=*/1,
            request_row_width,
            rows,
            state_size);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_gpu_copy_capture_terminal_rows_from_device_lengths] %s\n",
                    cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    bool cudaGDN_reassemble_modulo_linked_state(
        const float *gathered,
        float *full,
        int degree,
        int global_key_heads,
        int global_value_heads,
        int key_elements_per_head,
        int value_elements_per_head,
        int prefix_group_count,
        int device_idx,
        void *stream)
    {
        if (!gathered || !full || !stream ||
            degree <= 0 || global_key_heads <= 0 ||
            global_value_heads <= 0 ||
            global_key_heads % degree != 0 ||
            global_value_heads % global_key_heads != 0 ||
            (prefix_group_count != 0 && prefix_group_count != 2) ||
            (prefix_group_count > 0 && key_elements_per_head <= 0) ||
            value_elements_per_head <= 0)
        {
            return false;
        }

        const long long full_state_size_wide =
            static_cast<long long>(prefix_group_count) * global_key_heads *
                key_elements_per_head +
            static_cast<long long>(global_value_heads) *
                value_elements_per_head;
        if (full_state_size_wide <= 0 ||
            full_state_size_wide > std::numeric_limits<int>::max())
        {
            return false;
        }

        cudaError_t set_err = cudaSetDevice(device_idx);
        if (set_err != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_reassemble_modulo_linked_state] cudaSetDevice(%d) failed: %s\n",
                    device_idx,
                    cudaGetErrorString(set_err));
            return false;
        }
        (void)cudaGetLastError();
        const int full_state_size = static_cast<int>(full_state_size_wide);
        constexpr int threads = 256;
        int blocks = (full_state_size + threads - 1) / threads;
        if (blocks < 1)
            blocks = 1;
        if (blocks > 1024)
            blocks = 1024;
        cuda_gdn_reassemble_modulo_linked_state_kernel<<<
            blocks,
            threads,
            0,
            (cudaStream_t)stream>>>(
            gathered,
            full,
            degree,
            global_key_heads,
            global_value_heads / global_key_heads,
            key_elements_per_head,
            value_elements_per_head,
            prefix_group_count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_reassemble_modulo_linked_state] %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C" (memory helpers)

// =========================================================================
// QKV Deinterleave Wrapper
// =========================================================================

extern "C"
{

    bool cudaGDN_deinterleave_qkv(
        const float *merged, float *out_q, float *out_k, float *out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset,
        int device_idx, void *stream)
    {
        cudaError_t set_device_err = cudaSetDevice(device_idx);
        if (set_device_err != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_deinterleave_qkv] cudaSetDevice(%d) failed: %s "
                    "merged=%p out_q=%p out_k=%p out_v=%p stream=%p\n",
                    device_idx, cudaGetErrorString(set_device_err),
                    (const void *)merged, (void *)out_q, (void *)out_k, (void *)out_v, stream);
            return false;
        }

        int q_dst_dim = n_v_heads * d_k;
        int k_dst_dim = n_v_heads * d_k;
        int v_dim = n_v_heads * d_v;
        int total = seq_len * (q_dst_dim + k_dst_dim + v_dim);
        if (seq_len <= 0 || n_k_heads <= 0 || n_v_heads <= 0 ||
            d_k <= 0 || d_v <= 0 || total <= 0)
        {
            fprintf(stderr,
                    "[cudaGDN_deinterleave_qkv] invalid launch geometry: "
                    "seq_len=%d n_k_heads=%d n_v_heads=%d d_k=%d d_v=%d "
                    "global_v_offset=%d total=%d stream=%p\n",
                    seq_len, n_k_heads, n_v_heads, d_k, d_v,
                    global_v_offset, total, stream);
            return false;
        }

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cudaError_t pre_launch_sticky = cudaPeekAtLastError();
        cudaStreamCaptureStatus pre_capture_status = cudaStreamCaptureStatusNone;
        cudaError_t pre_capture_query =
            stream ? cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &pre_capture_status) : cudaSuccess;
        cudaError_t pre_stream_query = cudaSuccess;
        if (stream &&
            pre_capture_query == cudaSuccess &&
            pre_capture_status == cudaStreamCaptureStatusNone)
        {
            pre_stream_query = cudaStreamQuery(static_cast<cudaStream_t>(stream));
        }

        cuda_gdn_deinterleave_qkv_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            merged, out_q, out_k, out_v,
            seq_len, n_k_heads, n_v_heads, d_k, d_v, global_v_offset);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            int current_device = -1;
            (void)cudaGetDevice(&current_device);
            cudaError_t post_stream_query = cudaSuccess;
            if (stream &&
                pre_capture_query == cudaSuccess &&
                pre_capture_status == cudaStreamCaptureStatusNone)
            {
                post_stream_query = cudaStreamQuery(static_cast<cudaStream_t>(stream));
            }
            fprintf(stderr,
                    "[cudaGDN_deinterleave_qkv] %s "
                    "(seq_len=%d n_k_heads=%d n_v_heads=%d d_k=%d d_v=%d "
                    "global_v_offset=%d total=%d blocks=%d requested_device=%d current_device=%d "
                    "merged=%p out_q=%p out_k=%p out_v=%p stream=%p "
                    "pre_sticky=%s pre_stream_query=%s post_stream_query=%s "
                    "capture_query=%s capture_status=%d)\n",
                    cudaGetErrorString(err), seq_len, n_k_heads, n_v_heads,
                    d_k, d_v, global_v_offset, total, blocks,
                    device_idx, current_device,
                    (const void *)merged, (void *)out_q, (void *)out_k, (void *)out_v, stream,
                    cudaGetErrorString(pre_launch_sticky),
                    cudaGetErrorString(pre_stream_query),
                    cudaGetErrorString(post_stream_query),
                    cudaGetErrorString(pre_capture_query),
                    static_cast<int>(pre_capture_status));
            return false;
        }
        return true;
    }

} // extern "C" (deinterleave)

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaGDN_recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        return launch_cuda_gdn_recurrent_step(
            "cudaGDN_recurrent_step",
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            /*request_count=*/1, /*request_seq_len=*/1,
            n_heads, d_k, d_v,
            use_qk_l2norm,
            nullptr,
            -1,
            /*state_snapshots=*/nullptr,
            /*snapshot_stride_floats=*/0,
            /*max_snapshot_rows=*/0,
            static_cast<cudaStream_t>(stream));
    }

    bool cudaGDN_recurrent_step_effective_row(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_len,
        int row_idx,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        return launch_cuda_gdn_recurrent_step(
            "cudaGDN_recurrent_step_effective_row",
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            /*request_count=*/1, /*request_seq_len=*/1,
            n_heads, d_k, d_v,
            use_qk_l2norm,
            device_effective_seq_len,
            row_idx,
            /*state_snapshots=*/nullptr,
            /*snapshot_stride_floats=*/0,
            /*max_snapshot_rows=*/0,
            static_cast<cudaStream_t>(stream));
    }

    bool cudaGDN_chunk_forward_batched_kernel_route(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (!Q || !K || !V || !alpha || !beta_raw || !A_log || !dt_bias ||
            !output || !initial_state || !updated_state || !stream ||
            seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
            seq_len != request_count * request_seq_len ||
            n_heads <= 0 || (d_k != 64 && d_k != 128) || d_v <= 0)
        {
            return false;
        }

        const int required_snapshot_stride = n_heads * d_k * d_v;
        const bool complete_verifier_capture =
            state_snapshots != nullptr &&
            snapshot_stride_floats >= required_snapshot_stride &&
            max_snapshot_rows >= seq_len;

        /*
         * Small request matrices use the exact scalar recurrence because launch
         * economy is better than the row-split prefill route. More importantly,
         * every verifier transaction that supplied a complete post-row state
         * matrix must use this path regardless of M: publication can select any
         * row, so each snapshot and output must have scalar decode's arithmetic
         * schedule. Ordinary long prefill has no complete capture matrix and
         * retains the throughput-oriented row-split implementation below.
         */
        if (request_seq_len <= 8 || complete_verifier_capture)
        {
            if (llaminar2::debugEnv().runtime_debug.cuda_gdn_pointer_trace)
            {
                fprintf(
                    stderr,
                    "[cudaGDN_recurrent_step_batched] requests=%d heads=%d d_k=%d d_v=%d state=%p output=%p lengths=%p\n",
                    request_count,
                    n_heads,
                    d_k,
                    d_v,
                    static_cast<const void *>(initial_state),
                    static_cast<void *>(output),
                    static_cast<const void *>(device_effective_seq_len));
            }
            return launch_cuda_gdn_recurrent_step(
                "cudaGDN_recurrent_step_batched",
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, initial_state, updated_state,
                request_count, request_seq_len,
                n_heads, d_k, d_v,
                use_qk_l2norm,
                device_effective_seq_len,
                /*effective_row_idx=*/0,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                static_cast<cudaStream_t>(stream));
        }

        // Eight threads retain one deterministic key-row partition per output
        // column. A 64-thread block owns eight columns, giving the Qwen
        // D_V=128 geometry 512 independent blocks while retaining exactly the
        // same within-column arithmetic tree. Profiling selects ten resident
        // blocks as the best spill-free register/occupancy point on SM86.
        constexpr int col_threads = 64;
        constexpr int cols_per_block =
            col_threads / kGdnRecurrentRowSplit;
        const int num_col_blocks =
            (d_v + cols_per_block - 1) / cols_per_block;
        // Q/K use two shared stages so row t+1 transfers overlap row t's exact
        // recurrence. No reduction workspace or extra graph binding is needed.
        const int smem_size = 4 * d_k * sizeof(float);

        const int preprocess_blocks = seq_len * n_heads;
        cuda_gdn_prefill_preprocess_kernel<<<preprocess_blocks, 64, 0, (cudaStream_t)stream>>>(
            const_cast<float *>(Q), const_cast<float *>(K),
            const_cast<float *>(alpha), const_cast<float *>(beta_raw),
            A_log, dt_bias,
            device_effective_seq_len,
            request_count, request_seq_len,
            n_heads, d_k, use_qk_l2norm);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_prefill_preprocess] %s\n", cudaGetErrorString(err));
            return false;
        }

        const dim3 grid(n_heads, num_col_blocks, request_count);
        switch (d_k)
        {
        case 64:
            cuda_gdn_chunk_forward_kernel<64>
                <<<grid, col_threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
                    Q, K, V, alpha, beta_raw,
                    output, initial_state, updated_state,
                    device_effective_seq_len,
                    state_snapshots,
                    snapshot_stride_floats,
                    max_snapshot_rows,
                    request_count, request_seq_len,
                    n_heads, d_v);
            break;
        case 128:
            cuda_gdn_chunk_forward_kernel<128>
                <<<grid, col_threads, smem_size, static_cast<cudaStream_t>(stream)>>>(
                    Q, K, V, alpha, beta_raw,
                    output, initial_state, updated_state,
                    device_effective_seq_len,
                    state_snapshots,
                    snapshot_stride_floats,
                    max_snapshot_rows,
                    request_count, request_seq_len,
                    n_heads, d_v);
            break;
        default:
            fprintf(
                stderr,
                "[cudaGDN_chunk_forward] unsupported d_k=%d; expected 64 or 128\n",
                d_k);
            return false;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_chunk_forward] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_chunk_forward_kernel_route(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        return cudaGDN_chunk_forward_batched_kernel_route(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            seq_len, /*request_count=*/1, /*request_seq_len=*/seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            device_effective_seq_len,
            device_idx, stream);
    }

    bool cudaGDN_chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        /*
         * Small MTP verifier chunks must use the grouped recurrence kernel as
         * a first-class path.  It advances timesteps in the same mathematical
         * order as serial decode, but does so inside one graph-capturable
         * launch and publishes each post-row state snapshot directly.
         */
        return cudaGDN_chunk_forward_kernel_route(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, initial_state, updated_state,
            seq_len, n_heads, d_k, d_v, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            nullptr,
            device_idx, stream);
    }

    bool cudaGDN_chunk_forward_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        const bool ok = cudaGDN_chunk_forward_kernel_route(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            seq_len, n_heads, d_k, d_v, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            device_effective_seq_len,
            device_idx, stream);
        return ok;
    }

    /**
     * @brief Launch one request-grouped recurrence over a padded row matrix.
     *
     * The grid's Z dimension owns requests while X/Y retain the production
     * head/column tiling. Each block reads its request's real length directly
     * from device memory, so no host loop or scalar replay sits between graph
     * capture and the recurrent state bank.
     */
    bool cudaGDN_chunk_forward_batched_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_lens,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        if (!device_effective_seq_lens)
            return false;
        return cudaGDN_chunk_forward_batched_kernel_route(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            seq_len, request_count, request_seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            device_effective_seq_lens,
            device_idx, stream);
    }

    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output,
        const float *initial_conv_state,
        float *updated_conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        const int required_snapshot_stride = channels * (kernel_size - 1);
        const bool complete_verifier_capture =
            state_snapshots != nullptr &&
            snapshot_stride_floats >= required_snapshot_stride &&
            max_snapshot_rows >= seq_len;

        /*
         * A complete capture matrix identifies verifier execution. Route it
         * through the one-channel-owner kernel even at M=1 so every post-row
         * state is materialized for device-side publication. Ordinary decode
         * and prefill retain their dedicated throughput paths.
         */
        if (complete_verifier_capture)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                nullptr,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
        }
        else if (seq_len == 1)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_decode_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                channels, kernel_size, apply_silu);
        }
        else if (seq_len <= 4)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                nullptr,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
        }
        else
        {
            int total = seq_len * channels;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cuda_short_conv1d_prefill_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, initial_conv_state,
                nullptr,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
            int state_blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_state_update_kernel<<<state_blocks, threads, 0, (cudaStream_t)stream>>>(
                input, initial_conv_state, updated_conv_state, nullptr,
                seq_len, channels, kernel_size);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_short_conv1d] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output,
        const float *initial_conv_state,
        float *updated_conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        const int required_snapshot_stride = channels * (kernel_size - 1);
        const bool complete_verifier_capture =
            state_snapshots != nullptr &&
            snapshot_stride_floats >= required_snapshot_stride &&
            max_snapshot_rows >= seq_len;

        if (complete_verifier_capture)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                device_effective_seq_len,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
        }
        else if (seq_len == 1)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_decode_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                channels, kernel_size, apply_silu);
        }
        else if (seq_len <= 4)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output,
                initial_conv_state, updated_conv_state,
                device_effective_seq_len,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
        }
        else
        {
            int total = seq_len * channels;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cuda_short_conv1d_prefill_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, initial_conv_state,
                device_effective_seq_len,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
            int state_blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_state_update_kernel<<<state_blocks, threads, 0, (cudaStream_t)stream>>>(
                input, initial_conv_state, updated_conv_state,
                device_effective_seq_len,
                seq_len, channels, kernel_size);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_short_conv1d_effective] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Launch native grouped request-batched short convolution.
     *
     * The request dimension is encoded in the launch geometry, so this wrapper
     * performs no host request loop. Small verifier/decode groups require one
     * launch; longer prefill uses one output launch plus one race-free grouped
     * state commit, matching the scalar long-prefill algorithm.
     */
    bool cudaGDN_short_conv1d_batched(
        const float *input, const float *weight, const float *bias,
        float *output,
        const float *initial_request_states,
        float *updated_request_states,
        int request_count, int request_seq_len,
        int channels, int kernel_size,
        bool apply_silu,
        const int *device_request_seq_lens,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);
        if (!input || !weight || !output ||
            !initial_request_states || !updated_request_states ||
            !device_request_seq_lens || !stream ||
            request_count <= 0 || request_seq_len <= 0 ||
            channels <= 0 || kernel_size <= 1)
        {
            return false;
        }

        constexpr int threads = 256;
        const int required_snapshot_stride = channels * (kernel_size - 1);
        const bool complete_verifier_capture =
            state_snapshots != nullptr &&
            snapshot_stride_floats >= required_snapshot_stride &&
            max_snapshot_rows >= request_count * request_seq_len;
        if (request_seq_len <= 4 || complete_verifier_capture)
        {
            const int channel_blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_batched_small_m_kernel<<<
                dim3(channel_blocks, request_count),
                threads,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                input, weight, bias, output,
                initial_request_states, updated_request_states,
                device_request_seq_lens,
                state_snapshots, snapshot_stride_floats, max_snapshot_rows,
                request_count, request_seq_len,
                channels, kernel_size, apply_silu);
        }
        else
        {
            const int output_elements =
                request_count * request_seq_len * channels;
            const int output_blocks =
                (output_elements + threads - 1) / threads;
            cuda_short_conv1d_batched_prefill_kernel<<<
                output_blocks,
                threads,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                input, weight, bias, output, initial_request_states,
                device_request_seq_lens,
                state_snapshots, snapshot_stride_floats, max_snapshot_rows,
                request_count, request_seq_len,
                channels, kernel_size, apply_silu);

            const int state_elements = request_count * channels;
            const int state_blocks =
                (state_elements + threads - 1) / threads;
            cuda_short_conv1d_batched_state_update_kernel<<<
                state_blocks,
                threads,
                0,
                static_cast<cudaStream_t>(stream)>>>(
                input, initial_request_states, updated_request_states,
                device_request_seq_lens,
                request_count, request_seq_len, channels, kernel_size);
        }

        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr,
                    "[cudaGDN_short_conv1d_batched] %s\n",
                    cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    bool cudaGDN_gated_rmsnorm(
        const float *input, const float *gate, const float *gamma,
        float *output,
        int seq_len, int d_model, int norm_dim, int gamma_period,
        float eps, bool subtract_one, bool gate_silu,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        int n_groups = d_model / norm_dim;
        int total_work = seq_len * n_groups;
        int threads = (norm_dim <= 128) ? 128 : 256;
        int blocks = total_work;

        cuda_gated_rmsnorm_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            input, gate, gamma, output,
            total_work, n_groups, norm_dim, d_model, gamma_period,
            eps, subtract_one, gate_silu);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_gated_rmsnorm] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_attention_output_gate(
        const float *input, const float *gate, float *output,
        int size,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        cuda_attention_output_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            input, gate, output, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_attention_output_gate] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C" — end of first block (recurrence, chunk, rmsnorm, gate wrappers)

// =========================================================================
// Q+Gate Split kernel
// Deinterleaves per-head [Q, Gate] from FA Q projection output.
// Input layout: [seq_len, n_heads * head_dim * 2] with per-head
//   [q0_hd, g0_hd, q1_hd, g1_hd, ...]
// Output: separate Q[seq_len, n_heads*head_dim] and Gate[seq_len, n_heads*head_dim]
// =========================================================================
namespace
{
    __global__ void cuda_q_gate_split_kernel(
        const float *__restrict__ input,
        float *__restrict__ output_q,
        float *__restrict__ output_gate,
        int total_elements, int n_heads, int head_dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        int q_dim = n_heads * head_dim;
        int t = idx / q_dim;
        int offset = idx % q_dim;
        int h = offset / head_dim;
        int d = offset % head_dim;

        int input_dim = n_heads * head_dim * 2;
        int src_idx = t * input_dim + h * (head_dim * 2) + d;
        int gate_src_idx = t * input_dim + h * (head_dim * 2) + head_dim + d;

        output_q[idx] = input[src_idx];
        output_gate[idx] = input[gate_src_idx];
    }
} // anonymous namespace (cuda_q_gate_split_kernel)

extern "C"
{

    bool cudaGDN_q_gate_split(
        const float *input, float *output_q, float *output_gate,
        int seq_len, int n_heads, int head_dim,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        int total = seq_len * n_heads * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        cuda_q_gate_split_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            input, output_q, output_gate, total, n_heads, head_dim);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_q_gate_split] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
