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
#include <cstdio>
#include <cstdint>

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

    __global__ void cuda_gdn_recurrent_step_kernel(
        const float *__restrict__ q,        // [n_heads * d_k]
        const float *__restrict__ k,        // [n_heads * d_k]
        const float *__restrict__ v,        // [n_heads * d_v]
        const float *__restrict__ alpha,    // [n_heads]
        const float *__restrict__ beta_raw, // [n_heads]
        const float *__restrict__ A_log,    // [n_heads]
        const float *__restrict__ dt_bias,  // [n_heads]
        float *__restrict__ output,         // [n_heads * d_v]
        float *__restrict__ state,          // [n_heads, d_k, d_v]
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *__restrict__ effective_seq_len_ptr,
        int effective_row_idx)
    {
        const int h = blockIdx.x;
        if (h >= n_heads)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        // Column this thread handles (2D grid: blockIdx.y selects column tile)
        const int vi = blockIdx.y * block_size + tid;

        // Pointers for this head
        const float *q_head = q + h * d_k;
        const float *k_head = k + h * d_k;
        const float *v_head = v + h * d_v;
        float *S = state + (size_t)h * d_k * d_v;
        float *o_head = output + h * d_v;

        if (effective_seq_len_ptr && effective_row_idx >= *effective_seq_len_ptr)
        {
            if (vi < d_v)
                o_head[vi] = 0.0f;
            return;
        }

        // Shared memory for preprocessed Q and K
        extern __shared__ float smem[];
        float *q_local = smem;       // [d_k]
        float *k_local = smem + d_k; // [d_k]

        // ── Step 0: Preprocessing ──
        const float scale = rsqrtf((float)d_k);

        // Load Q and K into shared memory
        for (int i = tid; i < d_k; i += block_size)
        {
            q_local[i] = q_head[i];
            k_local[i] = k_head[i];
        }
        __syncthreads();

        // L2 normalize Q and K if requested
        if (use_qk_l2norm)
        {
            // Compute Q norm
            float q_sum = 0.0f;
            for (int i = tid; i < d_k; i += block_size)
                q_sum += q_local[i] * q_local[i];

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1)
                q_sum += __shfl_xor_sync(0xFFFFFFFF, q_sum, offset);

            // Cross-warp reduction via shared memory
            __shared__ float warp_sums[8]; // max 256 threads = 8 warps
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0)
                warp_sums[warp_id] = q_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                int num_warps = (block_size + 31) / 32;
                for (int w = 0; w < num_warps; w++)
                    total += warp_sums[w];
                warp_sums[0] = total;
            }
            __syncthreads();
            float q_norm_sq = warp_sums[0];
            float q_inv = scale / fmaxf(sqrtf(q_norm_sq), 1e-6f);

            // Apply Q scale
            for (int i = tid; i < d_k; i += block_size)
                q_local[i] *= q_inv;
            __syncthreads();

            // Compute K norm
            float k_sum = 0.0f;
            for (int i = tid; i < d_k; i += block_size)
                k_sum += k_local[i] * k_local[i];
            for (int offset = 16; offset > 0; offset >>= 1)
                k_sum += __shfl_xor_sync(0xFFFFFFFF, k_sum, offset);
            if (lane_id == 0)
                warp_sums[warp_id] = k_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                int num_warps = (block_size + 31) / 32;
                for (int w = 0; w < num_warps; w++)
                    total += warp_sums[w];
                warp_sums[0] = total;
            }
            __syncthreads();
            float k_norm_sq = warp_sums[0];
            float k_inv = 1.0f / fmaxf(sqrtf(k_norm_sq), 1e-6f);

            for (int i = tid; i < d_k; i += block_size)
                k_local[i] *= k_inv;
            __syncthreads();
        }
        else
        {
            // Just scale Q
            for (int i = tid; i < d_k; i += block_size)
                q_local[i] *= scale;
            __syncthreads();
        }

        // Compute gate and beta (single thread, broadcast via shared mem)
        __shared__ float decay_shared;
        __shared__ float beta_shared;
        if (tid == 0)
        {
            float x = alpha[h] + dt_bias[h];
            float sp = (x > 20.0f) ? x : log1pf(expf(x));
            decay_shared = expf(A_log[h] * sp);
            beta_shared = 1.0f / (1.0f + expf(-beta_raw[h]));
        }
        __syncthreads();
        float decay = decay_shared;
        float beta_h = beta_shared;

        // Fused per-column processing — NO separate decay pass
        if (vi < d_v)
        {
            // Pass 1: Fused decay + kv dot product (read-only, no write-back)
            float kv = 0.0f;
            for (int j = 0; j < d_k; j++)
            {
                float s_decayed = S[j * d_v + vi] * decay;
                kv += s_decayed * k_local[j];
            }

            float delta = (v_head[vi] - kv) * beta_h;

            // Pass 2: Fused decay + delta update + output (read-modify-write)
            float out_vi = 0.0f;
            for (int j = 0; j < d_k; j++)
            {
                float s_new = S[j * d_v + vi] * decay + k_local[j] * delta;
                S[j * d_v + vi] = s_new;
                out_vi += s_new * q_local[j];
            }
            o_head[vi] = out_vi;
        }
    }

    // =========================================================================
    // Prefill Recurrence Kernel (seq_len>1, multi-block per head)
    //
    // Optimizations vs V1:
    //   1. 2D grid: dim3(n_heads, col_blocks) — each column block processes
    //      a subset of the d_v columns. Columns are fully independent in the
    //      delta-rule recurrence, so no cross-block sync is needed.
    //   2. Fused 2-pass column processing (same as decode kernel):
    //      Pass 1: decay + kv (read-only); Pass 2: decay + update + output
    //      (read-modify-write). Eliminates the separate decay pass.
    //   3. Each thread owns one column vi — all per-column operations are
    //      independent, so no __syncthreads() in the column processing body.
    // =========================================================================

    __global__ __launch_bounds__(256, 2) void cuda_gdn_chunk_forward_kernel(
        const float *__restrict__ Q,        // [seq_len, n_heads * d_k]
        const float *__restrict__ K,        // [seq_len, n_heads * d_k]
        const float *__restrict__ V,        // [seq_len, n_heads * d_v]
        const float *__restrict__ alpha,    // [seq_len, n_heads]
        const float *__restrict__ beta_raw, // [seq_len, n_heads]
        const float *__restrict__ A_log,    // [n_heads]
        const float *__restrict__ dt_bias,  // [n_heads]
        float *__restrict__ output,         // [seq_len, n_heads * d_v]
        float *__restrict__ state,          // [n_heads, d_k, d_v]
        const int *__restrict__ effective_seq_len_ptr,
        float *__restrict__ state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        bool inputs_preprocessed)
    {
        const int h = blockIdx.x;
        if (h >= n_heads)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        // Row-split parallelism: ROW_SPLIT threads collaborate on each
        // column, splitting the d_k rows between them. With ROW_SPLIT=4 and
        // 256 threads/block: 8 warps/block — optimal for Ampere L1 latency.
        constexpr int ROW_SPLIT = 8;
        constexpr int ROWS_PER_SPLIT = 16;
        const int cols_per_block = block_size / ROW_SPLIT;
        const int split_id = tid / cols_per_block; // 0..ROW_SPLIT-1
        const int col_in_block = tid % cols_per_block;
        const int vi = blockIdx.y * cols_per_block + col_in_block;

        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;
        const float scale = rsqrtf((float)d_k);

        float *S = state + (size_t)h * d_k * d_v;

        extern __shared__ float smem[];
        float *q_local = smem;       // [d_k]
        float *k_local = smem + d_k; // [d_k]
        // Shared reduction scratch follows
        float *warp_sums = smem + 2 * d_k; // [16] for warp reduction (up to 16 warps)
        // Double-buffered reduction arrays for row-split partial sums
        float *reduce_kv = smem + 2 * d_k + 16;               // [block_size]
        float *reduce_out = smem + 2 * d_k + 16 + block_size; // [block_size]

        // Pre-load A_log and dt_bias only for the legacy in-kernel preprocessing
        // path. Captured prefill uses the separate preprocess kernel and skips
        // this uniform barrier on the hot recurrence path.
        __shared__ float A_log_h;
        __shared__ float dt_bias_h;
        if (!inputs_preprocessed)
        {
            if (tid == 0)
            {
                A_log_h = A_log[h];
                dt_bias_h = dt_bias[h];
            }
            __syncthreads();
        }

        int effective_seq_len = seq_len;
        if (effective_seq_len_ptr)
        {
            const int raw_effective = *effective_seq_len_ptr;
            effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
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
                sc[j] = S[(j_start + j) * d_v + vi];
        }
        else
        {
#pragma unroll
            for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                sc[j] = 0.0f;
        }

        // Process each timestep sequentially (inherent to recurrence)
        for (int t = 0; t < seq_len; t++)
        {
            const float *q_src = Q + t * qk_stride + h * d_k;
            const float *k_src = K + t * qk_stride + h * d_k;
            const float *v_src = V + t * v_stride + h * d_v;
            float *o_dst = output + t * v_stride + h * d_v;

            if (t >= effective_seq_len)
            {
                if (vi < d_v && split_id == 0)
                    o_dst[vi] = 0.0f;
                continue;
            }

            // Load Q and K into shared memory (all threads cooperate)
            for (int i = tid; i < d_k; i += block_size)
            {
                q_local[i] = q_src[i];
                k_local[i] = k_src[i];
            }
            __syncthreads();

            // Q/K and gate preprocessing can be shared by all column tiles. The
            // CUDA wrapper runs that separate graph-capturable kernel for
            // prefill so this recurrence kernel only handles the sequential
            // delta-rule state update.
            if (!inputs_preprocessed && use_qk_l2norm)
            {
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int num_warps = (block_size + 31) / 32;

                // Q norm
                float q_sum = 0.0f;
                for (int i = tid; i < d_k; i += block_size)
                    q_sum += q_local[i] * q_local[i];
                for (int offset = 16; offset > 0; offset >>= 1)
                    q_sum += __shfl_xor_sync(0xFFFFFFFF, q_sum, offset);
                if (lane_id == 0)
                    warp_sums[warp_id] = q_sum;
                __syncthreads();
                if (tid == 0)
                {
                    float total = 0.0f;
                    for (int w = 0; w < num_warps; w++)
                        total += warp_sums[w];
                    warp_sums[0] = total;
                }
                __syncthreads();
                float q_inv = scale / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
                for (int i = tid; i < d_k; i += block_size)
                    q_local[i] *= q_inv;
                __syncthreads();

                // K norm
                float k_sum = 0.0f;
                for (int i = tid; i < d_k; i += block_size)
                    k_sum += k_local[i] * k_local[i];
                for (int offset = 16; offset > 0; offset >>= 1)
                    k_sum += __shfl_xor_sync(0xFFFFFFFF, k_sum, offset);
                if (lane_id == 0)
                    warp_sums[warp_id] = k_sum;
                __syncthreads();
                if (tid == 0)
                {
                    float total = 0.0f;
                    for (int w = 0; w < num_warps; w++)
                        total += warp_sums[w];
                    warp_sums[0] = total;
                }
                __syncthreads();
                float k_inv = 1.0f / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
                for (int i = tid; i < d_k; i += block_size)
                    k_local[i] *= k_inv;
                __syncthreads();
            }
            else if (!inputs_preprocessed)
            {
                for (int i = tid; i < d_k; i += block_size)
                    q_local[i] *= scale;
                __syncthreads();
            }

            // Gate and beta (thread 0 computes, broadcast via shared mem)
            float decay, beta_h;
            if (inputs_preprocessed)
            {
                const int gate_idx = t * n_heads + h;
                decay = alpha[gate_idx];
                beta_h = beta_raw[gate_idx];
            }
            else
            {
                if (tid == 0)
                {
                    float x = alpha[t * n_heads + h] + dt_bias_h;
                    float sp = (x > 20.0f) ? x : log1pf(expf(x));
                    warp_sums[0] = expf(A_log_h * sp);
                    warp_sums[1] = 1.0f / (1.0f + expf(-beta_raw[t * n_heads + h]));
                }
                __syncthreads();
                decay = warp_sums[0];
                beta_h = warp_sums[1];
            }

            float partial_kv = 0.0f;
            if (vi < d_v)
            {
#pragma unroll
                for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                {
                    sc[j] *= decay;
                    partial_kv += sc[j] * k_local[j_start + j];
                }
            }
            reduce_kv[tid] = partial_kv;
            __syncthreads();

            float delta = 0.0f;
            if (vi < d_v)
            {
                float kv = 0.0f;
                for (int s = 0; s < ROW_SPLIT; s++)
                    kv += reduce_kv[col_in_block + s * cols_per_block];
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
            reduce_out[tid] = partial_out;
            __syncthreads();

            if (vi < d_v && split_id == 0)
            {
                float out_vi = 0.0f;
                for (int s = 0; s < ROW_SPLIT; s++)
                    out_vi += reduce_out[col_in_block + s * cols_per_block];
                o_dst[vi] = out_vi;
            }

            if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows && vi < d_v)
            {
                float *snapshot =
                    state_snapshots +
                    static_cast<size_t>(t) * static_cast<size_t>(snapshot_stride_floats) +
                    static_cast<size_t>(h) * static_cast<size_t>(d_k) * static_cast<size_t>(d_v);
#pragma unroll
                for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                    snapshot[(j_start + j) * d_v + vi] = sc[j];
            }
            __syncthreads();
        }

        if (vi < d_v)
        {
#pragma unroll
            for (int j = 0; j < ROWS_PER_SPLIT; ++j)
                S[(j_start + j) * d_v + vi] = sc[j];
        }
    }

    __global__ void cuda_gdn_prefill_preprocess_kernel(
        float *__restrict__ Q,
        float *__restrict__ K,
        float *__restrict__ alpha,
        float *__restrict__ beta_raw,
        const float *__restrict__ A_log,
        const float *__restrict__ dt_bias,
        int seq_len, int n_heads, int d_k,
        bool use_qk_l2norm)
    {
        const int h = blockIdx.x;
        const int t = blockIdx.y;
        if (h >= n_heads || t >= seq_len)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;
        const int qk_stride = n_heads * d_k;
        const float scale = rsqrtf(static_cast<float>(d_k));

        float *q_head = Q + t * qk_stride + h * d_k;
        float *k_head = K + t * qk_stride + h * d_k;

        __shared__ float warp_sums[8];
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int num_warps = (block_size + 31) / 32;

        if (use_qk_l2norm)
        {
            float q_sum = 0.0f;
            for (int i = tid; i < d_k; i += block_size)
            {
                const float qv = q_head[i];
                q_sum += qv * qv;
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                q_sum += __shfl_xor_sync(0xFFFFFFFF, q_sum, offset);
            if (lane_id == 0)
                warp_sums[warp_id] = q_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                for (int w = 0; w < num_warps; ++w)
                    total += warp_sums[w];
                warp_sums[0] = total;
            }
            __syncthreads();
            const float q_inv = scale / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
            for (int i = tid; i < d_k; i += block_size)
                q_head[i] *= q_inv;
            __syncthreads();

            float k_sum = 0.0f;
            for (int i = tid; i < d_k; i += block_size)
            {
                const float kv = k_head[i];
                k_sum += kv * kv;
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                k_sum += __shfl_xor_sync(0xFFFFFFFF, k_sum, offset);
            if (lane_id == 0)
                warp_sums[warp_id] = k_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                for (int w = 0; w < num_warps; ++w)
                    total += warp_sums[w];
                warp_sums[0] = total;
            }
            __syncthreads();
            const float k_inv = 1.0f / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
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
            const int gate_idx = t * n_heads + h;
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
        float *__restrict__ conv_state,   // [channels, kernel_size-1]
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
            sum += conv_state[ch * ks_minus1 + k] * weight[ch * kernel_size + k];
        sum += input[ch] * weight[ch * kernel_size + ks_minus1];

        if (bias)
            sum += bias[ch];

        // Apply SiLU activation
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));

        output[ch] = sum;

        // Now shift conv_state left by 1, insert new input at the end
        for (int k = 0; k < ks_minus1 - 1; k++)
            conv_state[ch * ks_minus1 + k] = conv_state[ch * ks_minus1 + k + 1];
        conv_state[ch * ks_minus1 + ks_minus1 - 1] = input[ch];
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
        float *__restrict__ conv_state,   // [channels, kernel_size-1] (updated at end)
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
                else if (conv_state)
                    val = conv_state[ch * ks_minus1 + ks_minus1 + src_t];
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
                                 : (conv_state ? conv_state[ch * ks_minus1 + ks_minus1 + src_t] : 0.0f);
            }
        }
    }

    /**
     * @brief Decode-equivalent grouped short-conv kernel for MTP verifier rows.
     *
     * MTP verifier groups are tiny (M=2..4).  The long-prefill kernel uses a
     * second launch to update live state safely; doing that for verifier rows
     * burns most of the M=2 win.  This kernel keeps one lane responsible for a
     * channel, walks the small row group in causal order, writes every row
     * snapshot, and only then publishes the channel's final live state.
     */
    __global__ void cuda_short_conv1d_small_m_kernel(
        const float *__restrict__ input,
        const float *__restrict__ weight,
        const float *__restrict__ bias,
        float *__restrict__ output,
        float *__restrict__ conv_state,
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
        const float *initial_state =
            conv_state
                ? conv_state + static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1)
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
                            : (initial_state
                                   ? initial_state[ks_minus1 + src_t]
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
                            : (initial_state
                                   ? initial_state[ks_minus1 + src_t]
                                   : 0.0f);
                }
            }
        }

        if (conv_state)
        {
            float *state =
                conv_state + static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = effective_seq_len - ks_minus1 + state_idx;
                state[state_idx] =
                    (src_t >= 0 && src_t < effective_seq_len)
                        ? input[static_cast<size_t>(src_t) * channels + ch]
                        : initial_state[ks_minus1 + src_t];
            }
        }
    }

    __global__ void cuda_short_conv1d_state_update_kernel(
        const float *__restrict__ input,
        float *__restrict__ conv_state,
        const int *__restrict__ effective_seq_len_ptr,
        int seq_len, int channels, int kernel_size)
    {
        int ch = blockIdx.x * blockDim.x + threadIdx.x;
        if (ch >= channels || !conv_state)
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

        float *state = conv_state + ch * ks_minus1;
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = effective_seq_len - ks_minus1 + state_idx;
            state[state_idx] =
                (src_t >= 0 && src_t < effective_seq_len) ? input[src_t * channels + ch]
                                                           : state[ks_minus1 + src_t];
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
            const int row = row_indices[request * row_index_stride];
            if (row < 0 || row >= rows)
                continue;
            dst[linear] =
                capture[static_cast<size_t>(row) * static_cast<size_t>(state_size) +
                        static_cast<size_t>(state_offset)];
        }
    }

} // anonymous namespace (deinterleave kernel)

// =========================================================================
// GPU Memory Management Helpers (called from headers via extern "C")
// =========================================================================

extern "C"
{

    bool cudaGDN_gpu_malloc(float **ptr, size_t count)
    {
        cudaError_t err = cudaMalloc(ptr, count * sizeof(float));
        return err == cudaSuccess;
    }

    void cudaGDN_gpu_free(float *ptr)
    {
        if (ptr)
            cudaFree(ptr);
    }

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
        cudaSetDevice(device_idx);

        int q_dst_dim = n_v_heads * d_k;
        int k_dst_dim = n_v_heads * d_k;
        int v_dim = n_v_heads * d_v;
        int total = seq_len * (q_dst_dim + k_dst_dim + v_dim);

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        cuda_gdn_deinterleave_qkv_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            merged, out_q, out_k, out_v,
            seq_len, n_k_heads, n_v_heads, d_k, d_v, global_v_offset);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_deinterleave_qkv] %s\n", cudaGetErrorString(err));
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
        float *output, float *state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        // 2D grid: x=heads, y=column tiles (32 threads = 1 warp per block)
        int col_threads = 32;
        if (d_v > 128)
            col_threads = 64;
        int num_col_blocks = (d_v + col_threads - 1) / col_threads;
        int smem_size = 2 * d_k * sizeof(float) + 8 * sizeof(float); // q_local + k_local + warp_sums

        dim3 grid(n_heads, num_col_blocks);
        cuda_gdn_recurrent_step_kernel<<<grid, col_threads, smem_size, (cudaStream_t)stream>>>(
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, state,
            n_heads, d_k, d_v, use_qk_l2norm,
            nullptr,
            -1);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_recurrent_step] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_recurrent_step_effective_row(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_len,
        int row_idx,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        int col_threads = 32;
        if (d_v > 128)
            col_threads = 64;
        const int num_col_blocks = (d_v + col_threads - 1) / col_threads;
        const int smem_size = 2 * d_k * sizeof(float) + 8 * sizeof(float);

        dim3 grid(n_heads, num_col_blocks);
        cuda_gdn_recurrent_step_kernel<<<grid, col_threads, smem_size, (cudaStream_t)stream>>>(
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, state,
            n_heads, d_k, d_v, use_qk_l2norm,
            device_effective_seq_len,
            row_idx);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_recurrent_step_effective_row] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_chunk_forward_kernel_route(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        const int *device_effective_seq_len,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        // Row-split: 256 threads per block, 8 threads per column = 32 cols/block.
        // More column blocks keep Ampere GPUs occupied during the sequential recurrence.
        int col_threads = 256;
        int cols_per_block = col_threads / 8; // 32 columns per block
        if (d_v <= 64)
        {
            col_threads = 128;
            cols_per_block = col_threads / 8; // 16 columns per block
        }
        int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
        // smem: q_local[d_k] + k_local[d_k] + warp_sums[16] + reduce_kv[col_threads] + reduce_out[col_threads]
        int smem_size = (2 * d_k + 16 + 2 * col_threads) * sizeof(float);

        cuda_gdn_prefill_preprocess_kernel<<<dim3(n_heads, seq_len), 64, 0, (cudaStream_t)stream>>>(
            const_cast<float *>(Q), const_cast<float *>(K),
            const_cast<float *>(alpha), const_cast<float *>(beta_raw),
            A_log, dt_bias,
            seq_len, n_heads, d_k, use_qk_l2norm);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_prefill_preprocess] %s\n", cudaGetErrorString(err));
            return false;
        }

        dim3 grid(n_heads, num_col_blocks);
        cuda_gdn_chunk_forward_kernel<<<grid, col_threads, smem_size, (cudaStream_t)stream>>>(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, n_heads, d_k, d_v, use_qk_l2norm,
            true);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_chunk_forward] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
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
            output, state,
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
        float *output, float *state,
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
            output, state,
            seq_len, n_heads, d_k, d_v, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            device_effective_seq_len,
            device_idx, stream);
        return ok;
    }

    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        if (seq_len == 1)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_decode_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, conv_state,
                channels, kernel_size, apply_silu);
        }
        else if (seq_len <= 4)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, conv_state,
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
                input, weight, bias, output, conv_state,
                nullptr,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
            int state_blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_state_update_kernel<<<state_blocks, threads, 0, (cudaStream_t)stream>>>(
                input, conv_state, nullptr,
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
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        if (seq_len == 1)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_decode_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, conv_state,
                channels, kernel_size, apply_silu);
        }
        else if (seq_len <= 4)
        {
            int threads = 256;
            int blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_small_m_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, conv_state,
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
                input, weight, bias, output, conv_state,
                device_effective_seq_len,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                seq_len, channels, kernel_size, apply_silu);
            int state_blocks = (channels + threads - 1) / threads;
            cuda_short_conv1d_state_update_kernel<<<state_blocks, threads, 0, (cudaStream_t)stream>>>(
                input, conv_state, device_effective_seq_len,
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
