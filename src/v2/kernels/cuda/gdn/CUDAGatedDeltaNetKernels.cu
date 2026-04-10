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
        bool use_qk_l2norm)
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

    __global__ void cuda_gdn_chunk_forward_kernel(
        const float *__restrict__ Q,        // [seq_len, n_heads * d_k]
        const float *__restrict__ K,        // [seq_len, n_heads * d_k]
        const float *__restrict__ V,        // [seq_len, n_heads * d_v]
        const float *__restrict__ alpha,    // [seq_len, n_heads]
        const float *__restrict__ beta_raw, // [seq_len, n_heads]
        const float *__restrict__ A_log,    // [n_heads]
        const float *__restrict__ dt_bias,  // [n_heads]
        float *__restrict__ output,         // [seq_len, n_heads * d_v]
        float *__restrict__ state,          // [n_heads, d_k, d_v]
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm)
    {
        const int h = blockIdx.x;
        if (h >= n_heads)
            return;

        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        // Row-split parallelism: ROW_SPLIT threads collaborate on each
        // column, splitting the d_k rows between them. With ROW_SPLIT=4 and
        // 256 threads/block: 8 warps/block — optimal for Ampere L1 latency.
        constexpr int ROW_SPLIT = 4;
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

        // Pre-load A_log and dt_bias (constant across timesteps)
        __shared__ float A_log_h;
        __shared__ float dt_bias_h;
        if (tid == 0)
        {
            A_log_h = A_log[h];
            dt_bias_h = dt_bias[h];
        }
        __syncthreads();

        // Process each timestep sequentially (inherent to recurrence)
        for (int t = 0; t < seq_len; t++)
        {
            const float *q_src = Q + t * qk_stride + h * d_k;
            const float *k_src = K + t * qk_stride + h * d_k;
            const float *v_src = V + t * v_stride + h * d_v;
            float *o_dst = output + t * v_stride + h * d_v;

            // Load Q and K into shared memory (all threads cooperate)
            for (int i = tid; i < d_k; i += block_size)
            {
                q_local[i] = q_src[i];
                k_local[i] = k_src[i];
            }
            __syncthreads();

            // L2 normalize Q and K
            if (use_qk_l2norm)
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
            else
            {
                for (int i = tid; i < d_k; i += block_size)
                    q_local[i] *= scale;
                __syncthreads();
            }

            // Gate and beta (thread 0 computes, broadcast via shared mem)
            float decay, beta_h;
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

            // Row-split column processing with register caching:
            // Each thread caches its d_k/ROW_SPLIT rows (32 floats for ROW_SPLIT=4)
            // in an array that nvcc keeps in registers. This eliminates the
            // second read of S per pass (1R+1W vs 2R+1W = 33% memory savings).
            // 32 floats + ~20 other regs ≈ 52 regs/thread — well within budget.
            {
                constexpr int ROWS_PER_SPLIT = 32; // d_k(128) / ROW_SPLIT(4)
                const int j_start = split_id * ROWS_PER_SPLIT;

                // Pass 1: Load S into register cache + partial kv dot product
                float s_cache[ROWS_PER_SPLIT];
                float partial_kv = 0.0f;
                if (vi < d_v)
                {
                    for (int j = 0; j < ROWS_PER_SPLIT; j++)
                    {
                        float s = S[(j_start + j) * d_v + vi] * decay;
                        s_cache[j] = s;
                        partial_kv += s * k_local[j_start + j];
                    }
                }
                reduce_kv[tid] = partial_kv;
                __syncthreads();

                // Reduce kv across all splits and compute delta
                float delta = 0.0f;
                if (vi < d_v)
                {
                    float kv = 0.0f;
                    for (int s = 0; s < ROW_SPLIT; s++)
                        kv += reduce_kv[col_in_block + s * cols_per_block];
                    delta = (v_src[vi] - kv) * beta_h;
                }

                // Pass 2: Update from register cache + partial output (no re-read)
                float partial_out = 0.0f;
                if (vi < d_v)
                {
                    for (int j = 0; j < ROWS_PER_SPLIT; j++)
                    {
                        float s_new = s_cache[j] + k_local[j_start + j] * delta;
                        S[(j_start + j) * d_v + vi] = s_new;
                        partial_out += s_new * q_local[j_start + j];
                    }
                }
                reduce_out[tid] = partial_out;
                __syncthreads();

                // Reduce output and write (only split_id=0 writes)
                if (vi < d_v && split_id == 0)
                {
                    float out_vi = 0.0f;
                    for (int s = 0; s < ROW_SPLIT; s++)
                        out_vi += reduce_out[col_in_block + s * cols_per_block];
                    o_dst[vi] = out_vi;
                }
            }
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
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = seq_len * channels;
        if (idx >= total)
            return;

        int t = idx / channels;
        int ch = idx % channels;

        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++)
        {
            int src_t = t - (kernel_size - 1) + k;
            float val = (src_t >= 0) ? input[src_t * channels + ch] : 0.0f;
            sum += val * weight[ch * kernel_size + k];
        }

        if (bias)
            sum += bias[ch];
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));
        output[t * channels + ch] = sum;

        // Update conv_state with the last (kernel_size-1) inputs for future decode
        int ks_minus1 = kernel_size - 1;
        if (t >= seq_len - ks_minus1 && conv_state)
        {
            int state_idx = t - (seq_len - ks_minus1);
            conv_state[ch * ks_minus1 + state_idx] = input[t * channels + ch];
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
     * Head mapping: k_head_for_v_head_j = (j + global_v_offset) % n_k_heads
     */
    __global__ void cuda_gdn_deinterleave_qkv_kernel(
        const float *__restrict__ merged,
        float *__restrict__ out_q,
        float *__restrict__ out_k,
        float *__restrict__ out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset)
    {
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
            n_heads, d_k, d_v, use_qk_l2norm);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_recurrent_step] %s\n", cudaGetErrorString(err));
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
        int device_idx, void *stream)
    {
        cudaSetDevice(device_idx);

        // Row-split: 256 threads per block, 4 threads per column = 64 cols/block.
        // 8 warps/block — optimal for Ampere L1 latency hiding.
        int col_threads = 256;
        int cols_per_block = col_threads / 4; // 64 columns per block
        if (d_v <= 64)
        {
            col_threads = 128;
            cols_per_block = col_threads / 4; // 32 columns per block
        }
        int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
        // smem: q_local[d_k] + k_local[d_k] + warp_sums[16] + reduce_kv[col_threads] + reduce_out[col_threads]
        int smem_size = (2 * d_k + 16 + 2 * col_threads) * sizeof(float);

        dim3 grid(n_heads, num_col_blocks);
        cuda_gdn_chunk_forward_kernel<<<grid, col_threads, smem_size, (cudaStream_t)stream>>>(
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, state,
            seq_len, n_heads, d_k, d_v, use_qk_l2norm);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_chunk_forward] %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
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
        else
        {
            int total = seq_len * channels;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cuda_short_conv1d_prefill_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
                input, weight, bias, output, conv_state,
                seq_len, channels, kernel_size, apply_silu);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[cudaGDN_short_conv1d] %s\n", cudaGetErrorString(err));
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
