/**
 * @file ROCmGatedDeltaNetKernels.hip
 * @brief HIP kernels for Gated Delta Net (GDN) recurrence and short convolution
 *
 * ROCm/HIP port of the CUDA GDN kernels. Implements:
 * - Delta rule recurrence (decode + prefill)
 * - Short causal conv1d + SiLU
 * - Gated RMSNorm
 * - Attention output gate (sigmoid)
 *
 * Key HIP differences from CUDA:
 * - 64-thread wavefronts (vs 32 warps), shuffle width adjusted
 * - hip_runtime.h / hip_fp16.h headers
 * - __expf → expf (HIP math)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace { // anonymous namespace to avoid symbol conflicts with CUDA kernels

// =========================================================================
// Decode Recurrence Kernel (seq_len=1, multi-block per head)
//
// Optimizations vs V1:
//   1. Fused decay into per-column loops — eliminates separate decay pass
//      and one full traversal of the d_k×d_v state matrix
//   2. Two-pass column processing: Pass 1 reads state (decay+kv), Pass 2
//      reads+writes (decay+update+output). Saves 50% memory traffic vs
//      original 3-pass + decay.
//   3. 2D grid: (n_heads, col_blocks) for better CU utilization on gfx906
//      (was: 16 blocks / 60 CUs = 27%, now: 32+ blocks = 53%+)
//   4. Single-wavefront blocks (64 threads) — L2 norm reduction is
//      intra-wavefront shuffle only, no cross-warp shared memory needed
// =========================================================================

__global__ void rocm_gdn_recurrent_step_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const float *__restrict__ alpha,
    const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    float *__restrict__ output,
    float *__restrict__ state,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Column this thread handles (2D grid: blockIdx.y selects column tile)
    const int vi = blockIdx.y * block_size + tid;

    const float *q_head = q + h * d_k;
    const float *k_head = k + h * d_k;
    const float *v_head = v + h * d_v;
    float *S = state + (size_t)h * d_k * d_v;
    float *o_head = output + h * d_v;

    extern __shared__ float smem[];
    float *q_local = smem;
    float *k_local = smem + d_k;

    const float scale = rsqrtf((float)d_k);

    // Load Q and K into shared memory (all threads cooperate)
    for (int i = tid; i < d_k; i += block_size) {
        q_local[i] = q_head[i];
        k_local[i] = k_head[i];
    }
    __syncthreads();

    if (use_qk_l2norm) {
        // Single-wavefront reduction — no cross-warp shared memory needed
        // Q norm
        float q_sum = 0.0f;
        for (int i = tid; i < d_k; i += block_size)
            q_sum += q_local[i] * q_local[i];
        for (int offset = 32; offset > 0; offset >>= 1)
            q_sum += __shfl_xor(q_sum, offset);
        // All lanes in wavefront now have the full sum
        float q_inv = scale / fmaxf(sqrtf(q_sum), 1e-6f);
        for (int i = tid; i < d_k; i += block_size)
            q_local[i] *= q_inv;
        __syncthreads();

        // K norm
        float k_sum = 0.0f;
        for (int i = tid; i < d_k; i += block_size)
            k_sum += k_local[i] * k_local[i];
        for (int offset = 32; offset > 0; offset >>= 1)
            k_sum += __shfl_xor(k_sum, offset);
        float k_inv = 1.0f / fmaxf(sqrtf(k_sum), 1e-6f);
        for (int i = tid; i < d_k; i += block_size)
            k_local[i] *= k_inv;
        __syncthreads();
    } else {
        for (int i = tid; i < d_k; i += block_size)
            q_local[i] *= scale;
        __syncthreads();
    }

    // Gate and beta — compute on lane 0, broadcast via shuffle
    float decay, beta_h;
    if (tid == 0) {
        float x = alpha[h] + dt_bias[h];
        float sp = (x > 20.0f) ? x : log1pf(expf(x));
        decay = expf(A_log[h] * sp);
        beta_h = 1.0f / (1.0f + expf(-beta_raw[h]));
    }
    decay = __shfl(decay, 0);
    beta_h = __shfl(beta_h, 0);

    // Fused per-column processing — NO separate decay pass
    // Each thread owns one column vi and processes it end-to-end.
    // Columns are independent so no sync needed between them.
    if (vi < d_v) {
        // Pass 1: Fused decay + kv dot product (read-only, no write-back)
        // Saves writing the decayed state back to global memory
        float kv = 0.0f;
        for (int j = 0; j < d_k; j++) {
            float s_decayed = S[j * d_v + vi] * decay;
            kv += s_decayed * k_local[j];
        }

        float delta = (v_head[vi] - kv) * beta_h;

        // Pass 2: Fused decay + delta update + output (read-modify-write)
        // Re-applies decay to avoid intermediate write-back
        float out_vi = 0.0f;
        for (int j = 0; j < d_k; j++) {
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

__global__ void rocm_gdn_chunk_forward_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ alpha,
    const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    float *__restrict__ output,
    float *__restrict__ state,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Row-split parallelism: ROW_SPLIT threads collaborate on each
    // column, splitting the d_k rows between them. With ROW_SPLIT=8 and
    // 512 threads/block: 8 wavefronts/block — optimal tradeoff between
    // latency hiding and reduction overhead on gfx906.
    constexpr int ROW_SPLIT = 8;
    const int cols_per_block = block_size / ROW_SPLIT;
    const int split_id = tid / cols_per_block;   // 0..ROW_SPLIT-1
    const int col_in_block = tid % cols_per_block;
    const int vi = blockIdx.y * cols_per_block + col_in_block;

    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;
    const float scale = rsqrtf((float)d_k);

    float *S = state + (size_t)h * d_k * d_v;

    extern __shared__ float smem[];
    float *q_local = smem;
    float *k_local = smem + d_k;
    float *warp_sums = smem + 2 * d_k; // [16] for warp reduction
    // Double-buffered reduction arrays for row-split partial sums
    float *reduce_kv = smem + 2 * d_k + 16;            // [block_size]
    float *reduce_out = smem + 2 * d_k + 16 + block_size; // [block_size]

    int warp_id = tid / 64;
    int lane_id = tid % 64;
    int num_warps = (block_size + 63) / 64;

    // Pre-load A_log and dt_bias (constant across timesteps)
    __shared__ float A_log_h;
    __shared__ float dt_bias_h;
    if (tid == 0)
    {
        A_log_h = A_log[h];
        dt_bias_h = dt_bias[h];
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float *q_src = Q + t * qk_stride + h * d_k;
        const float *k_src = K + t * qk_stride + h * d_k;
        const float *v_src = V + t * v_stride + h * d_v;
        float *o_dst = output + t * v_stride + h * d_v;

        // Load Q and K into shared memory (all threads cooperate)
        for (int i = tid; i < d_k; i += block_size) {
            q_local[i] = q_src[i];
            k_local[i] = k_src[i];
        }
        __syncthreads();

        if (use_qk_l2norm) {
            // Q norm
            float q_sum = 0.0f;
            for (int i = tid; i < d_k; i += block_size)
                q_sum += q_local[i] * q_local[i];
            for (int offset = 32; offset > 0; offset >>= 1)
                q_sum += __shfl_xor(q_sum, offset);
            if (lane_id == 0) warp_sums[warp_id] = q_sum;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_sums[w];
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
            for (int offset = 32; offset > 0; offset >>= 1)
                k_sum += __shfl_xor(k_sum, offset);
            if (lane_id == 0) warp_sums[warp_id] = k_sum;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_sums[w];
                warp_sums[0] = total;
            }
            __syncthreads();
            float k_inv = 1.0f / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
            for (int i = tid; i < d_k; i += block_size)
                k_local[i] *= k_inv;
            __syncthreads();
        } else {
            for (int i = tid; i < d_k; i += block_size)
                q_local[i] *= scale;
            __syncthreads();
        }

        // Gate and beta
        float decay, beta_h;
        if (tid == 0) {
            float x = alpha[t * n_heads + h] + dt_bias_h;
            float sp = (x > 20.0f) ? x : log1pf(expf(x));
            warp_sums[0] = expf(A_log_h * sp);
            warp_sums[1] = 1.0f / (1.0f + expf(-beta_raw[t * n_heads + h]));
        }
        __syncthreads();
        decay = warp_sums[0];
        beta_h = warp_sums[1];

        // Row-split column processing: each column is processed by
        // ROW_SPLIT threads. Double-buffered shared memory reduction.
        // NOTE: Register caching tested but regressed on gfx906 (AMD compiler
        // places arrays in scratch memory). 2-pass approach is faster here.
        {
            const int rows_per_split = d_k / ROW_SPLIT;
            const int j_start = split_id * rows_per_split;
            const int j_end = (split_id == ROW_SPLIT - 1) ? d_k : j_start + rows_per_split;

            // Pass 1: Fused decay + partial kv dot product
            float partial_kv = 0.0f;
            if (vi < d_v)
            {
                for (int j = j_start; j < j_end; j++)
                {
                    float s_decayed = S[j * d_v + vi] * decay;
                    partial_kv += s_decayed * k_local[j];
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

            // Pass 2: Fused decay + update + partial output
            float partial_out = 0.0f;
            if (vi < d_v)
            {
                for (int j = j_start; j < j_end; j++)
                {
                    float s_new = S[j * d_v + vi] * decay + k_local[j] * delta;
                    S[j * d_v + vi] = s_new;
                    partial_out += s_new * q_local[j];
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
        __syncthreads();
    }
}

// =========================================================================
// Short Conv1d Kernels
// =========================================================================

__global__ void rocm_short_conv1d_decode_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    float *__restrict__ conv_state,
    int channels, int kernel_size,
    bool apply_silu)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const int ks_minus1 = kernel_size - 1;

    // Compute convolution FIRST using OLD state, then shift
    // (must compute before shift to avoid overwriting state values)
    float sum = 0.0f;
    for (int k = 0; k < ks_minus1; k++)
        sum += conv_state[ch * ks_minus1 + k] * weight[ch * kernel_size + k];
    sum += input[ch] * weight[ch * kernel_size + ks_minus1];

    if (bias) sum += bias[ch];
    if (apply_silu)
        sum = sum / (1.0f + expf(-sum));
    output[ch] = sum;

    // Now shift conv_state left by 1, insert new input at the end
    for (int k = 0; k < ks_minus1 - 1; k++)
        conv_state[ch * ks_minus1 + k] = conv_state[ch * ks_minus1 + k + 1];
    conv_state[ch * ks_minus1 + ks_minus1 - 1] = input[ch];
}

__global__ void rocm_short_conv1d_prefill_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    float *__restrict__ conv_state,
    int seq_len, int channels, int kernel_size,
    bool apply_silu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * channels;
    if (idx >= total) return;

    int t = idx / channels;
    int ch = idx % channels;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int src_t = t - (kernel_size - 1) + k;
        float val = (src_t >= 0) ? input[src_t * channels + ch] : 0.0f;
        sum += val * weight[ch * kernel_size + k];
    }

    if (bias) sum += bias[ch];
    if (apply_silu)
        sum = sum / (1.0f + expf(-sum));
    output[t * channels + ch] = sum;

    int ks_minus1 = kernel_size - 1;
    if (t >= seq_len - ks_minus1 && conv_state) {
        int state_idx = t - (seq_len - ks_minus1);
        conv_state[ch * ks_minus1 + state_idx] = input[t * channels + ch];
    }
}

// =========================================================================
// GatedRMSNorm Kernel — Cooperative (one block per normalization group)
//
// Old kernel: 1 thread per group → for decode (seq_len=1), only 20 threads
//   across the entire GPU. Each thread sequentially loops over norm_dim.
// New kernel: 1 block per group → all threads cooperate on the reduction.
//   With norm_dim=128 and 128 threads/block, each thread handles 1 element.
//   Uses warp shuffle reduction (no shared memory for the sum).
//   Register-caches input values to avoid re-reading from global memory.
// =========================================================================

__global__ void rocm_gated_rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gate,
    const float *__restrict__ gamma,
    float *__restrict__ output,
    int total_work,
    int n_groups,
    int norm_dim,
    int d_model,
    int gamma_period,
    float eps,
    bool subtract_one,
    bool gate_silu)
{
    // One block per (seq_position, group) pair
    const int work_idx = blockIdx.x;
    if (work_idx >= total_work) return;

    const int t = work_idx / n_groups;
    const int g = work_idx % n_groups;
    const int offset = t * d_model + g * norm_dim;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Pass 1: Each thread handles its slice of norm_dim, accumulate sum_sq
    constexpr int kMaxPerThread = 8; // handles norm_dim up to 8*256=2048
    float local_vals[kMaxPerThread];
    int local_count = 0;

    float sum_sq = 0.0f;
    for (int j = tid; j < norm_dim; j += block_size) {
        float v = input[offset + j];
        local_vals[local_count++] = v;
        sum_sq += v * v;
    }

    // Warp shuffle reduction (64-wide wavefronts on GCN/CDNA)
    for (int offset_s = 32; offset_s > 0; offset_s >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset_s);

    // Cross-wavefront reduction via shared memory
    const int lane = tid & 63;
    const int warp_id = tid >> 6;
    const int num_warps = (block_size + 63) >> 6;

    __shared__ float warp_sums[4]; // max 256 threads = 4 wavefronts
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        warp_sums[0] = total;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / (float)norm_dim + eps);

    // Pass 2: Normalize with register-cached values
    int idx = 0;
    for (int j = tid; j < norm_dim; j += block_size) {
        float normalized = local_vals[idx++] * inv_rms;
        int gamma_idx = j % gamma_period;
        float gamma_eff = subtract_one ? (1.0f + gamma[gamma_idx]) : gamma[gamma_idx];
        float gate_val = gate[offset + j];
        float gate_act;
        if (gate_silu)
            gate_act = gate_val / (1.0f + expf(-gate_val));
        else
            gate_act = gate_val;
        output[offset + j] = normalized * gamma_eff * gate_act;
    }
}

// =========================================================================
// AttentionOutputGate Kernel
// =========================================================================

__global__ void rocm_attention_output_gate_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gate,
    float *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + expf(-gate[idx]));
        output[idx] = sig * input[idx];
    }
}

} // anonymous namespace (kernel functions)

// =========================================================================
// QKV Deinterleave Kernel (for merged QKV buffer on device)
// =========================================================================

namespace {

__global__ void rocm_gdn_deinterleave_qkv_kernel(
    const float *__restrict__ merged,
    float *__restrict__ out_q,
    float *__restrict__ out_k,
    float *__restrict__ out_v,
    int seq_len, int n_k_heads, int n_v_heads,
    int d_k, int d_v, int global_v_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int q_src_dim = n_k_heads * d_k;
    int k_src_dim = n_k_heads * d_k;
    int v_dim = n_v_heads * d_v;
    int stride = q_src_dim + k_src_dim + v_dim;

    int q_dst_dim = n_v_heads * d_k;
    int k_dst_dim = n_v_heads * d_k;

    int q_total = seq_len * q_dst_dim;
    int k_total = seq_len * k_dst_dim;
    int v_total = seq_len * v_dim;
    int total = q_total + k_total + v_total;

    if (idx >= total)
        return;

    if (idx < q_total)
    {
        int t = idx / q_dst_dim;
        int rem = idx % q_dst_dim;
        int j = rem / d_k;
        int d = rem % d_k;
        int k_idx = (j + global_v_offset) % n_k_heads;
        out_q[idx] = merged[t * stride + k_idx * d_k + d];
    }
    else if (idx < q_total + k_total)
    {
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

extern "C" {

bool rocmGDN_gpu_malloc(float **ptr, size_t count)
{
    hipError_t err = hipMalloc(ptr, count * sizeof(float));
    return err == hipSuccess;
}

void rocmGDN_gpu_free(float *ptr)
{
    if (ptr) hipFree(ptr);
}

void rocmGDN_gpu_memset_zero(float *ptr, size_t count)
{
    if (ptr && count > 0) hipMemset(ptr, 0, count * sizeof(float));
}

void rocmGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream)
{
    if (ptr && count > 0) hipMemsetAsync(ptr, 0, count * sizeof(float), (hipStream_t)stream);
}

void rocmGDN_gpu_set_device(int ordinal)
{
    hipSetDevice(ordinal);
}

void rocmGDN_gpu_memcpy(float *dst, const float *src, size_t count)
{
    hipMemcpy(dst, src, count * sizeof(float), hipMemcpyDefault);
}

void rocmGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream)
{
    hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDefault, (hipStream_t)stream);
}

void rocmGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count)
{
    hipMemcpy(host_dst, device_src, count * sizeof(float), hipMemcpyDeviceToHost);
}

void rocmGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream)
{
    hipMemcpyAsync(host_dst, device_src, count * sizeof(float), hipMemcpyDeviceToHost, (hipStream_t)stream);
}

void rocmGDN_stream_synchronize(void *stream)
{
    hipStreamSynchronize((hipStream_t)stream);
}

} // extern "C" (memory helpers)

// =========================================================================
// QKV Deinterleave Wrapper
// =========================================================================

extern "C" {

bool rocmGDN_deinterleave_qkv(
    const float *merged, float *out_q, float *out_k, float *out_v,
    int seq_len, int n_k_heads, int n_v_heads,
    int d_k, int d_v, int global_v_offset,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int q_dst_dim = n_v_heads * d_k;
    int k_dst_dim = n_v_heads * d_k;
    int v_dim = n_v_heads * d_v;
    int total = seq_len * (q_dst_dim + k_dst_dim + v_dim);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_gdn_deinterleave_qkv_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        merged, out_q, out_k, out_v,
        seq_len, n_k_heads, n_v_heads, d_k, d_v, global_v_offset);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGDN_deinterleave_qkv] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (deinterleave)

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C" {

bool rocmGDN_recurrent_step(
    const float *q, const float *k, const float *v,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, float *state,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    // 2D grid: x=heads, y=column tiles (64 threads = 1 wavefront per block)
    int col_threads = 64;
    int num_col_blocks = (d_v + col_threads - 1) / col_threads;
    int smem_size = 2 * d_k * sizeof(float);

    hipLaunchKernelGGL(rocm_gdn_recurrent_step_kernel,
        dim3(n_heads, num_col_blocks), dim3(col_threads), smem_size, (hipStream_t)stream,
        q, k, v, alpha, beta_raw, A_log, dt_bias,
        output, state,
        n_heads, d_k, d_v, use_qk_l2norm);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_recurrent_step] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_chunk_forward(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, float *state,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    // Row-split: 512 threads per block, 8 threads per column = 64 cols/block.
    // 8 wavefronts/block — optimal for gfx906 latency hiding.
    int col_threads = 512;
    int cols_per_block = col_threads / 8; // 64 columns per block
    if (d_v <= 64)
    {
        col_threads = 256;
        cols_per_block = col_threads / 8; // 32 columns per block
    }
    int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
    // smem: q_local[d_k] + k_local[d_k] + warp_sums[16] + reduce_kv[col_threads] + reduce_out[col_threads]
    int smem_size = (2 * d_k + 16 + 2 * col_threads) * sizeof(float);

    hipLaunchKernelGGL(rocm_gdn_chunk_forward_kernel,
        dim3(n_heads, num_col_blocks), dim3(col_threads), smem_size, (hipStream_t)stream,
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, state,
        seq_len, n_heads, d_k, d_v, use_qk_l2norm);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_chunk_forward] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_short_conv1d(
    const float *input, const float *weight, const float *bias,
    float *output, float *conv_state,
    int seq_len, int channels, int kernel_size,
    bool apply_silu,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (seq_len == 1) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_decode_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output, conv_state,
            channels, kernel_size, apply_silu);
    } else {
        int total = seq_len * channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_prefill_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output, conv_state,
            seq_len, channels, kernel_size, apply_silu);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_short_conv1d] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_gated_rmsnorm(
    const float *input, const float *gate, const float *gamma,
    float *output,
    int seq_len, int d_model, int norm_dim, int gamma_period,
    float eps, bool subtract_one, bool gate_silu,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int n_groups = d_model / norm_dim;
    int total_work = seq_len * n_groups;

    // Cooperative launch: one block per (seq_pos, group) pair.
    // Threads within a block cooperate on the norm_dim reduction.
    // Use min(norm_dim, 256) threads — for norm_dim=128, use 128 threads.
    int threads = (norm_dim <= 64) ? 64 : ((norm_dim <= 128) ? 128 : 256);
    int blocks = total_work;

    hipLaunchKernelGGL(rocm_gated_rmsnorm_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, gate, gamma, output,
        total_work, n_groups, norm_dim, d_model, gamma_period,
        eps, subtract_one, gate_silu);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_gated_rmsnorm] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_attention_output_gate(
    const float *input, const float *gate, float *output,
    int size,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_attention_output_gate_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, gate, output, size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_attention_output_gate] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" — end of first block

// =========================================================================
// Q+Gate Split kernel
// =========================================================================
namespace {
__global__ void rocm_q_gate_split_kernel(
    const float *__restrict__ input,
    float *__restrict__ output_q,
    float *__restrict__ output_gate,
    int total_elements, int n_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

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
} // anonymous namespace (rocm_q_gate_split_kernel)

extern "C" {

bool rocmGDN_q_gate_split(
    const float *input, float *output_q, float *output_gate,
    int seq_len, int n_heads, int head_dim,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int total = seq_len * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_q_gate_split_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, output_q, output_gate, total, n_heads, head_dim);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_q_gate_split] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
