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
#include "../../../utils/DebugEnv.h"

namespace {

const char *memory_type_name(hipMemoryType type)
{
    switch (type)
    {
    case hipMemoryTypeHost: return "host";
    case hipMemoryTypeDevice: return "device";
    case hipMemoryTypeArray: return "array";
    case hipMemoryTypeUnified: return "unified";
    default: return "unknown";
    }
}

void print_pointer_attrs(const char *label, const void *ptr)
{
    hipPointerAttribute_t attrs{};
    hipError_t err = hipPointerGetAttributes(&attrs, ptr);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGDN_ptr] %s=%p attrs=%s\n", label, ptr, hipGetErrorString(err));
        return;
    }
    fprintf(stderr, "[rocmGDN_ptr] %s=%p type=%s device=%d devicePointer=%p hostPointer=%p\n",
            label, ptr, memory_type_name(attrs.type), attrs.device,
            attrs.devicePointer, attrs.hostPointer);
}

} // anonymous namespace (diagnostics)

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace { // anonymous namespace to avoid symbol conflicts with CUDA kernels

// =========================================================================
// Decode Recurrence Kernel (seq_len=1, multi-block per head)
//
// V3 Optimizations:
//   1. ROW_SPLIT=4: 256 threads/block (4 wavefronts) — each column
//      processed by 4 threads splitting d_k rows. Increases occupancy
//      from 0.53 to 2.1 waves/CU on gfx906 (32→128 total wavefronts).
//   2. Fused decay into per-column loops — eliminates separate decay pass
//   3. Two-pass column processing: Pass 1 reads state (decay+kv), Pass 2
//      reads+writes (decay+update+output). 50% traffic savings.
//   4. 2D grid: (n_heads, col_blocks) with col_blocks = d_v / 64
//   5. Shared-memory reduction for cross-wavefront kv/output sums
//
// Memory: d_k × d_v × 4 bytes per head = 64KB state
// Occupancy: 128 wavefronts / 60 CUs = 2.1 waves/CU (was 0.53)
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

    // ROW_SPLIT=4: 4 threads collaborate on each column, splitting d_k rows
    constexpr int ROW_SPLIT = 4;
    const int cols_per_block = block_size / ROW_SPLIT; // 64
    const int split_id = tid / cols_per_block;         // 0..3
    const int col_in_block = tid % cols_per_block;     // 0..63
    const int vi = blockIdx.y * cols_per_block + col_in_block;

    const float *q_head = q + h * d_k;
    const float *k_head = k + h * d_k;
    const float *v_head = v + h * d_v;
    float *S = state + (size_t)h * d_k * d_v;
    float *o_head = output + h * d_v;

    extern __shared__ float smem[];
    float *q_local = smem;                              // [d_k]
    float *k_local = smem + d_k;                        // [d_k]
    float *warp_sums = smem + 2 * d_k;                  // [4] for norm reduction
    float *reduce_kv = smem + 2 * d_k + 4;             // [block_size]
    float *reduce_out = smem + 2 * d_k + 4 + block_size; // [block_size]

    const float scale = rsqrtf((float)d_k);
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;
    const int num_warps = block_size / 64; // 4

    // Load Q and K into shared memory (all 256 threads cooperate)
    for (int i = tid; i < d_k; i += block_size) {
        q_local[i] = q_head[i];
        k_local[i] = k_head[i];
    }
    __syncthreads();

    if (use_qk_l2norm) {
        // Multi-wavefront Q norm reduction
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

        // Multi-wavefront K norm reduction
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

    // Gate and beta — compute on thread 0, broadcast via shared memory
    float decay, beta_h;
    if (tid == 0) {
        float x = alpha[h] + dt_bias[h];
        float sp = (x > 20.0f) ? x : log1pf(expf(x));
        warp_sums[0] = expf(A_log[h] * sp);
        warp_sums[1] = 1.0f / (1.0f + expf(-beta_raw[h]));
    }
    __syncthreads();
    decay = warp_sums[0];
    beta_h = warp_sums[1];

    // Row-split column processing: ROW_SPLIT=4 threads collaborate on each
    // column, splitting d_k rows. Shared-memory reduction for kv and output.
    {
        const int rows_per_split = d_k / ROW_SPLIT;
        const int j_start = split_id * rows_per_split;
        const int j_end = (split_id == ROW_SPLIT - 1) ? d_k : j_start + rows_per_split;

        // Pass 1: Fused decay + partial kv dot product (read-only)
        float partial_kv = 0.0f;
        if (vi < d_v) {
            for (int j = j_start; j < j_end; j++) {
                float s_decayed = S[j * d_v + vi] * decay;
                partial_kv += s_decayed * k_local[j];
            }
        }
        reduce_kv[tid] = partial_kv;
        __syncthreads();

        // Reduce kv across splits and compute delta
        float delta = 0.0f;
        if (vi < d_v) {
            float kv = 0.0f;
            for (int s = 0; s < ROW_SPLIT; s++)
                kv += reduce_kv[col_in_block + s * cols_per_block];
            delta = (v_head[vi] - kv) * beta_h;
        }

        // Pass 2: Fused decay + update + partial output (read-modify-write)
        float partial_out = 0.0f;
        if (vi < d_v) {
            for (int j = j_start; j < j_end; j++) {
                float s_new = S[j * d_v + vi] * decay + k_local[j] * delta;
                S[j * d_v + vi] = s_new;
                partial_out += s_new * q_local[j];
            }
        }
        reduce_out[tid] = partial_out;
        __syncthreads();

        // Reduce output across splits — only split_id=0 writes final output
        if (vi < d_v && split_id == 0) {
            float out_vi = 0.0f;
            for (int s = 0; s < ROW_SPLIT; s++)
                out_vi += reduce_out[col_in_block + s * cols_per_block];
            o_head[vi] = out_vi;
        }
    }
}

// =========================================================================
// Prefill Recurrence Kernel (seq_len>1, register-cached state)
//
// V4 optimizations:
//   1. State cached in registers: each thread keeps its 32 state values
//      in VGPRs across all 595 timesteps. State is loaded from global
//      ONCE at kernel start and written back ONCE at end.
//   2. Single-pass per timestep: kv dot product → delta → state update →
//      output accumulation in ONE loop over the 32 cached rows.
//   3. 2D grid: dim3(n_heads, col_blocks) with ROW_SPLIT=4 for parity.
//   4. Eliminates ~1190 global memory state R/W passes per thread.
//
// Register budget: 32 state floats + ~20 other = 52 VGPRs per thread.
// gfx906 has 256 VGPRs → max occupancy = 4 waves/SIMD = 16 waves/CU.
// (Currently only 4 waves/CU from 1 block, so this is occupancy-neutral.)
// =========================================================================

__global__ __launch_bounds__(256, 2)
void rocm_gdn_chunk_forward_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ alpha,
    const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    float *__restrict__ output,
    float *__restrict__ state,
    const int *__restrict__ effective_seq_len_ptr,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    constexpr int ROW_SPLIT = 4;
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

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    // =====================================================================
    // REGISTER-CACHED STATE: Load state into registers ONCE
    // Each thread owns rows_per_split (=d_k/ROW_SPLIT) state values for
    // its column vi. With d_k=128, ROW_SPLIT=4: 32 floats per thread.
    // =====================================================================
    const int rows_per_split = d_k / ROW_SPLIT;
    const int j_start = split_id * rows_per_split;

    // Use individual variables in groups of 8 to help the compiler
    // keep them in VGPRs (AMD compiler more likely to register-allocate
    // fixed-size unrolled groups than a single large array).
    float sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7;
    float sc8, sc9, sc10, sc11, sc12, sc13, sc14, sc15;
    float sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23;
    float sc24, sc25, sc26, sc27, sc28, sc29, sc30, sc31;

    if (vi < d_v) {
        sc0  = S[(j_start +  0) * d_v + vi];
        sc1  = S[(j_start +  1) * d_v + vi];
        sc2  = S[(j_start +  2) * d_v + vi];
        sc3  = S[(j_start +  3) * d_v + vi];
        sc4  = S[(j_start +  4) * d_v + vi];
        sc5  = S[(j_start +  5) * d_v + vi];
        sc6  = S[(j_start +  6) * d_v + vi];
        sc7  = S[(j_start +  7) * d_v + vi];
        sc8  = S[(j_start +  8) * d_v + vi];
        sc9  = S[(j_start +  9) * d_v + vi];
        sc10 = S[(j_start + 10) * d_v + vi];
        sc11 = S[(j_start + 11) * d_v + vi];
        sc12 = S[(j_start + 12) * d_v + vi];
        sc13 = S[(j_start + 13) * d_v + vi];
        sc14 = S[(j_start + 14) * d_v + vi];
        sc15 = S[(j_start + 15) * d_v + vi];
        sc16 = S[(j_start + 16) * d_v + vi];
        sc17 = S[(j_start + 17) * d_v + vi];
        sc18 = S[(j_start + 18) * d_v + vi];
        sc19 = S[(j_start + 19) * d_v + vi];
        sc20 = S[(j_start + 20) * d_v + vi];
        sc21 = S[(j_start + 21) * d_v + vi];
        sc22 = S[(j_start + 22) * d_v + vi];
        sc23 = S[(j_start + 23) * d_v + vi];
        sc24 = S[(j_start + 24) * d_v + vi];
        sc25 = S[(j_start + 25) * d_v + vi];
        sc26 = S[(j_start + 26) * d_v + vi];
        sc27 = S[(j_start + 27) * d_v + vi];
        sc28 = S[(j_start + 28) * d_v + vi];
        sc29 = S[(j_start + 29) * d_v + vi];
        sc30 = S[(j_start + 30) * d_v + vi];
        sc31 = S[(j_start + 31) * d_v + vi];
    } else {
        sc0=sc1=sc2=sc3=sc4=sc5=sc6=sc7=0.0f;
        sc8=sc9=sc10=sc11=sc12=sc13=sc14=sc15=0.0f;
        sc16=sc17=sc18=sc19=sc20=sc21=sc22=sc23=0.0f;
        sc24=sc25=sc26=sc27=sc28=sc29=sc30=sc31=0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        const float *q_src = Q + t * qk_stride + h * d_k;
        const float *k_src = K + t * qk_stride + h * d_k;
        const float *v_src = V + t * v_stride + h * d_v;
        float *o_dst = output + t * v_stride + h * d_v;

        if (t >= effective_seq_len) {
            if (vi < d_v && split_id == 0)
                o_dst[vi] = 0.0f;
            continue;
        }

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

        // =================================================================
        // SINGLE-PASS register-cached state processing
        // Pass 1: decay state + accumulate kv dot product (from registers)
        // =================================================================
        float partial_kv = 0.0f;
        if (vi < d_v) {
            // Apply decay to cached state and compute kv dot product
            sc0 *= decay; partial_kv += sc0 * k_local[j_start + 0];
            sc1 *= decay; partial_kv += sc1 * k_local[j_start + 1];
            sc2 *= decay; partial_kv += sc2 * k_local[j_start + 2];
            sc3 *= decay; partial_kv += sc3 * k_local[j_start + 3];
            sc4 *= decay; partial_kv += sc4 * k_local[j_start + 4];
            sc5 *= decay; partial_kv += sc5 * k_local[j_start + 5];
            sc6 *= decay; partial_kv += sc6 * k_local[j_start + 6];
            sc7 *= decay; partial_kv += sc7 * k_local[j_start + 7];
            sc8 *= decay; partial_kv += sc8 * k_local[j_start + 8];
            sc9 *= decay; partial_kv += sc9 * k_local[j_start + 9];
            sc10 *= decay; partial_kv += sc10 * k_local[j_start + 10];
            sc11 *= decay; partial_kv += sc11 * k_local[j_start + 11];
            sc12 *= decay; partial_kv += sc12 * k_local[j_start + 12];
            sc13 *= decay; partial_kv += sc13 * k_local[j_start + 13];
            sc14 *= decay; partial_kv += sc14 * k_local[j_start + 14];
            sc15 *= decay; partial_kv += sc15 * k_local[j_start + 15];
            sc16 *= decay; partial_kv += sc16 * k_local[j_start + 16];
            sc17 *= decay; partial_kv += sc17 * k_local[j_start + 17];
            sc18 *= decay; partial_kv += sc18 * k_local[j_start + 18];
            sc19 *= decay; partial_kv += sc19 * k_local[j_start + 19];
            sc20 *= decay; partial_kv += sc20 * k_local[j_start + 20];
            sc21 *= decay; partial_kv += sc21 * k_local[j_start + 21];
            sc22 *= decay; partial_kv += sc22 * k_local[j_start + 22];
            sc23 *= decay; partial_kv += sc23 * k_local[j_start + 23];
            sc24 *= decay; partial_kv += sc24 * k_local[j_start + 24];
            sc25 *= decay; partial_kv += sc25 * k_local[j_start + 25];
            sc26 *= decay; partial_kv += sc26 * k_local[j_start + 26];
            sc27 *= decay; partial_kv += sc27 * k_local[j_start + 27];
            sc28 *= decay; partial_kv += sc28 * k_local[j_start + 28];
            sc29 *= decay; partial_kv += sc29 * k_local[j_start + 29];
            sc30 *= decay; partial_kv += sc30 * k_local[j_start + 30];
            sc31 *= decay; partial_kv += sc31 * k_local[j_start + 31];
        }
        reduce_kv[tid] = partial_kv;
        __syncthreads();

        // Reduce kv across splits and compute delta
        float delta = 0.0f;
        if (vi < d_v) {
            float kv = 0.0f;
            for (int s = 0; s < ROW_SPLIT; s++)
                kv += reduce_kv[col_in_block + s * cols_per_block];
            delta = (v_src[vi] - kv) * beta_h;
        }

        // =================================================================
        // Pass 2: Update state + accumulate output (all in registers)
        // =================================================================
        float partial_out = 0.0f;
        if (vi < d_v) {
            sc0 += k_local[j_start + 0] * delta; partial_out += sc0 * q_local[j_start + 0];
            sc1 += k_local[j_start + 1] * delta; partial_out += sc1 * q_local[j_start + 1];
            sc2 += k_local[j_start + 2] * delta; partial_out += sc2 * q_local[j_start + 2];
            sc3 += k_local[j_start + 3] * delta; partial_out += sc3 * q_local[j_start + 3];
            sc4 += k_local[j_start + 4] * delta; partial_out += sc4 * q_local[j_start + 4];
            sc5 += k_local[j_start + 5] * delta; partial_out += sc5 * q_local[j_start + 5];
            sc6 += k_local[j_start + 6] * delta; partial_out += sc6 * q_local[j_start + 6];
            sc7 += k_local[j_start + 7] * delta; partial_out += sc7 * q_local[j_start + 7];
            sc8 += k_local[j_start + 8] * delta; partial_out += sc8 * q_local[j_start + 8];
            sc9 += k_local[j_start + 9] * delta; partial_out += sc9 * q_local[j_start + 9];
            sc10 += k_local[j_start + 10] * delta; partial_out += sc10 * q_local[j_start + 10];
            sc11 += k_local[j_start + 11] * delta; partial_out += sc11 * q_local[j_start + 11];
            sc12 += k_local[j_start + 12] * delta; partial_out += sc12 * q_local[j_start + 12];
            sc13 += k_local[j_start + 13] * delta; partial_out += sc13 * q_local[j_start + 13];
            sc14 += k_local[j_start + 14] * delta; partial_out += sc14 * q_local[j_start + 14];
            sc15 += k_local[j_start + 15] * delta; partial_out += sc15 * q_local[j_start + 15];
            sc16 += k_local[j_start + 16] * delta; partial_out += sc16 * q_local[j_start + 16];
            sc17 += k_local[j_start + 17] * delta; partial_out += sc17 * q_local[j_start + 17];
            sc18 += k_local[j_start + 18] * delta; partial_out += sc18 * q_local[j_start + 18];
            sc19 += k_local[j_start + 19] * delta; partial_out += sc19 * q_local[j_start + 19];
            sc20 += k_local[j_start + 20] * delta; partial_out += sc20 * q_local[j_start + 20];
            sc21 += k_local[j_start + 21] * delta; partial_out += sc21 * q_local[j_start + 21];
            sc22 += k_local[j_start + 22] * delta; partial_out += sc22 * q_local[j_start + 22];
            sc23 += k_local[j_start + 23] * delta; partial_out += sc23 * q_local[j_start + 23];
            sc24 += k_local[j_start + 24] * delta; partial_out += sc24 * q_local[j_start + 24];
            sc25 += k_local[j_start + 25] * delta; partial_out += sc25 * q_local[j_start + 25];
            sc26 += k_local[j_start + 26] * delta; partial_out += sc26 * q_local[j_start + 26];
            sc27 += k_local[j_start + 27] * delta; partial_out += sc27 * q_local[j_start + 27];
            sc28 += k_local[j_start + 28] * delta; partial_out += sc28 * q_local[j_start + 28];
            sc29 += k_local[j_start + 29] * delta; partial_out += sc29 * q_local[j_start + 29];
            sc30 += k_local[j_start + 30] * delta; partial_out += sc30 * q_local[j_start + 30];
            sc31 += k_local[j_start + 31] * delta; partial_out += sc31 * q_local[j_start + 31];
        }
        reduce_out[tid] = partial_out;
        __syncthreads();

        // Reduce output and write (only split_id=0 writes)
        if (vi < d_v && split_id == 0) {
            float out_vi = 0.0f;
            for (int s = 0; s < ROW_SPLIT; s++)
                out_vi += reduce_out[col_in_block + s * cols_per_block];
            o_dst[vi] = out_vi;
        }
        if (state_snapshots && t < max_snapshot_rows && vi < d_v) {
            float *snapshot =
                state_snapshots +
                (size_t)t * snapshot_stride_floats +
                (size_t)h * d_k * d_v;
            snapshot[(j_start +  0) * d_v + vi] = sc0;
            snapshot[(j_start +  1) * d_v + vi] = sc1;
            snapshot[(j_start +  2) * d_v + vi] = sc2;
            snapshot[(j_start +  3) * d_v + vi] = sc3;
            snapshot[(j_start +  4) * d_v + vi] = sc4;
            snapshot[(j_start +  5) * d_v + vi] = sc5;
            snapshot[(j_start +  6) * d_v + vi] = sc6;
            snapshot[(j_start +  7) * d_v + vi] = sc7;
            snapshot[(j_start +  8) * d_v + vi] = sc8;
            snapshot[(j_start +  9) * d_v + vi] = sc9;
            snapshot[(j_start + 10) * d_v + vi] = sc10;
            snapshot[(j_start + 11) * d_v + vi] = sc11;
            snapshot[(j_start + 12) * d_v + vi] = sc12;
            snapshot[(j_start + 13) * d_v + vi] = sc13;
            snapshot[(j_start + 14) * d_v + vi] = sc14;
            snapshot[(j_start + 15) * d_v + vi] = sc15;
            snapshot[(j_start + 16) * d_v + vi] = sc16;
            snapshot[(j_start + 17) * d_v + vi] = sc17;
            snapshot[(j_start + 18) * d_v + vi] = sc18;
            snapshot[(j_start + 19) * d_v + vi] = sc19;
            snapshot[(j_start + 20) * d_v + vi] = sc20;
            snapshot[(j_start + 21) * d_v + vi] = sc21;
            snapshot[(j_start + 22) * d_v + vi] = sc22;
            snapshot[(j_start + 23) * d_v + vi] = sc23;
            snapshot[(j_start + 24) * d_v + vi] = sc24;
            snapshot[(j_start + 25) * d_v + vi] = sc25;
            snapshot[(j_start + 26) * d_v + vi] = sc26;
            snapshot[(j_start + 27) * d_v + vi] = sc27;
            snapshot[(j_start + 28) * d_v + vi] = sc28;
            snapshot[(j_start + 29) * d_v + vi] = sc29;
            snapshot[(j_start + 30) * d_v + vi] = sc30;
            snapshot[(j_start + 31) * d_v + vi] = sc31;
        }
        __syncthreads();
    }

    // =====================================================================
    // Write state back to global memory ONCE at the end
    // =====================================================================
    if (vi < d_v) {
        S[(j_start +  0) * d_v + vi] = sc0;
        S[(j_start +  1) * d_v + vi] = sc1;
        S[(j_start +  2) * d_v + vi] = sc2;
        S[(j_start +  3) * d_v + vi] = sc3;
        S[(j_start +  4) * d_v + vi] = sc4;
        S[(j_start +  5) * d_v + vi] = sc5;
        S[(j_start +  6) * d_v + vi] = sc6;
        S[(j_start +  7) * d_v + vi] = sc7;
        S[(j_start +  8) * d_v + vi] = sc8;
        S[(j_start +  9) * d_v + vi] = sc9;
        S[(j_start + 10) * d_v + vi] = sc10;
        S[(j_start + 11) * d_v + vi] = sc11;
        S[(j_start + 12) * d_v + vi] = sc12;
        S[(j_start + 13) * d_v + vi] = sc13;
        S[(j_start + 14) * d_v + vi] = sc14;
        S[(j_start + 15) * d_v + vi] = sc15;
        S[(j_start + 16) * d_v + vi] = sc16;
        S[(j_start + 17) * d_v + vi] = sc17;
        S[(j_start + 18) * d_v + vi] = sc18;
        S[(j_start + 19) * d_v + vi] = sc19;
        S[(j_start + 20) * d_v + vi] = sc20;
        S[(j_start + 21) * d_v + vi] = sc21;
        S[(j_start + 22) * d_v + vi] = sc22;
        S[(j_start + 23) * d_v + vi] = sc23;
        S[(j_start + 24) * d_v + vi] = sc24;
        S[(j_start + 25) * d_v + vi] = sc25;
        S[(j_start + 26) * d_v + vi] = sc26;
        S[(j_start + 27) * d_v + vi] = sc27;
        S[(j_start + 28) * d_v + vi] = sc28;
        S[(j_start + 29) * d_v + vi] = sc29;
        S[(j_start + 30) * d_v + vi] = sc30;
        S[(j_start + 31) * d_v + vi] = sc31;
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
    const int *__restrict__ effective_seq_len_ptr,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int seq_len, int channels, int kernel_size,
    bool apply_silu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ks_minus1 = kernel_size - 1;
    int total = seq_len * channels;
    if (idx >= total) return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    int t = idx / channels;
    int ch = idx % channels;

    float sum = 0.0f;
    if (t < effective_seq_len) {
        for (int k = 0; k < kernel_size; k++) {
            int src_t = t - ks_minus1 + k;
            float val = 0.0f;
            if (src_t >= 0)
                val = input[src_t * channels + ch];
            else if (conv_state)
                val = conv_state[ch * ks_minus1 + ks_minus1 + src_t];
            sum += val * weight[ch * kernel_size + k];
        }

        if (bias) sum += bias[ch];
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));
    }
    output[t * channels + ch] = sum;

    if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows) {
        float *snapshot =
            state_snapshots +
            (size_t)t * snapshot_stride_floats +
            (size_t)ch * ks_minus1;
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx) {
            int src_t = t - ks_minus1 + 1 + state_idx;
            snapshot[state_idx] =
                (src_t >= 0) ? input[src_t * channels + ch]
                             : (conv_state ? conv_state[ch * ks_minus1 + ks_minus1 + src_t] : 0.0f);
        }
    }
}

__global__ void rocm_short_conv1d_state_update_kernel(
    const float *__restrict__ input,
    float *__restrict__ conv_state,
    const int *__restrict__ effective_seq_len_ptr,
    int seq_len, int channels, int kernel_size)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels || !conv_state) return;

    int ks_minus1 = kernel_size - 1;
    if (ks_minus1 <= 0) return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    float *state = conv_state + ch * ks_minus1;
    for (int state_idx = 0; state_idx < ks_minus1; ++state_idx) {
        int src_t = effective_seq_len - ks_minus1 + state_idx;
        state[state_idx] =
            (src_t >= 0 && src_t < effective_seq_len) ? input[src_t * channels + ch]
                                                       : state[ks_minus1 + src_t];
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
    if (n_k_heads <= 0 || n_v_heads <= 0 || d_k <= 0 || d_v <= 0)
        return;

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
        if (k_idx < 0)
            k_idx += n_k_heads;
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
        if (k_idx < 0)
            k_idx += n_k_heads;
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

namespace {

__global__ void rocm_gdn_restore_state_from_capture_metadata_kernel(
    float *__restrict__ dst_state,
    const float *__restrict__ state_snapshots,
    const int32_t *__restrict__ device_accepted_state_slot_indices,
    int request_index,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int state_floats)
{
    const int row = device_accepted_state_slot_indices[request_index];
    if (row < 0 || row >= max_snapshot_rows)
        return;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_floats)
        return;

    dst_state[idx] =
        state_snapshots[static_cast<size_t>(row) *
                            static_cast<size_t>(snapshot_stride_floats) +
                        static_cast<size_t>(idx)];
}

} // anonymous namespace (state metadata restore kernel)

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

bool rocmGDN_restore_state_from_capture_metadata(
    float *dst_state,
    const float *state_snapshots,
    const int32_t *device_accepted_state_slot_indices,
    int request_index,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int state_floats,
    int device_idx,
    void *stream)
{
    if (!dst_state || !state_snapshots || !device_accepted_state_slot_indices ||
        request_index < 0 || snapshot_stride_floats < state_floats ||
        max_snapshot_rows <= 0 || state_floats <= 0 || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    const int blocks = (state_floats + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_gdn_restore_state_from_capture_metadata_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        dst_state,
        state_snapshots,
        device_accepted_state_slot_indices,
        request_index,
        snapshot_stride_floats,
        max_snapshot_rows,
        state_floats);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGDN_restore_state_from_capture_metadata] %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
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

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("merged", merged);
        print_pointer_attrs("out_q", out_q);
        print_pointer_attrs("out_k", out_k);
        print_pointer_attrs("out_v", out_v);
    }

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
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_deinterleave_qkv_sync] %s\n", hipGetErrorString(err));
            return false;
        }
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

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("q", q);
        print_pointer_attrs("k", k);
        print_pointer_attrs("v", v);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("state", state);
    }

    // ROW_SPLIT=4: 256 threads/block (4 wavefronts), 64 columns per block.
    // Increases occupancy from 0.53 to 2.1 waves/CU on gfx906 (32→128 wavefronts).
    // Shared memory: q[d_k] + k[d_k] + warp_sums[4] + reduce_kv[256] + reduce_out[256]
    int col_threads = 256;
    int cols_per_block = col_threads / 4; // 64 columns with ROW_SPLIT=4
    int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
    int smem_size = (2 * d_k + 4 + 2 * col_threads) * sizeof(float);

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
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_recurrent_step_sync] %s\n", hipGetErrorString(err));
            return false;
        }
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
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("Q", Q);
        print_pointer_attrs("K", K);
        print_pointer_attrs("V", V);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("state", state);
    }

    // Row-split: 256 threads per block, 4 threads per column = 64 cols/block.
    // 4 wavefronts/block — matches CUDA's ROW_SPLIT=4 to reduce per-step
    // reduction error that compounds through the GDN recurrence.
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

    hipLaunchKernelGGL(rocm_gdn_chunk_forward_kernel,
        dim3(n_heads, num_col_blocks), dim3(col_threads), smem_size, (hipStream_t)stream,
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, state,
        nullptr,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        seq_len, n_heads, d_k, d_v, use_qk_l2norm);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_chunk_forward] %s\n", hipGetErrorString(err));
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_chunk_forward_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmGDN_chunk_forward_effective(
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
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("Q", Q);
        print_pointer_attrs("K", K);
        print_pointer_attrs("V", V);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("state", state);
        print_pointer_attrs("effective_seq_len", device_effective_seq_len);
    }

    int col_threads = 256;
    int cols_per_block = col_threads / 4;
    if (d_v <= 64)
    {
        col_threads = 128;
        cols_per_block = col_threads / 4;
    }
    int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
    int smem_size = (2 * d_k + 16 + 2 * col_threads) * sizeof(float);

    hipLaunchKernelGGL(rocm_gdn_chunk_forward_kernel,
        dim3(n_heads, num_col_blocks), dim3(col_threads), smem_size, (hipStream_t)stream,
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, state,
        device_effective_seq_len,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        seq_len, n_heads, d_k, d_v, use_qk_l2norm);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_chunk_forward_effective] %s\n", hipGetErrorString(err));
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_chunk_forward_effective_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmGDN_short_conv1d(
    const float *input, const float *weight, const float *bias,
    float *output, float *conv_state,
    int seq_len, int channels, int kernel_size,
    bool apply_silu,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
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
            nullptr,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
        int state_blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_state_update_kernel,
            dim3(state_blocks), dim3(threads), 0, (hipStream_t)stream,
            input, conv_state, nullptr,
            seq_len, channels, kernel_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_short_conv1d] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_short_conv1d_effective(
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
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
        int state_blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_state_update_kernel,
            dim3(state_blocks), dim3(threads), 0, (hipStream_t)stream,
            input, conv_state, device_effective_seq_len,
            seq_len, channels, kernel_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_short_conv1d_effective] %s\n", hipGetErrorString(err));
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
