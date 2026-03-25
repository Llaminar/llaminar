/**
 * @file CUDAFusedOpsKernels.cu
 * @brief Fused CUDA element-wise kernels for LLM inference optimization
 *
 * Contains fused operations that eliminate intermediate memory round-trips:
 * 1. Fused SwiGLU + Blockwise INT8 Quantization (replaces SwiGLU + quant kernels)
 * 2. Fused Residual Add + RMSNorm (replaces residual_add + rmsnorm kernels)
 *
 * Design: Each fused kernel reads inputs once and produces final outputs,
 * avoiding writing intermediate FP32 tensors to global memory.
 */

#include "CUDAHelpers.cuh"
#include <cstdint>
#include <cstdio>

// =========================================================================
// 1. Fused SwiGLU + Blockwise INT8 Quantization
// =========================================================================

/**
 * @brief Fused SwiGLU activation + blockwise INT8 quantization
 *
 * Computes: silu(gate) * up, then quantizes each 32-element block to INT8.
 * Replaces two separate kernels (swiglu_fp32 + quantize_activations_blockwise)
 * by eliminating the intermediate FP32 write+read of SwiGLU output.
 *
 * Memory savings per call: 2 × M × K × sizeof(float) (write + read eliminated)
 *
 * Uses 2D grid for K-parallel execution:
 * Grid:  (grid_x, M) — multiple CUDA blocks share the K-blocks of each row
 * Block: (128, 1, 1) — 4 warps, each processes one 32-element quant block at a time
 *
 * Each 32-element K-block is independently quantized, so K-parallelism is trivial.
 * This is critical for decode (M=1) where a 1D grid of M blocks wastes 81 of 82 SMs.
 */
__global__ void fused_swiglu_quantize_blockwise_kernel(
    const float *__restrict__ gate,         // [M × K]
    const float *__restrict__ up,           // [M × K]
    int8_t *__restrict__ A_int8,            // [M × K] output
    float *__restrict__ scales_A_blockwise, // [M × num_blocks] output
    int M, int K)
{
    const int row = blockIdx.y;
    if (row >= M)
        return;

    constexpr int BLOCK_SIZE = 32;
    const int num_blocks = K / BLOCK_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    // Global warp index across all blocks in this row
    const int global_warp = blockIdx.x * num_warps + warp_id;
    const int total_warps = gridDim.x * num_warps;

    const float *row_gate = gate + row * K;
    const float *row_up = up + row * K;
    int8_t *row_int8 = A_int8 + row * K;
    float *row_scales = scales_A_blockwise + row * num_blocks;

    for (int b = global_warp; b < num_blocks; b += total_warps)
    {
        const int k_start = b * BLOCK_SIZE;

        // Coalesced load: each lane reads one element from gate and up
        float g = row_gate[k_start + lane];
        float u = row_up[k_start + lane];

        // SwiGLU: silu(gate) * up
        float val = (g / (1.0f + expf(-g))) * u;

        // Warp-level max_abs reduction via shuffle
        float abs_val = fabsf(val);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, mask));
        }

        float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        if (lane == 0)
            row_scales[b] = scale;

        // Quantize and coalesced write
        float qval = val * inv_scale;
        row_int8[k_start + lane] = static_cast<int8_t>(
            rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
    }
}

// =========================================================================
// 2. Fused Residual Add + RMSNorm
// =========================================================================

/**
 * @brief Fused residual add + RMSNorm in a single kernel
 *
 * Computes:
 *   hidden[row] = input[row] + residual[row]          (residual add)
 *   output[row] = rmsnorm(hidden[row], gamma, eps)     (normalize)
 *
 * Writes TWO outputs:
 *   - residual_output: the sum (for skip connections in next residual add)
 *   - norm_output: the normalized result (feeds into next GEMM)
 *
 * Eliminates the intermediate write of residual sum to global memory
 * followed by a read in the RMSNorm kernel.
 *
 * Memory savings: 2 × rows × cols × sizeof(float) per fusion point
 *
 * Grid:  (rows, 1, 1) — one CUDA block per row
 * Block: (256, 1, 1) — shared memory for reduction
 */
__global__ void fused_residual_rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    const float *__restrict__ gamma,
    float *__restrict__ residual_output, // hidden state (sum) for skip connections
    float *__restrict__ norm_output,     // normalized output for next GEMM
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float *row_input = input + row * cols;
    const float *row_residual = residual + row * cols;
    float *row_res_out = residual_output + row * cols;
    float *row_norm_out = norm_output + row * cols;

    // Pass 1: Add residual, compute sum of squares, and write residual output
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = row_input[i] + row_residual[i];
        row_res_out[i] = sum; // Write updated hidden state
        sum_sq += sum * sum;
    }

    // Reduce sum of squares
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    // Pass 2: Read back sum, normalize, and write normalized output
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = row_res_out[i] * inv_rms * gamma[i];
    }
}

/**
 * @brief Fused residual add + RMSNorm with BF16 I/O
 */
__global__ void fused_residual_rmsnorm_bf16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ residual_output,
    uint16_t *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = bf16_to_float(row_input[i]) + bf16_to_float(row_residual[i]);
        row_res_out[i] = float_to_bf16(sum);
        sum_sq += sum * sum;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    for (int i = tid; i < cols; i += stride)
    {
        float sum = bf16_to_float(row_res_out[i]);
        row_norm_out[i] = float_to_bf16(sum * inv_rms * gamma[i]);
    }
}

/**
 * @brief Fused residual add + RMSNorm with FP16 I/O
 */
__global__ void fused_residual_rmsnorm_fp16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ residual_output,
    uint16_t *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = fp16_to_float(row_input[i]) + fp16_to_float(row_residual[i]);
        row_res_out[i] = float_to_fp16(sum);
        sum_sq += sum * sum;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    for (int i = tid; i < cols; i += stride)
    {
        float sum = fp16_to_float(row_res_out[i]);
        row_norm_out[i] = float_to_fp16(sum * inv_rms * gamma[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    // =====================================================================
    // Fused SwiGLU + Blockwise Quantization
    // =====================================================================

    bool cudaOps_fused_swiglu_quantize_blockwise(
        const float *gate,
        const float *up,
        int8_t *A_int8,
        float *scales_A_blockwise,
        int M, int K,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);

        // 2D grid: blockIdx.x distributes K-blocks, blockIdx.y distributes rows.
        // For decode (M=1), this ensures all SMs participate instead of just 1.
        constexpr int NUM_WARPS = 4;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32; // 128 threads
        const int num_k_blocks = K / 32;
        // Target enough blocks per row to fill the GPU (~2 waves over all SMs)
        int grid_x = (num_k_blocks + NUM_WARPS - 1) / NUM_WARPS;
        if (grid_x > 256)
            grid_x = 256; // Cap to avoid excessive blocks for prefill

        dim3 grid(grid_x, M);
        dim3 block(BLOCK_SIZE);

        fused_swiglu_quantize_blockwise_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            gate, up, A_int8, scales_A_blockwise, M, K);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAFusedOps] fused_swiglu_quantize_blockwise launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // =====================================================================
    // Fused Residual Add + RMSNorm
    // =====================================================================

    bool cudaOps_fused_residual_rmsnorm_fp32(
        const float *input,
        const float *residual,
        const float *gamma,
        float *residual_output,
        float *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        fused_residual_rmsnorm_fp32_kernel<<<num_blocks, threads_per_block, shared_mem_size,
                                             static_cast<cudaStream_t>(stream)>>>(
            input, residual, gamma, residual_output, norm_output, cols, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAFusedOps] fused_residual_rmsnorm_fp32 launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_fused_residual_rmsnorm_bf16(
        const uint16_t *input,
        const uint16_t *residual,
        const float *gamma,
        uint16_t *residual_output,
        uint16_t *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        fused_residual_rmsnorm_bf16_kernel<<<num_blocks, threads_per_block, shared_mem_size,
                                             static_cast<cudaStream_t>(stream)>>>(
            input, residual, gamma, residual_output, norm_output, cols, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAFusedOps] fused_residual_rmsnorm_bf16 launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_fused_residual_rmsnorm_fp16(
        const uint16_t *input,
        const uint16_t *residual,
        const float *gamma,
        uint16_t *residual_output,
        uint16_t *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        fused_residual_rmsnorm_fp16_kernel<<<num_blocks, threads_per_block, shared_mem_size,
                                             static_cast<cudaStream_t>(stream)>>>(
            input, residual, gamma, residual_output, norm_output, cols, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAFusedOps] fused_residual_rmsnorm_fp16 launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
