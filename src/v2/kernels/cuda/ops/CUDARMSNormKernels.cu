/**
 * @file CUDARMSNormKernels.cu
 * @brief CUDA RMSNorm kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 RMSNorm kernels with extern "C" wrapper functions.
 *
 * Optimized design:
 * - Register-cached input values (Pass 2 reads from registers, not global memory)
 * - Warp shuffle reduction + cross-warp shared memory (2 barriers vs 8+)
 * - 1024 threads for cols > 256 (better SM utilization)
 * - No dynamic shared memory allocation
 */

#include "CUDAHelpers.cuh"
#include <cmath>
#include <cstdio>

// =========================================================================
// RMSNorm CUDA Kernels
// =========================================================================

/**
 * @brief FP32 RMSNorm kernel - one block per row
 *
 * Register-cached design: Pass 1 caches input values in registers while
 * computing sum-of-squares. Pass 2 reads from registers (no global re-read).
 * Uses warp shuffle + cross-warp shared memory reduction (2 barriers vs 8+).
 */
__global__ void rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gamma,
    float *__restrict__ output,
    int cols,
    float epsilon)
{
    __shared__ float warp_sums[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const float *row_input = input + row * cols;
    float *row_output = output + row * cols;

    // Pass 1: Load input values into registers, accumulate sum-of-squares
    constexpr int kMaxPerThread = 16;
    float local_vals[kMaxPerThread];
    float sum_sq = 0.0f;
    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        float val = row_input[i];
        local_vals[idx++] = val;
        sum_sq += val * val;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid < 32)
    {
        float val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (tid == 0)
            warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    // Pass 2: Normalize from register-cached values (no global re-read)
    idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_output[i] = local_vals[idx++] * inv_rms * gamma[i];
    }
}

/**
 * @brief BF16 RMSNorm kernel - FP32 computation with BF16 I/O
 *
 * Register-cached design with warp shuffle reduction.
 */
__global__ void rmsnorm_bf16_kernel(
    const uint16_t *__restrict__ input,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ output,
    int cols,
    float epsilon)
{
    __shared__ float warp_sums[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const uint16_t *row_input = input + row * cols;
    uint16_t *row_output = output + row * cols;

    // Pass 1: Load + convert to FP32, cache in registers, accumulate sum-of-squares
    constexpr int kMaxPerThread = 16;
    float local_vals[kMaxPerThread];
    float sum_sq = 0.0f;
    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        float val = bf16_to_float(row_input[i]);
        local_vals[idx++] = val;
        sum_sq += val * val;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid < 32)
    {
        float val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (tid == 0)
            warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    // Pass 2: Normalize from register-cached values (no global re-read)
    idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_output[i] = float_to_bf16(local_vals[idx++] * inv_rms * gamma[i]);
    }
}

/**
 * @brief FP16 RMSNorm kernel - FP32 computation with FP16 I/O
 *
 * Register-cached design with warp shuffle reduction.
 */
__global__ void rmsnorm_fp16_kernel(
    const uint16_t *__restrict__ input,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ output,
    int cols,
    float epsilon)
{
    __shared__ float warp_sums[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const uint16_t *row_input = input + row * cols;
    uint16_t *row_output = output + row * cols;

    // Pass 1: Load + convert to FP32, cache in registers, accumulate sum-of-squares
    constexpr int kMaxPerThread = 16;
    float local_vals[kMaxPerThread];
    float sum_sq = 0.0f;
    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        float val = fp16_to_float(row_input[i]);
        local_vals[idx++] = val;
        sum_sq += val * val;
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid < 32)
    {
        float val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (tid == 0)
            warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    // Pass 2: Normalize from register-cached values (no global re-read)
    idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_output[i] = float_to_fp16(local_vals[idx++] * inv_rms * gamma[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaOps_rmsnorm_fp32(
        const float *input,
        const float *gamma,
        float *output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        rmsnorm_fp32_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gamma, output, cols, epsilon);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RMSNorm FP32 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_rmsnorm_bf16(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        rmsnorm_bf16_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gamma, output, cols, epsilon);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RMSNorm BF16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_rmsnorm_fp16(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        rmsnorm_fp16_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gamma, output, cols, epsilon);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RMSNorm FP16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
