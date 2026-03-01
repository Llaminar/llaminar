/**
 * @file CUDARMSNormKernels.cu
 * @brief CUDA RMSNorm kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 RMSNorm kernels with extern "C" wrapper functions.
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
 * Uses shared memory for parallel reduction to compute RMS.
 * Each block handles one row: loads row → computes sum of squares →
 * reduce → normalize and scale.
 */
__global__ void rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gamma,
    float *__restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float *row_input = input + row * cols;
    float *row_output = output + row * cols;

    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = row_input[i];
        sum_sq += val * val;
    }

    // Store in shared memory
    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Compute inverse RMS scale factor
    // rsqrtf() maps to MUFU.RSQ (single instruction) instead of
    // sqrtf() (MUFU.SQRT) + rcp (MUFU.RCP)
    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    // Step 2: Normalize and scale by gamma
    for (int i = tid; i < cols; i += stride)
    {
        row_output[i] = row_input[i] * inv_rms * gamma[i];
    }
}

/**
 * @brief BF16 RMSNorm kernel - FP32 computation with BF16 I/O
 */
__global__ void rmsnorm_bf16_kernel(
    const uint16_t *__restrict__ input,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    uint16_t *row_output = output + row * cols;

    // Step 1: Compute sum of squares (in FP32)
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = bf16_to_float(row_input[i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    // Step 2: Normalize, scale, and convert back to BF16
    for (int i = tid; i < cols; i += stride)
    {
        float val = bf16_to_float(row_input[i]);
        float result = val * inv_rms * gamma[i];
        row_output[i] = float_to_bf16(result);
    }
}

/**
 * @brief FP16 RMSNorm kernel - FP32 computation with FP16 I/O
 */
__global__ void rmsnorm_fp16_kernel(
    const uint16_t *__restrict__ input,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    uint16_t *row_output = output + row * cols;

    // Step 1: Compute sum of squares (in FP32)
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = fp16_to_float(row_input[i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    // Step 2: Normalize, scale, and convert back to FP16
    for (int i = tid; i < cols; i += stride)
    {
        float val = fp16_to_float(row_input[i]);
        float result = val * inv_rms * gamma[i];
        row_output[i] = float_to_fp16(result);
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

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        rmsnorm_fp32_kernel<<<num_blocks, threads_per_block, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(
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

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        rmsnorm_bf16_kernel<<<num_blocks, threads_per_block, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(
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

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        rmsnorm_fp16_kernel<<<num_blocks, threads_per_block, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(
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
