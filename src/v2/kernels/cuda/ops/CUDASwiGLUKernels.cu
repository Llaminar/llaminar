/**
 * @file CUDASwiGLUKernels.cu
 * @brief CUDA SwiGLU kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 SwiGLU kernels with extern "C" wrapper functions.
 * SwiGLU computes: silu(gate) * up = gate / (1 + exp(-gate)) * up
 */

#include "CUDAHelpers.cuh"
#include <cstdio>

// =========================================================================
// SwiGLU CUDA Kernels
// =========================================================================

/**
 * @brief FP32 SwiGLU kernel - element-wise silu(gate) * up
 */
__global__ void swiglu_fp32_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = silu(gate[idx]) * up[idx];
    }
}

/**
 * @brief BF16 SwiGLU kernel - FP32 computation with BF16 I/O
 */
__global__ void swiglu_bf16_kernel(
    const uint16_t *__restrict__ gate,
    const uint16_t *__restrict__ up,
    uint16_t *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float g = bf16_to_float(gate[idx]);
        float u = bf16_to_float(up[idx]);
        float result = silu(g) * u;
        output[idx] = float_to_bf16(result);
    }
}

/**
 * @brief FP16 SwiGLU kernel - FP32 computation with FP16 I/O
 */
__global__ void swiglu_fp16_kernel(
    const uint16_t *__restrict__ gate,
    const uint16_t *__restrict__ up,
    uint16_t *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float g = fp16_to_float(gate[idx]);
        float u = fp16_to_float(up[idx]);
        float result = silu(g) * u;
        output[idx] = float_to_fp16(result);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaOps_swiglu_fp32(
        const float *gate,
        const float *up,
        float *output,
        int size,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        swiglu_fp32_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, output, size);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA SwiGLU FP32 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_swiglu_bf16(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        swiglu_bf16_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, output, size);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA SwiGLU BF16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_swiglu_fp16(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx,
        void *stream)
    {
        // Always set device — stream carries device context but kernel launch
        // uses the runtime's current-device for PTX code lookup.
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        swiglu_fp16_kernel<<<num_blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, output, size);

        (void)cudaGetLastError(); // Clear stale errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA SwiGLU FP16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
