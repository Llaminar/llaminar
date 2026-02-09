/**
 * @file ROCmSwiGLUKernels.hip
 * @brief HIP SwiGLU kernel implementations for ROCm
 * @author Llaminar Team
 *
 * Contains FP32, BF16, and FP16 SwiGLU kernels with extern "C" wrapper functions.
 * SwiGLU computes: silu(gate) * up = gate / (1 + exp(-gate)) * up
 *
 * This follows the same pattern as the CUDA SwiGLU kernels, using HIP for
 * AMD GPU compatibility.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace
{

// =========================================================================
// BF16 Conversion Helpers (HIP compatible)
// =========================================================================

__device__ __forceinline__ float bf16_to_float(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__device__ __forceinline__ uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

// =========================================================================
// FP16 Conversion Helpers (using HIP intrinsics)
// =========================================================================

__device__ __forceinline__ float fp16_to_float(uint16_t fp16)
{
    __half h;
    memcpy(&h, &fp16, sizeof(__half));
    return __half2float(h);
}

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    uint16_t result;
    memcpy(&result, &h, sizeof(uint16_t));
    return result;
}

// =========================================================================
// Activation Functions
// =========================================================================

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

} // anonymous namespace

// =========================================================================
// SwiGLU HIP Kernels
// =========================================================================

/**
 * @brief FP32 SwiGLU kernel - element-wise silu(gate) * up
 */
__global__ void rocm_swiglu_fp32_kernel(
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
__global__ void rocm_swiglu_bf16_kernel(
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
__global__ void rocm_swiglu_fp16_kernel(
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

    bool hipOps_swiglu_fp32(
        const float *gate,
        const float *up,
        float *output,
        int size,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        // Clear any stale error from previous operations
        (void)hipGetLastError();

        hipLaunchKernelGGL(rocm_swiglu_fp32_kernel, dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
                           gate, up, output, size);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP SwiGLU FP32 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_swiglu_bf16(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        // Clear any stale error from previous operations
        (void)hipGetLastError();

        hipLaunchKernelGGL(rocm_swiglu_bf16_kernel, dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
                           gate, up, output, size);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP SwiGLU BF16 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_swiglu_fp16(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        // Clear any stale error from previous operations
        (void)hipGetLastError();

        hipLaunchKernelGGL(rocm_swiglu_fp16_kernel, dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
                           gate, up, output, size);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP SwiGLU FP16 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
