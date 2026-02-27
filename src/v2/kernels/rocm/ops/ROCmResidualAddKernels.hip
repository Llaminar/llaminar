/**
 * @file ROCmResidualAddKernels.hip
 * @brief ROCm/HIP Residual Add kernel implementations
 *
 * Provides FP32, BF16, and FP16 element-wise addition kernels for AMD GPUs.
 * Used in transformer residual connections: output = input + residual
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"

namespace {

// ============================================================================
// BF16 Conversion Helpers
// ============================================================================

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
    // Round to nearest even (add 0x7FFF + bit 16)
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

// ============================================================================
// FP16 Conversion Helpers
// ============================================================================

__device__ __forceinline__ float fp16_to_float(uint16_t fp16)
{
    return __half2float(*reinterpret_cast<const __half*>(&fp16));
}

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

} // anonymous namespace

// ============================================================================
// FP32 Residual Add Kernel — V2 with float4 vectorization
// ============================================================================

__global__ void rocm_residual_add_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int size)
{
    const int size4 = size >> 2;
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx < size4)
    {
        const float4* in4  = reinterpret_cast<const float4*>(input);
        const float4* res4 = reinterpret_cast<const float4*>(residual);
        float4* out4       = reinterpret_cast<float4*>(output);

        float4 a = in4[vec_idx];
        float4 b = res4[vec_idx];
        float4 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        r.z = a.z + b.z;
        r.w = a.w + b.w;
        out4[vec_idx] = r;
    }

    // Scalar tail
    const int tail_start = size4 << 2;
    const int tail_idx = tail_start + (blockIdx.x * blockDim.x + threadIdx.x);
    if (tail_idx >= tail_start && tail_idx < size && vec_idx >= size4)
    {
        output[tail_idx] = input[tail_idx] + residual[tail_idx];
    }
}

// ============================================================================
// BF16 Residual Add Kernel (FP32 computation with BF16 I/O)
// ============================================================================

__global__ void rocm_residual_add_bf16_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float in_f = bf16_to_float(input[idx]);
        float res_f = bf16_to_float(residual[idx]);
        output[idx] = float_to_bf16(in_f + res_f);
    }
}

// ============================================================================
// FP16 Residual Add Kernel (FP32 computation with FP16 I/O)
// ============================================================================

__global__ void rocm_residual_add_fp16_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float in_f = fp16_to_float(input[idx]);
        float res_f = fp16_to_float(residual[idx]);
        output[idx] = float_to_fp16(in_f + res_f);
    }
}

// ============================================================================
// Extern "C" Wrapper Functions
// ============================================================================

extern "C"
{

bool rocmOps_residual_add_fp32(
    const float* input,
    const float* residual,
    float* output,
    int size,
    int device_idx,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int threads_per_block = 256;
    // Grid sized for float4 vectorized elements (+ tail handled internally)
    int size4 = size >> 2;
    int num_blocks = (size4 + threads_per_block - 1) / threads_per_block;
    if (num_blocks == 0 && size > 0) num_blocks = 1;

    hipLaunchKernelGGL(
        rocm_residual_add_fp32_kernel,
        dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
        input, residual, output, size);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Residual Add FP32 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_residual_add_bf16(
    const uint16_t* input,
    const uint16_t* residual,
    uint16_t* output,
    int size,
    int device_idx,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        rocm_residual_add_bf16_kernel,
        dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
        input, residual, output, size);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Residual Add BF16 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_residual_add_fp16(
    const uint16_t* input,
    const uint16_t* residual,
    uint16_t* output,
    int size,
    int device_idx,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        rocm_residual_add_fp16_kernel,
        dim3(num_blocks), dim3(threads_per_block), 0, static_cast<hipStream_t>(stream),
        input, residual, output, size);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Residual Add FP16 kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
