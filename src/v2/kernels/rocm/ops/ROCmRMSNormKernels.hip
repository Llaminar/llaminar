/**
 * @file ROCmRMSNormKernels.hip
 * @brief ROCm/HIP RMSNorm kernel implementations
 *
 * Provides FP32, BF16, and FP16 RMS normalization kernels for AMD GPUs.
 * Uses shared memory for parallel reduction of sum-of-squares.
 *
 * RMSNorm formula: output[i] = (input[i] / RMS(input)) * gamma[i]
 * where RMS(input) = sqrt(mean(input²) + epsilon)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"

// BF16 type support
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
#else
typedef struct { uint16_t x; } hip_bfloat16;
#endif

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

// ============================================================================
// FP32 RMSNorm Kernel
// ============================================================================

__global__ void rocm_rmsnorm_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;           // One block per row
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = row_input[i];
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Compute RMS scale factor
    float rms = sqrtf(sdata[0] / cols + epsilon);
    float inv_rms = 1.0f / rms;

    // Step 2: Normalize and scale by gamma
    for (int i = tid; i < cols; i += stride)
    {
        row_output[i] = row_input[i] * inv_rms * gamma[i];
    }
}

// ============================================================================
// BF16 RMSNorm Kernel (FP32 compute, BF16 I/O)
// ============================================================================

__global__ void rocm_rmsnorm_bf16_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ gamma,
    uint16_t* __restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t* row_input = input + row * cols;
    uint16_t* row_output = output + row * cols;

    // Step 1: Compute sum of squares in FP32
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = bf16_to_float(row_input[i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Compute RMS scale factor
    float rms = sqrtf(sdata[0] / cols + epsilon);
    float inv_rms = 1.0f / rms;

    // Step 2: Normalize and scale by gamma, convert back to BF16
    for (int i = tid; i < cols; i += stride)
    {
        float val = bf16_to_float(row_input[i]);
        float result = val * inv_rms * gamma[i];
        row_output[i] = float_to_bf16(result);
    }
}

// ============================================================================
// FP16 RMSNorm Kernel (FP32 compute, FP16 I/O)
// ============================================================================

__global__ void rocm_rmsnorm_fp16_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ gamma,
    uint16_t* __restrict__ output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t* row_input = input + row * cols;
    uint16_t* row_output = output + row * cols;

    // Step 1: Compute sum of squares in FP32
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float val = fp16_to_float(row_input[i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Compute RMS scale factor
    float rms = sqrtf(sdata[0] / cols + epsilon);
    float inv_rms = 1.0f / rms;

    // Step 2: Normalize and scale by gamma, convert back to FP16
    for (int i = tid; i < cols; i += stride)
    {
        float val = fp16_to_float(row_input[i]);
        float result = val * inv_rms * gamma[i];
        row_output[i] = float_to_fp16(result);
    }
}

} // anonymous namespace

// ============================================================================
// Extern "C" Wrappers
// ============================================================================

extern "C" bool hipOps_rmsnorm_fp32(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon,
    int device_idx,
    void* stream)
{
    if (!input || !gamma || !output || rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp32: Invalid parameters\n");
        return false;
    }

    hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx));
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp32: hipSetDevice failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Launch configuration: one block per row
    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Clear any stale error from previous operations (e.g., heterogeneous backend tests)
    (void)hipGetLastError();

    rocm_rmsnorm_fp32_kernel<<<grid, block, shared_mem_size, hip_stream>>>(
        input, gamma, output, cols, epsilon);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp32: kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    // No sync needed - coherence system handles sync when data is read
    return true;
}

extern "C" bool hipOps_rmsnorm_bf16(
    const uint16_t* input,
    const float* gamma,
    uint16_t* output,
    int rows,
    int cols,
    float epsilon,
    int device_idx,
    void* stream)
{
    if (!input || !gamma || !output || rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "hipOps_rmsnorm_bf16: Invalid parameters\n");
        return false;
    }

    hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx));
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_bf16: hipSetDevice failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Clear any stale error from previous operations
    (void)hipGetLastError();

    rocm_rmsnorm_bf16_kernel<<<grid, block, shared_mem_size, hip_stream>>>(
        input, gamma, output, cols, epsilon);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_bf16: kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    // No sync needed - coherence system handles sync when data is read
    return true;
}

extern "C" bool hipOps_rmsnorm_fp16(
    const uint16_t* input,
    const float* gamma,
    uint16_t* output,
    int rows,
    int cols,
    float epsilon,
    int device_idx,
    void* stream)
{
    if (!input || !gamma || !output || rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp16: Invalid parameters\n");
        return false;
    }

    hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx));
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp16: hipSetDevice failed: %s\n", hipGetErrorString(err));
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Clear any stale error from previous operations
    (void)hipGetLastError();

    rocm_rmsnorm_fp16_kernel<<<grid, block, shared_mem_size, hip_stream>>>(
        input, gamma, output, cols, epsilon);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "hipOps_rmsnorm_fp16: kernel launch failed: %s\n", hipGetErrorString(err));
        return false;
    }

    // No sync needed - coherence system handles sync when data is read
    return true;
}
