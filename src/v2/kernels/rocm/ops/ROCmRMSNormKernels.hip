/**
 * @file ROCmRMSNormKernels.hip
 * @brief ROCm/HIP RMSNorm kernel implementations — V2 optimized for gfx906
 *
 * Provides FP32, BF16, and FP16 RMS normalization kernels for AMD GPUs.
 *
 * V2 optimizations vs V1 baseline:
 *   1. Vectorized loads — float4 (FP32) / uint32 packed pairs (BF16/FP16)
 *   2. Warp shuffle reduction via ds_bpermute — replaces shared memory tree
 *      reduction.  Only 2 barriers (cross-warp) instead of 8+ in V1.
 *   3. Multiply by 1/cols instead of full-precision division
 *
 * ISA summary (gfx906, FP32 kernel):
 *   VGPR: 18 (vs 14 in V1), SGPR: 28, LDS: 16 bytes static
 *   No scratch memory, no spills
 *   global_load_dwordx4 for vectorized reads
 *   ds_bpermute_b32 for warp shuffles
 *   2 × s_barrier (cross-warp reduction)
 *
 * Performance note: At decode M=1, cols=3584 the kernel reads ~28 KB per call.
 * The 3.4µs latency is dominated by HIP kernel launch overhead (~2-3µs),
 * not memory bandwidth or compute.  Further improvement requires kernel fusion
 * with adjacent stages (e.g. fusing RMSNorm + GEMM).
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

/**
 * Pack 2 FP32 values into a uint32 of 2 packed FP16 values.
 * Maps to v_cvt_pkrtz_f16_f32 on GFX906 (single VALU instruction).
 */
__device__ __forceinline__ uint32_t pack_fp16_pair(float v0, float v1)
{
    float2 pair = make_float2(v0, v1);
    __half2 h2 = __float22half2_rn(pair);
    return *reinterpret_cast<uint32_t*>(&h2);
}

// ============================================================================
// Warp Shuffle Reduction (gfx906 wavefront = 64 lanes)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    // 6 steps for wavefront size 64: 32, 16, 8, 4, 2, 1
    // Each __shfl_xor maps to ds_swizzle or v_readlane — no barrier needed
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// ============================================================================
// FP32 RMSNorm Kernel — V2 Optimized
// ============================================================================

__global__ void rocm_rmsnorm_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int cols,
    float epsilon)
{
    // Shared memory: only need one float per warp for cross-warp reduction
    // (4 warps for 256 threads on gfx906 with wf=64)
    __shared__ float warp_sums[4];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;      // tid / 64 (gfx906 wavefront = 64)
    const int lane_id = tid & 63;      // tid % 64
    const int stride = blockDim.x;     // 256

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // ---- Pass 1: Compute sum of squares with vectorized float4 loads ----
    float sum_sq = 0.0f;

    const int cols4 = cols >> 2;  // Number of float4 elements in row
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);

    for (int i = tid; i < cols4; i += stride)
    {
        float4 v = row_input4[i];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle tail elements (cols not divisible by 4)
    const int tail_start = cols4 << 2;
    for (int i = tail_start + tid; i < cols; i += stride)
    {
        float val = row_input[i];
        sum_sq += val * val;
    }

    // ---- Reduction: warp shuffle + minimal shared memory ----
    // Step 1: Intra-warp reduction via shuffle (6 steps, no barriers)
    sum_sq = warp_reduce_sum(sum_sq);

    // Step 2: Cross-warp reduction via shared memory (1 write + 1 barrier + 1 read)
    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;

    __syncthreads();

    // Final reduction: lane 0 of warp 0 sums the 4 warp partial sums
    float total_sum;
    if (tid < 4)
    {
        total_sum = warp_sums[tid];
        // Reduce 4 values (only need 2 shuffle steps)
        total_sum += __shfl_xor(total_sum, 2);
        total_sum += __shfl_xor(total_sum, 1);
    }

    // Broadcast total_sum to all threads via shared memory
    if (tid == 0)
        warp_sums[0] = total_sum;

    __syncthreads();

    total_sum = warp_sums[0];

    // ---- Compute inv_rms: multiply by 1/cols instead of dividing ----
    // rsqrtf maps to v_rsq_f32 (4 cycles).
    // Avoids v_div_scale + v_rcp + 4×v_fma + v_div_fmas + v_div_fixup (~8 insns).
    const float inv_cols = 1.0f / static_cast<float>(cols);
    float inv_rms = rsqrtf(total_sum * inv_cols + epsilon);

    // ---- Pass 2: Re-read input, normalize and scale by gamma ----
    // Vectorized float4 loads/stores for bulk
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* row_output4 = reinterpret_cast<float4*>(row_output);

    for (int i = tid; i < cols4; i += stride)
    {
        float4 v = row_input4[i];
        float4 g = gamma4[i];
        float4 result;
        result.x = v.x * inv_rms * g.x;
        result.y = v.y * inv_rms * g.y;
        result.z = v.z * inv_rms * g.z;
        result.w = v.w * inv_rms * g.w;
        row_output4[i] = result;
    }

    // Handle tail elements
    for (int i = tail_start + tid; i < cols; i += stride)
    {
        row_output[i] = row_input[i] * inv_rms * gamma[i];
    }
}

// ============================================================================
// BF16 RMSNorm Kernel — V2 Optimized (FP32 compute, BF16 I/O)
// ============================================================================

__global__ void rocm_rmsnorm_bf16_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ gamma,
    uint16_t* __restrict__ output,
    int cols,
    float epsilon)
{
    __shared__ float warp_sums[4];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;
    const int stride = blockDim.x;

    const uint16_t* row_input = input + row * cols;
    uint16_t* row_output = output + row * cols;

    // ---- Pass 1: Vectorized BF16 loads, compute sum of squares ----
    float sum_sq = 0.0f;
    const int cols2 = cols >> 1;
    const uint32_t* row_input_u32 = reinterpret_cast<const uint32_t*>(row_input);

    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t packed = row_input_u32[i];
        float v0 = bf16_to_float(static_cast<uint16_t>(packed & 0xFFFF));
        float v1 = bf16_to_float(static_cast<uint16_t>(packed >> 16));
        sum_sq += v0 * v0 + v1 * v1;
    }

    if ((cols & 1) && (tid == 0))
    {
        float val = bf16_to_float(row_input[cols - 1]);
        sum_sq += val * val;
    }

    // ---- Warp shuffle + cross-warp reduction ----
    sum_sq = warp_reduce_sum(sum_sq);

    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    float total_sum;
    if (tid < 4)
    {
        total_sum = warp_sums[tid];
        total_sum += __shfl_xor(total_sum, 2);
        total_sum += __shfl_xor(total_sum, 1);
    }
    if (tid == 0)
        warp_sums[0] = total_sum;
    __syncthreads();

    total_sum = warp_sums[0];

    const float inv_cols = 1.0f / static_cast<float>(cols);
    float inv_rms = rsqrtf(total_sum * inv_cols + epsilon);

    // ---- Pass 2: Re-read input, normalize, scale, convert back to BF16 ----
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t packed = row_input_u32[i];
        int base = i << 1;
        float v0 = bf16_to_float(static_cast<uint16_t>(packed & 0xFFFF));
        float v1 = bf16_to_float(static_cast<uint16_t>(packed >> 16));
        v0 = v0 * inv_rms * gamma[base + 0];
        v1 = v1 * inv_rms * gamma[base + 1];
        uint32_t out_packed = static_cast<uint32_t>(float_to_bf16(v0)) |
                              (static_cast<uint32_t>(float_to_bf16(v1)) << 16);
        reinterpret_cast<uint32_t*>(row_output)[i] = out_packed;
    }

    if ((cols & 1) && (tid == 0))
    {
        float val = bf16_to_float(row_input[cols - 1]);
        row_output[cols - 1] = float_to_bf16(val * inv_rms * gamma[cols - 1]);
    }
}

// ============================================================================
// FP16 RMSNorm Kernel — V2 Optimized (FP32 compute, FP16 I/O)
// ============================================================================

__global__ void rocm_rmsnorm_fp16_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ gamma,
    uint16_t* __restrict__ output,
    int cols,
    float epsilon)
{
    __shared__ float warp_sums[4];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;
    const int stride = blockDim.x;

    const uint16_t* row_input = input + row * cols;
    uint16_t* row_output = output + row * cols;

    // ---- Pass 1: Vectorized FP16 loads, compute sum of squares ----
    float sum_sq = 0.0f;
    const int cols2 = cols >> 1;
    const uint32_t* row_input_u32 = reinterpret_cast<const uint32_t*>(row_input);

    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t packed = row_input_u32[i];
        float v0 = fp16_to_float(static_cast<uint16_t>(packed & 0xFFFF));
        float v1 = fp16_to_float(static_cast<uint16_t>(packed >> 16));
        sum_sq += v0 * v0 + v1 * v1;
    }

    if ((cols & 1) && (tid == 0))
    {
        float val = fp16_to_float(row_input[cols - 1]);
        sum_sq += val * val;
    }

    // ---- Warp shuffle + cross-warp reduction ----
    sum_sq = warp_reduce_sum(sum_sq);

    if (lane_id == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    float total_sum;
    if (tid < 4)
    {
        total_sum = warp_sums[tid];
        total_sum += __shfl_xor(total_sum, 2);
        total_sum += __shfl_xor(total_sum, 1);
    }
    if (tid == 0)
        warp_sums[0] = total_sum;
    __syncthreads();

    total_sum = warp_sums[0];

    const float inv_cols = 1.0f / static_cast<float>(cols);
    float inv_rms = rsqrtf(total_sum * inv_cols + epsilon);

    // ---- Pass 2: Re-read input, normalize, scale, convert back to FP16 ----
    // Output packing uses v_cvt_pkrtz_f16_f32 (single VALU op for 2 FP16 values)
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t packed = row_input_u32[i];
        int base = i << 1;
        float v0 = fp16_to_float(static_cast<uint16_t>(packed & 0xFFFF));
        float v1 = fp16_to_float(static_cast<uint16_t>(packed >> 16));
        v0 = v0 * inv_rms * gamma[base + 0];
        v1 = v1 * inv_rms * gamma[base + 1];
        reinterpret_cast<uint32_t*>(row_output)[i] = pack_fp16_pair(v0, v1);
    }

    if ((cols & 1) && (tid == 0))
    {
        float val = fp16_to_float(row_input[cols - 1]);
        row_output[cols - 1] = float_to_fp16(val * inv_rms * gamma[cols - 1]);
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

    // Launch configuration: one block per row, 256 threads
    // V2: shared memory is static (4 floats for warp_sums), not dynamic
    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);

    // Clear any stale error from previous operations (e.g., heterogeneous backend tests)
    (void)hipGetLastError();

    rocm_rmsnorm_fp32_kernel<<<grid, block, 0, hip_stream>>>(
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

    // V2: shared memory is static (4 floats for warp_sums), not dynamic
    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);

    // Clear any stale error from previous operations
    (void)hipGetLastError();

    rocm_rmsnorm_bf16_kernel<<<grid, block, 0, hip_stream>>>(
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

    // V2: shared memory is static (4 floats for warp_sums), not dynamic
    constexpr int threads_per_block = 256;
    dim3 grid(rows);
    dim3 block(threads_per_block);

    // Clear any stale error from previous operations
    (void)hipGetLastError();

    rocm_rmsnorm_fp16_kernel<<<grid, block, 0, hip_stream>>>(
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
