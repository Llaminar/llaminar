/**
 * @file ROCmFusedOpsKernels.hip
 * @brief Fused HIP element-wise kernels for LLM inference optimization (ROCm port)
 *
 * ROCm/HIP port of CUDAFusedOpsKernels.cu. Contains fused operations that
 * eliminate intermediate memory round-trips:
 * 1. Fused SwiGLU + Blockwise INT8 Quantization
 * 2. Fused Residual Add + RMSNorm (FP32, BF16, FP16)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../gemm/ROCmFastpackQuant.h"

namespace {

// =========================================================================
// BF16/FP16 Conversion Helpers
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
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

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

// =========================================================================
// 1. Fused SwiGLU + Blockwise INT8 Quantization
// =========================================================================

__global__ void rocm_fused_swiglu_quantize_blockwise_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    int8_t *__restrict__ A_int8,
    float *__restrict__ scales_A_blockwise,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M)
        return;

    constexpr int BLOCK_SIZE = 32;
    const int num_blocks = K / BLOCK_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int NUM_WARPS = 8;

    const float *row_gate = gate + row * K;
    const float *row_up = up + row * K;
    int8_t *row_int8 = A_int8 + row * K;
    float *row_scales = scales_A_blockwise + row * num_blocks;

    for (int b = warp_id; b < num_blocks; b += NUM_WARPS)
    {
        const int k_start = b * BLOCK_SIZE;

        float g = row_gate[k_start + lane];
        float u = row_up[k_start + lane];

        // SwiGLU: silu(gate) * up
        float val = (g / (1.0f + expf(-g))) * u;

        // Warp-level max_abs reduction via shuffle
        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));
        }

        float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        if (lane == 0)
            row_scales[b] = scale;

        // GFX906 fastpack: lanes 0,4,8,...,28 gather 4 neighbours via
        // shuffle, pack with v_cvt_pk_u8_f32, and emit one dword store.
        const int group_base = lane & ~3;  // 0,4,8,...,28
        float v0 = __shfl(val, group_base + 0);
        float v1 = __shfl(val, group_base + 1);
        float v2 = __shfl(val, group_base + 2);
        float v3 = __shfl(val, group_base + 3);
        if ((lane & 3) == 0)
        {
            unsigned pack = llaminar2::rocm::fastpack_q8_4x(v0, v1, v2, v3, inv_scale);
            *reinterpret_cast<uint32_t*>(row_int8 + k_start + lane) = pack;
        }
    }
}

// =========================================================================
// 2. Fused Residual Add + RMSNorm
// =========================================================================

__global__ void rocm_fused_residual_rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    const float *__restrict__ gamma,
    float *__restrict__ residual_output,
    float *__restrict__ norm_output,
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

    // Pass 1: Add residual, compute sum of squares, write residual output
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = row_input[i] + row_residual[i];
        row_res_out[i] = sum;
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

    // Pass 2: Normalize and write output
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = row_res_out[i] * inv_rms * gamma[i];
    }
}

__global__ void rocm_fused_residual_rmsnorm_bf16_kernel(
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

    const uint32_t *row_in_u32 = reinterpret_cast<const uint32_t*>(input + row * cols);
    const uint32_t *row_res_u32 = reinterpret_cast<const uint32_t*>(residual + row * cols);
    uint32_t *row_res_out_u32 = reinterpret_cast<uint32_t*>(residual_output + row * cols);
    uint32_t *row_norm_out_u32 = reinterpret_cast<uint32_t*>(norm_output + row * cols);
    const int cols2 = cols >> 1;

    // Pass 1: Vectorized residual add + sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t in_packed = row_in_u32[i];
        uint32_t res_packed = row_res_u32[i];
        float s0 = bf16_to_float(static_cast<uint16_t>(in_packed & 0xFFFF))
                 + bf16_to_float(static_cast<uint16_t>(res_packed & 0xFFFF));
        float s1 = bf16_to_float(static_cast<uint16_t>(in_packed >> 16))
                 + bf16_to_float(static_cast<uint16_t>(res_packed >> 16));
        uint32_t out_packed = static_cast<uint32_t>(float_to_bf16(s0)) |
                              (static_cast<uint32_t>(float_to_bf16(s1)) << 16);
        row_res_out_u32[i] = out_packed;
        sum_sq += s0 * s0 + s1 * s1;
    }
    // Scalar tail
    if ((cols & 1) && tid == 0)
    {
        const uint16_t *row_input = input + row * cols;
        const uint16_t *row_residual = residual + row * cols;
        float s = bf16_to_float(row_input[cols - 1]) + bf16_to_float(row_residual[cols - 1]);
        residual_output[row * cols + cols - 1] = float_to_bf16(s);
        sum_sq += s * s;
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

    // Pass 2: Vectorized normalize
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t res_packed = row_res_out_u32[i];
        int base = i << 1;
        float v0 = bf16_to_float(static_cast<uint16_t>(res_packed & 0xFFFF)) * inv_rms * gamma[base + 0];
        float v1 = bf16_to_float(static_cast<uint16_t>(res_packed >> 16)) * inv_rms * gamma[base + 1];
        row_norm_out_u32[i] = static_cast<uint32_t>(float_to_bf16(v0)) |
                              (static_cast<uint32_t>(float_to_bf16(v1)) << 16);
    }
    if ((cols & 1) && tid == 0)
    {
        float v = bf16_to_float(residual_output[row * cols + cols - 1]) * inv_rms * gamma[cols - 1];
        norm_output[row * cols + cols - 1] = float_to_bf16(v);
    }
}

__global__ void rocm_fused_residual_rmsnorm_fp16_kernel(
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

    const uint32_t *row_in_u32 = reinterpret_cast<const uint32_t*>(input + row * cols);
    const uint32_t *row_res_u32 = reinterpret_cast<const uint32_t*>(residual + row * cols);
    uint32_t *row_res_out_u32 = reinterpret_cast<uint32_t*>(residual_output + row * cols);
    uint32_t *row_norm_out_u32 = reinterpret_cast<uint32_t*>(norm_output + row * cols);
    const int cols2 = cols >> 1;

    // Pass 1: Vectorized residual add + sum of squares
    // Output packing uses v_cvt_pkrtz_f16_f32 on GFX906
    float sum_sq = 0.0f;
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t in_packed = row_in_u32[i];
        uint32_t res_packed = row_res_u32[i];
        float s0 = fp16_to_float(static_cast<uint16_t>(in_packed & 0xFFFF))
                 + fp16_to_float(static_cast<uint16_t>(res_packed & 0xFFFF));
        float s1 = fp16_to_float(static_cast<uint16_t>(in_packed >> 16))
                 + fp16_to_float(static_cast<uint16_t>(res_packed >> 16));
        float2 pair = make_float2(s0, s1);
        __half2 h2 = __float22half2_rn(pair);
        row_res_out_u32[i] = *reinterpret_cast<uint32_t*>(&h2);
        sum_sq += s0 * s0 + s1 * s1;
    }
    // Scalar tail
    if ((cols & 1) && tid == 0)
    {
        const uint16_t *row_input = input + row * cols;
        const uint16_t *row_residual = residual + row * cols;
        float s = fp16_to_float(row_input[cols - 1]) + fp16_to_float(row_residual[cols - 1]);
        residual_output[row * cols + cols - 1] = float_to_fp16(s);
        sum_sq += s * s;
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

    // Pass 2: Vectorized normalize with packed FP16 output
    for (int i = tid; i < cols2; i += stride)
    {
        uint32_t res_packed = row_res_out_u32[i];
        int base = i << 1;
        float v0 = fp16_to_float(static_cast<uint16_t>(res_packed & 0xFFFF)) * inv_rms * gamma[base + 0];
        float v1 = fp16_to_float(static_cast<uint16_t>(res_packed >> 16)) * inv_rms * gamma[base + 1];
        float2 pair = make_float2(v0, v1);
        __half2 h2 = __float22half2_rn(pair);
        row_norm_out_u32[i] = *reinterpret_cast<uint32_t*>(&h2);
    }
    if ((cols & 1) && tid == 0)
    {
        float v = fp16_to_float(residual_output[row * cols + cols - 1]) * inv_rms * gamma[cols - 1];
        norm_output[row * cols + cols - 1] = float_to_fp16(v);
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

    bool hipOps_fused_swiglu_quantize_blockwise(
        const float *gate,
        const float *up,
        int8_t *A_int8,
        float *scales_A_blockwise,
        int M, int K,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError(); // Clear stale errors

        dim3 grid(M);
        dim3 block(256);

        hipLaunchKernelGGL(rocm_fused_swiglu_quantize_blockwise_kernel,
                           grid, block, 0, static_cast<hipStream_t>(stream),
                           gate, up, A_int8, scales_A_blockwise, M, K);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_swiglu_quantize_blockwise launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =====================================================================
    // Fused Residual Add + RMSNorm
    // =====================================================================

    bool hipOps_fused_residual_rmsnorm_fp32(
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
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp32_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_fp32 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_fused_residual_rmsnorm_bf16(
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
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_bf16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_bf16 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_fused_residual_rmsnorm_fp16(
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
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_fp16 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
