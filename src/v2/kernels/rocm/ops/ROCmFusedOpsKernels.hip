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

/**
 * @brief SiLU math shared with the standalone ROCm SwiGLU decode kernel.
 *
 * The MTP grouped verifier compares multi-row kernels against serial decode
 * row-by-row.  Using the same fast exponential and reciprocal sequence here is
 * therefore a correctness requirement, not only a speed choice; a different
 * libm/division path can quantize activations differently and amplify through
 * later recurrent state.
 */
__device__ __forceinline__ float silu_fast(float x)
{
    float e = __expf(-x);
    return x * __builtin_amdgcn_rcpf(1.0f + e);
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
    int32_t *__restrict__ sums_A_blockwise,
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
    int32_t *row_sums = sums_A_blockwise ? sums_A_blockwise + row * num_blocks : nullptr;

    for (int b = warp_id; b < num_blocks; b += NUM_WARPS)
    {
        const int k_start = b * BLOCK_SIZE;

        float g = row_gate[k_start + lane];
        float u = row_up[k_start + lane];

        // SwiGLU: silu(gate) * up
        float val = silu_fast(g) * u;

        // Warp-level max_abs reduction via shuffle
        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));
        }

        float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        float qval = val * inv_scale;
        const int q = static_cast<int>(
            rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
        row_int8[k_start + lane] = static_cast<int8_t>(q);

        int q_sum = q;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            q_sum += __shfl_xor(q_sum, mask);
        }

        if (lane == 0)
        {
            row_scales[b] = scale;
            if (row_sums)
                row_sums[b] = q_sum;
        }
    }
}

// =========================================================================
// 2. Fused Residual Add + RMSNorm
// =========================================================================

// Optimized fused residual add + RMS norm kernel for ROCm.
// - Uses up to 1024 threads (better CU utilization for large hidden dims)
// - Keeps residual sums in registers across passes (no global re-read in Pass 2)
// - Warp shuffle reduction using 64-wide wavefronts (no shared memory for wavefront)
__global__ void rocm_fused_residual_rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    const float *__restrict__ gamma,
    float *__restrict__ residual_output,
    float *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float *row_input = input + row * cols;
    const float *row_residual = residual + row * cols;
    float *row_res_out = residual_output + row * cols;
    float *row_norm_out = norm_output + row * cols;

    // Pass 1: Add residual, compute sum of squares, write residual output.
    // Cache values in registers to avoid re-reading from global memory in Pass 2.
    constexpr int kMaxPerThread = 16; // handles up to cols=16384
    float local_sums[kMaxPerThread];
    int local_count = 0;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = row_input[i] + row_residual[i];
        row_res_out[i] = sum;
        local_sums[local_count++] = sum;
        sum_sq += sum * sum;
    }

    // Warp shuffle reduction (64-wide wavefronts on GCN/CDNA)
    for (int offset = 32; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset);

    // Cross-wavefront reduction via shared memory (one value per wavefront)
    __shared__ float warp_sums[16]; // max 1024 threads / 64 = 16 wavefronts
    const int lane = tid & 63;
    const int warp_id = tid >> 6;
    const int num_warps = (blockDim.x + 63) >> 6;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        float val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_xor(val, offset);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    // Pass 2: Normalize using register-cached residual sums (no global re-read)
    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = local_sums[idx++] * inv_rms * gamma[i];
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
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    constexpr int kMaxPerThread = 16;
    float local_sums[kMaxPerThread];
    int local_count = 0;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = bf16_to_float(row_input[i]) + bf16_to_float(row_residual[i]);
        row_res_out[i] = float_to_bf16(sum);
        local_sums[local_count++] = sum;
        sum_sq += sum * sum;
    }

    for (int offset = 32; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset);

    __shared__ float warp_sums[16];
    const int lane = tid & 63;
    const int warp_id = tid >> 6;
    const int num_warps = (blockDim.x + 63) >> 6;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        float val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_xor(val, offset);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = float_to_bf16(local_sums[idx++] * inv_rms * gamma[i]);
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
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    constexpr int kMaxPerThread = 16;
    float local_sums[kMaxPerThread];
    int local_count = 0;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = fp16_to_float(row_input[i]) + fp16_to_float(row_residual[i]);
        row_res_out[i] = float_to_fp16(sum);
        local_sums[local_count++] = sum;
        sum_sq += sum * sum;
    }

    for (int offset = 32; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset);

    __shared__ float warp_sums[16];
    const int lane = tid & 63;
    const int warp_id = tid >> 6;
    const int num_warps = (blockDim.x + 63) >> 6;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        float val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_xor(val, offset);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / cols + epsilon);

    int idx = 0;
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = float_to_fp16(local_sums[idx++] * inv_rms * gamma[i]);
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
                           gate, up, A_int8, scales_A_blockwise, nullptr, M, K);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_swiglu_quantize_blockwise launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_fused_swiglu_quantize_blockwise_with_sums(
        const float *gate,
        const float *up,
        int8_t *A_int8,
        float *scales_A_blockwise,
        int32_t *sums_A_blockwise,
        int M, int K,
        int device_idx,
        void *stream)
    {
        if (!sums_A_blockwise)
            return hipOps_fused_swiglu_quantize_blockwise(
                gate, up, A_int8, scales_A_blockwise, M, K, device_idx, stream);

        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        dim3 grid(M);
        dim3 block(256);

        hipLaunchKernelGGL(rocm_fused_swiglu_quantize_blockwise_kernel,
                           grid, block, 0, static_cast<hipStream_t>(stream),
                           gate, up, A_int8, scales_A_blockwise, sums_A_blockwise, M, K);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_swiglu_quantize_blockwise_with_sums launch failed: %s\n",
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

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp32_kernel,
                           dim3(num_blocks), dim3(threads_per_block), 0,
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

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_bf16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), 0,
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

        int threads_per_block = (cols <= 256) ? 256 : 1024;
        int num_blocks = rows;

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), 0,
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
