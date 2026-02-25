/**
 * @file ROCmQuantisedGemmKernel.hip
 * @brief Common HIP kernels and utilities for INT8 quantized GEMM on AMD ROCm GPUs
 *
 * This file contains shared infrastructure used by all quantized GEMM backends:
 *
 *   GPU Kernels:
 *     - quantizeActivationsQ8_kernel_t : FP32 → INT8 per-row symmetric quantization
 *     - applyScales_kernel             : INT32 → FP32 with per-row/col scales
 *     - applyScaling_full_kernel       : Full epilogue (alpha, beta, bias)
 *     - applyScaling_m1_vec4_kernel    : M=1 decode-optimized scaling
 *     - fillConstantFloat_kernel       : Fill buffer with constant value
 *     - biasAdd_kernel                 : In-place bias addition
 *
 *   Dispatch Functions (extern "C"):
 *     - rocmQuantGemm_setDevice()              : HIP device selection
 *     - rocmQuantGemm_uploadWeights()          : Weight upload H2D
 *     - rocmQuantGemm_ensureWorkBuffers()      : Work buffer management
 *     - rocmQuantGemm_quantizeActivations()    : FP32 → INT8 activation quantization
 *     - rocmQuantGemm_applyScaling()           : INT32 → FP32 scaling epilogue
 *     - rocmQuantGemm_biasAdd()                : In-place bias addition
 *     - rocmQuantGemm_executeTwoKernel*()      : CK GEMM + scaling orchestrators
 *     - rocmQuantGemm_alloc*(), copy*(), free* : Memory management utilities
 *
 * Backend-specific GEMM implementations live in separate files:
 *   - ROCmQuantisedGemmKernel_CK.hip       : ComposableKernel INT8 GEMM (gfx906)
 *   - ROCmQuantisedGemmKernel_INT8_VNNI.hip : INT8 VNNI prefill kernels (gfx906)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <chrono>

#include "../../backends/rocm/HipDeviceGuard.h"
#include "../../utils/DebugEnv.h"

// ============================================================================
// QUANTIZATION SHAPE HELPERS
// ============================================================================
//
// These helpers select thread counts and LDS policies for the quantization
// kernel based on the shape (aspect ratio) of the activation matrix.
//
// ============================================================================

namespace {

enum class QuantShapeClass
{
    ExtremeWide,
    Wide,
    Balanced,
    Tall,
};

struct QuantAspectThresholds
{
    double extreme_wide;
    double wide;
    double balanced_min;
};

struct QuantLdsPolicy
{
    int max_k;
    int min_m;
};

static QuantAspectThresholds getQuantAspectThresholds()
{
    static const QuantAspectThresholds thresholds{64.0, 8.0, 0.125};

    return thresholds;
}

static QuantLdsPolicy getQuantLdsPolicy()
{
    static const QuantLdsPolicy policy{4096, 4};

    return policy;
}

static QuantShapeClass classifyQuantShape(int M, int K)
{
    const double safe_m = static_cast<double>(std::max(M, 1));
    const double aspect_k_over_m = static_cast<double>(K) / safe_m;
    const QuantAspectThresholds thresholds = getQuantAspectThresholds();

    if (aspect_k_over_m >= thresholds.extreme_wide)
    {
        return QuantShapeClass::ExtremeWide;
    }
    if (aspect_k_over_m >= thresholds.wide)
    {
        return QuantShapeClass::Wide;
    }
    if (aspect_k_over_m >= thresholds.balanced_min)
    {
        return QuantShapeClass::Balanced;
    }
    return QuantShapeClass::Tall;
}

static int chooseQuantThreadsFromShape(int M, int K)
{
    switch (classifyQuantShape(M, K))
    {
        case QuantShapeClass::ExtremeWide:
            return 512;
        case QuantShapeClass::Wide:
            return 256;
        case QuantShapeClass::Balanced:
            return 256;
        case QuantShapeClass::Tall:
            return 128;
    }

    return 256;
}

} // anonymous namespace

// ============================================================================
// HIP Kernel: FP32 → INT8 Activation Quantization (Phase 2)
// ============================================================================

/**
 * @brief Per-row symmetric quantization: FP32 → INT8 with per-row scale
 *
 * Algorithm:
 *   1. Find max_abs = max(|x_i|) for each row
 *   2. scale = max_abs / 127.0 (or 1.0 if max_abs == 0)
 *   3. int8_i = round(x_i / scale) clamped to [-127, 127]
 *
 * Grid: (M, 1, 1) blocks
 * Block: shape-selected threads (128/256/512), chosen from matrix aspect ratio K/M
 *
 * @param d_A_fp32 Input FP32 matrix [M × K] RowMajor
 * @param d_A_int8 Output INT8 matrix [M × K] RowMajor
 * @param d_scales Output per-row scales [M]
 * @param M Number of rows
 * @param K Number of columns per row
 */
template <bool UseLdsRowStaging, bool NoTail>
__global__ void quantizeActivationsQ8_kernel_t(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory for wavefront-level max merge.
    // Largest launch uses 512 threads => 8 wave64 groups.
    __shared__ float s_wave_max[8];
    __shared__ float s_absmax;

    extern __shared__ float s_row[];

    // Step 1: Find max absolute value in this row (parallel reduction)
    float local_max = 0.0f;
    const float *row_ptr = d_A_fp32 + static_cast<size_t>(row) * K;
    const bool fp32_aligned16 = ((reinterpret_cast<uintptr_t>(row_ptr) & 0x0F) == 0);

    if constexpr (UseLdsRowStaging)
    {
        for (int k = tid; k < K; k += block_size)
        {
            s_row[k] = row_ptr[k];
        }
        __syncthreads();
    }

    const float* src_ptr = UseLdsRowStaging ? s_row : row_ptr;
    const int vec_len = NoTail ? K : ((K / 4) * 4);
    const bool src_aligned16 = ((reinterpret_cast<uintptr_t>(src_ptr) & 0x0F) == 0);

    if (src_aligned16)
    {
        for (int k4 = tid * 4; k4 < vec_len; k4 += block_size * 4)
        {
            const float4 v = *reinterpret_cast<const float4 *>(src_ptr + k4);
            local_max = fmaxf(local_max, fabsf(v.x));
            local_max = fmaxf(local_max, fabsf(v.y));
            local_max = fmaxf(local_max, fabsf(v.z));
            local_max = fmaxf(local_max, fabsf(v.w));
        }
        if constexpr (!NoTail)
        {
            for (int k = vec_len + tid; k < K; k += block_size)
            {
                local_max = fmaxf(local_max, fabsf(src_ptr[k]));
            }
        }
    }
    else
    {
        for (int k = tid; k < K; k += block_size)
        {
            float val = src_ptr[k];
            float abs_val = fabsf(val);
            local_max = fmaxf(local_max, abs_val);
        }
    }

    // Wavefront reduction (wave64 on gfx906)
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int wave_count = block_size >> 6;
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    if (lane == 0)
    {
        s_wave_max[wave] = local_max;
    }
    __syncthreads();

    // First wave merges all per-wave maxima
    if (wave == 0)
    {
        float v = (tid < wave_count) ? s_wave_max[tid] : 0.0f;
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            v = fmaxf(v, __shfl_down(v, offset));
        }
        if (tid == 0)
        {
            s_absmax = v;
        }
    }
    __syncthreads();

    // Step 2: Compute scale (thread 0 writes to global memory)
    float max_abs = s_absmax;
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0)
    {
        d_scales[row] = scale;
    }

    // Step 3: Quantize all elements in this row
    int8_t *out_row = d_A_int8 + static_cast<size_t>(row) * K;
    const bool int8_aligned4 = ((reinterpret_cast<uintptr_t>(out_row) & 0x03) == 0);

    if (src_aligned16 && int8_aligned4)
    {
        for (int k4 = tid * 4; k4 < vec_len; k4 += block_size * 4)
        {
            const float4 v = *reinterpret_cast<const float4 *>(src_ptr + k4);

            int q0 = static_cast<int>(rintf(v.x * inv_scale));
            int q1 = static_cast<int>(rintf(v.y * inv_scale));
            int q2 = static_cast<int>(rintf(v.z * inv_scale));
            int q3 = static_cast<int>(rintf(v.w * inv_scale));

            q0 = (q0 < -127) ? -127 : ((q0 > 127) ? 127 : q0);
            q1 = (q1 < -127) ? -127 : ((q1 > 127) ? 127 : q1);
            q2 = (q2 < -127) ? -127 : ((q2 > 127) ? 127 : q2);
            q3 = (q3 < -127) ? -127 : ((q3 > 127) ? 127 : q3);

            const uint32_t packed =
                static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);

            *reinterpret_cast<uint32_t *>(out_row + k4) = packed;
        }

        if constexpr (!NoTail)
        {
            for (int k = vec_len + tid; k < K; k += block_size)
            {
                int q = static_cast<int>(rintf(src_ptr[k] * inv_scale));
                q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
                out_row[k] = static_cast<int8_t>(q);
            }
        }
    }
    else
    {
        for (int k = tid; k < K; k += block_size)
        {
            float val = src_ptr[k];
            float scaled = val * inv_scale;

            // rintf() maps to v_rndne_f32 (1 instruction) instead of
            // roundf() which decomposes to trunc+sub+cmp+cndmask+bfi+add (6 instructions).
            // For int8 quantization, round-to-nearest-even vs round-half-away-from-zero
            // has negligible quality impact (differs only at exact x.5 tie points).
            int q = static_cast<int>(rintf(scaled));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);

            out_row[k] = static_cast<int8_t>(q);
        }
    }
}

// ============================================================================
// Apply Scales Kernel (for two-kernel approach)
// ============================================================================
//
// Applies per-row and per-column scales to INT32 GEMM output:
//   E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
//
// This is the second kernel in the two-kernel approach, which is more reliable
// than CK's fused D-tensor handling on gfx906.
//
// Grid: ((M*N + 255) / 256, 1, 1) blocks
// Block: (256, 1, 1) threads
//
__global__ void applyScales_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_E_fp32,           // [M×N] FP32 scaled output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    if (idx >= total) return;
    
    const int m = idx / N;
    const int n = idx % N;
    
    d_E_fp32[idx] = static_cast<float>(d_C_int32[idx]) * d_scale_A[m] * d_scale_B[n];
}

// ============================================================================
// Apply Scaling Kernel with Full Epilogue (alpha, beta, bias)
// ============================================================================
//
// Full-featured scaling epilogue matching CUDA's apply_scaling_kernel:
//   result = alpha * C_int32 * scale_A[m] * scale_B[n]
//   if (beta != 0) result += beta * C_existing[idx]
//   if (bias)      result += bias[n]
//
// Grid: ((N + 15) / 16, (M + 15) / 16, 1) blocks
// Block: (16, 16, 1) threads
//
__global__ void applyScaling_full_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_C_fp32,           // [M×N] FP32 output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N,
    float alpha, float beta,
    const float* __restrict__ d_C_existing, // [M×N] existing output (for beta != 0)
    const float* __restrict__ d_bias)       // [N] optional bias, broadcasted across rows
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        const int idx = row * N + col;
        const int32_t val_int32 = d_C_int32[idx];
        const float scale_combined = d_scale_A[row] * d_scale_B[col];
        float result = alpha * static_cast<float>(val_int32) * scale_combined;

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            result += beta * d_C_existing[idx];
        }

        if (d_bias != nullptr)
        {
            result += d_bias[col];
        }

        d_C_fp32[idx] = result;
    }
}

// ============================================================================
// Vectorized Scaling Kernel with 1D Flat Grid (M > 1 fast path)
// ============================================================================
//
// Replaces the 16×16 applyScaling_full_kernel which creates catastrophic
// block counts for large N (LM_Head N=151936: 360K blocks, 254ms).
//
// Uses a 1D flat grid-stride loop over the entire M×N output:
//   - Sequential memory access across the full buffer (HBM-friendly)
//   - Vec4 loads/stores for 128-bit transactions
//   - Fixed ~480 blocks regardless of shape (vs 360K for LM_Head)
//
// N must be divisible by 4 for vec4 correctness (all Llaminar shapes are).
// When N % 4 != 0, falls back to the legacy applyScaling_full_kernel.
//
// Grid:  (min((M*N/4 + BLOCK_X-1)/BLOCK_X, MAX_BLOCKS))
// Block: (BLOCK_X) where BLOCK_X = 256
//
__global__ void applyScaling_vec4_flat_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_C_fp32,           // [M×N] FP32 output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N, int total_vec4,           // total_vec4 = M * N / 4
    float alpha, float beta,
    const float* __restrict__ d_C_existing, // [M×N] existing output (for beta != 0)
    const float* __restrict__ d_bias)       // [N] optional bias, broadcasted across rows
{
    const int N4 = N / 4;  // number of vec4 chunks per row

    for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
         vi < total_vec4;
         vi += gridDim.x * blockDim.x)
    {
        const int row = vi / N4;
        const int col4 = (vi - row * N4) * 4;  // vi % N4 * 4
        const int idx = row * N + col4;

        const float row_scale = alpha * d_scale_A[row];
        const int4 vals = *reinterpret_cast<const int4*>(d_C_int32 + idx);
        const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + col4);

        float4 out;
        out.x = static_cast<float>(vals.x) * row_scale * sb.x;
        out.y = static_cast<float>(vals.y) * row_scale * sb.y;
        out.z = static_cast<float>(vals.z) * row_scale * sb.z;
        out.w = static_cast<float>(vals.w) * row_scale * sb.w;

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            const float4 ex = *reinterpret_cast<const float4*>(d_C_existing + idx);
            out.x += beta * ex.x;
            out.y += beta * ex.y;
            out.z += beta * ex.z;
            out.w += beta * ex.w;
        }

        if (d_bias != nullptr)
        {
            const float4 b = *reinterpret_cast<const float4*>(d_bias + col4);
            out.x += b.x;
            out.y += b.y;
            out.z += b.z;
            out.w += b.w;
        }

        *reinterpret_cast<float4*>(d_C_fp32 + idx) = out;
    }
}

// Decode-specialized scaling path (M=1, beta=0, no bias):
//   C_fp32[n] = alpha * C_int32[n] * scale_A[0] * scale_B[n]
// Uses 1D grid and vectorized x4 lanes to avoid the 16x16 underutilization
// of applyScaling_full_kernel when only a single row is active.
__global__ void applyScaling_m1_vec4_kernel(
    const int32_t* __restrict__ d_C_int32,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    float alpha)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = tid * 4;

    const float row_scale = alpha * d_scale_A[0];

    if (n4 + 3 < N)
    {
        const int4 v = *reinterpret_cast<const int4*>(d_C_int32 + n4);
        const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + n4);

        float4 out;
        out.x = static_cast<float>(v.x) * row_scale * sb.x;
        out.y = static_cast<float>(v.y) * row_scale * sb.y;
        out.z = static_cast<float>(v.z) * row_scale * sb.z;
        out.w = static_cast<float>(v.w) * row_scale * sb.w;

        *reinterpret_cast<float4*>(d_C_fp32 + n4) = out;
        return;
    }

    for (int n = n4; n < N && n < n4 + 4; ++n)
    {
        d_C_fp32[n] = static_cast<float>(d_C_int32[n]) * row_scale * d_scale_B[n];
    }
}

__global__ void fillConstantFloat_kernel(float* __restrict__ dst, int count, float value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] = value;
    }
}

// ============================================================================
// DISPATCH FUNCTIONS
// ============================================================================

extern "C"
{
    // Forward declaration: CK INT8 GEMM (defined in ROCmQuantisedGemmKernel_CK.hip)
    bool rocmQuantGemm_executeNoScale(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx);

    // -------------------------------------------------------------------------
    // Device management
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_setDevice(int rocm_device_id)
    {
        hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_setDevice] Failed to set device %d: %s\n",
                    rocm_device_id, hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Weight upload (Phase 1)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_uploadWeights(
        const int8_t *h_weights_int8,
        const float *h_scales_B,
        int8_t **d_weights_int8,
        float **d_scales_B,
        int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Allocate device memory for weights
        size_t weight_bytes = static_cast<size_t>(K) * N * sizeof(int8_t);
        hipError_t err = hipMalloc(d_weights_int8, weight_bytes);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc weights failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        // CRITICAL: Verify hipMalloc actually allocated memory
        if (*d_weights_int8 == nullptr)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] CRITICAL: hipMalloc returned success but pointer is NULL! "
                    "(bytes=%zu K=%d N=%d device=%d)\n",
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Allocate device memory for scales
        size_t scale_bytes = static_cast<size_t>(N) * sizeof(float);
        err = hipMalloc(d_scales_B, scale_bytes);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            *d_weights_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // Copy weights H2D
        err = hipMemcpy(*d_weights_int8, h_weights_int8, weight_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy weights failed: %s "
                    "(src=%p dst=%p bytes=%zu K=%d N=%d device=%d)\n",
                    hipGetErrorString(err), (void*)h_weights_int8, (void*)*d_weights_int8,
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Copy scales H2D
        err = hipMemcpy(*d_scales_B, h_scales_B, scale_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // -------------------------------------------------------------------------
    // Work buffer allocation (Phase 2)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Only reallocate if M increased
        if (M <= *work_buffer_M)
        {
            return true;
        }

        // Free existing buffers
        if (*d_A_int8) {
            hipFree(*d_A_int8);
        }
        if (*d_scales_A) {
            hipFree(*d_scales_A);
        }
        if (*d_C_int32) {
            hipFree(*d_C_int32);
        }

        *d_A_int8 = nullptr;
        *d_scales_A = nullptr;
        *d_C_int32 = nullptr;

        // Allocate new buffers
        hipError_t err = hipMalloc(d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc A_int8 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_scales_A, static_cast<size_t>(M) * sizeof(float));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            *d_A_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc scales_A failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            hipFree(*d_scales_A);
            *d_A_int8 = nullptr;
            *d_scales_A = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc C_int32 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        *work_buffer_M = M;
        return true;
    }

    // =========================================================================
    // TWO-KERNEL APPROACH: CK INT8→INT32 GEMM + applyScales_kernel
    // =========================================================================
    //
    // These functions orchestrate the two-kernel CK GEMM pipeline:
    //   1. rocmQuantGemm_executeNoScale() for INT8×INT8→INT32 (defined in CK.hip)
    //   2. applyScales_kernel for INT32→FP32 with per-row/col scales
    //
    // FORMULA:
    //   C_int32[m,n] = sum_k(A_int8[m,k] × B_int8[k,n])    [CK GEMM]
    //   E_fp32[m,n] = C_int32[m,n] × scale_A[m] × scale_B[n]  [applyScales]
    //
    // =========================================================================

    bool rocmQuantGemm_executeTwoKernel(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (row-major)
        float *d_E_fp32,              // [M × N] FP32 output (row-major)
        const float *d_scale_A,       // [M] per-row scales
        const float *d_scale_B,       // [N] per-column scales
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        // 1. Allocate INT32 accumulator buffer
        int32_t *d_C_int32 = nullptr;
        hipError_t err = hipMalloc(&d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] Failed to allocate INT32 buffer: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // 2. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] CK NoScale GEMM failed\n");
            hipFree(d_C_int32);
            return false;
        }

        // 3. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_C_int32);
            return false;
        }

        // 4. Cleanup
        hipFree(d_C_int32);
        return true;
    }

    // =========================================================================
    // ALLOCATION-FREE TWO-KERNEL CK APPROACH (HOT-PATH OPTIMIZED)
    // =========================================================================

    bool rocmQuantGemm_executeTwoKernel_cached(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (row-major)
        float *d_E_fp32,              // [M × N] FP32 output (row-major)
        const float *d_scale_A,       // [M] per-row scales
        const float *d_scale_B,       // [N] per-column scales
        int32_t *d_C_int32,           // Pre-allocated [M × N] INT32 accumulator
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        const bool profiling = llaminar2::debugEnv().profile.enabled;
        std::chrono::high_resolution_clock::time_point t_total_start{}, t_step_start{}, t_step_end{};
        if (profiling) {
            t_total_start = std::chrono::high_resolution_clock::now();
            t_step_start = t_total_start;
        }
        
        auto log_timing = [&](const char* step) {
            if (!profiling) return;
            t_step_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
            if (ms > 0.5) {
                fprintf(stderr, "[CK_GEMM_TIMING] %s: %.3f ms (M=%d N=%d K=%d)\n", step, ms, M, N, K);
            }
            t_step_start = t_step_end;
        };
        
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] Null INT32 buffer\n");
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] CK NoScale GEMM failed\n");
            return false;
        }
        log_timing("CK_executeNoScale");

        // 2. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("applyScales_kernel_launch");
        
        // Log total time
        if (profiling) {
            auto t_total_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
            if (total_ms > 1.0) {
                fprintf(stderr, "[CK_GEMM_TIMING] TOTAL: %.3f ms (M=%d N=%d K=%d)\n", total_ms, M, N, K);
            }
        }

        return true;
    }

    // =========================================================================
    // M-PADDED TWO-KERNEL CK APPROACH (FOR DECODE: M < 8)
    // =========================================================================

    bool rocmQuantGemm_executeTwoKernel_padded(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [N × K] INT8 weights (row-major, viewed as [K × N] col-major)
        float *d_E_fp32,              // [M × N] FP32 output (only first M rows written)
        const float *d_scale_A,       // [M] per-row activation scales
        const float *d_scale_B,       // [N] per-column weight scales
        int32_t *d_C_int32,           // [padded_m × N] Pre-allocated INT32 accumulator
        int M,                        // Actual M (output rows needed)
        int padded_m,                 // Padded M for CK (>= CK_SMALL_MIN_M = 8)
        int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Null INT32 buffer\n");
            return false;
        }

        if (padded_m < M)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] padded_m (%d) < M (%d)\n", padded_m, M);
            return false;
        }

        hipError_t err;

        // 1. Allocate padded activation buffer [padded_m × K]
        int8_t *d_A_padded = nullptr;
        err = hipMalloc(&d_A_padded, static_cast<size_t>(padded_m) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded A: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // Zero-initialize the entire padded buffer (padding rows will be zeros)
        err = hipMemset(d_A_padded, 0, static_cast<size_t>(padded_m) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to zero padded A: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // Copy actual activations into first M rows
        err = hipMemcpy(d_A_padded, d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to copy A to padded: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // 2. Allocate padded scale buffer [padded_m]
        float *d_scale_A_padded = nullptr;
        err = hipMalloc(&d_scale_A_padded, static_cast<size_t>(padded_m) * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded scales: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // Copy actual scales and fill padding with 1.0 (neutral scale)
        err = hipMemcpy(d_scale_A_padded, d_scale_A, static_cast<size_t>(M) * sizeof(float), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to copy scales: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            return false;
        }

        // Fill padding scales with 1.0 (so padded rows have neutral scaling)
        if (padded_m > M)
        {
            const int padding_count = padded_m - M;
            const int fill_block = 256;
            const int fill_grid = (padding_count + fill_block - 1) / fill_block;
            hipLaunchKernelGGL(
                fillConstantFloat_kernel,
                dim3(fill_grid), dim3(fill_block), 0, static_cast<hipStream_t>(stream),
                d_scale_A_padded + M, padding_count, 1.0f);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to fill padding scales: %s\n",
                        hipGetErrorString(err));
                hipFree(d_A_padded);
                hipFree(d_scale_A_padded);
                return false;
            }
        }

        // 3. Allocate padded output buffer [padded_m × N]
        float *d_E_padded = nullptr;
        err = hipMalloc(&d_E_padded, static_cast<size_t>(padded_m) * N * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded output: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            return false;
        }

        // 4. Run CK INT8 GEMM on padded dimensions: C_int32 = A_padded × B (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_padded, d_B_int8, d_C_int32, padded_m, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] CK NoScale GEMM failed\n");
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 5. Apply scales to full padded output: E_padded[m,n] = C_int32[m,n] * scale_A_padded[m] * scale_B[n]
        const int total_padded = padded_m * N;
        const int block_size = 256;
        const int grid_size = (total_padded + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_padded, d_scale_A_padded, d_scale_B, padded_m, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 6. Extract first M rows of padded output to actual output
        //    E_fp32[M × N] = E_padded[0:M, :]
        err = hipMemcpy(d_E_fp32, d_E_padded, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to extract output: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 7. Cleanup temporary buffers
        hipFree(d_A_padded);
        hipFree(d_scale_A_padded);
        hipFree(d_E_padded);

        return true;
    }

    // =========================================================================
    // M-PADDED TWO-KERNEL CK APPROACH WITH CACHED BUFFERS (ALLOCATION-FREE)
    // =========================================================================

    bool rocmQuantGemm_executeTwoKernel_padded_cached(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [N × K] INT8 weights (row-major, viewed as [K × N] col-major)
        float *d_E_fp32,              // [M × N] FP32 output (only first M rows written)
        const float *d_scale_A,       // [M] per-row activation scales
        const float *d_scale_B,       // [N] per-column weight scales
        int32_t *d_C_int32,           // [padded_m × N] Pre-allocated INT32 accumulator
        int8_t *d_A_padded,           // [padded_m × K] Pre-allocated padded activations
        float *d_scale_A_padded,      // [padded_m] Pre-allocated padded scales
        float *d_E_padded,            // [padded_m × N] Pre-allocated padded output
        int M,                        // Actual M (output rows needed)
        int padded_m,                 // Padded M for CK (>= CK_SMALL_MIN_M = 8)
        int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        const bool profiling = llaminar2::debugEnv().profile.enabled;
        std::chrono::high_resolution_clock::time_point t_total_start{}, t_step_start{}, t_step_end{};
        if (profiling) {
            t_total_start = std::chrono::high_resolution_clock::now();
            t_step_start = t_total_start;
        }
        
        auto log_timing = [&](const char* step) {
            if (!profiling) return;
            t_step_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
            if (ms > 0.5) {
                fprintf(stderr, "[CK_GEMM_TIMING] %s: %.3f ms (M=%d N=%d K=%d)\n", step, ms, M, N, K);
            }
            t_step_start = t_step_end;
        };
        
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32 || !d_A_padded || !d_scale_A_padded || !d_E_padded)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Null buffer\n");
            return false;
        }

        if (padded_m < M)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] padded_m (%d) < M (%d)\n", padded_m, M);
            return false;
        }

        hipError_t err;
        log_timing("param_validation");

        // 1. Zero-initialize the padded activation buffer (padding rows will be zeros)
        err = hipMemsetAsync(d_A_padded, 0, static_cast<size_t>(padded_m) * K * sizeof(int8_t), static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to zero padded A: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemsetAsync_A_padded");

        // Copy actual activations into first M rows (async)
        err = hipMemcpyAsync(d_A_padded, d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to copy A to padded: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_A_to_padded");

        // 2. Copy actual scales and fill padding with 1.0 (neutral scale)
        err = hipMemcpyAsync(d_scale_A_padded, d_scale_A, static_cast<size_t>(M) * sizeof(float), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to copy scales: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_scales");

        // Fill padding scales with 1.0 (allocation-free, stream-ordered)
        if (padded_m > M)
        {
            const int padding_count = padded_m - M;
            const int fill_block = 256;
            const int fill_grid = (padding_count + fill_block - 1) / fill_block;
            hipLaunchKernelGGL(
                fillConstantFloat_kernel,
                dim3(fill_grid), dim3(fill_block), 0, static_cast<hipStream_t>(stream),
                d_scale_A_padded + M, padding_count, 1.0f);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to fill padding scales: %s\n",
                        hipGetErrorString(err));
                return false;
            }
        }
        log_timing("fill_padding_scales");

        // 3. Run CK INT8 GEMM on padded dimensions: C_int32 = A_padded × B (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_padded, d_B_int8, d_C_int32, padded_m, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] CK NoScale GEMM failed\n");
            return false;
        }
        log_timing("CK_executeNoScale");

        // 4. Apply scales to full padded output: E_padded[m,n] = C_int32[m,n] * scale_A_padded[m] * scale_B[n]
        const int total_padded = padded_m * N;
        const int block_size = 256;
        const int grid_size = (total_padded + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_padded, d_scale_A_padded, d_scale_B, padded_m, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("applyScales_kernel_launch");

        // 5. Extract first M rows of padded output to actual output (async)
        //    E_fp32[M × N] = E_padded[0:M, :]
        err = hipMemcpyAsync(d_E_fp32, d_E_padded, static_cast<size_t>(M) * N * sizeof(float), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to extract output: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_extract_output");
        
        // Log total time
        if (profiling) {
            auto t_total_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
            if (total_ms > 1.0) {
                fprintf(stderr, "[CK_GEMM_TIMING] TOTAL: %.3f ms (M=%d N=%d K=%d)\n", total_ms, M, N, K);
            }
        }

        // No cleanup needed - buffers are caller-owned!
        return true;
    }

    // =========================================================================
    // INT8 ALLOCATION HELPER
    // =========================================================================
    
    bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt8] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // ACTIVATION QUANTIZATION (FP32 → INT8)
    // =========================================================================

    bool rocmQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int rocm_device_id, void* stream)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || K <= 0)
            return true; // Nothing to do

        // Launch kernel: one block per row, thread count selected by shape class.
        // Policy is based on aspect ratio K/M (row width vs row count), not fixed dimensions.
        const int block_size = chooseQuantThreadsFromShape(M, K);
        const bool no_tail = ((K & 0x3) == 0);
        const QuantLdsPolicy lds_policy = getQuantLdsPolicy();
        const bool use_lds_row_staging = (K <= lds_policy.max_k) && (M >= lds_policy.min_m);
        dim3 grid(M, 1, 1);
        dim3 block(block_size, 1, 1);

        size_t shared_mem_bytes = use_lds_row_staging ? static_cast<size_t>(K) * sizeof(float) : 0;

        if (use_lds_row_staging)
        {
            if (no_tail)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<true, true>),
                    grid, block, shared_mem_bytes, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<true, false>),
                    grid, block, shared_mem_bytes, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
        }
        else
        {
            if (no_tail)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<false, true>),
                    grid, block, 0, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<false, false>),
                    grid, block, 0, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
        }

        // Check for launch errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_quantizeActivations] Kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // No device-wide sync here; rely on stream ordering + a final sync/copy.

        return true;
    }

    // =========================================================================
    // MEMORY MANAGEMENT UTILITIES
    // =========================================================================

    void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id)
    {
        if (d_ptr)
        {
            // Set device before freeing - required by HIP/CUDA
            hipError_t set_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
            if (set_err == hipErrorDeinitialized || set_err == hipErrorNoDevice)
            {
                // HIP runtime is shutting down, skip cleanup
                // This is normal during static destruction at program exit
                return;
            }
            hipError_t err = hipFree(d_ptr);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                // Only log actual errors, not shutdown-related ones
                fprintf(stderr, "WARNING: hipFree failed: %s\n", hipGetErrorString(err));
            }
        }
    }

    bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocFloat] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt32] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(d_dst, h_src, count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyHostToDevice] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyDeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(int32_t), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyInt32DeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // KERNEL-ONLY TIMED EXECUTION (FOR BENCHMARKING)
    // =========================================================================

    bool rocmQuantGemm_executeTwoKernel_timed(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (already on device)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (already on device)
        float *d_E_fp32,              // [M × N] FP32 output (on device)
        const float *d_scale_A,       // [M] per-row scales (on device)
        const float *d_scale_B,       // [N] per-column scales (on device)
        int32_t *d_C_int32,           // Pre-allocated [M × N] INT32 accumulator
        int M, int N, int K,
        int rocm_device_id,
        float *kernel_time_ms, void* stream,
        void* kernel_ctx)        // OUTPUT: kernel execution time in ms
    {
        if (M <= 0 || N <= 0 || K <= 0)
        {
            if (kernel_time_ms) *kernel_time_ms = 0.0f;
            return true;
        }

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Null INT32 buffer\n");
            return false;
        }

        // Create HIP events for timing
        hipEvent_t start, stop;
        hipError_t err;
        
        err = hipEventCreate(&start);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to create start event: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        
        err = hipEventCreate(&stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to create stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            return false;
        }

        // Record start event
        err = hipEventRecord(start, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to record start event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] CK NoScale GEMM failed\n");
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // 2. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // Record stop event
        err = hipEventRecord(stop, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to record stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // Wait for completion and get elapsed time
        err = hipEventSynchronize(stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to sync stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        float elapsed_ms = 0.0f;
        err = hipEventElapsedTime(&elapsed_ms, start, stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to get elapsed time: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        if (kernel_time_ms)
            *kernel_time_ms = elapsed_ms;

        // Cleanup events
        hipEventDestroy(start);
        hipEventDestroy(stop);

        return true;
    }

    // =========================================================================
    // APPLY SCALING WITH FULL EPILOGUE (alpha, beta, bias)
    // =========================================================================

    bool rocmQuantGemm_applyScaling(
        const int32_t *d_C_int32,    // [M×N] INT32 GEMM output
        float *d_C_fp32,             // [M×N] FP32 output
        const float *d_scales_A,     // [M] per-row scales
        const float *d_scales_B,     // [N] per-column scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,   // For beta != 0 (nullable)
        const float *d_bias,         // [N] optional bias (nullable)
        int rocm_device_id, void* stream)
    {
        // Validate required pointers
        if (!d_C_int32 || !d_C_fp32 || !d_scales_A || !d_scales_B)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] Null pointer: "
                    "d_C_int32=%p d_C_fp32=%p d_scales_A=%p d_scales_B=%p\n",
                    (void *)d_C_int32, (void *)d_C_fp32,
                    (void *)d_scales_A, (void *)d_scales_B);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Decode-specialized fast path:
        // M=1 dominates INT8 VNNI decode and the generic 16x16 kernel wastes
        // 15/16 threads in this shape. Use a 1D vectorized kernel instead.
        if (M == 1 && beta == 0.0f && d_C_existing == nullptr && d_bias == nullptr)
        {
            const int block_size = 256;
            const int elems_per_thread = 4;
            const int grid_size = (N + block_size * elems_per_thread - 1) / (block_size * elems_per_thread);

            hipLaunchKernelGGL(
                applyScaling_m1_vec4_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B, N, alpha);

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_m1_vec4_kernel launch failed: %s "
                        "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
                return false;
            }
            return true;
        }

        // Vectorized 1D flat path for M > 1:
        // Uses sequential access through the entire M×N buffer (HBM-friendly)
        // with a fixed block count regardless of shape dimensions.
        if (N % 4 == 0)
        {
            constexpr int BLOCK_X = 256;
            const int total_vec4 = M * (N / 4);
            // Cap grid to ~480 blocks for MI60 (60 CUs × 8 waves/CU occupancy target)
            constexpr int MAX_BLOCKS = 480;
            const int natural_blocks = (total_vec4 + BLOCK_X - 1) / BLOCK_X;
            const int grid_size = (natural_blocks < MAX_BLOCKS) ? natural_blocks : MAX_BLOCKS;

            hipLaunchKernelGGL(
                applyScaling_vec4_flat_kernel,
                dim3(grid_size), dim3(BLOCK_X), 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
                M, N, total_vec4, alpha, beta, d_C_existing, d_bias);
        }
        else
        {
            // Fallback for N not divisible by 4 (shouldn't happen in practice)
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);
            hipLaunchKernelGGL(
                applyScaling_full_kernel,
                grid, block, 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
                M, N, alpha, beta, d_C_existing, d_bias);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_vec4_2d_kernel launch failed: %s "
                    "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

    // =========================================================================
    // Apply Scale-A Only Kernel (for grouped ratio-VNNI decode path)
    //
    // When sub-row grouping is used, the GEMV kernel already applies per-group
    // weight scales during FP32 partial accumulation. The remaining scaling is
    // just the activation scale:
    //   output[m,n] = alpha * C_fp32_grouped[m,n] * scale_A[m]
    //                 + beta * existing[m,n] + bias[n]
    // =========================================================================

    __global__ void applyScaleA_m1_vec4_kernel(
        const float* __restrict__ d_C_fp32_grouped,  // [N] FP32 with B-scales already applied
        float* __restrict__ d_output,                 // [N] FP32 output
        const float* __restrict__ d_scale_A,          // [1] activation scale
        int N,
        float alpha)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int n4 = tid * 4;

        const float row_scale = alpha * d_scale_A[0];

        if (n4 + 3 < N)
        {
            const float4 v = *reinterpret_cast<const float4*>(d_C_fp32_grouped + n4);

            float4 out;
            out.x = v.x * row_scale;
            out.y = v.y * row_scale;
            out.z = v.z * row_scale;
            out.w = v.w * row_scale;

            *reinterpret_cast<float4*>(d_output + n4) = out;
            return;
        }

        for (int n = n4; n < N && n < n4 + 4; ++n)
        {
            d_output[n] = d_C_fp32_grouped[n] * row_scale;
        }
    }

    __global__ void applyScaleA_full_kernel(
        const float* __restrict__ d_C_fp32_grouped,  // [M×N] FP32 with B-scales applied
        float* __restrict__ d_output,                 // [M×N] FP32 output
        const float* __restrict__ d_scale_A,          // [M] activation scales
        int M, int N,
        float alpha, float beta,
        const float* __restrict__ d_C_existing,       // [M×N] existing (for beta != 0)
        const float* __restrict__ d_bias)             // [N] optional bias
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total) return;

        const int m = idx / N;
        const int n = idx % N;

        float result = alpha * d_C_fp32_grouped[idx] * d_scale_A[m];

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            result += beta * d_C_existing[idx];
        }
        if (d_bias != nullptr)
        {
            result += d_bias[n];
        }

        d_output[idx] = result;
    }

    bool rocmQuantGemm_applyScaleA_fp32(
        const float *d_C_fp32_grouped,   // [M×N] FP32 with B-scales already applied
        float *d_output,                  // [M×N] FP32 output
        const float *d_scales_A,          // [M] per-row activation scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int rocm_device_id, void* stream)
    {
        if (!d_C_fp32_grouped || !d_output || !d_scales_A)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaleA_fp32] Null pointer\n");
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Decode-specialized fast path (M=1, no beta/bias)
        if (M == 1 && beta == 0.0f && d_C_existing == nullptr && d_bias == nullptr)
        {
            const int block_size = 256;
            const int elems_per_thread = 4;
            const int grid_size = (N + block_size * elems_per_thread - 1) / (block_size * elems_per_thread);

            hipLaunchKernelGGL(
                applyScaleA_m1_vec4_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_fp32_grouped, d_output, d_scales_A, N, alpha);
        }
        else
        {
            const int block_size = 256;
            const int total = M * N;
            const int grid_size = (total + block_size - 1) / block_size;

            hipLaunchKernelGGL(
                applyScaleA_full_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_fp32_grouped, d_output, d_scales_A,
                M, N, alpha, beta, d_C_existing, d_bias);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaleA_fp32] Kernel launch failed: %s (M=%d, N=%d)\n",
                    hipGetErrorString(err), M, N);
            return false;
        }

        return true;
    }

    // =========================================================================
    // IN-PLACE BIAS ADDITION KERNEL
    // =========================================================================

    __global__ void biasAdd_kernel(float *output, const float *bias, int M, int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N)
            return;
        
        const int col = idx % N;
        output[idx] += bias[col];
    }

    bool rocmQuantGemm_biasAdd(
        float *d_output,       // [M × N] FP32 output (modified in-place)
        const float *d_bias,   // [N] bias vector
        int M, int N,
        int rocm_device_id, void* stream)
    {
        if (!d_output || !d_bias)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] Null pointer: d_output=%p d_bias=%p\n",
                    (void *)d_output, (void *)d_bias);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            biasAdd_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_output, d_bias, M, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] biasAdd_kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

} // extern "C"
