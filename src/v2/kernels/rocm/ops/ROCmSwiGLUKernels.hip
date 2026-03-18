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

// =========================================================================
// Activation Functions
// =========================================================================

// SiLU activation using fast math intrinsics (gfx906 tuned)
// Uses __expf → v_exp_f32 SFU (~4 cycles) instead of expf (~15 insns)
// Uses __builtin_amdgcn_rcpf → bare v_rcp_f32 SFU (~4 cycles)
//   NOTE: __frcp_rn() on HIP still emits full IEEE division chain
//   (v_div_scale + v_rcp + v_fma×5 + v_div_fmas + v_div_fixup = 9 insns, ~40 cycles)
//   The GCN builtin maps directly to a single v_rcp_f32 instruction.
// Accuracy: v_rcp_f32 is ~1 ULP; for sigmoid ∈ [0,1] this is negligible.
// Net savings: ~50 cycles per element on gfx906
__device__ __forceinline__ float silu_fast(float x)
{
    float e = __expf(-x);                           // v_exp_f32 SFU (~4 cycles)
    return x * __builtin_amdgcn_rcpf(1.0f + e);      // v_rcp_f32 SFU (~4 cycles)
}

} // anonymous namespace

// =========================================================================
// SwiGLU HIP Kernels — V2 Optimized for gfx906
//
// V2 optimizations vs V1:
//   1. Float4 vectorized loads/stores — 4× fewer memory transactions
//   2. Fast math: __expf → v_exp_f32 SFU, __frcp_rn → v_rcp_f32
//      - Replaces IEEE-754 division chain (v_div_scale+v_rcp+5×v_fma
//        +v_div_fmas+v_div_fixup = 9 insns, ~40 cycles) with single
//        v_rcp_f32 (~4 cycles)
//   3. Scalar tail handling for non-float4-aligned sizes
//
// ISA profile (V2 FP32, float4 body):
//   4× v_exp_f32 + 4× v_rcp_f32 + 4× v_mul + 4× v_fma = ~16 insns
//   loads: 2× global_load_dwordx4 = 2 memory ops for 8 floats
//   vs V1: 1× global_load_dword + full exp + IEEE div = ~30 insns per float
// =========================================================================

/**
 * @brief FP32 SwiGLU kernel — V2 with float4 vectorization + fast math
 */
__global__ void rocm_swiglu_fp32_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ output,
    int size)
{
    const int size4 = size >> 2;
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx < size4)
    {
        const float4 *gate4 = reinterpret_cast<const float4 *>(gate);
        const float4 *up4   = reinterpret_cast<const float4 *>(up);
        float4 *out4        = reinterpret_cast<float4 *>(output);

        float4 g = gate4[vec_idx];
        float4 u = up4[vec_idx];
        float4 r;
        r.x = silu_fast(g.x) * u.x;
        r.y = silu_fast(g.y) * u.y;
        r.z = silu_fast(g.z) * u.z;
        r.w = silu_fast(g.w) * u.w;
        out4[vec_idx] = r;
    }

    // Scalar tail for elements not covered by float4
    const int tail_start = size4 << 2;
    const int tail_idx = tail_start + (blockIdx.x * blockDim.x + threadIdx.x);
    if (tail_idx >= tail_start && tail_idx < size && vec_idx >= size4)
    {
        output[tail_idx] = silu_fast(gate[tail_idx]) * up[tail_idx];
    }
}

/**
 * @brief BF16 SwiGLU kernel — V2 with fast math
 */
__global__ void rocm_swiglu_bf16_kernel(
    const uint16_t *__restrict__ gate,
    const uint16_t *__restrict__ up,
    uint16_t *__restrict__ output,
    int size)
{
    // Vectorized: each thread processes 2 BF16 elements via uint32 loads/stores
    const int size2 = size >> 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size2)
    {
        const uint32_t* gate_u32 = reinterpret_cast<const uint32_t*>(gate);
        const uint32_t* up_u32   = reinterpret_cast<const uint32_t*>(up);
        uint32_t* out_u32        = reinterpret_cast<uint32_t*>(output);

        uint32_t g_packed = gate_u32[idx];
        uint32_t u_packed = up_u32[idx];

        float g0 = bf16_to_float(static_cast<uint16_t>(g_packed & 0xFFFF));
        float g1 = bf16_to_float(static_cast<uint16_t>(g_packed >> 16));
        float u0 = bf16_to_float(static_cast<uint16_t>(u_packed & 0xFFFF));
        float u1 = bf16_to_float(static_cast<uint16_t>(u_packed >> 16));

        uint32_t out_packed = static_cast<uint32_t>(float_to_bf16(silu_fast(g0) * u0)) |
                              (static_cast<uint32_t>(float_to_bf16(silu_fast(g1) * u1)) << 16);
        out_u32[idx] = out_packed;
    }

    // Scalar tail for odd size
    if ((size & 1) && idx == 0)
    {
        int last = size - 1;
        float g = bf16_to_float(gate[last]);
        float u = bf16_to_float(up[last]);
        output[last] = float_to_bf16(silu_fast(g) * u);
    }
}

/**
 * @brief FP16 SwiGLU kernel — V2 with fast math
 */
__global__ void rocm_swiglu_fp16_kernel(
    const uint16_t *__restrict__ gate,
    const uint16_t *__restrict__ up,
    uint16_t *__restrict__ output,
    int size)
{
    // Vectorized: each thread processes 2 FP16 elements via uint32 loads/stores
    // Output packing uses v_cvt_pkrtz_f16_f32 on GFX906
    const int size2 = size >> 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size2)
    {
        const uint32_t* gate_u32 = reinterpret_cast<const uint32_t*>(gate);
        const uint32_t* up_u32   = reinterpret_cast<const uint32_t*>(up);
        uint32_t* out_u32        = reinterpret_cast<uint32_t*>(output);

        uint32_t g_packed = gate_u32[idx];
        uint32_t u_packed = up_u32[idx];

        float g0 = fp16_to_float(static_cast<uint16_t>(g_packed & 0xFFFF));
        float g1 = fp16_to_float(static_cast<uint16_t>(g_packed >> 16));
        float u0 = fp16_to_float(static_cast<uint16_t>(u_packed & 0xFFFF));
        float u1 = fp16_to_float(static_cast<uint16_t>(u_packed >> 16));

        out_u32[idx] = pack_fp16_pair(silu_fast(g0) * u0, silu_fast(g1) * u1);
    }

    // Scalar tail for odd size
    if ((size & 1) && idx == 0)
    {
        int last = size - 1;
        float g = fp16_to_float(gate[last]);
        float u = fp16_to_float(up[last]);
        output[last] = float_to_fp16(silu_fast(g) * u);
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
        // Grid sized for float4 vectorized elements (+ tail handled internally)
        int size4 = size >> 2;
        int num_blocks = (size4 + threads_per_block - 1) / threads_per_block;
        // Ensure at least 1 block for tail elements
        if (num_blocks == 0 && size > 0) num_blocks = 1;

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
        int size2 = size >> 1;
        int num_blocks = (size2 + threads_per_block - 1) / threads_per_block;
        if (num_blocks == 0 && size > 0) num_blocks = 1;

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
        int size2 = size >> 1;
        int num_blocks = (size2 + threads_per_block - 1) / threads_per_block;
        if (num_blocks == 0 && size > 0) num_blocks = 1;

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
