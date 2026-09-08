/**
 * @file ROCmSmallNFP32ProjectionKernels.hip
 * @brief Graph-capturable fixed-order small/output floating verifier projections.
 *
 * These kernels back MTP verifier publication for floating-point GDN-style
 * projections. They keep M=1 serial decode and runtime grouped verifier rows
 * on the same explicit HIP stream and the same reduction tree, avoiding
 * hipBLAS shape-dependent accumulation changes in publication-sensitive state.
 */

#include <hip/hip_runtime.h>

#include "kernels/common/FloatingPointVerifierLaunch.h"

#include <cstdio>
#include <cstdint>

#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"

namespace
{
    constexpr int SMALL_N_FP32_BLOCK = 256;
    constexpr int SMALL_N_FP32_MAX_BATCH = 8;
    constexpr int SMALL_N_FP32_MAX_N = 64;

    struct Fp32BatchedProjectionPointers
    {
        const float *A[SMALL_N_FP32_MAX_BATCH];
        const float *B[SMALL_N_FP32_MAX_BATCH];
        float *C[SMALL_N_FP32_MAX_BATCH];
    };

    enum SmallProjectionWeightDtype
    {
        SMALL_PROJECTION_WEIGHT_FP16 = 0,
        SMALL_PROJECTION_WEIGHT_BF16 = 1,
    };

    enum SmallSwiGLUDownWeightDtype
    {
        SMALL_SWIGLU_DOWN_WEIGHT_FP32 = 0,
        SMALL_SWIGLU_DOWN_WEIGHT_FP16 = 1,
        SMALL_SWIGLU_DOWN_WEIGHT_BF16 = 2,
    };

    __device__ __forceinline__ float uint32_bits_to_float(uint32_t bits)
    {
        union
        {
            uint32_t u;
            float f;
        } value{bits};
        return value.f;
    }

    /**
     * @brief Convert IEEE FP16 bits to FP32 in device scalar code.
     *
     * ROCm verifier rows use this conversion inside the fixed K loop so the
     * grouped path and serial M=1 path cannot diverge through different BLAS
     * kernels or hardware reduction policies.
     */
    __device__ __forceinline__ float fp16_bits_to_float(uint16_t h)
    {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        int32_t exp = (static_cast<int32_t>(h) & 0x7C00) >> 10;
        uint32_t mantissa = static_cast<uint32_t>(h) & 0x03FFu;

        uint32_t f = 0;
        if (exp == 0)
        {
            if (mantissa == 0)
            {
                f = sign;
            }
            else
            {
                exp = 1;
                while ((mantissa & 0x400u) == 0)
                {
                    mantissa <<= 1;
                    --exp;
                }
                mantissa &= 0x3FFu;
                f = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1F)
        {
            f = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            f = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mantissa << 13);
        }
        return uint32_bits_to_float(f);
    }

    __device__ __forceinline__ float bf16_bits_to_float(uint16_t bf16)
    {
        return uint32_bits_to_float(static_cast<uint32_t>(bf16) << 16);
    }

    /**
     * @brief Load one FP32/FP16/BF16 down weight as FP32.
     *
     * The fused verifier kernel receives the persistent weight allocation
     * through the common floating GEMM adapter as an untyped device pointer.
     * Keeping the dtype switch in this helper lets the hot loop share one
     * fixed K traversal and one reduction tree across all floating formats.
     */
    __device__ __forceinline__ float load_swiglu_down_weight(
        const void *__restrict__ weights,
        size_t index,
        int weight_dtype)
    {
        if (weight_dtype == SMALL_SWIGLU_DOWN_WEIGHT_FP32)
            return reinterpret_cast<const float *>(weights)[index];

        const uint16_t bits = reinterpret_cast<const uint16_t *>(weights)[index];
        return (weight_dtype == SMALL_SWIGLU_DOWN_WEIGHT_BF16)
                   ? bf16_bits_to_float(bits)
                   : fp16_bits_to_float(bits);
    }

    bool stream_capture_active(hipStream_t stream)
    {
        if (llaminar2::isGraphCaptureActive())
            return true;

        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t err = hipStreamIsCapturing(stream, &status);
        return err == hipSuccess && status != hipStreamCaptureStatusNone;
    }

    bool complete_kernel_launch(hipStream_t stream, const char *name)
    {
        if (stream_capture_active(stream))
            return true;

        const hipError_t err = hipGetLastError();
        if (err == hipSuccess)
            return true;

        std::fprintf(stderr, "[%s] launch failed: %s\n", name, hipGetErrorString(err));
        return false;
    }

    __global__ void fp32_stage_batched_projection_pointers_kernel(
        const Fp32BatchedProjectionPointers ptrs,
        const float **__restrict__ d_A_array,
        const float **__restrict__ d_B_array,
        float **__restrict__ d_C_array,
        int batch_count)
    {
        const int batch = static_cast<int>(threadIdx.x);
        if (batch >= batch_count)
            return;

        d_A_array[batch] = ptrs.A[batch];
        d_B_array[batch] = ptrs.B[batch];
        d_C_array[batch] = ptrs.C[batch];
    }

    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void fp32_small_n_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            llaminar2::DeviceRowRange row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= row_range.activeRows() || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            sum += A[static_cast<size_t>(m) * K + k] *
                   B[static_cast<size_t>(n) * K + k];
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * N + n] = scratch[0];
    }

    /**
     * @brief Fixed-order projection for FP32 activation rows and 16-bit weights.
     *
     * The current GPU floating GEMM adapters keep hidden rows and outputs in
     * FP32.  This kernel adds the missing grouped verifier implementation for
     * FP16/BF16 persistent weight tensors by reading the raw 16-bit weight rows
     * directly and converting each element inside the same scalar K loop used
     * for serial decode.
     */
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void fp32x16_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            int weight_dtype,
            llaminar2::DeviceRowRange row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= row_range.activeRows() || n >= N)
            return;

        const float *A = d_A_array[batch];
        const uint16_t *B = reinterpret_cast<const uint16_t *>(d_B_array[batch]);
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            const uint16_t weight_bits =
                B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
            const float weight =
                (weight_dtype == SMALL_PROJECTION_WEIGHT_BF16)
                    ? bf16_bits_to_float(weight_bits)
                    : fp16_bits_to_float(weight_bits);
            sum += A[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                   weight;
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }

    /**
     * @brief Fixed-order fused SwiGLU plus floating down projection.
     *
     * MTP shared-expert verifier rows publish the result of `down(swiglu(g,u))`
     * into live state. This kernel keeps serial M=1 and runtime grouped rows on
     * the same HIP code path: one block owns one output element, every thread
     * walks a deterministic strided K subsequence, and the block reduction tree
     * is identical for all M values.
     */
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void floating_swiglu_down_projection_kernel(
            const float *__restrict__ gate,
            const float *__restrict__ up,
            const void *__restrict__ weights,
            float *__restrict__ output,
            int M,
            int N,
            int K,
            int weight_dtype,
            llaminar2::DeviceRowRange row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        if (m >= row_range.activeRows() || n >= N)
            return;

        float sum = 0.0f;
        const size_t row_base = static_cast<size_t>(m) * static_cast<size_t>(K);
        const size_t weight_base = static_cast<size_t>(n) * static_cast<size_t>(K);
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            const float gate_value = gate[row_base + static_cast<size_t>(k)];
            const float up_value = up[row_base + static_cast<size_t>(k)];
            const float swiglu_value = (gate_value / (1.0f + expf(-gate_value))) * up_value;
            const float weight_value = load_swiglu_down_weight(
                weights,
                weight_base + static_cast<size_t>(k),
                weight_dtype);
            sum += swiglu_value * weight_value;
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }
} // namespace

extern "C" bool rocmFp32_stage_batched_projection_pointers(
    const float **d_A_array,
    const float **d_B_array,
    float **d_C_array,
    const float *const *h_A_ptrs,
    const float *const *h_B_ptrs,
    float *const *h_C_ptrs,
    int batch_count,
    int device_id,
    void *stream)
{
    if (!d_A_array || !d_B_array || !d_C_array ||
        !h_A_ptrs || !h_B_ptrs || !h_C_ptrs ||
        batch_count <= 0 || batch_count > SMALL_N_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_stage_batched_projection_pointers] invalid arguments: batch=%d stream=%p\n",
            batch_count,
            stream);
        return false;
    }

    Fp32BatchedProjectionPointers ptrs{};
    for (int i = 0; i < batch_count; ++i)
    {
        if (!h_A_ptrs[i] || !h_B_ptrs[i] || !h_C_ptrs[i])
        {
            std::fprintf(
                stderr,
                "[rocmFp32_stage_batched_projection_pointers] null pointer in batch %d\n",
                i);
            return false;
        }
        ptrs.A[i] = h_A_ptrs[i];
        ptrs.B[i] = h_B_ptrs[i];
        ptrs.C[i] = h_C_ptrs[i];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_stage_batched_projection_pointers] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        fp32_stage_batched_projection_pointers_kernel,
        dim3(1),
        dim3(SMALL_N_FP32_MAX_BATCH),
        0,
        hip_stream,
        ptrs,
        d_A_array,
        d_B_array,
        d_C_array,
        batch_count);

    return complete_kernel_launch(hip_stream, "rocmFp32_stage_batched_projection_pointers");
}

extern "C" bool rocmFp32_small_n_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int device_id,
    void *stream,
    const llaminar2::DeviceRowRange *row_range)
{
    if (row_range && row_range->physicalRows() != M)
        return false;
    const auto rows = row_range ? *row_range : llaminar2::DeviceRowRange::fullyActive(M);
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 ||
        N <= 0 || N > SMALL_N_FP32_MAX_N ||
        K <= 0 ||
        batch_count <= 0 || batch_count > SMALL_N_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_small_n_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d stream=%p\n",
            M,
            N,
            K,
            batch_count,
            stream);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32_small_n_batched_projection] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(SMALL_N_FP32_BLOCK);
    hipLaunchKernelGGL(
        fp32_small_n_batched_projection_kernel,
        grid,
        block,
        0,
        hip_stream,
        d_A_array,
        d_B_array,
        d_C_array,
        M,
        N,
        K,
        rows);

    return complete_kernel_launch(hip_stream, "rocmFp32_small_n_batched_projection");
}

extern "C" bool rocmFp32x16_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int weight_dtype,
    int device_id,
    void *stream,
    const llaminar2::DeviceRowRange *row_range)
{
    if (row_range && row_range->physicalRows() != M)
        return false;
    const auto rows = row_range ? *row_range : llaminar2::DeviceRowRange::fullyActive(M);
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        batch_count <= 0 || batch_count > SMALL_N_FP32_MAX_BATCH ||
        (weight_dtype != SMALL_PROJECTION_WEIGHT_FP16 &&
         weight_dtype != SMALL_PROJECTION_WEIGHT_BF16) ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFp32x16_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d dtype=%d stream=%p\n",
            M,
            N,
            K,
            batch_count,
            weight_dtype,
            stream);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFp32x16_batched_projection] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(SMALL_N_FP32_BLOCK);
    hipLaunchKernelGGL(
        fp32x16_batched_projection_kernel,
        grid,
        block,
        0,
        hip_stream,
        d_A_array,
        d_B_array,
        d_C_array,
        M,
        N,
        K,
        weight_dtype,
        rows);

    return complete_kernel_launch(hip_stream, "rocmFp32x16_batched_projection");
}

extern "C" bool rocmFloating_swiglu_down_projection(
    const float *d_gate,
    const float *d_up,
    const void *d_weights,
    float *d_output,
    int M,
    int N,
    int K,
    int weight_dtype,
    int device_id,
    void *stream,
    const llaminar2::DeviceRowRange *row_range)
{
    if (row_range && row_range->physicalRows() != M)
        return false;
    const auto rows = row_range ? *row_range : llaminar2::DeviceRowRange::fullyActive(M);
    if (!d_gate || !d_up || !d_weights || !d_output ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        (weight_dtype != SMALL_SWIGLU_DOWN_WEIGHT_FP32 &&
         weight_dtype != SMALL_SWIGLU_DOWN_WEIGHT_FP16 &&
         weight_dtype != SMALL_SWIGLU_DOWN_WEIGHT_BF16) ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFloating_swiglu_down_projection] invalid arguments: gate=%p up=%p weights=%p output=%p M=%d N=%d K=%d dtype=%d stream=%p\n",
            static_cast<const void *>(d_gate),
            static_cast<const void *>(d_up),
            d_weights,
            static_cast<void *>(d_output),
            M,
            N,
            K,
            weight_dtype,
            stream);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = hipSetDevice(device_id);
    if (err != hipSuccess)
    {
        std::fprintf(
            stderr,
            "[rocmFloating_swiglu_down_projection] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        floating_swiglu_down_projection_kernel,
        dim3(static_cast<unsigned>(N), static_cast<unsigned>(M), 1),
        dim3(SMALL_N_FP32_BLOCK),
        0,
        hip_stream,
        d_gate,
        d_up,
        d_weights,
        d_output,
        M,
        N,
        K,
        weight_dtype,
        rows);

    return complete_kernel_launch(hip_stream, "rocmFloating_swiglu_down_projection");
}
