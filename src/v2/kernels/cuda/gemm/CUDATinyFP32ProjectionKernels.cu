/**
 * @file CUDATinyFP32ProjectionKernels.cu
 * @brief Graph-capturable fixed-order small-N floating-point verifier projections.
 *
 * These kernels serve MTP verifier publication paths where grouped runtime-M rows
 * must match the backend's serial decode row contract.  They avoid cuBLAS
 * shape-dependent reduction choices for tiny output projections such as GDN
 * alpha/beta publication, while still grouping rows and projection batches in
 * one explicit-stream device path.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

namespace
{
    constexpr int TINY_FP32_BLOCK = 256;
    constexpr int TINY_FP32_MAX_BATCH = 8;

    struct Fp32BatchedProjectionPointers
    {
        const float *A[TINY_FP32_MAX_BATCH];
        const float *B[TINY_FP32_MAX_BATCH];
        float *C[TINY_FP32_MAX_BATCH];
    };

    enum TinyProjectionWeightDtype
    {
        TINY_PROJECTION_WEIGHT_FP16 = 0,
        TINY_PROJECTION_WEIGHT_BF16 = 1,
    };

    enum TinySwiGLUDownWeightDtype
    {
        TINY_SWIGLU_DOWN_WEIGHT_FP32 = 0,
        TINY_SWIGLU_DOWN_WEIGHT_FP16 = 1,
        TINY_SWIGLU_DOWN_WEIGHT_BF16 = 2,
    };

    /**
     * @brief Convert IEEE FP16 bits to FP32 using device-side scalar logic.
     *
     * The verifier kernels use explicit bit conversion instead of relying on a
     * library GEMM path so serial and grouped rows walk the same K loop and convert
     * each weight at the same point in that loop.
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
        return __uint_as_float(f);
    }

    __device__ __forceinline__ float bf16_bits_to_float(uint16_t bf16)
    {
        return __uint_as_float(static_cast<uint32_t>(bf16) << 16);
    }

    /**
     * @brief Load one down-projection weight as FP32 for fused verifier rows.
     *
     * The public C++ adapter stores the persistent weight pointer as `void *`.
     * This helper keeps the dtype-dependent reinterpret casts in one place so
     * the verifier kernel itself can keep a single fixed reduction body for
     * FP32, FP16, and BF16 weights.
     */
    __device__ __forceinline__ float load_swiglu_down_weight(
        const void *__restrict__ weights,
        size_t index,
        int weight_dtype)
    {
        if (weight_dtype == TINY_SWIGLU_DOWN_WEIGHT_FP32)
            return reinterpret_cast<const float *>(weights)[index];

        const uint16_t bits = reinterpret_cast<const uint16_t *>(weights)[index];
        return (weight_dtype == TINY_SWIGLU_DOWN_WEIGHT_BF16)
                   ? bf16_bits_to_float(bits)
                   : fp16_bits_to_float(bits);
    }

    /**
     * @brief Stage host-known projection pointers into workspace-owned device arrays.
     *
     * CUDA graph capture cannot safely depend on ad-hoc host-to-device copies or
     * hidden allocations.  The caller passes stable graph/workspace buffers, and
     * this tiny kernel writes the pointer arrays on the same explicit stream that
     * will run the projection kernel.
     */
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

    /**
     * @brief Deterministic small-N FP32 projection for GDN state publishers.
     *
     * Qwen3.6 GDN alpha/beta projections are FP32 with very small output width.
     * cuBLAS may choose different accumulation schedules as M changes, and even
     * sub-ULP alpha/beta drift can be amplified by recurrent state and prefix
     * cache restore.  This kernel uses one fixed reduction tree for every row,
     * so short cached-prefix blocks and longer full-prefill blocks publish the
     * same per-row values.
     */
    __global__ __launch_bounds__(TINY_FP32_BLOCK)
        void fp32_tiny_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= M || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += TINY_FP32_BLOCK)
        {
            sum += A[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                   B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
        }

        __shared__ float scratch[TINY_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = TINY_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }

    /**
     * @brief Deterministic small-N projection for FP32 activations and 16-bit weights.
     *
     * CUDAFloatingPointGemmKernel currently keeps hidden activations and
     * projection outputs in FP32 even when the persistent weight tensor is FP16
     * or BF16.  This kernel implements the grouped verifier contract for those
     * weight formats without routing through cuBLAS.  Every output element owns
     * one block, each thread accumulates a fixed strided K subsequence, and the
     * block reduction order is identical for serial M=1 and grouped runtime-M.
     */
    __global__ __launch_bounds__(TINY_FP32_BLOCK)
        void fp32x16_tiny_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            int weight_dtype)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= M || n >= N)
            return;

        const float *A = d_A_array[batch];
        const uint16_t *B = reinterpret_cast<const uint16_t *>(d_B_array[batch]);
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += TINY_FP32_BLOCK)
        {
            const uint16_t weight_bits =
                B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
            const float weight =
                (weight_dtype == TINY_PROJECTION_WEIGHT_BF16)
                    ? bf16_bits_to_float(weight_bits)
                    : fp16_bits_to_float(weight_bits);
            sum += A[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                   weight;
        }

        __shared__ float scratch[TINY_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = TINY_FP32_BLOCK / 2; stride > 0; stride >>= 1)
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
     * This is the floating-weight counterpart to the quantized grouped verifier
     * down path.  It intentionally does not materialize the intermediate
     * `silu(gate) * up` row: every output element walks K in the same strided
     * order for serial M=1 and grouped runtime-M, computes the SwiGLU product at
     * the same point in that loop, then reduces through the same block tree.
     */
    __global__ __launch_bounds__(TINY_FP32_BLOCK)
        void floating_swiglu_down_projection_kernel(
            const float *__restrict__ gate,
            const float *__restrict__ up,
            const void *__restrict__ weights,
            float *__restrict__ output,
            int M,
            int N,
            int K,
            int weight_dtype)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        if (m >= M || n >= N)
            return;

        float sum = 0.0f;
        const size_t row_base = static_cast<size_t>(m) * static_cast<size_t>(K);
        const size_t weight_base = static_cast<size_t>(n) * static_cast<size_t>(K);
        for (int k = static_cast<int>(threadIdx.x); k < K; k += TINY_FP32_BLOCK)
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

        __shared__ float scratch[TINY_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = TINY_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }
} // namespace

extern "C" bool cudaFp32_stage_batched_projection_pointers(
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
        batch_count <= 0 || batch_count > TINY_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_stage_batched_projection_pointers] invalid arguments: batch=%d stream=%p\n",
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
                "[cudaFp32_stage_batched_projection_pointers] null pointer in batch %d\n",
                i);
            return false;
        }
        ptrs.A[i] = h_A_ptrs[i];
        ptrs.B[i] = h_B_ptrs[i];
        ptrs.C[i] = h_C_ptrs[i];
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_stage_batched_projection_pointers] cudaSetDevice failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    fp32_stage_batched_projection_pointers_kernel<<<1, TINY_FP32_MAX_BATCH, 0, cuda_stream>>>(
        ptrs,
        d_A_array,
        d_B_array,
        d_C_array,
        batch_count);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_stage_batched_projection_pointers] launch failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }
    return true;
}

extern "C" bool cudaFp32_tiny_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int device_id,
    void *stream)
{
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        batch_count <= 0 || batch_count > TINY_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_tiny_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d stream=%p\n",
            M,
            N,
            K,
            batch_count,
            stream);
        return false;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_tiny_batched_projection] cudaSetDevice failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(TINY_FP32_BLOCK);
    fp32_tiny_batched_projection_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_A_array,
        d_B_array,
        d_C_array,
        M,
        N,
        K);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32_tiny_batched_projection] launch failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }
    return true;
}

extern "C" bool cudaFp32x16_tiny_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int weight_dtype,
    int device_id,
    void *stream)
{
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        batch_count <= 0 || batch_count > TINY_FP32_MAX_BATCH ||
        (weight_dtype != TINY_PROJECTION_WEIGHT_FP16 &&
         weight_dtype != TINY_PROJECTION_WEIGHT_BF16) ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[cudaFp32x16_tiny_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d dtype=%d stream=%p\n",
            M,
            N,
            K,
            batch_count,
            weight_dtype,
            stream);
        return false;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32x16_tiny_batched_projection] cudaSetDevice failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(TINY_FP32_BLOCK);
    fp32x16_tiny_batched_projection_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_A_array,
        d_B_array,
        d_C_array,
        M,
        N,
        K,
        weight_dtype);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFp32x16_tiny_batched_projection] launch failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }
    return true;
}

extern "C" bool cudaFloating_swiglu_down_projection(
    const float *d_gate,
    const float *d_up,
    const void *d_weights,
    float *d_output,
    int M,
    int N,
    int K,
    int weight_dtype,
    int device_id,
    void *stream)
{
    if (!d_gate || !d_up || !d_weights || !d_output ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        (weight_dtype != TINY_SWIGLU_DOWN_WEIGHT_FP32 &&
         weight_dtype != TINY_SWIGLU_DOWN_WEIGHT_FP16 &&
         weight_dtype != TINY_SWIGLU_DOWN_WEIGHT_BF16) ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[cudaFloating_swiglu_down_projection] invalid arguments: gate=%p up=%p weights=%p output=%p M=%d N=%d K=%d dtype=%d stream=%p\n",
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

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFloating_swiglu_down_projection] cudaSetDevice failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), 1);
    dim3 block(TINY_FP32_BLOCK);
    floating_swiglu_down_projection_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_gate,
        d_up,
        d_weights,
        d_output,
        M,
        N,
        K,
        weight_dtype);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "[cudaFloating_swiglu_down_projection] launch failed: %s\n",
            cudaGetErrorString(err));
        return false;
    }
    return true;
}
