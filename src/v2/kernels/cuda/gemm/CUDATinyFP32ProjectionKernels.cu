/**
 * @file CUDATinyFP32ProjectionKernels.cu
 * @brief Graph-capturable fixed-order small-N floating-point verifier projections.
 *
 * These kernels serve MTP verifier publication paths where grouped runtime-M rows
 * must match the backend's serial decode row contract.  They avoid cuBLAS
 * shape-dependent reduction choices for tiny output projections such as GDN
 * alpha/beta publication, while still grouping rows and projection batches in
 * one explicit-stream device path. Projection warps retain the legacy 256-way
 * arithmetic tree in registers, eliminating block barriers without changing
 * any serial, grouped-verifier, or prefill output bits. Large prefill batches
 * reuse native weights and FP32 activations in CTA-local shared tiles; they
 * never allocate another persistent weight or activation representation.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "CUDATinyProjectionSharedKernel.cuh"

#include "kernels/common/FloatingPointVerifierLaunch.h"

#include <cstdio>
#include <cstdint>

namespace
{
    constexpr int TINY_FP32_BLOCK = 256;
    constexpr int TINY_FP32_MAX_BATCH = 8;
    constexpr int TINY_PROJECTION_WARP_SIZE = 32;
    constexpr int TINY_PROJECTION_WARPS = TINY_FP32_BLOCK / TINY_PROJECTION_WARP_SIZE;
    // Measured crossover between the latency and throughput schedules. This is
    // launch geometry, never a different precision or a different reduction.
    constexpr int TINY_PROJECTION_PREFILL_MIN_ROWS = 32;
    // Captured floating-format tournaments place the shared-operand crossover
    // at large batches. Narrow columns and sub-tile K retain the warp schedule
    // because their unused lanes cannot amortize both CTA publication barriers.
    constexpr int TINY_PROJECTION_SHARED_MIN_ROWS = 512;
    constexpr int TINY_PROJECTION_SHARED_MIN_COLUMNS = 32;
    constexpr int TINY_PROJECTION_SHARED_MIN_K = 256;

    /** @brief By-value capture payload referencing already admitted persistent matrices. */
    struct Fp32BatchedProjectionPointers
    {
        const float *A[TINY_FP32_MAX_BATCH];
        const float *B[TINY_FP32_MAX_BATCH];
        float *C[TINY_FP32_MAX_BATCH];
    };

    /** @brief Native weight encodings accepted by the projection launch bridges. */
    enum TinyProjectionWeightDtype
    {
        TINY_PROJECTION_WEIGHT_FP32 = -1,
        TINY_PROJECTION_WEIGHT_FP16 = 0,
        TINY_PROJECTION_WEIGHT_BF16 = 1,
    };

    /** @brief Public SwiGLU bridge encodings, independent of the projection ABI. */
    enum TinySwiGLUDownWeightDtype
    {
        TINY_SWIGLU_DOWN_WEIGHT_FP32 = 0,
        TINY_SWIGLU_DOWN_WEIGHT_FP16 = 1,
        TINY_SWIGLU_DOWN_WEIGHT_BF16 = 2,
    };

    /**
     * @brief Convert IEEE FP16 bits to FP32 using device-side scalar logic.
     *
     * The fused SwiGLU-down kernel retains its established scalar conversion.
     * Tiny projections instead use the exactly equivalent finite-weight native
     * conversion instruction; the integration oracle checks all finite encodings.
     * @param h IEEE binary16 encoding.
     * @return Its FP32 representation.
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

    /** @brief Expand a BF16 encoding into FP32 without rounding. */
    __device__ __forceinline__ float bf16_bits_to_float(uint16_t bf16)
    {
        return __uint_as_float(static_cast<uint32_t>(bf16) << 16);
    }

    /**
     * @brief Read an unchanged persistent weight using its native encoding.
     * @tparam WeightDtype Encoding selected once by the public launch bridge.
     * @param weights Persistent native matrix, represented by the pointer-array ABI.
     * @param index Native element offset, not a byte offset.
     * @return Exactly expanded FP32 weight.
     */
    template <int WeightDtype>
    __device__ __forceinline__ float projection_weight(const float *weights, size_t index)
    {
        if constexpr (WeightDtype == TINY_PROJECTION_WEIGHT_FP32)
            return weights[index];
        else if constexpr (WeightDtype == TINY_PROJECTION_WEIGHT_BF16)
            return bf16_bits_to_float(reinterpret_cast<const uint16_t *>(weights)[index]);
        else
            // Native conversion is exact for every finite FP16 weight, including
            // subnormals. It avoids a divergent software exponent-normalizing loop.
            return __half2float(reinterpret_cast<const __half *>(weights)[index]);
    }

    /**
     * @brief Latency schedule for short batches, preserving the original tree.
     *
     * A 256-thread block distributes each output's K work across eight warps.
     * Short decode/verifier batches need that parallelism to occupy the device;
     * the register-tiled prefill schedule becomes economical with more rows.
     * @tparam WeightDtype Native FP32, FP16 or BF16 weight encoding.
     * @param d_A_array Device array of persistent FP32 activation matrices [M,K].
     * @param d_B_array Device array of persistent native weight matrices [N,K].
     * @param d_C_array Device array of disjoint FP32 output matrices [M,N].
     * @param M Number of activation rows.
     * @param N Number of output columns.
     * @param K Reduction width; each original thread visits stride-256 positions.
     */
    template <int WeightDtype>
    __global__ __launch_bounds__(TINY_FP32_BLOCK)
    void floating_tiny_block_projection_kernel(
        const float *const *d_A_array, const float *const *d_B_array,
        float *const *d_C_array, int M, int N, int K,
        llaminar2::DeviceRowRange row_range)
    {
        const int n = blockIdx.x, m = blockIdx.y, batch = blockIdx.z;
        const float *a = d_A_array[batch], *b = d_B_array[batch];
        float *c = d_C_array[batch];
        if (!a || !b || !c || m >= row_range.activeRows() || n >= N) return;
        float sum = 0.0f;
        for (int k = threadIdx.x; k < K; k += TINY_FP32_BLOCK)
            sum += a[static_cast<size_t>(m) * K + k] *
                   projection_weight<WeightDtype>(b, static_cast<size_t>(n) * K + k);
        __shared__ float parts[TINY_FP32_BLOCK];
        parts[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x < 32)
        {
            // The first warp evaluates the same cross-warp tree from the
            // published partials. Only that publication needs a block barrier;
            // the final five levels are warp-local and keep their old order.
            const int lane = threadIdx.x;
            const float low = (parts[lane] + parts[lane + 128]) +
                              (parts[lane + 64] + parts[lane + 192]);
            const float high = (parts[lane + 32] + parts[lane + 160]) +
                               (parts[lane + 96] + parts[lane + 224]);
            float result = low + high;
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1)
                result += __shfl_down_sync(0xffffffffu, result, stride);
            if (lane == 0) c[static_cast<size_t>(m) * N + n] = result;
        }
    }

    /**
     * @brief Load one down-projection weight as FP32 for fused verifier rows.
     *
     * The public C++ adapter stores the persistent weight pointer as `void *`.
     * This helper keeps the dtype-dependent reinterpret casts in one place so
     * the verifier kernel itself can keep a single fixed reduction body for
     * FP32, FP16, and BF16 weights.
     * @param weights Stable native down-projection matrix.
     * @param index Element offset in that native format.
     * @param weight_dtype Validated SwiGLU bridge encoding.
     * @return Exactly expanded FP32 weight.
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
     * @param ptrs By-value bundle of already allocated device addresses.
     * @param d_A_array Destination device activation-pointer array.
     * @param d_B_array Destination device weight-pointer array.
     * @param d_C_array Destination device output-pointer array.
     * @param batch_count Number of valid entries in the bundle.
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
     * @brief Project one output per warp with the fixed 256-part arithmetic tree.
     *
     * Lane L carries the old threads L, L+32, ..., L+224 in eight independent
     * accumulators. Each still visits K with stride 256. Register additions
     * implement the old strides 128, 64 and 32; warp shuffles implement strides
     * 16 through 1. Thus the physical launch changes but every FP32 addition
     * and multiply-add retains its original operands and parenthesization.
     * Eight warps produce eight neighboring columns without shared memory or
     * block barriers. Tail columns return as complete warps before shuffles.
     *
     * @tparam WeightDtype FP32, FP16 or BF16 persistent weight encoding.
     * @param d_A_array Device array of FP32 activation matrices [M,K].
     * @param d_B_array Device array of weight matrices [N,K].
     * @param d_C_array Device array of FP32 output matrices [M,N].
     * @param M Number of independent activation rows.
     * @param N Number of output columns per projection.
     * @param K Reduction width; positive unaligned tails are supported.
     */
    template <int WeightDtype>
    __global__ __launch_bounds__(TINY_FP32_BLOCK)
        void floating_tiny_warp_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            llaminar2::DeviceRowRange row_range)
    {
        constexpr int warp_size = TINY_PROJECTION_WARP_SIZE;
        constexpr int partials_per_lane = TINY_FP32_BLOCK / warp_size;
        const int lane = static_cast<int>(threadIdx.x) % warp_size;
        const int n = static_cast<int>(blockIdx.x) * TINY_PROJECTION_WARPS +
                      static_cast<int>(threadIdx.x) / warp_size;
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (m >= row_range.activeRows() || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sums[partials_per_lane] = {};
        const size_t activation_base = static_cast<size_t>(m) * K;
        const size_t weight_base = static_cast<size_t>(n) * K;
        for (int base = lane; base < K; base += TINY_FP32_BLOCK)
        {
            #pragma unroll
            for (int part = 0; part < partials_per_lane; ++part)
            {
                const int k = base + part * warp_size;
                if (k < K)
                {
                    const float weight = projection_weight<WeightDtype>(B, weight_base + k);
                    sums[part] += A[activation_base + k] * weight;
                }
            }
        }

        #pragma unroll
        for (int stride = partials_per_lane / 2; stride > 0; stride >>= 1)
        {
            #pragma unroll
            for (int part = 0; part < stride; ++part)
                sums[part] += sums[part + stride];
        }
        float result = sums[0];
        #pragma unroll
        for (int stride = warp_size / 2; stride > 0; stride >>= 1)
            result += __shfl_down_sync(0xffffffffu, result, stride);
        if (lane == 0)
            C[static_cast<size_t>(m) * N + n] = result;
    }

    /**
     * @brief Select a launch schedule without changing arithmetic or native storage.
     *
     * Decode/verifier batches distribute one dot product across a block; prefill
     * gives each warp a complete dot product. Large batches share both operands
     * across output rows/columns. Every schedule has the same stride-256 tree
     * and needs no persistent workspace. FP16 uses a smaller row tile because
     * its compiled conversion/load schedule favors four rows over eight.
     * @tparam WeightDtype Native weight encoding.
     * @param a Device activation-pointer array.
     * @param b Device weight-pointer array.
     * @param c Device output-pointer array with disjoint matrix destinations.
     * @param m Activation rows.
     * @param n Output columns.
     * @param k Reduction width.
     * @param batches Number of independent projections.
     * @param stream Exact non-null stream validated by the public launch bridge.
     */
    template <int WeightDtype>
    void launch_tiny_projection(const float *const *a, const float *const *b,
                                float *const *c, int m, int n, int k,
                                int batches, cudaStream_t stream,
                                llaminar2::DeviceRowRange row_range)
    {
        if (m < TINY_PROJECTION_PREFILL_MIN_ROWS)
            floating_tiny_block_projection_kernel<WeightDtype>
                <<<dim3(n, m, batches), TINY_FP32_BLOCK, 0, stream>>>(a, b, c, m, n, k, row_range);
        else if (m >= TINY_PROJECTION_SHARED_MIN_ROWS &&
                 n >= TINY_PROJECTION_SHARED_MIN_COLUMNS &&
                 k >= TINY_PROJECTION_SHARED_MIN_K)
        {
            using Weight = std::conditional_t<WeightDtype == TINY_PROJECTION_WEIGHT_FP32,
                float, std::conditional_t<WeightDtype == TINY_PROJECTION_WEIGHT_FP16,
                                         __half, __nv_bfloat16>>;
            constexpr int rows = WeightDtype == TINY_PROJECTION_WEIGHT_FP16 ? 4 : 8;
            llaminar2::cuda::detail::sharedTinyProjectionKernel<Weight, rows>
                <<<dim3((n-1)/TINY_PROJECTION_WARPS+1, (m-1)/rows+1, batches),
                   TINY_FP32_BLOCK, 0, stream>>>(a, b, c, m, n, k, row_range);
        }
        else
            floating_tiny_warp_projection_kernel<WeightDtype>
                <<<dim3((n + TINY_PROJECTION_WARPS - 1) / TINY_PROJECTION_WARPS, m, batches),
                   TINY_FP32_BLOCK, 0, stream>>>(a, b, c, m, n, k, row_range);
    }

    /**
     * @brief Fixed-order fused SwiGLU plus floating down projection.
     *
     * This is the floating-weight counterpart to the quantized grouped verifier
     * down path.  It intentionally does not materialize the intermediate
     * `silu(gate) * up` row: every output element walks K in the same strided
     * order for serial M=1 and grouped runtime-M, computes the SwiGLU product at
     * the same point in that loop, then reduces through the same block tree.
     * @param gate FP32 gate values [M,K].
     * @param up FP32 up values [M,K].
     * @param weights Native down-projection weights [N,K].
     * @param output FP32 destinations [M,N].
     * @param M Number of independent rows.
     * @param N Number of output columns.
     * @param K Fixed-order reduction width.
     * @param weight_dtype Validated SwiGLU bridge encoding.
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

/**
 * @brief Publish stable projection pointers through a capturable explicit-stream kernel.
 * @param d_A_array Persistent device storage for activation pointers.
 * @param d_B_array Persistent device storage for weight pointers.
 * @param d_C_array Persistent device storage for output pointers.
 * @param h_A_ptrs Host-known activation addresses embedded by value in the launch.
 * @param h_B_ptrs Host-known weight addresses embedded by value in the launch.
 * @param h_C_ptrs Host-known output addresses embedded by value in the launch.
 * @param batch_count Number of valid entries, from one through eight.
 * @param device_id Owning CUDA device.
 * @param stream Exact non-null CUDA stream.
 * @return False with a diagnostic on invalid input or CUDA submission failure.
 */
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

/**
 * @brief Enqueue fixed-order FP32 projections without allocation or synchronization.
 * @param d_A_array Device pointers to FP32 activation matrices [M,K].
 * @param d_B_array Device pointers to FP32 weight matrices [N,K].
 * @param d_C_array Device pointers to disjoint FP32 destinations [M,N].
 * @param M Positive activation-row count.
 * @param N Positive output-column count.
 * @param K Positive reduction width.
 * @param batch_count Projection count in [1,8].
 * @param device_id Owning device.
 * @param stream Exact non-null CUDA stream ordering pointer publication and execution.
 * @return False with a diagnostic on invalid input or CUDA submission failure.
 */
extern "C" bool cudaFp32_tiny_batched_projection(
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

    launch_tiny_projection<TINY_PROJECTION_WEIGHT_FP32>(
        d_A_array, d_B_array, d_C_array, M, N, K, batch_count, static_cast<cudaStream_t>(stream), rows);

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

/**
 * @brief Enqueue native FP16/BF16-weight projections with unchanged FP32 arithmetic.
 * @param d_A_array Device pointers to FP32 activation matrices [M,K].
 * @param d_B_array Device pointers to native 16-bit weight matrices [N,K].
 * @param d_C_array Device pointers to disjoint FP32 destinations [M,N].
 * @param M Positive activation-row count.
 * @param N Positive output-column count.
 * @param K Positive reduction width.
 * @param batch_count Projection count in [1,8].
 * @param weight_dtype Existing bridge encoding: zero for FP16, one for BF16.
 * @param device_id Owning device.
 * @param stream Exact non-null CUDA stream ordering pointer publication and execution.
 * @return False with a diagnostic on invalid input or CUDA submission failure.
 */
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

    if (weight_dtype == TINY_PROJECTION_WEIGHT_BF16)
        launch_tiny_projection<TINY_PROJECTION_WEIGHT_BF16>(
            d_A_array, d_B_array, d_C_array, M, N, K, batch_count, static_cast<cudaStream_t>(stream), rows);
    else
        launch_tiny_projection<TINY_PROJECTION_WEIGHT_FP16>(
            d_A_array, d_B_array, d_C_array, M, N, K, batch_count, static_cast<cudaStream_t>(stream), rows);

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

/**
 * @brief Enqueue the fixed-order floating SwiGLU/down verifier kernel.
 * @param d_gate Persistent FP32 gate matrix [M,K].
 * @param d_up Persistent FP32 up matrix [M,K].
 * @param d_weights Persistent native weight matrix [N,K].
 * @param d_output Persistent FP32 output matrix [M,N].
 * @param M Positive row count.
 * @param N Positive column count.
 * @param K Positive reduction width.
 * @param weight_dtype SwiGLU ABI encoding: FP32=0, FP16=1, BF16=2.
 * @param device_id Owning CUDA device.
 * @param stream Exact non-null stream ordering inputs and execution.
 * @return False with a diagnostic on invalid input or CUDA submission failure.
 */
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
        weight_dtype,
        rows);

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
