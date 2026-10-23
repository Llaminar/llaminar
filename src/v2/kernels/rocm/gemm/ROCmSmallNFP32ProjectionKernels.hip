/**
 * @file ROCmSmallNFP32ProjectionKernels.hip
 * @brief Graph-capturable fixed-order floating verifier and replicated-head projections.
 *
 * These kernels back MTP verifier publication for floating-point GDN-style
 * projections. They keep M=1 serial decode and runtime grouped verifier rows
 * on the same explicit HIP stream and the same reduction tree, avoiding
 * hipBLAS shape-dependent accumulation changes in publication-sensitive state.
 * Each output column has the same 256-lane K reduction regardless of physical
 * output width, so narrow shards and a mirrored full vocabulary remain exact.
 * Custom FP32/native16 consumers retain immutable matrix addresses by value;
 * pointer-table publication remains only for genuine table consumers.
 * Explicit half activation specializations read only live rows from the existing
 * retained half bank; serial and verifier dispatch share the same arithmetic.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "kernels/common/FloatingProjectionRows.h"
#include "kernels/rocm/gemm/ROCmFP32ProjectionLoadPolicy.h"

#include <cstdio>
#include <cstdint>

#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"

namespace
{
    constexpr int SMALL_N_FP32_BLOCK = 256;
    constexpr int SMALL_N_FP32_MAX_BATCH = 8;

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
     * @brief Expand every IEEE half word exactly using native widening.
     * @param h Physical half word; no activation or weight requantization occurs.
     * @return The exact legacy FP32 bits, including signed zero and NaN payload.
     *
     * Native widening handles all finite words and infinities exactly. Signaling
     * NaNs use the original bit construction so conversion itself cannot quiet
     * their payload. This replaces exponent/normalization branches in each K
     * iteration without changing any product or reduction.
     */
    __device__ __forceinline__ float fp16_bits_to_float(uint16_t h)
    {
        const uint32_t native = __float_as_uint(__half2float(__ushort_as_half(h)));
        const uint32_t original_nan = (uint32_t(h & 0x8000u) << 16) |
            0x7F800000u | (uint32_t(h & 0x03FFu) << 13);
        const uint32_t nan_mask = 0u - uint32_t((h & 0x7FFFu) > 0x7C00u);
        // A bit select retains every NaN payload without divergent EXEC masks.
        return uint32_bits_to_float(native ^ ((native ^ original_nan) & nan_mask));
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

    /**
     * @brief Preserve the original 256-part tree with a native wave64 tail.
     * @param value Original stride-256 lane accumulator, including zero K tails.
     * @param scratch Exclusive 256-float CTA storage, populated by every lane.
     * @return Original reduced result in lane zero; other lanes are not consumed.
     * @pre Exactly 256 threads enter collectively; no participant returns early.
     *
     * LDS retains edges128/64 and their producer barriers. All64 lanes of
     * the first wave then execute every shuffle; only original owning lanes
     * perform additions. No multiply or rounded partial is reassociated.
     */
    __device__ __forceinline__ float reduceFloating256(float value, float *scratch)
    {
        const unsigned lane = threadIdx.x;
        scratch[lane] = value;
        __syncthreads();
        for (unsigned stride = 128; stride >= 64; stride >>= 1)
        {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
        }
        if (lane < 64)
        {
            value = scratch[lane];
#pragma unroll
            for (unsigned stride = 32; stride; stride >>= 1)
            {
                const float partner = __shfl_down(value, stride, 64);
                if (lane < stride) value += partner;
            }
        }
        return value;
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

    /**
     * @brief Project FP32 operands with the original lane and reduction ownership.
     * @param projections Immutable input, weight and output bindings per projection.
     * @param M Admitted physical input/output rows.
     * @param N Output columns owned by this participant.
     * @param K Number of original FP32 products in each dot product.
     * @param row_range Device-owned live-row selection for this invocation.
     *
     * @tparam LOAD_PAIRS Compiled scalar, four-pair or eight-pair scheduling.
     * Each lane consumes k=lane+256*j in ascending order with one accumulator.
     * Complete load groups expose independent operand requests; scalar tails
     * preserve partial K without padded reads. Scalar scheduling retains the
     * original loop. Selection depends only on the immutable captured CTA grid.
     */
    template<int LOAD_PAIRS>
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void fp32_small_n_batched_projection_kernel(
            llaminar2::FixedFP32ProjectionGroup projections,
            int M,
            int N,
            int K,
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = projections.input(batch);
        const float *B = projections.weight(batch);
        float *C = projections.output(batch);
        if (!A || !B || !C)
            return;

        static_assert(LOAD_PAIRS == 1 || LOAD_PAIRS == 4 || LOAD_PAIRS == 8);
        float sum = 0.0f;
        if constexpr (LOAD_PAIRS == 1)
        {
            for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
            {
                sum += A[static_cast<size_t>(m) * K + k] *
                       B[static_cast<size_t>(n) * K + k];
            }
        }
        else
        {
            const float *input_row = A + static_cast<size_t>(m) * K;
            const float *weight_row = B + static_cast<size_t>(n) * K;
            int k = static_cast<int>(threadIdx.x);
            constexpr int LOAD_GROUP = LOAD_PAIRS * SMALL_N_FP32_BLOCK;
            const int grouped_end = K - K % LOAD_GROUP;
            for (; k < grouped_end; k += LOAD_GROUP)
            {
                float input[LOAD_PAIRS];
                float weight[LOAD_PAIRS];
#pragma unroll
                for (int pair = 0; pair < LOAD_PAIRS; ++pair)
                {
                    input[pair] = input_row[k + pair * SMALL_N_FP32_BLOCK];
                    weight[pair] = weight_row[k + pair * SMALL_N_FP32_BLOCK];
                }
                // One ordered chain, not independent partial accumulators: grouped rows
                // and serial decode must retain every original FP32 rounding step.
#pragma unroll
                for (int pair = 0; pair < LOAD_PAIRS; ++pair)
                    sum += input[pair] * weight[pair];
            }
            for (; k < K; k += SMALL_N_FP32_BLOCK)
            {
                sum += input_row[k] * weight_row[k];
            }
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        const float reduced = reduceFloating256(sum, scratch);

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * N + n] = reduced;
    }

    /**
     * @brief Fixed-order projection with 16-bit weights and original FP32 A.
     *
     * Original FP32 activation storage retains its 256 physical lane mapping.
     * Actual half A/B uses the compact kernel below with the same logical tree.
     * Inactive rows return before dereferencing either operand or output.
     * Native source encoding is immutable launch policy; specializing it removes
     * a uniform dtype branch from every K iteration without changing conversion.
     * @tparam WeightDtype Validated original FP16 or BF16 source encoding.
     */
    template<SmallProjectionWeightDtype WeightDtype>
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void fp32x16_batched_projection_kernel(
            llaminar2::FixedFP32x16ProjectionGroup projections,
            int M,
            int N,
            int K,
            llaminar2::FloatingProjectionRows row_range)
    {
        static_assert(WeightDtype == SMALL_PROJECTION_WEIGHT_FP16 ||
                      WeightDtype == SMALL_PROJECTION_WEIGHT_BF16);
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = projections.input(batch);
        const uint16_t *B = projections.weight(batch);
        float *C = projections.output(batch);
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            const uint16_t weight_bits =
                B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
            float weight;
            if constexpr (WeightDtype == SMALL_PROJECTION_WEIGHT_BF16)
                weight = bf16_bits_to_float(weight_bits);
            else
                weight = fp16_bits_to_float(weight_bits);
            const size_t activation_index =
                static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k);
            // Preserve the original FP32 products, strided lane ownership
            // and fixed 256-lane reduction tree.
            const float activation = A[activation_index];
            sum += activation * weight;
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        const float reduced = reduceFloating256(sum, scratch);

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = reduced;
    }

    /**
     * @brief Adjacent canonical lanes share one packed load, retaining their sums.
     *
     * Two independent logical lane sums per physical thread preserve stride256.
     * Packed aligned operand words cut memory instructions. Scalar edge loads
     * cover odd alignment/K without changing either accumulator. The first two
     * reduction levels use shared memory; the remaining wave64 tree is identical
     * on the configured gfx906 production target.
     * @param ap Device array of physical-half activation owners.
     * @param bp Device array of physical-half weight owners.
     * @param cp Device array of FP32 result owners.
     * @tparam FixedK Measured model widths specialize address/loop bounds; zero retains arbitrary K.
     * @param m Admitted physical rows. @param n Output columns. @param runtime_k Reduction width.
     * @param rows Device-owned live-row range, resolved before any operand load.
     */
    template<int FixedK = 0>
    __global__ __launch_bounds__(128)
    void fp16_packed_batched_projection_kernel(
        const float *const *ap,
        const float *const *bp,
        float *const *cp,
        int m,
        int n,
        int runtime_k,
        llaminar2::FloatingProjectionRows rows)
    {
        const int k = FixedK ? FixedK : runtime_k;
        const int col = static_cast<int>(blockIdx.x);
        const int row = static_cast<int>(blockIdx.y);
        const int lane = static_cast<int>(threadIdx.x);
        if (!rows.rowIsActive(row) || col >= n)
            return;

        const auto *a = reinterpret_cast<const uint16_t *>(ap[blockIdx.z]);
        const auto *b = reinterpret_cast<const uint16_t *>(bp[blockIdx.z]);
        float *c = cp[blockIdx.z];
        if (!a || !b || !c)
            return;
        a += static_cast<size_t>(row) * k;
        b += static_cast<size_t>(col) * k;
        const bool packed =
            ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 3u) == 0;

        // Physical lane i carries the original independent logical lanes 2i
        // and 2i+1. Combining their products before reduction would change MTP.
        float first = 0.0f;
        float second = 0.0f;
        int j = 2 * lane;
    #pragma unroll
        for (; j + 1 < k; j += SMALL_N_FP32_BLOCK)
        {
            uint32_t aw;
            uint32_t bw;
            if (packed)
            {
                aw = *reinterpret_cast<const uint32_t *>(a + j);
                bw = *reinterpret_cast<const uint32_t *>(b + j);
            }
            else
            {
                aw = uint32_t(a[j]) | (uint32_t(a[j + 1]) << 16);
                bw = uint32_t(b[j]) | (uint32_t(b[j + 1]) << 16);
            }
            const float w0 = fp16_bits_to_float(uint16_t(bw));
            const float w1 = fp16_bits_to_float(uint16_t(bw >> 16));
            first += fp16_bits_to_float(uint16_t(aw)) * w0;
            second += fp16_bits_to_float(uint16_t(aw >> 16)) * w1;
        }
        if (j < k)
            first += fp16_bits_to_float(a[j]) * fp16_bits_to_float(b[j]);

        // Publish every original lane sum before changing ownership. The first
        // two levels join offsets128 and64; the last wave retains offsets32..1.
        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[2 * lane] = first;
        scratch[2 * lane + 1] = second;
        __syncthreads();
        float value = scratch[lane] + scratch[lane + 128];
        scratch[lane] = value;
        __syncthreads();
        if (lane < 64)
        {
            value += scratch[lane + 64];
            for (int stride = 32; stride > 0; stride >>= 1)
                value += __shfl_down(value, stride, 64);
            if (lane == 0)
                c[static_cast<size_t>(row) * n + col] = value;
        }
    }


    /**
     * @brief Fuse ragged half projections while retaining each original FP32 operation.
     *
     * The compact column grid selects immutable bindings directly from kernel
     * arguments. There is no device pointer publication, padding, new workspace,
     * or changed reduction partition. Inactive rows retire before operand loads.
     * @tparam FixedK Measured width specialization; zero admits arbitrary K.
     * @param g Validated borrowed input/weight/output owners and column ends.
     * @param runtime_k Positive reduction width. @param rows Device-live row authority.
     */
    template<int FixedK = 0>
    __global__ __launch_bounds__(128)
    void short_half_projection_group_kernel(
        llaminar2::FixedHalfProjectionGroup g, int runtime_k,
        llaminar2::FloatingProjectionRows rows)
    {
        const int k = FixedK ? FixedK : runtime_k;
        const int row = blockIdx.y, flat = blockIdx.x, lane = threadIdx.x;
        if (!rows.rowIsActive(row)) return;
        int p = 0;
        while (flat >= g.columnEnd(p)) ++p;
        const int first_column = p ? g.columnEnd(p - 1) : 0;
        const int col = flat - first_column, n = g.columnEnd(p) - first_column;
        const uint16_t *a = g.input() + size_t(row) * k;
        const uint16_t *b = g.weight(p) + size_t(col) * k;
        const bool packed = ((reinterpret_cast<uintptr_t>(a) |
                              reinterpret_cast<uintptr_t>(b)) & 3u) == 0;
        float first = 0.0f, second = 0.0f;
        int j = 2 * lane;
    #pragma unroll
        for (; j + 1 < k; j += 256)
        {
            uint32_t aw, bw;
            if (packed)
            {
                aw = *reinterpret_cast<const uint32_t *>(a + j);
                bw = *reinterpret_cast<const uint32_t *>(b + j);
            }
            else
            {
                aw = uint32_t(a[j]) | (uint32_t(a[j + 1]) << 16);
                bw = uint32_t(b[j]) | (uint32_t(b[j + 1]) << 16);
            }
            const float w0 = fp16_bits_to_float(uint16_t(bw));
            const float w1 = fp16_bits_to_float(uint16_t(bw >> 16));
            first += fp16_bits_to_float(uint16_t(aw)) * w0;
            second += fp16_bits_to_float(uint16_t(aw >> 16)) * w1;
        }
        if (j < k) first += fp16_bits_to_float(a[j]) * fp16_bits_to_float(b[j]);

        // Each physical lane retains two independent original sums. Publish
        // them before the unchanged 128/64 and final wave64 reduction edges.
        __shared__ float scratch[256];
        scratch[2 * lane] = first;
        scratch[2 * lane + 1] = second;
        __syncthreads();
        float value = scratch[lane] + scratch[lane + 128];
        scratch[lane] = value;
        __syncthreads();
        if (lane < 64)
        {
            value += scratch[lane + 64];
            for (int stride = 32; stride > 0; stride >>= 1)
                value += __shfl_down(value, stride, 64);
            if (lane == 0) g.output(p)[size_t(row) * n + col] = value;
        }
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
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        if (!row_range.rowIsActive(m) || n >= N)
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
        const float reduced = reduceFloating256(sum, scratch);

        if (threadIdx.x == 0)
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = reduced;
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

/**
 * @brief Launch fixed-column FP32 reductions at any positive output width.
 *
 * N indexes independent blocks; it never changes shared storage or the K tree.
 * The former N<=64 restriction described a caller, not a kernel limitation.
 */
extern "C" bool rocmFp32_small_n_batched_projection(
    llaminar2::FixedFP32ProjectionGroup projections,
    int M,
    int N,
    int K,
    int device_id,
    void *stream,
    const llaminar2::DeviceRowRange *row_range,
    const llaminar2::DeviceRequestRowRanges *request_rows)
{
    const int batch_count = projections.size();
    if (M<=0 || (row_range && request_rows) ||
        (row_range && row_range->physicalRows()!=M) ||
        (request_rows && request_rows->physicalRows()!=M))return false;
    const auto rows=llaminar2::FloatingProjectionRows::bind(M,row_range,request_rows);
    if (M <= 0 ||
        N <= 0 ||
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
    // The actual chunk cardinality is already sealed in projections. Mutable
    // live rows cannot change the retained function or dispatch geometry.
    const auto launch = [&]<int LoadPairs>() {
        hipLaunchKernelGGL((fp32_small_n_batched_projection_kernel<LoadPairs>),
            grid, block, 0, hip_stream, projections, M, N, K, rows);
    };
    using LoadPairs = llaminar2::rocm::FP32ProjectionLoadPairs;
    switch (llaminar2::rocm::selectFP32ProjectionLoadPairs(M, N, batch_count, K))
    {
    case LoadPairs::Eight: launch.template operator()<8>(); break;
    case LoadPairs::Four: launch.template operator()<4>(); break;
    case LoadPairs::Scalar: launch.template operator()<1>(); break;
    }

    return complete_kernel_launch(hip_stream, "rocmFp32_small_n_batched_projection");
}

extern "C" bool rocmFp32x16_batched_projection(
    llaminar2::FixedFP32x16ProjectionGroup projections,
    int M,
    int N,
    int K,
    int weight_dtype,
    int device_id,
    void *stream,
    const llaminar2::DeviceRowRange *row_range,
    const llaminar2::DeviceRequestRowRanges *request_rows)
{
    const int batch_count = projections.size();
    if (M<=0 || (row_range && request_rows) ||
        (row_range && row_range->physicalRows()!=M) ||
        (request_rows && request_rows->physicalRows()!=M))return false;
    const auto rows=llaminar2::FloatingProjectionRows::bind(M,row_range,request_rows);
    if (M <= 0 ||
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
    // Admission above authenticates the original native encoding once. Each
    // retained graph records its exact specialization; live rows remain device
    // owned and never participate in this immutable source-format selection.
    if (weight_dtype == SMALL_PROJECTION_WEIGHT_BF16)
    {
        hipLaunchKernelGGL(
            (fp32x16_batched_projection_kernel<SMALL_PROJECTION_WEIGHT_BF16>),
            grid,
            block,
            0,
            hip_stream,
            projections,
            M,
            N,
            K,
            rows);
    }
    else
    {
        hipLaunchKernelGGL(
            (fp32x16_batched_projection_kernel<SMALL_PROJECTION_WEIGHT_FP16>),
            grid,
            block,
            0,
            hip_stream,
            projections,
            M,
            N,
            K,
            rows);
    }

    return complete_kernel_launch(hip_stream, "rocmFp32x16_batched_projection");
}

/**
 * @brief Submit fixed-order physical half A/B projections on exact live rows.
 * @details Pointer arrays are the existing opaque ABI; the specialization widens
 * each half operand and keeps the serial 256-lane accumulation/reduction tree.
 * No inactive suffix, host count, allocation or alternate launch is consumed.
 */
extern "C" bool rocmFp16_batched_projection(
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
    const auto rows = llaminar2::FloatingProjectionRows::bind(M,row_range,nullptr);
    if (!d_A_array || !d_B_array || !d_C_array ||
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        batch_count <= 0 || batch_count > SMALL_N_FP32_MAX_BATCH ||
        !stream)
    {
        std::fprintf(
            stderr,
            "[rocmFp16_batched_projection] invalid arguments: M=%d N=%d K=%d batch=%d stream=%p\n",
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
            "[rocmFp16_batched_projection] hipSetDevice failed: %s\n",
            hipGetErrorString(err));
        return false;
    }

    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(M), static_cast<unsigned>(batch_count));
    dim3 block(SMALL_N_FP32_BLOCK / 2);
    // All widths retain the same canonical lane order; the measured model
    // widths expose constant loop bounds without limiting generic geometry.
    const auto launch = [&]<int FixedK>() {
        hipLaunchKernelGGL((fp16_packed_batched_projection_kernel<FixedK>),
            grid, block, 0, hip_stream, d_A_array, d_B_array, d_C_array, M, N, K, rows);
    };
    if (K == 2048) launch.template operator()<2048>();
    else if (K == 4096) launch.template operator()<4096>();
    else launch.template operator()<0>();

    return complete_kernel_launch(hip_stream, "rocmFp16_batched_projection");
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
    const auto rows = llaminar2::FloatingProjectionRows::bind(M,row_range,nullptr);
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

/** @brief Submit validated short-row bindings; see the shared launch ABI contract. */
extern "C" bool rocmFp16_short_projection_group(
    const llaminar2::FixedHalfProjectionGroup &group, int m, int k,
    int device_id, void *stream, const llaminar2::DeviceRowRange *row_range)
{
    if (!stream || m <= 0 || m > llaminar2::kShortHalfProjectionMaxRows || k <= 0 ||
        (row_range && row_range->physicalRows() != m)) return false;
    const auto rows = llaminar2::FloatingProjectionRows::bind(m,row_range,nullptr);
    if (hipSetDevice(device_id) != hipSuccess) return false;
    const auto native = static_cast<hipStream_t>(stream);
    const auto launch = [&]<int K>() {
        hipLaunchKernelGGL((short_half_projection_group_kernel<K>),
            dim3(group.totalColumns(), m), dim3(128), 0, native, group, k, rows);
    };
    if (k == 2048) launch.template operator()<2048>();
    else if (k == 4096) launch.template operator()<4096>();
    else launch.template operator()<0>();
    return complete_kernel_launch(native, "rocmFp16_short_projection_group");
}
