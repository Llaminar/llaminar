/**
 * @file CUDATensorCoreGemmKernels.cu
 * @brief CUDA blockwise INT8 GEMM family backed by tensor-core MMA kernels.
 *
 * This file implements the blockwise-activation GEMM contract using CUTLASS
 * tensor-core micro-GEMMs over fixed K=32 slices. Each 32-wide activation
 * block launches an INT8 tensor-core GEMM that accumulates into INT32, then a
 * lightweight CUDA epilogue applies the per-block activation scale and the
 * per-column weight scale before accumulating into FP32 output.
 *
 * Unlike the decode GEMV path, GEMM here is intentionally tensor-core based:
 * CUTLASS is configured with OpClassTensorOp and the m16n8k32 INT8 MMA shape.
 * Dispatch selects among CUDA-native threadblock shapes instead of reusing the
 * ROCm-style per-thread DP4A structure.
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include <atomic>
#include <sstream>
#include <stdexcept>

namespace
{
    using CutlassInt8GemmBalanced = cutlass::gemm::device::Gemm<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    using CutlassInt8GemmSmallM = cutlass::gemm::device::Gemm<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    using CutlassInt8GemmWideN = cutlass::gemm::device::Gemm<
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    constexpr int kTensorCoreBlockK = 32;
    constexpr uint64_t kModeratelyWideWorkThreshold = 4ull * 1000ull * 1000ull * 1000ull;
    constexpr uint64_t kExtremelyWideWorkThreshold = 8ull * 1000ull * 1000ull * 1000ull;
    constexpr int kModeratelyWideAspectRatio = 4;
    constexpr int kExtremelyWideAspectRatio = 16;

    std::atomic<int> g_cuda_gemm_force_small_m{0};
    std::atomic<int> g_cuda_gemm_force_wide_n{0};
    std::atomic<int> g_cuda_gemm_force_balanced{0};

#define CUDA_GEMM_CHECK(call)                                                  \
    do                                                                         \
    {                                                                          \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess)                                              \
        {                                                                      \
            std::ostringstream oss__;                                          \
            oss__ << "[CUDATensorCoreGemm] CUDA error: "                      \
                  << cudaGetErrorString(err__) << " at " << __FILE__          \
                  << ":" << __LINE__;                                         \
            throw std::runtime_error(oss__.str());                             \
        }                                                                      \
    } while (0)

    __global__ void initialize_output_kernel(
        float *__restrict__ output,
        const float *__restrict__ existing,
        int total,
        float beta)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        const float base = existing ? existing[idx] : 0.0f;
        output[idx] = (beta == 0.0f) ? 0.0f : beta * base;
    }

    __global__ void accumulate_partial_blockwise_kernel(
        const int32_t *__restrict__ partial,
        float *__restrict__ output,
        const float *__restrict__ scales_A_blockwise,
        const float *__restrict__ scales_B,
        int M,
        int N,
        int blocks_per_row,
        int block_idx,
        float alpha)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total)
            return;

        const int m = idx / N;
        const int n = idx % N;
        const float scale_a = scales_A_blockwise[m * blocks_per_row + block_idx];
        const float scale_b = scales_B[n];
        output[idx] += alpha * static_cast<float>(partial[idx]) * scale_a * scale_b;
    }

    __global__ void add_bias_kernel(
        float *__restrict__ output,
        const float *__restrict__ bias,
        int M,
        int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total)
            return;

        const int n = idx % N;
        output[idx] += bias[n];
    }

    template <typename GemmT>
    cutlass::Status launch_tensor_core_partial(
        const int8_t *a_ptr,
        int lda,
        const int8_t *b_ptr,
        int ldb,
        int32_t *c_ptr,
        int ldc,
        int M,
        int N,
        int K,
        cudaStream_t stream)
    {
        GemmT op;
        typename GemmT::Arguments args(
            {M, N, K},
            {a_ptr, lda},
            {b_ptr, ldb},
            {c_ptr, ldc},
            {c_ptr, ldc},
            {1, 0});

        cutlass::Status can_status = GemmT::can_implement(args);
        if (can_status != cutlass::Status::kSuccess)
            return can_status;

        return op(args, nullptr, stream);
    }

    enum class GemmDispatchClass
    {
        SmallM,
        WideN,
        Balanced,
    };

    static GemmDispatchClass classifyGemmShape(int M, int N, int K)
    {
        if (g_cuda_gemm_force_small_m.load(std::memory_order_relaxed))
            return GemmDispatchClass::SmallM;
        if (g_cuda_gemm_force_wide_n.load(std::memory_order_relaxed))
            return GemmDispatchClass::WideN;
        if (g_cuda_gemm_force_balanced.load(std::memory_order_relaxed))
            return GemmDispatchClass::Balanced;

        const uint64_t total_work = static_cast<uint64_t>(M) * static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
        const bool extremely_wide = static_cast<uint64_t>(N) >= static_cast<uint64_t>(K) * kExtremelyWideAspectRatio;
        const bool moderately_wide = static_cast<uint64_t>(N) >= static_cast<uint64_t>(K) * kModeratelyWideAspectRatio;

        // Auto dispatch is intentionally driven by only two shape properties:
        // aspect ratio and total work. WideN only pays off once the matrix is
        // both wide enough and large enough; otherwise the smaller tile is the
        // more stable default across square and K-heavy projections.
        if ((extremely_wide && total_work >= kExtremelyWideWorkThreshold) ||
            (moderately_wide && total_work >= kModeratelyWideWorkThreshold))
            return GemmDispatchClass::WideN;

        return GemmDispatchClass::SmallM;
    }

    static cutlass::Status launchSelectedTensorCorePartial(
        GemmDispatchClass dispatch_class,
        const int8_t *a_ptr,
        int lda,
        const int8_t *b_ptr,
        int ldb,
        int32_t *c_ptr,
        int ldc,
        int M,
        int N,
        int K,
        cudaStream_t stream)
    {
        switch (dispatch_class)
        {
        case GemmDispatchClass::SmallM:
            return launch_tensor_core_partial<CutlassInt8GemmSmallM>(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, M, N, K, stream);
        case GemmDispatchClass::WideN:
        {
            cutlass::Status status = launch_tensor_core_partial<CutlassInt8GemmWideN>(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, M, N, K, stream);
            if (status == cutlass::Status::kSuccess)
                return status;
            return launch_tensor_core_partial<CutlassInt8GemmBalanced>(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, M, N, K, stream);
        }
        case GemmDispatchClass::Balanced:
        default:
            return launch_tensor_core_partial<CutlassInt8GemmBalanced>(a_ptr, lda, b_ptr, ldb, c_ptr, ldc, M, N, K, stream);
        }
    }
}

extern "C"
{
    void cudaTCGemm_setTuningOverrides(int force_small_m, int force_wide_n, int force_balanced, int, int)
    {
        g_cuda_gemm_force_small_m.store(force_small_m, std::memory_order_relaxed);
        g_cuda_gemm_force_wide_n.store(force_wide_n, std::memory_order_relaxed);
        g_cuda_gemm_force_balanced.store(force_balanced, std::memory_order_relaxed);
    }

    bool cudaTCGemm_blockwiseGemm(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8_tc_blocked,
        int32_t *d_partial_int32,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const float *d_scales_B,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_weights_int8_tc_blocked || !d_partial_int32 || !d_C_fp32 || !d_scales_A_block || !d_scales_B)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % kTensorCoreBlockK) != 0)
            return false;

        try
        {
            CUDA_GEMM_CHECK(cudaSetDevice(cuda_device_id));
            cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

            const int total = M * N;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            initialize_output_kernel<<<blocks, threads, 0, cuda_stream>>>(
                d_C_fp32,
                d_C_existing,
                total,
                beta);
            CUDA_GEMM_CHECK(cudaGetLastError());

            const int num_k_blocks = K / kTensorCoreBlockK;
            const GemmDispatchClass dispatch_class = classifyGemmShape(M, N, K);

            for (int block_idx = 0; block_idx < num_k_blocks; ++block_idx)
            {
                const int8_t *a_block = d_A_int8 + block_idx * kTensorCoreBlockK;
                const int8_t *b_block = d_weights_int8_tc_blocked + static_cast<size_t>(block_idx) * N * kTensorCoreBlockK;

                cutlass::Status status = launchSelectedTensorCorePartial(
                    dispatch_class,
                    a_block,
                    K,
                    b_block,
                    kTensorCoreBlockK,
                    d_partial_int32,
                    N,
                    M,
                    N,
                    kTensorCoreBlockK,
                    cuda_stream);

                if (status != cutlass::Status::kSuccess)
                    return false;

                accumulate_partial_blockwise_kernel<<<blocks, threads, 0, cuda_stream>>>(
                    d_partial_int32,
                    d_C_fp32,
                    d_scales_A_block,
                    d_scales_B,
                    M,
                    N,
                    num_k_blocks,
                    block_idx,
                    alpha);
                CUDA_GEMM_CHECK(cudaGetLastError());
            }

            if (d_bias)
            {
                add_bias_kernel<<<blocks, threads, 0, cuda_stream>>>(d_C_fp32, d_bias, M, N);
                CUDA_GEMM_CHECK(cudaGetLastError());
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }
}
