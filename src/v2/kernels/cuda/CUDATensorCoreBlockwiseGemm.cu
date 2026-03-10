/**
 * @file CUDATensorCoreBlockwiseGemm.cu
 * @brief Correct-first tensor-core scaffold for blockwise INT8 GEMM/GEMV on CUDA.
 *
 * This path mirrors the ROCm blockwise activation quantization contract without
 * claiming final-performance maturity yet. It repacks weights into a K=32-blocked
 * tensor-core layout and accumulates one tensor-core partial GEMM per activation
 * block, applying the per-block activation scale before accumulation into FP32 output.
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include <sstream>
#include <stdexcept>

namespace
{
    using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
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

    constexpr int kTensorCoreBlockK = 32;
    constexpr int kSmallMThreshold = 4;

#define CUDA_TC_CHECK(call)                                                   \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            std::ostringstream oss__;                                         \
            oss__ << "[CUDATensorCoreBlockwise] CUDA error: "                \
                  << cudaGetErrorString(err__) << " at " << __FILE__        \
                  << ":" << __LINE__;                                        \
            throw std::runtime_error(oss__.str());                            \
        }                                                                     \
    } while (0)

    __global__ void repack_weights_tc_blocked_kernel(
        const int8_t *__restrict__ weights_col_major,
        int8_t *__restrict__ weights_tc_blocked,
        int K, int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = K * N;
        if (idx >= total)
            return;

        const int k_block = idx / (N * kTensorCoreBlockK);
        const int rem = idx % (N * kTensorCoreBlockK);
        const int n = rem / kTensorCoreBlockK;
        const int k_in_block = rem % kTensorCoreBlockK;
        const int k = k_block * kTensorCoreBlockK + k_in_block;

        weights_tc_blocked[idx] = weights_col_major[n * K + k];
    }

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
        output[idx] = beta == 0.0f ? 0.0f : beta * base;
    }

    __global__ void accumulate_partial_blockwise_kernel(
        const int32_t *__restrict__ partial,
        float *__restrict__ output,
        const float *__restrict__ scales_A_blockwise,
        const float *__restrict__ scales_B,
        int M, int N,
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
        int M, int N)
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
}

extern "C"
{
    bool cudaQuantGemm_prepareTensorCoreBlockedWeights(
        const int8_t *d_weights_int8,
        int8_t **d_weights_int8_tc_blocked,
        int K, int N,
        int cuda_device_id,
        void *stream)
    {
        if (!d_weights_int8 || !d_weights_int8_tc_blocked || K <= 0 || N <= 0 || (K % kTensorCoreBlockK) != 0)
        {
            return false;
        }

        try
        {
            CUDA_TC_CHECK(cudaSetDevice(cuda_device_id));
            cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

            if (*d_weights_int8_tc_blocked == nullptr)
            {
                CUDA_TC_CHECK(cudaMalloc(d_weights_int8_tc_blocked, static_cast<size_t>(K) * N * sizeof(int8_t)));
            }

            const int total = K * N;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            repack_weights_tc_blocked_kernel<<<blocks, threads, 0, cuda_stream>>>(
                d_weights_int8,
                *d_weights_int8_tc_blocked,
                K,
                N);
            CUDA_TC_CHECK(cudaGetLastError());
            return true;
        }
        catch (...)
        {
            if (d_weights_int8_tc_blocked && *d_weights_int8_tc_blocked)
            {
                cudaFree(*d_weights_int8_tc_blocked);
                *d_weights_int8_tc_blocked = nullptr;
            }
            return false;
        }
    }

    bool cudaQuantGemm_blockwiseTensorCoreGemm(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8_tc_blocked,
        int32_t *d_partial_int32,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        const float *d_scales_B,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_weights_int8_tc_blocked || !d_partial_int32 || !d_C_fp32 || !d_scales_A_blockwise || !d_scales_B)
        {
            return false;
        }
        if (M <= 0 || N <= 0 || K <= 0 || (K % kTensorCoreBlockK) != 0)
        {
            return false;
        }

        try
        {
            CUDA_TC_CHECK(cudaSetDevice(cuda_device_id));
            cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

            const int total = M * N;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            initialize_output_kernel<<<blocks, threads, 0, cuda_stream>>>(
                d_C_fp32,
                d_C_existing,
                total,
                beta);
            CUDA_TC_CHECK(cudaGetLastError());

            const int num_k_blocks = K / kTensorCoreBlockK;
            for (int block_idx = 0; block_idx < num_k_blocks; ++block_idx)
            {
                const int8_t *a_block = d_A_int8 + block_idx * kTensorCoreBlockK;
                const int8_t *b_block = d_weights_int8_tc_blocked + static_cast<size_t>(block_idx) * N * kTensorCoreBlockK;

                cutlass::Status status = cutlass::Status::kErrorInternal;
                if (M <= kSmallMThreshold)
                {
                    status = launch_tensor_core_partial<CutlassInt8GemmSmallM>(
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
                }

                if (status != cutlass::Status::kSuccess)
                {
                    status = launch_tensor_core_partial<CutlassInt8Gemm>(
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
                }

                if (status != cutlass::Status::kSuccess)
                {
                    return false;
                }

                accumulate_partial_blockwise_kernel<<<blocks, threads, 0, cuda_stream>>>(
                    d_partial_int32,
                    d_C_fp32,
                    d_scales_A_blockwise,
                    d_scales_B,
                    M,
                    N,
                    num_k_blocks,
                    block_idx,
                    alpha);
                CUDA_TC_CHECK(cudaGetLastError());
            }

            if (d_bias)
            {
                add_bias_kernel<<<blocks, threads, 0, cuda_stream>>>(d_C_fp32, d_bias, M, N);
                CUDA_TC_CHECK(cudaGetLastError());
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }
}