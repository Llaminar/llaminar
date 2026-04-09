/**
 * @file CuBLASGemmKernel.h
 * @brief cuBLAS-based GEMM kernel for FP32/FP16/BF16 tensors
 *
 * **Purpose**: Provide GPU GEMM for non-quantized (floating-point) tensors using cuBLAS.
 *
 * **Design**:
 * - Inherits from CUDAKernelBase for workspace and device context support
 * - Uses cuBLAS sgemm (FP32), hgemm (FP16), or bfgemm (BF16)
 * - Expects input/output matrices already on GPU device
 * - Caller responsible for device memory management
 *
 * **Device Context Support (Phase 4)**:
 * - Can use cuBLAS handle from IWorkerGPUContext instead of creating own
 * - Backward compatible: existing constructor still creates own handle
 *
 * **Usage**:
 * ```cpp
 * // Legacy: creates own cuBLAS handle
 * auto kernel = std::make_unique<CuBLASGemmKernel>(device_id);
 *
 * // New: uses context's cuBLAS handle
 * auto kernel = std::make_unique<CuBLASGemmKernel>(context, Precision::FP32);
 * ```
 *
 * **Note**: For quantized tensors (Q4_0, Q8_0, IQ4_NL), use CudaQuantizedGemmKernel
 * with INT8 Tensor Cores instead of cuBLAS.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "kernels/cuda/CUDAKernelBase.h"
#include <cstddef>
#include <memory>
#include <stdexcept>

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cublasLt.h>
#endif

namespace llaminar2
{
    // Forward declaration
    class IWorkerGPUContext;

    namespace cuda
    {

        /**
         * @brief cuBLAS-based GEMM kernel for floating-point tensors
         *
         * Supports:
         * - FP32: cublasSgemm
         * - FP16: cublasHgemm with Tensor Core math mode
         * - BF16: cublasSgemmEx with CUDA_R_16BF (compute in FP32)
         *
         * **Memory Requirements**:
         * - All pointers (A, B, C) must be in GPU device memory
         * - Caller manages device allocation/deallocation
         *
         * **Performance**:
         * - Tensor Core math enabled for FP16 when available (SM 7.0+)
         * - Optimized for large matrices (M, N, K > 128)
         */
        class CuBLASGemmKernel : public CUDAKernelBase
        {
        public:
            /**
             * @brief Supported precision modes
             */
            enum class Precision
            {
                FP32, ///< 32-bit float (cublasSgemm)
                FP16, ///< 16-bit float (cublasHgemm with Tensor Cores)
                BF16  ///< BFloat16 (cublasSgemmEx with CUDA_R_16BF)
            };

            /**
             * @brief Create cuBLAS GEMM kernel (legacy constructor - creates own handle)
             *
             * @param device_id CUDA device ID (from cudaGetDevice)
             * @param precision Floating-point precision to use
             *
             * @throws std::runtime_error if cuBLAS handle creation fails
             */
            explicit CuBLASGemmKernel(int device_id, Precision precision = Precision::FP32);

            /**
             * @brief Create cuBLAS GEMM kernel using device context's handle
             *
             * This constructor uses the cuBLAS handle from the provided device context,
             * avoiding the overhead of creating a separate handle. The kernel does NOT
             * own the handle and will not destroy it.
             *
             * @param ctx Device context providing cuBLAS handle (must outlive this kernel)
             * @param precision Floating-point precision to use
             *
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit CuBLASGemmKernel(IWorkerGPUContext *ctx, Precision precision = Precision::FP32);

            /**
             * @brief Destructor - destroys cuBLAS handle only if owned
             */
            ~CuBLASGemmKernel();

            // Non-copyable
            CuBLASGemmKernel(const CuBLASGemmKernel &) = delete;
            CuBLASGemmKernel &operator=(const CuBLASGemmKernel &) = delete;

            // Movable
            CuBLASGemmKernel(CuBLASGemmKernel &&other) noexcept;
            CuBLASGemmKernel &operator=(CuBLASGemmKernel &&other) noexcept;

            // =================================================================
            // Device Query
            // =================================================================

            bool supports_device(int device_idx) const
            {
                return device_idx >= 0; // Supports any GPU device
            }

            // =================================================================
            // GEMM Operations
            // =================================================================

            /**
             * @brief GPU GEMM: C = alpha * A @ B + beta * C
             *
             * **Layout Convention**:
             * - A: Row-major [M × K]
             * - B: Row-major [N × K] if transpose_B=true, [K × N] otherwise
             * - C: Row-major [M × N]
             *
             * cuBLAS uses column-major, so we transpose the operation:
             * C^T = B^T @ A^T  (which gives C in row-major)
             *
             * @param d_A Device pointer to A [M × K]
             * @param d_B Device pointer to B [N × K] or [K × N]
             * @param d_C Device pointer to C [M × N]
             * @param M Rows in A and C
             * @param N Columns in C (rows in B if transposed)
             * @param K Columns in A (and B's inner dimension)
             * @param transA Transpose A (typically false)
             * @param transB Transpose B (typically true for weights)
             * @param alpha Scale for A@B
             * @param beta Scale for C
             *
             * @return true on success
             */
            bool execute(
                const float *d_A, const float *d_B, float *d_C,
                int M, int N, int K,
                bool transA = false, bool transB = false,
                float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief GPU GEMM with fused bias: C = alpha * A @ B + beta * C + bias
             *
             * Uses cuBLASLt epilogue for fused bias addition.
             * Bias is a 1D vector of length N, broadcast across rows.
             *
             * @param d_A Device pointer to A [M × K]
             * @param d_B Device pointer to B [N × K] or [K × N]
             * @param d_C Device pointer to C [M × N]
             * @param d_bias Device pointer to bias [N]
             * @param M Rows in A and C
             * @param N Columns in C (rows in B if transposed)
             * @param K Columns in A (and B's inner dimension)
             * @param transA Transpose A (typically false)
             * @param transB Transpose B (typically true for weights)
             * @param alpha Scale for A@B
             * @param beta Scale for C
             *
             * @return true on success
             */
            bool execute_with_bias(
                const float *d_A, const float *d_B, float *d_C,
                const float *d_bias,
                int M, int N, int K,
                bool transA = false, bool transB = false,
                float alpha = 1.0f, float beta = 0.0f);

            // Getters
            int device_id() const { return device_id_; }
            Precision precision() const { return precision_; }

            /**
             * @brief Check if this kernel owns its cuBLAS handle
             * @return true if destructor will destroy the handle, false if using context's handle
             */
            bool ownsHandle() const { return owns_handle_; }

            /**
             * @brief Set the stream on the underlying cuBLAS handle
             * @param stream CUDA stream (cast to cudaStream_t internally)
             */
            void setStream(void *stream);

        private:
#ifdef HAVE_CUDA
            cublasHandle_t handle_ = nullptr;
            cublasLtHandle_t lt_handle_ = nullptr;
#endif
            int device_id_ = 0;
            Precision precision_ = Precision::FP32;
            bool owns_handle_ = true;    ///< false when using context's cuBLAS handle
            bool owns_lt_handle_ = true; ///< false when using context's cuBLASLt handle
        };

        /**
         * @brief Factory function for cuBLAS GEMM kernel (legacy - creates own handle)
         *
         * @param device_id CUDA device ID
         * @param precision Desired precision (default FP32)
         * @return unique_ptr to CuBLASGemmKernel
         */
        std::unique_ptr<CuBLASGemmKernel> createCuBLASGemm(
            int device_id,
            CuBLASGemmKernel::Precision precision = CuBLASGemmKernel::Precision::FP32);

        /**
         * @brief Factory function for cuBLAS GEMM kernel using device context
         *
         * @param ctx Device context providing cuBLAS handle
         * @param precision Desired precision (default FP32)
         * @return unique_ptr to CuBLASGemmKernel
         */
        std::unique_ptr<CuBLASGemmKernel> createCuBLASGemm(
            IWorkerGPUContext *ctx,
            CuBLASGemmKernel::Precision precision = CuBLASGemmKernel::Precision::FP32);

    } // namespace cuda
} // namespace llaminar2
