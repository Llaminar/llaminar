/**
 * @file CUDAFloatingPointGemmKernel.h
 * @brief CUDA GEMM kernel for floating-point tensors (FP32/FP16/BF16)
 *
 * Implements ITensorGemm using cuBLAS for floating-point weight tensors.
 * This is the CUDA counterpart to CPU FloatingPointGemmKernel.
 *
 * **Design**:
 * - Primary entry point: multiply_tensor() with type introspection
 * - Wraps CuBLASGemmKernel for actual computation
 * - Handles tensor type dispatch (FP32, FP16, BF16)
 * - All tensor data must be on GPU before calling
 *
 * **Usage**:
 * ```cpp
 * auto kernel = std::make_unique<CUDAFloatingPointGemmKernel>(weights, cuda_device_id);
 * kernel->multiply_tensor(activations, output, ...);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include <memory>
#include <cstdint>

namespace llaminar2
{
    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias
    class FP32Tensor;
    class FP16Tensor;
    class BF16Tensor;

    namespace cuda
    {
        // Forward declare the low-level cuBLAS kernel
        class CuBLASGemmKernel;

        /**
         * @brief CUDA GEMM kernel for floating-point weight tensors
         *
         * Implements ITensorGemm for FP32, FP16, and BF16 weight tensors using cuBLAS.
         *
         * **Supported Weight Types**:
         * - FP32Tensor → cublasSgemm
         * - FP16Tensor → cublasHgemm (Tensor Core accelerated on Volta+)
         * - BF16Tensor → cublasSgemmEx with BF16 (compute in FP32)
         *
         * **Memory Requirements**:
         * - Weight tensor must be on GPU (ensureOnDevice() called)
         * - Input activations (A) must be on GPU
         * - Output (C) must be on GPU
         *
         * **Thread Safety**:
         * - Single kernel instance should be used from one thread
         * - cuBLAS handle is per-kernel (not shared)
         */
        class CUDAFloatingPointGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Precision mode for cuBLAS GEMM
             */
            enum class Precision
            {
                FP32, ///< 32-bit float (cublasSgemm)
                FP16, ///< 16-bit float (cublasHgemm)
                BF16  ///< BFloat16 (cublasSgemmEx)
            };

            /**
             * @brief Construct kernel for floating-point weight tensor
             *
             * @param weights FP32/FP16/BF16 tensor (must be on GPU)
             * @param cuda_device_id CUDA device ID (from cudaGetDevice, NOT global index)
             * @param precision Precision mode (default: auto-detect from tensor)
             *
             * @throws std::runtime_error if weight type not supported or not on GPU
             */
            CUDAFloatingPointGemmKernel(
                const TensorBase *weights,
                int cuda_device_id,
                Precision precision = Precision::FP32);

            ~CUDAFloatingPointGemmKernel() override;

            // Non-copyable
            CUDAFloatingPointGemmKernel(const CUDAFloatingPointGemmKernel &) = delete;
            CUDAFloatingPointGemmKernel &operator=(const CUDAFloatingPointGemmKernel &) = delete;

            // Movable
            CUDAFloatingPointGemmKernel(CUDAFloatingPointGemmKernel &&) noexcept;
            CUDAFloatingPointGemmKernel &operator=(CUDAFloatingPointGemmKernel &&) noexcept;

            // =========================================================================
            // ITensorGemm interface - Primary entry points
            // =========================================================================

            /**
             * @brief Tensor-based GEMM with type introspection
             *
             * Inspects A and C tensor types and routes to appropriate cuBLAS call.
             * Currently supports FP32 activations/output.
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Raw FP32 pointer GEMM
             *
             * Direct interface for callers with raw device pointers.
             */
            bool multiply(
                const float *A, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Activation-activation GEMM (not supported for FP CUDA kernel)
             *
             * CUDAFloatingPointGemmKernel is for weight projections only.
             * For activation-activation GEMM, use a dedicated attention kernel.
             */
            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Strided activation-activation GEMM (not supported)
             */
            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // =========================================================================
            // ITensorKernel interface
            // =========================================================================

            bool supports_device(int device_idx) const override;

            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // Accessors
            // =========================================================================

            int cuda_device_id() const { return cuda_device_id_; }
            size_t weight_rows() const { return N_; }
            size_t weight_cols() const { return K_; }
            Precision precision() const { return precision_; }

        private:
            const TensorBase *weights_; // Weight tensor (stored for metadata)
            const void *d_weights_;     // Device pointer to weight data
            int cuda_device_id_;
            Precision precision_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // cuBLAS kernel (created at construction)
            std::unique_ptr<CuBLASGemmKernel> cublas_kernel_;
        };

    } // namespace cuda
} // namespace llaminar2
