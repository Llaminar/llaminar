/**
 * @file ROCmFloatingPointGemmKernel.h
 * @brief ROCm GEMM kernel for floating-point tensors (FP32/FP16/BF16)
 *
 * Implements ITensorGemm using hipBLAS for floating-point weight tensors.
 * This is the ROCm counterpart to CUDA's CUDAFloatingPointGemmKernel.
 *
 * **Design**:
 * - Primary entry point: multiply_tensor() with type introspection
 * - Wraps HipBLASGemmKernel for actual computation
 * - Handles tensor type dispatch (FP32, FP16, BF16)
 * - All tensor data must be on GPU before calling
 *
 * **MI50 (gfx906) Notes**:
 * - FP32: 13.4 TFLOPS
 * - FP16: 26.5 TFLOPS (native support)
 * - BF16: NOT natively supported - emulated via FP32 compute
 *
 * **Usage**:
 * ```cpp
 * auto kernel = std::make_unique<ROCmFloatingPointGemmKernel>(weights, rocm_device_id);
 * kernel->multiply_tensor(activations, output, ...);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include <memory>
#include <cstdint>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class FP32Tensor;
    class FP16Tensor;
    class BF16Tensor;

    namespace rocm
    {
        // Forward declare the low-level hipBLAS kernel
        class HipBLASGemmKernel;

        /**
         * @brief ROCm GEMM kernel for floating-point weight tensors
         *
         * Implements ITensorGemm for FP32, FP16, and BF16 weight tensors using hipBLAS.
         *
         * **Supported Weight Types**:
         * - FP32Tensor → hipblasSgemm
         * - FP16Tensor → hipblasHgemm (good performance on MI50)
         * - BF16Tensor → emulated via FP32 (no native MI50 support)
         *
         * **Memory Requirements**:
         * - Weight tensor must be on GPU (ensureOnDevice() called)
         * - Input activations (A) must be on GPU
         * - Output (C) must be on GPU
         *
         * **Thread Safety**:
         * - Single kernel instance should be used from one thread
         * - hipBLAS handle is per-kernel (not shared)
         */
        class ROCmFloatingPointGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Precision mode for hipBLAS GEMM
             */
            enum class Precision
            {
                FP32, ///< 32-bit float (hipblasSgemm)
                FP16, ///< 16-bit float (hipblasHgemm)
                BF16  ///< BFloat16 (emulated - no native MI50 support)
            };

            /**
             * @brief Construct kernel for floating-point weight tensor
             *
             * @param weights FP32/FP16/BF16 tensor (must be on GPU)
             * @param rocm_device_id ROCm device ID (from hipGetDevice, NOT global index)
             * @param precision Precision mode (default: auto-detect from tensor)
             *
             * @throws std::runtime_error if weight type not supported or not on GPU
             */
            ROCmFloatingPointGemmKernel(
                const TensorBase *weights,
                int rocm_device_id,
                Precision precision = Precision::FP32);

            /**
             * @brief Construct kernel from raw device pointer (GPU pipeline path)
             *
             * Used when weights have been uploaded to a VRAM pool and the caller
             * already has the device pointer. No TensorBase needed.
             *
             * @param d_weights Device pointer to weight data (already on GPU)
             * @param N Output features (weight rows)
             * @param K Input features (weight cols)
             * @param rocm_device_id ROCm device ID
             * @param precision Precision mode
             * @param lifetime_owner Shared pointer that keeps the VRAM pool alive
             */
            ROCmFloatingPointGemmKernel(
                const void *d_weights,
                int N, int K,
                int rocm_device_id,
                Precision precision,
                std::shared_ptr<void> lifetime_owner);

            ~ROCmFloatingPointGemmKernel() override;

            // Non-copyable
            ROCmFloatingPointGemmKernel(const ROCmFloatingPointGemmKernel &) = delete;
            ROCmFloatingPointGemmKernel &operator=(const ROCmFloatingPointGemmKernel &) = delete;

            // Movable
            ROCmFloatingPointGemmKernel(ROCmFloatingPointGemmKernel &&) noexcept;
            ROCmFloatingPointGemmKernel &operator=(ROCmFloatingPointGemmKernel &&) noexcept;

            // =========================================================================
            // ITensorGemm interface - Primary entry points
            // =========================================================================

            /**
             * @brief Tensor-based GEMM with type introspection
             *
             * Inspects A and C tensor types and routes to appropriate hipBLAS call.
             * Currently supports FP32 activations/output.
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Activation-activation GEMM (not supported for FP ROCm kernel)
             *
             * ROCmFloatingPointGemmKernel is for weight projections only.
             * For activation-activation GEMM, use a dedicated attention kernel.
             */
            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            /**
             * @brief Strided activation-activation GEMM (not supported)
             */
            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // =========================================================================
            // ITensorKernel interface
            // =========================================================================

            bool supports_device(int device_idx) const override;

            void setGPUStream(void *stream) override;

            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // Accessors
            // =========================================================================

            int rocm_device_id() const { return rocm_device_id_; }
            size_t weight_rows() const { return N_; }
            size_t weight_cols() const { return K_; }
            Precision precision() const { return precision_; }

        private:
            const TensorBase *weights_; // Weight tensor (stored for metadata, may be null for pool path)
            const void *d_weights_;     // Device pointer to weight data
            int rocm_device_id_;
            Precision precision_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // hipBLAS kernel - shared across all ROCm GEMM kernels on same device
            // Owned by DeviceKernelCache, not this kernel instance
            HipBLASGemmKernel *hipblas_kernel_ = nullptr;

            // Lifetime owner: keeps VRAM pool alive when constructed from raw pointer
            std::shared_ptr<void> lifetime_owner_;

            // GPU stream for profiling (set via setGPUStream)
            void *gpu_stream_ = nullptr;

            // Cached HBM redirect buffer for mapped output memory
            // When output is host-mapped (e.g., logits), scattered GPU writes go over PCIe.
            // This provides an HBM staging buffer; we bulk-DMA to mapped memory after.
            float *d_mapped_redirect_ = nullptr;
            size_t mapped_redirect_capacity_ = 0;
        };

    } // namespace rocm
} // namespace llaminar2
