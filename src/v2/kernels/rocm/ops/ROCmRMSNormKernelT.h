/**
 * @file ROCmRMSNormKernelT.h
 * @brief ROCm RMSNorm kernel template header
 *
 * Provides FP32, BF16, and FP16 specializations of ITensorRMSNorm
 * for AMD GPUs via HIP.
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h" // For FP32Tensor, BF16Tensor, FP16Tensor (apply_tensor)
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    namespace rocm
    {

        // ============================================================================
        // Primary Template (static_assert for unsupported precisions)
        // ============================================================================

        template <ActivationPrecision Precision>
        class ROCmRMSNormKernelT
        {
            static_assert(
                Precision == ActivationPrecision::FP32 ||
                    Precision == ActivationPrecision::BF16 ||
                    Precision == ActivationPrecision::FP16,
                "ROCmRMSNormKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // ============================================================================
        // FP32 Specialization
        // ============================================================================

        template <>
        class ROCmRMSNormKernelT<ActivationPrecision::FP32> : public ITensorRMSNorm
        {
        public:
            using StorageType = float;

            explicit ROCmRMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmRMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRMSNormKernelT<FP32>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRMSNormKernelT<FP32>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRMSNormKernelT() = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

            // ITensorRMSNorm interface
            bool apply(
                const float *input,
                const float *weight,
                float *output,
                int rows,
                int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // Tensor-based API with automatic GPU pointer handling
            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // Typed API for direct device pointer access
            bool apply_typed(
                const float *d_input,
                const float *d_gamma,
                float *d_output,
                int rows,
                int cols,
                float epsilon,
                int device_idx);

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

        // ============================================================================
        // BF16 Specialization
        // ============================================================================

        template <>
        class ROCmRMSNormKernelT<ActivationPrecision::BF16> : public ITensorRMSNorm
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmRMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmRMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRMSNormKernelT<BF16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRMSNormKernelT<BF16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRMSNormKernelT() = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

            // ITensorRMSNorm interface
            bool apply(
                const float *input,
                const float *weight,
                float *output,
                int rows,
                int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                const uint16_t *input,
                const float *weight,
                uint16_t *output,
                int rows,
                int cols,
                float epsilon = 1e-6f,
                int device_idx = -1) override;

            // Tensor-based API with automatic GPU pointer handling
            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // Typed API for direct device pointer access
            bool apply_typed(
                const uint16_t *d_input,
                const float *d_gamma,
                uint16_t *d_output,
                int rows,
                int cols,
                float epsilon,
                int device_idx);

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

        // ============================================================================
        // FP16 Specialization
        // ============================================================================

        template <>
        class ROCmRMSNormKernelT<ActivationPrecision::FP16> : public ITensorRMSNorm
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmRMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmRMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRMSNormKernelT<FP16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRMSNormKernelT<FP16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRMSNormKernelT() = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

            // ITensorRMSNorm interface
            bool apply(
                const float *input,
                const float *weight,
                float *output,
                int rows,
                int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_fp16(
                const uint16_t *input,
                const float *weight,
                uint16_t *output,
                int rows,
                int cols,
                float epsilon = 1e-6f,
                int device_idx = -1) override;

            // Tensor-based API with automatic GPU pointer handling
            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // Typed API for direct device pointer access
            bool apply_typed(
                const uint16_t *d_input,
                const float *d_gamma,
                uint16_t *d_output,
                int rows,
                int cols,
                float epsilon,
                int device_idx);

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

    } // namespace rocm
} // namespace llaminar2
