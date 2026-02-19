/**
 * @file ROCmResidualAddKernelT.h
 * @brief ROCm Residual Add kernel template header
 *
 * Provides FP32, BF16, and FP16 specializations of ITensorResidualAdd
 * for AMD GPUs via HIP. Direct port of CUDA implementation.
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/ROCmKernelProfiler.h"
#include <cstdint>
#include <stdexcept>

// Forward declarations for HIP kernels (implemented in ROCmResidualAddKernels.hip)
extern "C"
{
    bool rocmOps_residual_add_fp32(const float *input, const float *residual, float *output, int size, int device_idx, void *stream);
    bool rocmOps_residual_add_bf16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx, void *stream);
    bool rocmOps_residual_add_fp16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx, void *stream);
}

namespace llaminar2
{
    namespace rocm
    {

        // ============================================================================
        // Primary Template (static_assert for unsupported precisions)
        // ============================================================================

        template <ActivationPrecision Precision>
        class ROCmResidualAddKernelT
        {
            static_assert(
                Precision == ActivationPrecision::FP32 ||
                    Precision == ActivationPrecision::BF16 ||
                    Precision == ActivationPrecision::FP16,
                "ROCmResidualAddKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // ============================================================================
        // FP32 Specialization
        // ============================================================================

        template <>
        class ROCmResidualAddKernelT<ActivationPrecision::FP32> : public ITensorResidualAdd
        {
        public:
            explicit ROCmResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmResidualAddKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmResidualAddKernelT<FP32>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmResidualAddKernelT<FP32>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmResidualAddKernelT() override = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0; // Supports any ROCm GPU device
            }

            // GPU stream for graph capture support
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            bool apply(
                const float *input, const float *residual, float *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::RESIDUAL_ADD, static_cast<hipStream_t>(gpu_stream_));
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::FP32] Executing on device " << dev);
                return rocmOps_residual_add_fp32(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *residual,
                TensorBase *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                if (!input || !residual || !output)
                    return false;
                if (input->native_type() != TensorType::FP32)
                    return false;

                // Use active_data_ptr() which returns GPU pointer when tensor is on GPU
                // (consistent with BF16/FP16 specializations and CUDA implementation)
                return apply(
                    static_cast<const float *>(input->active_data_ptr()),
                    static_cast<const float *>(residual->active_data_ptr()),
                    static_cast<float *>(output->active_mutable_data_ptr()),
                    num_elements,
                    mpi_ctx,
                    device_idx);
            }

        private:
            int device_idx_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

        // ============================================================================
        // BF16 Specialization
        // ============================================================================

        template <>
        class ROCmResidualAddKernelT<ActivationPrecision::BF16> : public ITensorResidualAdd
        {
        public:
            explicit ROCmResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmResidualAddKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmResidualAddKernelT<BF16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmResidualAddKernelT<BF16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmResidualAddKernelT() override = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0;
            }

            bool apply(
                const float *input, const float *residual, float *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)input;
                (void)residual;
                (void)output;
                (void)num_elements;
                (void)mpi_ctx;
                (void)device_idx;
                return false; // BF16 kernel doesn't handle FP32 pointers
            }

            bool apply_bf16(
                const uint16_t *input, const uint16_t *residual, uint16_t *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::RESIDUAL_ADD, static_cast<hipStream_t>(gpu_stream_));
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::BF16] Executing on device " << dev);
                return rocmOps_residual_add_bf16(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *residual,
                TensorBase *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                if (!input || !residual || !output)
                    return false;
                if (input->native_type() != TensorType::BF16)
                    return false;

                // Use active_data_ptr() which returns GPU pointer when tensor is on GPU
                return apply_bf16(
                    static_cast<const uint16_t *>(input->active_data_ptr()),
                    static_cast<const uint16_t *>(residual->active_data_ptr()),
                    static_cast<uint16_t *>(output->active_mutable_data_ptr()),
                    num_elements,
                    mpi_ctx,
                    device_idx);
            }

        private:
            int device_idx_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

        // ============================================================================
        // FP16 Specialization
        // ============================================================================

        template <>
        class ROCmResidualAddKernelT<ActivationPrecision::FP16> : public ITensorResidualAdd
        {
        public:
            explicit ROCmResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit ROCmResidualAddKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmResidualAddKernelT<FP16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmResidualAddKernelT<FP16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmResidualAddKernelT() override = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0;
            }

            // GPU stream for graph capture support
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            bool apply(
                const float *input, const float *residual, float *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)input;
                (void)residual;
                (void)output;
                (void)num_elements;
                (void)mpi_ctx;
                (void)device_idx;
                return false; // FP16 kernel doesn't handle FP32 pointers
            }

            bool apply_fp16(
                const uint16_t *input, const uint16_t *residual, uint16_t *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::RESIDUAL_ADD, static_cast<hipStream_t>(gpu_stream_));
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::FP16] Executing on device " << dev);
                return rocmOps_residual_add_fp16(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *residual,
                TensorBase *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                if (!input || !residual || !output)
                    return false;
                if (input->native_type() != TensorType::FP16)
                    return false;

                // Use active_data_ptr() which returns GPU pointer when tensor is on GPU
                return apply_fp16(
                    static_cast<const uint16_t *>(input->active_data_ptr()),
                    static_cast<const uint16_t *>(residual->active_data_ptr()),
                    static_cast<uint16_t *>(output->active_mutable_data_ptr()),
                    num_elements,
                    mpi_ctx,
                    device_idx);
            }

        private:
            int device_idx_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

    } // namespace rocm
} // namespace llaminar2
