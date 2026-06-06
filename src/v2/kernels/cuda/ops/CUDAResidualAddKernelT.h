/**
 * @file CUDAResidualAddKernelT.h
 * @brief CUDA implementation of residual add kernel
 * @author David Sanftenberg
 *
 * Template-specialized implementations of ITensorResidualAdd for CUDA.
 * Uses extern "C" wrappers to call CUDA kernels in CUDAResidualAddKernels.cu.
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include <stdexcept>

// Forward declarations for CUDA kernels
extern "C"
{
    bool cudaOps_residual_add_fp32(const float *input, const float *residual, float *output, int size, int device_idx, void *stream);
    bool cudaOps_residual_add_bf16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx, void *stream);
    bool cudaOps_residual_add_fp16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx, void *stream);
}

namespace llaminar2::cuda
{

    // ==========================================================================
    // FP32 Specialization
    // ==========================================================================

    template <ActivationPrecision Precision>
    class CUDAResidualAddKernelT;

    template <>
    class CUDAResidualAddKernelT<ActivationPrecision::FP32> : public ITensorResidualAdd
    {
    public:
        explicit CUDAResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

        /**
         * @brief Construct with device context (Phase 4 pattern)
         */
        explicit CUDAResidualAddKernelT(IWorkerGPUContext *ctx)
        {
            if (!ctx)
                throw std::runtime_error("CUDAResidualAddKernelT: Device context is null");
            if (!ctx->isInitialized())
                throw std::runtime_error("CUDAResidualAddKernelT: Device context not initialized");
            device_ctx_ = ctx;
            device_idx_ = ctx->deviceOrdinal();
        }

        ~CUDAResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // Supports any GPU device
        }

        // ===== Device Context Support (Phase 4) =====
        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

        // GPU stream for graph capture support
        void setGPUStream(void *stream) override { gpu_stream_ = stream; }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            LOG_DEBUG("[CUDAResidualAddKernelT::FP32] Executing on device " << dev);
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::RESIDUAL_ADD, gpu_stream_);
            return cudaOps_residual_add_fp32(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::FP32)
                return false;

            // Use active_data_ptr() which returns GPU pointer when tensor is on GPU
            // (consistent with BF16/FP16 specializations)
            return apply(
                static_cast<const float *>(input->active_data_ptr()),
                static_cast<const float *>(residual->active_data_ptr()),
                static_cast<float *>(output->active_mutable_data_ptr()),
                num_elements,
                mpi_ctx,
                device_idx);
        }

    private:
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
    };

    // ==========================================================================
    // BF16 Specialization
    // ==========================================================================

    template <>
    class CUDAResidualAddKernelT<ActivationPrecision::BF16> : public ITensorResidualAdd
    {
    public:
        explicit CUDAResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

        explicit CUDAResidualAddKernelT(IWorkerGPUContext *ctx)
        {
            if (!ctx)
                throw std::runtime_error("CUDAResidualAddKernelT: Device context is null");
            if (!ctx->isInitialized())
                throw std::runtime_error("CUDAResidualAddKernelT: Device context not initialized");
            device_ctx_ = ctx;
            device_idx_ = ctx->deviceOrdinal();
        }

        ~CUDAResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0;
        }

        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

        // GPU stream for graph capture support
        void setGPUStream(void *stream) override { gpu_stream_ = stream; }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)residual;
            (void)output;
            (void)num_elements;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // BF16 kernel doesn't handle FP32
        }

        bool apply_bf16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            LOG_DEBUG("[CUDAResidualAddKernelT::BF16] Executing on device " << dev);
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::RESIDUAL_ADD, gpu_stream_);
            return cudaOps_residual_add_bf16(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
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
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
    };

    // ==========================================================================
    // FP16 Specialization
    // ==========================================================================

    template <>
    class CUDAResidualAddKernelT<ActivationPrecision::FP16> : public ITensorResidualAdd
    {
    public:
        explicit CUDAResidualAddKernelT(int device_idx = -1) : device_idx_(device_idx) {}

        explicit CUDAResidualAddKernelT(IWorkerGPUContext *ctx)
        {
            if (!ctx)
                throw std::runtime_error("CUDAResidualAddKernelT: Device context is null");
            if (!ctx->isInitialized())
                throw std::runtime_error("CUDAResidualAddKernelT: Device context not initialized");
            device_ctx_ = ctx;
            device_idx_ = ctx->deviceOrdinal();
        }

        ~CUDAResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0;
        }

        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

        // GPU stream for graph capture support
        void setGPUStream(void *stream) override { gpu_stream_ = stream; }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)residual;
            (void)output;
            (void)num_elements;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // FP16 kernel doesn't handle FP32
        }

        bool apply_fp16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            LOG_DEBUG("[CUDAResidualAddKernelT::FP16] Executing on device " << dev);
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::RESIDUAL_ADD, gpu_stream_);
            return cudaOps_residual_add_fp16(input, residual, output, static_cast<int>(num_elements), dev, gpu_stream_);
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
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
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
    };

} // namespace llaminar2::cuda
