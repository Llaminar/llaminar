/**
 * @file CUDAResidualAddKernelT.h
 * @brief CUDA implementation of residual add kernel
 * @author David Sanftenberg
 *
 * Template-specialized implementations of ITensorResidualAdd for CUDA.
 * Uses extern "C" wrappers to call CUDA kernels in CUDAResidualAddKernels.cu.
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include <string>
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

    namespace residual_add_detail
    {
        /**
         * @brief Recover active verifier rows from the flat element contract.
         *
         * ResidualAdd is semantically flat, but production tensors retain a
         * logical column width. Dividing the active span by that width recovers
         * M even when a graph buffer reserves capacity for four rows.
         */
        inline int activeRows(
            const TensorBase *input,
            size_t num_elements)
        {
            if (!input || input->cols() == 0 || num_elements % input->cols() != 0)
                return 0;
            return static_cast<int>(num_elements / input->cols());
        }

        /**
         * @brief Record one successful device-owned CUDA grouped launch.
         *
         * The production implementation launches one flat elementwise grid for
         * all M rows. Serial M=1 witnesses intentionally emit no record, making
         * hidden row replay visible to the grouped regression gate.
         */
        inline void recordCUDAGroupedCall(
            const TensorBase *input,
            size_t num_elements,
            int device)
        {
            const int rows = activeRows(input, num_elements);
            if (rows < 2)
                return;

            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_residual_add_grouped_verifier_rows_calls",
                1.0,
                "verifier",
                DeviceId::cuda(device).to_string(),
                {{"tensor_format", tensorTypeName(input->native_type())},
                 {"verifier_rows", std::to_string(rows)},
                 {"cols", std::to_string(input->cols())},
                 {"active_elements", std::to_string(num_elements)},
                 {"capture_mode", isGraphCaptureActive() ? "graph_capture" : "direct"},
                 {"invocation_policy", "single_flat_launch"}});
        }
    } // namespace residual_add_detail

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
        void *getStream() const { return requireExplicitGPUStreamBinding(gpu_stream_, "GPU tensor kernel"); }

        // GPU stream for graph capture support
        void bindGPUStream(ExplicitGPUStream stream) override { gpu_stream_ = stream.get(); }
        void clearGPUStreamBinding() override { gpu_stream_ = nullptr; }

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
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAResidualAddKernelT::FP32] apply_tensor requires an explicit non-null CUDA stream");
                return false;
            }
            if (input->native_type() != TensorType::FP32 ||
                residual->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;

            // Graph execution owns residency. Never adopt a host-visible active
            // pointer for a GPU stage; all three operands must remain device-owned.
            const bool ok = apply(
                static_cast<const float *>(input->gpu_data_ptr()),
                static_cast<const float *>(residual->gpu_data_ptr()),
                static_cast<float *>(output->gpu_data_ptr()),
                num_elements,
                mpi_ctx,
                device_idx);
            const int dev = (device_idx >= 0) ? device_idx : device_idx_;
            if (ok)
                residual_add_detail::recordCUDAGroupedCall(input, num_elements, dev);
            return ok;
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
        void *getStream() const { return requireExplicitGPUStreamBinding(gpu_stream_, "GPU tensor kernel"); }

        // GPU stream for graph capture support
        void bindGPUStream(ExplicitGPUStream stream) override { gpu_stream_ = stream.get(); }
        void clearGPUStreamBinding() override { gpu_stream_ = nullptr; }

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
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAResidualAddKernelT::BF16] apply_tensor requires an explicit non-null CUDA stream");
                return false;
            }
            if (input->native_type() != TensorType::BF16 ||
                residual->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;

            const bool ok = apply_bf16(
                static_cast<const uint16_t *>(input->gpu_data_ptr()),
                static_cast<const uint16_t *>(residual->gpu_data_ptr()),
                static_cast<uint16_t *>(output->gpu_data_ptr()),
                num_elements,
                mpi_ctx,
                device_idx);
            const int dev = (device_idx >= 0) ? device_idx : device_idx_;
            if (ok)
                residual_add_detail::recordCUDAGroupedCall(input, num_elements, dev);
            return ok;
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
        void *getStream() const { return requireExplicitGPUStreamBinding(gpu_stream_, "GPU tensor kernel"); }

        // GPU stream for graph capture support
        void bindGPUStream(ExplicitGPUStream stream) override { gpu_stream_ = stream.get(); }
        void clearGPUStreamBinding() override { gpu_stream_ = nullptr; }

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
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAResidualAddKernelT::FP16] apply_tensor requires an explicit non-null CUDA stream");
                return false;
            }
            if (input->native_type() != TensorType::FP16 ||
                residual->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;

            const bool ok = apply_fp16(
                static_cast<const uint16_t *>(input->gpu_data_ptr()),
                static_cast<const uint16_t *>(residual->gpu_data_ptr()),
                static_cast<uint16_t *>(output->gpu_data_ptr()),
                num_elements,
                mpi_ctx,
                device_idx);
            const int dev = (device_idx >= 0) ? device_idx : device_idx_;
            if (ok)
                residual_add_detail::recordCUDAGroupedCall(input, num_elements, dev);
            return ok;
        }

    private:
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
    };

} // namespace llaminar2::cuda
