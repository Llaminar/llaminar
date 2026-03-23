/**
 * @file CUDASwiGLUKernelT.h
 * @brief CUDA SwiGLU activation kernel with typed precision support
 *
 * CUDA implementation of SwiGLU following the same template pattern as
 * CPUSwiGLUKernelT. Specialized for FP32, BF16, and FP16 precisions.
 *
 * @author Llaminar Team
 * @date 2025-01-14
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    namespace cuda
    {

        // =========================================================================
        // Primary Template (Unsupported Precisions)
        // =========================================================================

        template <ActivationPrecision Precision>
        class CUDASwiGLUKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "CUDASwiGLUKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class CUDASwiGLUKernelT<ActivationPrecision::FP32> : public ITensorSwiGLU
        {
        public:
            using StorageType = float;

            explicit CUDASwiGLUKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             */
            explicit CUDASwiGLUKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDASwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // GPU stream for graph capture support
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool apply_bf16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                const void *gate, const void *up, void *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *gate,
                const TensorBase *up,
                TensorBase *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const float *gate,
                const float *up,
                float *output,
                int size,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class CUDASwiGLUKernelT<ActivationPrecision::BF16> : public ITensorSwiGLU
        {
        public:
            using StorageType = uint16_t;

            explicit CUDASwiGLUKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            explicit CUDASwiGLUKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDASwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // GPU stream for graph capture support
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool apply_fp16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                const void *gate, const void *up, void *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *gate,
                const TensorBase *up,
                TensorBase *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const uint16_t *gate,
                const uint16_t *up,
                uint16_t *output,
                int size,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class CUDASwiGLUKernelT<ActivationPrecision::FP16> : public ITensorSwiGLU
        {
        public:
            using StorageType = uint16_t;

            explicit CUDASwiGLUKernelT(int device_idx = -1) : device_idx_(device_idx) {}

            explicit CUDASwiGLUKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDASwiGLUKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDASwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // GPU stream for graph capture support
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool apply_q8_1(
                const void *gate, const void *up, void *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1)
            {
                (void)gate;
                (void)up;
                (void)output;
                (void)rows;
                (void)cols;
                (void)add_residual;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *gate,
                const TensorBase *up,
                TensorBase *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const uint16_t *gate,
                const uint16_t *up,
                uint16_t *output,
                int size,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;
        };

    } // namespace cuda
} // namespace llaminar2
