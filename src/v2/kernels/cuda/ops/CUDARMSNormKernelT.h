/**
 * @file CUDARMSNormKernelT.h
 * @brief CUDA RMSNorm kernel with typed precision support
 *
 * CUDA implementation of RMSNorm following the same template pattern as
 * CPURMSNormKernelT. Specialized for FP32, BF16, and FP16 precisions.
 *
 * Architecture:
 * - Header declares class templates with ITensorRMSNorm interface
 * - CUDA kernels are in CUDAOpsKernels.cu with extern "C" API
 * - CUDAOpsKernels.cpp implements the ITensorRMSNorm methods
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

        /**
         * @brief Primary template - static_assert for unsupported precisions
         */
        template <ActivationPrecision Precision>
        class CUDARMSNormKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "CUDARMSNormKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class CUDARMSNormKernelT<ActivationPrecision::FP32> : public ITensorRMSNorm
        {
        public:
            using StorageType = float;

            explicit CUDARMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             */
            explicit CUDARMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDARMSNormKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDARMSNormKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== ITensorRMSNorm interface =====
            bool apply(
                const float *input, const float *weight, float *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                const Q8_1Block *input, const float *weight, Q8_1Block *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const float *input,
                const float *gamma,
                float *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class CUDARMSNormKernelT<ActivationPrecision::BF16> : public ITensorRMSNorm
        {
        public:
            using StorageType = uint16_t;

            explicit CUDARMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            explicit CUDARMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDARMSNormKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDARMSNormKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== ITensorRMSNorm interface =====
            bool apply(
                const float *input, const float *weight, float *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)use_bf16;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override;

            bool apply_fp16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                const Q8_1Block *input, const float *weight, Q8_1Block *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const uint16_t *input,
                const float *gamma,
                uint16_t *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class CUDARMSNormKernelT<ActivationPrecision::FP16> : public ITensorRMSNorm
        {
        public:
            using StorageType = uint16_t;

            explicit CUDARMSNormKernelT(int device_idx = 0) : device_idx_(device_idx) {}

            explicit CUDARMSNormKernelT(IWorkerGPUContext *ctx)
            {
                if (!ctx)
                    throw std::runtime_error("CUDARMSNormKernelT: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDARMSNormKernelT: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== ITensorRMSNorm interface =====
            bool apply(
                const float *input, const float *weight, float *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                bool use_bf16 = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)use_bf16;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                const uint16_t *input, const float *weight, uint16_t *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override;

            bool apply_q8_1(
                const Q8_1Block *input, const float *weight, Q8_1Block *output,
                int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override
            {
                (void)input;
                (void)weight;
                (void)output;
                (void)rows;
                (void)cols;
                (void)epsilon;
                (void)device_idx;
                return false;
            }

            bool apply_tensor(
                const TensorBase *input,
                const TensorBase *weight,
                TensorBase *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // ===== Typed API =====
            bool apply_typed(
                const uint16_t *input,
                const float *gamma,
                uint16_t *output,
                int rows, int cols,
                float epsilon = 1e-6f,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

        private:
            int device_idx_ = 0;
            IWorkerGPUContext *device_ctx_ = nullptr;
        };

    } // namespace cuda
} // namespace llaminar2
