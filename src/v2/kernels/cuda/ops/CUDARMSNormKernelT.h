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

#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>

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
            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
            int device_idx_;
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
            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
            int device_idx_;
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
            ~CUDARMSNormKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
            int device_idx_;
        };

    } // namespace cuda
} // namespace llaminar2
