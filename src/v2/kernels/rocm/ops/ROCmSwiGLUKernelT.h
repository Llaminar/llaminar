/**
 * @file ROCmSwiGLUKernelT.h
 * @brief ROCm SwiGLU activation kernel with typed precision support
 *
 * ROCm (HIP) implementation of SwiGLU following the same template pattern as
 * CUDASwiGLUKernelT. Specialized for FP32, BF16, and FP16 precisions.
 *
 * @author Llaminar Team
 * @date 2026-01-17
 */

#pragma once

#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // Primary Template (Unsupported Precisions)
        // =========================================================================

        template <ActivationPrecision Precision>
        class ROCmSwiGLUKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "ROCmSwiGLUKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class ROCmSwiGLUKernelT<ActivationPrecision::FP32> : public ITensorSwiGLU
        {
        public:
            using StorageType = float;

            explicit ROCmSwiGLUKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmSwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
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
                int device_idx = -1) override
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
                int device_idx = -1) override
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
            int device_idx_;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class ROCmSwiGLUKernelT<ActivationPrecision::BF16> : public ITensorSwiGLU
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmSwiGLUKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmSwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
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
                int device_idx = -1) override;

            bool apply_fp16(
                const uint16_t *gate, const uint16_t *up, uint16_t *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
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
                int device_idx = -1) override
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
            int device_idx_;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class ROCmSwiGLUKernelT<ActivationPrecision::FP16> : public ITensorSwiGLU
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmSwiGLUKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmSwiGLUKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorSwiGLU interface =====
            bool apply(
                const float *gate, const float *up, float *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
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
                int device_idx = -1) override
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
                int device_idx = -1) override;

            bool apply_q8_1(
                const void *gate, const void *up, void *output,
                int rows, int cols,
                bool add_residual,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
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
            int device_idx_;
        };

    } // namespace rocm
} // namespace llaminar2
