/**
 * @file ROCmResidualAddKernelT.h
 * @brief ROCm Residual Add kernel template header
 *
 * Provides FP32, BF16, and FP16 specializations of ITensorResidualAdd
 * for AMD GPUs via HIP. Direct port of CUDA implementation.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include <cstdint>

// Forward declarations for HIP kernels (implemented in ROCmResidualAddKernels.hip)
extern "C"
{
    bool rocmOps_residual_add_fp32(const float *input, const float *residual, float *output, int size, int device_idx);
    bool rocmOps_residual_add_bf16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx);
    bool rocmOps_residual_add_fp16(const uint16_t *input, const uint16_t *residual, uint16_t *output, int size, int device_idx);
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
            explicit ROCmResidualAddKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmResidualAddKernelT() override = default;

            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0; // Supports any ROCm GPU device
            }

            bool apply(
                const float *input, const float *residual, float *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::FP32] Executing on device " << dev);
                return rocmOps_residual_add_fp32(input, residual, output, static_cast<int>(num_elements), dev);
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

                return apply(
                    input->data(),
                    residual->data(),
                    output->mutable_data(),
                    num_elements,
                    mpi_ctx,
                    device_idx);
            }

        private:
            int device_idx_;
        };

        // ============================================================================
        // BF16 Specialization
        // ============================================================================

        template <>
        class ROCmResidualAddKernelT<ActivationPrecision::BF16> : public ITensorResidualAdd
        {
        public:
            explicit ROCmResidualAddKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmResidualAddKernelT() override = default;

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
                KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::BF16] Executing on device " << dev);
                return rocmOps_residual_add_bf16(input, residual, output, static_cast<int>(num_elements), dev);
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
        };

        // ============================================================================
        // FP16 Specialization
        // ============================================================================

        template <>
        class ROCmResidualAddKernelT<ActivationPrecision::FP16> : public ITensorResidualAdd
        {
        public:
            explicit ROCmResidualAddKernelT(int device_idx = 0) : device_idx_(device_idx) {}
            ~ROCmResidualAddKernelT() override = default;

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
                return false; // FP16 kernel doesn't handle FP32 pointers
            }

            bool apply_fp16(
                const uint16_t *input, const uint16_t *residual, uint16_t *output,
                size_t num_elements,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
                (void)mpi_ctx;
                int dev = (device_idx >= 0) ? device_idx : device_idx_;
                LOG_DEBUG("[ROCmResidualAddKernelT::FP16] Executing on device " << dev);
                return rocmOps_residual_add_fp16(input, residual, output, static_cast<int>(num_elements), dev);
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
        };

    } // namespace rocm
} // namespace llaminar2
