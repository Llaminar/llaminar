/**
 * @file CPUResidualAddKernelT.h
 * @brief CPU implementation of residual add kernel
 * @author David Sanftenberg
 *
 * Template-specialized implementations of ITensorResidualAdd for CPU.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/KernelProfiler.h"
#include "../CPUKernelBase.h"

namespace llaminar2
{

    // ==========================================================================
    // FP32 Specialization
    // ==========================================================================

    template <ActivationPrecision Precision>
    class CPUResidualAddKernelT;

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::FP32> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    output[i] = input[i] + residual[i];
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
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
    };

    // ==========================================================================
    // BF16 Specialization
    // ==========================================================================

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::BF16> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
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
            return false; // BF16 kernel doesn't handle FP32
        }

        bool apply_bf16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    float in_f = simd::bf16_to_fp32(input[i]);
                    float res_f = simd::bf16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_bf16(in_f + res_f);
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::BF16)
                return false;

            return apply_bf16(
                static_cast<const uint16_t *>(input->raw_data()),
                static_cast<const uint16_t *>(residual->raw_data()),
                static_cast<uint16_t *>(output->raw_mutable_data()),
                num_elements,
                mpi_ctx,
                device_idx);
        }
    };

    // ==========================================================================
    // FP16 Specialization
    // ==========================================================================

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::FP16> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
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
            return false; // FP16 kernel doesn't handle FP32
        }

        bool apply_fp16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    float in_f = simd::fp16_to_fp32(input[i]);
                    float res_f = simd::fp16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_fp16(in_f + res_f);
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::FP16)
                return false;

            return apply_fp16(
                static_cast<const uint16_t *>(input->raw_data()),
                static_cast<const uint16_t *>(residual->raw_data()),
                static_cast<uint16_t *>(output->raw_mutable_data()),
                num_elements,
                mpi_ctx,
                device_idx);
        }
    };

} // namespace llaminar2
