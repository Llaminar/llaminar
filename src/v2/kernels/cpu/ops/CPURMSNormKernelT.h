/**
 * @file CPURMSNormKernelT.h
 * @brief CPU implementation of RMSNorm kernel
 *
 * RMSNorm: output = input * weight * rsqrt(mean(input^2) + epsilon)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../CPUKernelBase.h"

namespace llaminar2
{
    // Forward declarations
    class FP32Tensor;
    class BF16Tensor;
    class FP16Tensor;
    class INT32Tensor;
    class Q8_1Tensor;

    /**
     * @brief CPU implementation of RMSNorm kernel
     */
    template <typename TensorT>
    class CPURMSNormKernelT : public ITensorRMSNorm, public CPUKernelBase
    {
    public:
        CPURMSNormKernelT() = default;
        ~CPURMSNormKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override;

        bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override;

        bool apply_int32_to_int8(
            const int32_t *input, const float *weight, int8_t *output,
            float *scales, int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) override;
    };

} // namespace llaminar2
