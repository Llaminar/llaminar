/**
 * @file CPUSoftmaxKernelT.h
 * @brief Templated CPU implementation of Softmax kernel
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../primitives/SoftmaxPrimitives_New.h"
#include "../CPUKernelBase.h"

namespace llaminar2
{
    // Forward declarations
    class FP32Tensor;
    class BF16Tensor;
    class FP16Tensor;
    class Q8_1Tensor;
    class INT32Tensor;

    /**
     * @brief Templated CPU implementation of Softmax kernel
     */
    template <typename TensorT>
    class CPUSoftmaxKernelT : public ITensorSoftmax, public CPUKernelBase
    {
    public:
        CPUSoftmaxKernelT() = default;
        ~CPUSoftmaxKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            const float *input, float *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool apply_bf16(
            const uint16_t *input, uint16_t *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool apply_fp16(
            const uint16_t *input, uint16_t *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;
    };

} // namespace llaminar2
