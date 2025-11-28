/**
 * @file CPUSwiGLUKernelT.h
 * @brief CPU implementation of SwiGLU activation
 *
 * SwiGLU: output = swish(gate) * up
 * where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
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
    class Q8_1Tensor;

    /**
     * @brief CPU implementation of SwiGLU kernel
     */
    template <typename TensorT>
    class CPUSwiGLUKernelT : public ITensorSwiGLU, public CPUKernelBase
    {
    public:
        CPUSwiGLUKernelT() = default;
        ~CPUSwiGLUKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

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
            int device_idx = -1) override;

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
            int device_idx = -1) override;
    };

    // Backward compatibility alias
    using CPUSwiGLUKernel = CPUSwiGLUKernelT<FP32Tensor>;

} // namespace llaminar2
