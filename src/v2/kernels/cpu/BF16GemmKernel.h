/**
 * @file BF16GemmKernel.h
 * @brief CPU BF16 GEMM kernel with MKL optimization
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"

namespace llaminar2
{

    /**
     * @brief CPU BF16 GEMM kernel
     *
     * Uses Intel MKL cblas_gemm_bf16bf16f32 when available (Ice Lake+),
     * otherwise falls back to software BF16→FP32 expansion + cblas_sgemm.
     */
    class BF16GemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct GEMM kernel for BF16 weight tensor
         *
         * @param weight_tensor BF16 weight tensor (B matrix)
         */
        explicit BF16GemmKernel(const BF16Tensor *weight_tensor)
            : weight_tensor_(weight_tensor)
        {
        }

        ~BF16GemmKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        const BF16Tensor *weight_tensor_;
    };

} // namespace llaminar2
