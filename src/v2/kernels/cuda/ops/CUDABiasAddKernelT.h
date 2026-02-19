/**
 * @file CUDABiasAddKernelT.h
 * @brief CUDA implementation of bias add kernel
 * @author David Sanftenberg
 *
 * Template-specialized implementations of bias addition for CUDA.
 * Uses extern "C" wrappers to call CUDA kernels in CUDABiasAddKernels.cu.
 *
 * Bias add broadcasts bias vector [N] across M rows: output[i,j] += bias[j]
 */

#pragma once

#include "../../../utils/Logger.h"

// Forward declarations for CUDA kernels
extern "C"
{
    bool cudaOps_bias_add_fp32(float *output, const float *bias, int M, int N, int device_idx);
    bool cudaOps_bias_add_fp32_scaled(float *output, const float *bias, float scale, int M, int N, int device_idx);
}

namespace llaminar2::cuda
{

    /**
     * @brief CUDA Bias Add helper class
     *
     * Provides static methods for adding bias to GEMM output on GPU.
     * Handles broadcasting bias [N] across rows of output [M, N].
     */
    class CUDABiasAdd
    {
    public:
        /**
         * @brief Add bias to FP32 output tensor (in-place)
         *
         * Computes: output[i,j] += bias[j] for all i in [0, M), j in [0, N)
         *
         * @param output Device pointer to output tensor [M, N] (modified in-place)
         * @param bias Device pointer to bias vector [N]
         * @param M Number of rows (typically seq_len or batch_size)
         * @param N Number of columns (bias length, typically output dimension)
         * @param device_idx CUDA device ID
         * @return true on success
         */
        static bool add_fp32(
            float *output,
            const float *bias,
            int M,
            int N,
            int device_idx = -1)
        {
            if (!output || !bias)
            {
                LOG_ERROR("[CUDABiasAdd] Null pointer: output=" << (void *)output
                                                                << " bias=" << (void *)bias);
                return false;
            }

            LOG_DEBUG("[CUDABiasAdd::add_fp32] M=" << M << " N=" << N
                                                   << " device=" << device_idx);

            return cudaOps_bias_add_fp32(output, bias, M, N, device_idx);
        }

        /**
         * @brief Add scaled bias to FP32 output tensor (in-place)
         *
         * Computes: output[i,j] = output[i,j] * scale + bias[j]
         *
         * @param output Device pointer to output tensor [M, N] (modified in-place)
         * @param bias Device pointer to bias vector [N]
         * @param scale Scale factor applied to output before bias add
         * @param M Number of rows
         * @param N Number of columns (bias length)
         * @param device_idx CUDA device ID
         * @return true on success
         */
        static bool add_fp32_scaled(
            float *output,
            const float *bias,
            float scale,
            int M,
            int N,
            int device_idx = -1)
        {
            if (!output || !bias)
            {
                LOG_ERROR("[CUDABiasAdd] Null pointer: output=" << (void *)output
                                                                << " bias=" << (void *)bias);
                return false;
            }

            LOG_DEBUG("[CUDABiasAdd::add_fp32_scaled] M=" << M << " N=" << N
                                                          << " scale=" << scale
                                                          << " device=" << device_idx);

            return cudaOps_bias_add_fp32_scaled(output, bias, scale, M, N, device_idx);
        }
    };

} // namespace llaminar2::cuda
