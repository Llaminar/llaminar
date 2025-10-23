/**
 * @file RmsnormT5.h
 * @brief T5-style RMSNorm implementation matching HuggingFace Transformers exactly
 * @author David Sanftenberg
 */
#pragma once

#include <cstddef>

namespace llaminar::kernels
{
    /**
     * @brief Compute T5-style RMSNorm exactly as in HuggingFace Transformers
     *
     * Formula: output[i,j] = weight[j] * input[i,j] / sqrt(mean(input[i,:]^2) + eps)
     *
     * Matches PyTorch T5LayerNorm:
     *   variance = hidden_states.pow(2).mean(-1, keepdim=True)
     *   hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
     *   return self.weight * hidden_states
     *
     * @param input Input tensor (rows x cols)
     * @param weight Scale weights (cols)
     * @param output Output tensor (rows x cols)
     * @param rows Number of rows (sequence length)
     * @param cols Number of columns (hidden dimension)
     * @param eps Variance epsilon (default 1e-6)
     * @param use_parallel Whether to parallelize over rows
     */
    void rmsnorm_t5_forward(const float *input,
                            const float *weight,
                            float *output,
                            size_t rows,
                            size_t cols,
                            float eps = 1e-6f,
                            bool use_parallel = true);

    /**
     * @brief T5-style RMSNorm with double precision accumulation for sum-of-squares
     *
     * @param input Input tensor (rows x cols)
     * @param weight Scale weights (cols)
     * @param output Output tensor (rows x cols)
     * @param rows Number of rows (sequence length)
     * @param cols Number of columns (hidden dimension)
     * @param eps Variance epsilon (default 1e-6)
     * @param use_parallel Whether to parallelize over rows
     */
    void rmsnorm_t5_forward_double_acc(const float *input,
                                       const float *weight,
                                       float *output,
                                       size_t rows,
                                       size_t cols,
                                       float eps = 1e-6f,
                                       bool use_parallel = true);

} // namespace llaminar::kernels
