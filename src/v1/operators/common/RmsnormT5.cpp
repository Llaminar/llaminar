/**
 * @file RmsnormT5.cpp
 * @brief T5-style RMSNorm implementation matching HuggingFace Transformers exactly
 * @author David Sanftenberg
 *
 * This implementation now delegates to RmsnormCore with t5_compat_mode enabled,
 * providing the same PyTorch T5LayerNorm semantics while eliminating code duplication.
 *
 * Formula: output[i,j] = weight[j] * input[i,j] / sqrt(mean(input[i,:]^2) + eps)
 *
 * Matches PyTorch T5LayerNorm:
 *   variance = hidden_states.pow(2).mean(-1, keepdim=True)
 *   hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
 *   return self.weight * hidden_states
 */
#include "RmsnormT5.h"
#include "RmsnormCore.h"

namespace llaminar::kernels
{
    void rmsnorm_t5_forward(const float *input,
                            const float *weight,
                            float *output,
                            size_t rows,
                            size_t cols,
                            float eps,
                            bool use_parallel)
    {
        // Delegate to RmsnormCore with T5 compatibility mode (float32 accumulation)
        RMSNormExecOptions opts;
        opts.allow_parallel = use_parallel;
        opts.t5_compat_mode = true; // Use float32 accumulation for PyTorch parity

        rmsnorm_row_major_fused(input, weight, output, rows, cols, eps,
                                GammaMode::REPLICATED, 0, opts);
    }

    void rmsnorm_t5_forward_double_acc(const float *input,
                                       const float *weight,
                                       float *output,
                                       size_t rows,
                                       size_t cols,
                                       float eps,
                                       bool use_parallel)
    {
        // Delegate to RmsnormCore with standard double precision accumulation
        // Note: Despite the name, this uses RmsnormCore's default double accumulation
        // rather than T5's float32 mode, providing better numerical stability
        RMSNormExecOptions opts;
        opts.allow_parallel = use_parallel;
        opts.t5_compat_mode = false; // Use double precision for better accuracy

        rmsnorm_row_major_fused(input, weight, output, rows, cols, eps,
                                GammaMode::REPLICATED, 0, opts);
    }

} // namespace llaminar::kernels
