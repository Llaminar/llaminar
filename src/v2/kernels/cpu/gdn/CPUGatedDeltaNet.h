/**
 * @file CPUGatedDeltaNet.h
 * @brief CPU implementation of ITensorGatedDeltaNet
 *
 * Delta rule recurrence for GDN linear attention.
 * OpenMP-parallelized across heads.
 *
 * The kernel owns ALL preprocessing:
 * - L2 normalization of Q and K (when use_qk_l2norm is true)
 * - Query scaling by 1/sqrt(d_k)
 * - Gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
 * - Beta sigmoid: beta_sig = sigmoid(beta_raw)
 *
 * This ensures a future CUDA kernel can do all math on-device.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"

#include <vector>

namespace llaminar2
{

    class CPUGatedDeltaNet : public ITensorGatedDeltaNet
    {
    public:
        bool chunk_forward(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm) override;

        bool recurrent_step(
            const float *q, const float *k, const float *v,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int n_heads, int d_k, int d_v,
            bool use_qk_l2norm) override;

    private:
        /// Compute gate values: g = A_log * softplus(alpha + dt_bias), beta_sig = sigmoid(beta_raw)
        static void computeGates(
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *g_out, float *beta_sig_out,
            int seq_len, int n_heads);

        /// L2 normalize vectors per head
        static void l2normalize(float *data, int seq_len, int n_heads, int head_dim);

        /// Ensure scratch buffers are large enough, reallocating only when needed
        void ensureScratch(int seq_len, int n_heads, int d_k, int d_v);

        // Reusable scratch buffers (grow-only, never shrink during lifetime)
        std::vector<float> q_scratch_;       ///< Preprocessed Q buffer
        std::vector<float> k_scratch_;       ///< Preprocessed K buffer
        std::vector<float> gate_scratch_;    ///< Gate values
        std::vector<float> beta_sig_scratch_; ///< Sigmoid of beta
    };

} // namespace llaminar2
