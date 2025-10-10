/**
 * @file rmsnorm_t5.cpp
 * @brief T5-style RMSNorm implementation matching HuggingFace Transformers exactly
 * @author David Sanftenberg
 *
 * This implementation replicates the exact computation from T5LayerNorm:
 * variance = hidden_states.pow(2).mean(-1, keepdim=True)
 * hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
 * return self.weight * hidden_states
 */
#include "rmsnorm_t5.h"
#include <cmath>
#include <algorithm>
#include "../../logger.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar::kernels
{
    /**
     * @brief Compute T5-style RMSNorm exactly as in HuggingFace Transformers
     *
     * Formula: output[i,j] = weight[j] * input[i,j] / sqrt(mean(input[i,:]^2) + eps)
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
                            float eps,
                            bool use_parallel)
    {
        if (!input || !weight || !output || rows == 0 || cols == 0)
        {
            return;
        }

        bool parallel = use_parallel && rows > 1;

#pragma omp parallel for if (parallel) schedule(static)
        for (long long r = 0; r < static_cast<long long>(rows); ++r)
        {
            const float *input_row = input + r * cols;
            float *output_row = output + r * cols;

            // Step 1: Accumulate sum of squares in float32 (matching PyTorch .to(torch.float32))
            // PyTorch: variance = hidden_states.pow(2).mean(-1, keepdim=True)
            float sum_sq = 0.0f;
            for (size_t c = 0; c < cols; ++c)
            {
                float val = input_row[c];
                sum_sq += val * val;
            }

            // Step 2: Compute mean variance and rsqrt
            // PyTorch: variance = sum_sq / cols
            // PyTorch: inv_rms = torch.rsqrt(variance + eps)
            float variance = sum_sq / static_cast<float>(cols);
            float inv_rms = 1.0f / std::sqrt(variance + eps);

            // Step 3: Apply normalization and weight scaling
            // PyTorch: output = weight * input * inv_rms

            // Debug logging for row 4, col 56 (first divergence point)
            static bool logged_row_4 = false;
            if (r == 4 && !logged_row_4 && cols > 56)
            {
                logged_row_4 = true;
                LOG_INFO("[T5_RMSNorm] Row 4 diagnostics:");
                LOG_INFO("  sum_sq: " << sum_sq);
                LOG_INFO("  variance: " << variance);
                LOG_INFO("  sqrt(variance + eps): " << std::sqrt(variance + eps));
                LOG_INFO("  inv_rms: " << inv_rms);
                LOG_INFO("  input[56]: " << input_row[56]);
                LOG_INFO("  weight[56]: " << weight[56]);
                LOG_INFO("  output[56] computed: " << (weight[56] * input_row[56] * inv_rms));
            }

            for (size_t c = 0; c < cols; ++c)
            {
                output_row[c] = weight[c] * input_row[c] * inv_rms;
            }
        }
    }

    /**
     * @brief T5-style RMSNorm with double precision accumulation for sum-of-squares
     *
     * This variant uses double precision for the variance accumulation to reduce
     * numerical errors on very long sequences, while still matching T5 semantics.
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
                                       float eps,
                                       bool use_parallel)
    {
        static bool first_call = true;
        if (first_call)
        {
            first_call = false;
            LOG_INFO("[T5_DOUBLE] First call: rows=" << rows << ", cols=" << cols << ", eps=" << eps);
        }

        if (!input || !weight || !output || rows == 0 || cols == 0)
        {
            return;
        }

        bool parallel = use_parallel && rows > 1;

#pragma omp parallel for if (parallel) schedule(static)
        for (long long r = 0; r < static_cast<long long>(rows); ++r)
        {
            const float *input_row = input + r * cols;
            float *output_row = output + r * cols;

            // Step 1: Accumulate sum of squares in double precision
            double sum_sq = 0.0;
            for (size_t c = 0; c < cols; ++c)
            {
                double val = static_cast<double>(input_row[c]);
                sum_sq += val * val;
            }

            // Step 2: Compute mean variance and rsqrt
            double variance = sum_sq / static_cast<double>(cols);
            double inv_rms = 1.0 / std::sqrt(variance + static_cast<double>(eps));

            // Debug logging for specific positions
            // Note: In MPI setup, global row 4 might be local row 1 on rank 1
            static int debug_count = 0;
            if (debug_count < 10)
            { // Log first 10 rows we process
                debug_count++;
                if (cols > 56)
                {
                    LOG_INFO("[T5_DOUBLE] Row " << r << " diagnostics:");
                    LOG_INFO("  sum_sq: " << sum_sq);
                    LOG_INFO("  variance: " << variance);
                    LOG_INFO("  sqrt(variance + eps): " << std::sqrt(variance + static_cast<double>(eps)));
                    LOG_INFO("  inv_rms: " << inv_rms);
                    LOG_INFO("  input[56]: " << static_cast<double>(input_row[56]));
                    LOG_INFO("  weight[56]: " << weight[56]);
                    float output_56 = static_cast<float>(weight[56] * static_cast<double>(input_row[56]) * inv_rms);
                    LOG_INFO("  output[56] computed: " << output_56);
                }
            }

            // Step 3: Apply normalization and weight scaling (cast back to float)
            float inv_rms_f = static_cast<float>(inv_rms);
            for (size_t c = 0; c < cols; ++c)
            {
                output_row[c] = weight[c] * input_row[c] * inv_rms_f;
            }
        }
    }

} // namespace llaminar::kernels
