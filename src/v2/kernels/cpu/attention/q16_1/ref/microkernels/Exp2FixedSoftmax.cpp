/**
 * @file Exp2FixedSoftmax.cpp
 * @brief Integer-only softmax via exp2 LUT approximation (v2)
 *
 * @see Exp2FixedSoftmax.h for algorithm details
 * @see Exp2Core.h for shared primitives
 */

#include "Exp2FixedSoftmax.h"
#include "Exp2Core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Core Algorithm
    // ============================================================================

    void exp2_softmax_int32(
        const int32_t *scores,
        int16_t *weights,
        int n,
        float alpha,
        int32_t *sum_out,
        const Exp2SoftmaxConfig &config)
    {
        // Early out for invalid inputs
        if (scores == nullptr || weights == nullptr || n <= 0)
        {
            if (sum_out)
                *sum_out = 0;
            return;
        }

        // ====================================================================
        // Pass 1: Find max score (for numerical stability)
        // ====================================================================

        int32_t max_score = exp2_find_block_max(scores, n, MASKED_SCORE);

        // All masked: output zeros
        if (max_score == MASKED_SCORE)
        {
            std::fill(weights, weights + n, static_cast<int16_t>(0));
            if (sum_out)
                *sum_out = 0;
            return;
        }

        // ====================================================================
        // Compute adaptive alpha configuration
        // ====================================================================

        AdaptiveAlphaConfig alpha_config = AdaptiveAlphaConfig::compute(alpha);

        // If alpha rounds to zero, all weights are equal (very small alpha)
        if (alpha_config.effective_alpha <= 0)
        {
            const int16_t w = config.weight_max;
            int64_t sum64 = 0;

            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == MASKED_SCORE)
                {
                    weights[i] = 0;
                }
                else
                {
                    weights[i] = w;
                    sum64 += w;
                }
            }

            if (sum_out)
            {
                *sum_out = static_cast<int32_t>(
                    std::min<int64_t>(sum64, std::numeric_limits<int32_t>::max()));
            }
            return;
        }

        // ====================================================================
        // Pass 2: Compute unnormalized exp2 values using shared primitives
        // ====================================================================

        // Temporary storage for exp values
        constexpr int STACK_THRESHOLD = 1024;
        int32_t stack_exp[STACK_THRESHOLD];
        std::unique_ptr<int32_t[]> heap_exp;
        int32_t *exp_vals;

        if (n <= STACK_THRESHOLD)
        {
            exp_vals = stack_exp;
        }
        else
        {
            heap_exp = std::make_unique<int32_t[]>(static_cast<size_t>(n));
            exp_vals = heap_exp.get();
        }

        // Use shared batch weight computation
        int64_t sum_exp = exp2_compute_block_weights(
            scores, exp_vals, n, max_score,
            alpha_config, config.frac_bits, config.lut_value_bits);

        // ====================================================================
        // Handle underflow: all exp values rounded to zero
        // ====================================================================

        if (sum_exp == 0)
        {
            // Fall back to uniform weights over unmasked positions
            const int16_t w = config.weight_max;
            int64_t sum64 = 0;

            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == MASKED_SCORE)
                {
                    weights[i] = 0;
                }
                else
                {
                    weights[i] = w;
                    sum64 += w;
                }
            }

            if (sum_out)
            {
                *sum_out = static_cast<int32_t>(
                    std::min<int64_t>(sum64, std::numeric_limits<int32_t>::max()));
            }
            return;
        }

        // ====================================================================
        // Pass 3: Normalize to INT16 weights using shared primitive
        // ====================================================================

        int32_t actual_sum = exp2_normalize_weights(
            exp_vals, weights, n, sum_exp, config.weight_max);

        if (sum_out)
        {
            *sum_out = actual_sum;
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
