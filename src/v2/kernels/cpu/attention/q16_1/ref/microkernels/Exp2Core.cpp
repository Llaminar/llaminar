/**
 * @file Exp2Core.cpp
 * @brief Core exp2 LUT primitives for integer softmax implementations
 *
 * @see Exp2Core.h for design rationale and API documentation
 */

#include "Exp2Core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // LUT Management (Thread-Safe Singleton)
    // ============================================================================

    namespace
    {
        /// Thread-safe singleton LUT for 2^(-frac) values
        /// 2048 entries × 4 bytes = 8KB, fits easily in L1 cache
        struct Exp2LUT
        {
            std::array<uint32_t, EXP2_LUT_SIZE> data{};
            int value_bits = 0;
            bool ready = false;
            std::mutex init_mutex;

            void initialize(int lut_value_bits)
            {
                std::lock_guard<std::mutex> lock(init_mutex);

                if (ready && value_bits == lut_value_bits)
                {
                    return;
                }

                const double scale = static_cast<double>(1ULL << lut_value_bits);

                for (int i = 0; i < EXP2_LUT_SIZE; ++i)
                {
                    // u ∈ [0, 1) at 2048 uniformly spaced points
                    const double u = static_cast<double>(i) / static_cast<double>(EXP2_LUT_SIZE);

                    // 2^(-u) ∈ (0.5, 1]
                    const double v = std::pow(2.0, -u);

                    // Quantize to integer with rounding
                    const double q = std::round(v * scale);

                    // Clamp to uint32 range (should never exceed with 30-bit scale)
                    uint64_t qi = static_cast<uint64_t>(q);
                    if (qi > 0xFFFFFFFFULL)
                    {
                        qi = 0xFFFFFFFFULL;
                    }

                    data[static_cast<size_t>(i)] = static_cast<uint32_t>(qi);
                }

                value_bits = lut_value_bits;
                ready = true;
            }
        };

        Exp2LUT &get_lut_instance()
        {
            static Exp2LUT instance;
            return instance;
        }

    } // namespace

    void ensure_exp2_lut_initialized(int lut_value_bits)
    {
        get_lut_instance().initialize(lut_value_bits);
    }

    const uint32_t *get_exp2_lut_data()
    {
        auto &lut = get_lut_instance();
        return lut.ready ? lut.data.data() : nullptr;
    }

    // ============================================================================
    // Adaptive Alpha Configuration
    // ============================================================================

    AdaptiveAlphaConfig AdaptiveAlphaConfig::compute(float alpha)
    {
        AdaptiveAlphaConfig config;
        config.original_alpha = alpha;

        // For very small alpha, we need to scale up to maintain precision
        // Target: effective_alpha should be in range [1e6, 1e9] for good precision
        if (alpha < 1e-6f)
        {
            // Very small alpha (e.g., 7.45e-9 for KV_CACHE_SCALE=8.0)
            // Scale up by 2^30 to get into workable range
            config.alpha_shift = 30;
            config.effective_alpha = static_cast<int64_t>(
                std::llround(static_cast<double>(alpha) * static_cast<double>(1ULL << 30)));
        }
        else if (alpha < 1e-3f)
        {
            // Small alpha
            config.alpha_shift = 20;
            config.effective_alpha = static_cast<int64_t>(
                std::llround(static_cast<double>(alpha) * static_cast<double>(1ULL << 20)));
        }
        else if (alpha < 1.0f)
        {
            // Normal alpha range
            config.alpha_shift = 10;
            config.effective_alpha = static_cast<int64_t>(
                std::llround(static_cast<double>(alpha) * static_cast<double>(1ULL << 10)));
        }
        else
        {
            // Large alpha (rare)
            config.alpha_shift = 0;
            config.effective_alpha = static_cast<int64_t>(std::llround(alpha));
        }

        return config;
    }

    // ============================================================================
    // Core Exp2 Weight Computation
    // ============================================================================

    uint32_t exp2_compute_weight(
        int32_t delta,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits,
        int lut_value_bits)
    {
        // Ensure LUT is ready
        ensure_exp2_lut_initialized(lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();

        // delta = 0 means this is the max position
        if (delta <= 0)
        {
            return static_cast<uint32_t>(1U << lut_value_bits);
        }

        // Compute t = delta * alpha * log2(e)
        // With adaptive scaling:
        //   t = delta * (effective_alpha / 2^alpha_shift) * log2(e)
        //     = delta * effective_alpha * log2(e) / 2^alpha_shift
        //
        // We compute M = effective_alpha * log2(e) * 2^beta_scale_bits
        // Then t_fixed = (delta * M) >> (beta_scale_bits - frac_bits + alpha_shift)

        constexpr int beta_scale_bits = 24;
        double effective_beta = static_cast<double>(alpha_config.effective_alpha) * LOG2E;
        int64_t M = static_cast<int64_t>(
            std::llround(effective_beta * static_cast<double>(1ULL << beta_scale_bits)));

        int shift_for_t = beta_scale_bits - frac_bits + alpha_config.alpha_shift;

        // t_fixed = delta * M >> shift_for_t
        int64_t prod = static_cast<int64_t>(delta) * M;
        int64_t t_fixed = (shift_for_t >= 0)
                              ? (prod >> shift_for_t)
                              : (prod << (-shift_for_t));

        // Decompose: t = ip + frac
        int64_t ip = t_fixed >> frac_bits;
        int frac_idx = static_cast<int>(t_fixed & ((1 << frac_bits) - 1));

        // Underflow check: if ip >= 31, result is effectively 0
        if (ip >= 31)
        {
            return 0;
        }

        // 2^(-t) = lut[frac] >> ip
        return lut[frac_idx] >> static_cast<int>(ip);
    }

    uint32_t exp2_compute_weight(
        int32_t delta,
        float alpha,
        int frac_bits,
        int lut_value_bits)
    {
        AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);
        return exp2_compute_weight(delta, config, frac_bits, lut_value_bits);
    }

    void exp2_compute_rescale(
        int32_t m_old,
        int32_t m_new,
        const AdaptiveAlphaConfig &alpha_config,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits,
        int lut_value_bits)
    {
        // m_new >= m_old always (running max is non-decreasing)
        // We need: correction = exp(m_old - m_new) = 2^((m_old - m_new) * alpha * log2e)
        //        = 2^(-delta * beta) where delta = m_new - m_old >= 0

        if (m_old == MASKED_SCORE || m_new <= m_old)
        {
            // No rescaling needed - return identity
            scale_num = 1 << lut_value_bits;
            scale_shift = lut_value_bits;
            return;
        }

        int32_t delta = m_new - m_old;

        // Compute t using same precision as weight computation
        double effective_beta = static_cast<double>(alpha_config.effective_alpha) * LOG2E;
        double t_scaled = static_cast<double>(delta) * effective_beta;
        double t = t_scaled / static_cast<double>(1ULL << alpha_config.alpha_shift);

        // Decompose t = ip + frac
        int ip = static_cast<int>(t);
        double frac = t - ip;

        // If ip >= 31, underflow to zero
        if (ip >= 31)
        {
            scale_num = 0;
            scale_shift = 0;
            return;
        }

        // Get exp2 LUT for fractional part
        ensure_exp2_lut_initialized(lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();

        // Index into LUT: frac * 2^frac_bits
        int lut_idx = static_cast<int>(frac * (1 << frac_bits));
        lut_idx = std::clamp(lut_idx, 0, EXP2_LUT_SIZE - 1);

        // 2^(-t) = 2^(-ip) * 2^(-frac) = lut[idx] >> ip
        scale_num = static_cast<int32_t>(lut[lut_idx]);
        scale_shift = lut_value_bits + ip;
    }

    // ============================================================================
    // Batch Weight Computation
    // ============================================================================

    int64_t exp2_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits,
        int lut_value_bits)
    {
        // Initialize LUT
        ensure_exp2_lut_initialized(lut_value_bits);
        const uint32_t *lut = get_exp2_lut_data();

        // Precompute M for the loop
        constexpr int beta_scale_bits = 24;
        double effective_beta = static_cast<double>(alpha_config.effective_alpha) * LOG2E;
        int64_t M = static_cast<int64_t>(
            std::llround(effective_beta * static_cast<double>(1ULL << beta_scale_bits)));

        int shift_for_t = beta_scale_bits - frac_bits + alpha_config.alpha_shift;

        uint32_t one = static_cast<uint32_t>(1U << lut_value_bits);

        int64_t sum = 0;

        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] == MASKED_SCORE)
            {
                weights[i] = 0;
                continue;
            }

            int32_t delta = running_max - scores[i];

            // delta = 0 means this is the max
            if (delta <= 0)
            {
                weights[i] = static_cast<int32_t>(one);
                sum += one;
                continue;
            }

            // t_fixed = delta * M >> shift_for_t
            int64_t prod = static_cast<int64_t>(delta) * M;
            int64_t t_fixed = (shift_for_t >= 0)
                                  ? (prod >> shift_for_t)
                                  : (prod << (-shift_for_t));

            // Decompose: t = ip + frac
            int64_t ip = t_fixed >> frac_bits;
            int frac_idx = static_cast<int>(t_fixed & ((1 << frac_bits) - 1));

            // Underflow check
            if (ip >= 31)
            {
                weights[i] = 0;
                continue;
            }

            // 2^(-t) = lut[frac] >> ip
            uint32_t w = lut[frac_idx] >> static_cast<int>(ip);
            weights[i] = static_cast<int32_t>(w);
            sum += w;
        }

        return sum;
    }

    int64_t exp2_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        float alpha,
        int frac_bits,
        int lut_value_bits)
    {
        AdaptiveAlphaConfig config = AdaptiveAlphaConfig::compute(alpha);
        return exp2_compute_block_weights(
            scores, weights, block_size, running_max,
            config, frac_bits, lut_value_bits);
    }

    // ============================================================================
    // Utility Functions
    // ============================================================================

    int32_t exp2_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value)
    {
        int32_t block_max = mask_value;

        for (int i = 0; i < block_size; ++i)
        {
            if (scores[i] != mask_value && scores[i] > block_max)
            {
                block_max = scores[i];
            }
        }

        return block_max;
    }

    int32_t exp2_normalize_weights(
        const int32_t *block_weights,
        int16_t *final_weights,
        int n,
        int64_t weight_sum,
        int16_t weight_max)
    {
        if (weight_sum == 0)
        {
            // No unmasked positions - output zeros
            std::fill(final_weights, final_weights + n, static_cast<int16_t>(0));
            return 0;
        }

        // Normalize: w_final = block_weights * weight_max / weight_sum
        int64_t half = weight_sum / 2; // For rounding
        int64_t sum = 0;

        for (int i = 0; i < n; ++i)
        {
            if (block_weights[i] == 0)
            {
                final_weights[i] = 0;
                continue;
            }

            int64_t num = static_cast<int64_t>(block_weights[i]) *
                          static_cast<int64_t>(weight_max);
            int64_t w = (num + half) / weight_sum;

            w = std::min<int64_t>(w, weight_max);
            final_weights[i] = static_cast<int16_t>(w);
            sum += w;
        }

        return static_cast<int32_t>(std::min<int64_t>(sum, std::numeric_limits<int32_t>::max()));
    }

} // namespace llaminar2::kernels::q16_1::microkernels
