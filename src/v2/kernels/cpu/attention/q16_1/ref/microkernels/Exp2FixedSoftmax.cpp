/**
 * @file Exp2FixedSoftmax.cpp
 * @brief Integer-only softmax via exp2 LUT approximation (v2)
 *
 * @see Exp2FixedSoftmax.h for algorithm details
 */

#include "Exp2FixedSoftmax.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Constants
    // ============================================================================

    namespace
    {
        /// log₂(e) ≈ 1.4427
        constexpr double LOG2E = 1.4426950408889634073599246810018921;

        /// Mask value for causal attention / padding
        constexpr int32_t MASKED = std::numeric_limits<int32_t>::min();

    } // namespace

    // ============================================================================
    // LUT Management
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

        int32_t max_score = MASKED;
        int unmasked_count = 0;

        for (int i = 0; i < n; ++i)
        {
            const int32_t s = scores[i];
            if (s != MASKED)
            {
                if (s > max_score)
                    max_score = s;
                ++unmasked_count;
            }
        }

        // All masked: output zeros
        if (unmasked_count == 0)
        {
            std::fill(weights, weights + n, static_cast<int16_t>(0));
            if (sum_out)
                *sum_out = 0;
            return;
        }

        // ====================================================================
        // Compute fixed-point β = α × log₂(e)
        // ====================================================================
        // This is the ONLY floating-point operation (done once, not per-element)

        const double beta = static_cast<double>(alpha) * LOG2E;
        const int64_t M = static_cast<int64_t>(
            std::llround(beta * static_cast<double>(1ULL << config.beta_scale_bits)));

        // If β rounds to zero, all weights are equal (very small alpha)
        if (M <= 0)
        {
            const int16_t w = config.weight_max;
            int64_t sum64 = 0;

            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == MASKED)
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
        // Initialize LUT (thread-safe, idempotent)
        // ====================================================================

        auto &lut = get_lut_instance();
        lut.initialize(config.lut_value_bits);

        // ====================================================================
        // Pass 2: Compute unnormalized exp2 values
        // ====================================================================
        // exp(-α×δ) = 2^(-t) where t = δ × β
        // t = ip + frac, so 2^(-t) = 2^(-ip) × 2^(-frac)
        // 2^(-ip) is a right-shift, 2^(-frac) is LUT lookup

        const int shift_for_t = config.beta_scale_bits - config.frac_bits;

        // Temporary storage for exp values (TODO: use scratch buffer from pool)
        // Using stack allocation for small n, heap for large
        constexpr int STACK_THRESHOLD = 1024;
        uint32_t stack_exp[STACK_THRESHOLD];
        std::unique_ptr<uint32_t[]> heap_exp;
        uint32_t *exp_vals;

        if (n <= STACK_THRESHOLD)
        {
            exp_vals = stack_exp;
        }
        else
        {
            heap_exp = std::make_unique<uint32_t[]>(static_cast<size_t>(n));
            exp_vals = heap_exp.get();
        }

        uint64_t sum_exp = 0;
        const uint32_t one = static_cast<uint32_t>(1U << config.lut_value_bits);

        for (int i = 0; i < n; ++i)
        {
            const int32_t s = scores[i];

            if (s == MASKED)
            {
                exp_vals[i] = 0;
                continue;
            }

            const int32_t delta = max_score - s;

            // δ = 0 → exp(0) = 1
            if (delta <= 0)
            {
                exp_vals[i] = one;
                sum_exp += one;
                continue;
            }

            // t_fixed = δ × M, in Q(frac_bits) format
            const int64_t prod = static_cast<int64_t>(delta) * M;
            const int64_t t_fixed = (shift_for_t >= 0)
                                        ? (prod >> shift_for_t)
                                        : (prod << (-shift_for_t));

            // Decompose: t = ip + frac
            const int64_t ip = t_fixed >> config.frac_bits;
            const int frac = static_cast<int>(t_fixed & ((1 << config.frac_bits) - 1));

            // If ip ≥ 31, 2^(-ip) underflows to 0
            if (ip >= 31)
            {
                exp_vals[i] = 0;
                continue;
            }

            // 2^(-t) = 2^(-ip) × 2^(-frac)
            //        = lut[frac] >> ip
            const uint32_t frac_val = lut.data[static_cast<size_t>(frac)];
            const uint32_t exp_val = frac_val >> static_cast<int>(ip);

            exp_vals[i] = exp_val;
            sum_exp += exp_val;
        }

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
                if (scores[i] == MASKED)
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
        // Pass 3: Normalize to INT16 weights
        // ====================================================================
        // w_i = exp_vals[i] × weight_max / sum_exp

        int64_t sum_w = 0;
        const uint64_t half = sum_exp / 2; // For rounding

        for (int i = 0; i < n; ++i)
        {
            if (scores[i] == MASKED)
            {
                weights[i] = 0;
                continue;
            }

            // Multiply first, then divide (with rounding)
            const uint64_t num = static_cast<uint64_t>(exp_vals[i]) *
                                 static_cast<uint64_t>(config.weight_max);
            const uint64_t w_u = (num + half) / sum_exp;

            // Clamp to weight_max (should rarely exceed due to rounding)
            const int16_t w = static_cast<int16_t>(
                std::min<uint64_t>(w_u, static_cast<uint64_t>(config.weight_max)));

            weights[i] = w;
            sum_w += w;
        }

        if (sum_out)
        {
            *sum_out = static_cast<int32_t>(
                std::min<int64_t>(sum_w, std::numeric_limits<int32_t>::max()));
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
