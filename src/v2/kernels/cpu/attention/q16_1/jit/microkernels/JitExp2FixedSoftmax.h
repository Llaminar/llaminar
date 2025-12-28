/**
 * @file JitExp2FixedSoftmax.h
 * @brief JIT Microkernel for Exp2 Fixed-Point Softmax
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated integer-domain softmax using exp2 LUT approximation.
 * Converts INT32 scores to INT16 attention weights [0, 32767].
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM: Exp2 Fixed-Point Softmax
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For each score s[kv]:
 *   1. delta = max_score - s[kv]                    // Distance from max
 *   2. t = delta × beta, where beta = alpha × log₂(e)
 *   3. Decompose t into integer (ip) and fractional (frac) parts
 *   4. result = LUT[frac × 256] >> ip               // 256-entry LUT lookup
 *   5. weight[kv] = clamp(result >> norm_shift, 0, 32767)
 *
 * LUT Design:
 *   - 256 entries for fractional part [0, 1)
 *   - Each entry: 2^(-frac/256) with 30-bit precision
 *   - Integer part handled by right shift
 *
 * Output: INT16 weights in [0, 32767] (VNNI-compatible signed INT16)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER CONTRACT
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * INPUTS (consumed, then overwritten):
 *   Scratch0-3 (zmm20-23): INT32 scores from Q16DotProduct
 *
 * OUTPUTS (overwrite inputs):
 *   Scratch0-3 (zmm20-23): INT16 weights [0, 32767]
 *
 * STATE (persistent across KV iterations - FA2 prefill):
 *   StateMax (zmm16):    Running max score (INT32)
 *   StateSum (zmm17):    Running sum of weights (INT64)
 *   StateWeight (zmm18): Current max weight (for rescaling check)
 *   StateCorr (zmm19):   Correction factor (rescale accumulator)
 *
 * SCRATCH:
 *   Scratch4-5 (zmm24-25): Delta computation, LUT indexing temps
 *   Input4-7 (zmm12-15):   LUT entries (loaded via gather)
 *
 * CONSTANTS:
 *   ConstLog2E (zmm28): log₂(e) ≈ 1.4427 for ln→log₂ conversion
 *   Const256 (zmm29):   256.0f for fractional index computation
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * DECODE vs PREFILL MODES
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * DECODE (single query):
 *   - Process all KV positions in one pass
 *   - StateZone unused (no online tracking needed)
 *   - Normalize at end using final sum
 *
 * PREFILL (FA2 tiling):
 *   - Process KV in tiles, StateZone tracks running (m, l)
 *   - Online softmax: rescale previous weights when max increases
 *   - Lazy rescaling (Option A): defer to finalization
 *
 * @see Q16RegisterAllocation.h for register zone definitions
 */

#pragma once

#include "../Q16RegisterAllocation.h"
#include "tensors/BlockStructures.h"
#include "../../../../jit/RegisterGuard.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <array>

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // Exp2 LUT Data
    // ============================================================================

    /**
     * @brief Static LUT for exp2 approximation
     *
     * LUT[i] = 2^(-i/256) × 2^30, for i in [0, 255]
     * This gives ~30 bits of precision for the fractional exp2.
     */
    struct Exp2LUT
    {
        static constexpr int LUT_SIZE = 256;
        static constexpr int LUT_BITS = 30;    ///< Precision bits in LUT entries
        static constexpr int WEIGHT_BITS = 15; ///< Output weight precision (INT16)

        alignas(64) uint32_t data[LUT_SIZE]; ///< LUT entries
        bool initialized = false;

        void initialize()
        {
            if (initialized)
                return;
            for (int i = 0; i < LUT_SIZE; ++i)
            {
                // 2^(-i/256) with 30-bit precision
                double val = std::pow(2.0, -static_cast<double>(i) / 256.0);
                data[i] = static_cast<uint32_t>(val * (1ULL << LUT_BITS));
            }
            initialized = true;
        }

        static Exp2LUT &instance()
        {
            static Exp2LUT lut;
            lut.initialize();
            return lut;
        }
    };

    // ============================================================================
    // JIT Code Emitter
    // ============================================================================

    /**
     * @brief JIT code emitter for Exp2 Fixed-Point Softmax microkernel
     *
     * Emits AVX-512 code for converting INT32 scores to INT16 attention weights.
     */
    class JitExp2FixedSoftmaxEmitter
    {
    public:
        /**
         * @brief Configuration for the softmax emitter
         */
        struct Config
        {
            float alpha = 1.0f;          ///< Score scaling factor (typically attention_scale)
            bool online_softmax = false; ///< Use FA2-style online softmax (prefill mode)
            int kv_micro_tile = 4;       ///< KV positions per micro-iteration
        };

        explicit JitExp2FixedSoftmaxEmitter(const Config &config) : config_(config) {}

        // ─────────────────────────────────────────────────────────────────────────
        // JIT Code Emission (TODO: Implement with Xbyak)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Emit LUT base address load
         * @param gen Xbyak code generator
         * @param lut_reg Register to hold LUT base address
         */
        template <typename CodeGen>
        void emit_load_lut_base(CodeGen &gen, const Xbyak::Reg64 &lut_reg)
        {
            (void)gen;
            (void)lut_reg;
            // TODO: Load address of Exp2LUT::instance().data into lut_reg
        }

        /**
         * @brief Emit max-finding for 4 scores
         * @param gen Xbyak code generator
         *
         * Finds max of Scratch0-3 scores, stores in Scratch4.
         * For online softmax, also updates StateMax if new max found.
         */
        template <typename CodeGen>
        void emit_find_max(CodeGen &gen)
        {
            (void)gen;
            // TODO: Implement horizontal max across 4 scores
            // vpmaxsd across Scratch0-3 → Scratch4
        }

        /**
         * @brief Emit exp2 computation for 4 scores
         * @param gen Xbyak code generator
         * @param lut_reg Register holding LUT base address
         *
         * Converts scores in Scratch0-3 to weights using exp2 LUT.
         *
         * INPUT:  Scratch0-3 (INT32 scores), Scratch4 (max_score)
         * OUTPUT: Scratch0-3 (INT16 weights [0, 32767])
         */
        template <typename CodeGen>
        void emit_exp2_weights(CodeGen &gen, const Xbyak::Reg64 &lut_reg)
        {
            (void)gen;
            (void)lut_reg;
            // TODO: Implement exp2 LUT lookup
            //
            // For each score in Scratch0-3:
            //   delta = max_score - score
            //   t = delta × beta (where beta = alpha × log2(e))
            //   ip = floor(t)        // Integer part
            //   frac = t - ip        // Fractional part
            //   idx = frac × 256     // LUT index
            //   raw = LUT[idx]       // 30-bit exp2 value
            //   result = raw >> ip   // Apply integer part
            //   weight = result >> (30 - 15)  // Scale to INT16 range
        }

        /**
         * @brief Emit weight sum accumulation
         * @param gen Xbyak code generator
         *
         * Sums weights in Scratch0-3, adds to StateSum (for online softmax)
         * or accumulates to a local sum (for decode mode).
         */
        template <typename CodeGen>
        void emit_accumulate_sum(CodeGen &gen)
        {
            (void)gen;
            // TODO: Horizontal sum of weights, add to running total
        }

        /**
         * @brief Emit online softmax rescaling (FA2 prefill only)
         * @param gen Xbyak code generator
         *
         * When max increases, rescale previous weights:
         *   old_weights *= exp2(old_max - new_max)
         */
        template <typename CodeGen>
        void emit_online_rescale(CodeGen &gen)
        {
            (void)gen;
            // TODO: Implement rescaling for online softmax
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Reference Implementation (for testing)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Compute softmax weights using exp2 LUT (reference)
         *
         * @param scores Input INT32 scores [n] (INT32_MIN = masked)
         * @param weights Output INT16 weights [n] (range [0, 32767])
         * @param n Number of positions
         * @param alpha Score scale factor
         * @param sum_out Output sum of weights (optional)
         */
        static void compute_reference(
            const int32_t *scores,
            int16_t *weights,
            int n,
            float alpha,
            int64_t *sum_out = nullptr)
        {
            auto &lut = Exp2LUT::instance();

            // Find max (excluding masked positions)
            int32_t max_score = std::numeric_limits<int32_t>::min();
            for (int i = 0; i < n; ++i)
            {
                if (scores[i] != std::numeric_limits<int32_t>::min())
                {
                    max_score = std::max(max_score, scores[i]);
                }
            }

            // If all masked, output zeros
            if (max_score == std::numeric_limits<int32_t>::min())
            {
                std::fill(weights, weights + n, static_cast<int16_t>(0));
                if (sum_out)
                    *sum_out = 0;
                return;
            }

            // Compute exp2 values
            constexpr float LOG2_E = 1.4426950408889634f;
            float beta = alpha * LOG2_E;

            int64_t sum = 0;
            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == std::numeric_limits<int32_t>::min())
                {
                    weights[i] = 0; // Masked position
                    continue;
                }

                // delta = max - score (positive)
                int32_t delta = max_score - scores[i];

                // t = delta × beta
                float t = static_cast<float>(delta) * beta;

                // Decompose into integer and fractional parts
                // Note: t is positive, so floor works as expected
                int ip = static_cast<int>(t);            // Integer part
                float frac = t - static_cast<float>(ip); // Fractional part [0, 1)

                // LUT index from fractional part
                int idx = static_cast<int>(frac * 256.0f);
                idx = std::clamp(idx, 0, 255);

                // Get exp2 value from LUT and apply integer part shift
                uint64_t raw = lut.data[idx];
                if (ip < 64)
                {
                    raw >>= ip; // Apply integer part as right shift
                }
                else
                {
                    raw = 0; // Underflow to zero
                }

                // Scale to INT16 range [0, 32767]
                // LUT has 30 bits, we want 15 bits
                int32_t weight = static_cast<int32_t>(raw >> (Exp2LUT::LUT_BITS - Exp2LUT::WEIGHT_BITS));
                weights[i] = static_cast<int16_t>(std::clamp(weight, 0, 32767));
                sum += weights[i];
            }

            if (sum_out)
                *sum_out = sum;
        }

        /**
         * @brief Compute softmax for 4 KV positions (reference)
         *
         * @param scores Input INT32 scores [4]
         * @param weights Output INT16 weights [4]
         * @param alpha Score scale factor
         * @param max_out Output max score (optional)
         * @param sum_out Output sum of weights (optional)
         */
        static void compute_4kv_reference(
            const int32_t *scores,
            int16_t *weights,
            float alpha,
            int32_t *max_out = nullptr,
            int64_t *sum_out = nullptr)
        {
            int64_t sum;
            compute_reference(scores, weights, 4, alpha, &sum);

            if (max_out)
            {
                *max_out = std::numeric_limits<int32_t>::min();
                for (int i = 0; i < 4; ++i)
                {
                    if (scores[i] != std::numeric_limits<int32_t>::min())
                    {
                        *max_out = std::max(*max_out, scores[i]);
                    }
                }
            }
            if (sum_out)
                *sum_out = sum;
        }

        /**
         * @brief Apply causal mask to scores
         *
         * @param scores INT32 scores [kv_len]
         * @param query_pos Query position (0-indexed)
         * @param kv_start Starting KV position in this batch
         * @param kv_count Number of KV positions
         */
        static void apply_causal_mask(
            int32_t *scores,
            int query_pos,
            int kv_start,
            int kv_count)
        {
            for (int i = 0; i < kv_count; ++i)
            {
                int kv_pos = kv_start + i;
                if (kv_pos > query_pos)
                {
                    scores[i] = std::numeric_limits<int32_t>::min();
                }
            }
        }

    private:
        Config config_;
    };

    // ============================================================================
    // Online Softmax State (for FA2 prefill)
    // ============================================================================

    /**
     * @brief State for FA2-style online softmax
     *
     * Tracks running max (m) and sum (l) across KV tiles.
     * When max increases, previous weights must be rescaled.
     */
    struct OnlineSoftmaxState
    {
        int32_t max_score = std::numeric_limits<int32_t>::min();
        int64_t weight_sum = 0;

        /**
         * @brief Update state with new tile
         *
         * @param tile_max Max score in new tile
         * @param tile_sum Sum of weights in new tile
         * @param rescale_factor_out Output: factor to rescale previous O accumulator
         * @return true if max increased (rescaling needed)
         */
        bool update(int32_t tile_max, int64_t tile_sum, float *rescale_factor_out)
        {
            if (tile_max > max_score)
            {
                // Max increased - need to rescale previous contributions
                if (max_score != std::numeric_limits<int32_t>::min())
                {
                    // Compute rescale factor: exp2(old_max - new_max)
                    int32_t delta = max_score - tile_max; // Negative
                    constexpr float LOG2_E = 1.4426950408889634f;
                    *rescale_factor_out = std::pow(2.0f, static_cast<float>(delta) * LOG2_E);
                }
                else
                {
                    *rescale_factor_out = 0.0f; // No previous contributions
                }

                // Update state
                weight_sum = static_cast<int64_t>(static_cast<float>(weight_sum) * (*rescale_factor_out));
                weight_sum += tile_sum;
                max_score = tile_max;
                return true;
            }
            else
            {
                // Max unchanged - new tile's weights are already correct
                weight_sum += tile_sum;
                *rescale_factor_out = 1.0f;
                return false;
            }
        }

        void reset()
        {
            max_score = std::numeric_limits<int32_t>::min();
            weight_sum = 0;
        }
    };

} // namespace llaminar2::kernels::q16_1::jit
