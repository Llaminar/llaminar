/**
 * @file AttentionKeyQ8Reference.h
 * @brief Scalar oracle for the attention-key Q8 physical format.
 *
 * This file defines the authoritative arithmetic used to certify optimized
 * CPU, CUDA, and ROCm attention-key cache kernels. It intentionally contains no
 * backend dispatch or allocation. Production implementations must produce the
 * same block bytes for finite inputs and must decode those bytes with the same
 * arithmetic order.
 */

#pragma once

#include "tensors/BlockStructures.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Encode a non-negative coordinate magnitude without evaluating cbrt.
     *
     * Rounding `sqrt(magnitude / max_abs) * 127` changes code between `q`
     * and `q + 1` at `((q + 0.5) / 127)^2`. Multiplying both sides by
     * `254^2` turns that boundary into the integer square `(2q + 1)^2`.
     * Seven comparisons therefore produce the exact codebook interval while
     * avoiding a transcendental instruction on cache append.
     *
     * @param magnitude Absolute finite input coordinate in [0, max_abs].
     * @param max_abs Positive finite maximum magnitude for the containing head.
     * @return Unsigned magnitude code in [0, 127].
     */
    inline int8_t attentionKeyQ8MagnitudeCodeReference(float magnitude, float max_abs)
    {
        constexpr int kThresholdDenominator = 2 * AttentionKeyQ8Block<8>::MAX_CODE;
        constexpr int kThresholdDenominatorSquared =
            kThresholdDenominator * kThresholdDenominator;

        const float scaled_magnitude =
            magnitude * static_cast<float>(kThresholdDenominatorSquared);
        int lower = 0;
        int upper = AttentionKeyQ8Block<8>::MAX_CODE;
        while (lower < upper)
        {
            const int midpoint = (lower + upper) / 2;
            const int odd_boundary = 2 * midpoint + 1;
            const int odd_boundary_squared = odd_boundary * odd_boundary;
            const float scaled_boundary =
                max_abs * static_cast<float>(odd_boundary_squared);
            if (scaled_magnitude < scaled_boundary)
            {
                upper = midpoint;
            }
            else
            {
                lower = midpoint + 1;
            }
        }
        return static_cast<int8_t>(lower);
    }

    /**
     * @brief Quantize one complete FP32 attention-key head.
     *
     * The encoder first finds the maximum absolute coordinate. Each value is
     * normalized into [-1, 1], transformed by signed square root, and rounded to
     * a signed int8 code. The transform gives small-but-score-relevant values
     * substantially more resolution than a linear max-abs quantizer.
     *
     * @tparam D Attention-head width.
     * @param input Finite FP32 coordinates for exactly one head.
     * @param output Destination physical block.
     * @throws std::domain_error if any input coordinate is non-finite.
     */
    template <int D>
    inline void attentionKeyQ8QuantizeReference(
        std::span<const float, D> input,
        AttentionKeyQ8Block<D> &output)
    {
        float max_abs = 0.0f;
        for (const float value : input)
        {
            if (!std::isfinite(value))
            {
                throw std::domain_error(
                    "attentionKeyQ8QuantizeReference requires finite key coordinates");
            }
            max_abs = std::max(max_abs, std::abs(value));
        }

        if (max_abs == 0.0f)
        {
            output.quadratic_scale = 0.0f;
            std::fill(std::begin(output.codes), std::end(output.codes), int8_t{0});
            return;
        }

        // Use one pre-rounded reciprocal on every backend. CUDA otherwise
        // lowers constant division differently from x86 and ROCm by one ULP.
        output.quadratic_scale =
            max_abs * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;

        // Encoding by rational interval comparisons makes block bytes stable
        // across CPU, CUDA, and ROCm math-library implementations.
        for (int coordinate = 0; coordinate < D; ++coordinate)
        {
            const int8_t magnitude_code = attentionKeyQ8MagnitudeCodeReference(
                std::abs(input[coordinate]), max_abs);
            output.codes[coordinate] = input[coordinate] < 0.0f
                                           ? static_cast<int8_t>(-magnitude_code)
                                           : magnitude_code;
        }
    }

    /**
     * @brief Decode one attention-key Q8 block to FP32.
     *
     * Multiplication order is fixed as `(scale * q) * abs(q)`. Optimized kernels
     * must preserve or prove equivalence to this order when compared against the
     * scalar oracle.
     *
     * @tparam D Attention-head width.
     * @param input Source physical block.
     * @param output Destination span for exactly one FP32 head.
     * @throws std::domain_error if the stored scale is negative/non-finite or a
     *         code contains the reserved int8 value -128.
     */
    template <int D>
    inline void attentionKeyQ8DequantizeReference(
        const AttentionKeyQ8Block<D> &input,
        std::span<float, D> output)
    {
        if (!std::isfinite(input.quadratic_scale) || input.quadratic_scale < 0.0f)
        {
            throw std::domain_error(
                "attentionKeyQ8DequantizeReference requires a finite non-negative scale");
        }

        for (int coordinate = 0; coordinate < D; ++coordinate)
        {
            const int code = static_cast<int>(input.codes[coordinate]);
            if (code == -128)
            {
                throw std::domain_error(
                    "attentionKeyQ8DequantizeReference encountered reserved code -128");
            }

            const float q = static_cast<float>(code);
            // The fixed order is cheap on every backend and avoids pow/cbrt on
            // cache reads, which are much more frequent than cache appends.
            output[coordinate] =
                (input.quadratic_scale * q) * std::abs(q);
        }
    }
} // namespace llaminar2
