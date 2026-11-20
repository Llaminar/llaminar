/**
 * @file CPUQ16AttentionMath.h
 * @brief Range proof shared by every CPU Q16 attention reduction.
 *
 * Native keys own their scale and may use the entire signed int16 range.
 * Bounding the ephemeral query instead of clipping stored keys preserves KV
 * precision without changing cache bytes, SIMD instructions or reduction order.
 *
 * Closed form: limit(T) = min(2047, floor(INT32_MAX / (32768 * T))) for the
 * T products of one integer reduction. Every whole-head integer reduction
 * runs only when head_dim <= block_elems; multi-block heads reduce at most
 * block_elems products per block and combine the blocks in FP32. Either way
 * one integer reduction has at most T = min(head_dim, block_elems) products. Worst magnitudes:
 * T=256 -> 255 (2,139,095,040), T=128 -> 511, T=64 -> 1023, T=32 -> 2047,
 * all at most INT32_MAX, so every partial sum in any order is exact in int32.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2::cpu
{
/**
 * @brief Largest permitted query code for one complete int32 dot product.
 *
 * The proof covers the horizontal reduction as well as each SIMD lane. For
 * any sign pattern, sum(abs(q*k)) is bounded by INT32_MAX, so every partial
 * sum is also safe. Include -32768 even though our encoder emits +/-32767:
 * native imported keys must not depend on that encoder detail.
 *
 * @param reduction_terms Number of products reduced before conversion to FP32.
 * @return Query magnitude capped at the existing 12-bit signed scratch range.
 * @throws std::invalid_argument If no positive query magnitude is safe.
 */
[[nodiscard]] constexpr int q16AttentionQueryLimit(int reduction_terms)
{
    constexpr std::int64_t key_magnitude = 32768;
    constexpr int query_ceiling = 2047;
    constexpr auto accumulator_limit = std::numeric_limits<std::int32_t>::max();
    if (reduction_terms <= 0 || reduction_terms > accumulator_limit / key_magnitude)
        throw std::invalid_argument("Q16 attention reduction has no safe int32 query range");
    return static_cast<int>(std::min<std::int64_t>(
        query_ceiling, accumulator_limit / (key_magnitude * reduction_terms)));
}

/**
 * @brief Typed proof that one integer attention reduction cannot leave int32.
 *
 * The same value feeds the query quantizer (its clamp limit) and every
 * integer dot (its maximum term count), so the two can never disagree. With
 * |q| <= queryLimit() from the quantizer clamp and |k| <= 32768 by the int16
 * type, sum(|q*k|) over at most terms() products is at most worstMagnitude()
 * <= INT32_MAX: every lane partial and the horizontal fold are exact in int32
 * and equal to the int64 sum.
 */
class Q16ReductionBound final
{
public:
    /**
     * @brief Bound one reduction width.
     * @param terms Products in one integer reduction, at most min(head_dim, block_elems).
     * @return Bound whose query limit is q16AttentionQueryLimit(terms).
     * @throws std::invalid_argument If no positive query magnitude is safe.
     */
    [[nodiscard]] static constexpr Q16ReductionBound forTerms(int terms)
    {
        return Q16ReductionBound(q16AttentionQueryLimit(terms), terms);
    }

    /** @return Largest query code magnitude the quantizer may emit. */
    [[nodiscard]] constexpr int queryLimit() const noexcept { return limit_; }
    /** @return Largest number of products one integer reduction may contain. */
    [[nodiscard]] constexpr int terms() const noexcept { return terms_; }
    /** @return Worst |sum| over terms() products, as a 64-bit witness. */
    [[nodiscard]] constexpr std::int64_t worstMagnitude() const noexcept
    { return std::int64_t(terms_) * limit_ * 32768; }

private:
    constexpr Q16ReductionBound(int limit, int terms) noexcept : limit_(limit), terms_(terms) {}

    int limit_; ///< Query clamp magnitude.
    int terms_; ///< Maximum products per integer reduction.
};

static_assert(Q16ReductionBound::forTerms(256).queryLimit() == 255 &&
              Q16ReductionBound::forTerms(256).worstMagnitude() == 2139095040);
static_assert(Q16ReductionBound::forTerms(1).queryLimit() == 2047);
static_assert(Q16ReductionBound::forTerms(32).worstMagnitude() <= std::numeric_limits<std::int32_t>::max());
static_assert(Q16ReductionBound::forTerms(65535).queryLimit() == 1 &&
              Q16ReductionBound::forTerms(65535).worstMagnitude() <= std::numeric_limits<std::int32_t>::max());
} // namespace llaminar2::cpu
