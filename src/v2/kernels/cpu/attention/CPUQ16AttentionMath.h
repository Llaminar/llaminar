/**
 * @file CPUQ16AttentionMath.h
 * @brief Range proof shared by every CPU Q16 attention reduction.
 *
 * Native keys own their scale and may use the entire signed int16 range.
 * Bounding the ephemeral query instead of clipping stored keys preserves KV
 * precision without changing cache bytes, SIMD instructions or reduction order.
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
} // namespace llaminar2::cpu
