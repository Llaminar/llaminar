/**
 * @file PreparedHalfReciprocal.h
 * @brief Exact CPU lookup of the device reciprocal contract for FP16 scales.
 *
 * Every quantized expert persists its activation scale as binary16. After
 * normalization this leaves only 1024 possible mantissas, independently of
 * model, codebook, row count or ISA. Evaluate the canonical five-step binary32
 * Newton tree at compile time for those mantissas, then install the exponent
 * using integer arithmetic. This is the same rounded result, not hardware
 * division or a new approximation. The immutable 4 KiB table has no startup
 * work, lazy initialization, allocation, mutable cache or GPU residency.
 * Exhaustive binary16 checks compare it with the independent memory-rounded
 * oracle in CPUExplicitRoundingContract on AVX2 and AVX512.
 */
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace llaminar2::cpu::prepared_half_reciprocal
{
namespace detail
{
/**
 * @brief Materialize the finite normalized binary16 reciprocal domain.
 * @return Exact binary32 words after the canonical five Newton iterations.
 *
 * Constant evaluation rounds each float assignment; none of these expressions
 * runs in the hot path or contracts into a runtime FMA. Keep the explicit
 * product/correction/product edges aligned with DeviceFP32NumericalContract.
 */
consteval std::array<std::uint32_t, 1024> normalizedReciprocalWords()
{
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
    std::array<std::uint32_t, 1024> words{};
    for (std::uint32_t mantissa = 0; mantissa < words.size(); ++mantissa)
    {
        const float normalized = std::bit_cast<float>(
            0x3f800000u | (mantissa << 13u));
        float reciprocal = 0.75f;
        for (int iteration = 0; iteration < 5; ++iteration)
        {
            const float product = normalized * reciprocal;
            const float correction = 2.0f - product;
            reciprocal = reciprocal * correction;
        }
        words[mantissa] = std::bit_cast<std::uint32_t>(reciprocal);
    }
    return words;
}

/// Read-only constant data: exactly one word per normalized half mantissa.
inline constexpr auto kNormalizedReciprocalWords = normalizedReciprocalWords();
} // namespace detail

/**
 * @brief Return the canonical reciprocal of a persisted binary16 scale.
 * @param half_bits Complete binary16 word, not a floating-point approximation.
 * @return The same binary32 word as reciprocalPositive(widen(half_bits)).
 *
 * Every finite nonzero binary16 reciprocal is normal binary32, so exponent
 * installation needs neither floating multiplication nor subnormal rounding.
 * Signed scales and NaNs retain the general contract's canonical invalid word;
 * positive zero and infinity retain its infinity/zero results.
 */
[[nodiscard]] inline float reciprocal(std::uint16_t half_bits) noexcept
{
    if ((half_bits & 0x8000u) != 0u)
        return std::bit_cast<float>(0x7fc00000u);
    const unsigned exponent = (half_bits >> 10u) & 31u;
    unsigned mantissa = half_bits & 1023u;
    if (exponent == 31u)
        return mantissa == 0u ? 0.0f : std::bit_cast<float>(0x7fc00000u);
    int unbiased_exponent = int(exponent) - 15;
    if (exponent == 0u)
    {
        if (mantissa == 0u)
            return std::bit_cast<float>(0x7f800000u);
        // Move the leading subnormal bit to the implicit-one position. Only
        // touched scale words are normalized; there is no runtime table fill.
        const int shift = std::countl_zero(std::uint32_t(mantissa)) - 21;
        mantissa = (mantissa << shift) & 1023u;
        unbiased_exponent = -14 - shift;
    }
    const std::uint32_t normalized = detail::kNormalizedReciprocalWords[mantissa];
    const unsigned reciprocal_exponent = unsigned(
        int(normalized >> 23u) - unbiased_exponent);
    return std::bit_cast<float>(
        (reciprocal_exponent << 23u) | (normalized & 0x007fffffu));
}
} // namespace llaminar2::cpu::prepared_half_reciprocal
