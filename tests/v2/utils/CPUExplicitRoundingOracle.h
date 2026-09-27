/**
 * @file CPUExplicitRoundingOracle.h
 * @brief Independent memory-rounded oracle for the cross-device FP32 contract.
 *
 * Production uses register dependencies to retain binary32 rounding without
 * store/load traffic. Tests deliberately retain volatile memory edges and use
 * frexp/scalbn instead of production's integer exponent reconstruction. This
 * prevents a shared optimized helper from blessing its own arithmetic error.
 */
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace llaminar2::test
{
/** @brief Retain a separately rounded product through observable memory. */
inline float memoryRoundedMultiply(float lhs, float rhs)
{
    volatile float result = lhs * rhs;
    return result;
}

/** @brief Retain a separately rounded addition through observable memory. */
inline float memoryRoundedAdd(float lhs, float rhs)
{
    volatile float result = lhs + rhs;
    return result;
}

/**
 * @brief Execute the published five-step Newton tree using memory edges.
 * @param denominator Non-negative scale; invalid signs/NaNs retain the ABI.
 * @return Bit-exact reference reciprocal, not the library division result.
 */
inline float memoryRoundedReciprocal(float denominator)
{
    const auto bits = std::bit_cast<std::uint32_t>(denominator);
    if ((bits & 0x80000000u) != 0u || std::isnan(denominator))
        return std::bit_cast<float>(0x7fc00000u);
    if (denominator == 0.0f)
        return std::numeric_limits<float>::infinity();
    if (std::isinf(denominator))
        return 0.0f;

    int exponent = 0;
    const float normalized = std::frexp(denominator, &exponent) * 2.0f;
    float reciprocal = 0.75f;
    for (int iteration = 0; iteration < 5; ++iteration)
    {
        const float product = memoryRoundedMultiply(normalized, reciprocal);
        const float correction = memoryRoundedAdd(2.0f, -product);
        reciprocal = memoryRoundedMultiply(reciprocal, correction);
    }
    return std::scalbn(reciprocal, 1 - exponent);
}
} // namespace llaminar2::test
