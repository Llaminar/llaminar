/**
 * @file Test__CPUExplicitRoundingContract.cpp
 * @brief Lock register-rounded CPU arithmetic to the existing device contract.
 *
 * Exhaust every positive finite binary16 scale used by every quantized expert
 * codebook, then cover binary32 exponent boundaries and adversarial mantissas.
 * The oracle retains the old memory-rounding semantics independently of the
 * production helper. Q8 publication additionally checks every partial-block
 * length and complete SIMD blocks against scalar oracle bytes.
 */
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h"
#include "utils/CPUExplicitRoundingOracle.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace llaminar2::test
{
/** @test Every persisted positive binary16 scale keeps the identical reciprocal. */
TEST(CPUExplicitRoundingContract, EveryHalfScalePreservesReciprocalBits)
{
    for (unsigned word = 0; word <= 0x7c00u; ++word)
    {
        const unsigned exponent = word >> 10u;
        const unsigned fraction = word & 1023u;
        const float scale = exponent == 31u
            ? std::numeric_limits<float>::infinity()
            : std::ldexp(float(exponent == 0u ? fraction : 1024u + fraction),
                         exponent == 0u ? -24 : int(exponent) - 25);
        ASSERT_EQ(std::bit_cast<std::uint32_t>(
                      device_fp32_contract::reciprocalPositive(scale)),
                  std::bit_cast<std::uint32_t>(memoryRoundedReciprocal(scale)))
            << "binary16 word=" << word;
        ASSERT_EQ(std::bit_cast<std::uint32_t>(
                      cpu::prepared_half_reciprocal::reciprocal(std::uint16_t(word))),
                  std::bit_cast<std::uint32_t>(memoryRoundedReciprocal(scale)))
            << "binary16 lookup word=" << word;
    }
}

/** @test Invalid scale signs and every NaN payload retain the general contract. */
TEST(CPUExplicitRoundingContract, HalfLookupRejectsSignsAndNaNsCanonically)
{
    for (unsigned word = 0x7c01u; word <= 0xffffu; ++word)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(
                      cpu::prepared_half_reciprocal::reciprocal(std::uint16_t(word))),
                  0x7fc00000u) << "binary16 word=" << word;
}

/** @test General positive FP32 values retain exponent, subnormal and tie handling. */
TEST(CPUExplicitRoundingContract, AllExponentBoundariesAndMantissaSamplesAgree)
{
    for (std::uint32_t exponent = 0; exponent <= 255u; ++exponent)
    {
        for (std::uint32_t index = 0; index < 1024u; ++index)
        {
            // Include the first and last mantissas and a distributed sample.
            const std::uint32_t mantissa = index < 4u ? index
                : index < 8u ? 0x007fffffu - (index - 4u)
                : (index * 2654435761u) & 0x007fffffu;
            const auto word = (exponent << 23u) | mantissa;
            const float denominator = std::bit_cast<float>(word);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(
                          device_fp32_contract::reciprocalPositive(denominator)),
                      std::bit_cast<std::uint32_t>(
                          memoryRoundedReciprocal(denominator))) << "word=" << word;
        }
    }
}

/** @test Multiplication cannot contract with its consumer across a rounding edge. */
TEST(CPUExplicitRoundingContract, AdversarialMultiplyAddKeepsTwoRoundingEdges)
{
    const float lhs = std::bit_cast<float>(0x3f800001u);
    const float rhs = std::bit_cast<float>(0x3f7ffffeu);
    const float expected = memoryRoundedAdd(memoryRoundedMultiply(lhs, rhs), -1.0f);
    ASSERT_NE(expected, std::fma(lhs, rhs, -1.0f));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(device_fp32_contract::add(
                  device_fp32_contract::multiply(lhs, rhs), -1.0f)),
              std::bit_cast<std::uint32_t>(expected));
}

/** @test Complete SIMD blocks and every tail match independently rounded Q8 bytes. */
TEST(CPUExplicitRoundingContract, ExpertQ8EveryTailAndMagnitudeMatchesMemoryOracle)
{
    for (int exponent : {-14, -7, 0, 7, 14})
    {
        for (int valid = 1; valid <= 32; ++valid)
        {
            std::array<float, 32> source{};
            for (int element = 0; element < valid; ++element)
                source[element] = std::ldexp(float((element * 71 + valid) % 255 - 127)
                                                * 0.03125f, exponent);
            float maximum = 0.0f;
            for (int element = 0; element < valid; ++element)
                maximum = std::max(maximum, std::fabs(source[element]));
            const float scale = canonicalPreparedHalfValue(maximum > 0.0f
                ? memoryRoundedMultiply(maximum, 1.0f / 127.0f) : 1.0f);
            const float inverse = memoryRoundedReciprocal(scale);
            Q8_1Block expected{}, actual{};
            expected.d = canonicalPreparedHalfBits(scale);
            for (int element = 0; element < valid; ++element)
            {
                const int value = std::clamp(int(std::rint(
                    memoryRoundedMultiply(source[element], inverse))), -127, 127);
                expected.qs[element] = std::int8_t(value);
                expected.sum_qs += value;
            }
            cpu::gpu_aligned_expert_q8::quantizeBlock(source.data(), actual, valid);
            ASSERT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0)
                << "valid=" << valid << " exponent=" << exponent;
        }
    }
}

/** @test Direct SIMD packing preserves every byte lane and extremal integer sums. */
TEST(CPUExplicitRoundingContract, ExpertQ8PackedLaneOrderAndExtremalSums)
{
    for (int pattern = 0; pattern < 68; ++pattern)
    {
        std::array<float, 32> source{};
        if (pattern < 2)
            source.fill(pattern == 0 ? 127.0f : -127.0f);
        else if (pattern == 2)
        {
            for (int lane = 0; lane < 32; ++lane) source[lane] = float(lane - 16);
            source[0] = -127.0f;
            source[31] = 127.0f;
        }
        else if (pattern >= 4)
            source[(pattern - 4) / 2] = pattern % 2 == 0 ? 127.0f : -127.0f;
        // Every nonzero pattern has max magnitude 127. Its canonical scale is
        // exactly one, as is the defined scale for the all-zero pattern.
        Q8_1Block actual{};
        cpu::gpu_aligned_expert_q8::quantizeBlock(source.data(), actual);
        ASSERT_EQ(actual.d, 0x3c00u) << "pattern=" << pattern;
        int expected_sum = 0;
        for (int lane = 0; lane < 32; ++lane)
        {
            const int expected = int(source[lane]);
            ASSERT_EQ(int(actual.qs[lane]), expected)
                << "pattern=" << pattern << " lane=" << lane;
            expected_sum += expected;
        }
        ASSERT_EQ(actual.sum_qs, expected_sum) << "pattern=" << pattern;
    }
}
} // namespace llaminar2::test
