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
#include <limits>
#include "utils/Q8ScaleUnderflowFixture.h"
#include "utils/FP32SubnormalContractCases.h"

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

#if defined(LLAMINAR_Q8_SCALE_TEST_SCALAR) || defined(LLAMINAR_Q8_SCALE_TEST_AVX2) || defined(LLAMINAR_Q8_SCALE_TEST_AVX512)
/** @test Dedicated targets compile the actual production ISA path and use real CPUID. */
TEST(CPUExplicitRoundingContract, Q8FiniteScaleCompiledISAPath)
{
#if defined(LLAMINAR_Q8_SCALE_TEST_SCALAR)
#if defined(__AVX2__) || defined(__AVX512F__)
    FAIL() << "Scalar target compiled vector publication code";
#endif
#elif defined(LLAMINAR_Q8_SCALE_TEST_AVX2)
#if !defined(__AVX2__) || defined(__AVX512F__)
    FAIL() << "AVX2 target has the wrong compiled ceiling";
#endif
    ASSERT_TRUE(cpu_supports_avx2()) << "Actual AVX2 hardware required";
#elif defined(LLAMINAR_Q8_SCALE_TEST_AVX512)
#if !defined(__AVX512F__)
    FAIL() << "AVX512 target omitted its production publication code";
#endif
    ASSERT_TRUE(cpu_supports_avx512()) << "Actual AVX512 hardware required";
#endif
}
#endif

/** @test CPU canonical operations match independent subnormal/signed-zero words. */
TEST(CPUExplicitRoundingContract, DeviceFP32SubnormalFixedWords)
{
    for (size_t row = 0; row < fp32SubnormalCases.size(); ++row)
    {
        const auto &test = fp32SubnormalCases[row];
        const float x = std::bit_cast<float>(test.lhs);
        const float y = std::bit_cast<float>(test.rhs);
        const float z = std::bit_cast<float>(test.accumulator);
        const std::array<float, 3> actual{
            device_fp32_contract::multiply(x, y), device_fp32_contract::add(x, y),
            device_fp32_contract::fusedMultiplyAdd(x, y, z)};
        for (size_t op = 0; op < actual.size(); ++op)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[op]), test.expected[op])
                << "case=" << row << " operation=" << op;
    }
}

/** @test Publishing finite inputs never creates an invalid half scale. */
TEST(CPUExplicitRoundingContract, Q8FiniteScaleDomainAndZeroBlocks)
{
    for (float maximum : {0.f, -0.f})
        EXPECT_EQ(std::bit_cast<std::uint32_t>(device_q8_activation_contract::scale(maximum)), 0x3f800000u);
    for (float maximum : {std::numeric_limits<float>::denorm_min(), 0x1p-24f,
                          0x1p-20f, 127.f * 0x1p-25f, 127.f * 0x1p-24f,
                          1.f, 127.f * 65504.f, 127.f * 65520.f,
                          std::numeric_limits<float>::max()})
    {
        const float scale = device_q8_activation_contract::scale(maximum);
        EXPECT_GE(scale, 0x1p-24f);
        EXPECT_LE(scale, 65504.f);
        EXPECT_TRUE(std::isfinite(scale));
    }
}

/** @test Every old positive finite publication, including RNE ties, stays unchanged. */
TEST(CPUExplicitRoundingContract, Q8FiniteScalePreservesEveryHalfBoundary)
{
    for (std::uint32_t word = 1; word < 0x7bffu; ++word)
    {
        const float low = canonicalPreparedHalfValue(
            std::ldexp(float(word < 1024u ? word : 1024u + (word & 1023u)), word < 1024u ? -24 : int(word >> 10u) - 25));
        const unsigned next = word + 1;
        const float high = std::ldexp(float(next < 1024u ? next : 1024u + (next & 1023u)), next < 1024u ? -24 : int(next >> 10u) - 25);
        const float midpoint = float((double(low) + double(high)) * .5);
        const float maximum_at_midpoint = midpoint * 127.f;
        for (float maximum : {low * 127.f,
                              std::nextafter(maximum_at_midpoint, 0.f),
                              maximum_at_midpoint,
                              std::nextafter(maximum_at_midpoint, INFINITY)})
        {
            const float old_scale = canonicalPreparedHalfValue(
                memoryRoundedMultiply(maximum, MoEProjectionNumericalContract::q8_scale_multiplier));
            const float new_scale = device_q8_activation_contract::scale(maximum);
            if (old_scale > 0.f && std::isfinite(old_scale))
                ASSERT_EQ(std::bit_cast<std::uint32_t>(new_scale), std::bit_cast<std::uint32_t>(old_scale))
                    << "word=" << word << " maximum=" << maximum;
            else
                ASSERT_TRUE(std::isfinite(new_scale) && new_scale > 0.f);
        }
    }
    EXPECT_EQ(canonicalPreparedHalfBits(0x1p-25f), 0u);
    EXPECT_EQ(canonicalPreparedHalfBits(std::nextafter(0x1p-25f, INFINITY)), 1u);
    EXPECT_EQ(canonicalPreparedHalfBits(3.f * 0x1p-25f), 2u);
    EXPECT_EQ(canonicalPreparedHalfBits(-0.f), 0u);
    EXPECT_EQ(canonicalPreparedHalfBits(65520.f), 0x7c00u);
}

/** @test Previously valid half-integer bytes keep RNE-before-saturation semantics. */
TEST(CPUExplicitRoundingContract, Q8FiniteScalePreservesRNETies)
{
    for (float scale : {0x1p-24f, 1.f, 65504.f})
        for (float quotient : {-127.5f, -127.f, -126.5f, -.5f, -0.f,
                               0.f, .5f, 126.5f, 127.f, 127.5f})
        {
            const float value = memoryRoundedMultiply(scale, quotient);
            const float rounded = std::rint(memoryRoundedMultiply(
                value, memoryRoundedReciprocal(scale)));
            const int old_byte = std::clamp(int(rounded), -127, 127);
            EXPECT_EQ(device_q8_activation_contract::quantize(value, scale), old_byte);
        }
}

/** @test The exact diagnostic GDN operand has a defined min-half publication. */
TEST(CPUExplicitRoundingContract, Q8FiniteScaleCapturedTinyBlock)
{
    std::array<float,32> source{};
    float maximum=0.f;
    for(std::size_t lane=0;lane<source.size();++lane) {
        source[lane]=q8ScaleUnderflowSourceValue(lane);
        maximum=std::max(maximum,std::fabs(source[lane]));
    }
    const float scale=device_q8_activation_contract::scale(maximum);
    ASSERT_TRUE(std::isfinite(scale)&&scale>0.f);
    ASSERT_EQ(canonicalPreparedHalfBits(scale),1u);
    Q8_1Block actual{};
    cpu::gpu_aligned_expert_q8::quantizeBlock(source.data(),actual);
    EXPECT_EQ(actual.d,1u);
    int sum=0;
    for(std::size_t lane=0;lane<source.size();++lane) {
        EXPECT_EQ(actual.qs[lane],q8ScaleUnderflowExpectedBytes[lane]);
        sum+=q8ScaleUnderflowExpectedBytes[lane];
    }
    EXPECT_EQ(actual.sum_qs,sum);
}

/** @test Scalar tails and complete SIMD blocks saturate only after explicit RNE. */
TEST(CPUExplicitRoundingContract, Q8FiniteScaleEveryTailHasDefinedBytesAndSums)
{
    for (int valid = 1; valid <= 32; ++valid)
        for (float magnitude : {0.f, 0x1p-24f, 0x1p-20f, 127.f * 0x1p-24f,
                                1.f, 127.f * 65520.f, std::numeric_limits<float>::max()})
        {
            std::array<float, 32> source{};
            for (int lane = 0; lane < valid; ++lane)
                source[lane] = lane % 4 == 0 ? magnitude : lane % 4 == 1 ? -magnitude : lane % 4 == 2 ? 0.f : -0.f;
            const float scale = device_q8_activation_contract::scale(magnitude);
            ASSERT_TRUE(std::isfinite(scale) && scale > 0.f);
            const float inverse = memoryRoundedReciprocal(scale);
            Q8_1Block expected{}, actual{};
            expected.d = canonicalPreparedHalfBits(scale);
            for (int lane = 0; lane < valid; ++lane)
            {
                const float rounded = std::rint(memoryRoundedMultiply(source[lane], inverse));
                const int value = int(std::clamp(rounded, -127.f, 127.f));
                expected.qs[lane] = std::int8_t(value);
                expected.sum_qs += value;
            }
            cpu::gpu_aligned_expert_q8::quantizeBlock(source.data(), actual, valid);
            ASSERT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0)
                << "valid=" << valid << " magnitude=" << magnitude;
        }
}

} // namespace llaminar2::test
