/**
 * @file Test__NEONGDNNormalization.cpp
 * @brief Byte-exact NEON GDN Q/K squared-norm and scaling programs.
 *
 * An independent oracle spells out the AVX2 eight-lane program with explicit
 * FP32 roundings: per-lane fused square accumulation, the hsum fold
 * ((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7)), then separately rounded tails. The
 * NEON leaves must equal it bit-for-bit for every width 0..80 and signed,
 * subnormal and large finite values. Device-free and engine-free.
 */
#include "kernels/cpu/gdn/NEONGDNNormalization.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Explicit-rounding oracle of the AVX2 l2norm squared-sum program. */
    float oracleNormSq(const std::vector<float> &x)
    {
        std::array<float, 8> lane{};
        std::size_t i = 0;
        for (; i + 8 <= x.size(); i += 8)
            for (int l = 0; l < 8; ++l)
                lane[l] = std::fma(x[i + l], x[i + l], lane[l]);
        const auto add = [](float a, float b) { return device_fp32_contract::add(a, b); };
        const auto mul = [](float a, float b) { return device_fp32_contract::multiply(a, b); };
        float result = add(add(add(lane[0], lane[4]), add(lane[1], lane[5])),
                           add(add(lane[2], lane[6]), add(lane[3], lane[7])));
        for (; i < x.size(); ++i)
            result = add(result, mul(x[i], x[i]));
        return result;
    }

    /** @brief Deterministic varied inputs including signs, subnormals and large finites. */
    std::vector<float> fixture(std::size_t n, std::uint32_t seed)
    {
        std::vector<float> x(n);
        std::uint32_t state = seed * 2654435761u + 1u;
        for (std::size_t i = 0; i < n; ++i)
        {
            state = state * 1664525u + 1013904223u;
            const float unit = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
            x[i] = unit * ((i % 7 == 3) ? 1.0e18f : (i % 11 == 5) ? 1.0e-40f : 3.0f);
        }
        return x;
    }
}

/** @test The NEON squared norm equals the explicit AVX2-program oracle bit-for-bit. */
TEST(NEONGDNNormalization, SquaredNormMatchesAVX2ProgramOracle)
{
    for (std::size_t n = 0; n <= 80; ++n)
        for (std::uint32_t seed : {1u, 2u, 3u})
        {
            const auto x = fixture(n, seed);
            const float neon = neon_gdn::norm_sq_avx2_program(x.data(), static_cast<int>(n));
            ASSERT_EQ(std::bit_cast<std::uint32_t>(neon), std::bit_cast<std::uint32_t>(oracleNormSq(x)))
                << "n=" << n << " seed=" << seed;
        }
}

/** @test Scaling is one separately rounded multiply per element, in place or not. */
TEST(NEONGDNNormalization, ScaleIsSeparatelyRoundedAndAliasSafe)
{
    for (std::size_t n = 0; n <= 37; ++n)
    {
        const auto x = fixture(n, 9u);
        const float factor = 0.7071067811865476f;
        std::vector<float> out(n + 2, -7.0f), inplace = x;
        neon_gdn::scale_rows(x.data(), out.data(), static_cast<int>(n), factor);
        neon_gdn::scale_rows(inplace.data(), inplace.data(), static_cast<int>(n), factor);
        for (std::size_t i = 0; i < n; ++i)
        {
            const float expected = device_fp32_contract::multiply(x[i], factor);
            ASSERT_EQ(std::bit_cast<std::uint32_t>(out[i]), std::bit_cast<std::uint32_t>(expected)) << n << ':' << i;
            ASSERT_EQ(std::bit_cast<std::uint32_t>(inplace[i]), std::bit_cast<std::uint32_t>(expected)) << n << ':' << i;
        }
        EXPECT_EQ(out[n], -7.0f);
        EXPECT_EQ(out[n + 1], -7.0f);
    }
}
