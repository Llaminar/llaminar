/**
 * @file ResidualAddTailCases.h
 * @brief Independent exact normal-FP32 oracle for residual vector/scalar tails.
 *
 * Dyadic operands and the integer-derived expected sum isolate live-extent
 * ownership from the separate near-zero numerical-contract regressions. Every
 * backend consumes the same changed operands, mod4 widths and grouped rows.
 */
#pragma once

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaminar2::test::residual_tail
{
inline constexpr std::array<size_t, 23> widths{
    6, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15,
    255, 256, 257, 258, 259, 1023, 1024, 1025, 1026, 1027};
inline constexpr std::array<size_t, 7> rows{1, 2, 3, 4, 5, 8, 17};
// Separate retained graph geometries reuse the same capacity. A reset does not
// recapture; the repeated largest graph must consume its restored live inputs.
// M1/width6 first reproduces the concrete missing index4 before the full sweep.
inline constexpr std::array<size_t, 9> replay_order{1, 17, 1, 8, 2, 5, 3, 4, 17};
inline constexpr size_t guards = 16;

/**
 * @brief Generate immutable finite operands and an exact independent equation.
 * @param count Complete source capacity to initialize, including unused rows.
 * @param replay Changed-input generation for a retained launch.
 * @param lhs Receives count FP32 dyadic source values.
 * @param rhs Receives count distinct FP32 dyadic residual values.
 * @param expected Receives count exact sums computed from INTEGER numerators.
 */
inline void operands(size_t count, size_t replay, std::vector<float> &lhs,
                     std::vector<float> &rhs, std::vector<float> &expected)
{
    lhs.resize(count); rhs.resize(count); expected.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const int a = static_cast<int>((i * 17 + replay * 7) % 257) - 128;
        const int b = static_cast<int>((i * 11 + replay * 13) % 129) - 64;
        lhs[i] = static_cast<float>(a) * 0x1p-4f;
        rhs[i] = static_cast<float>(b) * 0x1p-5f;
        expected[i] = static_cast<float>(a * 2 + b) * 0x1p-5f;
    }
}

/**
 * @brief Require every live sum and every unused/guard word, without tolerance.
 * @param actual Observed complete capacity plus trailing guards.
 * @param expected Independent complete-capacity exact equation.
 * @param live Number of elements the production call was authorized to write.
 * @param poison Expected unchanged word beyond the live extent.
 */
inline void expect(const std::vector<float> &actual,
                   const std::vector<float> &expected, size_t live,
                   std::uint32_t poison)
{
    ASSERT_GE(actual.size(), expected.size());
    ASSERT_LE(live, expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
            i < live ? std::bit_cast<std::uint32_t>(expected[i]) : poison)
            << "element=" << i << " live=" << live;
}
} // namespace llaminar2::test::residual_tail
