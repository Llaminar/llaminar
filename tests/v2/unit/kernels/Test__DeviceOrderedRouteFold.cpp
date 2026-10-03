/**
 * @file Test__DeviceOrderedRouteFold.cpp
 * @brief Device-free proof that load overlap preserves serial FP32 route sums.
 *
 * The oracle deliberately rounds after every addition without calling the
 * implementation's arithmetic helper. Cancellation, signed zero, odd strides,
 * and partial register windows detect reassociation or padded-route reads.
 */
#include "kernels/common/DeviceOrderedRouteFold.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

/** @test Every full/partial window matches an independent serial binary32 sum. */
TEST(DeviceOrderedRouteFold, ExactSerialOrderAcrossWindowsAndStrides)
{
    constexpr std::array<float, 10> pattern{
        16777216.0f, 1.0f, -16777216.0f, 0.25f, -0.5f,
        3.0f, -0.125f, 0.0625f, 0.0f, -0.0f};
    for (int routes = 1; routes <= 65; ++routes)
        for (std::size_t stride : {1u, 3u, 16u, 63u})
            for (std::size_t shift = 0; shift < pattern.size(); ++shift)
            {
                SCOPED_TRACE(::testing::Message() << "routes=" << routes
                    << " stride=" << stride << " shift=" << shift);
                std::vector<float> input((routes + 8) * stride,
                    std::numeric_limits<float>::quiet_NaN());
                volatile float expected = 0.0f;
                for (int route = 0; route < routes; ++route)
                {
                    const float value = pattern[(route + shift) % pattern.size()];
                    input[route * stride] = value;
                    expected = expected + value;
                }
                const float actual = llaminar2::device_ordered_route_fold::sum(
                    input.data(), stride, routes);
                const float serial = expected;
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(serial));
            }
}

/** @test An inactive route tensor retains the serial initial positive zero. */
TEST(DeviceOrderedRouteFold, NegativeZeroInputsStillStartFromPositiveZero)
{
    const std::array<float, 33> input = [] {
        std::array<float, 33> result{};
        result.fill(-0.0f);
        return result;
    }();
    for (int routes = 1; routes <= 33; ++routes)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(llaminar2::device_ordered_route_fold::sum(
            input.data(), 1, routes)), 0u);
}
