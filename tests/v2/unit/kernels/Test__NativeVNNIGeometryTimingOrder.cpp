/**
 * @file Test__NativeVNNIGeometryTimingOrder.cpp
 * @brief Device-free proof of the actual shared diagnostic timing order.
 *
 * Checks permutations, periodicity, positional exposure and both directions of
 * every candidate pair through the trainer's own helper. No GPU or runtime loads.
 */
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIGeometryTimingOrder.h"
using namespace llaminar2::test::native_vnni_dispatch;

/** @brief Balanced periods preserve candidate/position and pair attribution. */
TEST(NativeVNNIGeometryTimingOrder, CompleteBalancedPhysicalContests)
{
    std::array<std::array<unsigned, 5>, 5> positions{}, before{};
    for (std::size_t round = 0; round < kGeometryTimingOrders.size(); ++round)
    {
        const auto &order = geometryTimingOrder(round);
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t candidate = 0; candidate < sorted.size(); ++candidate)
            EXPECT_EQ(sorted[candidate], candidate);
        EXPECT_EQ(order, geometryTimingOrder(round + kGeometryTimingOrders.size()));
        for (std::size_t p = 0; p < order.size(); ++p)
        {
            ++positions[order[p]][p];
            for (std::size_t q = p + 1; q < order.size(); ++q) ++before[order[p]][order[q]];
        }
    }
    for (std::size_t a = 0; a < 5; ++a)
        for (std::size_t b = 0; b < 5; ++b)
        {
            EXPECT_EQ(positions[a][b], kGeometryTimingOrders.size() / 5);
            if (a != b) EXPECT_EQ(before[a][b], kGeometryTimingOrders.size() / 2);
        }
}

/** @brief Three real MoE contenders have independent balanced chronological exposure. */
TEST(NativeVNNIGeometryTimingOrder, CompleteBalancedMoERouteContests)
{
    std::array<std::array<unsigned, 3>, 3> positions{}, before{};
    for (std::size_t round = 0; round < kMoERouteGeometryTimingOrders.size(); ++round)
    {
        const auto &order = moeRouteGeometryTimingOrder(round);
        auto sorted = order; std::sort(sorted.begin(), sorted.end());
        for (std::size_t candidate = 0; candidate < sorted.size(); ++candidate)
            EXPECT_EQ(sorted[candidate], candidate);
        EXPECT_EQ(order, moeRouteGeometryTimingOrder(round + kMoERouteGeometryTimingOrders.size()));
        for (std::size_t p = 0; p < order.size(); ++p)
        {
            ++positions[order[p]][p];
            for (std::size_t q = p + 1; q < order.size(); ++q) ++before[order[p]][order[q]];
        }
    }
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = 0; b < 3; ++b)
        {
            EXPECT_EQ(positions[a][b], 2u);
            if (a != b) EXPECT_EQ(before[a][b], 3u);
        }
}
