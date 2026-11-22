/**
 * @file Test__ServingHttpPoolContract.cpp
 * @brief Unit tests for the typed HTTP pool width derived from serving capacity.
 *
 * A-2c: the HTTP worker pool width must come from the typed serving capacity
 * (R + bounded control headroom), not from a second ad hoc arithmetic rule
 * (the former "capacity + 2" expression). Device-free.
 */

#include <gtest/gtest.h>

#include "execution/config/ServingCapacity.h"

namespace llaminar2::test
{
    TEST(ServingHttpPoolContract, DefaultHeadroomDerivesFromResolvedCapacity)
    {
        const auto resolved =
            resolveInstalledServingCapacity(ServingCapacityRequest::automatic());
        EXPECT_EQ(servingHttpWorkerCount(resolved),
                  resolved.responseLeases() + kDefaultServingHttpControlHeadroom);
        // Automatic envelope: R = kDefaultServingResponsePool (32), S = 1, Q = R - S.
        EXPECT_EQ(servingHttpWorkerCount(resolved), 32 + 2);
    }

    TEST(ServingHttpPoolContract, ExplicitHeadroomAndAuthoredCapacity)
    {
        const auto resolved =
            resolveInstalledServingCapacity(ServingCapacityRequest::authored(1, 8, 4));
        EXPECT_EQ(resolved.responseLeases(), 8);
        EXPECT_EQ(servingHttpWorkerCount(resolved, 4), 12);
        EXPECT_EQ(servingHttpWorkerCount(resolved, 1), 9);
    }

    TEST(ServingHttpPoolContract, RejectsNonPositiveHeadroom)
    {
        const auto resolved =
            resolveInstalledServingCapacity(ServingCapacityRequest::automatic());
        EXPECT_THROW(servingHttpWorkerCount(resolved, 0), std::invalid_argument);
        EXPECT_THROW(servingHttpWorkerCount(resolved, -1), std::invalid_argument);
    }
} // namespace llaminar2::test
