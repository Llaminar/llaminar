/**
 * @file Test__MoEGroupedSourceRows.cpp
 * @brief Device-free admission and index-encoding proofs for original MoE rows.
 *
 * Factories must never inspect the borrowed device address. Boundary checks
 * reject absent owners and malformed token/route geometry before capture; the
 * same pure decoder rejects invalid device publications instead of clamping.
 */
#include "kernels/common/MoEGroupedSourceRows.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

namespace
{
    using llaminar2::MoEGroupedSourceRows;

    /** @test Neither factory dereferences an opaque device-only map. */
    TEST(MoEGroupedSourceRows, TokenAndRouteEncodingShareOriginalRows)
    {
        const auto *device_only = reinterpret_cast<const int *>(std::uintptr_t{4096});
        for (const int rows : {1, 16, 33, 65, 512, 4096})
            for (const int top_k : {1, 2, 3, 8, 16})
            {
                const auto tokens = MoEGroupedSourceRows::tokenIndices(device_only, rows);
                const auto routes = MoEGroupedSourceRows::routeIndices(device_only, rows, top_k);
                EXPECT_EQ(tokens.capacity(), rows);
                for (int row = 0; row < rows; ++row)
                {
                    EXPECT_EQ(tokens.decode(row), row);
                    for (int route = 0; route < top_k; ++route)
                        EXPECT_EQ(routes.decode(row * top_k + route), row);
                }
                for (const int invalid : {-1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()})
                {
                    EXPECT_EQ(tokens.decode(invalid), -1);
                    EXPECT_EQ(routes.decode(invalid), -1);
                }
                EXPECT_EQ(tokens.decode(rows), -1);
                EXPECT_EQ(routes.decode(rows * top_k), -1);
            }
    }

    /** @test Missing owners and nonpositive dimensions are unrepresentable. */
    TEST(MoEGroupedSourceRows, RejectsInvalidCaptureGeometry)
    {
        const int unused = 0;
        EXPECT_THROW((void)MoEGroupedSourceRows::tokenIndices(nullptr, 1), std::invalid_argument);
        EXPECT_THROW((void)MoEGroupedSourceRows::routeIndices(nullptr, 1, 1), std::invalid_argument);
        EXPECT_THROW((void)MoEGroupedSourceRows::routeIndices(&unused, std::numeric_limits<int>::max(), 2), std::invalid_argument);
        for (const int invalid : {0, -1, std::numeric_limits<int>::min()})
        {
            EXPECT_THROW((void)MoEGroupedSourceRows::tokenIndices(&unused, invalid), std::invalid_argument);
            EXPECT_THROW((void)MoEGroupedSourceRows::routeIndices(&unused, invalid, 1), std::invalid_argument);
            EXPECT_THROW((void)MoEGroupedSourceRows::routeIndices(&unused, 1, invalid), std::invalid_argument);
        }
    }
}
