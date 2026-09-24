/**
 * @file Test__CUDACanonicalKpartFold.cpp
 * @brief Device-free admission and capture-identity tests for CTA-local KPAR.
 *
 * The immutable plan owns all geometry constraints. Test every legal physical
 * partition count plus adjacent invalid values, without loading a GPU runtime.
 */
#include <gtest/gtest.h>
#include "kernels/cuda/gemm/CUDACanonicalKpartFold.h"

#include <array>
#include <limits>
#include <type_traits>

using llaminar2::CUDACanonicalKpartColumns;
using llaminar2::CUDACanonicalKpartFoldPlan;

/** @brief A raw/default plan cannot bypass geometry validation. */
TEST(CUDACanonicalKpartFold, ConstructionIsValidatedAndIdentityIsComplete)
{
    static_assert(!std::is_default_constructible_v<CUDACanonicalKpartFoldPlan>);
    constexpr auto plan = CUDACanonicalKpartFoldPlan::create(
        CUDACanonicalKpartColumns::Sixteen, 3, 160);
    static_assert(plan && plan->threads() == 48 && plan->sharedBytes() == 192);
    EXPECT_EQ(plan, CUDACanonicalKpartFoldPlan::create(CUDACanonicalKpartColumns::Sixteen, 3, 160));
    EXPECT_NE(plan, CUDACanonicalKpartFoldPlan::create(CUDACanonicalKpartColumns::ThirtyTwo, 3, 160));
    EXPECT_NE(plan, CUDACanonicalKpartFoldPlan::create(CUDACanonicalKpartColumns::Sixteen, 2, 160));
    EXPECT_NE(plan, CUDACanonicalKpartFoldPlan::create(CUDACanonicalKpartColumns::Sixteen, 3, 192));
}

/** @brief Every partition count is admitted exactly through the native CTA limit. */
TEST(CUDACanonicalKpartFold, PartitionCapacityAndReductionBoundsAreTotal)
{
    for (const auto columns : {CUDACanonicalKpartColumns::Sixteen,
                               CUDACanonicalKpartColumns::ThirtyTwo})
        for (int partitions = -1; partitions <= 65; ++partitions)
            for (const int k : {0, 31, 32, 33, 96, 672, 2048, 2080,
                                std::numeric_limits<int>::max() - 31})
            {
                const auto plan = CUDACanonicalKpartFoldPlan::create(columns, partitions, k);
                const bool expected = k > 0 && k % 32 == 0 && partitions > 0 &&
                    partitions <= k / 32 && partitions <= 1024 / static_cast<int>(columns);
                ASSERT_EQ(plan.has_value(), expected) << "K=" << k << " KB=" << partitions;
                if (plan)
                {
                    EXPECT_EQ(plan->partitions(), partitions);
                    EXPECT_EQ(plan->reductionElements(), k);
                    EXPECT_LE(plan->threads(), 1024);
                }
            }
    for (const int columns : {0, 1, 15, 17, 31, 33, 255})
        EXPECT_FALSE(CUDACanonicalKpartFoldPlan::create(
            static_cast<CUDACanonicalKpartColumns>(columns), 1, 32));
}
