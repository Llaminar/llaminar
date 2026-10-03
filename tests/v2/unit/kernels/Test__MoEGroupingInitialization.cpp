/**
 * @file Test__MoEGroupingInitialization.cpp
 * @brief Device-free admission checks for captured MoE grouping initialization.
 *
 * Validation only inspects borrowed pointer presence and independent array
 * extents. It must not dereference device addresses or initialize a backend.
 * Captured sentinel/guard semantics have separate CUDA/ROCm preflight proofs.
 */
#include "kernels/common/MoEGroupingInitialization.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{
    using llaminar2::MoEGroupingInitialization;
    static_assert(std::is_trivially_copyable_v<MoEGroupingInitialization>);

    /** @test Missing required banks and nonpositive extents cannot be launched. */
    TEST(MoEGroupingInitialization, RequiredBanksAndExtents)
    {
        auto *opaque = reinterpret_cast<int *>(std::uintptr_t{4096});
        const MoEGroupingInitialization valid{
            .expert_counts = opaque,
            .grouped_token_indices = reinterpret_cast<int *>(std::uintptr_t{8192}),
            .grouped_weights = reinterpret_cast<float *>(std::uintptr_t{12288}),
            .num_experts = 1,
            .total_slots = 1};
        EXPECT_TRUE(valid.valid());
        EXPECT_FALSE(MoEGroupingInitialization{}.valid());
        for (const int extent : {0, -1, std::numeric_limits<int>::min()})
        {
            auto bad = valid;
            bad.num_experts = extent;
            EXPECT_FALSE(bad.valid());
            bad = valid;
            bad.total_slots = extent;
            EXPECT_FALSE(bad.valid());
        }
        auto missing = valid;
        missing.expert_counts = nullptr;
        EXPECT_FALSE(missing.valid());
        missing = valid;
        missing.grouped_token_indices = nullptr;
        EXPECT_FALSE(missing.valid());
        missing = valid;
        missing.grouped_weights = nullptr;
        EXPECT_FALSE(missing.valid());
    }

    /** @test Independent expert/route capacities retain their exact maximum. */
    TEST(MoEGroupingInitialization, WorkExtentHasNoRoundingOverflow)
    {
        for (const int experts : {1, 31, 257, 1024, std::numeric_limits<int>::max()})
            for (const int slots : {1, 255, 513, 32768, std::numeric_limits<int>::max()})
            {
                const MoEGroupingInitialization binding{.num_experts = experts, .total_slots = slots};
                EXPECT_EQ(binding.workItems(), std::max(experts, slots));
                EXPECT_GT(1 + (binding.workItems() - 1) / 256, 0);
            }
    }
}
