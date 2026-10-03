/**
 * @file Test__SamplingWideTopK.cpp
 * @brief Device-free admission/extent properties of the wide Top-K launch plan.
 *
 * The existing stochastic arena admits 8192 partial entries per row. Sweep all
 * supported wide widths and MTP row counts, including unseen vocabulary sizes,
 * to prove the selector remains within that budget and the 32-KiB shared limit.
 */
#include "kernels/common/SamplingWideTopKDevice.inl"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

/** @brief Every supported width and row count fits the existing admitted arena. */
TEST(SamplingWideTopK, AllWidthsAndRowsRespectAdmittedCapacity)
{
    for (int rows = 1; rows <= 16; ++rows)
    for (int k = 65; k <= sampling_math::kMaxTopK; ++k)
    for (int vocabulary : {65, 129, 248320, 300001, std::numeric_limits<int>::max()})
    {
        const auto plan = sampling_wide::geometry(vocabulary, rows, k, rows * 8192);
        if (vocabulary < k)
        {
            EXPECT_FALSE(plan);
            continue;
        }
        ASSERT_TRUE(plan);
        EXPECT_GT(plan->partial_blocks, 0);
        EXPECT_LE(plan->partial_blocks, sampling_wide::kPartialLists);
        EXPECT_LE(static_cast<size_t>(rows) * plan->partial_blocks * k,
                  static_cast<size_t>(rows) * 8192);
        EXPECT_LE(plan->shared_bytes, 32u * 1024u);
    }
}

/** @brief Missing/minimal storage and invalid public geometry fail before launch. */
TEST(SamplingWideTopK, RejectsInvalidOrInsufficientGeometry)
{
    EXPECT_FALSE(sampling_wide::geometry(248320, 0, 256, 8192));
    EXPECT_FALSE(sampling_wide::geometry(0, 16, 256, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 64, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 257, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 256, 4095));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 256, 0));
    const auto minimum = sampling_wide::geometry(248320, 16, 256, 4096);
    ASSERT_TRUE(minimum);
    EXPECT_EQ(minimum->partial_blocks, 1);
}
