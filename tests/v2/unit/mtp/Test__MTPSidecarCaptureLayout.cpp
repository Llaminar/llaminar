/**
 * @file Test__MTPSidecarCaptureLayout.cpp
 * @brief Regression tests for total MTP sidecar graph-cache and token-slot ownership.
 */

#include <gtest/gtest.h>

#include "execution/mtp/MTPSidecarCaptureLayout.h"

#include <set>
#include <stdexcept>

using namespace llaminar2;

TEST(Test__MTPSidecarCaptureLayout, EveryConfiguredDepthOwnsOneCacheAndOneUniqueSlot)
{
    /*
     * MTP depth fifteen produces sixteen target rows. This was the first shape
     * class that exposed the historical fixed-array indexing bug at smaller M:
     * the code selected a cache by raw row count, then independently searched
     * for its token slot by address.
     */
    constexpr int kMaximumRows = 16;
    const MTPSidecarCaptureLayout layout(kMaximumRows);

    EXPECT_EQ(layout.maximumRows(), kMaximumRows);
    EXPECT_EQ(layout.kvOnlyBatchCacheCount(), 15u);
    EXPECT_EQ(layout.conditionTokenSlotCount(), 18);
    EXPECT_EQ(
        layout.conditionTokenSlot(MTPSidecarCaptureRole::Full, 1),
        0);
    EXPECT_EQ(
        layout.conditionTokenSlot(MTPSidecarCaptureRole::Chained, 1),
        1);
    EXPECT_EQ(
        layout.conditionTokenSlot(MTPSidecarCaptureRole::KVOnly, 1),
        2);

    std::set<int> observed_slots = {0, 1, 2};
    for (int rows = 2; rows <= kMaximumRows; ++rows)
    {
        const size_t expected_cache_index =
            static_cast<size_t>(rows - 2);
        const int expected_slot = rows + 1;

        EXPECT_EQ(
            layout.kvOnlyBatchCacheIndex(rows),
            expected_cache_index);
        EXPECT_EQ(
            layout.conditionTokenSlot(
                MTPSidecarCaptureRole::KVOnly,
                rows),
            expected_slot);
        EXPECT_TRUE(observed_slots.insert(expected_slot).second)
            << "Each captured graph shape must own a distinct input slot";
    }

    EXPECT_EQ(
        *observed_slots.rbegin(),
        layout.conditionTokenSlotCount() - 1);
    EXPECT_EQ(
        observed_slots.size(),
        static_cast<size_t>(layout.conditionTokenSlotCount()));
}

TEST(Test__MTPSidecarCaptureLayout, OutOfCapacityRowsFailBeforeCacheAccess)
{
    const MTPSidecarCaptureLayout layout(/*maximum_rows=*/5);

    EXPECT_THROW(
        static_cast<void>(layout.kvOnlyBatchCacheIndex(1)),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(layout.kvOnlyBatchCacheIndex(6)),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(
            layout.conditionTokenSlot(MTPSidecarCaptureRole::KVOnly, 0)),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(
            layout.conditionTokenSlot(MTPSidecarCaptureRole::Full, 6)),
        std::out_of_range);
}

TEST(Test__MTPSidecarCaptureLayout, NonPositiveCapacityIsRejected)
{
    EXPECT_THROW(
        MTPSidecarCaptureLayout(/*maximum_rows=*/0),
        std::invalid_argument);
    EXPECT_THROW(
        MTPSidecarCaptureLayout(/*maximum_rows=*/-1),
        std::invalid_argument);
}
