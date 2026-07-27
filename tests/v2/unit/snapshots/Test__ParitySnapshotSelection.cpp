/**
 * @file Test__ParitySnapshotSelection.cpp
 * @brief Unit tests for topology-aware tensor-parallel snapshot selection.
 */

#include <gtest/gtest.h>

#include "../../utils/ParitySnapshotSelection.h"

namespace llaminar2::test::parity
{
    TEST(Test__ParitySnapshotSelection, ReplicatedStageUsesSemanticSnapshotDirectly)
    {
        const auto selection = selectParitySnapshot(
            "layer3_ATTENTION_NORM",
            /*stage_requires_reduction=*/false,
            /*uses_in_process_local_tp=*/true);

        EXPECT_EQ(selection.key, "layer3_ATTENTION_NORM");
        EXPECT_EQ(selection.reduction, ParitySnapshotReduction::Direct);
        EXPECT_FALSE(selection.requires_post_collective_key);
    }

    TEST(Test__ParitySnapshotSelection, LocalTPRequiresExplicitPostCollectiveSnapshot)
    {
        const auto selection = selectParitySnapshot(
            "layer0_MOE_COMBINED_OUTPUT",
            /*stage_requires_reduction=*/true,
            /*uses_in_process_local_tp=*/true);

        EXPECT_EQ(
            selection.key,
            "layer0_MOE_COMBINED_OUTPUT_ALLREDUCED");
        EXPECT_EQ(selection.reduction, ParitySnapshotReduction::Direct);
        EXPECT_TRUE(selection.requires_post_collective_key);
    }

    TEST(Test__ParitySnapshotSelection, CrossRankTPRetainsPartialForMPISum)
    {
        const auto selection = selectParitySnapshot(
            "layer7_FFN_DOWN",
            /*stage_requires_reduction=*/true,
            /*uses_in_process_local_tp=*/false);

        EXPECT_EQ(selection.key, "layer7_FFN_DOWN");
        EXPECT_EQ(selection.reduction, ParitySnapshotReduction::CrossRankSum);
        EXPECT_FALSE(selection.requires_post_collective_key);
    }
} // namespace llaminar2::test::parity
