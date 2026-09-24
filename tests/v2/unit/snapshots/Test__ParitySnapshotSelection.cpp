/**
 * @file Test__ParitySnapshotSelection.cpp
 * @brief Unit tests for topology-aware tensor-parallel snapshot selection.
 */

#include <gtest/gtest.h>

#include "../../utils/ParitySnapshotSelection.h"

namespace llaminar2::test::parity
{
    TEST(Test__ParitySnapshotSelection, UnevenHeadAlignedTPSlicesCoverReferenceExactly)
    {
        const auto slices = makeParityTPColumnSlices(
            {256, 256, 256, 128}, 896);

        ASSERT_EQ(slices.size(), 4u);
        EXPECT_EQ(slices[0].start_column, 0u);
        EXPECT_EQ(slices[0].column_count, 256u);
        EXPECT_EQ(slices[1].start_column, 256u);
        EXPECT_EQ(slices[2].start_column, 512u);
        EXPECT_EQ(slices[3].start_column, 768u);
        EXPECT_EQ(slices[3].column_count, 128u);
    }

    TEST(Test__ParitySnapshotSelection, TPColumnSlicesRejectIncompleteOrZeroWidthGeometry)
    {
        EXPECT_THROW(
            makeParityTPColumnSlices({224, 224, 224, 224}, 1024),
            std::invalid_argument);
        EXPECT_THROW(
            makeParityTPColumnSlices({448, 0, 448}, 896),
            std::invalid_argument);
    }

    TEST(Test__ParitySnapshotSelection, TPReferenceWidthComesFromAuthenticatedCardinality)
    {
        EXPECT_EQ(parityTPReferenceColumnCount(9u * 2048u, 9u), 2048u);
        EXPECT_THROW(
            parityTPReferenceColumnCount(100u, 9u),
            std::invalid_argument);
    }

    TEST(Test__ParitySnapshotSelection, ReplicatedStageUsesSemanticSnapshotDirectly)
    {
        const auto selection = selectParitySnapshot(
            "layer3_ATTENTION_NORM",
            /*stage_requires_reduction=*/false,
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot);

        EXPECT_EQ(selection.key, "layer3_ATTENTION_NORM");
        EXPECT_EQ(selection.reduction, ParitySnapshotReduction::Direct);
        EXPECT_FALSE(selection.requires_post_collective_key);
    }

    TEST(Test__ParitySnapshotSelection, LocalTPRequiresExplicitPostCollectiveSnapshot)
    {
        const auto selection = selectParitySnapshot(
            "layer0_MOE_COMBINED_OUTPUT",
            /*stage_requires_reduction=*/true,
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot);

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
            ParityCollectiveEvidenceSource::CrossRankPartials);

        EXPECT_EQ(selection.key, "layer7_FFN_DOWN");
        EXPECT_EQ(selection.reduction, ParitySnapshotReduction::CrossRankSum);
        EXPECT_FALSE(selection.requires_post_collective_key);
    }
} // namespace llaminar2::test::parity
