#include <gtest/gtest.h>

#include "execution/mtp/MTPVerifierPolicy.h"

namespace llaminar2
{

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsBoundTheCapturedGraphFamily)
{
    constexpr int max_rows = 16;
    constexpr int expected[] = {
        1, 2, 4, 4, 8, 8, 8, 8,
        16, 16, 16, 16, 16, 16, 16, 16,
    };

    for (int logical_rows = 1; logical_rows <= max_rows; ++logical_rows)
    {
        EXPECT_EQ(
            mtpVerifierPhysicalRowBucket(logical_rows, max_rows),
            expected[logical_rows - 1])
            << "logical_rows=" << logical_rows;
    }
}

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsRemainTotalForNonPowerOfTwoCaps)
{
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(1, 15), 1);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(3, 15), 4);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(8, 15), 8);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(9, 15), 15);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(15, 15), 15);
}

TEST(Test__MTPVerifierPolicy, PhysicalRowBucketsRejectInvalidGeometry)
{
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(0, 16), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(-1, 16), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(1, 0), 0);
    EXPECT_EQ(mtpVerifierPhysicalRowBucket(17, 16), 0);
}

TEST(Test__MTPVerifierPolicy,
     PhysicalPaddedWidthBucketsOnlyTheScalarGpuLane)
{
    constexpr auto bucket =
        MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 5, 15, bucket), 8);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 9, 15, bucket), 15);

    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(2, 5, 15, bucket), 5);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(8, 9, 15, bucket), 9);

    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(0, 5, 15, bucket), 0);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 0, 15, bucket), 0);
    EXPECT_EQ(mtpVerifierPhysicalPaddedSeqLen(1, 16, 15, bucket), 0);
}

TEST(Test__MTPVerifierPolicy,
     DynamicDeviceEnvelopeIsStableAcrossEveryScalarLogicalWidth)
{
    constexpr int max_rows = 16;
    constexpr auto envelope =
        MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope;
    for (int logical_rows = 1; logical_rows <= max_rows; ++logical_rows)
    {
        EXPECT_EQ(
            mtpVerifierPhysicalPaddedSeqLen(
                /*request_count=*/1,
                logical_rows,
                max_rows,
                envelope),
            max_rows)
            << "logical_rows=" << logical_rows;
    }

    EXPECT_EQ(
        mtpVerifierPhysicalPaddedSeqLen(2, 5, max_rows, envelope),
        5)
        << "Request-batched metadata retains its explicit row stride.";
}

TEST(Test__MTPVerifierPolicy, GreedyUsesGroupedDecodeEquivalentOutcomeByDefault)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, StochasticUsesGroupedDecodeEquivalentOutcomeByDefault)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .stochastic_verify = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, GreedyCanUseAllPositionStatePublicationWhenRunnerSupportsIt)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, DirectPublicationWinsOverGroupedOutcome)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, StochasticCanUseAllPositionStatePublicationWhenRunnerSupportsIt)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .stochastic_verify = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, PenaltiesWithoutRowLocalSupportAreUnsupported)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(decision.path, MTPVerifierExecutionPath::Unsupported);
    EXPECT_STREQ(
        decision.reason,
        "row_local_penalty_application_required_for_grouped_verifier");
}

TEST(Test__MTPVerifierPolicy, PenaltiesCanUseAllPositionPublicationWithRowLocalSupport)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_row_local_penalty_application = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, PenaltiesCanUseGroupedOutcomeWithRowLocalSupport)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_row_local_penalty_application = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_grouped_decode_equivalent_outcome");
}

TEST(Test__MTPVerifierPolicy, NonGreedyWithoutStochasticVerifierIsUnsupported)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{});

    EXPECT_EQ(decision.path, MTPVerifierExecutionPath::Unsupported);
    EXPECT_STREQ(
        decision.reason,
        "sampling_mode_not_supported_by_grouped_verifier");
}

} // namespace llaminar2
