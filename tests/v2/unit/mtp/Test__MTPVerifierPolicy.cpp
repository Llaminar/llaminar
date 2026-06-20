#include <gtest/gtest.h>

#include "execution/mtp/MTPVerifierPolicy.h"

namespace llaminar2
{

TEST(Test__MTPVerifierPolicy, StatefulGreedyRunnerUsesDecodeEquivalentSequentialPath)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_shared_decode_equivalent_verifier");
}

TEST(Test__MTPVerifierPolicy, StatefulSamplingRunnerUsesDecodeEquivalentSequentialPath)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = false,
                .stochastic_verify = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_shared_decode_equivalent_verifier");
}

TEST(Test__MTPVerifierPolicy, GreedyPolicyIsSharedDecodeEquivalentOnly)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_shared_decode_equivalent_verifier");
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
    EXPECT_FALSE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, GreedyGroupedOutcomeRequiresDeviceResidentPublication)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .supports_grouped_decode_equivalent_outcome = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_grouped_decode_equivalent_outcome_with_device_resident_publication");
}

TEST(Test__MTPVerifierPolicy, StochasticGroupedOutcomeRequiresDeviceResidentPublication)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = false,
                .stochastic_verify = true,
                .supports_grouped_decode_equivalent_outcome = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_grouped_decode_equivalent_outcome_with_device_resident_publication");
}

TEST(Test__MTPVerifierPolicy, DirectPublicationWinsOverGroupedOutcome)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .supports_spec_state_publication = true,
                .supports_grouped_decode_equivalent_outcome = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_FALSE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, StochasticCanUseAllPositionStatePublicationWhenRunnerSupportsIt)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = false,
                .stochastic_verify = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionStatePublication);
    EXPECT_FALSE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "stochastic_uses_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesUseDecodeEquivalentSequentialPath)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_shared_decode_equivalent_verifier");
}

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesDoNotUseAllPositionPublication)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_spec_state_publication = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_shared_decode_equivalent_verifier");
}

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesCanUseAllPositionPublicationWithRowLocalPenaltySupport)
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
    EXPECT_FALSE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_all_position_state_publication");
}

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesDoNotUseGroupedOutcomeWithoutRowLocalPenaltySupport)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_grouped_decode_equivalent_outcome = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_shared_decode_equivalent_verifier");
}

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesCanUseGroupedOutcomeWithRowLocalPenaltySupport)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
                .supports_row_local_penalty_application = true,
                .supports_grouped_decode_equivalent_outcome = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalties_use_grouped_decode_equivalent_outcome_with_device_resident_publication");
}

TEST(Test__MTPVerifierPolicy, NonGreedyWithoutStochasticVerifierIsUnsupported)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = false,
                .stochastic_verify = false,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::Unsupported);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "sampling_mode_not_supported_by_shared_verifier");
}

} // namespace llaminar2
