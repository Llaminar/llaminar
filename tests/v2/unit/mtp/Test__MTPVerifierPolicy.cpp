#include <gtest/gtest.h>

#include "execution/mtp/MTPVerifierPolicy.h"

namespace llaminar2
{

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
