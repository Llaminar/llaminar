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

TEST(Test__MTPVerifierPolicy, GreedyPenaltiesAreUnsupportedUntilTransactionPathExists)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .greedy_sampling = true,
                .uses_sampling_penalties = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::Unsupported);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "greedy_penalty_mtp_requires_new_transaction_path");
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
