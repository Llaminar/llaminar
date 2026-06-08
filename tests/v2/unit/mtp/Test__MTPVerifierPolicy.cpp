#include <gtest/gtest.h>

#include "execution/mtp/MTPVerifierPolicy.h"

namespace llaminar2
{

TEST(Test__MTPVerifierPolicy, StatefulGreedyRunnerUsesDecodeEquivalentSequentialPath)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .runner_requires_decode_equivalent_replay = true,
                .runner_supports_verifier_state_row_restore = true,
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_FALSE(decision.allow_verifier_state_row_shortcut);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "stateful_runner_requires_decode_equivalent_replay");
}

TEST(Test__MTPVerifierPolicy, StatefulSamplingRunnerUsesDecodeEquivalentSequentialPath)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .runner_requires_decode_equivalent_replay = true,
                .runner_supports_verifier_state_row_restore = true,
                .greedy_sampling = false,
                .stochastic_verify = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::DecodeEquivalentSequential);
    EXPECT_FALSE(decision.allow_verifier_state_row_shortcut);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "stateful_runner_requires_decode_equivalent_replay");
}

TEST(Test__MTPVerifierPolicy, StatelessRunnerWithRowRestoreMayCommitVerifierRows)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .runner_requires_decode_equivalent_replay = false,
                .runner_supports_verifier_state_row_restore = true,
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionVerifier);
    EXPECT_TRUE(decision.allow_verifier_state_row_shortcut);
    EXPECT_FALSE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "verifier_prefill_rows_declared_decode_equivalent");
}

TEST(Test__MTPVerifierPolicy, MissingRowRestoreForcesReplayEvenForStatelessRunner)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .runner_requires_decode_equivalent_replay = false,
                .runner_supports_verifier_state_row_restore = false,
                .greedy_sampling = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionVerifier);
    EXPECT_FALSE(decision.allow_verifier_state_row_shortcut);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "verifier_state_row_restore_not_supported");
}

TEST(Test__MTPVerifierPolicy, DebugOverrideLeavesStatefulRunnerUnsafeByDesign)
{
    const MTPVerifierPolicyDecision decision =
        chooseMTPVerifierPolicy(
            MTPVerifierPolicyInput{
                .runner_requires_decode_equivalent_replay = true,
                .runner_supports_verifier_state_row_restore = true,
                .greedy_sampling = true,
                .disable_decode_equivalent_sequential = true,
            });

    EXPECT_EQ(
        decision.path,
        MTPVerifierExecutionPath::AllPositionVerifier);
    EXPECT_FALSE(decision.allow_verifier_state_row_shortcut);
    EXPECT_TRUE(decision.accepted_all_position_state_requires_replay);
    EXPECT_STREQ(
        decision.reason,
        "decode_equivalent_replay_debug_override");
}

} // namespace llaminar2
