#include <gtest/gtest.h>

#include "execution/mtp/MTPDepthController.h"

using namespace llaminar2;

namespace
{
    MTPDepthPolicyConfig dynamicConfig(
        int initial_depth,
        int max_depth,
        int window_size = 2,
        int cooldown_steps = 0)
    {
        MTPDepthPolicyConfig config;
        config.mode = MTPDepthPolicyMode::Dynamic;
        config.min_depth = 1;
        config.max_depth = max_depth;
        config.initial_depth = initial_depth;
        config.window_size = window_size;
        config.min_samples = window_size;
        config.cooldown_steps = cooldown_steps;
        config.promote_consecutive_windows = 1;
        config.promote_full_accept_rate = 0.75;
        config.demote_zero_accept_rate = 0.30;
        config.demote_acceptance_rate = 0.65;
        return config;
    }

    MTPDepthObservation observation(int depth, int accepted_prefix)
    {
        return MTPDepthObservation{
            .requested_depth = depth,
            .effective_depth = depth,
            .accepted_speculative_prefix = accepted_prefix,
            .budget_limited = false,
            .rollback = accepted_prefix < depth,
        };
    }
}

TEST(Test__MTPDepthController, FixedModeIgnoresAdaptiveBounds)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Fixed;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 3;
    config.window_size = 1;
    config.min_samples = 1;
    config.cooldown_steps = 0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/1);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 1);

    const auto decision = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_FALSE(decision.evaluated);
    EXPECT_FALSE(decision.changed);
    EXPECT_EQ(decision.reason, MTPDepthDecisionReason::FixedMode);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().windows, 0u);
}

TEST(Test__MTPDepthController, DynamicDefaultInitialDepthStartsAtMinimum)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 0;
    config.window_size = 16;
    config.min_samples = 4;
    config.cooldown_steps = 0;
    config.promote_consecutive_windows = 3;
    config.promote_full_accept_rate = 1.0;
    config.demote_zero_accept_rate = 0.30;
    config.demote_acceptance_rate = 0.55;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    EXPECT_EQ(controller.currentDepth(), 1)
        << "Dynamic mode should warm from the cheapest valid depth unless the "
           "user explicitly sets an initial depth.";
    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 3);
}

TEST(Test__MTPDepthController, DynamicDemotesOnZeroAcceptWindows)
{
    MTPDepthController controller(
        dynamicConfig(/*initial_depth=*/3, /*max_depth=*/3),
        /*configured_draft_tokens=*/3);
    EXPECT_EQ(controller.currentDepth(), 3);

    auto first = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    EXPECT_FALSE(first.evaluated);
    EXPECT_EQ(controller.currentDepth(), 3);

    auto second = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    EXPECT_TRUE(second.evaluated);
    EXPECT_TRUE(second.changed);
    EXPECT_EQ(second.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(second.old_depth, 3);
    EXPECT_EQ(second.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().windows, 1u);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, DynamicEarlyDemotesAfterMinSamplesWithoutFullWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/8,
        /*cooldown_steps=*/0);
    config.min_samples = 4;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
        EXPECT_FALSE(decision.evaluated);
        EXPECT_EQ(controller.currentDepth(), 3);
    }

    auto early = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
    ASSERT_TRUE(early.evaluated);
    ASSERT_TRUE(early.changed);
    EXPECT_EQ(early.reason, MTPDepthDecisionReason::DemoteLowAcceptanceRate);
    EXPECT_EQ(early.window.verifier_runs, 4u);
    EXPECT_EQ(early.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().windows, 1u);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, DynamicHealthyPartialWindowWaitsForFullWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/8,
        /*cooldown_steps=*/0);
    config.min_samples = 4;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 4; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
        EXPECT_FALSE(decision.evaluated);
        EXPECT_EQ(controller.currentDepth(), 3);
    }

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
        EXPECT_FALSE(decision.evaluated);
    }

    auto full = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    ASSERT_TRUE(full.evaluated);
    EXPECT_FALSE(full.changed);
    EXPECT_EQ(full.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(full.window.verifier_runs, 8u);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().windows, 1u);
}

TEST(Test__MTPDepthController, DynamicPromotesOnStableFullAcceptWindows)
{
    auto config = dynamicConfig(/*initial_depth=*/1, /*max_depth=*/3);
    config.promote_consecutive_windows = 1;
    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    auto d1a = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_FALSE(d1a.evaluated);
    auto d1b = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_TRUE(d1b.changed);
    EXPECT_EQ(d1b.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(controller.currentDepth(), 2);

    auto d2a = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    EXPECT_FALSE(d2a.evaluated);
    auto d2b = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    EXPECT_TRUE(d2b.changed);
    EXPECT_EQ(d2b.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 2u);
}

TEST(Test__MTPDepthController, DynamicPromotesPerfectProbeAfterMinSamples)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/16,
        /*cooldown_steps=*/0);
    config.min_samples = 4;
    config.promote_consecutive_windows = 3;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
        EXPECT_FALSE(decision.evaluated);
        EXPECT_EQ(controller.currentDepth(), 1);
    }

    auto promote = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(promote.evaluated);
    ASSERT_TRUE(promote.changed);
    EXPECT_EQ(promote.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(promote.window.verifier_runs, 4u);
    EXPECT_EQ(promote.old_depth, 1);
    EXPECT_EQ(promote.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, DynamicCanDemoteToZeroBypassAndProbeAfterCooldown)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/2);
    config.min_depth = 0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.requestedDepthForStep(), 1);

    auto demote = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(demote.evaluated);
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(demote.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(demote.old_depth, 1);
    EXPECT_EQ(demote.new_depth, 0);
    EXPECT_EQ(controller.currentDepth(), 0);
    EXPECT_EQ(controller.requestedDepthForStep(), 0);

    auto bypass1 = controller.recordBypassStep();
    EXPECT_FALSE(bypass1.evaluated);
    EXPECT_FALSE(bypass1.changed);
    EXPECT_EQ(bypass1.reason, MTPDepthDecisionReason::DepthZeroBypass);
    EXPECT_EQ(controller.requestedDepthForStep(), 0);

    auto bypass2 = controller.recordBypassStep();
    EXPECT_EQ(bypass2.reason, MTPDepthDecisionReason::DepthZeroBypass);
    EXPECT_EQ(controller.requestedDepthForStep(), 1)
        << "after the cooldown, depth zero should probe with one draft token";

    auto failed_probe = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(failed_probe.evaluated);
    EXPECT_FALSE(failed_probe.changed);
    EXPECT_EQ(failed_probe.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 0);
    EXPECT_EQ(controller.requestedDepthForStep(), 0)
        << "a failed probe should return to cooldown before probing again";

    controller.recordBypassStep();
    controller.recordBypassStep();

    auto successful_probe = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(successful_probe.evaluated);
    ASSERT_TRUE(successful_probe.changed);
    EXPECT_EQ(successful_probe.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(successful_probe.old_depth, 0);
    EXPECT_EQ(successful_probe.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().demotions, 1u);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, DynamicHoldsHealthyDepthOneWindowBeforeZeroBypass)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.min_depth = 0;
    config.min_samples = 4;

    MTPDepthController mixed(config, /*configured_draft_tokens=*/3);
    mixed.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    mixed.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    mixed.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto mixed_window = mixed.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(mixed_window.evaluated);
    EXPECT_FALSE(mixed_window.changed);
    EXPECT_EQ(mixed_window.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(mixed.currentDepth(), 1);

    MTPDepthController all_zero(config, /*configured_draft_tokens=*/3);
    for (int i = 0; i < 3; ++i)
    {
        auto decision = all_zero.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
        EXPECT_FALSE(decision.evaluated);
    }

    auto demote = all_zero.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(demote.evaluated);
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(demote.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(demote.old_depth, 1);
    EXPECT_EQ(demote.new_depth, 0);
    EXPECT_EQ(all_zero.currentDepth(), 0);
}

TEST(Test__MTPDepthController, DynamicKeepsDepthOneOnLowButNonzeroAcceptance)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.min_depth = 0;
    config.min_samples = 4;
    /*
     * Keep the zero-accept threshold above this 50% zero-accept window so the
     * regression isolates low acceptance.  Depth 1 should remain active unless
     * the dedicated zero-acceptance bypass signal fires.
     */
    config.demote_zero_accept_rate = 0.75;
    config.demote_acceptance_rate = 0.75;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto hold = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));

    ASSERT_TRUE(hold.evaluated);
    EXPECT_FALSE(hold.changed);
    EXPECT_EQ(hold.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(hold.old_depth, 1);
    EXPECT_EQ(hold.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().demotions, 0u);
}

TEST(Test__MTPDepthController, DynamicSlowlyExploresFromHealthyNonPerfectFloorWindows)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 0.50;
    config.demote_acceptance_rate = 0.45;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto first_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_FALSE(first_window.changed);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::PromotionHysteresisActive);
    EXPECT_EQ(controller.currentDepth(), 1);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto second_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(second_window.evaluated);
    ASSERT_TRUE(second_window.changed);
    EXPECT_EQ(second_window.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(second_window.old_depth, 1);
    EXPECT_EQ(second_window.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, DynamicDefaultFloorHoldsOnNonPerfectWindows)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 1.0;
    config.demote_acceptance_rate = 0.55;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    /*
     * This is profitable enough to keep depth 1 alive, but it is not strong
     * enough to pay for deeper probes under the production default.  The
     * default benchmark prompt often looks like this: d1 can win, while
     * probing d2/d3 spends more sidecar and verifier work than it earns.
     */
    for (int window = 0; window < 2; ++window)
    {
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
        auto decision = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
        ASSERT_TRUE(decision.evaluated);
        EXPECT_FALSE(decision.changed);
        EXPECT_EQ(decision.reason, MTPDepthDecisionReason::Hold);
        EXPECT_EQ(controller.currentDepth(), 1);
    }
    EXPECT_EQ(controller.stats().promotions, 0u);
}

TEST(Test__MTPDepthController, DynamicDoesNotExploreFromUnhealthyFloorWindows)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 1.0;
    config.demote_acceptance_rate = 0.75;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto first_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_FALSE(first_window.changed);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::Hold);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto second_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(second_window.evaluated);
    EXPECT_FALSE(second_window.changed);
    EXPECT_EQ(second_window.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 0u);
}

TEST(Test__MTPDepthController, DynamicRequiresConsecutivePromotableWindows)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 0.75;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto first_window = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_FALSE(first_window.changed);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::PromotionHysteresisActive);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 0u);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto second_window = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(second_window.evaluated);
    ASSERT_TRUE(second_window.changed);
    EXPECT_EQ(second_window.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(second_window.old_depth, 2);
    EXPECT_EQ(second_window.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, DynamicPromotionStreakResetsAfterBadWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 0.75;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto first_window = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::PromotionHysteresisActive);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto broken_window = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_TRUE(broken_window.evaluated);
    EXPECT_FALSE(broken_window.changed);
    EXPECT_EQ(broken_window.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 2);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto restart_window = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(restart_window.evaluated);
    EXPECT_FALSE(restart_window.changed);
    EXPECT_EQ(restart_window.reason, MTPDepthDecisionReason::PromotionHysteresisActive);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 0u);
}

TEST(Test__MTPDepthController, DefaultHysteresisHoldsAfterMiddlingAcceptanceWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
        EXPECT_FALSE(decision.evaluated);
    }
    auto demote = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
    ASSERT_TRUE(demote.evaluated);
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(demote.reason, MTPDepthDecisionReason::DemoteLowAcceptanceRate);
    EXPECT_EQ(controller.currentDepth(), 2);

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
        EXPECT_FALSE(decision.evaluated);
    }
    auto hold = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(hold.evaluated);
    EXPECT_FALSE(hold.changed);
    EXPECT_EQ(hold.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 0u);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, DynamicProbesHigherDepthOnceBeforeDemotingIntermediateDepth)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    /*
     * Depth 1 is perfect, so the controller first promotes to the normal
     * intermediate probe depth 2.
     */
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto promote_to_two = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(promote_to_two.changed);
    ASSERT_EQ(promote_to_two.new_depth, 2);

    /*
     * An ambiguous weak depth-2 window should not permanently hide depth 3.
     * The controller marks depth 2 as rejected and spends one window testing
     * depth 3 before it settles downward.
     */
    controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    auto probe_three = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_TRUE(probe_three.evaluated);
    ASSERT_TRUE(probe_three.changed);
    EXPECT_EQ(probe_three.reason, MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    EXPECT_EQ(probe_three.old_depth, 2);
    EXPECT_EQ(probe_three.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 2u);
    EXPECT_EQ(controller.stats().demotions, 0u);

    controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    auto hold_three = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    ASSERT_TRUE(hold_three.evaluated);
    EXPECT_FALSE(hold_three.changed);
    EXPECT_EQ(hold_three.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 3);
}

TEST(Test__MTPDepthController, DynamicDemotesInsteadOfProbingOnCatastrophicZeroAcceptWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto promote_to_two = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(promote_to_two.changed);
    ASSERT_EQ(controller.currentDepth(), 2);

    controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    auto demote_to_one = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_TRUE(demote_to_one.evaluated);
    ASSERT_TRUE(demote_to_one.changed);
    EXPECT_EQ(demote_to_one.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(demote_to_one.old_depth, 2);
    EXPECT_EQ(demote_to_one.new_depth, 1)
        << "a clearly bad depth-2 window should not pay for a depth-3 probe";
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 1u);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, DynamicRequiresFreshHysteresisBeforeRetryingRejectedDepth)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);
    auto probe_three = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_EQ(probe_three.reason, MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    ASSERT_EQ(controller.currentDepth(), 3);

    auto reject_three = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    ASSERT_EQ(reject_three.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    ASSERT_EQ(controller.currentDepth(), 2);

    auto reject_two_again = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_EQ(reject_two_again.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(reject_two_again.old_depth, 2);
    EXPECT_EQ(reject_two_again.new_depth, 1)
        << "rejected depth 3 must not be probed again immediately";
    EXPECT_EQ(controller.currentDepth(), 1);

    auto first_fresh_full_accept =
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(first_fresh_full_accept.evaluated);
    EXPECT_FALSE(first_fresh_full_accept.changed);
    EXPECT_EQ(first_fresh_full_accept.reason, MTPDepthDecisionReason::PromotionHysteresisActive)
        << "retrying a rejected depth should require the configured hysteresis";
    EXPECT_EQ(controller.currentDepth(), 1);

    auto second_fresh_full_accept =
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(second_fresh_full_accept.changed);
    EXPECT_EQ(second_fresh_full_accept.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(second_fresh_full_accept.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, DynamicDoesNotForgetRejectedHigherDepthDuringProbe)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);

    auto first_probe_three =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(first_probe_three.changed);
    ASSERT_EQ(first_probe_three.reason, MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    ASSERT_EQ(controller.currentDepth(), 3);

    auto reject_three =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    ASSERT_TRUE(reject_three.changed);
    ASSERT_EQ(reject_three.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    ASSERT_EQ(controller.currentDepth(), 2);

    /*
     * This is the ROCm stochastic failure shape from the Phase 4 benchmark:
     * depth 2 is weak but not catastrophic, so the old controller would clear
     * the rejected-depth bit while probing d3 and then repeat the bad d3 lane.
     */
    auto weak_two_after_bad_three =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(weak_two_after_bad_three.evaluated);
    EXPECT_FALSE(weak_two_after_bad_three.changed);
    EXPECT_EQ(weak_two_after_bad_three.reason,
              MTPDepthDecisionReason::Hold);
    EXPECT_EQ(weak_two_after_bad_three.old_depth, 2);
    EXPECT_EQ(weak_two_after_bad_three.new_depth, 2)
        << "rejected depth 3 must not be probed again, but d2 should remain "
           "active on a low-but-nonzero survivor window";
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, DynamicHoldsHighestUnrejectedDepthAfterDeeperRejection)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);

    auto probe_three =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(probe_three.changed);
    ASSERT_EQ(probe_three.reason, MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    ASSERT_EQ(controller.currentDepth(), 3);

    auto reject_three =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    ASSERT_TRUE(reject_three.changed);
    ASSERT_EQ(reject_three.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    ASSERT_EQ(controller.currentDepth(), 2);

    /*
     * This mirrors the current ROCm stochastic evidence: fixed d2 is the best
     * surviving lane after d3 proves bad, but individual d2 windows can still
     * be imperfect.  Hold d2 on a low-but-nonzero window instead of collapsing
     * to d1; the explicit zero-accept path remains the hard demotion signal.
     */
    auto weak_but_nonzero_two =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(weak_but_nonzero_two.evaluated);
    EXPECT_FALSE(weak_but_nonzero_two.changed);
    EXPECT_EQ(weak_but_nonzero_two.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(weak_but_nonzero_two.old_depth, 2);
    EXPECT_EQ(weak_but_nonzero_two.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);

    auto zero_two =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_TRUE(zero_two.evaluated);
    ASSERT_TRUE(zero_two.changed);
    EXPECT_EQ(zero_two.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(zero_two.old_depth, 2);
    EXPECT_EQ(zero_two.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
}

TEST(Test__MTPDepthController, CooldownPreventsImmediateOscillation)
{
    MTPDepthController controller(dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/2),
        /*configured_draft_tokens=*/3);

    auto demote = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(controller.currentDepth(), 2);

    auto cooldown = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    EXPECT_TRUE(cooldown.evaluated);
    EXPECT_FALSE(cooldown.changed);
    EXPECT_EQ(cooldown.reason, MTPDepthDecisionReason::CooldownActive);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, ObserveRecommendsWithoutChangingDepth)
{
    auto config = dynamicConfig(/*initial_depth=*/3, /*max_depth=*/3);
    config.mode = MTPDepthPolicyMode::Observe;
    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    auto decision = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    EXPECT_TRUE(decision.evaluated);
    EXPECT_FALSE(decision.changed);
    EXPECT_TRUE(decision.observe_recommendation);
    EXPECT_EQ(decision.recommended_depth, 2);
    EXPECT_EQ(decision.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().observe_recommendations, 1u);
}

TEST(Test__MTPDepthController, BudgetLimitedStepsDoNotTeachPolicy)
{
    MTPDepthController controller(dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0),
        /*configured_draft_tokens=*/3);

    auto budget = controller.recordStep(MTPDepthObservation{
        .requested_depth = 3,
        .effective_depth = 1,
        .accepted_speculative_prefix = 0,
        .budget_limited = true,
        .rollback = false,
    });
    EXPECT_FALSE(budget.evaluated);
    EXPECT_EQ(budget.reason, MTPDepthDecisionReason::BudgetLimited);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().windows, 0u);

    auto demote = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    EXPECT_TRUE(demote.changed);
    EXPECT_EQ(controller.currentDepth(), 2);
}
