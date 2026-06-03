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
    EXPECT_EQ(early.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
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
    MTPDepthController controller(
        dynamicConfig(/*initial_depth=*/1, /*max_depth=*/3),
        /*configured_draft_tokens=*/3);

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
    EXPECT_EQ(controller.currentDepth(), 1);

    for (int i = 0; i < 3; ++i)
    {
        auto decision = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
        EXPECT_FALSE(decision.evaluated);
    }
    auto hold = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(hold.evaluated);
    EXPECT_FALSE(hold.changed);
    EXPECT_EQ(hold.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 0u);
    EXPECT_EQ(controller.stats().demotions, 1u);
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
