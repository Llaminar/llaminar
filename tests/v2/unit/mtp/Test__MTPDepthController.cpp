/**
 * @file Test__MTPDepthController.cpp
 * @brief Device-free proofs of depth admission and measured-policy transitions.
 *
 * Synthetic observations exercise bounds, budget truncation, hysteresis and
 * generated policy independently of model execution. Learned startup tests
 * retain the complete adaptive capacity. Actual compact-publication fixtures prove
 * disjoint target confirmation, reset and unchanged trial ownership. An economical initial depth is not
 * an implicit ceiling. Captured backend tests prove the matching device ABI.
 */
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include "execution/mtp/DeviceGenerationContract.h"

#include "execution/mtp/MTPDepthController.h"
#include "execution/mtp/MTPDepthLearnedPolicy.h"
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
#include "kernels/common/SamplingMath.h"
#include "execution/mtp/MTPDepthPolicyScopeProjection.h"

using namespace llaminar2;

namespace
{
    /** @brief Build an explicit, untrained policy for transition unit tests. */
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
        config.use_generated_policy = false;
        config.promote_full_accept_rate = 0.75;
        config.demote_zero_accept_rate = 0.30;
        config.demote_acceptance_rate = 0.65;
        return config;
    }

    /** @brief Create one untruncated transaction with an accepted-prefix count. */
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
    /** @brief Real compact reducer plus publication counts for an authored host outcome.
     * @param depth Executed verifier depth. @param prefix Authored accepted prefix.
     * @param incoming Actual already-emitted condition; updated from raw publication.
     * @param budget Positive publication budget, or 0 for the full transaction.
     * @param stop Stop at the first rejected row; requires prefix 0.
     * @return Actual emitted Y and typed publication V, never an inferred mirror.
     */
    MTPDepthObservation committedObservation(int depth,int prefix,bool &incoming,int budget=0,bool stop=false)
    {
        using namespace sampling_math;
        if(stop&&prefix!=0)throw std::logic_error("stop fixture needs first rejected row");
        std::array<int,16> tokens{},flags{};std::array<int,17> output{};
        std::array<int,kSpeculativeBatchMetaCount> meta{};
        for(int i=0;i<16;++i){tokens[i]=1000+i;flags[i]=i<prefix;}
        const int stopped_token=tokens[0];const int rows=budget?budget:depth+1;
        summarize_speculative_verify_batch_at_commit_boundary(111,tokens.data(),flags.data(),depth,
            stop?&stopped_token:nullptr,stop?1:0,2000,1,rows,output.data(),output.size(),meta.data(),nullptr,incoming?1:0);
        if(meta[kSpecBatchMetaOk]!=1)throw std::logic_error("invalid compact fixture");
        MTPPublicationWitness publication;
        const int published=std::min(meta[kSpecBatchMetaTargetVerifierStateCommitCount],rows);
        if(!MTPPublicationWitness::fromCounts(meta[kSpecBatchMetaOutputCount],incoming?1:0,published,
                meta[kSpecBatchMetaStoppedOnOutput]!=0,&publication))throw std::logic_error("invalid publication fixture");
        auto actual=observation(depth,meta[kSpecBatchMetaAcceptedSpeculativePrefix]);
        actual.budget_limited=rows<depth+1;actual.newly_emitted_tokens=publication.newlyEmittedTokens();
        actual.publication=publication;incoming=!publication.modelStopped()&&publication.outgoingDebt();
        return actual;
    }
    /** @return The existing exact cost scope with explicitly authored bounds/start/cadence. */
    MTPDepthPolicyConfig survivalConfig(int initial=5,int minimum=1,int maximum=15)
    {
        auto config=dynamicConfig(initial,maximum,16,8);config.min_samples=4;
        config.min_depth=minimum;config.use_generated_policy=true;
        config.backend=MTPDepthPolicyBackend::ROCm;config.model_class=MTPDepthPolicyModelClass::MoE;
        config.scope={MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,4,1,15,2560,48,512,10,90,6};
        return config;
    }

}

TEST(Test__MTPDepthController, FixedModeIgnoresAdaptiveBounds)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Fixed;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 3;
    config.window_size = 0;
    config.min_samples = 0;
    config.cooldown_steps = -1;
    config.promote_consecutive_windows = 0;

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

TEST(Test__MTPDepthController, GreedyDynamicDefaultInitialDepthStartsAtDepthTwoWhenAvailable)
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

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    EXPECT_EQ(controller.currentDepth(), 2)
        << "Greedy dynamic mode should start at depth 2 when the configured "
           "range allows it, staying below the risky deepest lane while "
           "avoiding a short-run learning tax.";
    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 3);
}

/**
 * @test A direct controller caller gets the full adaptive range by default.
 *
 * This protects non-CLI callers from repeating the historical bug where the
 * automatic maximum inherited configured_draft_tokens=1 and made dynamic mode
 * indistinguishable from fixed depth one.
 */
TEST(Test__MTPDepthController, AutomaticMaximumUsesFullSupportedRange)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.min_depth = 1;
    config.max_depth = 0;
    config.initial_depth = 0;
    config.use_generated_policy = false;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/1,
        MTPVerifyMode::Greedy);

    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 15);
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, StochasticDynamicDefaultInitialDepthStartsAtDepthOne)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 0;
    config.window_size = 16;
    config.min_samples = 4;
    config.cooldown_steps = 0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    EXPECT_EQ(controller.currentDepth(), 1)
        << "Stochastic dynamic mode should begin in the cheapest useful lane; "
           "residual rejections cannot carry ready logits, so starting too "
           "deep pays a condition-forward learning tax.";
    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 3);
}

TEST(Test__MTPDepthController, GeneratedPolicyInitialDepthUsesLearnedMoEGreedyLane)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 0;
    config.window_size = 16;
    config.min_samples = 4;
    config.cooldown_steps = 0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    EXPECT_EQ(controller.currentDepth(), 3)
        << "generated hold rows identify the fixed-depth winner, so dynamic "
           "mode should warm-start there instead of probing through d2 first";
}

TEST(Test__MTPDepthController, GeneratedPolicyInitialDepthUsesRelearnedROCmMoEStochasticWinner)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_depth = 1;
    config.max_depth = 0;
    config.initial_depth = 0;
    config.window_size = 16;
    config.min_samples = 4;
    config.cooldown_steps = 0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    EXPECT_EQ(controller.currentDepth(), 3)
        << "fresh fixed-depth evidence replaces the stale depth-one hold; "
           "this is learned startup, not an authored depth override";
    EXPECT_EQ(controller.minDepth(), 1);
    EXPECT_EQ(controller.maxDepth(), 15);
}

/** @test Admission follows measured startup, without replacing explicit intent. */
TEST(Test__MTPDepthController, LearnedStartupSelectionHonorsTableAndExplicitBounds)
{
    for (const auto &choice : kMTPGeneratedDepthPolicyStartups)
    {
        MTPDepthPolicyConfig config;
        config.mode = MTPDepthPolicyMode::Dynamic;
        config.use_generated_policy = true;
        config.backend = choice.backend;
        config.model_class = choice.model_class;
        config.scope = choice.scope;
        config.min_depth = 1;
        config.max_depth = 0;
        config.initial_depth = 0;
        MTPDepthController automatic(config, 1, choice.verify_mode);
        EXPECT_EQ(automatic.currentDepth(), choice.initial_depth);
        EXPECT_EQ(automatic.minDepth(), 1);
        EXPECT_EQ(automatic.maxDepth(), 15);

        // Authored initialization and fixed execution remain separate intents.
        config.initial_depth = 15;
        MTPDepthController authored(config, 1, choice.verify_mode);
        EXPECT_EQ(authored.currentDepth(), 15);
        config.mode = MTPDepthPolicyMode::Fixed;
        MTPDepthController fixed(config, 2, choice.verify_mode);
        EXPECT_EQ(fixed.currentDepth(), 2);
        EXPECT_EQ(fixed.maxDepth(), 2);

        // A learned startup outside an authored interval must never escape it.
        config.mode = MTPDepthPolicyMode::Dynamic;
        config.initial_depth = 0;
        config.min_depth = 5;
        config.max_depth = 7;
        MTPDepthController bounded(config, 1, choice.verify_mode);
        EXPECT_GE(bounded.currentDepth(), 5);
        EXPECT_LE(bounded.currentDepth(), 7);
        EXPECT_EQ(bounded.maxDepth(), 7);
    }
}

/** @test Healthy stochastic windows follow the learned winner, not a d1 trap. */
TEST(Test__MTPDepthController, RelearnedROCmStochasticPolicyPromotesAndHoldsWithoutCappingCapacity)
{
    for (const int initial : {1, 2, 3})
    {
        auto config = dynamicConfig(initial, 15);
        config.use_generated_policy = true;
        config.backend = MTPDepthPolicyBackend::ROCm;
        config.model_class = MTPDepthPolicyModelClass::MoE;
        MTPDepthController controller(config, 1, MTPVerifyMode::SpeculativeSampling);
        controller.recordStep(observation(initial, initial));
        const auto decision = controller.recordStep(observation(initial, initial));
        EXPECT_EQ(controller.currentDepth(), 3) << initial;
        EXPECT_EQ(decision.reason, initial == 3 ? MTPDepthDecisionReason::GeneratedPolicyHold
                                               : MTPDepthDecisionReason::GeneratedPolicyPromote);
        EXPECT_EQ(controller.maxDepth(), 15);
    }
}

TEST(Test__MTPDepthController, DynamicDefaultInitialDepthPreservesDepthZeroBypass)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Dynamic;
    config.min_depth = 0;
    config.max_depth = 3;
    config.initial_depth = 0;
    config.window_size = 16;
    config.min_samples = 4;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    EXPECT_EQ(controller.currentDepth(), 0)
        << "A min-depth-zero policy is an explicit adaptive bypass request and "
           "must not be silently lifted to the depth-2 warm start.";
    EXPECT_EQ(controller.minDepth(), 0);
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

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

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

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

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
    auto config = dynamicConfig(/*initial_depth=*/1, /*max_depth=*/2);
    config.promote_consecutive_windows = 1;
    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/2,
        MTPVerifyMode::SpeculativeSampling);

    auto d1a = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_FALSE(d1a.evaluated);
    auto d1b = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_TRUE(d1b.changed);
    EXPECT_EQ(d1b.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(controller.currentDepth(), 2);

    auto d2a = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    EXPECT_FALSE(d2a.evaluated);
    auto d2b = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    EXPECT_FALSE(d2b.changed);
    EXPECT_EQ(d2b.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, RequestResetRestoresInitialDepthAndForgetsAdaptiveHistory)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    ASSERT_FALSE(
        controller.recordStep(
            observation(/*depth=*/1, /*accepted_prefix=*/1))
            .evaluated);
    const auto promotion =
        controller.recordStep(
            observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(promotion.changed);
    ASSERT_EQ(controller.currentDepth(), 2);
    ASSERT_EQ(controller.stats().promotions, 1u);

    /*
     * A new request must not inherit either the selected depth or the partial
     * policy evidence that selected it.  Prefix restore and ordinary prefill
     * both rely on this exact reset contract.
     */
    controller.reset();

    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.requestedDepthForStep(), 1);
    EXPECT_EQ(controller.stats().windows, 0u);
    EXPECT_EQ(controller.stats().updates, 0u);
    EXPECT_EQ(controller.stats().promotions, 0u);
    EXPECT_EQ(controller.stats().demotions, 0u);
    EXPECT_EQ(controller.stats().observe_recommendations, 0u);

    const auto first_new_request_observation =
        controller.recordStep(
            observation(/*depth=*/1, /*accepted_prefix=*/1));
    EXPECT_FALSE(first_new_request_observation.evaluated)
        << "Reset must discard the preceding request's partial/full window";
    EXPECT_EQ(controller.currentDepth(), 1);
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

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

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

TEST(Test__MTPDepthController, GeneratedPolicyDoesNotPromoteWithoutBackendMatch)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::Any;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    /*
     * Generated rows are keyed by backend.  A controller with no backend
     * evidence must not accidentally consume a CUDA promotion row and turn a
     * generic stochastic request into a depth increase.
     */
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto hold = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));

    ASSERT_TRUE(hold.evaluated);
    EXPECT_FALSE(hold.changed);
    EXPECT_EQ(hold.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(hold.old_depth, 1);
    EXPECT_EQ(hold.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 0u);
}

TEST(Test__MTPDepthController, GeneratedPolicyPromotesCUDAStochasticDepthOneToBestDepth)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::CUDA;
    config.model_class = MTPDepthPolicyModelClass::Dense;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    /*
     * CUDA dense stochastic fixed-depth evidence shows depth 3 is profitable
     * from a healthy depth-1 window.  The generated policy should allow that
     * first climb without waiting for the handwritten perfect-window fallback.
     */
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto promote = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));

    ASSERT_TRUE(promote.evaluated);
    ASSERT_TRUE(promote.changed);
    EXPECT_EQ(promote.reason, MTPDepthDecisionReason::GeneratedPolicyPromote);
    EXPECT_EQ(promote.old_depth, 1);
    EXPECT_EQ(promote.new_depth, 3)
        << "generated policy rows target the empirically best fixed-depth lane, "
           "not merely the next deeper lane";
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, GeneratedPolicyPromotesHealthyStochasticDepthTwo)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::CUDA;
    config.model_class = MTPDepthPolicyModelClass::Dense;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto promote = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));

    ASSERT_TRUE(promote.evaluated);
    ASSERT_TRUE(promote.changed);
    EXPECT_EQ(promote.reason, MTPDepthDecisionReason::GeneratedPolicyPromote);
    EXPECT_EQ(promote.old_depth, 2);
    EXPECT_EQ(promote.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, GeneratedPolicyPromotesROCmMoEGreedyAtMoEThreshold)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/5,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_samples = 5;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    /*
     * ROCm MoE greedy fixed-depth evidence shows depth 3 is the best lane even
     * when depth-2 acceptance is below the dense ROCm promotion threshold.
     * The generated MoE-specific row should climb without weakening dense.
     */
    for (int i = 0; i < 4; ++i)
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto promote = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));

    ASSERT_TRUE(promote.evaluated);
    ASSERT_TRUE(promote.changed);
    EXPECT_EQ(promote.reason, MTPDepthDecisionReason::GeneratedPolicyPromote);
    EXPECT_EQ(promote.old_depth, 2);
    EXPECT_EQ(promote.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, GeneratedPolicyDemotesWeakStochasticDepthTwo)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::Dense;
    config.min_samples = 4;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    auto demote = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));

    ASSERT_TRUE(demote.evaluated);
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(demote.reason, MTPDepthDecisionReason::GeneratedPolicyDemote);
    EXPECT_EQ(demote.old_depth, 2);
    EXPECT_EQ(demote.new_depth, 1);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, GeneratedPolicyDemotesWeakDepthThree)
{
    auto config = dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/16,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::Dense;
    config.min_samples = 4;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
    auto demote = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));

    ASSERT_TRUE(demote.evaluated);
    ASSERT_TRUE(demote.changed);
    EXPECT_EQ(demote.reason, MTPDepthDecisionReason::GeneratedPolicyDemote);
    EXPECT_EQ(demote.old_depth, 3);
    EXPECT_EQ(demote.new_depth, 1)
        << "generated policy rows can demote directly to the empirically best "
           "fixed-depth lane";
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().demotions, 1u);
}

TEST(Test__MTPDepthController, GeneratedPolicyHoldPinsBestMoEGreedyLane)
{
    auto config = dynamicConfig(
        /*initial_depth=*/3,
        /*max_depth=*/3,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;
    config.demote_acceptance_rate = 0.90;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    /*
     * ROCm MoE greedy fixed-depth evidence says depth 3 is the best steady
     * lane at roughly this acceptance level.  A generated hold row should
     * suppress an otherwise eager handwritten low-acceptance demotion.
     */
    controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/2));
    auto hold = controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));

    ASSERT_TRUE(hold.evaluated);
    EXPECT_FALSE(hold.changed);
    EXPECT_EQ(hold.reason, MTPDepthDecisionReason::GeneratedPolicyHold);
    EXPECT_EQ(hold.old_depth, 3);
    EXPECT_EQ(hold.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().demotions, 0u);
}

/**
 * @test A learned hold preserves the complete eligible window across clipped work.
 *
 * Perfect partial windows must not enter the generic early-probe branch and
 * reset evidence that the resident controller retains until the full window.
 * Repeated windows cover both spare retained capacity and a bound at the hold.
 */
TEST(Test__MTPDepthController, LearnedHoldPreservesFullWindowCadence)
{
    // Both an open capacity and a bound at the learned hold must retain evidence.
    for (const int maximum_depth : {3, 15})
    {
        SCOPED_TRACE(maximum_depth);
        auto config = dynamicConfig(/*initial_depth=*/3, maximum_depth,
                                    /*window_size=*/16, /*cooldown_steps=*/8);
        config.min_samples = 4;
        config.promote_consecutive_windows = 3;
        config.promote_full_accept_rate = 1.0;
        config.demote_acceptance_rate = 0.55;
        config.use_generated_policy = true;
        config.backend = MTPDepthPolicyBackend::CPU;
        config.model_class = MTPDepthPolicyModelClass::MoE;
        MTPDepthController controller(config, /*configured_draft_tokens=*/15,
                                      MTPVerifyMode::SpeculativeSampling);

        for (uint64_t window = 0; window < 3; ++window)
        {
            for (uint64_t run = 1; run <= 16; ++run)
            {
                // Clipped transactions cannot advance or clear eligible evidence.
                auto clipped = observation(/*depth=*/3, /*accepted_prefix=*/0);
                clipped.budget_limited = true;
                const auto ignored = controller.recordStep(clipped);
                EXPECT_FALSE(ignored.evaluated);
                EXPECT_EQ(ignored.reason, MTPDepthDecisionReason::BudgetLimited);
                EXPECT_EQ(controller.stats().windows, window);

                const auto decision = controller.recordStep(observation(3, 3));
                EXPECT_EQ(decision.window.verifier_runs, run);
                EXPECT_EQ(decision.window.attempted_draft_tokens, run * 3);
                EXPECT_EQ(decision.window.accepted_draft_tokens, run * 3);
                EXPECT_EQ(decision.window.rejected_draft_tokens, 0u);
                EXPECT_EQ(decision.window.full_accepts, run);
                EXPECT_EQ(decision.window.zero_accepts, 0u);
                EXPECT_EQ(decision.window.rollbacks, 0u);
                EXPECT_FALSE(decision.changed);
                EXPECT_EQ(controller.currentDepth(), 3);
                if (run < 16)
                {
                    EXPECT_FALSE(decision.evaluated);
                    EXPECT_EQ(decision.reason, MTPDepthDecisionReason::WindowNotReady);
                    EXPECT_EQ(controller.stats().windows, window);
                }
                else
                {
                    EXPECT_TRUE(decision.evaluated);
                    EXPECT_EQ(decision.reason, MTPDepthDecisionReason::GeneratedPolicyHold);
                    EXPECT_EQ(controller.stats().windows, window + 1);
                }
            }
        }
        EXPECT_EQ(controller.stats().updates, 0u);
        EXPECT_EQ(controller.stats().promotions, 0u);
        EXPECT_EQ(controller.stats().demotions, 0u);
    }
}

TEST(Test__MTPDepthController, GeneratedBestDepthWaitsForFullWindowBeforeNoisyDemotion)
{
    auto config = dynamicConfig(
        /*initial_depth=*/0,
        /*max_depth=*/3,
        /*window_size=*/16,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;
    config.demote_acceptance_rate = 0.90;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    ASSERT_EQ(controller.currentDepth(), 3);
    for (int i = 0; i < 4; ++i)
    {
        const auto decision =
            controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
        EXPECT_FALSE(decision.evaluated)
            << "small low-acceptance windows should not evict the learned "
               "best fixed-depth lane";
        EXPECT_EQ(controller.currentDepth(), 3);
    }

    for (int i = 0; i < 11; ++i)
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
    auto first_full_window =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));

    ASSERT_TRUE(first_full_window.evaluated);
    EXPECT_FALSE(first_full_window.changed)
        << "the learned fixed-depth winner gets one full-window grace period "
           "before a non-catastrophic demotion";
    EXPECT_EQ(first_full_window.reason, MTPDepthDecisionReason::GeneratedBestDepthGraceWindow);
    EXPECT_EQ(first_full_window.old_depth, 3);
    EXPECT_EQ(first_full_window.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);

    for (int i = 0; i < 15; ++i)
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));
    auto second_full_window =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/1));

    ASSERT_TRUE(second_full_window.evaluated);
    ASSERT_TRUE(second_full_window.changed)
        << "two consecutive bad full windows are request-local evidence that "
           "the generated best lane is wrong for this prompt";
    EXPECT_EQ(second_full_window.reason, MTPDepthDecisionReason::DemoteLowAcceptanceRate);
    EXPECT_EQ(second_full_window.old_depth, 3);
    EXPECT_EQ(second_full_window.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, GeneratedBestDepthGraceCoversObservedROCmMoEGreedyWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/0,
        /*max_depth=*/3,
        /*window_size=*/16,
        /*cooldown_steps=*/0);
    config.use_generated_policy = true;
    config.backend = MTPDepthPolicyBackend::ROCm;
    config.model_class = MTPDepthPolicyModelClass::MoE;
    config.min_samples = 4;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::Greedy);

    ASSERT_EQ(controller.currentDepth(), 3);

    /*
     * Regression fixture from the ROCm MoE greedy dynamic matrix:
     * acceptance=19/48=0.395833, zero_accept=6/16=0.375,
     * full_accept=2/16=0.125.  Fixed d3 still won the whole request, so this
     * first noisy window should not evict the generated best lane.
     */
    const int accepted_prefixes[] = {
        0, 2, 1, 3,
        0, 2, 1, 3,
        0, 2, 1, 0,
        2, 0, 2, 0,
    };
    MTPDepthDecision decision;
    for (int accepted_prefix : accepted_prefixes)
        decision = controller.recordStep(observation(/*depth=*/3, accepted_prefix));

    ASSERT_TRUE(decision.evaluated);
    EXPECT_FALSE(decision.changed);
    EXPECT_EQ(decision.reason, MTPDepthDecisionReason::GeneratedBestDepthGraceWindow);
    EXPECT_EQ(controller.currentDepth(), 3);
    EXPECT_EQ(controller.stats().demotions, 0u);
}

TEST(Test__MTPDepthController, GeneratedPolicyDoesNotAffectFixedMode)
{
    MTPDepthPolicyConfig config;
    config.mode = MTPDepthPolicyMode::Fixed;
    config.use_generated_policy = true;
    config.min_depth = 1;
    config.max_depth = 3;
    config.initial_depth = 3;
    config.window_size = 1;
    config.min_samples = 1;
    config.cooldown_steps = 0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/1);
    auto decision = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));

    EXPECT_FALSE(decision.evaluated);
    EXPECT_FALSE(decision.changed);
    EXPECT_EQ(decision.reason, MTPDepthDecisionReason::FixedMode);
    EXPECT_EQ(controller.currentDepth(), 1);
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
        /*initial_depth=*/1,
        /*max_depth=*/2,
        /*window_size=*/4,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 0.75;

    MTPDepthController controller(config, /*configured_draft_tokens=*/2);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto first_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_FALSE(first_window.changed);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::PromotionHysteresisActive);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 0u);

    for (int i = 0; i < 3; ++i)
        controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto second_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(second_window.evaluated);
    ASSERT_TRUE(second_window.changed);
    EXPECT_EQ(second_window.reason, MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(second_window.old_depth, 1);
    EXPECT_EQ(second_window.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
    EXPECT_EQ(controller.stats().promotions, 1u);
}

TEST(Test__MTPDepthController, DynamicPromotionStreakResetsAfterBadWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/2,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(config, /*configured_draft_tokens=*/2);

    auto first_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(first_window.evaluated);
    EXPECT_EQ(first_window.reason, MTPDepthDecisionReason::Hold);

    auto fresh_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(fresh_window.evaluated);
    EXPECT_TRUE(fresh_window.changed);
    EXPECT_EQ(fresh_window.reason, MTPDepthDecisionReason::PromoteFullAcceptRate)
        << "a perfect fresh probe may still promote immediately when the "
           "destination depth has not been rejected";
    EXPECT_EQ(controller.currentDepth(), 2);

    auto demote_to_one = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_TRUE(demote_to_one.evaluated);
    ASSERT_TRUE(demote_to_one.changed);
    EXPECT_EQ(demote_to_one.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(controller.currentDepth(), 1);

    auto restart_window = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/0));
    ASSERT_TRUE(restart_window.evaluated);
    EXPECT_FALSE(restart_window.changed);
    EXPECT_EQ(restart_window.reason, MTPDepthDecisionReason::Hold);
    EXPECT_EQ(controller.currentDepth(), 1);
    EXPECT_EQ(controller.stats().promotions, 1u);
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

TEST(Test__MTPDepthController, DynamicProbeCanContinueToConfiguredMaximumAfterFullAcceptance)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/4,
        /*window_size=*/2,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/4);

    /*
     * Depth 1 is perfect, so the controller first promotes to the normal
     * intermediate probe depth 2.
     */
    controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    auto promote_to_two = controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1));
    ASSERT_TRUE(promote_to_two.changed);
    ASSERT_EQ(promote_to_two.new_depth, 2);

    /*
     * With a generic 1..4 policy, depth 3 is still an intermediate lane.  An
     * ambiguous weak depth-2 window may probe it once before settling downward.
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
    auto promote_to_four =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/3));
    ASSERT_TRUE(promote_to_four.evaluated);
    EXPECT_TRUE(promote_to_four.changed);
    EXPECT_EQ(promote_to_four.reason,
              MTPDepthDecisionReason::PromoteFullAcceptRate);
    EXPECT_EQ(promote_to_four.new_depth, 4);
    EXPECT_EQ(controller.currentDepth(), 4);
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
        /*max_depth=*/2,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 2;

    MTPDepthController controller(config, /*configured_draft_tokens=*/2);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);
    auto reject_two = controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/0));
    ASSERT_EQ(reject_two.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(reject_two.old_depth, 2);
    EXPECT_EQ(reject_two.new_depth, 1);
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

TEST(Test__MTPDepthController, DynamicCanProbeConfiguredMaximumOnAmbiguousIntermediateWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);

    auto weak_two_with_higher_probe =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(weak_two_with_higher_probe.evaluated);
    EXPECT_TRUE(weak_two_with_higher_probe.changed);
    EXPECT_EQ(weak_two_with_higher_probe.reason,
              MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    EXPECT_EQ(weak_two_with_higher_probe.old_depth, 2);
    EXPECT_EQ(weak_two_with_higher_probe.new_depth, 3)
        << "The configured maximum is a real candidate, not a reserved depth-three sentinel.";
    EXPECT_EQ(controller.currentDepth(), 3);
}

TEST(Test__MTPDepthController, DynamicDemotesConfiguredMaximumAfterFailedProbe)
{
    auto config = dynamicConfig(
        /*initial_depth=*/1,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;

    MTPDepthController controller(config, /*configured_draft_tokens=*/3);

    ASSERT_EQ(controller.recordStep(observation(/*depth=*/1, /*accepted_prefix=*/1)).new_depth, 2);

    auto weak_but_nonzero_two =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/1));
    ASSERT_TRUE(weak_but_nonzero_two.evaluated);
    EXPECT_TRUE(weak_but_nonzero_two.changed);
    EXPECT_EQ(weak_but_nonzero_two.reason,
              MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    EXPECT_EQ(weak_but_nonzero_two.old_depth, 2);
    EXPECT_EQ(weak_but_nonzero_two.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);

    auto zero_three =
        controller.recordStep(observation(/*depth=*/3, /*accepted_prefix=*/0));
    ASSERT_TRUE(zero_three.evaluated);
    ASSERT_TRUE(zero_three.changed);
    EXPECT_EQ(zero_three.reason, MTPDepthDecisionReason::DemoteZeroAcceptRate);
    EXPECT_EQ(zero_three.old_depth, 3);
    EXPECT_EQ(zero_three.new_depth, 2);
    EXPECT_EQ(controller.currentDepth(), 2);
}

TEST(Test__MTPDepthController, DynamicFallbackEntersConfiguredMaximumOnPerfectLowerWindow)
{
    auto config = dynamicConfig(
        /*initial_depth=*/2,
        /*max_depth=*/3,
        /*window_size=*/1,
        /*cooldown_steps=*/0);
    config.promote_consecutive_windows = 1;
    config.promote_full_accept_rate = 1.0;

    MTPDepthController controller(
        config,
        /*configured_draft_tokens=*/3,
        MTPVerifyMode::SpeculativeSampling);

    auto perfect_two =
        controller.recordStep(observation(/*depth=*/2, /*accepted_prefix=*/2));
    ASSERT_TRUE(perfect_two.evaluated);
    EXPECT_TRUE(perfect_two.changed);
    EXPECT_EQ(perfect_two.reason, MTPDepthDecisionReason::PromoteFullAcceptRate)
        << "A perfect lower-depth window must be allowed to explore the configured maximum.";
    EXPECT_EQ(perfect_two.old_depth, 2);
    EXPECT_EQ(perfect_two.new_depth, 3);
    EXPECT_EQ(controller.currentDepth(), 3);
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

namespace
{
    /** @brief Synthetic measured overlay; no generated production defaults change. */
    constexpr MTPDepthPolicyScope exactTP4Scope{
        MTPVerifierPhysicalWidthPolicy::ExactLogicalRows, 4, 1, 15, 2048, 40, 256, 8, 90, 6};
    /** @brief Generic rows deliberately precede the measured domain to stress selection. */
    struct ScopeOrderingTables
    {
        static constexpr MTPGeneratedDepthPolicyRule rules[]{
            {MTPVerifyMode::Greedy, MTPDepthPolicyBackend::ROCm, MTPDepthPolicyModelClass::MoE,
             3, 0.75, 1.0, 1.0, 0.0, 0, "legacy_hold"},
            {MTPVerifyMode::Greedy, MTPDepthPolicyBackend::ROCm, MTPDepthPolicyModelClass::MoE,
             2, 0.0, 1.0, 1.0, 0.0, 1, "legacy_promote"},
            {MTPVerifyMode::Greedy, MTPDepthPolicyBackend::ROCm, MTPDepthPolicyModelClass::MoE,
             3, 0.9, 1.0, 1.0, 0.0, 2, "measured_promote", exactTP4Scope},
        };
        static constexpr MTPGeneratedDepthPolicyStartup startups[]{
            {MTPVerifyMode::Greedy, MTPDepthPolicyBackend::ROCm, MTPDepthPolicyModelClass::MoE, 3},
            {MTPVerifyMode::Greedy, MTPDepthPolicyBackend::ROCm, MTPDepthPolicyModelClass::MoE, 5, exactTP4Scope},
        };
    };
    /** @return Complete source-like controller identity, independent of live device state. */
    MTPLearnedDepthPolicyContext scopedContext()
    {
        return {.enabled = true, .backend = MTPDepthPolicyBackend::ROCm,
                .model_class = MTPDepthPolicyModelClass::MoE, .verify_mode = MTPVerifyMode::Greedy,
                .scope = exactTP4Scope};
    }
    /** @return Frozen local four-GPU metadata with no backend calls. */
    ClusterInventory scopedInventory()
    {
        ClusterInventory inventory;
        inventory.ranks.resize(1);
        inventory.ranks[0].hostname = "scope-host";
        for (int ordinal = 0; ordinal < 4; ++ordinal)
        {
            DeviceInfo gpu;
            gpu.type = DeviceType::ROCm;
            gpu.local_device_id = ordinal;
            gpu.compute_capability_major = 90;
            gpu.compute_capability_minor = 6;
            inventory.ranks[0].gpus.push_back(gpu);
        }
        return inventory;
    }
    /** @return Actual local TP membership rather than requested CLI degrees. */
    RankExecutionPlan scopedPlan()
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.primary_device = GlobalDeviceAddress::rocm(0);
        for (int ordinal = 0; ordinal < 4; ++ordinal)
            plan.local_tp_devices.push_back(GlobalDeviceAddress::rocm(ordinal));
        return plan;
    }
    /** @return Loaded main/predictor geometry with an explicitly excluded predictor. */
    ModelMemoryProfile scopedModel()
    {
        ModelMemoryProfile model;
        model.n_layers = 41;
        model.mtp_layer_count = 1;
        model.d_model = 2048;
        model.expert_count = 256;
        model.expert_used_count = 8;
        return model;
    }
}

TEST(Test__MTPDepthController, MeasuredScopeSelectsBeforePredicatesAndStartupBounds)
{
    auto context = scopedContext();
    const MTPLearnedDepthWindow high{100, 95, 10, 0, 5};
    const auto selected = matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, high);
    EXPECT_TRUE(selected.matched);
    EXPECT_EQ(selected.depth_delta, 2);
    const MTPLearnedDepthWindow below_interval{100, 80, 10, 0, 5};
    EXPECT_FALSE((matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, below_interval).matched));
    EXPECT_FALSE((matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 2, high).matched));
    EXPECT_EQ((resolveMTPLearnedStartup<0, ScopeOrderingTables>(context, 1, 15)), 5);
    EXPECT_EQ((resolveMTPLearnedStartup<0, ScopeOrderingTables>(context, 1, 4)), -1);
}

TEST(Test__MTPDepthController, UnmeasuredScopeKeepsGenericPolicyAndRejectsPartialIdentity)
{
    const auto base = scopedContext();
    std::vector<MTPDepthPolicyScope> unmeasured;
    auto add = [&](auto change) { auto scope = exactTP4Scope; change(scope); unmeasured.push_back(scope); };
    add([](auto &s) { s.verifier_width_policy = MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope; });
    add([](auto &s) { s.tp_width = 1; });
    add([](auto &s) { s.pp_stages = 2; });
    add([](auto &s) { s.retained_depth = 6; });
    add([](auto &s) { s.hidden_dim = 4096; });
    add([](auto &s) { s.main_layers = 48; });
    add([](auto &s) { s.expert_count = 512; });
    add([](auto &s) { s.experts_per_token = 4; });
    add([](auto &s) { s.compute_capability_major = 94; });
    add([](auto &s) { s.compute_capability_minor = 0; });
    unmeasured.push_back({});
    for (const auto &scope : unmeasured)
    {
        auto context = base;
        context.scope = scope;
        ASSERT_TRUE(context.valid());
        const auto match = matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(
            context, 3, {.attempted = 100, .accepted = 95, .runs = 10});
        EXPECT_TRUE(match.matched);
        EXPECT_EQ(match.depth_delta, 0);
        EXPECT_EQ((resolveMTPLearnedStartup<0, ScopeOrderingTables>(context, 1, 15)), 3);
    }
    auto partial = base;
    partial.scope.tp_width = 0;
    EXPECT_FALSE(partial.valid());
    EXPECT_FALSE((matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(
        partial, 3, {.attempted = 100, .accepted = 95, .runs = 10}).matched));
    EXPECT_EQ((resolveMTPLearnedStartup<0, ScopeOrderingTables>(partial, 1, 15)), -1);
    auto wrong_class = base;
    wrong_class.model_class = MTPDepthPolicyModelClass::Dense;
    EXPECT_FALSE(wrong_class.valid());
    auto malformed_config = dynamicConfig(3, 15);
    malformed_config.scope = partial.scope;
    EXPECT_THROW(MTPDepthController(malformed_config, 3, MTPVerifyMode::Greedy), std::invalid_argument);
}

TEST(Test__MTPDepthController, FrozenScopeUsesActualUniformTopologyModelAndHardware)
{
    auto plan = scopedPlan();
    auto inventory = scopedInventory();
    const auto model = scopedModel();
    auto projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
    ASSERT_TRUE(projection.scope.specified());
    auto expected = exactTP4Scope;
    expected.verifier_width_policy = MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
    EXPECT_EQ(projection.scope, expected); // Strategy is bound from the actual active execution policy separately.
    plan.local_tp_devices.clear();
    plan.local_pp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(2)};
    plan.local_pp_layer_boundaries = {0, 20, 40};
    plan.local_pp_stage_tp_info.resize(2);
    plan.local_pp_stage_tp_info[0].devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    plan.local_pp_stage_tp_info[1].devices = {GlobalDeviceAddress::rocm(2), GlobalDeviceAddress::rocm(3)};
    projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
    EXPECT_EQ(projection.scope.tp_width, 2);
    EXPECT_EQ(projection.scope.pp_stages, 2);
    plan.local_pp_stage_tp_info[1].devices.pop_back();
    projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
    EXPECT_FALSE(projection.scope.specified());
    EXPECT_STREQ(projection.generic_reason, "nonuniform_tp_pp");
    plan = scopedPlan();
    plan.prev_rank = 1;
    projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
    EXPECT_FALSE(projection.scope.specified());
    EXPECT_STREQ(projection.generic_reason, "remote_pp");
    plan = scopedPlan();
    inventory.ranks[0].gpus[3].compute_capability_major = 94;
    projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
    EXPECT_FALSE(projection.scope.specified());
    EXPECT_STREQ(projection.generic_reason, "missing_or_nonuniform_hardware_architecture");
    inventory = scopedInventory();
    inventory.ranks[0].gpus[2].compute_capability_major = 0;
    EXPECT_FALSE(projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory).scope.specified());
}

TEST(Test__MTPDepthController, FrozenScopeRejectsPartialMainCoverage)
{
    const auto model = scopedModel();
    const auto inventory = scopedInventory();
    const auto complete = scopedPlan();
    auto reject = [&](const RankExecutionPlan &plan) {
        const auto projection = projectMTPDepthPolicyScopeGeometry(plan, model, 15, inventory);
        EXPECT_FALSE(projection.scope.specified());
        EXPECT_STREQ(projection.generic_reason, "incomplete_main_model_coverage");
    };
    auto partial = complete;
    partial.first_layer = 1;
    reject(partial);
    partial = complete;
    partial.last_layer = model.mainLayerCount() - 2;
    reject(partial);
    partial.last_layer = model.mainLayerCount(); // Including a predictor is not main-only whole-model work.
    reject(partial);
    partial = complete;
    partial.has_embedding = false;
    reject(partial);
    partial = complete;
    partial.has_lm_head = false;
    reject(partial);
    // An explicit full interval and the canonical all-layers sentinel agree.
    auto explicit_full = complete;
    explicit_full.last_layer = model.mainLayerCount() - 1;
    EXPECT_EQ(projectMTPDepthPolicyScopeGeometry(explicit_full, model, 15, inventory).scope,
              projectMTPDepthPolicyScopeGeometry(complete, model, 15, inventory).scope);
    auto pipeline = complete;
    pipeline.local_tp_devices.clear();
    pipeline.local_pp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    pipeline.local_pp_layer_boundaries = {0, 20, 40};
    EXPECT_TRUE(projectMTPDepthPolicyScopeGeometry(pipeline, model, 15, inventory).scope.specified());
    for (const std::vector<int> boundaries : {std::vector<int>{}, {0, 20, 39}, {1, 20, 40},
                                            {0, 20, 20, 40}, {0, 30, 20, 40}, {0, 40, 40}})
    {
        pipeline.local_pp_layer_boundaries = boundaries;
        reject(pipeline);
    }
}

/** @brief One validation result preserves scoped hits, domain misses and fatal identity distinction. */
TEST(Test__MTPDepthController, LearnedDecisionValidatesOnceWithoutChangingScopedArithmetic)
{
    auto context = scopedContext();
    const MTPLearnedDepthWindow high{100, 95, 10, 0, 5};
    const auto selected = evaluateMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, high);
    ASSERT_TRUE(selected.valid);
    EXPECT_TRUE(selected.match.matched);
    EXPECT_EQ(selected.match.depth_delta, 2);
    const auto ordinary = matchMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, high);
    EXPECT_EQ(selected.match.matched, ordinary.matched);
    EXPECT_EQ(selected.match.depth_delta, ordinary.depth_delta);
    const auto miss = evaluateMTPLearnedDepthPolicy<0, ScopeOrderingTables>(
        context, 3, {.attempted = 100, .accepted = 80, .runs = 10, .full_accepts = 5});
    EXPECT_TRUE(miss.valid);
    EXPECT_FALSE(miss.match.matched); // A measured interval miss cannot enter the generic row.
    context.enabled = false;
    const auto disabled = evaluateMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, high);
    EXPECT_TRUE(disabled.valid);
    EXPECT_FALSE(disabled.match.matched);
    context.scope.tp_width = 0; // Partial non-generic identity remains invalid even when disabled.
    const auto invalid = evaluateMTPLearnedDepthPolicy<0, ScopeOrderingTables>(context, 3, high);
    EXPECT_FALSE(invalid.valid);
    EXPECT_FALSE(invalid.match.matched);
    EXPECT_EQ((mtpLearnedStartupAt<0, MTPInstalledDepthPolicyTables>().initial_depth),
        kMTPGeneratedDepthPolicyStartups[0].initial_depth);
    EXPECT_EQ((mtpLearnedRuleAt<0, MTPInstalledDepthPolicyTables>().depth_delta),
        kMTPGeneratedDepthPolicyRules[0].depth_delta);
}

/**
 * @test The installed QFN greedy overlay owns startup and integer-window decisions.
 *
 * This is the measured policy's complete frozen identity, rather than a fake
 * model or an alternate rule table. Installing only a generic policy must fail
 * this regression even if a synthetic table proves the interpreter separately.
 * Device integration and real-model Dynamic campaigns own execution/economy.
 */
TEST(Test__MTPDepthController, InstalledQFNGreedyExactRowsOwnsStartupAndIntegerWindows)
{
    const MTPDepthPolicyScope scope{
        MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,
        4, 1, 15, 2560, 48, 512, 10, 90, 6};
    const MTPLearnedDepthPolicyContext context{
        .enabled = true, .backend = MTPDepthPolicyBackend::ROCm,
        .model_class = MTPDepthPolicyModelClass::MoE,
        .verify_mode = MTPVerifyMode::Greedy, .scope = scope};
    ASSERT_TRUE(context.valid());
    ASSERT_TRUE(hasMTPMeasuredScope(context));
    EXPECT_EQ(resolveMTPLearnedStartup(context, 1, 15), 5);

    // All admitted depths have the same two measured economic targets. The
    // actual acceptance partition remains owned by the installed generated rows.
    for (int depth = 1; depth <= 15; ++depth)
    {
        SCOPED_TRACE(depth);
        int rows = 0;
        bool reaches_four = false, reaches_five = false;
        for (const auto &rule : kMTPGeneratedDepthPolicyRules)
        {
            if (rule.scope != scope || rule.verify_mode != MTPVerifyMode::Greedy ||
                rule.backend != MTPDepthPolicyBackend::ROCm ||
                rule.model_class != MTPDepthPolicyModelClass::MoE || rule.current_depth != depth)
                continue;
            ++rows;
            reaches_four |= depth + rule.depth_delta == 4;
            reaches_five |= depth + rule.depth_delta == 5;
        }
        EXPECT_EQ(rows, 2);
        EXPECT_TRUE(reaches_four);
        EXPECT_TRUE(reaches_five);
    }

    auto config = dynamicConfig(0, 15, /*window_size=*/16);
    config.use_generated_policy = true;
    config.backend = context.backend;
    config.model_class = context.model_class;
    config.scope = scope;
    MTPDepthController automatic(config, 15, MTPVerifyMode::Greedy);
    EXPECT_EQ(automatic.currentDepth(), 5);
    EXPECT_EQ(automatic.maxDepth(), 15); // Learned startup never shrinks admission.
    for (const int initial : {1, 3, 7, 15})
    {
        auto authored = config;
        authored.initial_depth = initial;
        MTPDepthController explicit_initial(authored, 15, MTPVerifyMode::Greedy);
        EXPECT_EQ(explicit_initial.currentDepth(), initial);
    }

    // Startup selection fixes the whole measured domain before request bounds.
    // Excluding five must not reactivate the preserved generic startup at three.
    EXPECT_EQ(resolveMTPLearnedStartup(context, 1, 4), -1);
    auto bounded = config;
    bounded.max_depth = 4;
    MTPDepthController bounded_initial(bounded, 15, MTPVerifyMode::Greedy);
    EXPECT_EQ(bounded_initial.currentDepth(), 2); // Ordinary bounded initialization.
    EXPECT_TRUE(hasMTPMeasuredScope(context));

    // The installed table remains an independently testable startup/interpreter
    // artifact. Measured runtime moves now belong to the shared cost selector.
    struct WindowCase { int depth; int accepted; int target; };
    for(const auto cell:{WindowCase{3,30,4},WindowCase{3,42,5},WindowCase{4,40,4},
        WindowCase{4,56,5},WindowCase{5,40,4},WindowCase{5,56,5}})
    {
        const auto match=matchMTPLearnedDepthPolicy(context,cell.depth,
            {.attempted=static_cast<uint64_t>(cell.depth*16),.accepted=static_cast<uint64_t>(cell.accepted),
             .runs=16,.zero_accepts=0,
             .full_accepts=static_cast<uint64_t>(std::max(cell.target>cell.depth?1:0,cell.accepted-16*(cell.depth-1)))});
        ASSERT_TRUE(match.matched);EXPECT_EQ(cell.depth+match.depth_delta,cell.target);
    }
    EXPECT_FALSE(matchMTPLearnedDepthPolicy(context,4,
        {.attempted=64,.accepted=0,.runs=16,.zero_accepts=16}).matched);

    auto neighbor = context;
    neighbor.scope.tp_width = 2;
    ASSERT_TRUE(neighbor.valid());
    EXPECT_FALSE(hasMTPMeasuredScope(neighbor));
    EXPECT_EQ(resolveMTPLearnedStartup(neighbor, 1, 15), 3);
    auto neighbor_config = config;
    neighbor_config.scope = neighbor.scope;
    MTPDepthController unmeasured(neighbor_config, 15, MTPVerifyMode::Greedy);
    EXPECT_EQ(unmeasured.currentDepth(), 3);
}

/**
 * @test The exact diagnostic table supplies an independent stochastic startup.
 * This tests installed arithmetic, not promotion: the frozen table's offline
 * economic gate remains failed and its local trial is explicitly diagnostic.
 */
TEST(Test__MTPDepthController, InstalledQFNStochasticExactRowsOwnsFrozenStartup)
{
    const MTPDepthPolicyScope scope{
        MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,
        4, 1, 15, 2560, 48, 512, 10, 90, 6};
    const MTPLearnedDepthPolicyContext context{
        .enabled = true, .backend = MTPDepthPolicyBackend::ROCm,
        .model_class = MTPDepthPolicyModelClass::MoE,
        .verify_mode = MTPVerifyMode::SpeculativeSampling, .scope = scope};
    ASSERT_TRUE(context.valid());
    ASSERT_TRUE(hasMTPMeasuredScope(context));
    EXPECT_EQ(resolveMTPLearnedStartup(context, 1, 15), 5);
    auto config = dynamicConfig(0, 15, 16);
    config.use_generated_policy = true;
    config.backend = context.backend;
    config.model_class = context.model_class;
    config.scope = scope;
    MTPDepthController automatic(config, 15, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(automatic.currentDepth(), 5);
    EXPECT_EQ(automatic.maxDepth(), 15);
    for (const int initial : {1, 3, 7, 15})
    {
        auto authored = config;
        authored.initial_depth = initial;
        MTPDepthController explicit_initial(authored, 15, MTPVerifyMode::SpeculativeSampling);
        EXPECT_EQ(explicit_initial.currentDepth(), initial);
    }
    EXPECT_EQ(resolveMTPLearnedStartup(context, 1, 4), -1);
    auto bounded = config;
    bounded.max_depth = 4;
    MTPDepthController bounded_initial(bounded, 15, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(bounded_initial.currentDepth(), 1); // Portable stochastic initialization, never generic learned three.
    auto neighbor = context;
    neighbor.scope.tp_width = 2;
    ASSERT_TRUE(neighbor.valid());
    EXPECT_FALSE(hasMTPMeasuredScope(neighbor));
    EXPECT_EQ(resolveMTPLearnedStartup(neighbor, 1, 15), 3);
}

/**
 * @test Installed six-decimal shared boundaries use exact integer first-match semantics.
 * Large but valid integer windows represent each PPM edge without floating-rate
 * rounding or inference work. They test both frozen sampling-law partitions.
 */
TEST(Test__MTPDepthController, InstalledQFNScopedPolicyKeepsSerializedPPMBoundaries)
{
    const MTPDepthPolicyScope scope{
        MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,
        4, 1, 15, 2560, 48, 512, 10, 90, 6};
    constexpr uint64_t scale = 1'000'000;
    for (const auto verify : {MTPVerifyMode::Greedy, MTPVerifyMode::SpeculativeSampling})
    {
        const MTPLearnedDepthPolicyContext context{
            .enabled = true, .backend = MTPDepthPolicyBackend::ROCm,
            .model_class = MTPDepthPolicyModelClass::MoE, .verify_mode = verify, .scope = scope};
        ASSERT_TRUE(hasMTPMeasuredScope(context));
        for (int depth = 1; depth <= 15; ++depth)
        {
            SCOPED_TRACE(::testing::Message() << "law=" << static_cast<int>(verify) << " depth=" << depth);
            std::vector<const MTPGeneratedDepthPolicyRule *> rows;
            for (const auto &row : kMTPGeneratedDepthPolicyRules)
                if (row.verify_mode == verify && row.backend == context.backend &&
                    row.model_class == context.model_class && row.scope == scope && row.current_depth == depth)
                    rows.push_back(&row);
            ASSERT_GE(rows.size(), 2u);
            // Construct valid prefix-count summaries: partial prefixes can
            // realize every remaining integer sum between one and depth−one.
            const auto window = [depth](uint64_t rate_ppm) {
                const uint64_t accepted = static_cast<uint64_t>(depth) * rate_ppm;
                const uint64_t partial_capacity = static_cast<uint64_t>(depth - 1) * scale;
                const uint64_t forced_full = accepted > partial_capacity ? accepted - partial_capacity : 0;
                const uint64_t full = forced_full > 0 ? forced_full : accepted >= static_cast<uint64_t>(depth) ? 1 : 0;
                const uint64_t remaining_runs = scale - full;
                const uint64_t remaining_accepted = accepted - full * static_cast<uint64_t>(depth);
                return MTPLearnedDepthWindow{
                    .attempted = static_cast<uint64_t>(depth) * scale,
                    .accepted = accepted, .runs = scale,
                    .zero_accepts = remaining_accepted < remaining_runs ? remaining_runs - remaining_accepted : 0,
                    .full_accepts = full};
            };
            for (std::size_t index = 1; index < rows.size(); ++index)
            {
                const auto &lower = *rows[index - 1];
                const auto &upper = *rows[index];
                const auto boundary = mtpLearnedRatePPM(lower.max_acceptance_rate);
                ASSERT_EQ(boundary, mtpLearnedRatePPM(upper.min_acceptance_rate));
                ASSERT_LT(boundary, scale);
                EXPECT_DOUBLE_EQ(lower.max_acceptance_rate, static_cast<double>(boundary) / scale);
                const auto on_edge = matchMTPLearnedDepthPolicy(context, depth, window(boundary));
                ASSERT_TRUE(on_edge.matched);
                EXPECT_EQ(on_edge.depth_delta, lower.depth_delta); // Earlier inclusive row owns the edge.
                const auto above_edge = matchMTPLearnedDepthPolicy(context, depth, window(boundary + 1));
                ASSERT_TRUE(above_edge.matched);
                EXPECT_EQ(above_edge.depth_delta, upper.depth_delta); // No PPM gap after the shared edge.
            }
        }
    }
}

/** @test A learned move needs one full prefix; partial means cannot invent extension evidence. */
TEST(Test__MTPDepthController, LearnedPromotionsRequireFullPrefixEvidence)
{
    const MTPDepthPolicyScope scope{
        MTPVerifierPhysicalWidthPolicy::ExactLogicalRows, 4, 1, 15, 2560, 48, 512, 10, 90, 6};
    const MTPLearnedDepthPolicyContext context{
        .enabled = true, .backend = MTPDepthPolicyBackend::ROCm,
        .model_class = MTPDepthPolicyModelClass::MoE, .verify_mode = MTPVerifyMode::Greedy, .scope = scope};
    for (const int depth : {3, 4})
    for (const uint64_t runs : {uint64_t{4}, uint64_t{1024}})
    {
        // The same accepted total can contain zero full prefixes or one full
        // prefix with redistributed partials, independently of window length.
        const uint64_t accepted = runs * static_cast<uint64_t>(depth - 1);
        auto window = MTPLearnedDepthWindow{
            .attempted = runs * static_cast<uint64_t>(depth), .accepted = accepted, .runs = runs};
        EXPECT_FALSE(matchMTPLearnedDepthPolicy(context, depth, window).matched);
        window.full_accepts = 1;
        const auto positive = matchMTPLearnedDepthPolicy(context, depth, window);
        ASSERT_TRUE(positive.matched);
        EXPECT_EQ(positive.depth_delta, 1);
    }
    const auto hold = matchMTPLearnedDepthPolicy(context, 4, {.attempted = 16, .accepted = 10, .runs = 4});
    ASSERT_TRUE(hold.matched);
    EXPECT_EQ(hold.depth_delta, 0);
    const auto demotion = matchMTPLearnedDepthPolicy(context, 5, {.attempted = 20, .accepted = 8, .runs = 4});
    ASSERT_TRUE(demotion.matched);
    EXPECT_EQ(demotion.depth_delta, -1);

    for (const int prefix : {2, 3})
    {
        MTPRuntimeConfig config;
        config.enabled = true;
        config.draft_tokens = config.graph_capacity_draft_tokens = 15;
        config.verify_mode = MTPVerifyMode::Greedy;
        config.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        config.depth_policy.backend = context.backend;
        config.depth_policy.model_class = context.model_class;
        config.depth_policy.scope = scope;
        config.depth_policy.initial_depth = 5;
        const auto policy = resolveMTPDeviceGenerationDepthPolicy(config);
        ASSERT_EQ(policy.window_size, 16);
        ASSERT_EQ(policy.minimum_samples, 4);
        ASSERT_EQ(policy.cooldown_steps, 8);
        ASSERT_EQ(policy.promote_consecutive_windows, 3);
        std::vector<int> control(sampling_math::kDeviceGenerationControlCount);
        ASSERT_TRUE(sampling_math::initialize_device_generation_control(4096, 4096, policy, control.data()));
        bool incoming = false;
        const int target = prefix == 2 ? 2 : 4;
        for (int window = 1; window <= policy.promote_consecutive_windows; ++window)
        {
            for (int step = 0; step < policy.minimum_samples; ++step)
            {
                const int depth = control[sampling_math::kDeviceGenerationControlCurrentDraftDepth];
                ASSERT_EQ(depth, 5);
                const auto actual = committedObservation(depth, prefix, incoming);
                control[sampling_math::kDeviceGenerationControlLastTransactionEmittedTokenCount] =
                    static_cast<int>(*actual.newly_emitted_tokens);
                ASSERT_TRUE(sampling_math::record_device_generation_depth_observation(
                    prefix, prefix < depth, false, control.data(), *actual.publication));
            }
            // Partial prefixes still rank the same conservative lower targets,
            // but only three disjoint confirmations may start the actual trial.
            const bool trial_started = window == policy.promote_consecutive_windows;
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlCurrentDraftDepth], trial_started ? target : 5);
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthLastRecommendedDepth], target);
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthPromotionStreak], trial_started ? 0 : window);
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthEvaluatedWindows], window);
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthUtilityProbes], trial_started ? 1 : 0);
            EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthPromotions], 0);
        }
        // Every observed prefix is partial, so deeper proposals have no extension
        // evidence. Rank conservative lower bounds: R4/A8 selects 2, while
        // R4/A12 selects 4 because the bound for depth 3 is not exact. The trial
        // reference owns only the last confirming window, never all twelve rows.
        EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthUtilityReferenceDepth], 5);
        EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthUtilityReferenceRuns], policy.minimum_samples);
        EXPECT_EQ(control[sampling_math::kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress],
            policy.minimum_samples * (prefix + 1));
    }
}


/** @test Complete cost-domain binding and scalar tables preserve every measured depth/law. */
TEST(Test__MTPDepthController, MeasuredTransactionCostsRequireCompleteScopeAndLaw)
{
    MTPLearnedDepthPolicyContext context{
        .enabled=true,.backend=MTPDepthPolicyBackend::ROCm,.model_class=MTPDepthPolicyModelClass::MoE,
        .verify_mode=MTPVerifyMode::Greedy,
        .scope={MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,4,1,15,2560,48,512,10,90,6}};
    const std::array<uint32_t,15> greedy{33853,38657,44485,50678,54652,59379,64089,72849,76018,81652,85536,90843,95966,101074,104359};
    const std::array<uint32_t,15> stochastic{34022,38758,44734,50902,54971,59622,64427,73224,76427,82064,86158,91538,96474,101630,104715};
    for (auto law : {MTPVerifyMode::Greedy,MTPVerifyMode::SpeculativeSampling})
    {
        context.verify_mode=law;const auto profile=resolveMTPDepthCostProfile(context);
        ASSERT_TRUE(profile.active());EXPECT_EQ(profile.costAt(0),0u);EXPECT_EQ(profile.costAt(16),0u);
        for(uint32_t d=1;d<=15;++d)EXPECT_EQ(profile.costAt(d),(law==MTPVerifyMode::Greedy?greedy:stochastic)[d-1]);
    }
    auto neighbor=context;neighbor.scope.tp_width=2;EXPECT_FALSE(resolveMTPDepthCostProfile(neighbor).active());
    neighbor=context;neighbor.scope.hidden_dim=2048;EXPECT_FALSE(resolveMTPDepthCostProfile(neighbor).active());
    neighbor=context;neighbor.scope.compute_capability_minor=0;EXPECT_FALSE(resolveMTPDepthCostProfile(neighbor).active());
    neighbor=context;neighbor.backend=MTPDepthPolicyBackend::CUDA;EXPECT_FALSE(resolveMTPDepthCostProfile(neighbor).active());
    neighbor=context;neighbor.enabled=false;EXPECT_FALSE(resolveMTPDepthCostProfile(neighbor).active());
}

/** @test Supported and unresolved analytical actions both execute actual host trials. */
TEST(Test__MTPDepthController, SurvivalActualPublicationSelectsBothDirections)
{
    struct Case{int depth,prefix,target;};
    for(const auto c:{Case{5,0,1},Case{5,3,4},Case{5,5,6},Case{4,4,5},Case{4,1,1}})
    {
        MTPDepthController controller(survivalConfig(c.depth),15);bool incoming=false;MTPDepthDecision decision;
        uint64_t actual_y=0;
        for(int i=0;i<4;++i)
        {
            auto actual=committedObservation(c.depth,c.prefix,incoming);actual_y+=*actual.newly_emitted_tokens;
            decision=controller.recordStep(actual);if(i<3)EXPECT_FALSE(decision.evaluated);
        }
        ASSERT_TRUE(decision.evaluated);EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeStarted);
        EXPECT_EQ(controller.currentDepth(),c.target);EXPECT_EQ(controller.stats().utility_probes,1u);
        const auto &state=controller.utilityState();EXPECT_EQ(state.phase,MTPReversibleProposalPhase::ProbePending);
        EXPECT_EQ(state.reference_depth,static_cast<uint32_t>(c.depth));EXPECT_EQ(state.reference_runs,4u);
        EXPECT_EQ(state.reference_verified_progress,static_cast<uint32_t>(4*(c.prefix+1)));
        EXPECT_EQ(state.window_emitted,0u);
        if(c.prefix<c.depth)EXPECT_EQ(actual_y,state.reference_verified_progress+1u);
        else EXPECT_EQ(actual_y,state.reference_verified_progress);
    }
}

/** @test Pending compares genuine published V, pauses across excluded work, and retains one cache. */
TEST(Test__MTPDepthController, SurvivalActualHostReturnCacheAndReset)
{
    auto config=survivalConfig(4,3,4);MTPDepthController controller(config,15);bool incoming=false;
    EXPECT_THROW(controller.recordStep(observation(4,2)),std::invalid_argument);
    EXPECT_EQ(controller.utilityState().window_emitted,0u);EXPECT_EQ(controller.stats().windows,0u);
    MTPDepthDecision decision;
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(4,2,incoming));
    ASSERT_EQ(controller.currentDepth(),3);EXPECT_EQ(controller.utilityState().reference_verified_progress,12u);
    ASSERT_EQ(controller.stats().utility_probes,1u);
    decision=controller.recordStep(committedObservation(3,0,incoming));EXPECT_FALSE(decision.evaluated);
    const auto prior=controller.utilityState();
    const auto clipped=committedObservation(3,0,incoming,1);
    EXPECT_EQ(controller.recordStep(clipped).reason,MTPDepthDecisionReason::BudgetLimited);
    EXPECT_EQ(controller.utilityState().window_emitted,prior.window_emitted);
    EXPECT_EQ(controller.utilityState().reference_verified_progress,prior.reference_verified_progress);
    for(int i=0;i<3;++i)decision=controller.recordStep(committedObservation(3,0,incoming));
    ASSERT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeReturned);ASSERT_EQ(controller.currentDepth(),4);
    EXPECT_EQ(controller.stats().utility_returns,1u);EXPECT_EQ(controller.utilityState().reference_depth,3u);
    EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);
    // The same failed target is filtered before ranking. With no other admitted
    // candidate, a full-window hold must preserve its reference and consume Y.
    for(int i=0;i<16;++i)decision=controller.recordStep(committedObservation(4,2,incoming));
    EXPECT_EQ(decision.reason,MTPDepthDecisionReason::SurvivalHold);EXPECT_EQ(controller.currentDepth(),4);
    EXPECT_EQ(controller.stats().utility_probes,1u);EXPECT_EQ(controller.utilityState().reference_depth,3u);
    EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);EXPECT_EQ(controller.utilityState().window_emitted,0u);
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(4,0,incoming));
    EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeStarted);EXPECT_EQ(controller.currentDepth(),3);
    EXPECT_EQ(controller.stats().utility_probes,2u);EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);
    controller.reset();EXPECT_EQ(controller.currentDepth(),4);EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::Clear);
    EXPECT_EQ(controller.utilityState().window_emitted,0u);EXPECT_EQ(controller.stats().utility_probes,0u);
    auto zero=config;zero.min_depth=0;EXPECT_THROW(MTPDepthController(zero,15),std::invalid_argument);
}

/** @test Pending keeps actual target V before cooldown, without applying another proposal. */
TEST(Test__MTPDepthController, SurvivalPendingKeepsActualPublicationBeforeCooldown)
{
    MTPDepthController controller(survivalConfig(),15);bool incoming=false;MTPDepthDecision decision;
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(5,3,incoming));
    ASSERT_EQ(controller.currentDepth(),4);ASSERT_EQ(controller.utilityState().reference_verified_progress,16u);
    uint32_t target_y=0;
    for(int i=0;i<4;++i)
    {
        auto actual=committedObservation(4,4,incoming);target_y+=*actual.newly_emitted_tokens;
        decision=controller.recordStep(actual);if(i<3)EXPECT_FALSE(decision.evaluated);
    }
    ASSERT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeKept);EXPECT_EQ(controller.currentDepth(),4);
    EXPECT_EQ(target_y,19u);EXPECT_EQ(controller.stats().utility_probes,1u);EXPECT_EQ(controller.stats().utility_keeps,1u);
    EXPECT_EQ(controller.stats().utility_returns,0u);EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::Clear);
    EXPECT_EQ(controller.utilityState().window_emitted,0u);
}

/** @test A real upward trial returns; ranking skips its cached target rather than forcing a retest. */
TEST(Test__MTPDepthController, SurvivalUpwardTrialReturnsAndCacheFiltersBeforeRanking)
{
    for(int maximum:{6,7})
    {
        MTPDepthController controller(survivalConfig(5,1,maximum),15);bool incoming=false;MTPDepthDecision decision;
        for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(5,5,incoming));
        ASSERT_EQ(controller.currentDepth(),6);ASSERT_EQ(controller.stats().utility_probes,1u);
        EXPECT_EQ(controller.utilityState().reference_verified_progress,24u);
        for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(6,0,incoming));
        ASSERT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeReturned);ASSERT_EQ(controller.currentDepth(),5);
        EXPECT_EQ(controller.utilityState().reference_depth,6u);EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);
        for(int i=0;i<(maximum==6?16:8);++i)decision=controller.recordStep(committedObservation(5,5,incoming));
        if(maximum==6)
        {
            EXPECT_EQ(decision.reason,MTPDepthDecisionReason::SurvivalHold);EXPECT_EQ(controller.currentDepth(),5);
            EXPECT_EQ(controller.stats().utility_probes,1u);EXPECT_EQ(controller.utilityState().reference_depth,6u);
        }
        else
        {
            EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeStarted);EXPECT_EQ(controller.currentDepth(),7);
            EXPECT_EQ(controller.stats().utility_probes,2u);EXPECT_EQ(controller.utilityState().reference_depth,5u);
            EXPECT_EQ(controller.utilityState().reference_verified_progress,48u);
        }
        EXPECT_EQ(controller.stats().utility_returns,1u);
    }
}

/** @test Unresolved shallower evidence is a trial even when exact adjacent movement loses. */
TEST(Test__MTPDepthController, SurvivalUnresolvedLowerCanCrossNonconvexCostGap)
{
    auto config=survivalConfig();config.min_samples=10; // Authored evidence geometry, not a production tuning choice.
    MTPDepthController controller(config,15);bool incoming=false;MTPDepthDecision decision;
    for(int prefix:{1,1,1,2,2,2,2,5,5,5})decision=controller.recordStep(committedObservation(5,prefix,incoming));
    ASSERT_TRUE(decision.evaluated);EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeStarted);
    EXPECT_EQ(decision.window.verifier_runs,10u);EXPECT_EQ(decision.window.accepted_draft_tokens,26u);
    EXPECT_EQ(controller.currentDepth(),3);EXPECT_EQ(controller.utilityState().reference_verified_progress,36u);
    EXPECT_EQ(controller.stats().utility_probes,1u);EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::ProbePending);
}

/** @test Stop and clipped frames preserve physical publication but cannot complete a trial. */
TEST(Test__MTPDepthController, SurvivalStopAndClipExcludedAndPublicationMismatchRejects)
{
    MTPDepthController controller(survivalConfig(),15);bool incoming=false;
    const auto stopped=committedObservation(5,0,incoming,0,true);
    EXPECT_EQ(controller.recordStep(stopped).reason,MTPDepthDecisionReason::ModelStopped);
    EXPECT_EQ(controller.stats().windows,0u);EXPECT_EQ(controller.utilityState().window_emitted,0u);
    controller.reset();incoming=false;const auto clipped=committedObservation(5,0,incoming,1);
    EXPECT_EQ(controller.recordStep(clipped).reason,MTPDepthDecisionReason::BudgetLimited);
    EXPECT_EQ(controller.utilityState().window_emitted,0u);
    auto bad=committedObservation(5,0,incoming);++*bad.newly_emitted_tokens;
    EXPECT_THROW(controller.recordStep(bad),std::invalid_argument);
    EXPECT_EQ(controller.utilityState().window_emitted,0u);
    // A coherent one-row debt witness is insufficient if actual eligible V
    // disagrees with the accepted-prefix identity.
    MTPPublicationWitness wrong;
    ASSERT_TRUE(MTPPublicationWitness::fromCounts(2,0,2,false,&wrong));
    bad=observation(5,0);bad.newly_emitted_tokens=2;bad.publication=wrong;
    EXPECT_THROW(controller.recordStep(bad),std::invalid_argument);EXPECT_EQ(controller.stats().windows,0u);
}

/** @test A neighboring unmeasured scope keeps the distinct actual CPU heuristic. */
TEST(Test__MTPDepthController, SurvivalUnmeasuredGenericUpwardKeepsLegacyCPUParent)
{
    auto config=dynamicConfig(4,15,16,8);config.min_samples=4;config.use_generated_policy=false;
    config.demote_zero_accept_rate=.45;config.demote_acceptance_rate=.55;
    MTPDepthController controller(config,15);MTPDepthDecision decision;
    ASSERT_FALSE(controller.costProfile().active());
    for(int i=0;i<4;++i)decision=controller.recordStep(observation(4,1));
    EXPECT_EQ(decision.reason,MTPDepthDecisionReason::ProbeHigherBeforeDemote);
    EXPECT_EQ(controller.currentDepth(),5);EXPECT_EQ(controller.stats().utility_probes,0u);
}

/** @test Ordinary execution cannot publish any reversible state or reporting word. */
TEST(Test__MTPDepthController, UtilityTerminalContractRejectsInactivePoisonedWords)
{
    using namespace sampling_math;
    std::array<int,kDeviceGenerationControlCount> row{};std::array<int32_t,1> response{};
    const DeviceGenerationAdmissionRequest admission{.request_count=1,.max_new_tokens=1,.depth_policy=DeviceGenerationPolicy::ordinary()};
    ASSERT_TRUE(initialize_device_generation_control(1,1,admission.depth_policy,row.data()));
    ASSERT_TRUE(append_ordinary_sample_to_device_generation(7,false,OrdinaryGenerationSampleSource::PrefillLogits,response.data(),1,row.data()));
    ASSERT_EQ(validateDeviceGenerationTerminal(row,admission,1,0,0),DeviceGenerationTerminalError::None);
    for(int i=kDeviceGenerationControlDepthUtilityPhase;i<=kDeviceGenerationControlDepthUtilityReturns;++i)
    for(int poison : {-1,1})
    {
        auto corrupt=row;corrupt[i]=poison;
        EXPECT_EQ(validateDeviceGenerationTerminal(corrupt,admission,1,0,0),DeviceGenerationTerminalError::InvalidDepthStatistics)<<i;
    }
}

/**
 * @test A real completed continuation may return a probe started in its predecessor.
 * Portable initialize/budget/summary/commit/publication helpers produce every
 * completion bit and counter. No controller row or terminal flag is fabricated.
 */
TEST(Test__MTPDepthController, UtilityTerminalContractAllowsPriorResponseProbeReturn)
{
    using namespace sampling_math;
    MTPRuntimeConfig config;
    config.enabled=true;config.draft_tokens=config.graph_capacity_draft_tokens=15;
    config.depth_policy.mode=MTPDepthPolicyMode::Dynamic;
    config.depth_policy.backend=MTPDepthPolicyBackend::ROCm;
    config.depth_policy.model_class=MTPDepthPolicyModelClass::MoE;
    config.depth_policy.initial_depth=4;config.depth_policy.min_depth=3;config.depth_policy.max_depth=4;
    config.depth_policy.min_samples=16;config.depth_policy.window_size=16;
    // Isolate continuation of an already started trial; threshold-three
    // confirmation and deferred terminal ownership have separate fixtures.
    config.depth_policy.promote_consecutive_windows=1;
    config.depth_policy.scope={MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,4,1,15,2560,48,512,10,90,6};
    const auto policy=resolveMTPDeviceGenerationDepthPolicy(config);
    std::array<int,kDeviceGenerationControlCount> row{};
    std::array<int32_t,64> response{};
    DeviceGenerationDispatchTicket ticket{};
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(7,9,&ticket));
    ASSERT_TRUE(initialize_device_generation_control(51,response.size(),policy,row.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse,
        {.kind=DeviceGenerationAdmissionKind::NewRequest,.session_epoch=7,.workspace_generation=9}));
    int condition=100,base=23;
    auto commit=[&](int accepted,int maintenance)
    {
        const int depth=row[kDeviceGenerationControlCurrentDraftDepth];
        const int leading=row[kDeviceGenerationControlNextLeadingCommittedOutputCount];
        std::array<int,16> tokens{},flags{},compact{};
        std::array<int,kSpeculativeBatchMetaCount> meta{};
        std::array<int,7> publication{};
        for(int i=0;i<16;++i){tokens[i]=1000+i;flags[i]=i<accepted?1:0;}
        summarize_speculative_verify_batch_at_commit_boundary(condition,tokens.data(),flags.data(),depth,
            nullptr,0,2000,1,maintenance,compact.data(),compact.size(),meta.data(),nullptr,leading);
        ASSERT_EQ(meta[kSpecBatchMetaOk],1);
        ASSERT_EQ(prepare_device_generation_transaction_budget(depth+1,maintenance,row.data()),maintenance);
        ASSERT_TRUE(commit_device_generation_and_derive_speculative_publication_metadata(
            compact.data(),compact.size(),meta.data(),meta.size(),0,depth+1,base,response.data(),response.size(),row.data(),
            publication.data(),publication.data()+1,publication.data()+2,publication.data()+3,
            publication.data()+4,publication.data()+5,publication.data()+6));
        condition=publication[4];base=publication[1];
        ASSERT_TRUE(publish_device_generation_dispatch_ticket(row.data(),nullptr,&ticket));
    };
    for(int i=0;i<16;++i)ASSERT_NO_FATAL_FAILURE(commit(2,5)); // Actual R16/A32 gives incumbent V48 and actual Y49.
    ASSERT_EQ(row[kDeviceGenerationControlCurrentDraftDepth],3);
    ASSERT_EQ(row[kDeviceGenerationControlLearnedDepthMatchedWindows],0);
    ASSERT_EQ(row[kDeviceGenerationControlDepthUtilityProbes],1);
    // The sixteenth eligible depth4 transaction needs its full five-row budget:
    // response46 + physical budget5 = 51. It emits3, leaving2 for two actual
    // maintenance-clipped commits; neither may change the Pending sample.
    ASSERT_NO_FATAL_FAILURE(commit(0,1));
    ASSERT_EQ(row[kDeviceGenerationControlRemainingTokenCount],1);
    ASSERT_EQ(row[kDeviceGenerationControlDepthUtilityPhase],1);
    ASSERT_EQ(row[kDeviceGenerationControlDepthWindowVerifierRuns],0);
    ASSERT_NO_FATAL_FAILURE(commit(0,1)); // Actual second clipped1 closes response51.
    ASSERT_EQ(row[kDeviceGenerationControlRequestComplete],1);
    ASSERT_EQ(row[kDeviceGenerationControlDepthUtilityPhase],1);
    const auto leading=row[kDeviceGenerationControlNextLeadingCommittedOutputCount]!=0
        ? DeviceGenerationLeadingRowDisposition::AlreadyEmitted : DeviceGenerationLeadingRowDisposition::PendingResponse;
    const DeviceGenerationAdmissionRequest admission{
        .request_count=1,.max_new_tokens=19,.kind=DeviceGenerationAdmissionKind::ContinueResponse,
        .depth_policy=policy,.initial_leading_row_disposition=leading};
    ASSERT_TRUE(initialize_device_generation_control(19,response.size(),policy,row.data(),leading,
        {.kind=DeviceGenerationAdmissionKind::ContinueResponse,.session_epoch=7,.workspace_generation=9,.prior_tickets=&ticket}));
    ASSERT_EQ(row[kDeviceGenerationControlDepthUtilityProbes],0);
    // The sixteenth eligible depth3 transaction starts after15 emitted tokens
    // and still needs its full four-row budget: 15+4=19. It emits1; the
    // remaining3 are then completed by the real clipped transaction below.
    for(int i=0;i<16;++i)ASSERT_NO_FATAL_FAILURE(commit(0,4)); // Actual target V16 loses to incumbent V48/R16.
    ASSERT_EQ(row[kDeviceGenerationControlDepthUtilityReturns],1);
    ASSERT_EQ(row[kDeviceGenerationControlCurrentDraftDepth],4);
    ASSERT_NO_FATAL_FAILURE(commit(15,row[kDeviceGenerationControlRemainingTokenCount]));
    ASSERT_EQ(row[kDeviceGenerationControlRequestComplete],1);
    ASSERT_EQ(row[kDeviceGenerationControlResponseTokenCount],19);
    EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityProbes],0);
    EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityReturns],1);
    EXPECT_EQ(validateDeviceGenerationTerminal(row,admission,response.size(),0,0),DeviceGenerationTerminalError::None);
}

/** @test Three actual consumed windows, not overlapping R4/R5/R6, admit a trial. */
TEST(Test__MTPDepthController, SurvivalConfirmationDisjointWindowsThenUnchangedPending)
{
    for(const int prefix:{3,5})
    {
        auto config=survivalConfig();config.promote_consecutive_windows=3;
        MTPDepthController controller(config,15);bool incoming=false;MTPDepthDecision decision;
        const int target=prefix==3?4:6;
        for(int transaction=1;transaction<=12;++transaction)
        {
            decision=controller.recordStep(committedObservation(5,prefix,incoming));
            ASSERT_NE(controller.proposalConfirmation(),nullptr);
            EXPECT_EQ(controller.stats().windows,static_cast<uint64_t>(transaction/4));
            if(transaction<12)
            {
                EXPECT_EQ(controller.currentDepth(),5);EXPECT_EQ(controller.stats().utility_probes,0u);
                EXPECT_EQ(controller.proposalConfirmation()->completed_windows,static_cast<uint32_t>(transaction/4));
                if(transaction>=4)EXPECT_EQ(decision.recommended_depth,target);
                if(transaction%4==0)
                {
                    EXPECT_EQ(decision.reason,MTPDepthDecisionReason::SurvivalAwaitingConfirmation);
                    EXPECT_EQ(controller.utilityState().window_emitted,0u);
                    EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::Clear);
                    EXPECT_EQ(controller.proposalConfirmation()->target_depth,static_cast<uint32_t>(target));
                }
                else EXPECT_FALSE(decision.evaluated);
            }
        }
        ASSERT_EQ(controller.currentDepth(),target);EXPECT_EQ(controller.stats().utility_probes,1u);
        EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeStarted);
        EXPECT_EQ(controller.utilityState().reference_depth,5u);EXPECT_EQ(controller.utilityState().reference_runs,4u);
        EXPECT_EQ(controller.utilityState().reference_verified_progress,static_cast<uint32_t>(4*(prefix+1)));
        EXPECT_EQ(controller.proposalConfirmation()->completed_windows,0u);
        EXPECT_EQ(controller.proposalConfirmation()->target_depth,static_cast<uint32_t>(target));
        for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(target,target,incoming));
        EXPECT_EQ(decision.reason,MTPDepthDecisionReason::UtilityProbeKept);
        EXPECT_EQ(controller.currentDepth(),target);EXPECT_EQ(controller.stats().windows,4u);
        EXPECT_EQ(controller.stats().utility_probes,1u);EXPECT_EQ(controller.stats().utility_keeps,1u);
        EXPECT_EQ(controller.stats().utility_returns,0u);EXPECT_EQ(controller.utilityState().window_emitted,0u);
    }
}

/** @test Changed target restarts; an actual full Hold clears a prior proposal. */
TEST(Test__MTPDepthController, SurvivalConfirmationChangedTargetAndCompletedHoldBreakStreak)
{
    auto config=survivalConfig();config.promote_consecutive_windows=3;
    MTPDepthController controller(config,15);bool incoming=false;MTPDepthDecision decision;
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(5,3,incoming));
    ASSERT_EQ(controller.proposalConfirmation()->target_depth,4u);
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(5,5,incoming));
    EXPECT_EQ(controller.proposalConfirmation()->target_depth,6u);
    EXPECT_EQ(controller.proposalConfirmation()->completed_windows,1u);EXPECT_EQ(controller.currentDepth(),5);
    config=survivalConfig(1,1,2);config.promote_consecutive_windows=3;
    controller.configure(config,15);incoming=false;
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(1,1,incoming));
    ASSERT_EQ(controller.proposalConfirmation()->target_depth,2u);
    for(int i=1;i<=16;++i)
    {
        decision=controller.recordStep(committedObservation(1,0,incoming));
        if(i<16){EXPECT_FALSE(decision.evaluated);EXPECT_EQ(controller.proposalConfirmation()->completed_windows,1u);}
    }
    EXPECT_EQ(decision.reason,MTPDepthDecisionReason::SurvivalHold);
    EXPECT_EQ(controller.proposalConfirmation()->target_depth,1u);EXPECT_EQ(controller.proposalConfirmation()->completed_windows,0u);
    EXPECT_EQ(controller.stats().windows,2u);EXPECT_EQ(controller.stats().utility_probes,0u);
}

/** @test Cache filtering precedes confirmation; genuine reopening requires three fresh windows. */
TEST(Test__MTPDepthController, SurvivalConfirmationCachedTargetAndReopenPreserveOwners)
{
    auto config=survivalConfig(4,3,4);config.promote_consecutive_windows=3;
    MTPDepthController controller(config,15);bool incoming=false;MTPDepthDecision decision;
    for(int i=0;i<12;++i)decision=controller.recordStep(committedObservation(4,2,incoming));
    ASSERT_EQ(controller.currentDepth(),3);ASSERT_EQ(controller.stats().utility_probes,1u);
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(3,0,incoming));
    ASSERT_EQ(controller.currentDepth(),4);ASSERT_EQ(controller.stats().utility_returns,1u);
    for(int i=0;i<16;++i)decision=controller.recordStep(committedObservation(4,2,incoming));
    ASSERT_EQ(decision.reason,MTPDepthDecisionReason::SurvivalHold);
    EXPECT_EQ(controller.proposalConfirmation()->completed_windows,0u);
    EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::RejectedComparison);
    for(int i=1;i<=12;++i)
    {
        decision=controller.recordStep(committedObservation(4,0,incoming));
        if(i<12)
        {
            EXPECT_EQ(controller.currentDepth(),4);EXPECT_EQ(controller.stats().utility_probes,1u);
            EXPECT_EQ(controller.utilityState().reference_depth,3u);
            EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);
        }
    }
    EXPECT_EQ(controller.currentDepth(),3);EXPECT_EQ(controller.stats().utility_probes,2u);
    EXPECT_EQ(controller.utilityState().phase,MTPReversibleProposalPhase::ProbePending);
    EXPECT_EQ(controller.utilityState().reference_depth,4u);EXPECT_EQ(controller.utilityState().reference_verified_progress,4u);
}

/** @test Excluded publication cannot confirm; reset discards partial confirmation. */
TEST(Test__MTPDepthController, SurvivalConfirmationClipStopPartialAndReset)
{
    auto config=survivalConfig();config.promote_consecutive_windows=3;
    MTPDepthController controller(config,15);bool incoming=false;MTPDepthDecision decision;
    for(int i=0;i<4;++i)decision=controller.recordStep(committedObservation(5,3,incoming));
    const auto before=*controller.proposalConfirmation();
    EXPECT_EQ(controller.recordStep(committedObservation(5,0,incoming,1)).reason,MTPDepthDecisionReason::BudgetLimited);
    EXPECT_EQ(controller.recordStep(committedObservation(5,0,incoming,0,true)).reason,MTPDepthDecisionReason::ModelStopped);
    EXPECT_EQ(controller.proposalConfirmation()->target_depth,before.target_depth);
    EXPECT_EQ(controller.proposalConfirmation()->completed_windows,before.completed_windows);
    EXPECT_EQ(controller.stats().windows,1u);EXPECT_EQ(controller.utilityState().window_emitted,0u);
    controller.reset();incoming=false;
    EXPECT_EQ(controller.proposalConfirmation()->target_depth,5u);EXPECT_EQ(controller.proposalConfirmation()->completed_windows,0u);
    for(int i=0;i<3;++i)decision=controller.recordStep(committedObservation(5,3,incoming));
    EXPECT_FALSE(decision.evaluated);EXPECT_EQ(controller.stats().windows,0u);EXPECT_EQ(controller.stats().utility_probes,0u);
    EXPECT_EQ(controller.proposalConfirmation()->completed_windows,0u);
}

/** @test Existing non-measured and Observe controller histories retain their actual policy. */
TEST(Test__MTPDepthController, SurvivalConfirmationLeavesLegacyAndObserveUnchanged)
{
    auto config=survivalConfig();config.mode=MTPDepthPolicyMode::Observe;config.promote_consecutive_windows=3;
    MTPDepthController observer(config,15);bool incoming=false;
    ASSERT_FALSE(observer.costProfile().active());EXPECT_EQ(observer.proposalConfirmation(),nullptr);
    for(int i=0;i<16;++i)observer.recordStep(committedObservation(5,0,incoming));
    EXPECT_EQ(observer.currentDepth(),5);EXPECT_EQ(observer.stats().utility_probes,0u);
    config.mode=MTPDepthPolicyMode::Dynamic;config.use_generated_policy=false;
    MTPDepthController legacy(config,15);ASSERT_FALSE(legacy.costProfile().active());
    EXPECT_EQ(legacy.proposalConfirmation(),nullptr);
    for(int i=0;i<4;++i)legacy.recordStep(observation(5,0));
    EXPECT_EQ(legacy.currentDepth(),4);EXPECT_EQ(legacy.stats().utility_probes,0u);
}

/** @test Actual host history poison is rejected before any eligible observation is recorded. */
TEST(Test__MTPDepthController, SurvivalConfirmationRejectsPoisonBeforeObservationMutation)
{
    auto config=survivalConfig();config.promote_consecutive_windows=3;
    MTPDepthController controller(config,15);
    for(const auto poison:{MTPProposalConfirmation{4,3},{4,0},{5,1},{16,1},{4,UINT32_MAX}})
    {
        controller.reset();bool incoming=false;
        // Corrupt the actual test-owned controller, never a production mutation API.
        *const_cast<MTPProposalConfirmation *>(controller.proposalConfirmation())=poison;
        EXPECT_THROW(controller.recordStep(committedObservation(5,3,incoming)),std::invalid_argument);
        EXPECT_EQ(controller.stats().windows,0u);EXPECT_EQ(controller.stats().updates,0u);
        EXPECT_EQ(controller.utilityState().window_emitted,0u);
        EXPECT_EQ(controller.currentDepth(),5);
        EXPECT_EQ(controller.proposalConfirmation()->target_depth,poison.target_depth);
        EXPECT_EQ(controller.proposalConfirmation()->completed_windows,poison.completed_windows);
    }
}

namespace
{
    /**
     * @brief Produce a genuine completed response with deferred or Pending confirmation.
     * @param pending Start the actual trial, then finish with fewer than four eligible target frames.
     * @param row Actual initialized/budgeted/committed controller, never fabricated accounting.
     * @param admission Exact immutable request used by terminal authentication.
     *
     * Deferred budget14 permits eight full-width depth5 zero outcomes (Y9),
     * then a maintenance-clipped full acceptance emits the remaining5 tokens.
     * Budget19 permits twelve zero outcomes (Y13) and three depth1 full outcomes
     * (Y5), then the final clipped token. The real reducer owns carry and P/Y.
     */
    void produceSurvivalConfirmationTerminal(bool pending,
        std::array<int,sampling_math::kDeviceGenerationControlCount> &row,
        DeviceGenerationAdmissionRequest &admission)
    {
        using namespace sampling_math;
        MTPRuntimeConfig config;config.enabled=true;config.draft_tokens=config.graph_capacity_draft_tokens=15;
        config.verify_mode=MTPVerifyMode::Greedy;config.depth_policy=survivalConfig();
        config.depth_policy.promote_consecutive_windows=3;
        const auto policy=resolveMTPDeviceGenerationDepthPolicy(config);
        ASSERT_TRUE(policy.valid());
        admission={.request_count=1,.max_new_tokens=pending?19:14,.depth_policy=policy};
        std::array<int32_t,64> response{};
        ASSERT_TRUE(initialize_device_generation_control(admission.max_new_tokens,response.size(),policy,row.data()));
        int condition=100,base=23;
        auto commit=[&](int prefix,int maintenance)
        {
            const int depth=row[kDeviceGenerationControlCurrentDraftDepth];
            const int leading=row[kDeviceGenerationControlNextLeadingCommittedOutputCount];
            std::array<int,16> tokens{},flags{},compact{};
            std::array<int,kSpeculativeBatchMetaCount> meta{};std::array<int,7> publication{};
            for(int i=0;i<16;++i){tokens[i]=1000+i;flags[i]=i<prefix?1:0;}
            summarize_speculative_verify_batch_at_commit_boundary(condition,tokens.data(),flags.data(),depth,
                nullptr,0,2000,1,maintenance,compact.data(),compact.size(),meta.data(),nullptr,leading);
            ASSERT_EQ(meta[kSpecBatchMetaOk],1);
            ASSERT_EQ(prepare_device_generation_transaction_budget(depth+1,maintenance,row.data()),maintenance);
            ASSERT_TRUE(commit_device_generation_and_derive_speculative_publication_metadata(
                compact.data(),compact.size(),meta.data(),meta.size(),0,depth+1,base,response.data(),response.size(),row.data(),
                publication.data(),publication.data()+1,publication.data()+2,publication.data()+3,
                publication.data()+4,publication.data()+5,publication.data()+6));
            condition=publication[4];base=publication[1];
        };
        for(int i=0;i<(pending?12:8);++i)ASSERT_NO_FATAL_FAILURE(commit(0,6));
        if(pending)
        {
            ASSERT_EQ(row[kDeviceGenerationControlCurrentDraftDepth],1);
            for(int i=0;i<3;++i)ASSERT_NO_FATAL_FAILURE(commit(1,2));
            ASSERT_EQ(row[kDeviceGenerationControlRemainingTokenCount],1);
        }
        else
        {
            ASSERT_EQ(row[kDeviceGenerationControlCurrentDraftDepth],5);
            ASSERT_EQ(row[kDeviceGenerationControlRemainingTokenCount],5);
        }
        ASSERT_NO_FATAL_FAILURE(commit(15,row[kDeviceGenerationControlRemainingTokenCount]));
        ASSERT_EQ(row[kDeviceGenerationControlRequestComplete],1);
        ASSERT_EQ(row[kDeviceGenerationControlResponseTokenCount],admission.max_new_tokens);
        ASSERT_EQ(row[kDeviceGenerationControlTransactionCount],pending?16:9);
    }
}

/** @test A valid deferred recommendation differs from current without invalidating the terminal. */
TEST(Test__MTPDepthController, SurvivalConfirmationTerminalAcceptsDeferredAndPartialPending)
{
    using namespace sampling_math;
    for(bool pending:{false,true})
    {
        std::array<int,kDeviceGenerationControlCount> row{};DeviceGenerationAdmissionRequest admission;
        ASSERT_NO_FATAL_FAILURE(produceSurvivalConfirmationTerminal(pending,row,admission));
        ASSERT_EQ(validateDeviceGenerationTerminal(row,admission,64,0,0),DeviceGenerationTerminalError::None);
        EXPECT_EQ(row[kDeviceGenerationControlDepthLastRecommendedDepth],1);
        EXPECT_EQ(row[kDeviceGenerationControlDepthPromotionStreak],pending?0:2);
        EXPECT_EQ(row[kDeviceGenerationControlCurrentDraftDepth],pending?1:5);
        EXPECT_EQ(row[kDeviceGenerationControlDepthEvaluatedWindows],pending?3:2);
        EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityPhase],pending?1:0);
        EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityProbes],pending?1:0);
        EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityKeeps],0);EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityReturns],0);
        EXPECT_EQ(row[kDeviceGenerationControlDepthWindowVerifierRuns],pending?3:0);
        EXPECT_EQ(row[kDeviceGenerationControlDepthUtilityWindowEmitted],pending?5:0);
    }
}

/** @test Terminal publication cannot omit confirmation validation on otherwise real physical accounting. */
TEST(Test__MTPDepthController, SurvivalConfirmationTerminalRejectsHistoryPoison)
{
    using namespace sampling_math;
    for(bool pending:{false,true})
    {
        std::array<int,kDeviceGenerationControlCount> row{};DeviceGenerationAdmissionRequest admission;
        ASSERT_NO_FATAL_FAILURE(produceSurvivalConfirmationTerminal(pending,row,admission));
        ASSERT_EQ(validateDeviceGenerationTerminal(row,admission,64,0,0),DeviceGenerationTerminalError::None);
        for(const auto poison:std::array<std::array<int,2>,7>{{
            {{-1,1}},{{1,-1}},{{0,1}},{{16,1}},{{1,3}},{{1,INT32_MAX}},
            {{row[kDeviceGenerationControlCurrentDraftDepth],1}}}})
        {
            auto corrupt=row;
            corrupt[kDeviceGenerationControlDepthLastRecommendedDepth]=poison[0];
            corrupt[kDeviceGenerationControlDepthPromotionStreak]=poison[1];
            EXPECT_EQ(validateDeviceGenerationTerminal(corrupt,admission,64,0,0),
                DeviceGenerationTerminalError::InvalidDepthStatistics)<<pending<<":"<<poison[0]<<","<<poison[1];
        }
        if(pending)
        {
            // An otherwise admitted different target with a positive streak is
            // invalid specifically because the independent Pending gate owns selection.
            auto corrupt=row;corrupt[kDeviceGenerationControlDepthLastRecommendedDepth]=2;
            corrupt[kDeviceGenerationControlDepthPromotionStreak]=1;
            EXPECT_EQ(validateDeviceGenerationTerminal(corrupt,admission,64,0,0),DeviceGenerationTerminalError::InvalidDepthStatistics);
            // Pending cannot retain an already completed min4 comparison at the terminal.
            corrupt=row;corrupt[kDeviceGenerationControlDepthWindowVerifierRuns]=4;
            EXPECT_EQ(validateDeviceGenerationTerminal(corrupt,admission,64,0,0),DeviceGenerationTerminalError::InvalidDepthStatistics);
        }
        else
        {
            auto corrupt=row;corrupt[kDeviceGenerationControlDepthPromotionStreak]=0;
            EXPECT_EQ(validateDeviceGenerationTerminal(corrupt,admission,64,0,0),DeviceGenerationTerminalError::InvalidDepthStatistics);
        }
    }
}
