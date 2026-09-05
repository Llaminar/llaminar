/**
 * @file NodeExpertOverlayParityShardOwnershipTests.cpp
 * @brief Model-free proof that fixture shards share one active-case authority.
 *
 * Policy lookup and configuration projection are defined in different object
 * files. They must observe the same scoped binding rather than acquire private
 * header-local copies when implementation is split for build parallelism.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{
    /** Verify cross-shard lookup for both model definitions without loading weights. */
    TEST(Qwen35MoEFixtureShardOwnership, SharesActiveCaseAcrossPolicyAndLifecycle)
    {
        ASSERT_EQ(activeModelParityCase(), nullptr);
        /** Restore fixture-global state even if a projection throws. */
        struct RestoreBinding
        {
            const ModelParityCase *previous = g_active_model_parity_case;
            /** Leave no model-specific binding behind for the next test. */
            ~RestoreBinding() { g_active_model_parity_case = previous; }
        } restore_binding;

        for (const auto *test_case : {
                 &qwen35GraphNativeParityCases().front(),
                 &qwen122ExpertOverlayParityCases().front()})
        {
            g_active_model_parity_case = test_case;
            EXPECT_EQ(activeModelParityCase(), test_case);
            EXPECT_EQ(&activeModelParityCaseOrThrow(), test_case);
            const auto &config =
                Qwen35MoENodeExpertOverlayParityTest::generatedConfig();
            EXPECT_EQ(config.name, test_case->testName());
            EXPECT_EQ(config.activation_precision, ActivationPrecision::FP32);
        }
    }
}
