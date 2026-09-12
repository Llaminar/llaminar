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
    /** Inspect the real fixture admission hook without loading a model. */
    class Qwen122RetainedEvidenceProbe final : public Qwen35MoENodeExpertOverlayParityTest
    {
    public:
        using Qwen35MoENodeExpertOverlayParityTest::productionParityRunnerEvidenceLifetime;
        using Qwen35MoENodeExpertOverlayParityTest::qwen122RunnerIdentity;
        /** The probe is manually owned; GoogleTest must not run model setup. */
        void TestBody() override {}
    };

    /** Retention requires actual sole ownership plus every immutable key field. */
    TEST(Qwen35MoEFixtureShardOwnership, EvidenceRequiresExactRetainedRunnerIdentity)
    {
        auto &cache = qwen122OverlayModelContextCampaignCache();
        ASSERT_EQ(cache.runner, nullptr);
        ASSERT_FALSE(cache.runner_identity.has_value());
        const auto &cases = qwen122ExpertOverlayParityCases();
        const auto selected = std::find_if(cases.begin(), cases.end(), [](const auto &cell)
        {
            return cell.mtp == ModelParityMTP::Depth1 && cell.expert_overlay &&
                   cell.expert_overlay->movement == ModelParityExpertMovement::Static;
        });
        ASSERT_NE(selected, cases.end());

        /** Restore process-wide bindings even when a key mutation fails. */
        struct RestoreBinding
        {
            const ModelParityCase *previous = g_active_model_parity_case;
            std::optional<std::string> process_campaign;
            /** Remember the environment before enabling the process reuse contract. */
            RestoreBinding()
            {
                if (const char *value = std::getenv("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN"))
                    process_campaign = value;
                setenv("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN", "1", 1);
            }
            /** Retire the uninitialized test owner and restore every scoped binding. */
            ~RestoreBinding()
            {
                auto &slot = qwen122OverlayModelContextCampaignCache();
                slot.runner.reset();
                slot.runner_identity.reset();
                g_active_model_parity_case = previous;
                if (process_campaign)
                    setenv("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN", process_campaign->c_str(), 1);
                else
                    unsetenv("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN");
            }
        } restore_binding;
        g_active_model_parity_case = &*selected;
        Qwen122RetainedEvidenceProbe probe;
        const auto key = probe.qwen122RunnerIdentity();
        EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::FreshRunner);

        // Construct only the ownership shell. No initialize(), model, graph,
        // allocation, device inventory, or inference is needed for admission.
        cache.runner = std::make_unique<OrchestrationRunner>(OrchestrationConfig{}, RankExecutionPlan{});
        EXPECT_THROW(probe.productionParityRunnerEvidenceLifetime(), std::logic_error);
        cache.runner_identity = key;
        EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::RetainedRunner);

        auto request = *selected;
        g_active_model_parity_case = &request;
        for (const auto depth : {ModelParityMTP::Depth2, ModelParityMTP::Depth3,
                                 ModelParityMTP::Depth15, ModelParityMTP::DynamicDepth})
        {
            request.mtp = depth;
            EXPECT_EQ(probe.qwen122RunnerIdentity(), key);
            EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::RetainedRunner);
        }
        request.mtp = ModelParityMTP::Off;
        EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::FreshRunner);
        request = *selected;
        request.expert_overlay->movement = ModelParityExpertMovement::Dynamic;
        EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::FreshRunner);
        g_active_model_parity_case = &*selected;

        for (int field = 0; field < 8; ++field)
        {
            auto changed = key;
            switch (field)
            {
            case 0: changed.physical.topology_id += "_different"; break;
            case 1: ++changed.physical.policy_slot; break;
            case 2: changed.activation_precision = ActivationPrecision::FP16; break;
            case 3: changed.kv_cache_precision = key.kv_cache_precision == KVCachePrecision::FP16
                        ? KVCachePrecision::FP32 : KVCachePrecision::FP16; break;
            case 4: ++changed.max_seq_len; break;
            case 5: ++changed.retained_mtp_draft_capacity; break;
            case 6: ++changed.prefill_capture_rows; break;
            case 7: ++changed.snapshot_mode; break;
            }
            cache.runner_identity = changed;
            EXPECT_EQ(probe.productionParityRunnerEvidenceLifetime(), ParityRunnerEvidenceLifetime::FreshRunner)
                << "Mismatched runner identity field " << field;
        }
        cache.runner_identity = key;
        cache.runner.reset();
        EXPECT_THROW(probe.productionParityRunnerEvidenceLifetime(), std::logic_error);
    }

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
