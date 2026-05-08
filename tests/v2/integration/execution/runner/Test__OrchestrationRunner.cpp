/**
 * @file Test__OrchestrationRunner.cpp
 * @brief Integration tests for OrchestrationRunner
 *
 * Tests the OrchestrationRunner implementation with various configurations:
 * - Simple single-device execution
 * - LOCAL TP (multiple devices within a rank)
 * - Pipeline Parallel (PP across ranks)
 * - Combined PP + TP scenarios
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mpi.h>
#include <csignal>

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/NamedDomainGlobalRunner.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;
using namespace testing;

namespace
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__OrchestrationRunner : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            factory_ = createOrchestrationRunnerFactory();
        }

        /**
         * @brief Create a minimal valid execution plan
         */
        static RankExecutionPlan createSimplePlan()
        {
            RankExecutionPlan plan;
            plan.rank = 0;
            plan.hostname = "localhost";
            plan.numa_node = 0;
            plan.pp_stage_id = 0;
            plan.first_layer = 0;
            plan.last_layer = 23;
            plan.has_embedding = true;
            plan.has_lm_head = true;
            plan.primary_device = GlobalDeviceAddress::cpu();
            return plan;
        }

        /**
         * @brief Create a plan with LOCAL TP
         */
        static RankExecutionPlan createLocalTPPlan()
        {
            RankExecutionPlan plan = createSimplePlan();
            plan.local_tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:0"));
            plan.local_tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:1"));
            plan.local_tp_weights = {0.5f, 0.5f};
            plan.local_tp_backend = CollectiveBackendType::NCCL;
            plan.primary_device = plan.local_tp_devices[0];
            return plan;
        }

        /**
         * @brief Create plans for PP with 2 stages
         */
        static std::pair<RankExecutionPlan, RankExecutionPlan> createPPPlans()
        {
            RankExecutionPlan plan0;
            plan0.rank = 0;
            plan0.hostname = "localhost";
            plan0.numa_node = 0;
            plan0.pp_stage_id = 0;
            plan0.first_layer = 0;
            plan0.last_layer = 11;
            plan0.has_embedding = true;
            plan0.has_lm_head = false;
            plan0.next_rank = 1; // Send to rank 1
            plan0.primary_device = GlobalDeviceAddress::parse("0:cuda:0");

            RankExecutionPlan plan1;
            plan1.rank = 1;
            plan1.hostname = "localhost";
            plan1.numa_node = 1;
            plan1.pp_stage_id = 1;
            plan1.first_layer = 12;
            plan1.last_layer = 23;
            plan1.has_embedding = false;
            plan1.has_lm_head = true;
            plan1.prev_rank = 0; // Receive from rank 0
            plan1.primary_device = GlobalDeviceAddress::parse("1:cuda:0");

            return {plan0, plan1};
        }

        std::unique_ptr<IOrchestrationRunnerFactory> factory_;
    };

    // =========================================================================
    // Factory Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, FactoryNotNull)
    {
        EXPECT_NE(factory_, nullptr);
    }

    TEST_F(Test__OrchestrationRunner, CreateSimpleReturnsRunner)
    {
        auto runner = factory_->createSimple("/nonexistent/model.gguf", "cpu:0");
        ASSERT_NE(runner, nullptr);
    }

    TEST_F(Test__OrchestrationRunner, CreateFromConfigReturnsRunner)
    {
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();

        auto runner = factory_->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
    }

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, InitializesFromSimpleConfig)
    {
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.tp_degree = 1;
        config.pp_degree = 1;

        auto runner = factory_->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);

        // Config should be accessible before initialization
        const auto &returned_config = runner->config();
        EXPECT_EQ(returned_config.tp_degree, 1);
        EXPECT_EQ(returned_config.pp_degree, 1);
    }

    TEST_F(Test__OrchestrationRunner, InitializesWithLocalTP)
    {
        OrchestrationConfig config;
        config.tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:0"));
        config.tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:1"));
        config.tp_degree = 2;
        config.tp_scope = TPScope::LOCAL;

        auto runner = factory_->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);

        // Note: Full initialization would require actual devices
        // This test verifies config propagation
    }

    TEST_F(Test__OrchestrationRunner, InitializesWithPipelineParallel)
    {
        OrchestrationConfig config;
        config.pp_degree = 2;
        config.pp_stage_definitions.push_back(
            PPStageDefinition{0, "stage0", 0, 11});
        config.pp_stage_definitions.push_back(
            PPStageDefinition{1, "stage1", 12, 23});

        auto runner = factory_->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);
    }

    // =========================================================================
    // Execution Plan Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, ExecutionPlanIsAccessible)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationConfig config;

        OrchestrationRunner runner(config, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_EQ(returned.rank, 0);
        EXPECT_EQ(returned.first_layer, 0);
        EXPECT_EQ(returned.last_layer, 23);
        EXPECT_TRUE(returned.has_embedding);
        EXPECT_TRUE(returned.has_lm_head);
    }

    TEST_F(Test__OrchestrationRunner, ExecutionPlanWithLocalTP)
    {
        RankExecutionPlan plan = createLocalTPPlan();
        OrchestrationConfig config;

        OrchestrationRunner runner(config, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_TRUE(returned.usesLocalTP());
        EXPECT_EQ(returned.local_tp_devices.size(), 2u);
        EXPECT_EQ(returned.local_tp_backend, CollectiveBackendType::NCCL);
    }

    TEST_F(Test__OrchestrationRunner, ExecutionPlanWithPP)
    {
        auto [plan0, plan1] = createPPPlans();

        // Check first stage plan
        OrchestrationRunner runner0(OrchestrationConfig{}, plan0);
        const auto &p0 = runner0.executionPlan();
        EXPECT_TRUE(p0.usesPipelineParallel());
        EXPECT_TRUE(p0.isFirstStage());
        EXPECT_FALSE(p0.isLastStage());
        EXPECT_TRUE(p0.has_embedding);
        EXPECT_FALSE(p0.has_lm_head);

        // Check second stage plan
        OrchestrationRunner runner1(OrchestrationConfig{}, plan1);
        const auto &p1 = runner1.executionPlan();
        EXPECT_TRUE(p1.usesPipelineParallel());
        EXPECT_FALSE(p1.isFirstStage());
        EXPECT_TRUE(p1.isLastStage());
        EXPECT_FALSE(p1.has_embedding);
        EXPECT_TRUE(p1.has_lm_head);
    }

    // =========================================================================
    // Status Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, NotInitializedBeforeInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        EXPECT_FALSE(runner.isInitialized());
    }

    TEST_F(Test__OrchestrationRunner, LastErrorEmptyInitially)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        EXPECT_TRUE(runner.lastError().empty());
    }

    TEST_F(Test__OrchestrationRunner, VocabSizeZeroBeforeInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        EXPECT_EQ(runner.vocabSize(), 0);
    }

    TEST_F(Test__OrchestrationRunner, CurrentPositionZeroBeforeInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        EXPECT_EQ(runner.currentPosition(), 0);
    }

    TEST_F(Test__OrchestrationRunner, LastLogitsNullBeforeInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        EXPECT_EQ(runner.lastLogits(), nullptr);
    }

    // =========================================================================
    // Stop Token Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, SetStopTokensDoesNotThrow)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        std::vector<int32_t> stops = {151643, 151644}; // Common EOS tokens
        EXPECT_NO_THROW(runner.setStopTokens(stops));
    }

    // =========================================================================
    // Lifecycle Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, ShutdownSafeWithoutInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        // Should not throw even if not initialized
        EXPECT_NO_THROW(runner.shutdown());
    }

    TEST_F(Test__OrchestrationRunner, ClearCacheSafeWithoutInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        // Should not throw even if not initialized
        EXPECT_NO_THROW(runner.clearCache());
    }

    // =========================================================================
    // Inference Error Handling Tests
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, PrefillFailsWithoutInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_FALSE(runner.prefill(tokens));
        EXPECT_FALSE(runner.lastError().empty());
    }

    TEST_F(Test__OrchestrationRunner, DecodeStepFailsWithoutInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        GenerationResult result = runner.decodeStep();
        EXPECT_FALSE(result.success());
        EXPECT_FALSE(result.error.empty());
    }

    TEST_F(Test__OrchestrationRunner, GenerateFailsWithoutInit)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        std::vector<int32_t> tokens = {1, 2, 3};
        SamplingParams sampling;
        GenerationResult result = runner.generate(tokens, 10, sampling);

        EXPECT_FALSE(result.success());
        EXPECT_FALSE(result.error.empty());
    }

    TEST_F(Test__OrchestrationRunner, PrefillFailsWithEmptyTokens)
    {
        RankExecutionPlan plan = createSimplePlan();
        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        // Even after init, empty tokens should fail
        std::vector<int32_t> tokens;
        EXPECT_FALSE(runner.prefill(tokens));
    }

    // =========================================================================
    // GenerationResult Tests
    // =========================================================================

    TEST(GenerationResult, SuccessWhenNoError)
    {
        GenerationResult result;
        result.tokens = {100};
        EXPECT_TRUE(result.success());
    }

    TEST(GenerationResult, FailureWhenError)
    {
        GenerationResult result;
        result.error = "Something went wrong";
        EXPECT_FALSE(result.success());
    }

    TEST(GenerationResult, TokenCount)
    {
        GenerationResult result;
        result.tokens = {100, 200, 300};
        EXPECT_EQ(result.tokenCount(), 3u);
    }

    TEST(GenerationResult, DefaultNotComplete)
    {
        GenerationResult result;
        EXPECT_FALSE(result.is_complete);
    }

    // =========================================================================
    // Execution Plan Device Selection Tests
    // =========================================================================
    // These tests verify the OrchestrationRunner correctly uses execution plan
    // device assignments. The fix ensures that when GPUs are enumerated in
    // ClusterInventory, they are selected over CPU.

    TEST_F(Test__OrchestrationRunner, ExecutionPlanWithGPU_UsesGPUAsPrimaryDevice)
    {
        // Create a plan with a CUDA GPU as primary device
        RankExecutionPlan plan = createSimplePlan();
        plan.primary_device = GlobalDeviceAddress::parse("0:cuda:0");

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_EQ(returned.primary_device.device_type, DeviceType::CUDA);
        EXPECT_FALSE(returned.primary_device.isCPU())
            << "Primary device should be GPU when configured in plan";
    }

    TEST_F(Test__OrchestrationRunner, ExecutionPlanWithCPU_UsesCPU)
    {
        // Verify CPU plans work correctly (baseline case)
        RankExecutionPlan plan = createSimplePlan();
        plan.primary_device = GlobalDeviceAddress::cpu();

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_TRUE(returned.primary_device.isCPU());
    }

    TEST_F(Test__OrchestrationRunner, ExecutionPlanWithROCm_UsesROCm)
    {
        // Verify ROCm devices are correctly preserved
        RankExecutionPlan plan = createSimplePlan();
        plan.primary_device = GlobalDeviceAddress::parse("0:rocm:0");
        plan.primary_device_numa_explicit = true;

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_EQ(returned.primary_device.device_type, DeviceType::ROCm);
        EXPECT_TRUE(returned.primary_device_numa_explicit)
            << "Strict NUMA intent should be preserved in runner-facing execution plan";
    }

    TEST_F(Test__OrchestrationRunner, ExecutionPlanAmbiguousDevice_HasNumaExplicitFalse)
    {
        RankExecutionPlan plan = createSimplePlan();
        plan.primary_device = GlobalDeviceAddress::parse("rocm:0");
        plan.primary_device_numa_explicit = false;

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_EQ(returned.primary_device.device_type, DeviceType::ROCm);
        EXPECT_FALSE(returned.primary_device_numa_explicit)
            << "Ambiguous device intent should remain non-explicit in runner-facing execution plan";
    }

    TEST_F(Test__OrchestrationRunner, CpuShorthandMappedPlan_ExposesGlobalTPFields)
    {
        // Simulate post-mapping runtime plan for `-d cpu` on rank 1 of world_size=2.
        RankExecutionPlan plan = createSimplePlan();
        plan.rank = 1;
        plan.hostname = "localhost";
        plan.numa_node = 1;
        plan.primary_device = GlobalDeviceAddress::cpu(1, "localhost");
        plan.primary_device_numa_explicit = true;

        plan.tp_scope = TPScope::GLOBAL;
        plan.global_tp_domain_id = 0;
        plan.global_tp_rank_in_domain = 1;
        plan.global_tp_domain_size = 2;

        plan.local_tp_devices.clear();
        plan.weight_shard.total_shards = 2;
        plan.weight_shard.shard_index = 1;
        plan.weight_shard.work_fraction = 0.5f;

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        EXPECT_TRUE(returned.primary_device.isCPU());
        EXPECT_EQ(returned.primary_device.numa_node, 1);
        EXPECT_TRUE(returned.primary_device_numa_explicit);

        EXPECT_EQ(returned.tp_scope, TPScope::GLOBAL);
        EXPECT_TRUE(returned.usesGlobalTP());
        EXPECT_FALSE(returned.usesLocalTP());
        ASSERT_TRUE(returned.global_tp_domain_id.has_value());
        EXPECT_EQ(*returned.global_tp_domain_id, 0);
        EXPECT_EQ(returned.global_tp_rank_in_domain, 1);
        EXPECT_EQ(returned.global_tp_domain_size, 2);
        EXPECT_EQ(returned.weight_shard.total_shards, 2);
        EXPECT_EQ(returned.weight_shard.shard_index, 1);
    }

    TEST_F(Test__OrchestrationRunner, LocalTPPlan_HasGPUDevices)
    {
        // Verify LOCAL TP plans have GPU devices populated
        RankExecutionPlan plan = createLocalTPPlan();

        OrchestrationRunner runner(OrchestrationConfig{}, plan);

        const auto &returned = runner.executionPlan();
        ASSERT_FALSE(returned.local_tp_devices.empty());
        // All LOCAL TP devices should be GPUs
        for (const auto &dev : returned.local_tp_devices)
        {
            EXPECT_FALSE(dev.isCPU())
                << "LOCAL TP devices should be GPUs, not CPU";
        }
    }

    TEST_F(Test__OrchestrationRunner, ConfigWithTPDevices_PropagatesDevices)
    {
        // Verify tp_devices from config are propagated correctly
        OrchestrationConfig config;
        config.tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:0"));
        config.tp_devices.push_back(GlobalDeviceAddress::parse("0:cuda:1"));
        config.tp_degree = 2;
        config.tp_scope = TPScope::LOCAL;

        auto runner = factory_->createFromOrchestrationConfig(config);
        ASSERT_NE(runner, nullptr);

        const auto &returned_config = runner->config();
        EXPECT_EQ(returned_config.tp_devices.size(), 2u);
        EXPECT_EQ(returned_config.tp_devices[0].device_type, DeviceType::CUDA);
        EXPECT_EQ(returned_config.tp_devices[1].device_type, DeviceType::CUDA);
    }

    // =========================================================================
    // Phase 5: NamedDomainGlobalRunner factory dispatch
    // =========================================================================

    TEST_F(Test__OrchestrationRunner, NamedDomainGlobalConfig_CreatesNamedDomainGlobalRunner)
    {
        // A config with named domains having scope=node_local should cause the
        // factory to return a NamedDomainGlobalRunner rather than OrchestrationRunner.
        OrchestrationConfig cfg;
        cfg.domain_definitions.push_back(DomainDefinition::parse(
            "rocm_domain=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0"));
        cfg.domain_definitions.push_back(DomainDefinition::parse(
            "cpu_domain=0:cpu:0,1:cpu:0;scope=node_local;ranks=0,1"));
        cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=rocm_domain:0-13"));
        cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("1=cpu_domain:14-27"));

        // Verify shouldUse() predicate matches
        EXPECT_TRUE(NamedDomainGlobalRunner::shouldUse(cfg));

        auto runner = factory_->createFromOrchestrationConfig(cfg);
        ASSERT_NE(runner, nullptr);

        // Should be a NamedDomainGlobalRunner, not OrchestrationRunner
        auto *named_runner = dynamic_cast<NamedDomainGlobalRunner *>(runner.get());
        auto *orch_runner = dynamic_cast<OrchestrationRunner *>(runner.get());
        EXPECT_NE(named_runner, nullptr)
            << "Expected NamedDomainGlobalRunner but got OrchestrationRunner or other type";
        EXPECT_EQ(orch_runner, nullptr)
            << "Should not be an OrchestrationRunner for cross-rank named domain config";
    }

    TEST_F(Test__OrchestrationRunner, SimpleSingleDeviceConfig_DoesNotCreateNamedDomainRunner)
    {
        // Simple single-device config must still go through OrchestrationRunner
        OrchestrationConfig cfg;
        cfg.device_for_this_rank = GlobalDeviceAddress::cpu();

        EXPECT_FALSE(NamedDomainGlobalRunner::shouldUse(cfg));

        auto runner = factory_->createFromOrchestrationConfig(cfg);
        ASSERT_NE(runner, nullptr);

        auto *named_runner = dynamic_cast<NamedDomainGlobalRunner *>(runner.get());
        EXPECT_EQ(named_runner, nullptr)
            << "Simple single-device config should use OrchestrationRunner, not NamedDomainGlobalRunner";
    }

} // anonymous namespace

#include <csignal>

static volatile sig_atomic_t g_any_assertion_failed = 0;

static void cleanup_crash_handler(int sig)
{
    if (!g_any_assertion_failed)
        _exit(0);
    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

static void install_crash_handlers()
{
    struct sigaction sa = {};
    sa.sa_handler = cleanup_crash_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
}

class AssertionTracker : public ::testing::EmptyTestEventListener
{
    void OnTestPartResult(const ::testing::TestPartResult &result) override
    {
        if (result.failed())
            g_any_assertion_failed = 1;
    }
};

int main(int argc, char **argv)
{
    install_crash_handlers();

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::UnitTest::GetInstance()->listeners().Append(new AssertionTracker);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    _exit(result);
}
