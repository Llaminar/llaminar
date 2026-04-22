/**
 * @file Test__GlobalOrchestratorRunner.cpp
 * @brief Unit tests for GlobalOrchestratorRunner (Phase 4)
 *
 * Tests the IOrchestrationRunner adapter that wraps GlobalOrchestrator.
 * Verifies lifecycle management, inference delegation, and sampling state.
 *
 * Uses MockMPIContext and MockDeviceRunner — no real MPI or devices needed.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>

#include "execution/global/GlobalOrchestratorRunner.h"
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "tensors/TensorClasses.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockRankOrchestrator.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__GlobalOrchestratorRunner : public ::testing::Test
    {
    protected:
        static constexpr int VOCAB_SIZE = 1000;
        static constexpr int D_MODEL = 128;
        static constexpr int TOTAL_LAYERS = 24;

        /**
         * @brief Build a single-stage global TP topology
         */
        static GlobalPPTopology buildSingleStageTopo(int world_size)
        {
            GlobalPPStageSpec stage;
            stage.stage_id = 0;
            stage.first_layer = 0;
            stage.last_layer = TOTAL_LAYERS - 1;
            stage.has_embedding = true;
            stage.has_lm_head = true;

            if (world_size == 1)
            {
                stage.is_global_tp = false;
                stage.owning_rank = 0;
                stage.inner_mode = InnerParallelism::SINGLE_DEVICE;
                stage.devices = {GlobalDeviceAddress::cpu()};
            }
            else
            {
                stage.is_global_tp = true;
                for (int r = 0; r < world_size; ++r)
                    stage.participating_ranks.push_back(r);
                stage.per_rank_device = GlobalDeviceAddress::cpu();
            }

            return GlobalPPTopology::build({stage}, TOTAL_LAYERS, world_size);
        }

        /**
         * @brief Build a 2-stage PP topology
         */
        static GlobalPPTopology buildTwoStagePPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.is_global_tp = false;
            s0.owning_rank = 0;
            s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s0.devices = {GlobalDeviceAddress::cpu()};

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_lm_head = true;
            s1.is_global_tp = false;
            s1.owning_rank = 1;
            s1.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s1.devices = {GlobalDeviceAddress::cpu()};

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 2);
        }

        /**
         * @brief Create a GlobalOrchestratorRunner::Config with a pre-built GlobalOrchestrator
         */
        struct TestHarness
        {
            std::shared_ptr<MockMPIContext> mpi_ctx;
            GlobalOrchestratorRunner::Config runner_config;
            MockDeviceRunner *mock_runner_ptr = nullptr; // Non-owning, for assertions
        };

        static TestHarness makeRunnerConfig(
            GlobalPPTopology topology,
            int rank,
            int world_size,
            bool has_hidden_state = false)
        {
            TestHarness h;
            h.mpi_ctx = std::make_shared<MockMPIContext>(rank, world_size);

            // Create mock device runner
            MockDeviceRunner::Config runner_cfg;
            runner_cfg.vocab_size = VOCAB_SIZE;
            runner_cfg.has_hidden_state = has_hidden_state;
            runner_cfg.hidden_state_dim = D_MODEL;
            auto mock_runner = std::make_unique<MockDeviceRunner>(runner_cfg);
            h.mock_runner_ptr = mock_runner.get();

            // Build GlobalOrchestrator
            GlobalOrchestrator::Config go_config;
            go_config.topology = topology;
            go_config.rank = rank;
            go_config.world_size = world_size;
            go_config.mpi_ctx = h.mpi_ctx.get();
            go_config.rank_runner = std::move(mock_runner);
            go_config.vocab_size = VOCAB_SIZE;
            go_config.d_model = D_MODEL;
            go_config.architecture_name = "test_qwen2";

            auto global_orch = std::make_unique<GlobalOrchestrator>(std::move(go_config));

            // Build runner config
            h.runner_config.orchestration_config = OrchestrationConfig::defaults();
            h.runner_config.topology = topology;
            h.runner_config.mpi_ctx = h.mpi_ctx;
            h.runner_config.global_orchestrator = std::move(global_orch);
            h.runner_config.tokenizer = nullptr;

            return h;
        }
    };

    // =========================================================================
    // Lifecycle Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, InitializeSucceeds)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        EXPECT_FALSE(runner.isInitialized());
        EXPECT_TRUE(runner.initialize());
        EXPECT_TRUE(runner.isInitialized());
    }

    TEST_F(Test__GlobalOrchestratorRunner, InitializeWithoutOrchestratorFails)
    {
        GlobalOrchestratorRunner::Config config;
        config.orchestration_config = OrchestrationConfig::defaults();
        // global_orchestrator is nullptr

        GlobalOrchestratorRunner runner(std::move(config));

        EXPECT_FALSE(runner.initialize());
        EXPECT_FALSE(runner.isInitialized());
        EXPECT_FALSE(runner.lastError().empty());
    }

    TEST_F(Test__GlobalOrchestratorRunner, DoubleInitializeIsIdempotent)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        EXPECT_TRUE(runner.initialize());
        EXPECT_TRUE(runner.initialize()); // Second call should succeed
        EXPECT_TRUE(runner.isInitialized());
    }

    TEST_F(Test__GlobalOrchestratorRunner, ShutdownResetsState)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        EXPECT_TRUE(runner.initialize());
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
        EXPECT_EQ(runner.globalOrchestrator(), nullptr);
    }

    // =========================================================================
    // Status & Query Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, VocabSizeDelegates)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        EXPECT_EQ(runner.vocabSize(), VOCAB_SIZE);
    }

    TEST_F(Test__GlobalOrchestratorRunner, ArchitectureDelegates)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        EXPECT_EQ(runner.architecture(), "test_qwen2");
    }

    TEST_F(Test__GlobalOrchestratorRunner, ConfigReturnsCopy)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        // config() should work even before initialize
        const auto &cfg = runner.config();
        (void)cfg; // Just verify it doesn't crash
    }

    TEST_F(Test__GlobalOrchestratorRunner, TopologyAccessor)
    {
        auto topo = buildTwoStagePPTopo();
        auto h = makeRunnerConfig(topo, 0, 2);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        EXPECT_EQ(runner.topology().numStages(), 2);
        EXPECT_EQ(runner.topology().total_layers, TOTAL_LAYERS);
    }

    // =========================================================================
    // Inference Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, PrefillDelegatesToOrchestrator)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
        EXPECT_TRUE(runner.prefill(tokens));

        // Mock runner should have received forward call
        EXPECT_EQ(h.mock_runner_ptr->forward_call_count(), 1u);
        EXPECT_EQ(h.mock_runner_ptr->last_seq_len(), 5);
    }

    TEST_F(Test__GlobalOrchestratorRunner, PrefillFailsWhenNotInitialized)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        // Don't initialize

        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_FALSE(runner.prefill(tokens));
    }

    TEST_F(Test__GlobalOrchestratorRunner, PrefillFailsWithEmptyTokens)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        std::vector<int32_t> empty;
        EXPECT_FALSE(runner.prefill(empty));
    }

    TEST_F(Test__GlobalOrchestratorRunner, DecodeStepSamplesFromPrefillLogits)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        // Configure greedy sampling
        SamplingParams params;
        params.temperature = 0.0f;
        runner.setSamplingParams(params);

        // Prefill
        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_TRUE(runner.prefill(tokens));

        // First decodeStep should NOT call forward again (uses prefill logits)
        size_t initial_forward_count = h.mock_runner_ptr->forward_call_count();
        GenerationResult result = runner.decodeStep();
        EXPECT_TRUE(result.success());
        EXPECT_EQ(result.tokens.size(), 1u);
        // Forward should not have been called again (prefill logits reused)
        EXPECT_EQ(h.mock_runner_ptr->forward_call_count(), initial_forward_count);
    }

    TEST_F(Test__GlobalOrchestratorRunner, DecodeStepCallsForwardAfterFirst)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        SamplingParams params;
        params.temperature = 0.0f;
        runner.setSamplingParams(params);

        // Prefill
        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_TRUE(runner.prefill(tokens));

        // First decode (uses prefill logits)
        runner.decodeStep();

        // Second decode should call forward
        size_t pre_count = h.mock_runner_ptr->forward_call_count();
        GenerationResult result = runner.decodeStep();
        EXPECT_TRUE(result.success());
        EXPECT_GT(h.mock_runner_ptr->forward_call_count(), pre_count);
    }

    TEST_F(Test__GlobalOrchestratorRunner, DecodeStepFailsWhenNotInitialized)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        GenerationResult result = runner.decodeStep();
        EXPECT_FALSE(result.success());
    }

    TEST_F(Test__GlobalOrchestratorRunner, GenerateRunsPrefillAndDecode)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        SamplingParams params;
        params.temperature = 0.0f;

        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner.generate(prompt, 5, params);

        EXPECT_TRUE(result.success());
        EXPECT_EQ(result.tokens.size(), 5u);
        // Should have: 1 prefill forward + 4 decode forwards (first decode uses prefill logits)
        EXPECT_EQ(h.mock_runner_ptr->forward_call_count(), 5u);
    }

    TEST_F(Test__GlobalOrchestratorRunner, GenerateFailsWhenNotInitialized)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));

        SamplingParams params;
        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner.generate(prompt, 5, params);
        EXPECT_FALSE(result.success());
    }

    // =========================================================================
    // Stop Token Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, StopTokenEndsGeneration)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        // Set mock logits so that argmax returns token index 5 (stop token)
        // (GPU greedy returns -1, so CPU fallback is used. Mock logits[5] = 1.0)
        auto& mock_logits = const_cast<std::vector<float>&>(
            *reinterpret_cast<const std::vector<float>*>(&h.mock_runner_ptr->logits()[0]));
        // Simpler: just set stop token to 0 (default argmax of all-zero logits)
        runner.setStopTokens({0});

        SamplingParams params;
        params.temperature = 0.0f;

        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner.generate(prompt, 10, params);

        EXPECT_TRUE(result.success());
        EXPECT_TRUE(result.is_complete);
        // Should have stopped at first decode (token 0 is the stop token)
        EXPECT_EQ(result.tokens.size(), 1u);
    }

    // =========================================================================
    // Cache Management Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, ClearCacheDelegates)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        runner.clearCache();
        EXPECT_EQ(h.mock_runner_ptr->clear_cache_call_count(), 1u);
    }

    // =========================================================================
    // GPU Sampling Delegation Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, SampleGreedyOnDeviceDelegates)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        int token = runner.sampleGreedyOnDevice();
        // MockDeviceRunner returns 0 by default for sampleGreedyOnDevice
        EXPECT_GE(token, 0);
    }

    // =========================================================================
    // Snapshot Delegation Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, SnapshotKeysDelegates)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        auto keys = runner.getSnapshotKeys();
        // Mock returns empty, just verify no crash
        EXPECT_TRUE(keys.empty());
    }

    // =========================================================================
    // Two-Stage PP Tests (verifies adapter works with PP topologies)
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, TwoStagePP_Rank0Prefill)
    {
        // PP rank 0 must produce hidden state for transfer to rank 1
        auto h = makeRunnerConfig(buildTwoStagePPTopo(), 0, 2, /*has_hidden_state=*/true);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_TRUE(runner.prefill(tokens));
    }

    TEST_F(Test__GlobalOrchestratorRunner, TwoStagePP_Rank1Prefill)
    {
        auto h = makeRunnerConfig(buildTwoStagePPTopo(), 1, 2);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        std::vector<int32_t> tokens = {1, 2, 3};
        EXPECT_TRUE(runner.prefill(tokens));
    }

    // =========================================================================
    // Profiling Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestratorRunner, SetSkipLogitsGatherDoesNotCrash)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        runner.setSkipLogitsGatherDecode(true);
        runner.setSkipLogitsGatherDecode(false);
        runner.setSkipLogitsGatherPrefill(true);
        runner.setSkipLogitsGatherPrefill(false);
    }

    TEST_F(Test__GlobalOrchestratorRunner, TimelineMethodsDoNotCrash)
    {
        auto h = makeRunnerConfig(buildSingleStageTopo(1), 0, 1);
        GlobalOrchestratorRunner runner(std::move(h.runner_config));
        runner.initialize();

        runner.setSuppressTimeline(true);
        runner.setAccumulatePrefill(true);
        runner.flushStageTimeline();
    }

} // namespace llaminar2::test
