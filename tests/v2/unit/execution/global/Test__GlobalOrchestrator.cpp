/**
 * @file Test__GlobalOrchestrator.cpp
 * @brief Unit tests for GlobalOrchestrator (Phase 1)
 *
 * Tests the cross-machine MPI cluster inference orchestrator using
 * MockMPIContext and MockDeviceRunner (no real MPI or devices needed).
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>

#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "tensors/TensorClasses.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockRankOrchestrator.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__GlobalOrchestrator : public ::testing::Test
    {
    protected:
        static constexpr int VOCAB_SIZE = 1000;
        static constexpr int D_MODEL = 128;
        static constexpr int TOTAL_LAYERS = 24;

        /**
         * @brief Build a single-stage topology (all layers on one rank)
         * Pure global-TP: all ranks run all layers
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
                // Global TP: all ranks participate
                stage.is_global_tp = true;
                for (int r = 0; r < world_size; ++r)
                    stage.participating_ranks.push_back(r);
                stage.per_rank_device = GlobalDeviceAddress::cpu();
            }

            return GlobalPPTopology::build({stage}, TOTAL_LAYERS, world_size);
        }

        /**
         * @brief Build a 2-stage PP topology (rank 0: layers 0-11, rank 1: layers 12-23)
         */
        static GlobalPPTopology buildTwoStagePPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = false;
            s0.owning_rank = 0;
            s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s0.devices = {GlobalDeviceAddress::cpu()};

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_embedding = false;
            s1.has_lm_head = true;
            s1.is_global_tp = false;
            s1.owning_rank = 1;
            s1.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s1.devices = {GlobalDeviceAddress::cpu()};

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 2);
        }

        /**
         * @brief Create a GlobalOrchestrator::Config with a mock runner
         */
        static GlobalOrchestrator::Config makeConfig(
            GlobalPPTopology topology,
            int rank,
            int world_size,
            MockMPIContext *mpi_ctx,
            std::unique_ptr<MockDeviceRunner> runner)
        {
            GlobalOrchestrator::Config config;
            config.topology = std::move(topology);
            config.rank = rank;
            config.world_size = world_size;
            config.mpi_ctx = mpi_ctx;
            config.rank_runner = std::move(runner);
            config.vocab_size = VOCAB_SIZE;
            config.d_model = D_MODEL;
            config.architecture_name = "test_qwen2";
            return config;
        }
    };

    // =========================================================================
    // Construction Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ConstructsSingleRankSingleStage)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 1);
        EXPECT_EQ(orch.vocab_size(), VOCAB_SIZE);
        EXPECT_STREQ(orch.architecture(), "test_qwen2");
    }

    TEST_F(Test__GlobalOrchestrator, ConstructsGlobalTP)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildSingleStageTopo(2);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        // Single stage = all ranks are both head and tail
        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 1);
    }

    TEST_F(Test__GlobalOrchestrator, ConstructsTwoStagePP_Rank0)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 2);
    }

    TEST_F(Test__GlobalOrchestrator, ConstructsTwoStagePP_Rank1)
    {
        MockMPIContext mpi(1, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 2, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 2);
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ThrowsOnNullMPIContext)
    {
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator::Config config;
        config.topology = std::move(topo);
        config.rank = 0;
        config.world_size = 1;
        config.mpi_ctx = nullptr;
        config.rank_runner = std::move(runner);
        config.vocab_size = VOCAB_SIZE;
        config.d_model = D_MODEL;

        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnNullRunner)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        GlobalOrchestrator::Config config;
        config.topology = std::move(topo);
        config.rank = 0;
        config.world_size = 1;
        config.mpi_ctx = &mpi;
        config.rank_runner = nullptr;
        config.vocab_size = VOCAB_SIZE;
        config.d_model = D_MODEL;

        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnEmptyTopology)
    {
        MockMPIContext mpi(0, 1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator::Config config;
        config.topology = GlobalPPTopology{}; // empty
        config.rank = 0;
        config.world_size = 1;
        config.mpi_ctx = &mpi;
        config.rank_runner = std::move(runner);
        config.vocab_size = VOCAB_SIZE;
        config.d_model = D_MODEL;

        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnInvalidRank)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        auto config = makeConfig(std::move(topo), 5, 1, &mpi, std::move(runner));
        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnZeroVocabSize)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        auto config = makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner));
        config.vocab_size = 0;
        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnZeroDModel)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        auto config = makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner));
        config.d_model = 0;
        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    // =========================================================================
    // Forward Pass Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ForwardDelegatesToRunner_SingleStage)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), static_cast<int>(tokens.size())));
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        EXPECT_EQ(runner_raw->last_seq_len(), 3);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardReturnsRunnerLogits_TailRank)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.mock_logits = std::vector<float>(VOCAB_SIZE, 0.0f);
        runner_config.mock_logits[42] = 1.0f; // Token 42 is the predicted token
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        const float *log = orch.logits();
        ASSERT_NE(log, nullptr);
        EXPECT_EQ(log[42], 1.0f);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardReturnsNullLogits_NonTailRank)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        // Rank 0 is head, not tail
        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        EXPECT_EQ(orch.logits(), nullptr);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardReturnsFalseOnRunnerFailure)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.forward_should_fail = true;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        EXPECT_FALSE(orch.forward(tokens.data(), 1));
    }

    // =========================================================================
    // Clear Cache Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ClearCacheDelegatesAndBarriers)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildSingleStageTopo(2);
        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        orch.clear_cache();
        EXPECT_EQ(runner_raw->clear_cache_call_count(), 1u);
        EXPECT_GE(mpi.barrier_call_count(), 1u);
    }

    // =========================================================================
    // Sampling Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, SampleGreedyBroadcasts_SingleStage)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.mock_logits = std::vector<float>(VOCAB_SIZE, 0.0f);
        runner_config.mock_logits[7] = 99.0f; // Token 7 wins argmax
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        // Forward first to populate logits
        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        int token = orch.sampleGreedyOnDevice();
        // MockDeviceRunner::sampleGreedyOnDevice() returns -1, so fallback to CPU argmax
        EXPECT_EQ(token, 7);
        // Should have called broadcast_int32 once
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, SampleOnDeviceBroadcasts)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.mock_logits = std::vector<float>(VOCAB_SIZE, 0.0f);
        runner_config.mock_logits[3] = 99.0f;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        SamplingParams params;
        params.temperature = 0.0f; // greedy
        int token = orch.sampleOnDevice(params);
        EXPECT_EQ(token, 3);
    }

    // =========================================================================
    // Rank Plan Query Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, RankPlanAccessible)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        EXPECT_EQ(plan.rank, 0);
        EXPECT_FALSE(plan.steps.empty());
        EXPECT_TRUE(plan.hasWork());
    }

    TEST_F(Test__GlobalOrchestrator, TopologyAccessible)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        EXPECT_EQ(orch.topology().numStages(), 1);
        EXPECT_EQ(orch.topology().total_layers, TOTAL_LAYERS);
    }

    // =========================================================================
    // Delegation Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, GetPositionDelegates)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.position = 42;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        EXPECT_EQ(orch.get_position(), 42);
    }

    TEST_F(Test__GlobalOrchestrator, ExecutionPathDelegates)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        EXPECT_EQ(orch.executionPath(), ExecutionPath::GRAPH);
    }

    TEST_F(Test__GlobalOrchestrator, PrimaryDeviceIdDelegates)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        EXPECT_EQ(orch.primaryDeviceId(), DeviceId::cpu());
    }

    // =========================================================================
    // Pipeline Head/Tail for Global TP
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, GlobalTP_BothRanksAreHeadAndTail)
    {
        // In pure global TP (single stage), all ranks are head AND tail
        for (int rank = 0; rank < 2; ++rank)
        {
            MockMPIContext mpi(rank, 2);
            auto topo = buildSingleStageTopo(2);
            auto runner = std::make_unique<MockDeviceRunner>();

            GlobalOrchestrator orch(makeConfig(std::move(topo), rank, 2, &mpi, std::move(runner)));

            EXPECT_TRUE(orch.isPipelineHead()) << "rank=" << rank;
            EXPECT_TRUE(orch.isPipelineTail()) << "rank=" << rank;
        }
    }

    // =========================================================================
    // Activation Buffer Allocation for PP
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, AllocatesActivationBufferForPP)
    {
        // 2-stage PP: rank 0 should have transfer steps → buffer allocated
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        // Just verify construction succeeds (buffer is private, but the plan
        // will have transfers which triggers buffer allocation)
        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();
        EXPECT_FALSE(transfers.empty());
    }

    TEST_F(Test__GlobalOrchestrator, NoActivationBufferForPureTP)
    {
        // Single-stage global TP: no transfers needed
        MockMPIContext mpi(0, 2);
        auto topo = buildSingleStageTopo(2);
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();
        EXPECT_TRUE(transfers.empty());
    }

} // namespace llaminar2::test
