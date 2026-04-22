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

        /**
         * @brief Build a 3-stage PP topology (3 ranks, each owning 8 layers)
         */
        static GlobalPPTopology buildThreeStagePPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 7;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = false;
            s0.owning_rank = 0;
            s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s0.devices = {GlobalDeviceAddress::cpu()};

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 8;
            s1.last_layer = 15;
            s1.has_embedding = false;
            s1.has_lm_head = false;
            s1.is_global_tp = false;
            s1.owning_rank = 1;
            s1.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s1.devices = {GlobalDeviceAddress::cpu()};

            GlobalPPStageSpec s2;
            s2.stage_id = 2;
            s2.first_layer = 16;
            s2.last_layer = 23;
            s2.has_embedding = false;
            s2.has_lm_head = true;
            s2.is_global_tp = false;
            s2.owning_rank = 2;
            s2.inner_mode = InnerParallelism::SINGLE_DEVICE;
            s2.devices = {GlobalDeviceAddress::cpu()};

            return GlobalPPTopology::build({s0, s1, s2}, TOTAL_LAYERS, 3);
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

    TEST_F(Test__GlobalOrchestrator, ThrowsOnNegativeRank)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        auto config = makeConfig(std::move(topo), -1, 1, &mpi, std::move(runner));
        EXPECT_THROW(GlobalOrchestrator(std::move(config)), std::invalid_argument);
    }

    TEST_F(Test__GlobalOrchestrator, ThrowsOnNegativeVocabSize)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);
        auto runner = std::make_unique<MockDeviceRunner>();

        auto config = makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner));
        config.vocab_size = -1;
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

    TEST_F(Test__GlobalOrchestrator, LogitsNullOnNonTailRank)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        // Rank 0 is head, not tail — logits() should always be nullptr
        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineTail());
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

    TEST_F(Test__GlobalOrchestrator, SampleGreedyCPUFallbackAndBroadcast)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.mock_logits = std::vector<float>(VOCAB_SIZE, 0.0f);
        runner_config.mock_logits[7] = 99.0f; // Token 7 wins argmax
        // greedy_sample_token = -1 (default) → forces CPU argmax fallback
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        size_t broadcast_before = mpi.broadcast_call_count();
        int token = orch.sampleGreedyOnDevice();

        EXPECT_EQ(token, 7); // CPU argmax picks token 7
        EXPECT_EQ(mpi.broadcast_call_count(), broadcast_before + 1);
    }

    TEST_F(Test__GlobalOrchestrator, SampleGreedyUsesRunnerWhenAvailable)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.greedy_sample_token = 42; // Runner returns valid token
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        size_t broadcast_before = mpi.broadcast_call_count();
        int token = orch.sampleGreedyOnDevice();

        // Runner returned 42 directly, no CPU fallback needed
        EXPECT_EQ(token, 42);
        EXPECT_EQ(mpi.broadcast_call_count(), broadcast_before + 1);
    }

    TEST_F(Test__GlobalOrchestrator, SampleOnDeviceFallsBackToGreedyWhenRunnerReturnsNegative)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.mock_logits = std::vector<float>(VOCAB_SIZE, 0.0f);
        runner_config.mock_logits[3] = 99.0f; // Token 3 wins argmax
        // sample_on_device_token = -1 (default) → triggers greedy fallback
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        SamplingParams params;
        params.temperature = 0.8f;
        size_t broadcast_before = mpi.broadcast_call_count();
        int token = orch.sampleOnDevice(params);

        // Fell back to greedy (CPU argmax) → token 3
        EXPECT_EQ(token, 3);
        // Greedy fallback internally broadcasts once
        EXPECT_GE(mpi.broadcast_call_count(), broadcast_before + 1);
    }

    TEST_F(Test__GlobalOrchestrator, SampleOnDeviceSuccessPathBroadcasts)
    {
        MockMPIContext mpi(0, 1);
        auto topo = buildSingleStageTopo(1);

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.sample_on_device_token = 99; // Runner returns valid token
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 1, &mpi, std::move(runner)));

        std::vector<int> tokens = {1};
        ASSERT_TRUE(orch.forward(tokens.data(), 1));

        SamplingParams params;
        params.temperature = 0.8f;
        size_t broadcast_before = mpi.broadcast_call_count();
        int token = orch.sampleOnDevice(params);

        // Runner returned 99, broadcast to all ranks
        EXPECT_EQ(token, 99);
        EXPECT_EQ(mpi.broadcast_call_count(), broadcast_before + 1);
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

    TEST_F(Test__GlobalOrchestrator, PPRankPlanHasTransfersForPP)
    {
        // 2-stage PP: rank 0 has EXECUTE + SEND steps
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();
        EXPECT_FALSE(transfers.empty());
        // Rank 0 sends to rank 1
        EXPECT_EQ(transfers[0]->direction, RankTransferAction::Direction::SEND);
        EXPECT_EQ(transfers[0]->peer_rank, 1);
    }

    TEST_F(Test__GlobalOrchestrator, PureTPHasNoTransfers)
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

    // =========================================================================
    // Forward on PP Topology — Error Paths
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ForwardPP_Rank0_FailsWhenHiddenStateNull)
    {
        // Rank 0 in 2-stage PP: EXECUTE succeeds, then SEND fails because
        // MockDeviceRunner::getHiddenState() returns nullptr (default).
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        // forward() should execute the EXECUTE_STAGE step (succeeds), then
        // attempt SEND transfer, which fails because getHiddenState() is null
        EXPECT_FALSE(orch.forward(tokens.data(), 3));
        // The EXECUTE step ran before the SEND failed
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
    }

    // =========================================================================
    // Non-Tail Rank Sampling Behavior
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, SampleGreedyOnNonTailRankSkipsLocalSampling)
    {
        // In PP, the non-tail rank should NOT do local sampling.
        // It only participates in the broadcast (receiving the token).
        // With MockMPIContext (no-op broadcast), the token stays -1 on non-tail
        // because the mock doesn't actually send data from the tail.
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.greedy_sample_token = 42; // Would be used if tail sampled
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        // Rank 0 = head, not tail
        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));
        ASSERT_FALSE(orch.isPipelineTail());

        // Non-tail rank: token starts as -1, broadcast is no-op in mock,
        // so token remains -1 (in real MPI, tail would supply the value)
        int token = orch.sampleGreedyOnDevice();
        EXPECT_EQ(token, -1);

        // Broadcast was still called (non-tail participates in collective)
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, SampleOnDeviceNonTailRankSkipsLocalSampling)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();

        auto runner_config = MockDeviceRunner::Config{};
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.sample_on_device_token = 77;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        // Rank 0 = head, not tail
        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));
        ASSERT_FALSE(orch.isPipelineTail());

        SamplingParams params;
        params.temperature = 0.8f;
        int token = orch.sampleOnDevice(params);

        // Non-tail: local sampling skipped, broadcast no-op → token stays -1
        EXPECT_EQ(token, -1);
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    // =========================================================================
    // hasLogitsLocal — Tail vs Non-Tail
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, HasLogitsLocalOnlyOnTailRank)
    {
        // Non-tail rank should return false regardless of runner
        {
            MockMPIContext mpi(0, 2);
            auto topo = buildTwoStagePPTopo();
            auto runner = std::make_unique<MockDeviceRunner>();
            GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));
            EXPECT_FALSE(orch.hasLogitsLocal());
        }

        // Tail rank delegates to runner (MockDeviceRunner returns false by default)
        {
            MockMPIContext mpi(1, 2);
            auto topo = buildTwoStagePPTopo();
            auto runner = std::make_unique<MockDeviceRunner>();
            GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 2, &mpi, std::move(runner)));
            // Still false because MockDeviceRunner::hasLogitsLocal() returns false,
            // but the path through GlobalOrchestrator delegates correctly
            EXPECT_FALSE(orch.hasLogitsLocal());
        }
    }

    // =========================================================================
    // Phase 2: PP Forward Path with IMPIContext Wrappers
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, ForwardPP_Rank0_SendsViaMPIContext)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        // SEND was called via IMPIContext (not raw MPI)
        EXPECT_EQ(mpi.send_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardPP_Rank1_RecvsViaMPIContext)
    {
        MockMPIContext mpi(1, 2);
        auto topo = buildTwoStagePPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 2, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        // RECV was called via IMPIContext
        EXPECT_EQ(mpi.recv_call_count(), 1u);
        // Forward was called (after receiving hidden state)
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        // setHiddenState was called on the runner
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardPP_Rank0_HiddenStateHasCorrectSize)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        // Prefill with 5 tokens
        std::vector<int> tokens = {1, 2, 3, 4, 5};
        EXPECT_TRUE(orch.forward(tokens.data(), 5));

        // The runner should have produced a hidden state of size seq_len * d_model
        auto *hs = runner_raw->getHiddenState();
        ASSERT_NE(hs, nullptr);
        EXPECT_EQ(hs->numel(), static_cast<size_t>(5 * D_MODEL));
    }

    TEST_F(Test__GlobalOrchestrator, ForwardPP_Rank1_BufferResizesForPrefillThenDecode)
    {
        // Rank 1 (tail) receives activations — buffer should resize for prefill
        // then work correctly for decode (seq_len=1)
        MockMPIContext mpi(1, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 2, &mpi, std::move(runner)));

        // Prefill: seq_len=10
        std::vector<int> tokens_prefill(10, 1);
        EXPECT_TRUE(orch.forward(tokens_prefill.data(), 10));
        EXPECT_EQ(mpi.recv_call_count(), 1u);

        // Decode: seq_len=1
        std::vector<int> tokens_decode = {42};
        EXPECT_TRUE(orch.forward(tokens_decode.data(), 1));
        EXPECT_EQ(mpi.recv_call_count(), 2u);
    }

    TEST_F(Test__GlobalOrchestrator, ThreeStagePP_Rank0_IsHeadOnly)
    {
        MockMPIContext mpi(0, 3);
        auto topo = buildThreeStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 3, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 3);
    }

    TEST_F(Test__GlobalOrchestrator, ThreeStagePP_Rank1_IsMiddle)
    {
        MockMPIContext mpi(1, 3);
        auto topo = buildThreeStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 3, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());

        // Middle rank has both RECV and SEND transfers
        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();
        ASSERT_GE(transfers.size(), 2u);

        // Find RECV and SEND
        bool has_recv = false, has_send = false;
        for (const auto *t : transfers)
        {
            if (t->direction == RankTransferAction::Direction::RECV) has_recv = true;
            if (t->direction == RankTransferAction::Direction::SEND) has_send = true;
        }
        EXPECT_TRUE(has_recv);
        EXPECT_TRUE(has_send);
    }

    TEST_F(Test__GlobalOrchestrator, ThreeStagePP_Rank2_IsTailOnly)
    {
        MockMPIContext mpi(2, 3);
        auto topo = buildThreeStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 3, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
    }

    TEST_F(Test__GlobalOrchestrator, ThreeStagePP_MiddleRank_ForwardRecvsAndSends)
    {
        MockMPIContext mpi(1, 3);
        auto topo = buildThreeStagePPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 3, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));

        // Middle rank: RECV from rank 0, EXECUTE, SEND to rank 2
        EXPECT_EQ(mpi.recv_call_count(), 1u);
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        EXPECT_EQ(mpi.send_call_count(), 1u);
        // setHiddenState called once (from RECV)
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, ForwardPP_WorksWithMockMPIContext)
    {
        // This test proves the raw MPI_Send/MPI_Recv calls were replaced with
        // IMPIContext wrappers, since MockMPIContext returns MPI_COMM_NULL
        // for communicator() — raw MPI calls would crash/hang.
        for (int rank = 0; rank < 2; ++rank)
        {
            MockMPIContext mpi(rank, 2);
            auto topo = buildTwoStagePPTopo();

            MockDeviceRunner::Config runner_config;
            runner_config.vocab_size = VOCAB_SIZE;
            runner_config.has_hidden_state = (rank == 0); // Head produces hidden state
            runner_config.hidden_state_dim = D_MODEL;
            auto runner = std::make_unique<MockDeviceRunner>(runner_config);

            GlobalOrchestrator orch(makeConfig(std::move(topo), rank, 2, &mpi, std::move(runner)));

            std::vector<int> tokens = {1, 2};
            // If raw MPI calls were still used, this would crash (MPI_COMM_NULL)
            EXPECT_TRUE(orch.forward(tokens.data(), 2))
                << "forward() failed for rank " << rank;
        }
    }

    TEST_F(Test__GlobalOrchestrator, ThreeStagePP_ClearCacheBarrier)
    {
        MockMPIContext mpi(1, 3);
        auto topo = buildThreeStagePPTopo();
        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 3, &mpi, std::move(runner)));

        size_t barrier_before = mpi.barrier_call_count();
        orch.clear_cache();
        EXPECT_EQ(runner_raw->clear_cache_call_count(), 1u);
        EXPECT_GT(mpi.barrier_call_count(), barrier_before);
    }

} // namespace llaminar2::test
