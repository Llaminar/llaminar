/**
 * @file Test__GlobalOrchestrator.cpp
 * @brief Unit tests for GlobalOrchestrator (Phase 1-3)
 *
 * Tests the cross-machine MPI cluster inference orchestrator using
 * MockMPIContext and MockDeviceRunner (no real MPI or devices needed).
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "tensors/TensorClasses.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockRankOrchestrator.h"

namespace llaminar2::test
{
    class MTPMockDeviceRunner : public MockDeviceRunner
    {
    public:
        explicit MTPMockDeviceRunner(int mtp_token = 17,
                                     int verifier_row0_token = 17,
                                     int verifier_row1_token = 23)
        {
            mtp_logits_.assign(1000, -10.0f);
            mtp_logits_[mtp_token] = 10.0f;
            all_position_logits_.assign(2 * 1000, -10.0f);
            all_position_logits_[verifier_row0_token] = 10.0f;
            all_position_logits_[1000 + verifier_row1_token] = 10.0f;
        }

        bool forwardMTP(int32_t draft_condition_token) override
        {
            ++forward_mtp_calls_;
            last_mtp_condition_token_ = draft_condition_token;
            return forward_mtp_ok_;
        }

        const float *mtpLogits() const override
        {
            return mtp_logits_.data();
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            ++set_all_position_calls_;
            all_position_enabled_ = enabled;
            return set_all_position_ok_;
        }

        const float *getAllPositionLogits() const override
        {
            return all_position_logits_.data();
        }

        std::string mtpDecodeUnsupportedReason() const override
        {
            return mtp_unsupported_reason_;
        }

        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
        {
            (void)seq_idx;
            PrefixStateSnapshot snapshot;
            snapshot.valid = capture_ok_;
            snapshot.cached_tokens = get_position();
            return snapshot;
        }

        PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override
        {
            (void)seq_idx;
            PrefixStateSnapshot snapshot;
            snapshot.valid = capture_ok_;
            snapshot.logical_checkpoint = true;
            snapshot.cached_tokens = get_position();
            return snapshot;
        }

        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override
        {
            (void)seq_idx;
            ++restore_live_calls_;
            if (!restore_ok_ || !snapshot.valid)
                return false;
            set_position(snapshot.cached_tokens);
            return true;
        }

        bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0) override
        {
            (void)seq_idx;
            ++truncate_live_calls_;
            if (!truncate_ok_)
                return false;
            set_position(cached_tokens);
            return true;
        }

        uint64_t moePlacementEpoch() const override { return moe_placement_epoch_; }

        int forward_mtp_calls() const { return forward_mtp_calls_; }
        int last_mtp_condition_token() const { return last_mtp_condition_token_; }
        int set_all_position_calls() const { return set_all_position_calls_; }
        bool all_position_enabled() const { return all_position_enabled_; }
        int restore_live_calls() const { return restore_live_calls_; }
        int truncate_live_calls() const { return truncate_live_calls_; }

        void set_mtp_unsupported_reason(std::string reason)
        {
            mtp_unsupported_reason_ = std::move(reason);
        }

        void set_moe_placement_epoch(uint64_t epoch)
        {
            moe_placement_epoch_ = epoch;
        }

    private:
        std::vector<float> mtp_logits_;
        std::vector<float> all_position_logits_;
        int forward_mtp_calls_ = 0;
        int set_all_position_calls_ = 0;
        int restore_live_calls_ = 0;
        int truncate_live_calls_ = 0;
        int last_mtp_condition_token_ = -1;
        bool forward_mtp_ok_ = true;
        bool set_all_position_ok_ = true;
        bool all_position_enabled_ = false;
        bool capture_ok_ = true;
        bool restore_ok_ = true;
        bool truncate_ok_ = true;
        uint64_t moe_placement_epoch_ = 0;
        std::string mtp_unsupported_reason_;
    };

    class ScriptedBroadcastMPIContext : public MockMPIContext
    {
    public:
        ScriptedBroadcastMPIContext(int rank, int world_size)
            : MockMPIContext(rank, world_size) {}

        void scriptInt32(std::vector<int32_t> data)
        {
            scripted_int32_.push_back(std::move(data));
        }

        void broadcast_int32(int32_t *data, size_t count, int root) const override
        {
            MockMPIContext::broadcast_int32(data, count, root);
            broadcast_roots_.push_back(root);
            if (rank() == root || scripted_int32_.empty())
                return;

            const std::vector<int32_t> scripted = scripted_int32_.front();
            scripted_int32_.pop_front();
            const size_t n = std::min(count, scripted.size());
            std::copy(scripted.begin(), scripted.begin() + static_cast<std::ptrdiff_t>(n), data);
        }

        size_t remainingScriptedInt32() const { return scripted_int32_.size(); }
        const std::vector<int> &broadcastRoots() const { return broadcast_roots_; }

    private:
        mutable std::deque<std::vector<int32_t>> scripted_int32_;
        mutable std::vector<int> broadcast_roots_;
    };

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

        static RankStageAction actionForStage(const GlobalPPTopology &topology,
                                              int rank,
                                              int stage_id)
        {
            GlobalPPRankPlan plan = GlobalPPRankPlanBuilder::build(topology, rank);
            for (const auto *action : plan.executeStages())
            {
                if (action->stage_id == stage_id)
                {
                    return *action;
                }
            }
            throw std::invalid_argument("test topology does not execute requested stage on rank");
        }

        static StageRunnerEntry makeStageRunnerEntry(const GlobalPPTopology &topology,
                                                     int rank,
                                                     int stage_id,
                                                     std::unique_ptr<MockDeviceRunner> runner)
        {
            StageRunnerEntry entry;
            entry.stage_id = stage_id;
            entry.action = actionForStage(topology, rank, stage_id);
            entry.domain_name = entry.action.domain_name;
            FactoryPPStageConfig pp_stage;
            pp_stage.first_layer = entry.action.first_layer;
            pp_stage.last_layer = entry.action.last_layer + 1;
            pp_stage.has_embedding = entry.action.has_embedding;
            pp_stage.has_lm_head = entry.action.has_lm_head;
            entry.pp_stage_config = pp_stage;
            entry.runner = std::move(runner);
            return entry;
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

        /**
         * @brief Build a 2-stage PP topology with 2-way global TP per stage
         *
         * 4 ranks total:
         *   Stage 0 (layers 0-11, has_embedding): ranks 0,1 (global TP)
         *   Stage 1 (layers 12-23, has_lm_head):  ranks 2,3 (global TP)
         *
         * Rank assignments:
         *   Rank 0: stage 0, tp_rank=0 (PP head)
         *   Rank 1: stage 0, tp_rank=1 (PP head)
         *   Rank 2: stage 1, tp_rank=0 (PP tail)
         *   Rank 3: stage 1, tp_rank=1 (PP tail)
         *
         * Disjoint rank sets → fan-out transfers from rank 0 (first rank
         * in sender domain) to ranks 2 and 3 (receiver domain ranks that
         * are not in the sender domain).
         */
        static GlobalPPTopology buildTwoStageTwoWayTPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = true;
            s0.participating_ranks = {0, 1};
            s0.per_rank_device = GlobalDeviceAddress::cpu();

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_embedding = false;
            s1.has_lm_head = true;
            s1.is_global_tp = true;
            s1.participating_ranks = {2, 3};
            s1.per_rank_device = GlobalDeviceAddress::cpu();

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 4);
        }

        /**
         * @brief Build a mixed topology: 2 PP-only stages + 1 global TP stage
         *
         * 4 ranks total:
         *   Stage 0 (layers 0-7, has_embedding): rank 0 only (PP-only)
         *   Stage 1 (layers 8-15): rank 1 only (PP-only)
         *   Stage 2 (layers 16-23, has_lm_head): ranks 2,3 (global TP)
         *
         * Transfers:
         *   Stage 0→1: rank 0 → rank 1 (single→single)
         *   Stage 1→2: rank 1 → rank 2, rank 1 → rank 3 (single→TP fan-out)
         */
        static GlobalPPTopology buildMixedPPAndTPTopo()
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
            s2.is_global_tp = true;
            s2.participating_ranks = {2, 3};
            s2.per_rank_device = GlobalDeviceAddress::cpu();

            return GlobalPPTopology::build({s0, s1, s2}, TOTAL_LAYERS, 4);
        }

        /**
         * @brief Build a 2-stage TP topology with the SAME rank set
         *
         * 2 ranks total:
         *   Stage 0 (layers 0-11, has_embedding): ranks 0,1 (global TP)
         *   Stage 1 (layers 12-23, has_lm_head):  ranks 0,1 (global TP)
         *
         * Same rank set — each rank needs a local handoff between distinct stages.
         */
        static GlobalPPTopology buildTwoStageSameTPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = true;
            s0.participating_ranks = {0, 1};
            s0.per_rank_device = GlobalDeviceAddress::cpu();

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_embedding = false;
            s1.has_lm_head = true;
            s1.is_global_tp = true;
            s1.participating_ranks = {0, 1};
            s1.per_rank_device = GlobalDeviceAddress::cpu();

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 2);
        }

        /**
         * @brief Build the Phase 6 named-domain shape: rank-local ROCm TP stage
         *        followed by node-local CPU TP across ranks 0 and 1.
         */
        static GlobalPPTopology buildNamedDomainHybridTopo(int gpu_owner_rank = 0)
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.domain_name = gpu_owner_rank == 0 ? "rocm_socket0" : "rocm_socket1";
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = false;
            s0.owning_rank = gpu_owner_rank;
            s0.inner_mode = InnerParallelism::LOCAL_TP;
            s0.backend = CollectiveBackendType::RCCL;
            s0.devices = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(1, 0)};

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.domain_name = "cpu_sockets";
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_embedding = false;
            s1.has_lm_head = true;
            s1.is_global_tp = true;
            s1.backend = CollectiveBackendType::UPI;
            s1.participating_ranks = {0, 1};
            s1.per_rank_device = GlobalDeviceAddress::cpu(0);
            s1.per_rank_devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 2);
        }

        /**
         * @brief Build a 2-stage TP topology with PARTIAL overlap
         *
         * 3 ranks total:
         *   Stage 0 (layers 0-11, has_embedding): ranks 0,1 (global TP)
         *   Stage 1 (layers 12-23, has_lm_head):  ranks 1,2 (global TP)
         *
         * Partial overlap: rank 1 is in both domains, rank 2 needs data.
         * Fan-out from rank 0 (first in sender domain) to rank 2.
         */
        static GlobalPPTopology buildPartialOverlapTPTopo()
        {
            GlobalPPStageSpec s0;
            s0.stage_id = 0;
            s0.first_layer = 0;
            s0.last_layer = 11;
            s0.has_embedding = true;
            s0.has_lm_head = false;
            s0.is_global_tp = true;
            s0.participating_ranks = {0, 1};
            s0.per_rank_device = GlobalDeviceAddress::cpu();

            GlobalPPStageSpec s1;
            s1.stage_id = 1;
            s1.first_layer = 12;
            s1.last_layer = 23;
            s1.has_embedding = false;
            s1.has_lm_head = true;
            s1.is_global_tp = true;
            s1.participating_ranks = {1, 2};
            s1.per_rank_device = GlobalDeviceAddress::cpu();

            return GlobalPPTopology::build({s0, s1}, TOTAL_LAYERS, 3);
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
    // MTP Delegation Tests
    // =========================================================================

    TEST_F(Test__GlobalOrchestrator, SingleStageGlobalTPDelegatesMTPSurfacesAndLiveSnapshots)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildSingleStageTopo(2);
        auto runner = std::make_unique<MTPMockDeviceRunner>();
        auto *runner_raw = runner.get();
        runner_raw->set_position(11);
        runner_raw->set_moe_placement_epoch(13);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.mtpDecodeUnsupportedReason().empty());
        EXPECT_TRUE(orch.supportsMTPTokenCoordination());
        EXPECT_EQ(orch.moePlacementEpoch(), 13u);
        EXPECT_TRUE(orch.forwardMTP(42));
        EXPECT_EQ(runner_raw->forward_mtp_calls(), 1);
        EXPECT_EQ(runner_raw->last_mtp_condition_token(), 42);

        const float *mtp_logits = orch.mtpLogits();
        ASSERT_NE(mtp_logits, nullptr);
        EXPECT_FLOAT_EQ(mtp_logits[17], 10.0f);
        EXPECT_EQ(orch.sampleGreedyFromMTPLogitsOnDevice(), 17);

        EXPECT_TRUE(orch.setComputeAllPositionLogits(true));
        EXPECT_EQ(runner_raw->set_all_position_calls(), 1);
        EXPECT_TRUE(runner_raw->all_position_enabled());

        const float *verifier_logits = orch.getAllPositionLogits();
        ASSERT_NE(verifier_logits, nullptr);
        EXPECT_FLOAT_EQ(verifier_logits[17], 10.0f);
        EXPECT_FLOAT_EQ(verifier_logits[VOCAB_SIZE + 23], 10.0f);
        EXPECT_EQ(orch.sampleGreedyFromAllPositionLogitsOnDevice(1), 23);

        PrefixStateSnapshot snapshot = orch.captureLivePrefixState();
        ASSERT_TRUE(snapshot.valid);
        EXPECT_EQ(snapshot.cached_tokens, 11);
        ASSERT_EQ(snapshot.participant_snapshots.size(), 1u);

        runner_raw->set_position(19);
        EXPECT_TRUE(orch.restoreLivePrefixState(snapshot));
        EXPECT_EQ(runner_raw->restore_live_calls(), 1);
        EXPECT_EQ(runner_raw->get_position(), 11);

        EXPECT_TRUE(orch.truncateLivePrefixState(5));
        EXPECT_EQ(runner_raw->truncate_live_calls(), 1);
        EXPECT_EQ(runner_raw->get_position(), 5);
    }

    TEST_F(Test__GlobalOrchestrator, GlobalTPMTPDraftSamplingUsesBroadcastRootTokenOnNonRootRank)
    {
        ScriptedBroadcastMPIContext mpi(1, 2);
        mpi.scriptInt32({17});
        mpi.scriptInt32({23});
        auto topo = buildSingleStageTopo(2);
        auto runner = std::make_unique<MTPMockDeviceRunner>(
            /*mtp_token=*/99,
            /*verifier_row0_token=*/88,
            /*verifier_row1_token=*/77);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 2, &mpi, std::move(runner)));
        ASSERT_TRUE(orch.isPipelineTail());
        EXPECT_TRUE(orch.supportsMTPTokenCoordination());
        EXPECT_TRUE(orch.forwardMTP(42));

        EXPECT_EQ(orch.sampleGreedyFromMTPLogitsOnDevice(), 17);
        EXPECT_TRUE(orch.setComputeAllPositionLogits(true));
        EXPECT_EQ(orch.sampleGreedyFromAllPositionLogitsOnDevice(1), 23);
        EXPECT_EQ(mpi.remainingScriptedInt32(), 0u);
        ASSERT_EQ(mpi.broadcastRoots().size(), 2u);
        EXPECT_EQ(mpi.broadcastRoots()[0], 0);
        EXPECT_EQ(mpi.broadcastRoots()[1], 0);
    }

    TEST_F(Test__GlobalOrchestrator, GlobalPPMTPReportsTopologyBypassBeforeDelegation)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MTPMockDeviceRunner>();
        auto *runner_raw = runner.get();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        EXPECT_NE(orch.mtpDecodeUnsupportedReason().find("GlobalPP"), std::string::npos);
        EXPECT_FALSE(orch.supportsMTPTokenCoordination());
        EXPECT_FALSE(orch.forwardMTP(7));
        EXPECT_FALSE(orch.setComputeAllPositionLogits(true));
        EXPECT_EQ(runner_raw->forward_mtp_calls(), 0);
        EXPECT_EQ(runner_raw->set_all_position_calls(), 0);
        EXPECT_EQ(orch.mtpLogits(), nullptr);
        EXPECT_EQ(orch.getAllPositionLogits(), nullptr);
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

    TEST_F(Test__GlobalOrchestrator, NamedDomainStageRunnerAccessorsExposeLocalRunners)
    {
        {
            MockMPIContext mpi(0, 2);
            auto topo = buildNamedDomainHybridTopo(/*gpu_owner_rank=*/0);

            auto gpu_raw = new MockDeviceRunner();
            auto gpu_runner = std::unique_ptr<MockDeviceRunner>(gpu_raw);
            auto cpu_raw = new MockDeviceRunner();
            auto cpu_runner = std::unique_ptr<MockDeviceRunner>(cpu_raw);

            GlobalOrchestrator::Config config;
            config.topology = topo;
            config.rank = 0;
            config.world_size = 2;
            config.mpi_ctx = &mpi;
            config.stage_runners.push_back(makeStageRunnerEntry(topo, 0, 0, std::move(gpu_runner)));
            config.stage_runners.push_back(makeStageRunnerEntry(topo, 0, 1, std::move(cpu_runner)));
            config.vocab_size = VOCAB_SIZE;
            config.d_model = D_MODEL;
            config.architecture_name = "test_qwen35moe";

            GlobalOrchestrator orch(std::move(config));

            EXPECT_EQ(orch.stageRunnerCount(), 2u);
            EXPECT_EQ(orch.stageRunnerForStage(0), gpu_raw);
            EXPECT_EQ(orch.stageRunnerForStage(1), cpu_raw);
            EXPECT_EQ(orch.stageRunnerForDomain("rocm_socket0"), gpu_raw);
            EXPECT_EQ(orch.stageRunnerForDomain("cpu_sockets"), cpu_raw);
            EXPECT_EQ(orch.stageRunnerForDomain("missing"), nullptr);

            const auto *gpu_entry = orch.stageRunnerEntryForDomain("rocm_socket0");
            ASSERT_NE(gpu_entry, nullptr);
            EXPECT_EQ(gpu_entry->action.backend, CollectiveBackendType::RCCL);
            EXPECT_EQ(gpu_entry->action.inner_mode, InnerParallelism::LOCAL_TP);
            ASSERT_EQ(gpu_entry->action.devices.size(), 2u);
            EXPECT_TRUE(gpu_entry->action.devices[0].isROCm());
            EXPECT_TRUE(gpu_entry->action.devices[1].isROCm());

            const auto *cpu_entry = orch.stageRunnerEntryForDomain("cpu_sockets");
            ASSERT_NE(cpu_entry, nullptr);
            EXPECT_TRUE(cpu_entry->action.is_global_tp);
            EXPECT_EQ(cpu_entry->action.backend, CollectiveBackendType::UPI);
            EXPECT_EQ(cpu_entry->action.tp_domain_size, 2);
            EXPECT_EQ(cpu_entry->action.device.device_type, DeviceType::CPU);
        }

        {
            MockMPIContext mpi(1, 2);
            auto topo = buildNamedDomainHybridTopo(/*gpu_owner_rank=*/0);

            auto cpu_raw = new MockDeviceRunner();
            auto cpu_runner = std::unique_ptr<MockDeviceRunner>(cpu_raw);

            GlobalOrchestrator::Config config;
            config.topology = topo;
            config.rank = 1;
            config.world_size = 2;
            config.mpi_ctx = &mpi;
            config.stage_runners.push_back(makeStageRunnerEntry(topo, 1, 1, std::move(cpu_runner)));
            config.vocab_size = VOCAB_SIZE;
            config.d_model = D_MODEL;
            config.architecture_name = "test_qwen35moe";

            GlobalOrchestrator orch(std::move(config));

            EXPECT_EQ(orch.stageRunnerCount(), 1u);
            EXPECT_EQ(orch.stageRunnerForDomain("rocm_socket0"), nullptr);
            EXPECT_EQ(orch.stageRunnerForStage(0), nullptr);
            EXPECT_EQ(orch.stageRunnerForDomain("cpu_sockets"), cpu_raw);
            EXPECT_EQ(orch.stageRunnerForStage(1), cpu_raw);
        }
    }

    TEST_F(Test__GlobalOrchestrator, SnapshotKeysAreGlobalizedForPPStageRunners)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildNamedDomainHybridTopo(/*gpu_owner_rank=*/0);

        auto gpu_runner = std::make_unique<MockDeviceRunner>();
        gpu_runner->add_snapshot("layer0_MOE_EXPERT_OUTPUT", {1.0f, 2.0f}, 1, 2);
        auto cpu_runner = std::make_unique<MockDeviceRunner>();
        cpu_runner->add_snapshot("layer0_MOE_EXPERT_OUTPUT", {12.0f, 13.0f}, 1, 2);
        cpu_runner->add_snapshot("layer13_ATTENTION_NORM", {130.0f}, 1, 1);

        GlobalOrchestrator::Config config;
        config.topology = topo;
        config.rank = 0;
        config.world_size = 2;
        config.mpi_ctx = &mpi;
        config.stage_runners.push_back(makeStageRunnerEntry(topo, 0, 0, std::move(gpu_runner)));
        config.stage_runners.push_back(makeStageRunnerEntry(topo, 0, 1, std::move(cpu_runner)));
        config.vocab_size = VOCAB_SIZE;
        config.d_model = D_MODEL;
        config.architecture_name = "test_qwen35moe";

        GlobalOrchestrator orch(std::move(config));
        auto keys = orch.getSnapshotKeys();

        EXPECT_NE(std::find(keys.begin(), keys.end(), "layer0_MOE_EXPERT_OUTPUT"), keys.end());
        EXPECT_NE(std::find(keys.begin(), keys.end(), "layer12_MOE_EXPERT_OUTPUT"), keys.end());
        EXPECT_NE(std::find(keys.begin(), keys.end(), "layer13_ATTENTION_NORM"), keys.end());

        size_t size = 0;
        const float *stage0 = orch.getSnapshot("layer0_MOE_EXPERT_OUTPUT", size);
        ASSERT_NE(stage0, nullptr);
        EXPECT_EQ(size, 2u);
        EXPECT_FLOAT_EQ(stage0[0], 1.0f);

        const float *stage1 = orch.getSnapshot("layer12_MOE_EXPERT_OUTPUT", size);
        ASSERT_NE(stage1, nullptr);
        EXPECT_EQ(size, 2u);
        EXPECT_FLOAT_EQ(stage1[0], 12.0f);

        const float *exact_global = orch.getSnapshot("layer13_ATTENTION_NORM", size);
        ASSERT_NE(exact_global, nullptr);
        EXPECT_EQ(size, 1u);
        EXPECT_FLOAT_EQ(exact_global[0], 130.0f);

        SnapshotInfo shaped = orch.getSnapshotWithShape("layer12_MOE_EXPERT_OUTPUT");
        ASSERT_TRUE(shaped);
        EXPECT_EQ(shaped.rows, 1u);
        EXPECT_EQ(shaped.cols, 2u);
        EXPECT_FLOAT_EQ(shaped.data[1], 13.0f);
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

    // =========================================================================
    // Phase 3: Global TP + PP Composition
    // =========================================================================

    // --- 2PP × 2TP Topology Construction ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank0_IsHeadNotTail)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 2);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank1_IsHeadNotTail)
    {
        MockMPIContext mpi(1, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank2_IsTailNotHead)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank3_IsTailNotHead)
    {
        MockMPIContext mpi(3, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 3, 4, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());
    }

    // --- TP Rank Plan Metadata ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank0_HasCorrectTPMetadata)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);

        const auto &action = *stages[0];
        EXPECT_TRUE(action.is_global_tp);
        EXPECT_EQ(action.tp_rank_in_domain, 0);
        EXPECT_EQ(action.tp_domain_size, 2);
        EXPECT_EQ(action.stage_id, 0);
        EXPECT_EQ(action.first_layer, 0);
        EXPECT_EQ(action.last_layer, 11);
        EXPECT_TRUE(action.has_embedding);
        EXPECT_FALSE(action.has_lm_head);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank1_HasCorrectTPMetadata)
    {
        MockMPIContext mpi(1, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);

        const auto &action = *stages[0];
        EXPECT_TRUE(action.is_global_tp);
        EXPECT_EQ(action.tp_rank_in_domain, 1);
        EXPECT_EQ(action.tp_domain_size, 2);
        EXPECT_EQ(action.stage_id, 0);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank2_HasCorrectTPMetadata)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);

        const auto &action = *stages[0];
        EXPECT_TRUE(action.is_global_tp);
        EXPECT_EQ(action.tp_rank_in_domain, 0);
        EXPECT_EQ(action.tp_domain_size, 2);
        EXPECT_EQ(action.stage_id, 1);
        EXPECT_EQ(action.first_layer, 12);
        EXPECT_EQ(action.last_layer, 23);
        EXPECT_FALSE(action.has_embedding);
        EXPECT_TRUE(action.has_lm_head);
    }

    // --- Weight Shard Info Access ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_WeightShardInfoAccessible)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        const auto *shard = orch.weightShardForStage(0);
        ASSERT_NE(shard, nullptr);
        // GlobalPPRankPlanBuilder populates weight_shard from TP metadata
        EXPECT_EQ(shard->total_shards, 2);
        EXPECT_EQ(shard->shard_index, 0);
        EXPECT_FLOAT_EQ(shard->work_fraction, 0.5f);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank1_WeightShardInfoShard1)
    {
        MockMPIContext mpi(1, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        const auto *shard = orch.weightShardForStage(0);
        ASSERT_NE(shard, nullptr);
        EXPECT_EQ(shard->total_shards, 2);
        EXPECT_EQ(shard->shard_index, 1);
        EXPECT_FLOAT_EQ(shard->work_fraction, 0.5f);
    }

    TEST_F(Test__GlobalOrchestrator, WeightShardForNonExistentStageReturnsNull)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        // Rank 0 doesn't execute stage 1, should return nullptr
        EXPECT_EQ(orch.weightShardForStage(1), nullptr);
        // Stage 99 doesn't exist at all
        EXPECT_EQ(orch.weightShardForStage(99), nullptr);
    }

    // --- Global TP Context Propagation ---

    TEST_F(Test__GlobalOrchestrator, GlobalTPContextPropagation)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        // Create a mock TP context
        MockLocalTPContext::Config tp_config;
        tp_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        auto tp_ctx = std::make_unique<MockLocalTPContext>(tp_config);
        auto *tp_ctx_raw = tp_ctx.get();

        auto config = makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner));
        config.global_tp_ctx = tp_ctx_raw;

        GlobalOrchestrator orch(std::move(config));

        EXPECT_EQ(orch.globalTPContext(), tp_ctx_raw);
    }

    TEST_F(Test__GlobalOrchestrator, GlobalTPContextNullForPurePP)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStagePPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        // Pure PP: no global TP context configured
        EXPECT_EQ(orch.globalTPContext(), nullptr);
    }

    // --- Forward with 2PP × 2TP (disjoint rank sets) ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank0_ForwardExecutesStage)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        // Rank 0 is the designated sender (first in sender domain {0,1}),
        // sends to rank 2 and rank 3 (disjoint receiver domain)
        EXPECT_EQ(mpi.send_call_count(), 2u);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank2_ForwardExecutesStage)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildTwoStageTwoWayTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        // Rank 2 receives hidden state from rank 0 (designated sender)
        EXPECT_EQ(mpi.recv_call_count(), 1u);
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank1_ForwardExecutesStage)
    {
        MockMPIContext mpi(1, 4);
        auto topo = buildTwoStageTwoWayTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        // Rank 1 is NOT the designated sender (only rank 0 fans out),
        // so it has no send operations
        EXPECT_EQ(mpi.send_call_count(), 0u);
    }

    // --- Mixed PP-only + Global TP Topology ---

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank0_IsPureHead)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildMixedPPAndTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        EXPECT_TRUE(orch.isPipelineHead());
        EXPECT_FALSE(orch.isPipelineTail());
        EXPECT_EQ(orch.pipelineDepth(), 3);

        // Stage 0 is NOT global TP (single rank)
        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);
        EXPECT_FALSE(stages[0]->is_global_tp);
        EXPECT_EQ(stages[0]->stage_id, 0);
    }

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank2_IsTPTail)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildMixedPPAndTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());

        // Stage 2 IS global TP
        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);
        EXPECT_TRUE(stages[0]->is_global_tp);
        EXPECT_EQ(stages[0]->tp_rank_in_domain, 0);
        EXPECT_EQ(stages[0]->tp_domain_size, 2);
    }

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank3_IsTPTail)
    {
        MockMPIContext mpi(3, 4);
        auto topo = buildMixedPPAndTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 3, 4, &mpi, std::move(runner)));

        EXPECT_FALSE(orch.isPipelineHead());
        EXPECT_TRUE(orch.isPipelineTail());

        const auto &plan = orch.rankPlan();
        auto stages = plan.executeStages();
        ASSERT_EQ(stages.size(), 1u);
        EXPECT_TRUE(stages[0]->is_global_tp);
        EXPECT_EQ(stages[0]->tp_rank_in_domain, 1);
    }

    // --- Mixed PP+TP Forward Path (exercises real send/recv via fan-out) ---

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank0_ForwardSendsToRank1)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildMixedPPAndTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        // Stage 0 (single) → Stage 1 (single): sends to rank 1
        EXPECT_EQ(mpi.send_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank2_ForwardRecvsAndExecutes)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildMixedPPAndTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        // Stage 1 (single, rank 1) → Stage 2 (TP, ranks 2,3): rank 2 receives
        EXPECT_EQ(mpi.recv_call_count(), 1u);
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank3_ForwardRecvsAndExecutes)
    {
        MockMPIContext mpi(3, 4);
        auto topo = buildMixedPPAndTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 3, 4, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        // Stage 1 (single, rank 1) → Stage 2 (TP, ranks 2,3): rank 3 receives
        EXPECT_EQ(mpi.recv_call_count(), 1u);
        EXPECT_EQ(runner_raw->forward_call_count(), 1u);
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
    }

    // --- Tail Rank Identification with Global TP ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_SamplingBroadcastsFromFirstTailTPRank)
    {
        // With global TP, the tail stage has multiple ranks (2, 3).
        // The first participating rank (2) is the broadcast root.
        // Rank 2 (tail, tp_rank=0): samples + broadcasts
        MockMPIContext mpi(2, 4);
        auto topo = buildTwoStageTwoWayTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.greedy_sample_token = 55;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));
        ASSERT_TRUE(orch.isPipelineTail());

        int token = orch.sampleGreedyOnDevice();
        EXPECT_EQ(token, 55);
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank3_IsTailAndSamples)
    {
        // Rank 3 is also in the tail stage (global TP) with has_lm_head = true
        MockMPIContext mpi(3, 4);
        auto topo = buildTwoStageTwoWayTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.greedy_sample_token = 66;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 3, 4, &mpi, std::move(runner)));
        ASSERT_TRUE(orch.isPipelineTail());

        // Rank 3 is pipeline tail (via global TP), so it samples locally too.
        // In the mock, broadcast is a no-op so the local value is retained.
        int token = orch.sampleGreedyOnDevice();
        EXPECT_EQ(token, 66);
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_Rank0_NonTailSampling)
    {
        // Rank 0 is head (not tail) — should not sample locally
        MockMPIContext mpi(0, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));
        ASSERT_FALSE(orch.isPipelineTail());

        int token = orch.sampleGreedyOnDevice();
        EXPECT_EQ(token, -1); // Non-tail: broadcast no-op in mock
        EXPECT_GE(mpi.broadcast_call_count(), 1u);
    }

    // --- Clear Cache with Global TP ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_ClearCacheBarriers)
    {
        MockMPIContext mpi(1, 4);
        auto topo = buildTwoStageTwoWayTPTopo();
        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        size_t barrier_before = mpi.barrier_call_count();
        orch.clear_cache();
        EXPECT_EQ(runner_raw->clear_cache_call_count(), 1u);
        EXPECT_GT(mpi.barrier_call_count(), barrier_before);
    }

    // --- Weight Shard Info for Non-TP Stages ---

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_SingleRankStage_WeightShardIsUnsharded)
    {
        MockMPIContext mpi(0, 4);
        auto topo = buildMixedPPAndTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 4, &mpi, std::move(runner)));

        const auto *shard = orch.weightShardForStage(0);
        ASSERT_NE(shard, nullptr);
        // Single-rank stage: no sharding
        EXPECT_EQ(shard->total_shards, 1);
        EXPECT_EQ(shard->shard_index, 0);
        EXPECT_FLOAT_EQ(shard->work_fraction, 1.0f);
        EXPECT_FALSE(shard->isSharded());
    }

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_TPStage_WeightShardIsSharded)
    {
        MockMPIContext mpi(2, 4);
        auto topo = buildMixedPPAndTPTopo();
        auto runner = std::make_unique<MockDeviceRunner>();

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 4, &mpi, std::move(runner)));

        const auto *shard = orch.weightShardForStage(2);
        ASSERT_NE(shard, nullptr);
        // Global TP stage with 2 participants
        EXPECT_EQ(shard->total_shards, 2);
        EXPECT_EQ(shard->shard_index, 0);
        EXPECT_FLOAT_EQ(shard->work_fraction, 0.5f);
        EXPECT_TRUE(shard->isSharded());
    }

    // --- 2PP × 2TP Topology Has Correct Fan-Out Transfers (TP→TP, disjoint) ---

    TEST_F(Test__GlobalOrchestrator, TwoPPTwoTP_HasCorrectTransfersForTPToTP)
    {
        // Disjoint rank sets: stage 0 {0,1} → stage 1 {2,3}
        // Rank 0: 2 SEND transfers (to rank 2, to rank 3)
        // Rank 1: 0 transfers (not the designated sender, not a receiver)
        // Rank 2: 1 RECV transfer (from rank 0)
        // Rank 3: 1 RECV transfer (from rank 0)

        auto verify_rank = [this](int rank, int expected_sends, int expected_recvs)
        {
            MockMPIContext mpi(rank, 4);
            auto topo = buildTwoStageTwoWayTPTopo();
            auto runner = std::make_unique<MockDeviceRunner>();

            GlobalOrchestrator orch(makeConfig(std::move(topo), rank, 4, &mpi, std::move(runner)));

            const auto &plan = orch.rankPlan();
            auto transfers = plan.transferActions();

            int send_count = 0, recv_count = 0;
            for (const auto *t : transfers)
            {
                if (t->direction == RankTransferAction::Direction::SEND) send_count++;
                if (t->direction == RankTransferAction::Direction::RECV) recv_count++;
            }
            EXPECT_EQ(send_count, expected_sends) << "rank " << rank << " send count";
            EXPECT_EQ(recv_count, expected_recvs) << "rank " << rank << " recv count";
        };

        verify_rank(0, 2, 0); // Designated sender → rank 2, rank 3
        verify_rank(1, 0, 0); // Not designated sender, not a receiver
        verify_rank(2, 0, 1); // Receiver from rank 0
        verify_rank(3, 0, 1); // Receiver from rank 0
    }

    // --- Mixed PP+TP Topology Has Correct Transfer Structure ---

    TEST_F(Test__GlobalOrchestrator, MixedPPTP_Rank1_HasSendFanOut)
    {
        // Rank 1 (middle, single) sends to both rank 2 and rank 3 (TP stage)
        MockMPIContext mpi(1, 4);
        auto topo = buildMixedPPAndTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner = std::make_unique<MockDeviceRunner>(runner_config);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 4, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();

        // Rank 1: RECV from rank 0 + SEND to rank 2 + SEND to rank 3
        int recv_count = 0, send_count = 0;
        for (const auto *t : transfers)
        {
            if (t->direction == RankTransferAction::Direction::RECV) recv_count++;
            if (t->direction == RankTransferAction::Direction::SEND) send_count++;
        }
        EXPECT_EQ(recv_count, 1); // From rank 0
        EXPECT_EQ(send_count, 2); // To rank 2 and rank 3 (fan-out)
    }

    // --- Same TP Rank Set (local handoff between distinct stages) ---

    TEST_F(Test__GlobalOrchestrator, SameTPTopo_LocalHandoffsForOverlappingRanks)
    {
        // Same rank set {0,1} in both stages. Phase 1 represents the overlap
        // as an explicit local handoff between distinct stage runners.
        for (int rank = 0; rank < 2; ++rank)
        {
            MockMPIContext mpi(rank, 2);
            auto topo = buildTwoStageSameTPTopo();
            auto runner = std::make_unique<MockDeviceRunner>();

            GlobalOrchestrator orch(makeConfig(std::move(topo), rank, 2, &mpi, std::move(runner)));

            const auto &plan = orch.rankPlan();
            auto transfers = plan.transferActions();
            ASSERT_EQ(transfers.size(), 1u) << "rank " << rank << " should have one local handoff";
            EXPECT_EQ(transfers[0]->direction, RankTransferAction::Direction::LOCAL_HANDOFF);
            EXPECT_EQ(transfers[0]->peer_rank, rank);
            EXPECT_EQ(transfers[0]->from_stage, 0);
            EXPECT_EQ(transfers[0]->to_stage, 1);
        }
    }

    TEST_F(Test__GlobalOrchestrator, SameTPTopo_Rank0_ForwardExecutesLocalHandoffCompatibilityRunner)
    {
        MockMPIContext mpi(0, 2);
        auto topo = buildTwoStageSameTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 2, &mpi, std::move(runner)));

        std::vector<int> tokens = {1, 2, 3};
        EXPECT_TRUE(orch.forward(tokens.data(), 3));
        EXPECT_EQ(mpi.send_call_count(), 0u);
        EXPECT_EQ(mpi.recv_call_count(), 0u);
        EXPECT_EQ(runner_raw->forward_call_count(), 2u);
        EXPECT_EQ(runner_raw->set_hidden_state_call_count(), 1u);
    }

    TEST_F(Test__GlobalOrchestrator, SameTPTopo_AnyRankExecutesTwoStageRunnersInOrder)
    {
        for (int rank = 0; rank < 2; ++rank)
        {
            MockMPIContext mpi(rank, 2);
            auto topo = buildTwoStageSameTPTopo();

            MockDeviceRunner::Config stage0_config;
            stage0_config.vocab_size = VOCAB_SIZE;
            stage0_config.device_idx = rank;
            stage0_config.has_hidden_state = true;
            stage0_config.hidden_state_dim = D_MODEL;
            auto stage0_raw = new MockDeviceRunner(stage0_config);
            auto stage0_runner = std::unique_ptr<MockDeviceRunner>(stage0_raw);

            MockDeviceRunner::Config stage1_config;
            stage1_config.vocab_size = VOCAB_SIZE;
            stage1_config.device_idx = rank + 10;
            stage1_config.mock_logits = {0.0f, 42.0f + rank, 3.0f};
            stage1_config.greedy_sample_token = 777 + rank;
            auto stage1_raw = new MockDeviceRunner(stage1_config);
            auto stage1_runner = std::unique_ptr<MockDeviceRunner>(stage1_raw);

            GlobalOrchestrator::Config config;
            config.topology = topo;
            config.rank = rank;
            config.world_size = 2;
            config.mpi_ctx = &mpi;
            config.stage_runners.push_back(makeStageRunnerEntry(topo, rank, 0, std::move(stage0_runner)));
            config.stage_runners.push_back(makeStageRunnerEntry(topo, rank, 1, std::move(stage1_runner)));
            config.vocab_size = VOCAB_SIZE;
            config.d_model = D_MODEL;
            config.architecture_name = "test_qwen2";

            GlobalOrchestrator orch(std::move(config));

            std::vector<int> tokens = {5, 6, 7};
            EXPECT_TRUE(orch.forward(tokens.data(), 3)) << "rank " << rank;

            EXPECT_EQ(stage0_raw->forward_call_count(), 1u) << "rank " << rank;
            EXPECT_EQ(stage1_raw->set_hidden_state_call_count(), 1u) << "rank " << rank;
            EXPECT_TRUE(stage1_raw->was_hidden_state_set()) << "rank " << rank;
            EXPECT_EQ(stage1_raw->forward_call_count(), 1u) << "rank " << rank;
            EXPECT_EQ(mpi.send_call_count(), 0u) << "rank " << rank;
            EXPECT_EQ(mpi.recv_call_count(), 0u) << "rank " << rank;

            TensorBase *handoff = stage1_raw->getHiddenState();
            ASSERT_NE(handoff, nullptr) << "rank " << rank;
            ASSERT_EQ(handoff->numel(), static_cast<size_t>(tokens.size()) * D_MODEL) << "rank " << rank;
            const float *handoff_data = handoff->data();
            EXPECT_FLOAT_EQ(handoff_data[0], static_cast<float>(rank * 1000)) << "rank " << rank;
            EXPECT_FLOAT_EQ(handoff_data[1], static_cast<float>(rank * 1000 + 1)) << "rank " << rank;
            EXPECT_FLOAT_EQ(handoff_data[D_MODEL], static_cast<float>(rank * 1000 + D_MODEL)) << "rank " << rank;

            ASSERT_NE(orch.logits(), nullptr) << "rank " << rank;
            EXPECT_FLOAT_EQ(orch.logits()[1], 42.0f + rank) << "rank " << rank;
            EXPECT_EQ(orch.sampleGreedyOnDevice(), 777 + rank) << "rank " << rank;

            orch.clear_cache();
            EXPECT_EQ(stage0_raw->clear_cache_call_count(), 1u) << "rank " << rank;
            EXPECT_EQ(stage1_raw->clear_cache_call_count(), 1u) << "rank " << rank;
        }
    }

    // --- Partial Overlap TP Rank Sets ---

    TEST_F(Test__GlobalOrchestrator, PartialOverlapTP_Rank2Receives)
    {
        // Rank 2 is in stage 1 {1,2} but NOT in stage 0 {0,1} — needs data
        MockMPIContext mpi(2, 3);
        auto topo = buildPartialOverlapTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 2, 3, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();

        int recv_count = 0;
        for (const auto *t : transfers)
        {
            if (t->direction == RankTransferAction::Direction::RECV) recv_count++;
        }
        EXPECT_EQ(recv_count, 1); // Receives from rank 0
    }

    TEST_F(Test__GlobalOrchestrator, PartialOverlapTP_Rank0Sends)
    {
        // Rank 0 is the first rank in sender domain {0,1} — designated sender
        MockMPIContext mpi(0, 3);
        auto topo = buildPartialOverlapTPTopo();

        MockDeviceRunner::Config runner_config;
        runner_config.vocab_size = VOCAB_SIZE;
        runner_config.has_hidden_state = true;
        runner_config.hidden_state_dim = D_MODEL;
        auto runner_raw = new MockDeviceRunner(runner_config);
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 0, 3, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();

        int send_count = 0;
        for (const auto *t : transfers)
        {
            if (t->direction == RankTransferAction::Direction::SEND) send_count++;
        }
        EXPECT_EQ(send_count, 1); // Sends to rank 2
    }

    TEST_F(Test__GlobalOrchestrator, PartialOverlapTP_Rank1LocalHandoff)
    {
        // Rank 1 is in BOTH domains {0,1} and {1,2}; Phase 1 makes the local
        // handoff explicit, while rank 0 remains the designated MPI sender.
        MockMPIContext mpi(1, 3);
        auto topo = buildPartialOverlapTPTopo();

        auto runner_raw = new MockDeviceRunner();
        auto runner = std::unique_ptr<MockDeviceRunner>(runner_raw);

        GlobalOrchestrator orch(makeConfig(std::move(topo), 1, 3, &mpi, std::move(runner)));

        const auto &plan = orch.rankPlan();
        auto transfers = plan.transferActions();
        ASSERT_EQ(transfers.size(), 1u) << "rank 1 should have one local handoff";
        EXPECT_EQ(transfers[0]->direction, RankTransferAction::Direction::LOCAL_HANDOFF);
        EXPECT_EQ(transfers[0]->peer_rank, 1);
        EXPECT_EQ(transfers[0]->from_stage, 0);
        EXPECT_EQ(transfers[0]->to_stage, 1);
    }

} // namespace llaminar2::test
