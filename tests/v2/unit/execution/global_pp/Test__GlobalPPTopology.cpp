/**
 * @file Test__GlobalPPTopology.cpp
 * @brief Unit tests for GlobalPPTopology and GlobalPPRankPlanBuilder
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for Global PP topology and rank plan building:
 * - Topology validation (layer coverage, contiguity, embedding/lm_head)
 * - Transfer derivation (single-rank to single-rank, single-rank to global TP)
 * - Rank plan building (correct step ordering, transfer matching)
 * - Edge cases (single stage, all-global-TP, 2-stage minimal)
 *
 * All tests are pure data-structure tests — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <initializer_list>
#include <vector>

#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/global_pp/GlobalPPRankPlan.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"

using namespace llaminar2;

// =============================================================================
// Helper: Build a standard 3-stage topology
// =============================================================================

/**
 * Creates the canonical test topology:
 *   Stage 0: rank=0, layers 0-9,   2x CUDA (LocalTP), +embedding
 *   Stage 1: rank=1, layers 10-19, 2x ROCm (LocalTP)
 *   Stage 2: global TP (ranks 0,1), layers 20-23, CPU, +lm_head
 */
static GlobalPPTopology buildCanonical3Stage()
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 9;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;
    s0.inner_mode = InnerParallelism::LOCAL_TP;
    s0.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 10;
    s1.last_layer = 19;
    s1.is_global_tp = false;
    s1.owning_rank = 1;
    s1.inner_mode = InnerParallelism::LOCAL_TP;
    s1.devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};

    GlobalPPStageSpec s2;
    s2.stage_id = 2;
    s2.first_layer = 20;
    s2.last_layer = 23;
    s2.has_lm_head = true;
    s2.is_global_tp = true;
    s2.participating_ranks = {0, 1};
    s2.per_rank_device = GlobalDeviceAddress::cpu(0);

    return GlobalPPTopology::build({s0, s1, s2}, 24, 2);
}

static GlobalPPStageSpec makeLocalTPStage(int stage_id,
                                          int first_layer,
                                          int last_layer,
                                          int owning_rank,
                                          std::initializer_list<GlobalDeviceAddress> devices,
                                          bool has_embedding = false,
                                          bool has_lm_head = false)
{
    GlobalPPStageSpec spec;
    spec.stage_id = stage_id;
    spec.first_layer = first_layer;
    spec.last_layer = last_layer;
    spec.has_embedding = has_embedding;
    spec.has_lm_head = has_lm_head;
    spec.is_global_tp = false;
    spec.owning_rank = owning_rank;
    spec.inner_mode = InnerParallelism::LOCAL_TP;
    spec.devices = devices;
    return spec;
}

static GlobalPPStageSpec makeNodeLocalCpuTPStage(int stage_id,
                                                 int first_layer,
                                                 int last_layer,
                                                 std::initializer_list<int> participating_ranks,
                                                 bool has_embedding = false,
                                                 bool has_lm_head = false)
{
    GlobalPPStageSpec spec;
    spec.stage_id = stage_id;
    spec.first_layer = first_layer;
    spec.last_layer = last_layer;
    spec.has_embedding = has_embedding;
    spec.has_lm_head = has_lm_head;
    spec.is_global_tp = true;
    spec.participating_ranks = participating_ranks;
    spec.per_rank_device = GlobalDeviceAddress::cpu(0);
    return spec;
}

static std::vector<int> executeStageIds(const GlobalPPRankPlan &plan)
{
    std::vector<int> ids;
    for (const auto *stage : plan.executeStages())
    {
        ids.push_back(stage->stage_id);
    }
    return ids;
}

static size_t countTransferActions(const GlobalPPRankPlan &plan, int from_stage, int to_stage)
{
    size_t count = 0;
    for (const auto *transfer : plan.transferActions())
    {
        if (transfer->from_stage == from_stage && transfer->to_stage == to_stage)
        {
            ++count;
        }
    }
    return count;
}

static size_t countTransferActions(const GlobalPPRankPlan &plan,
                                   int from_stage,
                                   int to_stage,
                                   RankTransferAction::Direction direction,
                                   int peer_rank)
{
    size_t count = 0;
    for (const auto *transfer : plan.transferActions())
    {
        if (transfer->from_stage == from_stage &&
            transfer->to_stage == to_stage &&
            transfer->direction == direction &&
            transfer->peer_rank == peer_rank)
        {
            ++count;
        }
    }
    return count;
}

// =============================================================================
// Topology Validation Tests
// =============================================================================

class Test__GlobalPPTopology : public ::testing::Test
{
};

/**
 * @test Valid 3-stage topology passes validation
 */
TEST_F(Test__GlobalPPTopology, ValidTopologyPasses)
{
    auto topo = buildCanonical3Stage();
    auto errors = topo.validate();

    EXPECT_TRUE(errors.empty()) << "Errors: ";
    for (const auto &e : errors)
    {
        std::cerr << "  " << e << "\n";
    }
}

/**
 * @test Topology has correct number of stages and total layers
 */
TEST_F(Test__GlobalPPTopology, BasicProperties)
{
    auto topo = buildCanonical3Stage();

    EXPECT_EQ(topo.numStages(), 3);
    EXPECT_EQ(topo.total_layers, 24);
    EXPECT_EQ(topo.world_size, 2);
}

/**
 * @test Layer coverage: each layer is in exactly one stage
 */
TEST_F(Test__GlobalPPTopology, LayerCoverage)
{
    auto topo = buildCanonical3Stage();

    for (int l = 0; l < 24; ++l)
    {
        auto *stage = topo.stageForLayer(l);
        ASSERT_NE(stage, nullptr) << "Layer " << l << " not found in any stage";

        if (l <= 9)
            EXPECT_EQ(stage->stage_id, 0);
        else if (l <= 19)
            EXPECT_EQ(stage->stage_id, 1);
        else
            EXPECT_EQ(stage->stage_id, 2);
    }

    // Layer outside range
    EXPECT_EQ(topo.stageForLayer(24), nullptr);
    EXPECT_EQ(topo.stageForLayer(-1), nullptr);
}

/**
 * @test Stages for rank: rank 0 participates in stages 0 and 2
 */
TEST_F(Test__GlobalPPTopology, StagesForRank0)
{
    auto topo = buildCanonical3Stage();
    auto stages = topo.stagesForRank(0);

    ASSERT_EQ(stages.size(), 2u);
    EXPECT_EQ(stages[0], 0);
    EXPECT_EQ(stages[1], 2);
}

/**
 * @test Stages for rank: rank 1 participates in stages 1 and 2
 */
TEST_F(Test__GlobalPPTopology, StagesForRank1)
{
    auto topo = buildCanonical3Stage();
    auto stages = topo.stagesForRank(1);

    ASSERT_EQ(stages.size(), 2u);
    EXPECT_EQ(stages[0], 1);
    EXPECT_EQ(stages[1], 2);
}

/**
 * @test Rank participation query
 */
TEST_F(Test__GlobalPPTopology, RankParticipation)
{
    auto topo = buildCanonical3Stage();

    EXPECT_TRUE(topo.rankParticipatesInStage(0, 0));
    EXPECT_FALSE(topo.rankParticipatesInStage(1, 0));
    EXPECT_FALSE(topo.rankParticipatesInStage(0, 1));
    EXPECT_TRUE(topo.rankParticipatesInStage(1, 1));
    EXPECT_TRUE(topo.rankParticipatesInStage(0, 2));
    EXPECT_TRUE(topo.rankParticipatesInStage(1, 2));
}

/**
 * @test Embedding is on first stage, LM head on last
 */
TEST_F(Test__GlobalPPTopology, EmbeddingAndLMHead)
{
    auto topo = buildCanonical3Stage();

    EXPECT_TRUE(topo.stages[0].has_embedding);
    EXPECT_FALSE(topo.stages[0].has_lm_head);
    EXPECT_FALSE(topo.stages[1].has_embedding);
    EXPECT_FALSE(topo.stages[1].has_lm_head);
    EXPECT_FALSE(topo.stages[2].has_embedding);
    EXPECT_TRUE(topo.stages[2].has_lm_head);
}

/**
 * @test toString produces non-empty output
 */
TEST_F(Test__GlobalPPTopology, ToStringNotEmpty)
{
    auto topo = buildCanonical3Stage();
    auto str = topo.toString();

    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("GlobalPPTopology"), std::string::npos);
    EXPECT_NE(str.find("total_layers=24"), std::string::npos);
}

// =============================================================================
// Validation Error Tests
// =============================================================================

/**
 * @test Validation catches missing embedding on first stage
 */
TEST_F(Test__GlobalPPTopology, ValidationCatchesMissingEmbedding)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 3;
    s0.has_embedding = false; // Missing!
    s0.has_lm_head = true;
    s0.owning_rank = 0;

    auto topo = GlobalPPTopology::build({s0}, 4, 1);
    auto errors = topo.validate();

    bool found_embedding_error = false;
    for (const auto &e : errors)
    {
        if (e.find("has_embedding") != std::string::npos)
        {
            found_embedding_error = true;
        }
    }
    EXPECT_TRUE(found_embedding_error);
}

/**
 * @test Validation catches layer gap between stages
 */
TEST_F(Test__GlobalPPTopology, ValidationCatchesLayerGap)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 3;
    s0.has_embedding = true;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 5; // Gap: layer 4 is missing!
    s1.last_layer = 7;
    s1.has_lm_head = true;
    s1.owning_rank = 1;

    auto topo = GlobalPPTopology::build({s0, s1}, 8, 2);
    auto errors = topo.validate();

    EXPECT_FALSE(errors.empty());
}

/**
 * @test Validation catches global TP with < 2 ranks
 */
TEST_F(Test__GlobalPPTopology, ValidationCatchesGlobalTPTooFewRanks)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 3;
    s0.has_embedding = true;
    s0.has_lm_head = true;
    s0.is_global_tp = true;
    s0.participating_ranks = {0}; // Only 1 rank!
    s0.per_rank_device = GlobalDeviceAddress::cpu(0);

    auto topo = GlobalPPTopology::build({s0}, 4, 2);
    auto errors = topo.validate();

    bool found_tp_error = false;
    for (const auto &e : errors)
    {
        if (e.find("participating ranks") != std::string::npos)
        {
            found_tp_error = true;
        }
    }
    EXPECT_TRUE(found_tp_error);
}

/**
 * @test Validation catches owning_rank >= world_size
 */
TEST_F(Test__GlobalPPTopology, ValidationCatchesRankOutOfRange)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 3;
    s0.has_embedding = true;
    s0.has_lm_head = true;
    s0.owning_rank = 5; // world_size is only 2!

    auto topo = GlobalPPTopology::build({s0}, 4, 2);
    auto errors = topo.validate();

    bool found_rank_error = false;
    for (const auto &e : errors)
    {
        if (e.find("world_size") != std::string::npos)
        {
            found_rank_error = true;
        }
    }
    EXPECT_TRUE(found_rank_error);
}

// =============================================================================
// Transfer Derivation Tests
// =============================================================================

/**
 * @test Transfers are derived for single-rank → single-rank transitions
 */
TEST_F(Test__GlobalPPTopology, TransferSingleToSingle)
{
    // 2 stages: rank 0 → rank 1
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.owning_rank = 1;

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2);

    ASSERT_EQ(topo.transfers.size(), 1u);
    EXPECT_EQ(topo.transfers[0].sender_rank, 0);
    EXPECT_EQ(topo.transfers[0].receiver_rank, 1);
    EXPECT_FALSE(topo.transfers[0].isNoop());
}

/**
 * @test Same-rank adjacent stages create an explicit local handoff
 */
TEST_F(Test__GlobalPPTopology, TransferSameRankCreatesLocalHandoff)
{
    // 2 stages both owned by rank 0 — no MPI transfer needed, but the
    // distinct stage runners still need an explicit local handoff.
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.owning_rank = 0;

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 1);

    ASSERT_EQ(topo.transfers.size(), 1u);
    EXPECT_EQ(topo.transfers[0].kind, GlobalPPTransferKind::LOCAL_HANDOFF);
    EXPECT_EQ(topo.transfers[0].sender_rank, 0);
    EXPECT_EQ(topo.transfers[0].receiver_rank, 0);
    EXPECT_FALSE(topo.transfers[0].isNoop());
}

/**
 * @test Transfers: single-rank → global TP creates fan-out
 */
TEST_F(Test__GlobalPPTopology, TransferSingleToGlobalTP)
{
    auto topo = buildCanonical3Stage();

    // Stage 0 (rank 0) → Stage 1 (rank 1): 1 transfer
    // Stage 1 (rank 1) → Stage 2 (global TP, ranks 0,1): fan-out from rank 1 to rank 0
    // Total: at least 2 non-noop transfers

    int non_noop = 0;
    for (const auto &t : topo.transfers)
    {
        if (!t.isNoop())
        {
            non_noop++;
        }
    }
    EXPECT_GE(non_noop, 2);
}

// =============================================================================
// GlobalPPStageSpec Tests
// =============================================================================

/**
 * @test Stage layer count
 */
TEST(Test__GlobalPPStageSpec, LayerCount)
{
    GlobalPPStageSpec spec;
    spec.first_layer = 5;
    spec.last_layer = 14;

    EXPECT_EQ(spec.layerCount(), 10);
}

/**
 * @test Stage hasLayer
 */
TEST(Test__GlobalPPStageSpec, HasLayer)
{
    GlobalPPStageSpec spec;
    spec.first_layer = 5;
    spec.last_layer = 14;

    EXPECT_FALSE(spec.hasLayer(4));
    EXPECT_TRUE(spec.hasLayer(5));
    EXPECT_TRUE(spec.hasLayer(10));
    EXPECT_TRUE(spec.hasLayer(14));
    EXPECT_FALSE(spec.hasLayer(15));
}

/**
 * @test innerParallelismName returns correct strings
 */
TEST(Test__GlobalPPStageSpec, InnerParallelismName)
{
    EXPECT_STREQ(innerParallelismName(InnerParallelism::SINGLE_DEVICE), "SINGLE_DEVICE");
    EXPECT_STREQ(innerParallelismName(InnerParallelism::LOCAL_TP), "LOCAL_TP");
    EXPECT_STREQ(innerParallelismName(InnerParallelism::LOCAL_PP), "LOCAL_PP");
}

/**
 * @test globalPPTransferKindName returns correct strings
 */
TEST(Test__GlobalPPTransferKind, GlobalPPTransferKindName)
{
    EXPECT_STREQ(globalPPTransferKindName(GlobalPPTransferKind::MPI), "MPI");
    EXPECT_STREQ(globalPPTransferKindName(GlobalPPTransferKind::LOCAL_HANDOFF), "LOCAL_HANDOFF");
}

/**
 * @test Domain names appear in topology and rank-plan output
 */
TEST_F(Test__GlobalPPTopology, DomainNamesAppearInOutputAndRankPlan)
{
    auto s0 = makeLocalTPStage(0, 0, 11, 0,
                               {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                               true,
                               false);
    s0.domain_name = "gpu_fast";

    auto s1 = makeNodeLocalCpuTPStage(1, 12, 23, {0, 1}, false, true);
    s1.domain_name = "cpu_tail";

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    const auto topo_string = topo.toString();
    EXPECT_NE(topo_string.find("domain=gpu_fast"), std::string::npos);
    EXPECT_NE(topo_string.find("domain=cpu_tail"), std::string::npos);

    const auto topo_table = topo.toTable();
    EXPECT_NE(topo_table.find("Domain"), std::string::npos);
    EXPECT_NE(topo_table.find("gpu_fast"), std::string::npos);
    EXPECT_NE(topo_table.find("cpu_tail"), std::string::npos);
    EXPECT_NE(topo_table.find("LOCAL_HANDOFF"), std::string::npos);

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto exec_stages = plan0.executeStages();
    ASSERT_EQ(exec_stages.size(), 2u);
    EXPECT_EQ(exec_stages[0]->domain_name, "gpu_fast");
    EXPECT_EQ(exec_stages[1]->domain_name, "cpu_tail");

    const auto plan_string = plan0.toString();
    EXPECT_NE(plan_string.find("domain=gpu_fast"), std::string::npos);
    EXPECT_NE(plan_string.find("domain=cpu_tail"), std::string::npos);
}

// =============================================================================
// Rank Plan Builder Tests
// =============================================================================

class Test__GlobalPPRankPlanBuilder : public ::testing::Test
{
};

/**
 * @test Rank 0 plan for canonical 3-stage topology
 *
 * Expected:
 *   1. EXECUTE stage 0 (layers 0-9, +embedding)
 *   2. SEND to rank 1 (stage 0 → stage 1)
 *   3. RECV from rank 1 (stage 1 → stage 2, if rank 0 doesn't have data)
 *   4. EXECUTE stage 2 (layers 20-23, global TP, +lm_head)
 */
TEST_F(Test__GlobalPPRankPlanBuilder, Rank0Plan)
{
    auto topo = buildCanonical3Stage();
    auto plan = GlobalPPRankPlanBuilder::build(topo, 0);

    EXPECT_EQ(plan.rank, 0);
    EXPECT_TRUE(plan.hasWork());

    // Should have at least 2 execute stages (stage 0 and stage 2)
    auto exec_stages = plan.executeStages();
    ASSERT_GE(exec_stages.size(), 2u);

    // First execute: stage 0 with embedding
    EXPECT_EQ(exec_stages[0]->stage_id, 0);
    EXPECT_EQ(exec_stages[0]->first_layer, 0);
    EXPECT_EQ(exec_stages[0]->last_layer, 9);
    EXPECT_TRUE(exec_stages[0]->has_embedding);
    EXPECT_FALSE(exec_stages[0]->is_global_tp);

    // Last execute: stage 2 with lm_head, global TP
    EXPECT_EQ(exec_stages.back()->stage_id, 2);
    EXPECT_EQ(exec_stages.back()->first_layer, 20);
    EXPECT_EQ(exec_stages.back()->last_layer, 23);
    EXPECT_TRUE(exec_stages.back()->has_lm_head);
    EXPECT_TRUE(exec_stages.back()->is_global_tp);
    EXPECT_EQ(exec_stages.back()->tp_rank_in_domain, 0);
    EXPECT_EQ(exec_stages.back()->tp_domain_size, 2);
}

/**
 * @test Rank 1 plan for canonical 3-stage topology
 *
 * Expected:
 *   1. RECV from rank 0 (stage 0 → stage 1)
 *   2. EXECUTE stage 1 (layers 10-19)
 *   3. SEND to rank 0 (stage 1 → stage 2, fan-out since stage 2 is global TP)
 *   4. EXECUTE stage 2 (layers 20-23, global TP)
 */
TEST_F(Test__GlobalPPRankPlanBuilder, Rank1Plan)
{
    auto topo = buildCanonical3Stage();
    auto plan = GlobalPPRankPlanBuilder::build(topo, 1);

    EXPECT_EQ(plan.rank, 1);
    EXPECT_TRUE(plan.hasWork());

    // Should have 2 execute stages (stage 1 and stage 2)
    auto exec_stages = plan.executeStages();
    ASSERT_GE(exec_stages.size(), 2u);

    // First execute: stage 1
    EXPECT_EQ(exec_stages[0]->stage_id, 1);
    EXPECT_EQ(exec_stages[0]->first_layer, 10);
    EXPECT_EQ(exec_stages[0]->last_layer, 19);
    EXPECT_FALSE(exec_stages[0]->has_embedding);
    EXPECT_FALSE(exec_stages[0]->has_lm_head);

    // Second execute: stage 2 (global TP)
    EXPECT_EQ(exec_stages[1]->stage_id, 2);
    EXPECT_TRUE(exec_stages[1]->is_global_tp);
    EXPECT_EQ(exec_stages[1]->tp_rank_in_domain, 1); // rank 1 is second in TP
}

/**
 * @test Transfer actions: rank 0 sends before stage 1, rank 1 receives
 */
TEST_F(Test__GlobalPPRankPlanBuilder, TransferMatching)
{
    auto topo = buildCanonical3Stage();
    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    auto transfers0 = plan0.transferActions();
    auto transfers1 = plan1.transferActions();

    // Rank 0 should have at least one SEND (stage 0 → stage 1)
    bool rank0_sends = false;
    for (auto *t : transfers0)
    {
        if (t->direction == RankTransferAction::Direction::SEND && t->peer_rank == 1)
        {
            rank0_sends = true;
        }
    }
    EXPECT_TRUE(rank0_sends) << "Rank 0 should send to rank 1";

    // Rank 1 should have at least one RECV (from rank 0, stage 0 → stage 1)
    bool rank1_recvs = false;
    for (auto *t : transfers1)
    {
        if (t->direction == RankTransferAction::Direction::RECV && t->peer_rank == 0)
        {
            rank1_recvs = true;
        }
    }
    EXPECT_TRUE(rank1_recvs) << "Rank 1 should receive from rank 0";
}

/**
 * @test Phase 0 red test: LocalTP rank 0 -> node-local CPU TP ranks 0 and 1.
 *
 * Rank 0 needs a local handoff into its CPU shard, alongside the SEND to rank 1.
 */
TEST_F(Test__GlobalPPRankPlanBuilder, Phase0_LocalTPToNodeLocalCpuTPRequiresSourceLocalHandoff)
{
    auto s0 = makeLocalTPStage(0, 0, 11, 0,
                               {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                               true,
                               false);
    auto s1 = makeNodeLocalCpuTPStage(1, 12, 23, {0, 1}, false, true);

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    EXPECT_EQ(executeStageIds(plan0), std::vector<int>({0, 1}));
    EXPECT_EQ(countTransferActions(plan0, 0, 1), 2u)
        << "Rank 0 needs the existing SEND plus a local handoff action for 0 -> 1";
    EXPECT_EQ(countTransferActions(plan0, 0, 1, RankTransferAction::Direction::SEND, 1), 1u);
    EXPECT_EQ(countTransferActions(plan0, 0, 1, RankTransferAction::Direction::LOCAL_HANDOFF, 0), 1u);

    EXPECT_EQ(executeStageIds(plan1), std::vector<int>({1}));
    EXPECT_EQ(countTransferActions(plan1, 0, 1, RankTransferAction::Direction::RECV, 0), 1u);
}

/**
 * @test Phase 0 red test: LocalTP ROCm rank 0 -> LocalTP CUDA rank 0 -> node-local CPU TP.
 *
 * Rank 0 should execute all three stage runners and have transfer actions for
 * both local domain handoffs plus the fan-out SEND to rank 1.
 */
TEST_F(Test__GlobalPPRankPlanBuilder, Phase0_ThreeStageRank0OwnershipRequiresTwoLocalHandoffs)
{
    auto s0 = makeLocalTPStage(0, 0, 7, 0,
                               {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                               true,
                               false);
    auto s1 = makeLocalTPStage(1, 8, 15, 0,
                               {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});
    auto s2 = makeNodeLocalCpuTPStage(2, 16, 23, {0, 1}, false, true);

    auto topo = GlobalPPTopology::build({s0, s1, s2}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    EXPECT_EQ(executeStageIds(plan0), std::vector<int>({0, 1, 2}));
    EXPECT_EQ(plan0.transferActions().size(), 3u)
        << "Rank 0 needs local handoff 0 -> 1, local handoff 1 -> 2, and SEND 1 -> 2 to rank 1";
    EXPECT_EQ(countTransferActions(plan0, 0, 1, RankTransferAction::Direction::LOCAL_HANDOFF, 0), 1u);
    EXPECT_EQ(countTransferActions(plan0, 1, 2, RankTransferAction::Direction::LOCAL_HANDOFF, 0), 1u);
    EXPECT_EQ(countTransferActions(plan0, 1, 2, RankTransferAction::Direction::SEND, 1), 1u);

    EXPECT_EQ(executeStageIds(plan1), std::vector<int>({2}));
    EXPECT_EQ(countTransferActions(plan1, 1, 2, RankTransferAction::Direction::RECV, 0), 1u);
}

/**
 * @test Phase 0 red test: mirrored socket ownership has rank 1 local GPU domains.
 *
 * Rank 1 should execute the node-local CPU shard and both local GPU TP stages,
 * with local handoff actions between 0 -> 1 and 1 -> 2. Rank 0 should execute
 * only stage 0; it does not need to send to rank 1 because rank 1 already
 * participates in the source node-local TP stage.
 */
TEST_F(Test__GlobalPPRankPlanBuilder, Phase0_MirroredSocketLocalityRequiresRank1LocalHandoffs)
{
    auto s0 = makeNodeLocalCpuTPStage(0, 0, 7, {0, 1}, true, false);
    auto s1 = makeLocalTPStage(1, 8, 15, 1,
                               {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    auto s2 = makeLocalTPStage(2, 16, 23, 1,
                               {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                               false,
                               true);

    auto topo = GlobalPPTopology::build({s0, s1, s2}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    EXPECT_EQ(executeStageIds(plan1), std::vector<int>({0, 1, 2}));
    EXPECT_EQ(plan1.transferActions().size(), 2u)
        << "Rank 1 needs local handoff actions for 0 -> 1 and 1 -> 2";
    EXPECT_EQ(countTransferActions(plan1, 0, 1, RankTransferAction::Direction::LOCAL_HANDOFF, 1), 1u);
    EXPECT_EQ(countTransferActions(plan1, 1, 2, RankTransferAction::Direction::LOCAL_HANDOFF, 1), 1u);

    EXPECT_EQ(executeStageIds(plan0), std::vector<int>({0}));
    EXPECT_EQ(countTransferActions(plan0, 0, 1), 0u)
        << "Rank 0 should not send because rank 1 already has stage 0 output from its node-local TP shard";
}

/**
 * @test Weight shard info for global TP stages
 */
TEST_F(Test__GlobalPPRankPlanBuilder, WeightShardForGlobalTP)
{
    auto topo = buildCanonical3Stage();
    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    auto exec0 = plan0.executeStages();
    auto exec1 = plan1.executeStages();

    // Find stage 2 in each plan
    const RankStageAction *gtp0 = nullptr;
    const RankStageAction *gtp1 = nullptr;
    for (auto *s : exec0) if (s->stage_id == 2) gtp0 = s;
    for (auto *s : exec1) if (s->stage_id == 2) gtp1 = s;

    ASSERT_NE(gtp0, nullptr);
    ASSERT_NE(gtp1, nullptr);

    // Rank 0: shard 0 of 2
    EXPECT_EQ(gtp0->weight_shard.shard_index, 0);
    EXPECT_EQ(gtp0->weight_shard.total_shards, 2);
    EXPECT_NEAR(gtp0->weight_shard.work_fraction, 0.5f, 0.01f);

    // Rank 1: shard 1 of 2
    EXPECT_EQ(gtp1->weight_shard.shard_index, 1);
    EXPECT_EQ(gtp1->weight_shard.total_shards, 2);
    EXPECT_NEAR(gtp1->weight_shard.work_fraction, 0.5f, 0.01f);
}

/**
 * @test Single-rank stages have no weight sharding
 */
TEST_F(Test__GlobalPPRankPlanBuilder, NoShardForSingleRankStage)
{
    auto topo = buildCanonical3Stage();
    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);

    auto exec = plan0.executeStages();
    ASSERT_GE(exec.size(), 1u);

    // Stage 0 is single-rank
    EXPECT_EQ(exec[0]->weight_shard.shard_index, 0);
    EXPECT_EQ(exec[0]->weight_shard.total_shards, 1);
    EXPECT_NEAR(exec[0]->weight_shard.work_fraction, 1.0f, 0.01f);
}

/**
 * @test Plan toString produces non-empty output
 */
TEST_F(Test__GlobalPPRankPlanBuilder, ToStringNotEmpty)
{
    auto topo = buildCanonical3Stage();
    auto plan = GlobalPPRankPlanBuilder::build(topo, 0);

    auto str = plan.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("GlobalPPRankPlan"), std::string::npos);
}

// =============================================================================
// Edge Cases
// =============================================================================

/**
 * @test Simple 2-stage pipeline (rank 0 → rank 1), no global TP
 */
TEST_F(Test__GlobalPPRankPlanBuilder, TwoStagePipeline)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.owning_rank = 1;

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    // Rank 0: execute stage 0, then send
    EXPECT_EQ(plan0.executeStages().size(), 1u);
    EXPECT_EQ(plan0.executeStages()[0]->stage_id, 0);

    // Rank 1: recv, then execute stage 1
    EXPECT_EQ(plan1.executeStages().size(), 1u);
    EXPECT_EQ(plan1.executeStages()[0]->stage_id, 1);

    // Verify transfer matching
    auto t0 = plan0.transferActions();
    auto t1 = plan1.transferActions();
    EXPECT_GE(t0.size(), 1u); // rank 0 sends
    EXPECT_GE(t1.size(), 1u); // rank 1 receives
}

/**
 * @test All-global-TP: single stage with all ranks
 */
TEST_F(Test__GlobalPPRankPlanBuilder, AllGlobalTP)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 23;
    s0.has_embedding = true;
    s0.has_lm_head = true;
    s0.is_global_tp = true;
    s0.participating_ranks = {0, 1};
    s0.per_rank_device = GlobalDeviceAddress::cpu(0);

    auto topo = GlobalPPTopology::build({s0}, 24, 2);
    EXPECT_TRUE(topo.validate().empty());

    auto plan0 = GlobalPPRankPlanBuilder::build(topo, 0);
    auto plan1 = GlobalPPRankPlanBuilder::build(topo, 1);

    // Both ranks execute all layers, no transfers needed
    EXPECT_EQ(plan0.executeStages().size(), 1u);
    EXPECT_EQ(plan1.executeStages().size(), 1u);
    EXPECT_TRUE(plan0.transferActions().empty());
    EXPECT_TRUE(plan1.transferActions().empty());
}

// =============================================================================
// Rank Locality Tests
// =============================================================================

/**
 * @test Two ranks on the same node → all transfers are INTRA_NODE
 */
TEST_F(Test__GlobalPPTopology, BuildWithLocalities_SameNode)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = false;
    s1.owning_rank = 1;

    std::vector<RankLocality> localities = {
        {0, "node-alpha", 0},
        {1, "node-alpha", 0},
    };
    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2, localities);

    ASSERT_EQ(topo.rank_localities.size(), 2u);
    EXPECT_TRUE(topo.areColocated(0, 1));
    EXPECT_EQ(topo.nodeCount(), 1);

    ASSERT_EQ(topo.transfers.size(), 1u);
    EXPECT_EQ(topo.transfers[0].locality, TransferLocality::INTRA_NODE);
}

/**
 * @test Two ranks on different nodes → transfers are INTER_NODE
 */
TEST_F(Test__GlobalPPTopology, BuildWithLocalities_DifferentNodes)
{
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = false;
    s1.owning_rank = 1;

    std::vector<RankLocality> localities = {
        {0, "node-alpha", 0},
        {1, "node-beta", 1},
    };
    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2, localities);

    EXPECT_FALSE(topo.areColocated(0, 1));
    EXPECT_EQ(topo.nodeCount(), 2);

    ASSERT_EQ(topo.transfers.size(), 1u);
    EXPECT_EQ(topo.transfers[0].locality, TransferLocality::INTER_NODE);
}

/**
 * @test No localities provided → existing build() path → locality stays UNKNOWN
 */
TEST_F(Test__GlobalPPTopology, BuildWithoutLocalities_TransfersAreUnknown)
{
    auto topo = buildCanonical3Stage();
    for (const auto &t : topo.transfers)
    {
        EXPECT_EQ(t.locality, TransferLocality::UNKNOWN);
    }
}

/**
 * @test areColocated returns false when no localities provided
 */
TEST_F(Test__GlobalPPTopology, AreColocated_NoLocalities)
{
    auto topo = buildCanonical3Stage();
    EXPECT_FALSE(topo.areColocated(0, 1));
}

/**
 * @test nodeCount returns 0 when no localities provided
 */
TEST_F(Test__GlobalPPTopology, NodeCount_NoLocalities)
{
    auto topo = buildCanonical3Stage();
    EXPECT_EQ(topo.nodeCount(), 0);
}

/**
 * @test ranksOnNode returns correct groupings for multi-node topology
 */
TEST_F(Test__GlobalPPTopology, RanksOnNode)
{
    std::vector<RankLocality> localities = {
        {0, "host-a", 0}, {1, "host-a", 0},
        {2, "host-b", 1}, {3, "host-b", 1},
    };

    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = true;
    s0.participating_ranks = {0, 1};
    s0.per_rank_device = GlobalDeviceAddress::cpu(0);

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = true;
    s1.participating_ranks = {2, 3};
    s1.per_rank_device = GlobalDeviceAddress::cpu(0);

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 4, localities);

    auto node0_ranks = topo.ranksOnNode(0);
    ASSERT_EQ(node0_ranks.size(), 2u);
    EXPECT_EQ(node0_ranks[0], 0);
    EXPECT_EQ(node0_ranks[1], 1);

    auto node1_ranks = topo.ranksOnNode(1);
    ASSERT_EQ(node1_ranks.size(), 2u);
    EXPECT_EQ(node1_ranks[0], 2);
    EXPECT_EQ(node1_ranks[1], 3);

    EXPECT_EQ(topo.nodeCount(), 2);
    EXPECT_TRUE(topo.areColocated(0, 1));
    EXPECT_TRUE(topo.areColocated(2, 3));
    EXPECT_FALSE(topo.areColocated(0, 2));
    EXPECT_FALSE(topo.areColocated(1, 3));
}

/**
 * @test Fan-out transfers with mixed locality (single → global TP, cross-node)
 */
TEST_F(Test__GlobalPPTopology, FanOutTransfers_MixedLocality)
{
    std::vector<RankLocality> localities = {
        {0, "node-a", 0},
        {1, "node-b", 1},
        {2, "node-b", 1},
    };

    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;
    s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
    s0.devices = {GlobalDeviceAddress::cpu(0)};

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = true;
    s1.participating_ranks = {1, 2};
    s1.per_rank_device = GlobalDeviceAddress::cpu(0);

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 3, localities);

    // Fan-out: 2 transfers (rank 0 → rank 1, rank 0 → rank 2)
    ASSERT_EQ(topo.transfers.size(), 2u);
    for (const auto &t : topo.transfers)
    {
        EXPECT_EQ(t.sender_rank, 0);
        EXPECT_EQ(t.locality, TransferLocality::INTER_NODE);
    }
}

/**
 * @test Fan-out transfers with all ranks on same node → INTRA_NODE
 */
TEST_F(Test__GlobalPPTopology, FanOutTransfers_IntraNodeLocality)
{
    std::vector<RankLocality> localities = {
        {0, "same-host", 0},
        {1, "same-host", 0},
        {2, "same-host", 0},
    };

    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;
    s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
    s0.devices = {GlobalDeviceAddress::cpu(0)};

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = true;
    s1.participating_ranks = {1, 2};
    s1.per_rank_device = GlobalDeviceAddress::cpu(0);

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 3, localities);

    ASSERT_EQ(topo.transfers.size(), 2u);
    for (const auto &t : topo.transfers)
    {
        EXPECT_EQ(t.locality, TransferLocality::INTRA_NODE);
    }
}

/**
 * @test toString includes locality and node info when localities are present
 */
TEST_F(Test__GlobalPPTopology, ToStringIncludesLocality)
{
    std::vector<RankLocality> localities = {
        {0, "host-a", 0},
        {1, "host-b", 1},
    };

    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;
    s0.inner_mode = InnerParallelism::SINGLE_DEVICE;
    s0.devices = {GlobalDeviceAddress::cpu(0)};

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_lm_head = true;
    s1.is_global_tp = false;
    s1.owning_rank = 1;
    s1.inner_mode = InnerParallelism::SINGLE_DEVICE;
    s1.devices = {GlobalDeviceAddress::cpu(0)};

    auto topo = GlobalPPTopology::build({s0, s1}, 24, 2, localities);

    std::string str = topo.toString();
    EXPECT_NE(str.find("INTER_NODE"), std::string::npos);
    EXPECT_NE(str.find("host-a"), std::string::npos);
    EXPECT_NE(str.find("host-b"), std::string::npos);
}

/**
 * @test transferLocalityName returns correct strings
 */
TEST(Test__TransferLocality, TransferLocalityName)
{
    EXPECT_STREQ(transferLocalityName(TransferLocality::INTRA_NODE), "INTRA_NODE");
    EXPECT_STREQ(transferLocalityName(TransferLocality::INTER_NODE), "INTER_NODE");
    EXPECT_STREQ(transferLocalityName(TransferLocality::UNKNOWN), "UNKNOWN");
}
