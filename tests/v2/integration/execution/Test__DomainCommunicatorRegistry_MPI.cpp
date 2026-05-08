/**
 * @file Test__DomainCommunicatorRegistry_MPI.cpp
 * @brief MPI integration tests for DomainCommunicatorRegistry (Phase 3)
 *
 * Runs with 3 MPI ranks. Topology has two global TP stages with overlapping
 * but different participant sets:
 *   - Stage 0: participants {0, 1}
 *   - Stage 1: participants {1, 2}
 *
 * Expected outcomes:
 *   - Rank 0: has context for stage 0, no context for stage 1
 *   - Rank 1: has contexts for stage 0 AND stage 1
 *   - Rank 2: has context for stage 1, no context for stage 0
 *
 * For each context held:
 *   - worldRanks() exactly matches the stage's participating_ranks
 *   - degree() equals the number of participants in that stage
 *   - domainId() equals the stage_id
 *
 * Acceptance: No rank receives a context for a stage it does not participate in.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/global/DomainCommunicatorRegistry.h"
#include "execution/global_pp/GlobalPPTopology.h"

namespace llaminar2::test
{

    // =========================================================================
    // Helper: Build topology with two overlapping global TP stages
    // =========================================================================

    static GlobalPPTopology buildOverlappingTopo()
    {
        // Stage 0: global TP, participants {0, 1}, layers 0-11
        GlobalPPStageSpec s0;
        s0.stage_id = 0;
        s0.domain_name = "domain_0";
        s0.first_layer = 0;
        s0.last_layer = 11;
        s0.has_embedding = true;
        s0.has_lm_head = false;
        s0.is_global_tp = true;
        s0.participating_ranks = {0, 1};
        s0.per_rank_device = GlobalDeviceAddress::cpu();

        // Stage 1: global TP, participants {1, 2}, layers 12-23
        GlobalPPStageSpec s1;
        s1.stage_id = 1;
        s1.domain_name = "domain_1";
        s1.first_layer = 12;
        s1.last_layer = 23;
        s1.has_embedding = false;
        s1.has_lm_head = true;
        s1.is_global_tp = true;
        s1.participating_ranks = {1, 2};
        s1.per_rank_device = GlobalDeviceAddress::cpu();

        return GlobalPPTopology::build({s0, s1}, /*total_layers=*/24, /*world_size=*/3);
    }

    // =========================================================================
    // Tests
    // =========================================================================

    class Test__DomainCommunicatorRegistry_MPI : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            topo_ = buildOverlappingTopo();
        }

        int rank_ = -1;
        int world_size_ = -1;
        GlobalPPTopology topo_;
    };

    // -------------------------------------------------------------------------
    // Context presence
    // -------------------------------------------------------------------------

    TEST_F(Test__DomainCommunicatorRegistry_MPI, Rank0_HasStage0_NoStage1)
    {
        ASSERT_EQ(world_size_, 3) << "Test requires exactly 3 MPI ranks";

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 0)
        {
            EXPECT_TRUE(registry.hasContextForStage(0));
            EXPECT_FALSE(registry.hasContextForStage(1));

            auto ids = registry.stageIds();
            ASSERT_EQ(ids.size(), 1u);
            EXPECT_EQ(ids[0], 0);
        }
    }

    TEST_F(Test__DomainCommunicatorRegistry_MPI, Rank1_HasBothStages)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 1)
        {
            EXPECT_TRUE(registry.hasContextForStage(0));
            EXPECT_TRUE(registry.hasContextForStage(1));

            auto ids = registry.stageIds();
            ASSERT_EQ(ids.size(), 2u);
        }
    }

    TEST_F(Test__DomainCommunicatorRegistry_MPI, Rank2_HasStage1_NoStage0)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 2)
        {
            EXPECT_FALSE(registry.hasContextForStage(0));
            EXPECT_TRUE(registry.hasContextForStage(1));

            auto ids = registry.stageIds();
            ASSERT_EQ(ids.size(), 1u);
            EXPECT_EQ(ids[0], 1);
        }
    }

    // -------------------------------------------------------------------------
    // Context correctness: worldRanks(), degree(), domainId()
    // -------------------------------------------------------------------------

    TEST_F(Test__DomainCommunicatorRegistry_MPI, Stage0_ContextCorrect)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 0 || rank_ == 1)
        {
            auto ctx = registry.globalTPContextForStage(0);
            ASSERT_NE(ctx, nullptr);

            // Domain should have exactly ranks {0, 1}
            EXPECT_EQ(ctx->degree(), 2);
            EXPECT_EQ(ctx->domainId(), 0);

            const auto &world_ranks = ctx->worldRanks();
            ASSERT_EQ(world_ranks.size(), 2u);
            EXPECT_EQ(world_ranks[0], 0);
            EXPECT_EQ(world_ranks[1], 1);
        }
    }

    TEST_F(Test__DomainCommunicatorRegistry_MPI, Stage1_ContextCorrect)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 1 || rank_ == 2)
        {
            auto ctx = registry.globalTPContextForStage(1);
            ASSERT_NE(ctx, nullptr);

            // Domain should have exactly ranks {1, 2}
            EXPECT_EQ(ctx->degree(), 2);
            EXPECT_EQ(ctx->domainId(), 1);

            const auto &world_ranks = ctx->worldRanks();
            ASSERT_EQ(world_ranks.size(), 2u);
            EXPECT_EQ(world_ranks[0], 1);
            EXPECT_EQ(world_ranks[1], 2);
        }
    }

    // -------------------------------------------------------------------------
    // myIndex correctness: rank 0 should be index 0 in stage 0,
    //                      rank 1 should be index 1 in stage 0 and index 0 in stage 1,
    //                      rank 2 should be index 1 in stage 1
    // -------------------------------------------------------------------------

    TEST_F(Test__DomainCommunicatorRegistry_MPI, MyIndex_Deterministic)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 0)
        {
            auto ctx = registry.globalTPContextForStage(0);
            ASSERT_NE(ctx, nullptr);
            EXPECT_EQ(ctx->myIndex(), 0); // rank 0 is index 0 in participants {0,1}
        }
        if (rank_ == 1)
        {
            auto ctx0 = registry.globalTPContextForStage(0);
            ASSERT_NE(ctx0, nullptr);
            EXPECT_EQ(ctx0->myIndex(), 1); // rank 1 is index 1 in participants {0,1}

            auto ctx1 = registry.globalTPContextForStage(1);
            ASSERT_NE(ctx1, nullptr);
            EXPECT_EQ(ctx1->myIndex(), 0); // rank 1 is index 0 in participants {1,2}
        }
        if (rank_ == 2)
        {
            auto ctx = registry.globalTPContextForStage(1);
            ASSERT_NE(ctx, nullptr);
            EXPECT_EQ(ctx->myIndex(), 1); // rank 2 is index 1 in participants {1,2}
        }
    }

    // -------------------------------------------------------------------------
    // Negative lookup: nullptr for non-participant stages
    // -------------------------------------------------------------------------

    TEST_F(Test__DomainCommunicatorRegistry_MPI, NullForNonParticipant)
    {
        ASSERT_EQ(world_size_, 3);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo_, MPI_COMM_WORLD, rank_);

        if (rank_ == 0)
        {
            EXPECT_EQ(registry.globalTPContextForStage(1), nullptr);
            EXPECT_EQ(registry.globalTPContextForStage(99), nullptr);
        }
        if (rank_ == 2)
        {
            EXPECT_EQ(registry.globalTPContextForStage(0), nullptr);
            EXPECT_EQ(registry.globalTPContextForStage(99), nullptr);
        }
    }

    // -------------------------------------------------------------------------
    // Empty topology: no global TP stages, no contexts
    // -------------------------------------------------------------------------

    TEST_F(Test__DomainCommunicatorRegistry_MPI, EmptyTopology_NoContexts)
    {
        // Single-rank single-stage topology with no global TP
        GlobalPPStageSpec s;
        s.stage_id = 0;
        s.first_layer = 0;
        s.last_layer = 11;
        s.has_embedding = true;
        s.has_lm_head = true;
        s.is_global_tp = false;
        s.owning_rank = 0;
        s.inner_mode = InnerParallelism::SINGLE_DEVICE;
        s.devices = {GlobalDeviceAddress::cpu()};

        // Build with world_size = actual world size
        auto topo = GlobalPPTopology::build({s}, /*total_layers=*/12, world_size_);

        DomainCommunicatorRegistry registry;
        registry.initialize(topo, MPI_COMM_WORLD, rank_);

        EXPECT_TRUE(registry.stageIds().empty());
        EXPECT_FALSE(registry.hasContextForStage(0));
        EXPECT_EQ(registry.globalTPContextForStage(0), nullptr);
    }

} // namespace llaminar2::test
