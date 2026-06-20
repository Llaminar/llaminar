/**
 * @file Test__NodeLeaderPageCache.cpp
 * @brief Unit tests for node-leader page cache coordination logic
 *
 * Tests that the page cache prepopulation strategy in OrchestrationRunner
 * correctly uses topology information to elect per-node leaders, and falls
 * back to rank-0 when topology is unavailable.
 *
 * This tests the decision logic in isolation using MockMPIContext +
 * MockMPITopology — no real MPI or I/O occurs.
 */

#include <gtest/gtest.h>

#include "mocks/MockMPIContext.h"
#include "mocks/MockMPITopology.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

// =========================================================================
// Helper: Simulate the page cache coordination logic from OrchestrationRunner
// =========================================================================
// This mirrors the logic in OrchestrationRunner::loadWeights() without the
// side effects (no real prepopulatePageCache or MPI_Barrier calls).

struct PageCacheDecision
{
    bool should_prepopulate = false;   ///< This rank should call prepopulatePageCache
    bool should_skip_cache_eviction = false; ///< This rank should keep the warmed page cache.
    bool used_intra_node_barrier = false; ///< Used intra-node comm for barrier
    bool used_world_barrier = false;      ///< Used world communicator for barrier
    bool used_fallback = false;           ///< Used rank-0 fallback path (no topology)
};

PageCacheDecision decide_page_cache_strategy(const IMPIContext *mpi_ctx, bool use_mmap, bool target_is_gpu)
{
    PageCacheDecision decision;
    (void)target_is_gpu;

    const bool is_multi_rank = mpi_ctx && mpi_ctx->world_size() > 1;
    if (!use_mmap)
        return decision;

    if (!is_multi_rank)
    {
        decision.should_prepopulate = true;
        decision.should_skip_cache_eviction = true;
        return decision;
    }

    const auto *topo = mpi_ctx->topology();
    if (topo)
    {
        if (topo->is_node_leader())
        {
            decision.should_prepopulate = true;
        }

        MPI_Comm intra = mpi_ctx->intra_node_comm();
        if (intra != MPI_COMM_NULL)
            decision.used_intra_node_barrier = true;
        else
            decision.used_world_barrier = true;
    }
    else
    {
        decision.used_fallback = true;
        if (mpi_ctx->rank() == 0)
        {
            decision.should_prepopulate = true;
        }
        decision.used_world_barrier = true;
    }

    decision.should_skip_cache_eviction = true;
    return decision;
}

// =========================================================================
// Test fixture
// =========================================================================

class Test__NodeLeaderPageCache : public ::testing::Test
{
};

// =========================================================================
// Tests: Topology-aware path
// =========================================================================

TEST(Test__NodeLeaderPageCache, NodeLeader_Prepopulates)
{
    // Rank 0 on node 0 — is node leader (local_rank = 0 % 2 = 0)
    auto topo = MockMPITopology::createSimple(/*rank=*/0, /*world_size=*/4,
                                               /*ranks_per_node=*/2);
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/4);
    ctx->set_topology(topo, MPI_COMM_WORLD); // Non-null intra comm

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_TRUE(d.should_prepopulate);
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_TRUE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_world_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, NonLeader_DoesNotPrepopulate)
{
    // Rank 1 on node 0 — local_rank = 1 % 2 = 1, NOT node leader
    auto topo = MockMPITopology::createSimple(/*rank=*/1, /*world_size=*/4,
                                               /*ranks_per_node=*/2);
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/1, /*world_size=*/4);
    ctx->set_topology(topo, MPI_COMM_WORLD);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_FALSE(d.should_prepopulate);
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_TRUE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_world_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, SecondNodeLeader_Prepopulates)
{
    // Rank 2 on node 1 — local_rank = 2 % 2 = 0, IS node leader for node 1
    auto topo = MockMPITopology::createSimple(/*rank=*/2, /*world_size=*/4,
                                               /*ranks_per_node=*/2);
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/2, /*world_size=*/4);
    ctx->set_topology(topo, MPI_COMM_WORLD);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_TRUE(d.should_prepopulate);
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_TRUE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, TopologyWithNullIntraComm_FallsToWorldBarrier)
{
    // Topology present but intra-node comm not configured → world barrier
    auto topo = MockMPITopology::createSimple(/*rank=*/0, /*world_size=*/2,
                                               /*ranks_per_node=*/2);
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/2);
    ctx->set_topology(topo); // No intra_comm → MPI_COMM_NULL default

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_TRUE(d.should_prepopulate); // Still node leader
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_FALSE(d.used_intra_node_barrier);
    EXPECT_TRUE(d.used_world_barrier);  // Falls back to world
    EXPECT_FALSE(d.used_fallback);       // Not the rank-0 fallback path
}

// =========================================================================
// Tests: Fallback path (no topology)
// =========================================================================

TEST(Test__NodeLeaderPageCache, NoTopology_Rank0Prepopulates)
{
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/4);
    // No topology set — topology() returns nullptr

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_TRUE(d.should_prepopulate);
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_TRUE(d.used_fallback);
    EXPECT_TRUE(d.used_world_barrier);
}

TEST(Test__NodeLeaderPageCache, NoTopology_NonRootDoesNotPrepopulate)
{
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/2, /*world_size=*/4);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_FALSE(d.should_prepopulate);
    EXPECT_TRUE(d.should_skip_cache_eviction);
    EXPECT_TRUE(d.used_fallback);
    EXPECT_TRUE(d.used_world_barrier);
}

// =========================================================================
// Tests: Edge cases
// =========================================================================

TEST(Test__NodeLeaderPageCache, SingleRankGpu_PrepopulatesAndSkipsEviction)
{
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/true);

    EXPECT_TRUE(d.should_prepopulate)
        << "single-rank GPU mmap must warm the page cache before tensor-sized upload faults";
    EXPECT_TRUE(d.should_skip_cache_eviction)
        << "MmapRegion must not evict the just-warmed cache before GPU staging copies";
    EXPECT_FALSE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_world_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, SingleRankCpu_PrepopulatesAndSkipsEviction)
{
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/1);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/true, /*target_is_gpu=*/false);

    EXPECT_TRUE(d.should_prepopulate)
        << "single-rank CPU mmap must warm the page cache before NUMA first-touch";
    EXPECT_TRUE(d.should_skip_cache_eviction)
        << "MmapRegion must not evict the just-warmed cache before first-touch";
    EXPECT_FALSE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_world_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, MmapDisabled_NoAction)
{
    auto ctx = std::make_shared<MockMPIContext>(/*rank=*/0, /*world_size=*/4);
    auto topo = MockMPITopology::createSimple(0, 4, 2);
    ctx->set_topology(topo, MPI_COMM_WORLD);

    auto d = decide_page_cache_strategy(ctx.get(), /*use_mmap=*/false, /*target_is_gpu=*/false);

    EXPECT_FALSE(d.should_prepopulate);
    EXPECT_FALSE(d.should_skip_cache_eviction);
    EXPECT_FALSE(d.used_intra_node_barrier);
    EXPECT_FALSE(d.used_world_barrier);
    EXPECT_FALSE(d.used_fallback);
}

TEST(Test__NodeLeaderPageCache, NullContext_TreatedAsSingleProcess)
{
    auto d = decide_page_cache_strategy(nullptr, /*use_mmap=*/true, /*target_is_gpu=*/true);

    EXPECT_TRUE(d.should_prepopulate)
        << "a missing MPI context is still a single-process mmap load";
    EXPECT_TRUE(d.should_skip_cache_eviction);
}

// =========================================================================
// Tests: Multi-node simulation — verify per-node independence
// =========================================================================

TEST(Test__NodeLeaderPageCache, MultiNode_EachNodeLeaderPrepopulates)
{
    // Simulate 4 ranks across 2 nodes:
    //   Node 0: rank 0 (leader, local_rank=0), rank 1 (local_rank=1)
    //   Node 1: rank 2 (leader, local_rank=0), rank 3 (local_rank=1)
    struct RankConfig
    {
        int rank;
        bool expect_prepopulate;
    };

    const RankConfig configs[] = {
        {0, true},   // Node 0 leader (local_rank=0)
        {1, false},  // Node 0 follower (local_rank=1)
        {2, true},   // Node 1 leader (local_rank=0)
        {3, false},  // Node 1 follower (local_rank=1)
    };

    for (const auto &c : configs)
    {
        auto topo = MockMPITopology::createSimple(c.rank, 4, /*ranks_per_node=*/2);
        auto ctx = std::make_shared<MockMPIContext>(c.rank, 4);
        ctx->set_topology(topo, MPI_COMM_WORLD);

        auto d = decide_page_cache_strategy(ctx.get(), true, /*target_is_gpu=*/false);

        EXPECT_EQ(d.should_prepopulate, c.expect_prepopulate)
            << "rank=" << c.rank;
        EXPECT_TRUE(d.should_skip_cache_eviction)
            << "rank=" << c.rank;
        EXPECT_TRUE(d.used_intra_node_barrier)
            << "rank=" << c.rank;
        EXPECT_FALSE(d.used_fallback)
            << "rank=" << c.rank;
    }
}

} // anonymous namespace
