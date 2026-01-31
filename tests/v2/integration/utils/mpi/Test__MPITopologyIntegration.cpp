/**
 * @file Test__MPITopologyIntegration.cpp
 * @brief Multi-rank integration tests for MPITopology (Phase 1)
 *
 * Tests the MPITopology class with real MPI communication:
 *   - Capability exchange across ranks
 *   - Consistent work distribution calculations
 *   - Deterministic placement plan computation
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>

#include "utils/MPITopology.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "execution/mpi_orchestration/PlacementPlan.h"
#include "tensors/TensorSlice.h"

using namespace llaminar2;

/**
 * @brief Test fixture for MPITopology multi-rank integration tests
 */
class Test__MPITopologyIntegration : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MPITopology> topology_;
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Create topology - this triggers capability exchange
        topology_ = std::make_unique<MPITopology>(MPI_COMM_WORLD);
    }

    void TearDown() override
    {
        topology_.reset();
        mpi_ctx_->barrier();
    }
};

// =============================================================================
// Basic Topology Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, AllRanksHaveSameWorldSize)
{
    // Every rank should see the same world_size
    int local_world_size = topology_->world_size();

    // Gather all values to rank 0 and verify consistency
    std::vector<int> all_world_sizes(world_size_);
    MPI_Allgather(&local_world_size, 1, MPI_INT,
                  all_world_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < world_size_; ++i)
    {
        EXPECT_EQ(all_world_sizes[i], world_size_)
            << "Rank " << i << " has inconsistent world_size";
    }
}

TEST_F(Test__MPITopologyIntegration, RanksHaveUniqueIDs)
{
    // Each rank should have a unique rank ID
    std::vector<int> all_ranks(world_size_);
    int my_rank = topology_->rank();

    MPI_Allgather(&my_rank, 1, MPI_INT,
                  all_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Check uniqueness
    std::set<int> unique_ranks(all_ranks.begin(), all_ranks.end());
    EXPECT_EQ(unique_ranks.size(), static_cast<size_t>(world_size_));

    // Check range [0, world_size)
    for (int r : all_ranks)
    {
        EXPECT_GE(r, 0);
        EXPECT_LT(r, world_size_);
    }
}

TEST_F(Test__MPITopologyIntegration, CapabilityExchangeCompletesSuccessfully)
{
    // After construction, all ranks should have placement info for all ranks
    const auto &all_placements = topology_->all_placements();

    EXPECT_EQ(all_placements.size(), static_cast<size_t>(world_size_))
        << "Rank " << rank_ << " should have placement info for all " << world_size_ << " ranks";

    // Verify each placement has valid data
    for (int r = 0; r < world_size_; ++r)
    {
        const auto &placement = topology_->get_placement(r);
        EXPECT_EQ(placement.rank, r) << "Placement for rank " << r << " has wrong rank ID";
        EXPECT_GE(placement.node_id, 0) << "Rank " << r << " has invalid node_id";
        EXPECT_GE(placement.local_rank, 0) << "Rank " << r << " has invalid local_rank";
    }
}

// =============================================================================
// Work Distribution Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, HeadRangesPartitionCompletely)
{
    // Test that head ranges from all ranks cover all heads exactly once
    constexpr int TOTAL_HEADS = 14; // Qwen2.5 0.5B has 14 heads

    WorkRange my_range = topology_->get_head_range(TOTAL_HEADS);

    // Gather all ranges
    std::vector<size_t> all_starts(world_size_), all_ends(world_size_);
    size_t my_start = my_range.start, my_end = my_range.end;

    MPI_Allgather(&my_start, 1, MPI_UNSIGNED_LONG,
                  all_starts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&my_end, 1, MPI_UNSIGNED_LONG,
                  all_ends.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // Verify complete partition
    std::vector<bool> covered(TOTAL_HEADS, false);
    for (int r = 0; r < world_size_; ++r)
    {
        for (size_t h = all_starts[r]; h < all_ends[r]; ++h)
        {
            EXPECT_FALSE(covered[h]) << "Head " << h << " assigned to multiple ranks";
            covered[h] = true;
        }
    }

    for (int h = 0; h < TOTAL_HEADS; ++h)
    {
        EXPECT_TRUE(covered[h]) << "Head " << h << " not assigned to any rank";
    }
}

TEST_F(Test__MPITopologyIntegration, ColumnRangesPartitionCompletely)
{
    // Test column-parallel work distribution
    constexpr size_t TOTAL_COLS = 896; // d_model for Qwen2.5 0.5B

    WorkRange my_range = topology_->get_column_range(TOTAL_COLS);

    std::vector<size_t> all_starts(world_size_), all_ends(world_size_);
    size_t my_start = my_range.start, my_end = my_range.end;

    MPI_Allgather(&my_start, 1, MPI_UNSIGNED_LONG,
                  all_starts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&my_end, 1, MPI_UNSIGNED_LONG,
                  all_ends.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // Verify ranges are contiguous and non-overlapping
    EXPECT_EQ(all_starts[0], 0u) << "First rank should start at 0";
    EXPECT_EQ(all_ends[world_size_ - 1], TOTAL_COLS) << "Last rank should end at TOTAL_COLS";

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_starts[r], all_ends[r - 1])
            << "Ranks " << (r - 1) << " and " << r << " don't have contiguous ranges";
    }
}

TEST_F(Test__MPITopologyIntegration, RowRangesPartitionCompletely)
{
    // Test row-parallel work distribution (for row-parallel GEMM)
    constexpr size_t TOTAL_ROWS = 1024;

    WorkRange my_range = topology_->get_row_range(TOTAL_ROWS);

    std::vector<size_t> all_starts(world_size_), all_ends(world_size_);
    size_t my_start = my_range.start, my_end = my_range.end;

    MPI_Allgather(&my_start, 1, MPI_UNSIGNED_LONG,
                  all_starts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&my_end, 1, MPI_UNSIGNED_LONG,
                  all_ends.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // Compute total coverage
    size_t total_coverage = 0;
    for (int r = 0; r < world_size_; ++r)
    {
        total_coverage += (all_ends[r] - all_starts[r]);
    }
    EXPECT_EQ(total_coverage, TOTAL_ROWS) << "Total coverage should equal TOTAL_ROWS";
}

TEST_F(Test__MPITopologyIntegration, WorkRangesAreBalanced)
{
    // Work should be distributed approximately equally
    constexpr size_t TOTAL_WORK = 1000;

    WorkRange my_range = topology_->get_column_range(TOTAL_WORK);
    size_t my_work = my_range.size();

    std::vector<size_t> all_work(world_size_);
    MPI_Allgather(&my_work, 1, MPI_UNSIGNED_LONG,
                  all_work.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // Find min and max work
    size_t min_work = *std::min_element(all_work.begin(), all_work.end());
    size_t max_work = *std::max_element(all_work.begin(), all_work.end());

    // Allow at most 1 element difference (due to integer division)
    EXPECT_LE(max_work - min_work, 1u)
        << "Work should be balanced (max diff <= 1)";
}

// =============================================================================
// Placement Plan Consistency Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, PlacementPlanIsDeterministicAcrossRanks)
{
    // All ranks should compute the exact same PlacementPlan
    // This is critical: if plans differ, ranks will have inconsistent views!

    // Create placement input for a small model
    PlacementPlan my_plan = topology_->computePlacement(
        "qwen2",           // architecture
        24,                // n_layers
        896,               // d_model
        4864,              // d_ff
        151936,            // vocab_size
        14,                // n_heads
        2,                 // n_kv_heads
        "Q4_0",            // quant_type
        500 * 1024 * 1024, // estimated_memory (500MB)
        ""                 // strategy_name (auto-select)
    );

    // Gather plan metadata from all ranks
    struct PlanSummary
    {
        int n_layers;
        int world_size;
        int strategy_hash; // Hash of strategy name
        int first_layer_device;
        int last_layer_device;
    };

    auto hash_string = [](const std::string &s)
    {
        int h = 0;
        for (char c : s)
            h = h * 31 + c;
        return h;
    };

    // Convert PlacementDevice to an int for comparison
    auto device_to_int = [](const PlacementDevice &d)
    {
        return (d.type << 8) | d.gpu_index;
    };

    PlanSummary my_summary{
        my_plan.n_layers,
        my_plan.world_size,
        hash_string(my_plan.strategy_name),
        my_plan.layers.empty() ? 0 : device_to_int(my_plan.layers[0].device),
        my_plan.layers.empty() ? 0 : device_to_int(my_plan.layers.back().device)};

    std::vector<PlanSummary> all_summaries(world_size_);
    MPI_Allgather(&my_summary, sizeof(PlanSummary), MPI_BYTE,
                  all_summaries.data(), sizeof(PlanSummary), MPI_BYTE, MPI_COMM_WORLD);

    // All summaries must match
    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_summaries[r].n_layers, all_summaries[0].n_layers)
            << "Rank " << r << " has different n_layers";
        EXPECT_EQ(all_summaries[r].world_size, all_summaries[0].world_size)
            << "Rank " << r << " has different world_size";
        EXPECT_EQ(all_summaries[r].strategy_hash, all_summaries[0].strategy_hash)
            << "Rank " << r << " selected different strategy";
        EXPECT_EQ(all_summaries[r].first_layer_device, all_summaries[0].first_layer_device)
            << "Rank " << r << " has different first layer device";
        EXPECT_EQ(all_summaries[r].last_layer_device, all_summaries[0].last_layer_device)
            << "Rank " << r << " has different last layer device";
    }
}

TEST_F(Test__MPITopologyIntegration, CPUOnlyStrategySelectedWithoutGPU)
{
    // Without GPUs, CPUOnlyStrategy should be auto-selected
    PlacementPlan plan = topology_->computePlacement(
        "qwen2", 24, 896, 4864, 151936, 14, 2, "Q4_0",
        500 * 1024 * 1024, "");

    // Should use CPUOnly strategy
    EXPECT_EQ(plan.strategy_name, "CPUOnly")
        << "Without GPUs, CPUOnly strategy should be selected";

    // All layers should be on CPU
    for (const auto &layer : plan.layers)
    {
        EXPECT_TRUE(layer.device.isCPU())
            << "Layer " << layer.layer_idx << " should be on CPU";
    }
}

// =============================================================================
// GQA Head Distribution Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, KVHeadRangesHandleGQA)
{
    // Test KV head distribution for GQA where n_kv_heads < n_heads
    constexpr int N_HEADS = 14;
    constexpr int N_KV_HEADS = 2;

    WorkRange q_range = topology_->get_head_range(N_HEADS);
    WorkRange kv_range = topology_->get_kv_head_range(N_KV_HEADS);

    // Log for debugging
    if (rank_ == 0)
    {
        LOG_INFO("[Test] world_size=" << world_size_
                                      << ", Q heads range: [" << q_range.start << ", " << q_range.end << ")"
                                      << ", KV heads range: [" << kv_range.start << ", " << kv_range.end << ")");
    }

    // If world_size > N_KV_HEADS, some ranks will have empty KV ranges
    // This is expected behavior for GQA
    if (world_size_ <= N_KV_HEADS)
    {
        EXPECT_GT(kv_range.size(), 0u)
            << "With world_size <= N_KV_HEADS, each rank should get some KV heads";
    }

    // Verify total KV coverage across all ranks
    std::vector<size_t> all_kv_sizes(world_size_);
    size_t my_kv_size = kv_range.size();
    MPI_Allgather(&my_kv_size, 1, MPI_UNSIGNED_LONG,
                  all_kv_sizes.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    size_t total_kv_coverage = 0;
    for (size_t s : all_kv_sizes)
        total_kv_coverage += s;

    EXPECT_EQ(total_kv_coverage, static_cast<size_t>(N_KV_HEADS))
        << "Total KV head coverage should equal N_KV_HEADS";
}

// =============================================================================
// SliceMetadata Creation Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, RowParallelMetadataIsConsistent)
{
    // Create row-parallel metadata and verify consistency
    constexpr size_t ROWS = 1024;
    constexpr size_t COLS = 896;

    SliceMetadata meta = topology_->createRowParallelMeta(ROWS, COLS, false);

    // Gather all metadata using the correct SliceMetadata fields
    struct MetaSummary
    {
        size_t original_rows;
        size_t original_cols;
        size_t slice_start;
        size_t slice_end;
        size_t slice_size;
    };

    MetaSummary my_meta{
        meta.original_rows,
        meta.original_cols,
        meta.slice_start,
        meta.slice_end,
        meta.slice_size()};

    std::vector<MetaSummary> all_meta(world_size_);
    MPI_Allgather(&my_meta, sizeof(MetaSummary), MPI_BYTE,
                  all_meta.data(), sizeof(MetaSummary), MPI_BYTE, MPI_COMM_WORLD);

    // All ranks should agree on original dimensions
    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_meta[r].original_rows, ROWS);
        EXPECT_EQ(all_meta[r].original_cols, COLS);
    }

    // Slice sizes should sum to original rows (row-parallel splits rows)
    size_t total_slice_size = 0;
    for (int r = 0; r < world_size_; ++r)
    {
        total_slice_size += all_meta[r].slice_size;
    }
    EXPECT_EQ(total_slice_size, ROWS);

    // Verify contiguous non-overlapping row coverage
    EXPECT_EQ(all_meta[0].slice_start, 0u);
    EXPECT_EQ(all_meta[world_size_ - 1].slice_end, ROWS);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_meta[r].slice_start, all_meta[r - 1].slice_end)
            << "Row slices should be contiguous between rank " << (r - 1) << " and " << r;
    }
}

TEST_F(Test__MPITopologyIntegration, ColumnParallelMetadataIsConsistent)
{
    // Create column-parallel metadata and verify consistency
    constexpr size_t ROWS = 896;
    constexpr size_t COLS = 4864; // FFN dimension

    SliceMetadata meta = topology_->createColumnParallelMeta(ROWS, COLS, false);

    // Gather all metadata using the correct SliceMetadata fields
    std::vector<size_t> all_slice_starts(world_size_), all_slice_ends(world_size_);
    size_t my_slice_start = meta.slice_start;
    size_t my_slice_end = meta.slice_end;

    MPI_Allgather(&my_slice_start, 1, MPI_UNSIGNED_LONG,
                  all_slice_starts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&my_slice_end, 1, MPI_UNSIGNED_LONG,
                  all_slice_ends.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // Verify contiguous non-overlapping column coverage
    EXPECT_EQ(all_slice_starts[0], 0u);
    EXPECT_EQ(all_slice_ends[world_size_ - 1], COLS);

    for (int r = 1; r < world_size_; ++r)
    {
        EXPECT_EQ(all_slice_starts[r], all_slice_ends[r - 1])
            << "Column slices should be contiguous between rank " << (r - 1) << " and " << r;
    }
}

// =============================================================================
// Coordinator and Node Topology Tests
// =============================================================================

TEST_F(Test__MPITopologyIntegration, ExactlyOneCoordinator)
{
    // Exactly one rank should be the coordinator
    int is_coordinator = topology_->is_coordinator() ? 1 : 0;
    int total_coordinators = 0;

    MPI_Allreduce(&is_coordinator, &total_coordinators, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(total_coordinators, 1) << "There should be exactly one coordinator";

    // Coordinator should be rank 0
    if (rank_ == 0)
    {
        EXPECT_TRUE(topology_->is_coordinator());
    }
    else
    {
        EXPECT_FALSE(topology_->is_coordinator());
    }
}

TEST_F(Test__MPITopologyIntegration, AllRanksAreComputeParticipantsByDefault)
{
    // By default, all ranks should participate in compute
    int is_participant = topology_->is_compute_participant() ? 1 : 0;
    int total_participants = 0;

    MPI_Allreduce(&is_participant, &total_participants, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(total_participants, world_size_)
        << "All ranks should be compute participants by default";
}

TEST_F(Test__MPITopologyIntegration, ComputeWorldSizeMatchesParticipants)
{
    // compute_world_size() should match actual participant count
    int compute_world = topology_->compute_world_size();

    // Gather to verify all ranks agree
    std::vector<int> all_compute_sizes(world_size_);
    MPI_Allgather(&compute_world, 1, MPI_INT,
                  all_compute_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 0; r < world_size_; ++r)
    {
        EXPECT_EQ(all_compute_sizes[r], world_size_)
            << "Rank " << r << " reports different compute_world_size";
    }
}

// MPI-aware main function
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
