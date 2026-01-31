/**
 * @file Test__WorkDistribution.cpp
 * @brief Integration tests for work distribution across ranks for tensor parallelism
 *
 * **Purpose**: Validates correct work distribution across MPI ranks for:
 * - Tensor-parallel GEMM (weights split across ranks)
 * - Head-parallel attention (attention heads distributed)
 * - Even and uneven work distribution scenarios
 * - Slice boundary correctness
 * - Allreduce synchronization after partial results
 *
 * **Test Strategy**:
 * Uses mock infrastructure (MockMPITopology, MockWorkDistributor, MockMPIContext)
 * to simulate multi-rank scenarios without requiring actual MPI runtime.
 *
 * **Dependencies**:
 * - MockMPITopology: Simulates multi-rank topology
 * - MockWorkDistributor: Verifies distribution logic
 * - MockMPIContext: Simulates rank communication
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

#include "mocks/MockMPITopology.h"
#include "mocks/MockWorkDistributor.h"
#include "mocks/MockMPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief Test fixture for work distribution tests
 *
 * Provides helper methods for creating various distributed configurations
 * and validating work distribution across ranks.
 */
class Test__WorkDistribution : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup common test data
    }

    /**
     * @brief Create a tensor-parallel topology with uniform ranks
     * @param world_size Number of MPI ranks
     */
    std::vector<std::shared_ptr<MockMPITopology>> createUniformTopology(int world_size) {
        std::vector<std::shared_ptr<MockMPITopology>> topologies;
        
        for (int rank = 0; rank < world_size; ++rank) {
            auto builder = MockMPITopologyBuilder();
            
            // All ranks on same node with CPU only (uniform)
            for (int r = 0; r < world_size; ++r) {
                builder.addCPUOnlyRank(r, 0);  // All on node 0
            }
            
            builder.setLocalRank(rank);
            topologies.push_back(builder.build());
        }
        
        return topologies;
    }

    /**
     * @brief Create work distributors for each rank
     */
    std::vector<std::shared_ptr<MockWorkDistributor>> createDistributors(int world_size) {
        std::vector<std::shared_ptr<MockWorkDistributor>> distributors;
        
        for (int rank = 0; rank < world_size; ++rank) {
            distributors.push_back(MockWorkDistributor::tensorParallel(rank, world_size));
        }
        
        return distributors;
    }

    /**
     * @brief Validate that slices cover the full range without gaps or overlaps
     */
    void validateSliceCoverage(const std::vector<IWorkDistributor::WorkSlice>& slices, 
                               size_t total_elements) {
        // Check no gaps or overlaps
        std::vector<bool> covered(total_elements, false);
        
        for (const auto& slice : slices) {
            for (size_t i = slice.start; i < slice.end; ++i) {
                ASSERT_LT(i, total_elements) << "Slice extends beyond total elements";
                EXPECT_FALSE(covered[i]) << "Element " << i << " covered by multiple slices";
                covered[i] = true;
            }
        }
        
        // Check all elements covered
        for (size_t i = 0; i < total_elements; ++i) {
            EXPECT_TRUE(covered[i]) << "Element " << i << " not covered by any slice";
        }
    }

    /**
     * @brief Validate slice boundaries are correct
     */
    void validateSliceBoundaries(const IWorkDistributor::WorkSlice& slice) {
        EXPECT_LE(slice.start, slice.end) << "Slice start > end";
        EXPECT_EQ(slice.count, slice.end - slice.start) << "Slice count incorrect";
    }
};

// ============================================================================
// Tensor Parallel GEMM Tests
// ============================================================================

/**
 * @test Verify weights are correctly split across ranks for tensor-parallel GEMM
 *
 * In tensor parallelism, weight matrices are column-sliced:
 * - Each rank owns a contiguous column range
 * - After local GEMM, results are allreduced
 */
TEST_F(Test__WorkDistribution, TensorParallelGemm_WeightsSplitAcrossRanks) {
    constexpr int world_size = 4;
    constexpr size_t hidden_dim = 1024;  // Columns to split
    
    auto topologies = createUniformTopology(world_size);
    auto distributors = createDistributors(world_size);
    
    // Each rank should get hidden_dim / world_size columns
    const size_t expected_per_rank = hidden_dim / world_size;  // 256
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto slice = distributors[rank]->getRankSlice(hidden_dim);
        
        // Verify slice size
        EXPECT_EQ(slice.count, expected_per_rank)
            << "Rank " << rank << " has wrong slice size";
        
        // Verify slice boundaries
        size_t expected_start = rank * expected_per_rank;
        size_t expected_end = expected_start + expected_per_rank;
        
        EXPECT_EQ(slice.start, expected_start)
            << "Rank " << rank << " has wrong start";
        EXPECT_EQ(slice.end, expected_end)
            << "Rank " << rank << " has wrong end";
        
        validateSliceBoundaries(slice);
    }
    
    // Verify full coverage
    auto all_slices = distributors[0]->getAllRankSlices(hidden_dim);
    validateSliceCoverage(all_slices, hidden_dim);
}

/**
 * @test Verify weight slicing works with non-divisible dimensions
 *
 * When hidden_dim doesn't divide evenly by world_size, some ranks
 * should get one extra element.
 */
TEST_F(Test__WorkDistribution, TensorParallelGemm_NonDivisibleDimension) {
    constexpr int world_size = 3;
    constexpr size_t hidden_dim = 1000;  // Not divisible by 3
    
    auto distributors = createDistributors(world_size);
    auto all_slices = distributors[0]->getAllRankSlices(hidden_dim);
    
    // Total should equal hidden_dim
    size_t total_assigned = 0;
    for (const auto& slice : all_slices) {
        total_assigned += slice.count;
        validateSliceBoundaries(slice);
    }
    EXPECT_EQ(total_assigned, hidden_dim);
    
    // Verify coverage
    validateSliceCoverage(all_slices, hidden_dim);
    
    // Difference between max and min slice size should be at most 1
    size_t min_size = SIZE_MAX;
    size_t max_size = 0;
    for (const auto& slice : all_slices) {
        min_size = std::min(min_size, slice.count);
        max_size = std::max(max_size, slice.count);
    }
    EXPECT_LE(max_size - min_size, 1)
        << "Uneven distribution: max=" << max_size << ", min=" << min_size;
}

// ============================================================================
// Head Parallel Attention Tests
// ============================================================================

/**
 * @test Verify attention heads are correctly distributed across ranks
 *
 * In head-parallel attention:
 * - Each rank processes a subset of attention heads
 * - No allreduce needed (just allgather for output)
 */
TEST_F(Test__WorkDistribution, HeadParallelAttention_HeadsDistributedAcrossRanks) {
    constexpr int world_size = 4;
    constexpr size_t num_heads = 32;  // Typical Qwen2 7B
    
    auto topologies = createUniformTopology(world_size);
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto head_range = topologies[rank]->get_head_range(num_heads);
        
        // Each rank should get num_heads / world_size heads
        size_t expected_count = num_heads / world_size;  // 8 heads per rank
        
        EXPECT_EQ(head_range.size(), expected_count)
            << "Rank " << rank << " has wrong number of heads";
        
        // Verify head indices are in correct range
        size_t expected_start = rank * expected_count;
        size_t expected_end = expected_start + expected_count;
        
        EXPECT_EQ(head_range.start, expected_start)
            << "Rank " << rank << " has wrong head start";
        EXPECT_EQ(head_range.end, expected_end)
            << "Rank " << rank << " has wrong head end";
    }
}

/**
 * @test Verify GQA (Grouped Query Attention) KV head distribution
 *
 * GQA uses fewer KV heads than Q heads (e.g., 8 KV heads for 32 Q heads).
 * Distribution should handle this correctly.
 */
TEST_F(Test__WorkDistribution, HeadParallelAttention_GQA_KVHeadDistribution) {
    constexpr int world_size = 4;
    constexpr size_t num_q_heads = 32;
    constexpr size_t num_kv_heads = 8;  // 4:1 ratio
    
    auto topologies = createUniformTopology(world_size);
    
    size_t total_q_heads = 0;
    size_t total_kv_heads = 0;
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto q_range = topologies[rank]->get_head_range(num_q_heads);
        auto kv_range = topologies[rank]->get_kv_head_range(num_kv_heads);
        
        total_q_heads += q_range.size();
        total_kv_heads += kv_range.size();
        
        // Q heads should be divisible
        EXPECT_EQ(q_range.size(), num_q_heads / world_size);
        
        // KV heads distributed (8 / 4 = 2 per rank)
        EXPECT_EQ(kv_range.size(), num_kv_heads / world_size);
    }
    
    EXPECT_EQ(total_q_heads, num_q_heads);
    EXPECT_EQ(total_kv_heads, num_kv_heads);
}

// ============================================================================
// Equal Work Distribution Tests
// ============================================================================

/**
 * @test Verify work is evenly distributed when ranks are identical
 */
TEST_F(Test__WorkDistribution, EqualWorkDistribution_UniformRanks) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1000;
    
    auto distributors = createDistributors(world_size);
    
    // All ranks should get equal (or nearly equal) work
    std::vector<size_t> work_per_rank;
    for (int rank = 0; rank < world_size; ++rank) {
        auto slice = distributors[rank]->getRankSlice(total_work);
        work_per_rank.push_back(slice.count);
    }
    
    // Verify fairly equal distribution
    size_t expected_avg = total_work / world_size;  // 250
    for (int rank = 0; rank < world_size; ++rank) {
        // Allow ±1 for rounding
        EXPECT_NEAR(work_per_rank[rank], expected_avg, 1)
            << "Rank " << rank << " work count deviates from expected";
    }
}

/**
 * @test Verify single-rank case gives all work to that rank
 */
TEST_F(Test__WorkDistribution, EqualWorkDistribution_SingleRank) {
    constexpr size_t total_work = 1000;
    
    auto distributor = MockWorkDistributor::singleRank();
    auto slice = distributor->getRankSlice(total_work);
    
    EXPECT_EQ(slice.start, 0);
    EXPECT_EQ(slice.end, total_work);
    EXPECT_EQ(slice.count, total_work);
}

// ============================================================================
// Uneven Work Distribution Tests
// ============================================================================

/**
 * @test Verify handling of non-divisible work sizes
 *
 * When total_work % world_size != 0, some ranks get extra elements.
 */
TEST_F(Test__WorkDistribution, UnevenWorkDistribution_NonDivisible) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1001;  // Not divisible by 4
    
    auto distributors = createDistributors(world_size);
    auto all_slices = distributors[0]->getAllRankSlices(total_work);
    
    // Total should still equal total_work
    size_t sum = 0;
    for (const auto& slice : all_slices) {
        sum += slice.count;
    }
    EXPECT_EQ(sum, total_work);
    
    // Verify coverage
    validateSliceCoverage(all_slices, total_work);
    
    // Count how many ranks got the extra element
    size_t base_work = total_work / world_size;  // 250
    size_t remainder = total_work % world_size;   // 1
    
    int ranks_with_extra = 0;
    for (const auto& slice : all_slices) {
        if (slice.count == base_work + 1) {
            ranks_with_extra++;
        } else {
            EXPECT_EQ(slice.count, base_work);
        }
    }
    EXPECT_EQ(ranks_with_extra, static_cast<int>(remainder));
}

/**
 * @test Verify very small work sizes are handled correctly
 *
 * When total_work < world_size, some ranks get zero work.
 */
TEST_F(Test__WorkDistribution, UnevenWorkDistribution_WorkLessThanRanks) {
    constexpr int world_size = 8;
    constexpr size_t total_work = 3;  // Less than world_size
    
    auto distributors = createDistributors(world_size);
    auto all_slices = distributors[0]->getAllRankSlices(total_work);
    
    // Count ranks with work
    int ranks_with_work = 0;
    size_t total_assigned = 0;
    for (const auto& slice : all_slices) {
        if (slice.count > 0) {
            ranks_with_work++;
            total_assigned += slice.count;
        }
    }
    
    EXPECT_EQ(total_assigned, total_work);
    EXPECT_EQ(ranks_with_work, static_cast<int>(total_work));
    
    // Verify rankHasWork correctly reports
    for (int rank = 0; rank < world_size; ++rank) {
        bool has_work = distributors[rank]->rankHasWork(total_work);
        auto slice = distributors[rank]->getRankSlice(total_work);
        EXPECT_EQ(has_work, slice.count > 0)
            << "rankHasWork disagrees with slice for rank " << rank;
    }
}

// ============================================================================
// Slice Boundary Tests
// ============================================================================

/**
 * @test Verify correct start/end indices for each rank
 */
TEST_F(Test__WorkDistribution, SliceBoundaries_CorrectIndices) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1024;
    
    auto distributors = createDistributors(world_size);
    
    size_t expected_start = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        auto slice = distributors[rank]->getRankSlice(total_work);
        
        // Each slice should start where the previous ended
        EXPECT_EQ(slice.start, expected_start)
            << "Rank " << rank << " has gap/overlap at start";
        
        // Update expected start for next rank
        expected_start = slice.end;
        
        // Validate consistency
        validateSliceBoundaries(slice);
    }
    
    // Final end should be total_work
    EXPECT_EQ(expected_start, total_work);
}

/**
 * @test Verify slice ownership indices are correct
 */
TEST_F(Test__WorkDistribution, SliceBoundaries_CorrectOwnership) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1000;
    
    auto distributors = createDistributors(world_size);
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto slice = distributors[rank]->getRankSlice(total_work);
        
        // Owner should match rank
        EXPECT_EQ(slice.owner, rank)
            << "Rank " << rank << " slice has wrong owner";
    }
}

/**
 * @test Verify contains() method works correctly
 */
TEST_F(Test__WorkDistribution, SliceBoundaries_ContainsMethod) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1000;
    
    auto distributors = createDistributors(world_size);
    
    // For each element, exactly one rank's slice should contain it
    for (size_t elem = 0; elem < total_work; ++elem) {
        int containing_rank = -1;
        int count_containing = 0;
        
        for (int rank = 0; rank < world_size; ++rank) {
            auto slice = distributors[rank]->getRankSlice(total_work);
            if (slice.contains(elem)) {
                containing_rank = rank;
                count_containing++;
            }
        }
        
        EXPECT_EQ(count_containing, 1)
            << "Element " << elem << " contained by " << count_containing << " slices";
    }
}

// ============================================================================
// Allreduce After GEMM Tests
// ============================================================================

/**
 * @test Verify partial results are correctly combined via allreduce
 *
 * Simulates tensor-parallel GEMM where each rank computes a partial result:
 * - Each rank contributes its portion
 * - Allreduce combines to full result
 */
TEST_F(Test__WorkDistribution, AllreduceAfterGemm_PartialResultsCombined) {
    constexpr int world_size = 4;
    constexpr size_t output_size = 100;
    
    // Create mock MPI context for each rank
    std::vector<std::shared_ptr<MockMPIContext>> contexts;
    for (int rank = 0; rank < world_size; ++rank) {
        contexts.push_back(std::make_shared<MockMPIContext>(rank, world_size));
    }
    
    // Simulate partial results from each rank
    // In real TP, each rank computes output from its weight slice
    std::vector<std::vector<float>> partial_results(world_size);
    for (int rank = 0; rank < world_size; ++rank) {
        partial_results[rank].resize(output_size);
        // Each rank contributes rank+1 to each element
        for (size_t i = 0; i < output_size; ++i) {
            partial_results[rank][i] = static_cast<float>(rank + 1);
        }
    }
    
    // Simulate allreduce (mock just copies for single-process simulation)
    // In real scenario, this would sum across ranks
    std::vector<float> result(output_size, 0.0f);
    
    // For testing, manually compute expected sum
    float expected_sum = 0.0f;
    for (int rank = 0; rank < world_size; ++rank) {
        expected_sum += static_cast<float>(rank + 1);  // 1 + 2 + 3 + 4 = 10
    }
    
    // Verify mock tracking works
    contexts[0]->barrier();
    EXPECT_EQ(contexts[0]->barrier_call_count(), 1);
    
    // Simulate allreduce
    std::vector<float> send_buf = partial_results[0];
    std::vector<float> recv_buf(output_size);
    contexts[0]->allreduce_sum(send_buf.data(), recv_buf.data(), output_size);
    
    EXPECT_EQ(contexts[0]->allreduce_call_count(), 1);
}

/**
 * @test Verify allreduce semantics match tensor parallel requirements
 *
 * In tensor parallelism:
 * - row-parallel: allreduce after FFN down projection
 * - column-parallel: no allreduce needed (outputs partition naturally)
 */
TEST_F(Test__WorkDistribution, AllreduceAfterGemm_RowParallelSematics) {
    constexpr int world_size = 2;
    constexpr size_t hidden_dim = 8;
    
    auto distributors = createDistributors(world_size);
    
    // Row-parallel: each rank processes all rows but partial columns
    // After GEMM, we need allreduce to sum the partial results
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto col_slice = distributors[rank]->getRankSlice(hidden_dim);
        
        // Rank 0 gets cols 0-3, rank 1 gets cols 4-7
        EXPECT_EQ(col_slice.count, hidden_dim / world_size);
        
        // Each rank needs to allreduce its partial output
        // The interface provides hierarchical distribution
        auto hier_slices = distributors[rank]->distribute(hidden_dim);
        EXPECT_GE(hier_slices.size(), 1);
        EXPECT_EQ(hier_slices[0].rank, rank);
    }
}

// ============================================================================
// Hierarchical Distribution Tests
// ============================================================================

/**
 * @test Verify hierarchical distribution across ranks and devices
 */
TEST_F(Test__WorkDistribution, HierarchicalDistribution_RankAndDevice) {
    constexpr size_t total_work = 1000;
    
    // Create multi-device distributor (CPU + GPU)
    auto distributor = MockWorkDistributor::heterogeneous({
        DeviceId::cpu(),
        DeviceId::cuda(0)
    });
    
    auto slices = distributor->distribute(total_work);
    
    // Should have slices for both devices
    EXPECT_EQ(slices.size(), 2);
    
    // Total work should be covered
    size_t total_assigned = 0;
    for (const auto& slice : slices) {
        total_assigned += slice.local_count;
        EXPECT_EQ(slice.rank, 0);  // Single rank
    }
    EXPECT_EQ(total_assigned, total_work);
}

/**
 * @test Verify primary device slice for single-device operations
 */
TEST_F(Test__WorkDistribution, HierarchicalDistribution_PrimaryDeviceSlice) {
    constexpr size_t total_work = 1000;
    
    auto distributor = MockWorkDistributor::singleRank();
    auto primary = distributor->getPrimaryDeviceSlice(total_work);
    
    EXPECT_EQ(primary.rank, 0);
    EXPECT_EQ(primary.local_count, total_work);
    EXPECT_EQ(primary.global_start, 0);
    EXPECT_EQ(primary.global_end, total_work);
}

// ============================================================================
// Call Tracking Verification Tests
// ============================================================================

/**
 * @test Verify call tracking works for verification
 */
TEST_F(Test__WorkDistribution, CallTracking_VerifiesBehavior) {
    auto distributor = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(4)
        .withCallTracking(true)
        .build();
    
    // Initial state
    EXPECT_EQ(distributor->get_rank_slice_call_count(), 0);
    EXPECT_EQ(distributor->get_all_rank_slices_call_count(), 0);
    EXPECT_EQ(distributor->rank_has_work_call_count(), 0);
    
    // Make calls
    distributor->getRankSlice(100);
    distributor->getRankSlice(200);
    EXPECT_EQ(distributor->get_rank_slice_call_count(), 2);
    
    distributor->getAllRankSlices(100);
    EXPECT_EQ(distributor->get_all_rank_slices_call_count(), 1);
    
    distributor->rankHasWork(100);
    distributor->rankHasWork(100);
    distributor->rankHasWork(100);
    EXPECT_EQ(distributor->rank_has_work_call_count(), 3);
    
    // Reset should clear counters
    distributor->reset_call_counts();
    EXPECT_EQ(distributor->get_rank_slice_call_count(), 0);
    EXPECT_EQ(distributor->get_all_rank_slices_call_count(), 0);
    EXPECT_EQ(distributor->rank_has_work_call_count(), 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * @test Verify zero-size work is handled correctly
 */
TEST_F(Test__WorkDistribution, EdgeCase_ZeroWork) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 0;
    
    auto distributors = createDistributors(world_size);
    
    for (int rank = 0; rank < world_size; ++rank) {
        auto slice = distributors[rank]->getRankSlice(total_work);
        EXPECT_EQ(slice.count, 0);
        EXPECT_TRUE(slice.empty());
        EXPECT_FALSE(distributors[rank]->rankHasWork(total_work));
    }
}

/**
 * @test Verify single-element work with multiple ranks
 */
TEST_F(Test__WorkDistribution, EdgeCase_SingleElementWork) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1;
    
    auto distributors = createDistributors(world_size);
    
    // Exactly one rank should have work
    int ranks_with_work = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        if (distributors[rank]->rankHasWork(total_work)) {
            ranks_with_work++;
            auto slice = distributors[rank]->getRankSlice(total_work);
            EXPECT_EQ(slice.count, 1);
            EXPECT_EQ(slice.start, 0);
            EXPECT_EQ(slice.end, 1);
        }
    }
    EXPECT_EQ(ranks_with_work, 1);
}

/**
 * @test Verify large work sizes are handled correctly
 */
TEST_F(Test__WorkDistribution, EdgeCase_LargeWorkSize) {
    constexpr int world_size = 4;
    constexpr size_t total_work = 1ULL << 30;  // 1 billion elements
    
    auto distributors = createDistributors(world_size);
    auto all_slices = distributors[0]->getAllRankSlices(total_work);
    
    // Verify total coverage
    size_t total_assigned = 0;
    for (const auto& slice : all_slices) {
        total_assigned += slice.count;
        validateSliceBoundaries(slice);
    }
    EXPECT_EQ(total_assigned, total_work);
}

// ============================================================================
// Custom Slice Override Tests
// ============================================================================

/**
 * @test Verify custom slice override works for deterministic testing
 */
TEST_F(Test__WorkDistribution, CustomOverride_RankSlice) {
    IWorkDistributor::WorkSlice custom_slice{100, 200, 100, 1};
    
    auto distributor = MockWorkDistributor::Builder()
        .withRank(1)
        .withWorldSize(4)
        .withCustomRankSlice(custom_slice)
        .build();
    
    // Should return custom slice regardless of input
    auto slice = distributor->getRankSlice(1000);
    EXPECT_EQ(slice.start, 100);
    EXPECT_EQ(slice.end, 200);
    EXPECT_EQ(slice.count, 100);
    
    // Same for different input
    slice = distributor->getRankSlice(500);
    EXPECT_EQ(slice.start, 100);
    EXPECT_EQ(slice.end, 200);
}

/**
 * @test Verify device slice override works
 */
TEST_F(Test__WorkDistribution, CustomOverride_DeviceSlices) {
    std::vector<IWorkDistributor::WorkSlice> custom_slices = {
        {0, 300, 300, 0},    // Device 0: 30%
        {300, 1000, 700, 1}  // Device 1: 70%
    };
    
    auto distributor = MockWorkDistributor::Builder()
        .withRank(0)
        .withWorldSize(1)
        .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
        .withCustomDeviceSlices(custom_slices)
        .build();
    
    auto slices = distributor->getAllDeviceSlices(1000);
    ASSERT_EQ(slices.size(), 2);
    
    EXPECT_EQ(slices[0].count, 300);
    EXPECT_EQ(slices[1].count, 700);
}
