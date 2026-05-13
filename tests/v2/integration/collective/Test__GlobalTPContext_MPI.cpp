/**
 * @file Test__GlobalTPContext_MPI.cpp
 * @brief MPI integration tests for GlobalTPContext
 *
 * These tests require multiple MPI ranks (mpirun -np 2) to verify
 * cross-rank collective operations and point-to-point communication.
 *
 * Test categories:
 * 1. Multi-Rank Creation Tests - Verify createWithSplit and domain setup
 * 2. Multi-Rank Collective Tests - AllReduce, Broadcast, AllGather
 * 3. Point-to-Point Tests - Send, Recv, Barrier
 * 4. Domain Ordering Tests - myIndex, degree, isGlobal
 *
 * Run with: mpirun -np 2 ./v2_integration_global_tp_mpi
 *
 * Test labels: V2, Integration, GlobalTP, MPI, TensorParallel
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "collective/GlobalTPContext.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for GlobalTPContext MPI integration tests
 *
 * Sets up MPI rank/size and provides helper methods for tensor creation
 * and data verification. Skips tests if not exactly 2 MPI ranks.
 */
class Test__GlobalTPContext_MPI : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // These tests require exactly 2 ranks
        if (world_size_ != 2)
        {
            GTEST_SKIP() << "Test requires exactly 2 MPI ranks (got " << world_size_ << ")";
        }
    }

    void TearDown() override
    {
        // Synchronize all ranks before test ends to prevent hangs
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int world_rank_;
    int world_size_;

    /**
     * @brief Create an FP32 tensor with given shape
     * @param shape Tensor dimensions
     * @return Unique pointer to FP32Tensor on CPU
     */
    std::unique_ptr<FP32Tensor> createTestTensor(const std::vector<size_t> &shape)
    {
        return std::make_unique<FP32Tensor>(shape);
    }

    /**
     * @brief Fill tensor with given values
     * @param t Tensor to fill
     * @param values Values to write (up to tensor size)
     */
    void fillTensor(FP32Tensor *t, const std::vector<float> &values)
    {
        float *data = t->mutable_data();
        for (size_t i = 0; i < values.size() && i < t->numel(); ++i)
        {
            data[i] = values[i];
        }
    }

    /**
     * @brief Verify tensor values match expected
     * @param t Tensor to check
     * @param expected Expected values
     * @param tolerance Comparison tolerance
     * @return true if all values match within tolerance
     */
    bool verifyTensor(const FP32Tensor *t, const std::vector<float> &expected, float tolerance = 1e-5f)
    {
        const float *data = t->data();
        for (size_t i = 0; i < expected.size() && i < t->numel(); ++i)
        {
            if (std::abs(data[i] - expected[i]) > tolerance)
            {
                LOG_ERROR("verifyTensor: Mismatch at index " << i << ": got " << data[i]
                                                             << ", expected " << expected[i]);
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Create a GlobalTPContext with all ranks in same domain
     * @param domain_id Domain identifier
     * @return Unique pointer to GlobalTPContext
     */
    std::unique_ptr<GlobalTPContext> createTwoRankContext(int domain_id = 42)
    {
        return GlobalTPContext::createWithSplit(
            MPI_COMM_WORLD,
            domain_id,
            /*color=*/0,        // Same color = same domain
            /*key=*/world_rank_ // Key determines ordering within domain
        );
    }
};

// =============================================================================
// Multi-Rank Creation Tests
// =============================================================================

/**
 * @test CreateWithSplit_TwoRanks
 * Create via createWithSplit, verify each rank has correct index (0 or 1)
 */
TEST_F(Test__GlobalTPContext_MPI, CreateWithSplit_TwoRanks)
{
    auto ctx = createTwoRankContext(42);

    ASSERT_NE(ctx, nullptr) << "createWithSplit returned nullptr on rank " << world_rank_;
    EXPECT_EQ(ctx->degree(), 2) << "Expected degree=2 for two-rank domain";
    EXPECT_EQ(ctx->myIndex(), world_rank_) << "myIndex should match world_rank when key=world_rank";
    EXPECT_EQ(ctx->domainId(), 42) << "Domain ID should be 42";
}

/**
 * @test CreateWithSplit_CorrectWorldRanks
 * Verify worldRanks() contains both world ranks (0 and 1)
 */
TEST_F(Test__GlobalTPContext_MPI, CreateWithSplit_CorrectWorldRanks)
{
    auto ctx = createTwoRankContext(100);

    ASSERT_NE(ctx, nullptr);

    const auto &world_ranks = ctx->worldRanks();
    EXPECT_EQ(world_ranks.size(), 2) << "Should have 2 world ranks";

    // worldRanks() is gathered via MPI_Allgather with key=world_rank
    // So we expect [0, 1] in sorted order
    std::vector<int> sorted_ranks = world_ranks;
    std::sort(sorted_ranks.begin(), sorted_ranks.end());
    EXPECT_EQ(sorted_ranks[0], 0);
    EXPECT_EQ(sorted_ranks[1], 1);
}

/**
 * @test CreateWithSplit_OwnsCommunicator
 * Verify context owns the created communicator (valid after creation)
 */
TEST_F(Test__GlobalTPContext_MPI, CreateWithSplit_OwnsCommunicator)
{
    auto ctx = createTwoRankContext(200);

    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isValid()) << "Context should be valid after creation";
    EXPECT_NE(ctx->communicator(), MPI_COMM_NULL) << "Communicator should not be NULL";
    EXPECT_NE(ctx->communicator(), MPI_COMM_WORLD) << "Should create new communicator, not reuse WORLD";
}

/**
 * @test CreateForTest_UsesExistingCommunicator
 * Verify createForTest works with an existing communicator
 */
TEST_F(Test__GlobalTPContext_MPI, CreateForTest_UsesExistingCommunicator)
{
    // Use MPI_COMM_WORLD directly for test
    std::vector<int> world_ranks = {0, 1};

    auto ctx = GlobalTPContext::createForTest(
        MPI_COMM_WORLD,
        /*domain_id=*/999,
        world_ranks);

    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isValid());
    EXPECT_EQ(ctx->communicator(), MPI_COMM_WORLD);
    EXPECT_EQ(ctx->domainId(), 999);
    EXPECT_EQ(ctx->degree(), 2);
}

// =============================================================================
// Multi-Rank Collective Tests
// =============================================================================

/**
 * @test Allreduce_TwoRanks_SumCorrect
 * Each rank sets [1,2,3,4], after allreduce verify [2,4,6,8]
 */
TEST_F(Test__GlobalTPContext_MPI, Allreduce_TwoRanks_SumCorrect)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // Each rank creates tensor with [1, 2, 3, 4]
    auto tensor = createTestTensor({4});
    fillTensor(tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f});

    // Perform allreduce (sum)
    bool success = ctx->allreduce(tensor.get());
    ASSERT_TRUE(success) << "allreduce failed on rank " << world_rank_;

    // After sum: [1+1, 2+2, 3+3, 4+4] = [2, 4, 6, 8]
    EXPECT_TRUE(verifyTensor(tensor.get(), {2.0f, 4.0f, 6.0f, 8.0f}))
        << "Allreduce result incorrect on rank " << world_rank_;
}

/**
 * @test Allreduce_LargerTensor
 * Test allreduce with larger tensor (256 elements)
 */
TEST_F(Test__GlobalTPContext_MPI, Allreduce_LargerTensor)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // Each rank creates tensor with values equal to (index + 1)
    const size_t size = 256;
    auto tensor = createTestTensor({size});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<float>(i + 1);
    }

    // Perform allreduce (sum)
    bool success = ctx->allreduce(tensor.get());
    ASSERT_TRUE(success);

    // After sum: each value should be 2*(i+1)
    const float *result = tensor->data();
    for (size_t i = 0; i < size; ++i)
    {
        float expected = 2.0f * static_cast<float>(i + 1);
        EXPECT_NEAR(result[i], expected, 1e-4f)
            << "Mismatch at index " << i << " on rank " << world_rank_;
    }
}

/**
 * @test Broadcast_FromRank0
 * Rank 0 has [10,20,30,40], rank 1 has [0,0,0,0]. After broadcast from 0,
 * verify both have [10,20,30,40]
 */
TEST_F(Test__GlobalTPContext_MPI, Broadcast_FromRank0)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // Rank 0 has [10, 20, 30, 40], rank 1 has zeros
    if (world_rank_ == 0)
    {
        fillTensor(tensor.get(), {10.0f, 20.0f, 30.0f, 40.0f});
    }
    else
    {
        fillTensor(tensor.get(), {0.0f, 0.0f, 0.0f, 0.0f});
    }

    // Broadcast from source_index=0 (which is rank 0 due to key=world_rank)
    bool success = ctx->broadcast(tensor.get(), /*source_index=*/0);
    ASSERT_TRUE(success) << "broadcast failed on rank " << world_rank_;

    // Both ranks should now have [10, 20, 30, 40]
    EXPECT_TRUE(verifyTensor(tensor.get(), {10.0f, 20.0f, 30.0f, 40.0f}))
        << "Broadcast result incorrect on rank " << world_rank_;
}

/**
 * @test Broadcast_FromRank1
 * Rank 1 has [99,88,77,66], after broadcast from 1, verify all have [99,88,77,66]
 */
TEST_F(Test__GlobalTPContext_MPI, Broadcast_FromRank1)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // Rank 1 has [99, 88, 77, 66], rank 0 has zeros
    if (world_rank_ == 1)
    {
        fillTensor(tensor.get(), {99.0f, 88.0f, 77.0f, 66.0f});
    }
    else
    {
        fillTensor(tensor.get(), {0.0f, 0.0f, 0.0f, 0.0f});
    }

    // Broadcast from source_index=1 (which is rank 1 due to key=world_rank)
    bool success = ctx->broadcast(tensor.get(), /*source_index=*/1);
    ASSERT_TRUE(success) << "broadcast failed on rank " << world_rank_;

    // Both ranks should now have [99, 88, 77, 66]
    EXPECT_TRUE(verifyTensor(tensor.get(), {99.0f, 88.0f, 77.0f, 66.0f}))
        << "Broadcast result incorrect on rank " << world_rank_;
}

/**
 * @test Allgather_TwoRanks
 * Rank 0 has [1,2], rank 1 has [3,4]. After allgather into size-4 tensor,
 * verify [1,2,3,4] (ordered by domain index)
 */
TEST_F(Test__GlobalTPContext_MPI, Allgather_TwoRanks)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // Each rank has 2 elements
    auto local_shard = createTestTensor({2});
    if (world_rank_ == 0)
    {
        fillTensor(local_shard.get(), {1.0f, 2.0f});
    }
    else
    {
        fillTensor(local_shard.get(), {3.0f, 4.0f});
    }

    // Global tensor has 4 elements (2 ranks * 2 elements each)
    auto global_tensor = createTestTensor({4});
    fillTensor(global_tensor.get(), {0.0f, 0.0f, 0.0f, 0.0f});

    // Perform allgather
    bool success = ctx->allgather(local_shard.get(), global_tensor.get());
    ASSERT_TRUE(success) << "allgather failed on rank " << world_rank_;

    // Result should be [1, 2, 3, 4] (rank 0's data first, then rank 1's)
    EXPECT_TRUE(verifyTensor(global_tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f}))
        << "Allgather result incorrect on rank " << world_rank_;
}

/**
 * @test Allgather_LargerShards
 * Test allgather with larger per-rank shards (64 elements each)
 */
TEST_F(Test__GlobalTPContext_MPI, Allgather_LargerShards)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    const size_t shard_size = 64;
    auto local_shard = createTestTensor({shard_size});
    float *local_data = local_shard->mutable_data();

    // Rank 0 fills [0, 1, 2, ..., 63], Rank 1 fills [64, 65, ..., 127]
    float offset = static_cast<float>(world_rank_ * shard_size);
    for (size_t i = 0; i < shard_size; ++i)
    {
        local_data[i] = offset + static_cast<float>(i);
    }

    // Global tensor has 128 elements
    auto global_tensor = createTestTensor({shard_size * 2});

    bool success = ctx->allgather(local_shard.get(), global_tensor.get());
    ASSERT_TRUE(success);

    // Verify: [0, 1, ..., 63, 64, 65, ..., 127]
    const float *result = global_tensor->data();
    for (size_t i = 0; i < shard_size * 2; ++i)
    {
        EXPECT_NEAR(result[i], static_cast<float>(i), 1e-4f)
            << "Mismatch at index " << i << " on rank " << world_rank_;
    }
}

// =============================================================================
// Point-to-Point Tests
// =============================================================================

/**
 * @test SendRecv_Rank0ToRank1
 * Rank 0 sends [100,200,300], rank 1 receives and verifies
 */
TEST_F(Test__GlobalTPContext_MPI, SendRecv_Rank0ToRank1)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({3});

    if (world_rank_ == 0)
    {
        // Rank 0: Send [100, 200, 300] to rank 1
        fillTensor(tensor.get(), {100.0f, 200.0f, 300.0f});
        bool success = ctx->send(tensor.get(), /*dest_index=*/1);
        EXPECT_TRUE(success) << "send failed on rank 0";
    }
    else
    {
        // Rank 1: Receive from rank 0
        fillTensor(tensor.get(), {0.0f, 0.0f, 0.0f}); // Initialize with zeros
        bool success = ctx->recv(tensor.get(), /*source_index=*/0);
        EXPECT_TRUE(success) << "recv failed on rank 1";

        // Verify received data
        EXPECT_TRUE(verifyTensor(tensor.get(), {100.0f, 200.0f, 300.0f}))
            << "Received data incorrect on rank 1";
    }
}

/**
 * @test SendRecv_Rank1ToRank0
 * Rank 1 sends [400,500,600], rank 0 receives and verifies
 */
TEST_F(Test__GlobalTPContext_MPI, SendRecv_Rank1ToRank0)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({3});

    if (world_rank_ == 1)
    {
        // Rank 1: Send [400, 500, 600] to rank 0
        fillTensor(tensor.get(), {400.0f, 500.0f, 600.0f});
        bool success = ctx->send(tensor.get(), /*dest_index=*/0);
        EXPECT_TRUE(success) << "send failed on rank 1";
    }
    else
    {
        // Rank 0: Receive from rank 1
        fillTensor(tensor.get(), {0.0f, 0.0f, 0.0f});
        bool success = ctx->recv(tensor.get(), /*source_index=*/1);
        EXPECT_TRUE(success) << "recv failed on rank 0";

        // Verify received data
        EXPECT_TRUE(verifyTensor(tensor.get(), {400.0f, 500.0f, 600.0f}))
            << "Received data incorrect on rank 0";
    }
}

/**
 * @test SendRecv_Bidirectional
 * Both ranks send and receive simultaneously
 */
TEST_F(Test__GlobalTPContext_MPI, SendRecv_Bidirectional)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto send_tensor = createTestTensor({2});
    auto recv_tensor = createTestTensor({2});

    if (world_rank_ == 0)
    {
        fillTensor(send_tensor.get(), {11.0f, 22.0f});
    }
    else
    {
        fillTensor(send_tensor.get(), {33.0f, 44.0f});
    }
    fillTensor(recv_tensor.get(), {0.0f, 0.0f});

    // Note: MPI_Send is blocking, so we need to order operations carefully
    // to avoid deadlock. Rank 0 sends first, rank 1 receives first.
    bool send_success = false;
    bool recv_success = false;

    if (world_rank_ == 0)
    {
        // Rank 0: Send first, then receive
        send_success = ctx->send(send_tensor.get(), 1);
        recv_success = ctx->recv(recv_tensor.get(), 1);
    }
    else
    {
        // Rank 1: Receive first, then send
        recv_success = ctx->recv(recv_tensor.get(), 0);
        send_success = ctx->send(send_tensor.get(), 0);
    }

    EXPECT_TRUE(send_success) << "send failed on rank " << world_rank_;
    EXPECT_TRUE(recv_success) << "recv failed on rank " << world_rank_;

    // Verify received data
    if (world_rank_ == 0)
    {
        EXPECT_TRUE(verifyTensor(recv_tensor.get(), {33.0f, 44.0f}));
    }
    else
    {
        EXPECT_TRUE(verifyTensor(recv_tensor.get(), {11.0f, 22.0f}));
    }
}

/**
 * @test Barrier_TwoRanks
 * Both ranks call barrier(), verify no hang
 */
TEST_F(Test__GlobalTPContext_MPI, Barrier_TwoRanks)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // Call barrier - if it returns, the test passes
    ctx->barrier();

    // Do some work, then barrier again
    auto tensor = createTestTensor({4});
    fillTensor(tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f});

    ctx->barrier();

    // If we reach here, barrier is working correctly
    SUCCEED() << "Barrier completed successfully on rank " << world_rank_;
}

/**
 * @test Barrier_Multiple
 * Call barrier multiple times
 */
TEST_F(Test__GlobalTPContext_MPI, Barrier_Multiple)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // Call barrier 5 times
    for (int i = 0; i < 5; ++i)
    {
        ctx->barrier();
    }

    SUCCEED() << "Multiple barriers completed on rank " << world_rank_;
}

// =============================================================================
// Domain Ordering Tests
// =============================================================================

/**
 * @test MyIndex_UniquePerRank
 * Verify myIndex() returns different values per rank (0 and 1)
 */
TEST_F(Test__GlobalTPContext_MPI, MyIndex_UniquePerRank)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    int my_index = ctx->myIndex();

    // Gather all indices
    int all_indices[2];
    MPI_Allgather(&my_index, 1, MPI_INT, all_indices, 1, MPI_INT, MPI_COMM_WORLD);

    // Verify uniqueness
    EXPECT_NE(all_indices[0], all_indices[1]) << "Indices should be unique";

    // With key=world_rank, index should equal world_rank
    EXPECT_EQ(my_index, world_rank_) << "myIndex should equal world_rank when key=world_rank";
}

/**
 * @test Degree_SameOnAllRanks
 * Verify degree()==2 on both ranks
 */
TEST_F(Test__GlobalTPContext_MPI, Degree_SameOnAllRanks)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    int degree = ctx->degree();

    // Gather all degrees
    int all_degrees[2];
    MPI_Allgather(&degree, 1, MPI_INT, all_degrees, 1, MPI_INT, MPI_COMM_WORLD);

    // All ranks should have degree=2
    EXPECT_EQ(all_degrees[0], 2);
    EXPECT_EQ(all_degrees[1], 2);
}

/**
 * @test ScopeReflectsNodePlacement_AllRanks
 * Verify scope reflects node placement on all ranks.
 * In a dev container (single node), all ranks are same-node → NODE_LOCAL.
 */
TEST_F(Test__GlobalTPContext_MPI, ScopeReflectsNodePlacement_AllRanks)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    // In a dev container, all MPI ranks are on the same physical node
    // Auto-detection via MPI_Get_processor_name identifies this and returns NODE_LOCAL
    EXPECT_FALSE(ctx->isLocal()) << "GlobalTPContext should not be LOCAL (intra-rank) on rank " << world_rank_;
    EXPECT_TRUE(ctx->isNodeLocal()) << "Same-node ranks should be NODE_LOCAL on rank " << world_rank_;
    EXPECT_TRUE(ctx->isAllRanksOnSameNode()) << "Dev container has single node";
    EXPECT_EQ(ctx->nodeCount(), 1) << "Should detect 1 node in dev container";
}

TEST_F(Test__GlobalTPContext_MPI, SameNodeUPIUsesShmemSpinBackend)
{
    auto ctx = GlobalTPContext::createWithSplit(
        MPI_COMM_WORLD,
        /*domain_id=*/6060,
        /*color=*/0,
        /*key=*/world_rank_,
        /*hostfile_path=*/"",
        CollectiveBackendType::UPI);
    ASSERT_NE(ctx, nullptr);

    if (ctx->isAllRanksOnSameNode())
    {
        EXPECT_EQ(ctx->collectiveBackendNameForDiagnostics(), "ShmemSpin")
            << "Same-node UPI GlobalTP domains should use the shared-memory fast path";
    }
    else
    {
        EXPECT_EQ(ctx->collectiveBackendNameForDiagnostics(), "UPI")
            << "Cross-node UPI domains should stay on the MPI/UPI backend";
    }
}

/**
 * @test DomainId_SameOnAllRanks
 * Verify domainId() returns same value on all ranks
 */
TEST_F(Test__GlobalTPContext_MPI, DomainId_SameOnAllRanks)
{
    const int expected_domain_id = 12345;
    auto ctx = createTwoRankContext(expected_domain_id);
    ASSERT_NE(ctx, nullptr);

    int domain_id = ctx->domainId();

    // Gather all domain IDs
    int all_domain_ids[2];
    MPI_Allgather(&domain_id, 1, MPI_INT, all_domain_ids, 1, MPI_INT, MPI_COMM_WORLD);

    // All ranks should have same domain ID
    EXPECT_EQ(all_domain_ids[0], expected_domain_id);
    EXPECT_EQ(all_domain_ids[1], expected_domain_id);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/**
 * @test Broadcast_InvalidSourceIndex
 * Verify broadcast fails gracefully with invalid source index
 */
TEST_F(Test__GlobalTPContext_MPI, Broadcast_InvalidSourceIndex)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});
    fillTensor(tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f});

    // Source index 5 is invalid for 2-rank domain
    bool success = ctx->broadcast(tensor.get(), /*source_index=*/5);
    EXPECT_FALSE(success) << "broadcast should fail with invalid source index";
}

/**
 * @test Send_InvalidDestIndex
 * Verify send fails gracefully with invalid destination index
 */
TEST_F(Test__GlobalTPContext_MPI, Send_InvalidDestIndex)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});
    fillTensor(tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f});

    // Dest index 10 is invalid for 2-rank domain
    bool success = ctx->send(tensor.get(), /*dest_index=*/10);
    EXPECT_FALSE(success) << "send should fail with invalid dest index";
}

/**
 * @test Recv_InvalidSourceIndex
 * Verify recv fails gracefully with invalid source index
 */
TEST_F(Test__GlobalTPContext_MPI, Recv_InvalidSourceIndex)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // Source index -1 is invalid
    bool success = ctx->recv(tensor.get(), /*source_index=*/-1);
    EXPECT_FALSE(success) << "recv should fail with invalid source index";
}

/**
 * @test Allreduce_NullTensor
 * Verify allreduce fails gracefully with null tensor
 */
TEST_F(Test__GlobalTPContext_MPI, Allreduce_NullTensor)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    bool success = ctx->allreduce(nullptr);
    EXPECT_FALSE(success) << "allreduce should fail with null tensor";
}

/**
 * @test Broadcast_NullTensor
 * Verify broadcast fails gracefully with null tensor
 */
TEST_F(Test__GlobalTPContext_MPI, Broadcast_NullTensor)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    bool success = ctx->broadcast(nullptr, 0);
    EXPECT_FALSE(success) << "broadcast should fail with null tensor";
}

/**
 * @test Allgather_NullTensors
 * Verify allgather fails gracefully with null tensors
 */
TEST_F(Test__GlobalTPContext_MPI, Allgather_NullTensors)
{
    auto ctx = createTwoRankContext();
    ASSERT_NE(ctx, nullptr);

    auto valid_tensor = createTestTensor({4});

    // Null local shard
    bool success1 = ctx->allgather(nullptr, valid_tensor.get());
    EXPECT_FALSE(success1) << "allgather should fail with null local_shard";

    // Null global tensor
    auto local_shard = createTestTensor({2});
    bool success2 = ctx->allgather(local_shard.get(), nullptr);
    EXPECT_FALSE(success2) << "allgather should fail with null global_tensor";
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

/**
 * @test MoveConstructor
 * Verify move constructor transfers ownership correctly
 */
TEST_F(Test__GlobalTPContext_MPI, MoveConstructor)
{
    auto ctx1 = createTwoRankContext();
    ASSERT_NE(ctx1, nullptr);
    ASSERT_TRUE(ctx1->isValid());

    MPI_Comm original_comm = ctx1->communicator();
    int original_degree = ctx1->degree();

    // Move construct
    GlobalTPContext ctx2(std::move(*ctx1));

    // ctx2 should have the resources
    EXPECT_TRUE(ctx2.isValid());
    EXPECT_EQ(ctx2.communicator(), original_comm);
    EXPECT_EQ(ctx2.degree(), original_degree);

    // ctx1 should be invalidated (can't check directly, but verify ctx2 works)
    auto tensor = createTestTensor({4});
    fillTensor(tensor.get(), {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_TRUE(ctx2.allreduce(tensor.get()));
}

/**
 * @test MoveAssignment
 * Verify move assignment transfers ownership correctly
 */
TEST_F(Test__GlobalTPContext_MPI, MoveAssignment)
{
    auto ctx1 = createTwoRankContext(111);
    auto ctx2 = createTwoRankContext(222);

    ASSERT_NE(ctx1, nullptr);
    ASSERT_NE(ctx2, nullptr);

    int domain_id_1 = ctx1->domainId();

    // Move assign ctx1 to ctx2 (ctx2's old resources should be freed)
    *ctx2 = std::move(*ctx1);

    EXPECT_TRUE(ctx2->isValid());
    EXPECT_EQ(ctx2->domainId(), domain_id_1);

    // Verify ctx2 still works with collective
    auto tensor = createTestTensor({2});
    fillTensor(tensor.get(), {5.0f, 6.0f});
    EXPECT_TRUE(ctx2->allreduce(tensor.get()));
}
