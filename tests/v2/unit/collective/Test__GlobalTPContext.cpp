/**
 * @file Test__GlobalTPContext.cpp
 * @brief Unit tests for GlobalTPContext (GLOBAL tensor parallelism)
 * @author David Sanftenberg
 * @date February 2026
 *
 * These tests run with a SINGLE MPI rank (np=1) to verify basic interface
 * functionality. Multi-rank collective tests are in the integration suite.
 *
 * Tests cover:
 * - Factory methods (createForTest, error cases)
 * - Interface accessors (degree, myIndex, domainId, worldRanks)
 * - Move semantics (move construct, move assign)
 * - Single-rank collective operations (trivial no-op cases)
 * - Error handling (null tensors, invalid indices)
 *
 * Note: MPI_COMM_SELF is used for single-rank tests (always size=1, rank=0)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstring>
#include <vector>

#include "collective/GlobalTPContext.h"
#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__GlobalTPContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure MPI is initialized (gtest_main_mpi.cpp handles this)
        int initialized;
        MPI_Initialized(&initialized);
        ASSERT_TRUE(initialized) << "MPI must be initialized for GlobalTPContext tests";
    }

    /**
     * @brief Create a test tensor with specified shape
     */
    std::unique_ptr<FP32Tensor> createTestTensor(const std::vector<size_t> &shape)
    {
        return TestTensorFactory::createFP32(shape);
    }

    /**
     * @brief Create a test tensor with random values
     */
    std::unique_ptr<FP32Tensor> createRandomTensor(const std::vector<size_t> &shape)
    {
        return TestTensorFactory::createFP32Random(shape);
    }

    /**
     * @brief Create a GlobalTPContext using MPI_COMM_SELF for single-rank tests
     *
     * MPI_COMM_SELF always has size=1 and rank=0, perfect for testing
     * interface correctness without requiring multi-process runs.
     */
    std::unique_ptr<GlobalTPContext> createSingleRankContext(
        int domain_id = 0,
        std::vector<int> world_ranks = {0})
    {
        return GlobalTPContext::createForTest(MPI_COMM_SELF, domain_id, std::move(world_ranks));
    }
};

// =============================================================================
// BasicInterface Tests
// =============================================================================

/**
 * @test Create via createForTest with MPI_COMM_SELF
 *
 * Verify that a single-rank context can be created with degree==1, myIndex==0.
 */
TEST_F(Test__GlobalTPContext, CreateForTest_Valid)
{
    auto ctx = createSingleRankContext(42, {0});

    ASSERT_NE(ctx, nullptr) << "createForTest should succeed with MPI_COMM_SELF";
    EXPECT_EQ(ctx->degree(), 1) << "MPI_COMM_SELF has size 1";
    EXPECT_EQ(ctx->myIndex(), 0) << "Single rank should have index 0";
    EXPECT_EQ(ctx->domainId(), 42) << "Domain ID should match what was passed";
}

/**
 * @test Verify createForTest returns nullptr for MPI_COMM_NULL
 *
 * MPI_COMM_NULL is an invalid communicator, creation should fail gracefully.
 */
TEST_F(Test__GlobalTPContext, CreateForTest_NullComm)
{
    auto ctx = GlobalTPContext::createForTest(MPI_COMM_NULL, 0, {0});

    EXPECT_EQ(ctx, nullptr) << "createForTest with MPI_COMM_NULL should return nullptr";
}

/**
 * @test Verify isLocal()==false, isGlobal()==true for GlobalTPContext
 *
 * GlobalTPContext is for cross-MPI-rank TP, so it's NOT local.
 */
TEST_F(Test__GlobalTPContext, IsGlobalNotLocal)
{
    auto ctx = createSingleRankContext();

    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->isLocal()) << "GlobalTPContext should NOT be local";
    EXPECT_TRUE(ctx->isGlobal()) << "GlobalTPContext should be global";
}

/**
 * @test Verify isValid()==true after successful creation
 */
TEST_F(Test__GlobalTPContext, IsValid_AfterCreate)
{
    auto ctx = createSingleRankContext();

    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isValid()) << "Context should be valid after creation";
}

/**
 * @test Verify domainId() returns what was passed to createForTest
 */
TEST_F(Test__GlobalTPContext, DomainId_Correct)
{
    // Test with various domain IDs
    {
        auto ctx = createSingleRankContext(0, {0});
        ASSERT_NE(ctx, nullptr);
        EXPECT_EQ(ctx->domainId(), 0);
    }
    {
        auto ctx = createSingleRankContext(123, {0});
        ASSERT_NE(ctx, nullptr);
        EXPECT_EQ(ctx->domainId(), 123);
    }
    {
        auto ctx = createSingleRankContext(999, {0});
        ASSERT_NE(ctx, nullptr);
        EXPECT_EQ(ctx->domainId(), 999);
    }
}

/**
 * @test Verify worldRanks() returns what was passed to createForTest
 */
TEST_F(Test__GlobalTPContext, WorldRanks_Correct)
{
    std::vector<int> ranks = {0};
    auto ctx = createSingleRankContext(0, ranks);

    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->worldRanks().size(), 1);
    EXPECT_EQ(ctx->worldRanks()[0], 0);

    // Test with different world rank value
    std::vector<int> ranks2 = {42};
    auto ctx2 = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, ranks2);

    ASSERT_NE(ctx2, nullptr);
    ASSERT_EQ(ctx2->worldRanks().size(), 1);
    EXPECT_EQ(ctx2->worldRanks()[0], 42);
}

// =============================================================================
// Accessor Tests
// =============================================================================

/**
 * @test Verify communicator() returns MPI_COMM_SELF when created with it
 */
TEST_F(Test__GlobalTPContext, Communicator_ReturnsSameComm)
{
    auto ctx = createSingleRankContext();

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->communicator(), MPI_COMM_SELF)
        << "communicator() should return the same communicator passed to createForTest";
}

/**
 * @test Verify degree()==1 for MPI_COMM_SELF
 */
TEST_F(Test__GlobalTPContext, Degree_SingleRank)
{
    auto ctx = createSingleRankContext();

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1) << "MPI_COMM_SELF has exactly 1 participant";
}

/**
 * @test Verify myIndex()==0 for MPI_COMM_SELF
 */
TEST_F(Test__GlobalTPContext, MyIndex_SingleRank)
{
    auto ctx = createSingleRankContext();

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->myIndex(), 0) << "The only rank in MPI_COMM_SELF has index 0";
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

/**
 * @test Move construct transfers ownership, source becomes invalid
 */
TEST_F(Test__GlobalTPContext, MoveConstruct_TransfersOwnership)
{
    auto original = createSingleRankContext(42, {0});
    ASSERT_NE(original, nullptr);
    ASSERT_TRUE(original->isValid());

    // Store original values for verification
    int orig_degree = original->degree();
    int orig_index = original->myIndex();
    int orig_domain_id = original->domainId();

    // Move construct
    GlobalTPContext moved(std::move(*original));

    // Moved-to context should have the original values
    EXPECT_TRUE(moved.isValid()) << "Moved-to context should be valid";
    EXPECT_EQ(moved.degree(), orig_degree);
    EXPECT_EQ(moved.myIndex(), orig_index);
    EXPECT_EQ(moved.domainId(), orig_domain_id);

    // Source should be invalid after move
    EXPECT_FALSE(original->isValid()) << "Source should be invalid after move";
}

/**
 * @test Move assign transfers ownership, source becomes invalid
 */
TEST_F(Test__GlobalTPContext, MoveAssign_TransfersOwnership)
{
    auto original = createSingleRankContext(42, {0});
    ASSERT_NE(original, nullptr);
    ASSERT_TRUE(original->isValid());

    // Create another context to overwrite
    auto target = createSingleRankContext(99, {1});
    ASSERT_NE(target, nullptr);

    // Store original values
    int orig_degree = original->degree();
    int orig_index = original->myIndex();
    int orig_domain_id = original->domainId();

    // Move assign
    *target = std::move(*original);

    // Target should have the original values
    EXPECT_TRUE(target->isValid()) << "Target should be valid after move assign";
    EXPECT_EQ(target->degree(), orig_degree);
    EXPECT_EQ(target->myIndex(), orig_index);
    EXPECT_EQ(target->domainId(), orig_domain_id);

    // Source should be invalid after move
    EXPECT_FALSE(original->isValid()) << "Source should be invalid after move assign";
}

// =============================================================================
// Single-Rank Collective Tests (Trivial Cases)
// =============================================================================

/**
 * @test Allreduce with single rank should be a no-op (data unchanged)
 *
 * When there's only one participant, allreduce has nothing to reduce.
 * The data should remain unchanged.
 */
TEST_F(Test__GlobalTPContext, Allreduce_SingleRank)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // Create tensor with known values
    auto tensor = createTestTensor({4, 4});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 16; ++i)
    {
        data[i] = static_cast<float>(i + 1); // 1, 2, 3, ..., 16
    }

    // Copy original values for comparison
    std::vector<float> original(data, data + 16);

    // Allreduce should succeed
    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result) << "Allreduce should succeed with single rank";

    // Data should be unchanged (single-rank allreduce is a no-op)
    for (size_t i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], original[i])
            << "Data should be unchanged after single-rank allreduce at index " << i;
    }
}

/**
 * @test Broadcast with single rank should be a no-op (source_index=0)
 *
 * When there's only one participant, broadcast has nothing to do.
 */
TEST_F(Test__GlobalTPContext, Broadcast_SingleRank)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // Create tensor with known values
    auto tensor = createRandomTensor({8});
    std::vector<float> original(tensor->data(), tensor->data() + 8);

    // Broadcast from rank 0 (the only rank) should succeed
    bool result = ctx->broadcast(tensor.get(), 0);
    EXPECT_TRUE(result) << "Broadcast from rank 0 should succeed";

    // Data should be unchanged
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(tensor->data()[i], original[i])
            << "Data should be unchanged after single-rank broadcast at index " << i;
    }
}

/**
 * @test Allgather with single rank should copy local to global
 *
 * With one participant, allgather is essentially a copy since
 * global_tensor has the same size as local_shard.
 */
TEST_F(Test__GlobalTPContext, Allgather_SingleRank)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // Create local shard with known values
    auto local = createTestTensor({4});
    float *local_data = local->mutable_data();
    local_data[0] = 1.0f;
    local_data[1] = 2.0f;
    local_data[2] = 3.0f;
    local_data[3] = 4.0f;

    // Global tensor should have same size (1 rank * 4 elements = 4 elements)
    auto global = createTestTensor({4});

    // Allgather should succeed
    bool result = ctx->allgather(local.get(), global.get());
    EXPECT_TRUE(result) << "Allgather should succeed with single rank";

    // Global should have the same values as local
    const float *global_data = global->data();
    EXPECT_FLOAT_EQ(global_data[0], 1.0f);
    EXPECT_FLOAT_EQ(global_data[1], 2.0f);
    EXPECT_FLOAT_EQ(global_data[2], 3.0f);
    EXPECT_FLOAT_EQ(global_data[3], 4.0f);
}

/**
 * @test Barrier with single rank should not hang
 *
 * Barrier is synchronization - with one rank it's trivially satisfied.
 */
TEST_F(Test__GlobalTPContext, Barrier_SingleRank)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // This should not hang or crash
    EXPECT_NO_THROW(ctx->barrier()) << "Barrier should complete without hanging";
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/**
 * @test Allreduce returns false for null tensor
 */
TEST_F(Test__GlobalTPContext, Allreduce_NullTensor)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->allreduce(nullptr);
    EXPECT_FALSE(result) << "Allreduce should return false for null tensor";
}

/**
 * @test Broadcast returns false for invalid source_index
 *
 * With degree=1, only source_index=0 is valid.
 */
TEST_F(Test__GlobalTPContext, Broadcast_InvalidSource)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // source_index=1 is out of range for degree=1
    bool result = ctx->broadcast(tensor.get(), 1);
    EXPECT_FALSE(result) << "Broadcast with source_index=1 should fail (degree=1)";

    // Negative index should also fail
    result = ctx->broadcast(tensor.get(), -1);
    EXPECT_FALSE(result) << "Broadcast with negative source_index should fail";
}

/**
 * @test Send returns false for invalid dest_index
 *
 * With degree=1, only dest_index=0 is valid (but sending to self is pointless).
 */
TEST_F(Test__GlobalTPContext, Send_InvalidDest)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // dest_index=1 is out of range for degree=1
    bool result = ctx->send(tensor.get(), 1);
    EXPECT_FALSE(result) << "Send with dest_index=1 should fail (degree=1)";

    // dest_index=100 should also fail
    result = ctx->send(tensor.get(), 100);
    EXPECT_FALSE(result) << "Send with dest_index=100 should fail";

    // Negative index should also fail
    result = ctx->send(tensor.get(), -1);
    EXPECT_FALSE(result) << "Send with negative dest_index should fail";
}

/**
 * @test Recv returns false for invalid source_index
 *
 * With degree=1, only source_index=0 is valid (but receiving from self is pointless).
 */
TEST_F(Test__GlobalTPContext, Recv_InvalidSource)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto tensor = createTestTensor({4});

    // source_index=1 is out of range for degree=1
    bool result = ctx->recv(tensor.get(), 1);
    EXPECT_FALSE(result) << "Recv with source_index=1 should fail (degree=1)";

    // source_index=100 should also fail
    result = ctx->recv(tensor.get(), 100);
    EXPECT_FALSE(result) << "Recv with source_index=100 should fail";

    // Negative index should also fail
    result = ctx->recv(tensor.get(), -1);
    EXPECT_FALSE(result) << "Recv with negative source_index should fail";
}

/**
 * @test Broadcast returns false for null tensor
 */
TEST_F(Test__GlobalTPContext, Broadcast_NullTensor)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->broadcast(nullptr, 0);
    EXPECT_FALSE(result) << "Broadcast should return false for null tensor";
}

/**
 * @test Allgather returns false for null local_shard
 */
TEST_F(Test__GlobalTPContext, Allgather_NullLocalShard)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto global = createTestTensor({4});

    bool result = ctx->allgather(nullptr, global.get());
    EXPECT_FALSE(result) << "Allgather should return false for null local_shard";
}

/**
 * @test Allgather returns false for null global_tensor
 */
TEST_F(Test__GlobalTPContext, Allgather_NullGlobalTensor)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto local = createTestTensor({4});

    bool result = ctx->allgather(local.get(), nullptr);
    EXPECT_FALSE(result) << "Allgather should return false for null global_tensor";
}

/**
 * @test Send returns false for null tensor
 */
TEST_F(Test__GlobalTPContext, Send_NullTensor)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->send(nullptr, 0);
    EXPECT_FALSE(result) << "Send should return false for null tensor";
}

/**
 * @test Recv returns false for null tensor
 */
TEST_F(Test__GlobalTPContext, Recv_NullTensor)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->recv(nullptr, 0);
    EXPECT_FALSE(result) << "Recv should return false for null tensor";
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @test Allreduce with zero-element tensor
 *
 * Zero-element tensors have no mutable_data(), so allreduce returns false.
 * This is expected behavior - there's nothing to reduce and no buffer to operate on.
 */
TEST_F(Test__GlobalTPContext, Allreduce_ZeroElements)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // Create empty tensor (0 elements)
    auto tensor = createTestTensor({0});

    // Zero-element tensors have no mutable_data(), so this should return false
    // (The tensor reports numel()==0 but mutable_data() returns nullptr)
    bool result = ctx->allreduce(tensor.get());
    EXPECT_FALSE(result) << "Allreduce returns false for zero-element tensor (no mutable_data)";
}

/**
 * @test Broadcast with zero-element tensor
 *
 * Zero-element tensors have no mutable_data(), so broadcast returns false.
 * This is expected behavior.
 */
TEST_F(Test__GlobalTPContext, Broadcast_ZeroElements)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    // Create empty tensor
    auto tensor = createTestTensor({0});

    // Zero-element tensors have no mutable_data(), so this returns false
    bool result = ctx->broadcast(tensor.get(), 0);
    EXPECT_FALSE(result) << "Broadcast returns false for zero-element tensor (no mutable_data)";
}

/**
 * @test Allgather with size mismatch should fail
 *
 * global_tensor must have size = local_shard.size * degree
 */
TEST_F(Test__GlobalTPContext, Allgather_SizeMismatch)
{
    auto ctx = createSingleRankContext();
    ASSERT_NE(ctx, nullptr);

    auto local = createTestTensor({4});
    // For degree=1, global should be size 4, but we give it size 8
    auto global_wrong_size = createTestTensor({8});

    bool result = ctx->allgather(local.get(), global_wrong_size.get());
    EXPECT_FALSE(result) << "Allgather should fail with size mismatch";
}

/**
 * @test Context with mismatched world_ranks size logs warning
 *
 * When world_ranks.size() != comm_size, createForTest should still
 * succeed but may log a warning (we can't verify logging, but ensure it doesn't crash).
 */
TEST_F(Test__GlobalTPContext, CreateForTest_MismatchedWorldRanksSize)
{
    // MPI_COMM_SELF has size 1, but we pass 3 world_ranks
    std::vector<int> ranks = {0, 1, 2};
    auto ctx = GlobalTPContext::createForTest(MPI_COMM_SELF, 0, ranks);

    // Should still create successfully (just with mismatched data)
    ASSERT_NE(ctx, nullptr) << "Should still create context despite size mismatch";
    EXPECT_EQ(ctx->degree(), 1) << "Degree comes from communicator size";
    EXPECT_EQ(ctx->worldRanks().size(), 3) << "worldRanks reflects what was passed";
}

