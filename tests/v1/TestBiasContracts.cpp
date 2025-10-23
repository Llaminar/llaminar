/**
 * @file TestBiasContracts.cpp
 * @brief Unit tests for BiasContract validation system
 * @author David Sanftenberg
 * @date 2025-10-12
 */

#include <gtest/gtest.h>
#include "BiasContracts.h"
#include "tensors/TensorFactory.h"
#include <mpi.h>

using namespace llaminar;

class BiasContractTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    int rank;
    int world_size;
};

TEST_F(BiasContractTest, ValidateFull_CorrectDimension)
{
    // Create a bias contract expecting full dim 896
    BiasContract contract("blk.0.attn_q.bias", "Q projection bias",
                          896, 448, rank, world_size);

    // Create a tensor with correct full dimension
    auto tensor = TensorFactory::create_simple({896});

    // Should pass validation
    EXPECT_TRUE(contract.validate_full(tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, ValidateFull_WrongDimension)
{
    // Create a bias contract expecting full dim 896
    BiasContract contract("blk.0.attn_q.bias", "Q projection bias",
                          896, 448, rank, world_size);

    // Create a tensor with WRONG full dimension
    auto tensor = TensorFactory::create_simple({448}); // Too small

    // Should FAIL validation
    EXPECT_FALSE(contract.validate_full(tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, ValidateFull_NullTensor)
{
    BiasContract contract("blk.0.attn_q.bias", "Q projection bias",
                          896, 448, rank, world_size);

    // Null tensor should fail
    std::shared_ptr<TensorBase> null_tensor = nullptr;
    EXPECT_FALSE(contract.validate_full(null_tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, ValidateSliced_CorrectDimension)
{
    // For a 2-rank system with 896 total elements:
    // Rank 0 gets 448, Rank 1 gets 448
    const int expected_local_dim = (rank == 0) ? 448 : 448;

    BiasContract contract("blk.0.attn_q.bias", "Q projection bias (sliced)",
                          896, expected_local_dim, rank, world_size);

    // Create a tensor with correct sliced dimension
    auto tensor = TensorFactory::create_simple({expected_local_dim});

    // Should pass validation
    EXPECT_TRUE(contract.validate(tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, ValidateSliced_WrongDimension)
{
    const int expected_local_dim = 448;
    const int wrong_dim = 896; // Full size when expecting sliced

    BiasContract contract("blk.0.attn_q.bias", "Q projection bias (sliced)",
                          896, expected_local_dim, rank, world_size);

    // Create a tensor with WRONG dimension (full instead of sliced)
    auto tensor = TensorFactory::create_simple({wrong_dim});

    // Should FAIL validation
    EXPECT_FALSE(contract.validate(tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, ValidateSliced_NullTensor)
{
    BiasContract contract("blk.0.attn_q.bias", "Q projection bias (sliced)",
                          896, 448, rank, world_size);

    // Null tensor should fail
    std::shared_ptr<TensorBase> null_tensor = nullptr;
    EXPECT_FALSE(contract.validate(null_tensor, 0, "blk.0.attn_q.bias"));
}

TEST_F(BiasContractTest, GetSliceRange_EvenDistribution)
{
    // For 896 elements across 2 ranks: each gets 448
    BiasContract contract("blk.0.attn_q.bias", "Q projection bias",
                          896, 448, rank, world_size);

    auto [offset, length] = contract.get_slice_range();

    if (world_size == 2)
    {
        if (rank == 0)
        {
            EXPECT_EQ(offset, 0);
            EXPECT_EQ(length, 448);
        }
        else if (rank == 1)
        {
            EXPECT_EQ(offset, 448);
            EXPECT_EQ(length, 448);
        }
    }
}

TEST_F(BiasContractTest, GetSliceRange_UnevenDistribution)
{
    // For 129 elements across 2 ranks:
    // Rank 0 gets 65 (128/2 + 1), Rank 1 gets 64 (128/2)
    const int total_dim = 129;
    const int per_rank = total_dim / world_size;
    const int remainder = total_dim % world_size;
    const int expected_local = per_rank + (rank < remainder ? 1 : 0);

    BiasContract contract("blk.0.test.bias", "Test bias (uneven)",
                          total_dim, expected_local, rank, world_size);

    auto [offset, length] = contract.get_slice_range();

    if (world_size == 2)
    {
        if (rank == 0)
        {
            EXPECT_EQ(offset, 0);
            EXPECT_EQ(length, 65); // Gets extra element
        }
        else if (rank == 1)
        {
            EXPECT_EQ(offset, 65);
            EXPECT_EQ(length, 64);
        }
    }
}

TEST_F(BiasContractTest, WorkflowValidation_FullThenSliced)
{
    // Simulate the QwenPipeline workflow:
    // 1. Load full tensor from GGUF
    // 2. Validate full dimension
    // 3. Slice for MPI rank
    // 4. Validate sliced dimension

    const int total_dim = 896;
    const int local_dim = 448; // For 2 ranks

    BiasContract contract("blk.0.attn_q.bias", "Q projection bias",
                          total_dim, local_dim, rank, world_size);

    // Step 1: Create "full" tensor as loaded from GGUF
    auto full_tensor = TensorFactory::create_simple({total_dim});

    // Step 2: Validate full dimension
    EXPECT_TRUE(contract.validate_full(full_tensor, 0, "blk.0.attn_q.bias"));

    // Step 3: Simulate slicing
    auto [offset, length] = contract.get_slice_range();
    auto sliced_tensor = TensorFactory::create_simple({length});
    memcpy(sliced_tensor->data(), full_tensor->data() + offset,
           length * sizeof(float));

    // Step 4: Validate sliced dimension
    EXPECT_TRUE(contract.validate(sliced_tensor, 0, "blk.0.attn_q.bias"));
    EXPECT_EQ(sliced_tensor->size(), static_cast<size_t>(local_dim));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
