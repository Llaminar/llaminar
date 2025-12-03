/**
 * @file Test__WeightManagerSharding.cpp
 * @brief Unit tests for WeightManager weight sharding functionality
 *
 * Tests the tensor parallelism weight sharding implementation:
 * - ShardingMode determination based on weight names
 * - Column slicing (for column-parallel weights)
 * - Row slicing (for row-parallel weights)
 * - Integration with WeightDistributionStrategy::SHARDED
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixtures
// =============================================================================

class WeightManagerShardingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test tensors
        createTestTensors();
    }

    void createTestTensors()
    {
        // Create a 16x8 FP32 tensor with known values
        // Value at (i, j) = i * 10 + j
        std::vector<size_t> shape = {16, 8};
        test_tensor_ = std::make_shared<FP32Tensor>(shape);
        float *data = test_tensor_->mutable_data();
        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                data[i * 8 + j] = static_cast<float>(i * 10 + j);
            }
        }
    }

    std::shared_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// ShardingMode Determination Tests
// =============================================================================

TEST_F(WeightManagerShardingTest, DeterminesRowParallelForAttnOutput)
{
    // attn_output.weight should be row-parallel (split input dimension)
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.attn_output.weight"),
              ShardingMode::ROW_PARALLEL);
    EXPECT_EQ(WeightManager::determineShardingMode("layers.5.self_attn.o_proj.weight"),
              ShardingMode::REPLICATE); // Different naming convention
    EXPECT_EQ(WeightManager::determineShardingMode("attn_output.weight"),
              ShardingMode::ROW_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesInputParallelForFFNDown)
{
    // Phase 4b-2: ffn_down.weight is INPUT_PARALLEL (column-sliced) to match Gate/Up output
    // Down splits its input dimension (columns) since Gate/Up produce [seq, d_ff_local]
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.ffn_down.weight"),
              ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(WeightManager::determineShardingMode("ffn_down.weight"),
              ShardingMode::INPUT_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesReplicateForQKV)
{
    // In Phase 1, QKV weights are replicated (column-parallel disabled)
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.attn_q.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.attn_k.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.attn_v.weight"),
              ShardingMode::REPLICATE);
}

TEST_F(WeightManagerShardingTest, DeterminesColumnParallelForGateUp)
{
    // Phase 4b-1: Gate/Up weights are column-parallel (split output dimension)
    // Each rank produces [seq, d_ff_local] where d_ff_local = d_ff / world_size
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.ffn_gate.weight"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.ffn_up.weight"),
              ShardingMode::COLUMN_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesReplicateForNorms)
{
    // Norms should always be replicated
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.attn_norm.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(WeightManager::determineShardingMode("blk.0.ffn_norm.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(WeightManager::determineShardingMode("output_norm.weight"),
              ShardingMode::REPLICATE);
}

TEST_F(WeightManagerShardingTest, DeterminesReplicateForEmbeddings)
{
    // Embeddings should always be replicated
    EXPECT_EQ(WeightManager::determineShardingMode("token_embd.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(WeightManager::determineShardingMode("output.weight"),
              ShardingMode::REPLICATE);
}

// =============================================================================
// Column Slicing Tests
// =============================================================================

TEST_F(WeightManagerShardingTest, SliceColumnsRank0Of2)
{
    // Slice [16, 8] tensor for rank 0 of 2 ranks
    // Should get rows 0-7: [8, 8]
    auto sliced = WeightManager::sliceColumns(test_tensor_, 0, 2);

    ASSERT_NE(sliced, nullptr);
    ASSERT_EQ(sliced->shape().size(), 2);
    EXPECT_EQ(sliced->shape()[0], 8); // out_local
    EXPECT_EQ(sliced->shape()[1], 8); // in_dim unchanged

    // Verify data
    const float *data = sliced->data();
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            float expected = static_cast<float>(i * 10 + j);
            EXPECT_FLOAT_EQ(data[i * 8 + j], expected)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(WeightManagerShardingTest, SliceColumnsRank1Of2)
{
    // Slice [16, 8] tensor for rank 1 of 2 ranks
    // Should get rows 8-15: [8, 8]
    auto sliced = WeightManager::sliceColumns(test_tensor_, 1, 2);

    ASSERT_NE(sliced, nullptr);
    ASSERT_EQ(sliced->shape().size(), 2);
    EXPECT_EQ(sliced->shape()[0], 8); // out_local
    EXPECT_EQ(sliced->shape()[1], 8); // in_dim unchanged

    // Verify data (rows 8-15 of original)
    const float *data = sliced->data();
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            float expected = static_cast<float>((i + 8) * 10 + j);
            EXPECT_FLOAT_EQ(data[i * 8 + j], expected)
                << "Mismatch at local row " << i << " (global " << (i + 8) << "), col " << j;
        }
    }
}

// =============================================================================
// Row Slicing Tests
// =============================================================================

TEST_F(WeightManagerShardingTest, SliceRowsRank0Of2)
{
    // Slice [16, 8] tensor for rank 0 of 2 ranks (row-parallel)
    // Should get columns 0-3: [16, 4]
    auto sliced = WeightManager::sliceRows(test_tensor_, 0, 2);

    ASSERT_NE(sliced, nullptr);
    ASSERT_EQ(sliced->shape().size(), 2);
    EXPECT_EQ(sliced->shape()[0], 16); // out_dim unchanged
    EXPECT_EQ(sliced->shape()[1], 4);  // in_local

    // Verify data (columns 0-3 of original)
    const float *data = sliced->data();
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float expected = static_cast<float>(i * 10 + j);
            EXPECT_FLOAT_EQ(data[i * 4 + j], expected)
                << "Mismatch at row " << i << ", local col " << j;
        }
    }
}

TEST_F(WeightManagerShardingTest, SliceRowsRank1Of2)
{
    // Slice [16, 8] tensor for rank 1 of 2 ranks (row-parallel)
    // Should get columns 4-7: [16, 4]
    auto sliced = WeightManager::sliceRows(test_tensor_, 1, 2);

    ASSERT_NE(sliced, nullptr);
    ASSERT_EQ(sliced->shape().size(), 2);
    EXPECT_EQ(sliced->shape()[0], 16); // out_dim unchanged
    EXPECT_EQ(sliced->shape()[1], 4);  // in_local

    // Verify data (columns 4-7 of original)
    const float *data = sliced->data();
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float expected = static_cast<float>(i * 10 + (j + 4));
            EXPECT_FLOAT_EQ(data[i * 4 + j], expected)
                << "Mismatch at row " << i << ", local col " << j << " (global " << (j + 4) << ")";
        }
    }
}

// =============================================================================
// Uneven Division Tests
// =============================================================================

TEST_F(WeightManagerShardingTest, SliceColumnsHandlesRemainder)
{
    // Create 15x8 tensor (not evenly divisible by 2)
    std::vector<size_t> shape = {15, 8};
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (int i = 0; i < 15; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            data[i * 8 + j] = static_cast<float>(i * 10 + j);
        }
    }

    // Rank 0 gets 7 rows (15/2 = 7)
    auto sliced0 = WeightManager::sliceColumns(tensor, 0, 2);
    ASSERT_NE(sliced0, nullptr);
    EXPECT_EQ(sliced0->shape()[0], 7);
    EXPECT_EQ(sliced0->shape()[1], 8);

    // Rank 1 gets remaining 8 rows (15 - 7 = 8)
    auto sliced1 = WeightManager::sliceColumns(tensor, 1, 2);
    ASSERT_NE(sliced1, nullptr);
    EXPECT_EQ(sliced1->shape()[0], 8);
    EXPECT_EQ(sliced1->shape()[1], 8);
}

TEST_F(WeightManagerShardingTest, SliceRowsHandlesRemainder)
{
    // Create 16x7 tensor (not evenly divisible by 2)
    std::vector<size_t> shape = {16, 7};
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            data[i * 7 + j] = static_cast<float>(i * 10 + j);
        }
    }

    // Rank 0 gets 3 columns (7/2 = 3)
    auto sliced0 = WeightManager::sliceRows(tensor, 0, 2);
    ASSERT_NE(sliced0, nullptr);
    EXPECT_EQ(sliced0->shape()[0], 16);
    EXPECT_EQ(sliced0->shape()[1], 3);

    // Rank 1 gets remaining 4 columns (7 - 3 = 4)
    auto sliced1 = WeightManager::sliceRows(tensor, 1, 2);
    ASSERT_NE(sliced1, nullptr);
    EXPECT_EQ(sliced1->shape()[0], 16);
    EXPECT_EQ(sliced1->shape()[1], 4);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(WeightManagerShardingTest, SliceSingleRankReturnsFullTensor)
{
    // Single rank (world_size=1) should return full tensor
    auto sliced_col = WeightManager::sliceColumns(test_tensor_, 0, 1);
    ASSERT_NE(sliced_col, nullptr);
    EXPECT_EQ(sliced_col->shape()[0], 16);
    EXPECT_EQ(sliced_col->shape()[1], 8);

    auto sliced_row = WeightManager::sliceRows(test_tensor_, 0, 1);
    ASSERT_NE(sliced_row, nullptr);
    EXPECT_EQ(sliced_row->shape()[0], 16);
    EXPECT_EQ(sliced_row->shape()[1], 8);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
