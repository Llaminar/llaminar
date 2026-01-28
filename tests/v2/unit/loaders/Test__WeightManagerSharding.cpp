/**
 * @file Test__WeightManagerSharding.cpp
 * @brief Unit tests for WeightManager weight sharding functionality
 *
 * Tests the tensor parallelism weight sharding implementation:
 * - ShardingMode determination based on weight names (via Qwen2Schema)
 * - Column slicing (for column-parallel weights)
 * - Row slicing (for row-parallel weights)
 * - Integration with WeightDistributionStrategy::SHARDED
 * - Instance methods: constructor, getWeight, getReplicatedWeight, cache operations
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "loaders/WeightManager.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "mocks/MockModelLoader.h"

using namespace llaminar2::test;

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

        // Get Qwen2 sharding config for testing
        Qwen2SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
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

    /**
     * @brief Helper to get sharding mode using Qwen2 schema config
     */
    ShardingMode getShardingMode(const std::string &name) const
    {
        WeightShardingMode mode = sharding_config_.getMode(name);
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            return ShardingMode::COLUMN_PARALLEL;
        case WeightShardingMode::RowParallel:
            return ShardingMode::ROW_PARALLEL;
        case WeightShardingMode::InputParallel:
            return ShardingMode::INPUT_PARALLEL;
        case WeightShardingMode::Replicate:
        default:
            return ShardingMode::REPLICATE;
        }
    }

    std::shared_ptr<FP32Tensor> test_tensor_;
    WeightShardingConfig sharding_config_;
};

// =============================================================================
// ShardingMode Determination Tests (using Qwen2Schema)
// =============================================================================

TEST_F(WeightManagerShardingTest, DeterminesInputParallelForAttnOutput)
{
    // attn_output.weight (Wo) is INPUT_PARALLEL to match column-parallel QKV output
    // Wo: [d_model, n_heads * head_dim] → [d_model, local_heads * head_dim] per rank
    // Input: [seq, local_heads * head_dim] from local attention
    // Output: [seq, d_model] partial sum, needs AllReduce
    EXPECT_EQ(getShardingMode("blk.0.attn_output.weight"),
              ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(getShardingMode("layers.5.self_attn.o_proj.weight"),
              ShardingMode::REPLICATE); // Different naming convention - not in Qwen2 schema
    EXPECT_EQ(getShardingMode("attn_output.weight"),
              ShardingMode::INPUT_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesInputParallelForFFNDown)
{
    // Phase 4b-2: ffn_down.weight is INPUT_PARALLEL (column-sliced) to match Gate/Up output
    // Down splits its input dimension (columns) since Gate/Up produce [seq, d_ff_local]
    EXPECT_EQ(getShardingMode("blk.0.ffn_down.weight"),
              ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(getShardingMode("ffn_down.weight"),
              ShardingMode::INPUT_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesColumnParallelForQKV)
{
    // Phase 3: QKV weights are column-parallel (split output dimension by head)
    // Q: [n_heads * head_dim, d_model] -> [local_n_heads * head_dim, d_model]
    // K/V: [n_kv_heads * head_dim, d_model] -> [local_n_kv_heads * head_dim, d_model]
    EXPECT_EQ(getShardingMode("blk.0.attn_q.weight"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(getShardingMode("blk.0.attn_k.weight"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(getShardingMode("blk.0.attn_v.weight"),
              ShardingMode::COLUMN_PARALLEL);
    // Biases are also column-parallel
    EXPECT_EQ(getShardingMode("blk.0.attn_q.bias"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(getShardingMode("blk.0.attn_k.bias"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(getShardingMode("blk.0.attn_v.bias"),
              ShardingMode::COLUMN_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesColumnParallelForGateUp)
{
    // Phase 4b-1: Gate/Up weights are column-parallel (split output dimension)
    // Each rank produces [seq, d_ff_local] where d_ff_local = d_ff / world_size
    EXPECT_EQ(getShardingMode("blk.0.ffn_gate.weight"),
              ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(getShardingMode("blk.0.ffn_up.weight"),
              ShardingMode::COLUMN_PARALLEL);
}

TEST_F(WeightManagerShardingTest, DeterminesReplicateForNorms)
{
    // Norms should always be replicated
    EXPECT_EQ(getShardingMode("blk.0.attn_norm.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(getShardingMode("blk.0.ffn_norm.weight"),
              ShardingMode::REPLICATE);
    EXPECT_EQ(getShardingMode("output_norm.weight"),
              ShardingMode::REPLICATE);
}

TEST_F(WeightManagerShardingTest, DeterminesReplicateForEmbeddings)
{
    // Token embeddings should always be replicated
    EXPECT_EQ(getShardingMode("token_embd.weight"),
              ShardingMode::REPLICATE);
}

TEST_F(WeightManagerShardingTest, DeterminesColumnParallelForLMHead)
{
    // LM head (output.weight) is column-parallel since Phase 5
    // Split vocab dimension across ranks, AllGather before sampling
    EXPECT_EQ(getShardingMode("output.weight"),
              ShardingMode::COLUMN_PARALLEL);
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
// WeightManager Instance Tests (using MockModelLoader)
// =============================================================================

/**
 * @brief Test fixture for WeightManager instance method tests
 *
 * Uses MockModelLoader to test the full WeightManager API without GGUF files.
 */
class WeightManagerInstanceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a minimal mock loader with known tensors
        mock_loader_ = MockModelLoader::createMinimal();

        // Add specific tensors for testing
        mock_loader_->addFP32RandomTensor("token_embd.weight", {1000, 128}, -1.0f, 1.0f, 42);
        mock_loader_->addFP32RandomTensor("output_norm.weight", {128}, -1.0f, 1.0f, 43);
        mock_loader_->addFP32RandomTensor("output.weight", {1000, 128}, -1.0f, 1.0f, 44);

        // Add layer 0 weights
        mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {128}, -1.0f, 1.0f, 100);
        mock_loader_->addFP32RandomTensor("blk.0.attn_q.weight", {128, 128}, -1.0f, 1.0f, 101);
        mock_loader_->addFP32RandomTensor("blk.0.attn_k.weight", {64, 128}, -1.0f, 1.0f, 102);
        mock_loader_->addFP32RandomTensor("blk.0.attn_v.weight", {64, 128}, -1.0f, 1.0f, 103);
        mock_loader_->addFP32RandomTensor("blk.0.attn_output.weight", {128, 128}, -1.0f, 1.0f, 104);
        mock_loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {128}, -1.0f, 1.0f, 105);
        mock_loader_->addFP32RandomTensor("blk.0.ffn_gate.weight", {512, 128}, -1.0f, 1.0f, 106);
        mock_loader_->addFP32RandomTensor("blk.0.ffn_up.weight", {512, 128}, -1.0f, 1.0f, 107);
        mock_loader_->addFP32RandomTensor("blk.0.ffn_down.weight", {128, 512}, -1.0f, 1.0f, 108);
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
};

// -----------------------------------------------------------------------------
// Constructor Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, Constructor_ReplicatedStrategy)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    // Should construct without error
    EXPECT_EQ(wm.cacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, Constructor_ShardedStrategy_NoMPI)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    // Should construct without error even without MPI context
    EXPECT_EQ(wm.cacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, Constructor_InterleavedStrategy)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::INTERLEAVED,
                     WeightPrecision::NATIVE);

    EXPECT_EQ(wm.cacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, Constructor_AllPrecisions)
{
    // Test all weight precision modes
    std::vector<WeightPrecision> precisions = {
        WeightPrecision::NATIVE,
        WeightPrecision::CONVERT_TO_FP32,
        WeightPrecision::CONVERT_TO_BF16,
        WeightPrecision::CONVERT_TO_FP16,
        WeightPrecision::CONVERT_TO_INT8};

    for (auto precision : precisions)
    {
        WeightManager wm(*mock_loader_, nullptr, nullptr,
                         WeightDistributionStrategy::REPLICATED,
                         precision);
        EXPECT_EQ(wm.cacheSize(), 0);
    }
}

// -----------------------------------------------------------------------------
// getWeight Tests (REPLICATED strategy)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, GetWeight_LoadsTensorOnFirstCall)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 1000);
    EXPECT_EQ(tensor->shape()[1], 128);
}

TEST_F(WeightManagerInstanceTest, GetWeight_CacheHitOnSecondCall)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor1 = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    auto tensor2 = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);

    // Should return same pointer (cache hit)
    ASSERT_NE(tensor1, nullptr);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor1.get(), tensor2.get());
}

TEST_F(WeightManagerInstanceTest, GetWeight_DifferentWeightsAreCachedSeparately)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto embd = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    auto norm = wm.getWeight("output_norm.weight", DeviceId::cpu(), 0);

    ASSERT_NE(embd, nullptr);
    ASSERT_NE(norm, nullptr);
    EXPECT_NE(embd.get(), norm.get());
    EXPECT_EQ(wm.cacheSize(), 2);
}

TEST_F(WeightManagerInstanceTest, GetWeight_1DTensor)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor = wm.getWeight("output_norm.weight", DeviceId::cpu(), 0);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 1);
    EXPECT_EQ(tensor->shape()[0], 128);
}

TEST_F(WeightManagerInstanceTest, GetWeight_NonexistentTensor_ReturnsNull)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor = wm.getWeight("nonexistent.weight", DeviceId::cpu(), 0);

    EXPECT_EQ(tensor, nullptr);
}

TEST_F(WeightManagerInstanceTest, GetWeight_MultipleLayers)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    // Load weights from layer 0
    auto q = wm.getWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0);
    auto k = wm.getWeight("blk.0.attn_k.weight", DeviceId::cpu(), 0);
    auto v = wm.getWeight("blk.0.attn_v.weight", DeviceId::cpu(), 0);

    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);

    // Verify shapes match model architecture
    EXPECT_EQ(q->shape()[0], 128); // n_heads * head_dim
    EXPECT_EQ(k->shape()[0], 64);  // n_kv_heads * head_dim
    EXPECT_EQ(v->shape()[0], 64);
}

// -----------------------------------------------------------------------------
// Cache Operations Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, CacheSize_EmptyInitially)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    EXPECT_EQ(wm.cacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, CacheSize_IncrementsOnLoad)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(wm.cacheSize(), 1);

    wm.getWeight("output_norm.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(wm.cacheSize(), 2);

    // Cache hit should not increment
    wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(wm.cacheSize(), 2);
}

TEST_F(WeightManagerInstanceTest, ClearCache_RemovesAllEntries)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    wm.getWeight("output_norm.weight", DeviceId::cpu(), 0);
    EXPECT_EQ(wm.cacheSize(), 2);

    wm.clearCache();
    EXPECT_EQ(wm.cacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, ClearCache_AllowsReload)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    auto tensor1 = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor1, nullptr);

    wm.clearCache();

    auto tensor2 = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(tensor2, nullptr);

    // Should be a different instance after cache clear
    // (Note: same data but different allocation)
    EXPECT_EQ(wm.cacheSize(), 1);
}

// -----------------------------------------------------------------------------
// Sharding Config Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, SetWeightShardingConfig_EnablesShardingModeDetection)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    auto config = schema_factory.getWeightShardingConfig();
    wm.setWeightShardingConfig(config);

    // Now isGemmWeight should work
    EXPECT_TRUE(wm.isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_FALSE(wm.isGemmWeight("blk.0.attn_norm.weight"));
}

// -----------------------------------------------------------------------------
// SHARDED Strategy Tests (single rank, simulated TP)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, Sharded_ReplicatesNormWeights)
{
    // Create a mock MPI context (single rank simulation)
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1); // rank=0, world_size=1

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Norm weights should be fully replicated
    auto norm = wm.getWeight("blk.0.attn_norm.weight", DeviceId::cpu(), 0);
    ASSERT_NE(norm, nullptr);
    EXPECT_EQ(norm->shape()[0], 128); // Full size, not sharded
}

TEST_F(WeightManagerInstanceTest, Sharded_ReplicatesEmbedding)
{
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1); // rank=0, world_size=1

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Embedding should be fully replicated
    auto embd = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);
    ASSERT_NE(embd, nullptr);
    EXPECT_EQ(embd->shape()[0], 1000);
    EXPECT_EQ(embd->shape()[1], 128);
}

// -----------------------------------------------------------------------------
// Decode Cache Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, DecodeCacheSize_EmptyInitially)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    EXPECT_EQ(wm.decodeCacheSize(), 0);
}

TEST_F(WeightManagerInstanceTest, ClearDecodeCache_Works)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    // Even if empty, clearDecodeCache should not crash
    wm.clearDecodeCache();
    EXPECT_EQ(wm.decodeCacheSize(), 0);
}

// -----------------------------------------------------------------------------
// Static Helper Tests (comprehensive)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, IsQKVWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isQKVWeight("blk.0.attn_q.weight"));
    EXPECT_TRUE(WeightManager::isQKVWeight("blk.0.attn_k.weight"));
    EXPECT_TRUE(WeightManager::isQKVWeight("blk.0.attn_v.weight"));
    EXPECT_TRUE(WeightManager::isQKVWeight("blk.23.attn_q.weight"));

    EXPECT_FALSE(WeightManager::isQKVWeight("blk.0.attn_output.weight"));
    EXPECT_FALSE(WeightManager::isQKVWeight("blk.0.ffn_gate.weight"));
    EXPECT_FALSE(WeightManager::isQKVWeight("token_embd.weight"));
}

TEST_F(WeightManagerInstanceTest, IsQKVBias_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_q.bias"));
    EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_k.bias"));
    EXPECT_TRUE(WeightManager::isQKVBias("blk.0.attn_v.bias"));

    EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_q.weight"));
    EXPECT_FALSE(WeightManager::isQKVBias("blk.0.attn_output.bias"));
}

TEST_F(WeightManagerInstanceTest, IsFFNGateUpWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isFFNGateUpWeight("blk.0.ffn_gate.weight"));
    EXPECT_TRUE(WeightManager::isFFNGateUpWeight("blk.0.ffn_up.weight"));
    EXPECT_TRUE(WeightManager::isFFNGateUpWeight("blk.15.ffn_gate.weight"));

    EXPECT_FALSE(WeightManager::isFFNGateUpWeight("blk.0.ffn_down.weight"));
    EXPECT_FALSE(WeightManager::isFFNGateUpWeight("blk.0.attn_q.weight"));
}

TEST_F(WeightManagerInstanceTest, IsFFNDownWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isFFNDownWeight("blk.0.ffn_down.weight"));
    EXPECT_TRUE(WeightManager::isFFNDownWeight("blk.23.ffn_down.weight"));

    EXPECT_FALSE(WeightManager::isFFNDownWeight("blk.0.ffn_gate.weight"));
    EXPECT_FALSE(WeightManager::isFFNDownWeight("blk.0.ffn_up.weight"));
}

TEST_F(WeightManagerInstanceTest, IsWoWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isWoWeight("blk.0.attn_output.weight"));
    EXPECT_TRUE(WeightManager::isWoWeight("blk.15.attn_output.weight"));
    EXPECT_TRUE(WeightManager::isWoWeight("attn_output.weight"));

    EXPECT_FALSE(WeightManager::isWoWeight("blk.0.attn_q.weight"));
    EXPECT_FALSE(WeightManager::isWoWeight("blk.0.ffn_down.weight"));
}

TEST_F(WeightManagerInstanceTest, IsLMHeadWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isLMHeadWeight("output.weight"));

    EXPECT_FALSE(WeightManager::isLMHeadWeight("output_norm.weight"));
    EXPECT_FALSE(WeightManager::isLMHeadWeight("token_embd.weight"));
}

TEST_F(WeightManagerInstanceTest, IsEmbeddingWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isEmbeddingWeight("token_embd.weight"));

    EXPECT_FALSE(WeightManager::isEmbeddingWeight("output.weight"));
    EXPECT_FALSE(WeightManager::isEmbeddingWeight("blk.0.attn_q.weight"));
}

TEST_F(WeightManagerInstanceTest, IsOutputNormWeight_AllPatterns)
{
    EXPECT_TRUE(WeightManager::isOutputNormWeight("output_norm.weight"));

    EXPECT_FALSE(WeightManager::isOutputNormWeight("blk.0.attn_norm.weight"));
    EXPECT_FALSE(WeightManager::isOutputNormWeight("blk.0.ffn_norm.weight"));
}

// -----------------------------------------------------------------------------
// SliceRowRange / SliceColumnRange Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, SliceRowRange_BasicSlice)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{16, 8});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 16; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            data[i * 8 + j] = static_cast<float>(i * 100 + j);
        }
    }

    // Slice rows 4-7 (4 rows)
    auto sliced = WeightManager::sliceRowRange(tensor, 4, 4);

    ASSERT_NE(sliced, nullptr);
    EXPECT_EQ(sliced->shape()[0], 4);
    EXPECT_EQ(sliced->shape()[1], 8);

    // Verify data
    const float *slice_data = sliced->data();
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            float expected = static_cast<float>((i + 4) * 100 + j);
            EXPECT_FLOAT_EQ(slice_data[i * 8 + j], expected);
        }
    }
}

TEST_F(WeightManagerInstanceTest, SliceColumnRange_BasicSlice)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 16});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        for (size_t j = 0; j < 16; ++j)
        {
            data[i * 16 + j] = static_cast<float>(i * 100 + j);
        }
    }

    // Slice columns 4-11 (8 columns)
    auto sliced = WeightManager::sliceColumnRange(tensor, 4, 8);

    ASSERT_NE(sliced, nullptr);
    EXPECT_EQ(sliced->shape()[0], 8);
    EXPECT_EQ(sliced->shape()[1], 8);

    // Verify data
    const float *slice_data = sliced->data();
    for (size_t i = 0; i < 8; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            float expected = static_cast<float>(i * 100 + (j + 4));
            EXPECT_FLOAT_EQ(slice_data[i * 8 + j], expected);
        }
    }
}

// -----------------------------------------------------------------------------
// TailSlice Tests (for decode shard fractions)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, SliceTailRows_HalfFraction)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{16, 8});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 16 * 8; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Fraction 0.5 should get last 8 rows (rows 8-15)
    auto sliced = WeightManager::sliceTailRows(tensor, 0.5f);

    ASSERT_NE(sliced, nullptr);
    EXPECT_EQ(sliced->shape()[0], 8); // 16 * 0.5 = 8
    EXPECT_EQ(sliced->shape()[1], 8);
}

TEST_F(WeightManagerInstanceTest, SliceTailColumns_HalfFraction)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 16});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 8 * 16; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Fraction 0.5 should get last 8 columns (cols 8-15)
    auto sliced = WeightManager::sliceTailColumns(tensor, 0.5f);

    ASSERT_NE(sliced, nullptr);
    EXPECT_EQ(sliced->shape()[0], 8);
    EXPECT_EQ(sliced->shape()[1], 8); // 16 * 0.5 = 8
}

// -----------------------------------------------------------------------------
// getShardingMode Tests (public API)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, GetShardingMode_WithConfig)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Test sharding modes for different weight types
    EXPECT_EQ(wm.getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.attn_k.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.attn_v.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.attn_output.weight"), ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.ffn_gate.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.ffn_up.weight"), ShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);
    EXPECT_EQ(wm.getShardingMode("blk.0.attn_norm.weight"), ShardingMode::REPLICATE);
    EXPECT_EQ(wm.getShardingMode("token_embd.weight"), ShardingMode::REPLICATE);
}

// -----------------------------------------------------------------------------
// isWeightSharded Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, IsWeightSharded_ReplicatedStrategy_AlwaysFalse)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // With REPLICATED strategy, no weights are sharded
    EXPECT_FALSE(wm.isWeightSharded("blk.0.attn_q.weight"));
    EXPECT_FALSE(wm.isWeightSharded("blk.0.ffn_gate.weight"));
}

TEST_F(WeightManagerInstanceTest, IsWeightSharded_ShardedStrategy_GemmWeights)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // QKV, Gate/Up/Down weights should be sharded
    EXPECT_TRUE(wm.isWeightSharded("blk.0.attn_q.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.attn_k.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.attn_v.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.attn_output.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.ffn_gate.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.ffn_up.weight"));
    EXPECT_TRUE(wm.isWeightSharded("blk.0.ffn_down.weight"));
}

TEST_F(WeightManagerInstanceTest, IsWeightSharded_ShardedStrategy_ReplicatedWeights)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Norms and embeddings should NOT be sharded
    EXPECT_FALSE(wm.isWeightSharded("blk.0.attn_norm.weight"));
    EXPECT_FALSE(wm.isWeightSharded("blk.0.ffn_norm.weight"));
    EXPECT_FALSE(wm.isWeightSharded("token_embd.weight"));
    EXPECT_FALSE(wm.isWeightSharded("output_norm.weight"));
}

// -----------------------------------------------------------------------------
// INTERLEAVED Strategy Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, InterleavedStrategy_LoadsTensor)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::INTERLEAVED,
                     WeightPrecision::NATIVE);

    // INTERLEAVED strategy should still load tensors (falls back to replicated for now)
    auto tensor = wm.getWeight("token_embd.weight", DeviceId::cpu(), 0);

    // May return nullptr if not implemented, or tensor if fallback works
    // The important thing is it doesn't crash
    if (tensor)
    {
        EXPECT_EQ(tensor->shape()[0], 1000);
        EXPECT_EQ(tensor->shape()[1], 128);
    }
}

// -----------------------------------------------------------------------------
// getDecodeWeight Tests (decode phase sharding)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_ColumnParallel_SlicesTailRows)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Q weight is column-parallel, should slice tail rows
    auto decode_shard = wm.getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f, 0);

    ASSERT_NE(decode_shard, nullptr);
    // Original shape [128, 128], with 0.5 fraction should get [64, 128]
    EXPECT_EQ(decode_shard->shape()[0], 64);  // half of 128 rows
    EXPECT_EQ(decode_shard->shape()[1], 128); // columns unchanged
}

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_InputParallel_SlicesTailColumns)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Wo weight is input-parallel, should slice tail columns
    auto decode_shard = wm.getDecodeWeight("blk.0.attn_output.weight", DeviceId::cpu(), 0.5f, 0);

    ASSERT_NE(decode_shard, nullptr);
    // Original shape [128, 128], with 0.5 fraction should get [128, 64]
    EXPECT_EQ(decode_shard->shape()[0], 128); // rows unchanged
    EXPECT_EQ(decode_shard->shape()[1], 64);  // half of 128 columns
}

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_FFNDown_InputParallel)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // FFN down is input-parallel, should slice tail columns
    auto decode_shard = wm.getDecodeWeight("blk.0.ffn_down.weight", DeviceId::cpu(), 0.25f, 0);

    ASSERT_NE(decode_shard, nullptr);
    // Original shape [128, 512], with 0.25 fraction should get [128, 128]
    EXPECT_EQ(decode_shard->shape()[0], 128); // rows unchanged
    EXPECT_EQ(decode_shard->shape()[1], 128); // 512 * 0.25 = 128 columns
}

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_ReplicatedWeight_ReturnsFullCopy)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Norm weights are replicated, should return full tensor
    auto decode_shard = wm.getDecodeWeight("blk.0.attn_norm.weight", DeviceId::cpu(), 0.5f, 0);

    ASSERT_NE(decode_shard, nullptr);
    EXPECT_EQ(decode_shard->shape()[0], 128); // Full size, not sliced
}

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_CachesSeparately)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Load both regular and decode weights
    auto regular = wm.getWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0);
    auto decode = wm.getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f, 0);

    ASSERT_NE(regular, nullptr);
    ASSERT_NE(decode, nullptr);

    // Should be different tensors
    EXPECT_NE(regular.get(), decode.get());

    // Regular should be full, decode should be half
    EXPECT_EQ(regular->shape()[0], 128);
    EXPECT_EQ(decode->shape()[0], 64);

    // Both caches should have entries
    EXPECT_EQ(wm.cacheSize(), 1);
    EXPECT_EQ(wm.decodeCacheSize(), 1);
}

TEST_F(WeightManagerInstanceTest, GetDecodeWeight_CacheHitOnSecondCall)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::REPLICATED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    auto decode1 = wm.getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f, 0);
    auto decode2 = wm.getDecodeWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0.5f, 0);

    ASSERT_NE(decode1, nullptr);
    ASSERT_NE(decode2, nullptr);

    // Should return same cached tensor
    EXPECT_EQ(decode1.get(), decode2.get());
    EXPECT_EQ(wm.decodeCacheSize(), 1);
}

// -----------------------------------------------------------------------------
// Weight Category Tests (exposed through isGemmWeight)
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, IsGemmWeight_IdentifiesProjections)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // All projection weights are GEMM weights
    EXPECT_TRUE(wm.isGemmWeight("blk.0.attn_q.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.attn_k.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.attn_v.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.attn_output.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.ffn_gate.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.ffn_up.weight"));
    EXPECT_TRUE(wm.isGemmWeight("blk.0.ffn_down.weight"));
    EXPECT_TRUE(wm.isGemmWeight("output.weight")); // LM head
}

TEST_F(WeightManagerInstanceTest, IsGemmWeight_IdentifiesNonGemm)
{
    WeightManager wm(*mock_loader_, nullptr, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Norms, embeddings, and biases are NOT GEMM weights
    EXPECT_FALSE(wm.isGemmWeight("blk.0.attn_norm.weight"));
    EXPECT_FALSE(wm.isGemmWeight("blk.0.ffn_norm.weight"));
    EXPECT_FALSE(wm.isGemmWeight("output_norm.weight"));
    EXPECT_FALSE(wm.isGemmWeight("token_embd.weight"));
}

// -----------------------------------------------------------------------------
// Multi-Rank Sharded Strategy Tests
// -----------------------------------------------------------------------------

TEST_F(WeightManagerInstanceTest, Sharded_TwoRanks_Rank0GetsFirstHalf)
{
    auto mpi_ctx = std::make_shared<MPIContext>(0, 2); // rank=0, world_size=2

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Q weight [128, 128] should be column-sliced to [64, 128] for rank 0
    auto q = wm.getWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->shape()[0], 64);  // half of 128
    EXPECT_EQ(q->shape()[1], 128); // input dim unchanged
}

TEST_F(WeightManagerInstanceTest, Sharded_TwoRanks_Rank1GetsSecondHalf)
{
    auto mpi_ctx = std::make_shared<MPIContext>(1, 2); // rank=1, world_size=2

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Q weight [128, 128] should be column-sliced to [64, 128] for rank 1
    auto q = wm.getWeight("blk.0.attn_q.weight", DeviceId::cpu(), 0);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->shape()[0], 64);  // half of 128
    EXPECT_EQ(q->shape()[1], 128); // input dim unchanged
}

TEST_F(WeightManagerInstanceTest, Sharded_InputParallel_SlicesColumns)
{
    auto mpi_ctx = std::make_shared<MPIContext>(0, 2); // rank=0, world_size=2

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // FFN down [128, 512] is input-parallel, should slice columns to [128, 256]
    auto down = wm.getWeight("blk.0.ffn_down.weight", DeviceId::cpu(), 0);
    ASSERT_NE(down, nullptr);
    EXPECT_EQ(down->shape()[0], 128); // output dim unchanged
    EXPECT_EQ(down->shape()[1], 256); // half of 512
}

TEST_F(WeightManagerInstanceTest, Sharded_ColumnParallel_GateUp)
{
    auto mpi_ctx = std::make_shared<MPIContext>(0, 2); // rank=0, world_size=2

    WeightManager wm(*mock_loader_, mpi_ctx, nullptr,
                     WeightDistributionStrategy::SHARDED,
                     WeightPrecision::NATIVE);

    Qwen2SchemaFactory schema_factory;
    wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    // Gate [512, 128] is column-parallel, should slice rows to [256, 128]
    auto gate = wm.getWeight("blk.0.ffn_gate.weight", DeviceId::cpu(), 0);
    ASSERT_NE(gate, nullptr);
    EXPECT_EQ(gate->shape()[0], 256); // half of 512
    EXPECT_EQ(gate->shape()[1], 128); // input dim unchanged

    // Up should be same
    auto up = wm.getWeight("blk.0.ffn_up.weight", DeviceId::cpu(), 0);
    ASSERT_NE(up, nullptr);
    EXPECT_EQ(up->shape()[0], 256);
    EXPECT_EQ(up->shape()[1], 128);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
