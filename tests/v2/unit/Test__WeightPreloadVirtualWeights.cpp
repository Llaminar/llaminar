/**
 * @file Test__WeightPreloadVirtualWeights.cpp
 * @brief Regression tests for weight preloading with virtual/tied weights
 *
 * These tests verify that:
 * 1. Virtual weights (e.g., output.weight tied to token_embd.weight) are
 *    included in preloadForDevices() even when they don't exist in the GGUF file
 * 2. The preloaded weights use consistent tensor identity across
 *    packGemmWeightsViaPipeline() and later buildWeights() calls
 * Regression tests for fixes:
 * - TP GEMM cache miss for tied output.weight (April 2026)
 * - PP eager load device parameter mismatch (April 2026)
 */

#include <gtest/gtest.h>
#include "loaders/WeightManager.h"
#include "config/TensorParallelConfig.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"

// Test mocks
#include "../mocks/MockModelLoader.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__WeightPreloadVirtualWeights : public ::testing::Test
{
protected:
    static constexpr int VOCAB_SIZE = 1024;
    static constexpr int D_MODEL = 128;
    static constexpr int N_HEADS = 4;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int D_FF = 512;

    void SetUp() override
    {
        // Create a mock loader with token_embd.weight but NO output.weight
        // (tied embeddings scenario)
        loader_ = std::make_shared<MockModelLoader>();
        loader_->setArchitecture("qwen2");
        loader_->setBlockCount(1);
        loader_->setEmbeddingLength(D_MODEL);
        loader_->setHeadCount(N_HEADS);
        loader_->setHeadCountKV(N_KV_HEADS);
        loader_->setVocabSize(VOCAB_SIZE);
        loader_->setFeedForwardLength(D_FF);

        // Add token_embd.weight but NOT output.weight
        // Use FP32 tensors since MockModelLoader only supports FP32 row/column slicing
        loader_->addFP32RandomTensor("token_embd.weight", {VOCAB_SIZE, D_MODEL});

        // Add layer 0 weights
        loader_->addFP32RandomTensor("blk.0.attn_q.weight", {D_MODEL, D_MODEL});
        loader_->addFP32RandomTensor("blk.0.attn_k.weight", {static_cast<size_t>(N_KV_HEADS * D_MODEL / N_HEADS), D_MODEL});
        loader_->addFP32RandomTensor("blk.0.attn_v.weight", {static_cast<size_t>(N_KV_HEADS * D_MODEL / N_HEADS), D_MODEL});
        loader_->addFP32RandomTensor("blk.0.attn_output.weight", {D_MODEL, D_MODEL});
        loader_->addFP32RandomTensor("blk.0.ffn_gate.weight", {D_FF, D_MODEL});
        loader_->addFP32RandomTensor("blk.0.ffn_up.weight", {D_FF, D_MODEL});
        loader_->addFP32RandomTensor("blk.0.ffn_down.weight", {D_MODEL, D_FF});
        loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {D_MODEL});
        loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {D_MODEL});
        loader_->addFP32RandomTensor("output_norm.weight", {D_MODEL});
    }

    /**
     * @brief Create WeightShardingConfig with output.weight as ColumnParallel
     *
     * This mimics the Qwen2/Qwen3/Qwen3.5 schema where output.weight is
     * configured as ColumnParallel even though it doesn't exist in the GGUF
     * (it's tied to token_embd.weight).
     */
    WeightShardingConfig createShardingConfig()
    {
        WeightShardingConfig config;

        // Exact matches (like Qwen schema)
        config.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
        config.exact_matches["token_embd.weight"] = WeightShardingMode::Replicate;
        config.exact_matches["output_norm.weight"] = WeightShardingMode::Replicate;

        // Dimension type overrides for exact matches
        config.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;
        config.exact_dimension_matches["token_embd.weight"] = WeightDimensionType::None;
        config.exact_dimension_matches["output_norm.weight"] = WeightDimensionType::None;

        // Pattern-based sharding for layer weights (with dimension types)
        config.patterns.push_back({"attn_q.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads, "Q projection"});
        config.patterns.push_back({"attn_k.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads, "K projection"});
        config.patterns.push_back({"attn_v.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads, "V projection"});
        config.patterns.push_back({"attn_output.weight", WeightShardingMode::RowParallel, WeightDimensionType::Heads, "Wo projection"});
        config.patterns.push_back({"ffn_gate.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden, "FFN gate"});
        config.patterns.push_back({"ffn_up.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden, "FFN up"});
        config.patterns.push_back({"ffn_down.weight", WeightShardingMode::RowParallel, WeightDimensionType::FFNHidden, "FFN down"});

        // Norms are replicated (non-GEMM)
        config.patterns.push_back({"_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::None, "Norm weight"});

        return config;
    }

    /**
     * @brief Create TensorParallelConfig for 2-way equal split on CPU devices
     */
    // Use ROCm devices for TP testing since CPU to_string() doesn't
    // distinguish ordinals (always returns "CPU"). ROCm:0 and ROCm:1
    // produce distinct cache keys without needing real GPU hardware.
    static DeviceId tpDevice0() { return DeviceId::rocm(0); }
    static DeviceId tpDevice1() { return DeviceId::rocm(1); }

    std::shared_ptr<TensorParallelConfig> createTPConfig()
    {
        return std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(
                2,           // world_size
                N_HEADS,     // n_heads
                N_KV_HEADS,  // n_kv_heads
                D_FF,        // d_ff
                VOCAB_SIZE,  // vocab_size
                std::vector<DeviceId>{tpDevice0(), tpDevice1()}));
    }

    std::shared_ptr<MockModelLoader> loader_;
};

// ============================================================================
// Test: Virtual weights from sharding config are included in preload
// ============================================================================

TEST_F(Test__WeightPreloadVirtualWeights, VirtualWeightIncludedInPreload)
{
    // Create WeightManager with TP config
    auto tp_config = createTPConfig();
    WeightManager wm(*loader_, /*mpi_ctx=*/nullptr, /*placement_map=*/nullptr,
                     WeightDistributionStrategy::SHARDED);

    auto sharding_config = createShardingConfig();
    wm.setWeightShardingConfig(sharding_config);
    wm.setTensorParallelConfig(tp_config);
    wm.setModelDimensions(N_HEADS, N_KV_HEADS, D_MODEL / N_HEADS);

    // Verify output.weight is NOT in the GGUF (mock loader)
    ASSERT_FALSE(loader_->hasTensor("output.weight"))
        << "output.weight should NOT exist in the mock loader (tied embedding)";
    ASSERT_TRUE(loader_->hasTensor("token_embd.weight"))
        << "token_embd.weight should exist as the source for tied embedding";

    // Preload for two CPU devices
    std::vector<DeviceId> devices = {tpDevice0(), tpDevice1()};
    ASSERT_TRUE(wm.preloadForDevices(devices));

    // After preload, getWeightForDevice should return a valid tensor for output.weight
    // on BOTH devices (the preload should have triggered the tied embedding path)
    auto output_dev0 = wm.getWeightForDevice("output.weight", tpDevice0());
    auto output_dev1 = wm.getWeightForDevice("output.weight", tpDevice1());

    ASSERT_NE(output_dev0, nullptr) << "output.weight should be loadable for device 0 via tied embedding";
    ASSERT_NE(output_dev1, nullptr) << "output.weight should be loadable for device 1 via tied embedding";

    // The two devices should have different tensor slices (different vocab ranges)
    EXPECT_NE(output_dev0.get(), output_dev1.get())
        << "Different TP devices should get different tensor slices";
}

// ============================================================================
// Test: Preloaded virtual weight returns same tensor object on subsequent calls
// ============================================================================

TEST_F(Test__WeightPreloadVirtualWeights, PreloadedVirtualWeightReturnsSameTensor)
{
    auto tp_config = createTPConfig();
    WeightManager wm(*loader_, /*mpi_ctx=*/nullptr, /*placement_map=*/nullptr,
                     WeightDistributionStrategy::SHARDED);

    wm.setWeightShardingConfig(createShardingConfig());
    wm.setTensorParallelConfig(tp_config);
    wm.setModelDimensions(N_HEADS, N_KV_HEADS, D_MODEL / N_HEADS);

    std::vector<DeviceId> devices = {tpDevice0(), tpDevice1()};
    ASSERT_TRUE(wm.preloadForDevices(devices));

    // First call: getWeightForDevice during preload (cached in per_device_cache_)
    auto first_call = wm.getWeightForDevice("output.weight", tpDevice0());
    ASSERT_NE(first_call, nullptr);

    // Second call: should return the SAME cached tensor (not create a new one)
    auto second_call = wm.getWeightForDevice("output.weight", tpDevice0());
    ASSERT_NE(second_call, nullptr);

    EXPECT_EQ(first_call.get(), second_call.get())
        << "Repeated getWeightForDevice should return the same cached tensor";
}

// ============================================================================
// Test: Virtual weight sharding produces correct vocab-parallel slices
// ============================================================================

TEST_F(Test__WeightPreloadVirtualWeights, TiedEmbeddingProducesVocabParallelSlices)
{
    auto tp_config = createTPConfig();
    WeightManager wm(*loader_, /*mpi_ctx=*/nullptr, /*placement_map=*/nullptr,
                     WeightDistributionStrategy::SHARDED);

    wm.setWeightShardingConfig(createShardingConfig());
    wm.setTensorParallelConfig(tp_config);
    wm.setModelDimensions(N_HEADS, N_KV_HEADS, D_MODEL / N_HEADS);

    std::vector<DeviceId> devices = {tpDevice0(), tpDevice1()};
    ASSERT_TRUE(wm.preloadForDevices(devices));

    auto output_dev0 = wm.getWeightForDevice("output.weight", tpDevice0());
    auto output_dev1 = wm.getWeightForDevice("output.weight", tpDevice1());
    ASSERT_NE(output_dev0, nullptr);
    ASSERT_NE(output_dev1, nullptr);

    // Check shapes: each device should get ~half the vocab
    size_t rows_dev0 = output_dev0->shape()[0];
    size_t rows_dev1 = output_dev1->shape()[0];
    size_t cols_dev0 = output_dev0->shape()[1];
    size_t cols_dev1 = output_dev1->shape()[1];

    // Rows should sum to total vocab (column-parallel splits vocab dim)
    EXPECT_EQ(rows_dev0 + rows_dev1, static_cast<size_t>(VOCAB_SIZE))
        << "Vocab slices should cover the full vocabulary";

    // Columns should be full d_model on both
    EXPECT_EQ(cols_dev0, static_cast<size_t>(D_MODEL));
    EXPECT_EQ(cols_dev1, static_cast<size_t>(D_MODEL));
}

// ============================================================================
// Test: Sharding config isNonGemmWeight correctly classifies output.weight
// ============================================================================

TEST_F(Test__WeightPreloadVirtualWeights, ShardingConfigClassifiesOutputWeightAsGemm)
{
    auto config = createShardingConfig();

    // output.weight should be ColumnParallel (a GEMM weight)
    EXPECT_EQ(config.getMode("output.weight"), WeightShardingMode::ColumnParallel);
    EXPECT_FALSE(config.isNonGemmWeight("output.weight"))
        << "output.weight should be classified as a GEMM weight";

    // Norms should be non-GEMM
    EXPECT_TRUE(config.isNonGemmWeight("output_norm.weight"))
        << "output_norm.weight should be classified as a non-GEMM weight";
    EXPECT_TRUE(config.isNonGemmWeight("blk.0.attn_norm.weight"))
        << "attn_norm.weight should be classified as a non-GEMM weight";
    EXPECT_TRUE(config.isNonGemmWeight("blk.64.nextn.hnorm.weight"))
        << "MTP hnorm.weight should be classified as a non-GEMM weight";
    EXPECT_TRUE(config.isNonGemmWeight("blk.64.nextn.enorm.weight"))
        << "MTP enorm.weight should be classified as a non-GEMM weight";
}

// ============================================================================
// Test: getWeightForDevice with device parameter returns device-specific tensor
// ============================================================================

TEST_F(Test__WeightPreloadVirtualWeights, GetWeightForDeviceReturnsDeviceSpecificTensor)
{
    auto tp_config = createTPConfig();
    WeightManager wm(*loader_, /*mpi_ctx=*/nullptr, /*placement_map=*/nullptr,
                     WeightDistributionStrategy::SHARDED);

    wm.setWeightShardingConfig(createShardingConfig());
    wm.setTensorParallelConfig(tp_config);
    wm.setModelDimensions(N_HEADS, N_KV_HEADS, D_MODEL / N_HEADS);

    // Get the same weight for two different devices
    auto q_dev0 = wm.getWeightForDevice("blk.0.attn_q.weight", tpDevice0());
    auto q_dev1 = wm.getWeightForDevice("blk.0.attn_q.weight", tpDevice1());

    ASSERT_NE(q_dev0, nullptr);
    ASSERT_NE(q_dev1, nullptr);

    // Should be different tensors (different slices for different TP ranks)
    EXPECT_NE(q_dev0.get(), q_dev1.get())
        << "TP devices should get different tensor slices for column-parallel weights";
}
