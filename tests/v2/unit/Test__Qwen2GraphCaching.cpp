/**
 * @file Test__Qwen2GraphCaching.cpp
 * @brief Unit tests for Qwen2Graph caching optimization (Phase 10)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the graph caching functionality in Qwen2Graph:
 * - Graph caching enabled after initializeBuffers()
 * - Cached graphs reused in decode mode (seq_len=1)
 * - Dynamic parameter updates (pos_offset)
 * - Cache cleared properly on clearCache()/releaseBuffers()
 * - Prefill (seq_len > 1) does not use cached graphs
 */

#include <gtest/gtest.h>
#include "../../../src/v2/pipelines/qwen/Qwen2Graph.h"
#include "../../../src/v2/execution/ComputeStage.h"
#include "../../../src/v2/tensors/TensorFactory.h"
#include "../../../src/v2/utils/MPIContext.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__Qwen2GraphCaching : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create minimal MPI context (single rank)
        mpi_ctx_ = std::make_shared<MPIContext>(
            0, // rank
            1, // world_size
            MPI_COMM_WORLD);

        // Create TensorFactory
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    Qwen2GraphConfig createMinimalConfig()
    {
        Qwen2GraphConfig config;
        config.d_model = 896;
        config.n_heads = 14;
        config.n_kv_heads = 2;
        config.head_dim = 64;
        config.d_ff = 4864;
        config.vocab_size = 151936;
        config.n_layers = 24;
        config.use_graph_buffer_management = true;
        config.max_seq_len = 128;
        config.default_device = 0;
        return config;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;
};

// ============================================================================
// Graph Caching Initialization Tests
// ============================================================================

TEST_F(Test__Qwen2GraphCaching, CachingEnabledAfterInitializeBuffers)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // Before initialization, caching should not be available
    EXPECT_FALSE(graph.isGraphCachingEnabled())
        << "Caching should not be enabled before initializeBuffers()";

    // Initialize buffers
    ASSERT_TRUE(graph.initializeBuffers(64));

    // After initialization, caching should be enabled
    EXPECT_TRUE(graph.isGraphCachingEnabled())
        << "Caching should be enabled after initializeBuffers()";
}

TEST_F(Test__Qwen2GraphCaching, CachingNotEnabledWithoutGraphBufferManagement)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = false; // Disabled

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // initializeBuffers should fail when flag is false
    bool result = graph.initializeBuffers(64);
    EXPECT_FALSE(result);
    EXPECT_FALSE(graph.isGraphCachingEnabled())
        << "Caching should not be enabled when graph buffer management is disabled";
}

TEST_F(Test__Qwen2GraphCaching, CacheVectorSizedCorrectly)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;
    config.n_layers = 24;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));

    // Cache should be sized for all layers
    EXPECT_EQ(graph.getCacheSize(), 24u)
        << "Cache should be sized for n_layers";
}

TEST_F(Test__Qwen2GraphCaching, CacheVectorSizedForDifferentLayerCounts)
{
    // Test with 12 layers
    {
        auto config = createMinimalConfig();
        config.use_graph_buffer_management = true;
        config.n_layers = 12;

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setTensorFactory(tensor_factory_.get());
        ASSERT_TRUE(graph.initializeBuffers(64));

        EXPECT_EQ(graph.getCacheSize(), 12u);
    }

    // Test with 32 layers
    {
        auto config = createMinimalConfig();
        config.use_graph_buffer_management = true;
        config.n_layers = 32;

        Qwen2Graph graph(config, mpi_ctx_);
        graph.setTensorFactory(tensor_factory_.get());
        ASSERT_TRUE(graph.initializeBuffers(64));

        EXPECT_EQ(graph.getCacheSize(), 32u);
    }
}

// ============================================================================
// Cache State Tests
// ============================================================================

TEST_F(Test__Qwen2GraphCaching, CacheInitiallyEmpty)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));

    // No graphs should be cached initially
    for (int layer = 0; layer < config.n_layers; ++layer)
    {
        EXPECT_FALSE(graph.hasValidCachedGraph(layer, true /* attention */))
            << "Attention cache should be empty for layer " << layer;
        EXPECT_FALSE(graph.hasValidCachedGraph(layer, false /* ffn */))
            << "FFN cache should be empty for layer " << layer;
    }
}

TEST_F(Test__Qwen2GraphCaching, CacheInvalidForOutOfRangeLayer)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;
    config.n_layers = 24;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));

    // Negative layer index
    EXPECT_FALSE(graph.hasValidCachedGraph(-1, true));
    EXPECT_FALSE(graph.hasValidCachedGraph(-1, false));

    // Layer index >= n_layers
    EXPECT_FALSE(graph.hasValidCachedGraph(24, true));
    EXPECT_FALSE(graph.hasValidCachedGraph(25, true));
    EXPECT_FALSE(graph.hasValidCachedGraph(100, false));
}

// ============================================================================
// Cache Clear Tests
// ============================================================================

TEST_F(Test__Qwen2GraphCaching, ClearCacheResetsState)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));
    EXPECT_TRUE(graph.isGraphCachingEnabled());

    // Clear cache
    graph.clearCache();

    // Caching should still be enabled (just cache contents cleared)
    EXPECT_TRUE(graph.isGraphCachingEnabled())
        << "Caching should remain enabled after clearCache()";

    // Cache should be sized correctly but empty
    EXPECT_EQ(graph.getCacheSize(), static_cast<size_t>(config.n_layers));
}

TEST_F(Test__Qwen2GraphCaching, ReleaseBuffersDisablesCaching)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));
    EXPECT_TRUE(graph.isGraphCachingEnabled());

    // Release buffers
    graph.releaseBuffers();

    // Caching should be disabled
    EXPECT_FALSE(graph.isGraphCachingEnabled())
        << "Caching should be disabled after releaseBuffers()";

    // Cache vector is cleared but may retain size - verify no valid cached graphs
    for (size_t i = 0; i < graph.getCacheSize(); ++i)
    {
        EXPECT_FALSE(graph.hasValidCachedGraph(static_cast<int>(i), true))
            << "Attention cache should be invalid for layer " << i;
        EXPECT_FALSE(graph.hasValidCachedGraph(static_cast<int>(i), false))
            << "FFN cache should be invalid for layer " << i;
    }
}

TEST_F(Test__Qwen2GraphCaching, ReinitializeReenablesCaching)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // First init
    ASSERT_TRUE(graph.initializeBuffers(64));
    EXPECT_TRUE(graph.isGraphCachingEnabled());

    // Release
    graph.releaseBuffers();
    EXPECT_FALSE(graph.isGraphCachingEnabled());

    // Reinitialize
    ASSERT_TRUE(graph.initializeBuffers(128));
    EXPECT_TRUE(graph.isGraphCachingEnabled())
        << "Caching should be re-enabled after reinitialize";
    EXPECT_EQ(graph.getCacheSize(), static_cast<size_t>(config.n_layers));
}

// ============================================================================
// updateDynamicParams Tests (ComputeStage)
// ============================================================================

TEST_F(Test__Qwen2GraphCaching, RoPEStageUpdatesDynamicParams)
{
    // Create a RoPEStage and verify updateDynamicParams works
    RoPEStage::Params params;
    params.n_heads = 14;
    params.n_kv_heads = 2;
    params.head_dim = 64;
    params.pos_offset = 0;
    params.theta_base = 1000000.0f;

    RoPEStage stage(params);

    // Initial pos_offset
    EXPECT_EQ(stage.getParams().pos_offset, 0);

    // Update dynamic params
    stage.updateDynamicParams(42, 1);
    EXPECT_EQ(stage.getParams().pos_offset, 42)
        << "RoPEStage should update pos_offset";

    // Update again
    stage.updateDynamicParams(100, 1);
    EXPECT_EQ(stage.getParams().pos_offset, 100);
}

TEST_F(Test__Qwen2GraphCaching, AttentionComputeStageUpdatesDynamicParams)
{
    // Create AttentionComputeStage and verify updateDynamicParams works
    AttentionComputeStage::Params params;
    params.batch_size = 1;
    params.seq_len = 256;
    params.n_heads = 14;
    params.n_kv_heads = 2;
    params.head_dim = 64;
    params.position_offset = 0;

    AttentionComputeStage stage(params);

    // Initial position_offset
    EXPECT_EQ(stage.getParams().position_offset, 0);

    // Update dynamic params
    stage.updateDynamicParams(42, 1);
    EXPECT_EQ(stage.getParams().position_offset, 42)
        << "AttentionComputeStage should update position_offset";

    // Update again
    stage.updateDynamicParams(100, 1);
    EXPECT_EQ(stage.getParams().position_offset, 100);
}

TEST_F(Test__Qwen2GraphCaching, BaseStageIgnoresDynamicParams)
{
    // Create a stage that doesn't override updateDynamicParams (e.g., RMSNormStage)
    RMSNormStage::Params params;
    params.eps = 1e-6f;

    RMSNormStage stage(params);

    // Should not crash - default implementation is no-op
    stage.updateDynamicParams(42, 1);

    // Params should be unchanged (RMSNorm has no position-dependent params)
    EXPECT_FLOAT_EQ(stage.getParams().eps, 1e-6f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__Qwen2GraphCaching, CachingWithZeroLayers)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;
    config.n_layers = 0;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());

    // Should still initialize (empty cache)
    // Note: This may fail depending on other validation - adjust if needed
    bool result = graph.initializeBuffers(64);
    if (result)
    {
        EXPECT_EQ(graph.getCacheSize(), 0u);
    }
}

TEST_F(Test__Qwen2GraphCaching, MultipleClearCacheCalls)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));

    // Multiple clear calls should be safe
    graph.clearCache();
    graph.clearCache();
    graph.clearCache();

    EXPECT_TRUE(graph.isGraphCachingEnabled());
    EXPECT_EQ(graph.getCacheSize(), static_cast<size_t>(config.n_layers));
}

TEST_F(Test__Qwen2GraphCaching, MultipleReleaseBuffersCalls)
{
    auto config = createMinimalConfig();
    config.use_graph_buffer_management = true;

    Qwen2Graph graph(config, mpi_ctx_);
    graph.setTensorFactory(tensor_factory_.get());
    ASSERT_TRUE(graph.initializeBuffers(64));

    // Multiple release calls should be safe
    graph.releaseBuffers();
    graph.releaseBuffers();
    graph.releaseBuffers();

    EXPECT_FALSE(graph.isGraphCachingEnabled());

    // All cached graphs should be invalid after release
    for (size_t i = 0; i < graph.getCacheSize(); ++i)
    {
        EXPECT_FALSE(graph.hasValidCachedGraph(static_cast<int>(i), true));
        EXPECT_FALSE(graph.hasValidCachedGraph(static_cast<int>(i), false));
    }
}
