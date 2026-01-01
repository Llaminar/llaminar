/**
 * @file Test__GraphOrchestrator.cpp
 * @brief Unit tests for GraphOrchestrator class
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the orchestrator's execution, caching, and device context management
 * functionality, verifying clean separation from graph building.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/execution/GraphOrchestrator.h"
#include "../../../src/v2/execution/RuntimeConfig.h"
#include "../../../src/v2/models/qwen/Qwen2Graph.h"
#include "../../../src/v2/backends/ComputeBackend.h"
#include "../../../src/v2/utils/Logger.h"
#include "../../../src/v2/utils/DebugEnv.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/tensors/TensorFactory.h"
#include "../../../src/v2/tensors/UnifiedKVCache.h"
#include <memory>

using namespace llaminar2;

/**
 * @brief Test fixture for GraphOrchestrator unit tests
 */
class Test__GraphOrchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager (required for DeviceContext creation)
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

        // Create minimal config for testing
        config_.d_model = 896;
        config_.d_ff = 4864;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.n_layers = 24;
        config_.vocab_size = 151936;
        config_.rms_norm_eps = 1e-6f;
        config_.rope_theta = 1000000.0f;
        config_.default_device = 0;

        // Create graph builder
        graph_builder_ = std::make_shared<Qwen2Graph>(config_, nullptr);
    }

    Qwen2GraphConfig config_;
    std::shared_ptr<Qwen2Graph> graph_builder_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, ConstructWithGraphBuilder)
{
    // Test construction with existing graph builder
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    EXPECT_NE(orchestrator, nullptr);
    EXPECT_EQ(orchestrator->graphBuilder(), graph_builder_.get());
    EXPECT_TRUE(orchestrator->isGraphCachingEnabled());
}

TEST_F(Test__GraphOrchestrator, ConstructWithConfig)
{
    // Test construction with config (creates internal graph builder)
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    EXPECT_NE(orchestrator, nullptr);
    EXPECT_NE(orchestrator->graphBuilder(), nullptr);
    EXPECT_TRUE(orchestrator->isGraphCachingEnabled());
}

TEST_F(Test__GraphOrchestrator, ConstructWithCacheDisabled)
{
    GraphCacheConfig cache_config;
    cache_config.enabled = false;

    auto orchestrator = std::make_unique<GraphOrchestrator>(
        graph_builder_, nullptr, cache_config);

    EXPECT_FALSE(orchestrator->isGraphCachingEnabled());
}

TEST_F(Test__GraphOrchestrator, NullGraphBuilderThrows)
{
    // Construction with null graph builder should throw
    EXPECT_THROW(
        GraphOrchestrator(std::shared_ptr<Qwen2Graph>(nullptr), nullptr),
        std::invalid_argument);
}

// =============================================================================
// Cache Management Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, InitializeGraphCache)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Before initialization, no cached graphs
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(0, true));
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(0, false));

    // Initialize cache
    orchestrator->initializeGraphCache(24);

    // Still no cached graphs (they're created on first execution)
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(0, true));
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(0, false));
}

TEST_F(Test__GraphOrchestrator, SetGraphCachingEnabled)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    EXPECT_TRUE(orchestrator->isGraphCachingEnabled());

    orchestrator->setGraphCachingEnabled(false);
    EXPECT_FALSE(orchestrator->isGraphCachingEnabled());

    orchestrator->setGraphCachingEnabled(true);
    EXPECT_TRUE(orchestrator->isGraphCachingEnabled());
}

TEST_F(Test__GraphOrchestrator, ClearCacheResetsCacheStats)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    // Clear cache
    orchestrator->clearCache();

    // Verify stats are reset
    auto stats = orchestrator->getCacheStats();
    EXPECT_EQ(stats.attention_cache_hits, 0u);
    EXPECT_EQ(stats.attention_cache_misses, 0u);
    EXPECT_EQ(stats.ffn_cache_hits, 0u);
    EXPECT_EQ(stats.ffn_cache_misses, 0u);
}

TEST_F(Test__GraphOrchestrator, InvalidateGraphCacheSpecificLayer)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    // Invalidate specific layer (should not throw even if nothing cached)
    EXPECT_NO_THROW(orchestrator->invalidateGraphCache(5));
}

TEST_F(Test__GraphOrchestrator, InvalidateGraphCacheAllLayers)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    // Invalidate all layers
    EXPECT_NO_THROW(orchestrator->invalidateGraphCache(-1));
}

// =============================================================================
// Device Context Tests
// Note: These tests may skip in environments without DeviceManager initialization
// =============================================================================

TEST_F(Test__GraphOrchestrator, GetDeviceContextLazyCreation)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // First call creates context (may fail without DeviceManager init)
    IDeviceContext *ctx1 = orchestrator->getDeviceContext(0);
    if (ctx1 == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Second call returns same context (cached)
    IDeviceContext *ctx2 = orchestrator->getDeviceContext(0);
    EXPECT_EQ(ctx1, ctx2);
}

TEST_F(Test__GraphOrchestrator, GetDeviceContextMultipleDevices)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Get context for device 0 (may fail without DeviceManager init)
    IDeviceContext *ctx0 = orchestrator->getDeviceContext(0);
    if (ctx0 == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Device 1 might not exist, which is fine
    IDeviceContext *ctx1 = orchestrator->getDeviceContext(1);
    if (ctx1 != nullptr)
    {
        EXPECT_NE(ctx0, ctx1);
    }
}

TEST_F(Test__GraphOrchestrator, ClearCacheClearsDeviceContexts)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create a device context (may fail without DeviceManager init)
    IDeviceContext *ctx_before = orchestrator->getDeviceContext(0);
    if (ctx_before == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Clear all caches
    orchestrator->clearCache();

    // New context should be created (different pointer)
    IDeviceContext *ctx_after = orchestrator->getDeviceContext(0);
    EXPECT_NE(ctx_after, nullptr);
    // Note: Can't guarantee different pointer since OS might reuse memory
}

// =============================================================================
// Cache Statistics Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, CacheStatsInitialValues)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    auto stats = orchestrator->getCacheStats();
    EXPECT_EQ(stats.attention_cache_hits, 0u);
    EXPECT_EQ(stats.attention_cache_misses, 0u);
    EXPECT_EQ(stats.ffn_cache_hits, 0u);
    EXPECT_EQ(stats.ffn_cache_misses, 0u);
    EXPECT_EQ(stats.cached_layers, 0u);
}

TEST_F(Test__GraphOrchestrator, CacheStatsAfterInitialize)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    auto stats = orchestrator->getCacheStats();
    EXPECT_EQ(stats.cached_layers, 24u);
}

// =============================================================================
// GraphCacheConfig Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, CustomCacheConfig)
{
    GraphCacheConfig cache_config;
    cache_config.enabled = true;
    cache_config.decode_seq_len = 2;      // Custom decode threshold
    cache_config.cache_attention = false; // Disable attention caching
    cache_config.cache_ffn = true;

    auto orchestrator = std::make_unique<GraphOrchestrator>(
        graph_builder_, nullptr, cache_config);

    EXPECT_TRUE(orchestrator->isGraphCachingEnabled());
    // Note: We can't directly verify decode_seq_len or per-graph-type caching
    // without executing, but the construction should work
}

// =============================================================================
// Executor Access Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, ExecutorAccess)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Non-const access
    GraphExecutor &exec = orchestrator->executor();
    (void)exec; // Just verify we can access it

    // Const access
    const GraphOrchestrator *const_orch = orchestrator.get();
    const GraphExecutor &const_exec = const_orch->executor();
    (void)const_exec; // Just verify we can access it
}

TEST_F(Test__GraphOrchestrator, GraphBuilderAccess)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Non-const access
    Qwen2Graph *builder = orchestrator->graphBuilder();
    EXPECT_EQ(builder, graph_builder_.get());

    // Const access
    const GraphOrchestrator *const_orch = orchestrator.get();
    const Qwen2Graph *const_builder = const_orch->graphBuilder();
    EXPECT_EQ(const_builder, graph_builder_.get());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, HasValidCachedGraphOutOfBounds)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    // Layer index out of bounds should return false
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(-1, true));
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(100, true));
    EXPECT_FALSE(orchestrator->hasValidCachedGraph(24, false)); // Exactly at boundary
}

TEST_F(Test__GraphOrchestrator, InvalidateGraphCacheOutOfBounds)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator->initializeGraphCache(24);

    // Should not throw for out-of-bounds indices
    EXPECT_NO_THROW(orchestrator->invalidateGraphCache(100));
    EXPECT_NO_THROW(orchestrator->invalidateGraphCache(-2));
}

TEST_F(Test__GraphOrchestrator, MoveConstruction)
{
    auto orchestrator1 = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator1->initializeGraphCache(24);

    // Move construct
    auto orchestrator2 = std::move(orchestrator1);

    EXPECT_NE(orchestrator2, nullptr);
    EXPECT_EQ(orchestrator2->graphBuilder(), graph_builder_.get());
}

TEST_F(Test__GraphOrchestrator, MoveAssignment)
{
    auto orchestrator1 = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);
    orchestrator1->initializeGraphCache(24);

    auto orchestrator2 = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Move assign
    *orchestrator2 = std::move(*orchestrator1);

    EXPECT_EQ(orchestrator2->graphBuilder(), graph_builder_.get());
}

// =============================================================================
// Weight Configuration Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, SetWeightsDelegatesToGraphBuilder)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create mock weights (don't need valid data, just pointers)
    Qwen2ModelWeights weights;
    std::unique_ptr<FP32Tensor> embed = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, 0);
    std::unique_ptr<FP32Tensor> norm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{896}, 0);
    std::unique_ptr<FP32Tensor> lm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, 0);

    weights.embedding_table = embed.get();
    weights.final_norm = norm.get();
    weights.lm_head = lm.get();
    weights.get_layer_weights = [](int) -> Qwen2LayerWeights
    { return {}; };

    // Set weights via orchestrator
    orchestrator->setWeights(weights);

    // Verify graph builder's isInitialized returns true
    EXPECT_TRUE(orchestrator->graphBuilder()->isInitialized());
}

TEST_F(Test__GraphOrchestrator, SetBuffersDelegatesToGraphBuilder)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create mock buffers
    Qwen2ModelBuffers buffers;
    std::unique_ptr<FP32Tensor> hidden = std::make_unique<FP32Tensor>(
        std::vector<size_t>{128, 896}, 0);
    std::unique_ptr<FP32Tensor> logits = std::make_unique<FP32Tensor>(
        std::vector<size_t>{128, 151936}, 0);

    buffers.current_hidden = hidden.get();
    buffers.logits = logits.get();

    // Set buffers via orchestrator (should not throw)
    EXPECT_NO_THROW(orchestrator->setBuffers(buffers));
}

TEST_F(Test__GraphOrchestrator, HasGlobalWeightsReturnsFalseWhenNotSet)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Before setting weights, hasGlobalWeights should return false
    // (isInitialized checks get_layer_weights callback)
    EXPECT_FALSE(orchestrator->hasGlobalWeights());
}

TEST_F(Test__GraphOrchestrator, HasGlobalWeightsReturnsTrueWhenSet)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Set up minimal weights with layer accessor
    Qwen2ModelWeights weights;
    std::unique_ptr<FP32Tensor> embed = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, 0);
    std::unique_ptr<FP32Tensor> norm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{896}, 0);
    std::unique_ptr<FP32Tensor> lm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, 0);

    weights.embedding_table = embed.get();
    weights.final_norm = norm.get();
    weights.lm_head = lm.get();
    weights.get_layer_weights = [](int) -> Qwen2LayerWeights
    { return {}; };

    orchestrator->setWeights(weights);

    // After setting weights with layer accessor, hasGlobalWeights should return true
    EXPECT_TRUE(orchestrator->hasGlobalWeights());
}
// =============================================================================
// Inference State Management Tests (Phase 5)
// =============================================================================

TEST_F(Test__GraphOrchestrator, InferenceStateNotInitializedByDefault)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // State should not be initialized by default
    EXPECT_FALSE(orchestrator->hasInferenceState());
}

TEST_F(Test__GraphOrchestrator, InitializeInferenceStateSuccess)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Initialize state
    int batch_size = 2;
    int max_seq_len = 64;
    int device_idx = 0;

    bool success = orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx);

    EXPECT_TRUE(success);
    EXPECT_TRUE(orchestrator->hasInferenceState());
}

TEST_F(Test__GraphOrchestrator, InitializeInferenceStateAllocatesBuffers)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    int batch_size = 2;
    int max_seq_len = 64;
    int device_idx = 0;

    bool success = orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx);
    ASSERT_TRUE(success);

    // Should be able to access logits (nullptr check, not actual data yet)
    // Logits are initialized but not computed until forward() is called
    EXPECT_NE(orchestrator->logits(), nullptr);
}

TEST_F(Test__GraphOrchestrator, ClearInferenceStateResetsPositions)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    int batch_size = 2;
    int max_seq_len = 64;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // Clear state
    orchestrator->clearInferenceState();

    // Positions should be reset to 0
    EXPECT_EQ(orchestrator->getPosition(0), 0);
    EXPECT_EQ(orchestrator->getPosition(1), 0);
}

TEST_F(Test__GraphOrchestrator, GetPositionReturnsZeroForUninitializedState)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Without initialized state, getPosition should return 0
    EXPECT_EQ(orchestrator->getPosition(0), 0);
    EXPECT_EQ(orchestrator->getPosition(99), 0); // Out of bounds also returns 0
}

TEST_F(Test__GraphOrchestrator, LogitsReturnsNullptrWhenNotInitialized)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Without initialized state, logits() should return nullptr
    EXPECT_EQ(orchestrator->logits(), nullptr);
}

TEST_F(Test__GraphOrchestrator, ForwardFailsWithoutState)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // forward() should fail without initialized state
    std::vector<int> tokens = {1, 2, 3};
    const float *result = orchestrator->forward(tokens.data(), tokens.size(), 1);

    EXPECT_EQ(result, nullptr);
}

TEST_F(Test__GraphOrchestrator, ForwardFailsWithoutWeights)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Initialize state but not weights
    ASSERT_TRUE(orchestrator->initializeInferenceState(1, 64, 0));

    // forward() should fail without weights
    std::vector<int> tokens = {1, 2, 3};
    const float *result = orchestrator->forward(tokens.data(), tokens.size(), 1);

    EXPECT_EQ(result, nullptr);
}

TEST_F(Test__GraphOrchestrator, InferenceStateMultipleBatches)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    int batch_size = 4;
    int max_seq_len = 128;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));

    // All batch positions should start at 0
    for (int b = 0; b < batch_size; ++b)
    {
        EXPECT_EQ(orchestrator->getPosition(b), 0);
    }
}

// =============================================================================
// KV Cache Layout Mode Tests
// =============================================================================

TEST_F(Test__GraphOrchestrator, KVCacheLayoutMode_FP32_UsesPositionMajor)
{
    // Create config with FP32 precision
    auto fp32_config = config_;
    fp32_config.activation_precision = ActivationPrecision::FP32;
    auto orchestrator = std::make_unique<GraphOrchestrator>(fp32_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should exist and use POSITION_MAJOR layout
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    EXPECT_EQ(state.kv_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

TEST_F(Test__GraphOrchestrator, KVCacheLayoutMode_BF16_UsesPositionMajor)
{
    // Create config with BF16 precision
    auto bf16_config = config_;
    bf16_config.activation_precision = ActivationPrecision::BF16;
    auto orchestrator = std::make_unique<GraphOrchestrator>(bf16_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use POSITION_MAJOR layout for BF16
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    EXPECT_EQ(state.kv_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

TEST_F(Test__GraphOrchestrator, KVCacheLayoutMode_HybridQ16_UsesHeadMajor)
{
    // Create config with HybridQ16 precision - this resolves KV cache to Q16_1
    auto hybrid_config = config_;
    hybrid_config.activation_precision = ActivationPrecision::HybridQ16;
    auto orchestrator = std::make_unique<GraphOrchestrator>(hybrid_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use HEAD_MAJOR layout for Q16_1 (required by Q16IntegerAttention)
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    EXPECT_EQ(state.kv_cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
}

TEST_F(Test__GraphOrchestrator, KVCacheLayoutMode_Hybrid_UsesPositionMajor)
{
    // Create config with Hybrid precision - KV cache should be BF16
    auto hybrid_config = config_;
    hybrid_config.activation_precision = ActivationPrecision::Hybrid;
    auto orchestrator = std::make_unique<GraphOrchestrator>(hybrid_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;
    int device_idx = 0;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, device_idx));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use POSITION_MAJOR layout for Hybrid (BF16 KV cache)
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    EXPECT_EQ(state.kv_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}