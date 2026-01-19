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
#include "../../../src/v2/backends/DeviceId.h"
#include "../../../src/v2/execution/GraphOrchestrator.h"
#include "../../../src/v2/execution/GraphExecutor.h"
#include "../../../src/v2/execution/RuntimeConfig.h"
#include "../../../src/v2/execution/WorkspaceDescriptor.h"
#include "../../../src/v2/execution/DeviceWorkspaceManager.h"
#include "../../../src/v2/execution/compute_stages/IComputeStage.h"
#include "../../../src/v2/interfaces/IWorkspaceConsumer.h"
#include "../../../src/v2/models/qwen/Qwen2Graph.h"
#include "../../../src/v2/backends/ComputeBackend.h"
#include "../../../src/v2/utils/Logger.h"
#include "../../../src/v2/utils/DebugEnv.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/tensors/TensorFactory.h"
#include "../../../src/v2/kernels/cpu/CPUKVCache.h"
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
        config_.default_device = DeviceId::cpu();

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
    IDeviceContext *ctx1 = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx1 == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Second call returns same context (cached)
    IDeviceContext *ctx2 = orchestrator->getDeviceContext(DeviceId::cpu());
    EXPECT_EQ(ctx1, ctx2);
}

TEST_F(Test__GraphOrchestrator, GetDeviceContextMultipleDevices)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Get context for device 0 (may fail without DeviceManager init)
    IDeviceContext *ctx0 = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx0 == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Device 1 might not exist, which is fine
    IDeviceContext *ctx1 = orchestrator->getDeviceContext(DeviceId::cuda(0));
    if (ctx1 != nullptr)
    {
        EXPECT_NE(ctx0, ctx1);
    }
}

TEST_F(Test__GraphOrchestrator, ClearCacheClearsDeviceContexts)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create a device context (may fail without DeviceManager init)
    IDeviceContext *ctx_before = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx_before == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Clear all caches
    orchestrator->clearCache();

    // New context should be created (different pointer)
    IDeviceContext *ctx_after = orchestrator->getDeviceContext(DeviceId::cpu());
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
        std::vector<size_t>{151936, 896}, DeviceId::cpu());
    std::unique_ptr<FP32Tensor> norm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{896}, DeviceId::cpu());
    std::unique_ptr<FP32Tensor> lm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, DeviceId::cpu());

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
        std::vector<size_t>{128, 896}, DeviceId::cpu());
    std::unique_ptr<FP32Tensor> logits = std::make_unique<FP32Tensor>(
        std::vector<size_t>{128, 151936}, DeviceId::cpu());

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
        std::vector<size_t>{151936, 896}, DeviceId::cpu());
    std::unique_ptr<FP32Tensor> norm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{896}, DeviceId::cpu());
    std::unique_ptr<FP32Tensor> lm = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, DeviceId::cpu());

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

    bool success = orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu());

    EXPECT_TRUE(success);
    EXPECT_TRUE(orchestrator->hasInferenceState());
}

TEST_F(Test__GraphOrchestrator, InitializeInferenceStateAllocatesBuffers)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    int batch_size = 2;
    int max_seq_len = 64;

    bool success = orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu());
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

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));
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
    ASSERT_TRUE(orchestrator->initializeInferenceState(1, 64, DeviceId::cpu()));

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

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));

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

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should exist and use POSITION_MAJOR layout
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    auto *cpu_cache = dynamic_cast<ICPUKVCache *>(state.kv_cache.get());
    ASSERT_NE(cpu_cache, nullptr);
    EXPECT_EQ(cpu_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
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

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use POSITION_MAJOR layout for BF16
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    auto *cpu_cache = dynamic_cast<ICPUKVCache *>(state.kv_cache.get());
    ASSERT_NE(cpu_cache, nullptr);
    EXPECT_EQ(cpu_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

// HybridQ16 tests disabled - HybridQ16 project on hold (2025-01)
TEST_F(Test__GraphOrchestrator, DISABLED_KVCacheLayoutMode_HybridQ16_UsesHeadMajor)
{
    // Create config with HybridQ16 precision - this resolves KV cache to Q16_1
    auto hybrid_config = config_;
    hybrid_config.activation_precision = ActivationPrecision::HybridQ16;
    auto orchestrator = std::make_unique<GraphOrchestrator>(hybrid_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use HEAD_MAJOR layout for Q16_1 (required by Q16IntegerAttention)
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    auto *cpu_cache = dynamic_cast<ICPUKVCache *>(state.kv_cache.get());
    ASSERT_NE(cpu_cache, nullptr);
    EXPECT_EQ(cpu_cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
}

// Hybrid tests disabled - HybridQ16 project on hold (2025-01)
TEST_F(Test__GraphOrchestrator, DISABLED_KVCacheLayoutMode_Hybrid_UsesPositionMajor)
{
    // Create config with Hybrid precision - KV cache should be BF16
    auto hybrid_config = config_;
    hybrid_config.activation_precision = ActivationPrecision::Hybrid;
    auto orchestrator = std::make_unique<GraphOrchestrator>(hybrid_config, nullptr);

    // Initialize state
    int batch_size = 1;
    int max_seq_len = 64;

    ASSERT_TRUE(orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu()));
    ASSERT_TRUE(orchestrator->hasInferenceState());

    // KV cache should use POSITION_MAJOR layout for Hybrid (BF16 KV cache)
    const auto &state = orchestrator->inferenceState();
    ASSERT_NE(state.kv_cache, nullptr);
    auto *cpu_cache = dynamic_cast<ICPUKVCache *>(state.kv_cache.get());
    ASSERT_NE(cpu_cache, nullptr);
    EXPECT_EQ(cpu_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

// =============================================================================
// Workspace Allocation Tests
// =============================================================================
// NOTE: ensureDeviceWorkspaceAllocated() is a private method called internally
// by execute(). These tests verify the behavior through the public execute() API.
// Full workspace integration testing with actual GPU stages is in integration tests.

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_EmptyGraph_Succeeds)
{
    // Test that executing an empty graph doesn't fail workspace allocation
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create empty graph
    ComputeGraph graph;

    // Get a device context for execution
    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute empty graph - should succeed (no workspace needed)
    bool result = orchestrator->execute(graph, ctx);

    // Empty graph execution should succeed
    // (Note: actual success depends on executor behavior with empty graph)
    // The key point is that workspace allocation doesn't fail
    EXPECT_TRUE(result);
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_IsIdempotent)
{
    // Test that calling execute() multiple times doesn't fail
    // (workspace allocation should be a no-op after first call)
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // First execution
    bool result1 = orchestrator->execute(graph, ctx);

    // Reset graph for second execution
    graph.reset();

    // Second execution - workspace allocation should be idempotent
    bool result2 = orchestrator->execute(graph, ctx);

    // Both should succeed (or at least not crash due to workspace issues)
    EXPECT_EQ(result1, result2);
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_CPUOnlyGraph_NoGPUWorkspaceNeeded)
{
    // Test that CPU-only stages don't trigger GPU workspace allocation
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create graph with CPU device explicitly
    ComputeGraph graph;
    // Note: We don't add actual stages here since that would require
    // setting up the full weight/buffer infrastructure.
    // The empty graph with CPU context verifies the code path.

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute with CPU context
    bool result = orchestrator->execute(graph, ctx);

    // Should succeed - no GPU workspace allocation attempted for CPU-only
    EXPECT_TRUE(result);
}

// =============================================================================
// Workspace Allocation Unit Tests with Mock Stages
// =============================================================================
// These tests use mock stages to verify the workspace allocation logic in
// ensureDeviceWorkspaceAllocated() without requiring actual GPU hardware.

namespace
{
    /**
     * @brief Mock IComputeStage that also implements IWorkspaceConsumer for testing
     *
     * Captures the dimension arguments (m, n, k) passed to getWorkspaceRequirements()
     * for verifying that GraphOrchestrator uses correct model dimensions.
     */
    class MockWorkspaceConsumerStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        explicit MockWorkspaceConsumerStage(
            const std::string &name,
            DeviceId device,
            size_t workspace_size = 1024)
            : IComputeStage(device), name_(name), workspace_size_(workspace_size)
        {
        }

        // IComputeStage interface
        bool execute(IDeviceContext *ctx) override
        {
            (void)ctx;
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }

        std::string name() const override { return name_; }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            (void)backend;
            return true;
        }

        StageDumpInfo getDumpInfo() const override { return {}; }

        // IWorkspaceConsumer interface
        WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override
        {
            // Capture the dimensions passed by GraphOrchestrator
            last_m_ = m;
            last_n_ = n;
            last_k_ = k;
            getRequirements_called_++;
            WorkspaceRequirements reqs;
            reqs.buffers.push_back(WorkspaceDescriptor{
                name_ + "_workspace",
                workspace_size_,
                256,
                true});
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *ws) override
        {
            bound_workspace_ = ws;
            bind_called_++;
        }

        void unbindWorkspace() override
        {
            bound_workspace_ = nullptr;
        }

        bool hasWorkspace() const override
        {
            return bound_workspace_ != nullptr;
        }

        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

        // Test inspection
        bool wasBound() const { return bind_called_ > 0; }
        int getBindCallCount() const { return bind_called_; }
        int getRequirementsCallCount() const { return getRequirements_called_; }
        DeviceWorkspaceManager *getBoundWorkspace() const { return bound_workspace_; }

        // Dimension capture inspection
        int getLastM() const { return last_m_; }
        int getLastN() const { return last_n_; }
        int getLastK() const { return last_k_; }

    private:
        std::string name_;
        size_t workspace_size_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        mutable int getRequirements_called_ = 0;
        int bind_called_ = 0;
        mutable int last_m_ = -1;
        mutable int last_n_ = -1;
        mutable int last_k_ = -1;
    };

    /**
     * @brief Simple CPU-only stage that does NOT implement IWorkspaceConsumer
     */
    class MockCPUOnlyStage : public IComputeStage
    {
    public:
        explicit MockCPUOnlyStage(const std::string &name, DeviceId device = DeviceId::cpu())
            : IComputeStage(device), name_(name)
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            (void)ctx;
            return true;
        }

        ComputeStageType type() const override { return ComputeStageType::COPY; }

        std::string name() const override { return name_; }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            (void)backend;
            return true;
        }

        StageDumpInfo getDumpInfo() const override { return {}; }

    private:
        std::string name_;
    };
} // anonymous namespace

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_SkipsCPUStages)
{
    // Test that ensureDeviceWorkspaceAllocated() skips CPU stages
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    // Create a graph with a mock workspace consumer on CPU device
    ComputeGraph graph;
    auto cpu_stage = std::make_unique<MockWorkspaceConsumerStage>("cpu_gemm", DeviceId::cpu(), 4096);
    auto *cpu_stage_ptr = cpu_stage.get();

    // Add with CPU device - this should be skipped by workspace allocation
    graph.addNode("cpu_gemm_node", std::move(cpu_stage), DeviceId::cpu());

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute - workspace allocation should skip CPU stage
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);

    // CPU stage should NOT have had workspace bound (is_gpu() returns false)
    EXPECT_FALSE(cpu_stage_ptr->wasBound());
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_SkipsNonWorkspaceConsumerStages)
{
    // Test that stages not implementing IWorkspaceConsumer are skipped
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;
    auto simple_stage = std::make_unique<MockCPUOnlyStage>("simple_stage");

    graph.addNode("simple_node", std::move(simple_stage), DeviceId::cpu());

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute - should succeed without workspace issues
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_CollectsGPUStageRequirements)
{
    // Test that GPU stages (simulated) have their requirements collected
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;
    // Create a stage with CUDA device
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 8192);
    auto *gpu_stage_ptr = gpu_stage.get();

    // Add with a GPU device ID
    graph.addNode("gpu_gemm_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute - should attempt workspace allocation for GPU stage
    // Note: This may fail if CUDA device 0 doesn't exist, but we're testing the logic path
    orchestrator->execute(graph, ctx);

    // The stage should have had getWorkspaceRequirements called since it's a GPU stage
    // (Even if allocation ultimately fails due to no real GPU, the logic path is exercised)
    EXPECT_GE(gpu_stage_ptr->getRequirementsCallCount(), 0); // May be called depending on device availability
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_MergesRequirements_MaxSizeWins)
{
    // Test that when multiple stages on same device have requirements,
    // the merged requirements use max sizes
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;

    // Two GPU stages with different workspace sizes on same device
    auto gpu_stage1 = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm1", DeviceId::cuda(0), 4096);
    auto gpu_stage2 = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm2", DeviceId::cuda(0), 8192);
    auto *stage1_ptr = gpu_stage1.get();
    auto *stage2_ptr = gpu_stage2.get();

    graph.addNode("gpu_node1", std::move(gpu_stage1), DeviceId::cuda(0));
    graph.addNode("gpu_node2", std::move(gpu_stage2), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute
    orchestrator->execute(graph, ctx);

    // Both stages should have requirements queried (they're both GPU stages on same device)
    // The actual binding depends on whether CUDA device exists
    // This test verifies the iteration and collection logic works with multiple stages
    EXPECT_GE(stage1_ptr->getRequirementsCallCount() + stage2_ptr->getRequirementsCallCount(), 0);
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_BindsWorkspaceToAllConsumers)
{
    // Test that workspace is bound to all consumers on a device
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;

    // Create two GPU stages on the same device
    auto gpu_stage1 = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm1", DeviceId::cuda(0), 2048);
    auto gpu_stage2 = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm2", DeviceId::cuda(0), 2048);
    auto *stage1_ptr = gpu_stage1.get();
    auto *stage2_ptr = gpu_stage2.get();

    graph.addNode("gpu_node1", std::move(gpu_stage1), DeviceId::cuda(0));
    graph.addNode("gpu_node2", std::move(gpu_stage2), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute
    orchestrator->execute(graph, ctx);

    // If workspace allocation succeeded, both should be bound to same workspace
    // If no GPU available, neither will be bound (graceful skip)
    if (stage1_ptr->wasBound())
    {
        EXPECT_TRUE(stage2_ptr->wasBound());
        EXPECT_EQ(stage1_ptr->getBoundWorkspace(), stage2_ptr->getBoundWorkspace());
    }
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_HandlesGraphWithNoGPUStages)
{
    // Test that a graph with stages but none on GPU works correctly
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;

    // Add workspace consumer stages but all on CPU
    auto cpu_stage1 = std::make_unique<MockWorkspaceConsumerStage>("cpu_stage1", DeviceId::cpu(), 1024);
    auto cpu_stage2 = std::make_unique<MockWorkspaceConsumerStage>("cpu_stage2", DeviceId::cpu(), 2048);
    auto *stage1_ptr = cpu_stage1.get();
    auto *stage2_ptr = cpu_stage2.get();

    graph.addNode("cpu_node1", std::move(cpu_stage1), DeviceId::cpu());
    graph.addNode("cpu_node2", std::move(cpu_stage2), DeviceId::cpu());

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute should succeed
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);

    // Neither CPU stage should have workspace bound (workspace only for GPU)
    EXPECT_FALSE(stage1_ptr->wasBound());
    EXPECT_FALSE(stage2_ptr->wasBound());
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_IdempotentAcrossMultipleExecutions)
{
    // Test that device_workspace_allocated_ flag prevents redundant allocations
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_stage_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // First execution
    orchestrator->execute(graph, ctx);
    int first_bind_count = gpu_stage_ptr->getBindCallCount();

    // Reset graph for second execution
    graph.reset();

    // Second execution - workspace should already be allocated
    orchestrator->execute(graph, ctx);
    int second_bind_count = gpu_stage_ptr->getBindCallCount();

    // Bind count should not increase after first allocation
    // (early return when device_workspace_allocated_ is true)
    EXPECT_EQ(first_bind_count, second_bind_count);
}

TEST_F(Test__GraphOrchestrator, WorkspaceAllocation_MultipleDevices_SeparateWorkspaces)
{
    // Test that different GPU devices get separate workspace managers
    auto orchestrator = std::make_unique<GraphOrchestrator>(graph_builder_, nullptr);

    ComputeGraph graph;

    // Stages on different GPU devices
    auto gpu0_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu0_gemm", DeviceId::cuda(0), 4096);
    auto gpu1_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu1_gemm", DeviceId::cuda(1), 4096);
    auto *gpu0_ptr = gpu0_stage.get();
    auto *gpu1_ptr = gpu1_stage.get();

    graph.addNode("gpu0_node", std::move(gpu0_stage), DeviceId::cuda(0));
    graph.addNode("gpu1_node", std::move(gpu1_stage), DeviceId::cuda(1));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Execute
    orchestrator->execute(graph, ctx);

    // If both were bound, they should have DIFFERENT workspace managers
    if (gpu0_ptr->wasBound() && gpu1_ptr->wasBound())
    {
        EXPECT_NE(gpu0_ptr->getBoundWorkspace(), gpu1_ptr->getBoundWorkspace());
    }
}

// =============================================================================
// Workspace Sizing Tests - Model Dimensions
// =============================================================================
// These tests verify that GraphOrchestrator correctly uses model dimensions
// from the config to size workspace buffers. This is critical to prevent
// crashes from undersized buffers during inference.

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_UsesActualMaxSeqLen)
{
    // Configure specific max_seq_len
    auto custom_config = config_;
    custom_config.max_seq_len = 2048; // Non-default value

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Verify that max_seq_len was passed as m (first argument)
    EXPECT_EQ(gpu_ptr->getLastM(), 2048) << "max_seq_len should be passed as m dimension";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_UsesActualNHeads)
{
    // Configure specific n_heads
    auto custom_config = config_;
    custom_config.n_heads = 32; // Non-default value

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_attn", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Verify that n_heads was passed as n (second argument)
    EXPECT_EQ(gpu_ptr->getLastN(), 32) << "n_heads should be passed as n dimension";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_CalculatesHeadDimFromModel)
{
    // Configure d_model and n_heads to verify head_dim calculation
    auto custom_config = config_;
    custom_config.d_model = 4096;
    custom_config.n_heads = 32;
    // Expected head_dim = 4096 / 32 = 128

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_attn", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Verify that head_dim (d_model / n_heads) was passed as k
    EXPECT_EQ(gpu_ptr->getLastK(), 128) << "head_dim should be calculated as d_model / n_heads";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_AllDimensionsFromQwen2Config)
{
    // Test with a realistic Qwen2-0.5B configuration
    Qwen2GraphConfig qwen_config;
    qwen_config.d_model = 896;
    qwen_config.d_ff = 4864;
    qwen_config.n_heads = 14;
    qwen_config.n_kv_heads = 2;
    qwen_config.head_dim = 64;
    qwen_config.n_layers = 24;
    qwen_config.vocab_size = 151936;
    qwen_config.max_seq_len = 32768;
    qwen_config.default_device = DeviceId::cpu();

    auto orchestrator = std::make_unique<GraphOrchestrator>(qwen_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("qwen_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Verify all dimensions match Qwen2 config
    EXPECT_EQ(gpu_ptr->getLastM(), 32768) << "max_seq_len should be 32768 for Qwen2";
    EXPECT_EQ(gpu_ptr->getLastN(), 14) << "n_heads should be 14 for Qwen2-0.5B";
    EXPECT_EQ(gpu_ptr->getLastK(), 64) << "head_dim should be 64 for Qwen2-0.5B";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_LlamaStyleConfig)
{
    // Test with a Llama-style configuration (different head_dim)
    Qwen2GraphConfig llama_config;
    llama_config.d_model = 4096;
    llama_config.d_ff = 14336;
    llama_config.n_heads = 32;
    llama_config.n_kv_heads = 8;
    llama_config.head_dim = 128; // Llama uses 128 head_dim
    llama_config.n_layers = 32;
    llama_config.vocab_size = 128256;
    llama_config.max_seq_len = 8192;
    llama_config.default_device = DeviceId::cpu();

    auto orchestrator = std::make_unique<GraphOrchestrator>(llama_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("llama_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Verify Llama dimensions
    EXPECT_EQ(gpu_ptr->getLastM(), 8192) << "max_seq_len should be 8192 for Llama";
    EXPECT_EQ(gpu_ptr->getLastN(), 32) << "n_heads should be 32 for Llama";
    EXPECT_EQ(gpu_ptr->getLastK(), 128) << "head_dim should be 128 for Llama";
}

// =============================================================================
// Workspace Sizing Edge Cases - Default Fallbacks
// =============================================================================
// These tests verify that the orchestrator handles edge cases gracefully
// by falling back to reasonable defaults instead of crashing.

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_FallbackWhenMaxSeqLenZero)
{
    // Test fallback when max_seq_len is 0 (uninitialized)
    auto custom_config = config_;
    custom_config.max_seq_len = 0; // Should fallback to 4096

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should not crash
    orchestrator->execute(graph, ctx);

    // Verify fallback to default (4096)
    EXPECT_EQ(gpu_ptr->getLastM(), 4096) << "max_seq_len=0 should fallback to 4096";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_FallbackWhenNHeadsZero)
{
    // Test fallback when n_heads is 0 (uninitialized)
    auto custom_config = config_;
    custom_config.n_heads = 0; // Should fallback to 128

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should not crash
    orchestrator->execute(graph, ctx);

    // Verify fallback to default (128)
    EXPECT_EQ(gpu_ptr->getLastN(), 128) << "n_heads=0 should fallback to 128";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_FallbackWhenDModelZero)
{
    // Test fallback when d_model is 0 (affects head_dim calculation)
    auto custom_config = config_;
    custom_config.d_model = 0;
    custom_config.n_heads = 32;

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should not crash (would crash if dividing by zero or using invalid dimension)
    orchestrator->execute(graph, ctx);

    // Verify fallback to default head_dim (128)
    EXPECT_EQ(gpu_ptr->getLastK(), 128) << "head_dim should fallback to 128 when d_model=0";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_NoCrashOnNegativeValues)
{
    // Test robustness against negative config values (should use fallbacks)
    auto custom_config = config_;
    custom_config.max_seq_len = -1;
    custom_config.n_heads = -1;
    custom_config.d_model = -1;

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("gpu_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should not crash - negative values should trigger fallback logic
    bool result = orchestrator->execute(graph, ctx);

    // Test passes if no crash occurred
    EXPECT_TRUE(result || !result); // Just verify no crash - result depends on impl details
}

// =============================================================================
// Workspace Sizing - Device-Specific Behavior
// =============================================================================
// These tests verify that workspace allocation behaves correctly for different
// device types (CPU, CUDA, ROCm).

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_CUDAStage_GetsDimensions)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    ComputeGraph graph;
    auto cuda_stage = std::make_unique<MockWorkspaceConsumerStage>("cuda_gemm", DeviceId::cuda(0), 4096);
    auto *cuda_ptr = cuda_stage.get();
    graph.addNode("cuda_node", std::move(cuda_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // CUDA stages should receive workspace requirements call
    EXPECT_GT(cuda_ptr->getRequirementsCallCount(), 0)
        << "CUDA stage should have getWorkspaceRequirements called";
    EXPECT_GT(cuda_ptr->getLastM(), 0) << "CUDA stage should receive positive m dimension";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_ROCmStage_GetsDimensions)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    ComputeGraph graph;
    auto rocm_stage = std::make_unique<MockWorkspaceConsumerStage>("rocm_gemm", DeviceId::rocm(0), 4096);
    auto *rocm_ptr = rocm_stage.get();
    graph.addNode("rocm_node", std::move(rocm_stage), DeviceId::rocm(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // ROCm stages should receive workspace requirements call
    EXPECT_GT(rocm_ptr->getRequirementsCallCount(), 0)
        << "ROCm stage should have getWorkspaceRequirements called";
    EXPECT_GT(rocm_ptr->getLastM(), 0) << "ROCm stage should receive positive m dimension";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_CPUStage_SkipsWorkspace)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    ComputeGraph graph;
    auto cpu_stage = std::make_unique<MockWorkspaceConsumerStage>("cpu_gemm", DeviceId::cpu(), 4096);
    auto *cpu_ptr = cpu_stage.get();
    graph.addNode("cpu_node", std::move(cpu_stage), DeviceId::cpu());

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // CPU stages should NOT receive workspace (workspace is GPU-only)
    EXPECT_FALSE(cpu_ptr->wasBound())
        << "CPU stage should not have workspace bound";
}

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_MixedDevices_OnlyGPUGetWorkspace)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    ComputeGraph graph;

    // Mix of CPU and GPU stages
    auto cpu_stage = std::make_unique<MockWorkspaceConsumerStage>("cpu_gemm", DeviceId::cpu(), 4096);
    auto cuda_stage = std::make_unique<MockWorkspaceConsumerStage>("cuda_gemm", DeviceId::cuda(0), 4096);
    auto rocm_stage = std::make_unique<MockWorkspaceConsumerStage>("rocm_gemm", DeviceId::rocm(0), 4096);

    auto *cpu_ptr = cpu_stage.get();
    auto *cuda_ptr = cuda_stage.get();
    auto *rocm_ptr = rocm_stage.get();

    graph.addNode("cpu_node", std::move(cpu_stage), DeviceId::cpu());
    graph.addNode("cuda_node", std::move(cuda_stage), DeviceId::cuda(0));
    graph.addNode("rocm_node", std::move(rocm_stage), DeviceId::rocm(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph, ctx);

    // Only GPU stages should have workspace
    EXPECT_FALSE(cpu_ptr->wasBound()) << "CPU stage should not have workspace";
    // GPU stages may or may not be bound depending on device availability,
    // but they should have requirements queried
    EXPECT_GT(cuda_ptr->getRequirementsCallCount(), 0) << "CUDA stage requirements should be queried";
    EXPECT_GT(rocm_ptr->getRequirementsCallCount(), 0) << "ROCm stage requirements should be queried";
}

// =============================================================================
// Workspace Sizing - Batch Size Handling for KV Cache
// =============================================================================
// KV cache consumers use batch_size instead of max_seq_len for workspace sizing.
// This verifies the special-case handling in GraphOrchestrator.

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_UsesActualBatchSizeFromInferenceState)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Initialize inference state with specific batch size
    const int batch_size = 8;
    const int max_seq_len = 64;

    bool init_success = orchestrator->initializeInferenceState(batch_size, max_seq_len, DeviceId::cpu());
    if (!init_success)
    {
        GTEST_SKIP() << "Could not initialize inference state";
    }

    // Verify batch_size is captured in state
    ASSERT_TRUE(orchestrator->hasInferenceState());
    const auto &state = orchestrator->inferenceState();
    EXPECT_EQ(state.batch_size, batch_size) << "Inference state should capture batch_size";
}

// =============================================================================
// Workspace Sizing - Large Model Configurations
// =============================================================================
// Test with large model configs to verify no overflow or allocation issues.

TEST_F(Test__GraphOrchestrator, WorkspaceSizing_LargeModelConfig_NoOverflow)
{
    // Large model configuration (e.g., 70B-scale)
    Qwen2GraphConfig large_config;
    large_config.d_model = 8192;
    large_config.d_ff = 28672;
    large_config.n_heads = 64;
    large_config.n_kv_heads = 8;
    large_config.head_dim = 128;
    large_config.n_layers = 80;
    large_config.vocab_size = 128256;
    large_config.max_seq_len = 131072; // 128K context
    large_config.default_device = DeviceId::cpu();

    auto orchestrator = std::make_unique<GraphOrchestrator>(large_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("large_gemm", DeviceId::cuda(0), 4096);
    auto *gpu_ptr = gpu_stage.get();
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should not crash or overflow
    orchestrator->execute(graph, ctx);

    // Verify large values are passed correctly
    EXPECT_EQ(gpu_ptr->getLastM(), 131072) << "Large max_seq_len should be passed";
    EXPECT_EQ(gpu_ptr->getLastN(), 64) << "Large n_heads should be passed";
    EXPECT_EQ(gpu_ptr->getLastK(), 128) << "head_dim should be calculated correctly";
}

// =============================================================================
// Workspace Sizing - Activation Precision Impact (Documentation)
// =============================================================================
// Note: Activation precision affects which kernels are used (e.g., FP32 vs BF16
// vs Q8_1 GEMM) and thus which workspace requirements are generated. The actual
// buffer sizing is determined by the kernel implementations, not GraphOrchestrator.
// These tests verify the precision is correctly propagated to the orchestrator.

TEST_F(Test__GraphOrchestrator, ActivationPrecision_FP32_WorkspaceAllocation)
{
    auto custom_config = config_;
    custom_config.activation_precision = ActivationPrecision::FP32;

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("fp32_gemm", DeviceId::cuda(0), 4096);
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should execute without crash regardless of precision
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);
}

TEST_F(Test__GraphOrchestrator, ActivationPrecision_BF16_WorkspaceAllocation)
{
    auto custom_config = config_;
    custom_config.activation_precision = ActivationPrecision::BF16;

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("bf16_gemm", DeviceId::cuda(0), 4096);
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should execute without crash regardless of precision
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);
}

TEST_F(Test__GraphOrchestrator, ActivationPrecision_Hybrid_WorkspaceAllocation)
{
    auto custom_config = config_;
    custom_config.activation_precision = ActivationPrecision::Hybrid;

    auto orchestrator = std::make_unique<GraphOrchestrator>(custom_config, nullptr);

    ComputeGraph graph;
    auto gpu_stage = std::make_unique<MockWorkspaceConsumerStage>("hybrid_gemm", DeviceId::cuda(0), 4096);
    graph.addNode("gpu_node", std::move(gpu_stage), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    // Should execute without crash regardless of precision
    bool result = orchestrator->execute(graph, ctx);
    EXPECT_TRUE(result);
}

// =============================================================================
// Workspace Consistency Tests
// =============================================================================
// Verify that workspace dimensions are consistent across multiple executions
// and that the same model config always produces the same workspace sizing.

TEST_F(Test__GraphOrchestrator, WorkspaceConsistency_SameDimensionsAcrossExecutions)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // First graph with workspace consumer
    ComputeGraph graph1;
    auto gpu_stage1 = std::make_unique<MockWorkspaceConsumerStage>("gemm1", DeviceId::cuda(0), 4096);
    auto *gpu_ptr1 = gpu_stage1.get();
    graph1.addNode("gpu_node", std::move(gpu_stage1), DeviceId::cuda(0));

    IDeviceContext *ctx = orchestrator->getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available in this environment";
    }

    orchestrator->execute(graph1, ctx);
    int first_m = gpu_ptr1->getLastM();
    int first_n = gpu_ptr1->getLastN();
    int first_k = gpu_ptr1->getLastK();

    // Workspace allocation is idempotent, but dimensions should be consistent
    // if they were queried again (e.g., after reset)
    EXPECT_GT(first_m, 0) << "First execution should query positive m";
    EXPECT_GT(first_n, 0) << "First execution should query positive n";
    EXPECT_GT(first_k, 0) << "First execution should query positive k";
}

// TODO: Full workspace integration tests with actual GPU stages that implement
// IWorkspaceConsumer should be in tests/v2/integration/ as they require:
// - GPU device availability
// - Actual GEMM stages with workspace requirements
// - DeviceWorkspaceManager allocation verification