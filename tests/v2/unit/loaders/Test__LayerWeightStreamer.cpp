/**
 * @file Test__LayerWeightStreamer.cpp
 * @brief Device-free construction, CPU no-transfer and statistics contracts.
 *
 * A real WeightManager borrows the shared in-memory loader fixture. These
 * checks need neither model files nor native devices: construction validates
 * layer geometry and CPU requests exercise the no-transfer lifecycle. GPU
 * transfer, eviction and asynchronous behavior require native integration tests.
 */

#include "loaders/LayerWeightStreamer.h"
#include "loaders/WeightManager.h"
#include "../../mocks/MockModelLoader.h"
#include "backends/DeviceId.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for LayerWeightStreamer
 *
 * The loader outlives its borrowing manager. No tensors are required for
 * constructor checks or CPU no-transfer requests, and no device is initialized.
 */
class LayerWeightStreamerTest : public ::testing::Test
{
protected:
    /** @brief Construct the real manager with a model-free source and a small declared budget. */
    void SetUp() override
    {
        // Test configuration with small budget for testing eviction
        config_.gpu_memory_budget = 3 * 9 * 1024; // A small explicit test budget; no allocation occurs.
        config_.cache_layer_count = 2;
        config_.eviction_policy = StreamingEvictionPolicy::LRU;
        config_.enable_prefetch = true;
        config_.log_transfer_stats = false;

        weight_manager_ = std::make_shared<WeightManager>(loader_);
    }

    test::MockModelLoader loader_; ///< Shared source fixture, retained past manager destruction.
    std::shared_ptr<WeightManager> weight_manager_; ///< Actual production manager, with no native/model resources.
    StreamingConfig config_;
    static constexpr int NUM_LAYERS = 24;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, ConstructionWithNullWeightManagerThrows)
{
    EXPECT_THROW(
        LayerWeightStreamer(nullptr, NUM_LAYERS, config_),
        std::invalid_argument);
}

TEST_F(LayerWeightStreamerTest, ConstructionWithZeroLayersThrows)
{
    EXPECT_THROW(LayerWeightStreamer(weight_manager_, 0, config_), std::invalid_argument);
}

TEST_F(LayerWeightStreamerTest, ConstructionWithNegativeLayersThrows)
{
    EXPECT_THROW(LayerWeightStreamer(weight_manager_, -1, config_), std::invalid_argument);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, MemoryBudgetReturnsConfiguredValue)
{
    LayerWeightStreamer streamer(weight_manager_, NUM_LAYERS, config_);
    EXPECT_EQ(streamer.memoryBudget(), config_.gpu_memory_budget);
    EXPECT_EQ(streamer.currentDeviceMemoryUsage(), 0u);
}

TEST_F(LayerWeightStreamerTest, InitialCacheIsEmpty)
{
    LayerWeightStreamer streamer(weight_manager_, NUM_LAYERS, config_);
    EXPECT_EQ(streamer.currentDeviceMemoryUsage(), 0u);
    for (int layer = 0; layer < NUM_LAYERS; ++layer)
    {
        EXPECT_FALSE(streamer.isLayerCached(layer, DeviceId::cpu()));
        EXPECT_FALSE(streamer.isPrefetchInProgress(layer));
        EXPECT_FALSE(streamer.evictLayer(layer));
    }
}

/** @test A rejected streaming request inspects source metadata without loading CPU weights or touching a GPU. */
TEST_F(LayerWeightStreamerTest, SourceSizeProbeDoesNotMaterializeCPUWeights)
{
    loader_.addFP32OnesTensor("blk.0.attn_q.weight", {32, 64});
    config_.gpu_memory_budget = 1;
    config_.eviction_policy = StreamingEvictionPolicy::NONE;
    config_.enable_prefetch = false;
    LayerWeightStreamer streamer(weight_manager_, NUM_LAYERS, config_);
    EXPECT_EQ(weight_manager_->originalWeightSourceBytes("blk.0.attn_q.weight"), 32u * 64u * sizeof(float));
    EXPECT_EQ(weight_manager_->originalWeightSourceBytes("absent.weight"), 0u);
    // Nonexistent ordinals prove the capacity refusal precedes native work.
    for (const auto device : {DeviceId::cuda(71), DeviceId::rocm(93)})
        EXPECT_FALSE(streamer.ensureLayerOnDevice(0, device));
    EXPECT_EQ(loader_.loadTensorCallCount(), 0u);
    EXPECT_EQ(loader_.nativeSourceReadCallCount(), 0u);
    EXPECT_EQ(weight_manager_->cacheSize(), 0u);
    EXPECT_EQ(streamer.currentDeviceMemoryUsage(), 0u);
}

// =============================================================================
// CPU Device Tests (No actual streaming needed)
// =============================================================================

/**
 * @test Valid CPU layer requests perform no transfer or cache mutation.
 *
 * The streamer does not certify loaded model weights. Its CPU branch only
 * validates the layer index and leaves transfer ownership with the caller.
 */
TEST_F(LayerWeightStreamerTest, CPUDeviceAlwaysSucceeds)
{
    LayerWeightStreamer streamer(weight_manager_, NUM_LAYERS, config_);
    for (int layer : {0, NUM_LAYERS - 1})
    {
        EXPECT_TRUE(streamer.ensureLayerOnDevice(layer, DeviceId::cpu()));
        streamer.prefetchLayer(layer, DeviceId::cpu());
        streamer.releaseLayer(layer);
        EXPECT_FALSE(streamer.isLayerCached(layer, DeviceId::cpu()));
        EXPECT_FALSE(streamer.isPrefetchInProgress(layer));
    }
    EXPECT_FALSE(streamer.ensureLayerOnDevice(-1, DeviceId::cpu()));
    EXPECT_FALSE(streamer.ensureLayerOnDevice(NUM_LAYERS, DeviceId::cpu()));
    streamer.synchronize();
    streamer.clearCache();
    EXPECT_EQ(streamer.currentDeviceMemoryUsage(), 0u);
    const auto observed = streamer.stats();
    EXPECT_EQ(observed.layers_transferred, 0u);
    EXPECT_EQ(observed.cache_hits, 0u);
    EXPECT_EQ(observed.cache_misses, 0u);
    EXPECT_EQ(observed.bytes_transferred, 0u);
}

// =============================================================================
// Stats Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, InitialStatsAreZero)
{
    StreamingStats stats;

    // Verify initial state of stats struct
    EXPECT_EQ(stats.layers_transferred, 0);
    EXPECT_EQ(stats.layers_evicted, 0);
    EXPECT_EQ(stats.cache_hits, 0);
    EXPECT_EQ(stats.cache_misses, 0);
    EXPECT_EQ(stats.prefetch_hits, 0);
    EXPECT_EQ(stats.prefetch_wasted, 0);
    EXPECT_EQ(stats.bytes_transferred, 0);
    EXPECT_EQ(stats.bytes_evicted, 0);
    EXPECT_EQ(stats.total_transfer_time.count(), 0);
    EXPECT_EQ(stats.total_wait_time.count(), 0);
    EXPECT_EQ(stats.max_transfer_latency.count(), 0);
    EXPECT_EQ(stats.current_cache_size, 0);
    EXPECT_EQ(stats.peak_cache_size, 0);
    EXPECT_EQ(stats.current_layers_cached, 0);
}

TEST_F(LayerWeightStreamerTest, HitRateCalculation)
{
    StreamingStats stats;
    stats.cache_hits = 0;
    stats.cache_misses = 0;
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.0); // 0/0 = 0

    stats.cache_hits = 5;
    stats.cache_misses = 5;
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.5); // 5/10 = 50%

    stats.cache_hits = 10;
    stats.cache_misses = 0;
    EXPECT_DOUBLE_EQ(stats.hitRate(), 1.0); // 10/10 = 100%

    stats.cache_hits = 0;
    stats.cache_misses = 10;
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.0); // 0/10 = 0%
}

TEST_F(LayerWeightStreamerTest, BandwidthCalculation)
{
    StreamingStats stats;

    // No transfers = 0 bandwidth
    stats.bytes_transferred = 0;
    stats.total_transfer_time = std::chrono::nanoseconds(0);
    EXPECT_DOUBLE_EQ(stats.averageBandwidthGBps(), 0.0);

    // 1 GB in 1 second = 1 GB/s
    stats.bytes_transferred = 1024ULL * 1024 * 1024; // 1 GB
    stats.total_transfer_time = std::chrono::nanoseconds(1000000000); // 1 second
    EXPECT_NEAR(stats.averageBandwidthGBps(), 1.0, 0.001);

    // 10 GB in 0.5 seconds = 20 GB/s
    stats.bytes_transferred = 10ULL * 1024 * 1024 * 1024; // 10 GB
    stats.total_transfer_time = std::chrono::nanoseconds(500000000); // 0.5 seconds
    EXPECT_NEAR(stats.averageBandwidthGBps(), 20.0, 0.001);
}

TEST_F(LayerWeightStreamerTest, PrefetchEffectivenessCalculation)
{
    StreamingStats stats;

    // No prefetches = 0 effectiveness
    stats.prefetch_hits = 0;
    stats.prefetch_wasted = 0;
    EXPECT_DOUBLE_EQ(stats.prefetchEffectiveness(), 0.0);

    // 50% hit rate
    stats.prefetch_hits = 5;
    stats.prefetch_wasted = 5;
    EXPECT_DOUBLE_EQ(stats.prefetchEffectiveness(), 0.5);

    // 100% hit rate
    stats.prefetch_hits = 10;
    stats.prefetch_wasted = 0;
    EXPECT_DOUBLE_EQ(stats.prefetchEffectiveness(), 1.0);

    // 0% hit rate
    stats.prefetch_hits = 0;
    stats.prefetch_wasted = 10;
    EXPECT_DOUBLE_EQ(stats.prefetchEffectiveness(), 0.0);
}

// =============================================================================
// StreamingConfig Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, DefaultConfigValues)
{
    StreamingConfig default_config;

    // Memory defaults
    EXPECT_EQ(default_config.gpu_memory_budget, 0); // Auto-detect
    EXPECT_EQ(default_config.cache_layer_count, 2);
    EXPECT_EQ(default_config.min_free_memory, 512 * 1024 * 1024); // 512 MB

    // Prefetching defaults
    EXPECT_TRUE(default_config.enable_prefetch);
    EXPECT_EQ(default_config.prefetch_depth, 1);
    EXPECT_TRUE(default_config.use_prefetch_stream);

    // Eviction defaults
    EXPECT_EQ(default_config.eviction_policy, StreamingEvictionPolicy::LRU);

    // Transfer defaults
    EXPECT_TRUE(default_config.use_pinned_memory);
    EXPECT_TRUE(default_config.enable_compute_transfer_overlap);

    // Phase defaults
    EXPECT_TRUE(default_config.prefill_preload_enabled);
    EXPECT_TRUE(default_config.decode_aggressive_prefetch);

    // Debug defaults
    EXPECT_FALSE(default_config.log_transfer_stats);
    EXPECT_FALSE(default_config.verify_transfers);
}

// =============================================================================
// Eviction Policy Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, EvictionPolicyEnum)
{
    // Verify enum values exist
    EXPECT_NE(StreamingEvictionPolicy::LRU, StreamingEvictionPolicy::FIFO);
    EXPECT_NE(StreamingEvictionPolicy::LRU, StreamingEvictionPolicy::NONE);
    EXPECT_NE(StreamingEvictionPolicy::FIFO, StreamingEvictionPolicy::NONE);
}

// =============================================================================
// WeightResidencyMode Tests
// =============================================================================

TEST_F(LayerWeightStreamerTest, ResidencyModeEnum)
{
    // Verify enum values exist
    EXPECT_NE(WeightResidencyMode::RESIDENT, WeightResidencyMode::STREAMING);
    EXPECT_NE(WeightResidencyMode::RESIDENT, WeightResidencyMode::UNIFIED);
    EXPECT_NE(WeightResidencyMode::STREAMING, WeightResidencyMode::UNIFIED);
}

// =============================================================================
// Layer Name Generation Tests (internal method behavior)
// =============================================================================

TEST_F(LayerWeightStreamerTest, QwenStyleWeightNaming)
{
    // This tests the expected weight naming convention
    // The actual method is private, so we just verify the pattern

    std::string prefix_0 = "blk.0.";
    std::string prefix_5 = "blk.5.";
    std::string prefix_23 = "blk.23.";

    // Verify expected weight names
    EXPECT_EQ(prefix_0 + "attn_q.weight", "blk.0.attn_q.weight");
    EXPECT_EQ(prefix_5 + "ffn_gate.weight", "blk.5.ffn_gate.weight");
    EXPECT_EQ(prefix_23 + "attn_norm.weight", "blk.23.attn_norm.weight");

    // Full list of weights per layer
    std::vector<std::string> weight_suffixes = {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ffn_norm.weight"};

    EXPECT_EQ(weight_suffixes.size(), 9); // 9 weights per layer
}
