/**
 * @file Test__WeightStreaming.cpp
 * @brief Integration tests for weight streaming with layer prefetch and eviction
 *
 * **Purpose**: Validates weight streaming functionality including:
 * - Layer prefetching ahead of compute
 * - Eviction when memory budget exceeded
 * - LRU and FIFO eviction policies
 * - Streaming disabled by default behavior
 * - Concurrent prefetch with computation
 * - Statistics tracking accuracy
 *
 * **Test Strategy**:
 * Uses MockWeightStreamer to simulate GPU memory constraints and verify
 * streaming behavior without requiring actual GPU memory management.
 *
 * **Dependencies**:
 * - MockWeightStreamer: Simulates weight streaming with callbacks
 * - MockModelLoader: Provides model metadata (layer count, etc.)
 * - MockMPITopology: Simulates device configuration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <atomic>

#include "mocks/MockWeightStreamer.h"
#include "mocks/MockModelLoader.h"
#include "mocks/MockMPITopology.h"
#include "backends/DeviceId.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief Test fixture for weight streaming tests
 *
 * Provides helper methods for creating streamers and simulating
 * layer execution patterns.
 */
class Test__WeightStreaming : public ::testing::Test {
protected:
    void SetUp() override {
        // Default: 24 layers, cache holds 3 layers
        default_num_layers_ = 24;
        default_cache_capacity_ = 3;
        layer_size_ = 100 * 1024 * 1024;  // 100MB per layer
    }

    /**
     * @brief Create a streaming-enabled mock
     */
    std::shared_ptr<MockWeightStreamer> createStreamingMock(
        int num_layers = -1,
        int cache_capacity = -1) {
        
        if (num_layers < 0) num_layers = default_num_layers_;
        if (cache_capacity < 0) cache_capacity = default_cache_capacity_;
        
        return MockWeightStreamerBuilder()
            .setStreamingEnabled(true)
            .setNumLayers(num_layers)
            .setLayerSize(layer_size_)
            .setMemoryBudget(cache_capacity * layer_size_)
            .setEvictionPolicy(StreamingEvictionPolicy::LRU)
            .build();
    }

    /**
     * @brief Create a mock with streaming disabled (resident mode)
     */
    std::shared_ptr<MockWeightStreamer> createResidentMock() {
        return MockWeightStreamer::createResident();
    }

    /**
     * @brief Simulate forward pass through layers with prefetch
     * @param streamer The weight streamer
     * @param prefetch_depth How many layers to prefetch ahead
     */
    void simulateForwardPass(
        std::shared_ptr<MockWeightStreamer> streamer,
        int prefetch_depth = 1) {
        
        DeviceId gpu = DeviceId::cuda(0);
        int num_layers = streamer->numLayers();
        
        for (int layer = 0; layer < num_layers; ++layer) {
            // Ensure current layer is on device
            ASSERT_TRUE(streamer->ensureLayerOnDevice(layer, gpu))
                << "Failed to load layer " << layer;
            
            // Prefetch upcoming layers
            for (int p = 1; p <= prefetch_depth && (layer + p) < num_layers; ++p) {
                streamer->prefetchLayer(layer + p, gpu);
            }
            
            // Simulate completing prefetches (synchronous simulation)
            for (int p = 1; p <= prefetch_depth && (layer + p) < num_layers; ++p) {
                streamer->completePrefetch(layer + p);
            }
            
            // Release current layer after use (hint for eviction)
            streamer->releaseLayer(layer);
        }
    }

    int default_num_layers_;
    int default_cache_capacity_;
    size_t layer_size_;
};

// ============================================================================
// Basic Streaming Tests
// ============================================================================

/**
 * @test Verify streaming is disabled by default and all layers appear cached
 */
TEST_F(Test__WeightStreaming, StreamingDisabledByDefault) {
    auto streamer = createResidentMock();
    DeviceId gpu = DeviceId::cuda(0);
    
    // All layers should appear cached when streaming disabled
    for (int i = 0; i < 24; ++i) {
        EXPECT_TRUE(streamer->isLayerCached(i, gpu))
            << "Layer " << i << " should be cached when streaming disabled";
    }
    
    // ensureLayerOnDevice should always succeed
    EXPECT_TRUE(streamer->ensureLayerOnDevice(0, gpu));
    EXPECT_TRUE(streamer->ensureLayerOnDevice(10, gpu));
    EXPECT_TRUE(streamer->ensureLayerOnDevice(23, gpu));
    
    // No actual transfers should occur
    auto stats = streamer->stats();
    EXPECT_EQ(stats.layers_transferred, 0);
    EXPECT_EQ(stats.bytes_transferred, 0);
}

/**
 * @test Verify layer prefetch and eviction during forward pass
 */
TEST_F(Test__WeightStreaming, LayerPrefetchAndEviction) {
    auto streamer = createStreamingMock(8, 3);  // 8 layers, cache holds 3
    
    std::vector<int> evicted_layers;
    std::vector<int> prefetched_layers;
    std::vector<int> loaded_layers;
    
    streamer->set_evict_callback([&](int layer) {
        evicted_layers.push_back(layer);
    });
    streamer->set_prefetch_callback([&](int layer) {
        prefetched_layers.push_back(layer);
    });
    streamer->set_load_callback([&](int layer) {
        loaded_layers.push_back(layer);
    });
    
    simulateForwardPass(streamer, 1);
    
    // Load callback is only called for cache misses.
    // With prefetch depth 1: layer 0 is a miss, layers 1-7 are prefetch hits
    // So only layer 0 triggers the load callback
    EXPECT_EQ(loaded_layers.size(), 1) << "Only layer 0 should be a cache miss";
    EXPECT_EQ(loaded_layers[0], 0);
    
    // All layers transferred: 1 via ensureLayerOnDevice, 7 via completePrefetch
    auto stats = streamer->stats();
    EXPECT_EQ(stats.layers_transferred, 8);
    
    // With 3-layer cache and 8 layers, we need to evict at least 5 times
    // (first 3 loads fit, then need to evict for each subsequent)
    EXPECT_GE(evicted_layers.size(), 5);
    
    // Layers 1-7 should be prefetched
    EXPECT_EQ(prefetched_layers.size(), 7);
    
    // Verify load order is sequential
    for (size_t i = 0; i < loaded_layers.size(); ++i) {
        EXPECT_EQ(loaded_layers[i], static_cast<int>(i))
            << "Load order should be sequential";
    }
}

/**
 * @test Verify prefetch depth configuration affects behavior
 */
TEST_F(Test__WeightStreaming, PrefetchDepthConfiguration) {
    // Test with different prefetch depths
    for (size_t prefetch_depth : {1, 2, 3}) {
        auto streamer = createStreamingMock(12, 4);  // 12 layers, cache 4
        streamer->setPrefetchDepth(prefetch_depth);
        
        std::vector<int> prefetched;
        streamer->set_prefetch_callback([&](int layer) {
            prefetched.push_back(layer);
        });
        
        streamer->clearHistory();
        prefetched.clear();
        
        simulateForwardPass(streamer, static_cast<int>(prefetch_depth));
        
        // With prefetch depth N, we should prefetch layers 1..N for layer 0,
        // 2..N+1 for layer 1, etc. Total prefetches depends on depth.
        // Each layer (except last N) triggers N prefetches, but duplicates removed
        
        // At minimum, layers 1 through 11 should be prefetched at some point
        std::unordered_set<int> unique_prefetched(prefetched.begin(), prefetched.end());
        EXPECT_GE(unique_prefetched.size(), std::min(11UL, prefetch_depth * 10))
            << "Prefetch depth " << prefetch_depth << " should trigger more prefetches";
    }
}

/**
 * @test Verify memory budget is respected
 */
TEST_F(Test__WeightStreaming, MemoryBudgetRespected) {
    auto streamer = createStreamingMock(10, 3);  // 10 layers, cache 3
    DeviceId gpu = DeviceId::cuda(0);
    
    size_t budget = streamer->memoryBudget();
    
    // Load layers sequentially
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(streamer->ensureLayerOnDevice(i, gpu));
        
        // Memory usage should never exceed budget
        EXPECT_LE(streamer->currentDeviceMemoryUsage(), budget)
            << "Memory usage exceeded budget after loading layer " << i;
        
        // Complete any pending prefetches to get accurate memory count
        streamer->synchronize();
        EXPECT_LE(streamer->currentDeviceMemoryUsage(), budget)
            << "Memory exceeded after synchronize for layer " << i;
    }
    
    // Should have exactly 3 layers cached (the budget limit)
    EXPECT_EQ(streamer->cachedLayerCount(), 3);
}

/**
 * @test Verify LRU eviction policy works correctly
 */
TEST_F(Test__WeightStreaming, LRUEvictionPolicy) {
    auto streamer = createStreamingMock(6, 3);  // 6 layers, cache 3
    streamer->setEvictionPolicy(StreamingEvictionPolicy::LRU);
    DeviceId gpu = DeviceId::cuda(0);
    
    std::vector<int> evicted;
    streamer->set_evict_callback([&](int layer) {
        evicted.push_back(layer);
    });
    
    // Load layers 0, 1, 2 (fills cache)
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    EXPECT_TRUE(evicted.empty()) << "No eviction needed yet";
    
    // Touch layer 0 again (makes it MRU)
    streamer->ensureLayerOnDevice(0, gpu);
    
    // Load layer 3 - should evict layer 1 (LRU)
    streamer->ensureLayerOnDevice(3, gpu);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0], 1) << "Layer 1 should be evicted (LRU)";
    
    // Load layer 4 - should evict layer 2 (now LRU)
    streamer->ensureLayerOnDevice(4, gpu);
    ASSERT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[1], 2) << "Layer 2 should be evicted next";
    
    // Load layer 5 - should evict layer 0 (now LRU)
    streamer->ensureLayerOnDevice(5, gpu);
    ASSERT_EQ(evicted.size(), 3);
    EXPECT_EQ(evicted[2], 0) << "Layer 0 should be evicted now";
}

/**
 * @test Verify FIFO eviction policy works correctly
 */
TEST_F(Test__WeightStreaming, FIFOEvictionPolicy) {
    auto streamer = createStreamingMock(6, 3);
    streamer->setEvictionPolicy(StreamingEvictionPolicy::FIFO);
    DeviceId gpu = DeviceId::cuda(0);
    
    std::vector<int> evicted;
    streamer->set_evict_callback([&](int layer) {
        evicted.push_back(layer);
    });
    
    // Load layers 0, 1, 2 (fills cache)
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    EXPECT_TRUE(evicted.empty());
    
    // Touch layer 0 again - FIFO ignores this (no reordering)
    streamer->ensureLayerOnDevice(0, gpu);
    
    // Load layer 3 - with FIFO, should evict layer 0 (first loaded)
    streamer->ensureLayerOnDevice(3, gpu);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0], 0) << "Layer 0 should be evicted (FIFO - first loaded)";
    
    // Load layer 4 - should evict layer 1 (next in FIFO order)
    streamer->ensureLayerOnDevice(4, gpu);
    ASSERT_EQ(evicted.size(), 2);
    EXPECT_EQ(evicted[1], 1) << "Layer 1 should be evicted next";
}

/**
 * @test Verify cache hits and misses are tracked correctly
 */
TEST_F(Test__WeightStreaming, CacheStatisticsTracking) {
    auto streamer = createStreamingMock(6, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    streamer->resetStats();
    
    // Load 3 layers (all misses)
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    
    auto stats1 = streamer->stats();
    EXPECT_EQ(stats1.cache_misses, 3) << "3 cache misses for initial loads";
    EXPECT_EQ(stats1.cache_hits, 0) << "No hits yet";
    EXPECT_EQ(stats1.layers_transferred, 3);
    
    // Access same layers again (all hits)
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    
    auto stats2 = streamer->stats();
    EXPECT_EQ(stats2.cache_hits, 2) << "2 cache hits";
    EXPECT_EQ(stats2.cache_misses, 3) << "Still 3 misses";
    
    // Hit rate should be 2/5 = 0.4
    EXPECT_NEAR(stats2.hitRate(), 2.0 / 5.0, 0.001);
}

/**
 * @test Verify prefetch hits are tracked
 */
TEST_F(Test__WeightStreaming, PrefetchHitsTracking) {
    auto streamer = createStreamingMock(6, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    streamer->resetStats();
    
    // Load layer 0
    streamer->ensureLayerOnDevice(0, gpu);
    
    // Prefetch layer 1 and complete it
    streamer->prefetchLayer(1, gpu);
    // Layer 1 is now in "prefetching" state
    
    // Now access layer 1 while it's still "prefetching"
    // This should count as a prefetch hit (layer was found in prefetch queue)
    streamer->ensureLayerOnDevice(1, gpu);
    
    auto stats = streamer->stats();
    EXPECT_EQ(stats.prefetch_hits, 1) << "Should have 1 prefetch hit";
}

/**
 * @test Verify concurrent prefetch with compute simulation
 */
TEST_F(Test__WeightStreaming, ConcurrentPrefetchAndCompute) {
    auto streamer = createStreamingMock(8, 4);  // 8 layers, cache 4
    DeviceId gpu = DeviceId::cuda(0);
    
    std::atomic<int> prefetch_count{0};
    int transfer_count = 0;
    
    streamer->set_prefetch_callback([&](int /*layer*/) {
        prefetch_count++;
    });
    
    // Simulate: load layer 0, prefetch 1, complete prefetch before next load, etc.
    for (int i = 0; i < 8; ++i) {
        // Load current layer
        ASSERT_TRUE(streamer->ensureLayerOnDevice(i, gpu));
        
        // Start prefetch of next layer (async)
        if (i + 1 < 8) {
            streamer->prefetchLayer(i + 1, gpu);
        }
        
        // "Compute" happens here - prefetch runs concurrently
        
        // Complete prefetch before next iteration
        if (i + 1 < 8) {
            streamer->completePrefetch(i + 1);
        }
    }
    
    // All layers should be transferred
    auto stats = streamer->stats();
    EXPECT_EQ(stats.layers_transferred, 8) << "All 8 layers should be transferred";
    EXPECT_GE(prefetch_count.load(), 7) << "At least 7 prefetches initiated";
}

/**
 * @test Verify layer release hints affect eviction order
 */
TEST_F(Test__WeightStreaming, ReleaseLayerAffectsEviction) {
    auto streamer = createStreamingMock(5, 3);
    streamer->setEvictionPolicy(StreamingEvictionPolicy::LRU);
    DeviceId gpu = DeviceId::cuda(0);
    
    std::vector<int> evicted;
    streamer->set_evict_callback([&](int layer) {
        evicted.push_back(layer);
    });
    
    // Load layers 0, 1, 2
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    
    // Release layer 2 (marks as eviction candidate)
    streamer->releaseLayer(2);
    
    // Touch layer 0 and 1 to make them MRU
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    
    // Load layer 3 - layer 2 should be evicted (it was released)
    streamer->ensureLayerOnDevice(3, gpu);
    ASSERT_EQ(evicted.size(), 1);
    EXPECT_EQ(evicted[0], 2) << "Released layer 2 should be evicted first";
}

/**
 * @test Verify clearCache evicts all layers
 */
TEST_F(Test__WeightStreaming, ClearCacheEvictsAll) {
    auto streamer = createStreamingMock(6, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Load some layers
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    
    EXPECT_EQ(streamer->cachedLayerCount(), 3);
    
    std::vector<int> evicted;
    streamer->set_evict_callback([&](int layer) {
        evicted.push_back(layer);
    });
    
    // Clear cache
    streamer->clearCache();
    
    EXPECT_EQ(streamer->cachedLayerCount(), 0);
    EXPECT_EQ(evicted.size(), 3) << "All 3 layers should be evicted";
    EXPECT_EQ(streamer->currentDeviceMemoryUsage(), 0);
}

/**
 * @test Verify phase transitions are handled
 */
TEST_F(Test__WeightStreaming, PhaseTransitionHandled) {
    auto streamer = createStreamingMock();
    
    // Transition from PREFILL to DECODE
    streamer->onPhaseTransition(InferencePhase::PREFILL, InferencePhase::DECODE);
    
    // Transition back
    streamer->onPhaseTransition(InferencePhase::DECODE, InferencePhase::PREFILL);
    
    // No crash, no error - phase management is a hook for future optimization
    SUCCEED() << "Phase transitions handled without error";
}

/**
 * @test Verify statistics are reset correctly
 */
TEST_F(Test__WeightStreaming, StatsReset) {
    auto streamer = createStreamingMock(4, 2);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Do some operations
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);  // Triggers eviction
    
    auto stats1 = streamer->stats();
    EXPECT_GT(stats1.layers_transferred, 0);
    EXPECT_GT(stats1.layers_evicted, 0);
    
    // Reset stats
    streamer->resetStats();
    
    auto stats2 = streamer->stats();
    EXPECT_EQ(stats2.layers_transferred, 0);
    EXPECT_EQ(stats2.layers_evicted, 0);
    EXPECT_EQ(stats2.cache_hits, 0);
    EXPECT_EQ(stats2.cache_misses, 0);
    
    // But cache state should remain
    EXPECT_EQ(stats2.current_layers_cached, 2);
}

/**
 * @test Verify invalid layer indices are handled gracefully
 */
TEST_F(Test__WeightStreaming, InvalidLayerHandling) {
    auto streamer = createStreamingMock(10, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Negative layer index
    EXPECT_FALSE(streamer->ensureLayerOnDevice(-1, gpu));
    EXPECT_FALSE(streamer->isLayerCached(-1, gpu));
    
    // Layer beyond model size
    EXPECT_FALSE(streamer->ensureLayerOnDevice(100, gpu));
    EXPECT_FALSE(streamer->isLayerCached(100, gpu));
    
    // Prefetch invalid layer should be no-op
    streamer->prefetchLayer(-1, gpu);
    streamer->prefetchLayer(100, gpu);
    EXPECT_TRUE(streamer->prefetchingLayers().empty());
}

/**
 * @test Verify synchronize completes all pending prefetches
 */
TEST_F(Test__WeightStreaming, SynchronizeCompletesPrefetches) {
    auto streamer = createStreamingMock(6, 4);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Start multiple prefetches
    streamer->prefetchLayer(0, gpu);
    streamer->prefetchLayer(1, gpu);
    streamer->prefetchLayer(2, gpu);
    
    EXPECT_EQ(streamer->prefetchingLayers().size(), 3);
    
    // Synchronize should complete all
    streamer->synchronize();
    
    EXPECT_TRUE(streamer->prefetchingLayers().empty());
    EXPECT_TRUE(streamer->isLayerCached(0, gpu));
    EXPECT_TRUE(streamer->isLayerCached(1, gpu));
    EXPECT_TRUE(streamer->isLayerCached(2, gpu));
}

/**
 * @test Verify eviction with NONE policy fails gracefully
 */
TEST_F(Test__WeightStreaming, NoneEvictionPolicyFailsGracefully) {
    auto streamer = createStreamingMock(6, 2);
    streamer->setEvictionPolicy(StreamingEvictionPolicy::NONE);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Load 2 layers (fills cache)
    EXPECT_TRUE(streamer->ensureLayerOnDevice(0, gpu));
    EXPECT_TRUE(streamer->ensureLayerOnDevice(1, gpu));
    
    // Third layer should fail (no eviction allowed)
    EXPECT_FALSE(streamer->ensureLayerOnDevice(2, gpu));
    
    // Original layers still cached
    EXPECT_TRUE(streamer->isLayerCached(0, gpu));
    EXPECT_TRUE(streamer->isLayerCached(1, gpu));
    EXPECT_FALSE(streamer->isLayerCached(2, gpu));
}

/**
 * @test Verify peak memory tracking
 */
TEST_F(Test__WeightStreaming, PeakMemoryTracking) {
    auto streamer = createStreamingMock(10, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    size_t layer_size = streamer->layerSize();
    
    // Load 3 layers
    streamer->ensureLayerOnDevice(0, gpu);
    streamer->ensureLayerOnDevice(1, gpu);
    streamer->ensureLayerOnDevice(2, gpu);
    
    auto stats1 = streamer->stats();
    EXPECT_EQ(stats1.peak_cache_size, 3 * layer_size);
    
    // Clear cache
    streamer->clearCache();
    
    // Peak should still show 3 layers
    auto stats2 = streamer->stats();
    EXPECT_EQ(stats2.peak_cache_size, 3 * layer_size);
    EXPECT_EQ(stats2.current_cache_size, 0);
}

/**
 * @test Verify duplicate prefetch is ignored
 */
TEST_F(Test__WeightStreaming, DuplicatePrefetchIgnored) {
    auto streamer = createStreamingMock(6, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    int prefetch_count = 0;
    streamer->set_prefetch_callback([&](int /*layer*/) {
        prefetch_count++;
    });
    
    // Prefetch same layer multiple times
    streamer->prefetchLayer(0, gpu);
    streamer->prefetchLayer(0, gpu);
    streamer->prefetchLayer(0, gpu);
    
    // Should only trigger once
    EXPECT_EQ(prefetch_count, 1);
    EXPECT_EQ(streamer->prefetchingLayers().size(), 1);
}

/**
 * @test Verify prefetch on cached layer is no-op
 */
TEST_F(Test__WeightStreaming, PrefetchCachedLayerNoop) {
    auto streamer = createStreamingMock(6, 3);
    DeviceId gpu = DeviceId::cuda(0);
    
    // Load layer 0
    streamer->ensureLayerOnDevice(0, gpu);
    
    int prefetch_count = 0;
    streamer->set_prefetch_callback([&](int /*layer*/) {
        prefetch_count++;
    });
    
    // Try to prefetch already-cached layer
    streamer->prefetchLayer(0, gpu);
    
    EXPECT_EQ(prefetch_count, 0);
    EXPECT_TRUE(streamer->prefetchingLayers().empty());
}

// ============================================================================
// Integration with Model Metadata
// ============================================================================

/**
 * @test Verify streaming configuration from model metadata
 */
TEST_F(Test__WeightStreaming, ConfigurationFromModelMetadata) {
    // Create mock model loader with specific layer count
    auto model = MockModelLoader::createQwen2_05B();  // 24 layers
    
    // Create streamer based on model
    int num_layers = static_cast<int>(model->blockCount());
    auto streamer = createStreamingMock(num_layers, 4);  // Cache 4 layers
    
    EXPECT_EQ(streamer->numLayers(), 24);
    
    DeviceId gpu = DeviceId::cuda(0);
    
    // Simulate full forward pass
    for (int i = 0; i < num_layers; ++i) {
        ASSERT_TRUE(streamer->ensureLayerOnDevice(i, gpu))
            << "Failed at layer " << i;
    }
    
    auto stats = streamer->stats();
    EXPECT_EQ(stats.layers_transferred, static_cast<size_t>(num_layers));
    EXPECT_GE(stats.layers_evicted, static_cast<size_t>(num_layers - 4));
}

