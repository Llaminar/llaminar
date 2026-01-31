/**
 * @file Test__NullWeightStreamer.cpp
 * @brief Unit tests for NullWeightStreamer (Option B - RESIDENT mode)
 *
 * Tests the no-op weight streamer implementation used when weights fit in GPU VRAM.
 * Verifies that all methods return expected values and have no side effects.
 */

#include "loaders/NullWeightStreamer.h"
#include "backends/DeviceId.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class NullWeightStreamerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        streamer_ = std::make_unique<NullWeightStreamer>();
        streamer_with_budget_ = std::make_unique<NullWeightStreamer>(1024 * 1024 * 1024); // 1GB
    }

    std::unique_ptr<NullWeightStreamer> streamer_;
    std::unique_ptr<NullWeightStreamer> streamer_with_budget_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, DefaultConstructionZeroBudget)
{
    NullWeightStreamer streamer;
    EXPECT_EQ(streamer.memoryBudget(), 0);
}

TEST_F(NullWeightStreamerTest, ConstructionWithBudget)
{
    const size_t budget = 4ULL * 1024 * 1024 * 1024; // 4GB
    NullWeightStreamer streamer(budget);
    EXPECT_EQ(streamer.memoryBudget(), budget);
}

// =============================================================================
// Layer Management Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, EnsureLayerOnDeviceAlwaysReturnsTrue)
{
    // Test with various layer indices and devices
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(0, DeviceId::cpu()));
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(0, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(31, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(100, DeviceId::cuda(1)));
    
    // Negative layer index (edge case)
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(-1, DeviceId::cpu()));
}

TEST_F(NullWeightStreamerTest, PrefetchLayerIsNoOp)
{
    // Should not throw or have any observable effect
    EXPECT_NO_THROW(streamer_->prefetchLayer(0, DeviceId::cuda(0)));
    EXPECT_NO_THROW(streamer_->prefetchLayer(10, DeviceId::cuda(1)));
    EXPECT_NO_THROW(streamer_->prefetchLayer(-1, DeviceId::cpu()));
    
    // Verify no state change
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
}

TEST_F(NullWeightStreamerTest, ReleaseLayerIsNoOp)
{
    // Should not throw or have any observable effect
    EXPECT_NO_THROW(streamer_->releaseLayer(0));
    EXPECT_NO_THROW(streamer_->releaseLayer(31));
    EXPECT_NO_THROW(streamer_->releaseLayer(-1));
    
    // Verify layers are still "cached" after release
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cuda(0)));
}

// =============================================================================
// Phase Management Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, OnPhaseTransitionIsNoOp)
{
    // Should not throw or have any observable effect
    EXPECT_NO_THROW(streamer_->onPhaseTransition(InferencePhase::PREFILL, InferencePhase::DECODE));
    EXPECT_NO_THROW(streamer_->onPhaseTransition(InferencePhase::DECODE, InferencePhase::PREFILL));
    EXPECT_NO_THROW(streamer_->onPhaseTransition(InferencePhase::DECODE, InferencePhase::DECODE));
    
    // Verify no state change
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, CurrentDeviceMemoryUsageAlwaysZero)
{
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
    EXPECT_EQ(streamer_with_budget_->currentDeviceMemoryUsage(), 0);
    
    // Even after "loading" layers
    streamer_->ensureLayerOnDevice(0, DeviceId::cuda(0));
    streamer_->ensureLayerOnDevice(1, DeviceId::cuda(0));
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
}

TEST_F(NullWeightStreamerTest, MemoryBudgetReturnsConfiguredValue)
{
    EXPECT_EQ(streamer_->memoryBudget(), 0);
    EXPECT_EQ(streamer_with_budget_->memoryBudget(), 1024 * 1024 * 1024);
}

TEST_F(NullWeightStreamerTest, EvictLayerAlwaysReturnsFalse)
{
    // Nothing to evict when weights are resident
    EXPECT_FALSE(streamer_->evictLayer(0));
    EXPECT_FALSE(streamer_->evictLayer(31));
    EXPECT_FALSE(streamer_->evictLayer(-1));
    
    // Verify layer is still "cached" after eviction attempt
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cuda(0)));
}

TEST_F(NullWeightStreamerTest, ClearCacheIsNoOp)
{
    EXPECT_NO_THROW(streamer_->clearCache());
    
    // Verify no state change - layers still "cached"
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, SynchronizeIsNoOp)
{
    EXPECT_NO_THROW(streamer_->synchronize());
    
    // Should return immediately, no blocking
    auto start = std::chrono::steady_clock::now();
    streamer_->synchronize();
    auto elapsed = std::chrono::steady_clock::now() - start;
    
    // Should be essentially instant (< 1ms)
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1);
}

// =============================================================================
// Diagnostics Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, IsLayerCachedAlwaysReturnsTrue)
{
    // All layers are "cached" because they're resident
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cpu()));
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer_->isLayerCached(31, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer_->isLayerCached(100, DeviceId::cuda(1)));
    EXPECT_TRUE(streamer_->isLayerCached(-1, DeviceId::cpu()));
}

TEST_F(NullWeightStreamerTest, IsPrefetchInProgressAlwaysReturnsFalse)
{
    // No prefetch operations when weights are resident
    EXPECT_FALSE(streamer_->isPrefetchInProgress(0));
    EXPECT_FALSE(streamer_->isPrefetchInProgress(31));
    EXPECT_FALSE(streamer_->isPrefetchInProgress(-1));
    
    // Even after calling prefetchLayer
    streamer_->prefetchLayer(0, DeviceId::cuda(0));
    EXPECT_FALSE(streamer_->isPrefetchInProgress(0));
}

TEST_F(NullWeightStreamerTest, StatsReturnsEmptyStatistics)
{
    StreamingStats stats = streamer_->stats();
    
    // All counters should be zero
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
    
    // Derived metrics
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.0);
    EXPECT_DOUBLE_EQ(stats.averageBandwidthGBps(), 0.0);
    EXPECT_DOUBLE_EQ(stats.prefetchEffectiveness(), 0.0);
}

TEST_F(NullWeightStreamerTest, ResetStatsIsNoOp)
{
    // Should not throw
    EXPECT_NO_THROW(streamer_->resetStats());
    
    // Stats should still be empty
    StreamingStats stats = streamer_->stats();
    EXPECT_EQ(stats.layers_transferred, 0);
}

// =============================================================================
// Polymorphism Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, WorksThroughIWeightStreamerInterface)
{
    // Use through base interface pointer
    std::unique_ptr<IWeightStreamer> base_ptr = std::make_unique<NullWeightStreamer>(1024);
    
    // All methods should work correctly through the interface
    EXPECT_TRUE(base_ptr->ensureLayerOnDevice(0, DeviceId::cuda(0)));
    EXPECT_NO_THROW(base_ptr->prefetchLayer(1, DeviceId::cuda(0)));
    EXPECT_NO_THROW(base_ptr->releaseLayer(0));
    EXPECT_NO_THROW(base_ptr->onPhaseTransition(InferencePhase::PREFILL, InferencePhase::DECODE));
    EXPECT_EQ(base_ptr->currentDeviceMemoryUsage(), 0);
    EXPECT_EQ(base_ptr->memoryBudget(), 1024);
    EXPECT_FALSE(base_ptr->evictLayer(0));
    EXPECT_NO_THROW(base_ptr->clearCache());
    EXPECT_NO_THROW(base_ptr->synchronize());
    EXPECT_TRUE(base_ptr->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_FALSE(base_ptr->isPrefetchInProgress(0));
    
    StreamingStats stats = base_ptr->stats();
    EXPECT_EQ(stats.layers_transferred, 0);
    
    EXPECT_NO_THROW(base_ptr->resetStats());
}

// =============================================================================
// Idempotency Tests
// =============================================================================

TEST_F(NullWeightStreamerTest, MultipleCallsAreIdempotent)
{
    // Multiple calls to the same method should have no cumulative effect
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_TRUE(streamer_->ensureLayerOnDevice(0, DeviceId::cuda(0)));
    }
    
    for (int i = 0; i < 100; ++i)
    {
        streamer_->prefetchLayer(0, DeviceId::cuda(0));
    }
    
    for (int i = 0; i < 100; ++i)
    {
        streamer_->releaseLayer(0);
    }
    
    // State should be unchanged
    EXPECT_EQ(streamer_->currentDeviceMemoryUsage(), 0);
    EXPECT_TRUE(streamer_->isLayerCached(0, DeviceId::cuda(0)));
    
    StreamingStats stats = streamer_->stats();
    EXPECT_EQ(stats.layers_transferred, 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(NullWeightStreamerTest, HandlesMaxLayerIndex)
{
    const int max_layer = std::numeric_limits<int>::max();
    
    EXPECT_TRUE(streamer_->ensureLayerOnDevice(max_layer, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer_->isLayerCached(max_layer, DeviceId::cuda(0)));
    EXPECT_FALSE(streamer_->isPrefetchInProgress(max_layer));
    EXPECT_FALSE(streamer_->evictLayer(max_layer));
}

TEST_F(NullWeightStreamerTest, HandlesMaxMemoryBudget)
{
    const size_t max_budget = std::numeric_limits<size_t>::max();
    NullWeightStreamer streamer(max_budget);
    
    EXPECT_EQ(streamer.memoryBudget(), max_budget);
    EXPECT_EQ(streamer.currentDeviceMemoryUsage(), 0); // Still no actual usage
}
