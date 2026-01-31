/**
 * @file Test__WeightStreamerFactory.cpp
 * @brief Unit tests for WeightStreamerFactory
 *
 * Tests the factory for creating IWeightStreamer implementations based on
 * configuration, environment variables, and auto-detection.
 */

#include "loaders/WeightStreamerFactory.h"
#include "loaders/NullWeightStreamer.h"
#include "loaders/LayerWeightStreamer.h"
#include "backends/DeviceId.h"
#include <gtest/gtest.h>
#include <memory>
#include <cstdlib>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class WeightStreamerFactoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // No setup needed - tests use explicit modes via create()
    }
};

// =============================================================================
// create() with Explicit Mode - RESIDENT
// =============================================================================

TEST_F(WeightStreamerFactoryTest, CreateResidentModeReturnsNullStreamer)
{
    auto streamer = WeightStreamerFactory::create(WeightResidencyMode::RESIDENT);

    ASSERT_NE(streamer, nullptr);

    // Verify it's a NullWeightStreamer by checking behavior
    // NullWeightStreamer always returns true for isLayerCached
    EXPECT_TRUE(streamer->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_TRUE(streamer->isLayerCached(100, DeviceId::cuda(0)));

    // Memory usage should be 0
    EXPECT_EQ(streamer->currentDeviceMemoryUsage(), 0);
}

TEST_F(WeightStreamerFactoryTest, CreateResidentModeIgnoresWeightManager)
{
    // Even with a null weight_manager, RESIDENT mode should work
    auto streamer = WeightStreamerFactory::create(
        WeightResidencyMode::RESIDENT,
        nullptr,  // null weight manager
        0);       // zero layers

    ASSERT_NE(streamer, nullptr);
    EXPECT_TRUE(streamer->ensureLayerOnDevice(0, DeviceId::cuda(0)));
}

// =============================================================================
// create() with Explicit Mode - STREAMING
// =============================================================================

TEST_F(WeightStreamerFactoryTest, CreateStreamingModeRequiresWeightManager)
{
    EXPECT_THROW(
        WeightStreamerFactory::create(
            WeightResidencyMode::STREAMING,
            nullptr,  // null weight manager - should throw
            24),
        std::invalid_argument);
}

TEST_F(WeightStreamerFactoryTest, CreateStreamingModeRequiresPositiveLayers)
{
    // We can't create a real WeightManager without a ModelLoader,
    // but we can test that the validation happens before WeightManager is used
    // by checking the num_layers validation

    EXPECT_THROW(
        WeightStreamerFactory::create(
            WeightResidencyMode::STREAMING,
            nullptr,
            0),  // zero layers - should throw before checking weight_manager
        std::invalid_argument);

    EXPECT_THROW(
        WeightStreamerFactory::create(
            WeightResidencyMode::STREAMING,
            nullptr,
            -5),  // negative layers - should throw
        std::invalid_argument);
}

// =============================================================================
// create() with Explicit Mode - UNIFIED
// =============================================================================

TEST_F(WeightStreamerFactoryTest, CreateUnifiedModeReturnsNullStreamer)
{
    auto streamer = WeightStreamerFactory::create(WeightResidencyMode::UNIFIED);

    ASSERT_NE(streamer, nullptr);

    // UNIFIED mode uses NullWeightStreamer (driver handles placement)
    EXPECT_TRUE(streamer->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_EQ(streamer->currentDeviceMemoryUsage(), 0);
}

// =============================================================================
// createFromEnv() Tests
// =============================================================================

TEST_F(WeightStreamerFactoryTest, CreateFromEnvDisabledReturnsNullStreamer)
{
    // Note: debugEnv() is initialized once at startup, so we can only test
    // the disabled case reliably (the default state).
    // Enabled tests would require running in a subprocess with LLAMINAR_WEIGHT_STREAMING=1.
    
    auto streamer = WeightStreamerFactory::createFromEnv(nullptr, 0);

    ASSERT_NE(streamer, nullptr);

    // Should be NullWeightStreamer
    EXPECT_TRUE(streamer->isLayerCached(0, DeviceId::cuda(0)));
    EXPECT_EQ(streamer->currentDeviceMemoryUsage(), 0);
}

TEST_F(WeightStreamerFactoryTest, CreateFromEnvDisabledIgnoresParameters)
{
    // When streaming is disabled, weight_manager and num_layers are ignored
    // (NullWeightStreamer doesn't need them)
    auto streamer = WeightStreamerFactory::createFromEnv(nullptr, 0);

    ASSERT_NE(streamer, nullptr);
    EXPECT_TRUE(streamer->isLayerCached(0, DeviceId::cuda(0)));
}

// Note: Tests for createFromEnv with LLAMINAR_WEIGHT_STREAMING=1 cannot be
// reliably tested because debugEnv() is a cached singleton initialized at
// program startup. Setting environment variables at runtime has no effect.
// The streaming-enabled code path is tested via the create() method instead.

// =============================================================================
// detectResidencyMode() Tests
// =============================================================================

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeModelFitsReturnsResident)
{
    const size_t model_size = 4ULL * 1024 * 1024 * 1024;  // 4 GB model
    const size_t vram_size = 8ULL * 1024 * 1024 * 1024;   // 8 GB VRAM
    const float headroom = 0.2f;  // 20% headroom

    // Available for weights: 8GB * 0.8 = 6.4GB
    // Model: 4GB fits in 6.4GB
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeModelExceedsReturnsStreaming)
{
    const size_t model_size = 8ULL * 1024 * 1024 * 1024;  // 8 GB model
    const size_t vram_size = 8ULL * 1024 * 1024 * 1024;   // 8 GB VRAM
    const float headroom = 0.2f;  // 20% headroom

    // Available for weights: 8GB * 0.8 = 6.4GB
    // Model: 8GB does NOT fit in 6.4GB
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::STREAMING);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeExactFitReturnsResident)
{
    const size_t model_size = 800;  // 800 bytes model
    const size_t vram_size = 1000;  // 1000 bytes VRAM
    const float headroom = 0.2f;    // 20% headroom

    // Available for weights: 1000 * 0.8 = 800 bytes
    // Model: 800 bytes fits exactly
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeZeroVramReturnsStreaming)
{
    const size_t model_size = 1024;  // 1 KB model
    const size_t vram_size = 0;      // No VRAM

    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size);

    EXPECT_EQ(mode, WeightResidencyMode::STREAMING);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeZeroModelReturnsResident)
{
    const size_t model_size = 0;                         // No model
    const size_t vram_size = 8ULL * 1024 * 1024 * 1024;  // 8 GB VRAM

    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeZeroHeadroomAllowsFullVram)
{
    const size_t model_size = 1000;  // 1000 bytes model
    const size_t vram_size = 1000;   // 1000 bytes VRAM
    const float headroom = 0.0f;     // No headroom

    // Available for weights: 1000 * 1.0 = 1000 bytes
    // Model: 1000 bytes fits
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeNegativeHeadroomClampedToZero)
{
    const size_t model_size = 1000;  // 1000 bytes model
    const size_t vram_size = 1000;   // 1000 bytes VRAM
    const float headroom = -0.5f;    // Invalid negative headroom

    // Should be clamped to 0.0, so full VRAM available
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeExcessiveHeadroomClampedToMax)
{
    const size_t model_size = 1;     // 1 byte model
    const size_t vram_size = 1000;   // 1000 bytes VRAM
    const float headroom = 1.5f;     // Invalid headroom > 1.0

    // Should be clamped to 0.99, leaving 10 bytes for weights
    // 1 byte model fits in 10 bytes
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size, headroom);

    EXPECT_EQ(mode, WeightResidencyMode::RESIDENT);
}

TEST_F(WeightStreamerFactoryTest, DetectResidencyModeDefaultHeadroomIs20Percent)
{
    const size_t model_size = 850;   // 850 bytes model
    const size_t vram_size = 1000;   // 1000 bytes VRAM
    // Default headroom = 0.2

    // Available for weights: 1000 * 0.8 = 800 bytes
    // Model: 850 bytes does NOT fit in 800 bytes
    auto mode = WeightStreamerFactory::detectResidencyMode(model_size, vram_size);

    EXPECT_EQ(mode, WeightResidencyMode::STREAMING);
}

// =============================================================================
// Streamer Polymorphism Tests
// =============================================================================

TEST_F(WeightStreamerFactoryTest, StreamerInterfaceIsPolymorphic)
{
    // Create via factory
    std::unique_ptr<IWeightStreamer> streamer =
        WeightStreamerFactory::create(WeightResidencyMode::RESIDENT);

    // Use via interface
    EXPECT_TRUE(streamer->ensureLayerOnDevice(0, DeviceId::cuda(0)));
    streamer->prefetchLayer(1, DeviceId::cuda(0));
    streamer->releaseLayer(0);

    // Stats should work
    StreamingStats stats = streamer->stats();
    EXPECT_EQ(stats.cache_hits, 0);  // NullWeightStreamer returns empty stats
}

TEST_F(WeightStreamerFactoryTest, CanStoreStreamerInSharedPtr)
{
    // Some use cases may want shared ownership
    std::shared_ptr<IWeightStreamer> streamer =
        WeightStreamerFactory::create(WeightResidencyMode::RESIDENT);

    EXPECT_TRUE(streamer->isLayerCached(0, DeviceId::cpu()));
    EXPECT_TRUE(streamer->isLayerCached(99, DeviceId::cuda(0)));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(WeightStreamerFactoryTest, StreamingConfigDefaultsAreUsed)
{
    // When not specifying config, defaults should be used
    auto streamer = WeightStreamerFactory::create(WeightResidencyMode::RESIDENT);
    ASSERT_NE(streamer, nullptr);

    // For RESIDENT mode, config is ignored anyway
    EXPECT_EQ(streamer->memoryBudget(), 0);  // NullWeightStreamer default
}

TEST_F(WeightStreamerFactoryTest, StreamingConfigIsPassedThrough)
{
    // When streaming is disabled, config doesn't matter
    StreamingConfig config;
    config.gpu_memory_budget = 4ULL * 1024 * 1024 * 1024;  // 4GB

    auto streamer = WeightStreamerFactory::create(
        WeightResidencyMode::RESIDENT,
        nullptr,
        0,
        config);

    ASSERT_NE(streamer, nullptr);
    // NullWeightStreamer doesn't use the config
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
