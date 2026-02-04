/**
 * @file Test__PerStageBufferPool.cpp
 * @brief Unit tests for PerStageBufferPool
 *
 * Tests per-PP-stage buffer allocation for heterogeneous execution.
 * Uses CPU-only configurations so tests run without GPU hardware.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/device/PerStageBufferPool.h"
#include "config/PipelineConfig.h"
#include "backends/BackendManager.h"

namespace llaminar2
{
namespace
{

/**
 * @brief Test fixture for PerStageBufferPool tests
 *
 * Uses CPU-only configurations so tests run without GPU hardware.
 */
class Test__PerStageBufferPool : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure CPU backend is initialized for tensor allocation
        if (!hasCPUBackend())
        {
            initCPUBackend(-1); // System-wide memory (no NUMA binding)
        }

        // Create a simple 2-stage PP config for testing
        // Stage 0: layers 0-11, Stage 1: layers 12-23
        config_ = createTwoStageCPUConfig(24);
    }

    /**
     * @brief Create a 2-stage PP config on CPU devices
     *
     * Both stages on CPU to avoid GPU requirements in unit tests.
     */
    static PipelineConfig createTwoStageCPUConfig(int total_layers)
    {
        PipelineConfig config;
        config.total_layers = total_layers;

        // Two CPU domains (different logical CPUs for testing)
        config.tp_domains = {
            TPDomainConfig{
                .name = "cpu_stage0",
                .devices = {DeviceId::cpu()},
                .tp_backend = CollectiveBackendType::HOST},
            TPDomainConfig{
                .name = "cpu_stage1",
                .devices = {DeviceId::cpu()},
                .tp_backend = CollectiveBackendType::HOST}};

        int split = total_layers / 2;
        config.pp_stages = {
            PPStageConfig::firstStage(0, "cpu_stage0", 0, split),
            PPStageConfig::lastStage(1, "cpu_stage1", split, total_layers)};

        return config;
    }

    /**
     * @brief Create a small buffer spec for testing
     */
    static PPStageBufferSpec createSmallBufferSpec()
    {
        PPStageBufferSpec spec;
        spec.batch_size = 1;
        spec.seq_len = 4; // Small for fast tests
        spec.d_model = 64;
        spec.n_heads = 2;
        spec.n_kv_heads = 2;
        spec.head_dim = 32;
        spec.intermediate_size = 128;
        spec.vocab_size = 1000;
        spec.precision = ActivationPrecision::FP32;
        spec.enable_snapshots = false;
        return spec;
    }

    PipelineConfig config_;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, InitializeWithTwoStages)
{
    PerStageBufferPool pool;
    EXPECT_FALSE(pool.isInitialized());

    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    EXPECT_TRUE(pool.isInitialized());
    EXPECT_EQ(pool.numStages(), 2);
}

TEST_F(Test__PerStageBufferPool, InitializeWithEmptyConfigSucceedsWithZeroStages)
{
    PerStageBufferPool pool;
    PipelineConfig empty_config;
    auto spec = createSmallBufferSpec();

    // Empty config initializes with 0 stages (valid but useless)
    EXPECT_TRUE(pool.initialize(empty_config, spec));
    EXPECT_TRUE(pool.isInitialized());
    EXPECT_EQ(pool.numStages(), 0);
}

TEST_F(Test__PerStageBufferPool, InitializeWithNoStagesSucceedsWithZeroStages)
{
    PerStageBufferPool pool;
    PipelineConfig config;
    config.total_layers = 24;
    config.tp_domains = {
        TPDomainConfig{
            .name = "cpu_domain",
            .devices = {DeviceId::cpu()},
            .tp_backend = CollectiveBackendType::HOST}};
    // No pp_stages added

    auto spec = createSmallBufferSpec();
    // Config with domains but no stages initializes with 0 stages
    EXPECT_TRUE(pool.initialize(config, spec));
    EXPECT_TRUE(pool.isInitialized());
    EXPECT_EQ(pool.numStages(), 0);
}

// ============================================================================
// Stage Buffer Access Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, ForStageReturnsValidBuffers)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    // Access stage 0 buffers
    auto &buffers0 = pool.forStage(0);
    EXPECT_NE(buffers0.residual, nullptr);
    EXPECT_NE(buffers0.Q, nullptr);
    EXPECT_NE(buffers0.K, nullptr);
    EXPECT_NE(buffers0.V, nullptr);
    EXPECT_NE(buffers0.current_hidden, nullptr);

    // Access stage 1 buffers
    auto &buffers1 = pool.forStage(1);
    EXPECT_NE(buffers1.residual, nullptr);
    EXPECT_NE(buffers1.Q, nullptr);

    // Different stages should have different buffer pointers
    EXPECT_NE(buffers0.residual, buffers1.residual);
    EXPECT_NE(buffers0.Q, buffers1.Q);
}

TEST_F(Test__PerStageBufferPool, ForStageThrowsOnInvalidStage)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    EXPECT_THROW(pool.forStage(-1), std::out_of_range);
    EXPECT_THROW(pool.forStage(2), std::out_of_range); // Only stages 0,1 exist
    EXPECT_THROW(pool.forStage(100), std::out_of_range);
}

TEST_F(Test__PerStageBufferPool, ConstForStageWorks)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    const PerStageBufferPool &const_pool = pool;
    const auto &buffers = const_pool.forStage(0);
    EXPECT_NE(buffers.residual, nullptr);
}

TEST_F(Test__PerStageBufferPool, ForStageThrowsIfNotInitialized)
{
    PerStageBufferPool pool;
    // Implementation throws out_of_range for invalid stage IDs (including when not initialized)
    EXPECT_THROW(pool.forStage(0), std::out_of_range);
}

// ============================================================================
// Layer-to-Stage Mapping Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, ForLayerReturnsCorrectStageBuffers)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    // Stage 0 owns layers 0-11, Stage 1 owns layers 12-23
    auto &stage0_buffers = pool.forStage(0);
    auto &stage1_buffers = pool.forStage(1);

    // Layers in stage 0 should return stage 0 buffers
    EXPECT_EQ(&pool.forLayer(0), &stage0_buffers);
    EXPECT_EQ(&pool.forLayer(5), &stage0_buffers);
    EXPECT_EQ(&pool.forLayer(11), &stage0_buffers);

    // Layers in stage 1 should return stage 1 buffers
    EXPECT_EQ(&pool.forLayer(12), &stage1_buffers);
    EXPECT_EQ(&pool.forLayer(18), &stage1_buffers);
    EXPECT_EQ(&pool.forLayer(23), &stage1_buffers);
}

TEST_F(Test__PerStageBufferPool, ForLayerThrowsOnOutOfRange)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    EXPECT_THROW(pool.forLayer(-1), std::out_of_range);
    EXPECT_THROW(pool.forLayer(24), std::out_of_range); // total_layers = 24
    EXPECT_THROW(pool.forLayer(100), std::out_of_range);
}

TEST_F(Test__PerStageBufferPool, ForLayerThrowsIfNotInitialized)
{
    PerStageBufferPool pool;
    EXPECT_THROW(pool.forLayer(0), std::runtime_error);
}

// ============================================================================
// Device Placement Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, DeviceForStageReturnsPrimaryDevice)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    // Both stages on CPU in our test config
    EXPECT_EQ(pool.deviceForStage(0).type, DeviceType::CPU);
    EXPECT_EQ(pool.deviceForStage(1).type, DeviceType::CPU);
}

TEST_F(Test__PerStageBufferPool, DeviceForStageThrowsOnInvalid)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    EXPECT_THROW(pool.deviceForStage(-1), std::out_of_range);
    EXPECT_THROW(pool.deviceForStage(5), std::out_of_range);
}

TEST_F(Test__PerStageBufferPool, DeviceForStageThrowsIfNotInitialized)
{
    PerStageBufferPool pool;
    EXPECT_THROW(pool.deviceForStage(0), std::runtime_error);
}

// ============================================================================
// Release and Reinitialize Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, ReleaseFreesAllBuffers)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));
    EXPECT_TRUE(pool.isInitialized());

    pool.release();

    EXPECT_FALSE(pool.isInitialized());
    EXPECT_EQ(pool.numStages(), 0);
}

TEST_F(Test__PerStageBufferPool, CanReinitializeAfterRelease)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();

    ASSERT_TRUE(pool.initialize(config_, spec));
    pool.release();

    // Should be able to reinitialize
    ASSERT_TRUE(pool.initialize(config_, spec));
    EXPECT_TRUE(pool.isInitialized());
    EXPECT_EQ(pool.numStages(), 2);

    // Buffers should be valid
    auto &buffers = pool.forStage(0);
    EXPECT_NE(buffers.Q, nullptr);
}

TEST_F(Test__PerStageBufferPool, ReleaseResetsStats)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    EXPECT_GT(pool.stats().total_bytes(), 0u);

    pool.release();

    EXPECT_EQ(pool.stats().total_bytes(), 0u);
    EXPECT_EQ(pool.stats().total_buffers(), 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, StatsReportAllocation)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    const auto &stats = pool.stats();

    // Should have allocated some bytes
    EXPECT_GT(stats.total_bytes(), 0u);
    EXPECT_GT(stats.total_buffers(), 0);

    // CPU allocations (our test config uses CPU)
    EXPECT_GT(stats.cpu_bytes_allocated, 0u);
    EXPECT_GT(stats.cpu_buffer_count, 0);

    // No GPU allocations in this test
    EXPECT_EQ(stats.gpu_bytes_allocated, 0u);
    EXPECT_EQ(stats.gpu_buffer_count, 0);
}

TEST_F(Test__PerStageBufferPool, StatsTrackPerDevice)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    const auto &stats = pool.stats();

    // Should have per-device tracking
    EXPECT_FALSE(stats.bytes_per_device.empty());
    EXPECT_FALSE(stats.buffers_per_device.empty());
}

TEST_F(Test__PerStageBufferPool, StatsEmptyBeforeInit)
{
    PerStageBufferPool pool;
    const auto &stats = pool.stats();

    EXPECT_EQ(stats.total_bytes(), 0u);
    EXPECT_EQ(stats.total_buffers(), 0);
    EXPECT_EQ(stats.cpu_bytes_allocated, 0u);
    EXPECT_EQ(stats.gpu_bytes_allocated, 0u);
}

// ============================================================================
// Buffer Content Tests
// ============================================================================

TEST_F(Test__PerStageBufferPool, BufferShapesMatchSpec)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    auto &buffers = pool.forStage(0);

    // Check Q buffer shape: [tokens, n_heads * head_dim]
    size_t expected_q_elements = spec.total_tokens() * spec.n_heads * spec.head_dim;
    EXPECT_EQ(buffers.Q->numel(), expected_q_elements);

    // Check residual shape: [tokens, d_model]
    size_t expected_residual = spec.total_tokens() * spec.d_model;
    // For FP32 mode, should match exactly
    if (spec.precision == ActivationPrecision::FP32)
    {
        EXPECT_EQ(buffers.residual->numel(), expected_residual);
    }
}

TEST_F(Test__PerStageBufferPool, StagesHaveIndependentBuffers)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    auto &buf0 = pool.forStage(0);
    auto &buf1 = pool.forStage(1);

    // All buffer pointers should be different between stages
    EXPECT_NE(buf0.Q, buf1.Q);
    EXPECT_NE(buf0.K, buf1.K);
    EXPECT_NE(buf0.V, buf1.V);
    EXPECT_NE(buf0.residual, buf1.residual);
    EXPECT_NE(buf0.normalized, buf1.normalized);
    EXPECT_NE(buf0.gate, buf1.gate);
    EXPECT_NE(buf0.up, buf1.up);
}

// ============================================================================
// Three-Stage Configuration Test
// ============================================================================

TEST_F(Test__PerStageBufferPool, SupportsThreeStages)
{
    // Create a 3-stage config
    PipelineConfig config;
    config.total_layers = 24;

    config.tp_domains = {
        TPDomainConfig{.name = "stage0", .devices = {DeviceId::cpu()}, .tp_backend = CollectiveBackendType::HOST},
        TPDomainConfig{.name = "stage1", .devices = {DeviceId::cpu()}, .tp_backend = CollectiveBackendType::HOST},
        TPDomainConfig{.name = "stage2", .devices = {DeviceId::cpu()}, .tp_backend = CollectiveBackendType::HOST}};

    config.pp_stages = {
        PPStageConfig::firstStage(0, "stage0", 0, 8),
        PPStageConfig::middleStage(1, "stage1", 8, 16),
        PPStageConfig::lastStage(2, "stage2", 16, 24)};

    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config, spec));

    EXPECT_EQ(pool.numStages(), 3);

    // Each stage should have its own buffers
    auto &buf0 = pool.forStage(0);
    auto &buf1 = pool.forStage(1);
    auto &buf2 = pool.forStage(2);

    EXPECT_NE(buf0.Q, buf1.Q);
    EXPECT_NE(buf1.Q, buf2.Q);
    EXPECT_NE(buf0.Q, buf2.Q);

    // Layer mapping should work
    EXPECT_EQ(&pool.forLayer(5), &buf0);
    EXPECT_EQ(&pool.forLayer(10), &buf1);
    EXPECT_EQ(&pool.forLayer(20), &buf2);
}

// ============================================================================
// Single Stage Configuration Test
// ============================================================================

TEST_F(Test__PerStageBufferPool, SupportsSingleStage)
{
    // Create a single-stage config (no PP)
    PipelineConfig config = PipelineConfig::singleDevice(24, DeviceId::cpu());

    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config, spec));

    EXPECT_EQ(pool.numStages(), 1);
    EXPECT_TRUE(pool.isInitialized());

    // Should be able to access the single stage
    auto &buffers = pool.forStage(0);
    EXPECT_NE(buffers.Q, nullptr);
    EXPECT_NE(buffers.residual, nullptr);

    // All layers map to stage 0
    EXPECT_EQ(&pool.forLayer(0), &buffers);
    EXPECT_EQ(&pool.forLayer(12), &buffers);
    EXPECT_EQ(&pool.forLayer(23), &buffers);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__PerStageBufferPool, NumStagesZeroBeforeInit)
{
    PerStageBufferPool pool;
    EXPECT_EQ(pool.numStages(), 0);
}

TEST_F(Test__PerStageBufferPool, BuffersWritable)
{
    PerStageBufferPool pool;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool.initialize(config_, spec));

    auto &buffers = pool.forStage(0);

    // Verify we can write to the buffers (FP32 mode)
    if (buffers.Q && buffers.Q->numel() > 0)
    {
        float *data = static_cast<float *>(buffers.Q->mutable_data());
        data[0] = 1.0f; // Should not crash
        EXPECT_EQ(data[0], 1.0f);
    }
}

TEST_F(Test__PerStageBufferPool, MoveConstruction)
{
    PerStageBufferPool pool1;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool1.initialize(config_, spec));

    // Move construct
    PerStageBufferPool pool2 = std::move(pool1);

    EXPECT_TRUE(pool2.isInitialized());
    EXPECT_EQ(pool2.numStages(), 2);

    // pool2 should have valid buffers
    auto &buffers = pool2.forStage(0);
    EXPECT_NE(buffers.Q, nullptr);
}

TEST_F(Test__PerStageBufferPool, MoveAssignment)
{
    PerStageBufferPool pool1;
    auto spec = createSmallBufferSpec();
    ASSERT_TRUE(pool1.initialize(config_, spec));

    // Move assign
    PerStageBufferPool pool2;
    pool2 = std::move(pool1);

    EXPECT_TRUE(pool2.isInitialized());
    EXPECT_EQ(pool2.numStages(), 2);

    // pool2 should have valid buffers
    auto &buffers = pool2.forStage(0);
    EXPECT_NE(buffers.Q, nullptr);
}

} // namespace
} // namespace llaminar2
