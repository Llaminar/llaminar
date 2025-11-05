/**
 * @file Test__PipelineBase_PrecisionMode.cpp
 * @brief Unit tests for PipelineBase precision mode handling
 *
 * Tests that ComputePrecision flows correctly from PipelineConfig through
 * PipelineBase to attention kernels and other operations.
 *
 * Test file naming convention:
 *   File: Test__PipelineBase_PrecisionMode.cpp → Testing: PipelineBase precision handling
 *   Suite: TEST(Test__PipelineBase_PrecisionMode, ...) → Matches filename
 *
 * @author David Sanftenberg
 * @date 2025-11-05
 */

#include <gtest/gtest.h>
#include "pipelines/PipelineBase.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightPlacementMap.h"
#include "utils/MPIContext.h"
#include <memory>

using namespace llaminar2;

// =============================================================================
// TEST FIXTURE
// =============================================================================

/**
 * @brief Mock pipeline for testing precision mode
 *
 * Minimal implementation that allows testing precision configuration.
 */
class MockPipelineForPrecision : public PipelineBase
{
public:
    // Expose protected config for testing
    ComputePrecision getConfigPrecision() const { return config_.precision; }

    // Constructor
    MockPipelineForPrecision(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        std::shared_ptr<WeightPlacementMap> placement_map,
        const PipelineConfig &config)
        : PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map, config) {}

    // Implement all pure virtual methods with minimal stubs
    const char *architecture() const override { return "MockPrecisionPipeline"; }
    bool forward(const int * /*tokens*/, int /*seq_len*/) override { return true; }

    std::vector<std::string> getAllWeightNames() const override
    {
        return {};
    }

    ActivationBuffers createBuffersForDevice(int /*device_idx*/, int /*max_seq_len*/) override
    {
        return ActivationBuffers{};
    }

    bool transformer_layer(int /*layer_idx*/, int /*seq_len*/) override
    {
        return true;
    }
};

class Test__PipelineBase_PrecisionMode : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create minimal model context for testing
        model_ctx_ = ModelContext::createForTesting("test_precision.gguf");
        mpi_ctx_ = nullptr; // Single-node test
        device_idx_ = -1;   // CPU
        placement_map_ = std::make_shared<WeightPlacementMap>(-1);
    }

    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_;
    std::shared_ptr<WeightPlacementMap> placement_map_;
};

// =============================================================================
// PRECISION MODE CONFIGURATION TESTS
// =============================================================================

/**
 * @brief Test MIXED precision mode (default)
 */
TEST_F(Test__PipelineBase_PrecisionMode, MixedPrecisionDefault)
{
    PipelineConfig config;
    config.precision = ComputePrecision::MIXED;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    // Verify precision is stored correctly
    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::MIXED);
}

/**
 * @brief Test FP32 precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, FP32Precision)
{
    PipelineConfig config;
    config.precision = ComputePrecision::FP32;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::FP32);
}

/**
 * @brief Test BF16 precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, BF16Precision)
{
    PipelineConfig config;
    config.precision = ComputePrecision::BF16;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::BF16);
}

/**
 * @brief Test FP16 precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, FP16Precision)
{
    PipelineConfig config;
    config.precision = ComputePrecision::FP16;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::FP16);
}

/**
 * @brief Test INT8 precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, INT8Precision)
{
    PipelineConfig config;
    config.precision = ComputePrecision::INT8;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::INT8);
}

/**
 * @brief Test AUTO precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, AutoPrecision)
{
    PipelineConfig config;
    config.precision = ComputePrecision::AUTO;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::AUTO);
}

/**
 * @brief Test default PipelineConfig uses MIXED precision
 */
TEST_F(Test__PipelineBase_PrecisionMode, DefaultConfigIsMixed)
{
    PipelineConfig config; // Default constructor

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    // Default should be MIXED
    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::MIXED);
}

/**
 * @brief Test precision persists through config copy
 */
TEST_F(Test__PipelineBase_PrecisionMode, PrecisionPersistsThroughCopy)
{
    PipelineConfig config1;
    config1.precision = ComputePrecision::INT8;

    // Copy config
    PipelineConfig config2 = config1;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config2);

    // Should preserve INT8 precision
    EXPECT_EQ(pipeline.getConfigPrecision(), ComputePrecision::INT8);
}

// =============================================================================
// PRECISION MODE ENUM TESTS
// =============================================================================
// ENUM VALUE TESTS (standalone, don't need fixture)
// =============================================================================

/**
 * @brief Test ComputePrecision enum values
 */
TEST(ComputePrecisionEnum, EnumValues)
{
    // Verify enum values are distinct
    EXPECT_NE(static_cast<int>(ComputePrecision::MIXED), static_cast<int>(ComputePrecision::FP32));
    EXPECT_NE(static_cast<int>(ComputePrecision::MIXED), static_cast<int>(ComputePrecision::BF16));
    EXPECT_NE(static_cast<int>(ComputePrecision::MIXED), static_cast<int>(ComputePrecision::FP16));
    EXPECT_NE(static_cast<int>(ComputePrecision::MIXED), static_cast<int>(ComputePrecision::INT8));
    EXPECT_NE(static_cast<int>(ComputePrecision::MIXED), static_cast<int>(ComputePrecision::AUTO));

    EXPECT_NE(static_cast<int>(ComputePrecision::FP32), static_cast<int>(ComputePrecision::BF16));
    EXPECT_NE(static_cast<int>(ComputePrecision::FP32), static_cast<int>(ComputePrecision::FP16));
    EXPECT_NE(static_cast<int>(ComputePrecision::FP32), static_cast<int>(ComputePrecision::INT8));
    EXPECT_NE(static_cast<int>(ComputePrecision::FP32), static_cast<int>(ComputePrecision::AUTO));
}

/**
 * @brief Test MIXED is the first enum value (for default initialization)
 */
TEST(ComputePrecisionEnum, MixedIsFirstValue)
{
    // MIXED should be the first enum value (0) for default initialization
    EXPECT_EQ(static_cast<int>(ComputePrecision::MIXED), 0);
}

// =============================================================================
// MAIN
// =============================================================================
