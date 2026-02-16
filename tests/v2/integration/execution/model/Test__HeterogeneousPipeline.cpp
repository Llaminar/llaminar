/**
 * @file Test__HeterogeneousPipeline.cpp
 * @brief End-to-end integration tests for heterogeneous pipeline execution
 *
 * **Phase 5.5**: Comprehensive integration tests verifying the complete
 * heterogeneous execution pipeline:
 *
 * 1. LayerPlacementConfig - Specifies which layers run where
 * 2. HeterogeneousLayerExecutor - Routes execution correctly
 * 3. CrossDomainTransfer - Handles activation transfers
 * 4. DomainAwareBufferManager - Allocates buffers correctly
 *
 * These tests exercise the full pipeline with actual GPU hardware when
 * available, with graceful fallback to CPU-only testing when not.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <numeric>

#include "execution/local_execution/model/HeterogeneousLayerExecutor.h"
#include "execution/local_execution/device/DomainAwareBufferManager.h"
#include "execution/local_execution/coherence/CrossDomainTransfer.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h" // Contains ComputeGraph definition
#include "config/LayerPlacementConfig.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h" // For DeviceManager
#include "tensors/TensorClasses.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// Macro to skip tests that require GPU when no GPU is available
// GTEST_SKIP() doesn't return from a helper function, so we need a macro
#define REQUIRE_GPU()                                                       \
    do                                                                      \
    {                                                                       \
        if (!gpu_available_)                                                \
        {                                                                   \
            GTEST_SKIP() << "No GPU available, skipping GPU-required test"; \
            return;                                                         \
        }                                                                   \
    } while (0)

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for heterogeneous pipeline integration tests
 *
 * Sets up the complete heterogeneous execution environment including:
 * - GPU detection and device selection
 * - Buffer manager with domain awareness
 * - Cross-domain transfer utility
 * - Test graph creation
 */
class HeterogeneousPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Detect GPU availability
        auto &dm = DeviceManager::instance();
        gpu_available_ = dm.has_gpu();

        if (gpu_available_)
        {
#ifdef HAVE_CUDA
            if (dm.cuda_device_count() > 0)
            {
                gpu_device_ = DeviceId::cuda(dm.get_device_id_for_type(
                    ComputeBackendType::GPU_CUDA, 0));
            }
#endif
#ifdef HAVE_ROCM
            if (!gpu_device_.is_gpu() && dm.rocm_device_count() > 0)
            {
                gpu_device_ = DeviceId::rocm(dm.get_device_id_for_type(
                    ComputeBackendType::GPU_ROCM, 0));
            }
#endif
        }

        // Initialize cross-domain transfer
        CrossDomainTransfer::Config transfer_cfg;
        transfer_cfg.use_pinned_memory = true;
        transfer_ = std::make_unique<CrossDomainTransfer>(transfer_cfg);
    }

    void TearDown() override
    {
        transfer_.reset();
        placement_config_.reset();
        buffer_manager_.reset();
        executor_.reset();
    }

    /**
     * @brief Create executor with the given placement config
     */
    void createExecutorWithPlacement(LayerPlacementConfig &placement,
                                     bool enable_profiling = true)
    {
        HeterogeneousLayerExecutor::Config config;
        config.placement_config = &placement;
        config.enable_profiling = enable_profiling;
        executor_ = std::make_unique<HeterogeneousLayerExecutor>(config);
    }

    /**
     * @brief Create buffer manager with the given placement config
     */
    void createBufferManagerWithPlacement(LayerPlacementConfig &placement)
    {
        DomainAwareBufferConfig config;
        config.placement_config = &placement;
        config.default_device = DeviceId::cpu();
        buffer_manager_ = std::make_unique<DomainAwareBufferManager>(config);
    }

    /**
     * @brief Create test tensors for pipeline testing
     * @param shape Tensor shape
     * @param seed Random seed for reproducibility
     * @return Vector of test tensors
     */
    std::vector<std::unique_ptr<FP32Tensor>> createTestTensors(
        const std::vector<size_t> &shape,
        int count,
        uint32_t base_seed = 42)
    {
        std::vector<std::unique_ptr<FP32Tensor>> tensors;
        for (int i = 0; i < count; ++i)
        {
            tensors.push_back(TestTensorFactory::createFP32Random(
                shape, -1.0f, 1.0f, base_seed + i));
        }
        return tensors;
    }

    /**
     * @brief Verify activations are preserved after transfer
     * @param original Original tensor data
     * @param transferred Transferred tensor
     * @param tolerance Comparison tolerance
     * @return true if values match within tolerance
     */
    bool verifyActivations(const std::vector<float> &original,
                           const TensorBase *transferred,
                           float tolerance = 1e-6f)
    {
        auto *fp32_tensor = dynamic_cast<const FP32Tensor *>(transferred);
        if (!fp32_tensor)
            return false;

        const float *data = fp32_tensor->data();
        for (size_t i = 0; i < original.size(); ++i)
        {
            if (std::abs(original[i] - data[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Compute mean squared error between two buffers
     */
    float computeMSE(const float *a, const float *b, size_t count)
    {
        float mse = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = a[i] - b[i];
            mse += diff * diff;
        }
        return mse / static_cast<float>(count);
    }

    // Members
    bool gpu_available_ = false;
    DeviceId gpu_device_ = DeviceId::cpu();

    std::unique_ptr<LayerPlacementConfig> placement_config_;
    std::unique_ptr<HeterogeneousLayerExecutor> executor_;
    std::unique_ptr<DomainAwareBufferManager> buffer_manager_;
    std::unique_ptr<CrossDomainTransfer> transfer_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, AllLayersOnCPU)
{
    // Setup: 10 layers all on CPU
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    createExecutorWithPlacement(placement);
    createBufferManagerWithPlacement(placement);

    // Verify placement
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(placement.isCPULayer(i)) << "Layer " << i << " should be CPU";
        EXPECT_EQ(executor_->getDeviceForLayer(i), DeviceId::cpu());
    }

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    // Verify stats
    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 10);
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

TEST_F(HeterogeneousPipelineTest, AllLayersOnGPU)
{
    REQUIRE_GPU();

    // Setup: 10 layers all on GPU
    auto placement = LayerPlacementConfig::allOnDevice(gpu_device_, 10);
    createExecutorWithPlacement(placement);
    createBufferManagerWithPlacement(placement);

    // Verify placement
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(placement.isGPULayer(i)) << "Layer " << i << " should be GPU";
        EXPECT_EQ(executor_->getDeviceForLayer(i), gpu_device_);
    }

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    // Verify stats
    auto stats = executor_->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 10);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

TEST_F(HeterogeneousPipelineTest, FirstHalfCPUSecondHalfGPU)
{
    REQUIRE_GPU();

    // Setup: First 5 on CPU, last 5 on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(5, 10, gpu_device_);
    createExecutorWithPlacement(placement);
    createBufferManagerWithPlacement(placement);

    // Verify placement
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(placement.isCPULayer(i)) << "Layer " << i << " should be CPU";
    }
    for (int i = 5; i < 10; ++i)
    {
        EXPECT_TRUE(placement.isGPULayer(i)) << "Layer " << i << " should be GPU";
    }

    // Verify single boundary
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 5);

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    // Verify stats
    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 5);
    EXPECT_EQ(stats.gpu_layers_executed, 5);
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}

TEST_F(HeterogeneousPipelineTest, FirstHalfGPUSecondHalfCPU)
{
    REQUIRE_GPU();

    // Setup: First 5 on GPU, last 5 on CPU
    auto placement = LayerPlacementConfig::cpuLastLayers(5, 10, gpu_device_);
    createExecutorWithPlacement(placement);
    createBufferManagerWithPlacement(placement);

    // Verify placement
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(placement.isGPULayer(i)) << "Layer " << i << " should be GPU";
    }
    for (int i = 5; i < 10; ++i)
    {
        EXPECT_TRUE(placement.isCPULayer(i)) << "Layer " << i << " should be CPU";
    }

    // Verify single boundary
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 5);

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    // Verify stats
    auto stats = executor_->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 5);
    EXPECT_EQ(stats.cpu_layers_executed, 5);
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}

// =============================================================================
// Domain Boundary Handling Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, SingleBoundaryTransfer)
{
    REQUIRE_GPU();

    // Setup: First 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, gpu_device_);
    createExecutorWithPlacement(placement);

    // Verify boundary detection
    EXPECT_FALSE(executor_->requiresCrossDomainTransfer(2, 3)); // Both CPU
    EXPECT_TRUE(executor_->requiresCrossDomainTransfer(3, 4));  // CPU -> GPU
    EXPECT_FALSE(executor_->requiresCrossDomainTransfer(4, 5)); // Both GPU

    // Execute and verify transfer counted
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}

TEST_F(HeterogeneousPipelineTest, MultipleBoundaryTransfers)
{
    REQUIRE_GPU();

    // Setup: CPU-GPU-CPU sandwich (0-2: CPU, 3-6: GPU, 7-9: CPU)
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 3; ++i)
        assignments.push_back({i, DeviceId::cpu(), 0});
    for (int i = 3; i < 7; ++i)
        assignments.push_back({i, gpu_device_, 0});
    for (int i = 7; i < 10; ++i)
        assignments.push_back({i, DeviceId::cpu(), 0});

    auto placement = LayerPlacementConfig::custom(assignments);
    createExecutorWithPlacement(placement);

    // Verify two boundaries
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 2);
    EXPECT_EQ(boundaries[0], 3); // CPU -> GPU
    EXPECT_EQ(boundaries[1], 7); // GPU -> CPU

    // Verify transfer requirements
    EXPECT_TRUE(executor_->requiresCrossDomainTransfer(2, 3));  // CPU -> GPU
    EXPECT_FALSE(executor_->requiresCrossDomainTransfer(5, 6)); // Both GPU
    EXPECT_TRUE(executor_->requiresCrossDomainTransfer(6, 7));  // GPU -> CPU

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 6); // 0-2, 7-9
    EXPECT_EQ(stats.gpu_layers_executed, 4); // 3-6
    EXPECT_EQ(stats.cross_domain_transfers, 2);
}

TEST_F(HeterogeneousPipelineTest, AlternatingDomains)
{
    REQUIRE_GPU();

    // Setup: Alternating CPU/GPU pattern
    auto placement = LayerPlacementConfig::alternating(10, DeviceId::cpu(), gpu_device_);
    createExecutorWithPlacement(placement);

    // Verify every transition requires transfer
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_TRUE(executor_->requiresCrossDomainTransfer(i, i + 1))
            << "Transfer should be required between " << i << " and " << (i + 1);
    }

    // 9 boundaries for alternating pattern
    auto boundaries = placement.getDomainBoundaries();
    EXPECT_EQ(boundaries.size(), 9);

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 5); // Even: 0, 2, 4, 6, 8
    EXPECT_EQ(stats.gpu_layers_executed, 5); // Odd: 1, 3, 5, 7, 9
    EXPECT_EQ(stats.cross_domain_transfers, 9);
}

// =============================================================================
// Buffer Allocation Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, GPULayersGetGPUBuffers)
{
    REQUIRE_GPU();

    // Setup: All GPU layers
    auto placement = LayerPlacementConfig::allOnDevice(gpu_device_, 10);
    createBufferManagerWithPlacement(placement);

    // Allocate buffers for layers
    for (int layer = 0; layer < 10; ++layer)
    {
        auto *buffer = buffer_manager_->allocateForLayer(
            layer, "activation_" + std::to_string(layer),
            {64, 128}, BufferTensorType::FP32);

        ASSERT_NE(buffer, nullptr) << "Failed to allocate buffer for layer " << layer;
        // Buffer should be allocated for GPU
    }

    // Check allocation stats
    auto alloc_stats = buffer_manager_->getStats();
    EXPECT_EQ(alloc_stats.gpu_buffer_count, 10);
    EXPECT_EQ(alloc_stats.cpu_buffer_count, 0);
}

TEST_F(HeterogeneousPipelineTest, CPULayersGetCPUBuffers)
{
    // Setup: All CPU layers
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    createBufferManagerWithPlacement(placement);

    // Allocate buffers for layers
    for (int layer = 0; layer < 10; ++layer)
    {
        auto *buffer = buffer_manager_->allocateForLayer(
            layer, "activation_" + std::to_string(layer),
            {64, 128}, BufferTensorType::FP32);

        ASSERT_NE(buffer, nullptr) << "Failed to allocate buffer for layer " << layer;
    }

    // Check allocation stats
    auto alloc_stats = buffer_manager_->getStats();
    EXPECT_EQ(alloc_stats.cpu_buffer_count, 10);
    EXPECT_EQ(alloc_stats.gpu_buffer_count, 0);
}

TEST_F(HeterogeneousPipelineTest, BuffersAllocatedCorrectly)
{
    REQUIRE_GPU();

    // Setup: Mixed placement
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, gpu_device_);
    createBufferManagerWithPlacement(placement);

    // Allocate buffers for all layers
    for (int layer = 0; layer < 10; ++layer)
    {
        auto *buffer = buffer_manager_->allocateForLayer(
            layer, "activation_" + std::to_string(layer),
            {64, 128}, BufferTensorType::FP32);

        ASSERT_NE(buffer, nullptr);
    }

    // Check allocation stats
    auto alloc_stats = buffer_manager_->getStats();
    EXPECT_EQ(alloc_stats.cpu_buffer_count, 4); // Layers 0-3
    EXPECT_EQ(alloc_stats.gpu_buffer_count, 6); // Layers 4-9
}

// =============================================================================
// Data Integrity Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, ActivationsPreservedAcrossTransfer)
{
    REQUIRE_GPU();

    // Create source tensor with known data
    auto src = TestTensorFactory::createFP32Random({64, 128}, -10.0f, 10.0f, 12345);

    // Store original data
    std::vector<float> original_data(src->numel());
    std::memcpy(original_data.data(), src->data(), src->size_bytes());

    // Create destination tensor
    auto dst = TestTensorFactory::createFP32({64, 128});

    // Transfer CPU -> GPU -> CPU round trip
    ASSERT_TRUE(transfer_->cpuToGpu(src.get(), dst.get(), gpu_device_));

    auto roundtrip = TestTensorFactory::createFP32({64, 128});
    ASSERT_TRUE(transfer_->gpuToCpu(dst.get(), roundtrip.get()));

    // Verify data integrity
    const float *roundtrip_data = roundtrip->data();
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(original_data[i], roundtrip_data[i], 1e-6f)
            << "Mismatch at index " << i;
    }

    // Verify transfer stats
    auto stats = transfer_->getStats();
    EXPECT_EQ(stats.cpu_to_gpu_count, 1);
    EXPECT_EQ(stats.gpu_to_cpu_count, 1);
}

TEST_F(HeterogeneousPipelineTest, NumericalStabilityAcrossTransfers)
{
    REQUIRE_GPU();

    // Test with extreme values
    auto extreme = TestTensorFactory::createFP32({32, 32});
    float *data = extreme->mutable_data();

    // Fill with extreme but valid float values
    for (size_t i = 0; i < extreme->numel(); ++i)
    {
        if (i % 4 == 0)
            data[i] = 1e-38f; // Small positive
        else if (i % 4 == 1)
            data[i] = -1e-38f; // Small negative
        else if (i % 4 == 2)
            data[i] = 1e38f; // Large positive
        else
            data[i] = -1e38f; // Large negative
    }

    // Store original
    std::vector<float> original_data(extreme->numel());
    std::memcpy(original_data.data(), extreme->data(), extreme->size_bytes());

    // Round-trip transfer
    auto gpu_tensor = TestTensorFactory::createFP32({32, 32});
    ASSERT_TRUE(transfer_->cpuToGpu(extreme.get(), gpu_tensor.get(), gpu_device_));

    auto roundtrip = TestTensorFactory::createFP32({32, 32});
    ASSERT_TRUE(transfer_->gpuToCpu(gpu_tensor.get(), roundtrip.get()));

    // Verify extreme values preserved
    const float *roundtrip_data = roundtrip->data();
    for (size_t i = 0; i < original_data.size(); ++i)
    {
        // Use relative tolerance for large values
        float expected = original_data[i];
        float actual = roundtrip_data[i];

        if (std::abs(expected) > 1.0f)
        {
            EXPECT_NEAR(actual / expected, 1.0f, 1e-6f)
                << "Relative mismatch at index " << i;
        }
        else
        {
            EXPECT_NEAR(expected, actual, 1e-6f)
                << "Absolute mismatch at index " << i;
        }
    }
}

// =============================================================================
// Performance Statistics Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, ExecutorStatsAccurate)
{
    REQUIRE_GPU();

    // Setup with profiling enabled
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, gpu_device_);
    createExecutorWithPlacement(placement, /*enable_profiling=*/true);

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    auto stats = executor_->getStats();

    // Counts should match placement
    EXPECT_EQ(stats.cpu_layers_executed, 4);
    EXPECT_EQ(stats.gpu_layers_executed, 6);
    EXPECT_EQ(stats.cross_domain_transfers, 1);

    // Timing should be non-negative
    EXPECT_GE(stats.cpu_time_ms, 0.0);
    EXPECT_GE(stats.gpu_time_ms, 0.0);
    EXPECT_GE(stats.transfer_time_ms, 0.0);

    // Total should match sum
    double expected_total = stats.cpu_time_ms + stats.gpu_time_ms + stats.transfer_time_ms;
    EXPECT_DOUBLE_EQ(stats.totalTimeMs(), expected_total);

    // Percentages should sum to 100 (or 0 if total is 0)
    double total_percent = stats.cpuTimePercent() + stats.gpuTimePercent() +
                           stats.transferTimePercent();
    if (stats.totalTimeMs() > 0)
    {
        EXPECT_NEAR(total_percent, 100.0, 0.1);
    }
}

TEST_F(HeterogeneousPipelineTest, TransferStatsAccurate)
{
    REQUIRE_GPU();

    // Create test tensors of known size
    const size_t rows = 64;
    const size_t cols = 128;
    const size_t expected_bytes = rows * cols * sizeof(float);

    auto src = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f, 42);
    auto dst = TestTensorFactory::createFP32({rows, cols});

    // Perform multiple transfers
    const int num_transfers = 5;
    for (int i = 0; i < num_transfers; ++i)
    {
        ASSERT_TRUE(transfer_->cpuToGpu(src.get(), dst.get(), gpu_device_));
        ASSERT_TRUE(transfer_->gpuToCpu(dst.get(), src.get()));
    }

    auto stats = transfer_->getStats();

    // Verify counts
    EXPECT_EQ(stats.cpu_to_gpu_count, num_transfers);
    EXPECT_EQ(stats.gpu_to_cpu_count, num_transfers);

    // Verify byte counts
    EXPECT_EQ(stats.cpu_to_gpu_bytes, expected_bytes * num_transfers);
    EXPECT_EQ(stats.gpu_to_cpu_bytes, expected_bytes * num_transfers);

    // Verify timing is positive
    EXPECT_GT(stats.cpu_to_gpu_time_ms, 0.0);
    EXPECT_GT(stats.gpu_to_cpu_time_ms, 0.0);
}

TEST_F(HeterogeneousPipelineTest, BufferStatsAccurate)
{
    REQUIRE_GPU();

    // Setup: Mixed placement
    auto placement = LayerPlacementConfig::cpuFirstLayers(3, 10, gpu_device_);
    createBufferManagerWithPlacement(placement);

    // Allocate buffers of known size
    const std::vector<size_t> shape = {64, 128};
    const size_t expected_size = 64 * 128 * sizeof(float);

    for (int layer = 0; layer < 10; ++layer)
    {
        buffer_manager_->allocateForLayer(
            layer, "buf_" + std::to_string(layer), shape, BufferTensorType::FP32);
    }

    auto stats = buffer_manager_->getStats();

    // Verify counts
    EXPECT_EQ(stats.cpu_buffer_count, 3);
    EXPECT_EQ(stats.gpu_buffer_count, 7);

    // Verify byte totals
    EXPECT_EQ(stats.cpu_bytes_allocated, expected_size * 3);
    EXPECT_EQ(stats.gpu_bytes_allocated, expected_size * 7);
    EXPECT_EQ(stats.total_bytes(), expected_size * 10);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, SingleLayer)
{
    REQUIRE_GPU();

    // Single GPU layer
    auto placement = LayerPlacementConfig::allOnDevice(gpu_device_, 1);
    createExecutorWithPlacement(placement);

    EXPECT_TRUE(placement.isGPULayer(0));
    EXPECT_EQ(placement.numLayers(), 1);
    EXPECT_EQ(placement.getDomainBoundaries().size(), 0); // No transitions

    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayer(0, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 1);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

TEST_F(HeterogeneousPipelineTest, TwoLayers)
{
    REQUIRE_GPU();

    // Two layers: CPU then GPU
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, gpu_device_, 0}};
    auto placement = LayerPlacementConfig::custom(assignments);
    createExecutorWithPlacement(placement);

    EXPECT_EQ(placement.numLayers(), 2);
    EXPECT_TRUE(executor_->requiresCrossDomainTransfer(0, 1));

    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 2, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 1);
    EXPECT_EQ(stats.gpu_layers_executed, 1);
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}

TEST_F(HeterogeneousPipelineTest, ManyLayers28)
{
    REQUIRE_GPU();

    // 28 layers (Qwen2.5-7B like): first 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, gpu_device_);
    createExecutorWithPlacement(placement);

    EXPECT_EQ(placement.numLayers(), 28);
    EXPECT_EQ(placement.getCPULayers().size(), 4);
    EXPECT_EQ(placement.getGPULayers().size(), 24);

    // Single boundary at layer 4
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 4);

    // Execute all layers
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 28, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 4);
    EXPECT_EQ(stats.gpu_layers_executed, 24);
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}

TEST_F(HeterogeneousPipelineTest, EmptyRange)
{
    // Empty range should be a no-op
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    createExecutorWithPlacement(placement);

    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(5, 5, &graph)); // Empty range

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 0);
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

// =============================================================================
// Stats Reset Tests
// =============================================================================

TEST_F(HeterogeneousPipelineTest, StatsResetBetweenRuns)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    createExecutorWithPlacement(placement);

    // First run
    ComputeGraph graph;
    executor_->executeLayerRange(0, 10, &graph);

    auto stats1 = executor_->getStats();
    EXPECT_EQ(stats1.cpu_layers_executed, 10);

    // Reset
    executor_->resetStats();

    // Verify reset
    auto stats2 = executor_->getStats();
    EXPECT_EQ(stats2.cpu_layers_executed, 0);
    EXPECT_EQ(stats2.gpu_layers_executed, 0);
    EXPECT_EQ(stats2.cross_domain_transfers, 0);
    EXPECT_DOUBLE_EQ(stats2.cpu_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats2.gpu_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats2.transfer_time_ms, 0.0);

    // Second run
    executor_->executeLayerRange(0, 5, &graph);

    auto stats3 = executor_->getStats();
    EXPECT_EQ(stats3.cpu_layers_executed, 5);
}

TEST_F(HeterogeneousPipelineTest, TransferStatsReset)
{
    REQUIRE_GPU();

    auto src = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f, 42);
    auto dst = TestTensorFactory::createFP32({32, 64});

    // First transfer
    ASSERT_TRUE(transfer_->cpuToGpu(src.get(), dst.get(), gpu_device_));

    auto stats1 = transfer_->getStats();
    EXPECT_EQ(stats1.cpu_to_gpu_count, 1);

    // Reset
    transfer_->resetStats();

    auto stats2 = transfer_->getStats();
    EXPECT_EQ(stats2.cpu_to_gpu_count, 0);
    EXPECT_EQ(stats2.gpu_to_cpu_count, 0);
    EXPECT_EQ(stats2.cpu_to_gpu_bytes, 0);
    EXPECT_EQ(stats2.gpu_to_cpu_bytes, 0);

    // Another transfer
    ASSERT_TRUE(transfer_->cpuToGpu(src.get(), dst.get(), gpu_device_));

    auto stats3 = transfer_->getStats();
    EXPECT_EQ(stats3.cpu_to_gpu_count, 1); // Fresh count
}

// =============================================================================
// Multi-GPU Tests (when available)
// =============================================================================

TEST_F(HeterogeneousPipelineTest, MultipleGPUsDifferentLayers)
{
    auto &dm = DeviceManager::instance();

    int cuda_count = 0;
    int rocm_count = 0;
#ifdef HAVE_CUDA
    cuda_count = dm.cuda_device_count();
#endif
#ifdef HAVE_ROCM
    rocm_count = dm.rocm_device_count();
#endif

    int total_gpus = cuda_count + rocm_count;
    if (total_gpus < 2)
    {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU test";
    }

    // Get two GPU devices
    DeviceId gpu0, gpu1;
#ifdef HAVE_CUDA
    if (cuda_count >= 2)
    {
        gpu0 = DeviceId::cuda(dm.get_device_id_for_type(ComputeBackendType::GPU_CUDA, 0));
        gpu1 = DeviceId::cuda(dm.get_device_id_for_type(ComputeBackendType::GPU_CUDA, 1));
    }
    else if (cuda_count == 1 && rocm_count >= 1)
    {
        gpu0 = DeviceId::cuda(dm.get_device_id_for_type(ComputeBackendType::GPU_CUDA, 0));
        gpu1 = DeviceId::rocm(dm.get_device_id_for_type(ComputeBackendType::GPU_ROCM, 0));
    }
#endif
#ifdef HAVE_ROCM
    if (!gpu0.is_gpu() && rocm_count >= 2)
    {
        gpu0 = DeviceId::rocm(dm.get_device_id_for_type(ComputeBackendType::GPU_ROCM, 0));
        gpu1 = DeviceId::rocm(dm.get_device_id_for_type(ComputeBackendType::GPU_ROCM, 1));
    }
#endif

    // Create placement: first 5 on GPU0, last 5 on GPU1
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 5; ++i)
        assignments.push_back({i, gpu0, 0});
    for (int i = 5; i < 10; ++i)
        assignments.push_back({i, gpu1, 0});

    auto placement = LayerPlacementConfig::custom(assignments);
    createExecutorWithPlacement(placement);

    // Verify routing
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(executor_->getDeviceForLayer(i), gpu0);
    }
    for (int i = 5; i < 10; ++i)
    {
        EXPECT_EQ(executor_->getDeviceForLayer(i), gpu1);
    }

    // Transfer required between GPUs
    EXPECT_TRUE(executor_->requiresCrossDomainTransfer(4, 5));

    // Execute
    ComputeGraph graph;
    EXPECT_TRUE(executor_->executeLayerRange(0, 10, &graph));

    auto stats = executor_->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 10);
    EXPECT_EQ(stats.cross_domain_transfers, 1);
}
