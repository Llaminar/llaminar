/**
 * @file Test__HeterogeneousLayerExecutorIntegration.cpp
 * @brief Integration tests for HeterogeneousLayerExecutor with LayerPlacementConfig
 *
 * Tests the integration between HeterogeneousLayerExecutor and LayerPlacementConfig
 * to verify:
 * - Correct device routing based on placement configuration
 * - Domain boundary detection and transfer triggering
 * - Statistics tracking across mixed CPU/GPU execution
 * - Various placement patterns (first N, last N, alternating, etc.)
 *
 * Part of Phase 5.4: HeterogeneousLayerExecutor + LayerPlacementConfig Integration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/model/HeterogeneousLayerExecutor.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "config/LayerPlacementConfig.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__HeterogeneousIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a basic compute graph for testing
        graph_ = std::make_unique<ComputeGraph>();
    }

    /**
     * @brief Create a HeterogeneousLayerExecutor with the given placement config
     */
    std::unique_ptr<HeterogeneousLayerExecutor> createExecutor(
        LayerPlacementConfig &placement,
        bool enable_profiling = true)
    {
        HeterogeneousLayerExecutor::Config config;
        config.placement_config = &placement;
        config.enable_profiling = enable_profiling;
        return std::make_unique<HeterogeneousLayerExecutor>(config);
    }

    std::unique_ptr<ComputeGraph> graph_;
};

// =============================================================================
// All GPU Layer Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, AllGPULayersExecuteOnGPU)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto executor = createExecutor(placement);

    // Execute all layers
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(executor->executeLayer(i, graph_.get()))
            << "Layer " << i << " should execute successfully";
    }

    // Verify all went to GPU
    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 10);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

TEST_F(Test__HeterogeneousIntegration, AllGPULayersRouteCorrectly)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto executor = createExecutor(placement);

    // Verify routing queries
    for (int i = 0; i < 10; ++i)
    {
        DeviceId device = executor->getDeviceForLayer(i);
        EXPECT_TRUE(device.is_gpu()) << "Layer " << i << " should route to GPU";
        EXPECT_EQ(device, DeviceId::cuda(0));
    }
}

// =============================================================================
// All CPU Layer Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, AllCPULayersExecuteOnCPU)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    auto executor = createExecutor(placement);

    // Execute all layers
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(executor->executeLayer(i, graph_.get()))
            << "Layer " << i << " should execute successfully";
    }

    // Verify all went to CPU
    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 10);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
}

TEST_F(Test__HeterogeneousIntegration, AllCPULayersRouteCorrectly)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    auto executor = createExecutor(placement);

    // Verify routing queries
    for (int i = 0; i < 10; ++i)
    {
        DeviceId device = executor->getDeviceForLayer(i);
        EXPECT_TRUE(device.is_cpu()) << "Layer " << i << " should route to CPU";
    }
}

// =============================================================================
// Mixed Placement Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, MixedPlacementRoutesCorrectly)
{
    // First 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Verify routing
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(placement.isCPULayer(i))
            << "Layer " << i << " should be CPU according to config";
        EXPECT_TRUE(executor->getDeviceForLayer(i).is_cpu())
            << "Layer " << i << " should route to CPU";
    }

    for (int i = 4; i < 10; ++i)
    {
        EXPECT_TRUE(placement.isGPULayer(i))
            << "Layer " << i << " should be GPU according to config";
        EXPECT_TRUE(executor->getDeviceForLayer(i).is_gpu())
            << "Layer " << i << " should route to GPU";
    }
}

TEST_F(Test__HeterogeneousIntegration, MixedPlacementExecutesCorrectly)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Execute all layers individually
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(executor->executeLayer(i, graph_.get()));
    }

    auto stats = executor->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 4);
    EXPECT_EQ(stats.gpu_layers_executed, 6);
}

// =============================================================================
// Domain Boundary Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, DomainBoundaryTriggersTransfer)
{
    // First 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Verify domain boundaries match between config and executor
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 4);

    // Verify transfer detection at boundary
    EXPECT_FALSE(executor->requiresCrossDomainTransfer(2, 3)); // Same domain (CPU)
    EXPECT_TRUE(executor->requiresCrossDomainTransfer(3, 4));  // Cross boundary
    EXPECT_FALSE(executor->requiresCrossDomainTransfer(4, 5)); // Same domain (GPU)
}

TEST_F(Test__HeterogeneousIntegration, DomainBoundaryInRangeExecution)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Execute range that crosses boundary
    EXPECT_TRUE(executor->executeLayerRange(0, 10, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 4);
    EXPECT_EQ(stats.gpu_layers_executed, 6);
    EXPECT_EQ(stats.cross_domain_transfers, 1); // One transfer at layer 3->4
}

TEST_F(Test__HeterogeneousIntegration, MultipleDomainBoundaries)
{
    // Create a sandwich pattern: CPU, GPU, CPU
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 3; ++i)
        assignments.push_back({i, DeviceId::cpu(), 0});
    for (int i = 3; i < 7; ++i)
        assignments.push_back({i, DeviceId::cuda(0), 0});
    for (int i = 7; i < 10; ++i)
        assignments.push_back({i, DeviceId::cpu(), 0});

    auto placement = LayerPlacementConfig::custom(assignments);
    auto executor = createExecutor(placement);

    // Verify boundaries
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 2);
    EXPECT_EQ(boundaries[0], 3); // CPU -> GPU
    EXPECT_EQ(boundaries[1], 7); // GPU -> CPU

    // Execute full range
    EXPECT_TRUE(executor->executeLayerRange(0, 10, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 6);    // 0-2 + 7-9
    EXPECT_EQ(stats.gpu_layers_executed, 4);    // 3-6
    EXPECT_EQ(stats.cross_domain_transfers, 2); // Two boundaries
}

// =============================================================================
// Multiple GPU Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, MultipleGPUsDifferentLayers)
{
    // Layers on different GPUs
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 5; ++i)
        assignments.push_back({i, DeviceId::cuda(0), 0});
    for (int i = 5; i < 10; ++i)
        assignments.push_back({i, DeviceId::cuda(1), 0});

    auto placement = LayerPlacementConfig::custom(assignments);
    auto executor = createExecutor(placement);

    // Verify routing
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(executor->getDeviceForLayer(i), DeviceId::cuda(0));
    }
    for (int i = 5; i < 10; ++i)
    {
        EXPECT_EQ(executor->getDeviceForLayer(i), DeviceId::cuda(1));
    }

    // Domain boundary between different GPUs
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 5);

    // Transfer required between GPUs
    EXPECT_TRUE(executor->requiresCrossDomainTransfer(4, 5));
}

TEST_F(Test__HeterogeneousIntegration, MixedCUDAAndROCm)
{
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 4; ++i)
        assignments.push_back({i, DeviceId::cuda(0), 0});
    for (int i = 4; i < 8; ++i)
        assignments.push_back({i, DeviceId::rocm(0), 0});

    auto placement = LayerPlacementConfig::custom(assignments);
    auto executor = createExecutor(placement);

    // All are GPU layers
    auto gpu_layers = placement.getGPULayers();
    EXPECT_EQ(gpu_layers.size(), 8);

    // But they're on different device types
    EXPECT_EQ(placement.getDeviceTypeForLayer(0), DeviceType::CUDA);
    EXPECT_EQ(placement.getDeviceTypeForLayer(4), DeviceType::ROCm);

    // Domain boundary exists
    EXPECT_TRUE(executor->requiresCrossDomainTransfer(3, 4));
}

// =============================================================================
// First N / Last N Pattern Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, FirstNLayersOnGPU)
{
    // First 6 on GPU, rest on CPU
    auto placement = LayerPlacementConfig::cpuLastLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Verify first 6 are GPU
    auto gpu_layers = placement.getGPULayers();
    EXPECT_EQ(gpu_layers, (std::vector<int>{0, 1, 2, 3, 4, 5}));

    // Verify last 4 are CPU
    auto cpu_layers = placement.getCPULayers();
    EXPECT_EQ(cpu_layers, (std::vector<int>{6, 7, 8, 9}));

    // Execute and verify
    EXPECT_TRUE(executor->executeLayerRange(0, 10, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 6);
    EXPECT_EQ(stats.cpu_layers_executed, 4);
}

TEST_F(Test__HeterogeneousIntegration, LastNLayersOnCPU)
{
    // Same as above but explicit test
    auto placement = LayerPlacementConfig::cpuLastLayers(3, 10, DeviceId::cuda(0));

    // Boundary should be at layer 7
    auto boundaries = placement.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 7);

    // Layer 6 is GPU, layer 7 is CPU
    EXPECT_TRUE(placement.isGPULayer(6));
    EXPECT_TRUE(placement.isCPULayer(7));
}

// =============================================================================
// Alternating Pattern Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, AlternatingPlacement)
{
    // Even on CPU, odd on GPU
    auto placement = LayerPlacementConfig::alternating(10, DeviceId::cpu(), DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Verify alternating pattern
    for (int i = 0; i < 10; ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_TRUE(placement.isCPULayer(i)) << "Layer " << i << " should be CPU";
        }
        else
        {
            EXPECT_TRUE(placement.isGPULayer(i)) << "Layer " << i << " should be GPU";
        }
    }

    // Every adjacent pair requires transfer
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_TRUE(executor->requiresCrossDomainTransfer(i, i + 1))
            << "Transfer should be required between " << i << " and " << (i + 1);
    }

    // Boundaries at every transition
    auto boundaries = placement.getDomainBoundaries();
    EXPECT_EQ(boundaries.size(), 9); // Layers 1,2,3,4,5,6,7,8,9
}

TEST_F(Test__HeterogeneousIntegration, AlternatingExecutionStats)
{
    auto placement = LayerPlacementConfig::alternating(10, DeviceId::cpu(), DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Execute full range
    EXPECT_TRUE(executor->executeLayerRange(0, 10, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 5);    // Even layers: 0, 2, 4, 6, 8
    EXPECT_EQ(stats.gpu_layers_executed, 5);    // Odd layers: 1, 3, 5, 7, 9
    EXPECT_EQ(stats.cross_domain_transfers, 9); // 9 transitions
}

// =============================================================================
// Single Layer Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, SingleGPULayer)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 1);
    auto executor = createExecutor(placement);

    // Single layer execution
    EXPECT_TRUE(executor->executeLayer(0, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 1);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);

    // No boundaries possible with single layer
    EXPECT_EQ(placement.getDomainBoundaries().size(), 0);
}

TEST_F(Test__HeterogeneousIntegration, SingleCPULayer)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 1);
    auto executor = createExecutor(placement);

    EXPECT_TRUE(executor->executeLayer(0, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 1);
}

// =============================================================================
// Empty/Edge Case Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, EmptyPlacementDefaultBehavior)
{
    // Empty config - not really valid but test behavior
    LayerPlacementConfig placement;

    // Should have no layers
    EXPECT_EQ(placement.numLayers(), 0);
    EXPECT_EQ(placement.getGPULayers().size(), 0);
    EXPECT_EQ(placement.getCPULayers().size(), 0);
    EXPECT_EQ(placement.getDomainBoundaries().size(), 0);
}

TEST_F(Test__HeterogeneousIntegration, EmptyRangeExecution)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto executor = createExecutor(placement);

    // Empty range should succeed with no-op
    EXPECT_TRUE(executor->executeLayerRange(5, 5, graph_.get()));

    auto stats = executor->getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
}

// =============================================================================
// Statistics Reset Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, StatsResetBetweenRuns)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // First run
    executor->executeLayerRange(0, 10, graph_.get());
    auto stats1 = executor->getStats();
    EXPECT_EQ(stats1.cpu_layers_executed, 4);
    EXPECT_EQ(stats1.gpu_layers_executed, 6);

    // Reset stats
    executor->resetStats();

    // Verify reset
    auto stats2 = executor->getStats();
    EXPECT_EQ(stats2.cpu_layers_executed, 0);
    EXPECT_EQ(stats2.gpu_layers_executed, 0);
    EXPECT_EQ(stats2.cross_domain_transfers, 0);

    // Second run
    executor->executeLayerRange(0, 5, graph_.get());
    auto stats3 = executor->getStats();
    EXPECT_EQ(stats3.cpu_layers_executed, 4);
    EXPECT_EQ(stats3.gpu_layers_executed, 1);
}

// =============================================================================
// Consistency Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, PlacementAndExecutorConsistency)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(5, 12, DeviceId::cuda(0));
    auto executor = createExecutor(placement);

    // Verify executor delegates correctly to placement
    for (int i = 0; i < 12; ++i)
    {
        DeviceId from_placement = placement.deviceForLayer(i);
        DeviceId from_executor = executor->getDeviceForLayer(i);

        EXPECT_EQ(from_placement, from_executor)
            << "Layer " << i << " device mismatch";

        // Also verify predicates match
        bool is_gpu_placement = placement.isGPULayer(i);
        bool is_gpu_executor = from_executor.is_gpu();
        EXPECT_EQ(is_gpu_placement, is_gpu_executor);
    }
}

TEST_F(Test__HeterogeneousIntegration, TransitionPointsMatchBoundaries)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    auto transitions = placement.transitionPoints();
    auto boundaries = placement.getDomainBoundaries();

    // Should have same count
    EXPECT_EQ(transitions.size(), boundaries.size());

    // Boundaries are the "to_layer" of each transition
    for (size_t i = 0; i < transitions.size(); ++i)
    {
        EXPECT_EQ(transitions[i].to_layer, boundaries[i]);
    }
}

// =============================================================================
// Profiling Tests
// =============================================================================

TEST_F(Test__HeterogeneousIntegration, ProfilingTracksTime)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));
    auto executor = createExecutor(placement, /*enable_profiling=*/true);

    // Execute with profiling
    executor->executeLayerRange(0, 10, graph_.get());

    auto stats = executor->getStats();

    // Times should be non-negative (may be 0 if execution is very fast)
    EXPECT_GE(stats.gpu_time_ms, 0.0);
    EXPECT_GE(stats.cpu_time_ms, 0.0);
    EXPECT_GE(stats.transfer_time_ms, 0.0);

    // Total should be sum of parts
    double expected_total = stats.gpu_time_ms + stats.cpu_time_ms + stats.transfer_time_ms;
    EXPECT_DOUBLE_EQ(stats.totalTimeMs(), expected_total);
}

TEST_F(Test__HeterogeneousIntegration, ProfilingDisabledZeroTimes)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto executor = createExecutor(placement, /*enable_profiling=*/false);

    // Execute without profiling
    executor->executeLayerRange(0, 10, graph_.get());

    auto stats = executor->getStats();

    // Counts should still work
    EXPECT_EQ(stats.gpu_layers_executed, 10);

    // Times should be zero when profiling disabled
    EXPECT_DOUBLE_EQ(stats.gpu_time_ms, 0.0);
    EXPECT_DOUBLE_EQ(stats.cpu_time_ms, 0.0);
}
