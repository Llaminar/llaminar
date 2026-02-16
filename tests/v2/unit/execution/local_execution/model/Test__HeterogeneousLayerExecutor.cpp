/**
 * @file Test__HeterogeneousLayerExecutor.cpp
 * @brief Unit tests for HeterogeneousLayerExecutor
 *
 * Tests heterogeneous layer execution with mock placement configurations
 * and verifies device routing, statistics tracking, and cross-domain
 * transfer detection.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/model/HeterogeneousLayerExecutor.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "config/LayerPlacementConfig.h"
#include "config/TPDomain.h"

using namespace llaminar2;

// =============================================================================
// Construction Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, ConstructsWithConfig)
{
    // Create a simple placement config
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    // Should construct without throwing
    EXPECT_NO_THROW({
        HeterogeneousLayerExecutor executor(config);
    });
}

TEST(Test__HeterogeneousLayerExecutor, ThrowsOnNullPlacementConfig)
{
    HeterogeneousLayerExecutor::Config config;
    config.placement_config = nullptr; // Required field not set

    // Should throw due to null placement_config
    EXPECT_THROW({ HeterogeneousLayerExecutor executor(config); }, std::invalid_argument);
}

TEST(Test__HeterogeneousLayerExecutor, ConfigAccessible)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor(config);

    EXPECT_EQ(executor.config().placement_config, &placement);
    EXPECT_TRUE(executor.config().enable_profiling);
}

// =============================================================================
// Device Lookup Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, GetDeviceForLayerDelegatesToPlacement)
{
    // Create a mixed placement: first 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // First 4 layers should be on CPU
    for (int i = 0; i < 4; ++i)
    {
        DeviceId device = executor.getDeviceForLayer(i);
        EXPECT_TRUE(device.is_cpu()) << "Layer " << i << " should be on CPU";
    }

    // Rest should be on GPU
    for (int i = 4; i < 10; ++i)
    {
        DeviceId device = executor.getDeviceForLayer(i);
        EXPECT_TRUE(device.is_gpu()) << "Layer " << i << " should be on GPU";
        EXPECT_EQ(device, DeviceId::cuda(0));
    }
}

TEST(Test__HeterogeneousLayerExecutor, GetDeviceForLayerThrowsOnInvalidIndex)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // Out of range should throw
    EXPECT_THROW({ executor.getDeviceForLayer(100); }, std::out_of_range);

    EXPECT_THROW({ executor.getDeviceForLayer(-1); }, std::out_of_range);
}

// =============================================================================
// Domain Lookup Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, GetDomainForLayerReturnsNullWithNoTPConfig)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.tp_config = nullptr; // No TP config

    HeterogeneousLayerExecutor executor(config);

    // Should return nullptr when no TP config
    EXPECT_EQ(executor.getDomainForLayer(0), nullptr);
    EXPECT_EQ(executor.getDomainForLayer(5), nullptr);
}

TEST(Test__HeterogeneousLayerExecutor, GetDomainForLayerDelegatesToTPConfig)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    // Create a test TP config with a GPU domain
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    gpu_domain.name = "test_gpu_domain";

    auto tp_config = MultiDomainTPConfig::createForTest({gpu_domain});

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.tp_config = &tp_config;

    HeterogeneousLayerExecutor executor(config);

    // Should return a domain pointer (may be the gpu domain)
    const TPDomain *domain = executor.getDomainForLayer(0);
    // Note: actual result depends on MultiDomainTPConfig::domainForLayer implementation
    // For now we just verify no crash occurs
}

// =============================================================================
// Cross-Domain Transfer Detection Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, RequiresCrossDomainTransferDetectsChange)
{
    // Layers 0-3 on CPU, 4-9 on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // No transfer needed within CPU layers
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(0, 1));
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(1, 2));
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(2, 3));

    // Transfer needed at CPU->GPU boundary
    EXPECT_TRUE(executor.requiresCrossDomainTransfer(3, 4));

    // No transfer needed within GPU layers
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(4, 5));
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(5, 6));
}

TEST(Test__HeterogeneousLayerExecutor, RequiresCrossDomainTransferSameDevice)
{
    // All layers on same device
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // No transfers needed when all on same device
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_FALSE(executor.requiresCrossDomainTransfer(i, i + 1))
            << "No transfer should be needed between layers " << i << " and " << (i + 1);
    }
}

TEST(Test__HeterogeneousLayerExecutor, RequiresCrossDomainTransferAlternating)
{
    // Alternating CPU/GPU pattern
    auto placement = LayerPlacementConfig::alternating(10, DeviceId::cpu(), DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // Every transition should require transfer (alternating pattern)
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_TRUE(executor.requiresCrossDomainTransfer(i, i + 1))
            << "Transfer should be needed between layers " << i << " and " << (i + 1);
    }
}

TEST(Test__HeterogeneousLayerExecutor, RequiresCrossDomainTransferOutOfBounds)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // Out of bounds should return false (safe behavior)
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(-1, 0));
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(9, 100));
    EXPECT_FALSE(executor.requiresCrossDomainTransfer(-5, -3));
}

// =============================================================================
// Execute Layer Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerOnGPU)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor(config);

    // Create a minimal graph
    ComputeGraph graph;

    // Execute layer 0 (should go to GPU)
    bool success = executor.executeLayer(0, &graph);
    EXPECT_TRUE(success);

    // Check stats
    auto stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 1);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
}

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerOnCPU)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor(config);

    // Create a minimal graph
    ComputeGraph graph;

    // Execute layer 0 (should go to CPU)
    bool success = executor.executeLayer(0, &graph);
    EXPECT_TRUE(success);

    // Check stats
    auto stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 1);
}

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerWithNullGraph)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // Null graph should return false
    bool success = executor.executeLayer(0, nullptr);
    EXPECT_FALSE(success);
}

// =============================================================================
// Execute Layer Range Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerRangeHandlesBoundaries)
{
    // First 4 on CPU, rest on GPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Execute layers 0-6 (crosses CPU->GPU boundary at layer 4)
    bool success = executor.executeLayerRange(0, 7, &graph);
    EXPECT_TRUE(success);

    // Check stats
    auto stats = executor.getStats();
    EXPECT_EQ(stats.cpu_layers_executed, 4);    // Layers 0-3
    EXPECT_EQ(stats.gpu_layers_executed, 3);    // Layers 4-6
    EXPECT_EQ(stats.cross_domain_transfers, 1); // One transfer at boundary
}

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerRangeEmptyRange)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Empty range should succeed with no-op
    bool success = executor.executeLayerRange(5, 5, &graph);
    EXPECT_TRUE(success);

    // Inverted range should also succeed with no-op
    success = executor.executeLayerRange(7, 3, &graph);
    EXPECT_TRUE(success);

    // Stats should be zero
    auto stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
}

TEST(Test__HeterogeneousLayerExecutor, ExecuteLayerRangeNullGraph)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    // Null graph should return false
    bool success = executor.executeLayerRange(0, 5, nullptr);
    EXPECT_FALSE(success);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, StatsTrackGPULayers)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Execute 5 GPU layers
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(executor.executeLayer(i, &graph));
    }

    auto stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 5);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
}

TEST(Test__HeterogeneousLayerExecutor, StatsTrackCPULayers)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Execute 3 CPU layers
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(executor.executeLayer(i, &graph));
    }

    auto stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 3);
}

TEST(Test__HeterogeneousLayerExecutor, StatsTrackTransfers)
{
    // Alternating pattern: every layer transition needs a transfer
    auto placement = LayerPlacementConfig::alternating(10, DeviceId::cpu(), DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Execute layers 0-5 (should have 4 transfers)
    EXPECT_TRUE(executor.executeLayerRange(0, 5, &graph));

    auto stats = executor.getStats();
    EXPECT_EQ(stats.cross_domain_transfers, 4); // 4 boundaries in 5 layers
}

TEST(Test__HeterogeneousLayerExecutor, ResetStatsClearsAll)
{
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor(config);

    ComputeGraph graph;

    // Execute some layers to accumulate stats
    executor.executeLayerRange(0, 6, &graph);

    // Verify stats are non-zero
    auto stats = executor.getStats();
    EXPECT_GT(stats.cpu_layers_executed + stats.gpu_layers_executed, 0);

    // Reset
    executor.resetStats();

    // Verify all stats are cleared
    stats = executor.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 0);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
    EXPECT_EQ(stats.cross_domain_transfers, 0);
    EXPECT_EQ(stats.gpu_time_ms, 0.0);
    EXPECT_EQ(stats.cpu_time_ms, 0.0);
    EXPECT_EQ(stats.transfer_time_ms, 0.0);
}

// =============================================================================
// ExecutionStats Helper Methods Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, StatsTotalTimeMs)
{
    HeterogeneousLayerExecutor::ExecutionStats stats;
    stats.gpu_time_ms = 10.0;
    stats.cpu_time_ms = 5.0;
    stats.transfer_time_ms = 2.0;

    EXPECT_DOUBLE_EQ(stats.totalTimeMs(), 17.0);
}

TEST(Test__HeterogeneousLayerExecutor, StatsTimePercentages)
{
    HeterogeneousLayerExecutor::ExecutionStats stats;
    stats.gpu_time_ms = 80.0;
    stats.cpu_time_ms = 10.0;
    stats.transfer_time_ms = 10.0;

    EXPECT_DOUBLE_EQ(stats.gpuTimePercent(), 80.0);
    EXPECT_DOUBLE_EQ(stats.cpuTimePercent(), 10.0);
    EXPECT_DOUBLE_EQ(stats.transferTimePercent(), 10.0);
}

TEST(Test__HeterogeneousLayerExecutor, StatsTimePercentagesZeroTotal)
{
    HeterogeneousLayerExecutor::ExecutionStats stats;
    // All zeros

    // Should return 0 when total is 0 (no division by zero)
    EXPECT_DOUBLE_EQ(stats.gpuTimePercent(), 0.0);
    EXPECT_DOUBLE_EQ(stats.cpuTimePercent(), 0.0);
    EXPECT_DOUBLE_EQ(stats.transferTimePercent(), 0.0);
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(Test__HeterogeneousLayerExecutor, MoveConstruction)
{
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    HeterogeneousLayerExecutor::Config config;
    config.placement_config = &placement;
    config.enable_profiling = true;

    HeterogeneousLayerExecutor executor1(config);

    ComputeGraph graph;
    executor1.executeLayer(0, &graph);

    // Move construct
    HeterogeneousLayerExecutor executor2(std::move(executor1));

    // executor2 should work
    EXPECT_TRUE(executor2.executeLayer(1, &graph));

    auto stats = executor2.getStats();
    // Stats may or may not be preserved depending on implementation
    // At minimum, execution should work
    EXPECT_GE(stats.gpu_layers_executed, 1);
}

TEST(Test__HeterogeneousLayerExecutor, MoveAssignment)
{
    auto placement1 = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto placement2 = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 5);

    HeterogeneousLayerExecutor::Config config1;
    config1.placement_config = &placement1;

    HeterogeneousLayerExecutor::Config config2;
    config2.placement_config = &placement2;

    HeterogeneousLayerExecutor executor1(config1);
    HeterogeneousLayerExecutor executor2(config2);

    // Move assign
    executor2 = std::move(executor1);

    // executor2 should now use placement1
    ComputeGraph graph;
    EXPECT_TRUE(executor2.executeLayer(0, &graph));

    // Should be on GPU (from placement1)
    auto stats = executor2.getStats();
    EXPECT_EQ(stats.gpu_layers_executed, 1);
    EXPECT_EQ(stats.cpu_layers_executed, 0);
}
