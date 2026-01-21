/**
 * @file Test__LayerPlacementConfigExtended.cpp
 * @brief Unit tests for extended LayerPlacementConfig methods
 *
 * Tests the new heterogeneous integration methods:
 * - getDeviceTypeForLayer() - Get DeviceType for a layer
 * - getDeviceIdForLayer() - Alias for deviceForLayer()
 * - isGPULayer() / isCPULayer() - Layer device predicates
 * - getGPULayers() / getCPULayers() - Filtered layer lists
 * - getDomainBoundaries() - Layer indices where device changes
 *
 * Part of Phase 5.4: HeterogeneousLayerExecutor + LayerPlacementConfig Integration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/LayerPlacementConfig.h"

using namespace llaminar2;

// =============================================================================
// getDeviceTypeForLayer() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, GetDeviceTypeForGPULayer)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    // All layers should return CUDA type
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(config.getDeviceTypeForLayer(i), DeviceType::CUDA)
            << "Layer " << i << " should be CUDA";
    }
}

TEST(Test__LayerPlacementConfigExtended, GetDeviceTypeForCPULayer)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    // All layers should return CPU type
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(config.getDeviceTypeForLayer(i), DeviceType::CPU)
            << "Layer " << i << " should be CPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, GetDeviceTypeForMixedLayers)
{
    // First 4 on CPU, rest on GPU
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    // First 4 should be CPU
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(config.getDeviceTypeForLayer(i), DeviceType::CPU)
            << "Layer " << i << " should be CPU";
    }

    // Rest should be CUDA
    for (int i = 4; i < 10; ++i)
    {
        EXPECT_EQ(config.getDeviceTypeForLayer(i), DeviceType::CUDA)
            << "Layer " << i << " should be CUDA";
    }
}

TEST(Test__LayerPlacementConfigExtended, GetDeviceTypeForROCmLayer)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::rocm(0), 5);

    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(config.getDeviceTypeForLayer(i), DeviceType::ROCm)
            << "Layer " << i << " should be ROCm";
    }
}

TEST(Test__LayerPlacementConfigExtended, GetDeviceTypeThrowsOnInvalidLayer)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    EXPECT_THROW(config.getDeviceTypeForLayer(-1), std::out_of_range);
    EXPECT_THROW(config.getDeviceTypeForLayer(10), std::out_of_range);
    EXPECT_THROW(config.getDeviceTypeForLayer(100), std::out_of_range);
}

// =============================================================================
// getDeviceIdForLayer() Tests (alias for deviceForLayer)
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, GetDeviceIdForLayer)
{
    auto config = LayerPlacementConfig::cpuFirstLayers(3, 8, DeviceId::cuda(1));

    // Test CPU layers
    EXPECT_EQ(config.getDeviceIdForLayer(0), DeviceId::cpu());
    EXPECT_EQ(config.getDeviceIdForLayer(2), DeviceId::cpu());

    // Test GPU layers
    EXPECT_EQ(config.getDeviceIdForLayer(3), DeviceId::cuda(1));
    EXPECT_EQ(config.getDeviceIdForLayer(7), DeviceId::cuda(1));

    // Verify it's equivalent to deviceForLayer
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_EQ(config.getDeviceIdForLayer(i), config.deviceForLayer(i));
    }
}

// =============================================================================
// isGPULayer() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, IsGPULayerReturnsTrue)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(config.isGPULayer(i)) << "Layer " << i << " should be GPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsGPULayerReturnsFalse)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    for (int i = 0; i < 10; ++i)
    {
        EXPECT_FALSE(config.isGPULayer(i)) << "Layer " << i << " should not be GPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsGPULayerMixed)
{
    // Last 4 layers on CPU
    auto config = LayerPlacementConfig::cpuLastLayers(4, 10, DeviceId::cuda(0));

    // First 6 should be GPU
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_TRUE(config.isGPULayer(i)) << "Layer " << i << " should be GPU";
    }

    // Last 4 should not be GPU
    for (int i = 6; i < 10; ++i)
    {
        EXPECT_FALSE(config.isGPULayer(i)) << "Layer " << i << " should not be GPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsGPULayerIncludesROCm)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::rocm(0), 5);

    for (int i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(config.isGPULayer(i)) << "ROCm layer " << i << " should be GPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsGPULayerThrowsOnInvalid)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    EXPECT_THROW(config.isGPULayer(-1), std::out_of_range);
    EXPECT_THROW(config.isGPULayer(10), std::out_of_range);
}

// =============================================================================
// isCPULayer() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, IsCPULayerReturnsTrue)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(config.isCPULayer(i)) << "Layer " << i << " should be CPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsCPULayerReturnsFalse)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    for (int i = 0; i < 10; ++i)
    {
        EXPECT_FALSE(config.isCPULayer(i)) << "Layer " << i << " should not be CPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsCPULayerMixed)
{
    // First 4 layers on CPU
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    // First 4 should be CPU
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(config.isCPULayer(i)) << "Layer " << i << " should be CPU";
    }

    // Rest should not be CPU
    for (int i = 4; i < 10; ++i)
    {
        EXPECT_FALSE(config.isCPULayer(i)) << "Layer " << i << " should not be CPU";
    }
}

TEST(Test__LayerPlacementConfigExtended, IsCPULayerThrowsOnInvalid)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    EXPECT_THROW(config.isCPULayer(-1), std::out_of_range);
    EXPECT_THROW(config.isCPULayer(10), std::out_of_range);
}

// =============================================================================
// getGPULayers() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, GetGPULayersReturnsAll)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    auto gpu_layers = config.getGPULayers();

    EXPECT_EQ(gpu_layers.size(), 10);
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(gpu_layers[i], i);
    }
}

TEST(Test__LayerPlacementConfigExtended, GetGPULayersReturnsNone)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    auto gpu_layers = config.getGPULayers();

    EXPECT_EQ(gpu_layers.size(), 0);
}

TEST(Test__LayerPlacementConfigExtended, GetGPULayersReturnsPartial)
{
    // First 4 on CPU, rest on GPU
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    auto gpu_layers = config.getGPULayers();

    ASSERT_EQ(gpu_layers.size(), 6);
    EXPECT_EQ(gpu_layers, (std::vector<int>{4, 5, 6, 7, 8, 9}));
}

TEST(Test__LayerPlacementConfigExtended, GetGPULayersIncludesROCm)
{
    // Create mixed CUDA and ROCm config
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cuda(0), 0},
        {2, DeviceId::rocm(0), 0},
        {3, DeviceId::cuda(1), 0},
        {4, DeviceId::cpu(), 0}};
    auto config = LayerPlacementConfig::custom(assignments);

    auto gpu_layers = config.getGPULayers();

    ASSERT_EQ(gpu_layers.size(), 3);
    EXPECT_EQ(gpu_layers, (std::vector<int>{1, 2, 3}));
}

TEST(Test__LayerPlacementConfigExtended, GetGPULayersIsSorted)
{
    // Alternating pattern
    auto config = LayerPlacementConfig::alternating(10, DeviceId::cuda(0), DeviceId::cpu());

    auto gpu_layers = config.getGPULayers();

    // Even layers are GPU
    ASSERT_EQ(gpu_layers.size(), 5);
    EXPECT_EQ(gpu_layers, (std::vector<int>{0, 2, 4, 6, 8}));

    // Verify sorted
    for (size_t i = 1; i < gpu_layers.size(); ++i)
    {
        EXPECT_LT(gpu_layers[i - 1], gpu_layers[i]);
    }
}

// =============================================================================
// getCPULayers() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, GetCPULayersReturnsAll)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    auto cpu_layers = config.getCPULayers();

    EXPECT_EQ(cpu_layers.size(), 10);
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(cpu_layers[i], i);
    }
}

TEST(Test__LayerPlacementConfigExtended, GetCPULayersReturnsNone)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    auto cpu_layers = config.getCPULayers();

    EXPECT_EQ(cpu_layers.size(), 0);
}

TEST(Test__LayerPlacementConfigExtended, GetCPULayersReturnsPartial)
{
    // First 4 on CPU, rest on GPU
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    auto cpu_layers = config.getCPULayers();

    ASSERT_EQ(cpu_layers.size(), 4);
    EXPECT_EQ(cpu_layers, (std::vector<int>{0, 1, 2, 3}));
}

TEST(Test__LayerPlacementConfigExtended, GetCPULayersIsSorted)
{
    // Alternating pattern (CPU on odd layers)
    auto config = LayerPlacementConfig::alternating(10, DeviceId::cuda(0), DeviceId::cpu());

    auto cpu_layers = config.getCPULayers();

    // Odd layers are CPU
    ASSERT_EQ(cpu_layers.size(), 5);
    EXPECT_EQ(cpu_layers, (std::vector<int>{1, 3, 5, 7, 9}));

    // Verify sorted
    for (size_t i = 1; i < cpu_layers.size(); ++i)
    {
        EXPECT_LT(cpu_layers[i - 1], cpu_layers[i]);
    }
}

// =============================================================================
// getDomainBoundaries() Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesNone)
{
    // All on same device - no boundaries
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    auto boundaries = config.getDomainBoundaries();

    EXPECT_EQ(boundaries.size(), 0);
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesSingle)
{
    // CPU first 4, GPU rest - one boundary at layer 4
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    auto boundaries = config.getDomainBoundaries();

    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 4);
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesMultiple)
{
    // Alternating: boundaries at every layer transition
    auto config = LayerPlacementConfig::alternating(6, DeviceId::cpu(), DeviceId::cuda(0));

    auto boundaries = config.getDomainBoundaries();

    // Transitions: 0->1, 1->2, 2->3, 3->4, 4->5
    ASSERT_EQ(boundaries.size(), 5);
    EXPECT_EQ(boundaries, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesCPULastLayers)
{
    // GPU first 6, CPU last 4 - one boundary at layer 6
    auto config = LayerPlacementConfig::cpuLastLayers(4, 10, DeviceId::cuda(0));

    auto boundaries = config.getDomainBoundaries();

    ASSERT_EQ(boundaries.size(), 1);
    EXPECT_EQ(boundaries[0], 6);
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesMultiGPU)
{
    // Multiple GPUs: 0-2 on cuda:0, 3-5 on cuda:1, 6-9 on rocm:0
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 3; ++i)
        assignments.push_back({i, DeviceId::cuda(0), 0});
    for (int i = 3; i < 6; ++i)
        assignments.push_back({i, DeviceId::cuda(1), 0});
    for (int i = 6; i < 10; ++i)
        assignments.push_back({i, DeviceId::rocm(0), 0});

    auto config = LayerPlacementConfig::custom(assignments);

    auto boundaries = config.getDomainBoundaries();

    // Boundaries at layer 3 (cuda:0 -> cuda:1) and layer 6 (cuda:1 -> rocm:0)
    ASSERT_EQ(boundaries.size(), 2);
    EXPECT_EQ(boundaries[0], 3);
    EXPECT_EQ(boundaries[1], 6);
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesSingleLayer)
{
    // Single layer - no boundaries possible
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 1);

    auto boundaries = config.getDomainBoundaries();

    EXPECT_EQ(boundaries.size(), 0);
}

TEST(Test__LayerPlacementConfigExtended, GetDomainBoundariesEmptyConfig)
{
    // Empty config
    LayerPlacementConfig config;

    auto boundaries = config.getDomainBoundaries();

    EXPECT_EQ(boundaries.size(), 0);
}

// =============================================================================
// First/Last Layer Placement Tests
// =============================================================================

TEST(Test__LayerPlacementConfigExtended, FirstLastLayerPlacement)
{
    // First 2 on CPU, middle on GPU, last 2 on CPU (sandwich pattern)
    std::vector<LayerDeviceAssignment> assignments;
    assignments.push_back({0, DeviceId::cpu(), 0});
    assignments.push_back({1, DeviceId::cpu(), 0});
    for (int i = 2; i < 8; ++i)
    {
        assignments.push_back({i, DeviceId::cuda(0), 0});
    }
    assignments.push_back({8, DeviceId::cpu(), 0});
    assignments.push_back({9, DeviceId::cpu(), 0});

    auto config = LayerPlacementConfig::custom(assignments);

    // Verify first/last layers
    EXPECT_TRUE(config.isCPULayer(0));
    EXPECT_TRUE(config.isCPULayer(1));
    EXPECT_TRUE(config.isGPULayer(2));
    EXPECT_TRUE(config.isGPULayer(7));
    EXPECT_TRUE(config.isCPULayer(8));
    EXPECT_TRUE(config.isCPULayer(9));

    // Verify boundaries
    auto boundaries = config.getDomainBoundaries();
    ASSERT_EQ(boundaries.size(), 2);
    EXPECT_EQ(boundaries[0], 2); // CPU -> GPU
    EXPECT_EQ(boundaries[1], 8); // GPU -> CPU

    // Verify layer lists
    auto cpu_layers = config.getCPULayers();
    auto gpu_layers = config.getGPULayers();

    EXPECT_EQ(cpu_layers, (std::vector<int>{0, 1, 8, 9}));
    EXPECT_EQ(gpu_layers, (std::vector<int>{2, 3, 4, 5, 6, 7}));
}

TEST(Test__LayerPlacementConfigExtended, ConsistencyBetweenPredicatesAndLists)
{
    auto config = LayerPlacementConfig::cpuFirstLayers(5, 12, DeviceId::cuda(0));

    auto cpu_layers = config.getCPULayers();
    auto gpu_layers = config.getGPULayers();

    // Verify every layer in CPU list returns true for isCPULayer
    for (int layer : cpu_layers)
    {
        EXPECT_TRUE(config.isCPULayer(layer));
        EXPECT_FALSE(config.isGPULayer(layer));
    }

    // Verify every layer in GPU list returns true for isGPULayer
    for (int layer : gpu_layers)
    {
        EXPECT_TRUE(config.isGPULayer(layer));
        EXPECT_FALSE(config.isCPULayer(layer));
    }

    // Total should equal numLayers
    EXPECT_EQ(cpu_layers.size() + gpu_layers.size(), 12);
}
