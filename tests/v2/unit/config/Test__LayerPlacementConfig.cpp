/**
 * @file Test__LayerPlacementConfig.cpp
 * @brief Unit tests for LayerPlacementConfig
 *
 * Tests layer-to-device placement configuration for CPU pipeline stage support.
 */

#include <gtest/gtest.h>
#include "config/LayerPlacementConfig.h"

using namespace llaminar2;

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, AllOnGPU)
{
    // Standard pattern: all 28 layers on CUDA:0
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 28);

    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_FALSE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 1);

    // Verify all layers are on CUDA:0
    for (int i = 0; i < 28; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cuda(0));
    }

    // No transition points when all on same device
    auto transitions = config.transitionPoints();
    EXPECT_EQ(transitions.size(), 0);
}

TEST(Test__LayerPlacementConfig, AllOnCPU)
{
    // All 28 layers on CPU
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 28);

    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_FALSE(config.hasLayersOnGPU());
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 1);

    // Verify all layers are on CPU
    for (int i = 0; i < 28; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cpu());
    }
}

TEST(Test__LayerPlacementConfig, CPUFirstFour)
{
    // Layers 0-3 on CPU, 4-27 on GPU
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));

    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 2);

    // First 4 layers on CPU
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cpu())
            << "Layer " << i << " should be on CPU";
    }

    // Rest on GPU
    for (int i = 4; i < 28; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cuda(0))
            << "Layer " << i << " should be on CUDA:0";
    }
}

TEST(Test__LayerPlacementConfig, CPULastTwo)
{
    // Layers 0-25 on GPU, 26-27 on CPU
    auto config = LayerPlacementConfig::cpuLastLayers(2, 28, DeviceId::cuda(0));

    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 2);

    // First 26 layers on GPU
    for (int i = 0; i < 26; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cuda(0))
            << "Layer " << i << " should be on CUDA:0";
    }

    // Last 2 on CPU
    for (int i = 26; i < 28; ++i)
    {
        EXPECT_EQ(config.deviceForLayer(i), DeviceId::cpu())
            << "Layer " << i << " should be on CPU";
    }
}

TEST(Test__LayerPlacementConfig, Alternating)
{
    // Even layers on CPU, odd on GPU
    auto config = LayerPlacementConfig::alternating(10, DeviceId::cpu(), DeviceId::cuda(0));

    EXPECT_EQ(config.numLayers(), 10);
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 2);

    for (int i = 0; i < 10; ++i)
    {
        if (i % 2 == 0)
        {
            EXPECT_EQ(config.deviceForLayer(i), DeviceId::cpu())
                << "Layer " << i << " (even) should be on CPU";
        }
        else
        {
            EXPECT_EQ(config.deviceForLayer(i), DeviceId::cuda(0))
                << "Layer " << i << " (odd) should be on CUDA:0";
        }
    }
}

TEST(Test__LayerPlacementConfig, Custom_Mixed)
{
    // Custom arbitrary assignment
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cpu(), 0},
        {2, DeviceId::cuda(0), 0},
        {3, DeviceId::cuda(0), 0},
        {4, DeviceId::rocm(0), 0},
        {5, DeviceId::cpu(), 0}};

    auto config = LayerPlacementConfig::custom(assignments);

    EXPECT_EQ(config.numLayers(), 6);
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_EQ(config.deviceCount(), 3); // CPU, CUDA:0, ROCm:0

    EXPECT_EQ(config.deviceForLayer(0), DeviceId::cpu());
    EXPECT_EQ(config.deviceForLayer(1), DeviceId::cpu());
    EXPECT_EQ(config.deviceForLayer(2), DeviceId::cuda(0));
    EXPECT_EQ(config.deviceForLayer(3), DeviceId::cuda(0));
    EXPECT_EQ(config.deviceForLayer(4), DeviceId::rocm(0));
    EXPECT_EQ(config.deviceForLayer(5), DeviceId::cpu());
}

// =============================================================================
// Lookup Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, DeviceForLayer_Valid)
{
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    // Valid lookups
    EXPECT_EQ(config.deviceForLayer(0), DeviceId::cpu());
    EXPECT_EQ(config.deviceForLayer(3), DeviceId::cpu());
    EXPECT_EQ(config.deviceForLayer(4), DeviceId::cuda(0));
    EXPECT_EQ(config.deviceForLayer(9), DeviceId::cuda(0));
}

TEST(Test__LayerPlacementConfig, DeviceForLayer_Invalid)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);

    // Out of range should throw
    EXPECT_THROW(config.deviceForLayer(-1), std::out_of_range);
    EXPECT_THROW(config.deviceForLayer(10), std::out_of_range);
    EXPECT_THROW(config.deviceForLayer(100), std::out_of_range);
}

TEST(Test__LayerPlacementConfig, LayersForDevice_Filter)
{
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 10, DeviceId::cuda(0));

    // CPU layers
    auto cpu_layers = config.layersForDevice(DeviceId::cpu());
    EXPECT_EQ(cpu_layers.size(), 4);
    EXPECT_EQ(cpu_layers, (std::vector<int>{0, 1, 2, 3}));

    // GPU layers
    auto gpu_layers = config.layersForDevice(DeviceId::cuda(0));
    EXPECT_EQ(gpu_layers.size(), 6);
    EXPECT_EQ(gpu_layers, (std::vector<int>{4, 5, 6, 7, 8, 9}));

    // Non-existent device
    auto rocm_layers = config.layersForDevice(DeviceId::rocm(0));
    EXPECT_EQ(rocm_layers.size(), 0);
}

// =============================================================================
// Transition Point Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, TransitionPoints_CPUFirst)
{
    // CPU layers 0-3, GPU layers 4-27
    // Should have exactly one transition: layer 3 → layer 4
    auto config = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));

    auto transitions = config.transitionPoints();
    ASSERT_EQ(transitions.size(), 1);

    EXPECT_EQ(transitions[0].from_layer, 3);
    EXPECT_EQ(transitions[0].to_layer, 4);
    EXPECT_EQ(transitions[0].from_device, DeviceId::cpu());
    EXPECT_EQ(transitions[0].to_device, DeviceId::cuda(0));
}

TEST(Test__LayerPlacementConfig, TransitionPoints_CPULast)
{
    // GPU layers 0-25, CPU layers 26-27
    // Should have exactly one transition: layer 25 → layer 26
    auto config = LayerPlacementConfig::cpuLastLayers(2, 28, DeviceId::cuda(0));

    auto transitions = config.transitionPoints();
    ASSERT_EQ(transitions.size(), 1);

    EXPECT_EQ(transitions[0].from_layer, 25);
    EXPECT_EQ(transitions[0].to_layer, 26);
    EXPECT_EQ(transitions[0].from_device, DeviceId::cuda(0));
    EXPECT_EQ(transitions[0].to_device, DeviceId::cpu());
}

TEST(Test__LayerPlacementConfig, TransitionPoints_Multiple)
{
    // Alternating: many transitions
    auto config = LayerPlacementConfig::alternating(6, DeviceId::cpu(), DeviceId::cuda(0));

    auto transitions = config.transitionPoints();
    // Transitions at: 0→1, 1→2, 2→3, 3→4, 4→5 = 5 transitions
    EXPECT_EQ(transitions.size(), 5);

    // Verify first transition
    EXPECT_EQ(transitions[0].from_layer, 0);
    EXPECT_EQ(transitions[0].to_layer, 1);
    EXPECT_EQ(transitions[0].from_device, DeviceId::cpu());
    EXPECT_EQ(transitions[0].to_device, DeviceId::cuda(0));

    // Verify second transition (back to CPU)
    EXPECT_EQ(transitions[1].from_layer, 1);
    EXPECT_EQ(transitions[1].to_layer, 2);
    EXPECT_EQ(transitions[1].from_device, DeviceId::cuda(0));
    EXPECT_EQ(transitions[1].to_device, DeviceId::cpu());
}

TEST(Test__LayerPlacementConfig, TransitionPoints_NoTransitions)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);

    auto transitions = config.transitionPoints();
    EXPECT_EQ(transitions.size(), 0);
}

TEST(Test__LayerPlacementConfig, TransitionPoints_SingleLayer)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 1);

    auto transitions = config.transitionPoints();
    EXPECT_EQ(transitions.size(), 0);
}

// =============================================================================
// Device Detection Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, HasLayersOnCPU_True)
{
    auto config = LayerPlacementConfig::cpuFirstLayers(1, 10, DeviceId::cuda(0));
    EXPECT_TRUE(config.hasLayersOnCPU());

    auto config2 = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    EXPECT_TRUE(config2.hasLayersOnCPU());
}

TEST(Test__LayerPlacementConfig, HasLayersOnCPU_False)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    EXPECT_FALSE(config.hasLayersOnCPU());

    auto config2 = LayerPlacementConfig::allOnDevice(DeviceId::rocm(0), 10);
    EXPECT_FALSE(config2.hasLayersOnCPU());
}

TEST(Test__LayerPlacementConfig, HasLayersOnGPU_True)
{
    auto config = LayerPlacementConfig::cpuLastLayers(1, 10, DeviceId::cuda(0));
    EXPECT_TRUE(config.hasLayersOnGPU());
}

TEST(Test__LayerPlacementConfig, HasLayersOnGPU_False)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 10);
    EXPECT_FALSE(config.hasLayersOnGPU());
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, Validate_Valid)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 28);
    EXPECT_TRUE(config.validate(28));
    EXPECT_TRUE(config.validationError().empty());
}

TEST(Test__LayerPlacementConfig, Validate_MissingLayer)
{
    // Create config with missing layer 5
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 10; ++i)
    {
        if (i != 5)
        { // Skip layer 5
            assignments.push_back({i, DeviceId::cpu(), 0});
        }
    }

    auto config = LayerPlacementConfig::custom(assignments);
    EXPECT_FALSE(config.validate(10));
    EXPECT_FALSE(config.validationError().empty());
    // Validation fails because count is wrong (9 vs 10 expected)
    EXPECT_NE(config.validationError().find("Expected"), std::string::npos);
}

TEST(Test__LayerPlacementConfig, Validate_Gap)
{
    // Create config with gap in layer indices: 0,1,2,3,5,6,7,8,9 (missing 4)
    // But pass 9 as expected count so count check passes, then gap check catches it
    std::vector<LayerDeviceAssignment> assignments;
    for (int i = 0; i < 10; ++i)
    {
        if (i != 4)
        { // Skip layer 4
            assignments.push_back({i, DeviceId::cpu(), 0});
        }
    }

    auto config = LayerPlacementConfig::custom(assignments);
    EXPECT_FALSE(config.validate(9)); // Pass 9 so count check passes
    EXPECT_FALSE(config.validationError().empty());
    // Now it should report missing layer 4
    EXPECT_NE(config.validationError().find("Missing layer"), std::string::npos);
}

TEST(Test__LayerPlacementConfig, Validate_DuplicateLayer)
{
    // Create config with duplicate layer 3
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cpu(), 0},
        {2, DeviceId::cpu(), 0},
        {3, DeviceId::cpu(), 0},
        {3, DeviceId::cuda(0), 0}, // Duplicate!
    };

    auto config = LayerPlacementConfig::custom(assignments);
    EXPECT_FALSE(config.validate(5));
    // Should fail because we have 5 assignments but layer_indices set has only 4
}

TEST(Test__LayerPlacementConfig, Validate_WrongCount)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    EXPECT_FALSE(config.validate(28)); // Expected 28, got 10
    EXPECT_NE(config.validationError().find("Expected"), std::string::npos);
}

TEST(Test__LayerPlacementConfig, Validate_OutOfRange)
{
    // Create config with out-of-range layer index
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cpu(), 0},
        {99, DeviceId::cpu(), 0}, // Out of range for 5-layer model
    };

    auto config = LayerPlacementConfig::custom(assignments);
    EXPECT_FALSE(config.validate(5));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(Test__LayerPlacementConfig, EmptyConfig)
{
    LayerPlacementConfig config;
    EXPECT_EQ(config.numLayers(), 0);
    EXPECT_FALSE(config.hasLayersOnCPU());
    EXPECT_FALSE(config.hasLayersOnGPU());
    EXPECT_EQ(config.deviceCount(), 0);
    EXPECT_EQ(config.transitionPoints().size(), 0);
}

TEST(Test__LayerPlacementConfig, SingleLayerCPU)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cpu(), 1);
    EXPECT_EQ(config.numLayers(), 1);
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_FALSE(config.hasLayersOnGPU());
    EXPECT_EQ(config.deviceForLayer(0), DeviceId::cpu());
    EXPECT_TRUE(config.validate(1));
}

TEST(Test__LayerPlacementConfig, AllCPULayers)
{
    // Edge case: all layers on CPU (total offload)
    auto config = LayerPlacementConfig::cpuFirstLayers(28, 28, DeviceId::cuda(0));
    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_TRUE(config.hasLayersOnCPU());
    EXPECT_FALSE(config.hasLayersOnGPU());
    EXPECT_EQ(config.transitionPoints().size(), 0);
}

TEST(Test__LayerPlacementConfig, AllGPULayersCPULast)
{
    // Edge case: 0 CPU layers means all on GPU
    auto config = LayerPlacementConfig::cpuLastLayers(0, 28, DeviceId::cuda(0));
    EXPECT_EQ(config.numLayers(), 28);
    EXPECT_FALSE(config.hasLayersOnCPU());
    EXPECT_TRUE(config.hasLayersOnGPU());
    EXPECT_EQ(config.transitionPoints().size(), 0);
}

// =============================================================================
// Device Inventory Tests
// =============================================================================

TEST(Test__LayerPlacementConfig, UniqueDevices_Single)
{
    auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 10);
    auto devices = config.uniqueDevices();
    ASSERT_EQ(devices.size(), 1);
    EXPECT_EQ(devices[0], DeviceId::cuda(0));
}

TEST(Test__LayerPlacementConfig, UniqueDevices_Multiple)
{
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 0},
        {1, DeviceId::cuda(0), 0},
        {2, DeviceId::rocm(0), 0},
        {3, DeviceId::cuda(1), 0},
    };

    auto config = LayerPlacementConfig::custom(assignments);
    auto devices = config.uniqueDevices();

    EXPECT_EQ(devices.size(), 4);
    EXPECT_EQ(config.deviceCount(), 4);
}

// =============================================================================
// Priority Tests (for future scheduling)
// =============================================================================

TEST(Test__LayerPlacementConfig, AssignmentWithPriority)
{
    std::vector<LayerDeviceAssignment> assignments = {
        {0, DeviceId::cpu(), 10},  // High priority
        {1, DeviceId::cuda(0), 5}, // Medium priority
        {2, DeviceId::cuda(0), 0}, // Default priority
    };

    auto config = LayerPlacementConfig::custom(assignments);

    // Verify assignments preserve priority
    const auto &stored = config.assignments();
    ASSERT_EQ(stored.size(), 3);
    EXPECT_EQ(stored[0].priority, 10);
    EXPECT_EQ(stored[1].priority, 5);
    EXPECT_EQ(stored[2].priority, 0);
}

// =============================================================================
// Factory Method Edge Cases
// =============================================================================

TEST(Test__LayerPlacementConfig, CpuFirstLayers_InvalidRange)
{
    EXPECT_THROW(
        LayerPlacementConfig::cpuFirstLayers(-1, 10, DeviceId::cuda(0)),
        std::invalid_argument);

    EXPECT_THROW(
        LayerPlacementConfig::cpuFirstLayers(11, 10, DeviceId::cuda(0)),
        std::invalid_argument);
}

TEST(Test__LayerPlacementConfig, CpuLastLayers_InvalidRange)
{
    EXPECT_THROW(
        LayerPlacementConfig::cpuLastLayers(-1, 10, DeviceId::cuda(0)),
        std::invalid_argument);

    EXPECT_THROW(
        LayerPlacementConfig::cpuLastLayers(11, 10, DeviceId::cuda(0)),
        std::invalid_argument);
}
