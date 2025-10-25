/**
 * @file Test__DeviceAwareExecution.cpp
 * @brief Phase 4.3 tests - Device-aware execution with weight placement
 *
 * Tests that pipelines correctly:
 * - Query weight device placement
 * - Transfer activations to target device
 * - Execute operations on the correct device
 * - Track device index through execution
 *
 * @author David Sanftenberg
 * @date 2025-01-20
 */

#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/loaders/WeightPlacementMap.h"

using namespace llaminar2;

/**
 * @brief Test Phase 4.3: Device-aware tensor operations
 *
 * These tests verify the infrastructure for device-aware execution without
 * requiring full pipeline integration (which would need GGUF models).
 */

/**
 * @test Verify device transfer preserves data correctly (CPU only for now)
 */
TEST(Test__DeviceAwareExecution, DeviceTransferCorrectness_CPU)
{
    // Create two FP32 tensors on CPU (device -1)
    std::vector<size_t> shape = {4, 8}; // 4 tokens, 8 features
    auto tensor_src = std::make_unique<FP32Tensor>(shape, -1);
    auto tensor_dst = std::make_unique<FP32Tensor>(shape, -1);

    // Fill source with test data
    float *src_data = tensor_src->mutable_data();
    for (size_t i = 0; i < 4 * 8; ++i)
    {
        src_data[i] = static_cast<float>(i) * 0.1f;
    }

    // Transfer from CPU to CPU
    bool transfer_success = tensor_dst->copyFrom(tensor_src.get());
    EXPECT_TRUE(transfer_success) << "CPU → CPU transfer should succeed";

    // Verify data was copied correctly
    const float *dst_data = tensor_dst->data();
    for (size_t i = 0; i < 4 * 8; ++i)
    {
        EXPECT_FLOAT_EQ(src_data[i], dst_data[i])
            << "Data mismatch at index " << i << " after transfer";
    }

    // Verify device indices
    EXPECT_EQ(tensor_src->device_index(), -1) << "Source tensor should remain on CPU";
    EXPECT_EQ(tensor_dst->device_index(), -1) << "Dest tensor should be on CPU";
}

/**
 * @test Verify WeightPlacementMap interface
 */
TEST(Test__DeviceAwareExecution, WeightPlacementMapInterface)
{
    auto placement_map = std::make_shared<WeightPlacementMap>();

    // Set some weight placements
    placement_map->setTensorDevice("layer.0.attn_q", 0);
    placement_map->setTensorDevice("layer.0.attn_k", 0);
    placement_map->setTensorDevice("layer.1.attn_q", 1);

    // Query placements
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.0.attn_q"), 0)
        << "layer.0.attn_q should be on device 0";
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.1.attn_q"), 1)
        << "layer.1.attn_q should be on device 1";

    // Query non-existent weight (should return default device 0)
    EXPECT_EQ(placement_map->getDeviceForWeight("nonexistent"), 0)
        << "Non-existent weight should return default device 0";
}

/**
 * @test Verify device index tracking
 */
TEST(Test__DeviceAwareExecution, DeviceIndexTracking)
{
    auto cpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4}, -1);
    EXPECT_EQ(cpu_tensor->device_index(), -1) << "CPU tensor should have device_idx = -1";

    // Note: set_device() is not fully implemented yet (GPU backends are stubs)
    // This test just documents the interface
    SUCCEED() << "Device index tracking interface documented";
}

/**
 * @test Verify no-op transfer (same device) - CPU
 */
TEST(Test__DeviceAwareExecution, SameDeviceTransfer_CPU)
{
    std::vector<size_t> shape = {2, 4};
    auto src = std::make_unique<FP32Tensor>(shape, -1); // CPU
    auto dst = std::make_unique<FP32Tensor>(shape, -1); // CPU

    // Fill source
    float *src_data = src->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        src_data[i] = static_cast<float>(i);
    }

    // Transfer between same device (should succeed)
    bool success = dst->copyFrom(src.get());
    EXPECT_TRUE(success) << "Same-device transfer should succeed";

    // Verify data copied
    const float *dst_data = dst->data();
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i));
    }
}

/**
 * @test Verify transfer with different data values (CPU)
 */
TEST(Test__DeviceAwareExecution, TransferDataIntegrity_CPU)
{
    std::vector<size_t> shape = {3, 5};                 // Non-power-of-2 dimensions
    auto src = std::make_unique<FP32Tensor>(shape, -1); // CPU
    auto dst = std::make_unique<FP32Tensor>(shape, -1); // CPU

    // Fill with pattern: sin(i/10.0)
    float *src_data = src->mutable_data();
    for (size_t i = 0; i < 15; ++i)
    {
        src_data[i] = std::sin(static_cast<float>(i) / 10.0f);
    }

    // Transfer
    bool success = dst->copyFrom(src.get());
    EXPECT_TRUE(success);

    // Verify exact match
    const float *dst_data = dst->data();
    for (size_t i = 0; i < 15; ++i)
    {
        EXPECT_FLOAT_EQ(dst_data[i], std::sin(static_cast<float>(i) / 10.0f))
            << "Data mismatch at index " << i;
    }
}

/**
 * @test Verify WeightPlacementMap strategy application
 */
TEST(Test__DeviceAwareExecution, PlacementStrategyBasic)
{
    auto placement_map = std::make_shared<WeightPlacementMap>();

    // Simulate a simple placement strategy: even layers on device 0, odd on device 1
    for (int layer = 0; layer < 4; ++layer)
    {
        int device = (layer % 2 == 0) ? 0 : 1;
        std::string prefix = "layer." + std::to_string(layer) + ".";

        placement_map->setTensorDevice(prefix + "attn_q", device);
        placement_map->setTensorDevice(prefix + "attn_k", device);
        placement_map->setTensorDevice(prefix + "attn_v", device);
        placement_map->setTensorDevice(prefix + "ffn_gate", device);
    }

    // Verify placements
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.0.attn_q"), 0);
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.1.attn_q"), 1);
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.2.ffn_gate"), 0);
    EXPECT_EQ(placement_map->getDeviceForWeight("layer.3.ffn_gate"), 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
