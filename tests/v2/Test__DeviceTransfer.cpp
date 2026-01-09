/**
 * @file Test__DeviceTransfer.cpp
 * @brief Test device transfer functionality (Phase 4.2)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../src/v2/tensors/Tensors.h"
#include <vector>
#include <memory>
#include <cmath>

using namespace llaminar2;

/**
 * @test Verify CPU → CPU transfer works correctly
 */
TEST(Test__DeviceTransfer, CPUtoCPU_BasicTransfer)
{
    // Create source tensor on CPU
    std::vector<size_t> shape = {4, 8};
    auto src = std::make_unique<FP32Tensor>(shape, -1); // device_idx = -1 (CPU)

    // Fill with test data
    float *src_data = src->mutable_data();
    for (size_t i = 0; i < 32; ++i)
    {
        src_data[i] = static_cast<float>(i) * 1.5f;
    }

    // Create destination tensor on CPU
    auto dst = std::make_unique<FP32Tensor>(shape, -1);

    // Perform copy
    bool success = dst->copyFrom(src.get());
    EXPECT_TRUE(success) << "CPU → CPU transfer should succeed";

    // Verify data was copied correctly
    const float *dst_data = dst->data();
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i) * 1.5f)
            << "Mismatch at index " << i;
    }
}

/**
 * @test Verify shape mismatch detection
 */
TEST(Test__DeviceTransfer, ShapeMismatchDetection)
{
    auto src = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8}, -1);
    auto dst = std::make_unique<FP32Tensor>(std::vector<size_t>{8, 4}, -1);

    bool success = dst->copyFrom(src.get());
    EXPECT_FALSE(success) << "Shape mismatch should be detected and fail";
}

/**
 * @test Verify null source detection
 */
TEST(Test__DeviceTransfer, NullSourceDetection)
{
    auto dst = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 8}, -1);

    bool success = dst->copyFrom(nullptr);
    EXPECT_FALSE(success) << "Null source should be detected and fail";
}

/**
 * @test Verify device index tracking
 */
TEST(Test__DeviceTransfer, DeviceIndexTracking)
{
    auto cpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4}, -1);
    EXPECT_TRUE(cpu_tensor->home_device().is_cpu()) << "CPU tensor should have CPU device";

    auto gpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4}, 0);
    EXPECT_TRUE(gpu_tensor->home_device().is_cpu()) << "Legacy device_idx=0 maps to CPU";
}

/**
 * @test Verify CPU → GPU transfer stub behavior (Phase 4 CUDA will implement)
 */
TEST(Test__DeviceTransfer, CPUtoGPU_StubBehavior)
{
    auto src = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4}, -1); // CPU
    auto dst = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 4}, 0);  // GPU

    // Fill source with data
    float *src_data = src->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        src_data[i] = static_cast<float>(i);
    }

    // Try transfer (should fail with stub message)
    bool success = dst->copyFrom(src.get());
    EXPECT_FALSE(success) << "CPU → GPU transfer should fail (not implemented yet)";
}

/**
 * @test Verify quantized tensor copyFrom stub behavior
 */
TEST(Test__DeviceTransfer, QuantizedTensor_ReadOnlyStub)
{
    // Create IQ4_NL tensor (quantized, read-only)
    std::vector<size_t> shape = {64, 32};         // 64 rows × 32 cols
    size_t blocks_per_row = (shape[1] + 31) / 32; // 1 block per row (32 elements/block)
    size_t total_blocks = shape[0] * blocks_per_row;
    std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_NLBlock), 0);

    auto quantized = std::make_unique<IQ4_NLTensor>(shape, raw_data);
    auto fp32_tensor = std::make_unique<FP32Tensor>(shape, -1);

    // Try to copy from FP32 to quantized (should fail - quantized is read-only)
    bool success = quantized->copyFrom(fp32_tensor.get());
    EXPECT_FALSE(success) << "copyFrom on quantized tensor should fail (read-only)";
}

/**
 * @test Verify large tensor transfer performance
 */
TEST(Test__DeviceTransfer, LargeTensorTransfer)
{
    std::vector<size_t> shape = {512, 1024}; // 512K elements
    auto src = std::make_unique<FP32Tensor>(shape, -1);
    auto dst = std::make_unique<FP32Tensor>(shape, -1);

    // Fill with pattern
    float *src_data = src->mutable_data();
    for (size_t i = 0; i < 512 * 1024; ++i)
    {
        src_data[i] = std::sin(static_cast<float>(i) * 0.001f);
    }

    // Transfer
    bool success = dst->copyFrom(src.get());
    EXPECT_TRUE(success) << "Large tensor transfer should succeed";

    // Spot check (first, middle, last)
    const float *dst_data = dst->data();
    EXPECT_FLOAT_EQ(dst_data[0], std::sin(0.0f));
    EXPECT_FLOAT_EQ(dst_data[256 * 1024], std::sin(256 * 1024 * 0.001f));
    EXPECT_FLOAT_EQ(dst_data[512 * 1024 - 1], std::sin((512 * 1024 - 1) * 0.001f));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
