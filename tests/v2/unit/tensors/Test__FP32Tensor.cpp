/**
 * @file Test__FP32Tensor.cpp
 * @brief Unit tests for FP32Tensor class
 * @author David Sanftenberg
 *
 * This is an example unit test demonstrating the V2 test infrastructure.
 * It tests the FP32Tensor class without requiring model loading.
 *
 * Naming convention: Test file and test suite are named after the class under test.
 * File: Test__FP32Tensor.cpp → Testing: FP32Tensor class
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include <memory>

using namespace llaminar2;

/**
 * @brief Test FP32 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, FP32Creation)
{
    std::vector<size_t> shape = {2, 3}; // 2x3 matrix

    auto tensor = std::make_shared<FP32Tensor>(shape);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->native_type(), TensorType::FP32);

    // Calculate expected sizes
    size_t expected_elements = 2 * 3;                          // 6 elements
    size_t expected_bytes = expected_elements * sizeof(float); // 24 bytes

    // V2 TensorBase doesn't expose size() directly, but we can verify via data pointer
    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test FP16 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, FP16Creation)
{
    std::vector<size_t> shape = {4, 4};     // 4x4 matrix
    std::vector<uint16_t> data(16, 0x3C00); // FP16 value for 1.0

    auto tensor = std::make_shared<FP16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 4);
    EXPECT_EQ(tensor->shape()[1], 4);
    EXPECT_EQ(tensor->native_type(), TensorType::FP16);

    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test BF16 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, BF16Creation)
{
    std::vector<size_t> shape = {3, 5};     // 3x5 matrix
    std::vector<uint16_t> data(15, 0x3F80); // BF16 value for 1.0

    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->shape()[1], 5);
    EXPECT_EQ(tensor->native_type(), TensorType::BF16);

    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test various tensor shapes (1D, 2D, 3D)
 */
TEST(Test__FP32Tensor, ShapeValidation)
{
    // 1D tensor (vector)
    auto tensor1d = std::make_shared<FP32Tensor>(std::vector<size_t>{10});
    EXPECT_EQ(tensor1d->shape().size(), 1);
    EXPECT_EQ(tensor1d->shape()[0], 10);

    // 2D tensor (matrix)
    auto tensor2d = std::make_shared<FP32Tensor>(std::vector<size_t>{5, 8});
    EXPECT_EQ(tensor2d->shape().size(), 2);
    EXPECT_EQ(tensor2d->shape()[0], 5);
    EXPECT_EQ(tensor2d->shape()[1], 8);

    // 3D tensor (batch of matrices)
    auto tensor3d = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 3, 4});
    EXPECT_EQ(tensor3d->shape().size(), 3);
    EXPECT_EQ(tensor3d->shape()[0], 2);
    EXPECT_EQ(tensor3d->shape()[1], 3);
    EXPECT_EQ(tensor3d->shape()[2], 4);
}

/**
 * @brief Test device affinity defaults (CPU = device -1)
 */
TEST(Test__FP32Tensor, DeviceAffinity)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{10, 10});

    // Default device should be host (CPU = -1)
    EXPECT_EQ(tensor->device_index(), -1);
    EXPECT_TRUE(tensor->is_on_device(-1));
}
