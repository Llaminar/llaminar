/**
 * @file Test__FP16Tensor.cpp
 * @brief Unit tests for FP16Tensor class including view support
 * @author David Sanftenberg
 *
 * Tests FP16 tensor operations including:
 * - Basic tensor creation and properties
 * - FP16 ↔ FP32 conversion accuracy
 * - View creation and bounds checking
 * - View lifetime management
 * - View chaining (view of a view)
 * - Data access through views
 *
 * Naming convention: Test file and test suite are named after the class under test.
 * File: Test__FP16Tensor.cpp → Testing: FP16Tensor class
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <memory>
#include <cmath>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Test FP16 tensor creation and basic properties
 */
TEST(Test__FP16Tensor, BasicCreation)
{
    std::vector<size_t> shape = {3, 5}; // 3x5 matrix
    std::vector<uint16_t> data(15);

    // Initialize with FP16 representation of 1.0
    for (size_t i = 0; i < 15; ++i)
    {
        data[i] = fp32_to_fp16(1.0f);
    }

    auto tensor = std::make_shared<FP16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->shape()[1], 5);
    EXPECT_EQ(tensor->native_type(), TensorType::FP16);
    EXPECT_EQ(tensor->device_index(), -1); // Default to CPU
    ASSERT_NE(tensor->data(), nullptr);
    EXPECT_FALSE(tensor->is_view());
}

/**
 * @brief Test FP16 ↔ FP32 conversion accuracy
 *
 * FP16 has ~3-4 decimal digits of precision (10 mantissa bits vs FP32's 23).
 * Conversion should preserve sign, exponent, and mantissa bits.
 */
TEST(Test__FP16Tensor, ConversionAccuracy)
{
    // Test values covering different ranges
    std::vector<float> test_values = {
        0.0f,     // Zero
        1.0f,     // Exact representation
        -1.0f,    // Negative
        3.14159f, // π (loses precision)
        0.125f,   // Power of 2 (exact)
        100.5f,   // Larger value
        0.001f,   // Small value
        -50.25f   // Negative larger value
    };

    for (float original : test_values)
    {
        uint16_t fp16 = fp32_to_fp16(original);
        float converted = fp16_to_fp32(fp16);

        if (original == 0.0f)
        {
            EXPECT_EQ(converted, 0.0f);
        }
        else
        {
            // FP16 has ~0.05% relative precision for normalized values
            float rel_error = std::abs((converted - original) / original);
            EXPECT_LT(rel_error, 0.001f) << "Value: " << original
                                         << " converted to: " << converted;
        }
    }
}

/**
 * @brief Test FP16 tensor data access
 */
TEST(Test__FP16Tensor, DataAccess)
{
    std::vector<size_t> shape = {2, 3};
    std::vector<uint16_t> fp16_data = {
        fp32_to_fp16(1.0f), fp32_to_fp16(2.0f), fp32_to_fp16(3.0f),
        fp32_to_fp16(4.0f), fp32_to_fp16(5.0f), fp32_to_fp16(6.0f)};

    auto tensor = std::make_shared<FP16Tensor>(shape, fp16_data);

    const float *fp32_data = tensor->data();
    ASSERT_NE(fp32_data, nullptr);

    // Verify conversion to FP32
    EXPECT_NEAR(fp32_data[0], 1.0f, 1e-3f);
    EXPECT_NEAR(fp32_data[1], 2.0f, 1e-3f);
    EXPECT_NEAR(fp32_data[2], 3.0f, 1e-3f);
    EXPECT_NEAR(fp32_data[3], 4.0f, 1e-3f);
    EXPECT_NEAR(fp32_data[4], 5.0f, 1e-3f);
    EXPECT_NEAR(fp32_data[5], 6.0f, 1e-3f);
}

// ========== View Tests ==========

/**
 * @brief Test basic view creation
 */
TEST(Test__FP16Tensor, BasicViewCreation)
{
    // Create parent tensor [10, 20] = 200 elements
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    // Fill with test data (0, 1, 2, ..., 199)
    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create a view of first 5 rows
    auto view = parent->create_view({5, 20}, 0);

    ASSERT_NE(view, nullptr) << "View creation failed";
    EXPECT_EQ(view->shape().size(), 2);
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 20);
    EXPECT_TRUE(view->is_view());

    // Verify data pointer is valid
    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr) << "View data pointer is null";

    // Verify first element
    EXPECT_NEAR(view_data[0], 0.0f, 1e-3f);
}

/**
 * @brief Test view creation with offset
 */
TEST(Test__FP16Tensor, ViewWithOffset)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    // Fill with test data
    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create a view starting at element 100 (row 5)
    auto view = parent->create_view({3, 20}, 100);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());

    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr);

    // First element should be 100
    EXPECT_NEAR(view_data[0], 100.0f, 1e-2f) << "First element should be ~100";
}

/**
 * @brief Test view bounds checking
 */
TEST(Test__FP16Tensor, ViewBoundsChecking)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    // Try to create a view that exceeds parent bounds
    auto view = parent->create_view({20, 20}, 0); // 400 elements > 200 available

    EXPECT_EQ(view, nullptr) << "View creation should fail for out-of-bounds request";
}

/**
 * @brief Test view with offset that exceeds bounds
 */
TEST(Test__FP16Tensor, ViewOffsetBoundsChecking)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    // Try to create a view with offset that exceeds bounds
    auto view = parent->create_view({5, 20}, 150); // offset 150 + 100 elements > 200

    EXPECT_EQ(view, nullptr) << "View creation should fail when offset + size exceeds bounds";
}

/**
 * @brief Test view lifetime (parent stays alive via shared_ptr)
 */
TEST(Test__FP16Tensor, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    const float *view_data_ptr = nullptr;

    {
        auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});
        std::vector<float> test_data(200);
        for (size_t i = 0; i < 200; ++i)
        {
            test_data[i] = static_cast<float>(i * 2);
        }
        parent->from_fp32(test_data.data(), 200);

        // Create view in inner scope
        view = parent->create_view({5, 20}, 0);
        ASSERT_NE(view, nullptr);
        view_data_ptr = view->data();
        ASSERT_NE(view_data_ptr, nullptr);

        // parent goes out of scope here
    }

    // View still exists, should keep parent alive
    EXPECT_NE(view->data(), nullptr) << "View data should still be valid";
    EXPECT_TRUE(view->is_view());

    // Data should still be accessible
    const float *current_data = view->data();
    EXPECT_NEAR(current_data[0], 0.0f, 1e-2f) << "View data should still be accessible";
}

/**
 * @brief Test view chaining (view of a view)
 */
TEST(Test__FP16Tensor, ViewChaining)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create first view (rows 2-6)
    auto view1 = parent->create_view({5, 20}, 40); // offset = 2 rows * 20 cols
    ASSERT_NE(view1, nullptr);
    EXPECT_TRUE(view1->is_view());

    // Cast to FP16Tensor to create view of view
    auto fp16_view1 = std::dynamic_pointer_cast<FP16Tensor>(view1);
    ASSERT_NE(fp16_view1, nullptr);

    // Create view of view (first 2 rows of view1)
    auto view2 = fp16_view1->create_view({2, 20}, 0);
    ASSERT_NE(view2, nullptr);
    EXPECT_TRUE(view2->is_view());

    // Verify view2 points to correct data in original parent
    const float *view2_data = view2->data();
    ASSERT_NE(view2_data, nullptr);

    // First element should be 40 (offset 40 in parent)
    EXPECT_NEAR(view2_data[0], 40.0f, 1e-2f) << "Chained view should point to element 40";
}

/**
 * @brief Test view data modification affects parent
 */
TEST(Test__FP16Tensor, ViewModification)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200, 1.0f);
    parent->from_fp32(test_data.data(), 200);

    // Create view
    auto view = parent->create_view({5, 20}, 0);
    ASSERT_NE(view, nullptr);

    auto fp16_view = std::dynamic_pointer_cast<FP16Tensor>(view);
    ASSERT_NE(fp16_view, nullptr);

    // Modify through view
    std::vector<float> new_data(100, 42.0f);
    fp16_view->from_fp32(new_data.data(), 100);

    // Verify parent data changed
    const float *parent_data = parent->data();
    EXPECT_NEAR(parent_data[0], 42.0f, 1e-2f) << "Parent should reflect view modification";
    EXPECT_NEAR(parent_data[99], 42.0f, 1e-2f);
}

/**
 * @brief Test view with different shape (reshape)
 */
TEST(Test__FP16Tensor, ViewReshape)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create view with different shape but same total elements
    auto view = parent->create_view({20, 10}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], 20);
    EXPECT_EQ(view->shape()[1], 10);

    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr);

    // Data should still be accessible in new shape
    EXPECT_NEAR(view_data[0], 0.0f, 1e-3f);
    EXPECT_NEAR(view_data[10], 10.0f, 1e-2f);
}

/**
 * @brief Test view of subset with reshape
 */
TEST(Test__FP16Tensor, ViewSubsetReshape)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Take 120 elements starting at offset 40, reshape to [6, 20]
    auto view = parent->create_view({6, 20}, 40);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], 6);
    EXPECT_EQ(view->shape()[1], 20);

    const float *view_data = view->data();
    EXPECT_NEAR(view_data[0], 40.0f, 1e-2f);
}

/**
 * @brief Test multiple views of same parent
 */
TEST(Test__FP16Tensor, MultipleViews)
{
    auto parent = std::make_shared<FP16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create multiple non-overlapping views
    auto view1 = parent->create_view({3, 20}, 0);   // rows 0-2
    auto view2 = parent->create_view({3, 20}, 60);  // rows 3-5
    auto view3 = parent->create_view({4, 20}, 120); // rows 6-9

    ASSERT_NE(view1, nullptr);
    ASSERT_NE(view2, nullptr);
    ASSERT_NE(view3, nullptr);

    EXPECT_NEAR(view1->data()[0], 0.0f, 1e-3f);
    EXPECT_NEAR(view2->data()[0], 60.0f, 1e-2f);
    EXPECT_NEAR(view3->data()[0], 120.0f, 1e-1f);
}

/**
 * @brief Test FP16 precision with typical values
 */
TEST(Test__FP16Tensor, PrecisionTypicalValues)
{
    std::vector<float> original_values;
    std::vector<uint16_t> fp16_values;

    // Generate test values from -10 to 10 with 0.1 steps
    for (float val = -10.0f; val <= 10.0f; val += 0.1f)
    {
        original_values.push_back(val);
        fp16_values.push_back(fp32_to_fp16(val));
    }

    auto tensor = std::make_shared<FP16Tensor>(
        std::vector<size_t>{original_values.size()},
        fp16_values);

    const float *converted = tensor->data();

    for (size_t i = 0; i < original_values.size(); ++i)
    {
        float original = original_values[i];
        float after_conversion = converted[i];

        if (std::abs(original) < 1e-4f)
        {
            // Near-zero values
            EXPECT_NEAR(after_conversion, 0.0f, 1e-3f);
        }
        else
        {
            // Relative error should be very small for FP16
            float rel_error = std::abs((after_conversion - original) / original);
            EXPECT_LT(rel_error, 0.002f) << "Value: " << original
                                         << " -> " << after_conversion;
        }
    }
}
