/**
 * @file Test__TensorViews.cpp
 * @brief Unit tests for tensor view functionality
 * @author David Sanftenberg
 *
 * Tests the tensor view system including:
 * - View creation from parent tensors
 * - View lifetime management (shared_ptr)
 * - Data pointer validity
 * - Bounds checking
 * - View chaining (view of a view)
 */

#include <gtest/gtest.h>
#include "v2/tensors/Tensors.h"
#include <memory>
#include <vector>

using namespace llaminar2;

/**
 * @brief Test fixture for tensor view tests
 */
class TensorViewTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a parent tensor with known data
        parent_ = std::make_shared<FP32Tensor>(std::vector<size_t>{10, 20}, -1);

        // Fill with test data
        float *data = parent_->mutable_data();
        for (size_t i = 0; i < 200; ++i)
        {
            data[i] = static_cast<float>(i);
        }
    }

    std::shared_ptr<FP32Tensor> parent_;
};

/**
 * @brief Test basic view creation
 */
TEST_F(TensorViewTest, BasicViewCreation)
{
    // Create a view of first 5 rows
    auto view = parent_->create_view({5, 20}, 0);

    ASSERT_NE(view, nullptr) << "View creation failed";
    EXPECT_EQ(view->shape().size(), 2);
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 20);

    // Verify data pointer is valid
    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr) << "View data pointer is null";

    // Verify view points to parent data
    const float *parent_data = parent_->data();
    EXPECT_EQ(view_data, parent_data) << "View should point to parent data at offset 0";
}

/**
 * @brief Test view creation with offset
 */
TEST_F(TensorViewTest, ViewWithOffset)
{
    // Create a view starting at element 100 (row 5)
    auto view = parent_->create_view({3, 20}, 100);

    ASSERT_NE(view, nullptr);

    const float *view_data = view->data();
    const float *parent_data = parent_->data();

    EXPECT_EQ(view_data, parent_data + 100) << "View should point to parent data + offset";
    EXPECT_FLOAT_EQ(view_data[0], 100.0f) << "First element should be 100";
}

/**
 * @brief Test view bounds checking
 */
TEST_F(TensorViewTest, ViewBoundsChecking)
{
    // Try to create a view that exceeds parent bounds
    auto view = parent_->create_view({20, 20}, 0); // 400 elements > 200 available

    EXPECT_EQ(view, nullptr) << "View creation should fail for out-of-bounds request";
}

/**
 * @brief Test view with offset that exceeds bounds
 */
TEST_F(TensorViewTest, ViewOffsetBoundsChecking)
{
    // Try to create a view with offset that exceeds bounds
    auto view = parent_->create_view({5, 20}, 150); // offset 150 + 100 elements > 200

    EXPECT_EQ(view, nullptr) << "View creation should fail when offset + size exceeds bounds";
}

/**
 * @brief Test view lifetime (parent stays alive via shared_ptr)
 */
TEST_F(TensorViewTest, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    const float *view_data = nullptr;

    {
        // Create view in inner scope
        view = parent_->create_view({5, 20}, 0);
        ASSERT_NE(view, nullptr);
        view_data = view->data();
        ASSERT_NE(view_data, nullptr);
    }

    // View still exists, should keep parent alive
    EXPECT_NE(view->data(), nullptr) << "View data should still be valid";
    EXPECT_FLOAT_EQ(view->data()[0], 0.0f) << "View data should be accessible";
}

/**
 * @brief Test view chaining (view of a view)
 */
TEST_F(TensorViewTest, ViewChaining)
{
    // Create first view (rows 2-6)
    auto view1 = parent_->create_view({5, 20}, 40); // offset = 2 rows * 20 cols
    ASSERT_NE(view1, nullptr);

    // Cast to FP32Tensor to create view of view
    auto fp32_view1 = std::dynamic_pointer_cast<FP32Tensor>(view1);
    ASSERT_NE(fp32_view1, nullptr);

    // Create view of view (first 2 rows of view1)
    auto view2 = fp32_view1->create_view({2, 20}, 0);
    ASSERT_NE(view2, nullptr);

    // Verify view2 points to correct data in original parent
    const float *view2_data = view2->data();
    const float *parent_data = parent_->data();
    EXPECT_EQ(view2_data, parent_data + 40) << "Chained view should point to parent data + 40";
    EXPECT_FLOAT_EQ(view2_data[0], 40.0f) << "First element should be 40";
}

/**
 * @brief Test view data modification affects parent
 */
TEST_F(TensorViewTest, ViewDataModification)
{
    auto view = parent_->create_view({5, 20}, 0);
    ASSERT_NE(view, nullptr);

    // Modify data through view
    auto fp32_view = std::dynamic_pointer_cast<FP32Tensor>(view);
    ASSERT_NE(fp32_view, nullptr);

    float *view_data = fp32_view->mutable_data();
    view_data[0] = 999.0f;

    // Verify parent sees the change
    EXPECT_FLOAT_EQ(parent_->data()[0], 999.0f) << "Parent should see view's data modification";
}

/**
 * @brief Test that view creation works when parent is managed by shared_ptr
 *
 * This is a regression test for the shared_from_this() requirement
 */
TEST_F(TensorViewTest, ViewFromSharedPtrManaged)
{
    // Parent is already managed by shared_ptr (created in SetUp)
    auto view = parent_->create_view({5, 20}, 0);

    EXPECT_NE(view, nullptr) << "View creation should succeed when parent is shared_ptr managed";
}

/**
 * @brief Test multiple views from same parent
 */
TEST_F(TensorViewTest, MultipleViewsFromParent)
{
    auto view1 = parent_->create_view({3, 20}, 0);
    auto view2 = parent_->create_view({3, 20}, 60);
    auto view3 = parent_->create_view({4, 20}, 120);

    ASSERT_NE(view1, nullptr);
    ASSERT_NE(view2, nullptr);
    ASSERT_NE(view3, nullptr);

    // Verify each view points to correct offset
    EXPECT_FLOAT_EQ(view1->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(view2->data()[0], 60.0f);
    EXPECT_FLOAT_EQ(view3->data()[0], 120.0f);
}

/**
 * @brief Test view with 1D shape
 */
TEST_F(TensorViewTest, OneDimensionalView)
{
    // Create 1D view (flatten first 50 elements)
    auto view = parent_->create_view({50}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape().size(), 1);
    EXPECT_EQ(view->shape()[0], 50);

    const float *view_data = view->data();
    EXPECT_FLOAT_EQ(view_data[25], 25.0f);
}

/**
 * @brief Test view with 3D shape (reshape)
 */
TEST_F(TensorViewTest, ThreeDimensionalView)
{
    // Create 3D view: [2, 5, 10] = 100 elements
    auto view = parent_->create_view({2, 5, 10}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape().size(), 3);
    EXPECT_EQ(view->shape()[0], 2);
    EXPECT_EQ(view->shape()[1], 5);
    EXPECT_EQ(view->shape()[2], 10);
}

/**
 * @brief Test empty view (zero elements)
 */
TEST_F(TensorViewTest, EmptyView)
{
    // Create view with 0 elements
    auto view = parent_->create_view({0, 20}, 0);

    // Should succeed (0 elements is valid)
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], 0);
}

/**
 * @brief Test view creation with invalid shape (negative dimensions)
 *
 * Note: This would require size_t overflow checking, which may not be practical
 * Documenting expected behavior
 */
TEST_F(TensorViewTest, DISABLED_InvalidShapeHandling)
{
    // This test is disabled because size_t is unsigned, so negative values
    // would overflow. Real validation would need to happen at a higher level.
    GTEST_SKIP() << "Size_t overflow checking not implemented";
}
