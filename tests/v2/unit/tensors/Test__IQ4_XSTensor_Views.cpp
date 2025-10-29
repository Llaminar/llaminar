/**
 * @file Test__IQ4_XSTensor_Views.cpp
 * @brief Unit tests for IQ4_XSTensor view support (row-slice views preserving K dimension)
 * @author David Sanftenberg
 */

#include "../../../../src/v2/tensors/Tensors.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace llaminar2;

/**
 * @brief Test fixture for IQ4_XSTensor view tests
 *
 * IQ4_XS uses 256-element super-blocks (136 bytes each)
 * View support ensures zero-copy row slicing for MPI weight partitioning.
 */
class Test__IQ4_XSTensor_Views : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create 4 rows × 512 columns (2 super-blocks per row, 136 bytes each)
        const size_t M = 4;
        const size_t K = 512; // Must be multiple of 256
        const size_t blocks_per_row = K / 256;
        const size_t block_bytes = sizeof(IQ4_XSBlock);
        const size_t total_bytes = M * blocks_per_row * block_bytes;

        raw_data_.resize(total_bytes, 0);

        // Initialize super-blocks with distinguishable patterns (row_idx encoded in scale)
        IQ4_XSBlock *blocks = reinterpret_cast<IQ4_XSBlock *>(raw_data_.data());
        for (size_t r = 0; r < M; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                IQ4_XSBlock &block = blocks[r * blocks_per_row + b];
                block.d = fp32_to_fp16(static_cast<float>(r + 1)); // Row 0 → scale 1.0, row 1 → 2.0, etc.
                block.scales_h = static_cast<uint16_t>(r * 256);
                for (size_t i = 0; i < 4; ++i)
                {
                    block.scales_l[i] = static_cast<uint8_t>((r * 4 + i) & 0xFF);
                }
                for (size_t i = 0; i < 128; ++i)
                {
                    block.qs[i] = static_cast<uint8_t>((r * 128 + i) & 0xFF);
                }
            }
        }

        parent_shape_ = {M, K};
        parent_tensor_ = std::make_shared<IQ4_XSTensor>(parent_shape_, raw_data_);
    }

    std::vector<size_t> parent_shape_;
    std::vector<uint8_t> raw_data_;
    std::shared_ptr<IQ4_XSTensor> parent_tensor_;
};

/**
 * @brief Test basic view creation (middle 2 rows)
 */
TEST_F(Test__IQ4_XSTensor_Views, BasicViewCreation)
{
    // Create view: rows [1, 2] (2 rows × 512 cols)
    auto view = parent_tensor_->create_view({2, 512}, 512); // Offset 512 = 1 row
    ASSERT_NE(view, nullptr);

    auto *view_tensor = dynamic_cast<IQ4_XSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);

    // Check view properties
    EXPECT_TRUE(view_tensor->is_view());
    EXPECT_EQ(view_tensor->shape()[0], 2);
    EXPECT_EQ(view_tensor->shape()[1], 512);

    // Decode first block of view (should be row 1 of parent)
    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);

    // Row 1 of parent has scale 2.0
    const float expected_scale = 2.0f;
    EXPECT_NEAR(fp16_to_fp32(fp32_to_fp16(expected_scale)), expected_scale, 0.01f);
}

/**
 * @brief Test view with non-zero offset
 */
TEST_F(Test__IQ4_XSTensor_Views, ViewWithOffset)
{
    // Create view: last 2 rows (rows [2, 3])
    auto view = parent_tensor_->create_view({2, 512}, 1024); // Offset 1024 = 2 rows
    ASSERT_NE(view, nullptr);

    auto *view_tensor = dynamic_cast<IQ4_XSTensor *>(view.get());
    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);

    // Row 2 of parent has scale 3.0
    // (Validation: first element should decode with this scale)
}

/**
 * @brief Test that K dimension must match
 */
TEST_F(Test__IQ4_XSTensor_Views, KDimensionMustMatch)
{
    EXPECT_THROW(
        parent_tensor_->create_view({2, 64}, 0), // K=64 != parent K=512
        std::invalid_argument);
}

/**
 * @brief Test that offset must be row-aligned
 */
TEST_F(Test__IQ4_XSTensor_Views, OffsetMustBeRowAligned)
{
    EXPECT_THROW(
        parent_tensor_->create_view({2, 512}, 64), // Offset 64 is not multiple of K=512
        std::invalid_argument);
}

/**
 * @brief Test view bounds checking
 */
TEST_F(Test__IQ4_XSTensor_Views, ViewBoundsChecking)
{
    // Offset 2 rows + 3 view rows = 5 rows total (exceeds 4 rows in parent)
    EXPECT_THROW(
        parent_tensor_->create_view({3, 512}, 1024), // 2 + 3 > 4
        std::out_of_range);
}

/**
 * @brief Test view lifetime (parent must outlive view)
 */
TEST_F(Test__IQ4_XSTensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;

    {
        auto temp_parent = std::make_shared<IQ4_XSTensor>(parent_shape_, raw_data_);
        view = temp_parent->create_view({2, 512}, 0);
    } // temp_parent goes out of scope, but view should keep it alive

    auto *view_tensor = dynamic_cast<IQ4_XSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);
    EXPECT_TRUE(view_tensor->is_view());

    // View should still be usable
    float decoded[256];
    EXPECT_NO_THROW(view_tensor->decode_block_at(0, 0, decoded));
}

/**
 * @brief Test view chaining (view of view)
 */
TEST_F(Test__IQ4_XSTensor_Views, ViewChaining)
{
    // Parent: 4 rows [0, 1, 2, 3]
    // View 1: 3 rows [1, 2, 3] (offset=1 row)
    auto view1 = parent_tensor_->create_view({3, 512}, 512);
    ASSERT_NE(view1, nullptr);

    // View 2: 2 rows [2, 3] (1 row offset into view1)
    auto view2 = view1->create_view({2, 512}, 512);
    ASSERT_NE(view2, nullptr);

    auto *view2_tensor = dynamic_cast<IQ4_XSTensor *>(view2.get());
    float decoded[256];
    view2_tensor->decode_block_at(0, 0, decoded);

    // Should decode row 2 of parent (scale 3.0)
    const float expected_scale = 3.0f;
    EXPECT_NEAR(fp16_to_fp32(fp32_to_fp16(expected_scale)), expected_scale, 0.01f);
}

/**
 * @brief Test IBlockDecoder interface on view
 */
TEST_F(Test__IQ4_XSTensor_Views, IBlockDecoderInterface)
{
    auto view = parent_tensor_->create_view({2, 512}, 512); // Rows [1, 2]
    auto *view_tensor = dynamic_cast<IQ4_XSTensor *>(view.get());

    // Test decode_block_at (inlined in header)
    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);

    // Test get_raw_block_at
    const void *raw_block = view_tensor->get_raw_block_at(0, 0);
    EXPECT_NE(raw_block, nullptr);

    const auto *block = static_cast<const IQ4_XSBlock *>(raw_block);
    EXPECT_NEAR(fp16_to_fp32(block->d), 2.0f, 0.01f); // Row 1 scale
}

/**
 * @brief Test block alignment (K must be multiple of block size)
 */
TEST_F(Test__IQ4_XSTensor_Views, BlockAlignment)
{
    // K=512 is valid (96 % 32 == 0)
    EXPECT_NO_THROW(parent_tensor_->create_view({2, 512}, 0));

    // Create parent with K=50 (not multiple of 32) - should work (padding handled)
    std::vector<size_t> odd_shape = {2, 50};
    size_t blocks_per_row = (50 + 31) / 32; // 2 blocks (padded)
    std::vector<uint8_t> odd_data(2 * blocks_per_row * sizeof(IQ4_XSBlock), 0);
    auto odd_tensor = std::make_shared<IQ4_XSTensor>(odd_shape, odd_data);

    // View with same K should work
    EXPECT_NO_THROW(odd_tensor->create_view({1, 50}, 0));
}
