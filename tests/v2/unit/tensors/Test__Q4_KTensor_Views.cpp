/**
 * @file Test__Q4_KTensor_Views.cpp
 * @brief Unit tests for Q4_KTensor row-slice view support
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <cmath>

using namespace llaminar2;

namespace
{

    // Helper to create a test Q4_K tensor with unique values
    std::shared_ptr<Q4_KTensor> createTestTensor(size_t rows, size_t cols)
    {
        // Fill with unique block values for verification
        // Q4_K has 256 elements per block (super-block)
        const size_t blocks_per_row = (cols + 255) / 256;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_KBlock));
        Q4_KBlock *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());

        for (size_t i = 0; i < total_blocks; ++i)
        {
            // Set unique delta and dmin values for each block
            blocks[i].d = fp32_to_fp16(static_cast<float>(i) * 0.01f);
            blocks[i].dmin = fp32_to_fp16(static_cast<float>(i) * 0.001f);

            // Set unique scale values (12 bytes, packed 6-bit scales)
            for (size_t j = 0; j < 12; ++j)
            {
                blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) & 0xFF);
            }

            // Set pattern in quantized data (128 bytes, 4 bits × 2 = 256 elements)
            for (size_t j = 0; j < 128; ++j)
            {
                blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
            }
        }

        return std::make_shared<Q4_KTensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

} // anonymous namespace

TEST(Test__Q4_KTensor, BasicViewCreation)
{
    auto parent = createTestTensor(128, 512); // 512 = 2 × 256

    // Create view of first 64 rows
    auto view = parent->create_view({64, 512}, 0);
    ASSERT_NE(view, nullptr);

    // Verify shape
    EXPECT_EQ(view->shape().size(), 2);
    EXPECT_EQ(view->shape()[0], 64);
    EXPECT_EQ(view->shape()[1], 512);
}

TEST(Test__Q4_KTensor, ViewWithOffset)
{
    auto parent = createTestTensor(128, 1024); // 1024 = 4 × 256

    // Create view starting at row 32 (offset = 32 * 1024 = 32768 elements)
    auto view = parent->create_view({64, 1024}, 32 * 1024);
    ASSERT_NE(view, nullptr);

    // Verify shape
    EXPECT_EQ(view->shape()[0], 64);
    EXPECT_EQ(view->shape()[1], 1024);
}

TEST(Test__Q4_KTensor, KDimensionMustMatch)
{
    auto parent = createTestTensor(128, 512);

    // Attempt to create view with different K dimension (should throw)
    EXPECT_THROW({
        auto view = parent->create_view({64, 256}, 0); // K changed from 512 to 256
    },
                 std::invalid_argument);
}

TEST(Test__Q4_KTensor, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(128, 512);

    // Offset must be multiple of K (row-aligned)
    // Valid: 0, 512, 1024, 1536, ... (multiples of 512)
    auto view_valid = parent->create_view({64, 512}, 512);
    EXPECT_NE(view_valid, nullptr);

    // Invalid: not a multiple of K (should throw)
    EXPECT_THROW({
        auto view_invalid = parent->create_view({64, 512}, 256); // 256 is not multiple of 512
    },
                 std::invalid_argument);
}

TEST(Test__Q4_KTensor, ViewBoundsChecking)
{
    auto parent = createTestTensor(128, 512);

    // Valid: view fits within parent
    auto view_valid = parent->create_view({64, 512}, 32 * 512); // Rows 32-95
    EXPECT_NE(view_valid, nullptr);

    // Invalid: view extends beyond parent (should throw)
    EXPECT_THROW({
        auto view_invalid = parent->create_view({96, 512}, 64 * 512); // Would need rows 64-159
    },
                 std::out_of_range);
}

TEST(Test__Q4_KTensor, ViewLifetime)
{
    std::weak_ptr<Q4_KTensor> weak_parent;
    std::shared_ptr<TensorBase> view;

    {
        auto parent = createTestTensor(128, 512);
        weak_parent = parent;

        // Create view
        view = parent->create_view({64, 512}, 0);
        ASSERT_NE(view, nullptr);

        // Parent goes out of scope, but should be kept alive by view
    }

    // Parent should still be alive because view holds reference
    EXPECT_FALSE(weak_parent.expired());

    // Verify view is still valid
    EXPECT_EQ(view->shape()[0], 64);
    EXPECT_EQ(view->shape()[1], 512);

    // Destroy view
    view.reset();

    // Now parent should be destroyed
    EXPECT_TRUE(weak_parent.expired());
}

TEST(Test__Q4_KTensor, ViewChaining)
{
    auto parent = createTestTensor(128, 1024);

    // Create first view (rows 32-95)
    auto view1 = parent->create_view({64, 1024}, 32 * 1024);
    ASSERT_NE(view1, nullptr);

    // Create view of view (rows 16-47 of view1 = rows 48-79 of parent)
    auto view2 = view1->create_view({32, 1024}, 16 * 1024);
    ASSERT_NE(view2, nullptr);

    // Verify shapes
    EXPECT_EQ(view2->shape()[0], 32);
    EXPECT_EQ(view2->shape()[1], 1024);
}

TEST(Test__Q4_KTensor, IBlockDecoderInterface)
{
    auto tensor = createTestTensor(64, 512);

    // Q4_K should support IBlockDecoder interface
    EXPECT_EQ(tensor->block_size(), 256); // Q4_K uses 256-element super-blocks

    // Test decode_block_at
    float decoded[256];
    tensor->decode_block_at(0, 0, decoded);

    // Decoded values should be finite
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_TRUE(std::isfinite(decoded[i]));
    }

    // Test get_raw_block_at
    const void *block_ptr = tensor->get_raw_block_at(0, 0);
    EXPECT_NE(block_ptr, nullptr);
}

TEST(Test__Q4_KTensor, SuperBlockAlignment)
{
    // Q4_K uses 256-element super-blocks
    // Column count must be multiple of 256 for proper alignment

    auto tensor_aligned = createTestTensor(32, 512); // 512 = 2 × 256 ✓
    EXPECT_NE(tensor_aligned, nullptr);

    auto tensor_aligned2 = createTestTensor(32, 1024); // 1024 = 4 × 256 ✓
    EXPECT_NE(tensor_aligned2, nullptr);

    // Non-aligned column counts still work (blocks_per_row rounds up)
    // but may waste memory
    auto tensor_nonaligned = createTestTensor(32, 300); // Rounds to 2 blocks (512 elements)
    EXPECT_NE(tensor_nonaligned, nullptr);

    // Verify blocks_per_row calculation
    // 300 elements → (300 + 255) / 256 = 2 blocks
    const size_t expected_blocks_per_row = (300 + 255) / 256;
    EXPECT_EQ(expected_blocks_per_row, 2);
}
