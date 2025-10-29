/**
 * @file Test__Q6_KTensor_Views.cpp
 * @brief Unit tests for Q6_KTensor view support (256-element super-blocks)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>

using namespace llaminar2;

namespace
{

    // Helper: Create Q6_K tensor with unique scale per super-block for verification
    // Q6_K uses 256-element super-blocks, so cols must be multiple of 256
    std::shared_ptr<Q6_KTensor> createTestTensor(size_t rows, size_t cols)
    {
        constexpr size_t BLOCK_SIZE = Q6_KBlock::BLOCK_SIZE; // 256
        const size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q6_KBlock));
        Q6_KBlock *blocks = reinterpret_cast<Q6_KBlock *>(raw_data.data());

        for (size_t i = 0; i < total_blocks; ++i)
        {
            // Unique super-block scale for verification
            blocks[i].d = fp32_to_fp16(1.0f + 0.1f * static_cast<float>(i));
            // Unique sub-block scales (16 per super-block)
            for (size_t j = 0; j < 16; ++j)
            {
                blocks[i].scales[j] = static_cast<int8_t>(j * 8 + static_cast<int8_t>(i % 16));
            }
            // Pattern in quantized data
            for (size_t j = 0; j < 128; ++j)
            {
                blocks[i].ql[j] = static_cast<uint8_t>((j * 17 + i) % 256);
            }
            for (size_t j = 0; j < 64; ++j)
            {
                blocks[i].qh[j] = static_cast<uint8_t>((j * 13 + i) % 256);
            }
        }

        return std::make_shared<Q6_KTensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

} // namespace

// Test 1: Basic view creation
TEST(Q6_KTensorViews, BasicViewCreation)
{
    auto parent = createTestTensor(64, 512); // 512 = 2 * 256
    ASSERT_NE(parent, nullptr);

    // Create view of first 32 rows
    auto view = parent->create_view({32, 512}, 0);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 512);
}

// Test 2: View with offset
TEST(Q6_KTensorViews, ViewWithOffset)
{
    auto parent = createTestTensor(64, 512);
    ASSERT_NE(parent, nullptr);

    // Create view starting at row 16
    size_t offset = 16 * 512;
    auto view = parent->create_view({32, 512}, offset);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 512);
}

// Test 3: K dimension must match
TEST(Q6_KTensorViews, KDimensionMustMatch)
{
    auto parent = createTestTensor(64, 512);
    ASSERT_NE(parent, nullptr);

    // Attempting to change column count should throw
    EXPECT_THROW({ auto view = parent->create_view({32, 256}, 0); }, std::invalid_argument);
}

// Test 4: Offset must be row-aligned
TEST(Q6_KTensorViews, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(64, 512);
    ASSERT_NE(parent, nullptr);

    // Offset not at row boundary (512 elements per row)
    EXPECT_THROW({
        auto view = parent->create_view({32, 512}, 256); // 256 is not row-aligned (need 512)
    },
                 std::invalid_argument);
}

// Test 5: View bounds checking
TEST(Q6_KTensorViews, ViewBoundsChecking)
{
    auto parent = createTestTensor(64, 512);
    ASSERT_NE(parent, nullptr);

    // View exceeds parent bounds
    EXPECT_THROW({
        auto view = parent->create_view({50, 512}, 20 * 512); // 20 + 50 = 70 > 64
    },
                 std::out_of_range);
}

// Test 6: View keeps parent alive
TEST(Q6_KTensorViews, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto parent = createTestTensor(64, 512);
        view = parent->create_view({32, 512}, 0);
        ASSERT_NE(view, nullptr);
        // parent goes out of scope here
    }

    // View should still be valid (parent kept alive)
    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 512);
}

// Test 7: View chaining
TEST(Q6_KTensorViews, ViewChaining)
{
    auto parent = createTestTensor(128, 1024); // 1024 = 4 * 256
    ASSERT_NE(parent, nullptr);

    // First view: rows 16-79 (64 rows)
    auto view1 = parent->create_view({64, 1024}, 16 * 1024);
    ASSERT_NE(view1, nullptr);

    // Second view: rows 16-47 of view1 (which is rows 32-63 of parent)
    auto view2 = view1->create_view({32, 1024}, 16 * 1024);
    ASSERT_NE(view2, nullptr);

    EXPECT_EQ(view2->shape()[0], 32);
    EXPECT_EQ(view2->shape()[1], 1024);
}

// Test 8: IBlockDecoder interface works with views
TEST(Q6_KTensorViews, IBlockDecoderInterface)
{
    auto parent = createTestTensor(64, 512);
    ASSERT_NE(parent, nullptr);

    auto view = parent->create_view({32, 512}, 16 * 512); // Rows 16-47
    ASSERT_NE(view, nullptr);

    // Cast to IBlockDecoder
    auto *decoder = dynamic_cast<IBlockDecoder *>(view.get());
    ASSERT_NE(decoder, nullptr);

    // Verify decoder metadata
    EXPECT_EQ(decoder->decoder_rows(), 32);
    EXPECT_EQ(decoder->decoder_cols(), 512);
    EXPECT_EQ(decoder->block_size(), Q6_KBlock::BLOCK_SIZE); // 256

    // Decode a super-block from the view
    float output[Q6_KBlock::BLOCK_SIZE];
    decoder->decode_block_at(0, 0, output);

    // Output should be non-zero (pattern-based data with scales)
    bool has_nonzero = false;
    for (size_t i = 0; i < Q6_KBlock::BLOCK_SIZE; ++i)
    {
        if (output[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test 9: 256-element alignment requirement
TEST(Q6_KTensorViews, SuperBlockAlignment)
{
    // Q6_K requires columns to be multiple of 256 (super-block size)
    auto parent = createTestTensor(64, 512); // Valid: 512 = 2 * 256
    ASSERT_NE(parent, nullptr);

    // Verify we can create views
    auto view = parent->create_view({32, 512}, 0);
    ASSERT_NE(view, nullptr);

    // Note: Non-256-aligned columns would be caught during tensor creation,
    // not during view creation (view preserves parent's column count)
}
