/**
 * @file Test__Q4_0Tensor_Views.cpp
 * @brief Unit tests for Q4_0Tensor view support
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>

using namespace llaminar2;

namespace
{

    // Helper: Create Q4_0 tensor with unique scale per block for verification
    std::shared_ptr<Q4_0Tensor> createTestTensor(size_t rows, size_t cols)
    {
        constexpr size_t BLOCK_SIZE = Q4_0Block::BLOCK_SIZE;
        const size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t total_blocks = rows * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
        Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

        for (size_t i = 0; i < total_blocks; ++i)
        {
            // Unique scale per block for verification
            blocks[i].d = fp32_to_fp16(1.0f + 0.1f * static_cast<float>(i));
            // Simple pattern in quantized values
            for (size_t j = 0; j < 16; ++j)
            {
                blocks[i].qs[j] = static_cast<uint8_t>(j * 17); // Varied pattern
            }
        }

        return std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

} // namespace

// Test 1: Basic view creation
TEST(Q4_0TensorViews, BasicViewCreation)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    // Create view of first 32 rows
    auto view = parent->create_view({32, 128}, 0);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 128);
}

// Test 2: View with offset
TEST(Q4_0TensorViews, ViewWithOffset)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    // Create view starting at row 16
    size_t offset = 16 * 128;
    auto view = parent->create_view({32, 128}, offset);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 128);
}

// Test 3: K dimension must match
TEST(Q4_0TensorViews, KDimensionMustMatch)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    // Attempting to change column count should throw
    EXPECT_THROW({ auto view = parent->create_view({32, 64}, 0); }, std::invalid_argument);
}

// Test 4: Offset must be row-aligned
TEST(Q4_0TensorViews, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    // Offset not at row boundary (128 elements per row)
    EXPECT_THROW({ auto view = parent->create_view({32, 128}, 64); }, std::invalid_argument);
}

// Test 5: View bounds checking
TEST(Q4_0TensorViews, ViewBoundsChecking)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    // View exceeds parent bounds
    EXPECT_THROW({
        auto view = parent->create_view({50, 128}, 20 * 128); // 20 + 50 = 70 > 64
    },
                 std::out_of_range);
}

// Test 6: View keeps parent alive
TEST(Q4_0TensorViews, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto parent = createTestTensor(64, 128);
        view = parent->create_view({32, 128}, 0);
        ASSERT_NE(view, nullptr);
        // parent goes out of scope here
    }

    // View should still be valid (parent kept alive)
    EXPECT_EQ(view->shape()[0], 32);
    EXPECT_EQ(view->shape()[1], 128);
}

// Test 7: View chaining
TEST(Q4_0TensorViews, ViewChaining)
{
    auto parent = createTestTensor(128, 256);
    ASSERT_NE(parent, nullptr);

    // First view: rows 16-79 (64 rows)
    auto view1 = parent->create_view({64, 256}, 16 * 256);
    ASSERT_NE(view1, nullptr);

    // Second view: rows 16-47 of view1 (which is rows 32-63 of parent)
    auto view2 = view1->create_view({32, 256}, 16 * 256);
    ASSERT_NE(view2, nullptr);

    EXPECT_EQ(view2->shape()[0], 32);
    EXPECT_EQ(view2->shape()[1], 256);
}

// Test 8: ITensorGemmTileDataProvider interface works with views
TEST(Q4_0TensorViews, IBlockDecoderInterface)
{
    auto parent = createTestTensor(64, 128);
    ASSERT_NE(parent, nullptr);

    auto view = parent->create_view({32, 128}, 16 * 128); // Rows 16-47
    ASSERT_NE(view, nullptr);

    // Cast to ITensorGemmTileDataProvider
    auto *decoder = dynamic_cast<ITensorGemmTileDataProvider *>(view.get());
    ASSERT_NE(decoder, nullptr);

    // Verify decoder metadata
    EXPECT_EQ(decoder->decoder_rows(), 32);
    EXPECT_EQ(decoder->decoder_cols(), 128);
    EXPECT_EQ(decoder->block_size(), Q4_0Block::BLOCK_SIZE);

    // Decode a block from the view
    float output[Q4_0Block::BLOCK_SIZE];
    decoder->decode_block_at(0, 0, output);

    // Output should be non-zero (pattern-based data)
    bool has_nonzero = false;
    for (size_t i = 0; i < Q4_0Block::BLOCK_SIZE; ++i)
    {
        if (output[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// --- 3D Parent → 2D View regression tests ---

TEST(Q4_0TensorViews, View3DParentBasicSlice)
{
    // 3D GGUF shape: [cols=128(ne[0]), rows_per_expert=8(ne[1]), num_experts=4(ne[2])]
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / 32;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q4_0Block));
    auto tensor_3d = std::make_shared<Q4_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 0 as 2D view
    auto view = tensor_3d->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST(Q4_0TensorViews, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / 32;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q4_0Block));
    auto tensor_3d = std::make_shared<Q4_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 2 (middle expert)
    auto view = tensor_3d->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST(Q4_0TensorViews, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / 32;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q4_0Block));
    auto tensor_3d = std::make_shared<Q4_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // View starting at row 25 with 8 rows needs rows 25-32, but total is 32 → overflow
    EXPECT_THROW(
        tensor_3d->create_view({R, K}, (N * R - R + 1) * K),
        std::out_of_range);
}
