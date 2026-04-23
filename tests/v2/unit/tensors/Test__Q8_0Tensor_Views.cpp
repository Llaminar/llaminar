/**
 * @file Test__Q8_0Tensor_Views.cpp
 * @brief Unit tests for Q8_0 tensor view support
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Test fixture for Q8_0 tensor view tests
 */
class Test__Q8_0Tensor_Views : public ::testing::Test
{
protected:
    /**
     * @brief Create a Q8_0 tensor with known pattern for testing
     */
    std::shared_ptr<Q8_0Tensor> createTestTensor(size_t rows, size_t cols)
    {
        size_t blocks_per_row = (cols + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;
        size_t raw_size = total_blocks * sizeof(Q8_0Block);

        std::vector<uint8_t> raw_data(raw_size);
        Q8_0Block *blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

        // Fill with pattern: each block has unique scale
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                size_t block_idx = r * blocks_per_row + b;
                float scale = static_cast<float>(block_idx + 1);

                // Convert to FP16
                uint32_t bits;
                std::memcpy(&bits, &scale, sizeof(float));
                uint32_t sign = (bits >> 16) & 0x8000;
                int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
                uint32_t mantissa = (bits >> 13) & 0x3FF;

                if (exponent <= 0)
                    blocks[block_idx].d = static_cast<uint16_t>(sign);
                else if (exponent >= 0x1F)
                    blocks[block_idx].d = static_cast<uint16_t>(sign | 0x7C00);
                else
                    blocks[block_idx].d = static_cast<uint16_t>(sign | (exponent << 10) | mantissa);

                // Fill quantized values
                for (size_t i = 0; i < 32; ++i)
                {
                    blocks[block_idx].qs[i] = static_cast<int8_t>((block_idx + i) & 0x7F);
                }
            }
        }

        return std::make_shared<Q8_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }
};

TEST_F(Test__Q8_0Tensor_Views, BasicViewCreation)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 64);
}

TEST_F(Test__Q8_0Tensor_Views, ViewWithOffset)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());
}

TEST_F(Test__Q8_0Tensor_Views, KDimensionMustMatch)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 32}, 0), std::invalid_argument);
}

TEST_F(Test__Q8_0Tensor_Views, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 64}, 32), std::invalid_argument);
}

TEST_F(Test__Q8_0Tensor_Views, ViewBoundsChecking)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({8, 64}, 5 * 64), std::out_of_range);
}

TEST_F(Test__Q8_0Tensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto parent = createTestTensor(10, 64);
        view = parent->create_view({5, 64}, 0);
        ASSERT_NE(view, nullptr);
    }

    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape()[0], 5);
}

TEST_F(Test__Q8_0Tensor_Views, ViewChaining)
{
    auto parent = createTestTensor(20, 64);
    auto view1 = parent->create_view({10, 64}, 5 * 64);
    ASSERT_NE(view1, nullptr);

    auto view2 = view1->create_view({5, 64}, 2 * 64);
    ASSERT_NE(view2, nullptr);
    EXPECT_TRUE(view2->is_view());
}

TEST_F(Test__Q8_0Tensor_Views, IBlockDecoderInterface)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64);
    ASSERT_NE(view, nullptr);

    auto *view_q8 = dynamic_cast<Q8_0Tensor *>(view.get());
    ASSERT_NE(view_q8, nullptr);

    float view_block[Q8_0Block::BLOCK_SIZE];
    view_q8->decode_block_at(0, 0, view_block);

    auto *parent_q8 = dynamic_cast<Q8_0Tensor *>(parent.get());
    float parent_block[Q8_0Block::BLOCK_SIZE];
    parent_q8->decode_block_at(5, 0, parent_block);

    for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(view_block[i], parent_block[i]);
    }
}

// --- 3D Parent → 2D View regression tests ---

TEST_F(Test__Q8_0Tensor_Views, View3DParentBasicSlice)
{
    // 3D GGUF shape: [cols=128(ne[0]), rows_per_expert=8(ne[1]), num_experts=4(ne[2])]
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q8_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q8_0Block));
    auto tensor_3d = std::make_shared<Q8_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 0 as 2D view
    auto view = tensor_3d->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q8_0Tensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q8_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q8_0Block));
    auto tensor_3d = std::make_shared<Q8_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 2 (middle expert)
    auto view = tensor_3d->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q8_0Tensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q8_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(Q8_0Block));
    auto tensor_3d = std::make_shared<Q8_0Tensor>(std::vector<size_t>{K, R, N}, raw);

    // View starting at row 25 with 8 rows needs rows 25-32, but total is 32 → overflow
    EXPECT_THROW(tensor_3d->create_view({R, K}, (N * R - R + 1) * K), std::out_of_range);
}
