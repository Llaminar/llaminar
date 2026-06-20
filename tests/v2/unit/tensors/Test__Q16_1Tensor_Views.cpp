/**
 * @file Test__Q16_1Tensor_Views.cpp
 * @brief Unit tests for Q16_1 tensor view support including 3D→2D MoE expert slicing
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstring>

using namespace llaminar2;

class Test__Q16_1Tensor_Views : public ::testing::Test
{
protected:
    /**
     * @brief Create a Q16_1 tensor with known pattern for testing
     *
     * Q16_1Block has BLOCK_SIZE=32 with: float d, int32_t sum_qs, int16_t qs[32]
     * Constructor: Q16_1Tensor(shape, raw_data) where shape is 2D {rows, cols}
     * and cols must be divisible by BLOCK_SIZE (32).
     */
    std::shared_ptr<Q16_1Tensor> createTestTensor(size_t rows, size_t cols)
    {
        size_t blocks_per_row = (cols + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;
        size_t raw_size = total_blocks * sizeof(Q16_1Block);

        std::vector<uint8_t> raw_data(raw_size);
        Q16_1Block *blocks = reinterpret_cast<Q16_1Block *>(raw_data.data());

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                size_t block_idx = r * blocks_per_row + b;

                // Q16_1Block uses native float for d
                blocks[block_idx].d = static_cast<float>(block_idx + 1) * 0.01f;

                // Fill quantized values and compute sum
                int32_t sum = 0;
                for (size_t i = 0; i < 32; ++i)
                {
                    int16_t val = static_cast<int16_t>((block_idx * 32 + i) & 0x7FFF);
                    blocks[block_idx].qs[i] = val;
                    sum += val;
                }
                blocks[block_idx].sum_qs = sum;
            }
        }

        return std::make_shared<Q16_1Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }
};

TEST_F(Test__Q16_1Tensor_Views, BasicViewCreation)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 64);
}

TEST_F(Test__Q16_1Tensor_Views, ViewWithOffset)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());
}

TEST_F(Test__Q16_1Tensor_Views, KDimensionMustMatch)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 32}, 0), std::invalid_argument);
}

TEST_F(Test__Q16_1Tensor_Views, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 64}, 32), std::invalid_argument);
}

TEST_F(Test__Q16_1Tensor_Views, ViewBoundsChecking)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({8, 64}, 5 * 64), std::out_of_range);
}

TEST_F(Test__Q16_1Tensor_Views, ViewLifetime)
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

// --- 3D Parent → 2D View tests ---

TEST_F(Test__Q16_1Tensor_Views, View3DParentBasicSlice)
{
    // Q16_1Tensor constructor only supports 2D; use flat [N*R, K] layout
    // to test expert-style slicing (equivalent to 3D [N=4, R=8, K=128])
    const size_t N = 4, R = 8, K = 128;
    auto parent = createTestTensor(N * R, K);

    // Extract "expert 0" as 2D view
    auto view = parent->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q16_1Tensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 128;
    auto parent = createTestTensor(N * R, K);

    // Extract "expert 2" (middle expert)
    auto view = parent->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q16_1Tensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 128;
    auto parent = createTestTensor(N * R, K);

    // View starting at row 25 with 8 rows needs rows 25-32, but total is 32 → overflow
    EXPECT_THROW(parent->create_view({R, K}, (N * R - R + 1) * K), std::out_of_range);
}
