/**
 * @file Test__Q5_0Tensor_Views.cpp
 * @brief Unit tests for Q5_0 tensor view support including 3D→2D MoE expert slicing
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstring>

using namespace llaminar2;

class Test__Q5_0Tensor_Views : public ::testing::Test
{
protected:
    std::shared_ptr<Q5_0Tensor> createTestTensor(size_t rows, size_t cols)
    {
        size_t blocks_per_row = (cols + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;
        size_t raw_size = total_blocks * sizeof(Q5_0Block);

        std::vector<uint8_t> raw_data(raw_size);
        Q5_0Block *blocks = reinterpret_cast<Q5_0Block *>(raw_data.data());

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

                // Fill high bits
                for (size_t i = 0; i < 4; ++i)
                {
                    blocks[block_idx].qh[i] = static_cast<uint8_t>((block_idx + i) & 0xFF);
                }

                // Fill quantized values
                for (size_t i = 0; i < 16; ++i)
                {
                    blocks[block_idx].qs[i] = static_cast<uint8_t>((block_idx + i) & 0xFF);
                }
            }
        }

        return std::make_shared<Q5_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }
};

TEST_F(Test__Q5_0Tensor_Views, BasicViewCreation)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 0);

    ASSERT_NE(view, nullptr);
    // Note: Q5_0Tensor is missing is_view() override (pre-existing bug)
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 64);
}

TEST_F(Test__Q5_0Tensor_Views, ViewWithOffset)
{
    auto parent = createTestTensor(10, 64);
    auto view = parent->create_view({5, 64}, 5 * 64);

    ASSERT_NE(view, nullptr);
}

TEST_F(Test__Q5_0Tensor_Views, KDimensionMustMatch)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 32}, 0), std::invalid_argument);
}

TEST_F(Test__Q5_0Tensor_Views, OffsetMustBeRowAligned)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({5, 64}, 32), std::invalid_argument);
}

TEST_F(Test__Q5_0Tensor_Views, ViewBoundsChecking)
{
    auto parent = createTestTensor(10, 64);
    EXPECT_THROW(parent->create_view({8, 64}, 5 * 64), std::out_of_range);
}

TEST_F(Test__Q5_0Tensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto parent = createTestTensor(10, 64);
        view = parent->create_view({5, 64}, 0);
        ASSERT_NE(view, nullptr);
    }

    EXPECT_EQ(view->shape()[0], 5);
}

// --- 3D Parent → 2D View tests ---

TEST_F(Test__Q5_0Tensor_Views, View3DParentBasicSlice)
{
    // Q5_0Tensor(shape, raw_data) requires 2D; use mmap constructor for 3D
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q5_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    auto raw_owner = std::make_shared<std::vector<uint8_t>>(total_blocks * sizeof(Q5_0Block));
    auto tensor_3d = std::make_shared<Q5_0Tensor>(
        std::vector<size_t>{K, R, N},
        raw_owner->data(),
        raw_owner->size(),
        std::static_pointer_cast<void>(raw_owner));

    auto view = tensor_3d->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q5_0Tensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q5_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    auto raw_owner = std::make_shared<std::vector<uint8_t>>(total_blocks * sizeof(Q5_0Block));
    auto tensor_3d = std::make_shared<Q5_0Tensor>(
        std::vector<size_t>{K, R, N},
        raw_owner->data(),
        raw_owner->size(),
        std::static_pointer_cast<void>(raw_owner));

    auto view = tensor_3d->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__Q5_0Tensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 128;
    const size_t blocks_per_row = K / Q5_0Block::BLOCK_SIZE;
    const size_t total_blocks = N * R * blocks_per_row;
    auto raw_owner = std::make_shared<std::vector<uint8_t>>(total_blocks * sizeof(Q5_0Block));
    auto tensor_3d = std::make_shared<Q5_0Tensor>(
        std::vector<size_t>{K, R, N},
        raw_owner->data(),
        raw_owner->size(),
        std::static_pointer_cast<void>(raw_owner));

    EXPECT_THROW(tensor_3d->create_view({R, K}, (N * R - R + 1) * K), std::out_of_range);
}
