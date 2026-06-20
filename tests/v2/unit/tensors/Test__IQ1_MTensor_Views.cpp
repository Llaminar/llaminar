/**
 * @file Test__IQ1_MTensor_Views.cpp
 * @brief Unit tests for IQ1_MTensor view support (row-slice views preserving K dimension)
 * @author David Sanftenberg
 */

#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace llaminar2;

class Test__IQ1_MTensor_Views : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const size_t M = 4;
        const size_t K = 512;
        const size_t blocks_per_row = K / 256;
        const size_t block_bytes = sizeof(IQ1_MBlock);
        const size_t total_bytes = M * blocks_per_row * block_bytes;

        raw_data_.resize(total_bytes, 0);

        IQ1_MBlock *blocks = reinterpret_cast<IQ1_MBlock *>(raw_data_.data());
        for (size_t r = 0; r < M; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                IQ1_MBlock &block = blocks[r * blocks_per_row + b];
                // IQ1_M has scales[8] instead of single d field
                for (size_t i = 0; i < 8; ++i)
                {
                    block.scales[i] = static_cast<uint8_t>(r + 1);
                }
                for (size_t i = 0; i < 16; ++i)
                {
                    block.qs[i] = static_cast<uint8_t>((r * 16 + i) & 0xFF);
                }
                for (size_t i = 0; i < 8; ++i)
                {
                    block.qh[i] = static_cast<uint8_t>((r * 8 + i) & 0xFF);
                }
            }
        }

        parent_shape_ = {M, K};
        parent_tensor_ = std::make_shared<IQ1_MTensor>(parent_shape_, raw_data_);
    }

    std::vector<size_t> parent_shape_;
    std::vector<uint8_t> raw_data_;
    std::shared_ptr<IQ1_MTensor> parent_tensor_;
};

TEST_F(Test__IQ1_MTensor_Views, BasicViewCreation)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    ASSERT_NE(view, nullptr);
    auto *view_tensor = dynamic_cast<IQ1_MTensor *>(view.get());
    EXPECT_TRUE(view_tensor->is_view());
}

TEST_F(Test__IQ1_MTensor_Views, ViewWithOffset)
{
    auto view = parent_tensor_->create_view({2, 512}, 1024);
    ASSERT_NE(view, nullptr);
}

TEST_F(Test__IQ1_MTensor_Views, KDimensionMustMatch)
{
    EXPECT_THROW(parent_tensor_->create_view({2, 256}, 0), std::invalid_argument);
}

TEST_F(Test__IQ1_MTensor_Views, OffsetMustBeRowAligned)
{
    EXPECT_THROW(parent_tensor_->create_view({2, 512}, 256), std::invalid_argument);
}

TEST_F(Test__IQ1_MTensor_Views, ViewBoundsChecking)
{
    EXPECT_THROW(parent_tensor_->create_view({3, 512}, 1024), std::out_of_range);
}

TEST_F(Test__IQ1_MTensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto temp_parent = std::make_shared<IQ1_MTensor>(parent_shape_, raw_data_);
        view = temp_parent->create_view({2, 512}, 0);
    }
    auto *view_tensor = dynamic_cast<IQ1_MTensor *>(view.get());
    EXPECT_TRUE(view_tensor->is_view());
}

TEST_F(Test__IQ1_MTensor_Views, ViewChaining)
{
    auto view1 = parent_tensor_->create_view({3, 512}, 512);
    auto view2 = view1->create_view({2, 512}, 512);
    ASSERT_NE(view2, nullptr);
}

TEST_F(Test__IQ1_MTensor_Views, IBlockDecoderInterface)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    auto *view_tensor = dynamic_cast<IQ1_MTensor *>(view.get());
    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);
    const void *raw_block = view_tensor->get_raw_block_at(0, 0);
    EXPECT_NE(raw_block, nullptr);
}

TEST_F(Test__IQ1_MTensor_Views, SuperBlockAlignment)
{
    EXPECT_EQ(IQ1_MBlock::BLOCK_SIZE, 256);
}

// --- 3D Parent → 2D View regression tests ---

TEST_F(Test__IQ1_MTensor_Views, View3DParentBasicSlice)
{
    // 3D GGUF shape: [cols=512(ne[0]), rows_per_expert=8(ne[1]), num_experts=4(ne[2])]
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ1_MBlock));
    auto tensor_3d = std::make_shared<IQ1_MTensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 0 as 2D view
    auto view = tensor_3d->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__IQ1_MTensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ1_MBlock));
    auto tensor_3d = std::make_shared<IQ1_MTensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 2 (middle expert)
    auto view = tensor_3d->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__IQ1_MTensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ1_MBlock));
    auto tensor_3d = std::make_shared<IQ1_MTensor>(std::vector<size_t>{K, R, N}, raw);

    // View starting at row 25 with 8 rows needs rows 25-32, but total is 32 → overflow
    EXPECT_THROW(
        tensor_3d->create_view({R, K}, (N * R - R + 1) * K),
        std::out_of_range);
}
