/**
 * @file Test__IQ2_XSTensor_Views.cpp
 * @brief Unit tests for IQ2_XSTensor view support (row-slice views preserving K dimension)
 * @author David Sanftenberg
 */

#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace llaminar2;

class Test__IQ2_XSTensor_Views : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const size_t M = 4;
        const size_t K = 512;
        const size_t blocks_per_row = K / 256;
        const size_t block_bytes = sizeof(IQ2_XSBlock);
        const size_t total_bytes = M * blocks_per_row * block_bytes;

        raw_data_.resize(total_bytes, 0);

        IQ2_XSBlock *blocks = reinterpret_cast<IQ2_XSBlock *>(raw_data_.data());
        for (size_t r = 0; r < M; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                IQ2_XSBlock &block = blocks[r * blocks_per_row + b];
                block.d = fp32_to_fp16(static_cast<float>(r + 1));
                // IQ2_XS has qs[32] of type uint16_t + scales[8]
                for (size_t i = 0; i < 32; ++i)
                {
                    block.qs[i] = static_cast<uint16_t>((r * 16 + i) & 0xFFFF);
                }
                for (size_t i = 0; i < 8; ++i)
                {
                    block.scales[i] = static_cast<uint8_t>(r + 1);
                }
            }
        }

        parent_shape_ = {M, K};
        parent_tensor_ = std::make_shared<IQ2_XSTensor>(parent_shape_, raw_data_);
    }

    std::vector<size_t> parent_shape_;
    std::vector<uint8_t> raw_data_;
    std::shared_ptr<IQ2_XSTensor> parent_tensor_;
};

TEST_F(Test__IQ2_XSTensor_Views, BasicViewCreation)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    ASSERT_NE(view, nullptr);
    auto *view_tensor = dynamic_cast<IQ2_XSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);
    EXPECT_TRUE(view_tensor->is_view());
}

TEST_F(Test__IQ2_XSTensor_Views, ViewWithOffset)
{
    auto view = parent_tensor_->create_view({2, 512}, 1024);
    ASSERT_NE(view, nullptr);
}

TEST_F(Test__IQ2_XSTensor_Views, KDimensionMustMatch)
{
    EXPECT_THROW(parent_tensor_->create_view({2, 256}, 0), std::invalid_argument);
}

TEST_F(Test__IQ2_XSTensor_Views, OffsetMustBeRowAligned)
{
    EXPECT_THROW(parent_tensor_->create_view({2, 512}, 256), std::invalid_argument);
}

TEST_F(Test__IQ2_XSTensor_Views, ViewBoundsChecking)
{
    EXPECT_THROW(parent_tensor_->create_view({3, 512}, 1024), std::out_of_range);
}

TEST_F(Test__IQ2_XSTensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    {
        auto temp_parent = std::make_shared<IQ2_XSTensor>(parent_shape_, raw_data_);
        view = temp_parent->create_view({2, 512}, 0);
    }
    auto *view_tensor = dynamic_cast<IQ2_XSTensor *>(view.get());
    EXPECT_TRUE(view_tensor->is_view());
}

TEST_F(Test__IQ2_XSTensor_Views, ViewChaining)
{
    auto view1 = parent_tensor_->create_view({3, 512}, 512);
    auto view2 = view1->create_view({2, 512}, 512);
    ASSERT_NE(view2, nullptr);
}

TEST_F(Test__IQ2_XSTensor_Views, IBlockDecoderInterface)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    auto *view_tensor = dynamic_cast<IQ2_XSTensor *>(view.get());
    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);
    const void *raw_block = view_tensor->get_raw_block_at(0, 0);
    EXPECT_NE(raw_block, nullptr);
}

TEST_F(Test__IQ2_XSTensor_Views, SuperBlockAlignment)
{
    EXPECT_EQ(IQ2_XSBlock::BLOCK_SIZE, 256);
}
