/**
 * @file Test__IQ2_XXSTensor_Views.cpp
 * @brief Unit tests for IQ2_XXSTensor view support (row-slice views preserving K dimension)
 * @author David Sanftenberg
 */

#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace llaminar2;

/**
 * @brief Test fixture for IQ2_XXSTensor view tests
 *
 * IQ2_XXS uses 256-element super-blocks (66 bytes each)
 * View support ensures zero-copy row slicing for MPI weight partitioning.
 */
class Test__IQ2_XXSTensor_Views : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create 4 rows × 512 columns (2 blocks per row, 66 bytes each)
        const size_t M = 4;
        const size_t K = 512; // Must be multiple of 256
        const size_t blocks_per_row = K / 256;
        const size_t block_bytes = sizeof(IQ2_XXSBlock);
        const size_t total_bytes = M * blocks_per_row * block_bytes;

        raw_data_.resize(total_bytes, 0);

        // Initialize blocks with distinguishable patterns
        IQ2_XXSBlock *blocks = reinterpret_cast<IQ2_XXSBlock *>(raw_data_.data());
        for (size_t r = 0; r < M; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                IQ2_XXSBlock &block = blocks[r * blocks_per_row + b];
                block.d = fp32_to_fp16(static_cast<float>(r + 1));
                // IQ2_XXS has qs[32] of type uint16_t (not uint8_t)
                for (size_t i = 0; i < 32; ++i)
                {
                    block.qs[i] = static_cast<uint16_t>((r * 16 + i) & 0xFFFF);
                }
            }
        }

        parent_shape_ = {M, K};
        parent_tensor_ = std::make_shared<IQ2_XXSTensor>(parent_shape_, raw_data_);
    }

    std::vector<size_t> parent_shape_;
    std::vector<uint8_t> raw_data_;
    std::shared_ptr<IQ2_XXSTensor> parent_tensor_;
};

TEST_F(Test__IQ2_XXSTensor_Views, BasicViewCreation)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    ASSERT_NE(view, nullptr);

    auto *view_tensor = dynamic_cast<IQ2_XXSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);

    EXPECT_TRUE(view_tensor->is_view());
    EXPECT_EQ(view_tensor->shape()[0], 2);
    EXPECT_EQ(view_tensor->shape()[1], 512);
}

TEST_F(Test__IQ2_XXSTensor_Views, ViewWithOffset)
{
    auto view = parent_tensor_->create_view({2, 512}, 1024); // 2 rows offset
    ASSERT_NE(view, nullptr);

    auto *view_tensor = dynamic_cast<IQ2_XXSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);
}

TEST_F(Test__IQ2_XXSTensor_Views, KDimensionMustMatch)
{
    EXPECT_THROW(
        parent_tensor_->create_view({2, 256}, 0),
        std::invalid_argument);
}

TEST_F(Test__IQ2_XXSTensor_Views, OffsetMustBeRowAligned)
{
    EXPECT_THROW(
        parent_tensor_->create_view({2, 512}, 256),
        std::invalid_argument);
}

TEST_F(Test__IQ2_XXSTensor_Views, ViewBoundsChecking)
{
    EXPECT_THROW(
        parent_tensor_->create_view({3, 512}, 1024),
        std::out_of_range);
}

TEST_F(Test__IQ2_XXSTensor_Views, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;

    {
        auto temp_parent = std::make_shared<IQ2_XXSTensor>(parent_shape_, raw_data_);
        view = temp_parent->create_view({2, 512}, 0);
    }

    auto *view_tensor = dynamic_cast<IQ2_XXSTensor *>(view.get());
    ASSERT_NE(view_tensor, nullptr);
    EXPECT_TRUE(view_tensor->is_view());

    float decoded[256];
    EXPECT_NO_THROW(view_tensor->decode_block_at(0, 0, decoded));
}

TEST_F(Test__IQ2_XXSTensor_Views, ViewChaining)
{
    auto view1 = parent_tensor_->create_view({3, 512}, 512);
    ASSERT_NE(view1, nullptr);

    auto view2 = view1->create_view({2, 512}, 512);
    ASSERT_NE(view2, nullptr);

    auto *view2_tensor = dynamic_cast<IQ2_XXSTensor *>(view2.get());
    ASSERT_NE(view2_tensor, nullptr);
}

TEST_F(Test__IQ2_XXSTensor_Views, IBlockDecoderInterface)
{
    auto view = parent_tensor_->create_view({2, 512}, 512);
    auto *view_tensor = dynamic_cast<IQ2_XXSTensor *>(view.get());

    float decoded[256];
    view_tensor->decode_block_at(0, 0, decoded);

    const void *raw_block = view_tensor->get_raw_block_at(0, 0);
    EXPECT_NE(raw_block, nullptr);
}

TEST_F(Test__IQ2_XXSTensor_Views, SuperBlockAlignment)
{
    // Verify 256-element block boundaries
    EXPECT_EQ(IQ2_XXSBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(parent_tensor_->shape()[1] % IQ2_XXSBlock::BLOCK_SIZE, 0);
}

// --- 3D Parent → 2D View regression tests ---

TEST_F(Test__IQ2_XXSTensor_Views, View3DParentBasicSlice)
{
    // 3D GGUF shape: [cols=512(ne[0]), rows_per_expert=8(ne[1]), num_experts=4(ne[2])]
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ2_XXSBlock));
    auto tensor_3d = std::make_shared<IQ2_XXSTensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 0 as 2D view
    auto view = tensor_3d->create_view({R, K}, 0);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__IQ2_XXSTensor_Views, View3DParentWithOffset)
{
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ2_XXSBlock));
    auto tensor_3d = std::make_shared<IQ2_XXSTensor>(std::vector<size_t>{K, R, N}, raw);

    // Extract expert 2 (middle expert)
    auto view = tensor_3d->create_view({R, K}, 2 * R * K);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], R);
    EXPECT_EQ(view->shape()[1], K);
}

TEST_F(Test__IQ2_XXSTensor_Views, View3DParentBoundsCheck)
{
    const size_t N = 4, R = 8, K = 512;
    const size_t blocks_per_row = K / 256;
    const size_t total_blocks = N * R * blocks_per_row;
    std::vector<uint8_t> raw(total_blocks * sizeof(IQ2_XXSBlock));
    auto tensor_3d = std::make_shared<IQ2_XXSTensor>(std::vector<size_t>{K, R, N}, raw);

    // View starting at row 25 with 8 rows needs rows 25-32, but total is 32 → overflow
    EXPECT_THROW(
        tensor_3d->create_view({R, K}, (N * R - R + 1) * K),
        std::out_of_range);
}
