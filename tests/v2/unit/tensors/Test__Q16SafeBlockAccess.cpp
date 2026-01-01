/**
 * @file Test__Q16SafeBlockAccess.cpp
 * @brief Unit tests for Q16_1Tensor safe block access API (Phase 2)
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the safe block access API including:
 * - as_block_32/64/128() accessors
 * - dequant_element(), quantized_element(), block_scale() generic access
 * - dispatchQ16Block() template dispatch
 * - forEachQ16Block() iteration
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/Q16BlockDispatch.h"
#include "tensors/BlockStructures.h"
#include <cmath>
#include <vector>

namespace llaminar2::test
{

    class Q16SafeBlockAccessTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create tensors with different block sizes for testing
            // Shape: [4, 64] - 4 rows, 64 columns

            // BLOCK_32: 2 blocks per row
            tensor_32_ = std::make_shared<Q16_1Tensor>(
                std::vector<size_t>{4, 64}, Q16BlockSize::BLOCK_32);

            // BLOCK_64: 1 block per row
            tensor_64_ = std::make_shared<Q16_1Tensor>(
                std::vector<size_t>{4, 64}, Q16BlockSize::BLOCK_64);

            // BLOCK_128: For [4, 128] shape
            tensor_128_ = std::make_shared<Q16_1Tensor>(
                std::vector<size_t>{4, 128}, Q16BlockSize::BLOCK_128);

            // Initialize with known values for testing
            initializeTensor(tensor_32_.get());
            initializeTensor(tensor_64_.get());
            initializeTensor(tensor_128_.get());
        }

        void initializeTensor(Q16_1Tensor *tensor)
        {
            // Initialize by copying from FP32 data
            const size_t rows = tensor->shape()[0];
            const size_t cols = tensor->shape()[1];
            std::vector<float> fp32_data(rows * cols);

            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    // Use a pattern that's easy to verify
                    fp32_data[r * cols + c] = static_cast<float>(r * 100 + c) * 0.01f;
                }
            }

            tensor->copyFrom_fp32(fp32_data.data());
        }

        std::shared_ptr<Q16_1Tensor> tensor_32_;
        std::shared_ptr<Q16_1Tensor> tensor_64_;
        std::shared_ptr<Q16_1Tensor> tensor_128_;
    };

    // =============================================================================
    // as_block_XX() Safe Accessor Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, AsBlock32_ReturnsCorrectType_ForBlock32Tensor)
    {
        const Q16_1Block *blocks = tensor_32_->as_block_32();
        ASSERT_NE(blocks, nullptr);
        EXPECT_EQ(tensor_32_->q16_block_size(), Q16BlockSize::BLOCK_32);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock32_ReturnsNullptr_ForBlock64Tensor)
    {
        const Q16_1Block *blocks = tensor_64_->as_block_32();
        EXPECT_EQ(blocks, nullptr);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock32_ReturnsNullptr_ForBlock128Tensor)
    {
        const Q16_1Block *blocks = tensor_128_->as_block_32();
        EXPECT_EQ(blocks, nullptr);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock64_ReturnsCorrectType_ForBlock64Tensor)
    {
        const Q16_1Block_64 *blocks = tensor_64_->as_block_64();
        ASSERT_NE(blocks, nullptr);
        EXPECT_EQ(tensor_64_->q16_block_size(), Q16BlockSize::BLOCK_64);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock64_ReturnsNullptr_ForBlock32Tensor)
    {
        const Q16_1Block_64 *blocks = tensor_32_->as_block_64();
        EXPECT_EQ(blocks, nullptr);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock128_ReturnsCorrectType_ForBlock128Tensor)
    {
        const Q16_1Block_128 *blocks = tensor_128_->as_block_128();
        ASSERT_NE(blocks, nullptr);
        EXPECT_EQ(tensor_128_->q16_block_size(), Q16BlockSize::BLOCK_128);
    }

    TEST_F(Q16SafeBlockAccessTest, AsBlock128_ReturnsNullptr_ForNonBlock128Tensor)
    {
        EXPECT_EQ(tensor_32_->as_block_128(), nullptr);
        EXPECT_EQ(tensor_64_->as_block_128(), nullptr);
    }

    // =============================================================================
    // Mutable as_block_XX() Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, MutableAsBlock32_ReturnsCorrectType_ForBlock32Tensor)
    {
        Q16_1Block *blocks = tensor_32_->mutable_as_block_32();
        ASSERT_NE(blocks, nullptr);

        // Verify we can modify the data
        float original_scale = blocks[0].d;
        blocks[0].d = 999.0f;
        EXPECT_FLOAT_EQ(tensor_32_->as_block_32()[0].d, 999.0f);
        blocks[0].d = original_scale; // Restore
    }

    TEST_F(Q16SafeBlockAccessTest, MutableAsBlock64_ReturnsCorrectType_ForBlock64Tensor)
    {
        Q16_1Block_64 *blocks = tensor_64_->mutable_as_block_64();
        ASSERT_NE(blocks, nullptr);

        // Verify we can modify the data
        float original_scale = blocks[0].d;
        blocks[0].d = 888.0f;
        EXPECT_FLOAT_EQ(tensor_64_->as_block_64()[0].d, 888.0f);
        blocks[0].d = original_scale; // Restore
    }

    // =============================================================================
    // Generic Element Access Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, DequantElement_ReturnsApproximateValue_Block32)
    {
        // We initialized with pattern: value = (row * 100 + col) * 0.01
        // After quantization/dequantization, should be approximately the same

        float expected_0_0 = 0.0f;              // row=0, col=0 -> 0 * 0.01 = 0
        float expected_1_50 = (100 + 50) * 0.01f; // row=1, col=50 -> 150 * 0.01 = 1.5

        float actual_0_0 = tensor_32_->dequant_element(0, 0);
        float actual_1_50 = tensor_32_->dequant_element(1, 50);

        // Allow for quantization error
        EXPECT_NEAR(actual_0_0, expected_0_0, 0.1f);
        EXPECT_NEAR(actual_1_50, expected_1_50, 0.1f);
    }

    TEST_F(Q16SafeBlockAccessTest, DequantElement_ReturnsApproximateValue_Block64)
    {
        float expected_2_30 = (200 + 30) * 0.01f; // row=2, col=30 -> 230 * 0.01 = 2.3

        float actual = tensor_64_->dequant_element(2, 30);

        EXPECT_NEAR(actual, expected_2_30, 0.1f);
    }

    TEST_F(Q16SafeBlockAccessTest, DequantElement_ReturnsApproximateValue_Block128)
    {
        float expected_3_100 = (300 + 100) * 0.01f; // row=3, col=100 -> 400 * 0.01 = 4.0

        float actual = tensor_128_->dequant_element(3, 100);

        EXPECT_NEAR(actual, expected_3_100, 0.1f);
    }

    TEST_F(Q16SafeBlockAccessTest, BlockScale_ReturnsNonZeroValue)
    {
        // Scales should be computed during quantization
        EXPECT_NE(tensor_32_->block_scale(0, 0), 0.0f);
        EXPECT_NE(tensor_64_->block_scale(0, 0), 0.0f);
        EXPECT_NE(tensor_128_->block_scale(0, 0), 0.0f);
    }

    // =============================================================================
    // dispatchQ16Block() Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, DispatchQ16Block_CallsCorrectOverload_Block32)
    {
        size_t received_block_size = 0;
        dispatchQ16Block(tensor_32_.get(), [&](auto * /*blocks*/, size_t bs)
                         {
            received_block_size = bs;
            return 0; });

        EXPECT_EQ(received_block_size, 32u);
    }

    TEST_F(Q16SafeBlockAccessTest, DispatchQ16Block_CallsCorrectOverload_Block64)
    {
        size_t received_block_size = 0;
        dispatchQ16Block(tensor_64_.get(), [&](auto * /*blocks*/, size_t bs)
                         {
            received_block_size = bs;
            return 0; });

        EXPECT_EQ(received_block_size, 64u);
    }

    TEST_F(Q16SafeBlockAccessTest, DispatchQ16Block_CallsCorrectOverload_Block128)
    {
        size_t received_block_size = 0;
        dispatchQ16Block(tensor_128_.get(), [&](auto * /*blocks*/, size_t bs)
                         {
            received_block_size = bs;
            return 0; });

        EXPECT_EQ(received_block_size, 128u);
    }

    TEST_F(Q16SafeBlockAccessTest, DispatchQ16Block_CanReadFirstBlockScale)
    {
        // This pattern works for any block size
        float scale_32 = dispatchQ16Block(tensor_32_.get(), [](auto *blocks, size_t /*bs*/)
                                          { return blocks[0].d; });

        float scale_64 = dispatchQ16Block(tensor_64_.get(), [](auto *blocks, size_t /*bs*/)
                                          { return blocks[0].d; });

        float scale_128 = dispatchQ16Block(tensor_128_.get(), [](auto *blocks, size_t /*bs*/)
                                           { return blocks[0].d; });

        // All should be valid non-zero scales
        EXPECT_NE(scale_32, 0.0f);
        EXPECT_NE(scale_64, 0.0f);
        EXPECT_NE(scale_128, 0.0f);
    }

    // =============================================================================
    // forEachQ16Block() Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16Block_IteratesAllBlocks_Block32)
    {
        size_t count = 0;
        forEachQ16Block(tensor_32_.get(), [&](const auto & /*block*/, size_t /*idx*/)
                        { ++count; });

        // [4, 64] with block_size=32 -> 2 blocks per row * 4 rows = 8 blocks
        EXPECT_EQ(count, 8u);
    }

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16Block_IteratesAllBlocks_Block64)
    {
        size_t count = 0;
        forEachQ16Block(tensor_64_.get(), [&](const auto & /*block*/, size_t /*idx*/)
                        { ++count; });

        // [4, 64] with block_size=64 -> 1 block per row * 4 rows = 4 blocks
        EXPECT_EQ(count, 4u);
    }

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16Block_IteratesAllBlocks_Block128)
    {
        size_t count = 0;
        forEachQ16Block(tensor_128_.get(), [&](const auto & /*block*/, size_t /*idx*/)
                        { ++count; });

        // [4, 128] with block_size=128 -> 1 block per row * 4 rows = 4 blocks
        EXPECT_EQ(count, 4u);
    }

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16Block_IndicesAreSequential)
    {
        std::vector<size_t> indices;
        forEachQ16Block(tensor_32_.get(), [&](const auto & /*block*/, size_t idx)
                        { indices.push_back(idx); });

        // Should be 0, 1, 2, 3, 4, 5, 6, 7
        ASSERT_EQ(indices.size(), 8u);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            EXPECT_EQ(indices[i], i);
        }
    }

    // =============================================================================
    // blocks_per_row() and total_blocks() Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, BlocksPerRow_CorrectForBlock32)
    {
        // [4, 64] with block_size=32 -> 64/32 = 2 blocks per row
        EXPECT_EQ(tensor_32_->blocks_per_row(), 2u);
    }

    TEST_F(Q16SafeBlockAccessTest, BlocksPerRow_CorrectForBlock64)
    {
        // [4, 64] with block_size=64 -> 64/64 = 1 block per row
        EXPECT_EQ(tensor_64_->blocks_per_row(), 1u);
    }

    TEST_F(Q16SafeBlockAccessTest, BlocksPerRow_CorrectForBlock128)
    {
        // [4, 128] with block_size=128 -> 128/128 = 1 block per row
        EXPECT_EQ(tensor_128_->blocks_per_row(), 1u);
    }

    TEST_F(Q16SafeBlockAccessTest, TotalBlocks_CorrectForAllSizes)
    {
        EXPECT_EQ(tensor_32_->total_blocks(), 8u);  // 4 rows * 2 blocks/row
        EXPECT_EQ(tensor_64_->total_blocks(), 4u);  // 4 rows * 1 block/row
        EXPECT_EQ(tensor_128_->total_blocks(), 4u); // 4 rows * 1 block/row
    }

    // =============================================================================
    // forEachQ16BlockInRow() Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16BlockInRow_IteratesRowBlocks_Block32)
    {
        std::vector<size_t> col_offsets;
        forEachQ16BlockInRow(tensor_32_.get(), 0, [&](const auto & /*block*/, size_t col_offset)
                             { col_offsets.push_back(col_offset); });

        // Row with block_size=32 has 2 blocks at offsets 0 and 32
        ASSERT_EQ(col_offsets.size(), 2u);
        EXPECT_EQ(col_offsets[0], 0u);
        EXPECT_EQ(col_offsets[1], 32u);
    }

    TEST_F(Q16SafeBlockAccessTest, ForEachQ16BlockInRow_IteratesRowBlocks_Block64)
    {
        std::vector<size_t> col_offsets;
        forEachQ16BlockInRow(tensor_64_.get(), 2, [&](const auto & /*block*/, size_t col_offset)
                             { col_offsets.push_back(col_offset); });

        // Row with block_size=64 has 1 block at offset 0
        ASSERT_EQ(col_offsets.size(), 1u);
        EXPECT_EQ(col_offsets[0], 0u);
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    TEST_F(Q16SafeBlockAccessTest, EmptyTensor_HandlesGracefully)
    {
        // Create a minimal valid tensor
        auto small_tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{1, 32}, Q16BlockSize::BLOCK_32);

        EXPECT_EQ(small_tensor->blocks_per_row(), 1u);
        EXPECT_EQ(small_tensor->total_blocks(), 1u);
        EXPECT_NE(small_tensor->as_block_32(), nullptr);
    }

    TEST_F(Q16SafeBlockAccessTest, DispatchQ16BlockMutable_CanModifyBlocks)
    {
        // Modify first block's scale via dispatch
        dispatchQ16BlockMutable(tensor_32_.get(), [](auto *blocks, size_t /*bs*/)
                                {
            blocks[0].d = 12345.0f;
            return 0; });

        // Verify the change persisted
        float scale = dispatchQ16Block(tensor_32_.get(), [](auto *blocks, size_t /*bs*/)
                                       { return blocks[0].d; });

        EXPECT_FLOAT_EQ(scale, 12345.0f);
    }

} // namespace llaminar2::test
