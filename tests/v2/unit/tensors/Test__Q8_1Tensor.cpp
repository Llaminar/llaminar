/**
 * @file Test__Q8_1Tensor.cpp
 * @brief Unit tests for Q8_1Tensor quantized activation storage
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <vector>

using namespace llaminar2;

namespace
{
    std::shared_ptr<Q8_1Tensor> create_empty_q8_1_tensor(int rows, int cols)
    {
        const size_t total_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const size_t total_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q8_1Block), 0);
        return std::make_shared<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            raw);
    }
}

/**
 * @brief Ensure Q8_1Tensor::from_int32_with_scales scales/quantizes activations and preserves precomputed sums.
 */
TEST(Test__Q8_1Tensor, FromInt32WithScalesQuantizesAndStoresSum)
{
    constexpr int rows = 1;
    constexpr int cols = 4;
    auto tensor = create_empty_q8_1_tensor(rows, cols);

    const std::vector<int32_t> accum = {8, -8, 16, -16};
    const std::vector<float> row_scales = {0.5f};
    const std::vector<float> col_scales = {1.0f, 1.0f, 0.5f, 0.5f};
    const std::vector<float> bias = {0.0f, 0.5f, 0.0f, -0.5f};

    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        row_scales.data(),
        col_scales.data(),
        bias.data()));

    std::vector<float> dequant(rows * cols, 0.0f);
    tensor->to_fp32(dequant.data());

    const std::vector<float> expected = {
        4.0f,  // 8 * 0.5 * 1.0 + 0.0
        -3.5f, // -8 * 0.5 * 1.0 + 0.5
        4.0f,  // 16 * 0.5 * 0.5 + 0.0
        -4.5f  // -16 * 0.5 * 0.5 - 0.5
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(dequant[i], expected[i], 0.2f) << "Mismatch at index " << i;
    }

    const Q8_1Block *block = tensor->decode_to_q8_1(0, 0);
    ASSERT_NE(block, nullptr);

    int32_t raw_sum = 0;
    for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
    {
        raw_sum += block->qs[i];
    }
    EXPECT_EQ(block->sum_qs, raw_sum);
    EXPECT_GT(fp16_to_fp32(block->d), 0.0f);
    if (cols < static_cast<int>(Q8_1Block::BLOCK_SIZE))
    {
        EXPECT_EQ(block->qs[cols], 0) << "First padded element should remain zero";
    }
}
