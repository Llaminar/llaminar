/**
 * @file Test__Q8_1_BlockAlignedSlicing.cpp
 * @brief Unit tests for Q8_1Tensor::slice_k_blocks() - column-slicing at block boundaries
 * @author David Sanftenberg
 *
 * Tests the block-aligned k-slicing functionality used for tensor-parallel GEMM
 * where k-dimension is split across MPI ranks. The k-slice must occur at
 * 32-element block boundaries to preserve Q8_1 block structure.
 *
 * Phase 6.4: Q8_1 Block-Aligned Slicing
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

using namespace llaminar2;

namespace
{
    /**
     * @brief Create a Q8_1Tensor with known quantized values for testing
     * @param rows Number of rows
     * @param cols Number of columns (must be multiple of BLOCK_SIZE for clean tests)
     * @param seed Random seed for reproducibility
     *
     * Quantizes FP32 values into Q8_1 blocks with proper scale and sum_qs computation.
     */
    std::shared_ptr<Q8_1Tensor> create_test_q8_1_tensor(int rows, int cols, int seed = 42)
    {
        // Generate FP32 source data
        std::vector<float> fp32_data(static_cast<size_t>(rows) * cols);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : fp32_data)
        {
            val = dist(rng);
        }

        // Calculate number of blocks
        const size_t total_elements = static_cast<size_t>(rows) * cols;
        const size_t total_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const size_t blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

        // Allocate raw block storage
        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q8_1Block));
        Q8_1Block *blocks = reinterpret_cast<Q8_1Block *>(raw_data.data());

        // Quantize each block
        for (int r = 0; r < rows; ++r)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                size_t block_idx = r * blocks_per_row + b;
                Q8_1Block &block = blocks[block_idx];

                // Find max absolute value in this block for scale computation
                float max_abs = 0.0f;
                size_t elem_start = r * cols + b * Q8_1Block::BLOCK_SIZE;
                size_t elem_count = std::min(static_cast<size_t>(Q8_1Block::BLOCK_SIZE),
                                             static_cast<size_t>(cols) - b * Q8_1Block::BLOCK_SIZE);

                for (size_t i = 0; i < elem_count; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(fp32_data[elem_start + i]));
                }

                // Compute scale (scale = max_abs / 127)
                float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
                block.d = fp32_to_fp16(scale);

                // Quantize values and compute sum
                int32_t sum = 0;
                for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
                {
                    if (i < elem_count)
                    {
                        float val = fp32_data[elem_start + i];
                        int8_t qval = static_cast<int8_t>(std::round(val / scale));
                        qval = std::max(static_cast<int8_t>(-127), std::min(static_cast<int8_t>(127), qval));
                        block.qs[i] = qval;
                    }
                    else
                    {
                        block.qs[i] = 0; // Padding
                    }
                    sum += block.qs[i];
                }
                block.sum_qs = static_cast<int16_t>(sum);
            }
        }

        return std::make_shared<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            raw_data);
    }

    /**
     * @brief Compute mean absolute error between two float arrays
     */
    float compute_mae(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(a[i] - b[i]);
        }
        return static_cast<float>(sum / n);
    }
}

/**
 * @brief Test is_k_aligned() static helper
 */
TEST(Test__Q8_1BlockAlignedSlicing, IsKAligned)
{
    // Block size is 32
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(0));
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(32));
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(64));
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(128));
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(1024));

    EXPECT_FALSE(Q8_1Tensor::is_k_aligned(1));
    EXPECT_FALSE(Q8_1Tensor::is_k_aligned(16));
    EXPECT_FALSE(Q8_1Tensor::is_k_aligned(31));
    EXPECT_FALSE(Q8_1Tensor::is_k_aligned(33));
    EXPECT_FALSE(Q8_1Tensor::is_k_aligned(63));
}

/**
 * @brief Test slice_k_blocks returns nullptr for unaligned k_start
 */
TEST(Test__Q8_1BlockAlignedSlicing, RejectsUnalignedKStart)
{
    constexpr int rows = 4;
    constexpr int cols = 128; // 4 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Try unaligned k_start values - should return nullptr
    EXPECT_EQ(tensor->slice_k_blocks(1, 32), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(16, 32), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(31, 32), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(33, 64), nullptr);
}

/**
 * @brief Test slice_k_blocks returns nullptr for unaligned k_size
 */
TEST(Test__Q8_1BlockAlignedSlicing, RejectsUnalignedKSize)
{
    constexpr int rows = 4;
    constexpr int cols = 128; // 4 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Try unaligned k_size values - should return nullptr
    EXPECT_EQ(tensor->slice_k_blocks(0, 1), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(0, 16), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(0, 31), nullptr);
    EXPECT_EQ(tensor->slice_k_blocks(32, 33), nullptr);
}

/**
 * @brief Test slice_k_blocks returns nullptr for out-of-bounds slice
 */
TEST(Test__Q8_1BlockAlignedSlicing, RejectsOutOfBounds)
{
    constexpr int rows = 4;
    constexpr int cols = 128; // 4 blocks (k=128)

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // k_start + k_size > cols
    EXPECT_EQ(tensor->slice_k_blocks(128, 32), nullptr); // Start at end
    EXPECT_EQ(tensor->slice_k_blocks(96, 64), nullptr);  // 96+64=160 > 128
    EXPECT_EQ(tensor->slice_k_blocks(0, 256), nullptr);  // Way too large
}

/**
 * @brief Test basic k-slice: first half of columns
 */
TEST(Test__Q8_1BlockAlignedSlicing, SliceFirstHalf)
{
    constexpr int rows = 4;
    constexpr int cols = 128; // 4 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Slice first 64 columns (2 blocks)
    auto sliced = tensor->slice_k_blocks(0, 64);
    ASSERT_NE(sliced, nullptr);

    // Check shape
    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), 64);

    // Dequantize both and compare overlapping region
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * 64);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    // First 64 columns should match
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < 64; ++c)
        {
            float orig = original_fp32[r * cols + c];
            float slice = sliced_fp32[r * 64 + c];
            EXPECT_NEAR(slice, orig, 1e-5f)
                << "Mismatch at row=" << r << ", col=" << c;
        }
    }
}

/**
 * @brief Test k-slice: second half of columns
 */
TEST(Test__Q8_1BlockAlignedSlicing, SliceSecondHalf)
{
    constexpr int rows = 4;
    constexpr int cols = 128; // 4 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Slice columns 64-127 (last 2 blocks)
    auto sliced = tensor->slice_k_blocks(64, 64);
    ASSERT_NE(sliced, nullptr);

    // Check shape
    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), 64);

    // Dequantize both and compare
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * 64);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    // Columns 64-127 in original should match columns 0-63 in sliced
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < 64; ++c)
        {
            float orig = original_fp32[r * cols + 64 + c];
            float slice = sliced_fp32[r * 64 + c];
            EXPECT_NEAR(slice, orig, 1e-5f)
                << "Mismatch at row=" << r << ", orig_col=" << (64 + c);
        }
    }
}

/**
 * @brief Test k-slice: middle blocks
 */
TEST(Test__Q8_1BlockAlignedSlicing, SliceMiddleBlocks)
{
    constexpr int rows = 4;
    constexpr int cols = 256; // 8 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Slice columns 64-191 (4 middle blocks)
    auto sliced = tensor->slice_k_blocks(64, 128);
    ASSERT_NE(sliced, nullptr);

    // Check shape
    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), 128);

    // Dequantize both and compare
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * 128);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    // Columns 64-191 in original should match columns 0-127 in sliced
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < 128; ++c)
        {
            float orig = original_fp32[r * cols + 64 + c];
            float slice = sliced_fp32[r * 128 + c];
            EXPECT_NEAR(slice, orig, 1e-5f)
                << "Mismatch at row=" << r << ", orig_col=" << (64 + c);
        }
    }
}

/**
 * @brief Test k-slice: single block
 */
TEST(Test__Q8_1BlockAlignedSlicing, SliceSingleBlock)
{
    constexpr int rows = 8;
    constexpr int cols = 128; // 4 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Slice just the third block (columns 64-95)
    auto sliced = tensor->slice_k_blocks(64, 32);
    ASSERT_NE(sliced, nullptr);

    // Check shape
    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), 32);

    // Dequantize both and compare
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * 32);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    // Columns 64-95 in original should match columns 0-31 in sliced
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < 32; ++c)
        {
            float orig = original_fp32[r * cols + 64 + c];
            float slice = sliced_fp32[r * 32 + c];
            EXPECT_NEAR(slice, orig, 1e-5f)
                << "Mismatch at row=" << r << ", orig_col=" << (64 + c);
        }
    }
}

/**
 * @brief Test k-slice: entire tensor (k_start=0, k_size=cols)
 */
TEST(Test__Q8_1BlockAlignedSlicing, SliceEntireTensor)
{
    constexpr int rows = 4;
    constexpr int cols = 128;

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Slice entire tensor
    auto sliced = tensor->slice_k_blocks(0, cols);
    ASSERT_NE(sliced, nullptr);

    // Check shape
    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), cols);

    // Dequantize both and compare all values
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * cols);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    float mae = compute_mae(original_fp32.data(), sliced_fp32.data(), rows * cols);
    EXPECT_LT(mae, 1e-5f) << "Full slice should match original exactly";
}

/**
 * @brief Test k-slice: realistic tensor parallel dimensions
 *
 * Simulates tensor-parallel split of d_model=896 across 2 ranks
 * Each rank gets k=448 (14 blocks of 32)
 */
TEST(Test__Q8_1BlockAlignedSlicing, TensorParallelDimensions)
{
    constexpr int rows = 32;  // Batch of tokens
    constexpr int cols = 896; // d_model for Qwen2.5 0.5B
    constexpr int world_size = 2;
    constexpr int k_local = cols / world_size; // 448

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Verify alignment
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(0));
    EXPECT_TRUE(Q8_1Tensor::is_k_aligned(k_local)); // 448 % 32 == 0

    // Slice for rank 0: columns 0-447
    auto rank0_slice = tensor->slice_k_blocks(0, k_local);
    ASSERT_NE(rank0_slice, nullptr);
    EXPECT_EQ(rank0_slice->decoder_cols(), k_local);

    // Slice for rank 1: columns 448-895
    auto rank1_slice = tensor->slice_k_blocks(k_local, k_local);
    ASSERT_NE(rank1_slice, nullptr);
    EXPECT_EQ(rank1_slice->decoder_cols(), k_local);

    // Dequantize all three
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> rank0_fp32(rows * k_local);
    std::vector<float> rank1_fp32(rows * k_local);

    tensor->to_fp32(original_fp32.data());
    rank0_slice->to_fp32(rank0_fp32.data());
    rank1_slice->to_fp32(rank1_fp32.data());

    // Verify rank 0 slice matches columns 0-447
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < k_local; ++c)
        {
            EXPECT_NEAR(rank0_fp32[r * k_local + c],
                        original_fp32[r * cols + c], 1e-5f)
                << "Rank 0 mismatch at row=" << r << ", col=" << c;
        }
    }

    // Verify rank 1 slice matches columns 448-895
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < k_local; ++c)
        {
            EXPECT_NEAR(rank1_fp32[r * k_local + c],
                        original_fp32[r * cols + k_local + c], 1e-5f)
                << "Rank 1 mismatch at row=" << r << ", col=" << c;
        }
    }
}

/**
 * @brief Test k-slice: 4-way tensor parallel split
 *
 * Simulates tensor-parallel split of d_model=4096 across 4 ranks
 */
TEST(Test__Q8_1BlockAlignedSlicing, FourWayTensorParallel)
{
    constexpr int rows = 16;
    constexpr int cols = 4096; // Typical d_model for 7B model
    constexpr int world_size = 4;
    constexpr int k_local = cols / world_size; // 1024

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    // Verify alignment for all ranks
    for (int rank = 0; rank < world_size; ++rank)
    {
        size_t k_start = rank * k_local;
        EXPECT_TRUE(Q8_1Tensor::is_k_aligned(k_start))
            << "k_start=" << k_start << " should be aligned";
    }

    // Dequantize original
    std::vector<float> original_fp32(rows * cols);
    tensor->to_fp32(original_fp32.data());

    // Test each rank's slice
    for (int rank = 0; rank < world_size; ++rank)
    {
        size_t k_start = rank * k_local;
        auto slice = tensor->slice_k_blocks(k_start, k_local);
        ASSERT_NE(slice, nullptr) << "Failed to slice for rank " << rank;
        EXPECT_EQ(slice->decoder_cols(), k_local);

        std::vector<float> slice_fp32(rows * k_local);
        slice->to_fp32(slice_fp32.data());

        // Verify slice matches expected columns
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < k_local; ++c)
            {
                EXPECT_NEAR(slice_fp32[r * k_local + c],
                            original_fp32[r * cols + k_start + c], 1e-5f)
                    << "Rank " << rank << " mismatch at row=" << r << ", col=" << c;
            }
        }
    }
}

/**
 * @brief Test that sliced tensor's blocks are valid Q8_1 blocks
 */
TEST(Test__Q8_1BlockAlignedSlicing, BlockIntegrity)
{
    constexpr int rows = 4;
    constexpr int cols = 128;

    auto tensor = create_test_q8_1_tensor(rows, cols);
    ASSERT_NE(tensor, nullptr);

    auto sliced = tensor->slice_k_blocks(32, 64); // 2 middle blocks
    ASSERT_NE(sliced, nullptr);

    // Sliced tensor has 2 blocks per row (64/32)
    constexpr size_t sliced_blocks_per_row = 64 / Q8_1Block::BLOCK_SIZE; // 2

    // Access blocks directly via decode_to_q8_1
    for (int r = 0; r < rows; ++r)
    {
        for (size_t b = 0; b < sliced_blocks_per_row; ++b)
        {
            // decode_to_q8_1 takes block INDEX (0, 1, 2...), not column offset
            const Q8_1Block *block = sliced->decode_to_q8_1(r, b);
            ASSERT_NE(block, nullptr) << "Block at row=" << r << ", block=" << b;

            // Verify sum_qs matches actual sum of qs values
            int32_t computed_sum = 0;
            for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                computed_sum += block->qs[i];
            }
            EXPECT_EQ(block->sum_qs, static_cast<int16_t>(computed_sum))
                << "sum_qs mismatch at row=" << r << ", block=" << b;

            // d should be positive (non-zero scale)
            EXPECT_GT(fp16_to_fp32(block->d), 0.0f)
                << "Zero scale at row=" << r << ", block=" << b;
        }
    }
}

/**
 * @brief Performance-oriented test: large tensor slicing
 */
TEST(Test__Q8_1BlockAlignedSlicing, LargeTensorSlicing)
{
    constexpr int rows = 512;
    constexpr int cols = 4096; // 128 blocks

    auto tensor = create_test_q8_1_tensor(rows, cols, 123);
    ASSERT_NE(tensor, nullptr);

    // Slice half (2048 columns = 64 blocks)
    auto sliced = tensor->slice_k_blocks(1024, 2048);
    ASSERT_NE(sliced, nullptr);

    EXPECT_EQ(sliced->decoder_rows(), rows);
    EXPECT_EQ(sliced->decoder_cols(), 2048);

    // Spot-check a few values
    std::vector<float> original_fp32(rows * cols);
    std::vector<float> sliced_fp32(rows * 2048);

    tensor->to_fp32(original_fp32.data());
    sliced->to_fp32(sliced_fp32.data());

    // Check corners and middle
    const std::vector<std::pair<int, int>> check_points = {
        {0, 0}, {0, 1023}, {0, 2047}, {255, 512}, {255, 1500}, {511, 0}, {511, 2047}};

    for (const auto &[r, c] : check_points)
    {
        float orig = original_fp32[r * cols + 1024 + c];
        float slice = sliced_fp32[r * 2048 + c];
        EXPECT_NEAR(slice, orig, 1e-5f)
            << "Large tensor mismatch at row=" << r << ", slice_col=" << c;
    }
}
