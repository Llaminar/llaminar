/**
 * @file Test__Q16HeadNormalization.cpp
 * @brief Unit tests for Q16_1 per-head scale normalization primitives
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include "kernels/cpu/primitives/Q16HeadNormalization.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Decode a Q16_1 block to FP32 values
 */
template <typename BlockType>
void decode_block(const BlockType &block, float *output)
{
    for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
    {
        output[i] = static_cast<float>(block.qs[i]) * block.d;
    }
}

/**
 * @brief Create a Q16_1 block from FP32 values
 */
template <typename BlockType>
void quantize_to_block(const float *input, BlockType &block)
{
    float max_val = 0.0f;
    for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
    {
        max_val = std::max(max_val, std::abs(input[i]));
    }

    if (max_val == 0.0f)
    {
        block.d = 0.0f;
        block.sum_qs = 0;
        std::memset(block.qs, 0, BlockType::BLOCK_SIZE * sizeof(int16_t));
        return;
    }

    // Scale to use full INT16 range
    block.d = max_val / 32767.0f;
    float inv_d = 1.0f / block.d;
    int32_t sum = 0;

    for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
    {
        int32_t val = static_cast<int32_t>(std::round(input[i] * inv_d));
        val = std::max(-32768, std::min(32767, val));
        block.qs[i] = static_cast<int16_t>(val);
        sum += val;
    }
    block.sum_qs = sum;
}

/**
 * @brief Compute MSE between two float arrays
 */
float compute_mse(const float *a, const float *b, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / count);
}

/**
 * @brief Compute cosine similarity between two float arrays
 */
float compute_cosine_similarity(const float *a, const float *b, size_t count)
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a == 0.0 || norm_b == 0.0)
        return 0.0f;
    return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

// ============================================================================
// Test Fixtures
// ============================================================================

class Q16HeadNormalizationTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    template <typename BlockType>
    void fill_block_random(BlockType &block, float scale_range = 1.0f)
    {
        std::uniform_real_distribution<float> dist(-scale_range, scale_range);
        std::vector<float> values(BlockType::BLOCK_SIZE);
        for (auto &v : values)
        {
            v = dist(rng_);
        }
        quantize_to_block(values.data(), block);
    }

    template <typename BlockType>
    void fill_blocks_random(std::vector<BlockType> &blocks, float scale_range = 1.0f)
    {
        for (auto &block : blocks)
        {
            fill_block_random(block, scale_range);
        }
    }
};

// ============================================================================
// find_head_max_scale Tests
// ============================================================================

TEST_F(Q16HeadNormalizationTest, FindMaxScale_SingleBlock_ReturnsBlockScale)
{
    Q16_1Block_128 block;
    fill_block_random(block, 2.5f);

    float max_scale = find_head_max_scale(&block, 1);
    EXPECT_FLOAT_EQ(max_scale, std::abs(block.d));
}

TEST_F(Q16HeadNormalizationTest, FindMaxScale_MultipleBlocks_ReturnsMax)
{
    std::vector<Q16_1Block> blocks(4);
    blocks[0].d = 0.5f;
    blocks[1].d = 2.0f;  // Max
    blocks[2].d = -1.5f; // Negative scale
    blocks[3].d = 0.8f;

    float max_scale = find_head_max_scale(blocks.data(), blocks.size());
    EXPECT_FLOAT_EQ(max_scale, 2.0f);
}

TEST_F(Q16HeadNormalizationTest, FindMaxScale_NegativeScales_ReturnsAbsMax)
{
    std::vector<Q16_1Block_64> blocks(3);
    blocks[0].d = -3.0f; // Max absolute
    blocks[1].d = 2.0f;
    blocks[2].d = -1.0f;

    float max_scale = find_head_max_scale(blocks.data(), blocks.size());
    EXPECT_FLOAT_EQ(max_scale, 3.0f);
}

TEST_F(Q16HeadNormalizationTest, FindMaxScale_AllZero_ReturnsZero)
{
    std::vector<Q16_1Block_64> blocks(2);
    blocks[0].d = 0.0f;
    blocks[1].d = 0.0f;

    float max_scale = find_head_max_scale(blocks.data(), blocks.size());
    EXPECT_FLOAT_EQ(max_scale, 0.0f);
}

TEST_F(Q16HeadNormalizationTest, FindMaxScale_Empty_ReturnsZero)
{
    float max_scale = find_head_max_scale<Q16_1Block>(nullptr, 0);
    EXPECT_FLOAT_EQ(max_scale, 0.0f);
}

TEST_F(Q16HeadNormalizationTest, FindMaxScale_AllBlockTypes)
{
    // Test with all block sizes
    {
        std::vector<Q16_1Block> blocks(4);
        fill_blocks_random(blocks, 1.0f);
        float max_scale = find_head_max_scale(blocks.data(), blocks.size());
        EXPECT_GT(max_scale, 0.0f);
    }
    {
        std::vector<Q16_1Block_64> blocks(2);
        fill_blocks_random(blocks, 1.0f);
        float max_scale = find_head_max_scale(blocks.data(), blocks.size());
        EXPECT_GT(max_scale, 0.0f);
    }
    {
        std::vector<Q16_1Block_128> blocks(2);
        fill_blocks_random(blocks, 1.0f);
        float max_scale = find_head_max_scale(blocks.data(), blocks.size());
        EXPECT_GT(max_scale, 0.0f);
    }
}

// ============================================================================
// requantize_blocks_to_scale Tests
// ============================================================================

TEST_F(Q16HeadNormalizationTest, Requantize_SingleBlock_JustUpdatesScale)
{
    Q16_1Block_64 block;
    fill_block_random(block, 1.5f);

    // Save original values
    std::vector<float> original(Q16_1Block_64::BLOCK_SIZE);
    decode_block(block, original.data());

    float head_scale = std::abs(block.d);
    requantize_blocks_to_scale(&block, 1, head_scale);

    // Decode after requantization
    std::vector<float> after(Q16_1Block_64::BLOCK_SIZE);
    decode_block(block, after.data());

    // Values should be nearly identical
    float mse = compute_mse(original.data(), after.data(), Q16_1Block_64::BLOCK_SIZE);
    EXPECT_LT(mse, 1e-8f) << "Single block requantization should preserve values";
    EXPECT_FLOAT_EQ(block.d, head_scale);
}

TEST_F(Q16HeadNormalizationTest, Requantize_MultipleBlocks_PreservesValues)
{
    // 128-dim head with 4 x 32-element blocks
    constexpr size_t NUM_BLOCKS = 4;
    std::vector<Q16_1Block> blocks(NUM_BLOCKS);
    fill_blocks_random(blocks, 2.0f);

    // Save original decoded values
    std::vector<float> original(NUM_BLOCKS * Q16_1Block::BLOCK_SIZE);
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        decode_block(blocks[i], original.data() + i * Q16_1Block::BLOCK_SIZE);
    }

    // Normalize
    float head_scale = find_head_max_scale(blocks.data(), NUM_BLOCKS);
    requantize_blocks_to_scale(blocks.data(), NUM_BLOCKS, head_scale);

    // Decode after normalization
    std::vector<float> after(NUM_BLOCKS * Q16_1Block::BLOCK_SIZE);
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        decode_block(blocks[i], after.data() + i * Q16_1Block::BLOCK_SIZE);
    }

    // Check cosine similarity (should be very high)
    float cosine = compute_cosine_similarity(original.data(), after.data(),
                                             NUM_BLOCKS * Q16_1Block::BLOCK_SIZE);
    EXPECT_GT(cosine, 0.9999f) << "Normalization should preserve values";

    // All blocks should have the same scale
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        EXPECT_FLOAT_EQ(blocks[i].d, head_scale) << "Block " << i << " scale mismatch";
    }
}

TEST_F(Q16HeadNormalizationTest, Requantize_AllBlocksUnified_SameScale)
{
    // 128-dim head with 2 x 64-element blocks
    std::vector<Q16_1Block_64> blocks(2);
    fill_blocks_random(blocks, 1.0f);

    float head_scale = find_head_max_scale(blocks.data(), blocks.size());
    requantize_blocks_to_scale(blocks.data(), blocks.size(), head_scale);

    EXPECT_FLOAT_EQ(blocks[0].d, head_scale);
    EXPECT_FLOAT_EQ(blocks[1].d, head_scale);
}

TEST_F(Q16HeadNormalizationTest, Requantize_ZeroBlock_HandledGracefully)
{
    std::vector<Q16_1Block> blocks(2);
    fill_block_random(blocks[0], 1.0f);

    // Second block is all zeros
    blocks[1].d = 0.0f;
    blocks[1].sum_qs = 0;
    std::memset(blocks[1].qs, 0, sizeof(blocks[1].qs));

    float head_scale = find_head_max_scale(blocks.data(), blocks.size());
    requantize_blocks_to_scale(blocks.data(), blocks.size(), head_scale);

    // Both blocks should have head_scale
    EXPECT_FLOAT_EQ(blocks[0].d, head_scale);
    EXPECT_FLOAT_EQ(blocks[1].d, head_scale);

    // Second block should still be all zeros
    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
    {
        EXPECT_EQ(blocks[1].qs[i], 0);
    }
}

// ============================================================================
// normalize_q16_head Tests
// ============================================================================

TEST_F(Q16HeadNormalizationTest, Normalize_SingleBlock_OptimalPath)
{
    Q16_1Block_128 block;
    fill_block_random(block, 3.0f);

    float original_scale = std::abs(block.d);
    float head_scale = normalize_q16_head(&block, 1);

    // Should return the original scale without modification
    EXPECT_FLOAT_EQ(head_scale, original_scale);
}

TEST_F(Q16HeadNormalizationTest, Normalize_MultipleBlocks_AllUnified)
{
    constexpr size_t NUM_BLOCKS = 4;
    std::vector<Q16_1Block> blocks(NUM_BLOCKS);
    fill_blocks_random(blocks, 2.0f);

    float head_scale = normalize_q16_head(blocks.data(), NUM_BLOCKS);

    // All blocks should have unified scale
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        EXPECT_FLOAT_EQ(blocks[i].d, head_scale);
    }
}

TEST_F(Q16HeadNormalizationTest, Normalize_PreservesSum_Approximately)
{
    constexpr size_t NUM_BLOCKS = 2;
    std::vector<Q16_1Block_64> blocks(NUM_BLOCKS);
    fill_blocks_random(blocks, 1.5f);

    // Compute original element sum
    double original_sum = 0.0;
    for (const auto &block : blocks)
    {
        for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
        {
            original_sum += static_cast<double>(block.qs[i]) * block.d;
        }
    }

    normalize_q16_head(blocks.data(), NUM_BLOCKS);

    // Compute normalized element sum
    double normalized_sum = 0.0;
    for (const auto &block : blocks)
    {
        for (size_t i = 0; i < Q16_1Block_64::BLOCK_SIZE; ++i)
        {
            normalized_sum += static_cast<double>(block.qs[i]) * block.d;
        }
    }

    // Sums should be approximately equal
    double rel_error = std::abs(original_sum - normalized_sum) /
                       std::max(1e-10, std::abs(original_sum));
    EXPECT_LT(rel_error, 0.01) << "Normalization changed sum by more than 1%";
}

// Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
// with independent scales, not a combined 192-dim block.

// ============================================================================
// needs_normalization Tests
// ============================================================================

TEST_F(Q16HeadNormalizationTest, NeedsNormalization_SingleBlock_ReturnsFalse)
{
    Q16_1Block_128 block;
    fill_block_random(block, 1.0f);

    EXPECT_FALSE(needs_normalization(&block, 1));
}

TEST_F(Q16HeadNormalizationTest, NeedsNormalization_DifferentScales_ReturnsTrue)
{
    std::vector<Q16_1Block> blocks(3);
    blocks[0].d = 1.0f;
    blocks[1].d = 2.0f; // Different
    blocks[2].d = 1.0f;

    EXPECT_TRUE(needs_normalization(blocks.data(), blocks.size()));
}

TEST_F(Q16HeadNormalizationTest, NeedsNormalization_SameScales_ReturnsFalse)
{
    std::vector<Q16_1Block_64> blocks(2);
    blocks[0].d = 1.5f;
    blocks[1].d = 1.5f;

    EXPECT_FALSE(needs_normalization(blocks.data(), blocks.size()));
}

TEST_F(Q16HeadNormalizationTest, NeedsNormalization_WithTolerance)
{
    std::vector<Q16_1Block> blocks(2);
    blocks[0].d = 1.0f;
    blocks[1].d = 1.000001f; // Within tolerance

    EXPECT_FALSE(needs_normalization(blocks.data(), blocks.size(), 1e-5f));
    EXPECT_TRUE(needs_normalization(blocks.data(), blocks.size(), 1e-8f));
}

TEST_F(Q16HeadNormalizationTest, NeedsNormalization_AfterNormalize_ReturnsFalse)
{
    std::vector<Q16_1Block> blocks(4);
    fill_blocks_random(blocks, 2.0f);

    // Before normalization, scales likely differ
    bool before = needs_normalization(blocks.data(), blocks.size());
    // (May or may not be true depending on random values)

    normalize_q16_head(blocks.data(), blocks.size());

    // After normalization, should always be false
    EXPECT_FALSE(needs_normalization(blocks.data(), blocks.size()));
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(Q16HeadNormalizationTest, BlocksPerHead_Calculations)
{
    EXPECT_EQ(blocks_per_head(64, 32), 2u);
    EXPECT_EQ(blocks_per_head(64, 64), 1u);
    EXPECT_EQ(blocks_per_head(128, 32), 4u);
    EXPECT_EQ(blocks_per_head(128, 64), 2u);
    EXPECT_EQ(blocks_per_head(128, 128), 1u);
    EXPECT_EQ(blocks_per_head(192, 32), 6u);
    EXPECT_EQ(blocks_per_head(192, 64), 3u);
    EXPECT_EQ(blocks_per_head(192, 192), 1u);
}

TEST_F(Q16HeadNormalizationTest, IsOptimalBlockSize_Calculations)
{
    // Optimal: head_dim == block_size
    EXPECT_TRUE(is_optimal_block_size(64, 64));
    EXPECT_TRUE(is_optimal_block_size(128, 128));
    EXPECT_TRUE(is_optimal_block_size(192, 192));

    // Not optimal: head_dim != block_size
    EXPECT_FALSE(is_optimal_block_size(64, 32));
    EXPECT_FALSE(is_optimal_block_size(128, 64));
    EXPECT_FALSE(is_optimal_block_size(192, 64));
}

// ============================================================================
// Precision Tests (Critical for Integer Attention)
// ============================================================================

TEST_F(Q16HeadNormalizationTest, Precision_LargeScaleDifference)
{
    // Test with blocks having moderately different scales (realistic case)
    // Note: Extreme scale differences (e.g., 0.001 vs 100.0) will cause
    // precision loss as small values get quantized to near-zero after
    // normalization. This is expected behavior.
    std::vector<Q16_1Block_64> blocks(2);

    // Block 0: moderate values
    {
        std::vector<float> values(64, 1.0f);
        quantize_to_block(values.data(), blocks[0]);
    }

    // Block 1: larger values (10× difference - realistic for attention)
    {
        std::vector<float> values(64, 10.0f);
        quantize_to_block(values.data(), blocks[1]);
    }

    // Save original decoded values
    std::vector<float> original(128);
    decode_block(blocks[0], original.data());
    decode_block(blocks[1], original.data() + 64);

    normalize_q16_head(blocks.data(), 2);

    // Decode after normalization
    std::vector<float> after(128);
    decode_block(blocks[0], after.data());
    decode_block(blocks[1], after.data() + 64);

    // The large-scale block should be perfectly preserved
    float cosine_large = compute_cosine_similarity(
        original.data() + 64, after.data() + 64, 64);
    EXPECT_GT(cosine_large, 0.9999f);

    // The smaller-scale block should still be well preserved with 10× difference
    float cosine_small = compute_cosine_similarity(
        original.data(), after.data(), 64);
    EXPECT_GT(cosine_small, 0.999f); // Good precision with 10× scale ratio
}

TEST_F(Q16HeadNormalizationTest, Precision_TypicalDistribution)
{
    // Test with realistic value distributions
    std::vector<Q16_1Block_128> blocks(2);

    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &block : blocks)
    {
        std::vector<float> values(128);
        for (auto &v : values)
        {
            v = dist(rng_);
        }
        quantize_to_block(values.data(), block);
    }

    // Save original
    std::vector<float> original(256);
    decode_block(blocks[0], original.data());
    decode_block(blocks[1], original.data() + 128);

    normalize_q16_head(blocks.data(), 2);

    // Decode after
    std::vector<float> after(256);
    decode_block(blocks[0], after.data());
    decode_block(blocks[1], after.data() + 128);

    float cosine = compute_cosine_similarity(original.data(), after.data(), 256);
    EXPECT_GT(cosine, 0.9999f) << "Typical distribution should have excellent precision";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Q16HeadNormalizationTest, EdgeCase_NullPointer)
{
    EXPECT_FLOAT_EQ(find_head_max_scale<Q16_1Block>(nullptr, 5), 0.0f);
    EXPECT_NO_THROW(requantize_blocks_to_scale<Q16_1Block>(nullptr, 5, 1.0f));
    EXPECT_FLOAT_EQ(normalize_q16_head<Q16_1Block>(nullptr, 5), 0.0f);
    EXPECT_FALSE(needs_normalization<Q16_1Block>(nullptr, 5));
}

TEST_F(Q16HeadNormalizationTest, EdgeCase_ZeroNumBlocks)
{
    Q16_1Block block;
    EXPECT_FLOAT_EQ(find_head_max_scale(&block, 0), 0.0f);
    EXPECT_NO_THROW(requantize_blocks_to_scale(&block, 0, 1.0f));
    EXPECT_FLOAT_EQ(normalize_q16_head(&block, 0), 0.0f);
    EXPECT_FALSE(needs_normalization(&block, 0));
}

TEST_F(Q16HeadNormalizationTest, EdgeCase_ZeroHeadScale)
{
    Q16_1Block_64 block;
    block.d = 1.0f;
    std::fill(block.qs, block.qs + 64, static_cast<int16_t>(100));

    // Passing zero head_scale should be a no-op
    EXPECT_NO_THROW(requantize_blocks_to_scale(&block, 1, 0.0f));
    EXPECT_FLOAT_EQ(block.d, 1.0f); // Unchanged
}

TEST_F(Q16HeadNormalizationTest, EdgeCase_MaxINT16Values)
{
    Q16_1Block block;
    block.d = 1.0f;
    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
    {
        block.qs[i] = std::numeric_limits<int16_t>::max();
    }
    block.sum_qs = static_cast<int32_t>(Q16_1Block::BLOCK_SIZE) * std::numeric_limits<int16_t>::max();

    // Should handle without overflow
    EXPECT_NO_THROW(requantize_blocks_to_scale(&block, 1, 2.0f));

    // Check sum_qs is consistent
    int32_t expected_sum = 0;
    for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
    {
        expected_sum += block.qs[i];
    }
    EXPECT_EQ(block.sum_qs, expected_sum);
}
