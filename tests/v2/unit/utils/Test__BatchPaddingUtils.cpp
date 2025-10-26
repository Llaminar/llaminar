/**
 * @file Test__BatchPaddingUtils.cpp
 * @brief Unit tests for batch padding utilities
 * @author David Sanftenberg
 * @date October 26, 2025
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/utils/BatchPaddingUtils.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include <vector>
#include <cmath>

using namespace llaminar2;

class Test__BatchPaddingUtils : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test: Padding Variable-Length Sequences
// =============================================================================

TEST_F(Test__BatchPaddingUtils, PaddingVariableLengthSequences)
{
    // Input: 3 sequences with different lengths
    std::vector<std::vector<int>> token_sequences = {
        {1, 2, 3},   // length 3
        {4, 5},      // length 2
        {6, 7, 8, 9} // length 4 (longest)
    };

    auto batch = createPaddedBatch(token_sequences, 0);

    // Check batch metadata
    EXPECT_EQ(batch.batch_size, 3);
    EXPECT_EQ(batch.max_length, 4); // Longest sequence
    EXPECT_EQ(batch.actual_lengths.size(), 3);
    EXPECT_EQ(batch.actual_lengths[0], 3);
    EXPECT_EQ(batch.actual_lengths[1], 2);
    EXPECT_EQ(batch.actual_lengths[2], 4);

    // Check tensor shape: [batch_size, max_length]
    ASSERT_NE(batch.tokens, nullptr);
    const auto &shape = batch.tokens->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3); // batch_size
    EXPECT_EQ(shape[1], 4); // max_length

    // Check token values (verify padding)
    const float *data = batch.tokens->data();

    // Sequence 0: [1, 2, 3, 0]
    EXPECT_EQ(static_cast<int>(data[0 * 4 + 0]), 1);
    EXPECT_EQ(static_cast<int>(data[0 * 4 + 1]), 2);
    EXPECT_EQ(static_cast<int>(data[0 * 4 + 2]), 3);
    EXPECT_EQ(static_cast<int>(data[0 * 4 + 3]), 0); // padding

    // Sequence 1: [4, 5, 0, 0]
    EXPECT_EQ(static_cast<int>(data[1 * 4 + 0]), 4);
    EXPECT_EQ(static_cast<int>(data[1 * 4 + 1]), 5);
    EXPECT_EQ(static_cast<int>(data[1 * 4 + 2]), 0); // padding
    EXPECT_EQ(static_cast<int>(data[1 * 4 + 3]), 0); // padding

    // Sequence 2: [6, 7, 8, 9]
    EXPECT_EQ(static_cast<int>(data[2 * 4 + 0]), 6);
    EXPECT_EQ(static_cast<int>(data[2 * 4 + 1]), 7);
    EXPECT_EQ(static_cast<int>(data[2 * 4 + 2]), 8);
    EXPECT_EQ(static_cast<int>(data[2 * 4 + 3]), 9);

    // Check padding mask (1=real, 0=pad)
    EXPECT_EQ(batch.padding_mask.size(), 3 * 4);

    // Sequence 0: [1, 1, 1, 0]
    EXPECT_EQ(batch.padding_mask[0 * 4 + 0], 1);
    EXPECT_EQ(batch.padding_mask[0 * 4 + 1], 1);
    EXPECT_EQ(batch.padding_mask[0 * 4 + 2], 1);
    EXPECT_EQ(batch.padding_mask[0 * 4 + 3], 0);

    // Sequence 1: [1, 1, 0, 0]
    EXPECT_EQ(batch.padding_mask[1 * 4 + 0], 1);
    EXPECT_EQ(batch.padding_mask[1 * 4 + 1], 1);
    EXPECT_EQ(batch.padding_mask[1 * 4 + 2], 0);
    EXPECT_EQ(batch.padding_mask[1 * 4 + 3], 0);

    // Sequence 2: [1, 1, 1, 1]
    EXPECT_EQ(batch.padding_mask[2 * 4 + 0], 1);
    EXPECT_EQ(batch.padding_mask[2 * 4 + 1], 1);
    EXPECT_EQ(batch.padding_mask[2 * 4 + 2], 1);
    EXPECT_EQ(batch.padding_mask[2 * 4 + 3], 1);

    // Test is_padding() helper
    EXPECT_FALSE(batch.is_padding(0, 0));
    EXPECT_FALSE(batch.is_padding(0, 2));
    EXPECT_TRUE(batch.is_padding(0, 3)); // padding
    EXPECT_FALSE(batch.is_padding(1, 1));
    EXPECT_TRUE(batch.is_padding(1, 2));  // padding
    EXPECT_FALSE(batch.is_padding(2, 3)); // no padding
}

TEST_F(Test__BatchPaddingUtils, UniformLengthBatch)
{
    // All sequences have same length (no padding needed)
    std::vector<std::vector<int>> token_sequences = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};

    auto batch = createPaddedBatch(token_sequences, 0);

    EXPECT_EQ(batch.batch_size, 3);
    EXPECT_EQ(batch.max_length, 3);

    // All actual lengths are 3
    for (const auto &len : batch.actual_lengths)
    {
        EXPECT_EQ(len, 3);
    }

    // No padding in mask (all 1s)
    for (const auto &mask_val : batch.padding_mask)
    {
        EXPECT_EQ(mask_val, 1);
    }
}

TEST_F(Test__BatchPaddingUtils, CustomPadToken)
{
    std::vector<std::vector<int>> token_sequences = {
        {1, 2},
        {3, 4, 5}};

    auto batch = createPaddedBatch(token_sequences, 999); // pad_token_id = 999

    const float *data = batch.tokens->data();

    // Sequence 0: [1, 2, 999]
    EXPECT_EQ(static_cast<int>(data[0 * 3 + 0]), 1);
    EXPECT_EQ(static_cast<int>(data[0 * 3 + 1]), 2);
    EXPECT_EQ(static_cast<int>(data[0 * 3 + 2]), 999); // custom pad token
}

TEST_F(Test__BatchPaddingUtils, EmptyBatch)
{
    std::vector<std::vector<int>> token_sequences = {};

    auto batch = createPaddedBatch(token_sequences, 0);

    EXPECT_EQ(batch.batch_size, 0);
    EXPECT_EQ(batch.max_length, 0);
    EXPECT_EQ(batch.actual_lengths.size(), 0);
    EXPECT_EQ(batch.padding_mask.size(), 0);
}

TEST_F(Test__BatchPaddingUtils, SingleSequence)
{
    std::vector<std::vector<int>> token_sequences = {
        {10, 20, 30, 40}};

    auto batch = createPaddedBatch(token_sequences, 0);

    EXPECT_EQ(batch.batch_size, 1);
    EXPECT_EQ(batch.max_length, 4);
    EXPECT_EQ(batch.actual_lengths[0], 4);

    // No padding needed
    for (const auto &mask_val : batch.padding_mask)
    {
        EXPECT_EQ(mask_val, 1);
    }
}

// =============================================================================
// Test: Attention Padding Mask Generation
// =============================================================================

TEST_F(Test__BatchPaddingUtils, AttentionMaskGeneration)
{
    std::vector<int> actual_lengths = {3, 2, 4};
    size_t max_length = 4;

    auto mask = createAttentionPaddingMask(actual_lengths, max_length);

    ASSERT_NE(mask, nullptr);
    const auto &shape = mask->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3); // batch_size
    EXPECT_EQ(shape[1], 4); // max_length

    const float *data = mask->data();

    // Sequence 0 (length=3): [0, 0, 0, -inf]
    EXPECT_EQ(data[0 * 4 + 0], 0.0f);
    EXPECT_EQ(data[0 * 4 + 1], 0.0f);
    EXPECT_EQ(data[0 * 4 + 2], 0.0f);
    EXPECT_TRUE(std::isinf(data[0 * 4 + 3]) && data[0 * 4 + 3] < 0); // -INFINITY

    // Sequence 1 (length=2): [0, 0, -inf, -inf]
    EXPECT_EQ(data[1 * 4 + 0], 0.0f);
    EXPECT_EQ(data[1 * 4 + 1], 0.0f);
    EXPECT_TRUE(std::isinf(data[1 * 4 + 2]) && data[1 * 4 + 2] < 0);
    EXPECT_TRUE(std::isinf(data[1 * 4 + 3]) && data[1 * 4 + 3] < 0);

    // Sequence 2 (length=4): [0, 0, 0, 0]
    EXPECT_EQ(data[2 * 4 + 0], 0.0f);
    EXPECT_EQ(data[2 * 4 + 1], 0.0f);
    EXPECT_EQ(data[2 * 4 + 2], 0.0f);
    EXPECT_EQ(data[2 * 4 + 3], 0.0f);
}

TEST_F(Test__BatchPaddingUtils, AttentionMaskNoMasking)
{
    // All sequences use full length (no masking needed)
    std::vector<int> actual_lengths = {4, 4, 4};
    size_t max_length = 4;

    auto mask = createAttentionPaddingMask(actual_lengths, max_length);

    const float *data = mask->data();

    // All values should be 0.0 (no masking)
    for (size_t i = 0; i < 3 * 4; ++i)
    {
        EXPECT_EQ(data[i], 0.0f);
    }
}

// =============================================================================
// Test: Bucketing Sequences by Length
// =============================================================================

TEST_F(Test__BatchPaddingUtils, BucketingByLength)
{
    std::vector<std::vector<int>> token_sequences = {
        {1, 2},                                                           // length 2 → bucket [0, 8)
        {3, 4, 5, 6, 7},                                                  // length 5 → bucket [0, 8)
        {8, 9, 10, 11, 12, 13, 14, 15, 16},                               // length 9 → bucket [8, 16)
        {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, // length 16 → bucket [8, 16)
        {33, 34, 35}                                                      // length 3 → bucket [0, 8)
    };

    std::vector<size_t> bucket_boundaries = {8, 16, 32};

    auto buckets = bucketSequencesByLength(token_sequences, bucket_boundaries);

    // Expect 2 buckets: [0, 8) and [8, 16)
    EXPECT_EQ(buckets.size(), 2);

    // Bucket 0: lengths 2, 5, 3 (max_length should be 8 after padding to boundary)
    EXPECT_EQ(buckets[0].batch_size, 3);
    EXPECT_EQ(buckets[0].max_length, 8); // Padded to boundary

    // Bucket 1: lengths 9, 16 (max_length should be 16)
    EXPECT_EQ(buckets[1].batch_size, 2);
    EXPECT_EQ(buckets[1].max_length, 16);
}

TEST_F(Test__BatchPaddingUtils, BucketingSingleBucket)
{
    std::vector<std::vector<int>> token_sequences = {
        {1, 2, 3},
        {4, 5},
        {6, 7, 8, 9}};

    std::vector<size_t> bucket_boundaries = {16}; // All fit in first bucket

    auto buckets = bucketSequencesByLength(token_sequences, bucket_boundaries);

    EXPECT_EQ(buckets.size(), 1);
    EXPECT_EQ(buckets[0].batch_size, 3);
    EXPECT_EQ(buckets[0].max_length, 16); // Padded to boundary
}

TEST_F(Test__BatchPaddingUtils, BucketingDefaultBoundaries)
{
    std::vector<std::vector<int>> token_sequences;

    // Create sequences of various lengths
    for (int len : {5, 10, 20, 50, 100, 200, 300, 600})
    {
        std::vector<int> seq(len);
        for (int i = 0; i < len; ++i)
            seq[i] = i;
        token_sequences.push_back(seq);
    }

    // Use default boundaries
    auto buckets = bucketSequencesByLength(token_sequences);

    // Should have multiple buckets
    EXPECT_GT(buckets.size(), 1);

    // Each bucket should have at least one sequence
    for (const auto &bucket : buckets)
    {
        EXPECT_GT(bucket.batch_size, 0);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
