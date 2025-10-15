/**
 * @file test_batch_padding.cpp
 * @brief Test batch padding utilities
 */

#include "BatchPaddingUtils.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace llaminar;
using namespace llaminar::batch;

TEST(BatchPaddingTest, CreatePaddedBatch_SameLength)
{
    // Arrange: Three sequences of same length
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    // Act
    auto padded = createPaddedBatch(sequences);
    
    // Assert
    EXPECT_EQ(padded.batch_size, 3);
    EXPECT_EQ(padded.max_length, 3);
    EXPECT_EQ(padded.actual_lengths.size(), 3);
    EXPECT_EQ(padded.actual_lengths[0], 3);
    EXPECT_EQ(padded.actual_lengths[1], 3);
    EXPECT_EQ(padded.actual_lengths[2], 3);
    
    // Check tokens
    ASSERT_NE(padded.tokens, nullptr);
    const float* data = padded.tokens->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);  // [0, 0]
    EXPECT_FLOAT_EQ(data[1], 2.0f);  // [0, 1]
    EXPECT_FLOAT_EQ(data[2], 3.0f);  // [0, 2]
    EXPECT_FLOAT_EQ(data[3], 4.0f);  // [1, 0]
    EXPECT_FLOAT_EQ(data[4], 5.0f);  // [1, 1]
    EXPECT_FLOAT_EQ(data[5], 6.0f);  // [1, 2]
    
    // Check padding mask (all should be 1 - no padding)
    for (size_t i = 0; i < 9; ++i)
    {
        EXPECT_EQ(padded.padding_mask[i], 1) << "Position " << i;
    }
}

TEST(BatchPaddingTest, CreatePaddedBatch_DifferentLengths)
{
    // Arrange: Sequences with different lengths
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3, 4},      // length 4
        {5, 6},            // length 2
        {7, 8, 9}          // length 3
    };
    
    // Act
    auto padded = createPaddedBatch(sequences);
    
    // Assert
    EXPECT_EQ(padded.batch_size, 3);
    EXPECT_EQ(padded.max_length, 4);  // Max of [4, 2, 3]
    EXPECT_EQ(padded.actual_lengths[0], 4);
    EXPECT_EQ(padded.actual_lengths[1], 2);
    EXPECT_EQ(padded.actual_lengths[2], 3);
    
    // Check tokens
    const float* data = padded.tokens->data();
    
    // Batch 0: [1, 2, 3, 4] - no padding
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
    
    // Batch 1: [5, 6, 0, 0] - padded
    EXPECT_FLOAT_EQ(data[4], 5.0f);
    EXPECT_FLOAT_EQ(data[5], 6.0f);
    EXPECT_FLOAT_EQ(data[6], 0.0f);  // padding
    EXPECT_FLOAT_EQ(data[7], 0.0f);  // padding
    
    // Batch 2: [7, 8, 9, 0] - padded
    EXPECT_FLOAT_EQ(data[8], 7.0f);
    EXPECT_FLOAT_EQ(data[9], 8.0f);
    EXPECT_FLOAT_EQ(data[10], 9.0f);
    EXPECT_FLOAT_EQ(data[11], 0.0f);  // padding
    
    // Check padding mask
    // Batch 0: [1, 1, 1, 1]
    EXPECT_EQ(padded.padding_mask[0], 1);
    EXPECT_EQ(padded.padding_mask[1], 1);
    EXPECT_EQ(padded.padding_mask[2], 1);
    EXPECT_EQ(padded.padding_mask[3], 1);
    
    // Batch 1: [1, 1, 0, 0]
    EXPECT_EQ(padded.padding_mask[4], 1);
    EXPECT_EQ(padded.padding_mask[5], 1);
    EXPECT_EQ(padded.padding_mask[6], 0);  // padding
    EXPECT_EQ(padded.padding_mask[7], 0);  // padding
    
    // Batch 2: [1, 1, 1, 0]
    EXPECT_EQ(padded.padding_mask[8], 1);
    EXPECT_EQ(padded.padding_mask[9], 1);
    EXPECT_EQ(padded.padding_mask[10], 1);
    EXPECT_EQ(padded.padding_mask[11], 0);  // padding
}

TEST(BatchPaddingTest, IsPadding)
{
    std::vector<std::vector<int>> sequences = {
        {1, 2, 3},
        {4, 5}
    };
    
    auto padded = createPaddedBatch(sequences);
    
    // Batch 0: all real
    EXPECT_FALSE(padded.is_padding(0, 0));
    EXPECT_FALSE(padded.is_padding(0, 1));
    EXPECT_FALSE(padded.is_padding(0, 2));
    
    // Batch 1: real + padding
    EXPECT_FALSE(padded.is_padding(1, 0));
    EXPECT_FALSE(padded.is_padding(1, 1));
    EXPECT_TRUE(padded.is_padding(1, 2));  // padding
}

TEST(BatchPaddingTest, AttentionPaddingMask)
{
    std::vector<int> actual_lengths = {3, 2, 4};
    size_t max_length = 4;
    
    auto mask = createAttentionPaddingMask(actual_lengths, max_length);
    
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ(mask->shape()[0], 3);  // batch
    EXPECT_EQ(mask->shape()[1], 4);  // max_length
    
    const float* data = mask->data();
    
    // Batch 0 (length 3): [0, 0, 0, -inf]
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_TRUE(std::isinf(data[3]) && data[3] < 0);
    
    // Batch 1 (length 2): [0, 0, -inf, -inf]
    EXPECT_FLOAT_EQ(data[4], 0.0f);
    EXPECT_FLOAT_EQ(data[5], 0.0f);
    EXPECT_TRUE(std::isinf(data[6]) && data[6] < 0);
    EXPECT_TRUE(std::isinf(data[7]) && data[7] < 0);
    
    // Batch 2 (length 4): [0, 0, 0, 0]
    EXPECT_FLOAT_EQ(data[8], 0.0f);
    EXPECT_FLOAT_EQ(data[9], 0.0f);
    EXPECT_FLOAT_EQ(data[10], 0.0f);
    EXPECT_FLOAT_EQ(data[11], 0.0f);
}

TEST(BatchPaddingTest, EmptySequence)
{
    std::vector<std::vector<int>> empty;
    auto padded = createPaddedBatch(empty);
    
    EXPECT_EQ(padded.batch_size, 0);
    EXPECT_EQ(padded.max_length, 0);
}

TEST(BatchPaddingTest, BucketSequences)
{
    std::vector<std::vector<int>> sequences = {
        {1, 2},           // len 2
        {3, 4, 5},        // len 3
        {6, 7, 8, 9},     // len 4
        {10, 11},         // len 2
        {12, 13, 14, 15, 16, 17, 18, 19, 20}  // len 9
    };
    
    std::vector<size_t> boundaries = {4, 8, 16};
    auto buckets = bucketSequencesByLength(sequences, boundaries);
    
    // Should have 2 buckets:
    // Bucket 0 (≤4): sequences with length 2, 3, 4, 2 = 4 sequences
    // Bucket 1 (4-8): sequence with length 9 = 1 sequence
    EXPECT_EQ(buckets.size(), 2);
    
    if (buckets.size() >= 1)
    {
        EXPECT_EQ(buckets[0].batch_size, 4);  // len ≤4
        EXPECT_EQ(buckets[0].max_length, 4);
    }
    
    if (buckets.size() >= 2)
    {
        EXPECT_EQ(buckets[1].batch_size, 1);  // len 4-8 (actually 9)
        EXPECT_EQ(buckets[1].max_length, 9);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
