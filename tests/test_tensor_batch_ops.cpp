/**
 * @file test_tensor_batch_ops.cpp
 * @brief Test SimpleTensor batch dimension support
 */

#include "tensors/SimpleTensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>

using namespace llaminar;

TEST(TensorBatchOps, BatchSize_3D)
{
    // Arrange: Create [batch=4, seq=8, hidden=896]
    SimpleTensor tensor({4, 8, 896});
    
    // Act & Assert
    EXPECT_EQ(tensor.batch_size(), 4);
}

TEST(TensorBatchOps, BatchSize_2D)
{
    // Arrange: Create [seq=8, hidden=896] (no explicit batch)
    SimpleTensor tensor({8, 896});
    
    // Act & Assert
    EXPECT_EQ(tensor.batch_size(), 1);  // Default to batch=1
}

TEST(TensorBatchOps, BatchSize_1D)
{
    // Arrange: Create [hidden=896] (single vector)
    SimpleTensor tensor({896});
    
    // Act & Assert
    EXPECT_EQ(tensor.batch_size(), 1);  // Default to batch=1
}

TEST(TensorBatchOps, SeqLen_3D)
{
    // Arrange: Create [batch=4, seq=8, hidden=896]
    SimpleTensor tensor({4, 8, 896});
    
    // Act & Assert
    EXPECT_EQ(tensor.seq_len(), 8);
}

TEST(TensorBatchOps, SeqLen_2D)
{
    // Arrange: Create [seq=8, hidden=896]
    SimpleTensor tensor({8, 896});
    
    // Act & Assert
    EXPECT_EQ(tensor.seq_len(), 8);
}

TEST(TensorBatchOps, ReshapeCopy_Correctness)
{
    // Arrange: Create [2, 3, 4] with known values
    SimpleTensor original({2, 3, 4});
    float* data = original.data();
    for (int i = 0; i < 24; ++i)
    {
        data[i] = static_cast<float>(i);
    }
    
    // Act: Reshape to [24]
    auto reshaped = original.reshape_copy({24});
    
    // Assert: Shape changed
    EXPECT_EQ(reshaped->shape().size(), 1);
    EXPECT_EQ(reshaped->shape()[0], 24);
    
    // Assert: Data preserved
    const float* reshaped_data = reshaped->data();
    for (int i = 0; i < 24; ++i)
    {
        EXPECT_FLOAT_EQ(reshaped_data[i], static_cast<float>(i)) << "Position " << i;
    }
    
    // Assert: Original unchanged
    EXPECT_EQ(original.shape().size(), 3);
    EXPECT_EQ(original.shape()[0], 2);
}

TEST(TensorBatchOps, ReshapeCopy_RowMajor)
{
    // Arrange: Create [2, 3] = [[1,2,3], [4,5,6]]
    SimpleTensor original({2, 3});
    float* data = original.data();
    for (int i = 0; i < 6; ++i)
    {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Act: Reshape to [3, 2]
    auto reshaped = original.reshape_copy({3, 2});
    
    // Assert: Data in row-major order
    const float* reshaped_data = reshaped->data();
    // Original: [1,2,3,4,5,6]
    // Reshaped as [3,2]: [[1,2], [3,4], [5,6]]
    EXPECT_FLOAT_EQ(reshaped_data[0], 1.0f);  // [0,0]
    EXPECT_FLOAT_EQ(reshaped_data[1], 2.0f);  // [0,1]
    EXPECT_FLOAT_EQ(reshaped_data[2], 3.0f);  // [1,0]
    EXPECT_FLOAT_EQ(reshaped_data[3], 4.0f);  // [1,1]
    EXPECT_FLOAT_EQ(reshaped_data[4], 5.0f);  // [2,0]
    EXPECT_FLOAT_EQ(reshaped_data[5], 6.0f);  // [2,1]
}

TEST(TensorBatchOps, SliceBatch_3D)
{
    // Arrange: Create [batch=3, seq=2, hidden=4]
    SimpleTensor batched({3, 2, 4});
    float* data = batched.data();
    
    // Fill with pattern: batch_idx * 100 + position
    for (size_t b = 0; b < 3; ++b)
    {
        for (size_t s = 0; s < 2; ++s)
        {
            for (size_t h = 0; h < 4; ++h)
            {
                size_t idx = b * 8 + s * 4 + h;
                data[idx] = static_cast<float>(b * 100 + idx);
            }
        }
    }
    
    // Act: Extract batch 1
    auto slice = batched.slice_batch(1);
    
    // Assert: Shape is [seq=2, hidden=4]
    EXPECT_EQ(slice->shape().size(), 2);
    EXPECT_EQ(slice->shape()[0], 2);
    EXPECT_EQ(slice->shape()[1], 4);
    
    // Assert: Data from batch 1 only (positions 8-15)
    const float* slice_data = slice->data();
    for (size_t i = 0; i < 8; ++i)
    {
        float expected = static_cast<float>(1 * 100 + (8 + i));  // batch_idx=1, offset=8
        EXPECT_FLOAT_EQ(slice_data[i], expected) << "Position " << i;
    }
}

TEST(TensorBatchOps, SliceBatch_FirstAndLast)
{
    // Arrange
    SimpleTensor batched({4, 2, 2});  // [batch=4, seq=2, hidden=2]
    float* data = batched.data();
    for (int i = 0; i < 16; ++i)
    {
        data[i] = static_cast<float>(i);
    }
    
    // Act: Extract first and last batches
    auto first = batched.slice_batch(0);
    auto last = batched.slice_batch(3);
    
    // Assert: First batch has [0,1,2,3]
    const float* first_data = first->data();
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(first_data[i], static_cast<float>(i));
    }
    
    // Assert: Last batch has [12,13,14,15]
    const float* last_data = last->data();
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(last_data[i], static_cast<float>(12 + i));
    }
}

TEST(TensorBatchOps, StackBatch_Simple)
{
    // Arrange: Create 3 sequences of shape [2, 4]
    auto seq0 = std::make_shared<SimpleTensor>(std::vector<int>{2, 4});
    auto seq1 = std::make_shared<SimpleTensor>(std::vector<int>{2, 4});
    auto seq2 = std::make_shared<SimpleTensor>(std::vector<int>{2, 4});
    
    // Fill with known values
    float* data0 = seq0->data();
    float* data1 = seq1->data();
    float* data2 = seq2->data();
    
    for (int i = 0; i < 8; ++i)
    {
        data0[i] = static_cast<float>(i);           // 0-7
        data1[i] = static_cast<float>(100 + i);     // 100-107
        data2[i] = static_cast<float>(200 + i);     // 200-207
    }
    
    // Act: Stack into batch
    std::vector<std::shared_ptr<SimpleTensor>> sequences = {seq0, seq1, seq2};
    auto batched = SimpleTensor::stack_batch(sequences);
    
    // Assert: Shape is [batch=3, seq=2, hidden=4]
    EXPECT_EQ(batched->shape().size(), 3);
    EXPECT_EQ(batched->shape()[0], 3);  // batch
    EXPECT_EQ(batched->shape()[1], 2);  // seq
    EXPECT_EQ(batched->shape()[2], 4);  // hidden
    
    // Assert: Data layout
    const float* batched_data = batched->data();
    
    // Batch 0: [0-7]
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(batched_data[i], static_cast<float>(i));
    }
    
    // Batch 1: [100-107]
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(batched_data[8 + i], static_cast<float>(100 + i));
    }
    
    // Batch 2: [200-207]
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(batched_data[16 + i], static_cast<float>(200 + i));
    }
}

TEST(TensorBatchOps, StackBatch_SingleSequence)
{
    // Arrange
    auto single = std::make_shared<SimpleTensor>(std::vector<int>{3, 5});
    
    // Act
    auto batched = SimpleTensor::stack_batch({single});
    
    // Assert: Shape is [batch=1, seq=3, hidden=5]
    EXPECT_EQ(batched->shape()[0], 1);
    EXPECT_EQ(batched->shape()[1], 3);
    EXPECT_EQ(batched->shape()[2], 5);
}

TEST(TensorBatchOps, StackBatch_Empty)
{
    // Arrange
    std::vector<std::shared_ptr<SimpleTensor>> empty;
    
    // Act
    auto batched = SimpleTensor::stack_batch(empty);
    
    // Assert: Returns nullptr for empty input
    EXPECT_EQ(batched, nullptr);
}

TEST(TensorBatchOps, SliceAndStack_RoundTrip)
{
    // Arrange: Create original batch
    SimpleTensor original({3, 4, 5});  // [batch=3, seq=4, hidden=5]
    float* data = original.data();
    for (int i = 0; i < 60; ++i)
    {
        data[i] = static_cast<float>(i);
    }
    
    // Act: Slice into individual sequences
    auto seq0 = original.slice_batch(0);
    auto seq1 = original.slice_batch(1);
    auto seq2 = original.slice_batch(2);
    
    // Act: Stack back together
    auto reconstructed = SimpleTensor::stack_batch({seq0, seq1, seq2});
    
    // Assert: Shape matches original
    EXPECT_EQ(reconstructed->shape()[0], 3);
    EXPECT_EQ(reconstructed->shape()[1], 4);
    EXPECT_EQ(reconstructed->shape()[2], 5);
    
    // Assert: Data matches original
    const float* reconstructed_data = reconstructed->data();
    for (int i = 0; i < 60; ++i)
    {
        EXPECT_FLOAT_EQ(reconstructed_data[i], static_cast<float>(i)) << "Position " << i;
    }
}

TEST(TensorBatchOps, BatchSize_AfterStack)
{
    // Arrange
    auto seq1 = std::make_shared<SimpleTensor>(std::vector<int>{8, 896});
    auto seq2 = std::make_shared<SimpleTensor>(std::vector<int>{8, 896});
    auto seq3 = std::make_shared<SimpleTensor>(std::vector<int>{8, 896});
    
    // Act
    auto batched = SimpleTensor::stack_batch({seq1, seq2, seq3});
    
    // Assert
    EXPECT_EQ(batched->batch_size(), 3);
    EXPECT_EQ(batched->seq_len(), 8);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
