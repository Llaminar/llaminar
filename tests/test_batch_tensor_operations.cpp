/**
 * @file test_batch_tensor_operations.cpp
 * @brief Comprehensive tests for SimpleTensor batch dimension support
 * @author David Sanftenberg
 * @date October 15, 2025
 * 
 * Tests for Day 2: SimpleTensor Batch Dimension Support
 * Part of Option A: Full Parallel Batching implementation
 */

#include <gtest/gtest.h>
#include "../src/tensors/SimpleTensor.h"
#include <memory>
#include <vector>

using namespace llaminar;

/**
 * Test Fixture for Batch Tensor Operations
 */
class BatchTensorOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test dimensions
        batch_size_ = 4;
        seq_len_ = 8;
        hidden_dim_ = 16;
    }

    void TearDown() override {
        // Cleanup if needed
    }

    size_t batch_size_;
    size_t seq_len_;
    size_t hidden_dim_;
};

// =============================================================================
// Test 1: 3D Tensor Creation
// =============================================================================

TEST_F(BatchTensorOperationsTest, Create3DTensor) {
    // Create [batch=4, seq=8, hidden=16] tensor
    std::vector<int> shape = {
        static_cast<int>(batch_size_),
        static_cast<int>(seq_len_),
        static_cast<int>(hidden_dim_)
    };
    
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // Verify shape
    ASSERT_EQ(tensor->shape().size(), 3);
    EXPECT_EQ(tensor->shape()[0], batch_size_);
    EXPECT_EQ(tensor->shape()[1], seq_len_);
    EXPECT_EQ(tensor->shape()[2], hidden_dim_);
    
    // Verify total size
    size_t expected_size = batch_size_ * seq_len_ * hidden_dim_;
    EXPECT_EQ(tensor->size(), expected_size);
    
    // Verify zero initialization
    const float* data = tensor->data();
    for (size_t i = 0; i < expected_size; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f) << "Non-zero at index " << i;
    }
}

TEST_F(BatchTensorOperationsTest, Create3DTensorWithData) {
    std::vector<int> shape = {2, 3, 4};  // Small batch for testing
    std::vector<float> data(24);  // 2*3*4 = 24 elements
    
    // Fill with test pattern
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto tensor = std::make_shared<SimpleTensor>(shape, data);
    
    // Verify shape
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->shape()[2], 4);
    
    // Verify data
    const float* tensor_data = tensor->data();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(tensor_data[i], static_cast<float>(i));
    }
}

// =============================================================================
// Test 2: Batch Size Query
// =============================================================================

TEST_F(BatchTensorOperationsTest, BatchSizeQuery3D) {
    std::vector<int> shape = {4, 8, 16};
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    EXPECT_EQ(tensor->batch_size(), 4);
}

TEST_F(BatchTensorOperationsTest, BatchSizeQuery2D) {
    std::vector<int> shape = {8, 16};  // [seq, hidden]
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // 2D tensors have implicit batch_size=1
    EXPECT_EQ(tensor->batch_size(), 1);
}

TEST_F(BatchTensorOperationsTest, BatchSizeQuery1D) {
    std::vector<int> shape = {16};  // [hidden]
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // 1D tensors have implicit batch_size=1
    EXPECT_EQ(tensor->batch_size(), 1);
}

// =============================================================================
// Test 3: Sequence Length Query
// =============================================================================

TEST_F(BatchTensorOperationsTest, SequenceLengthQuery3D) {
    std::vector<int> shape = {4, 8, 16};  // [batch, seq, hidden]
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    EXPECT_EQ(tensor->seq_len(), 8);
}

TEST_F(BatchTensorOperationsTest, SequenceLengthQuery2D) {
    std::vector<int> shape = {8, 16};  // [seq, hidden]
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    EXPECT_EQ(tensor->seq_len(), 8);
}

// =============================================================================
// Test 4: Batch Slicing (Extract Single Sequence)
// =============================================================================

TEST_F(BatchTensorOperationsTest, SliceBatchBasic) {
    // Create [batch=4, seq=3, hidden=2] with test pattern
    std::vector<int> shape = {4, 3, 2};
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto batched = std::make_shared<SimpleTensor>(shape, data);
    
    // Slice batch index 1
    auto sliced = batched->slice_batch(1);
    
    // Verify shape: should be [3, 2]
    ASSERT_EQ(sliced->shape().size(), 2);
    EXPECT_EQ(sliced->shape()[0], 3);  // seq_len
    EXPECT_EQ(sliced->shape()[1], 2);  // hidden
    
    // Verify data: batch 1 starts at index 6 (1 * 3 * 2)
    const float* sliced_data = sliced->data();
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(sliced_data[i], static_cast<float>(6 + i))
            << "Mismatch at index " << i;
    }
}

TEST_F(BatchTensorOperationsTest, SliceBatchAllIndices) {
    std::vector<int> shape = {4, 2, 3};
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto batched = std::make_shared<SimpleTensor>(shape, data);
    
    // Test each batch index
    for (size_t b = 0; b < 4; ++b) {
        auto sliced = batched->slice_batch(b);
        
        ASSERT_EQ(sliced->shape().size(), 2);
        EXPECT_EQ(sliced->shape()[0], 2);
        EXPECT_EQ(sliced->shape()[1], 3);
        
        // Verify data offset
        const float* sliced_data = sliced->data();
        size_t expected_offset = b * 6;  // Each batch has 2*3=6 elements
        for (size_t i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(sliced_data[i], static_cast<float>(expected_offset + i));
        }
    }
}

TEST_F(BatchTensorOperationsTest, SliceBatchOutOfRange) {
    std::vector<int> shape = {4, 2, 3};
    auto batched = std::make_shared<SimpleTensor>(shape);
    
    // Should throw for out-of-range index
    EXPECT_THROW(batched->slice_batch(4), std::out_of_range);
    EXPECT_THROW(batched->slice_batch(10), std::out_of_range);
}

TEST_F(BatchTensorOperationsTest, SliceBatchScalarTensor) {
    std::vector<int> shape = {};  // Scalar
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // Should throw for scalar tensor
    EXPECT_THROW(tensor->slice_batch(0), std::invalid_argument);
}

// =============================================================================
// Test 5: Batch Stacking (Combine Multiple Sequences)
// =============================================================================

TEST_F(BatchTensorOperationsTest, StackBatchBasic) {
    // Create 3 sequences of [2, 3]
    std::vector<std::shared_ptr<SimpleTensor>> sequences;
    for (size_t b = 0; b < 3; ++b) {
        std::vector<int> shape = {2, 3};
        std::vector<float> data(6);
        for (size_t i = 0; i < 6; ++i) {
            data[i] = static_cast<float>(b * 10 + i);  // Pattern: 0-5, 10-15, 20-25
        }
        sequences.push_back(std::make_shared<SimpleTensor>(shape, data));
    }
    
    // Stack into batch
    auto batched = SimpleTensor::stack_batch(sequences);
    
    ASSERT_NE(batched, nullptr);
    
    // Verify shape: should be [3, 2, 3]
    ASSERT_EQ(batched->shape().size(), 3);
    EXPECT_EQ(batched->shape()[0], 3);  // batch
    EXPECT_EQ(batched->shape()[1], 2);  // seq
    EXPECT_EQ(batched->shape()[2], 3);  // hidden
    
    // Verify data
    const float* data = batched->data();
    for (size_t b = 0; b < 3; ++b) {
        for (size_t i = 0; i < 6; ++i) {
            float expected = static_cast<float>(b * 10 + i);
            size_t idx = b * 6 + i;
            EXPECT_FLOAT_EQ(data[idx], expected) 
                << "Mismatch at batch=" << b << ", element=" << i;
        }
    }
}

TEST_F(BatchTensorOperationsTest, StackBatchEmpty) {
    std::vector<std::shared_ptr<SimpleTensor>> sequences;
    
    auto result = SimpleTensor::stack_batch(sequences);
    
    EXPECT_EQ(result, nullptr);
}

TEST_F(BatchTensorOperationsTest, StackBatchSingleSequence) {
    std::vector<std::shared_ptr<SimpleTensor>> sequences;
    std::vector<int> shape = {2, 3};
    sequences.push_back(std::make_shared<SimpleTensor>(shape));
    
    auto batched = SimpleTensor::stack_batch(sequences);
    
    ASSERT_NE(batched, nullptr);
    ASSERT_EQ(batched->shape().size(), 3);
    EXPECT_EQ(batched->shape()[0], 1);  // batch=1
    EXPECT_EQ(batched->shape()[1], 2);
    EXPECT_EQ(batched->shape()[2], 3);
}

TEST_F(BatchTensorOperationsTest, StackBatchMismatchedShapes) {
    std::vector<std::shared_ptr<SimpleTensor>> sequences;
    sequences.push_back(std::make_shared<SimpleTensor>(std::vector<int>{2, 3}));
    sequences.push_back(std::make_shared<SimpleTensor>(std::vector<int>{2, 4}));  // Different!
    
    // Should throw due to shape mismatch
    EXPECT_THROW(SimpleTensor::stack_batch(sequences), std::invalid_argument);
}

// =============================================================================
// Test 6: Stack Then Slice Round-Trip
// =============================================================================

TEST_F(BatchTensorOperationsTest, StackSliceRoundTrip) {
    // Create original sequences
    std::vector<std::shared_ptr<SimpleTensor>> original_sequences;
    for (size_t b = 0; b < 4; ++b) {
        std::vector<int> shape = {3, 5};
        std::vector<float> data(15);
        for (size_t i = 0; i < 15; ++i) {
            data[i] = static_cast<float>(b * 100 + i);
        }
        original_sequences.push_back(std::make_shared<SimpleTensor>(shape, data));
    }
    
    // Stack into batch
    auto batched = SimpleTensor::stack_batch(original_sequences);
    ASSERT_NE(batched, nullptr);
    
    // Slice back out and compare
    for (size_t b = 0; b < 4; ++b) {
        auto sliced = batched->slice_batch(b);
        
        // Compare shape
        ASSERT_EQ(sliced->shape().size(), original_sequences[b]->shape().size());
        for (size_t dim = 0; dim < sliced->shape().size(); ++dim) {
            EXPECT_EQ(sliced->shape()[dim], original_sequences[b]->shape()[dim]);
        }
        
        // Compare data
        const float* sliced_data = sliced->data();
        const float* original_data = original_sequences[b]->data();
        for (size_t i = 0; i < 15; ++i) {
            EXPECT_FLOAT_EQ(sliced_data[i], original_data[i])
                << "Mismatch at batch=" << b << ", element=" << i;
        }
    }
}

// =============================================================================
// Test 7: Reshape Operations with Batch Dimension
// =============================================================================

TEST_F(BatchTensorOperationsTest, ReshapeWithBatchDimension) {
    // Create [batch=2, seq=3, hidden=4] = 24 elements
    std::vector<int> shape = {2, 3, 4};
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = std::make_shared<SimpleTensor>(shape, data);
    
    // Reshape to [batch=2, seq*hidden=12]
    auto reshaped = tensor->reshape_copy({2, 12});
    
    ASSERT_EQ(reshaped->shape().size(), 2);
    EXPECT_EQ(reshaped->shape()[0], 2);
    EXPECT_EQ(reshaped->shape()[1], 12);
    
    // Data should be identical (just different view)
    const float* reshaped_data = reshaped->data();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(reshaped_data[i], static_cast<float>(i));
    }
}

TEST_F(BatchTensorOperationsTest, ReshapeInPlace) {
    std::vector<int> shape = {4, 6};
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // Fill with test data
    float* data = tensor->data();
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape in-place to [2, 3, 4]
    tensor->reshape({2, 3, 4});
    
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->shape()[2], 4);
    
    // Data should be unchanged
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

TEST_F(BatchTensorOperationsTest, ReshapeSizeMismatch) {
    std::vector<int> shape = {2, 3, 4};  // 24 elements
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // Try to reshape to different total size
    EXPECT_THROW(tensor->reshape_copy({2, 3, 5}), std::invalid_argument);  // 30 elements
    EXPECT_THROW(tensor->reshape({5, 5}), std::invalid_argument);  // 25 elements
}

// =============================================================================
// Test 8: Batch Dimension Edge Cases
// =============================================================================

TEST_F(BatchTensorOperationsTest, LargeBatchSize) {
    // Test with large batch size
    std::vector<int> shape = {128, 4, 8};  // 128 sequences
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    EXPECT_EQ(tensor->batch_size(), 128);
    EXPECT_EQ(tensor->size(), 128 * 4 * 8);
    
    // Slice first and last batch
    auto first = tensor->slice_batch(0);
    auto last = tensor->slice_batch(127);
    
    EXPECT_EQ(first->shape()[0], 4);
    EXPECT_EQ(first->shape()[1], 8);
    EXPECT_EQ(last->shape()[0], 4);
    EXPECT_EQ(last->shape()[1], 8);
}

TEST_F(BatchTensorOperationsTest, BatchSizeOne) {
    // Edge case: batch size = 1
    std::vector<int> shape = {1, 8, 16};
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    EXPECT_EQ(tensor->batch_size(), 1);
    
    auto sliced = tensor->slice_batch(0);
    EXPECT_EQ(sliced->shape()[0], 8);
    EXPECT_EQ(sliced->shape()[1], 16);
}

// =============================================================================
// Test 9: Memory Efficiency Tests
// =============================================================================

TEST_F(BatchTensorOperationsTest, NUMAFirstTouchApplied) {
    // Large tensor to trigger NUMA first-touch
    std::vector<int> shape = {8, 512, 896};  // ~3.5M elements, ~14MB
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    // Just verify it constructs successfully and is zero-initialized
    EXPECT_EQ(tensor->size(), 8 * 512 * 896);
    
    // Spot check some values
    const float* data = tensor->data();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1000], 0.0f);
    EXPECT_FLOAT_EQ(data[tensor->size() - 1], 0.0f);
}

// =============================================================================
// Test 10: Copy and Zero Operations
// =============================================================================

TEST_F(BatchTensorOperationsTest, CopyBatchedTensor) {
    std::vector<int> shape = {2, 3, 4};
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto original = std::make_shared<SimpleTensor>(shape, data);
    
    auto copied = std::dynamic_pointer_cast<SimpleTensor>(original->copy());
    
    ASSERT_NE(copied, nullptr);
    EXPECT_EQ(copied->shape(), original->shape());
    
    const float* orig_data = original->data();
    const float* copy_data = copied->data();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(copy_data[i], orig_data[i]);
    }
    
    // Modify copy, original should be unchanged
    copied->data()[0] = 999.0f;
    EXPECT_FLOAT_EQ(original->data()[0], 0.0f);
}

TEST_F(BatchTensorOperationsTest, ZeroBatchedTensor) {
    std::vector<int> shape = {2, 3, 4};
    std::vector<float> data(24, 5.0f);  // Fill with 5.0
    auto tensor = std::make_shared<SimpleTensor>(shape, data);
    
    // Verify filled
    EXPECT_FLOAT_EQ(tensor->data()[0], 5.0f);
    
    // Zero it
    tensor->zero();
    
    // Verify all zeros
    const float* tensor_data = tensor->data();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(tensor_data[i], 0.0f);
    }
}

TEST_F(BatchTensorOperationsTest, FillBatchedTensor) {
    std::vector<int> shape = {2, 3, 4};
    auto tensor = std::make_shared<SimpleTensor>(shape);
    
    tensor->fill(3.14f);
    
    const float* data = tensor->data();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }
}

// =============================================================================
// Test 11: Batch Dimension Queries for Various Shapes
// =============================================================================

TEST_F(BatchTensorOperationsTest, DimensionQueriesComprehensive) {
    // Test all dimension queries for various tensor shapes
    
    // 1D tensor
    {
        auto t = std::make_shared<SimpleTensor>(std::vector<int>{10});
        EXPECT_EQ(t->batch_size(), 1);
        EXPECT_EQ(t->seq_len(), 1);
        EXPECT_EQ(t->ndim(), 1);
    }
    
    // 2D tensor [seq, hidden]
    {
        auto t = std::make_shared<SimpleTensor>(std::vector<int>{8, 16});
        EXPECT_EQ(t->batch_size(), 1);
        EXPECT_EQ(t->seq_len(), 8);
        EXPECT_EQ(t->ndim(), 2);
    }
    
    // 3D tensor [batch, seq, hidden]
    {
        auto t = std::make_shared<SimpleTensor>(std::vector<int>{4, 8, 16});
        EXPECT_EQ(t->batch_size(), 4);
        EXPECT_EQ(t->seq_len(), 8);
        EXPECT_EQ(t->ndim(), 3);
    }
    
    // 4D tensor (hypothetical future use)
    {
        auto t = std::make_shared<SimpleTensor>(std::vector<int>{2, 4, 8, 16});
        EXPECT_EQ(t->batch_size(), 2);
        EXPECT_EQ(t->seq_len(), 4);
        EXPECT_EQ(t->ndim(), 4);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
