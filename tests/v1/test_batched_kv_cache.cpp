/**
 * @file test_batched_kv_cache.cpp
 * @brief Comprehensive tests for BatchedKVCache
 * @author David Sanftenberg
 * @date October 15, 2025
 * 
 * Test Coverage:
 * - Construction and initialization
 * - Get/Set operations
 * - Append operations with concatenation
 * - Sequence length tracking
 * - Batch isolation (no cross-contamination)
 * - Reset and clear operations
 * - Error handling (bounds, max length)
 * - Edge cases (empty, batch=1, max length)
 */

#include <gtest/gtest.h>
#include "../src/tensors/BatchedKVCache.h"
#include "../src/tensors/SimpleTensor.h"
#include <stdexcept>

using namespace llaminar;

class BatchedKVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test configuration
        num_layers_ = 4;
        batch_size_ = 8;
        max_seq_len_ = 2048;
        hidden_dim_ = 512;
        num_heads_ = 8;
        head_dim_ = hidden_dim_ / num_heads_;  // 64
    }
    
    // Helper: Create KV tensor [num_heads, seq_len, head_dim]
    std::shared_ptr<SimpleTensor> createKVTensor(int seq_len, float fill_value = 0.0f) {
        std::vector<int> shape = {static_cast<int>(num_heads_), seq_len, static_cast<int>(head_dim_)};
        auto tensor = std::make_shared<SimpleTensor>(shape);
        
        // Fill with value for easy verification
        float* data = tensor->data();
        size_t total_size = num_heads_ * seq_len * head_dim_;
        std::fill(data, data + total_size, fill_value);
        
        return tensor;
    }
    
    size_t num_layers_;
    size_t batch_size_;
    size_t max_seq_len_;
    size_t hidden_dim_;
    size_t num_heads_;
    size_t head_dim_;
};

// ========================================
// Construction and Initialization Tests
// ========================================

TEST_F(BatchedKVCacheTest, ConstructionInitializesCorrectly) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_EQ(cache.num_layers(), num_layers_);
    EXPECT_EQ(cache.batch_size(), batch_size_);
    EXPECT_EQ(cache.max_sequence_length(), max_seq_len_);
    EXPECT_EQ(cache.hidden_dim(), hidden_dim_);
}

TEST_F(BatchedKVCacheTest, InitiallyAllEmpty) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        for (size_t batch = 0; batch < batch_size_; ++batch) {
            EXPECT_TRUE(cache.is_empty(layer, batch));
            EXPECT_EQ(cache.sequence_length(layer, batch), 0);
            EXPECT_EQ(cache.get_k(layer, batch), nullptr);
            EXPECT_EQ(cache.get_v(layer, batch), nullptr);
        }
    }
}

TEST_F(BatchedKVCacheTest, SmallCacheConfiguration) {
    BatchedKVCache small_cache(2, 1, 128, 64);
    
    EXPECT_EQ(small_cache.num_layers(), 2);
    EXPECT_EQ(small_cache.batch_size(), 1);
    EXPECT_EQ(small_cache.max_sequence_length(), 128);
}

// ========================================
// Get/Set Operations Tests
// ========================================

TEST_F(BatchedKVCacheTest, SetAndGetKVTensors) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k_tensor = createKVTensor(16, 1.5f);
    auto v_tensor = createKVTensor(16, 2.5f);
    
    cache.set_kv(0, 0, k_tensor, v_tensor);
    
    auto retrieved_k = cache.get_k(0, 0);
    auto retrieved_v = cache.get_v(0, 0);
    
    EXPECT_NE(retrieved_k, nullptr);
    EXPECT_NE(retrieved_v, nullptr);
    EXPECT_EQ(cache.sequence_length(0, 0), 16);
    EXPECT_FALSE(cache.is_empty(0, 0));
}

TEST_F(BatchedKVCacheTest, SetUpdatesSequenceLength) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k1 = createKVTensor(10);
    auto v1 = createKVTensor(10);
    cache.set_kv(1, 2, k1, v1);
    EXPECT_EQ(cache.sequence_length(1, 2), 10);
    
    auto k2 = createKVTensor(25);
    auto v2 = createKVTensor(25);
    cache.set_kv(1, 2, k2, v2);
    EXPECT_EQ(cache.sequence_length(1, 2), 25);
}

TEST_F(BatchedKVCacheTest, SetNullClearsPreviousData) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    cache.set_kv(0, 3, k, v);
    EXPECT_FALSE(cache.is_empty(0, 3));
    
    cache.set_kv(0, 3, nullptr, nullptr);
    EXPECT_TRUE(cache.is_empty(0, 3));
    EXPECT_EQ(cache.sequence_length(0, 3), 0);
}

// ========================================
// Append Operations Tests
// ========================================

TEST_F(BatchedKVCacheTest, AppendToEmptyCache) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(8, 1.0f);
    auto v = createKVTensor(8, 2.0f);
    
    cache.append_kv(0, 0, k, v);
    
    EXPECT_EQ(cache.sequence_length(0, 0), 8);
    EXPECT_FALSE(cache.is_empty(0, 0));
    
    auto retrieved_k = cache.get_k(0, 0);
    EXPECT_EQ(retrieved_k->shape()[1], 8);  // seq dimension
}

TEST_F(BatchedKVCacheTest, AppendConcatenatesSequences) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k1 = createKVTensor(10, 1.0f);
    auto v1 = createKVTensor(10, 2.0f);
    cache.append_kv(1, 2, k1, v1);
    
    auto k2 = createKVTensor(5, 3.0f);
    auto v2 = createKVTensor(5, 4.0f);
    cache.append_kv(1, 2, k2, v2);
    
    EXPECT_EQ(cache.sequence_length(1, 2), 15);
    
    auto final_k = cache.get_k(1, 2);
    EXPECT_EQ(final_k->shape()[1], 15);  // Total sequence length
}

TEST_F(BatchedKVCacheTest, AppendPreservesData) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k1 = createKVTensor(4, 1.0f);
    auto v1 = createKVTensor(4, 2.0f);
    cache.append_kv(0, 0, k1, v1);
    
    auto k2 = createKVTensor(3, 5.0f);
    auto v2 = createKVTensor(3, 6.0f);
    cache.append_kv(0, 0, k2, v2);
    
    auto final_k = cache.get_k(0, 0);
    const float* k_data = final_k->data();
    
    // Verify first sequence (head 0)
    for (int i = 0; i < 4 * head_dim_; ++i) {
        EXPECT_FLOAT_EQ(k_data[i], 1.0f);
    }
    
    // Verify second sequence (head 0)
    for (int i = 4 * head_dim_; i < 7 * head_dim_; ++i) {
        EXPECT_FLOAT_EQ(k_data[i], 5.0f);
    }
}

TEST_F(BatchedKVCacheTest, MultipleAppendsIncrementLength) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    for (int i = 0; i < 5; ++i) {
        auto k = createKVTensor(3);
        auto v = createKVTensor(3);
        cache.append_kv(2, 1, k, v);
        EXPECT_EQ(cache.sequence_length(2, 1), (i + 1) * 3);
    }
}

// ========================================
// Batch Isolation Tests
// ========================================

TEST_F(BatchedKVCacheTest, DifferentBatchesAreIsolated) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k0 = createKVTensor(10, 1.0f);
    auto v0 = createKVTensor(10, 2.0f);
    cache.set_kv(0, 0, k0, v0);
    
    auto k1 = createKVTensor(20, 3.0f);
    auto v1 = createKVTensor(20, 4.0f);
    cache.set_kv(0, 1, k1, v1);
    
    EXPECT_EQ(cache.sequence_length(0, 0), 10);
    EXPECT_EQ(cache.sequence_length(0, 1), 20);
    EXPECT_TRUE(cache.is_empty(0, 2));  // Other batches unaffected
}

TEST_F(BatchedKVCacheTest, DifferentLayersAreIsolated) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(15);
    auto v = createKVTensor(15);
    
    cache.set_kv(0, 3, k, v);
    cache.set_kv(2, 3, k, v);
    
    EXPECT_EQ(cache.sequence_length(0, 3), 15);
    EXPECT_EQ(cache.sequence_length(2, 3), 15);
    EXPECT_TRUE(cache.is_empty(1, 3));  // Layer 1 unaffected
}

// ========================================
// Reset and Clear Tests
// ========================================

TEST_F(BatchedKVCacheTest, ResetBatchClearsAllLayers) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    
    // Set data in all layers for batch 2
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        cache.set_kv(layer, 2, k, v);
    }
    
    cache.reset_batch(2);
    
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        EXPECT_TRUE(cache.is_empty(layer, 2));
        EXPECT_EQ(cache.sequence_length(layer, 2), 0);
    }
}

TEST_F(BatchedKVCacheTest, ResetBatchLeavesOtherBatchesIntact) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    
    cache.set_kv(0, 0, k, v);
    cache.set_kv(0, 1, k, v);
    
    cache.reset_batch(0);
    
    EXPECT_TRUE(cache.is_empty(0, 0));
    EXPECT_FALSE(cache.is_empty(0, 1));  // Batch 1 unaffected
}

TEST_F(BatchedKVCacheTest, ClearAllRemovesEverything) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    
    // Fill cache
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        for (size_t batch = 0; batch < batch_size_; ++batch) {
            cache.set_kv(layer, batch, k, v);
        }
    }
    
    cache.clear_all();
    
    // Verify all empty
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        for (size_t batch = 0; batch < batch_size_; ++batch) {
            EXPECT_TRUE(cache.is_empty(layer, batch));
            EXPECT_EQ(cache.sequence_length(layer, batch), 0);
        }
    }
}

// ========================================
// Error Handling Tests
// ========================================

TEST_F(BatchedKVCacheTest, GetInvalidLayerThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_THROW(cache.get_k(num_layers_, 0), std::out_of_range);
    EXPECT_THROW(cache.get_v(num_layers_ + 1, 0), std::out_of_range);
}

TEST_F(BatchedKVCacheTest, GetInvalidBatchThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_THROW(cache.get_k(0, batch_size_), std::out_of_range);
    EXPECT_THROW(cache.get_v(0, batch_size_ + 1), std::out_of_range);
}

TEST_F(BatchedKVCacheTest, SetInvalidIndicesThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    
    EXPECT_THROW(cache.set_kv(num_layers_, 0, k, v), std::out_of_range);
    EXPECT_THROW(cache.set_kv(0, batch_size_, k, v), std::out_of_range);
}

TEST_F(BatchedKVCacheTest, AppendNullTensorsThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_THROW(cache.append_kv(0, 0, nullptr, nullptr), std::invalid_argument);
}

TEST_F(BatchedKVCacheTest, AppendExceedingMaxLengthThrows) {
    BatchedKVCache cache(4, 2, 20, 512);  // max_seq = 20
    
    auto k1 = createKVTensor(15);
    auto v1 = createKVTensor(15);
    cache.append_kv(0, 0, k1, v1);
    
    auto k2 = createKVTensor(10);  // Would make total 25 > 20
    auto v2 = createKVTensor(10);
    
    EXPECT_THROW(cache.append_kv(0, 0, k2, v2), std::runtime_error);
}

TEST_F(BatchedKVCacheTest, AppendIncompatibleShapesThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k1 = createKVTensor(10);  // [8, 10, 64]
    auto v1 = createKVTensor(10);
    cache.append_kv(0, 0, k1, v1);
    
    // Create incompatible tensor (different num_heads)
    std::vector<int> bad_shape = {4, 5, static_cast<int>(head_dim_)};  // 4 heads instead of 8
    auto bad_k = std::make_shared<SimpleTensor>(bad_shape);
    auto bad_v = std::make_shared<SimpleTensor>(bad_shape);
    
    EXPECT_THROW(cache.append_kv(0, 0, bad_k, bad_v), std::invalid_argument);
}

TEST_F(BatchedKVCacheTest, ResetInvalidBatchThrows) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_THROW(cache.reset_batch(batch_size_), std::out_of_range);
    EXPECT_THROW(cache.reset_batch(batch_size_ + 5), std::out_of_range);
}

// ========================================
// Edge Case Tests
// ========================================

TEST_F(BatchedKVCacheTest, SingleBatchConfiguration) {
    BatchedKVCache cache(4, 1, 512, 256);
    
    auto k = createKVTensor(10);
    auto v = createKVTensor(10);
    
    cache.set_kv(0, 0, k, v);
    EXPECT_EQ(cache.sequence_length(0, 0), 10);
    
    // No batch 1
    EXPECT_THROW(cache.get_k(0, 1), std::out_of_range);
}

TEST_F(BatchedKVCacheTest, MaxSequenceLengthExactly) {
    BatchedKVCache cache(2, 2, 50, 256);
    
    auto k = createKVTensor(50);
    auto v = createKVTensor(50);
    
    // Should succeed at exactly max
    EXPECT_NO_THROW(cache.append_kv(0, 0, k, v));
    EXPECT_EQ(cache.sequence_length(0, 0), 50);
    
    // One more should fail
    auto k_extra = createKVTensor(1);
    auto v_extra = createKVTensor(1);
    EXPECT_THROW(cache.append_kv(0, 0, k_extra, v_extra), std::runtime_error);
}

TEST_F(BatchedKVCacheTest, SequenceLengthOne) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    auto k = createKVTensor(1);
    auto v = createKVTensor(1);
    
    cache.set_kv(0, 0, k, v);
    EXPECT_EQ(cache.sequence_length(0, 0), 1);
}

TEST_F(BatchedKVCacheTest, QueryEmptySequenceLength) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    EXPECT_NO_THROW({
        size_t len = cache.sequence_length(0, 0);
        EXPECT_EQ(len, 0);
    });
}

// ========================================
// Integration Test
// ========================================

TEST_F(BatchedKVCacheTest, SimulatePrefillAndDecode) {
    BatchedKVCache cache(num_layers_, batch_size_, max_seq_len_, hidden_dim_);
    
    // Prefill: large initial sequence
    auto k_prefill = createKVTensor(32, 1.0f);
    auto v_prefill = createKVTensor(32, 2.0f);
    cache.append_kv(0, 0, k_prefill, v_prefill);
    EXPECT_EQ(cache.sequence_length(0, 0), 32);
    
    // Decode: append one token at a time
    for (int step = 0; step < 10; ++step) {
        auto k_token = createKVTensor(1, 3.0f);
        auto v_token = createKVTensor(1, 4.0f);
        cache.append_kv(0, 0, k_token, v_token);
    }
    
    EXPECT_EQ(cache.sequence_length(0, 0), 42);  // 32 + 10
    
    auto final_k = cache.get_k(0, 0);
    EXPECT_EQ(final_k->shape()[1], 42);
}

// ========================================
// Main
// ========================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
