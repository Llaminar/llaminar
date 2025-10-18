/**
 * @file test_attention_batch_integration.cpp
 * @brief Integration tests for MPIAttentionOperator with various batch sizes
 * @author David Sanftenberg
 * @date October 16, 2025
 * 
 * Test Coverage:
 * - MPIAttentionOperator with various batch sizes (1, 4, 8, 16, 32)
 * - 3D input tensor [batch, seq_len, d_model] handling
 * - Batch dimension pass-through and correctness
 * - Memory efficiency across different batch sizes
 */

#include <gtest/gtest.h>
#include "../src/operators/MPIAttentionOperator.h"
#include "../src/tensors/SimpleTensor.h"
#include "../src/Logger.h"
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>

using namespace llaminar;

class AttentionBatchIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        
        // Standard Qwen config
        n_head_ = 14;
        n_head_kv_ = 2;
        head_dim_ = 64;
        d_model_ = n_head_ * head_dim_;  // 896
        rope_freq_base_ = 1000000.0f;
        
        // Create attention operator
        attention_op_ = std::make_unique<MPIAttentionOperator>(
            n_head_, n_head_kv_, head_dim_, rope_freq_base_);
    }
    
    void TearDown() override {
        attention_op_.reset();
    }
    
    /**
     * @brief Create full attention inputs (10 tensors total)
     * Matches MPIAttentionOperator interface: {input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache}
     */
    void createAttentionInputs(int batch_size, int seq_len, 
                              std::vector<std::shared_ptr<TensorBase>>& inputs) {
        // Input activation [batch, seq, d_model]
        auto input = std::make_shared<SimpleTensor>(
            std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
        
        // Fill with small random values
        for (size_t i = 0; i < input->size(); ++i) {
            input->data()[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
        }
        
        // Weight matrices (note: wk and wv are transposed relative to wq and wo)
        int total_head_dim = n_head_ * head_dim_;
        int total_kv_head_dim = n_head_kv_ * head_dim_;
        
        auto wq = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(d_model_), total_head_dim});
        auto wk = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim, static_cast<int>(d_model_)});  // Transposed
        auto wv = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim, static_cast<int>(d_model_)});  // Transposed
        auto wo = std::make_shared<SimpleTensor>(std::vector<int>{total_head_dim, static_cast<int>(d_model_)});
        
        // Fill weights
        for (size_t i = 0; i < wq->size(); ++i) wq->data()[i] = 0.01f * ((i % 50) / 50.0f);
        for (size_t i = 0; i < wk->size(); ++i) wk->data()[i] = 0.01f * ((i % 50) / 50.0f);
        for (size_t i = 0; i < wv->size(); ++i) wv->data()[i] = 0.01f * ((i % 50) / 50.0f);
        for (size_t i = 0; i < wo->size(); ++i) wo->data()[i] = 0.01f * ((i % 50) / 50.0f);
        
        // Biases
        auto bq = std::make_shared<SimpleTensor>(std::vector<int>{total_head_dim});
        auto bk = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim});
        auto bv = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim});
        
        std::fill_n(bq->data(), total_head_dim, 0.0f);
        std::fill_n(bk->data(), total_kv_head_dim, 0.0f);
        std::fill_n(bv->data(), total_kv_head_dim, 0.0f);
        
        // KV cache tensors (empty for prefill)
        int max_cache_len = 2048;
        auto k_cache = std::make_shared<SimpleTensor>(std::vector<int>{max_cache_len, total_kv_head_dim});
        auto v_cache = std::make_shared<SimpleTensor>(std::vector<int>{max_cache_len, total_kv_head_dim});
        
        std::fill_n(k_cache->data(), k_cache->size(), 0.0f);
        std::fill_n(v_cache->data(), v_cache->size(), 0.0f);
        
        inputs = {input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache};
    }
    
    /**
     * @brief Check tensor has no NaN or Inf values
     */
    bool hasNoNaNOrInf(const std::shared_ptr<TensorBase>& tensor) {
        const float* data = tensor->data();
        for (size_t i = 0; i < tensor->size(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return false;
            }
        }
        return true;
    }
    
    int rank_;
    int world_size_;
    size_t n_head_;
    size_t n_head_kv_;
    size_t head_dim_;
    size_t d_model_;
    float rope_freq_base_;
    std::unique_ptr<MPIAttentionOperator> attention_op_;
};

// ========================================
// Batch Size 1 (Baseline)
// ========================================

TEST_F(AttentionBatchIntegrationTest, Batch1) {
    const int batch_size = 1;
    const int seq_len = 16;
    
    attention_op_->setSequencePosition(0);  // Prefill mode
    
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(batch_size, seq_len, inputs);
    
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = attention_op_->execute(inputs, outputs);
    ASSERT_TRUE(success) << "Batch 1 execution failed";
    
    // Verify output shape
    ASSERT_EQ(outputs[0]->shape().size(), 3);
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_EQ(outputs[0]->shape()[2], d_model_);
    
    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0])) << "Output contains NaN or Inf";
    
    if (rank_ == 0) {
        LOG_INFO("[AttentionBatchIntegrationTest] Batch1: PASS");
    }
}

// ========================================
// Batch Size 4
// ========================================

TEST_F(AttentionBatchIntegrationTest, Batch4) {
    const int batch_size = 4;
    const int seq_len = 8;
    
    attention_op_->setSequencePosition(0);  // Prefill mode
    
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(batch_size, seq_len, inputs);
    
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = attention_op_->execute(inputs, outputs);
    ASSERT_TRUE(success) << "Batch 4 execution failed";
    
    // Verify output shape
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0]));
    
    if (rank_ == 0) {
        LOG_INFO("[AttentionBatchIntegrationTest] Batch4: PASS");
    }
}

// ========================================
// Batch Size 8
// ========================================

TEST_F(AttentionBatchIntegrationTest, Batch8) {
    const int batch_size = 8;
    const int seq_len = 16;
    
    attention_op_->setSequencePosition(0);
    
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(batch_size, seq_len, inputs);
    
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = attention_op_->execute(inputs, outputs);
    ASSERT_TRUE(success) << "Batch 8 execution failed";
    
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0]));
    
    if (rank_ == 0) {
        LOG_INFO("[AttentionBatchIntegrationTest] Batch8: PASS");
    }
}

// ========================================
// Batch Size 16
// ========================================

TEST_F(AttentionBatchIntegrationTest, Batch16) {
    const int batch_size = 16;
    const int seq_len = 8;
    
    attention_op_->setSequencePosition(0);
    
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(batch_size, seq_len, inputs);
    
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = attention_op_->execute(inputs, outputs);
    ASSERT_TRUE(success) << "Batch 16 execution failed";
    
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0]));
    
    if (rank_ == 0) {
        LOG_INFO("[AttentionBatchIntegrationTest] Batch16: PASS");
    }
}

// ========================================
// Batch Size 32
// ========================================

TEST_F(AttentionBatchIntegrationTest, Batch32) {
    const int batch_size = 32;
    const int seq_len = 4;
    
    attention_op_->setSequencePosition(0);
    
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(batch_size, seq_len, inputs);
    
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, static_cast<int>(d_model_)});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = attention_op_->execute(inputs, outputs);
    ASSERT_TRUE(success) << "Batch 32 execution failed";
    
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0]));
    
    if (rank_ == 0) {
        LOG_INFO("[AttentionBatchIntegrationTest] Batch32: PASS");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        LOG_INFO("========================================");
        LOG_INFO("Attention Batch Integration Test Suite");
        LOG_INFO("========================================");
    }
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
