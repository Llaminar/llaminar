/**
 * @file TestBatchPrefillSkeleton.cpp
 * @brief Basic test for BatchQwenPipeline prefill functionality
 * 
 * Validates:
 * - Weight loading
 * - Batched prefill execution
 * - Output shape correctness
 * - Padding and sequence length tracking
 * - Real embedding lookup and projection
 */

#include <gtest/gtest.h>
#include "BatchQwenPipeline.h"
#include "ModelLoader.h"
#include "Logger.h"
#include <memory>
#include <cmath>

using namespace llaminar;

class BatchPrefillTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create minimal config for testing
        TransformerLayerConfig layer_cfg;
        layer_cfg.n_head = 14;
        layer_cfg.n_head_kv = 2;
        layer_cfg.head_dim = 64;
        layer_cfg.d_model = 896;
        layer_cfg.d_ff = 4864;
        layer_cfg.vocab_size = 151936;
        layer_cfg.max_seq_len = 2048;
        layer_cfg.n_layers = 24;
        layer_cfg.eps = 1e-6f;
        layer_cfg.rope_freq_base = 1000000.0f;
        
        config_ = ModelConfig(layer_cfg, "qwen");
        
        // Load real model weights for validation tests
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    }
    
    ModelConfig config_;
    std::string model_path_;
};

TEST_F(BatchPrefillTest, ConstructorBasic) {
    BatchQwenPipeline pipeline(config_);
    EXPECT_EQ(pipeline.name(), "BatchQwenPipeline");
    EXPECT_EQ(pipeline.config().getLayerConfig().d_model, 896);
}

TEST_F(BatchPrefillTest, PrefillBatchShapes) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    // Create batch of sequences with varying lengths
    std::vector<std::vector<int>> token_batches = {
        {1, 2, 3, 4, 5},           // 5 tokens
        {10, 11, 12},              // 3 tokens  
        {20, 21, 22, 23, 24, 25}   // 6 tokens
    };
    
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    
    // Execute prefill with real weights
    bool success = pipeline.prefillBatch(token_batches, *weights, ctx, logits);
    
    // Should succeed with shape validation
    ASSERT_TRUE(success);
    ASSERT_NE(logits, nullptr);
    
    // Check output shape: [batch_size, vocab_size]
    const auto& shape = logits->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);  // batch size
    EXPECT_EQ(shape[1], config_.getLayerConfig().vocab_size);
    
    // Verify logits are non-zero (real computation happened)
    const float* logits_data = logits->data();
    bool has_nonzero = false;
    for (size_t i = 0; i < 100 && i < logits->size(); ++i) {
        if (std::abs(logits_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Logits should contain non-zero values from real projection";
}

TEST_F(BatchPrefillTest, EmptyBatchHandling) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    std::vector<std::vector<int>> empty_batch;
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    
    // Should fail gracefully even with valid weights
    bool success = pipeline.prefillBatch(empty_batch, *weights, ctx, logits);
    EXPECT_FALSE(success);
}

TEST_F(BatchPrefillTest, SingleSequenceBatch) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    std::vector<std::vector<int>> single_batch = {
        {1, 2, 3, 4, 5, 6, 7, 8}
    };
    
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    
    bool success = pipeline.prefillBatch(single_batch, *weights, ctx, logits);
    
    ASSERT_TRUE(success);
    ASSERT_NE(logits, nullptr);
    
    const auto& shape = logits->shape();
    EXPECT_EQ(shape[0], 1);  // batch size = 1
    EXPECT_EQ(shape[1], config_.getLayerConfig().vocab_size);
}

TEST_F(BatchPrefillTest, LogitsRetrieval) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    std::vector<std::vector<int>> batch = {{1, 2, 3}};
    StageContext ctx;
    std::shared_ptr<TensorBase> logits1;
    
    pipeline.prefillBatch(batch, *weights, ctx, logits1);
    
    // Retrieve logits via interface
    std::shared_ptr<TensorBase> logits2;
    bool retrieved = pipeline.logits(logits2);
    
    ASSERT_TRUE(retrieved);
    ASSERT_NE(logits2, nullptr);
    
    // Should be same tensor
    EXPECT_EQ(logits1.get(), logits2.get());
}

TEST_F(BatchPrefillTest, DecodeSingleStep) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    // First do prefill
    std::vector<std::vector<int>> prefill_batch = {
        {1, 2, 3, 4},
        {10, 11, 12}
    };
    
    StageContext ctx;
    std::shared_ptr<TensorBase> prefill_logits;
    bool prefill_ok = pipeline.prefillBatch(prefill_batch, *weights, ctx, prefill_logits);
    ASSERT_TRUE(prefill_ok);
    
    // Now decode single step (one token per sequence)
    std::vector<int> decode_tokens = {5, 13};  // Next token for each sequence
    std::shared_ptr<TensorBase> decode_logits;
    
    bool decode_ok = pipeline.decodeBatch(decode_tokens, *weights, ctx, decode_logits);
    
    ASSERT_TRUE(decode_ok);
    ASSERT_NE(decode_logits, nullptr);
    
    // Check output shape: [batch_size, vocab_size]
    const auto& shape = decode_logits->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);  // Same batch size
    EXPECT_EQ(shape[1], config_.getLayerConfig().vocab_size);
    
    // Verify logits are non-zero (real computation)
    const float* logits_data = decode_logits->data();
    bool has_nonzero = false;
    for (size_t i = 0; i < 100 && i < decode_logits->size(); ++i) {
        if (std::abs(logits_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Decode logits should contain non-zero values";
}

TEST_F(BatchPrefillTest, DecodeMultiStep) {
    BatchQwenPipeline pipeline(config_);
    
    // Load real weights
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    // Prefill
    std::vector<std::vector<int>> prefill_batch = {{1, 2, 3}};
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    
    pipeline.prefillBatch(prefill_batch, *weights, ctx, logits);
    
    // Generate 10 decode steps
    const int num_steps = 10;
    std::vector<int> generated_tokens;
    
    for (int step = 0; step < num_steps; ++step) {
        // For simplicity, always decode with token 4
        std::vector<int> next_tokens = {4};
        
        bool success = pipeline.decodeBatch(next_tokens, *weights, ctx, logits);
        ASSERT_TRUE(success) << "Decode step " << step << " failed";
        ASSERT_NE(logits, nullptr);
        
        // Verify shape
        const auto& shape = logits->shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], config_.getLayerConfig().vocab_size);
        
        generated_tokens.push_back(4);
    }
    
    EXPECT_EQ(generated_tokens.size(), num_steps);
}

TEST_F(BatchPrefillTest, DecodeBatchSizeValidation) {
    BatchQwenPipeline pipeline(config_);
    
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    // Prefill with 3 sequences
    std::vector<std::vector<int>> prefill_batch = {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    pipeline.prefillBatch(prefill_batch, *weights, ctx, logits);
    
    // Try to decode with wrong batch size (should fail)
    std::vector<int> wrong_size_tokens = {7, 8};  // Only 2 tokens, need 3
    
    bool should_fail = pipeline.decodeBatch(wrong_size_tokens, *weights, ctx, logits);
    EXPECT_FALSE(should_fail) << "Should reject mismatched batch size";
    
    // Now try with correct batch size
    std::vector<int> correct_tokens = {7, 8, 9};
    bool should_succeed = pipeline.decodeBatch(correct_tokens, *weights, ctx, logits);
    EXPECT_TRUE(should_succeed) << "Should accept matching batch size";
}

TEST_F(BatchPrefillTest, DecodeWithoutPrefill) {
    BatchQwenPipeline pipeline(config_);
    
    auto weights = pipeline.loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);
    
    // Try decode without prefill (should work - decode initializes state)
    std::vector<int> decode_tokens = {1, 2, 3};
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    
    bool success = pipeline.decodeBatch(decode_tokens, *weights, ctx, logits);
    
    ASSERT_TRUE(success);
    ASSERT_NE(logits, nullptr);
    
    const auto& shape = logits->shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], config_.getLayerConfig().vocab_size);
}

// Main for standalone execution
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize MPI for pipeline (required by PipelineBase)
    MPI_Init(&argc, &argv);
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
