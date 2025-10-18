/**
 * @file test_qwen_pipeline_batch.cpp
 * @brief Unit tests for QwenPipeline batch processing methods
 * @author David Sanftenberg
 */

#include "QwenPipelineAdapter.h"
#include "BatchQwenPipelineAdapter.h"
#include "AbstractPipeline.h"
#include "ModelLoader.h"
#include "TransformerConfig.h"
#include "tensors/SimpleTensor.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <memory>
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace llaminar;

/**
 * Test fixture for QwenPipeline batch operations
 */
class QwenPipelineBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Model path - use smallest available model for testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Disable COSMA for simpler testing
        setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);

        // Register batch pipeline
        registerBatchQwenPipeline();
    }

    int rank_;
    int world_size_;
    std::string model_path_;
};

/**
 * @brief Test basic prefillBatch with single sequence
 *
 * Verifies that batch prefill with batch_size=1 produces valid output.
 */
TEST_F(QwenPipelineBatchTest, PrefillBatch_SingleSequence)
{
    if (!std::ifstream(model_path_).good())
    {
        GTEST_SKIP() << "Model file not found: " << model_path_;
    }

    // Load model
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path_)) << "Failed to load model: " << model_path_;

    TransformerLayerConfig base_config = loader.createLayerConfig();
    ModelConfig model_cfg(base_config, "qwen_batch");

    // Create pipeline
    auto pipeline = PipelineFactory::instance().create(model_cfg);
    ASSERT_NE(pipeline, nullptr) << "Failed to create pipeline";

    auto weights = pipeline->loadWeights(model_path_);
    ASSERT_NE(weights, nullptr) << "Failed to load weights";

    // Test sequence
    std::vector<int> tokens = {1, 2, 3, 4, 5};

    // Batch prefill with single sequence
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    std::vector<std::vector<int>> batch = {tokens};

    bool success = pipeline->prefillBatch(batch, *weights, ctx, logits);
    ASSERT_TRUE(success) << "Batch prefill failed";
    ASSERT_NE(logits, nullptr) << "Batch logits is null";

    // Verify output shape
    ASSERT_FALSE(logits->shape().empty());

    const int vocab_size = base_config.vocab_size;
    ASSERT_EQ(logits->shape().back(), vocab_size) << "Last dimension should be vocab_size";

    if (rank_ == 0)
    {
        std::cout << "Batch prefill logits shape: [";
        for (size_t i = 0; i < logits->shape().size(); ++i)
        {
            std::cout << logits->shape()[i];
            if (i < logits->shape().size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

/**
 * @brief Test prefillBatch with multiple sequences of same length
 */
TEST_F(QwenPipelineBatchTest, PrefillBatch_MultipleSequences_SameLength)
{
    if (!std::ifstream(model_path_).good())
    {
        GTEST_SKIP() << "Model file not found: " << model_path_;
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path_));

    TransformerLayerConfig base_config = loader.createLayerConfig();
    ModelConfig model_cfg(base_config, "qwen_batch");
    auto pipeline = PipelineFactory::instance().create(model_cfg);
    ASSERT_NE(pipeline, nullptr);

    auto weights = pipeline->loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);

    // Create batch of sequences with same length
    std::vector<std::vector<int>> batch = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}};

    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    bool success = pipeline->prefillBatch(batch, *weights, ctx, logits);

    ASSERT_TRUE(success) << "Batch prefill failed";
    ASSERT_NE(logits, nullptr);

    // Verify output shape
    const auto &shape = logits->shape();
    ASSERT_GE(shape.size(), 2) << "Expected at least 2D output";

    const int vocab_size = base_config.vocab_size;
    ASSERT_EQ(shape.back(), vocab_size) << "Last dimension should be vocab_size";

    if (rank_ == 0)
    {
        std::cout << "Batch prefill (3 sequences, len=4) shape: [";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i];
            if (i < shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

/**
 * @brief Test error handling with empty batch
 */
TEST_F(QwenPipelineBatchTest, ErrorHandling_EmptyBatch)
{
    if (!std::ifstream(model_path_).good())
    {
        GTEST_SKIP() << "Model file not found: " << model_path_;
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path_));

    TransformerLayerConfig base_config = loader.createLayerConfig();
    ModelConfig model_cfg(base_config, "qwen_batch");
    auto pipeline = PipelineFactory::instance().create(model_cfg);
    ASSERT_NE(pipeline, nullptr);

    auto weights = pipeline->loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);

    // Empty batch should fail gracefully
    std::vector<std::vector<int>> empty_batch;
    StageContext ctx;
    std::shared_ptr<TensorBase> logits;
    bool success = pipeline->prefillBatch(empty_batch, *weights, ctx, logits);

    ASSERT_FALSE(success) << "Empty batch should fail";
}

/**
 * @brief Test multi-step batch generation (prefill + decode)
 *
 * This tests Phase 3.3: batch state tracking across multiple decode steps
 */
TEST_F(QwenPipelineBatchTest, MultiStepGeneration_BatchState)
{
    if (!std::ifstream(model_path_).good())
    {
        GTEST_SKIP() << "Model file not found: " << model_path_;
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path_));

    TransformerLayerConfig base_config = loader.createLayerConfig();
    ModelConfig model_cfg(base_config, "qwen_batch");
    auto pipeline = PipelineFactory::instance().create(model_cfg);
    ASSERT_NE(pipeline, nullptr);

    auto weights = pipeline->loadWeights(model_path_);
    ASSERT_NE(weights, nullptr);

    // Prefill 2 sequences
    std::vector<std::vector<int>> batch = {
        {1, 2, 3},
        {4, 5, 6, 7} // Different length
    };

    StageContext ctx;
    std::shared_ptr<TensorBase> prefill_logits;
    bool success = pipeline->prefillBatch(batch, *weights, ctx, prefill_logits);

    ASSERT_TRUE(success) << "Batch prefill failed";
    ASSERT_NE(prefill_logits, nullptr);
    ASSERT_EQ(prefill_logits->shape()[0], 2) << "Should have 2 sequences";

    // Now decode next token for each sequence (using greedy sampling)
    std::vector<int> next_tokens(2);
    const float *prefill_data = prefill_logits->data();
    int vocab_size = prefill_logits->shape()[1];

    // Argmax for each sequence
    for (int i = 0; i < 2; ++i)
    {
        const float *seq_logits = prefill_data + i * vocab_size;
        int best_token = 0;
        float best_val = seq_logits[0];
        for (int j = 1; j < vocab_size; ++j)
        {
            if (seq_logits[j] > best_val)
            {
                best_val = seq_logits[j];
                best_token = j;
            }
        }
        next_tokens[i] = best_token;
    }

    // First decode step
    std::shared_ptr<TensorBase> decode1_logits;
    success = pipeline->decodeBatch(next_tokens, *weights, ctx, decode1_logits);
    ASSERT_TRUE(success) << "First decode batch failed";
    ASSERT_NE(decode1_logits, nullptr);
    ASSERT_EQ(decode1_logits->shape()[0], 2) << "Should have 2 sequences";

    // Second decode step (reusing next_tokens - in real case would sample again)
    std::shared_ptr<TensorBase> decode2_logits;
    success = pipeline->decodeBatch(next_tokens, *weights, ctx, decode2_logits);
    ASSERT_TRUE(success) << "Second decode batch failed";
    ASSERT_NE(decode2_logits, nullptr);
    ASSERT_EQ(decode2_logits->shape()[0], 2) << "Should have 2 sequences";

    // Third decode step
    std::shared_ptr<TensorBase> decode3_logits;
    success = pipeline->decodeBatch(next_tokens, *weights, ctx, decode3_logits);
    ASSERT_TRUE(success) << "Third decode batch failed";
    ASSERT_NE(decode3_logits, nullptr);
    ASSERT_EQ(decode3_logits->shape()[0], 2) << "Should have 2 sequences";

    if (rank_ == 0)
    {
        std::cout << "Multi-step batch generation successful: prefill + 3 decode steps" << std::endl;
    }
}

/**
 * @brief Main entry point with MPI initialization
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
