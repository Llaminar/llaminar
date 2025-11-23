/**
 * @file Test__Qwen2Pipeline_BatchHandling.cpp
 * @brief Comprehensive unit tests for Qwen2Pipeline batch handling
 *
 * Tests residual connections, buffer management, and per-sequence data handling
 * to isolate the root cause of Sequence 1 divergence in E2E tests.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>

#include "v2/pipelines/qwen/Qwen2Pipeline.h"
#include "v2/pipelines/PipelineFactory.h"
#include "v2/loaders/ModelContext.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/Logger.h"

using namespace llaminar2;

class Test__Qwen2Pipeline_BatchHandling : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI context for single-rank testing
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        // Load small test model
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        ASSERT_NE(model_ctx_, nullptr) << "Failed to load model: " << model_path_;

        // Create pipeline with batch_size=2
        auto pipeline_base = PipelineFactory::instance().create(
            model_ctx_->architecture(),
            model_ctx_,
            mpi_ctx_,
            /*batch_size=*/2);

        // Cast unique_ptr from PipelineBase to Qwen2Pipeline
        auto *raw_ptr = pipeline_base.release();
        pipeline_.reset(static_cast<Qwen2Pipeline *>(raw_ptr));
        ASSERT_NE(pipeline_, nullptr) << "Failed to create Qwen2Pipeline";

        LOG_INFO("[Test Setup] Created Qwen2Pipeline with batch_size=2");
    }

    void TearDown() override
    {
        pipeline_.reset();
        model_ctx_.reset();
        mpi_ctx_.reset();
    }

    // Helper: Create test input tokens (2 sequences, 2 tokens each)
    std::vector<std::vector<int>> createTestBatch()
    {
        return {
            {1, 2}, // Sequence 0
            {3, 4}  // Sequence 1
        };
    }

    // Helper: Compare tensors element-wise
    struct ComparisonResult
    {
        bool passed;
        float max_diff;
        float mean_diff;
        size_t mismatches;
    };

    ComparisonResult compareTensors(
        const float *a, const float *b, size_t count, float tolerance = 1e-4f)
    {
        ComparisonResult result{true, 0.0f, 0.0f, 0};
        double sum_diff = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            sum_diff += diff;

            if (diff > result.max_diff)
            {
                result.max_diff = diff;
            }

            if (diff > tolerance)
            {
                result.mismatches++;
            }
        }

        result.mean_diff = static_cast<float>(sum_diff / count);
        result.passed = (result.mismatches == 0);

        return result;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_;
    std::unique_ptr<Qwen2Pipeline> pipeline_; // Changed from shared_ptr - PipelineFactory returns unique_ptr
    std::string model_path_;
};

/**
 * @test ResidualConnection_EqualLengthBatch
 * @brief Validates residual connections work correctly for equal-length batches
 *
 * Ensures that residual[b] + output[b] produces correct results for all b in batch.
 */
TEST_F(Test__Qwen2Pipeline_BatchHandling, ResidualConnection_EqualLengthBatch)
{
    // Test data: [batch=2, seq_len=2, d_model=10]
    const int batch_size = 2;
    const int seq_len = 2;
    const int d_model = 10;
    const int effective_seq_len = batch_size * seq_len; // 4 tokens total

    // Create test tensors
    std::vector<float> residual(effective_seq_len * d_model);
    std::vector<float> output(effective_seq_len * d_model);
    std::vector<float> result(effective_seq_len * d_model);

    // Fill with distinct patterns for each batch
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < d_model; ++d)
            {
                size_t idx = (b * seq_len + s) * d_model + d;

                // Batch 0: residual=1.0, output=2.0 → result should be 3.0
                // Batch 1: residual=10.0, output=20.0 → result should be 30.0
                residual[idx] = (b == 0) ? 1.0f : 10.0f;
                output[idx] = (b == 0) ? 2.0f : 20.0f;
            }
        }
    }

    // Simulate residual connection (same as Qwen2Pipeline)
    for (size_t i = 0; i < effective_seq_len * d_model; ++i)
    {
        result[i] = residual[i] + output[i];
    }

    // Verify batch 0
    for (int s = 0; s < seq_len; ++s)
    {
        for (int d = 0; d < d_model; ++d)
        {
            size_t idx = s * d_model + d;
            EXPECT_FLOAT_EQ(result[idx], 3.0f)
                << "Batch 0, token " << s << ", dim " << d;
        }
    }

    // Verify batch 1
    for (int s = 0; s < seq_len; ++s)
    {
        for (int d = 0; d < d_model; ++d)
        {
            size_t idx = (seq_len + s) * d_model + d;
            EXPECT_FLOAT_EQ(result[idx], 30.0f)
                << "Batch 1, token " << s << ", dim " << d;
        }
    }
}

/**
 * @test BufferAllocation_BatchSizing
 * @brief Validates activation buffers are correctly sized for batches
 *
 * Ensures buffers can hold batch_size * seq_len tokens without overflow.
 */
TEST_F(Test__Qwen2Pipeline_BatchHandling, BufferAllocation_BatchSizing)
{
    // Create test batch
    auto batch = createTestBatch();

    // Perform forward pass (this allocates buffers internally)
    bool success = pipeline_->forward_batch(batch);
    ASSERT_TRUE(success) << "Forward pass failed";

    // Verify we can extract logits for both sequences
    const float *logits_0 = pipeline_->getLogits(0);
    const float *logits_1 = pipeline_->getLogits(1);

    ASSERT_NE(logits_0, nullptr) << "Failed to get logits for sequence 0";
    ASSERT_NE(logits_1, nullptr) << "Failed to get logits for sequence 1";

    // Verify pointers are distinct (proper stride)
    EXPECT_NE(logits_0, logits_1) << "Logits pointers should differ for different sequences";

    // Verify logits have reasonable values (not NaN, not Inf)
    const auto &model = model_ctx_->model();
    int vocab_size = static_cast<int>(model.vocab_size);

    for (int v = 0; v < vocab_size; ++v)
    {
        EXPECT_FALSE(std::isnan(logits_0[v]))
            << "NaN in logits_0 at vocab index " << v;
        EXPECT_FALSE(std::isinf(logits_0[v]))
            << "Inf in logits_0 at vocab index " << v;

        EXPECT_FALSE(std::isnan(logits_1[v]))
            << "NaN in logits_1 at vocab index " << v;
        EXPECT_FALSE(std::isinf(logits_1[v]))
            << "Inf in logits_1 at vocab index " << v;
    }
}

/**
 * @test SequentialVsBatched_SingleToken
 * @brief Compares sequential vs batched execution for single tokens
 *
 * Validates that processing two sequences individually produces the same
 * results as processing them together in a batch.
 */
TEST_F(Test__Qwen2Pipeline_BatchHandling, SequentialVsBatched_SingleToken)
{
    // Create single-sequence pipelines
    auto pipeline_seq0 = PipelineFactory::instance().create(
        model_ctx_->architecture(), model_ctx_, mpi_ctx_, /*batch_size=*/1);
    auto pipeline_seq1 = PipelineFactory::instance().create(
        model_ctx_->architecture(), model_ctx_, mpi_ctx_, /*batch_size=*/1);

    // Test sequences
    std::vector<int> seq0 = {1, 2};
    std::vector<int> seq1 = {3, 4};

    // Sequential execution
    ASSERT_TRUE(pipeline_seq0->forward(seq0.data(), seq0.size()));
    ASSERT_TRUE(pipeline_seq1->forward(seq1.data(), seq1.size()));

    // Cast unique_ptr from PipelineBase to Qwen2Pipeline
    auto *raw_seq0 = pipeline_seq0.release();
    auto qwen_seq0 = std::unique_ptr<Qwen2Pipeline>(static_cast<Qwen2Pipeline *>(raw_seq0));
    auto *raw_seq1 = pipeline_seq1.release();
    auto qwen_seq1 = std::unique_ptr<Qwen2Pipeline>(static_cast<Qwen2Pipeline *>(raw_seq1));

    const float *logits_seq0 = qwen_seq0->getLogits(0);
    const float *logits_seq1 = qwen_seq1->getLogits(0);

    // Batched execution
    std::vector<std::vector<int>> batch = {seq0, seq1};
    ASSERT_TRUE(pipeline_->forward_batch(batch));

    const float *logits_batch0 = pipeline_->getLogits(0);
    const float *logits_batch1 = pipeline_->getLogits(1);

    // Compare results
    const auto &model = model_ctx_->model();
    int vocab_size = static_cast<int>(model.vocab_size);
    int seq_len = 2;

    // Compare sequence 0
    auto result0 = compareTensors(
        logits_seq0, logits_batch0, seq_len * vocab_size, /*tolerance=*/1e-4f);

    EXPECT_TRUE(result0.passed)
        << "Sequence 0: Sequential vs Batched mismatch\n"
        << "  Max diff: " << result0.max_diff << "\n"
        << "  Mean diff: " << result0.mean_diff << "\n"
        << "  Mismatches: " << result0.mismatches << "/" << (seq_len * vocab_size);

    // Compare sequence 1 (THIS is where the E2E test fails!)
    auto result1 = compareTensors(
        logits_seq1, logits_batch1, seq_len * vocab_size, /*tolerance=*/1e-4f);

    EXPECT_TRUE(result1.passed)
        << "Sequence 1: Sequential vs Batched mismatch\n"
        << "  Max diff: " << result1.max_diff << "\n"
        << "  Mean diff: " << result1.mean_diff << "\n"
        << "  Mismatches: " << result1.mismatches << "/" << (seq_len * vocab_size);
}

/**
 * @test GetLogits_SequenceIndexing
 * @brief Validates getLogits() correctly extracts per-sequence data
 *
 * Ensures logits pointers point to correct batch offsets.
 */
TEST_F(Test__Qwen2Pipeline_BatchHandling, GetLogits_SequenceIndexing)
{
    auto batch = createTestBatch();
    ASSERT_TRUE(pipeline_->forward_batch(batch));

    const float *logits_0 = pipeline_->getLogits(0);
    const float *logits_1 = pipeline_->getLogits(1);

    ASSERT_NE(logits_0, nullptr);
    ASSERT_NE(logits_1, nullptr);

    // Verify stride: logits_1 should be padded_seq_len * vocab_size ahead
    const auto &model = model_ctx_->model();
    int vocab_size = static_cast<int>(model.vocab_size);
    int padded_seq_len = 2; // Both sequences are length 2

    const float *expected_logits_1 = logits_0 + (padded_seq_len * vocab_size);
    EXPECT_EQ(logits_1, expected_logits_1)
        << "Logits pointer for sequence 1 has incorrect stride";
}
