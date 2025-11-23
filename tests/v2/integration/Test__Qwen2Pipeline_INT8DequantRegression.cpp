/**
 * @file Test__Qwen2Pipeline_INT8DequantRegression.cpp
 * @brief Integration regression test for INT8 dequantization buffer bounds bug
 *
 * @details
 * This integration test prevents regression of the bug where INT8 dequantization
 * loops iterated over total_rows instead of effective_seq_len, accessing
 * uninitialized scale values and causing 97 billion % divergence in batched inference.
 *
 * Classified as Integration test because it:
 * - Loads full GGUF model file (qwen2.5-0.5b-instruct-q4_0.gguf)
 * - Runs complete forward passes through all 24 layers
 * - Tests end-to-end batch vs sequential parity
 * - Takes ~60-90 seconds to execute (too slow for unit test suite)
 *
 * Bug History:
 * - Date: 2025-11-23
 * - Location: Qwen2Pipeline.cpp lines 584, 1009
 * - Impact: Batched inference diverged catastrophically (97B% error)
 * - Root Cause: Dequantization loops used total_rows = batch_size * padded_seq_len
 *               instead of effective_seq_len, accessing garbage scale values
 * - Fix: Changed loops to only process effective_seq_len rows
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

#include "v2/pipelines/qwen/Qwen2Pipeline.h"
#include "v2/pipelines/PipelineFactory.h"
#include "v2/loaders/ModelContext.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Regression test fixture for INT8 dequantization bug
 *
 * Tests that batched inference with padding produces identical results to
 * sequential inference, ensuring dequantization loops respect effective_seq_len.
 */
class Test__Qwen2Pipeline_INT8DequantRegression : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use small Qwen model for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Create MPI context (rank 0, world size 1 for unit test)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);

        // Load model
        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);
        ASSERT_NE(model_ctx_, nullptr) << "Failed to load model: " << model_path_;

        LOG_INFO("[Test Setup] Loaded model: " << model_path_);
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_.reset();
    }

    /**
     * @brief Helper to run inference on a sequence
     * @return Logits vector for the last token
     */
    std::vector<float> runInference(PipelineBase *pipeline, const std::vector<int> &tokens)
    {
        const int seq_len = tokens.size();
        const int vocab_size = 151936; // Qwen 2.5 vocab size

        // Flatten token IDs
        std::vector<int> input_ids = tokens;

        // Run forward pass
        bool success = pipeline->forward(input_ids.data(), seq_len);
        EXPECT_TRUE(success) << "Forward pass failed";

        // Extract logits for last token (batch index 0)
        const float *logits_ptr = pipeline->getLogits(0);
        EXPECT_NE(logits_ptr, nullptr) << "getLogits() returned null";

        std::vector<float> result(vocab_size);
        if (logits_ptr)
        {
            std::memcpy(result.data(), logits_ptr, vocab_size * sizeof(float));
        }

        return result;
    }

    /**
     * @brief Compare two vectors element-wise
     */
    struct ComparisonResult
    {
        bool passed;
        float max_diff;
        float mean_diff;
        float max_rel_diff;
        size_t mismatches;
    };

    ComparisonResult compareVectors(
        const std::vector<float> &a,
        const std::vector<float> &b,
        float abs_tolerance = 1e-4f,
        float rel_tolerance = 1e-3f)
    {
        ComparisonResult result{true, 0.0f, 0.0f, 0.0f, 0};

        EXPECT_EQ(a.size(), b.size()) << "Vector sizes differ";
        if (a.size() != b.size())
        {
            result.passed = false;
            return result;
        }

        double sum_diff = 0.0;
        size_t count = a.size();

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            float rel_diff = 0.0f;

            if (std::abs(b[i]) > 1e-8f)
            {
                rel_diff = diff / std::abs(b[i]);
            }

            result.max_diff = std::max(result.max_diff, diff);
            result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
            sum_diff += diff;

            if (diff > abs_tolerance && rel_diff > rel_tolerance)
            {
                result.mismatches++;
                if (result.mismatches <= 5)
                { // Log first 5 mismatches
                    LOG_ERROR("Mismatch at index " << i << ": a=" << a[i]
                                                   << ", b=" << b[i] << ", diff=" << diff
                                                   << ", rel_diff=" << rel_diff);
                }
            }
        }

        result.mean_diff = static_cast<float>(sum_diff / count);
        result.passed = (result.mismatches == 0);

        return result;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_;
    std::string model_path_;
};

/**
 * @test BatchedVsSequential_PaddedBatch
 * @brief Core regression test: batched inference with padding must match sequential
 *
 * This is the PRIMARY regression test that catches the original bug:
 * - Create batch with unequal sequence lengths (forcing padding)
 * - Run sequences individually (sequential mode)
 * - Run sequences together (batched mode)
 * - Verify outputs are identical
 *
 * If INT8 dequantization loops incorrectly use total_rows instead of effective_seq_len,
 * this test will FAIL with large divergence (previously 97B%).
 */
TEST_F(Test__Qwen2Pipeline_INT8DequantRegression, BatchedVsSequential_PaddedBatch)
{
    LOG_INFO("=== Regression Test: Batched vs Sequential with Padding ===");

    // Test sequences with different lengths (forces padding)
    std::vector<int> seq0 = {151644, 9906, 77091, 103383}; // 4 tokens
    std::vector<int> seq1 = {151644, 9906};                // 2 tokens (will be padded)

    // NOTE: We use FP32 activations instead of INT8 because INT8 activation support
    //       has separate bugs (memory corruption in FusedRMSNormQuantize).
    //       The batch padding divergence bug we're testing occurs in the INT8
    //       dequantization loops, which are ALSO used when weights are INT8
    //       (even if activations are FP32).
    PipelineConfig config;
    config.activation_precision = ActivationPrecision::FP32;

    // PART 1: Run sequences individually (sequential baseline)
    LOG_INFO("Running Sequence 0 individually (4 tokens)...");
    config.batch_size = 1;
    auto pipeline_seq0 = PipelineFactory::instance().create(
        model_ctx_->architecture(), model_ctx_, mpi_ctx_, -1, config);
    ASSERT_NE(pipeline_seq0, nullptr) << "Failed to create pipeline for seq0";
    auto *qwen2_seq0 = dynamic_cast<Qwen2Pipeline *>(pipeline_seq0.get());
    ASSERT_NE(qwen2_seq0, nullptr) << "Pipeline seq0 is not a Qwen2Pipeline";
    auto output_seq0 = runInference(qwen2_seq0, seq0);

    LOG_INFO("Running Sequence 1 individually (2 tokens)...");
    config.batch_size = 1;
    auto pipeline_seq1 = PipelineFactory::instance().create(
        model_ctx_->architecture(), model_ctx_, mpi_ctx_, -1, config);
    ASSERT_NE(pipeline_seq1, nullptr) << "Failed to create pipeline for seq1";
    auto *qwen2_seq1 = dynamic_cast<Qwen2Pipeline *>(pipeline_seq1.get());
    ASSERT_NE(qwen2_seq1, nullptr) << "Pipeline seq1 is not a Qwen2Pipeline";
    auto output_seq1 = runInference(qwen2_seq1, seq1);

    // PART 2: Run sequences together in batch (with padding)
    LOG_INFO("Running both sequences in batch (with padding)...");
    config.batch_size = 2;
    auto pipeline_batch = PipelineFactory::instance().create(
        model_ctx_->architecture(), model_ctx_, mpi_ctx_, -1, config);
    ASSERT_NE(pipeline_batch, nullptr) << "Failed to create batched pipeline";

    // Create batched input using forward_batch API
    const int vocab_size = 151936; // Qwen 2.5 vocab size
    std::vector<std::vector<int>> token_batches;
    token_batches.push_back(std::vector<int>(seq0.begin(), seq0.end()));
    token_batches.push_back(std::vector<int>(seq1.begin(), seq1.end()));

    // Cast to Qwen2Pipeline to access forward_batch (architecture-specific API)
    auto *qwen2_batch = dynamic_cast<Qwen2Pipeline *>(pipeline_batch.get());
    ASSERT_NE(qwen2_batch, nullptr) << "Pipeline is not a Qwen2Pipeline";

    bool success = qwen2_batch->forward_batch(token_batches);
    ASSERT_TRUE(success) << "Batched forward pass failed";

    // Extract logits from batch (each sequence gets logits for its last valid token)
    const float *logits_seq0 = pipeline_batch->getLogits(0);
    const float *logits_seq1 = pipeline_batch->getLogits(1);
    ASSERT_NE(logits_seq0, nullptr) << "getLogits(0) returned null";
    ASSERT_NE(logits_seq1, nullptr) << "getLogits(1) returned null";

    std::vector<float> batched_seq0(vocab_size);
    std::vector<float> batched_seq1(vocab_size);
    std::memcpy(batched_seq0.data(), logits_seq0, vocab_size * sizeof(float));
    std::memcpy(batched_seq1.data(), logits_seq1, vocab_size * sizeof(float));

    // PART 3: Verify batched outputs match sequential outputs
    LOG_INFO("Comparing Sequence 0 outputs...");
    auto cmp0 = compareVectors(output_seq0, batched_seq0, 1e-4f, 1e-3f);
    LOG_INFO("  Max diff: " << cmp0.max_diff
                            << ", Mean diff: " << cmp0.mean_diff
                            << ", Max rel diff: " << (cmp0.max_rel_diff * 100.0f) << "%"
                            << ", Mismatches: " << cmp0.mismatches);

    LOG_INFO("Comparing Sequence 1 outputs (THE CRITICAL TEST)...");
    auto cmp1 = compareVectors(output_seq1, batched_seq1, 1e-4f, 1e-3f);
    LOG_INFO("  Max diff: " << cmp1.max_diff
                            << ", Mean diff: " << cmp1.mean_diff
                            << ", Max rel diff: " << (cmp1.max_rel_diff * 100.0f) << "%"
                            << ", Mismatches: " << cmp1.mismatches);

    // ASSERT: Both sequences must match within tolerance
    EXPECT_TRUE(cmp0.passed) << "Sequence 0 diverged: "
                             << cmp0.mismatches << " mismatches, "
                             << "max_rel_diff=" << (cmp0.max_rel_diff * 100.0f) << "%";

    EXPECT_TRUE(cmp1.passed) << "Sequence 1 diverged (INT8 DEQUANT BUG DETECTED!): "
                             << cmp1.mismatches << " mismatches, "
                             << "max_rel_diff=" << (cmp1.max_rel_diff * 100.0f) << "%"
                             << "\n\nIf this test fails, check Qwen2Pipeline.cpp INT8 dequantization loops:"
                             << "\n  - attention_block() around line 584"
                             << "\n  - ffn_block() around line 1009"
                             << "\n\nEnsure loops use effective_seq_len, NOT total_rows!";

    // Additional diagnostic: if Seq1 fails, likely indicates the bug has returned
    if (!cmp1.passed && cmp1.max_rel_diff > 1000.0f)
    {
        FAIL() << "\n\n"
               << "╔════════════════════════════════════════════════════════════════╗\n"
               << "║  INT8 DEQUANTIZATION BUG REGRESSION DETECTED!                  ║\n"
               << "╠════════════════════════════════════════════════════════════════╣\n"
               << "║  Sequence 1 diverged by " << (cmp1.max_rel_diff * 100.0f) << "%\n"
               << "║  This indicates INT8 dequantization loops are accessing       ║\n"
               << "║  uninitialized scale values beyond effective_seq_len.         ║\n"
               << "║                                                                ║\n"
               << "║  CHECK: Qwen2Pipeline.cpp                                     ║\n"
               << "║    - attention_block() dequant loop (~line 584)               ║\n"
               << "║    - ffn_block() dequant loop (~line 1009)                    ║\n"
               << "║                                                                ║\n"
               << "║  MUST USE: for (int r = 0; r < effective_seq_len; ++r)       ║\n"
               << "║  NOT:      for (int r = 0; r < total_rows; ++r)               ║\n"
               << "╚════════════════════════════════════════════════════════════════╝\n";
    }

    LOG_INFO("✅ Regression test PASSED - INT8 dequantization bug not present");
}

/**
 * @test MultipleBatchSizes
 * @brief Test various batch size configurations to ensure robustness
 *
 * Validates the fix works across different batch sizes and sequence length ratios.
 */
TEST_F(Test__Qwen2Pipeline_INT8DequantRegression, MultipleBatchSizes)
{
    LOG_INFO("=== Testing Multiple Batch Size Configurations ===");

    // Test configurations: (batch_size, seq_lengths)
    struct TestConfig
    {
        int batch_size;
        std::vector<std::vector<int>> sequences;
        std::string description;
    };

    std::vector<TestConfig> configs = {
        {2, {{151644, 9906}, {151644, 9906, 77091}}, "Batch=2, lengths=[2,3]"},
        {3, {{151644}, {151644, 9906}, {151644, 9906, 77091}}, "Batch=3, lengths=[1,2,3]"},
        {4, {{151644, 9906}, {151644}, {151644, 9906, 77091}, {151644, 9906}}, "Batch=4, mixed"},
    };

    for (const auto &test_config : configs)
    {
        LOG_INFO("Testing: " << test_config.description);

        // Find max sequence length for padding
        int max_len = 0;
        for (const auto &seq : test_config.sequences)
        {
            max_len = std::max(max_len, static_cast<int>(seq.size()));
        }

        // Run each sequence individually
        PipelineConfig config;
        config.activation_precision = ActivationPrecision::FP32;
        config.batch_size = 1;
        std::vector<std::vector<float>> sequential_outputs;
        for (const auto &seq : test_config.sequences)
        {
            auto pipeline = PipelineFactory::instance().create(
                model_ctx_->architecture(), model_ctx_, mpi_ctx_, -1, config);
            ASSERT_NE(pipeline, nullptr) << "Failed to create sequential pipeline";
            auto *qwen2_pipeline = dynamic_cast<Qwen2Pipeline *>(pipeline.get());
            ASSERT_NE(qwen2_pipeline, nullptr) << "Pipeline is not a Qwen2Pipeline";
            sequential_outputs.push_back(runInference(qwen2_pipeline, seq));
        }

        // Run all sequences in batch
        config.batch_size = test_config.batch_size;
        auto pipeline_batch = PipelineFactory::instance().create(
            model_ctx_->architecture(), model_ctx_, mpi_ctx_, -1, config);
        ASSERT_NE(pipeline_batch, nullptr) << "Failed to create batched pipeline";

        // Use forward_batch API with vector of sequences
        std::vector<std::vector<int>> token_batches = test_config.sequences;

        // Cast to Qwen2Pipeline to access forward_batch (architecture-specific API)
        auto *qwen2_batch = dynamic_cast<Qwen2Pipeline *>(pipeline_batch.get());
        ASSERT_NE(qwen2_batch, nullptr) << "Pipeline is not a Qwen2Pipeline";

        bool success = qwen2_batch->forward_batch(token_batches);
        ASSERT_TRUE(success) << "Batch forward failed: " << test_config.description;

        // Compare outputs (logits for each sequence)
        const int vocab_size = 151936;

        bool all_passed = true;
        for (size_t b = 0; b < test_config.sequences.size(); ++b)
        {
            const float *logits_ptr = pipeline_batch->getLogits(b);
            ASSERT_NE(logits_ptr, nullptr) << "getLogits(" << b << ") returned null";

            std::vector<float> batched_output(vocab_size);
            std::memcpy(batched_output.data(), logits_ptr, vocab_size * sizeof(float));

            auto cmp = compareVectors(sequential_outputs[b], batched_output, 1e-4f, 1e-3f);
            if (!cmp.passed)
            {
                LOG_ERROR("  Sequence " << b << " FAILED: max_rel_diff="
                                        << (cmp.max_rel_diff * 100.0f) << "%");
                all_passed = false;
            }
        }

        EXPECT_TRUE(all_passed) << "Config failed: " << test_config.description;
        if (all_passed)
        {
            LOG_INFO("  ✅ " << test_config.description << " PASSED");
        }
    }
}
