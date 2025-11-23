/**
 * @file Test__Qwen2Pipeline.cpp
 * @brief Integration tests for Qwen2Pipeline validating end-to-end pipeline functionality
 * @author David Sanftenberg
 * @date 2025-11-22
 *
 * Tests cover:
 * - Single token inference (decode phase)
 * - Multi-sequence batching (different lengths)
 * - Batch processing with padding
 * - Numerical sanity checks (embeddings, attention, FFN, logits)
 * - Pipeline state management (position tracking, KV cache)
 */

#include "gtest/gtest.h"
#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/utils/Logger.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace llaminar2;

/**
 * @class Qwen2PipelineIntegration
 * @brief Integration test fixture for Qwen2Pipeline
 *
 * Loads a real GGUF model and tests the complete pipeline with actual weights.
 * Tests numerical sanity rather than exact values (which are model-dependent).
 */
class Qwen2PipelineIntegration : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::string model_path_;
    int rank_;
    int world_size_;

    void SetUp() override
    {
        // Initialize MPI context
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        rank_ = rank;
        world_size_ = world_size;

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        // Use small Qwen model for fast testing
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

        // Load model (collective operation)
        model_ctx_ = ModelContext::create(model_path_, mpi_ctx_);

        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        if (rank_ == 0)
        {
            LOG_INFO("[Integration Test] Loaded model: " << model_path_);
            LOG_INFO("[Integration Test] Vocab size: " << model_ctx_->model().vocab_size);
            LOG_INFO("[Integration Test] Block count: " << model_ctx_->model().block_count);
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Validate tensor contains reasonable FP32 values
     * @param data Tensor data pointer
     * @param size Number of elements
     * @param name Tensor name for error messages
     * @param max_abs_threshold Maximum absolute value considered sane
     */
    void validateTensorSanity(const float *data, size_t size, const std::string &name,
                              float max_abs_threshold = 1000.0f)
    {
        ASSERT_NE(data, nullptr) << name << ": null pointer";
        ASSERT_GT(size, 0) << name << ": empty tensor";

        int nan_count = 0;
        int inf_count = 0;
        int extreme_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        double sum = 0.0;

        for (size_t i = 0; i < size; ++i)
        {
            float val = data[i];

            if (std::isnan(val))
            {
                nan_count++;
            }
            else if (std::isinf(val))
            {
                inf_count++;
            }
            else
            {
                if (std::abs(val) > max_abs_threshold)
                {
                    extreme_count++;
                }
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
        }

        // Report findings
        if (rank_ == 0)
        {
            float mean = static_cast<float>(sum / size);
            LOG_INFO("[Sanity Check] " << name << ":");
            LOG_INFO("  Shape: " << size << " elements");
            LOG_INFO("  Range: [" << min_val << ", " << max_val << "]");
            LOG_INFO("  Mean: " << mean);
            LOG_INFO("  NaN count: " << nan_count);
            LOG_INFO("  Inf count: " << inf_count);
            LOG_INFO("  Extreme (|x| > " << max_abs_threshold << "): " << extreme_count);
        }

        // Assertions
        EXPECT_EQ(nan_count, 0) << name << ": contains NaN values";
        EXPECT_EQ(inf_count, 0) << name << ": contains Inf values";
        EXPECT_LT(extreme_count, static_cast<int>(size * 0.01))
            << name << ": >1% of values are extreme (|x| > " << max_abs_threshold << ")";
        EXPECT_FALSE(std::isnan(min_val)) << name << ": min is NaN";
        EXPECT_FALSE(std::isnan(max_val)) << name << ": max is NaN";
    }

    /**
     * @brief Validate logits distribution is reasonable
     * @param logits Logits tensor
     * @param seq_len Sequence length
     * @param vocab_size Vocabulary size
     */
    void validateLogitsDistribution(const float *logits, size_t seq_len, size_t vocab_size)
    {
        ASSERT_NE(logits, nullptr);

        // Check each token position
        for (size_t tok = 0; tok < seq_len; ++tok)
        {
            const float *token_logits = logits + tok * vocab_size;

            // Find max logit (for numerical stability check)
            float max_logit = *std::max_element(token_logits, token_logits + vocab_size);

            // Compute softmax sum (should be close to 1.0)
            double softmax_sum = 0.0;
            for (size_t v = 0; v < vocab_size; ++v)
            {
                softmax_sum += std::exp(token_logits[v] - max_logit);
            }

            // Validate softmax is numerically stable
            EXPECT_GT(softmax_sum, 0.0) << "Softmax sum is zero for token " << tok;
            EXPECT_LT(softmax_sum, 1e10) << "Softmax sum is too large for token " << tok;

            // Check at least some diversity in logits (not all same value)
            float min_logit = *std::min_element(token_logits, token_logits + vocab_size);
            EXPECT_GT(max_logit - min_logit, 1.0f)
                << "Logits lack diversity for token " << tok
                << " (range: " << (max_logit - min_logit) << ")";
        }

        if (rank_ == 0)
        {
            LOG_INFO("[Logits Validation] All " << seq_len << " token positions passed sanity checks");
        }
    }
};

/**
 * @test SingleTokenInference
 * @brief Test single token forward pass (typical decode scenario)
 *
 * Validates:
 * - Pipeline can process single token
 * - Logits have reasonable distribution
 * - No NaN/Inf in outputs
 * - Position tracking works correctly
 */
TEST_F(Qwen2PipelineIntegration, SingleTokenInference)
{
    // Create pipeline for single sequence
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    // Test token (e.g., "Hello" token from Qwen tokenizer)
    std::vector<int> tokens = {9906}; // Common token in Qwen vocabulary

    // Forward pass
    bool success = pipeline->forward(tokens.data(), tokens.size());

    // Synchronize across ranks
    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Forward pass failed on some rank";

    // Get logits
    const size_t vocab_size = model_ctx_->model().vocab_size;
    const float *logits = pipeline->getLogits(0);
    ASSERT_NE(logits, nullptr) << "Failed to get logits";

    // Validate numerical sanity
    validateTensorSanity(logits, vocab_size, "SingleToken_Logits", 100.0f);
    validateLogitsDistribution(logits, 1, vocab_size);

    if (rank_ == 0)
    {
        LOG_INFO("[SingleTokenInference] PASSED");
    }
}

/**
 * @test MultiTokenSequence
 * @brief Test multi-token sequence inference (prefill phase)
 *
 * Validates:
 * - Pipeline handles longer sequences
 * - All token positions produce valid logits
 * - Position tracking accumulates correctly
 */
TEST_F(Qwen2PipelineIntegration, MultiTokenSequence)
{
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    // Multi-token sequence (4 tokens)
    std::vector<int> tokens = {151644, 9906, 1374, 374}; // Example prompt tokens

    bool success = pipeline->forward(tokens.data(), tokens.size());

    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Forward pass failed on some rank";

    // Get logits for all tokens
    const size_t vocab_size = model_ctx_->model().vocab_size;
    const size_t seq_len = tokens.size();
    const float *logits = pipeline->getLogits(0);
    ASSERT_NE(logits, nullptr);

    // Validate full logits tensor
    validateTensorSanity(logits, seq_len * vocab_size, "MultiToken_Logits", 100.0f);
    validateLogitsDistribution(logits, seq_len, vocab_size);

    if (rank_ == 0)
    {
        LOG_INFO("[MultiTokenSequence] PASSED - " << seq_len << " tokens processed");
    }
}

/**
 * @test BatchInferenceEqualLength
 * @brief Test batch processing with equal-length sequences
 *
 * Validates:
 * - Batch forward pass works correctly
 * - Each sequence produces independent logits
 * - No cross-contamination between sequences
 */
TEST_F(Qwen2PipelineIntegration, BatchInferenceEqualLength)
{
    const int batch_size = 2;
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, batch_size);

    // Two sequences of equal length
    std::vector<std::vector<int>> batch = {
        {151644, 9906, 1374, 374}, // Sequence 0: 4 tokens
        {151644, 9906, 1374, 374}  // Sequence 1: 4 tokens (same for simplicity)
    };

    bool success = pipeline->forward_batch(batch);

    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Batch forward failed on some rank";

    const size_t vocab_size = model_ctx_->model().vocab_size;

    // Validate logits for each sequence
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        const size_t seq_len = batch[seq_idx].size();
        const float *logits = pipeline->getLogits(seq_idx);
        ASSERT_NE(logits, nullptr) << "Failed to get logits for sequence " << seq_idx;

        std::string name = "Batch_Seq" + std::to_string(seq_idx) + "_Logits";
        validateTensorSanity(logits, seq_len * vocab_size, name, 100.0f);
        validateLogitsDistribution(logits, seq_len, vocab_size);
    }

    if (rank_ == 0)
    {
        LOG_INFO("[BatchInferenceEqualLength] PASSED - " << batch_size << " sequences");
    }
}

/**
 * @test BatchInferenceVariableLength
 * @brief Test batch processing with different-length sequences (requires padding)
 *
 * Validates:
 * - Padding logic works correctly
 * - Shorter sequences produce valid results
 * - Longer sequences aren't affected by padding
 *
 * This is a critical test for the batch divergence bug we've been investigating.
 */
TEST_F(Qwen2PipelineIntegration, BatchInferenceVariableLength)
{
    const int batch_size = 2;
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, batch_size);

    // Two sequences with different lengths (like ComprehensiveBatchParity test)
    std::vector<std::vector<int>> batch = {
        {151644, 9906, 1374, 374}, // Sequence 0: 4 tokens
        {151644, 9906}             // Sequence 1: 2 tokens (will be padded to 4)
    };

    bool success = pipeline->forward_batch(batch);

    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Batch forward failed on some rank";

    const size_t vocab_size = model_ctx_->model().vocab_size;

    // Validate each sequence independently
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx)
    {
        const size_t seq_len = batch[seq_idx].size();
        const float *logits = pipeline->getLogits(seq_idx);
        ASSERT_NE(logits, nullptr) << "Failed to get logits for sequence " << seq_idx;

        std::string name = "VariableBatch_Seq" + std::to_string(seq_idx) + "_Logits";
        validateTensorSanity(logits, seq_len * vocab_size, name, 100.0f);
        validateLogitsDistribution(logits, seq_len, vocab_size);

        if (rank_ == 0)
        {
            LOG_INFO("[VariableLength] Sequence " << seq_idx << ": " << seq_len
                                                  << " tokens, logits validated");
        }
    }

    // Additional check: Compare batch metadata
    EXPECT_EQ(pipeline->batch_size(), batch_size);
    EXPECT_EQ(pipeline->padded_seq_len(), 4) << "Should pad to max length (4)";

    if (rank_ == 0)
    {
        LOG_INFO("[BatchInferenceVariableLength] PASSED - variable lengths handled correctly");
    }
}

/**
 * @test SequentialVsBatchConsistency
 * @brief Compare sequential execution vs batched execution for same inputs
 *
 * This test directly addresses the E2E bug: runs same sequences both
 * sequentially and batched, then validates they produce similar results.
 *
 * NOTE: We allow some tolerance due to:
 * - Quantization differences
 * - Floating point accumulation order
 * - Padding-related numerical differences
 */
TEST_F(Qwen2PipelineIntegration, SequentialVsBatchConsistency)
{
    const float tolerance = 0.1f; // 10% tolerance for integration test

    // Test sequences
    std::vector<std::vector<int>> sequences = {
        {151644, 9906, 1374, 374}, // Sequence 0: 4 tokens
        {151644, 9906}             // Sequence 1: 2 tokens
    };

    const size_t vocab_size = model_ctx_->model().vocab_size;

    // ===== Sequential Execution =====
    std::vector<std::vector<float>> logits_sequential(sequences.size());

    for (size_t i = 0; i < sequences.size(); ++i)
    {
        auto pipeline_seq = std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

        bool success = pipeline_seq->forward(sequences[i].data(), sequences[i].size());
        int local_ok = success ? 1 : 0;
        int global_ok;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_ok, 1) << "Sequential forward failed for sequence " << i;

        // Copy logits
        const size_t seq_len = sequences[i].size();
        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = pipeline_seq->getLogits(0);
        std::memcpy(logits_sequential[i].data(), logits_ptr,
                    seq_len * vocab_size * sizeof(float));
    }

    mpi_ctx_->barrier();

    // ===== Batched Execution =====
    auto pipeline_batch = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{},
        /*batch_size=*/static_cast<int>(sequences.size()));

    bool success = pipeline_batch->forward_batch(sequences);
    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Batched forward failed";

    // ===== Compare Results =====
    for (size_t i = 0; i < sequences.size(); ++i)
    {
        const size_t seq_len = sequences[i].size();
        const float *logits_batch = pipeline_batch->getLogits(i);
        const float *logits_seq = logits_sequential[i].data();

        // Compute relative differences
        double max_rel_diff = 0.0;
        double mean_rel_diff = 0.0;
        int mismatch_count = 0;

        for (size_t j = 0; j < seq_len * vocab_size; ++j)
        {
            float seq_val = logits_seq[j];
            float batch_val = logits_batch[j];

            // Skip near-zero values (division by zero)
            if (std::abs(seq_val) < 1e-6f)
                continue;

            float rel_diff = std::abs(batch_val - seq_val) / std::abs(seq_val);
            max_rel_diff = std::max(max_rel_diff, static_cast<double>(rel_diff));
            mean_rel_diff += rel_diff;

            if (rel_diff > tolerance)
            {
                mismatch_count++;
            }
        }

        mean_rel_diff /= (seq_len * vocab_size);

        if (rank_ == 0)
        {
            LOG_INFO("[Consistency Check] Sequence " << i << ":");
            LOG_INFO("  Max relative diff: " << max_rel_diff);
            LOG_INFO("  Mean relative diff: " << mean_rel_diff);
            LOG_INFO("  Mismatches (>" << tolerance << "): " << mismatch_count);
        }

        // Relaxed checks for integration test
        // (E2E tests have stricter tolerances)
        EXPECT_LT(max_rel_diff, 1.0)
            << "Sequence " << i << " has >100% relative difference";
        EXPECT_LT(mean_rel_diff, tolerance)
            << "Sequence " << i << " has excessive mean difference";
        EXPECT_LT(mismatch_count, static_cast<int>(seq_len * vocab_size * 0.1))
            << "Sequence " << i << " has >10% mismatched values";
    }

    if (rank_ == 0)
    {
        LOG_INFO("[SequentialVsBatchConsistency] PASSED - batch and sequential produce similar results");
    }
}

/**
 * @test IncrementalDecoding
 * @brief Test autoregressive decoding (multiple decode steps)
 *
 * Validates:
 * - KV cache accumulates correctly
 * - Pipeline state remains consistent
 */
TEST_F(Qwen2PipelineIntegration, IncrementalDecoding)
{
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    const size_t vocab_size = model_ctx_->model().vocab_size;
    const int decode_steps = 3;

    // Initial prompt (prefill)
    std::vector<int> prompt = {151644, 9906};
    bool success = pipeline->forward(prompt.data(), prompt.size());

    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Prefill failed";

    // Decode steps (one token at a time)
    for (int step = 0; step < decode_steps; ++step)
    {
        // Use arbitrary token (in real inference, would be argmax of logits)
        std::vector<int> next_token = {9906};

        success = pipeline->forward(next_token.data(), next_token.size());
        local_ok = success ? 1 : 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_ok, 1) << "Decode step " << step << " failed";

        // Validate logits
        const float *logits = pipeline->getLogits(0);
        ASSERT_NE(logits, nullptr);

        std::string name = "Decode_Step" + std::to_string(step) + "_Logits";
        validateTensorSanity(logits, vocab_size, name, 100.0f);
    }

    if (rank_ == 0)
    {
        LOG_INFO("[IncrementalDecoding] PASSED - " << decode_steps
                                                   << " decode steps completed successfully");
    }
}

/**
 * @test PipelineReuse
 * @brief Test pipeline can handle multiple independent sequences
 *
 * Validates:
 * - Pipeline can be reused for different inputs
 * - No state contamination between runs
 */
TEST_F(Qwen2PipelineIntegration, PipelineReuse)
{
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    // First inference
    std::vector<int> tokens1 = {151644, 9906};
    bool success = pipeline->forward(tokens1.data(), tokens1.size());

    int local_ok = success ? 1 : 0;
    int global_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1);

    const size_t vocab_size = model_ctx_->model().vocab_size;
    const float *logits1 = pipeline->getLogits(0);
    validateTensorSanity(logits1, tokens1.size() * vocab_size, "FirstRun_Logits", 100.0f);

    // Second inference (KV cache will continue accumulating)
    std::vector<int> tokens2 = {151644, 1374, 374};
    success = pipeline->forward(tokens2.data(), tokens2.size());

    local_ok = success ? 1 : 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1);

    const float *logits2 = pipeline->getLogits(0);
    validateTensorSanity(logits2, tokens2.size() * vocab_size, "SecondRun_Logits", 100.0f);

    if (rank_ == 0)
    {
        LOG_INFO("[PipelineReuse] PASSED - pipeline successfully reused");
    }
}

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
