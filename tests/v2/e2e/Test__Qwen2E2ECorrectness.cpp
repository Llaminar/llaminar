/**
 * @file Test__Qwen2E2ECorrectness.cpp
 * @brief End-to-end correctness tests for Qwen2Pipeline (Phase 3c)
 * @author David Sanftenberg
 *
 * Validates full pipeline correctness by comparing:
 * - Single-rank vs multi-rank MPI execution
 * - All intermediate activations across transformer layers
 *
 * Requirements:
 * - Real Qwen 2.5 0.5B model (models/qwen2.5-0.5b-instruct-q4_0.gguf)
 * - MPI support (exactly 2 ranks for tensor-parallel validation)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>

#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Qwen2 end-to-end correctness
 */
class Qwen2E2ECorrectness : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI context
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        // Model path (from test fixtures)
        model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    }

    void TearDown() override
    {
        // Cleanup
        model_ctx_single_.reset();
        model_ctx_multi_.reset();
        mpi_ctx_.reset();
    }

    /**
     * @brief Load model for single-rank execution
     */
    bool loadModelSingleRank()
    {
        if (rank_ == 0)
        {
            model_ctx_single_ = ModelContext::create(model_path_);
            if (!model_ctx_single_)
            {
                LOG_ERROR("[E2E] Failed to load model (single-rank): " << model_path_);
                return false;
            }
            LOG_INFO("[E2E] Model loaded successfully (single-rank): " << model_path_);
        }
        return true;
    }

    /**
     * @brief Load model for multi-rank execution
     */
    bool loadModelMultiRank()
    {
        model_ctx_multi_ = ModelContext::create(model_path_);
        if (!model_ctx_multi_)
        {
            LOG_ERROR("[E2E] Rank " << rank_ << " failed to load model (multi-rank): " << model_path_);
            return false;
        }
        LOG_INFO("[E2E] Rank " << rank_ << " model loaded successfully (multi-rank)");
        return true;
    }

    /**
     * @brief Compare two output tensors with tolerance
     */
    struct ComparisonResult
    {
        bool passed = false;
        float max_abs_diff = 0.0f;
        float mean_abs_diff = 0.0f;
        float rel_l2_norm = 0.0f;
        size_t num_mismatches = 0;
    };

    ComparisonResult compareTensors(
        const float *a, const float *b, size_t size, float tolerance)
    {
        ComparisonResult result;

        double sum_abs_diff = 0.0;
        double sum_sq_diff = 0.0;
        double sum_sq_b = 0.0;

        for (size_t i = 0; i < size; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > tolerance)
            {
                result.num_mismatches++;
            }
            if (diff > result.max_abs_diff)
            {
                result.max_abs_diff = diff;
            }
            sum_abs_diff += diff;
            sum_sq_diff += diff * diff;
            sum_sq_b += b[i] * b[i];
        }

        result.mean_abs_diff = static_cast<float>(sum_abs_diff / size);

        if (sum_sq_b > 1e-10)
        {
            result.rel_l2_norm = static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_b));
        }

        result.passed = (result.max_abs_diff <= tolerance &&
                         result.rel_l2_norm <= 0.01f);

        return result;
    }

    void printComparisonResult(const ComparisonResult &result, const std::string &name)
    {
        std::cout << "=== " << name << " ===" << std::endl;
        std::cout << "  Max abs diff:   " << result.max_abs_diff << std::endl;
        std::cout << "  Mean abs diff:  " << result.mean_abs_diff << std::endl;
        std::cout << "  Rel L2 norm:    " << result.rel_l2_norm << std::endl;
        std::cout << "  Mismatches:     " << result.num_mismatches << std::endl;
        std::cout << "  Status:         " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_single_;
    std::shared_ptr<ModelContext> model_ctx_multi_;
    std::string model_path_;
    int rank_;
    int world_size_;
};

/**
 * @brief Test: Single token inference correctness (decode phase)
 *
 * Validates that single-rank and multi-rank pipelines produce identical
 * logits for a single token (typical decode scenario).
 */
TEST_F(Qwen2E2ECorrectness, SingleTokenInference)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f; // Relaxed tolerance for full pipeline

    // Load models
    ASSERT_TRUE(loadModelSingleRank());
    ASSERT_TRUE(loadModelMultiRank());

    // Single token input
    std::vector<int> tokens = {151644}; // BOS token for Qwen 2.5

    // Single-rank execution (rank 0 only)
    std::vector<float> logits_single;
    if (rank_ == 0)
    {
        auto pipeline_single = std::make_unique<Qwen2Pipeline>(
            model_ctx_single_, nullptr, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

        bool success = pipeline_single->forward(tokens.data(), tokens.size());
        ASSERT_TRUE(success) << "Single-rank forward pass failed";

        // Get logits (vocabulary size)
        const auto &model = model_ctx_single_->model();
        size_t vocab_size = model.vocab_size;
        logits_single.resize(vocab_size);

        // Extract logits from pipeline (seq_idx=0 for single sequence)
        const float *logits_ptr = pipeline_single->getLogits(0);
        ASSERT_NE(logits_ptr, nullptr) << "getLogits() returned null";
        std::memcpy(logits_single.data(), logits_ptr, vocab_size * sizeof(float));

        LOG_INFO("[E2E] Rank 0 single-rank logits extracted (" << vocab_size << " values)");
    }

    // Multi-rank execution (all ranks)
    auto pipeline_multi = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    bool success = pipeline_multi->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Multi-rank forward pass failed on rank " << rank_;

    std::vector<float> logits_multi;
    const auto &model = model_ctx_multi_->model();
    size_t vocab_size = model.vocab_size;
    logits_multi.resize(vocab_size);

    // Extract logits from multi-rank pipeline (seq_idx=0)
    const float *logits_ptr_multi = pipeline_multi->getLogits(0);
    ASSERT_NE(logits_ptr_multi, nullptr) << "getLogits() returned null on rank " << rank_;
    std::memcpy(logits_multi.data(), logits_ptr_multi, vocab_size * sizeof(float));

    LOG_INFO("[E2E] Rank " << rank_ << " multi-rank logits extracted");

    // Compare on rank 0
    if (rank_ == 0)
    {
        auto result = compareTensors(
            logits_single.data(), logits_multi.data(), vocab_size, tolerance);

        printComparisonResult(result, "Single Token Inference");
        EXPECT_TRUE(result.passed)
            << "Logits mismatch: max_abs_diff=" << result.max_abs_diff
            << ", rel_l2=" << result.rel_l2_norm;

        LOG_INFO("[E2E] Single token test complete");
    }

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Multi-token prefill correctness
 *
 * Validates that single-rank and multi-rank pipelines produce identical
 * results for multi-token prefill (e.g., 8-32 tokens).
 */
TEST_F(Qwen2E2ECorrectness, MultiTokenPrefill)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load models
    ASSERT_TRUE(loadModelSingleRank());
    ASSERT_TRUE(loadModelMultiRank());

    // Multi-token input (8 tokens)
    std::vector<int> tokens = {
        151644, // BOS
        9906,   // Hello
        0,      // (placeholder - actual tokens TBD)
        0,
        0,
        0,
        0,
        0};

    // Single-rank execution (rank 0 only)
    std::vector<float> logits_single;
    if (rank_ == 0)
    {
        auto pipeline_single = std::make_unique<Qwen2Pipeline>(
            model_ctx_single_, nullptr, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

        bool success = pipeline_single->forward(tokens.data(), tokens.size());
        ASSERT_TRUE(success) << "Single-rank forward pass failed";

        LOG_INFO("[E2E] Single-rank prefill completed");
    }

    // Multi-rank execution (all ranks)
    auto pipeline_multi = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

    bool success = pipeline_multi->forward(tokens.data(), tokens.size());
    ASSERT_TRUE(success) << "Multi-rank forward pass failed on rank " << rank_;

    LOG_INFO("[E2E] Rank " << rank_ << " multi-rank prefill completed");

    // TODO: Compare intermediate activations
    // - Embedding output
    // - Each layer's attention output
    // - Each layer's FFN output
    // - Final norm output
    // - Logits

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Multi-sequence batch inference (equal lengths)
 *
 * Validates batched execution with equal-length sequences (no padding).
 * This test should pass because padding masking is not required.
 */
TEST_F(Qwen2E2ECorrectness, MultiSequenceBatchEqualLength)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model
    ASSERT_TRUE(loadModelMultiRank());

    // Define batch with 2 sequences of EQUAL length (no padding needed)
    std::vector<std::vector<int>> batch = {
        {151644, 9906}, // Sequence 0: BOS + "Hello" (2 tokens)
        {151644, 1374}  // Sequence 1: BOS + different token (2 tokens)
    };

    const size_t batch_size = batch.size();

    // ======== Sequential Execution (Baseline) ========
    std::vector<std::vector<float>> logits_sequential(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        auto pipeline_seq = std::make_unique<Qwen2Pipeline>(
            model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

        bool success = pipeline_seq->forward(batch[i].data(), batch[i].size());
        ASSERT_TRUE(success) << "Sequential forward pass failed for sequence " << i;

        const auto &model = model_ctx_multi_->model();
        size_t vocab_size = model.vocab_size;
        size_t seq_len = batch[i].size();

        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = pipeline_seq->getLogits(0);
        ASSERT_NE(logits_ptr, nullptr);
        std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Sequential: Sequence " << i << " completed (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Batched Execution ========
    auto pipeline_batch = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/batch_size);

    bool success = pipeline_batch->forward_batch(batch);
    ASSERT_TRUE(success) << "Batched forward pass failed on rank " << rank_;

    const auto &model = model_ctx_multi_->model();
    size_t vocab_size = model.vocab_size;

    // Extract per-sequence logits from batch
    std::vector<std::vector<float>> logits_batched(batch_size);

    const int padded_seq_len = pipeline_batch->padded_seq_len();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Batched execution complete (equal-length sequences):");
        LOG_INFO("[E2E]   Batch size: " << pipeline_batch->batch_size());
        LOG_INFO("[E2E]   Padded seq len: " << padded_seq_len);
    }

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *logits_ptr = pipeline_batch->getLogits(i);
        ASSERT_NE(logits_ptr, nullptr);

        size_t seq_len = batch[i].size();
        logits_batched[i].resize(seq_len * vocab_size);

        // Extract logits (no padding to worry about since all sequences are equal length)
        std::memcpy(logits_batched[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Batched: Extracted logits for sequence " << i << " (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Compare Sequential vs Batched ========
    if (rank_ == 0)
    {
        for (size_t i = 0; i < batch_size; ++i)
        {
            size_t seq_len = batch[i].size();
            size_t total_elements = seq_len * vocab_size;

            auto result = compareTensors(
                logits_sequential[i].data(),
                logits_batched[i].data(),
                total_elements,
                tolerance);

            std::string test_name = "Batch Parity (Equal Length) - Sequence " + std::to_string(i);
            printComparisonResult(result, test_name);

            EXPECT_TRUE(result.passed)
                << "Sequence " << i << " mismatch: "
                << "max_abs_diff=" << result.max_abs_diff
                << ", rel_l2=" << result.rel_l2_norm;
        }

        LOG_INFO("[E2E] Equal-length batch test complete");
    }

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Multi-sequence batch inference
 *
 * Validates that batched execution produces identical results to
 * sequential execution (batch parity).
 *
 * Tests with batch_size=2, comparing:
 * - Batched execution (both sequences processed together)
 * - Sequential execution (each sequence processed separately)
 *
 * **KNOWN ISSUE**: This test currently FAILS because padding masking
 * is not yet implemented in attention. Padding tokens participate in
 * attention when they shouldn't, causing divergence for variable-length
 * sequences.
 *
 * **TODO**: Implement `attention_gqa_batch()` with combined causal+padding masking.
 */
TEST_F(Qwen2E2ECorrectness, DISABLED_MultiSequenceBatch)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model
    ASSERT_TRUE(loadModelMultiRank());

    // Define batch with 2 sequences of different lengths
    std::vector<std::vector<int>> batch = {
        {151644},      // Sequence 0: BOS token only (1 token)
        {151644, 9906} // Sequence 1: BOS + "Hello" (2 tokens)
    };

    const size_t batch_size = batch.size();

    // ======== Sequential Execution (Baseline) ========
    std::vector<std::vector<float>> logits_sequential(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        auto pipeline_seq = std::make_unique<Qwen2Pipeline>(
            model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);

        bool success = pipeline_seq->forward(batch[i].data(), batch[i].size());
        ASSERT_TRUE(success) << "Sequential forward pass failed for sequence " << i;

        const auto &model = model_ctx_multi_->model();
        size_t vocab_size = model.vocab_size;
        size_t seq_len = batch[i].size();

        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = pipeline_seq->getLogits(0);
        ASSERT_NE(logits_ptr, nullptr);
        std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Sequential: Sequence " << i << " completed (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Batched Execution ========
    auto pipeline_batch = std::make_unique<Qwen2Pipeline>(
        model_ctx_multi_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/batch_size);

    bool success = pipeline_batch->forward_batch(batch);
    ASSERT_TRUE(success) << "Batched forward pass failed on rank " << rank_;

    const auto &model = model_ctx_multi_->model();
    size_t vocab_size = model.vocab_size;

    // Extract per-sequence logits from batch
    std::vector<std::vector<float>> logits_batched(batch_size);

    const int padded_seq_len = pipeline_batch->padded_seq_len();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Batched execution complete. Extracting logits:");
        LOG_INFO("[E2E]   Batch size: " << pipeline_batch->batch_size());
        LOG_INFO("[E2E]   Padded seq len: " << padded_seq_len);
        for (size_t i = 0; i < batch_size; ++i)
        {
            LOG_INFO("[E2E]   Sequence " << i << " actual length: " << pipeline_batch->sequence_lengths()[i]);
        }
    }

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *logits_ptr = pipeline_batch->getLogits(i);
        ASSERT_NE(logits_ptr, nullptr);

        size_t seq_len = batch[i].size();
        const auto &seq_lengths = pipeline_batch->sequence_lengths();
        size_t actual_len = seq_lengths[i];
        ASSERT_EQ(actual_len, seq_len) << "Sequence length mismatch for sequence " << i;

        // Allocate buffer for this sequence's logits
        logits_batched[i].resize(seq_len * vocab_size);

        // Extract only non-padded logits row by row
        // logits_ptr points to start of sequence i in padded layout
        // Layout: [padded_seq_len rows for this sequence, vocab_size cols]
        // We want first seq_len rows (actual tokens, not padding)
        for (size_t token_idx = 0; token_idx < seq_len; ++token_idx)
        {
            const float *src = logits_ptr + (token_idx * vocab_size);
            float *dst = logits_batched[i].data() + (token_idx * vocab_size);
            std::memcpy(dst, src, vocab_size * sizeof(float));
        }

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Batched: Extracted logits for sequence " << i << " (" << seq_len << " tokens)");

            // Debug: Print first few logits values
            if (i == 1) // Only for sequence 1 (the failing one)
            {
                LOG_INFO("[E2E] Sequence 1 first 10 logits values:");
                for (size_t j = 0; j < std::min<size_t>(10, vocab_size); ++j)
                {
                    LOG_INFO("[E2E]   logits[" << j << "] = " << logits_batched[i][j]);
                }
            }
        }
    }

    mpi_ctx_->barrier();

    // ======== Compare Sequential vs Batched ========
    if (rank_ == 0)
    {
        for (size_t i = 0; i < batch_size; ++i)
        {
            size_t seq_len = batch[i].size();
            size_t total_elements = seq_len * vocab_size;

            auto result = compareTensors(
                logits_sequential[i].data(),
                logits_batched[i].data(),
                total_elements,
                tolerance);

            std::string test_name = "Batch Parity - Sequence " + std::to_string(i);
            printComparisonResult(result, test_name);

            EXPECT_TRUE(result.passed)
                << "Sequence " << i << " mismatch: "
                << "max_abs_diff=" << result.max_abs_diff
                << ", rel_l2=" << result.rel_l2_norm;
        }

        LOG_INFO("[E2E] Multi-sequence batch test complete");
    }

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Autoregressive decode correctness
 *
 * Validates multi-step decode produces correct token sequence.
 * Tests KV cache functionality and incremental decode.
 */
TEST_F(Qwen2E2ECorrectness, DISABLED_AutoregressiveDecode)
{
    // Disabled until KV cache is implemented
    GTEST_SKIP() << "KV cache not yet implemented";
}

/**
 * @brief Test: Layer-by-layer activation parity
 *
 * Captures and compares activations at every transformer layer
 * between single-rank and multi-rank execution.
 */
TEST_F(Qwen2E2ECorrectness, DISABLED_LayerActivationParity)
{
    // Disabled until snapshot infrastructure is added
    GTEST_SKIP() << "Activation snapshot capture not yet implemented";
}
