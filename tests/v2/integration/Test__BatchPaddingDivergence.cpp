/**
 * @file Test__BatchPaddingDivergence.cpp
 * @brief Minimal reproducer for batch padding divergence bug
 * @author David Sanftenberg
 * @date 2025-11-22
 *
 * Minimal unit test that reproduces the batch padding divergence bug found in
 * integration tests. This test isolates just the problematic scenario:
 * - Sequential execution of 2 sequences (4 tokens, then 2 tokens)
 * - vs Batched execution (batch_size=2, padded to 4 tokens each)
 *
 * Expected: Logits should match within 10% tolerance
 * Actual: Sequence 1 (shorter, padded) diverges by 96.8%
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "v2/loaders/ModelContext.h"
#include "v2/pipelines/qwen/Qwen2Pipeline.h"
#include "v2/pipelines/PipelineFactory.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

/**
 * @brief Minimal reproducer for batch padding divergence
 *
 * This test fixture uses the smallest possible setup to reproduce the bug:
 * - Uses actual Qwen2Pipeline (not mocked - need real weights to see divergence)
 * - Minimal model (qwen2.5-0.5b-instruct-q4_0.gguf)
 * - Just 2 sequences: 4 tokens vs 2 tokens (padded to 4)
 */
class BatchPaddingDivergenceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get MPI rank
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        // Load model on all ranks (MPI requires all ranks to participate)
        const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
        model_ctx_ = ModelContext::create(model_path, mpi_ctx_);
        ASSERT_NE(model_ctx_, nullptr) << "Failed to load model: " << model_path;
    }

    void TearDown() override
    {
        // Cleanup handled by shared_ptr
    }

    /**
     * @brief Dump first few values of a tensor for debugging
     */
    void dumpTensorSample(const char *name, const float *data, size_t count, size_t sample_size = 10)
    {
        if (mpi_ctx_->rank() != 0)
            return;

        std::stringstream ss;
        ss << "[" << name << "] First " << std::min(sample_size, count) << " values: [";
        for (size_t i = 0; i < std::min(sample_size, count); ++i)
        {
            ss << data[i];
            if (i < std::min(sample_size, count) - 1)
                ss << ", ";
        }
        ss << "]";
        LOG_ERROR(ss.str());
    }

    /**
     * @brief Compare two logits tensors element-wise
     * @return Max relative difference as percentage
     */
    float compareLogits(const float *expected, const float *actual, size_t count)
    {
        float max_rel_diff = 0.0f;
        size_t mismatch_count = 0;
        size_t first_mismatch = SIZE_MAX;

        for (size_t i = 0; i < count; ++i)
        {
            float exp = expected[i];
            float act = actual[i];

            // Skip near-zero values (both < 0.01)
            if (std::abs(exp) < 0.01f && std::abs(act) < 0.01f)
            {
                continue;
            }

            float abs_diff = std::abs(act - exp);
            float rel_diff = abs_diff / (std::abs(exp) + 1e-8f);

            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if (rel_diff > 0.1f)
            { // 10% threshold
                if (first_mismatch == SIZE_MAX)
                {
                    first_mismatch = i;
                }
                mismatch_count++;
            }
        }

        if (mpi_ctx_->rank() == 0)
        {
            LOG_ERROR("[compareLogits] Max relative diff: " << (max_rel_diff * 100.0f)
                                                            << "%, Mismatches: " << mismatch_count << "/" << count
                                                            << " (" << (100.0f * mismatch_count / count) << "%)");
            if (first_mismatch != SIZE_MAX)
            {
                LOG_ERROR("[compareLogits] First mismatch at index " << first_mismatch
                                                                     << ": expected=" << expected[first_mismatch]
                                                                     << ", actual=" << actual[first_mismatch]);
            }
        }

        return max_rel_diff;
    }

    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

/**
 * @brief Validate batch execution with padding produces correct results
 *
 * Test Flow:
 * 1. Run each sequence INDEPENDENTLY in batch_size=1 mode (isolated baseline)
 * 2. Run both sequences together in batch_size=2 mode (with padding)
 * 3. Compare: Each sequence's logits should match its isolated baseline
 *
 * This validates that padding does not corrupt the computation for
 * sequences in a batch. Each sequence should produce identical results
 * whether run alone or in a batch with padding.
 *
 * Expected Result: Both sequences match their isolated baselines within 10%
 */
TEST_F(BatchPaddingDivergenceTest, SequentialVsBatchedWithPadding)
{
    ASSERT_NE(model_ctx_, nullptr);

    const int rank = mpi_ctx_->rank();
    const size_t vocab_size = model_ctx_->model().vocab_size;

    // Test sequences
    std::vector<int> seq0 = {1, 2, 3, 4}; // 4 tokens
    std::vector<int> seq1 = {5, 6};       // 2 tokens (will be padded to 4)

    // Pre-allocate logit vectors
    std::vector<float> seq0_sequential_logits;
    std::vector<float> seq1_sequential_logits;

    // ========================================================================
    // PART 1: Sequential Execution (isolated baseline)
    // ========================================================================
    if (rank == 0)
    {
        LOG_ERROR("[SEQUENTIAL] Running sequences INDEPENDENTLY (isolated KV cache)");
        LOG_ERROR("[SEQUENTIAL] Each sequence starts from empty KV cache");
    }

    // Sequence 0: Run in isolation
    {
        auto pipeline_seq0 = std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);
        ASSERT_NE(pipeline_seq0, nullptr);

        bool success = pipeline_seq0->forward(seq0.data(), seq0.size());
        ASSERT_TRUE(success) << "Sequential sequence 0 forward pass failed";

        // Get logits for last token
        seq0_sequential_logits.resize(vocab_size);
        const float *logits_ptr = pipeline_seq0->getLogits();
        size_t logits_offset = (seq0.size() - 1) * vocab_size;
        std::copy(logits_ptr + logits_offset,
                  logits_ptr + logits_offset + vocab_size,
                  seq0_sequential_logits.begin());
    }

    // Sequence 1: Run in isolation (separate pipeline, fresh KV cache)
    {
        auto pipeline_seq1 = std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);
        ASSERT_NE(pipeline_seq1, nullptr);

        bool success = pipeline_seq1->forward(seq1.data(), seq1.size());
        ASSERT_TRUE(success) << "Sequential sequence 1 forward pass failed";

        // Get logits for last token
        seq1_sequential_logits.resize(vocab_size);
        const float *logits_ptr = pipeline_seq1->getLogits();
        size_t logits_offset = (seq1.size() - 1) * vocab_size;
        std::copy(logits_ptr + logits_offset,
                  logits_ptr + logits_offset + vocab_size,
                  seq1_sequential_logits.begin());
    }

    if (rank == 0)
    {
        LOG_ERROR("[SEQUENTIAL] Captured isolated baseline logits");
        dumpTensorSample("Seq0_Sequential_Logits", seq0_sequential_logits.data(), vocab_size);
        dumpTensorSample("Seq1_Sequential_Logits", seq1_sequential_logits.data(), vocab_size);
    }

    // ========================================================================
    // PART 2: Batched Execution (test case)
    // ========================================================================
    if (rank == 0)
    {
        LOG_ERROR("[BATCHED] Running batch_size=2 with padding...");
    }

    auto pipeline_batch = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/2);
    ASSERT_NE(pipeline_batch, nullptr);

    // Prepare batched input (pad seq1 to match seq0 length)
    std::vector<std::vector<int>> batch = {
        {1, 2, 3, 4}, // sequence 0: 4 tokens
        {5, 6}        // sequence 1: 2 tokens (will be internally padded)
    };

    if (rank == 0)
    {
        LOG_ERROR("[BATCHED] Input layout: [[1,2,3,4], [5,6]] (batch automatically padded)");
    }

    // Run batched forward
    bool success_batch = pipeline_batch->forward_batch(batch);
    ASSERT_TRUE(success_batch) << "Batched forward pass failed";

    if (rank == 0)
    {
        LOG_ERROR("[BATCHED] Forward pass completed successfully");
    }

    // Extract logits for each sequence (last valid token only)
    const float *batch_logits_ptr = pipeline_batch->getLogits();

    // Sequence 0: last valid token at position 3 (0-indexed within seq0's slot)
    std::vector<float> seq0_batch_logits(vocab_size);
    size_t seq0_batch_offset = 3 * vocab_size; // Last token of seq0
    std::copy(batch_logits_ptr + seq0_batch_offset,
              batch_logits_ptr + seq0_batch_offset + vocab_size,
              seq0_batch_logits.begin());

    // Sequence 1: last valid token at position 5 (4 + 1, accounting for padded layout)
    // Layout: [seq0_tok0, seq0_tok1, seq0_tok2, seq0_tok3, seq1_tok0, seq1_tok1, seq1_pad0, seq1_pad1]
    // Position 5 = seq1's last real token
    std::vector<float> seq1_batch_logits(vocab_size);
    size_t seq1_batch_offset = 5 * vocab_size;
    std::copy(batch_logits_ptr + seq1_batch_offset,
              batch_logits_ptr + seq1_batch_offset + vocab_size,
              seq1_batch_logits.begin());

    if (rank == 0)
    {
        LOG_ERROR("[BATCHED] Extracted logits from batch execution");
        dumpTensorSample("Seq0_Batch_Logits", seq0_batch_logits.data(), vocab_size);
        dumpTensorSample("Seq1_Batch_Logits", seq1_batch_logits.data(), vocab_size);
    }

    // ========================================================================
    // PART 3: Comparison
    // ========================================================================
    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Comparing isolated vs batched logits...");
    }

    // Compare sequence 0 (should match - no padding)
    float seq0_rel_diff = compareLogits(
        seq0_sequential_logits.data(),
        seq0_batch_logits.data(),
        vocab_size);

    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Sequence 0 (no padding): "
                  << (seq0_rel_diff * 100.0f) << "% max rel diff");
    }

    // Compare sequence 1 (padded - this is the critical test)
    float seq1_rel_diff = compareLogits(
        seq1_sequential_logits.data(),
        seq1_batch_logits.data(),
        vocab_size);

    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Sequence 1 (padded): "
                  << (seq1_rel_diff * 100.0f) << "% max rel diff");
    }

    // Final assertions
    EXPECT_LT(seq0_rel_diff, 0.1f)
        << "Sequence 0 (no padding) should match within 10% tolerance. "
        << "Divergence: " << (seq0_rel_diff * 100.0f) << "%";

    EXPECT_LT(seq1_rel_diff, 0.1f)
        << "Sequence 1 (with padding) should match within 10% tolerance. "
        << "Divergence: " << (seq1_rel_diff * 100.0f) << "%";

    if (rank == 0)
    {
        if (seq0_rel_diff < 0.1f && seq1_rel_diff < 0.1f)
        {
            LOG_ERROR("[TEST PASSED] Both sequences match their isolated baselines!");
            LOG_ERROR("  Padding does not corrupt batch execution");
        }
        else
        {
            LOG_ERROR("[TEST FAILED] Batch execution diverges from isolated baseline");
            LOG_ERROR("  Seq0 divergence: " << (seq0_rel_diff * 100.0f) << "%");
            LOG_ERROR("  Seq1 divergence: " << (seq1_rel_diff * 100.0f) << "%");
            LOG_ERROR("  This indicates a bug in batch padding handling");
        }
    }
}

/**
 * @brief Even simpler test - just validate batch execution runs without crashing
 *
 * This test exists to verify the minimal setup works before running the full comparison.
 */
TEST_F(BatchPaddingDivergenceTest, BatchExecutionBasicSanity)
{
    ASSERT_NE(model_ctx_, nullptr);

    const int rank = mpi_ctx_->rank();

    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/2);
    ASSERT_NE(pipeline, nullptr);

    // Simple batch: 2 sequences, both 3 tokens
    std::vector<std::vector<int>> batch = {
        {1, 2, 3},
        {4, 5, 6}};

    bool success = pipeline->forward_batch(batch);
    ASSERT_TRUE(success) << "Basic batch forward pass failed";

    // Validate logits are sane (no NaN/Inf)
    const float *logits = pipeline->getLogits();
    const size_t vocab_size = model_ctx_->model().vocab_size;
    size_t total_logits = 6 * vocab_size; // 6 tokens total

    size_t nan_count = 0;
    size_t inf_count = 0;
    for (size_t i = 0; i < total_logits; ++i)
    {
        if (std::isnan(logits[i]))
            nan_count++;
        if (std::isinf(logits[i]))
            inf_count++;
    }

    EXPECT_EQ(nan_count, 0) << "Found NaN values in logits";
    EXPECT_EQ(inf_count, 0) << "Found Inf values in logits";

    if (rank == 0)
    {
        LOG_ERROR("[BasicSanity] PASSED - batch execution produces valid logits");
    }
}

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
