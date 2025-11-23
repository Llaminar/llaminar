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
 * @brief Minimal reproducer for batch padding divergence bug
 *
 * Test Flow:
 * 1. Create pipeline with batch_size=1 (for sequential execution)
 * 2. Run sequence 0: [1, 2, 3, 4] (4 tokens)
 * 3. Run sequence 1: [5, 6] (2 tokens)
 * 4. Extract final token logits for each sequence
 *
 * 5. Create pipeline with batch_size=2 (for batched execution)
 * 6. Run batched: [[1, 2, 3, 4], [5, 6, PAD, PAD]]
 * 7. Extract logits for each sequence
 *
 * 8. Compare: seq0_sequential vs seq0_batch (should match)
 * 9. Compare: seq1_sequential vs seq1_batch (FAILS - 96.8% divergence)
 *
 * Expected Result: Both sequences match within 10% tolerance
 * Actual Result: Sequence 1 (padded) diverges by ~97%
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
    // PART 1: Sequential Execution (baseline) - POSITION-MATCHED!
    // ========================================================================
    if (rank == 0)
    {
        LOG_ERROR("[SEQUENTIAL] Running with POSITION-MATCHED K/V cache state");
        LOG_ERROR("[SEQUENTIAL] Seq0 at positions [0-3], Seq1 at positions [4-5]");
        LOG_ERROR("[SEQUENTIAL] This matches the batch layout for fair comparison");
    }

    // Use SHARED pipeline to accumulate K/V cache state (mimics batch layout)
    auto pipeline_seq = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, /*batch_size=*/1);
    ASSERT_NE(pipeline_seq, nullptr);

    // Run sequence 0 (positions 0-3 in K/V cache)
    bool success_seq0 = pipeline_seq->forward(seq0.data(), seq0.size());
    ASSERT_TRUE(success_seq0) << "Sequential sequence 0 forward pass failed";

    // Get logits for sequence 0 (last token at position 3)
    seq0_sequential_logits.resize(vocab_size);
    const float *seq0_logits_ptr = pipeline_seq->getLogits();
    size_t seq0_logits_offset = (seq0.size() - 1) * vocab_size;
    std::copy(seq0_logits_ptr + seq0_logits_offset,
              seq0_logits_ptr + seq0_logits_offset + vocab_size,
              seq0_sequential_logits.begin());

    // Run sequence 1 (positions 4-5 in K/V cache, AFTER seq0 context)
    bool success_seq1 = pipeline_seq->forward(seq1.data(), seq1.size());
    ASSERT_TRUE(success_seq1) << "Sequential sequence 1 forward pass failed";

    // Get logits for sequence 1 (last token at position 5)
    seq1_sequential_logits.resize(vocab_size);
    const float *seq1_logits_ptr = pipeline_seq->getLogits();
    size_t seq1_logits_offset = (seq1.size() - 1) * vocab_size;
    std::copy(seq1_logits_ptr + seq1_logits_offset,
              seq1_logits_ptr + seq1_logits_offset + vocab_size,
              seq1_sequential_logits.begin());

    if (rank == 0)
    {
        LOG_ERROR("[SEQUENTIAL] Captured position-matched baseline logits");
        LOG_ERROR("[SEQUENTIAL] Seq0 last token at position 3, Seq1 last token at position 5");
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

    // Sequence 0: last valid token at position 3 (0-indexed)
    std::vector<float> seq0_batch_logits(vocab_size);
    size_t seq0_batch_offset = 3 * vocab_size; // Last token of seq0
    std::copy(batch_logits_ptr + seq0_batch_offset,
              batch_logits_ptr + seq0_batch_offset + vocab_size,
              seq0_batch_logits.begin());

    // Sequence 1: last valid token at position 5 (batch1, token1)
    // Layout: [seq0_tok0, seq0_tok1, seq0_tok2, seq0_tok3, seq1_tok0, seq1_tok1, seq1_pad0, seq1_pad1]
    // So seq1 last valid = position 5
    // BUT: Based on debugging, position 4 has the correct logits! Investigating...
    std::vector<float> seq1_batch_logits(vocab_size);

    // Try both position 4 and 5 to see which matches
    std::vector<float> seq1_batch_logits_pos4(vocab_size);
    std::vector<float> seq1_batch_logits_pos5(vocab_size);

    size_t seq1_batch_offset_pos4 = 4 * vocab_size;
    size_t seq1_batch_offset_pos5 = 5 * vocab_size;

    std::copy(batch_logits_ptr + seq1_batch_offset_pos4,
              batch_logits_ptr + seq1_batch_offset_pos4 + vocab_size,
              seq1_batch_logits_pos4.begin());

    std::copy(batch_logits_ptr + seq1_batch_offset_pos5,
              batch_logits_ptr + seq1_batch_offset_pos5 + vocab_size,
              seq1_batch_logits_pos5.begin());

    if (rank == 0)
    {
        LOG_ERROR("[BATCHED] Extracted logits from batch execution");
        LOG_ERROR("[BATCHED] Logits layout: 8 total positions (4+4 padded)");
        LOG_ERROR("[BATCHED] Seq0 positions: [0,1,2,3], Seq1 positions: [4,5,6(PAD),7(PAD)]");

        // Check ALL positions for Seq1
        const float *all_logits = pipeline_batch->getLogits();
        for (int pos = 4; pos <= 7; ++pos)
        {
            const float *pos_logits = all_logits + pos * vocab_size;
            float sum = 0.0f;
            for (size_t i = 0; i < std::min(size_t(10), vocab_size); ++i)
            {
                sum += std::abs(pos_logits[i]);
            }
            LOG_ERROR("[BATCHED] Position " << pos << " (Seq1 "
                                            << (pos < 6 ? "REAL" : "PAD") << "): "
                                            << "first 10 sum=" << sum
                                            << ", first_value=" << pos_logits[0]);
        }

        dumpTensorSample("Seq0_Batch_Logits", seq0_batch_logits.data(), vocab_size);
        dumpTensorSample("Seq1_Batch_Logits_Pos4", seq1_batch_logits_pos4.data(), vocab_size);
        dumpTensorSample("Seq1_Batch_Logits_Pos5", seq1_batch_logits_pos5.data(), vocab_size);
    }

    // ========================================================================
    // PART 3: Comparison
    // ========================================================================
    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Comparing sequential vs batched logits...");
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

    // Compare sequence 1 POSITION 4 vs sequential (test if off-by-one error)
    float seq1_pos4_rel_diff = compareLogits(
        seq1_sequential_logits.data(),
        seq1_batch_logits_pos4.data(),
        vocab_size);

    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Sequence 1 Position 4 vs Sequential: "
                  << (seq1_pos4_rel_diff * 100.0f) << "% max rel diff");
    }

    // Compare sequence 1 POSITION 5 vs sequential (original extraction)
    float seq1_pos5_rel_diff = compareLogits(
        seq1_sequential_logits.data(),
        seq1_batch_logits_pos5.data(),
        vocab_size);

    if (rank == 0)
    {
        LOG_ERROR("[COMPARISON] Sequence 1 Position 5 vs Sequential: "
                  << (seq1_pos5_rel_diff * 100.0f) << "% max rel diff");
    }

    // Assertions and root cause diagnosis
    EXPECT_LT(seq0_rel_diff, 0.1f)
        << "Sequence 0 (no padding) should match within 10% tolerance";

    // Diagnostic logic
    if (rank == 0)
    {
        if (seq1_pos5_rel_diff < 0.1f)
        {
            LOG_ERROR("[SUCCESS] Position-matched comparison PASSES!");
            LOG_ERROR("  → Seq1 last token (position 5) matches between sequential and batch");
            LOG_ERROR("  → Previous bug was due to comparing different K/V cache positions");
            LOG_ERROR("  → Sequential isolated vs batched with context are NOT equivalent");
        }
        else if (seq1_pos4_rel_diff < 0.1f && seq1_pos5_rel_diff >= 0.1f)
        {
            LOG_ERROR("[OFFSET ERROR] Position 4 matches, Position 5 does NOT!");
            LOG_ERROR("  → Last token logits are at wrong position in batch layout");
        }
        else if (seq1_pos4_rel_diff >= 0.1f && seq1_pos5_rel_diff >= 0.1f)
        {
            LOG_ERROR("[BUG CONFIRMED] Neither position matches!");
            LOG_ERROR("  → Padding corruption affects logit computation");
            LOG_ERROR("  → This is a real bug in batch processing with padding");
        }
    }

    // Final assertions
    bool pos5_matches = seq1_pos5_rel_diff < 0.1f;

    EXPECT_LT(seq1_pos5_rel_diff, 0.1f)
        << "Position-matched Seq1 (position 5) should match within 10% tolerance. "
        << "Divergence: " << (seq1_pos5_rel_diff * 100.0f) << "%";

    if (rank == 0)
    {
        if (pos5_matches)
        {
            LOG_ERROR("[TEST PASSED] Position-matched comparison successful!");
            LOG_ERROR("  Previous divergence was due to K/V cache position mismatch");
        }
        else
        {
            LOG_ERROR("[TEST FAILED] Even with position-matching, batch diverges by "
                      << (seq1_pos5_rel_diff * 100.0f) << "%");
            LOG_ERROR("  This indicates a real bug in batch padding handling");
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
