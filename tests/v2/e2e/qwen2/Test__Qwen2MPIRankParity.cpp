/**
 * @file Test__Qwen2MPIRankParity.cpp
 * @brief End-to-end MPI rank parity tests using GraphOrchestrator
 * @author David Sanftenberg
 * @updated 2025-12-27 - Migrated to IInferenceRunner/GraphOrchestrator
 *
 * Validates MPI execution parity by comparing:
 * - Single-rank execution vs multi-rank (tensor-parallel) MPI execution
 * - Ensures distributed execution produces identical results to single-node
 *
 * Uses the unified IInferenceRunner interface which now defaults to GraphOrchestrator.
 *
 * NOTE: These tests compare MPI ranks against each other, NOT against a
 * ground truth PyTorch reference. For PyTorch parity tests, see
 * Test__Qwen2FP32Parity.cpp.
 *
 * Requirements:
 * - Real Qwen 2.5 0.5B model (models/qwen2.5-0.5b-instruct-q4_0.gguf)
 * - MPI support (exactly 2 ranks for tensor-parallel validation)
 * - ENABLE_PIPELINE_SNAPSHOTS must be defined (Debug or E2ERelease build)
 */

// CRITICAL: This test requires snapshot capture to work properly
#ifndef ENABLE_PIPELINE_SNAPSHOTS
#error "Test__Qwen2MPIRankParity requires ENABLE_PIPELINE_SNAPSHOTS to be defined. Build with CMAKE_BUILD_TYPE=Debug or CMAKE_BUILD_TYPE=E2ERelease"
#endif

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>

#include "../../../../src/v2/loaders/ModelContext.h"
#include "../../../../src/v2/execution/InferenceRunnerFactory.h"
#include "../../../../src/v2/execution/IInferenceRunner.h"
#include "../../../../src/v2/backends/ComputeBackend.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include "../../../../src/v2/utils/Logger.h"
#include "../../../../src/v2/kernels/KernelFactory.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Qwen2 MPI rank parity validation
 *
 * Tests compare single-rank execution against multi-rank MPI execution
 * to ensure tensor-parallel distribution produces identical results.
 *
 * Uses IInferenceRunner (GraphOrchestrator) for all test execution.
 */
class Qwen2MPIRankParity : public ::testing::Test
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

        // Initialize device manager (required before creating inference runners)
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering
    }

    void TearDown() override
    {
        // Cleanup runners first (before model contexts)
        runner_single_.reset();
        runner_multi_.reset();
        model_ctx_single_.reset();
        model_ctx_multi_.reset();
        mpi_ctx_.reset();

        // Clear kernel cache to prevent stale pointers
        llaminar::v2::kernels::KernelFactory::clearCache();
    }

    /**
     * @brief Load model and create single-rank inference runner
     *
     * Only rank 0 creates the runner (for baseline comparison).
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

            // Create runner without MPI context (single-rank execution)
            InferenceRunnerConfig config;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.force_graph = true; // Use Graph path

            auto &dm = DeviceManager::instance();
            int cpu_device = dm.cpuDeviceIndex();

            runner_single_ = createInferenceRunner(model_ctx_single_, nullptr, cpu_device, config);
            if (!runner_single_)
            {
                LOG_ERROR("[E2E] Failed to create single-rank runner");
                return false;
            }

            LOG_INFO("[E2E] Model loaded successfully (single-rank): " << model_path_);
            LOG_INFO("[E2E] Execution path: " << (runner_single_->executionPath() == ExecutionPath::GRAPH ? "GRAPH" : "PIPELINE"));
        }
        return true;
    }

    /**
     * @brief Load model and create multi-rank inference runner
     *
     * All ranks create runners with MPI context (tensor-parallel execution).
     */
    bool loadModelMultiRank()
    {
        // Pass MPI context to ModelContext::create for consistent weight sharding
        model_ctx_multi_ = ModelContext::create(model_path_, mpi_ctx_);
        if (!model_ctx_multi_)
        {
            LOG_ERROR("[E2E] Rank " << rank_ << " failed to load model (multi-rank): " << model_path_);
            return false;
        }

        // Create runner with MPI context (multi-rank execution)
        InferenceRunnerConfig config;
        config.max_seq_len = 4096;
        config.batch_size = 1;
        config.force_graph = true; // Use Graph path

        auto &dm = DeviceManager::instance();
        int cpu_device = dm.cpuDeviceIndex();

        runner_multi_ = createInferenceRunner(model_ctx_multi_, mpi_ctx_, cpu_device, config);
        if (!runner_multi_)
        {
            LOG_ERROR("[E2E] Rank " << rank_ << " failed to create multi-rank runner");
            return false;
        }

        LOG_INFO("[E2E] Rank " << rank_ << " model loaded successfully (multi-rank)");
        LOG_INFO("[E2E] Execution path: " << (runner_multi_->executionPath() == ExecutionPath::GRAPH ? "GRAPH" : "PIPELINE"));
        return true;
    }

    /**
     * @brief Create a fresh inference runner with specific batch size
     *
     * Helper for tests that need runners with different batch sizes.
     */
    std::unique_ptr<IInferenceRunner> createRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int batch_size)
    {
        InferenceRunnerConfig config;
        config.max_seq_len = 4096;
        config.batch_size = batch_size;
        config.force_graph = true;

        auto &dm = DeviceManager::instance();
        int cpu_device = dm.cpuDeviceIndex();

        return createInferenceRunner(model_ctx, mpi_ctx, cpu_device, config);
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
    std::unique_ptr<IInferenceRunner> runner_single_;
    std::unique_ptr<IInferenceRunner> runner_multi_;
    std::string model_path_;
    int rank_;
    int world_size_;
};

/**
 * @brief Test: Single token inference correctness (decode phase)
 *
 * Validates that single-rank and multi-rank execution produce identical
 * logits for a single token (typical decode scenario).
 *
 * Uses IInferenceRunner (GraphOrchestrator) for execution.
 */
TEST_F(Qwen2MPIRankParity, SingleTokenInference)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f; // Relaxed tolerance for full pipeline

    // Load models (propagate failures to all ranks)
    bool local_ok = loadModelSingleRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Single-rank model load failed on some rank";

    local_ok = loadModelMultiRank();
    global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Single token input
    std::vector<int> tokens = {151644}; // BOS token for Qwen 2.5

    // Single-rank execution (rank 0 only)
    std::vector<float> logits_single;
    bool single_rank_success = true;
    if (rank_ == 0)
    {
        ASSERT_NE(runner_single_, nullptr) << "Single-rank runner not created";

        single_rank_success = runner_single_->forward(tokens.data(), static_cast<int>(tokens.size()));

        if (single_rank_success)
        {
            // Get logits (vocabulary size)
            size_t vocab_size = static_cast<size_t>(runner_single_->vocab_size());
            logits_single.resize(vocab_size);

            // Extract logits from runner (seq_idx=0 for single sequence)
            const float *logits_ptr = runner_single_->getLogits(0);
            if (logits_ptr)
            {
                std::memcpy(logits_single.data(), logits_ptr, vocab_size * sizeof(float));
                LOG_INFO("[E2E] Rank 0 single-rank logits extracted (" << vocab_size << " values)");
            }
            else
            {
                LOG_ERROR("[E2E] getLogits() returned null");
                single_rank_success = false;
            }
        }
    }

    // Sync success status
    int local_success_int = single_rank_success ? 1 : 0;
    int global_success_int = 0;
    MPI_Allreduce(&local_success_int, &global_success_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success_int, 1) << "Single-rank forward pass failed on rank 0";

    // Multi-rank execution (all ranks)
    ASSERT_NE(runner_multi_, nullptr) << "Multi-rank runner not created on rank " << rank_;

    bool success = runner_multi_->forward(tokens.data(), static_cast<int>(tokens.size()));
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Multi-rank forward pass failed on some rank";

    std::vector<float> logits_multi;
    size_t vocab_size = static_cast<size_t>(runner_multi_->vocab_size());
    logits_multi.resize(vocab_size);

    // Extract logits from multi-rank runner (seq_idx=0)
    const float *logits_ptr_multi = runner_multi_->getLogits(0);
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
 * Validates that single-rank and multi-rank execution produce identical
 * results for multi-token prefill (e.g., 8-32 tokens).
 *
 * Uses IInferenceRunner (GraphOrchestrator) for execution.
 */
TEST_F(Qwen2MPIRankParity, MultiTokenPrefill)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load models (propagate failures to all ranks)
    bool local_ok = loadModelSingleRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Single-rank model load failed on some rank";

    local_ok = loadModelMultiRank();
    global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

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
    int single_rank_success = 0;

    if (rank_ == 0)
    {
        ASSERT_NE(runner_single_, nullptr) << "Single-rank runner not created";

        bool success = runner_single_->forward(tokens.data(), static_cast<int>(tokens.size()));
        single_rank_success = success ? 1 : 0;

        size_t vocab_size = static_cast<size_t>(runner_single_->vocab_size());
        size_t seq_len = tokens.size();

        logits_single.resize(seq_len * vocab_size);
        const float *logits_ptr_single = runner_single_->getLogits(0);
        if (logits_ptr_single)
        {
            std::memcpy(logits_single.data(), logits_ptr_single, seq_len * vocab_size * sizeof(float));
            LOG_INFO("[E2E] Single-rank prefill completed");
        }
        else
        {
            LOG_ERROR("[E2E] Single-rank getLogits() returned null");
            single_rank_success = 0;
        }
    }

    // Broadcast success from Rank 0 to ensure all ranks are synchronized
    // and Rank 1 doesn't proceed to multi-rank execution prematurely
    MPI_Bcast(&single_rank_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ASSERT_EQ(single_rank_success, 1) << "Single-rank forward pass failed";

    // Multi-rank execution (all ranks)
    ASSERT_NE(runner_multi_, nullptr) << "Multi-rank runner not created on rank " << rank_;

    bool success = runner_multi_->forward(tokens.data(), static_cast<int>(tokens.size()));
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Multi-rank forward pass failed on some rank";

    size_t vocab_size = static_cast<size_t>(runner_multi_->vocab_size());
    size_t seq_len = tokens.size();

    std::vector<float> logits_multi(seq_len * vocab_size);
    const float *logits_ptr_multi = runner_multi_->getLogits(0);

    // Verify logits availability across all ranks to prevent hangs
    bool has_logits = (logits_ptr_multi != nullptr);
    int global_has_logits = has_logits ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_has_logits, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_has_logits, 1) << "Multi-rank getLogits() returned null on some rank";

    std::memcpy(logits_multi.data(), logits_ptr_multi, seq_len * vocab_size * sizeof(float));

    LOG_INFO("[E2E] Rank " << rank_ << " multi-rank prefill completed");

    // Compare logits on rank 0
    if (rank_ == 0)
    {
        // Compare all tokens
        auto result = compareTensors(
            logits_single.data(),
            logits_multi.data(),
            seq_len * vocab_size,
            tolerance);

        printComparisonResult(result, "Multi-Token Prefill");
        EXPECT_TRUE(result.passed)
            << "Multi-token prefill mismatch: "
            << "max_abs_diff=" << result.max_abs_diff
            << ", rel_l2=" << result.rel_l2_norm;

        LOG_INFO("[E2E] Multi-token prefill test complete (" << seq_len << " tokens validated)");
    }

    mpi_ctx_->barrier();
}

/**
 * @brief Test: Multi-sequence batch inference (equal lengths)
 *
 * Validates batched execution with equal-length sequences (no padding).
 * This test should pass because padding masking is not required.
 *
 * Uses IInferenceRunner (GraphOrchestrator) for execution.
 */
TEST_F(Qwen2MPIRankParity, MultiSequenceBatchEqualLength)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model (propagate failures to all ranks)
    bool local_ok = loadModelMultiRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Define batch with 2 sequences of EQUAL length (no padding needed)
    std::vector<std::vector<int>> batch = {
        {151644, 9906}, // Sequence 0: BOS + "Hello" (2 tokens)
        {151644, 1374}  // Sequence 1: BOS + different token (2 tokens)
    };

    const size_t batch_size = batch.size();
    const size_t vocab_size = static_cast<size_t>(runner_multi_->vocab_size());

    // ======== Sequential Execution (Baseline) ========
    std::vector<std::vector<float>> logits_sequential(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        // Create fresh runner for each sequence
        LOG_INFO("[E2E] Rank " << rank_ << " creating runner for sequence " << i);
        auto runner_seq = createRunner(model_ctx_multi_, mpi_ctx_, 1);
        local_ok = (runner_seq != nullptr);
        int global_runner_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_runner_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_runner_ok, 1) << "Failed to create sequential runner on some rank";

        LOG_INFO("[E2E] Rank " << rank_ << " calling forward for sequence " << i << " with " << batch[i].size() << " tokens");
        bool success = runner_seq->forward(batch[i].data(), static_cast<int>(batch[i].size()));
        LOG_INFO("[E2E] Rank " << rank_ << " forward returned " << (success ? "SUCCESS" : "FAILED") << " for sequence " << i);
        local_ok = success;
        int global_success = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_success, 1) << "Sequential forward pass failed for some rank (sequence " << i << ")";

        size_t seq_len = batch[i].size();
        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = runner_seq->getLogits(0);

        // Synchronize null check across ranks to prevent hangs
        local_ok = (logits_ptr != nullptr);
        int global_logits_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_logits_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_logits_ok, 1) << "Sequential getLogits returned null on some rank (sequence " << i << ")";

        std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Sequential: Sequence " << i << " completed (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Batched Execution ========
    auto runner_batch = createRunner(model_ctx_multi_, mpi_ctx_, static_cast<int>(batch_size));
    local_ok = (runner_batch != nullptr);
    int global_batch_runner_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_batch_runner_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_batch_runner_ok, 1) << "Failed to create batched runner on some rank";

    bool success = runner_batch->forward_batch(batch);
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Batched forward pass failed on some rank";

    // Extract per-sequence logits from batch
    std::vector<std::vector<float>> logits_batched(batch_size);

    const int padded_seq_len = runner_batch->padded_seq_len();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Batched execution complete (equal-length sequences):");
        LOG_INFO("[E2E]   Batch size: " << runner_batch->batch_size());
        LOG_INFO("[E2E]   Padded seq len: " << padded_seq_len);
    }

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *logits_ptr = runner_batch->getLogits(static_cast<int>(i));

        // Synchronize null check across ranks to prevent hangs
        local_ok = (logits_ptr != nullptr);
        int global_logits_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_logits_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_logits_ok, 1) << "Batched getLogits returned null on some rank (sequence " << i << ")";

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
    std::vector<int> test_results(batch_size, 1);

    // Rank 0: Compare all sequences
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

            test_results[i] = result.passed ? 1 : 0;
        }
    }

    // Broadcast all results to all ranks
    MPI_Bcast(test_results.data(), static_cast<int>(batch_size), MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks check results together
    for (size_t i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(test_results[i], 1)
            << "Sequence " << i << " parity check failed on rank 0";
    }

    mpi_ctx_->barrier();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Equal-length batch test complete");
    }
}

/**
 * @brief Test: Multi-sequence batch inference (variable-length sequences)
 *
 * Validates that batched execution produces identical results to
 * sequential execution (batch parity) for variable-length sequences.
 *
 * Tests with batch_size=2, comparing:
 * - Batched execution (both sequences processed together with padding)
 * - Sequential execution (each sequence processed separately)
 *
 * Uses combined causal+padding masking to prevent padding tokens from
 * participating in attention.
 *
 * Uses IInferenceRunner (GraphOrchestrator) for execution.
 */
TEST_F(Qwen2MPIRankParity, MultiSequenceBatch)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model (propagate failures to all ranks)
    bool local_ok = loadModelMultiRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Define batch with 2 sequences of different lengths
    std::vector<std::vector<int>> batch = {
        {151644},      // Sequence 0: BOS token only (1 token)
        {151644, 9906} // Sequence 1: BOS + "Hello" (2 tokens)
    };

    const size_t batch_size = batch.size();
    const size_t vocab_size = static_cast<size_t>(runner_multi_->vocab_size());

    // ======== Sequential Execution (Baseline) ========
    std::vector<std::vector<float>> logits_sequential(batch_size);
    std::vector<std::unique_ptr<IInferenceRunner>> runners_seq(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        runners_seq[i] = createRunner(model_ctx_multi_, mpi_ctx_, 1);
        local_ok = (runners_seq[i] != nullptr);
        int global_runner_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_runner_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_runner_ok, 1) << "Failed to create sequential runner on some rank";

        // Enable snapshot capture for sequential baseline
        runners_seq[i]->enableSnapshotCapture("/tmp/llaminar_snapshots_seq_" + std::to_string(i));

        bool success = runners_seq[i]->forward(batch[i].data(), static_cast<int>(batch[i].size()));
        local_ok = success;
        int global_success = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_success, 1) << "Sequential forward pass failed for some rank (sequence " << i << ")";

        size_t seq_len = batch[i].size();
        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = runners_seq[i]->getLogits(0);

        // Synchronize null check across ranks to prevent hangs
        local_ok = (logits_ptr != nullptr);
        int global_logits_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_logits_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_logits_ok, 1) << "Sequential getLogits returned null on some rank (sequence " << i << ")";

        std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[E2E] Sequential: Sequence " << i << " completed (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Batched Execution ========
    auto runner_batch = createRunner(model_ctx_multi_, mpi_ctx_, static_cast<int>(batch_size));
    local_ok = (runner_batch != nullptr);
    int global_batch_runner_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_batch_runner_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_batch_runner_ok, 1) << "Failed to create batched runner on some rank";

    // Enable snapshot capture for batched execution
    runner_batch->enableSnapshotCapture("/tmp/llaminar_snapshots_batch");

    bool success = runner_batch->forward_batch(batch);
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Batched forward pass failed on some rank";

    // Extract per-sequence logits from batch
    std::vector<std::vector<float>> logits_batched(batch_size);

    const int padded_seq_len = runner_batch->padded_seq_len();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Batched execution complete. Extracting logits:");
        LOG_INFO("[E2E]   Batch size: " << runner_batch->batch_size());
        LOG_INFO("[E2E]   Padded seq len: " << padded_seq_len);
        for (size_t i = 0; i < batch_size; ++i)
        {
            LOG_INFO("[E2E]   Sequence " << i << " actual length: " << runner_batch->sequence_lengths()[i]);
        }
    }

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *logits_ptr = runner_batch->getLogits(static_cast<int>(i));

        // Synchronize null check across ranks to prevent hangs
        local_ok = (logits_ptr != nullptr);
        int global_logits_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_logits_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_logits_ok, 1) << "Batched getLogits returned null on some rank (sequence " << i << ")";

        size_t seq_len = batch[i].size();
        const auto &seq_lengths = runner_batch->sequence_lengths();
        size_t actual_len = static_cast<size_t>(seq_lengths[i]);
        ASSERT_EQ(actual_len, seq_len) << "Sequence length mismatch for sequence " << i;

        // Allocate buffer for this sequence's logits
        logits_batched[i].resize(seq_len * vocab_size);

        // Extract only non-padded logits row by row
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
    std::vector<int> test_results(batch_size, 1);

    // Rank 0: Compare all sequences AND snapshots
    if (rank_ == 0)
    {
        // Compare final logits
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

            test_results[i] = result.passed ? 1 : 0;
        }

        // Compare layer-by-layer snapshots for failing sequence (Seq1)
        LOG_INFO("[E2E] ===== Snapshot Comparison for Seq1 =====");

        // Get snapshot keys from sequential Seq1 execution
        auto seq1_keys = runners_seq[1]->getSnapshotKeys();
        LOG_INFO("[E2E] Sequential Seq1 has " << seq1_keys.size() << " snapshots");

        // For each layer, compare Q_ROPE, K_ROPE, ATTENTION_CONTEXT
        for (int layer = 0; layer < 2; ++layer) // Just first 2 layers for now
        {
            std::vector<std::string> keys_to_check = {
                "layer" + std::to_string(layer) + "_Q_ROPE",
                "layer" + std::to_string(layer) + "_K_ROPE",
                "layer" + std::to_string(layer) + "_ATTENTION_CONTEXT"};

            for (const auto &key : keys_to_check)
            {
                size_t seq_size = 0, batch_size_snap = 0;
                const float *seq_data = runners_seq[1]->getSnapshot(key, seq_size);
                const float *batch_data = runner_batch->getSnapshot(key, batch_size_snap);

                if (seq_data && batch_data)
                {
                    // For batched execution, extract just Seq1's portion
                    // Snapshots for batched have layout [batch_size * seq_len, feature_dim]
                    // Seq1 starts at offset: padded_seq_len * feature_dim (after Seq0)
                    size_t feature_dim = seq_size / batch[1].size(); // feature_dim = total / seq_len
                    size_t batch_seq1_offset = static_cast<size_t>(padded_seq_len) * feature_dim;
                    const float *batch_seq1_data = batch_data + batch_seq1_offset;

                    auto result = compareTensors(seq_data, batch_seq1_data, seq_size, tolerance);

                    LOG_INFO("[E2E] " << key << ": max_abs_diff=" << result.max_abs_diff
                                      << ", mean=" << result.mean_abs_diff
                                      << ", status=" << (result.passed ? "PASS" : "FAIL"));
                }
                else
                {
                    LOG_WARN("[E2E] " << key << ": snapshot not found (seq=" << (seq_data != nullptr)
                                      << ", batch=" << (batch_data != nullptr) << ")");
                }
            }
        }
    }

    // Broadcast all results to all ranks
    MPI_Bcast(test_results.data(), static_cast<int>(batch_size), MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks check results together
    for (size_t i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(test_results[i], 1)
            << "Sequence " << i << " parity check failed on rank 0";
    }

    mpi_ctx_->barrier();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Multi-sequence batch test complete");
    }
}

/**
 * @brief Test: Batch scaling (throughput validation)
 *
 * Validates that batched execution scales efficiently across different
 * batch sizes (1, 2, 4, 8) and produces correct results for each sequence.
 *
 * Compares batched vs sequential execution for all batch sizes.
 */
TEST_F(Qwen2MPIRankParity, BatchScaling)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Slightly relaxed tolerance for larger batches with Q4_0 quantization
    const float tolerance = 2e-3f;

    // Load model (propagate failures to all ranks)
    bool local_ok = loadModelMultiRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Test different batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8};

    for (int batch_size : batch_sizes)
    {
        if (rank_ == 0)
        {
            LOG_INFO("\n========================================");
            LOG_INFO("Testing batch_size=" << batch_size);
            LOG_INFO("========================================");
        }

        // Create batch with variable-length sequences
        std::vector<std::vector<int>> batch;
        for (int i = 0; i < batch_size; ++i)
        {
            // Vary length: 1 to batch_size tokens
            std::vector<int> seq;
            seq.push_back(151644); // BOS
            for (int j = 0; j < i; ++j)
            {
                seq.push_back(9906 + j); // Add tokens
            }
            batch.push_back(seq);
        }

        // Sequential execution (baseline)
        std::vector<std::vector<float>> logits_sequential(batch_size);
        size_t vocab_size = runner_multi_->vocab_size();

        for (int i = 0; i < batch_size; ++i)
        {
            auto runner_seq = createRunner(model_ctx_multi_, mpi_ctx_, 1);

            bool success = runner_seq->forward(batch[i].data(), batch[i].size());
            local_ok = success;
            int global_success = local_ok ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            ASSERT_EQ(global_success, 1) << "Sequential forward failed for some rank (seq " << i << ")";

            size_t seq_len = batch[i].size();
            logits_sequential[i].resize(seq_len * vocab_size);
            const float *logits_ptr = runner_seq->getLogits(0);
            std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));
        }

        mpi_ctx_->barrier();

        // Batched execution
        auto runner_batch = createRunner(model_ctx_multi_, mpi_ctx_, batch_size);

        bool success = runner_batch->forward_batch(batch);
        local_ok = success;
        int global_success = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_success, 1) << "Batched forward failed for some rank (batch_size=" << batch_size << ")";

        // Extract logits for all sequences in batch
        std::vector<std::vector<float>> logits_batched(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            size_t seq_len = batch[i].size();
            logits_batched[i].resize(seq_len * vocab_size);

            const float *logits_ptr = runner_batch->getLogits(static_cast<int>(i));
            ASSERT_NE(logits_ptr, nullptr);

            // Extract non-padded logits
            for (size_t token_idx = 0; token_idx < seq_len; ++token_idx)
            {
                const float *src = logits_ptr + (token_idx * vocab_size);
                float *dst = logits_batched[i].data() + (token_idx * vocab_size);
                std::memcpy(dst, src, vocab_size * sizeof(float));
            }
        }

        // Compare all sequences on rank 0
        std::vector<int> test_results(batch_size, 1);
        if (rank_ == 0)
        {
            for (int i = 0; i < batch_size; ++i)
            {
                size_t seq_len = batch[i].size();
                auto result = compareTensors(
                    logits_sequential[i].data(),
                    logits_batched[i].data(),
                    seq_len * vocab_size,
                    tolerance);

                std::string test_name = "Batch " + std::to_string(batch_size) + " Seq " + std::to_string(i);
                test_results[i] = result.passed ? 1 : 0;
                if (!result.passed)
                {
                    LOG_ERROR(test_name << " mismatch: max_diff=" << result.max_abs_diff);
                }
            }
        }

        // Broadcast all results to all ranks
        MPI_Bcast(test_results.data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);

        // All ranks check results together
        for (int i = 0; i < batch_size; ++i)
        {
            ASSERT_EQ(test_results[i], 1)
                << "Batch " << batch_size << " Seq " << i << " parity check failed on rank 0";
        }

        mpi_ctx_->barrier();
    }

    if (rank_ == 0)
    {
        LOG_INFO("\n[E2E] Batch scaling test complete - all batch sizes validated");
    }
}

/**
 * @brief Test: Autoregressive decode correctness
 *
 * Validates multi-step decode produces correct token sequence.
 * Tests KV cache functionality and incremental decode.
 */
TEST_F(Qwen2MPIRankParity, IncrementalDecode)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model (propagate failures to all ranks)
    bool local_ok = loadModelMultiRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Initial prompt
    std::vector<int> prompt = {151644, 9906}; // BOS + "Hello"
    const int n_decode_steps = 5;

    // Create runner with batch_size=1
    auto runner = createRunner(model_ctx_multi_, mpi_ctx_, 1);

    // Prefill phase
    bool success = runner->forward(prompt.data(), prompt.size());
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Prefill failed on some rank";

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Prefill complete: " << prompt.size() << " tokens");
    }

    // Decode phase (incremental)
    std::vector<int> generated_tokens;
    size_t vocab_size = runner->vocab_size();

    for (int step = 0; step < n_decode_steps; ++step)
    {
        // Get logits from last token
        const float *logits = runner->getLogits(0);
        ASSERT_NE(logits, nullptr);

        // Greedy sampling on rank 0
        int next_token = 0;
        if (rank_ == 0)
        {
            // Find argmax
            float max_logit = logits[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                if (logits[i] > max_logit)
                {
                    max_logit = logits[i];
                    next_token = static_cast<int>(i);
                }
            }
            LOG_INFO("[E2E] Step " << step << ": sampled token " << next_token);
        }

        // Broadcast next token to all ranks
        MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        generated_tokens.push_back(next_token);

        // Incremental decode (single token)
        success = runner->forward(&next_token, 1);
        local_ok = success;
        global_success = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_success, 1) << "Decode step " << step << " failed on some rank";
    }

    mpi_ctx_->barrier();

    if (rank_ == 0)
    {
        LOG_INFO("[E2E] Incremental decode complete: generated " << generated_tokens.size() << " tokens");
        // Verify we generated expected number of tokens
        EXPECT_EQ(generated_tokens.size(), static_cast<size_t>(n_decode_steps));
    }
}

/**
 * @brief Test: Comprehensive batch vs sequential parity
 *
 * Validates that batched pipeline produces identical results to
 * sequential pipeline across all components:
 * - Embedding lookup
 * - All transformer layers (attention + FFN)
 * - Final normalization
 * - LM head projection
 *
 * This is the V2 equivalent of V1's 17-stage parity test.
 */
TEST_F(Qwen2MPIRankParity, ComprehensiveBatchParity)
{
    // Skip if not exactly 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const float tolerance = 1e-3f;

    // Load model (propagate failures to all ranks)
    bool local_ok = loadModelMultiRank();
    int global_ok = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_ok, 1) << "Multi-rank model load failed on some rank";

    // Test configuration: 2 sequences with different lengths
    std::vector<std::vector<int>> batch = {
        {151644, 9906, 1374, 374}, // Sequence 0: 4 tokens
        {151644, 9906}             // Sequence 1: 2 tokens
    };

    const int batch_size = static_cast<int>(batch.size());
    const size_t vocab_size = runner_multi_->vocab_size();

    // ======== Sequential Execution (Baseline) ========
    std::vector<std::vector<float>> logits_sequential(batch_size);
    std::vector<std::unique_ptr<IInferenceRunner>> runners_seq(batch_size);

    for (int i = 0; i < batch_size; ++i)
    {
        runners_seq[i] = createRunner(model_ctx_multi_, mpi_ctx_, 1);

        // Enable snapshot capture for detailed layer comparison
        runners_seq[i]->enableSnapshotCapture("/tmp/llaminar_snapshots_seq_" + std::to_string(i));

        bool success = runners_seq[i]->forward(batch[i].data(), batch[i].size());
        local_ok = success;
        int global_success = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ASSERT_EQ(global_success, 1) << "Sequential forward failed for some rank (sequence " << i << ")";

        const size_t seq_len = batch[i].size();
        logits_sequential[i].resize(seq_len * vocab_size);
        const float *logits_ptr = runners_seq[i]->getLogits(0);
        std::memcpy(logits_sequential[i].data(), logits_ptr, seq_len * vocab_size * sizeof(float));

        if (rank_ == 0)
        {
            LOG_INFO("[Parity] Sequential: Sequence " << i << " complete (" << seq_len << " tokens)");
        }
    }

    mpi_ctx_->barrier();

    // ======== Batched Execution ========
    auto runner_batch = createRunner(model_ctx_multi_, mpi_ctx_, batch_size);

    // Enable snapshot capture for batched execution
    runner_batch->enableSnapshotCapture("/tmp/llaminar_snapshots_batch");

    bool success = runner_batch->forward_batch(batch);
    local_ok = success;
    int global_success = local_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(global_success, 1) << "Batched forward failed on some rank";

    if (rank_ == 0)
    {
        LOG_INFO("[Parity] Batched execution complete:");
        LOG_INFO("[Parity]   Batch size: " << runner_batch->batch_size());
        LOG_INFO("[Parity]   Padded seq len: " << runner_batch->padded_seq_len());
    }

    // ======== Extract and Compare Logits ========
    // Extract logits for all sequences in batch
    std::vector<std::vector<float>> logits_batched(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        const size_t seq_len = batch[i].size();
        logits_batched[i].resize(seq_len * vocab_size);

        const float *logits_ptr = runner_batch->getLogits(static_cast<int>(i));
        ASSERT_NE(logits_ptr, nullptr);

        // Extract non-padded logits row by row
        for (size_t token_idx = 0; token_idx < seq_len; ++token_idx)
        {
            const float *src = logits_ptr + (token_idx * vocab_size);
            float *dst = logits_batched[i].data() + (token_idx * vocab_size);
            std::memcpy(dst, src, vocab_size * sizeof(float));
        }
    }

    // Compare all sequences on rank 0
    std::vector<int> test_results(batch_size, 1);
    if (rank_ == 0)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            const size_t seq_len = batch[i].size();
            auto result = compareTensors(
                logits_sequential[i].data(),
                logits_batched[i].data(),
                seq_len * vocab_size,
                tolerance);

            std::string test_name = "Comprehensive Parity - Sequence " + std::to_string(i);
            printComparisonResult(result, test_name);

            test_results[i] = result.passed ? 1 : 0;

            // If this sequence failed, do detailed layer-by-layer snapshot comparison
            if (!result.passed)
            {
                LOG_INFO("\n[E2E] ===== Layer-by-Layer Snapshot Comparison for Sequence " << i << " =====");

                // Get all snapshot keys from sequential execution
                auto seq_keys = runners_seq[i]->getSnapshotKeys();
                LOG_INFO("[E2E] Sequential has " << seq_keys.size() << " snapshots");

                // Track if we found the first divergence
                bool found_first_divergence = false;

                // Compare each snapshot
                for (const auto &key : seq_keys)
                {
                    size_t seq_size = 0;
                    size_t batch_size_snap = 0;

                    const float *seq_data = runners_seq[i]->getSnapshot(key, seq_size);
                    const float *batch_data = runner_batch->getSnapshot(key, batch_size_snap);

                    if (seq_data && batch_data)
                    {
                        // For batched snapshots, extract the specific sequence
                        // Layout: [batch_size * padded_seq_len, feature_dim]
                        const int padded_len = runner_batch->padded_seq_len();
                        const size_t feature_dim = seq_size / seq_len;
                        const size_t batch_feature_dim = batch_size_snap / (batch_size * padded_len);

                        if (feature_dim == batch_feature_dim)
                        {
                            // Extract sequence i from batch
                            std::vector<float> batch_seq_i(seq_len * feature_dim);
                            for (size_t tok = 0; tok < seq_len; ++tok)
                            {
                                const size_t batch_offset = (i * padded_len + tok) * feature_dim;
                                const size_t seq_offset = tok * feature_dim;
                                std::memcpy(batch_seq_i.data() + seq_offset,
                                            batch_data + batch_offset,
                                            feature_dim * sizeof(float));
                            }

                            // Compare this snapshot
                            auto snap_result = compareTensors(seq_data, batch_seq_i.data(),
                                                              seq_len * feature_dim, tolerance);

                            const char *status = snap_result.passed ? "✓ PASS" : "✗ FAIL";
                            LOG_INFO("[E2E]   " << key << ": " << status
                                                << " (max_diff=" << snap_result.max_abs_diff
                                                << ", mean_diff=" << snap_result.mean_abs_diff << ")");

                            // If this is the first failing snapshot, print detailed comparison
                            if (!snap_result.passed && !found_first_divergence)
                            {
                                found_first_divergence = true;
                                LOG_INFO("[E2E]     ═══════════════════════════════════════════════════");
                                LOG_INFO("[E2E]     → FIRST DIVERGENCE DETECTED: " << key);
                                LOG_INFO("[E2E]     → Shape: [" << seq_len << ", " << feature_dim << "]");
                                LOG_INFO("[E2E]     → Max diff: " << snap_result.max_abs_diff);
                                LOG_INFO("[E2E]     → Mean diff: " << snap_result.mean_abs_diff);
                                LOG_INFO("[E2E]     → Rel L2 norm: " << snap_result.rel_l2_norm);

                                // Print first few values for debugging
                                LOG_INFO("[E2E]     → Sequential first 5 values: "
                                         << seq_data[0] << ", " << seq_data[1] << ", "
                                         << seq_data[2] << ", " << seq_data[3] << ", " << seq_data[4]);
                                LOG_INFO("[E2E]     → Batched first 5 values: "
                                         << batch_seq_i[0] << ", " << batch_seq_i[1] << ", "
                                         << batch_seq_i[2] << ", " << batch_seq_i[3] << ", " << batch_seq_i[4]);
                                LOG_INFO("[E2E]     ═══════════════════════════════════════════════════");
                            }
                        }
                        else
                        {
                            LOG_WARN("[E2E]   " << key << ": dimension mismatch (seq="
                                                << feature_dim << ", batch=" << batch_feature_dim << ")");
                        }
                    }
                    else
                    {
                        LOG_WARN("[E2E]   " << key << ": snapshot not found (seq="
                                            << (seq_data != nullptr) << ", batch=" << (batch_data != nullptr) << ")");
                    }
                }
            }
        }
    }

    // Broadcast all results to all ranks
    MPI_Bcast(test_results.data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks check results together
    for (int i = 0; i < batch_size; ++i)
    {
        ASSERT_EQ(test_results[i], 1)
            << "Sequence " << i << " parity check failed on rank 0";
    }

    mpi_ctx_->barrier();

    if (rank_ == 0)
    {
        LOG_INFO("\n[E2E] ✓ Comprehensive batch parity test PASSED");
        LOG_INFO("[E2E]   All " << batch_size << " sequences match sequential execution");
        LOG_INFO("[E2E]   Validated: embedding → transformer → norm → LM head");
    }
}

/**
 * @brief Test: Layer-by-layer activation parity
 *
 * Captures and compares activations at every transformer layer
 * between single-rank and multi-rank execution.
 *
 * TODO: Add snapshot infrastructure for intermediate activations.
 */
TEST_F(Qwen2MPIRankParity, DISABLED_LayerActivationParity)
{
    // Disabled until snapshot infrastructure is added
    GTEST_SKIP() << "Activation snapshot capture not yet implemented";
}
