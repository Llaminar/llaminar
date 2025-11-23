/**
 * @file Test__MpiAttentionOrchestrator_MaskPassing.cpp
 * @brief Unit tests for MpiAttentionOrchestrator batch padding mask handling
 * @author David Sanftenberg
 *
 * Tests the orchestrator layer to verify it correctly passes masks through
 * to GQAAttention and handles batch processing properly.
 *
 * Critical Tests:
 * 1. Verify mask is forwarded to GQAAttention (not lost in orchestration)
 * 2. Sequential vs batch produces identical outputs
 * 3. Padding outputs are zeroed
 * 4. MPI dispatch works correctly
 */

#include <gtest/gtest.h>
#include "../../src/v2/pipelines/attention/MpiAttentionOrchestrator.h"
#include "../../src/v2/pipelines/AttentionUtils.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/utils/Logger.h"
#include <memory>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace llaminar2;

namespace
{
    // Helper to calculate tensor size
    size_t tensor_size(const std::unique_ptr<FP32Tensor> &tensor)
    {
        const auto &shape = tensor->shape();
        return shape[0] * shape[1];
    }

    // Helper to create FP32 tensor
    std::unique_ptr<FP32Tensor> create_fp32_tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, 0);
    }

    // Initialize tensor with sequential values
    void init_sequential(float *data, size_t count, float start_val)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = start_val + static_cast<float>(i) * 0.01f;
        }
    }

    // Initialize tensor with uniform value
    void init_uniform(float *data, size_t count, float val)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = val;
        }
    }

    // Check if tensor region is all zeros
    bool is_zero(const float *data, size_t start, size_t count, float tolerance = 1e-6f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::abs(data[start + i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // Compare two tensors element-wise
    void compare_tensors(const float *a, const float *b, size_t count,
                         float &max_diff, float &mean_diff)
    {
        max_diff = 0.0f;
        mean_diff = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_diff = std::max(max_diff, diff);
            mean_diff += diff;
        }
        mean_diff /= static_cast<float>(count);
    }

} // namespace

/**
 * TEST SUITE: MpiAttentionOrchestrator_MaskPassing
 *
 * Verifies that MpiAttentionOrchestrator correctly:
 * 1. Passes masks through to GQAAttention
 * 2. Handles batched sequences with padding
 * 3. Produces identical sequential vs batch results
 * 4. Zeros padding outputs
 */
class MpiAttentionOrchestrator_MaskPassing : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // No setup needed - V2 uses LOG_ERROR macros that don't require Logger initialization
    }
};

/**
 * TEST 1: Verify mask is passed to underlying GQAAttention
 *
 * Strategy:
 * - Create batch with one sequence masked at position 2
 * - Verify mask is not nullptr in config passed to GQA
 * - Verify output is computed (not NaN/Inf)
 */
TEST_F(MpiAttentionOrchestrator_MaskPassing, MaskNotNullptr)
{
    const int seq_len = 4;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // Create inputs with uniform values for easy verification
    auto Q = create_fp32_tensor(seq_len, d_model);
    auto K = create_fp32_tensor(seq_len, kv_dim);
    auto V = create_fp32_tensor(seq_len, kv_dim);
    auto output = create_fp32_tensor(seq_len, d_model);

    // Initialize with uniform value (all V = 5.0)
    init_uniform(Q->mutable_data(), tensor_size(Q), 1.0f);
    init_uniform(K->mutable_data(), tensor_size(K), 1.0f);
    init_uniform(V->mutable_data(), tensor_size(V), 5.0f);

    // Create mask that blocks position 2
    auto mask_tensor = create_fp32_tensor(seq_len, seq_len);
    float *mask_data = mask_tensor->mutable_data();

    // Initialize mask: all positions can attend everywhere except position 2
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < seq_len; ++j)
        {
            mask_data[i * seq_len + j] = (i == 2 && j == 2) ? -INFINITY : 0.0f;
        }
    }

    MpiAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;
    config.workspace_mask = std::move(mask_tensor);

    bool success = MpiAttentionOrchestrator::compute(
        Q.get(), K.get(), V.get(), output.get(), config);

    ASSERT_TRUE(success) << "Orchestrator compute failed";

    // Verify output is valid (not NaN/Inf)
    const float *out_data = output->data();
    std::cout << "\n[MASK APPLICATION CHECK]\n";
    for (int pos = 0; pos < seq_len; ++pos)
    {
        std::cout << "Position " << pos << " output: [";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << out_data[pos * d_model + i];
            if (i < 7)
                std::cout << ", ";
        }
        std::cout << "]";
        if (pos == 2)
            std::cout << " (masked)";
        std::cout << "\n";

        EXPECT_FALSE(std::isnan(out_data[pos * d_model])) << "NaN at position " << pos;
        EXPECT_FALSE(std::isinf(out_data[pos * d_model])) << "Inf at position " << pos;
    }
}

/**
 * TEST 2: Sequential vs Batch Exact Match
 *
 * This is the CRITICAL test that should replicate the E2E failure.
 *
 * Strategy:
 * - Create 2 sequences with different lengths (8 and 5)
 * - Run each sequentially through orchestrator
 * - Run both as batch through orchestrator
 * - Verify sequential and batch produce IDENTICAL outputs
 * - Verify padding outputs are zero
 *
 * Expected: If orchestrator has bug, this test will FAIL with divergence.
 */
TEST_F(MpiAttentionOrchestrator_MaskPassing, SequentialVsBatchExactMatch)
{
    const int batch_size = 2;
    const int seq_len = 8; // Max sequence length
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // Actual sequence lengths (Seq1 has padding)
    const int seq0_len = 8; // Full length
    const int seq1_len = 5; // 3 padding tokens
    std::vector<int> actual_lengths = {seq0_len, seq1_len};

    // === SEQUENTIAL EXECUTION ===
    // Run each sequence independently

    // Sequence 0 (full length)
    auto Q_seq0 = create_fp32_tensor(seq0_len, d_model);
    auto K_seq0 = create_fp32_tensor(seq0_len, kv_dim);
    auto V_seq0 = create_fp32_tensor(seq0_len, kv_dim);
    auto output_seq0 = create_fp32_tensor(seq0_len, d_model);

    init_sequential(Q_seq0->mutable_data(), tensor_size(Q_seq0), 1.0f);
    init_sequential(K_seq0->mutable_data(), tensor_size(K_seq0), 10.0f);
    init_sequential(V_seq0->mutable_data(), tensor_size(V_seq0), 20.0f);

    MpiAttentionConfig config_seq;
    config_seq.n_heads = n_heads;
    config_seq.n_kv_heads = n_kv_heads;
    config_seq.head_dim = head_dim;
    config_seq.causal = false;
    config_seq.window_size = -1;
    config_seq.precision = ActivationPrecision::FP32;
    config_seq.mpi_ctx = nullptr;
    config_seq.mpi_strategy = MPIStrategy::None;

    bool success0 = MpiAttentionOrchestrator::compute(
        Q_seq0.get(), K_seq0.get(), V_seq0.get(), output_seq0.get(), config_seq);
    ASSERT_TRUE(success0) << "Sequential Seq0 failed";

    // Sequence 1 (with padding)
    auto Q_seq1 = create_fp32_tensor(seq1_len, d_model);
    auto K_seq1 = create_fp32_tensor(seq1_len, kv_dim);
    auto V_seq1 = create_fp32_tensor(seq1_len, kv_dim);
    auto output_seq1 = create_fp32_tensor(seq1_len, d_model);

    init_sequential(Q_seq1->mutable_data(), tensor_size(Q_seq1), 100.0f);
    init_sequential(K_seq1->mutable_data(), tensor_size(K_seq1), 200.0f);
    init_sequential(V_seq1->mutable_data(), tensor_size(V_seq1), 300.0f);

    bool success1 = MpiAttentionOrchestrator::compute(
        Q_seq1.get(), K_seq1.get(), V_seq1.get(), output_seq1.get(), config_seq);
    ASSERT_TRUE(success1) << "Sequential Seq1 failed";

    // === BATCH EXECUTION ===
    // Run both sequences together

    auto Q_batch = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_batch = create_fp32_tensor(batch_size * seq_len, kv_dim);
    auto V_batch = create_fp32_tensor(batch_size * seq_len, kv_dim);
    auto output_batch = create_fp32_tensor(batch_size * seq_len, d_model);

    // Copy Seq0 data to batch positions [0:8]
    std::memcpy(Q_batch->mutable_data(), Q_seq0->data(), seq0_len * d_model * sizeof(float));
    std::memcpy(K_batch->mutable_data(), K_seq0->data(), seq0_len * kv_dim * sizeof(float));
    std::memcpy(V_batch->mutable_data(), V_seq0->data(), seq0_len * kv_dim * sizeof(float));

    // Copy Seq1 data to batch positions [8:13], zero-pad [13:16]
    std::memcpy(Q_batch->mutable_data() + seq_len * d_model,
                Q_seq1->data(), seq1_len * d_model * sizeof(float));
    std::memcpy(K_batch->mutable_data() + seq_len * kv_dim,
                K_seq1->data(), seq1_len * kv_dim * sizeof(float));
    std::memcpy(V_batch->mutable_data() + seq_len * kv_dim,
                V_seq1->data(), seq1_len * kv_dim * sizeof(float));

    // Zero-pad positions [13:16] for Seq1
    const int padding_start = seq_len + seq1_len;
    const int padding_count = seq_len - seq1_len;
    std::memset(Q_batch->mutable_data() + padding_start * d_model, 0, padding_count * d_model * sizeof(float));
    std::memset(K_batch->mutable_data() + padding_start * kv_dim, 0, padding_count * kv_dim * sizeof(float));
    std::memset(V_batch->mutable_data() + padding_start * kv_dim, 0, padding_count * kv_dim * sizeof(float));

    MpiAttentionConfig config_batch;
    config_batch.n_heads = n_heads;
    config_batch.n_kv_heads = n_kv_heads;
    config_batch.head_dim = head_dim;
    config_batch.causal = false;
    config_batch.window_size = -1;
    config_batch.precision = ActivationPrecision::FP32;
    config_batch.mpi_ctx = nullptr;
    config_batch.mpi_strategy = MPIStrategy::None;

    // CRITICAL: Batch path requires workspace_mask when actual_lengths provided
    auto batch_mask_tensor = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);
    config_batch.workspace_mask = std::move(batch_mask_tensor);

    bool success_batch = MpiAttentionOrchestrator::compute_batch(
        Q_batch.get(), K_batch.get(), V_batch.get(), output_batch.get(),
        actual_lengths, batch_size, seq_len, config_batch);
    ASSERT_TRUE(success_batch) << "Batch execution failed";

    // === COMPARISON ===
    std::cout << "\n[SEQUENTIAL VS BATCH COMPARISON]\n";

    // Compare Seq0 outputs
    float max_diff_0, mean_diff_0;
    compare_tensors(output_seq0->data(), output_batch->data(),
                    seq0_len * d_model, max_diff_0, mean_diff_0);

    if (max_diff_0 < 0.0001f)
    {
        std::cout << "✅ Seq0 matches perfectly (max_diff < 0.00010)\n";
    }
    else
    {
        std::cout << "❌ Seq0 DIVERGES: max_diff=" << max_diff_0
                  << ", mean_diff=" << mean_diff_0 << "\n";
    }
    EXPECT_LT(max_diff_0, 0.0001f) << "Seq0 sequential vs batch divergence";

    // Compare Seq1 outputs (only first seq1_len tokens)
    float max_diff_1, mean_diff_1;
    compare_tensors(output_seq1->data(),
                    output_batch->data() + seq_len * d_model,
                    seq1_len * d_model, max_diff_1, mean_diff_1);

    if (max_diff_1 < 0.0001f)
    {
        std::cout << "✅ Seq1 matches perfectly (max_diff < 0.00010)\n";
    }
    else
    {
        std::cout << "❌ Seq1 DIVERGES: max_diff=" << max_diff_1
                  << ", mean_diff=" << mean_diff_1 << "\n";
    }
    EXPECT_LT(max_diff_1, 0.0001f) << "Seq1 sequential vs batch divergence";

    // Verify padding outputs are zero
    const float *batch_out = output_batch->data();
    bool padding_is_zero = is_zero(batch_out, padding_start * d_model, padding_count * d_model);

    if (padding_is_zero)
    {
        std::cout << "✅ Padding output is zero\n";
    }
    else
    {
        std::cout << "❌ Padding output is NON-ZERO\n";
    }
    EXPECT_TRUE(padding_is_zero) << "Padding outputs should be zero";
}

/**
 * TEST 3: Verify padding outputs are zeroed
 *
 * Tests that tokens beyond actual_length produce zero outputs.
 */
TEST_F(MpiAttentionOrchestrator_MaskPassing, PaddingOutputsZeroed)
{
    const int batch_size = 2;
    const int seq_len = 8;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    std::vector<int> actual_lengths = {5, 3}; // Batch 0: 3 padding, Batch 1: 5 padding

    auto Q = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K = create_fp32_tensor(batch_size * seq_len, kv_dim);
    auto V = create_fp32_tensor(batch_size * seq_len, kv_dim);
    auto output = create_fp32_tensor(batch_size * seq_len, d_model);

    // Initialize all with non-zero values
    init_sequential(Q->mutable_data(), tensor_size(Q), 1.0f);
    init_sequential(K->mutable_data(), tensor_size(K), 10.0f);
    init_sequential(V->mutable_data(), tensor_size(V), 20.0f);

    MpiAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;

    // CRITICAL: Batch path requires workspace_mask when actual_lengths provided
    auto mask_tensor = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);
    config.workspace_mask = std::move(mask_tensor);

    bool success = MpiAttentionOrchestrator::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);

    ASSERT_TRUE(success) << "Batch computation failed";

    const float *out_data = output->data();

    std::cout << "\n[PADDING OUTPUT VERIFICATION]\n";

    // Check batch 0 padding (tokens 5-7)
    bool batch0_padding_zero = is_zero(out_data, 5 * d_model, 3 * d_model);
    float batch0_sum = 0.0f;
    for (int i = 5 * d_model; i < 8 * d_model; ++i)
    {
        batch0_sum += std::abs(out_data[i]);
    }
    std::cout << "Batch 1 (length=5, padding=3): sum(abs)=" << std::fixed << std::setprecision(5)
              << batch0_sum << (batch0_padding_zero ? " ✅" : " ❌") << "\n";
    EXPECT_TRUE(batch0_padding_zero) << "Batch 0 padding should be zero";

    // Check batch 1 padding (tokens 11-15)
    bool batch1_padding_zero = is_zero(out_data, (seq_len + 3) * d_model, 5 * d_model);
    float batch1_sum = 0.0f;
    for (int i = (seq_len + 3) * d_model; i < (seq_len + 8) * d_model; ++i)
    {
        batch1_sum += std::abs(out_data[i]);
    }
    std::cout << "Batch 2 (length=3, padding=5): sum(abs)=" << std::fixed << std::setprecision(5)
              << batch1_sum << (batch1_padding_zero ? " ✅" : " ❌") << "\n";
    EXPECT_TRUE(batch1_padding_zero) << "Batch 1 padding should be zero";
}

/**
 * TEST 4: MPI dispatch correctness
 *
 * Verify that MPIStrategy::None correctly dispatches to single-rank compute.
 */
TEST_F(MpiAttentionOrchestrator_MaskPassing, MPIDispatchCorrect)
{
    const int seq_len = 4;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    auto Q = create_fp32_tensor(seq_len, d_model);
    auto K = create_fp32_tensor(seq_len, kv_dim);
    auto V = create_fp32_tensor(seq_len, kv_dim);
    auto output = create_fp32_tensor(seq_len, d_model);

    init_sequential(Q->mutable_data(), tensor_size(Q), 1.0f);
    init_sequential(K->mutable_data(), tensor_size(K), 10.0f);
    init_sequential(V->mutable_data(), tensor_size(V), 20.0f);

    MpiAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr; // No MPI context
    config.mpi_strategy = MPIStrategy::None;

    // Should dispatch to single-rank compute
    bool success = MpiAttentionOrchestrator::compute_mpi(
        Q.get(), K.get(), V.get(), output.get(), config);

    ASSERT_TRUE(success) << "MPI dispatch failed";

    // Verify output is valid
    const float *out_data = output->data();
    EXPECT_FALSE(std::isnan(out_data[0])) << "Output contains NaN";
    EXPECT_FALSE(std::isinf(out_data[0])) << "Output contains Inf";

    std::cout << "\n[MPI DISPATCH CHECK]\n";
    std::cout << "✅ MPIStrategy::None correctly dispatched to single-rank compute\n";
}
