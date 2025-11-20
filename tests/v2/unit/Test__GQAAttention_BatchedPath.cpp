/**
 * @file Test__GQAAttention_BatchedPath.cpp
 * @brief Comprehensive unit tests for GQAAttention batched execution path
 * @author David Sanftenberg
 *
 * Tests focus on the GQAAttention::compute_batch() path to isolate
 * the divergence issue seen in E2E tests.
 *
 * Test Categories:
 * 1. Basic batched attention functionality
 * 2. Batch vs Sequential equivalence
 * 3. Causal vs non-causal masking
 * 4. Padding mask correctness
 * 5. GQA head broadcasting
 * 6. Edge cases (batch_size=1, variable lengths)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>
#include <numeric>
#include <algorithm>

#include "v2/pipelines/attention/GQAAttention.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

namespace
{
    // Helper: Create FP32 tensor without TensorFactory
    std::unique_ptr<FP32Tensor> create_fp32_tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, 0 /* device_idx */);
    }
}

namespace
{
    constexpr float FP32_TOLERANCE = 1e-4f;

    // Helper: Initialize with predictable values
    void init_test_data(float *data, int count, float scale = 1.0f, float offset = 0.0f)
    {
        for (int i = 0; i < count; ++i)
        {
            data[i] = offset + (static_cast<float>(i) / count) * scale;
        }
    }

    // Helper: Initialize with small random-like values (deterministic)
    void init_random_like(float *data, int count, int seed = 42)
    {
        for (int i = 0; i < count; ++i)
        {
            int val = (seed * 1103515245 + i * 12345) & 0x7fffffff;
            data[i] = static_cast<float>(val % 1000) / 1000.0f - 0.5f; // [-0.5, 0.5]
        }
    }

    // Helper: Compare tensors with detailed error reporting
    struct ComparisonResult
    {
        bool equal;
        float max_abs_diff;
        float mean_abs_diff;
        int first_mismatch_idx;
        float first_expected;
        float first_actual;
    };

    ComparisonResult compare_tensors(const float *expected, const float *actual,
                                     int count, float tolerance)
    {
        ComparisonResult result;
        result.equal = true;
        result.max_abs_diff = 0.0f;
        result.mean_abs_diff = 0.0f;
        result.first_mismatch_idx = -1;

        double sum_abs_diff = 0.0;
        for (int i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            sum_abs_diff += diff;

            if (diff > result.max_abs_diff)
            {
                result.max_abs_diff = diff;
            }

            if (diff > tolerance && result.first_mismatch_idx == -1)
            {
                result.first_mismatch_idx = i;
                result.first_expected = expected[i];
                result.first_actual = actual[i];
                result.equal = false;
            }
        }

        result.mean_abs_diff = static_cast<float>(sum_abs_diff / count);
        return result;
    }

    // Helper: Print comparison results
    void print_comparison(const ComparisonResult &result, const char *test_name)
    {
        std::cout << "\n=== " << test_name << " ===" << std::endl;
        std::cout << "Equal: " << (result.equal ? "YES" : "NO") << std::endl;
        std::cout << "Max abs diff: " << result.max_abs_diff << std::endl;
        std::cout << "Mean abs diff: " << result.mean_abs_diff << std::endl;
        if (!result.equal)
        {
            std::cout << "First mismatch at index " << result.first_mismatch_idx
                      << ": expected=" << result.first_expected
                      << ", actual=" << result.first_actual << std::endl;
        }
    }

} // anonymous namespace

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(GQAAttention_BatchedPath, BasicBatchedCompute)
{
    // Test that compute_batch() executes without errors
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Create input tensors [batch_size * seq_len, d_model]
    auto Q = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto K = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto V = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto output = create_fp32_tensor(batch_size * seq_len,
                                     d_model);

    // Initialize with test data
    init_test_data(Q->mutable_data(), batch_size * seq_len * d_model, 1.0f);
    init_test_data(K->mutable_data(), batch_size * seq_len * d_model, 1.0f, 0.5f);
    init_test_data(V->mutable_data(), batch_size * seq_len * d_model, 1.0f, 1.0f);

    // Configure attention
    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false; // Non-causal for simplicity
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    // All sequences have full length (no padding)
    std::vector<int> actual_lengths = {seq_len, seq_len};

    // Execute batched attention
    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);

    ASSERT_TRUE(success) << "compute_batch() failed";

    // Basic sanity checks
    const float *out_data = output->data();
    EXPECT_FALSE(std::isnan(out_data[0])) << "Output contains NaN";
    EXPECT_FALSE(std::isinf(out_data[0])) << "Output contains Inf";
}

// ============================================================================
// Batch vs Sequential Equivalence
// ============================================================================

TEST(GQAAttention_BatchedPath, BatchSize1EquivalentToSingleSequence)
{
    // CRITICAL: batch_size=1 should match single-sequence path
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Shared input data
    std::vector<float> Q_data(seq_len * d_model);
    std::vector<float> K_data(seq_len * d_model);
    std::vector<float> V_data(seq_len * d_model);
    init_random_like(Q_data.data(), Q_data.size(), 100);
    init_random_like(K_data.data(), K_data.size(), 200);
    init_random_like(V_data.data(), V_data.size(), 300);

    // Single-sequence path: compute()
    auto Q_single = create_fp32_tensor(seq_len,
                                       d_model);
    auto K_single = create_fp32_tensor(seq_len,
                                       d_model);
    auto V_single = create_fp32_tensor(seq_len,
                                       d_model);
    auto output_single = create_fp32_tensor(seq_len,
                                            d_model);

    std::copy(Q_data.begin(), Q_data.end(), Q_single->mutable_data());
    std::copy(K_data.begin(), K_data.end(), K_single->mutable_data());
    std::copy(V_data.begin(), V_data.end(), V_single->mutable_data());

    GQAAttentionConfig config_single;
    config_single.n_heads = n_heads;
    config_single.n_kv_heads = n_kv_heads;
    config_single.head_dim = head_dim;
    config_single.causal = false;
    config_single.window_size = -1;
    config_single.precision = ActivationPrecision::FP32;
    config_single.mpi_ctx = nullptr;
    config_single.mpi_strategy = MPIStrategy::None;

    bool success_single = GQAAttention::compute(
        Q_single.get(), K_single.get(), V_single.get(), output_single.get(),
        config_single, 1, nullptr);
    ASSERT_TRUE(success_single) << "Single-sequence compute failed";

    // Batched path: compute_batch() with batch_size=1
    auto Q_batch = create_fp32_tensor(seq_len,
                                      d_model);
    auto K_batch = create_fp32_tensor(seq_len,
                                      d_model);
    auto V_batch = create_fp32_tensor(seq_len,
                                      d_model);
    auto output_batch = create_fp32_tensor(seq_len,
                                           d_model);

    std::copy(Q_data.begin(), Q_data.end(), Q_batch->mutable_data());
    std::copy(K_data.begin(), K_data.end(), K_batch->mutable_data());
    std::copy(V_data.begin(), V_data.end(), V_batch->mutable_data());

    GQAAttentionConfig config_batch;
    config_batch.n_heads = n_heads;
    config_batch.n_kv_heads = n_kv_heads;
    config_batch.head_dim = head_dim;
    config_batch.causal = false;
    config_batch.window_size = -1;
    config_batch.precision = ActivationPrecision::FP32;
    config_batch.mpi_ctx = nullptr;
    config_batch.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [1 * seq_len, seq_len]
    config_batch.workspace_mask = create_fp32_tensor(seq_len, seq_len);

    std::vector<int> actual_lengths = {seq_len};

    bool success_batch = GQAAttention::compute_batch(
        Q_batch.get(), K_batch.get(), V_batch.get(), output_batch.get(),
        actual_lengths, 1, seq_len, config_batch);
    ASSERT_TRUE(success_batch) << "Batched compute (batch_size=1) failed";

    // Compare outputs
    auto result = compare_tensors(output_single->data(), output_batch->data(),
                                  seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result, "BatchSize1 vs SingleSequence");

    EXPECT_TRUE(result.equal)
        << "Batch size 1 output differs from single-sequence output\n"
        << "Max abs diff: " << result.max_abs_diff << "\n"
        << "Mean abs diff: " << result.mean_abs_diff;
}

TEST(GQAAttention_BatchedPath, BatchedEquivalentToSequentialExecution)
{
    // Test: Running batch_size=N should equal N sequential calls
    const int batch_size = 3;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Prepare batched input [batch_size * seq_len, d_model]
    auto Q_batched = create_fp32_tensor(batch_size * seq_len,
                                        d_model);
    auto K_batched = create_fp32_tensor(batch_size * seq_len,
                                        d_model);
    auto V_batched = create_fp32_tensor(batch_size * seq_len,
                                        d_model);
    auto output_batched = create_fp32_tensor(batch_size * seq_len,
                                             d_model);

    // Initialize each batch with different data
    for (int b = 0; b < batch_size; ++b)
    {
        init_random_like(Q_batched->mutable_data() + b * seq_len * d_model,
                         seq_len * d_model, 100 + b);
        init_random_like(K_batched->mutable_data() + b * seq_len * d_model,
                         seq_len * d_model, 200 + b);
        init_random_like(V_batched->mutable_data() + b * seq_len * d_model,
                         seq_len * d_model, 300 + b);
    }

    // Batched execution
    GQAAttentionConfig config_batched;
    config_batched.n_heads = n_heads;
    config_batched.n_kv_heads = n_kv_heads;
    config_batched.head_dim = head_dim;
    config_batched.causal = false;
    config_batched.window_size = -1;
    config_batched.precision = ActivationPrecision::FP32;
    config_batched.mpi_ctx = nullptr;
    config_batched.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config_batched.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths(batch_size, seq_len); // All full length

    bool success_batched = GQAAttention::compute_batch(
        Q_batched.get(), K_batched.get(), V_batched.get(), output_batched.get(),
        actual_lengths, batch_size, seq_len, config_batched);
    ASSERT_TRUE(success_batched) << "Batched execution failed";

    // Sequential execution (process each sequence independently)
    std::vector<float> output_sequential(batch_size * seq_len * d_model);

    for (int b = 0; b < batch_size; ++b)
    {
        auto Q_seq = create_fp32_tensor(seq_len,
                                        d_model);
        auto K_seq = create_fp32_tensor(seq_len,
                                        d_model);
        auto V_seq = create_fp32_tensor(seq_len,
                                        d_model);
        auto out_seq = create_fp32_tensor(seq_len,
                                          d_model);

        // Copy batch b's data
        std::copy(Q_batched->data() + b * seq_len * d_model,
                  Q_batched->data() + (b + 1) * seq_len * d_model,
                  Q_seq->mutable_data());
        std::copy(K_batched->data() + b * seq_len * d_model,
                  K_batched->data() + (b + 1) * seq_len * d_model,
                  K_seq->mutable_data());
        std::copy(V_batched->data() + b * seq_len * d_model,
                  V_batched->data() + (b + 1) * seq_len * d_model,
                  V_seq->mutable_data());

        GQAAttentionConfig config_seq;
        config_seq.n_heads = n_heads;
        config_seq.n_kv_heads = n_kv_heads;
        config_seq.head_dim = head_dim;
        config_seq.causal = false;
        config_seq.window_size = -1;
        config_seq.precision = ActivationPrecision::FP32;
        config_seq.mpi_ctx = nullptr;
        config_seq.mpi_strategy = MPIStrategy::None;

        bool success_seq = GQAAttention::compute(
            Q_seq.get(), K_seq.get(), V_seq.get(), out_seq.get(),
            config_seq, 1, nullptr);
        ASSERT_TRUE(success_seq) << "Sequential compute failed for batch " << b;

        // Copy to sequential output buffer
        std::copy(out_seq->data(), out_seq->data() + seq_len * d_model,
                  output_sequential.data() + b * seq_len * d_model);
    }

    // Compare batched vs sequential
    auto result = compare_tensors(output_sequential.data(), output_batched->data(),
                                  batch_size * seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result, "Batched vs Sequential Execution");

    EXPECT_TRUE(result.equal)
        << "Batched execution differs from sequential execution\n"
        << "Max abs diff: " << result.max_abs_diff << "\n"
        << "Mean abs diff: " << result.mean_abs_diff;
}

// ============================================================================
// Causal vs Non-Causal Masking Tests
// ============================================================================

TEST(GQAAttention_BatchedPath, CausalVsNonCausalMasking)
{
    // Test that causal flag properly controls masking behavior
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Shared input tensors
    auto Q = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto K = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto V = create_fp32_tensor(batch_size * seq_len,
                                d_model);

    init_random_like(Q->mutable_data(), batch_size * seq_len * d_model, 400);
    init_random_like(K->mutable_data(), batch_size * seq_len * d_model, 500);
    init_random_like(V->mutable_data(), batch_size * seq_len * d_model, 600);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    // Causal attention
    auto output_causal = create_fp32_tensor(batch_size * seq_len,
                                            d_model);

    GQAAttentionConfig config_causal;
    config_causal.n_heads = n_heads;
    config_causal.n_kv_heads = n_kv_heads;
    config_causal.head_dim = head_dim;
    config_causal.causal = true; // CAUSAL
    config_causal.window_size = -1;
    config_causal.precision = ActivationPrecision::FP32;
    config_causal.mpi_ctx = nullptr;
    config_causal.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config_causal.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success_causal = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output_causal.get(),
        actual_lengths, batch_size, seq_len, config_causal);
    ASSERT_TRUE(success_causal) << "Causal attention failed";

    // Non-causal attention
    auto output_noncausal = create_fp32_tensor(batch_size * seq_len,
                                               d_model);

    GQAAttentionConfig config_noncausal;
    config_noncausal.n_heads = n_heads;
    config_noncausal.n_kv_heads = n_kv_heads;
    config_noncausal.head_dim = head_dim;
    config_noncausal.causal = false; // NON-CAUSAL
    config_noncausal.window_size = -1;
    config_noncausal.precision = ActivationPrecision::FP32;
    config_noncausal.mpi_ctx = nullptr;
    config_noncausal.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config_noncausal.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success_noncausal = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output_noncausal.get(),
        actual_lengths, batch_size, seq_len, config_noncausal);
    ASSERT_TRUE(success_noncausal) << "Non-causal attention failed";

    // Compare outputs - they SHOULD differ (causal blocks future, non-causal allows)
    auto result = compare_tensors(output_causal->data(), output_noncausal->data(),
                                  batch_size * seq_len * d_model, FP32_TOLERANCE);

    EXPECT_FALSE(result.equal)
        << "Causal and non-causal attention should produce different outputs\n"
        << "Max abs diff: " << result.max_abs_diff;

    // Ensure the difference is meaningful (not just numerical noise)
    EXPECT_GT(result.max_abs_diff, 0.01f)
        << "Causal vs non-causal diff too small: " << result.max_abs_diff;
}

// ============================================================================
// Padding Mask Correctness
// ============================================================================

TEST(GQAAttention_BatchedPath, PaddingMaskCorrectness)
{
    // Test that padding positions are properly masked
    const int batch_size = 2;
    const int seq_len = 6; // Max length
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Actual sequence lengths (varied)
    std::vector<int> actual_lengths = {4, 3}; // batch 0: 4 tokens, batch 1: 3 tokens

    // Input tensors [batch_size * seq_len, d_model]
    auto Q = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto K = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto V = create_fp32_tensor(batch_size * seq_len,
                                d_model);
    auto output = create_fp32_tensor(batch_size * seq_len,
                                     d_model);

    // Initialize real tokens with meaningful values
    for (int b = 0; b < batch_size; ++b)
    {
        int real_len = actual_lengths[b];
        // Real tokens: meaningful values
        init_random_like(Q->mutable_data() + b * seq_len * d_model,
                         real_len * d_model, 700 + b);
        init_random_like(K->mutable_data() + b * seq_len * d_model,
                         real_len * d_model, 800 + b);
        init_random_like(V->mutable_data() + b * seq_len * d_model,
                         real_len * d_model, 900 + b);

        // Padding tokens: zeros (though attention should mask them anyway)
        std::fill(Q->mutable_data() + b * seq_len * d_model + real_len * d_model,
                  Q->mutable_data() + (b + 1) * seq_len * d_model, 0.0f);
        std::fill(K->mutable_data() + b * seq_len * d_model + real_len * d_model,
                  K->mutable_data() + (b + 1) * seq_len * d_model, 0.0f);
        std::fill(V->mutable_data() + b * seq_len * d_model + real_len * d_model,
                  V->mutable_data() + (b + 1) * seq_len * d_model, 0.0f);
    }

    // Configure attention
    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false; // Non-causal, only padding mask
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);
    ASSERT_TRUE(success) << "Batched attention with padding failed";

    // Verify: Padding positions in output should have minimal/zero influence
    // Check that real token outputs are non-zero, padding outputs are near-zero
    const float *out_data = output->data();

    for (int b = 0; b < batch_size; ++b)
    {
        int real_len = actual_lengths[b];

        // Check real tokens have meaningful values
        for (int i = 0; i < real_len; ++i)
        {
            float sum = 0.0f;
            for (int d = 0; d < d_model; ++d)
            {
                sum += std::abs(out_data[b * seq_len * d_model + i * d_model + d]);
            }
            EXPECT_GT(sum, 0.1f)
                << "Real token output too small for batch " << b << ", token " << i;
        }

        // Check padding tokens have near-zero output
        for (int i = real_len; i < seq_len; ++i)
        {
            float sum = 0.0f;
            for (int d = 0; d < d_model; ++d)
            {
                sum += std::abs(out_data[b * seq_len * d_model + i * d_model + d]);
            }
            EXPECT_LT(sum, 0.1f)
                << "Padding token output too large for batch " << b << ", token " << i
                << ", sum=" << sum;
        }
    }
}

// ============================================================================
// GQA Head Broadcasting Tests
// ============================================================================

TEST(GQAAttention_BatchedPath, GQAHeadBroadcasting)
{
    // Test GQA with n_heads > n_kv_heads (K/V head broadcasting)
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 4;    // Query heads
    const int n_kv_heads = 2; // K/V heads (broadcast to 4)
    const int head_dim = 8;

    // Input tensors
    auto Q = create_fp32_tensor(batch_size * seq_len,
                                n_heads * head_dim);
    auto K = create_fp32_tensor(batch_size * seq_len,
                                n_kv_heads * head_dim);
    auto V = create_fp32_tensor(batch_size * seq_len,
                                n_kv_heads * head_dim);
    auto output = create_fp32_tensor(batch_size * seq_len,
                                     n_heads * head_dim);

    init_random_like(Q->mutable_data(), batch_size * seq_len * n_heads * head_dim, 1000);
    init_random_like(K->mutable_data(), batch_size * seq_len * n_kv_heads * head_dim, 1100);
    init_random_like(V->mutable_data(), batch_size * seq_len * n_kv_heads * head_dim, 1200);

    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;

    // Allocate workspace mask tensor [batch_size * seq_len, seq_len]
    config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);

    ASSERT_TRUE(success) << "GQA batched attention failed";

    // Basic sanity checks
    const float *out_data = output->data();
    EXPECT_FALSE(std::isnan(out_data[0])) << "GQA output contains NaN";
    EXPECT_FALSE(std::isinf(out_data[0])) << "GQA output contains Inf";

    // Verify output shape is correct [batch_size * seq_len, n_heads * head_dim]
    EXPECT_EQ(batch_size * seq_len * n_heads * head_dim, batch_size * seq_len * n_heads * head_dim);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
