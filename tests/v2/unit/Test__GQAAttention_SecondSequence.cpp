/**
 * @file Test__GQAAttention_SecondSequence.cpp
 * @brief Unit tests specifically targeting second sequence in batched GQA attention
 * @author David Sanftenberg
 *
 * Purpose: Diagnose E2E test failures where sequence 0 passes but sequence 1 fails.
 * These tests isolate the handling of the second sequence in a batch to identify:
 * - Buffer offset calculation bugs
 * - Mask application issues for non-first sequences
 * - Cross-sequence interference
 * - Output buffer addressing for sequence 1
 */

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>

#include "v2/pipelines/attention/GQAAttention.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-4f;

    std::unique_ptr<FP32Tensor> create_fp32_tensor(size_t rows, size_t cols, int device_idx = 0)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, device_idx);
    }

    void init_random_like(float *data, size_t size, int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = dis(gen);
        }
    }

    void init_constant(float *data, size_t size, float value)
    {
        std::fill(data, data + size, value);
    }

    void init_sequential(float *data, size_t size, float start = 0.0f, float step = 1.0f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = start + i * step;
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
}

// ============================================================================
// Test 1: NonZeroOutput - Verify sequence 1 produces non-zero results
// ============================================================================

TEST(GQAAttention_SecondSequence, NonZeroOutput)
{
    // Purpose: Detect if sequence 1 output is all zeros (indicates buffer offset bug)
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Create batched tensors [batch_size * seq_len, d_model]
    auto Q = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output = create_fp32_tensor(batch_size * seq_len, d_model);

    // Initialize with random data
    init_random_like(Q->mutable_data(), batch_size * seq_len * d_model, 100);
    init_random_like(K->mutable_data(), batch_size * seq_len * d_model, 200);
    init_random_like(V->mutable_data(), batch_size * seq_len * d_model, 300);

    // Setup config
    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;
    config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    // Run batched attention
    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);
    ASSERT_TRUE(success) << "Batched attention failed";

    // Extract sequence 1 output
    const size_t seq1_offset = seq_len * d_model;
    const float *seq1_output = output->data() + seq1_offset;

    // Verify sequence 1 is not all zeros
    float sum = 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        sum += std::abs(seq1_output[i]);
        max_abs = std::max(max_abs, std::abs(seq1_output[i]));
    }

    EXPECT_GT(sum, 0.0f) << "Sequence 1 output is all zeros (buffer offset bug)";
    EXPECT_GT(max_abs, FP32_TOLERANCE) << "Sequence 1 output has no significant values";
}

// ============================================================================
// Test 2: IsolatedComparison - Compare seq1 batched vs standalone
// ============================================================================

TEST(GQAAttention_SecondSequence, IsolatedComparison)
{
    // Purpose: Verify sequence 1 in batch matches standalone execution
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Prepare sequence 1 input data
    std::vector<float> seq1_Q_data(seq_len * d_model);
    std::vector<float> seq1_K_data(seq_len * d_model);
    std::vector<float> seq1_V_data(seq_len * d_model);
    init_random_like(seq1_Q_data.data(), seq1_Q_data.size(), 400);
    init_random_like(seq1_K_data.data(), seq1_K_data.size(), 500);
    init_random_like(seq1_V_data.data(), seq1_V_data.size(), 600);

    // --- Standalone execution for sequence 1 ---
    auto Q_standalone = create_fp32_tensor(seq_len, d_model);
    auto K_standalone = create_fp32_tensor(seq_len, d_model);
    auto V_standalone = create_fp32_tensor(seq_len, d_model);
    auto output_standalone = create_fp32_tensor(seq_len, d_model);

    std::copy(seq1_Q_data.begin(), seq1_Q_data.end(), Q_standalone->mutable_data());
    std::copy(seq1_K_data.begin(), seq1_K_data.end(), K_standalone->mutable_data());
    std::copy(seq1_V_data.begin(), seq1_V_data.end(), V_standalone->mutable_data());

    GQAAttentionConfig config_standalone;
    config_standalone.n_heads = n_heads;
    config_standalone.n_kv_heads = n_kv_heads;
    config_standalone.head_dim = head_dim;
    config_standalone.causal = false;
    config_standalone.window_size = -1;
    config_standalone.precision = ActivationPrecision::FP32;
    config_standalone.mpi_ctx = nullptr;
    config_standalone.mpi_strategy = MPIStrategy::None;

    bool success_standalone = GQAAttention::compute(
        Q_standalone.get(), K_standalone.get(), V_standalone.get(), output_standalone.get(),
        config_standalone, 1, nullptr);
    ASSERT_TRUE(success_standalone) << "Standalone attention failed";

    // --- Batched execution with sequence 1 ---
    auto Q_batched = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_batched = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V_batched = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_batched = create_fp32_tensor(batch_size * seq_len, d_model);

    // Initialize sequence 0 with different data, sequence 1 with test data
    init_random_like(Q_batched->mutable_data(), seq_len * d_model, 700);
    init_random_like(K_batched->mutable_data(), seq_len * d_model, 800);
    init_random_like(V_batched->mutable_data(), seq_len * d_model, 900);

    const size_t seq1_offset = seq_len * d_model;
    std::copy(seq1_Q_data.begin(), seq1_Q_data.end(), Q_batched->mutable_data() + seq1_offset);
    std::copy(seq1_K_data.begin(), seq1_K_data.end(), K_batched->mutable_data() + seq1_offset);
    std::copy(seq1_V_data.begin(), seq1_V_data.end(), V_batched->mutable_data() + seq1_offset);

    GQAAttentionConfig config_batched;
    config_batched.n_heads = n_heads;
    config_batched.n_kv_heads = n_kv_heads;
    config_batched.head_dim = head_dim;
    config_batched.causal = false;
    config_batched.window_size = -1;
    config_batched.precision = ActivationPrecision::FP32;
    config_batched.mpi_ctx = nullptr;
    config_batched.mpi_strategy = MPIStrategy::None;
    config_batched.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success_batched = GQAAttention::compute_batch(
        Q_batched.get(), K_batched.get(), V_batched.get(), output_batched.get(),
        actual_lengths, batch_size, seq_len, config_batched);
    ASSERT_TRUE(success_batched) << "Batched attention failed";

    // Extract and compare sequence 1 outputs
    const float *seq1_output_batched = output_batched->data() + seq1_offset;
    auto result = compare_tensors(output_standalone->data(), seq1_output_batched,
                                  seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result, "Sequence 1 Standalone vs Batched");

    EXPECT_TRUE(result.equal)
        << "Sequence 1 output differs between standalone and batched execution\n"
        << "Max abs diff: " << result.max_abs_diff << "\n"
        << "Mean abs diff: " << result.mean_abs_diff;
}

// ============================================================================
// Test 3: BufferOffsetValidation - Verify Q/K/V offsets with known pattern
// ============================================================================

TEST(GQAAttention_SecondSequence, BufferOffsetValidation)
{
    // Purpose: Use sequential pattern to verify buffer offsets are correct
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    auto Q = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output = create_fp32_tensor(batch_size * seq_len, d_model);

    // Initialize with sequential patterns for easy visual verification
    // Sequence 0: 0.0, 0.1, 0.2, ...
    // Sequence 1: 100.0, 100.1, 100.2, ...
    init_sequential(Q->mutable_data(), seq_len * d_model, 0.0f, 0.1f);
    init_sequential(Q->mutable_data() + seq_len * d_model, seq_len * d_model, 100.0f, 0.1f);

    init_sequential(K->mutable_data(), seq_len * d_model, 0.0f, 0.1f);
    init_sequential(K->mutable_data() + seq_len * d_model, seq_len * d_model, 100.0f, 0.1f);

    init_sequential(V->mutable_data(), seq_len * d_model, 0.0f, 0.1f);
    init_sequential(V->mutable_data() + seq_len * d_model, seq_len * d_model, 100.0f, 0.1f);

    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = -1;
    config.precision = ActivationPrecision::FP32;
    config.mpi_ctx = nullptr;
    config.mpi_strategy = MPIStrategy::None;
    config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths, batch_size, seq_len, config);
    ASSERT_TRUE(success) << "Batched attention failed";

    // Verify sequence 1 input was read correctly (not zeros, in expected range)
    const size_t seq1_offset = seq_len * d_model;
    const float *seq1_output = output->data() + seq1_offset;

    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        min_val = std::min(min_val, seq1_output[i]);
        max_val = std::max(max_val, seq1_output[i]);
    }

    // With sequential inputs starting at 100.0, output should be non-zero and reasonable
    EXPECT_GT(max_val, 0.0f) << "Sequence 1 output has no positive values";
    EXPECT_LT(min_val, 1000.0f) << "Sequence 1 output values unreasonably large (offset bug?)";
    EXPECT_GT(min_val, -1000.0f) << "Sequence 1 output values unreasonably small (offset bug?)";
}

// ============================================================================
// Test 4: CausalMaskApplication - Verify causal mask works for seq1
// ============================================================================

TEST(GQAAttention_SecondSequence, CausalMaskApplication)
{
    // Purpose: Test that causal mask is correctly applied to sequence 1
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    auto Q = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_causal = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_noncausal = create_fp32_tensor(batch_size * seq_len, d_model);

    // Initialize with random data
    init_random_like(Q->mutable_data(), batch_size * seq_len * d_model, 1000);
    init_random_like(K->mutable_data(), batch_size * seq_len * d_model, 1100);
    init_random_like(V->mutable_data(), batch_size * seq_len * d_model, 1200);

    // Run with causal mask
    GQAAttentionConfig config_causal;
    config_causal.n_heads = n_heads;
    config_causal.n_kv_heads = n_kv_heads;
    config_causal.head_dim = head_dim;
    config_causal.causal = true;
    config_causal.window_size = -1;
    config_causal.precision = ActivationPrecision::FP32;
    config_causal.mpi_ctx = nullptr;
    config_causal.mpi_strategy = MPIStrategy::None;
    config_causal.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success_causal = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output_causal.get(),
        actual_lengths, batch_size, seq_len, config_causal);
    ASSERT_TRUE(success_causal) << "Causal attention failed";

    // Run with non-causal mask (different output expected)
    GQAAttentionConfig config_noncausal;
    config_noncausal.n_heads = n_heads;
    config_noncausal.n_kv_heads = n_kv_heads;
    config_noncausal.head_dim = head_dim;
    config_noncausal.causal = false;
    config_noncausal.window_size = -1;
    config_noncausal.precision = ActivationPrecision::FP32;
    config_noncausal.mpi_ctx = nullptr;
    config_noncausal.mpi_strategy = MPIStrategy::None;
    config_noncausal.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success_noncausal = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output_noncausal.get(),
        actual_lengths, batch_size, seq_len, config_noncausal);
    ASSERT_TRUE(success_noncausal) << "Non-causal attention failed";

    // Extract sequence 1 outputs
    const size_t seq1_offset = seq_len * d_model;
    const float *seq1_causal = output_causal->data() + seq1_offset;
    const float *seq1_noncausal = output_noncausal->data() + seq1_offset;

    // Causal and non-causal should produce different results for sequence 1
    auto result = compare_tensors(seq1_causal, seq1_noncausal, seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result, "Sequence 1 Causal vs Non-Causal");

    EXPECT_FALSE(result.equal)
        << "Causal and non-causal attention produce identical results for sequence 1\n"
        << "This suggests causal mask is not being applied correctly";
}

// ============================================================================
// Test 5: CrossSequenceIsolation - Ensure seq0 and seq1 don't interfere
// ============================================================================

TEST(GQAAttention_SecondSequence, CrossSequenceIsolation)
{
    // Purpose: Verify changing seq0 input doesn't affect seq1 output
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Sequence 1 input (constant across both runs)
    std::vector<float> seq1_Q_data(seq_len * d_model);
    std::vector<float> seq1_K_data(seq_len * d_model);
    std::vector<float> seq1_V_data(seq_len * d_model);
    init_random_like(seq1_Q_data.data(), seq1_Q_data.size(), 2000);
    init_random_like(seq1_K_data.data(), seq1_K_data.size(), 2100);
    init_random_like(seq1_V_data.data(), seq1_V_data.size(), 2200);

    // --- First run: seq0 with pattern A ---
    auto Q_run1 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_run1 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V_run1 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_run1 = create_fp32_tensor(batch_size * seq_len, d_model);

    init_constant(Q_run1->mutable_data(), seq_len * d_model, 1.0f);
    init_constant(K_run1->mutable_data(), seq_len * d_model, 1.0f);
    init_constant(V_run1->mutable_data(), seq_len * d_model, 1.0f);

    const size_t seq1_offset = seq_len * d_model;
    std::copy(seq1_Q_data.begin(), seq1_Q_data.end(), Q_run1->mutable_data() + seq1_offset);
    std::copy(seq1_K_data.begin(), seq1_K_data.end(), K_run1->mutable_data() + seq1_offset);
    std::copy(seq1_V_data.begin(), seq1_V_data.end(), V_run1->mutable_data() + seq1_offset);

    GQAAttentionConfig config_run1;
    config_run1.n_heads = n_heads;
    config_run1.n_kv_heads = n_kv_heads;
    config_run1.head_dim = head_dim;
    config_run1.causal = false;
    config_run1.window_size = -1;
    config_run1.precision = ActivationPrecision::FP32;
    config_run1.mpi_ctx = nullptr;
    config_run1.mpi_strategy = MPIStrategy::None;
    config_run1.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success_run1 = GQAAttention::compute_batch(
        Q_run1.get(), K_run1.get(), V_run1.get(), output_run1.get(),
        actual_lengths, batch_size, seq_len, config_run1);
    ASSERT_TRUE(success_run1) << "First batched attention failed";

    // --- Second run: seq0 with pattern B (different) ---
    auto Q_run2 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_run2 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V_run2 = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_run2 = create_fp32_tensor(batch_size * seq_len, d_model);

    init_constant(Q_run2->mutable_data(), seq_len * d_model, -1.0f); // Different from run1
    init_constant(K_run2->mutable_data(), seq_len * d_model, -1.0f);
    init_constant(V_run2->mutable_data(), seq_len * d_model, -1.0f);

    std::copy(seq1_Q_data.begin(), seq1_Q_data.end(), Q_run2->mutable_data() + seq1_offset);
    std::copy(seq1_K_data.begin(), seq1_K_data.end(), K_run2->mutable_data() + seq1_offset);
    std::copy(seq1_V_data.begin(), seq1_V_data.end(), V_run2->mutable_data() + seq1_offset);

    GQAAttentionConfig config_run2;
    config_run2.n_heads = n_heads;
    config_run2.n_kv_heads = n_kv_heads;
    config_run2.head_dim = head_dim;
    config_run2.causal = false;
    config_run2.window_size = -1;
    config_run2.precision = ActivationPrecision::FP32;
    config_run2.mpi_ctx = nullptr;
    config_run2.mpi_strategy = MPIStrategy::None;
    config_run2.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success_run2 = GQAAttention::compute_batch(
        Q_run2.get(), K_run2.get(), V_run2.get(), output_run2.get(),
        actual_lengths, batch_size, seq_len, config_run2);
    ASSERT_TRUE(success_run2) << "Second batched attention failed";

    // Compare sequence 1 outputs (should be identical)
    const float *seq1_output_run1 = output_run1->data() + seq1_offset;
    const float *seq1_output_run2 = output_run2->data() + seq1_offset;

    auto result = compare_tensors(seq1_output_run1, seq1_output_run2,
                                  seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result, "Sequence 1 Output: Run1 vs Run2");

    EXPECT_TRUE(result.equal)
        << "Sequence 1 output changed when sequence 0 input changed\n"
        << "This indicates cross-sequence interference\n"
        << "Max abs diff: " << result.max_abs_diff << "\n"
        << "Mean abs diff: " << result.mean_abs_diff;
}

// ============================================================================
// Test 6: SymmetricProcessing - Verify input/output symmetry
// ============================================================================

TEST(GQAAttention_SecondSequence, SymmetricProcessing)
{
    // Purpose: Test that swapping sequence positions produces swapped outputs
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;

    // Two distinct input patterns
    std::vector<float> pattern_A_Q(seq_len * d_model);
    std::vector<float> pattern_A_K(seq_len * d_model);
    std::vector<float> pattern_A_V(seq_len * d_model);
    init_random_like(pattern_A_Q.data(), pattern_A_Q.size(), 3000);
    init_random_like(pattern_A_K.data(), pattern_A_K.size(), 3100);
    init_random_like(pattern_A_V.data(), pattern_A_V.size(), 3200);

    std::vector<float> pattern_B_Q(seq_len * d_model);
    std::vector<float> pattern_B_K(seq_len * d_model);
    std::vector<float> pattern_B_V(seq_len * d_model);
    init_random_like(pattern_B_Q.data(), pattern_B_Q.size(), 3300);
    init_random_like(pattern_B_K.data(), pattern_B_K.size(), 3400);
    init_random_like(pattern_B_V.data(), pattern_B_V.size(), 3500);

    // --- First run: A at pos 0, B at pos 1 ---
    auto Q_AB = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_AB = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V_AB = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_AB = create_fp32_tensor(batch_size * seq_len, d_model);

    std::copy(pattern_A_Q.begin(), pattern_A_Q.end(), Q_AB->mutable_data());
    std::copy(pattern_B_Q.begin(), pattern_B_Q.end(), Q_AB->mutable_data() + seq_len * d_model);

    std::copy(pattern_A_K.begin(), pattern_A_K.end(), K_AB->mutable_data());
    std::copy(pattern_B_K.begin(), pattern_B_K.end(), K_AB->mutable_data() + seq_len * d_model);

    std::copy(pattern_A_V.begin(), pattern_A_V.end(), V_AB->mutable_data());
    std::copy(pattern_B_V.begin(), pattern_B_V.end(), V_AB->mutable_data() + seq_len * d_model);

    GQAAttentionConfig config_AB;
    config_AB.n_heads = n_heads;
    config_AB.n_kv_heads = n_kv_heads;
    config_AB.head_dim = head_dim;
    config_AB.causal = false;
    config_AB.window_size = -1;
    config_AB.precision = ActivationPrecision::FP32;
    config_AB.mpi_ctx = nullptr;
    config_AB.mpi_strategy = MPIStrategy::None;
    config_AB.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    std::vector<int> actual_lengths = {seq_len, seq_len};

    bool success_AB = GQAAttention::compute_batch(
        Q_AB.get(), K_AB.get(), V_AB.get(), output_AB.get(),
        actual_lengths, batch_size, seq_len, config_AB);
    ASSERT_TRUE(success_AB) << "First batched attention (AB) failed";

    // --- Second run: B at pos 0, A at pos 1 (swapped) ---
    auto Q_BA = create_fp32_tensor(batch_size * seq_len, d_model);
    auto K_BA = create_fp32_tensor(batch_size * seq_len, d_model);
    auto V_BA = create_fp32_tensor(batch_size * seq_len, d_model);
    auto output_BA = create_fp32_tensor(batch_size * seq_len, d_model);

    std::copy(pattern_B_Q.begin(), pattern_B_Q.end(), Q_BA->mutable_data());
    std::copy(pattern_A_Q.begin(), pattern_A_Q.end(), Q_BA->mutable_data() + seq_len * d_model);

    std::copy(pattern_B_K.begin(), pattern_B_K.end(), K_BA->mutable_data());
    std::copy(pattern_A_K.begin(), pattern_A_K.end(), K_BA->mutable_data() + seq_len * d_model);

    std::copy(pattern_B_V.begin(), pattern_B_V.end(), V_BA->mutable_data());
    std::copy(pattern_A_V.begin(), pattern_A_V.end(), V_BA->mutable_data() + seq_len * d_model);

    GQAAttentionConfig config_BA;
    config_BA.n_heads = n_heads;
    config_BA.n_kv_heads = n_kv_heads;
    config_BA.head_dim = head_dim;
    config_BA.causal = false;
    config_BA.window_size = -1;
    config_BA.precision = ActivationPrecision::FP32;
    config_BA.mpi_ctx = nullptr;
    config_BA.mpi_strategy = MPIStrategy::None;
    config_BA.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);

    bool success_BA = GQAAttention::compute_batch(
        Q_BA.get(), K_BA.get(), V_BA.get(), output_BA.get(),
        actual_lengths, batch_size, seq_len, config_BA);
    ASSERT_TRUE(success_BA) << "Second batched attention (BA) failed";

    // Extract outputs
    const size_t seq1_offset = seq_len * d_model;
    const float *output_AB_pos0 = output_AB->data();
    const float *output_AB_pos1 = output_AB->data() + seq1_offset;
    const float *output_BA_pos0 = output_BA->data();
    const float *output_BA_pos1 = output_BA->data() + seq1_offset;

    // Verify swapped outputs: AB[0] should match BA[1], AB[1] should match BA[0]
    auto result_A = compare_tensors(output_AB_pos0, output_BA_pos1,
                                    seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result_A, "Pattern A: AB[pos0] vs BA[pos1]");

    auto result_B = compare_tensors(output_AB_pos1, output_BA_pos0,
                                    seq_len * d_model, FP32_TOLERANCE);
    print_comparison(result_B, "Pattern B: AB[pos1] vs BA[pos0]");

    EXPECT_TRUE(result_A.equal)
        << "Pattern A output changes when sequence position changes\n"
        << "Max abs diff: " << result_A.max_abs_diff << "\n"
        << "Mean abs diff: " << result_A.mean_abs_diff;

    EXPECT_TRUE(result_B.equal)
        << "Pattern B output changes when sequence position changes\n"
        << "Max abs diff: " << result_B.max_abs_diff << "\n"
        << "Mean abs diff: " << result_B.mean_abs_diff;
}
