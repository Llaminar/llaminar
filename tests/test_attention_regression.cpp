/**
 * @file test_attention_regression.cpp
 * @brief Regression tests for attention kernel bugs (bias overflow, RoPE GQA, dimension validation)
 * @author David Sanftenberg
 *
 * This file contains targeted regression tests for specific bugs discovered during development:
 * 1. Heap buffer overflow in bias access (reading past 1-element dummy bias)
 * 2. RoPE GQA buffer overflow (reading K buffer with Q head count)
 * 3. Dimension mismatch detection before cblas operations
 *
 * These tests use assertions and memory safety checks to catch regressions early.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include "kernels/MPIAttentionKernel.h"
#include "kernels/common/attention_primitives.h"
#include "tensors/simple_tensor.h"
#include "logger.h"

using namespace llaminar;

class AttentionRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void TearDown() override
    {
        // Don't finalize - other tests may still need MPI
    }

    int rank = 0;
    int world_size = 1;
};

/**
 * @brief Test that RoPE correctly handles different head counts for Q and K (GQA)
 *
 * Regression test for bug where apply_rope() assumed Q and K have same number of heads.
 * For GQA, Q has more heads than K, and the old code would read past the end of K buffer.
 *
 * This test verifies:
 * - No buffer overflow when q_heads > k_heads
 * - Both Q and K are correctly rotated
 * - Values are modified (not zeros)
 */
TEST_F(AttentionRegressionTest, RopeGQABufferSafety)
{
    const int seq_len = 4;
    const int head_dim = 32;
    const int q_heads = 4; // MHA: 4 query heads
    const int k_heads = 2; // GQA: 2 key-value heads
    const int n_past = 0;
    const float freq_base = 10000.0f;

    // Allocate Q and K with correct sizes
    const size_t q_size = seq_len * q_heads * head_dim;
    const size_t k_size = seq_len * k_heads * head_dim;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);

    // Initialize with non-zero values to detect corruption
    for (size_t i = 0; i < q_size; ++i)
    {
        q_data[i] = 0.01f * (i % 100 - 50);
    }
    for (size_t i = 0; i < k_size; ++i)
    {
        k_data[i] = 0.01f * (i % 100 - 50);
    }

    // Store original values
    std::vector<float> q_orig = q_data;
    std::vector<float> k_orig = k_data;

    // Apply RoPE - this should NOT crash or overflow
    ASSERT_NO_THROW({
        llaminar::attn::apply_rope(
            q_data.data(), k_data.data(),
            seq_len, head_dim, q_heads, k_heads,
            n_past, freq_base);
    }) << "RoPE should not crash with different head counts (GQA)";

    // Verify Q was modified (RoPE applied)
    bool q_modified = false;
    for (size_t i = 0; i < q_size; ++i)
    {
        if (std::abs(q_data[i] - q_orig[i]) > 1e-6f)
        {
            q_modified = true;
            break;
        }
    }
    EXPECT_TRUE(q_modified) << "Q should be modified by RoPE";

    // Verify K was modified (RoPE applied)
    bool k_modified = false;
    for (size_t i = 0; i < k_size; ++i)
    {
        if (std::abs(k_data[i] - k_orig[i]) > 1e-6f)
        {
            k_modified = true;
            break;
        }
    }
    EXPECT_TRUE(k_modified) << "K should be modified by RoPE";

    // Verify no NaN/Inf corruption
    for (size_t i = 0; i < q_size; ++i)
    {
        ASSERT_FALSE(std::isnan(q_data[i])) << "Q[" << i << "] is NaN";
        ASSERT_FALSE(std::isinf(q_data[i])) << "Q[" << i << "] is Inf";
    }
    for (size_t i = 0; i < k_size; ++i)
    {
        ASSERT_FALSE(std::isnan(k_data[i])) << "K[" << i << "] is NaN";
        ASSERT_FALSE(std::isinf(k_data[i])) << "K[" << i << "] is Inf";
    }
}

/**
 * @brief Test that RoPE works correctly when Q and K have same head count
 *
 * Sanity check that the optimized path (when q_heads == k_heads) still works.
 */
TEST_F(AttentionRegressionTest, RopeMHABufferSafety)
{
    const int seq_len = 4;
    const int head_dim = 32;
    const int heads = 4; // Same for both Q and K (MHA)
    const int n_past = 0;
    const float freq_base = 10000.0f;

    const size_t size = seq_len * heads * head_dim;
    std::vector<float> q_data(size);
    std::vector<float> k_data(size);

    // Initialize with varying values (not uniform)
    for (size_t i = 0; i < size; ++i)
    {
        q_data[i] = 0.01f * (i % 100 - 50);
        k_data[i] = 0.01f * (i % 100 - 50);
    }

    // Store original values
    std::vector<float> q_orig = q_data;
    std::vector<float> k_orig = k_data;

    ASSERT_NO_THROW({
        llaminar::attn::apply_rope(
            q_data.data(), k_data.data(),
            seq_len, head_dim, heads, heads,
            n_past, freq_base);
    }) << "RoPE should not crash with same head counts (MHA)";

    // Basic sanity: values should have changed (check if any element changed)
    bool q_changed = false;
    bool k_changed = false;
    for (size_t i = 0; i < size; ++i)
    {
        if (std::abs(q_data[i] - q_orig[i]) > 1e-6f)
            q_changed = true;
        if (std::abs(k_data[i] - k_orig[i]) > 1e-6f)
            k_changed = true;
    }
    EXPECT_TRUE(q_changed) << "Q should be rotated by RoPE";
    EXPECT_TRUE(k_changed) << "K should be rotated by RoPE";
}

/**
 * @brief Test bias buffer size validation
 *
 * Regression test for bug where size-1 dummy bias tensors were passed directly to
 * matmul_with_bias, causing buffer overflow when reading bias[n] for n=0..127.
 *
 * The kernel should:
 * - Accept nullptr bias (no bias)
 * - Accept properly sized bias tensors
 * - NOT crash with size-1 dummy bias (should convert to nullptr)
 */
TEST_F(AttentionRegressionTest, BiasBufferValidation)
{
    const int seq_len = 4;
    const int d_model = 128;
    const int n_head = 4;
    const int head_dim = 32;
    const int layer_idx = -1;

    // Create minimal input tensors
    auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});
    auto wq = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wk = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wv = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wo = std::make_shared<SimpleTensor>(std::vector<int>{d_model, n_head * head_dim});

    // Initialize weights with small values
    for (size_t i = 0; i < wq->size(); ++i)
        wq->data()[i] = 0.001f;
    for (size_t i = 0; i < wk->size(); ++i)
        wk->data()[i] = 0.001f;
    for (size_t i = 0; i < wv->size(); ++i)
        wv->data()[i] = 0.001f;
    for (size_t i = 0; i < wo->size(); ++i)
        wo->data()[i] = 0.001f;

    // Create SIZE-1 dummy bias tensors (the bug case!)
    auto dummy_bq = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_bk = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_bv = std::make_shared<SimpleTensor>(std::vector<int>{1});
    dummy_bq->data()[0] = 0.0f;
    dummy_bk->data()[0] = 0.0f;
    dummy_bv->data()[0] = 0.0f;

    // Create dummy KV cache tensors (size-1 to indicate no cache)
    auto dummy_k_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_v_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});

    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Create kernel
    MPIAttentionKernel kernel(n_head, n_head, head_dim);

    // Execute with dummy (size-1) bias - should NOT crash!
    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq, wk, wv, wo, dummy_bq, dummy_bk, dummy_bv, dummy_k_cache, dummy_v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    ASSERT_NO_THROW({
        bool success = kernel.execute(inputs, outputs);
        EXPECT_TRUE(success) << "Kernel should succeed with size-1 dummy bias";
    }) << "Kernel should not crash with size-1 bias tensors";

    // Verify output is not garbage (no NaN/Inf)
    for (size_t i = 0; i < output->size(); ++i)
    {
        ASSERT_FALSE(std::isnan(output->data()[i])) << "Output[" << i << "] is NaN";
        ASSERT_FALSE(std::isinf(output->data()[i])) << "Output[" << i << "] is Inf";
    }
}

/**
 * @brief Test dimension mismatch detection
 *
 * Regression test for missing validation that caused cblas crashes.
 * The kernel should detect and reject incorrect weight dimensions.
 */
TEST_F(AttentionRegressionTest, DimensionMismatchDetection)
{
    const int seq_len = 4;
    const int d_model = 128;
    const int n_head = 4;
    const int head_dim = 32;
    const int layer_idx = -1;

    auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Create WRONG-SIZED weight matrices (this is the bug we're testing for)
    // Expected: [n_head * head_dim, d_model] = [128, 128]
    // But provide: [256, 128] instead
    const int wrong_size = 256;
    auto wq_wrong = std::make_shared<SimpleTensor>(std::vector<int>{wrong_size, d_model});
    auto wk = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wv = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wo = std::make_shared<SimpleTensor>(std::vector<int>{d_model, n_head * head_dim});

    auto dummy_bias = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Create dummy KV cache tensors
    auto dummy_k_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_v_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});

    MPIAttentionKernel kernel(n_head, n_head, head_dim);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq_wrong, wk, wv, wo, dummy_bias, dummy_bias, dummy_bias, dummy_k_cache, dummy_v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Should detect dimension mismatch and return false (not crash!)
    bool success = kernel.execute(inputs, outputs);
    EXPECT_FALSE(success) << "Kernel should reject wrong-sized wq matrix";
}

/**
 * @brief Test GQA dimension handling
 *
 * Verify that GQA (q_heads != k_heads) works correctly with proper dimensions.
 */
TEST_F(AttentionRegressionTest, GQADimensionHandling)
{
    const int seq_len = 4;
    const int d_model = 128;
    const int n_head = 4;    // Q heads
    const int n_head_kv = 2; // K/V heads (GQA)
    const int head_dim = 32;
    const int layer_idx = -1;

    auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Q: [n_head * head_dim, d_model] = [128, 128]
    auto wq = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});

    // K/V: [n_head_kv * head_dim, d_model] = [64, 128] (GQA!)
    auto wk = std::make_shared<SimpleTensor>(std::vector<int>{n_head_kv * head_dim, d_model});
    auto wv = std::make_shared<SimpleTensor>(std::vector<int>{n_head_kv * head_dim, d_model});

    auto wo = std::make_shared<SimpleTensor>(std::vector<int>{d_model, n_head * head_dim});

    // Initialize with small values
    for (size_t i = 0; i < wq->size(); ++i)
        wq->data()[i] = 0.001f;
    for (size_t i = 0; i < wk->size(); ++i)
        wk->data()[i] = 0.001f;
    for (size_t i = 0; i < wv->size(); ++i)
        wv->data()[i] = 0.001f;
    for (size_t i = 0; i < wo->size(); ++i)
        wo->data()[i] = 0.001f;

    auto dummy_bias = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Create dummy KV cache tensors
    auto dummy_k_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_v_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});

    // Create GQA kernel (n_head_kv < n_head)
    MPIAttentionKernel kernel(n_head, n_head_kv, head_dim);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq, wk, wv, wo, dummy_bias, dummy_bias, dummy_bias, dummy_k_cache, dummy_v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Should succeed with GQA configuration
    ASSERT_NO_THROW({
        bool success = kernel.execute(inputs, outputs);
        EXPECT_TRUE(success) << "Kernel should succeed with GQA (different Q/KV head counts)";
    }) << "GQA execution should not crash";

    // Verify output is valid
    for (size_t i = 0; i < output->size(); ++i)
    {
        ASSERT_FALSE(std::isnan(output->data()[i])) << "GQA output[" << i << "] is NaN";
        ASSERT_FALSE(std::isinf(output->data()[i])) << "GQA output[" << i << "] is Inf";
    }
}

/**
 * @brief Test edge case: single token, single head
 *
 * Minimal configuration to test boundary conditions.
 */
TEST_F(AttentionRegressionTest, MinimalConfiguration)
{
    const int seq_len = 1;
    const int d_model = 64;
    const int n_head = 1;
    const int head_dim = 64;
    const int layer_idx = -1;

    auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});
    auto wq = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wk = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wv = std::make_shared<SimpleTensor>(std::vector<int>{n_head * head_dim, d_model});
    auto wo = std::make_shared<SimpleTensor>(std::vector<int>{d_model, n_head * head_dim});

    for (size_t i = 0; i < wq->size(); ++i)
        wq->data()[i] = 0.001f;
    for (size_t i = 0; i < wk->size(); ++i)
        wk->data()[i] = 0.001f;
    for (size_t i = 0; i < wv->size(); ++i)
        wv->data()[i] = 0.001f;
    for (size_t i = 0; i < wo->size(); ++i)
        wo->data()[i] = 0.001f;

    auto dummy_bias = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});

    // Create dummy KV cache tensors
    auto dummy_k_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});
    auto dummy_v_cache = std::make_shared<SimpleTensor>(std::vector<int>{1});

    MPIAttentionKernel kernel(n_head, n_head, head_dim);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq, wk, wv, wo, dummy_bias, dummy_bias, dummy_bias, dummy_k_cache, dummy_v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    ASSERT_NO_THROW({
        bool success = kernel.execute(inputs, outputs);
        EXPECT_TRUE(success) << "Minimal configuration should work";
    });
}

/**
 * @brief Test RoPE with various head_dim values
 *
 * Ensure RoPE works correctly with different head dimensions (64, 128, etc.)
 */
TEST_F(AttentionRegressionTest, RopeVariousHeadDimensions)
{
    const int seq_len = 2;
    const int n_past = 0;
    const float freq_base = 10000.0f;

    // Test different head_dim values
    std::vector<int> head_dims = {32, 64, 80, 96, 128};

    for (int head_dim : head_dims)
    {
        const int q_heads = 4;
        const int k_heads = 2;

        const size_t q_size = seq_len * q_heads * head_dim;
        const size_t k_size = seq_len * k_heads * head_dim;

        std::vector<float> q_data(q_size, 0.01f);
        std::vector<float> k_data(k_size, 0.02f);

        ASSERT_NO_THROW({
            llaminar::attn::apply_rope(
                q_data.data(), k_data.data(),
                seq_len, head_dim, q_heads, k_heads,
                n_past, freq_base);
        }) << "RoPE should work with head_dim="
           << head_dim;

        // Verify no corruption
        for (size_t i = 0; i < q_size; ++i)
        {
            ASSERT_FALSE(std::isnan(q_data[i])) << "head_dim=" << head_dim << " Q[" << i << "] is NaN";
        }
        for (size_t i = 0; i < k_size; ++i)
        {
            ASSERT_FALSE(std::isnan(k_data[i])) << "head_dim=" << head_dim << " K[" << i << "] is NaN";
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        MPI_Init(&argc, &argv);
    }

    int result = RUN_ALL_TESTS();

    // Don't finalize - let MPI clean up naturally
    return result;
}
