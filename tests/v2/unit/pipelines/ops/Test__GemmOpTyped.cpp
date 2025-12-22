/**
 * @file Test__GemmOpTyped.cpp
 * @brief Unit tests for GemmOpTyped<P> typed GEMM operations
 * @author David Sanftenberg
 *
 * Tests the typed GEMM (matrix multiplication) operations across precision modes:
 * - FP32: Standard float GEMM
 * - Q8_1: Native quantized GEMM via precomputed blocks
 *
 * Key tests:
 * 1. Basic FP32 GEMM correctness (C = A @ W^T)
 * 2. Q8_1 GEMM parity vs FP32 reference
 * 3. Q8_1→Q8_1 path (input Q8_1, output Q8_1)
 * 4. FP32→Q8_1 path (input FP32, output Q8_1)
 * 5. Pipeline-realistic dimensions (Qwen 2.5 0.5B QKV projection)
 * 6. Factory function consistency
 *
 * These tests use real model weights from models/qwen2.5-0.5b-instruct-q4_0.gguf
 * to verify the QKV projection path produces correct Q8_1 output.
 */

#include <gtest/gtest.h>

#include "../../../../src/v2/pipelines/ops/GemmOp.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/kernels/KernelFactory.h"
#include "../../../../src/v2/pipelines/RuntimeConfig.h"
#include "../../../../src/v2/loaders/ModelContext.h"

#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__GemmOpTyped : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    void SetUp() override
    {
        // Seed RNG for reproducibility
        gen_.seed(42);

        // Load model for tests that need real weights
        model_ctx_ = ModelContext::create(MODEL_PATH);
    }

    void TearDown() override
    {
        // Clear kernel cache after each test to avoid stale entries
        llaminar::v2::kernels::KernelFactory::clearCache();
        model_ctx_.reset();
    }

    // Generate random FP32 data in range [-range, range]
    std::vector<float> generateRandomFP32(size_t count, float range = 1.0f)
    {
        std::vector<float> data(count);
        std::uniform_real_distribution<float> dist(-range, range);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen_);
        }
        return data;
    }

    // Compute FP32 reference GEMM: C = A @ W^T
    // A: [m, k], W: [n, k] (stored row-major, so W^T is [k, n])
    std::vector<float> computeFP32Reference(
        const std::vector<float> &A, // [m, k]
        const std::vector<float> &W, // [n, k]
        int m, int n, int k)
    {
        std::vector<float> C(m * n, 0.0f);
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l)
                {
                    // A[i, l] * W[j, l] (W is row-major, so W^T[l, j] = W[j, l])
                    sum += A[i * k + l] * W[j * k + l];
                }
                C[i * n + j] = sum;
            }
        }
        return C;
    }

    // Compute cosine similarity between two vectors
    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12));
    }

    // Compute relative L2 error
    float relativeL2Error(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    // Extract float data from Q8_1Tensor (dequantizes)
    std::vector<float> extractQ8_1Data(const Q8_1Tensor *tensor, size_t count)
    {
        const float *data = tensor->fp32_data(); // Explicit dequantization via fp32_data()
        return std::vector<float>(data, data + count);
    }

    std::mt19937 gen_;
};

// =============================================================================
// FP32 GEMM Tests (using synthetic data)
// =============================================================================

/**
 * @brief Test FP32 GEMM - basic correctness (small dimensions)
 */
TEST_F(Test__GemmOpTyped, FP32_BasicCorrectness)
{
    const int m = 4;  // seq_len
    const int n = 8;  // output features
    const int k = 16; // input features

    // Generate random input
    auto A_data = generateRandomFP32(m * k);
    auto W_data = generateRandomFP32(n * k);

    // Compute FP32 reference
    auto expected = computeFP32Reference(A_data, W_data, m, n, k);

    // Create tensors
    auto A = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)}, -1);
    auto W = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, -1);
    auto C = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)}, -1);

    std::memcpy(A->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    std::memcpy(W->mutable_data(), W_data.data(), W_data.size() * sizeof(float));

    // Execute typed op
    GemmOpTyped<ActivationPrecision::FP32> gemm;
    ASSERT_TRUE(gemm.execute(A.get(), W.get(), C.get(), m, n, k));

    // Verify output
    const float *C_data = C->data();
    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        float error = std::abs(C_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Basic] Max error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-4f); // Allow small FP32 accumulation error
}

/**
 * @brief Test FP32 GEMM - Qwen 2.5 0.5B dimensions (Q projection)
 *
 * seq_len=9, d_model=896, q_dim=896 (14 heads × 64 head_dim)
 */
TEST_F(Test__GemmOpTyped, FP32_QwenDimensions)
{
    const int m = 9;   // seq_len
    const int n = 896; // q_dim = n_heads * head_dim
    const int k = 896; // d_model

    // Generate random input
    auto A_data = generateRandomFP32(m * k);
    auto W_data = generateRandomFP32(n * k);

    // Compute FP32 reference
    auto expected = computeFP32Reference(A_data, W_data, m, n, k);

    // Create tensors
    auto A = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)}, -1);
    auto W = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, -1);
    auto C = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)}, -1);

    std::memcpy(A->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    std::memcpy(W->mutable_data(), W_data.data(), W_data.size() * sizeof(float));

    // Execute typed op
    GemmOpTyped<ActivationPrecision::FP32> gemm;
    ASSERT_TRUE(gemm.execute(A.get(), W.get(), C.get(), m, n, k));

    // Verify output
    const float *C_data = C->data();
    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        float error = std::abs(C_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Qwen Dims] Max error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-3f); // Larger matrices have more accumulation error
}

/**
 * @brief Test explicit dimensions with pre-allocated buffers
 *
 * This test verifies the bug fix for MPI tensor-parallel attention segfault.
 * The bug was: multiply_tensor() inferred m,n,k from tensor shapes, but in
 * Llaminar, activation buffers are pre-allocated for max_seq_len. When
 * processing just 1 token, tensor.shape()[0] = 512 but actual m = 1.
 *
 * The kernel would try to write 512 * n floats into an output buffer sized
 * for just 1 * n floats, causing heap-buffer-overflow.
 *
 * This test simulates that scenario:
 * - Create large input tensor (shape [max_seq, k] but only first row has data)
 * - Create correctly-sized output tensor (shape [actual_m, n])
 * - Call GEMM with explicit m = actual_m (< tensor shape)
 * - Verify no crash and correct output
 */
TEST_F(Test__GemmOpTyped, FP32_ExplicitDimensions_PreallocatedBuffer)
{
    // Simulate pre-allocated buffer scenario
    const int max_seq_len = 512; // Pre-allocated buffer size
    const int actual_m = 1;      // Actual tokens to process (single token decode)
    const int n = 448;           // Output features (sharded n_local = 896/2)
    const int k = 896;           // Input features (d_model)

    // Generate random data for the input row we'll actually use
    auto A_data_full = generateRandomFP32(max_seq_len * k);
    auto W_data = generateRandomFP32(n * k);

    // Compute reference for just the first row
    std::vector<float> A_first_row(A_data_full.begin(), A_data_full.begin() + k);
    auto expected = computeFP32Reference(A_first_row, W_data, actual_m, n, k);

    // Create tensors:
    // A: Large pre-allocated buffer [max_seq_len, k] - simulates activation buffer
    // W: Weight tensor [n, k]
    // C: Correctly-sized output [actual_m, n] - this is what project_row_parallel creates
    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(k)}, -1);
    auto W = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, -1);
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(actual_m), static_cast<size_t>(n)}, -1);

    std::memcpy(A->mutable_data(), A_data_full.data(), A_data_full.size() * sizeof(float));
    std::memcpy(W->mutable_data(), W_data.data(), W_data.size() * sizeof(float));

    // Execute typed op with EXPLICIT dimensions
    // This is the key: m=actual_m (1), NOT A->shape()[0] (512)
    GemmOpTyped<ActivationPrecision::FP32> gemm;
    ASSERT_TRUE(gemm.execute(A.get(), W.get(), C.get(), actual_m, n, k))
        << "GEMM with explicit dimensions should succeed";

    // Verify output
    const float *C_data = C->data();
    float max_error = 0.0f;
    for (int i = 0; i < actual_m * n; ++i)
    {
        float error = std::abs(C_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Explicit Dims] Max error: " << max_error
              << " (input shape [" << max_seq_len << "," << k << "], actual m=" << actual_m << ")" << std::endl;
    EXPECT_LT(max_error, 1e-4f);
}

/**
 * @brief Test explicit dimensions with quantized weights (Q4_0)
 *
 * Same scenario as above but with real quantized model weights.
 * This is the actual code path that was crashing in MPI tensor-parallel mode.
 */
TEST_F(Test__GemmOpTyped, Q4_0_ExplicitDimensions_PreallocatedBuffer)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    // Simulate pre-allocated buffer scenario
    const int max_seq_len = 512; // Pre-allocated buffer size
    const int actual_m = 1;      // Single token decode
    const int n = 896;           // Output features (d_model, full Wo projection)
    const int k = 896;           // Input features

    // Load real Wo weight
    auto wo = model_ctx_->loader().loadTensor("blk.0.attn_output.weight", -1);
    if (!wo)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_output.weight";
    }

    // Generate random input data
    auto A_data_full = generateRandomFP32(max_seq_len * k);

    // Create tensors
    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len), static_cast<size_t>(k)}, -1);
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(actual_m), static_cast<size_t>(n)}, -1);

    std::memcpy(A->mutable_data(), A_data_full.data(), A_data_full.size() * sizeof(float));

    // Execute with explicit dimensions
    GemmOpTyped<ActivationPrecision::FP32> gemm;
    ASSERT_TRUE(gemm.execute(A.get(), wo.get(), C.get(), actual_m, n, k))
        << "Q4_0 GEMM with explicit dimensions should succeed";

    // Basic sanity check: output should have non-zero values
    const float *C_data = C->data();
    float sum = 0.0f;
    for (int i = 0; i < actual_m * n; ++i)
    {
        sum += std::abs(C_data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should have non-zero values";

    std::cout << "[Q4_0 Explicit Dims] Output sum: " << sum
              << " (input shape [" << max_seq_len << "," << k << "], actual m=" << actual_m << ")" << std::endl;
}

// =============================================================================
// Q8_1 GEMM Tests (using real model weights)
// =============================================================================

/**
 * @brief Test Q8_1 GEMM with real model weights - Q projection
 *
 * Uses layer 0 Wq from Qwen 2.5 0.5B Q4_0.
 * Compares Q8_1 output vs FP32 reference (dequant→FP32 GEMM).
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_QProjection)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_heads = static_cast<int>(metadata.head_count);       // 14
    const int d_model = static_cast<int>(metadata.embedding_length); // 896
    const int head_dim = d_model / n_heads;                          // 64
    const int q_dim = n_heads * head_dim;                            // 896

    // Test with typical sequence length
    const int seq_len = 9;

    // Load Wq weight directly from ModelLoader (doesn't release raw data)
    // This allows us to dequantize to FP32 for reference
    auto wq_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_q.weight", -1);
    if (!wq_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_q.weight";
    }

    // Also load via getWeight which creates the packed GEMM kernel
    auto wq = model_ctx_->getWeight("blk.0.attn_q.weight", -1);
    ASSERT_NE(wq, nullptr);

    std::cout << "[Q8_1 Q Projection] Wq shape: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]" << std::endl;
    std::cout << "[Q8_1 Q Projection] Wq type: " << static_cast<int>(wq->native_type()) << std::endl;

    // Generate random input activations (simulating RMSNorm output)
    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // ========================================================================
    // FP32 Reference: Dequant weight → FP32 GEMM
    // ========================================================================
    std::vector<float> W_fp32(q_dim * d_model);
    wq_for_fp32->to_fp32(W_fp32.data()); // Use unpacked tensor for dequant

    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, q_dim, d_model);

    // ========================================================================
    // Q8_1 Path: Q8_1 input → Q4_0 weight → Q8_1 output
    // ========================================================================

    // Create Q8_1 input tensor
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    ASSERT_NE(input_q8, nullptr);

    // Create Q8_1 output tensor
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, -1);

    // Execute typed op (uses cached GEMM kernel from getWeight)
    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), wq.get(), output_q8.get(),
                             seq_len, q_dim, d_model));

    // Extract Q8_1 output (dequantizes)
    auto actual = extractQ8_1Data(output_q8.get(), seq_len * q_dim);

    // Compute metrics
    float cosine = cosineSimilarity(actual, fp32_expected);
    float l2_error = relativeL2Error(actual, fp32_expected) * 100.0f;

    std::cout << "[Q8_1 Q Projection] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 Q Projection] Relative L2 error: " << std::fixed << std::setprecision(6) << l2_error << "%" << std::endl;

    // Q8_1 output has quantization noise but should maintain high cosine
    EXPECT_GT(cosine, 0.98f);
    EXPECT_LT(l2_error, 10.0f);
}

/**
 * @brief Test Q8_1 GEMM with real model weights - K projection (GQA)
 *
 * Uses layer 0 Wk from Qwen 2.5 0.5B Q4_0.
 * Tests the smaller KV head dimensions (2 heads × 64 = 128).
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_KProjection)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_kv_heads = static_cast<int>(metadata.head_count_kv); // 2
    const int d_model = static_cast<int>(metadata.embedding_length); // 896
    const int head_dim = 64;                                         // Fixed for Qwen
    const int kv_dim = n_kv_heads * head_dim;                        // 128

    const int seq_len = 9;

    // Load Wk weight directly from ModelLoader (doesn't release raw data)
    auto wk_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_k.weight", -1);
    if (!wk_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_k.weight";
    }
    // Also load via getWeight for GEMM kernel
    auto wk = model_ctx_->getWeight("blk.0.attn_k.weight", -1);
    ASSERT_NE(wk, nullptr);

    std::cout << "[Q8_1 K Projection] Wk shape: [" << wk->shape()[0] << ", " << wk->shape()[1] << "]" << std::endl;

    // Generate random input
    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // FP32 Reference (use unpacked tensor for dequant)
    std::vector<float> W_fp32(kv_dim * d_model);
    wk_for_fp32->to_fp32(W_fp32.data());
    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, kv_dim, d_model);

    // Q8_1 Path
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);

    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), wk.get(), output_q8.get(),
                             seq_len, kv_dim, d_model));

    auto actual = extractQ8_1Data(output_q8.get(), seq_len * kv_dim);

    float cosine = cosineSimilarity(actual, fp32_expected);
    float l2_error = relativeL2Error(actual, fp32_expected) * 100.0f;

    std::cout << "[Q8_1 K Projection] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 K Projection] Relative L2 error: " << std::fixed << std::setprecision(6) << l2_error << "%" << std::endl;

    EXPECT_GT(cosine, 0.98f);
    EXPECT_LT(l2_error, 10.0f);
}

/**
 * @brief Test Q8_1 GEMM with real model weights - V projection
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_VProjection)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_kv_heads = static_cast<int>(metadata.head_count_kv);
    const int d_model = static_cast<int>(metadata.embedding_length);
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    const int seq_len = 9;

    auto wv_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_v.weight", -1);
    if (!wv_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_v.weight";
    }
    auto wv = model_ctx_->getWeight("blk.0.attn_v.weight", -1);
    ASSERT_NE(wv, nullptr);

    std::cout << "[Q8_1 V Projection] Wv shape: [" << wv->shape()[0] << ", " << wv->shape()[1] << "]" << std::endl;

    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // FP32 Reference (use unpacked tensor for dequant)
    std::vector<float> W_fp32(kv_dim * d_model);
    wv_for_fp32->to_fp32(W_fp32.data());
    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, kv_dim, d_model);

    // Q8_1 Path
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);

    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), wv.get(), output_q8.get(),
                             seq_len, kv_dim, d_model));

    auto actual = extractQ8_1Data(output_q8.get(), seq_len * kv_dim);

    float cosine = cosineSimilarity(actual, fp32_expected);
    float l2_error = relativeL2Error(actual, fp32_expected) * 100.0f;

    std::cout << "[Q8_1 V Projection] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 V Projection] Relative L2 error: " << std::fixed << std::setprecision(6) << l2_error << "%" << std::endl;

    EXPECT_GT(cosine, 0.98f);
    EXPECT_LT(l2_error, 10.0f);
}

/**
 * @brief Test single-token decode scenario (m=1)
 */
TEST_F(Test__GemmOpTyped, Q8_1_SingleTokenDecode)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_heads = static_cast<int>(metadata.head_count);
    const int d_model = static_cast<int>(metadata.embedding_length);
    const int head_dim = d_model / n_heads;
    const int q_dim = n_heads * head_dim;

    const int seq_len = 1; // Single token

    auto wq_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_q.weight", -1);
    if (!wq_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_q.weight";
    }
    auto wq = model_ctx_->getWeight("blk.0.attn_q.weight", -1);
    ASSERT_NE(wq, nullptr);

    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // FP32 Reference (use unpacked tensor for dequant)
    std::vector<float> W_fp32(q_dim * d_model);
    wq_for_fp32->to_fp32(W_fp32.data());
    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, q_dim, d_model);

    // Q8_1 Path
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, -1);

    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), wq.get(), output_q8.get(),
                             seq_len, q_dim, d_model));

    auto actual = extractQ8_1Data(output_q8.get(), seq_len * q_dim);

    float cosine = cosineSimilarity(actual, fp32_expected);

    std::cout << "[Q8_1 Single Token] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.98f);
}

// =============================================================================
// Full QKV Projection Test
// =============================================================================

/**
 * @brief Test full QKV projection pattern with real weights
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_FullQKV)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_heads = static_cast<int>(metadata.head_count);
    const int n_kv_heads = static_cast<int>(metadata.head_count_kv);
    const int d_model = static_cast<int>(metadata.embedding_length);
    const int head_dim = d_model / n_heads;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    const int seq_len = 9;

    // Load QKV weights directly from ModelLoader (doesn't release raw data)
    auto wq_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_q.weight", -1);
    auto wk_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_k.weight", -1);
    auto wv_for_fp32 = model_ctx_->loader().loadTensor("blk.0.attn_v.weight", -1);
    if (!wq_for_fp32 || !wk_for_fp32 || !wv_for_fp32)
    {
        GTEST_SKIP() << "Failed to load QKV weights";
    }

    // Also load via getWeight for GEMM kernel
    auto wq = model_ctx_->getWeight("blk.0.attn_q.weight", -1);
    auto wk = model_ctx_->getWeight("blk.0.attn_k.weight", -1);
    auto wv = model_ctx_->getWeight("blk.0.attn_v.weight", -1);
    ASSERT_NE(wq, nullptr);
    ASSERT_NE(wk, nullptr);
    ASSERT_NE(wv, nullptr);

    // Generate shared input (simulating RMSNorm output)
    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // Create shared Q8_1 input
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    // Compute FP32 references (use unpacked tensors for dequant)
    std::vector<float> wq_fp32(q_dim * d_model), wk_fp32(kv_dim * d_model), wv_fp32(kv_dim * d_model);
    wq_for_fp32->to_fp32(wq_fp32.data());
    wk_for_fp32->to_fp32(wk_fp32.data());
    wv_for_fp32->to_fp32(wv_fp32.data());

    auto q_expected = computeFP32Reference(input_data, wq_fp32, seq_len, q_dim, d_model);
    auto k_expected = computeFP32Reference(input_data, wk_fp32, seq_len, kv_dim, d_model);
    auto v_expected = computeFP32Reference(input_data, wv_fp32, seq_len, kv_dim, d_model);

    // Create Q8_1 outputs
    auto Q_out = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, -1);
    auto K_out = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);
    auto V_out = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);

    // Execute QKV projections
    GemmOpTyped<ActivationPrecision::Q8_1> gemm;

    ASSERT_TRUE(gemm.execute(input_q8.get(), wq.get(), Q_out.get(), seq_len, q_dim, d_model));
    ASSERT_TRUE(gemm.execute(input_q8.get(), wk.get(), K_out.get(), seq_len, kv_dim, d_model));
    ASSERT_TRUE(gemm.execute(input_q8.get(), wv.get(), V_out.get(), seq_len, kv_dim, d_model));

    // Extract and compare
    auto q_actual = extractQ8_1Data(Q_out.get(), seq_len * q_dim);
    auto k_actual = extractQ8_1Data(K_out.get(), seq_len * kv_dim);
    auto v_actual = extractQ8_1Data(V_out.get(), seq_len * kv_dim);

    float q_cosine = cosineSimilarity(q_actual, q_expected);
    float k_cosine = cosineSimilarity(k_actual, k_expected);
    float v_cosine = cosineSimilarity(v_actual, v_expected);

    std::cout << "[Full QKV]" << std::endl;
    std::cout << "  Q cosine: " << std::fixed << std::setprecision(6) << q_cosine << std::endl;
    std::cout << "  K cosine: " << std::fixed << std::setprecision(6) << k_cosine << std::endl;
    std::cout << "  V cosine: " << std::fixed << std::setprecision(6) << v_cosine << std::endl;

    EXPECT_GT(q_cosine, 0.98f);
    EXPECT_GT(k_cosine, 0.98f);
    EXPECT_GT(v_cosine, 0.98f);
}

// =============================================================================
// FFN Projection Tests
// =============================================================================

/**
 * @brief Test FFN gate/up projections (larger dimensions)
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_FFN_GateUp)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int d_model = static_cast<int>(metadata.embedding_length); // 896
    // d_ff isn't directly exposed in GGUFModel struct, hardcode for Qwen2.5-0.5B
    const int d_ff = 4864;

    const int seq_len = 9;

    // Load FFN gate weight directly from ModelLoader (doesn't release raw data)
    auto w_gate_for_fp32 = model_ctx_->loader().loadTensor("blk.0.ffn_gate.weight", -1);
    if (!w_gate_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.ffn_gate.weight";
    }
    auto w_gate = model_ctx_->getWeight("blk.0.ffn_gate.weight", -1);
    ASSERT_NE(w_gate, nullptr);

    std::cout << "[Q8_1 FFN Gate] w_gate shape: [" << w_gate->shape()[0] << ", " << w_gate->shape()[1] << "]" << std::endl;

    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // FP32 Reference (use unpacked tensor for dequant)
    std::vector<float> W_fp32(d_ff * d_model);
    w_gate_for_fp32->to_fp32(W_fp32.data());
    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, d_ff, d_model);

    // Q8_1 Path
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_ff)}, -1);

    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), w_gate.get(), output_q8.get(),
                             seq_len, d_ff, d_model));

    auto actual = extractQ8_1Data(output_q8.get(), seq_len * d_ff);

    float cosine = cosineSimilarity(actual, fp32_expected);
    float l2_error = relativeL2Error(actual, fp32_expected) * 100.0f;

    std::cout << "[Q8_1 FFN Gate] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 FFN Gate] Relative L2 error: " << std::fixed << std::setprecision(6) << l2_error << "%" << std::endl;

    EXPECT_GT(cosine, 0.98f);
    EXPECT_LT(l2_error, 10.0f);
}

/**
 * @brief Test FFN down projection (d_ff → d_model)
 */
TEST_F(Test__GemmOpTyped, Q8_1_RealWeights_FFN_Down)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int d_model = static_cast<int>(metadata.embedding_length); // 896
    // d_ff isn't directly exposed in GGUFModel struct, hardcode for Qwen2.5-0.5B
    const int d_ff = 4864;

    const int seq_len = 9;

    // Load FFN down weight directly from ModelLoader (doesn't release raw data)
    auto w_down_for_fp32 = model_ctx_->loader().loadTensor("blk.0.ffn_down.weight", -1);
    if (!w_down_for_fp32)
    {
        GTEST_SKIP() << "Failed to load blk.0.ffn_down.weight";
    }
    auto w_down = model_ctx_->getWeight("blk.0.ffn_down.weight", -1);
    ASSERT_NE(w_down, nullptr);

    std::cout << "[Q8_1 FFN Down] w_down shape: [" << w_down->shape()[0] << ", " << w_down->shape()[1] << "]" << std::endl;

    // Input is d_ff (output of SwiGLU)
    auto input_data = generateRandomFP32(seq_len * d_ff, 1.0f);

    // FP32 Reference (use unpacked tensor for dequant)
    std::vector<float> W_fp32(d_model * d_ff);
    w_down_for_fp32->to_fp32(W_fp32.data());
    auto fp32_expected = computeFP32Reference(input_data, W_fp32, seq_len, d_model, d_ff);

    // Q8_1 Path
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                   {static_cast<size_t>(seq_len), static_cast<size_t>(d_ff)});
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)}, -1);

    GemmOpTyped<ActivationPrecision::Q8_1> gemm;
    ASSERT_TRUE(gemm.execute(input_q8.get(), w_down.get(), output_q8.get(),
                             seq_len, d_model, d_ff));

    auto actual = extractQ8_1Data(output_q8.get(), seq_len * d_model);

    float cosine = cosineSimilarity(actual, fp32_expected);
    float l2_error = relativeL2Error(actual, fp32_expected) * 100.0f;

    std::cout << "[Q8_1 FFN Down] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 FFN Down] Relative L2 error: " << std::fixed << std::setprecision(6) << l2_error << "%" << std::endl;

    EXPECT_GT(cosine, 0.98f);
    EXPECT_LT(l2_error, 10.0f);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

/**
 * @brief Test factory creates ops for all precisions
 */
TEST_F(Test__GemmOpTyped, Factory_CreateAllPrecisions)
{
    auto fp32_op = createGemmOp(ActivationPrecision::FP32);
    ASSERT_NE(fp32_op, nullptr);
    EXPECT_EQ(fp32_op->precision(), ActivationPrecision::FP32);

    auto bf16_op = createGemmOp(ActivationPrecision::BF16);
    ASSERT_NE(bf16_op, nullptr);
    EXPECT_EQ(bf16_op->precision(), ActivationPrecision::BF16);

    auto fp16_op = createGemmOp(ActivationPrecision::FP16);
    ASSERT_NE(fp16_op, nullptr);
    EXPECT_EQ(fp16_op->precision(), ActivationPrecision::FP16);

    auto q8_1_op = createGemmOp(ActivationPrecision::Q8_1);
    ASSERT_NE(q8_1_op, nullptr);
    EXPECT_EQ(q8_1_op->precision(), ActivationPrecision::Q8_1);
}

/**
 * @brief Test factory-created Q8_1 op produces same results as direct instantiation
 */
TEST_F(Test__GemmOpTyped, Factory_Q8_1_Consistency)
{
    if (!model_ctx_)
    {
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;
    }

    const auto &metadata = model_ctx_->model();
    const int n_heads = static_cast<int>(metadata.head_count);
    const int d_model = static_cast<int>(metadata.embedding_length);
    const int head_dim = d_model / n_heads;
    const int q_dim = n_heads * head_dim;

    const int seq_len = 4;

    auto wq = model_ctx_->getWeight("blk.0.attn_q.weight", -1);
    if (!wq)
    {
        GTEST_SKIP() << "Failed to load weight";
    }

    auto input_data = generateRandomFP32(seq_len * d_model, 1.0f);

    // Create two Q8_1 inputs and outputs
    auto input1 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                 {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto input2 = Q8_1Tensor::quantize_from_fp32(input_data.data(),
                                                 {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    auto output1 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, -1);
    auto output2 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, -1);

    // Execute with direct instantiation
    GemmOpTyped<ActivationPrecision::Q8_1> direct_op;
    ASSERT_TRUE(direct_op.execute(input1.get(), wq.get(), output1.get(), seq_len, q_dim, d_model));

    // Execute with factory-created op
    auto factory_op = createGemmOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(factory_op->execute(input2.get(), wq.get(), output2.get(), seq_len, q_dim, d_model));

    // Extract outputs
    auto out1 = extractQ8_1Data(output1.get(), seq_len * q_dim);
    auto out2 = extractQ8_1Data(output2.get(), seq_len * q_dim);

    // Should be identical
    float cosine = cosineSimilarity(out1, out2);
    std::cout << "[Factory Consistency] Direct vs Factory cosine: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.9999f);
}

// =============================================================================
// =============================================================================
// NOTE: Legacy GemmOp tests with Q8_1 outputs are NOT valid.
// The legacy GemmOp writes FP32 to output->mutable_data(), which Q8_1 does not
// support (by design - Q8_1 is a bandwidth optimization format).
// Use GemmOpTyped<ActivationPrecision::Q8_1> for Q8_1 workflows.
// =============================================================================
