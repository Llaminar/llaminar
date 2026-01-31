/**
 * @file Test__FloatingPointGemmKernel.cpp
 * @brief Unit tests for FloatingPointGemmKernel (OneDNN-based FP32/BF16/FP16 GEMM)
 *
 * Tests cover:
 * - Basic GEMM correctness (C = A @ B^T)
 * - Alpha scaling (C = alpha * A @ B^T)
 * - Beta accumulation (C = alpha * A @ B^T + beta * C)
 * - Strided GEMM for multi-head attention
 * - Fused GEMM + Softmax
 * - Causal masking
 * - Mask addition
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{
    /**
     * @brief Generate random FP32 data
     */
    std::vector<float> generate_random_fp32(size_t count, float min_val = -1.0f, float max_val = 1.0f, int seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        std::vector<float> data(count);
        for (auto &x : data)
            x = dist(gen);
        return data;
    }

    /**
     * @brief Compute reference GEMM: C = alpha * A @ B^T + beta * C
     *
     * @param A Input matrix [M, K]
     * @param B Weight matrix [N, K] (stored row-major, transposed in computation)
     * @param C Output matrix [M, N]
     * @param M Rows of A and C
     * @param N Rows of B (columns of B^T) and columns of C
     * @param K Columns of A and B
     * @param alpha Scaling factor for A @ B^T
     * @param beta Scaling factor for existing C
     */
    void reference_gemm(const float *A, const float *B, float *C,
                        int M, int N, int K,
                        float alpha = 1.0f, float beta = 0.0f)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    // A[m, k] @ B[n, k]^T = A[m, k] * B[n, k]
                    sum += A[m * K + k] * B[n * K + k];
                }
                C[m * N + n] = alpha * sum + beta * C[m * N + n];
            }
        }
    }

    /**
     * @brief Compute reference strided GEMM: C = alpha * A @ B^T
     *
     * For multi-head attention where data is interleaved.
     */
    void reference_gemm_strided(const float *A, const float *B, float *C,
                                int M, int N, int K,
                                int lda, int ldb, int ldc,
                                float alpha = 1.0f)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    // A at row m, col k: A[m * lda + k]
                    // B at row n, col k: B[n * ldb + k]
                    sum += A[m * lda + k] * B[n * ldb + k];
                }
                C[m * ldc + n] = alpha * sum;
            }
        }
    }

    /**
     * @brief Compute reference softmax over rows
     */
    void reference_softmax(float *data, int rows, int cols, bool causal = false, int row_offset = 0)
    {
        for (int r = 0; r < rows; ++r)
        {
            float *row = data + r * cols;

            // Apply causal mask (set future positions to -inf)
            if (causal)
            {
                int max_visible = row_offset + r + 1;
                for (int c = max_visible; c < cols; ++c)
                {
                    row[c] = -std::numeric_limits<float>::infinity();
                }
            }

            // Find max for numerical stability
            float max_val = *std::max_element(row, row + cols);

            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                row[c] = std::exp(row[c] - max_val);
                sum += row[c];
            }

            // Normalize
            for (int c = 0; c < cols; ++c)
            {
                row[c] /= sum;
            }
        }
    }

    /**
     * @brief Compare two arrays with tolerance
     */
    bool compare_arrays(const float *actual, const float *expected, size_t count,
                        float rtol = 1e-5f, float atol = 1e-5f,
                        std::string *error_msg = nullptr)
    {
        float max_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t max_diff_idx = 0;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(actual[i] - expected[i]);
            float rel_diff = diff / (std::abs(expected[i]) + atol);

            if (diff > max_diff)
            {
                max_diff = diff;
                max_rel_diff = rel_diff;
                max_diff_idx = i;
            }

            // Check if within tolerance
            if (diff > atol + rtol * std::abs(expected[i]))
            {
                if (error_msg)
                {
                    std::ostringstream oss;
                    oss << "Mismatch at index " << i << ": "
                        << "actual=" << actual[i] << ", expected=" << expected[i]
                        << ", diff=" << diff << ", rel_diff=" << rel_diff;
                    *error_msg = oss.str();
                }
                return false;
            }
        }
        return true;
    }

} // namespace

// =============================================================================
// Basic GEMM Tests
// =============================================================================

/**
 * @brief Test basic FP32 GEMM: C = A @ B^T
 */
TEST(Test__FloatingPointGemmKernel, BasicMatMul)
{
    int M = 4;
    int N = 8;
    int K = 16;

    // Create weight tensor [N, K]
    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    // Create kernel
    FloatingPointGemmKernel kernel(weights_tensor.get());

    // Create input A [M, K]
    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    // Compute reference
    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K);

    // Compute actual
    std::vector<float> C_act(M * N, 0.0f);
    bool success = kernel.multiply(A.data(), C_act.data(), M, N, K);
    ASSERT_TRUE(success);

    // Compare
    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test GEMM with single element (M=1, N=1, K=1)
 */
TEST(Test__FloatingPointGemmKernel, SingleElement)
{
    int M = 1, N = 1, K = 1;

    std::vector<float> weights_data = {2.0f};
    auto weights_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1, 1});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    std::vector<float> A = {3.0f};
    std::vector<float> C(1, 0.0f);

    ASSERT_TRUE(kernel.multiply(A.data(), C.data(), M, N, K));
    EXPECT_NEAR(C[0], 6.0f, 1e-6f); // 3 * 2 = 6
}

/**
 * @brief Test GEMM with larger matrices
 */
TEST(Test__FloatingPointGemmKernel, LargerMatrices)
{
    int M = 32;
    int N = 64;
    int K = 128;

    auto weights_data = generate_random_fp32(N * K, -0.5f, 0.5f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -0.5f, 0.5f, 123);

    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K);

    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K));

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

// =============================================================================
// Alpha Scaling Tests
// =============================================================================

/**
 * @brief Test GEMM with alpha scaling: C = alpha * A @ B^T
 */
TEST(Test__FloatingPointGemmKernel, AlphaScaling)
{
    int M = 4;
    int N = 8;
    int K = 16;
    float alpha = 0.125f; // 1/sqrt(64) - common attention scale

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    // Reference with alpha
    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K, alpha);

    // Actual
    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K, true, alpha));

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test GEMM with alpha=0 (should produce zeros)
 */
TEST(Test__FloatingPointGemmKernel, AlphaZero)
{
    int M = 4;
    int N = 8;
    int K = 16;
    float alpha = 0.0f;

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    std::vector<float> C_act(M * N, 999.0f); // Initialize with non-zero to verify it gets zeroed
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K, true, alpha));

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], 0.0f, 1e-6f) << "Expected zero at index " << i;
    }
}

/**
 * @brief Test GEMM with negative alpha
 */
TEST(Test__FloatingPointGemmKernel, AlphaNegative)
{
    int M = 4;
    int N = 8;
    int K = 16;
    float alpha = -2.0f;

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K, alpha);

    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K, true, alpha));

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test GEMM with very small alpha (attention scale for large head_dim)
 */
TEST(Test__FloatingPointGemmKernel, AlphaSmall)
{
    int M = 4;
    int N = 8;
    int K = 64;
    float alpha = 1.0f / std::sqrt(128.0f); // ~0.0884

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K, alpha);

    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K, true, alpha));

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

// =============================================================================
// Strided GEMM Tests
// =============================================================================

/**
 * @brief Test strided GEMM for multi-head attention pattern
 *
 * Simulates Q @ K^T where Q and K are interleaved across heads.
 * Q shape: [seq_len, n_heads, head_dim] stored as [seq_len * n_heads * head_dim]
 * K shape: [seq_len, n_heads, head_dim] stored as [seq_len * n_heads * head_dim]
 *
 * For a single head h:
 *   Q_h starts at offset h * head_dim
 *   K_h starts at offset h * head_dim
 *   Stride between rows = n_heads * head_dim
 */
TEST(Test__FloatingPointGemmKernel, StridedGemm_AttentionPattern)
{
    int seq_len = 9;
    int n_heads = 14;
    int head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    int total_q_elements = seq_len * n_heads * head_dim;

    // Generate Q and K data
    auto Q_data = generate_random_fp32(total_q_elements, -1.0f, 1.0f, 42);
    auto K_data = generate_random_fp32(total_q_elements, -1.0f, 1.0f, 123);

    // Test for head 0
    int h = 0;
    const float *Q_h = Q_data.data() + h * head_dim;
    const float *K_h = K_data.data() + h * head_dim;

    int lda = n_heads * head_dim;
    int ldb = n_heads * head_dim;
    int ldc = seq_len;

    // Compute reference for this head
    std::vector<float> C_ref(seq_len * seq_len, 0.0f);
    reference_gemm_strided(Q_h, K_h, C_ref.data(), seq_len, seq_len, head_dim, lda, ldb, ldc, scale);

    // Compute actual using run_onednn_fp32_matmul_strided
    std::vector<float> C_act(seq_len * seq_len, 0.0f);
    bool success = run_onednn_fp32_matmul_strided(
        Q_h, K_h, C_act.data(),
        seq_len, seq_len, head_dim,
        lda, ldb, ldc,
        true, // transpose_B
        scale,
        0.0f);
    ASSERT_TRUE(success);

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), seq_len * seq_len, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test strided GEMM with non-square matrices
 */
TEST(Test__FloatingPointGemmKernel, StridedGemm_NonSquare)
{
    int M = 4;
    int N = 8;
    int K = 16;
    int lda = 32; // Larger than K
    int ldb = 32;
    int ldc = 16; // Larger than N

    // Allocate with strides
    std::vector<float> A(M * lda, 0.0f);
    std::vector<float> B(N * ldb, 0.0f);
    std::vector<float> C_ref(M * ldc, 0.0f);
    std::vector<float> C_act(M * ldc, 0.0f);

    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A[m * lda + k] = dist(gen);

    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            B[n * ldb + k] = dist(gen);

    float alpha = 0.5f;

    // Reference
    reference_gemm_strided(A.data(), B.data(), C_ref.data(), M, N, K, lda, ldb, ldc, alpha);

    // Actual
    bool success = run_onednn_fp32_matmul_strided(
        A.data(), B.data(), C_act.data(),
        M, N, K,
        lda, ldb, ldc,
        true, // transpose_B
        alpha,
        0.0f);
    ASSERT_TRUE(success);

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

// =============================================================================
// Fused GEMM + Softmax Tests
// =============================================================================

/**
 * @brief Test fused GEMM + Softmax (attention Q@K^T with softmax)
 */
TEST(Test__FloatingPointGemmKernel, FusedGemmSoftmax)
{
    int M = 4;
    int N = 8;
    int K = 16;
    // multiply_with_softmax interface does not support scaling (assumes alpha=1.0)
    // Use multiply_with_softmax_typed for scaling support
    float scale = 1.0f;

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    // Reference: GEMM then softmax
    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K, scale);
    reference_softmax(C_ref.data(), M, N);

    // Actual: fused GEMM + softmax
    std::vector<float> C_act(M * N, 0.0f);
    bool success = kernel.multiply_with_softmax(
        A.data(), weights_data.data(), C_act.data(),
        M, N, K,
        true, // transpose_B
        1,    // softmax_axis (row-wise)
        nullptr,
        nullptr, -1);

    // Check if multiply_with_softmax is supported
    if (!success)
    {
        // If not supported, skip this test
        GTEST_SKIP() << "multiply_with_softmax not supported by FloatingPointGemmKernel";
    }

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test fused strided GEMM + Softmax
 */
TEST(Test__FloatingPointGemmKernel, FusedStridedGemmSoftmax)
{
    int seq_len = 9;
    int n_heads = 14;
    int head_dim = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    int total_elements = seq_len * n_heads * head_dim;

    auto Q_data = generate_random_fp32(total_elements, -1.0f, 1.0f, 42);
    auto K_data = generate_random_fp32(total_elements, -1.0f, 1.0f, 123);

    // Create a dummy tensor for the kernel
    auto dummy_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1});
    FloatingPointGemmKernel kernel(dummy_tensor.get());

    int h = 0;
    const float *Q_h = Q_data.data() + h * head_dim;
    const float *K_h = K_data.data() + h * head_dim;

    int lda = n_heads * head_dim;
    int ldb = n_heads * head_dim;
    int ldc = seq_len;

    // Reference: strided GEMM then softmax
    std::vector<float> C_ref(seq_len * seq_len, 0.0f);
    reference_gemm_strided(Q_h, K_h, C_ref.data(), seq_len, seq_len, head_dim, lda, ldb, ldc, scale);
    reference_softmax(C_ref.data(), seq_len, seq_len);

    // Actual: fused strided GEMM + softmax via multiply_with_softmax_strided_typed_impl
    std::vector<float> C_act(seq_len * seq_len, 0.0f);
    bool success = kernel.multiply_with_softmax_strided_typed_impl(
        Q_h, K_h, C_act.data(),
        seq_len, seq_len, head_dim,
        lda, ldb, ldc,
        scale,
        true, // transpose_B
        1,    // softmax_axis
        nullptr,
        false, // is_causal
        nullptr, -1,
        ActivationFormat::FP32,
        ActivationFormat::FP32);
    ASSERT_TRUE(success);

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), seq_len * seq_len, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

// =============================================================================
// Causal Masking Tests
// =============================================================================

/**
 * @brief Test fused GEMM + Softmax with causal masking
 */
TEST(Test__FloatingPointGemmKernel, FusedGemmSoftmax_Causal)
{
    int seq_len = 8;
    int head_dim = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto Q = generate_random_fp32(seq_len * head_dim, -1.0f, 1.0f, 42);
    auto K = generate_random_fp32(seq_len * head_dim, -1.0f, 1.0f, 123);

    auto dummy_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1});
    FloatingPointGemmKernel kernel(dummy_tensor.get());

    // Reference: GEMM then causal softmax
    std::vector<float> C_ref(seq_len * seq_len, 0.0f);
    // Compute Q @ K^T
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < seq_len; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; ++k)
            {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            C_ref[i * seq_len + j] = scale * sum;
        }
    }
    reference_softmax(C_ref.data(), seq_len, seq_len, true);

    // Actual: fused strided GEMM + causal softmax
    std::vector<float> C_act(seq_len * seq_len, 0.0f);
    bool success = kernel.multiply_with_softmax_strided_typed_impl(
        Q.data(), K.data(), C_act.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale,
        true, // transpose_B
        1,    // softmax_axis
        nullptr,
        true, // is_causal
        nullptr, -1,
        ActivationFormat::FP32,
        ActivationFormat::FP32);
    ASSERT_TRUE(success);

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), seq_len * seq_len, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Verify causal mask produces correct attention pattern
 *
 * For causal attention, position i can only attend to positions 0..i.
 * After softmax, positions j > i should have weight 0.
 */
TEST(Test__FloatingPointGemmKernel, CausalMask_AttentionPattern)
{
    int seq_len = 5;
    int head_dim = 4;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Use simple values for easier verification
    std::vector<float> Q(seq_len * head_dim, 1.0f);
    std::vector<float> K(seq_len * head_dim, 1.0f);

    auto dummy_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1});
    FloatingPointGemmKernel kernel(dummy_tensor.get());

    std::vector<float> C(seq_len * seq_len, 0.0f);
    bool success = kernel.multiply_with_softmax_strided_typed_impl(
        Q.data(), K.data(), C.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale,
        true, // transpose_B
        1,
        nullptr,
        true, // causal
        nullptr, -1,
        ActivationFormat::FP32,
        ActivationFormat::FP32);
    ASSERT_TRUE(success);

    // Verify causal pattern: C[i,j] should be 0 for j > i
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < seq_len; ++j)
        {
            if (j > i)
            {
                EXPECT_NEAR(C[i * seq_len + j], 0.0f, 1e-6f)
                    << "Expected 0 at position [" << i << "," << j << "] due to causal masking";
            }
        }
    }

    // Verify each row sums to 1 (softmax property)
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            row_sum += C[i * seq_len + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " does not sum to 1";
    }
}

// =============================================================================
// Mask Addition Tests
// =============================================================================

/**
 * @brief Test GEMM + Softmax with additive mask
 */
TEST(Test__FloatingPointGemmKernel, FusedGemmSoftmax_WithMask)
{
    int seq_len = 6;
    int head_dim = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto Q = generate_random_fp32(seq_len * head_dim, -1.0f, 1.0f, 42);
    auto K = generate_random_fp32(seq_len * head_dim, -1.0f, 1.0f, 123);

    // Create a mask that zeros out the last 2 columns
    std::vector<float> mask(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = seq_len - 2; j < seq_len; ++j)
        {
            mask[i * seq_len + j] = -std::numeric_limits<float>::infinity();
        }
    }

    auto dummy_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1});
    FloatingPointGemmKernel kernel(dummy_tensor.get());

    // Reference: compute GEMM, add mask, then softmax
    std::vector<float> C_ref(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < seq_len; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; ++k)
            {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            C_ref[i * seq_len + j] = scale * sum + mask[i * seq_len + j];
        }
    }
    reference_softmax(C_ref.data(), seq_len, seq_len);

    // Actual
    std::vector<float> C_act(seq_len * seq_len, 0.0f);
    bool success = kernel.multiply_with_softmax_strided_typed_impl(
        Q.data(), K.data(), C_act.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale,
        true, // transpose_B
        1,
        mask.data(),
        false, // is_causal
        nullptr, -1,
        ActivationFormat::FP32,
        ActivationFormat::FP32);
    ASSERT_TRUE(success);

    // Verify masked positions are 0
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = seq_len - 2; j < seq_len; ++j)
        {
            EXPECT_NEAR(C_act[i * seq_len + j], 0.0f, 1e-6f)
                << "Expected 0 at masked position [" << i << "," << j << "]";
        }
    }

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), seq_len * seq_len, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

// =============================================================================
// Edge Cases
// =============================================================================

/**
 * @brief Test with M=1 (single query, common in decoding)
 */
TEST(Test__FloatingPointGemmKernel, SingleQuery)
{
    int M = 1;
    int N = 16;
    int K = 64;
    float alpha = 1.0f / std::sqrt(64.0f);

    auto weights_data = generate_random_fp32(N * K, -1.0f, 1.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -1.0f, 1.0f, 123);

    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K, alpha);

    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K, true, alpha));

    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-4f, 1e-4f, &error_msg)) << error_msg;
}

/**
 * @brief Test with identical Q and K (self-attention with same inputs)
 */
TEST(Test__FloatingPointGemmKernel, SelfAttention_IdenticalQK)
{
    int seq_len = 8;
    int head_dim = 16;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto QK = generate_random_fp32(seq_len * head_dim, -1.0f, 1.0f, 42);

    auto dummy_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1});
    FloatingPointGemmKernel kernel(dummy_tensor.get());

    std::vector<float> C(seq_len * seq_len, 0.0f);
    bool success = run_onednn_fp32_matmul_strided(
        QK.data(), QK.data(), C.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        true,
        scale,
        0.0f);
    ASSERT_TRUE(success);

    // The matrix should be symmetric for Q == K
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = i; j < seq_len; ++j)
        {
            EXPECT_NEAR(C[i * seq_len + j], C[j * seq_len + i], 1e-5f)
                << "Matrix not symmetric at [" << i << "," << j << "]";
        }
    }

    // Diagonal should be positive (squared norm scaled)
    for (int i = 0; i < seq_len; ++i)
    {
        EXPECT_GT(C[i * seq_len + i], 0.0f) << "Diagonal element should be positive at [" << i << "," << i << "]";
    }
}

/**
 * @brief Test numerical stability with large values
 */
TEST(Test__FloatingPointGemmKernel, NumericalStability_LargeValues)
{
    int M = 4;
    int N = 8;
    int K = 16;

    // Use larger values that could cause overflow if not handled properly
    auto weights_data = generate_random_fp32(N * K, -10.0f, 10.0f, 42);
    auto weights_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)});
    std::memcpy(weights_tensor->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

    FloatingPointGemmKernel kernel(weights_tensor.get());

    auto A = generate_random_fp32(M * K, -10.0f, 10.0f, 123);

    std::vector<float> C_ref(M * N, 0.0f);
    reference_gemm(A.data(), weights_data.data(), C_ref.data(), M, N, K);

    std::vector<float> C_act(M * N, 0.0f);
    ASSERT_TRUE(kernel.multiply(A.data(), C_act.data(), M, N, K));

    // Check for NaN/Inf
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_FALSE(std::isnan(C_act[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(C_act[i])) << "Inf at index " << i;
    }

    // Use relaxed tolerance for large values
    std::string error_msg;
    EXPECT_TRUE(compare_arrays(C_act.data(), C_ref.data(), M * N, 1e-3f, 1e-2f, &error_msg)) << error_msg;
}
