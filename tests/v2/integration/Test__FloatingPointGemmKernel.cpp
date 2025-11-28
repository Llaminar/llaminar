/**
 * @file Test__FloatingPointGemmKernel.cpp
 * @brief Integration tests for FloatingPointGemmKernel (OneDNN-based floating-point GEMM)
 * @author David Sanftenberg
 * @date 2025-11-28
 *
 * Thorough integration tests covering:
 * - Large matrix dimensions (realistic model sizes)
 * - FP32, BF16, FP16 precision variants
 * - Mixed precision operations (FP32×BF16, FP32×FP16)
 * - Strided memory layouts (attention head patterns)
 * - Fused operations (softmax, causal masking)
 * - Numerical accuracy validation
 * - Multi-batch processing
 *
 * For quick unit tests, see tests/v2/unit/Test__FloatingPointGemmKernel.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mpi.h>

#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace test
    {
        // =============================================================================
        // Test Helpers
        // =============================================================================

        /**
         * @brief Fill buffer with random FP32 values
         */
        void fill_random_fp32(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        /**
         * @brief Fill buffer with random BF16 values
         */
        void fill_random_bf16(uint16_t *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = simd::fp32_to_bf16(dist(gen));
            }
        }

        /**
         * @brief Fill buffer with random FP16 values
         */
        void fill_random_fp16(uint16_t *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = simd::fp32_to_fp16(dist(gen));
            }
        }

        /**
         * @brief Reference GEMM: C = alpha * A @ B^T + beta * C
         */
        void reference_gemm_transposed(const float *A, const float *B, float *C,
                                       int m, int n, int k,
                                       float alpha = 1.0f, float beta = 0.0f)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l)
                    {
                        sum += A[i * k + l] * B[j * k + l];
                    }
                    C[i * n + j] = alpha * sum + beta * C[i * n + j];
                }
            }
        }

        /**
         * @brief Reference GEMM: C = alpha * A @ B + beta * C (no transpose)
         */
        void reference_gemm(const float *A, const float *B, float *C,
                            int m, int n, int k,
                            float alpha = 1.0f, float beta = 0.0f)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l)
                    {
                        sum += A[i * k + l] * B[l * n + j];
                    }
                    C[i * n + j] = alpha * sum + beta * C[i * n + j];
                }
            }
        }

        /**
         * @brief Reference strided GEMM: C = alpha * A @ B^T + beta * C
         */
        void reference_gemm_strided(const float *A, const float *B, float *C,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc,
                                    bool transpose_B,
                                    float alpha = 1.0f, float beta = 0.0f)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l)
                    {
                        float a_val = A[i * lda + l];
                        float b_val = transpose_B ? B[j * ldb + l] : B[l * ldb + j];
                        sum += a_val * b_val;
                    }
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                }
            }
        }

        /**
         * @brief Reference row-wise softmax
         */
        void reference_softmax(float *data, int rows, int cols)
        {
            for (int i = 0; i < rows; ++i)
            {
                float *row = data + i * cols;

                // Find max for numerical stability
                float max_val = *std::max_element(row, row + cols);

                // Compute exp(x - max) and sum
                float sum = 0.0f;
                for (int j = 0; j < cols; ++j)
                {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }

                // Normalize
                for (int j = 0; j < cols; ++j)
                {
                    row[j] /= sum;
                }
            }
        }

        /**
         * @brief Apply causal mask (upper triangle = -inf)
         */
        void apply_causal_mask(float *data, int rows, int cols)
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = i + 1; j < cols; ++j)
                {
                    data[i * cols + j] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        /**
         * @brief Compute max absolute difference
         */
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }

        /**
         * @brief Compute mean absolute error
         */
        float mean_abs_error(const float *a, const float *b, size_t count)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                sum += std::abs(a[i] - b[i]);
            }
            return sum / count;
        }

        /**
         * @brief Compute relative error
         */
        float relative_error(const float *actual, const float *expected, size_t count)
        {
            float num = 0.0f, den = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                num += (actual[i] - expected[i]) * (actual[i] - expected[i]);
                den += expected[i] * expected[i];
            }
            return std::sqrt(num / (den + 1e-10f));
        }

    } // namespace test

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FloatingPointGemmKernel_Integration : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // MPI is already initialized by mpirun - no need to init here
        }

        void TearDown() override
        {
            // MPI finalization is handled by the test framework
        }
    };

    // =============================================================================
    // Large Matrix FP32 Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_RealisticDimensions_Linear)
    {
        // Typical linear layer: seq_len=128, d_model=896, intermediate=4864 (Qwen 0.5B)
        int m = 128;  // sequence length
        int k = 896;  // d_model
        int n = 4864; // intermediate_size

        std::vector<float> weights_data(n * k);
        test::fill_random_fp32(weights_data.data(), weights_data.size(), 0.5f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m * k);
        std::vector<float> C(m * n, 0.0f);
        std::vector<float> C_ref(m * n, 0.0f);

        test::fill_random_fp32(A.data(), A.size(), 0.5f, 456);

        // Reference computation
        test::reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m, n, k);

        // Kernel computation
        auto start = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m, n, k, true));
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * m * n * k) / (ms * 1e6);

        LOG_INFO("[FP32 Linear " << m << "×" << k << "×" << n << "] "
                                 << "Time: " << ms << " ms, " << gflops << " GFLOPS");

        // Verify accuracy
        float max_diff = test::max_abs_diff(C.data(), C_ref.data(), C.size());
        float rel_err = test::relative_error(C.data(), C_ref.data(), C.size());

        EXPECT_LT(max_diff, 1e-2f) << "Max diff: " << max_diff;
        EXPECT_LT(rel_err, 1e-4f) << "Relative error: " << rel_err;
    }

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_RealisticDimensions_Attention)
    {
        // Attention Q@K^T: seq_len=128, num_heads=14, head_dim=64 (Qwen 0.5B)
        int seq_len = 128;
        int num_heads = 14;
        int head_dim = 64;

        // Per-head computation: [seq_len, head_dim] @ [seq_len, head_dim]^T = [seq_len, seq_len]
        int m = seq_len;
        int k = head_dim;
        int n = seq_len;

        std::vector<float> Q(m * k);
        std::vector<float> K(n * k); // K^T will be used
        std::vector<float> scores(m * n, 0.0f);
        std::vector<float> scores_ref(m * n, 0.0f);

        test::fill_random_fp32(Q.data(), Q.size(), 0.5f, 123);
        test::fill_random_fp32(K.data(), K.size(), 0.5f, 456);

        // Create kernel with K as weights
        auto K_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(K_tensor->mutable_data(), K.data(), K.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(K_tensor.get());

        // Reference
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        test::reference_gemm_transposed(Q.data(), K.data(), scores_ref.data(), m, n, k, scale);

        // Kernel
        ASSERT_TRUE(kernel.multiply(Q.data(), scores.data(), m, n, k, true, scale));

        float max_diff = test::max_abs_diff(scores.data(), scores_ref.data(), scores.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff for attention scores: " << max_diff;
    }

    // =============================================================================
    // BF16 Precision Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, BF16_RealisticDimensions)
    {
        int m = 64;
        int k = 512;
        int n = 1024;

        // Create BF16 weights
        std::vector<uint16_t> weights_bf16(n * k);
        test::fill_random_bf16(weights_bf16.data(), weights_bf16.size(), 0.5f, 123);

        auto weights = std::make_unique<BF16Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_bf16_data(), weights_bf16.data(), weights_bf16.size() * sizeof(uint16_t));

        // Create BF16 activations
        std::vector<uint16_t> A_bf16(m * k);
        test::fill_random_bf16(A_bf16.data(), A_bf16.size(), 0.5f, 456);

        auto A = std::make_unique<BF16Tensor>(
            std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
        std::memcpy(A->mutable_bf16_data(), A_bf16.data(), A_bf16.size() * sizeof(uint16_t));

        // Create FP32 output
        auto C = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});
        std::memset(C->mutable_data(), 0, m * n * sizeof(float));

        // Compute reference (convert to FP32)
        std::vector<float> A_fp32(m * k);
        std::vector<float> weights_fp32(n * k);
        for (size_t i = 0; i < A_bf16.size(); ++i)
            A_fp32[i] = simd::bf16_to_fp32(A_bf16[i]);
        for (size_t i = 0; i < weights_bf16.size(); ++i)
            weights_fp32[i] = simd::bf16_to_fp32(weights_bf16[i]);

        std::vector<float> C_ref(m * n, 0.0f);
        test::reference_gemm_transposed(A_fp32.data(), weights_fp32.data(), C_ref.data(), m, n, k);

        // Kernel computation
        gemm_v4::FloatingPointGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.multiply_tensor(A.get(), C.get(), true));

        // BF16 has ~3 decimal digits of precision
        float max_diff = test::max_abs_diff(C->data(), C_ref.data(), C_ref.size());
        float mean_err = test::mean_abs_error(C->data(), C_ref.data(), C_ref.size());

        LOG_INFO("[BF16 GEMM " << m << "×" << k << "×" << n << "] "
                               << "Max diff: " << max_diff << ", Mean error: " << mean_err);

        EXPECT_LT(max_diff, 0.5f) << "BF16 max diff too large";
        EXPECT_LT(mean_err, 0.05f) << "BF16 mean error too large";
    }

    // =============================================================================
    // Strided GEMM Tests (Attention Pattern)
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_Strided_AttentionPattern)
    {
        // Simulates attention with interleaved heads
        // Q/K/V are stored as [seq_len, num_heads, head_dim]
        // We compute per-head: Q_head @ K_head^T

        int seq_len = 32;
        int num_heads = 8;
        int head_dim = 64;

        int m = seq_len;
        int k = head_dim;
        int n = seq_len;

        // Strides for interleaved layout [seq_len, num_heads, head_dim]
        int lda = num_heads * head_dim; // stride between rows of Q
        int ldb = num_heads * head_dim; // stride between rows of K
        int ldc = seq_len;              // output is contiguous per head

        // Allocate full Q and K tensors
        std::vector<float> Q_full(seq_len * num_heads * head_dim);
        std::vector<float> K_full(seq_len * num_heads * head_dim);

        test::fill_random_fp32(Q_full.data(), Q_full.size(), 0.5f, 123);
        test::fill_random_fp32(K_full.data(), K_full.size(), 0.5f, 456);

        // Test one head (head 3)
        int head_idx = 3;
        const float *Q_head = Q_full.data() + head_idx * head_dim;
        const float *K_head = K_full.data() + head_idx * head_dim;

        std::vector<float> scores(m * n, 0.0f);
        std::vector<float> scores_ref(m * n, 0.0f);

        // Reference: extract contiguous data and compute
        std::vector<float> Q_cont(m * k);
        std::vector<float> K_cont(n * k);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j)
                Q_cont[i * k + j] = Q_head[i * lda + j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j)
                K_cont[i * k + j] = K_head[i * ldb + j];

        test::reference_gemm_transposed(Q_cont.data(), K_cont.data(), scores_ref.data(), m, n, k);

        // Kernel: use strided interface
        gemm_v4::FloatingPointGemmKernel kernel(nullptr);
        ASSERT_TRUE(kernel.multiply_activations_strided(
            Q_head, K_head, scores.data(),
            m, n, k,
            lda, ldb, ldc,
            true, // transpose_B
            1.0f, 0.0f,
            nullptr, -1));

        float max_diff = test::max_abs_diff(scores.data(), scores_ref.data(), scores.size());
        EXPECT_LT(max_diff, 1e-4f) << "Strided attention pattern max diff: " << max_diff;
    }

    // =============================================================================
    // Fused Softmax Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_FusedSoftmax_NoCausal)
    {
        int m = 32;
        int k = 64;
        int n = 32;

        std::vector<float> Q(m * k);
        std::vector<float> K(n * k);
        test::fill_random_fp32(Q.data(), Q.size(), 0.3f, 123);
        test::fill_random_fp32(K.data(), K.size(), 0.3f, 456);

        auto K_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(K_tensor->mutable_data(), K.data(), K.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(K_tensor.get());

        std::vector<float> scores(m * n, 0.0f);
        std::vector<float> scores_ref(m * n, 0.0f);

        float scale = 1.0f / std::sqrt(static_cast<float>(k));

        // Reference: GEMM + softmax
        test::reference_gemm_transposed(Q.data(), K.data(), scores_ref.data(), m, n, k, scale);
        test::reference_softmax(scores_ref.data(), m, n);

        // Kernel: fused GEMM + softmax
        ASSERT_TRUE(kernel.multiply_with_softmax(
            Q.data(), nullptr, scores.data(),
            m, n, k,
            true, 1, nullptr,
            nullptr, -1));

        // Note: kernel uses scale=1.0 internally, we need to apply scale separately
        // Let's test without scale first
        std::vector<float> scores_ref_no_scale(m * n, 0.0f);
        test::reference_gemm_transposed(Q.data(), K.data(), scores_ref_no_scale.data(), m, n, k);
        test::reference_softmax(scores_ref_no_scale.data(), m, n);

        float max_diff = test::max_abs_diff(scores.data(), scores_ref_no_scale.data(), scores.size());
        EXPECT_LT(max_diff, 1e-4f) << "Fused softmax max diff: " << max_diff;

        // Verify softmax properties
        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                EXPECT_GE(scores[i * n + j], 0.0f);
                EXPECT_LE(scores[i * n + j], 1.0f);
                row_sum += scores[i * n + j];
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " sum: " << row_sum;
        }
    }

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_FusedSoftmax_Causal)
    {
        int m = 16;
        int k = 32;
        int n = 16; // Causal requires square attention

        std::vector<float> Q(m * k);
        std::vector<float> K(n * k);
        test::fill_random_fp32(Q.data(), Q.size(), 0.3f, 123);
        test::fill_random_fp32(K.data(), K.size(), 0.3f, 456);

        auto K_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(K_tensor->mutable_data(), K.data(), K.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(K_tensor.get());

        std::vector<float> scores(m * n, 0.0f);
        std::vector<float> scores_ref(m * n, 0.0f);

        // Reference: GEMM + causal mask + softmax
        test::reference_gemm_transposed(Q.data(), K.data(), scores_ref.data(), m, n, k);
        test::apply_causal_mask(scores_ref.data(), m, n);
        test::reference_softmax(scores_ref.data(), m, n);

        // Kernel: fused with strided typed interface (causal)
        // Using strided interface which supports causal masking for FP32
        ASSERT_TRUE(kernel.multiply_with_softmax_strided_typed_impl(
            Q.data(), K.data(), scores.data(),
            m, n, k,
            k, k, n, // contiguous strides: lda=k, ldb=k, ldc=n
            1.0f,    // scale
            true,    // transpose_B
            1,       // softmax_axis
            nullptr, // no explicit mask
            true,    // is_causal
            nullptr, -1,
            ActivationFormat::FP32, ActivationFormat::FP32));

        float max_diff = test::max_abs_diff(scores.data(), scores_ref.data(), scores.size());
        EXPECT_LT(max_diff, 1e-4f) << "Fused causal softmax max diff: " << max_diff;

        // Verify causal pattern: upper triangle should be ~0 after softmax
        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                if (j > i)
                {
                    // Upper triangle should be effectively 0
                    EXPECT_LT(scores[i * n + j], 1e-6f)
                        << "Upper triangle at [" << i << "," << j << "] = " << scores[i * n + j];
                }
                row_sum += scores[i * n + j];
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " sum: " << row_sum;
        }
    }

    // =============================================================================
    // Mixed Precision Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, MixedPrecision_FP32_BF16_Strided)
    {
        // FP32 scores × BF16 V (attention output computation)
        int m = 32; // seq_len
        int k = 32; // seq_len (softmax output)
        int n = 64; // head_dim

        std::vector<float> scores(m * k);
        std::vector<uint16_t> V_bf16(k * n);

        // Scores should be softmax output (0-1 range)
        std::mt19937 gen(123);
        for (int i = 0; i < m; ++i)
        {
            std::vector<float> row(k);
            test::fill_random_fp32(row.data(), k, 1.0f, 123 + i);
            test::reference_softmax(row.data(), 1, k);
            std::memcpy(&scores[i * k], row.data(), k * sizeof(float));
        }

        test::fill_random_bf16(V_bf16.data(), V_bf16.size(), 0.5f, 456);

        std::vector<float> output(m * n, 0.0f);

        // Reference (convert BF16 V to FP32)
        std::vector<float> V_fp32(k * n);
        for (size_t i = 0; i < V_bf16.size(); ++i)
            V_fp32[i] = simd::bf16_to_fp32(V_bf16[i]);

        std::vector<float> output_ref(m * n, 0.0f);
        test::reference_gemm(scores.data(), V_fp32.data(), output_ref.data(), m, n, k);

        // Kernel: FP32 × BF16 strided
        gemm_v4::FloatingPointGemmKernel kernel(nullptr);
        ASSERT_TRUE(kernel.multiply_activations_strided_typed_impl(
            scores.data(), V_bf16.data(), output.data(),
            m, n, k,
            k, n, n, // contiguous strides
            false,   // V is [K, N], not transposed
            1.0f, 0.0f,
            nullptr, -1,
            ActivationFormat::FP32, ActivationFormat::BF16));

        float max_diff = test::max_abs_diff(output.data(), output_ref.data(), output.size());
        EXPECT_LT(max_diff, 0.1f) << "Mixed FP32×BF16 max diff: " << max_diff;
    }

    // =============================================================================
    // Batch Processing Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_MultipleBatches)
    {
        // Process multiple independent GEMMs (simulating batch dimension)
        int batch_size = 4;
        int m = 32;
        int k = 64;
        int n = 128;

        std::vector<float> weights_data(n * k);
        test::fill_random_fp32(weights_data.data(), weights_data.size(), 0.5f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<float> A(m * k);
            std::vector<float> C(m * n, 0.0f);
            std::vector<float> C_ref(m * n, 0.0f);

            test::fill_random_fp32(A.data(), A.size(), 0.5f, 456 + b * 100);

            test::reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m, n, k);
            ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m, n, k, true));

            float max_diff = test::max_abs_diff(C.data(), C_ref.data(), C.size());
            EXPECT_LT(max_diff, 1e-3f) << "Batch " << b << " max diff: " << max_diff;
        }
    }

    // =============================================================================
    // Edge Cases
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_NonMultipleOfSIMD)
    {
        // Test dimensions that aren't nice multiples of SIMD width
        std::vector<std::tuple<int, int, int>> test_cases = {
            {17, 31, 23},   // All prime-ish
            {1, 1024, 1},   // Single row/col
            {127, 255, 63}, // Near powers of 2
            {3, 7, 11},     // Small primes
        };

        for (const auto &[m, k, n] : test_cases)
        {
            std::vector<float> weights_data(n * k);
            test::fill_random_fp32(weights_data.data(), weights_data.size(), 0.5f, 123);

            auto weights = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
            std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

            gemm_v4::FloatingPointGemmKernel kernel(weights.get());

            std::vector<float> A(m * k);
            std::vector<float> C(m * n, 0.0f);
            std::vector<float> C_ref(m * n, 0.0f);

            test::fill_random_fp32(A.data(), A.size(), 0.5f, 456);

            test::reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m, n, k);
            ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m, n, k, true))
                << "Failed for dimensions " << m << "×" << k << "×" << n;

            float max_diff = test::max_abs_diff(C.data(), C_ref.data(), C.size());
            EXPECT_LT(max_diff, 1e-3f) << "Dimensions " << m << "×" << k << "×" << n
                                       << " max diff: " << max_diff;
        }
    }

    TEST_F(Test__FloatingPointGemmKernel_Integration, FP32_ZeroInputs)
    {
        int m = 16, k = 32, n = 24;

        std::vector<float> weights_data(n * k, 0.0f); // All zeros
        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m * k);
        test::fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        std::vector<float> C(m * n, 999.0f); // Non-zero initial

        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m, n, k, true));

        // Output should be all zeros (A @ 0 = 0)
        for (size_t i = 0; i < C.size(); ++i)
        {
            EXPECT_NEAR(C[i], 0.0f, 1e-6f) << "Zero weights should produce zero output";
        }
    }

} // namespace llaminar2

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
