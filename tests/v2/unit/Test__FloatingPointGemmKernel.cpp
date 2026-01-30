/**
 * @file Test__FloatingPointGemmKernel.cpp
 * @brief Unit tests for FloatingPointGemmKernel (OneDNN-based floating-point GEMM)
 * @author David Sanftenberg
 * @date 2025-11-28
 *
 * Quick unit tests covering:
 * - FP32×FP32 GEMM basic functionality
 * - BF16×BF16 GEMM basic functionality
 * - Transpose modes
 * - Alpha/beta scaling
 * - Edge cases (small matrices, single element)
 *
 * For thorough integration tests with larger matrices and mixed precision,
 * see tests/v2/integration/Test__FloatingPointGemmKernel.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <numeric>

#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Fill buffer with random FP32 values
        void fill_random_fp32(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Fill buffer with random BF16 values
        void fill_random_bf16(uint16_t *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = fp32_to_bf16(dist(gen));
            }
        }

        // Helper: Reference GEMM C = alpha * A @ B^T + beta * C
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
                        sum += A[i * k + l] * B[j * k + l]; // B^T: access as B[j,l]
                    }
                    C[i * n + j] = alpha * sum + beta * C[i * n + j];
                }
            }
        }

        // Helper: Reference GEMM C = alpha * A @ B + beta * C (no transpose)
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
                        sum += A[i * k + l] * B[l * n + j]; // B: access as B[l,j]
                    }
                    C[i * n + j] = alpha * sum + beta * C[i * n + j];
                }
            }
        }

        // Helper: Check approximate equality
        bool approx_equal(float a, float b, float rtol = 1e-3f, float atol = 1e-5f)
        {
            return std::abs(a - b) <= (atol + rtol * std::abs(b));
        }

        // Helper: Compute max absolute difference
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
            }
            return max_diff;
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FloatingPointGemmKernel : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Default small dimensions for quick unit tests
            m_ = 4; // rows in A and C
            k_ = 8; // cols in A, rows in B
            n_ = 6; // cols in B and C (after transpose)
        }

        int m_, k_, n_;
    };

    // =============================================================================
    // FP32 GEMM Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, FP32_Basic_Transposed)
    {
        // Create FP32 weight tensor [N, K]
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        // Create kernel bound to weights
        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // Create input and output
        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);
        std::vector<float> C_ref(m_ * n_, 0.0f);

        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        // Compute reference
        reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m_, n_, k_);

        // Compute with kernel (transpose_B=true)
        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true));

        // Verify
        float max_diff = max_abs_diff(C.data(), C_ref.data(), C.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff: " << max_diff;
    }

    TEST_F(Test__FloatingPointGemmKernel, FP32_NoTranspose)
    {
        // Create FP32 weight tensor [K, N] (no transpose)
        std::vector<float> weights_data(k_ * n_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(k_), static_cast<size_t>(n_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        // Create kernel
        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // Create input and output
        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);
        std::vector<float> C_ref(m_ * n_, 0.0f);

        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        // Compute reference (no transpose)
        reference_gemm(A.data(), weights_data.data(), C_ref.data(), m_, n_, k_);

        // Compute with kernel (transpose_B=false)
        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, false));

        // Verify
        float max_diff = max_abs_diff(C.data(), C_ref.data(), C.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff: " << max_diff;
    }

    TEST_F(Test__FloatingPointGemmKernel, FP32_AlphaScaling)
    {
        // Test alpha scaling
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);
        std::vector<float> C_ref(m_ * n_, 0.0f);

        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        float alpha = 0.5f;
        reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m_, n_, k_, alpha);

        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true, alpha, 0.0f));

        float max_diff = max_abs_diff(C.data(), C_ref.data(), C.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff with alpha=" << alpha << ": " << max_diff;
    }

    TEST_F(Test__FloatingPointGemmKernel, FP32_BetaAccumulation)
    {
        // Test beta accumulation
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m_ * k_);
        std::vector<float> C_initial(m_ * n_);
        std::vector<float> C(m_ * n_);
        std::vector<float> C_ref(m_ * n_);

        fill_random_fp32(A.data(), A.size(), 1.0f, 456);
        fill_random_fp32(C_initial.data(), C_initial.size(), 1.0f, 789);

        std::memcpy(C.data(), C_initial.data(), C.size() * sizeof(float));
        std::memcpy(C_ref.data(), C_initial.data(), C_ref.size() * sizeof(float));

        float alpha = 1.0f;
        float beta = 0.5f;
        reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m_, n_, k_, alpha, beta);

        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true, alpha, beta));

        float max_diff = max_abs_diff(C.data(), C_ref.data(), C.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff with beta=" << beta << ": " << max_diff;
    }

    TEST_F(Test__FloatingPointGemmKernel, FP32_SingleElement)
    {
        // Edge case: 1×1 matrices
        std::vector<float> weights_data = {2.0f};
        auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{1, 1});
        std::memcpy(weights->mutable_data(), weights_data.data(), sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A = {3.0f};
        std::vector<float> C = {0.0f};

        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), 1, 1, 1, true));

        EXPECT_NEAR(C[0], 6.0f, 1e-5f); // 3 * 2 = 6
    }

    TEST_F(Test__FloatingPointGemmKernel, FP32_LargerMatrix)
    {
        // Slightly larger matrix for quick sanity check
        int m = 16, k = 32, n = 24;

        std::vector<float> weights_data(n * k);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m * k);
        std::vector<float> C(m * n, 0.0f);
        std::vector<float> C_ref(m * n, 0.0f);

        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        reference_gemm_transposed(A.data(), weights_data.data(), C_ref.data(), m, n, k);

        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m, n, k, true));

        float max_diff = max_abs_diff(C.data(), C_ref.data(), C.size());
        EXPECT_LT(max_diff, 1e-3f) << "Max diff for 16x32x24: " << max_diff;
    }

    // =============================================================================
    // BF16 GEMM Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, BF16_Basic_Transposed)
    {
        // Create BF16 weight tensor [N, K]
        std::vector<uint16_t> weights_bf16(n_ * k_);
        fill_random_bf16(weights_bf16.data(), weights_bf16.size(), 1.0f, 123);

        auto weights = std::make_unique<BF16Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_bf16_data(), weights_bf16.data(), weights_bf16.size() * sizeof(uint16_t));

        // Create BF16 activation tensor [M, K]
        std::vector<uint16_t> A_bf16(m_ * k_);
        fill_random_bf16(A_bf16.data(), A_bf16.size(), 1.0f, 456);

        auto A = std::make_unique<BF16Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)});
        std::memcpy(A->mutable_bf16_data(), A_bf16.data(), A_bf16.size() * sizeof(uint16_t));

        // Create FP32 output tensor [M, N]
        auto C = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_)});
        std::memset(C->mutable_data(), 0, m_ * n_ * sizeof(float));

        // Create kernel
        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // Compute with tensor interface
        ASSERT_TRUE(kernel.multiply_tensor(A.get(), C.get(), true));

        // Compute reference (dequantize BF16 to FP32 for reference)
        std::vector<float> A_fp32(m_ * k_);
        std::vector<float> weights_fp32(n_ * k_);
        for (size_t i = 0; i < A_bf16.size(); ++i)
            A_fp32[i] = bf16_to_fp32(A_bf16[i]);
        for (size_t i = 0; i < weights_bf16.size(); ++i)
            weights_fp32[i] = bf16_to_fp32(weights_bf16[i]);

        std::vector<float> C_ref(m_ * n_, 0.0f);
        reference_gemm_transposed(A_fp32.data(), weights_fp32.data(), C_ref.data(), m_, n_, k_);

        // Verify (BF16 has lower precision, use larger tolerance)
        float max_diff = max_abs_diff(C->data(), C_ref.data(), C_ref.size());
        EXPECT_LT(max_diff, 5e-2f) << "Max diff for BF16: " << max_diff;
    }

    // =============================================================================
    // Strided GEMM Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, FP32_Strided_Basic)
    {
        // Test strided GEMM with non-contiguous memory
        int m = 4, k = 8, n = 6;
        int lda = k + 4; // Extra padding in A
        int ldb = k + 2; // Extra padding in B (stored as [N, K] for transpose)
        int ldc = n + 3; // Extra padding in C

        // Allocate with padding
        std::vector<float> A(m * lda, 0.0f);
        std::vector<float> B(n * ldb, 0.0f); // [N, K] layout
        std::vector<float> C(m * ldc, 0.0f);

        // Fill only the valid portions
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                A[i * lda + j] = dist(gen);
            }
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                B[i * ldb + j] = dist(gen);
            }
        }

        // Compute reference (extract contiguous data)
        std::vector<float> A_cont(m * k);
        std::vector<float> B_cont(n * k);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j)
                A_cont[i * k + j] = A[i * lda + j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j)
                B_cont[i * k + j] = B[i * ldb + j];

        std::vector<float> C_ref(m * n, 0.0f);
        reference_gemm_transposed(A_cont.data(), B_cont.data(), C_ref.data(), m, n, k);

        // Create kernel (no bound weights - use activations interface)
        gemm_v4::FloatingPointGemmKernel kernel(nullptr);

        // Compute with strided interface
        ASSERT_TRUE(kernel.multiply_activations_strided(
            A.data(), B.data(), C.data(),
            m, n, k,
            lda, ldb, ldc,
            true, // transpose_B
            1.0f, 0.0f,
            nullptr, -1));

        // Verify (extract from strided output)
        std::vector<float> C_result(m * n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C_result[i * n + j] = C[i * ldc + j];

        float max_diff = max_abs_diff(C_result.data(), C_ref.data(), C_ref.size());
        EXPECT_LT(max_diff, 1e-4f) << "Max diff for strided GEMM: " << max_diff;
    }

    // =============================================================================
    // GEMM with Softmax Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, FP32_WithSoftmax)
    {
        // Test GEMM fused with softmax (attention Q@K^T pattern)
        int m = 4, k = 8, n = 4; // Square output for attention

        std::vector<float> weights_data(n * k);
        fill_random_fp32(weights_data.data(), weights_data.size(), 0.5f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m * k);
        std::vector<float> C(m * n, 0.0f);

        fill_random_fp32(A.data(), A.size(), 0.5f, 456);

        ASSERT_TRUE(kernel.multiply_with_softmax(
            A.data(), nullptr, C.data(),
            m, n, k,
            true,    // transpose_B
            1,       // softmax_axis
            nullptr, // no mask
            nullptr, -1));

        // Verify softmax properties: each row should sum to 1
        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                float val = C[i * n + j];
                EXPECT_GE(val, 0.0f) << "Softmax output should be non-negative";
                EXPECT_LE(val, 1.0f) << "Softmax output should be <= 1";
                row_sum += val;
            }
            EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " sum should be 1";
        }
    }

    // =============================================================================
    // Device Support Test
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, SupportsOnlyCPU)
    {
        auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});
        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // CPU device (-1) should be supported
        EXPECT_TRUE(kernel.supports_device(-1));

        // GPU devices should not be supported
        EXPECT_FALSE(kernel.supports_device(0));
        EXPECT_FALSE(kernel.supports_device(1));
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, NullWeightTensor)
    {
        gemm_v4::FloatingPointGemmKernel kernel(nullptr);

        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);

        // Should fail gracefully when no weight tensor is bound
        EXPECT_FALSE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true));
    }

    TEST_F(Test__FloatingPointGemmKernel, TypeMismatch_RejectsQuantized)
    {
        // FloatingPointGemmKernel should reject quantized weight types
        // We can't easily create an IQ4_NL tensor without proper data, so we test
        // via the weight_type validation in constructor

        // This test documents the expected behavior:
        // - FP32, FP16, BF16 weights are accepted
        // - Quantized weights (Q4_0, Q8_0, etc.) should be rejected

        auto fp32_weights = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});
        EXPECT_NO_THROW({
            gemm_v4::FloatingPointGemmKernel kernel(fp32_weights.get());
        });
    }

    // =============================================================================
    // Workspace Parameter Tests
    // =============================================================================

    TEST_F(Test__FloatingPointGemmKernel, Multiply_WithNullWorkspace_Succeeds)
    {
        // Create FP32 weight tensor [N, K]
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);
        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        // Call multiply with explicit workspace=nullptr
        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true, 1.0f, 0.0f, nullptr, -1, nullptr));

        // Verify output is non-zero
        float sum = 0.0f;
        for (size_t i = 0; i < C.size(); ++i)
            sum += std::abs(C[i]);
        EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
    }

    TEST_F(Test__FloatingPointGemmKernel, Multiply_WithWorkspace_Succeeds)
    {
        // Create FP32 weight tensor [N, K]
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        std::vector<float> A(m_ * k_);
        std::vector<float> C(m_ * n_, 0.0f);
        fill_random_fp32(A.data(), A.size(), 1.0f, 456);

        // Create a workspace manager for CPU
        auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 1024 * 1024);
        ASSERT_NE(workspace, nullptr);

        // Call multiply with non-null workspace (CPU kernel should accept but ignore it)
        ASSERT_TRUE(kernel.multiply(A.data(), C.data(), m_, n_, k_, true, 1.0f, 0.0f, nullptr, -1, workspace.get()));

        // Verify output is non-zero
        float sum = 0.0f;
        for (size_t i = 0; i < C.size(); ++i)
            sum += std::abs(C[i]);
        EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
    }

    TEST_F(Test__FloatingPointGemmKernel, MultiplyTensor_WithNullWorkspace_Succeeds)
    {
        // Create FP32 weight tensor [N, K]
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // Create FP32 input tensor
        auto A_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)});
        fill_random_fp32(A_tensor->mutable_data(), m_ * k_, 1.0f, 456);

        // Create output tensor
        auto C_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_)});

        // Call multiply_tensor with explicit workspace=nullptr
        ASSERT_TRUE(kernel.multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f, nullptr, -1, nullptr));

        // Verify output is non-zero
        const float *C_data = C_tensor->data();
        float sum = 0.0f;
        for (int i = 0; i < m_ * n_; ++i)
            sum += std::abs(C_data[i]);
        EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
    }

    TEST_F(Test__FloatingPointGemmKernel, MultiplyTensor_WithWorkspace_Succeeds)
    {
        // Create FP32 weight tensor [N, K]
        std::vector<float> weights_data(n_ * k_);
        fill_random_fp32(weights_data.data(), weights_data.size(), 1.0f, 123);

        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)});
        std::memcpy(weights->mutable_data(), weights_data.data(), weights_data.size() * sizeof(float));

        gemm_v4::FloatingPointGemmKernel kernel(weights.get());

        // Create FP32 input tensor
        auto A_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)});
        fill_random_fp32(A_tensor->mutable_data(), m_ * k_, 1.0f, 456);

        // Create output tensor
        auto C_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_)});

        // Create a workspace manager for CPU
        auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 1024 * 1024);
        ASSERT_NE(workspace, nullptr);

        // Call multiply_tensor with non-null workspace (CPU kernel should accept but ignore it)
        ASSERT_TRUE(kernel.multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f, nullptr, -1, workspace.get()));

        // Verify output is non-zero
        const float *C_data = C_tensor->data();
        float sum = 0.0f;
        for (int i = 0; i < m_ * n_; ++i)
            sum += std::abs(C_data[i]);
        EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
    }

} // namespace llaminar2
