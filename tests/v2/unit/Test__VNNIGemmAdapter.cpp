/**
 * @file Test__VNNIGemmAdapter.cpp
 * @brief Unit tests for VNNI GEMM adapter layer
 * @author David Sanftenberg
 *
 * Tests the vnni_gemm_adapter function which bridges the Tensor API
 * to the low-level VNNI GEMM kernel. Validates:
 * - Correctness against naive FP32 GEMM
 * - Quantization semantics (symmetric, signed int8)
 * - Proper handling of scales and bias
 * - Edge cases (zeros, small values, large values)
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v3/VNNIGemmAdapter.h"
#include "tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

using namespace llaminar2;

namespace
{
    /**
     * @brief Naive reference FP32 GEMM for validation
     * C = A * B + bias (broadcast over rows)
     */
    void naive_gemm_fp32(
        const float *A, const float *B, const float *bias,
        float *C,
        int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float acc = bias ? bias[j] : 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    acc += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = acc;
            }
        }
    }

    /**
     * @brief Compare two FP32 matrices with relative and absolute tolerance
     * @return true if all elements match within tolerance
     */
    bool compare_matrices(
        const float *A, const float *B,
        int M, int N,
        float rel_tol = 1e-3f, float abs_tol = 1e-5f,
        bool verbose = false)
    {
        bool all_match = true;
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        int mismatch_count = 0;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                const int idx = i * N + j;
                const float a = A[idx];
                const float b = B[idx];
                const float abs_diff = std::abs(a - b);
                const float max_val = std::max(std::abs(a), std::abs(b));
                const float rel_diff = (max_val > 1e-6f) ? (abs_diff / max_val) : 0.0f;

                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);

                if (abs_diff > abs_tol && rel_diff > rel_tol)
                {
                    all_match = false;
                    mismatch_count++;

                    if (verbose && mismatch_count <= 10)
                    {
                        std::cout << "Mismatch at [" << i << "," << j << "]: "
                                  << "ref=" << a << ", vnni=" << b
                                  << ", abs_diff=" << abs_diff
                                  << ", rel_diff=" << rel_diff << std::endl;
                    }
                }
            }
        }

        if (verbose || !all_match)
        {
            std::cout << "Matrix comparison: "
                      << "max_abs_diff=" << max_abs_diff
                      << ", max_rel_diff=" << max_rel_diff
                      << ", mismatches=" << mismatch_count << "/" << (M * N)
                      << std::endl;
        }

        return all_match;
    }

    /**
     * @brief Create a Q8_0 tensor from FP32 data with symmetric quantization
     */
    std::shared_ptr<Q8_0Tensor> create_q8_0_from_fp32(
        const std::vector<float> &data,
        const std::vector<size_t> &shape)
    {
        auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
        std::memcpy(fp32_tensor->mutable_data(), data.data(), data.size() * sizeof(float));

        // Use TensorBase::to_q8_0 for conversion
        const size_t block_count = (data.size() + 31) / 32; // Q8_0 has 32-element blocks
        std::vector<Q8_0Block> blocks(block_count);
        fp32_tensor->to_q8_0(blocks.data());

        // Convert blocks to raw uint8 data for Q8_0Tensor constructor
        std::vector<uint8_t> raw_data(block_count * sizeof(Q8_0Block));
        std::memcpy(raw_data.data(), blocks.data(), block_count * sizeof(Q8_0Block));

        return std::make_shared<Q8_0Tensor>(shape, raw_data);
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class VNNIGemmAdapterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    std::mt19937 rng_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(VNNIGemmAdapterTest, SmallMatrixMultiplication)
{
    // Test dimensions that are multiples of tile sizes
    constexpr int M = 32;  // 8 * 4
    constexpr int N = 64;  // 16 * 4
    constexpr int K = 128; // 32 * 4

    // Template parameters for VNNI kernel
    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 2;
    constexpr int PREFETCH_B_L1 = 0;

    // Generate random FP32 inputs
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);
    std::vector<float> bias_data(N);

    for (auto &x : A_data)
        x = dist(rng_);
    for (auto &x : B_data)
        x = dist(rng_);
    for (auto &x : bias_data)
        x = dist(rng_);

    // Create tensors
    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    // Compute reference result
    std::vector<float> C_ref(M * N);
    naive_gemm_fp32(A_data.data(), B_data.data(), bias_data.data(),
                    C_ref.data(), M, N, K);

    // Compute VNNI result
    std::vector<float> C_vnni(M * N);
    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        bias_data.data());

    // Compare results
    // Note: Quantization introduces error, so we use relaxed tolerances
    EXPECT_TRUE(compare_matrices(C_ref.data(), C_vnni.data(), M, N,
                                 0.02f,  // 2% relative tolerance
                                 0.01f,  // 0.01 absolute tolerance
                                 true)); // verbose
}

TEST_F(VNNIGemmAdapterTest, IdentityMatrix)
{
    // Test with identity-like pattern to verify correctness
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 32;

    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 1;
    constexpr int PREFETCH_B_L1 = 0;

    // A = ones, B = identity (extended), bias = zeros
    // Expected: C[i,j] = 1.0 for all i,j (since we're summing K ones)
    std::vector<float> A_data(M * K, 1.0f);
    std::vector<float> B_data(K * N, 0.0f);

    // Fill B as extended identity
    for (int k = 0; k < K && k < N; ++k)
    {
        B_data[k * N + k] = 1.0f;
    }

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    std::vector<float> C_vnni(M * N);
    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        nullptr);

    // Verify pattern: first min(K,N) columns should be ~1.0, rest ~0.0
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const float expected = (j < K) ? 1.0f : 0.0f;
            const float actual = C_vnni[i * N + j];
            EXPECT_NEAR(actual, expected, 0.05f) << "Mismatch at [" << i << "," << j << "]";
        }
    }
}

TEST_F(VNNIGemmAdapterTest, ZeroInputs)
{
    // Test with all zeros
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 32;

    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 1;
    constexpr int PREFETCH_B_L1 = 0;

    std::vector<float> A_data(M * K, 0.0f);
    std::vector<float> B_data(K * N, 0.0f);

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    std::vector<float> C_vnni(M * N, 999.0f); // Fill with sentinel value

    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        nullptr);

    // All outputs should be zero
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_vnni[i], 0.0f, 1e-5f) << "Non-zero at index " << i;
    }
}

TEST_F(VNNIGemmAdapterTest, BiasAddition)
{
    // Test that bias is correctly added
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 32;

    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 1;
    constexpr int PREFETCH_B_L1 = 0;

    // Use zeros for A and B, non-zero bias
    std::vector<float> A_data(M * K, 0.0f);
    std::vector<float> B_data(K * N, 0.0f);
    std::vector<float> bias_data(N);

    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto &x : bias_data)
        x = dist(rng_);

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    std::vector<float> C_vnni(M * N);
    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        bias_data.data());

    // Each row should equal the bias vector (since A*B=0)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            EXPECT_NEAR(C_vnni[i * N + j], bias_data[j], 1e-4f)
                << "Bias mismatch at [" << i << "," << j << "]";
        }
    }
}

// ============================================================================
// Quantization Semantics Tests
// ============================================================================

TEST_F(VNNIGemmAdapterTest, SymmetricQuantizationRange)
{
    // Test that symmetric quantization preserves sign and scales correctly
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 32;

    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 1;
    constexpr int PREFETCH_B_L1 = 0;

    // Create data with clear positive/negative patterns
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);

    for (int i = 0; i < M * K; ++i)
    {
        A_data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    for (int i = 0; i < K * N; ++i)
    {
        B_data[i] = (i % 3 == 0) ? 1.0f : ((i % 3 == 1) ? -1.0f : 0.5f);
    }

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    // Reference
    std::vector<float> C_ref(M * N);
    naive_gemm_fp32(A_data.data(), B_data.data(), nullptr, C_ref.data(), M, N, K);

    // VNNI
    std::vector<float> C_vnni(M * N);
    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        nullptr);

    // Compare with relaxed tolerance (quantization error expected)
    EXPECT_TRUE(compare_matrices(C_ref.data(), C_vnni.data(), M, N,
                                 0.05f, 0.05f, true));
}

// ============================================================================
// Larger Matrix Tests
// ============================================================================

TEST_F(VNNIGemmAdapterTest, MediumMatrixMultiplication)
{
    // Test with larger dimensions
    constexpr int M = 64;  // 8 * 8
    constexpr int N = 128; // 16 * 8
    constexpr int K = 256; // 32 * 8

    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 32;
    constexpr int UNROLL_K = 2;
    constexpr int PREFETCH_B_L1 = 64;

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);

    for (auto &x : A_data)
        x = dist(rng_);
    for (auto &x : B_data)
        x = dist(rng_);

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    std::vector<float> C_ref(M * N);
    naive_gemm_fp32(A_data.data(), B_data.data(), nullptr, C_ref.data(), M, N, K);

    std::vector<float> C_vnni(M * N);
    vnni_gemm_adapter<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1>(
        M, N, K,
        *A_fp32,
        *B_q8_0,
        C_vnni.data(), N,
        nullptr);

    EXPECT_TRUE(compare_matrices(C_ref.data(), C_vnni.data(), M, N,
                                 0.02f, 0.01f, false));
}

// ============================================================================
// Different Tile Configurations
// ============================================================================

TEST_F(VNNIGemmAdapterTest, DifferentTileConfigurations)
{
    // Test with different M_R/N_R/K_BLK settings
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 128;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);

    for (auto &x : A_data)
        x = dist(rng_);
    for (auto &x : B_data)
        x = dist(rng_);

    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_fp32->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    auto B_q8_0 = create_q8_0_from_fp32(B_data, {static_cast<size_t>(K), static_cast<size_t>(N)});

    std::vector<float> C_ref(M * N);
    naive_gemm_fp32(A_data.data(), B_data.data(), nullptr, C_ref.data(), M, N, K);

    // Configuration 1: M_R=8, N_R=16, K_BLK=32
    {
        std::vector<float> C_vnni(M * N);
        vnni_gemm_adapter<8, 16, 32, 2, 0>(
            M, N, K, *A_fp32, *B_q8_0, C_vnni.data(), N, nullptr);

        EXPECT_TRUE(compare_matrices(C_ref.data(), C_vnni.data(), M, N,
                                     0.02f, 0.01f, false))
            << "Config 1 (8x16x32) failed";
    }

    // Configuration 2: M_R=16, N_R=32, K_BLK=64
    {
        std::vector<float> C_vnni(M * N);
        vnni_gemm_adapter<16, 32, 64, 1, 0>(
            M, N, K, *A_fp32, *B_q8_0, C_vnni.data(), N, nullptr);

        EXPECT_TRUE(compare_matrices(C_ref.data(), C_vnni.data(), M, N,
                                     0.02f, 0.01f, false))
            << "Config 2 (16x32x64) failed";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
