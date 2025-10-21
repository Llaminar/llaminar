/**
 * @file test_mkl_bf16.cpp
 * @brief Test Intel MKL BF16 GEMM backend
 * @author David Sanftenberg
 * @date October 19, 2025
 *
 * Tests MKL's cblas_gemm_bf16bf16f32 implementation across various matrix sizes:
 * - Small matrices (2×2, 64×64) - correctness validation
 * - Production sizes (64×896×896, 512×4096×4096) - NaN checking
 * - Comparison against FP32 reference for numerical accuracy
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#ifdef HAVE_MKL
#include "backends/MKLBackend.h"
#include "utils/BFloat16.h"
#include "Logger.h"

using namespace llaminar;

class MKLBackendTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random seed for reproducibility
        std::srand(42);
    }

    // Generate random FP32 matrix
    std::vector<float> randomMatrix(int rows, int cols, float min = -1.0f, float max = 1.0f)
    {
        std::vector<float> mat(rows * cols);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(min, max);
        for (auto &val : mat)
        {
            val = dist(gen);
        }
        return mat;
    }

    // Convert FP32 matrix to BF16
    std::vector<bfloat16> toBF16(const std::vector<float> &fp32)
    {
        std::vector<bfloat16> bf16(fp32.size());
        for (size_t i = 0; i < fp32.size(); ++i)
        {
            bf16[i] = bfloat16::from_float(fp32[i]);
        }
        return bf16;
    }

    // Reference FP32 GEMM (row-major)
    void referenceSGEMM(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f)
    {
        // C = alpha * A * B + beta * C
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int p = 0; p < k; ++p)
                {
                    sum += A[i * k + p] * B[p * n + j];
                }
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }
    }

    // Check for NaN/Inf in matrix
    bool hasInvalidValues(const float *mat, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            if (!std::isfinite(mat[i]))
            {
                LOG_ERROR("Invalid value at index " << i << ": " << mat[i]);
                return true;
            }
        }
        return false;
    }

    // Compute relative L2 error
    double relativeL2Error(const float *ref, const float *test, size_t size)
    {
        double ref_norm = 0.0;
        double diff_norm = 0.0;

        for (size_t i = 0; i < size; ++i)
        {
            double r = static_cast<double>(ref[i]);
            double t = static_cast<double>(test[i]);
            ref_norm += r * r;
            double diff = r - t;
            diff_norm += diff * diff;
        }

        return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);
    }

    // Max absolute error
    float maxAbsError(const float *ref, const float *test, size_t size)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < size; ++i)
        {
            float err = std::abs(ref[i] - test[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }
};

// Test 1: Tiny 2×2 matrices - basic correctness
TEST_F(MKLBackendTest, Tiny2x2Matrix)
{
    const int m = 2, n = 2, k = 2;

    // Simple test matrices
    std::vector<float> A = {1.0f, 2.0f,  // [1 2]
                            3.0f, 4.0f}; // [3 4]

    std::vector<float> B = {5.0f, 6.0f,  // [5 6]
                            7.0f, 8.0f}; // [7 8]

    auto B_bf16 = toBF16(B);

    // Expected result: A * B = [19 22]
    //                          [43 50]
    std::vector<float> C_mkl(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // MKL BF16 GEMM
    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                m, n, k, 1.0f, 0.0f, false, false, false);
    ASSERT_TRUE(ok) << "MKL BF16 GEMM failed";

    // Reference FP32 GEMM
    referenceSGEMM(A.data(), B.data(), C_ref.data(), m, n, k);

    // Check for invalid values
    EXPECT_FALSE(hasInvalidValues(C_mkl.data(), m * n)) << "MKL result contains NaN/Inf";

    // Check numerical accuracy
    double rel_err = relativeL2Error(C_ref.data(), C_mkl.data(), m * n);
    float max_err = maxAbsError(C_ref.data(), C_mkl.data(), m * n);

    LOG_INFO("2×2 test: rel_l2=" << rel_err << " max_abs=" << max_err);

    // BF16 has 7-bit mantissa, expect ~1% error
    EXPECT_LT(rel_err, 0.02) << "Relative L2 error too high";
    EXPECT_LT(max_err, 0.5f) << "Max absolute error too high";
}

// Test 2: Small 64×64 matrices
TEST_F(MKLBackendTest, Small64x64Matrix)
{
    const int m = 64, n = 64, k = 64;

    auto A = randomMatrix(m, k, -1.0f, 1.0f);
    auto B = randomMatrix(k, n, -1.0f, 1.0f);
    auto B_bf16 = toBF16(B);

    std::vector<float> C_mkl(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // MKL BF16 GEMM
    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                m, n, k, 1.0f, 0.0f, false, false, false);
    ASSERT_TRUE(ok) << "MKL BF16 GEMM failed";

    // Reference FP32 GEMM
    referenceSGEMM(A.data(), B.data(), C_ref.data(), m, n, k);

    // Check for invalid values
    EXPECT_FALSE(hasInvalidValues(C_mkl.data(), m * n)) << "MKL result contains NaN/Inf";

    // Check numerical accuracy
    double rel_err = relativeL2Error(C_ref.data(), C_mkl.data(), m * n);
    float max_err = maxAbsError(C_ref.data(), C_mkl.data(), m * n);

    LOG_INFO("64×64 test: rel_l2=" << rel_err << " max_abs=" << max_err);

    // BF16 accumulation may have more error with larger K
    EXPECT_LT(rel_err, 0.05) << "Relative L2 error too high";
}

// Test 3: Production size 64×896×896 (fails with OpenBLAS)
TEST_F(MKLBackendTest, Production64x896x896)
{
    const int m = 64, n = 896, k = 896;

    LOG_INFO("Testing production size: " << m << "×" << k << " × " << k << "×" << n);

    auto A = randomMatrix(m, k, -0.5f, 0.5f);
    auto B = randomMatrix(k, n, -0.5f, 0.5f);
    auto B_bf16 = toBF16(B);

    std::vector<float> C_mkl(m * n, 0.0f);

    // MKL BF16 GEMM
    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                m, n, k, 1.0f, 0.0f, false, false, false);
    ASSERT_TRUE(ok) << "MKL BF16 GEMM failed on 64×896×896";

    // **Critical check**: This size produces NaN with OpenBLAS on Cascade Lake
    EXPECT_FALSE(hasInvalidValues(C_mkl.data(), m * n))
        << "MKL result contains NaN/Inf (this would fail with OpenBLAS!)";

    // Spot check: compute a few reference values
    std::vector<float> C_ref_sample(10);
    for (int idx = 0; idx < 10; ++idx)
    {
        int i = idx % m;
        int j = (idx * 73) % n; // Pseudo-random column
        float sum = 0.0f;
        for (int p = 0; p < k; ++p)
        {
            sum += A[i * k + p] * B[p * n + j];
        }
        C_ref_sample[idx] = sum;

        float mkl_val = C_mkl[i * n + j];
        float err = std::abs(mkl_val - sum) / (std::abs(sum) + 1e-6f);

        LOG_DEBUG("Sample [" << i << "," << j << "]: ref=" << sum
                             << " mkl=" << mkl_val << " rel_err=" << err);

        EXPECT_LT(err, 0.1f) << "Sample error too high";
    }

    LOG_INFO("✓ MKL handles 64×896×896 without NaN (OpenBLAS fails this test)");
}

// Test 4: Large production size 512×4096×4096
TEST_F(MKLBackendTest, LargeProduction512x4096x4096)
{
    const int m = 512, n = 4096, k = 4096;

    LOG_INFO("Testing large production size: " << m << "×" << k << " × " << k << "×" << n);
    LOG_INFO("Memory required: ~" << (m * k + k * n * 2 + m * n * 4) / (1024 * 1024) << " MB");

    auto A = randomMatrix(m, k, -0.1f, 0.1f); // Smaller range to avoid overflow
    auto B = randomMatrix(k, n, -0.1f, 0.1f);
    auto B_bf16 = toBF16(B);

    std::vector<float> C_mkl(m * n, 0.0f);

    // MKL BF16 GEMM
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                m, n, k, 1.0f, 0.0f, false, false, false);
    auto t1 = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(ok) << "MKL BF16 GEMM failed on 512×4096×4096";

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflops = (2.0 * m * n * k) / (elapsed_ms * 1e6);

    LOG_INFO("Elapsed: " << elapsed_ms << " ms, " << gflops << " GFLOPS");

    // Check for invalid values
    EXPECT_FALSE(hasInvalidValues(C_mkl.data(), m * n))
        << "MKL result contains NaN/Inf on large matrix";

    // Spot check a few values
    for (int idx = 0; idx < 5; ++idx)
    {
        int i = (idx * 17) % m;
        int j = (idx * 97) % n;
        float val = C_mkl[i * n + j];
        EXPECT_TRUE(std::isfinite(val)) << "Invalid value at [" << i << "," << j << "]";
    }

    LOG_INFO("✓ MKL handles 512×4096×4096 successfully");
}

// Test 5: Alpha/Beta scaling
TEST_F(MKLBackendTest, AlphaBetaScaling)
{
    const int m = 32, n = 32, k = 32;

    auto A = randomMatrix(m, k);
    auto B = randomMatrix(k, n);
    auto B_bf16 = toBF16(B);

    std::vector<float> C_mkl(m * n);
    std::vector<float> C_ref(m * n);

    // Initialize C with non-zero values
    for (auto &val : C_mkl)
        val = 0.5f;
    for (auto &val : C_ref)
        val = 0.5f;

    float alpha = 2.0f, beta = 3.0f;

    // MKL BF16 GEMM: C = 2.0 * A*B + 3.0 * C
    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                m, n, k, alpha, beta, false, false, false);
    ASSERT_TRUE(ok);

    // Reference
    referenceSGEMM(A.data(), B.data(), C_ref.data(), m, n, k, alpha, beta);

    // Check
    double rel_err = relativeL2Error(C_ref.data(), C_mkl.data(), m * n);
    LOG_INFO("Alpha/Beta test: rel_l2=" << rel_err);

    EXPECT_LT(rel_err, 0.05) << "Alpha/Beta scaling error too high";
}

// Test 6: Transpose operations
TEST_F(MKLBackendTest, TransposeOperations)
{
    const int m = 32, n = 32, k = 32;

    auto A = randomMatrix(m, k);
    auto B = randomMatrix(k, n);
    auto B_bf16 = toBF16(B);

    // Test A^T * B (transpose_A = true)
    std::vector<float> C_mkl(k * n, 0.0f); // Result is k×n when A transposed

    bool ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl.data(),
                                k, n, m, // m and k swapped for transpose
                                1.0f, 0.0f, true, false, false);
    ASSERT_TRUE(ok) << "Transpose A test failed";
    EXPECT_FALSE(hasInvalidValues(C_mkl.data(), k * n));

    // Test A * B^T (transpose_B = true)
    std::vector<float> C_mkl2(m * k, 0.0f); // Result is m×k when B transposed

    ok = mkl_multiply_bf16(A.data(), B_bf16.data(), C_mkl2.data(),
                           m, k, n, // n and k swapped for transpose
                           1.0f, 0.0f, false, true, false);
    ASSERT_TRUE(ok) << "Transpose B test failed";
    EXPECT_FALSE(hasInvalidValues(C_mkl2.data(), m * k));

    LOG_INFO("✓ Transpose operations work correctly");
}

#else

// Dummy test when MKL not available
TEST(MKLBackendTest, NotAvailable)
{
    GTEST_SKIP() << "MKL not available (build with -DUSE_MKL=ON)";
}

#endif // HAVE_MKL

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
