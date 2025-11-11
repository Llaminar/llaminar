/**
 * @file Test__INT8GemmImpl.cpp
 * @brief Unit tests for multi-ISA INT8 GEMM implementations
 *
 * Tests verify correctness by comparing:
 * - AVX512-VNNI vs Scalar (ground truth)
 * - AVX2 vs Scalar (ground truth)
 * - AVX512-VNNI vs AVX2 (consistency)
 *
 * All implementations should produce identical INT32 results.
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include "kernels/cpu/gemm/int8/INT8GemmImpl.h"
#include "utils/Logger.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <cstring>

using namespace llaminar2::kernels::gemm::int8_impl;

/**
 * @brief Test fixture for INT8 GEMM implementation tests
 */
class INT8GemmImplTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random number generator with fixed seed for reproducibility
        rng.seed(42);
    }

    /**
     * @brief Generate random int8 matrix
     *
     * Now supports full signed range [-128, 127] with proper signed×signed correction.
     * AVX512-VNNI and AVX2 apply offset method: A_u = A + 128, then C = C_raw - 128*sum(B).
     */
    std::vector<int8_t> generate_random_int8(size_t rows, size_t cols, int8_t min_val = -127, int8_t max_val = 127)
    {
        std::vector<int8_t> matrix(rows * cols);
        std::uniform_int_distribution<int> dist(min_val, max_val);

        for (size_t i = 0; i < rows * cols; ++i)
        {
            matrix[i] = static_cast<int8_t>(dist(rng));
        }

        return matrix;
    }

    /**
     * @brief Generate simple pattern int8 matrix for debugging
     */
    std::vector<int8_t> generate_pattern_int8(size_t rows, size_t cols)
    {
        std::vector<int8_t> matrix(rows * cols);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                matrix[i * cols + j] = static_cast<int8_t>((i * 7 + j * 3) % 128);
            }
        }

        return matrix;
    }

    /**
     * @brief Compare two INT32 matrices for equality
     *
     * @return Pair of (max_abs_diff, num_mismatches)
     */
    std::pair<int32_t, size_t> compare_int32_matrices(
        const int32_t *C1,
        const int32_t *C2,
        size_t m, size_t n,
        size_t ldc1, size_t ldc2)
    {
        int32_t max_abs_diff = 0;
        size_t num_mismatches = 0;

        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                int32_t v1 = C1[i * ldc1 + j];
                int32_t v2 = C2[i * ldc2 + j];
                int32_t diff = std::abs(v1 - v2);

                if (diff > 0)
                {
                    num_mismatches++;
                    max_abs_diff = std::max(max_abs_diff, diff);
                }
            }
        }

        return {max_abs_diff, num_mismatches};
    }

    std::mt19937 rng;
};

// ============================================================================
// Test 1: Scalar Implementation (Baseline)
// ============================================================================

/**
 * @brief Test scalar implementation with known matrices
 */
TEST_F(INT8GemmImplTest, ScalarKnownMatrices)
{
    // A = [1, 2, 3]    B^T = [4, 5, 6]    (B = [4]
    //     [4, 5, 6]          [7, 8, 9]           [5]
    //                                            [6]
    //                                            [7]
    //                                            [8]
    //                                            [9])
    //
    // C = A * B^T = [1*4+2*5+3*6,  1*7+2*8+3*9]   = [32,  50]
    //               [4*4+5*5+6*6,  4*7+5*8+6*9]     [77, 122]

    std::vector<int8_t> A = {1, 2, 3, 4, 5, 6};
    std::vector<int8_t> B = {4, 5, 6, 7, 8, 9};
    std::vector<int32_t> C(4, 0); // 2×2 matrix

    gemm_scalar_impl(A.data(), B.data(), C.data(), 2, 2, 3, 2);

    EXPECT_EQ(C[0], 32);  // C[0,0]
    EXPECT_EQ(C[1], 50);  // C[0,1]
    EXPECT_EQ(C[2], 77);  // C[1,0]
    EXPECT_EQ(C[3], 122); // C[1,1]

    LOG_INFO("✓ Scalar implementation produces correct results for known matrices");
}

/**
 * @brief Test scalar with negative values
 */
TEST_F(INT8GemmImplTest, ScalarNegativeValues)
{
    // Test with negative values to verify signed×signed multiplication
    // A = [-1, 2, -3]    B^T = [4, -5, 6]
    // C = A * B^T = [-1*4 + 2*-5 + -3*6] = [-4 - 10 - 18] = [-32]

    std::vector<int8_t> A = {-1, 2, -3};
    std::vector<int8_t> B = {4, -5, 6};
    std::vector<int32_t> C(1, 0); // 1×1 matrix

    gemm_scalar_impl(A.data(), B.data(), C.data(), 1, 1, 3, 1);

    EXPECT_EQ(C[0], -32);

    LOG_INFO("✓ Scalar implementation handles negative values correctly");
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

TEST_F(INT8GemmImplTest, AVX512VNNI_NegativeValues)
{
    // Same test as scalar, but with AVX512-VNNI
    std::vector<int8_t> A = {-1, 2, -3};
    std::vector<int8_t> B = {4, -5, 6};
    std::vector<int32_t> C(1, 0);

    gemm_avx512vnni_impl(A.data(), B.data(), C.data(), 1, 1, 3, 1);

    EXPECT_EQ(C[0], -32) << "AVX512-VNNI should handle negative values correctly with offset correction";

    LOG_INFO("✓ AVX512-VNNI handles negative values correctly");
}

#endif

#if defined(__AVX2__)

TEST_F(INT8GemmImplTest, AVX2_NegativeValues)
{
    // Same test as scalar, but with AVX2
    std::vector<int8_t> A = {-1, 2, -3};
    std::vector<int8_t> B = {4, -5, 6};
    std::vector<int32_t> C(1, 0);

    gemm_avx2_impl(A.data(), B.data(), C.data(), 1, 1, 3, 1);

    EXPECT_EQ(C[0], -32) << "AVX2 should handle negative values correctly with offset correction";

    LOG_INFO("✓ AVX2 handles negative values correctly");
}

#endif

// ============================================================================
// Test 2: Scalar vs AVX512-VNNI Correctness
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

TEST_F(INT8GemmImplTest, AVX512VNNIvsScalar_SmallMatrix)
{
    // Test 4×4 matrix (64 int8s, exactly 1 AVX512 vector load)
    constexpr int m = 4, n = 4, k = 64;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx512(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx512vnni_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx512.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "AVX512-VNNI and Scalar should produce identical results";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX512-VNNI matches Scalar for 4×4×64 matrix");
}

TEST_F(INT8GemmImplTest, AVX512VNNIvsScalar_LargeMatrix)
{
    // Test 16×16 matrix with k=128 (2 AVX512 vector loads)
    constexpr int m = 16, n = 16, k = 128;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx512(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx512vnni_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx512.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0);
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX512-VNNI matches Scalar for 16×16×128 matrix");
}

TEST_F(INT8GemmImplTest, AVX512VNNIvsScalar_NonAlignedK)
{
    // Test non-aligned k (not multiple of 64) - tests scalar tail handling
    constexpr int m = 8, n = 8, k = 100; // k=100 → 64 vectorized + 36 scalar

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx512(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx512vnni_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx512.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "AVX512-VNNI tail handling should match Scalar";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX512-VNNI matches Scalar for non-aligned k=100");
}

TEST_F(INT8GemmImplTest, AVX512VNNIvsScalar_AlphaBeta)
{
    // Test alpha/beta scaling
    constexpr int m = 4, n = 4, k = 64;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx512(m * n, 10); // Initial values
    std::vector<int32_t> C_scalar(m * n, 10);
    std::vector<int32_t> C_avx512_copy = C_avx512;
    std::vector<int32_t> C_scalar_copy = C_scalar;

    int32_t alpha = 2, beta = 3;

    gemm_avx512vnni_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n, alpha, beta);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, alpha, beta);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx512.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "Alpha/beta scaling should match";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX512-VNNI matches Scalar with alpha=2, beta=3");
}

#endif // AVX512VNNI

// ============================================================================
// Test 3: Scalar vs AVX2 Correctness
// ============================================================================

#if defined(__AVX2__)

TEST_F(INT8GemmImplTest, AVX2vsScalar_SmallMatrix)
{
    // Test 4×4 matrix (32 int8s, exactly 1 AVX2 vector load)
    constexpr int m = 4, n = 4, k = 32;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx2(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx2_impl(A.data(), B.data(), C_avx2.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx2.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "AVX2 and Scalar should produce identical results";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX2 matches Scalar for 4×4×32 matrix");
}

TEST_F(INT8GemmImplTest, AVX2vsScalar_LargeMatrix)
{
    // Test 16×16 matrix with k=96 (3 AVX2 vector loads)
    constexpr int m = 16, n = 16, k = 96;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx2(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx2_impl(A.data(), B.data(), C_avx2.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx2.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0);
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX2 matches Scalar for 16×16×96 matrix");
}

TEST_F(INT8GemmImplTest, AVX2vsScalar_NonAlignedK)
{
    // Test non-aligned k (not multiple of 32) - tests scalar tail handling
    constexpr int m = 8, n = 8, k = 50; // k=50 → 32 vectorized + 18 scalar

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx2(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_avx2_impl(A.data(), B.data(), C_avx2.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx2.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "AVX2 tail handling should match Scalar";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX2 matches Scalar for non-aligned k=50");
}

#endif // AVX2

// ============================================================================
// Test 4: AVX512-VNNI vs AVX2 Consistency
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX2__)

TEST_F(INT8GemmImplTest, AVX512VNNIvsAVX2_Consistency)
{
    // Both ISAs should produce identical results
    constexpr int m = 16, n = 16, k = 128;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_avx512(m * n, 0);
    std::vector<int32_t> C_avx2(m * n, 0);

    gemm_avx512vnni_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n);
    gemm_avx2_impl(A.data(), B.data(), C_avx2.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_avx512.data(), C_avx2.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "AVX512-VNNI and AVX2 should produce identical results";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ AVX512-VNNI matches AVX2 for 16×16×128 matrix");
}

#endif // AVX512VNNI && AVX2

// ============================================================================
// Test 5: Auto-Dispatch Mechanism
// ============================================================================

TEST_F(INT8GemmImplTest, AutoDispatch)
{
    constexpr int m = 8, n = 8, k = 64;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_auto(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    const char *isa_used = gemm_int8_auto(A.data(), B.data(), C_auto.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_auto.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0) << "Auto-dispatch should match Scalar";
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ Auto-dispatch selected: " << isa_used);
}

// ============================================================================
// Test 6: Edge Cases
// ============================================================================

TEST_F(INT8GemmImplTest, EdgeCase_1x1Matrix)
{
    // Minimal matrix: 1×1×k
    constexpr int m = 1, n = 1, k = 100;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_auto(1, 0);
    std::vector<int32_t> C_scalar(1, 0);

    gemm_int8_auto(A.data(), B.data(), C_auto.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    EXPECT_EQ(C_auto[0], C_scalar[0]);

    LOG_INFO("✓ Edge case: 1×1 matrix handled correctly");
}

TEST_F(INT8GemmImplTest, EdgeCase_VeryLargeK)
{
    // Test large k (multiple vector loads)
    constexpr int m = 4, n = 4, k = 1024;

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_auto(m * n, 0);
    std::vector<int32_t> C_scalar(m * n, 0);

    gemm_int8_auto(A.data(), B.data(), C_auto.data(), m, n, k, n);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_auto.data(), C_scalar.data(), m, n, n, n);

    EXPECT_EQ(max_diff, 0);
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ Edge case: Large k=1024 handled correctly");
}

TEST_F(INT8GemmImplTest, EdgeCase_AllZeros)
{
    // Test with zero matrices
    constexpr int m = 8, n = 8, k = 64;

    std::vector<int8_t> A(m * k, 0);
    std::vector<int8_t> B(n * k, 0);

    std::vector<int32_t> C(m * n, 999); // Non-zero initial values

    gemm_int8_auto(A.data(), B.data(), C.data(), m, n, k, n, 1, 0);

    for (int32_t val : C)
    {
        EXPECT_EQ(val, 0) << "Zero matrices should produce zero result";
    }

    LOG_INFO("✓ Edge case: All-zero matrices produce zero result");
}

// ============================================================================
// Test 7: Strided Access (ldc != n)
// ============================================================================

TEST_F(INT8GemmImplTest, StridedAccess)
{
    // Test non-contiguous C matrix (ldc > n)
    constexpr int m = 4, n = 4, k = 64;
    constexpr int ldc = 8; // Stride larger than n

    auto A = generate_random_int8(m, k);
    auto B = generate_random_int8(n, k);

    std::vector<int32_t> C_auto(m * ldc, 0);
    std::vector<int32_t> C_scalar(m * ldc, 0);

    gemm_int8_auto(A.data(), B.data(), C_auto.data(), m, n, k, ldc);
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, ldc);

    auto [max_diff, mismatches] = compare_int32_matrices(
        C_auto.data(), C_scalar.data(), m, n, ldc, ldc);

    EXPECT_EQ(max_diff, 0);
    EXPECT_EQ(mismatches, 0);

    LOG_INFO("✓ Strided access (ldc=" << ldc << ") handled correctly");
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
