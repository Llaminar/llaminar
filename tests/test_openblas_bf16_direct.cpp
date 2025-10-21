/**
 * @file test_openblas_bf16_direct.cpp
 * @brief Test OpenBLAS cblas_sbgemm directly to verify NaN bug
 * @author David Sanftenberg
 * @date 2025-10-20
 *
 * This test bypasses CPU feature detection to force the OpenBLAS BF16 path
 * on Cascade Lake (no AVX512_BF16) to verify if the NaN bug actually exists.
 *
 * Expected result on Cascade Lake: NaN outputs (confirming bug)
 * Expected result on Cooper Lake+: Correct outputs (native BF16)
 */

#include <gtest/gtest.h>
#include "utils/BFloat16.h"
#include "utils/CpuFeatures.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace llaminar;

// Forward declare OpenBLAS cblas_sbgemm
extern "C"
{
    void cblas_sbgemm(
        const int Order,
        const int TransA, const int TransB,
        const int M, const int N, const int K,
        const float alpha,
        const void *A, const int lda,
        const void *B, const int ldb,
        const float beta,
        float *C, const int ldc);
}

// CBLAS constants
#define CblasRowMajor 101
#define CblasNoTrans 111

/**
 * @brief Test small matrix (2x2) - should work even without native BF16
 */
TEST(OpenBLASBF16DirectTest, SmallMatrix2x2)
{
    std::cout << "\n=== Testing OpenBLAS cblas_sbgemm: 2×2 Matrix ===" << std::endl;
    std::cout << "CPU: " << CpuFeatures::instance().summary() << std::endl;

    const int m = 2, k = 2, n = 2;

    // A = [[1, 2], [3, 4]]
    std::vector<float> A_fp32 = {1.0f, 2.0f, 3.0f, 4.0f};

    // B = [[1, 0], [2, 1]]
    std::vector<float> B_fp32 = {1.0f, 0.0f, 2.0f, 1.0f};

    // Convert to BF16
    std::vector<uint16_t> A_bf16(m * k);
    std::vector<uint16_t> B_bf16(k * n);
    for (int i = 0; i < m * k; i++)
    {
        A_bf16[i] = bfloat16::from_float(A_fp32[i]).data;
    }
    for (int i = 0; i < k * n; i++)
    {
        B_bf16[i] = bfloat16::from_float(B_fp32[i]).data;
    }

    // Output
    std::vector<float> C(m * n, 0.0f);

    // Call cblas_sbgemm directly (bypassing CPU feature check)
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f,
                 A_bf16.data(), k,
                 B_bf16.data(), n,
                 0.0f,
                 C.data(), n);

    // Expected: C = [[5, 2], [11, 4]]
    std::cout << "Output C:" << std::endl;
    std::cout << "  [" << C[0] << ", " << C[1] << "]" << std::endl;
    std::cout << "  [" << C[2] << ", " << C[3] << "]" << std::endl;

    // Check for NaN
    bool has_nan = false;
    for (const auto &val : C)
    {
        if (std::isnan(val))
        {
            has_nan = true;
            break;
        }
    }

    EXPECT_FALSE(has_nan) << "Small matrix produced NaN!";

    // Check numerical accuracy
    EXPECT_NEAR(C[0], 5.0f, 0.1f);
    EXPECT_NEAR(C[1], 2.0f, 0.1f);
    EXPECT_NEAR(C[2], 11.0f, 0.1f);
    EXPECT_NEAR(C[3], 4.0f, 0.1f);

    std::cout << "Result: " << (has_nan ? "FAIL (NaN)" : "PASS") << std::endl;
}

/**
 * @brief Test medium matrix (64×64) - check where bug starts
 */
TEST(OpenBLASBF16DirectTest, MediumMatrix64x64)
{
    std::cout << "\n=== Testing OpenBLAS cblas_sbgemm: 64×64 Matrix ===" << std::endl;
    std::cout << "CPU: " << CpuFeatures::instance().summary() << std::endl;

    const int m = 64, k = 64, n = 64;

    // Initialize with simple pattern (all 1.0)
    std::vector<float> A_fp32(m * k, 1.0f);
    std::vector<float> B_fp32(k * n, 1.0f);

    // Convert to BF16
    std::vector<uint16_t> A_bf16(m * k);
    std::vector<uint16_t> B_bf16(k * n);
    for (size_t i = 0; i < A_bf16.size(); i++)
    {
        A_bf16[i] = bfloat16::from_float(A_fp32[i]).data;
    }
    for (size_t i = 0; i < B_bf16.size(); i++)
    {
        B_bf16[i] = bfloat16::from_float(B_fp32[i]).data;
    }

    // Output
    std::vector<float> C(m * n, 0.0f);

    // Call cblas_sbgemm directly
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f,
                 A_bf16.data(), k,
                 B_bf16.data(), n,
                 0.0f,
                 C.data(), n);

    // Count NaN
    size_t nan_count = 0;
    for (const auto &val : C)
    {
        if (std::isnan(val))
            nan_count++;
    }

    std::cout << "NaN count: " << nan_count << " / " << (m * n) << " ("
              << (100.0 * nan_count / (m * n)) << "%)" << std::endl;

    // Sample first few outputs
    std::cout << "First 10 outputs: ";
    for (int i = 0; i < 10 && i < m * n; i++)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Expected: Each element should be ~64.0 (sum of 64 1.0s)
    // On Cascade Lake without native BF16: May produce NaN
    if (nan_count > 0)
    {
        std::cout << "Result: FAIL (NaN detected - OpenBLAS BF16 emulation bug)" << std::endl;
        if (!can_use_native_bf16_gemm())
        {
            std::cout << "NOTE: CPU lacks AVX512_BF16 - bug expected on Cascade Lake" << std::endl;
        }
    }
    else
    {
        std::cout << "Result: PASS (no NaN)" << std::endl;
        // Verify numerical correctness
        float expected = static_cast<float>(k); // Sum of k 1.0s
        bool numerically_correct = true;
        for (const auto &val : C)
        {
            if (std::abs(val - expected) > 1.0f)
            {
                numerically_correct = false;
                break;
            }
        }
        std::cout << "Numerical correctness: " << (numerically_correct ? "PASS" : "FAIL") << std::endl;
    }

    // On CPUs without native BF16, we expect this to fail
    if (!can_use_native_bf16_gemm())
    {
        std::cout << "WARNING: CPU lacks AVX512_BF16 - bug may or may not reproduce" << std::endl;
        // Don't fail the test if NaN is present (it's the bug we're investigating)
    }
    else
    {
        // On CPUs with native BF16, we expect no NaN
        EXPECT_EQ(nan_count, 0) << "Native BF16 CPU should not produce NaN";
    }
}

/**
 * @brief Test large matrix (64×896×896) - known to fail on Cascade Lake
 */
TEST(OpenBLASBF16DirectTest, LargeMatrix64x896x896)
{
    std::cout << "\n=== Testing OpenBLAS cblas_sbgemm: 64×896×896 Matrix ===" << std::endl;
    std::cout << "CPU: " << CpuFeatures::instance().summary() << std::endl;
    std::cout << "NOTE: This test forces OpenBLAS BF16 path, bypassing CPU feature check" << std::endl;
    std::cout << "      On Cascade Lake (no AVX512_BF16), we expect NaN outputs" << std::endl;

    const int m = 64, k = 896, n = 896;

    std::cout << "Allocating matrices: A[" << m << "×" << k << "], B[" << k << "×" << n
              << "], C[" << m << "×" << n << "]" << std::endl;

    // Initialize with simple pattern (all 1.0)
    std::vector<float> A_fp32(m * k, 1.0f);
    std::vector<float> B_fp32(k * n, 1.0f);

    std::cout << "Converting to BF16..." << std::endl;

    // Convert to BF16
    std::vector<uint16_t> A_bf16(m * k);
    std::vector<uint16_t> B_bf16(k * n);
    for (size_t i = 0; i < A_bf16.size(); i++)
    {
        A_bf16[i] = bfloat16::from_float(A_fp32[i]).data;
    }
    for (size_t i = 0; i < B_bf16.size(); i++)
    {
        B_bf16[i] = bfloat16::from_float(B_fp32[i]).data;
    }

    // Verify inputs are valid (no NaN)
    bool A_has_nan = false, B_has_nan = false;
    for (const auto &val : A_bf16)
    {
        bfloat16 bf;
        bf.data = val;
        if (std::isnan(static_cast<float>(bf)))
        {
            A_has_nan = true;
            break;
        }
    }
    for (const auto &val : B_bf16)
    {
        bfloat16 bf;
        bf.data = val;
        if (std::isnan(static_cast<float>(bf)))
        {
            B_has_nan = true;
            break;
        }
    }
    std::cout << "Input validation: A_has_nan=" << A_has_nan << ", B_has_nan=" << B_has_nan << std::endl;
    EXPECT_FALSE(A_has_nan) << "Input A should not contain NaN";
    EXPECT_FALSE(B_has_nan) << "Input B should not contain NaN";

    // Output
    std::vector<float> C(m * n, 0.0f);

    std::cout << "Calling cblas_sbgemm (directly, bypassing CPU check)..." << std::endl;

    // Call cblas_sbgemm directly (THIS IS WHERE THE BUG SHOULD OCCUR)
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f,
                 A_bf16.data(), k,
                 B_bf16.data(), n,
                 0.0f,
                 C.data(), n);

    std::cout << "cblas_sbgemm completed" << std::endl;

    // Count NaN
    size_t nan_count = 0;
    for (const auto &val : C)
    {
        if (std::isnan(val))
            nan_count++;
    }

    std::cout << "Output NaN count: " << nan_count << " / " << (m * n) << " ("
              << (100.0 * nan_count / (m * n)) << "%)" << std::endl;

    // Sample first few outputs
    std::cout << "First 10 outputs: ";
    for (int i = 0; i < 10 && i < m * n; i++)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Expected: Each element should be ~896.0 (sum of 896 1.0s)
    // On Cascade Lake without native BF16: SHOULD produce NaN (the bug)
    if (nan_count > 0)
    {
        std::cout << "Result: BUG REPRODUCED - NaN outputs detected" << std::endl;
        if (!can_use_native_bf16_gemm())
        {
            std::cout << "✓ CPU lacks AVX512_BF16 - bug confirmed in OpenBLAS emulation" << std::endl;
            std::cout << "✓ This confirms the bug exists and justifies our CPU feature check" << std::endl;
        }
        else
        {
            std::cout << "✗ CPU has AVX512_BF16 but still produces NaN - unexpected!" << std::endl;
        }
    }
    else
    {
        std::cout << "Result: NO BUG - cblas_sbgemm worked correctly" << std::endl;
        if (!can_use_native_bf16_gemm())
        {
            std::cout << "✗ CPU lacks AVX512_BF16 but didn't produce NaN - bug may be fixed?" << std::endl;
        }
        else
        {
            std::cout << "✓ CPU has AVX512_BF16 - expected behavior" << std::endl;
        }

        // Verify numerical correctness
        float expected = static_cast<float>(k); // Sum of k 1.0s
        bool numerically_correct = true;
        for (const auto &val : C)
        {
            if (std::abs(val - expected) > 10.0f)
            { // Allow some BF16 error
                numerically_correct = false;
                break;
            }
        }
        std::cout << "Numerical correctness: " << (numerically_correct ? "PASS" : "FAIL") << std::endl;
    }

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "CPU has AVX512_BF16: " << (can_use_native_bf16_gemm() ? "YES" : "NO") << std::endl;
    std::cout << "NaN detected: " << (nan_count > 0 ? "YES" : "NO") << std::endl;

    if (!can_use_native_bf16_gemm() && nan_count > 0)
    {
        std::cout << "\n✓ BUG CONFIRMED: OpenBLAS cblas_sbgemm emulation is broken on Cascade Lake" << std::endl;
        std::cout << "  This justifies our CPU feature check and automatic fallback to FP32 expansion" << std::endl;
        // Don't fail the test - we're documenting the bug, not asserting it shouldn't exist
    }
    else if (!can_use_native_bf16_gemm() && nan_count == 0)
    {
        std::cout << "\n? BUG NOT REPRODUCED: OpenBLAS may have fixed the emulation bug" << std::endl;
        std::cout << "  Consider removing CPU feature check if consistently working" << std::endl;
    }
    else if (can_use_native_bf16_gemm() && nan_count > 0)
    {
        std::cout << "\n✗ UNEXPECTED: Native BF16 CPU produced NaN - investigate!" << std::endl;
        FAIL() << "Native BF16 CPU should not produce NaN";
    }
    else
    {
        std::cout << "\n✓ EXPECTED: Native BF16 CPU working correctly" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
