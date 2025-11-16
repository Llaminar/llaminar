/**
 * @file Test__VNNIGemm_Focused.cpp
 * @brief Focused kernel-level tests for VNNI GEMM correctness with current API
 *
 * This file contains simplified tests that directly invoke the VNNI GEMM kernel
 * with the current API (no zero-point parameters). These tests isolate kernel
 * correctness from the adapter layer to help debug numerical issues.
 *
 * The original Test__VNNIGemm.cpp contains tests using an older API with
 * column_sums and zero-point parameters that are no longer supported.
 */

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

#include "kernels/cpu/gemm_v3/VNNIGemm.h"

namespace llaminar2
{

    class Test__VNNIGemm_Focused : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Seed RNG for reproducibility
            std::srand(42);
        }

        // Simple reference INT8 GEMM with scales (for validation)
        static void simpleReferenceGemmINT8(
            const int8_t *A, const int8_t *B, float *C,
            const float *act_scales, const float *wgt_scales,
            int M, int N, int K, int K_BLK)
        {
            const int T = K / K_BLK;
            for (int m = 0; m < M; ++m)
            {
                for (int n = 0; n < N; ++n)
                {
                    float sum = 0.0f;
                    for (int t = 0; t < T; ++t)
                    {
                        for (int kb = 0; kb < K_BLK; ++kb)
                        {
                            int k = t * K_BLK + kb;
                            float a_val = static_cast<float>(A[m * K + k]) * act_scales[t];
                            float b_val = static_cast<float>(B[k * N + n]) * wgt_scales[n];
                            sum += a_val * b_val;
                        }
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        // Pack B matrix into VNNI format (column-major, 4-way interleaved)
        static std::vector<int8_t> packBMatrixVNNI(const int8_t *B, int K, int N, int K_BLK)
        {
            const int T = K / K_BLK;
            std::vector<int8_t> B_packed((K_BLK / 4) * (4 * N) * T);

            for (int t = 0; t < T; ++t)
            {
                for (int kb = 0; kb < K_BLK / 4; ++kb)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        for (int lane = 0; lane < 4; ++lane)
                        {
                            int k = t * K_BLK + kb * 4 + lane;
                            int packed_idx = t * (K_BLK / 4) * (4 * N) + kb * (4 * N) + n * 4 + lane;
                            B_packed[packed_idx] = B[k * N + n];
                        }
                    }
                }
            }
            return B_packed;
        }
    };

    /**
     * @brief Test kernel with all-ones matrices
     *
     * A = all 1s (int8), B = all 1s (int8), scales = 1.0
     * Expected: C[m,n] = K (sum of K ones)
     */
    TEST_F(Test__VNNIGemm_Focused, AllOnesMatrix)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8;
        constexpr int N_R = 16;
        constexpr int K_BLK = 32;
        constexpr int UNROLL_K = 1;
        constexpr int PREFETCH_B_L1 = 0;
        constexpr int PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::cout << "\n=== Kernel: All Ones (" << M << "×" << N << "×" << K << ") ===" << std::endl;

        // All ones
        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);
        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        // Pack B
        auto B_packed = packBMatrixVNNI(B_unpacked.data(), K, N, K_BLK);
        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        // Expected: C[m,n] = 1 * 1 * K = K
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            M, N, K);

        // Verify all elements = K
        bool all_match = true;
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                const float expected = static_cast<float>(K);
                const float actual = C_test[m * N + n];
                const float abs_err = std::abs(expected - actual);

                if (abs_err > 0.01f)
                {
                    std::cout << "MISMATCH at [" << m << "," << n << "]: "
                              << "expected=" << expected << ", actual=" << actual
                              << ", error=" << abs_err << std::endl;
                    all_match = false;
                }
            }
        }

        EXPECT_TRUE(all_match) << "All-ones test failed";
        if (all_match)
            std::cout << "✅ All-ones test PASSED (all elements = " << K << ")" << std::endl;
    }

    /**
     * @brief Test kernel with row/column pattern
     *
     * A[m,k] = m+1 (each row has constant value)
     * B[k,n] = n+1 (each column has constant value)
     * Expected: C[m,n] = (m+1) * (n+1) * K
     */
    TEST_F(Test__VNNIGemm_Focused, RowColumnPattern)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8;
        constexpr int N_R = 16;
        constexpr int K_BLK = 32;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 0;
        constexpr int PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::cout << "\n=== Kernel: Row/Column Pattern (" << M << "×" << N << "×" << K << ") ===" << std::endl;

        // A[m,k] = m+1
        std::vector<int8_t> A(M * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K + k] = static_cast<int8_t>(m + 1);

        // B[k,n] = n+1
        std::vector<int8_t> B_unpacked(K * N);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = static_cast<int8_t>(n + 1);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        // Pack B
        auto B_packed = packBMatrixVNNI(B_unpacked.data(), K, N, K_BLK);
        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        // Reference
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(),
                                M, N, K, K_BLK);

        // VNNI kernel
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            M, N, K);

        // Verify: C[m,n] = (m+1) * (n+1) * K
        bool all_match = true;
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                const float expected = static_cast<float>((m + 1) * (n + 1) * K);
                const float actual = C_test[m * N + n];
                const float abs_err = std::abs(expected - actual);
                const float rel_err = (expected > 1e-6f) ? (abs_err / std::abs(expected)) : 0.0f;

                if (abs_err > 0.5f && rel_err > 0.01f)
                {
                    std::cout << "MISMATCH at [" << m << "," << n << "]: "
                              << "expected=" << expected << ", actual=" << actual
                              << ", abs_err=" << abs_err << ", rel_err=" << rel_err << std::endl;
                    all_match = false;
                }
            }
        }

        EXPECT_TRUE(all_match) << "Row/column pattern test failed";
        if (all_match)
        {
            std::cout << "✅ Row/column pattern test PASSED" << std::endl;
            std::cout << "   Sample: C[0,0] = " << C_test[0]
                      << " (expected " << (1 * 1 * K) << ")" << std::endl;
            std::cout << "   Sample: C[1,1] = " << C_test[N + 1]
                      << " (expected " << (2 * 2 * K) << ")" << std::endl;
            std::cout << "   Sample: C[7,15] = " << C_test[7 * N + 15]
                      << " (expected " << (8 * 16 * K) << ")" << std::endl;
        }
    }

    /**
     * @brief Test kernel with different scales
     *
     * Uses simple pattern with non-unit scales to verify scale application
     */
    TEST_F(Test__VNNIGemm_Focused, WithScales)
    {
        const int M = 8, N = 16, K = 64; // K=64 for 2 K-blocks
        constexpr int M_R = 8;
        constexpr int N_R = 16;
        constexpr int K_BLK = 32;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 0;
        constexpr int PREFETCH_B_L2 = 0;
        const int T = K / K_BLK; // T = 2

        std::cout << "\n=== Kernel: With Scales (" << M << "×" << N << "×" << K << ") ===" << std::endl;

        // Simple pattern: all values = 1 (int8)
        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        // Non-unit scales
        std::vector<float> act_scales(T);
        act_scales[0] = 2.0f; // First K-block: scale = 2.0
        act_scales[1] = 3.0f; // Second K-block: scale = 3.0

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = 0.5f * (n + 1); // wgt_scales = [0.5, 1.0, 1.5, ...]

        std::vector<float> bias(N, 0.0f);

        // Pack B
        auto B_packed = packBMatrixVNNI(B_unpacked.data(), K, N, K_BLK);
        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        // Reference
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(),
                                M, N, K, K_BLK);

        // VNNI kernel
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            M, N, K);

        // Verify against reference
        bool all_match = true;
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                const float expected = C_ref[m * N + n];
                const float actual = C_test[m * N + n];
                const float abs_err = std::abs(expected - actual);
                const float rel_err = (expected > 1e-6f) ? (abs_err / std::abs(expected)) : 0.0f;

                if (abs_err > 0.5f && rel_err > 0.01f)
                {
                    std::cout << "MISMATCH at [" << m << "," << n << "]: "
                              << "expected=" << expected << ", actual=" << actual
                              << ", abs_err=" << abs_err << ", rel_err=" << rel_err << std::endl;
                    all_match = false;
                }
            }
        }

        EXPECT_TRUE(all_match) << "Scales test failed";
        if (all_match)
        {
            std::cout << "✅ Scales test PASSED" << std::endl;
            // Expected C[0,0] = sum over k blocks:
            //   K_BLK * 1 * 1 * act_scales[0] * wgt_scales[0] +
            //   K_BLK * 1 * 1 * act_scales[1] * wgt_scales[0]
            // = 32 * 2.0 * 0.5 + 32 * 3.0 * 0.5 = 32 + 48 = 80
            float expected_00 = 32.0f * 2.0f * 0.5f + 32.0f * 3.0f * 0.5f;
            std::cout << "   Sample: C[0,0] = " << C_test[0]
                      << " (expected " << expected_00 << ")" << std::endl;
        }
    }

} // namespace llaminar2
