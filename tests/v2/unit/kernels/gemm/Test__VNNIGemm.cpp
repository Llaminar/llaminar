/**
 * @file Test__VNNIGemm.cpp
 * @brief Comprehensive VNNI INT8 GEMM kernel tests
 * @author David Sanftenberg
 * @date 2025-11-16
 *
 * Combines minimal debugging tests and full correctness validation tests
 * for the gemm_v3 VNNI INT8 kernel.
 *
 * Test Suite:
 * 1. TinyMatrix_4x16x4: Simple diagonal pattern (debugging)
 * 2. SmallMatrix_8x16x8: Simple structured values (debugging)
 * 3. SmallMatrix_64x64x64: Random data with quantization (correctness)
 * 4. MediumMatrix_Qwen05B_Dims: Realistic model dimensions (512×896×896)
 */

#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "v2/kernels/cpu/gemm_v3/VNNIGemm.h"

// OneDNN for reference GEMM
#include "oneapi/dnnl/dnnl.hpp"

namespace llaminar2
{
    class Test__VNNIGemm : public ::testing::Test
    {
    protected:
        /**
         * @brief Simple reference GEMM (triple loop, FP32)
         */
        static void simpleReferenceGemmFP32(
            const float *A, const float *B, float *C,
            int M, int N, int K)
        {
            for (int m = 0; m < M; ++m)
            {
                for (int n = 0; n < N; ++n)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * @brief Simple reference GEMM (triple loop, INT8 with scales)
         */
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
                    for (int k = 0; k < K; ++k)
                    {
                        const int t = k / K_BLK;
                        const int8_t a_val = A[m * K + k];
                        const int8_t b_val = B[k * N + n];
                        sum += (float)a_val * (float)b_val * act_scales[t] * wgt_scales[n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * @brief Compute relative L2 error
         */
        static double computeRelativeL2Error(
            const float *C_ref, const float *C_test,
            int M, int N)
        {
            double diff_norm = 0.0, ref_norm = 0.0;
            for (int i = 0; i < M * N; ++i)
            {
                double diff = C_ref[i] - C_test[i];
                diff_norm += diff * diff;
                ref_norm += C_ref[i] * C_ref[i];
            }
            return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-10);
        }

        /**
         * @brief Compute max absolute error
         */
        static float computeMaxAbsError(
            const float *C_ref, const float *C_test,
            int M, int N)
        {
            float max_err = 0.0f;
            for (int i = 0; i < M * N; ++i)
            {
                float err = std::abs(C_ref[i] - C_test[i]);
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        /**
         * @brief Generate random FP32 matrix
         */
        static std::vector<float> generateRandomMatrix(int rows, int cols, int seed = 42)
        {
            std::vector<float> mat(rows * cols);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &val : mat)
                val = dist(rng);
            return mat;
        }

        /**
         * @brief Pack B matrix into VNNI-interleaved layout
         */
        static std::vector<int8_t> packBMatrixVNNI(
            const int8_t *B_unpacked, int K, int N, int K_BLK)
        {
            const int T = K / K_BLK;
            std::vector<int8_t> B_packed(T * (K_BLK / 4) * (4 * N));

            for (int t = 0; t < T; ++t)
            {
                for (int kk = 0; kk < K_BLK; kk += 4)
                {
                    int offset = t * (K_BLK / 4) * (4 * N) + (kk / 4) * (4 * N);

                    for (int n = 0; n < N; ++n)
                    {
                        for (int k_sub = 0; k_sub < 4; ++k_sub)
                        {
                            const int k = t * K_BLK + kk + k_sub;
                            B_packed[offset + n * 4 + k_sub] = B_unpacked[k * N + n];
                        }
                    }
                }
            }

            return B_packed;
        }

        /**
         * @brief Compute per-K-block column sums for zero-point correction
         */
        static std::vector<int32_t> computeColumnSums(
            const int8_t *B_unpacked, int K, int N, int K_BLK)
        {
            const int T = K / K_BLK;
            std::vector<int32_t> column_sums(T * N, 0);

            for (int t = 0; t < T; ++t)
            {
                for (int n = 0; n < N; ++n)
                {
                    for (int kk = 0; kk < K_BLK; ++kk)
                    {
                        const int k = t * K_BLK + kk;
                        column_sums[t * N + n] += B_unpacked[k * N + n];
                    }
                }
            }

            return column_sums;
        }
    };

    // ========================================
    // Minimal Debugging Tests
    // ========================================

    /**
     * @brief Simplest possible test: 4×16×4 with diagonal pattern
     *
     * Uses identity-like matrices to verify basic functionality.
     */
    TEST_F(Test__VNNIGemm, TinyMatrix_4x16x4)
    {
        const int M = 4, N = 16, K = 4;
        constexpr int M_R = 4;
        constexpr int N_R = 16;
        constexpr int K_BLK = 4;
        constexpr int UNROLL_K = 1;
        constexpr int PREFETCH_B_L1 = 0;
        constexpr int PREFETCH_B_L2 = 0;
        const int T = K / K_BLK; // T = 1

        std::cout << "\n=== Tiny Matrix Test: " << M << "×" << N << "×" << K << " ===" << std::endl;

        // Create simple diagonal pattern
        std::vector<int8_t> A(M * K, 0);
        std::vector<int8_t> B_unpacked(K * N, 0);

        for (int i = 0; i < std::min(M, K); ++i)
            A[i * K + i] = 1;

        for (int i = 0; i < std::min(K, N); ++i)
            B_unpacked[i * N + i] = 1;

        // Unit scales
        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);

        // Pack B and setup PackedB structure
        auto B_packed = packBMatrixVNNI(B_unpacked.data(), K, N, K_BLK);

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        std::vector<float> bias(N, 0.0f);

        // Reference computation
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
            nullptr,  // No zero-point correction for signed A
            M, N, K);

        // Compare
        bool all_match = true;
        for (int i = 0; i < M * N; ++i)
        {
            if (std::abs(C_ref[i] - C_test[i]) > 0.01f)
            {
                std::cout << "MISMATCH at [" << i << "]: ref=" << C_ref[i]
                          << ", test=" << C_test[i] << std::endl;
                all_match = false;
            }
        }

        EXPECT_TRUE(all_match) << "Tiny matrix test failed";
        if (all_match)
            std::cout << "✅ Tiny matrix test PASSED" << std::endl;
    }

    /**
     * @brief Small test with structured values: 8×16×8
     *
     * Uses simple pattern: A[i,j] = i+1, B[i,j] = j+1
     * Expected: C[m,n] = (m+1) * (n+1) * K
     */
    TEST_F(Test__VNNIGemm, SmallMatrix_8x16x8)
    {
        const int M = 8, N = 16, K = 8;
        constexpr int M_R = 8;
        constexpr int N_R = 16;
        constexpr int K_BLK = 8;
        constexpr int UNROLL_K = 1;
        constexpr int PREFETCH_B_L1 = 0;
        constexpr int PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::cout << "\n=== Small Matrix Test: " << M << "×" << N << "×" << K << " ===" << std::endl;

        // Simple structured values
        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);

        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K + k] = (m + 1);

        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = (n + 1);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);

        // Pack B
        auto B_packed = packBMatrixVNNI(B_unpacked.data(), K, N, K_BLK);

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        std::vector<float> bias(N, 0.0f);

        // Reference
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(),
                                M, N, K, K_BLK);

        // VNNI
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            nullptr,  // No zero-point correction for signed A
            M, N, K);

        // Expected: C[m,n] = (m+1) * (n+1) * K
        std::cout << "Expected C[0,0] = " << (1 * 1 * K) << ", Actual = " << C_test[0] << std::endl;
        std::cout << "Expected C[1,1] = " << (2 * 2 * K) << ", Actual = " << C_test[N + 1] << std::endl;

        // Compare
        bool all_match = true;
        for (int i = 0; i < M * N; ++i)
        {
            if (std::abs(C_ref[i] - C_test[i]) > 0.01f)
            {
                const int m = i / N;
                const int n = i % N;
                std::cout << "MISMATCH at [" << m << "," << n << "]: ref=" << C_ref[i]
                          << ", test=" << C_test[i] << std::endl;
                all_match = false;
            }
        }

        EXPECT_TRUE(all_match) << "Small matrix test failed";
        if (all_match)
            std::cout << "✅ Small matrix test PASSED" << std::endl;
    }

    // ========================================
    // Correctness Tests (Random Data + Quantization)
    // ========================================

    /**
     * @brief Basic correctness test: 64×64×64 with random data
     *
     * Validates:
     * 1. FP32 SGEMM (ground truth)
     * 2. Quantize FP32 → INT8 (A: per-K-block scale, B: per-column scale)
     * 3. VNNI INT8 GEMM with scales and zero-point correction
     * 4. Compare outputs (expect <2% relative L2 error due to INT8 quantization)
     */
    TEST_F(Test__VNNIGemm, SmallMatrix_64x64x64)
    {
        const int M = 64, N = 64, K = 64;

        constexpr int M_R = 16;
        constexpr int N_R = 64;
        constexpr int K_BLK = 64;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 64;
        constexpr int PREFETCH_B_L2 = 256;

        std::cout << "\n=== VNNI GEMM Correctness: " << M << "×" << N << "×" << K << " ===" << std::endl;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K, 42);
        auto B_fp32 = generateRandomMatrix(K, N, 43);

        // Quantize to INT8
        const int T = K / K_BLK;

        // Use random INT8 for A with unit scales
        std::mt19937 rng_int8(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A_int8(M * K);
        for (auto &x : A_int8)
            x = static_cast<int8_t>(int8_dist(rng_int8));

        std::vector<float> act_scales(T, 1.0f);

        // Recompute A_fp32 from quantized A
        for (int i = 0; i < M; ++i)
        {
            for (int k = 0; k < K; ++k)
            {
                const int t = k / K_BLK;
                A_fp32[i * K + k] = static_cast<float>(A_int8[i * K + k]) * act_scales[t];
            }
        }

        // Quantize B (per-column scale)
        std::vector<float> wgt_scales(N);
        std::vector<int8_t> B_int8_unpacked(K * N);

        for (int n = 0; n < N; ++n)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int k = 0; k < K; ++k)
            {
                float val = B_fp32[k * N + n];
                float quantized = val / wgt_scales[n];
                quantized = std::max(-127.0f, std::min(127.0f, quantized));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        // Dequantize B for reference
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K; ++k)
            {
                B_fp32[k * N + n] = static_cast<float>(B_int8_unpacked[k * N + n]) * wgt_scales[n];
            }
        }

        // FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
        std::cout << "✓ FP32 reference completed" << std::endl;

        // Pack B and compute column sums
        auto B_packed = packBMatrixVNNI(B_int8_unpacked.data(), K, N, K_BLK);
        auto column_sums = computeColumnSums(B_int8_unpacked.data(), K, N, K_BLK);

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;
        Bp.column_sums = column_sums.data();

        std::vector<float> bias(N, 0.0f);

        // VNNI requires unsigned A, convert signed to unsigned
        std::vector<uint8_t> A_uint8(M * K);
        for (size_t i = 0; i < A_int8.size(); ++i)
        {
            A_uint8[i] = static_cast<uint8_t>(static_cast<int16_t>(A_int8[i]) + 128);
        }
        std::vector<float> act_zero_points(T, 128.0f);

        // VNNI INT8 GEMM
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            reinterpret_cast<const int8_t *>(A_uint8.data()),
            Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            act_zero_points.data(),
            M, N, K);

        std::cout << "✓ VNNI INT8 GEMM completed" << std::endl;

        // Compare results
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Relative L2 error: " << (rel_l2 * 100) << "%" << std::endl;
        std::cout << "Max absolute error: " << max_abs << std::endl;

        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2% for INT8 GEMM";
        EXPECT_LT(max_abs, 50.0f) << "Max absolute error should be reasonable";

        std::cout << "✅ Correctness test PASSED" << std::endl;
    }

    /**
     * @brief Medium matrix test: Qwen 0.5B dimensions (512×896×896)
     *
     * Tests realistic model dimensions with tighter error bounds.
     */
    TEST_F(Test__VNNIGemm, MediumMatrix_Qwen05B_Dims)
    {
        const int M = 512, N = 896, K = 896;

        constexpr int M_R = 16;
        constexpr int N_R = 64;
        constexpr int K_BLK = 64;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 64;
        constexpr int PREFETCH_B_L2 = 256;

        std::cout << "\n=== VNNI GEMM Correctness: " << M << "×" << N << "×" << K << " (Qwen 0.5B) ===" << std::endl;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K, 100);
        auto B_fp32 = generateRandomMatrix(K, N, 101);

        // FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
        std::cout << "✓ FP32 reference completed" << std::endl;

        // Quantize to INT8
        const int T = K / K_BLK;

        std::vector<float> wgt_scales(N);
        std::vector<int8_t> B_int8_unpacked(K * N);
        for (int n = 0; n < N; ++n)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int k = 0; k < K; ++k)
            {
                float val = B_fp32[k * N + n];
                float quantized = std::max(-127.0f, std::min(127.0f, val / wgt_scales[n]));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        std::vector<float> act_scales(T);
        std::vector<int8_t> A_int8(M * K);
        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            float max_abs = 0.0f;
            for (int m = 0; m < M; ++m)
                for (int kk = 0; kk < K_BLK; ++kk)
                    max_abs = std::max(max_abs, std::abs(A_fp32[m * K + k0 + kk]));
            act_scales[t] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int m = 0; m < M; ++m)
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    float val = A_fp32[m * K + k0 + kk];
                    float quantized = std::max(-127.0f, std::min(127.0f, val / act_scales[t]));
                    A_int8[m * K + k0 + kk] = static_cast<int8_t>(std::round(quantized));
                }
        }

        // Pack B and compute column sums
        auto B_packed = packBMatrixVNNI(B_int8_unpacked.data(), K, N, K_BLK);
        auto column_sums = computeColumnSums(B_int8_unpacked.data(), K, N, K_BLK);

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_col = 4 * N;
        Bp.N = N;
        Bp.K_BLK = K_BLK;
        Bp.column_sums = column_sums.data();

        std::vector<float> bias(N, 0.0f);

        // Convert signed A to unsigned
        std::vector<uint8_t> A_uint8(M * K);
        for (size_t i = 0; i < A_int8.size(); ++i)
        {
            A_uint8[i] = static_cast<uint8_t>(static_cast<int16_t>(A_int8[i]) + 128);
        }
        std::vector<float> act_zero_points(T, 128.0f);

        // VNNI INT8 GEMM
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            reinterpret_cast<const int8_t *>(A_uint8.data()),
            Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            act_zero_points.data(),
            M, N, K);
        std::cout << "✓ VNNI INT8 GEMM completed" << std::endl;

        // Compare
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Relative L2 error: " << (rel_l2 * 100) << "%" << std::endl;
        std::cout << "Max absolute error: " << max_abs << std::endl;

        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2%";
        EXPECT_LT(max_abs, 100.0f) << "Max absolute error should be reasonable";

        std::cout << "✅ Correctness test PASSED" << std::endl;
    }

} // namespace llaminar2
