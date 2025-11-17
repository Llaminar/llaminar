/**
 * @file Test__VNNIGemmKernel.cpp
 * @brief Comprehensive unit tests for gemm_int8_vnni_kernel function
 * @author David Sanftenberg
 * @date 2025-11-16
 *
 * Tests the low-level VNNI INT8 GEMM kernel (gemm_int8_vnni_kernel) with:
 * - Simple patterns for debugging (all-ones, row/column patterns)
 * - Various scales configurations
 * - Small to large matrix dimensions
 * - Quantization accuracy validation
 *
 * Consolidated from:
 * - Test__VNNIGemm_Focused.cpp (focused kernel tests)
 * - Test__VNNIGemm_Minimal.cpp (minimal debugging tests)
 * - Test__VNNIGemm_Correctness.cpp (correctness validation)
 * - Test__VNNIGemm.cpp (comprehensive tests)
 */

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "kernels/cpu/gemm_v3/VNNIGemm.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace llaminar2
{

    namespace
    {
        inline int8_t apply_sign_bias(int8_t value)
        {
            return static_cast<int8_t>(static_cast<uint8_t>(value) ^ 0x80u);
        }

        constexpr int8_t kBiasedZero = static_cast<int8_t>(0x80);
    }

    class Test__VNNIGemmKernel : public ::testing::Test
    {
    protected:
        static constexpr double kOneDNNReferenceRelTol = 1e-5;

        void SetUp() override
        {
            std::srand(42); // Reproducibility
        }

        /**
         * @brief Simple reference INT8 GEMM with per-K-block activation scales
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

        /**
         * @brief Simple reference FP32 GEMM (triple loop)
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
         * @brief Pack B matrix into VNNI chunk-interleaved format
         */
        template <int KBlk>
        static std::vector<int8_t> packBMatrixVNNI(
            const int8_t *B, int K, int N,
            int &ld_block_out, int &ld_chunk_out, int &ld_col_out)
        {
            static_assert(KBlk % 4 == 0, "KBlk must be multiple of 4");
            const int T = K / KBlk;
            const int panel_size = N * KBlk;
            std::vector<int8_t> B_packed(T * panel_size, 0);

            for (int t = 0; t < T; ++t)
            {
                int block_ld = 0;
                int chunk_ld = 0;
                int col_ld = 0;
                int8_t *panel_ptr = B_packed.data() + t * panel_size;
                pack_B_panel_vnni<KBlk>(
                    B, K, N, t * KBlk, 0, N, panel_ptr,
                    block_ld, chunk_ld, col_ld);

                if (t == 0)
                {
                    ld_block_out = block_ld;
                    ld_chunk_out = chunk_ld;
                    ld_col_out = col_ld;
                }
            }
            return B_packed;
        }

        template <int KBlk>
        static std::vector<int8_t> packBMatrixVNNIWithWidth(
            const int8_t *B, int K, int N,
            int padded_nr,
            int &ld_block_out, int &ld_chunk_out, int &ld_col_out)
        {
            static_assert(KBlk % 4 == 0, "KBlk must be multiple of 4");
            const int T = K / KBlk;
            const int panel_size = padded_nr * KBlk;
            std::vector<int8_t> B_packed(T * panel_size, 0);

            for (int t = 0; t < T; ++t)
            {
                int block_ld = 0;
                int chunk_ld = 0;
                int col_ld = 0;
                int8_t *panel_ptr = B_packed.data() + static_cast<size_t>(t) * panel_size;
                pack_B_panel_vnni<KBlk>(
                    B, K, N,
                    t * KBlk,
                    0,
                    padded_nr,
                    panel_ptr,
                    block_ld, chunk_ld, col_ld);

                if (t == 0)
                {
                    ld_block_out = block_ld;
                    ld_chunk_out = chunk_ld;
                    ld_col_out = col_ld;
                }
            }

            return B_packed;
        }

        /**
         * @brief Compute relative L2 error
         */
        static double computeRelativeL2Error(const float *ref, const float *test, int M, int N)
        {
            double ref_norm = 0.0;
            double diff_norm = 0.0;
            for (int i = 0; i < M * N; ++i)
            {
                double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
                diff_norm += diff * diff;
                ref_norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            }
            return (ref_norm > 1e-12) ? std::sqrt(diff_norm / ref_norm) : 0.0;
        }

        /**
         * @brief Compute max absolute error
         */
        static float computeMaxAbsError(const float *ref, const float *test, int M, int N)
        {
            float max_err = 0.0f;
            for (int i = 0; i < M * N; ++i)
            {
                max_err = std::max(max_err, std::abs(ref[i] - test[i]));
            }
            return max_err;
        }

        /**
         * @brief Dequantize inputs and compute FP32 GEMM via OneDNN for ground truth
         */
        static std::vector<float> computeOneDNNReference(
            const int8_t *A, const int8_t *B,
            const float *act_scales, const float *wgt_scales,
            const float *bias,
            int M, int N, int K, int K_BLK)
        {
            const int T = K / K_BLK;

            std::vector<float> A_fp32(M * K, 0.0f);
            for (int m = 0; m < M; ++m)
            {
                for (int k = 0; k < K; ++k)
                {
                    const int t = k / K_BLK;
                    A_fp32[m * K + k] = static_cast<float>(A[m * K + k]) * act_scales[t];
                }
            }

            std::vector<float> B_fp32(K * N, 0.0f);
            for (int k = 0; k < K; ++k)
            {
                for (int n = 0; n < N; ++n)
                {
                    B_fp32[k * N + n] = static_cast<float>(B[k * N + n]) * wgt_scales[n];
                }
            }

            std::vector<float> C_fp32(M * N, 0.0f);
            dnnl::status status = dnnl::sgemm(
                'N', 'N',
                M, N, K,
                1.0f,
                A_fp32.data(), K,
                B_fp32.data(), N,
                0.0f,
                C_fp32.data(), N);

            if (status != dnnl::status::success)
            {
                throw std::runtime_error("OneDNN SGEMM failed during VNNI reference computation");
            }

            if (bias != nullptr)
            {
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        C_fp32[m * N + n] += bias[n];
                    }
                }
            }

            return C_fp32;
        }
    };

    // ========================================
    // FOCUSED KERNEL TESTS (Simple Patterns)
    // ========================================

    /**
     * @brief Test kernel with all-ones matrices
     * Expected: C[m,n] = K (sum of K ones)
     */
    TEST_F(Test__VNNIGemmKernel, AllOnesMatrix)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);
        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], static_cast<float>(K), 0.01f)
                << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test kernel with row/column pattern
     * A[m,k] = m+1, B[k,n] = n+1
     * Expected: C[m,n] = (m+1) * (n+1) * K
     */
    TEST_F(Test__VNNIGemmKernel, RowColumnPattern)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K + k] = static_cast<int8_t>(m + 1);

        std::vector<int8_t> B_unpacked(K * N);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = static_cast<int8_t>(n + 1);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float expected = static_cast<float>((m + 1) * (n + 1) * K);
                EXPECT_NEAR(C_test[m * N + n], expected, 0.5f)
                    << "Mismatch at [" << m << "," << n << "]";
            }
        }
    }

    /**
     * @brief Test kernel with non-unit scales
     * Validates scale application across K-blocks
     */
    TEST_F(Test__VNNIGemmKernel, WithScales)
    {
        const int M = 8, N = 16, K = 64; // 2 K-blocks
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        std::vector<float> act_scales(T);
        act_scales[0] = 2.0f;
        act_scales[1] = 3.0f;

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = 0.5f * (n + 1);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_qwen = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_qwen = computeRelativeL2Error(
            C_ref.data(), C_onednn_qwen.data(), M, N);
        float ref_vs_onednn_qwen_max = computeMaxAbsError(
            C_ref.data(), C_onednn_qwen.data(), M, N);
        EXPECT_LT(ref_vs_onednn_qwen, kOneDNNReferenceRelTol)
            << "QwenRealisticDims: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_qwen_max << ")";

        auto C_onednn = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn = computeRelativeL2Error(C_ref.data(), C_onednn.data(), M, N);
        float ref_vs_onednn_max = computeMaxAbsError(C_ref.data(), C_onednn.data(), M, N);
        EXPECT_LT(ref_vs_onednn, 1e-6)
            << "Simple reference diverges from OneDNN baseline (max abs error="
            << ref_vs_onednn_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    // ========================================
    // MINIMAL DEBUGGING TESTS
    // ========================================

    /**
     * @brief Tiny 4×16×4 diagonal pattern test
     */
    TEST_F(Test__VNNIGemmKernel, TinyMatrix_4x16x4)
    {
        const int M = 4, N = 16, K = 4;
        constexpr int M_R = 4, N_R = 16, K_BLK = 4;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 0);
        std::vector<int8_t> B_unpacked(K * N, 0);

        for (int i = 0; i < std::min(M, K); ++i)
            A[i * K + i] = 1;
        for (int i = 0; i < std::min(K, N); ++i)
            B_unpacked[i * N + i] = 1;

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_extreme = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_extreme = computeRelativeL2Error(
            C_ref.data(), C_onednn_extreme.data(), M, N);
        float ref_vs_onednn_extreme_max = computeMaxAbsError(
            C_ref.data(), C_onednn_extreme.data(), M, N);
        EXPECT_LT(ref_vs_onednn_extreme, kOneDNNReferenceRelTol)
            << "ExtremeValues: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_extreme_max << ")";

        auto C_onednn = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn = computeRelativeL2Error(C_ref.data(), C_onednn.data(), M, N);
        float ref_vs_onednn_max = computeMaxAbsError(C_ref.data(), C_onednn.data(), M, N);
        EXPECT_LT(ref_vs_onednn, 1e-6)
            << "Simple reference diverges from OneDNN baseline (max abs error="
            << ref_vs_onednn_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.01f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Small 8×16×8 structured values test
     */
    TEST_F(Test__VNNIGemmKernel, SmallMatrix_8x16x8)
    {
        const int M = 8, N = 16, K = 8;
        constexpr int M_R = 8, N_R = 16, K_BLK = 8;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

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
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn = computeRelativeL2Error(C_ref.data(), C_onednn.data(), M, N);
        float ref_vs_onednn_max = computeMaxAbsError(C_ref.data(), C_onednn.data(), M, N);
        EXPECT_LT(ref_vs_onednn, 1e-6)
            << "Simple reference diverges from OneDNN baseline (max abs error="
            << ref_vs_onednn_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.01f) << "Mismatch at index " << i;
        }
    }

    // ========================================
    // CORRECTNESS TESTS (Random + Quantization)
    // ========================================

    /**
     * @brief 64×64×64 random data with quantization
     * Validates INT8 quantization accuracy
     *
     * DISABLED: This test was copied from an old version that expected
     * PackedB to have column_sums field for zero-point correction.
     * Current kernel doesn't support zero-point correction.
     */
    TEST_F(Test__VNNIGemmKernel, DISABLED_SmallMatrix_64x64x64)
    {
        const int M = 64, N = 64, K = 64;
        constexpr int M_R = 16, N_R = 64, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A_fp32(M * K);
        std::vector<float> B_fp32(K * N);
        for (auto &val : A_fp32)
            val = dist(rng);
        for (auto &val : B_fp32)
            val = dist(rng);

        // Quantize B (per-column symmetric)
        std::vector<int8_t> B_int8_unpacked(K * N);
        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int k = 0; k < K; ++k)
            {
                float quantized = std::max(-127.0f, std::min(127.0f, B_fp32[k * N + n] / wgt_scales[n]));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        // Quantize A (per-K-block symmetric)
        std::vector<int8_t> A_int8(M * K);
        std::vector<float> act_scales(T);
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
                    float quantized = std::max(-127.0f, std::min(127.0f, A_fp32[m * K + k0 + kk] / act_scales[t]));
                    A_int8[m * K + k0 + kk] = static_cast<int8_t>(std::round(quantized));
                }
        }

        // Dequantize A and B for FP32 reference (to match quantization error)
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
            {
                const int t = k / K_BLK;
                A_fp32[m * K + k] = static_cast<float>(A_int8[m * K + k]) * act_scales[t];
            }

        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                B_fp32[k * N + n] = static_cast<float>(B_int8_unpacked[k * N + n]) * wgt_scales[n];

        // FP32 reference using dequantized values
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Pack B
        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_int8_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> bias(N, 0.0f);
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A_int8.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        // Debug: Print first few values
        std::cout << "\nDEBUG SmallMatrix_64x64x64:" << std::endl;
        std::cout << "  C_ref[0]=" << C_ref[0] << ", C_test[0]=" << C_test[0] << std::endl;
        std::cout << "  C_ref[1]=" << C_ref[1] << ", C_test[1]=" << C_test[1] << std::endl;
        std::cout << "  ld_block=" << ld_block << ", ld_chunk=" << ld_chunk << ", ld_col=" << ld_col << std::endl;

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2% for INT8 GEMM";
    }

    /**
     * @brief 512×896×896 realistic model dimensions (Qwen 0.5B)
     * Validates performance and accuracy at production scale
     *
     * DISABLED: This test was copied from an old version that expected
     * PackedB to have column_sums field for zero-point correction.
     * Current kernel doesn't support zero-point correction.
     */
    TEST_F(Test__VNNIGemmKernel, DISABLED_MediumMatrix_Qwen05B_Dims)
    {
        const int M = 512, N = 896, K = 896;
        constexpr int M_R = 16, N_R = 64, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A_fp32(M * K);
        std::vector<float> B_fp32(K * N);
        for (auto &val : A_fp32)
            val = dist(rng);
        for (auto &val : B_fp32)
            val = dist(rng);

        // Quantize B
        std::vector<int8_t> B_int8_unpacked(K * N);
        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int k = 0; k < K; ++k)
            {
                float quantized = std::max(-127.0f, std::min(127.0f, B_fp32[k * N + n] / wgt_scales[n]));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        // Quantize A
        std::vector<int8_t> A_int8(M * K);
        std::vector<float> act_scales(T);
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
                    float quantized = std::max(-127.0f, std::min(127.0f, A_fp32[m * K + k0 + kk] / act_scales[t]));
                    A_int8[m * K + k0 + kk] = static_cast<int8_t>(std::round(quantized));
                }
        }

        // Dequantize A and B for FP32 reference (to match quantization error)
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
            {
                const int t = k / K_BLK;
                A_fp32[m * K + k] = static_cast<float>(A_int8[m * K + k]) * act_scales[t];
            }

        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                B_fp32[k * N + n] = static_cast<float>(B_int8_unpacked[k * N + n]) * wgt_scales[n];

        // FP32 reference using dequantized values
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Pack B
        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_int8_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> bias(N, 0.0f);
        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A_int8.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2%";
        EXPECT_LT(max_abs, 100.0f) << "Max absolute error should be reasonable";
    }

    // ========================================
    // EDGE CASES AND ADDITIONAL CORRECTNESS TESTS
    // ========================================

    /**
     * @brief Test with K > K_BLK (multiple K-blocks)
     * Validates multi-block accumulation with different scales per block
     */
    TEST_F(Test__VNNIGemmKernel, MultipleKBlocks)
    {
        const int M = 16, N = 32, K = 128; // 4 K-blocks
        constexpr int M_R = 16, N_R = 32, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 2);
        std::vector<int8_t> B_unpacked(K * N, 3);

        // Different scale per K-block
        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = 1.0f + 0.5f * t;

        std::vector<float> wgt_scales(N, 2.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn = computeRelativeL2Error(C_ref.data(), C_onednn.data(), M, N);
        float ref_vs_onednn_max = computeMaxAbsError(C_ref.data(), C_onednn.data(), M, N);
        EXPECT_LT(ref_vs_onednn, 1e-6)
            << "Simple reference diverges from OneDNN baseline (max abs error="
            << ref_vs_onednn_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with non-tile-aligned M dimension
     * M < M_R to test edge handling
     */
    TEST_F(Test__VNNIGemmKernel, PartialMTile)
    {
        const int M = 5, N = 16, K = 32; // M < M_R
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        std::vector<float> act_scales(T, 1.5f);
        std::vector<float> wgt_scales(N, 2.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_partial_m = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_partial_m = computeRelativeL2Error(
            C_ref.data(), C_onednn_partial_m.data(), M, N);
        float ref_vs_onednn_partial_m_max = computeMaxAbsError(
            C_ref.data(), C_onednn_partial_m.data(), M, N);
        EXPECT_LT(ref_vs_onednn_partial_m, kOneDNNReferenceRelTol)
            << "PartialMTile: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_partial_m_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Exercises A-row padding, B-column padding, and microkernel tail loads
     */
    TEST_F(Test__VNNIGemmKernel, PartialTileColumnTailCoverage)
    {
        const int M = 10, N = 20, K = 128;
        constexpr int M_R = 16, N_R = 64, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 64, PREFETCH_B_L2 = 256;
        const int T = K / K_BLK;

        std::mt19937 rng(321);
        std::uniform_int_distribution<int> int8_dist(-32, 32);
        std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (auto &s : act_scales)
            s = scale_dist(rng);
        std::vector<float> wgt_scales(N);
        for (auto &s : wgt_scales)
            s = scale_dist(rng);
        std::vector<float> bias(N, 0.0f);

        int ld_block = 0;
        int ld_chunk = 0;
        int ld_col = 0;
        auto B_packed = packBMatrixVNNIWithWidth<K_BLK>(
            B_unpacked.data(), K, N, N_R,
            ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        const int chunk_count = K_BLK / 4;
        ASSERT_GT(N_R - N, 0);
        for (int chunk = 0; chunk < chunk_count; ++chunk)
        {
            const int8_t *chunk_base = B_packed.data() + chunk * ld_chunk;
            for (int n = N; n < N_R; ++n)
            {
                const int8_t *col_ptr = chunk_base + n * ld_col;
                for (int lane = 0; lane < 4; ++lane)
                {
                    ASSERT_EQ(col_ptr[lane], 0)
                        << "Expected zero padding for chunk " << chunk
                        << ", column " << n << ", lane " << lane;
                }
            }
        }
        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn = computeRelativeL2Error(C_ref.data(), C_onednn.data(), M, N);
        float ref_vs_onednn_max = computeMaxAbsError(C_ref.data(), C_onednn.data(), M, N);
        EXPECT_LT(ref_vs_onednn, kOneDNNReferenceRelTol)
            << "PartialTileColumnTailCoverage: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.6f) << "Mismatch at index " << i;
        }
    }

    TEST_F(Test__VNNIGemmKernel, PackAOddKChunkTailCoverage)
    {
        constexpr int M_R = 16;
        constexpr int K_BLK = 64;
        const int M = M_R;
        const int K = 60; // Force odd K_chunks (15)
        const int kblk = 60;
        const int mr = M_R;
        const int num_groups = M_R / 4;
        const int K_chunks = kblk / 4;
        const int group_stride = K_chunks * 16;

        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        for (int m = 0; m < M; ++m)
        {
            for (int k = 0; k < K; ++k)
            {
                A[m * K + k] = static_cast<int8_t>((m * K + k) % 97);
            }
        }

        std::vector<int8_t> A_packed(static_cast<size_t>(num_groups) * group_stride, 0);

        pack_A_tile_4x4_grouped<M_R, K_BLK>(
            A.data(), M, K,
            /*M0=*/0, /*k0=*/0,
            mr, kblk,
            A_packed.data());

        const int tail_chunk = K_chunks - 1; // kk index 14
        for (int g = 0; g < num_groups; ++g)
        {
            const int8_t *group_ptr = A_packed.data() + static_cast<size_t>(g) * group_stride;
            const int8_t *chunk_ptr = group_ptr + tail_chunk * 16;
            for (int lane = 0; lane < 4; ++lane)
            {
                const int row = g * 4 + lane;
                for (int offset = 0; offset < 4; ++offset)
                {
                    const int k_idx = tail_chunk * 4 + offset;
                    const int8_t expected = apply_sign_bias(A[row * K + k_idx]);
                    const int8_t packed_val = chunk_ptr[lane * 4 + offset];
                    ASSERT_EQ(packed_val, expected)
                        << "Mismatch at group " << g
                        << ", lane " << lane
                        << ", offset " << offset;
                }
            }
        }
    }

    TEST_F(Test__VNNIGemmKernel, PackAPartialRowAndZeroPaddingCoverage)
    {
        constexpr int M_R = 16;
        constexpr int K_BLK = 64;
        const int M = 10;
        const int M0 = 8;
        const int mr = M - M0; // 2 rows remain, triggers slow path
        const int K = K_BLK;
        const int kblk = K_BLK;
        const int num_groups = M_R / 4;
        const int K_chunks = kblk / 4;
        const int group_stride = K_chunks * 16;

        std::vector<int8_t> A(static_cast<size_t>(M) * K, 0);
        for (int m = 0; m < M; ++m)
        {
            for (int k = 0; k < K; ++k)
            {
                A[m * K + k] = static_cast<int8_t>((m * K + k) % 89);
            }
        }

        std::vector<int8_t> A_packed(static_cast<size_t>(num_groups) * group_stride, -1);

        pack_A_tile_4x4_grouped<M_R, K_BLK>(
            A.data(), M, K,
            M0, /*k0=*/0,
            mr, kblk,
            A_packed.data());

        const int8_t *group0 = A_packed.data();
        for (int lane = 0; lane < 4; ++lane)
        {
            for (int offset = 0; offset < 4; ++offset)
            {
                const int8_t val = group0[lane * 4 + offset];
                if (lane < mr)
                {
                    const int row = M0 + lane;
                    const int8_t expected = apply_sign_bias(A[row * K + offset]);
                    ASSERT_EQ(val, expected)
                        << "Row copy mismatch for lane " << lane << ", offset " << offset;
                }
                else
                {
                    ASSERT_EQ(val, kBiasedZero)
                        << "Expected zero-padding for lane " << lane << ", offset " << offset;
                }
            }
        }

        for (int g = 1; g < num_groups; ++g)
        {
            const int8_t *group_ptr = A_packed.data() + static_cast<size_t>(g) * group_stride;
            for (int i = 0; i < group_stride; ++i)
            {
                ASSERT_EQ(group_ptr[i], kBiasedZero)
                    << "Expected zero in padded group " << g << ", byte " << i;
            }
        }
    }

    // ========================================
    // MATRIX SIZE COVERAGE TESTS
    // ========================================

    TEST_F(Test__VNNIGemmKernel, SingleRowMatrix)
    {
        const int M = 1, N = 64, K = 64;
        constexpr int M_R = 4, N_R = 32, K_BLK = 32;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(123);
        std::uniform_int_distribution<int> int8_dist(-32, 32);
        std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        auto C_ref = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_err = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 5e-5)
            << "SingleRowMatrix relative L2 too high (max abs error=" << max_err << ")";
    }

    TEST_F(Test__VNNIGemmKernel, SingleColumnMatrix)
    {
        const int M = 48, N = 1, K = 64;
        constexpr int M_R = 16, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(321);
        std::uniform_int_distribution<int> int8_dist(-40, 40);
        std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        wgt_scales[0] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        auto C_ref = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_err = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 5e-5)
            << "SingleColumnMatrix relative L2 too high (max abs error=" << max_err << ")";
    }

    TEST_F(Test__VNNIGemmKernel, SmallRectangularMatrix)
    {
        const int M = 7, N = 32, K = 64;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(777);
        std::uniform_int_distribution<int> int8_dist(-50, 50);
        std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        auto C_ref = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_err = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 1e-4)
            << "SmallRectangularMatrix relative L2 too high (max abs error=" << max_err << ")";
    }

    TEST_F(Test__VNNIGemmKernel, MediumMatrix)
    {
        const int M = 32, N = 48, K = 192;
        constexpr int M_R = 16, N_R = 32, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(888);
        std::uniform_int_distribution<int> int8_dist(-75, 75);
        std::uniform_real_distribution<float> scale_dist(0.4f, 1.6f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        auto C_ref = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_err = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 2e-4)
            << "MediumMatrix relative L2 too high (max abs error=" << max_err << ")";
    }

    TEST_F(Test__VNNIGemmKernel, LargeMatrix)
    {
        const int M = 64, N = 128, K = 256;
        constexpr int M_R = 16, N_R = 64, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(999);
        std::uniform_int_distribution<int> int8_dist(-100, 100);
        std::uniform_real_distribution<float> scale_dist(0.3f, 1.7f);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        auto C_ref = computeOneDNNReference(
            A.data(), B_unpacked.data(),
            act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_err = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 3e-4)
            << "LargeMatrix relative L2 too high (max abs error=" << max_err << ")";
    }

    /**
     * @brief Test with non-tile-aligned N dimension
     * N < N_R to test edge handling
     *
     * DISABLED: Triggers AddressSanitizer error - out-of-bounds read in microkernel
     * when loading B matrix with N=10 < N_R=16. Kernel needs bounds checking for
     * partial N tiles in B prefetch/load logic.
     */
    TEST_F(Test__VNNIGemmKernel, PartialNTile)
    {
        const int M = 8, N = 10, K = 32; // N < N_R
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 2);
        std::vector<int8_t> B_unpacked(K * N, 3);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.5f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_negative = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_negative = computeRelativeL2Error(
            C_ref.data(), C_onednn_negative.data(), M, N);
        float ref_vs_onednn_negative_max = computeMaxAbsError(
            C_ref.data(), C_onednn_negative.data(), M, N);
        EXPECT_LT(ref_vs_onednn_negative, kOneDNNReferenceRelTol)
            << "NegativeValues: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_negative_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with both M and N non-tile-aligned
     *
     * DISABLED: Would trigger same bugs as PartialMTile + PartialNTile.
     * Re-enable after fixing partial tile handling in kernel.
     */
    TEST_F(Test__VNNIGemmKernel, PartialMNTile)
    {
        const int M = 13, N = 21, K = 64; // Both M < M_R and N < N_R
        constexpr int M_R = 16, N_R = 32, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        std::vector<float> act_scales(T, 2.0f);
        std::vector<float> wgt_scales(N, 1.5f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_qwen = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_qwen = computeRelativeL2Error(
            C_ref.data(), C_onednn_qwen.data(), M, N);
        float ref_vs_onednn_qwen_max = computeMaxAbsError(
            C_ref.data(), C_onednn_qwen.data(), M, N);
        EXPECT_LT(ref_vs_onednn_qwen, kOneDNNReferenceRelTol)
            << "QwenRealisticDims: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_qwen_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with varying scales across columns
     * Validates per-column weight scaling
     */
    TEST_F(Test__VNNIGemmKernel, VaryingColumnScales)
    {
        const int M = 8, N = 32, K = 64;
        constexpr int M_R = 8, N_R = 32, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = 0.5f + 0.1f * n; // Linearly increasing scales

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        auto C_onednn_extreme = computeOneDNNReference(
            A.data(), B_unpacked.data(), act_scales.data(), wgt_scales.data(), bias.data(),
            M, N, K, K_BLK);
        double ref_vs_onednn_extreme = computeRelativeL2Error(
            C_ref.data(), C_onednn_extreme.data(), M, N);
        float ref_vs_onednn_extreme_max = computeMaxAbsError(
            C_ref.data(), C_onednn_extreme.data(), M, N);
        EXPECT_LT(ref_vs_onednn_extreme, kOneDNNReferenceRelTol)
            << "ExtremeValues: simple reference diverges from OneDNN (max abs error="
            << ref_vs_onednn_extreme_max << ")";

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with non-zero bias
     * Validates bias addition
     */
    TEST_F(Test__VNNIGemmKernel, NonZeroBias)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 1);

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);

        std::vector<float> bias(N);
        for (int n = 0; n < N; ++n)
            bias[n] = 10.0f * (n + 1); // Non-trivial bias values

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);
        // Add bias to reference
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                C_ref[m * N + n] += bias[n];

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with negative values
     * Validates signed INT8 handling
     */
    TEST_F(Test__VNNIGemmKernel, NegativeValues)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);

        // Mix of positive and negative values
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = (i % 2 == 0) ? 10 : -10;
        for (size_t i = 0; i < B_unpacked.size(); ++i)
            B_unpacked[i] = (i % 3 == 0) ? 5 : -5;

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with larger K_BLK (128)
     * Validates different K-block sizes
     */
    TEST_F(Test__VNNIGemmKernel, LargeKBlock)
    {
        const int M = 16, N = 32, K = 256;
        constexpr int M_R = 16, N_R = 32, K_BLK = 128; // Larger K_BLK
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K, 1);
        std::vector<int8_t> B_unpacked(K * N, 2);

        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = 1.0f + 0.2f * t;

        std::vector<float> wgt_scales(N, 1.5f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with realistic Qwen 0.5B layer dimensions
     * M=32 (batch), N=896 (d_model), K=896 (d_model)
     */
    TEST_F(Test__VNNIGemmKernel, QwenRealisticDims)
    {
        const int M = 32, N = 896, K = 896;
        constexpr int M_R = 16, N_R = 64, K_BLK = 64;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);
        for (auto &val : A)
            val = static_cast<int8_t>(int8_dist(rng));
        for (auto &val : B_unpacked)
            val = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T);
        std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 0.01) << "Relative L2 error should be <1% for random INT8 values";
    }

    /**
     * @brief Test with extreme INT8 values (boundary conditions)
     * Uses min/max INT8 values to test overflow handling
     */
    TEST_F(Test__VNNIGemmKernel, ExtremeValues)
    {
        const int M = 8, N = 16, K = 32;
        constexpr int M_R = 8, N_R = 16, K_BLK = 32;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;
        const int T = K / K_BLK;

        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B_unpacked(K * N);

        // Alternate between min and max INT8 values
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = (i % 2 == 0) ? 127 : -127;
        for (size_t i = 0; i < B_unpacked.size(); ++i)
            B_unpacked[i] = (i % 2 == 0) ? 127 : -127;

        std::vector<float> act_scales(T, 0.01f); // Small scales to prevent overflow
        std::vector<float> wgt_scales(N, 0.01f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K);

        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with K dimension not aligned to K_BLK
     * K=130 with K_BLK=128 exercises the edge case where the last K-block
     * has lanes that extend beyond K, triggering nullptr assignment in pack_B_panel_vnni
     *
     * This test packs B with UNPADDED K to exercise the lane_ptrs[lane]=nullptr path
     */
    TEST_F(Test__VNNIGemmKernel, NonAlignedKDimension_Small)
    {
        const int M = 8, N = 16, K = 130; // K not multiple of K_BLK=128
        constexpr int M_R = 8, N_R = 16, K_BLK = 128;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;

        // Pack with UNPADDED K to trigger nullptr lanes in last chunk
        const int T = (K + K_BLK - 1) / K_BLK; // 2 blocks (128 + partial)
        const int K_packed = T * K_BLK;        // 256 for computation

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A(M * K_packed, 0);
        std::vector<int8_t> B_unpacked(K_packed * N, 0);

        // Fill only the valid K range (0..129)
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K_packed + k] = static_cast<int8_t>(int8_dist(rng));

        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        // Pack B with ACTUAL K (not padded) to trigger nullptr for lanes beyond K
        int ld_block, ld_chunk, ld_col;
        const int panel_size = N * K_BLK;
        std::vector<int8_t> B_packed(T * panel_size, 0);

        for (int t = 0; t < T; ++t)
        {
            int8_t *panel_ptr = B_packed.data() + t * panel_size;
            pack_B_panel_vnni<K_BLK>(
                B_unpacked.data(), K, N, // Pass UNPADDED K=130
                t * K_BLK, 0, N, panel_ptr,
                ld_block, ld_chunk, ld_col);
        }

        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K_packed, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K_packed);

        // Result should be identical since padding is zero
        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i;
        }
    }

    /**
     * @brief Test with larger K dimension not aligned to K_BLK
     * K=450 with K_BLK=128 (3.5 blocks) exercises multiple partial K-blocks
     */
    TEST_F(Test__VNNIGemmKernel, NonAlignedKDimension_Medium)
    {
        const int M = 16, N = 32, K = 450; // K not multiple of K_BLK=128
        constexpr int M_R = 16, N_R = 32, K_BLK = 128;
        constexpr int UNROLL_K = 2, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;

        const int K_padded = ((K + K_BLK - 1) / K_BLK) * K_BLK; // 512
        const int T = K_padded / K_BLK;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A(M * K_padded, 0);
        std::vector<int8_t> B_unpacked(K_padded * N, 0);

        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K_padded + k] = static_cast<int8_t>(int8_dist(rng));

        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = static_cast<int8_t>(int8_dist(rng));

        std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);
        std::vector<float> act_scales(T);
        for (int t = 0; t < T; ++t)
            act_scales[t] = scale_dist(rng);

        std::vector<float> wgt_scales(N);
        for (int n = 0; n < N; ++n)
            wgt_scales[n] = scale_dist(rng);

        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K_padded, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K_padded, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K_padded);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        EXPECT_LT(rel_l2, 0.01) << "Relative L2 error should be <1% even with non-aligned K";
    }

    /**
     * @brief Test with N dimension requiring scalar tail in packing (exercises line 359)
     * N=20 (not multiple of 16) with panel width causes scalar tail handling in pack_B_panel_vnni
     * Combined with non-aligned K to trigger nullptr lane handling
     */
    TEST_F(Test__VNNIGemmKernel, NonAlignedKDimension_WithPartialN)
    {
        const int M = 8, N = 20, K = 130; // K not aligned to K_BLK, N has scalar tail in packing
        constexpr int M_R = 8, N_R = 32, K_BLK = 128;
        constexpr int UNROLL_K = 1, PREFETCH_B_L1 = 0, PREFETCH_B_L2 = 0;

        const int K_padded = ((K + K_BLK - 1) / K_BLK) * K_BLK; // 256
        const int T = K_padded / K_BLK;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A(M * K_padded, 0);
        std::vector<int8_t> B_unpacked(K_padded * N, 0);

        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[m * K_padded + k] = static_cast<int8_t>(int8_dist(rng));

        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_unpacked[k * N + n] = static_cast<int8_t>(int8_dist(rng));

        std::vector<float> act_scales(T, 1.0f);
        std::vector<float> wgt_scales(N, 1.0f);
        std::vector<float> bias(N, 0.0f);

        int ld_block, ld_chunk, ld_col;
        auto B_packed = packBMatrixVNNI<K_BLK>(B_unpacked.data(), K_padded, N, ld_block, ld_chunk, ld_col);
        PackedB Bp{B_packed.data(), ld_block, ld_chunk, ld_col, N, K_BLK};

        std::vector<float> C_ref(M * N, 0.0f);
        simpleReferenceGemmINT8(A.data(), B_unpacked.data(), C_ref.data(),
                                act_scales.data(), wgt_scales.data(), M, N, K_padded, K_BLK);

        std::vector<float> C_test(M * N, 0.0f);
        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            A.data(), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(), M, N, K_padded);

        // Verify correctness with non-aligned K and partial N (scalar tail in packing)
        for (int i = 0; i < M * N; ++i)
        {
            EXPECT_NEAR(C_test[i], C_ref[i], 0.5f) << "Mismatch at index " << i
                                                   << " with non-aligned K and N dimensions";
        }
    }

} // namespace llaminar2
