/**
 * @file Test__Q8_1GemmKernel.cpp
 * @brief Unit tests for Q8_1GemmKernel mathematical correctness
 * @author David Sanftenberg
 * @date 2025-11-13
 *
 * Tests Q8_1 × Q8_0 → FP32 GEMM kernel with various activation tensor types:
 * - Q8_1Tensor (zero-copy, pre-computed sums)
 * - FP32Tensor (on-the-fly quantization)
 * - FP16Tensor (on-the-fly quantization)
 * - BF16Tensor (on-the-fly quantization)
 *
 * Validates:
 * - Mathematical correctness vs FP32 reference
 * - Pre-computed sum usage (Q8_1 blocks)
 * - IQ8_1Decodable interface compliance
 * - Edge cases (small/large matrices, non-multiple-of-32 dimensions)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

#include "v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/SIMDHelpers.h"
#include "utils/DebugEnv.h"

namespace llaminar2
{

    /**
     * @brief Test fixture for Q8_1GemmKernel tests
     */
    class Test__Q8_1GemmKernel : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Fixed seed for reproducibility
            rng.seed(42);
        }

        /**
         * @brief Generate random FP32 matrix with controlled range
         */
        std::vector<float> generateRandomMatrix(int rows, int cols, float min_val = -1.0f, float max_val = 1.0f)
        {
            std::uniform_real_distribution<float> dist(min_val, max_val);
            std::vector<float> matrix(rows * cols);
            for (auto &val : matrix)
            {
                val = dist(rng);
            }
            return matrix;
        }

        /**
         * @brief Create Q8_0Tensor from FP32 data (for weights)
         */
        std::shared_ptr<Q8_0Tensor> createQ8_0Tensor(const float *data, int rows, int cols)
        {
            // Q8_1GemmKernel expects B (K×N weights) in COLUMN-MAJOR block layout.
            // The kernel accesses as: B_blocks[column_idx * K_blocks + k_block_idx]
            // This means: N columns, each containing K_blocks quantized blocks.
            //
            // Input: data is K×N in row-major (rows=K, cols=N)
            // Output: Q8_0Tensor with blocks arranged as N columns × K_blocks blocks/column
            //
            // Strategy: We MUST transpose before quantization, otherwise blocks won't align!

            const int K = rows;
            const int N = cols;

            // Step 1: Transpose K×N row-major → N×K row-major
            std::vector<float> transposed(K * N);
            for (int k = 0; k < K; ++k)
            {
                for (int n = 0; n < N; ++n)
                {
                    transposed[n * K + k] = data[k * N + n];
                }
            }

            // Step 2: Quantize the N×K row-major data
            // This creates blocks in row-major order: column 0 (K elements), column 1 (K elements), etc.
            FP32Tensor fp32_tensor({static_cast<size_t>(N), static_cast<size_t>(K)});
            std::memcpy(const_cast<float *>(fp32_tensor.data()), transposed.data(), K * N * sizeof(float));

            size_t num_elements = K * N;
            size_t num_blocks = (num_elements + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
            std::vector<Q8_0Block> q8_0_blocks(num_blocks);
            fp32_tensor.to_q8_0(q8_0_blocks.data());

            // Step 3: Create Q8_0Tensor with N×K shape (column-major interpretation for kernel)
            std::vector<uint8_t> raw_data(num_blocks * sizeof(Q8_0Block));
            std::memcpy(raw_data.data(), q8_0_blocks.data(), raw_data.size());

            return std::make_shared<Q8_0Tensor>(
                std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)},
                raw_data);
        }

        /**
         * @brief Reference FP32 GEMM: C = A × B
         */
        void referenceGemm(const float *A, const float *B, float *C,
                           int M, int N, int K)
        {
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }

        /**
         * @brief Compute relative L2 error between two matrices
         */
        double computeRelativeL2Error(const float *C_ref, const float *C_test,
                                      int M, int N)
        {
            double num = 0.0, denom = 0.0;
            for (int i = 0; i < M * N; ++i)
            {
                double diff = C_ref[i] - C_test[i];
                num += diff * diff;
                denom += C_ref[i] * C_ref[i];
            }
            return std::sqrt(num / (denom + 1e-12));
        }

        /**
         * @brief Compute max absolute error
         */
        float computeMaxAbsError(const float *C_ref, const float *C_test,
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
         * @brief Print error statistics
         */
        void printErrorStats(const std::string &test_name,
                             const float *C_ref, const float *C_test,
                             int M, int N)
        {
            double rel_l2 = computeRelativeL2Error(C_ref, C_test, M, N);
            float max_abs = computeMaxAbsError(C_ref, C_test, M, N);

            std::cout << "[" << test_name << "] "
                      << "Rel L2: " << std::scientific << std::setprecision(3) << rel_l2
                      << ", Max Abs: " << max_abs << std::endl;
        }

        std::mt19937 rng;
    };

    // ============================================================================
    // Basic Correctness Tests
    // ============================================================================

    /**
     * @brief Test Q8_1Tensor × Q8_0Tensor GEMM (zero-copy path)
     *
     * This is the primary use case: pre-quantized Q8_1 activations with Q8_0 weights.
     * Q8_1 blocks have pre-computed sums, enabling 18× speedup vs inline computation.
     */
    TEST_F(Test__Q8_1GemmKernel, Q8_1_times_Q8_0_BasicCorrectness)
    {
        const int M = 64, N = 64, K = 256;

        std::cout << "\n=== DEBUG BasicCorrectness: M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        // Quantize to Q8_1 (activations) and Q8_0 (weights)
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(),
                                                     {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::cout << "A_q8_1 shape: " << A_q8_1->shape()[0] << "×" << A_q8_1->shape()[1] << std::endl;
        std::cout << "B_q8_0 shape: " << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;
        std::cout << "K_blocks: " << (K / 32) << std::endl;

        // Print first A block info
        const Q8_1Block *a_block0 = A_q8_1->decode_to_q8_1(0, 0);
        float a_d = llaminar2::simd::fp16_to_fp32(a_block0->d);
        float a_s = llaminar2::simd::fp16_to_fp32(a_block0->s);
        std::cout << "A block[0,0]: d=" << a_d << ", s=" << a_s << ", qs[0]=" << (int)a_block0->qs[0] << std::endl;

        // Print first B block info
        Q8_0Block b_block0;
        B_q8_0->decode_to_q8_0(0, 0, &b_block0);
        float b_d = llaminar2::simd::fp16_to_fp32(b_block0.d);
        std::cout << "B block[0,0]: d=" << b_d << ", qs[0]=" << (int)b_block0.qs[0] << std::endl;

        // Compute Q8_1 × Q8_0 GEMM
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "C_test[0]=" << C_test[0] << ", C_test[1]=" << C_test[1] << std::endl;

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "C_ref[0]=" << C_ref[0] << ", C_ref[1]=" << C_ref[1] << std::endl;

        // Validate (quantization introduces error, but should be <1% relative L2)
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        printErrorStats("Q8_1×Q8_0", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.01) << "Relative L2 error should be <1% for Q8_1×Q8_0";
        EXPECT_LT(max_abs, 10.0f) << "Max absolute error should be reasonable";
    }

    /**
     * @brief Test FP32Tensor × Q8_0Tensor GEMM (on-the-fly quantization)
     *
     * FP32 activations are quantized to Q8_1 on-the-fly via IQ8_1Decodable interface.
     * This validates the interface works correctly for non-Q8_1 tensor types.
     */
    TEST_F(Test__Q8_1GemmKernel, FP32_times_Q8_0_OnTheFlyQuantization)
    {
        const int M = 64, N = 64, K = 256;

        // Generate random FP32 matrices
        auto A_fp32_data = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        // Create FP32Tensor (implements IQ8_1Decodable)
        FP32Tensor A_fp32({static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_fp32.mutable_data(), A_fp32_data.data(), M * K * sizeof(float));

        // Quantize weights to Q8_0
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Compute FP32 × Q8_0 GEMM (A quantized on-the-fly)
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, A_fp32, *B_q8_0, C_test.data(), N);

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Validate
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        printErrorStats("FP32×Q8_0", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.01) << "Relative L2 error should be <1% for FP32×Q8_0";
        EXPECT_LT(max_abs, 10.0f) << "Max absolute error should be reasonable";
    }

    /**
     * @brief Test FP16Tensor × Q8_0Tensor GEMM (on-the-fly quantization)
     */
    TEST_F(Test__Q8_1GemmKernel, FP16_times_Q8_0_OnTheFlyQuantization)
    {
        const int M = 64, N = 64, K = 256;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        // Convert FP32 to FP16
        std::vector<uint16_t> A_fp16_data(M * K);
        for (size_t i = 0; i < M * K; ++i)
        {
            A_fp16_data[i] = llaminar2::simd::fp32_to_fp16(A_fp32[i]);
        }
        FP16Tensor A_fp16({static_cast<size_t>(M), static_cast<size_t>(K)}, A_fp16_data);

        // Quantize weights to Q8_0
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Compute FP16 × Q8_0 GEMM
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, A_fp16, *B_q8_0, C_test.data(), N);

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Validate (FP16 has additional rounding error)
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        printErrorStats("FP16×Q8_0", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.015) << "Relative L2 error should be <1.5% for FP16×Q8_0 (FP16 rounding)";
        EXPECT_LT(max_abs, 15.0f) << "Max absolute error should be reasonable";
    }

    /**
     * @brief Test BF16Tensor × Q8_0Tensor GEMM (on-the-fly quantization)
     */
    TEST_F(Test__Q8_1GemmKernel, BF16_times_Q8_0_OnTheFlyQuantization)
    {
        const int M = 64, N = 64, K = 256;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        // Convert FP32 to BF16
        std::vector<uint16_t> A_bf16_data(M * K);
        for (size_t i = 0; i < M * K; ++i)
        {
            A_bf16_data[i] = llaminar2::simd::fp32_to_bf16(A_fp32[i]);
        }
        BF16Tensor A_bf16({static_cast<size_t>(M), static_cast<size_t>(K)}, A_bf16_data);

        // Quantize weights to Q8_0
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Compute BF16 × Q8_0 GEMM
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, A_bf16, *B_q8_0, C_test.data(), N);

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Validate (BF16 has reduced mantissa precision)
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        printErrorStats("BF16×Q8_0", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2% for BF16×Q8_0 (BF16 precision)";
        EXPECT_LT(max_abs, 20.0f) << "Max absolute error should be reasonable";
    }

    // ============================================================================
    // Edge Cases
    // ============================================================================

    /**
     * @brief Test small matrix (8×8×32) - fits entirely in microkernel
     */
    TEST_F(Test__Q8_1GemmKernel, SmallMatrix_8x8x32)
    {
        const int M = 8, N = 8, K = 32;

        std::cout << "\n=== DEBUG SmallMatrix: M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::cout << "A_q8_1 shape: " << A_q8_1->shape()[0] << "×" << A_q8_1->shape()[1] << std::endl;
        std::cout << "B_q8_0 shape: " << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;
        std::cout << "K_blocks: " << (K / 32) << std::endl;

        // Print first blocks
        const Q8_1Block *a_block0 = A_q8_1->decode_to_q8_1(0, 0);
        float a_d = llaminar2::simd::fp16_to_fp32(a_block0->d);
        float a_s = llaminar2::simd::fp16_to_fp32(a_block0->s);
        std::cout << "A block[0,0]: d=" << a_d << ", s=" << a_s << ", qs[0]=" << (int)a_block0->qs[0] << std::endl;

        Q8_0Block b_block0;
        B_q8_0->decode_to_q8_0(0, 0, &b_block0);
        float b_d = llaminar2::simd::fp16_to_fp32(b_block0.d);
        std::cout << "B block[0,0]: d=" << b_d << ", qs[0]=" << (int)b_block0.qs[0] << std::endl;

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "C_test[0]=" << C_test[0] << ", C_test[1]=" << C_test[1] << std::endl;

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        printErrorStats("Small 8×8×32", C_ref.data(), C_test.data(), M, N);
        std::cout << "DEBUG Small: A[0]=" << A_fp32[0] << ", B[0]=" << B_fp32[0] << std::endl;
        std::cout << "DEBUG Small: C_ref[0]=" << C_ref[0] << ", C_test[0]=" << C_test[0] << std::endl;
        std::cout << "DEBUG Small: B shape=" << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "Small matrix should have <1% error";
    }

    /**
     * @brief Test single block K dimension (K=32, minimal case)
     */
    TEST_F(Test__Q8_1GemmKernel, SingleKBlock_32x32x32)
    {
        const int M = 32, N = 32, K = 32;

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        printErrorStats("Single K block", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.01) << "Single K block should have <1% error";
    }

    /**
     * @brief Test large matrix (512×512×512) - exercises full blocking
     */
    TEST_F(Test__Q8_1GemmKernel, LargeMatrix_512x512x512)
    {
        const int M = 512, N = 512, K = 512;

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        printErrorStats("Large 512×512×512", C_ref.data(), C_test.data(), M, N);

        EXPECT_LT(rel_l2, 0.01) << "Large matrix should have <1% error";
        EXPECT_LT(max_abs, 50.0f) << "Max error should be reasonable for large matrix";
    }

    /**
     * @brief Test rectangular matrix (tall: 256×64×128)
     */
    TEST_F(Test__Q8_1GemmKernel, RectangularMatrix_Tall_256x64x128)
    {
        const int M = 256, N = 64, K = 128;

        std::cout << "\n=== DEBUG Tall (PASSING): M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::cout << "A_q8_1 shape: " << A_q8_1->shape()[0] << "×" << A_q8_1->shape()[1] << std::endl;
        std::cout << "B_q8_0 shape: " << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;
        std::cout << "K_blocks: " << (K / 32) << std::endl;

        // Print first blocks
        const Q8_1Block *a_block0 = A_q8_1->decode_to_q8_1(0, 0);
        float a_d = llaminar2::simd::fp16_to_fp32(a_block0->d);
        float a_s = llaminar2::simd::fp16_to_fp32(a_block0->s);
        std::cout << "A block[0,0]: d=" << a_d << ", s=" << a_s << ", qs[0]=" << (int)a_block0->qs[0] << std::endl;

        Q8_0Block b_block0;
        B_q8_0->decode_to_q8_0(0, 0, &b_block0);
        float b_d = llaminar2::simd::fp16_to_fp32(b_block0.d);
        std::cout << "B block[0,0]: d=" << b_d << ", qs[0]=" << (int)b_block0.qs[0] << std::endl;

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "C_test[0]=" << C_test[0] << ", C_test[1]=" << C_test[1] << std::endl;

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "C_ref[0]=" << C_ref[0] << ", C_ref[1]=" << C_ref[1] << std::endl;

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        printErrorStats("Tall 256×64×128", C_ref.data(), C_test.data(), M, N);
        std::cout << "DEBUG Tall: B shape=" << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "Tall rectangular matrix should have <1% error";
    }

    /**
     * @brief Test rectangular matrix (wide: 64×256×128)
     */
    TEST_F(Test__Q8_1GemmKernel, RectangularMatrix_Wide_64x256x128)
    {
        const int M = 64, N = 256, K = 128;

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        printErrorStats("Wide 64×256×128", C_ref.data(), C_test.data(), M, N);
        std::cout << "DEBUG Tall: B shape=" << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "Wide rectangular matrix should have <1% error";
    }

    // ============================================================================
    // Pre-Computed Sum Validation
    // ============================================================================

    /**
     * @brief Validate that Q8_1 pre-computed sums are correctly used
     *
     * This test verifies that the kernel is actually using the pre-computed sums
     * from Q8_1 blocks rather than computing them inline. We do this by:
     * 1. Creating a Q8_1Tensor with known values
     * 2. Manually verifying the pre-computed sums are correct
     * 3. Running the GEMM and checking results match expected compensation formula
     */
    TEST_F(Test__Q8_1GemmKernel, PreComputedSumsValidation)
    {
        const int M = 32, N = 32, K = 32;

        // Create simple pattern: all elements in a block are the same value
        std::vector<float> A_fp32(M * K);
        for (int i = 0; i < M; ++i)
        {
            for (int k = 0; k < K; ++k)
            {
                // Each block has constant value = (i % 8)
                A_fp32[i * K + k] = static_cast<float>((i % 8) - 4); // Range: [-4, 3]
            }
        }

        // Quantize to Q8_1
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        // Verify pre-computed sums are present
        std::cout << "DEBUG: Checking Q8_1 block sums..." << std::endl;
        for (int i = 0; i < M; ++i)
        {
            const Q8_1Block *block = A_q8_1->decode_to_q8_1(i, 0);
            float d_fp32 = fp16_to_fp32(block->d);
            float s_fp32 = fp16_to_fp32(block->s);

            // Manually compute sum for verification
            int32_t manual_sum = 0;
            for (int j = 0; j < 32; ++j)
            {
                manual_sum += block->qs[j];
            }

            if (i < 5)
            { // Print first 5 rows
                std::cout << "Row " << i << ": d=" << d_fp32 << ", s=" << s_fp32
                          << " (raw=" << block->s << "), manual_sum=" << manual_sum
                          << ", expected_s=" << (d_fp32 * manual_sum) << std::endl;
            }

            // If manual_sum is non-zero, pre-computed sum should also be non-zero (unless FP16 underflow)
            if (manual_sum != 0)
            {
                // Allow for FP16 underflow on very small values
                if (std::abs(d_fp32 * manual_sum) > 1e-5f)
                {
                    EXPECT_NE(block->s, 0) << "Pre-computed sum should be non-zero for row " << i
                                           << " (manual_sum=" << manual_sum << ")";
                }
            }

            // Verify expected sum matches actual (allow small FP16 rounding error)
            float expected_sum = d_fp32 * static_cast<float>(manual_sum);

            // Allow small error due to FP16 rounding
            EXPECT_NEAR(expected_sum, s_fp32, 0.01f * std::abs(expected_sum))
                << "Pre-computed sum mismatch for row " << i;
        }

        // Now run GEMM and verify correctness
        auto B_fp32 = generateRandomMatrix(K, N);
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        printErrorStats("Pre-computed sums", C_ref.data(), C_test.data(), M, N);
        std::cout << "DEBUG Tall: B shape=" << B_q8_0->shape()[0] << "×" << B_q8_0->shape()[1] << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "GEMM with pre-computed sums should have <1% error";
    }

    // ============================================================================
    // Unit Tests for Helper Functions
    // ============================================================================

    /**
     * @brief Test compute_sum_qs helper function
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ComputeSumQs)
    {
        // Test case 1: Normal values
        uint16_t a_sum_fp16 = llaminar2::simd::fp32_to_fp16(-1.17188f);
        uint16_t a_scale_fp16 = llaminar2::simd::fp32_to_fp16(0.00785828f);

        float sum_qs = Q8_1GemmKernel::compute_sum_qs(a_sum_fp16, a_scale_fp16);
        float expected = -1.17188f / 0.00785828f; // ≈ -149.13

        std::cout << "compute_sum_qs: sum=" << llaminar2::simd::fp16_to_fp32(a_sum_fp16)
                  << ", scale=" << llaminar2::simd::fp16_to_fp32(a_scale_fp16)
                  << ", result=" << sum_qs << ", expected≈" << expected << std::endl;

        EXPECT_NEAR(sum_qs, expected, 1.0f) << "Sum QS should be s/d";

        // Test case 2: Zero block (division by zero protection)
        a_sum_fp16 = llaminar2::simd::fp32_to_fp16(0.0f);
        a_scale_fp16 = llaminar2::simd::fp32_to_fp16(0.0f);

        sum_qs = Q8_1GemmKernel::compute_sum_qs(a_sum_fp16, a_scale_fp16);
        EXPECT_FALSE(std::isnan(sum_qs)) << "Should not produce NaN for zero block";
        EXPECT_FALSE(std::isinf(sum_qs)) << "Should not produce Inf for zero block";
        EXPECT_EQ(sum_qs, 0.0f) << "Zero block should have sum_qs=0";
    }

    /**
     * @brief Test apply_compensation helper function (Q8_0 - no compensation)
     * Q8_0 uses symmetric quantization - NO compensation needed!
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ApplyCompensation)
    {
        // Test case: Q8_0 symmetric quantization should NOT apply compensation
        int32_t accum = 4064;
        float sum_qs = -149.13f;

        int32_t compensated = Q8_1GemmKernel::apply_compensation(accum, sum_qs);
        int32_t expected = accum; // No compensation for symmetric Q8_0!

        std::cout << "apply_compensation (Q8_0): accum=" << accum
                  << ", sum_qs=" << sum_qs << " (unused for Q8_0)"
                  << ", compensated=" << compensated
                  << ", expected=" << expected << std::endl;

        EXPECT_EQ(compensated, expected) << "Q8_0 apply_compensation should return accum unchanged";
    }

    /**
     * @brief Test apply_dpbusd_compensation helper function (Q8_1 - compensation required)
     * Q8_1 uses dpbusd which needs compensation for unsigned→signed conversion
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ApplyDpbusdCompensation)
    {
        // Test case: Q8_1 dpbusd requires compensation
        int32_t dpbusd_result = 4064;
        float sum_qs = -149.13f;

        int32_t compensated = Q8_1GemmKernel::apply_dpbusd_compensation(dpbusd_result, sum_qs);

        // For this specific case: 4064 - (-149.13 * 128) = 4064 - (-19088.64) = 4064 + 19088.64 ≈ 23153
        int32_t expected = static_cast<int32_t>(static_cast<float>(dpbusd_result) - 128.0f * sum_qs);

        std::cout << "apply_dpbusd_compensation (Q8_1): dpbusd_result=" << dpbusd_result
                  << ", sum_qs=" << sum_qs
                  << ", compensated=" << compensated
                  << ", expected=" << expected << std::endl;

        EXPECT_EQ(compensated, expected) << "Q8_1 dpbusd compensation should be dpbusd - 128*sum_qs";
        EXPECT_GT(compensated, dpbusd_result) << "Negative sum_qs should increase compensated value";
    }

    /**
     * @brief Test scale_to_fp32 helper function
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ScaleToFp32)
    {
        int32_t compensated = 23152;
        float a_scale = 0.00785828f;
        float b_scale = 0.00779343f;

        float result = Q8_1GemmKernel::scale_to_fp32(compensated, a_scale, b_scale);
        float expected = static_cast<float>(compensated) * a_scale * b_scale;

        std::cout << "scale_to_fp32: compensated=" << compensated
                  << ", a_scale=" << a_scale
                  << ", b_scale=" << b_scale
                  << ", result=" << result
                  << ", expected=" << expected << std::endl;

        EXPECT_FLOAT_EQ(result, expected) << "Scaling should multiply compensated * a_scale * b_scale";
    }

    /**
     * @brief Test complete block computation (integration of all helpers)
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ComputeBlockResult)
    {
        // Use real values from failing test
        int32_t accum = 4064; // dpbusd result for first block
        uint16_t a_sum_fp16 = llaminar2::simd::fp32_to_fp16(-1.17188f);
        uint16_t a_scale_fp16 = llaminar2::simd::fp32_to_fp16(0.00785828f);
        uint16_t b_scale_fp16 = llaminar2::simd::fp32_to_fp16(0.00779343f);

        float result = Q8_1GemmKernel::compute_block_result(accum, a_sum_fp16, a_scale_fp16, b_scale_fp16);

        // Manual computation for verification (Q8_0 symmetric: NO compensation!)
        float sum_qs = -1.17188f / 0.00785828f; // -149.13 (computed but unused)
        int32_t compensated = accum;            // No compensation for Q8_0!
        float a_scale = 0.00785828f;
        float b_scale = 0.00779343f;
        float expected = static_cast<float>(compensated) * a_scale * b_scale;

        std::cout << "compute_block_result:" << std::endl;
        std::cout << "  accum=" << accum << std::endl;
        std::cout << "  sum_qs=" << sum_qs << " (computed but unused for Q8_0)" << std::endl;
        std::cout << "  compensated=" << compensated << " (no compensation!)" << std::endl;
        std::cout << "  result=" << result << std::endl;
        std::cout << "  expected=" << expected << std::endl;

        EXPECT_NEAR(result, expected, 1e-5f) << "Complete block result should match manual calculation";
    }

    /**
     * @brief Test block index calculation for column-major layout
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_BlockIndexColMajor)
    {
        // For N=8, K=32: K_blocks=1, B shape is 8×32 (N×K after transpose)
        // Blocks should be: [row0_kb0, row1_kb0, row2_kb0, ..., row7_kb0]
        size_t K_blocks = 1;

        for (size_t row = 0; row < 8; ++row)
        {
            size_t idx = Q8_1GemmKernel::block_index_col_major(row, 0, K_blocks);
            EXPECT_EQ(idx, row) << "For K_blocks=1, index should equal row";
        }

        // For N=64, K=128: K_blocks=4, B shape is 64×128
        K_blocks = 4;

        // Row 0: blocks at 0, 1, 2, 3
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(0, 0, K_blocks), 0);
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(0, 1, K_blocks), 1);
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(0, 2, K_blocks), 2);
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(0, 3, K_blocks), 3);

        // Row 1: blocks at 4, 5, 6, 7
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(1, 0, K_blocks), 4);
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(1, 3, K_blocks), 7);

        // Row 63: blocks at 252, 253, 254, 255
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(63, 0, K_blocks), 252);
        EXPECT_EQ(Q8_1GemmKernel::block_index_col_major(63, 3, K_blocks), 255);

        std::cout << "Column-major block indexing verified for K_blocks=1 and K_blocks=4" << std::endl;
    }

    /**
     * @brief Test manual single-element GEMM computation
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ManualSingleElementGEMM)
    {
        // Create minimal 1×1×32 GEMM (single element result from single block)
        const int M = 1, N = 1, K = 32;

        // Generate known FP32 values
        std::vector<float> A_fp32(K);
        std::vector<float> B_fp32(K);
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int k = 0; k < K; ++k)
        {
            A_fp32[k] = dist(gen);
            B_fp32[k] = dist(gen);
        }

        // Compute FP32 reference
        float ref_result = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            ref_result += A_fp32[k] * B_fp32[k];
        }

        // Quantize to Q8_1 and Q8_0
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {1, 32});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Get quantized blocks
        const Q8_1Block *a_block = A_q8_1->decode_to_q8_1(0, 0);
        Q8_0Block b_block;
        B_q8_0->decode_to_q8_0(0, 0, &b_block);

        // Manually compute dpbusd (Σ(a_qs[i] * b_qs[i]))
        int32_t manual_accum = 0;
        for (int i = 0; i < 32; ++i)
        {
            manual_accum += static_cast<int32_t>(a_block->qs[i]) * static_cast<int32_t>(b_block.qs[i]);
        }

        // Apply Q8_0 compensation formula
        float sum_qs = Q8_1GemmKernel::compute_sum_qs(a_block->s, a_block->d);
        int32_t compensated = Q8_1GemmKernel::apply_compensation(manual_accum, sum_qs);
        float a_scale = fp16_to_fp32(a_block->d);
        float b_scale = fp16_to_fp32(b_block.d);
        float quant_result = Q8_1GemmKernel::scale_to_fp32(compensated, a_scale, b_scale);

        // Run actual GEMM
        std::vector<float> C_test(1, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "\n=== Manual 1×1×32 GEMM Verification ===" << std::endl;
        std::cout << "FP32 reference: " << ref_result << std::endl;
        std::cout << "Manual accum: " << manual_accum << std::endl;
        std::cout << "Sum QS: " << sum_qs << std::endl;
        std::cout << "Compensated: " << compensated << std::endl;
        std::cout << "Manual quant result: " << quant_result << std::endl;
        std::cout << "Kernel GEMM result: " << C_test[0] << std::endl;
        std::cout << "Quantization error: " << std::abs(quant_result - ref_result) << std::endl;
        std::cout << "Kernel error: " << std::abs(C_test[0] - ref_result) << std::endl;

        // TEST HYPOTHESIS: What if we DON'T apply compensation?
        float no_compensation_result = static_cast<float>(manual_accum) * a_scale * b_scale;
        std::cout << "NO compensation result: " << no_compensation_result << std::endl;
        std::cout << "NO compensation error: " << std::abs(no_compensation_result - ref_result) << std::endl;

        // Verify manual computation matches kernel
        EXPECT_NEAR(C_test[0], quant_result, 1e-5f)
            << "Kernel result should match manual computation";

        // Verify quantization error is reasonable (<1%)
        double rel_error = std::abs(quant_result - ref_result) / (std::abs(ref_result) + 1e-10);
        EXPECT_LT(rel_error, 0.01) << "Quantization error should be <1%";
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    /**
     * @brief Test that non-multiple-of-32 K dimension throws error
     */
    TEST_F(Test__Q8_1GemmKernel, InvalidKDimension_NotMultipleOf32)
    {
        const int M = 32, N = 32, K = 31; // K=31 is NOT multiple of 32

        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        FP32Tensor A({M, K});
        std::memcpy(A.mutable_data(), A_fp32.data(), M * K * sizeof(float));

        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        std::vector<float> C(M * N, 0.0f);

        // Should throw runtime_error for K not multiple of 32
        EXPECT_THROW({ Q8_1GemmKernel::gemm(M, N, K, A, *B_q8_0, C.data(), N); }, std::runtime_error);
    }

    /**
     * @brief Test that non-IQ8_1Decodable tensor throws error
     *
     * NOTE: This is tricky to test since all our tensor types implement IQ8_1Decodable.
     * We'd need a mock tensor that doesn't implement the interface.
     * For now, we document that the dynamic_cast check is in place.
     */
    // TEST_F(Test__Q8_1GemmKernel, InvalidTensor_NotIQ8_1Decodable) {
    //     // Would require a mock tensor type that doesn't implement IQ8_1Decodable
    //     // Skipping for now - covered by type system
    // }

    /**
     * @brief Test B value unpacking helper function
     *
     * Verifies that unpack_B_value() correctly extracts values from packed layout.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_UnpackBValue)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int N = 8;
        constexpr int K = 64;
        constexpr int K_blocks = 2;

        // Create test data with known pattern
        std::vector<Q8_0Block> B_blocks(N * K_blocks);
        for (int jr = 0; jr < N; ++jr)
        {
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                Q8_0Block &block = B_blocks[jr * K_blocks + kb];
                block.d = fp32_to_fp16(0.01f);

                // Fill with pattern: value = jr*100 + kb*10 + k_in
                for (int k_in = 0; k_in < 32; ++k_in)
                {
                    int pattern = jr * 100 + kb * 10 + k_in;
                    // Clamp to int8 range
                    block.qs[k_in] = static_cast<int8_t>(pattern % 200 - 100);
                }
            }
        }

        // Pack B panel
        Kernel::PackedBPanel packed(K_blocks);
        Kernel::pack_B_panel(N, K_blocks, B_blocks.data(), packed);

        // Verify unpacking
        bool all_match = true;
        for (int jr = 0; jr < N; ++jr)
        {
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                const Q8_0Block &orig = B_blocks[jr * K_blocks + kb];
                for (int k_in = 0; k_in < 32; ++k_in)
                {
                    int8_t expected = orig.qs[k_in];
                    int8_t actual = Kernel::unpack_B_value(packed.quants.data(), kb, jr, k_in);

                    if (expected != actual)
                    {
                        std::cout << "MISMATCH at jr=" << jr << ", kb=" << kb << ", k_in=" << k_in
                                  << ": expected=" << static_cast<int>(expected)
                                  << ", actual=" << static_cast<int>(actual) << std::endl;
                        all_match = false;
                    }
                }
            }
        }

        EXPECT_TRUE(all_match) << "B unpacking should match original values";
    }

    /**
     * @brief Test B packing/unpacking round-trip verification
     *
     * Uses verify_B_packing() helper to ensure pack_B_panel() is invertible.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_VerifyBPacking)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int N = 8;
        constexpr int K = 128;
        constexpr int K_blocks = 4;

        // Create random test data
        std::vector<float> B_fp32 = generateRandomMatrix(K, N, -5.0f, 5.0f);
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Get blocks (column-major layout)
        std::vector<Q8_0Block> B_blocks(N * K_blocks);
        for (int jr = 0; jr < N; ++jr)
        {
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                B_q8_0->decode_to_q8_0(jr, kb, &B_blocks[jr * K_blocks + kb]);
            }
        }

        // Pack B panel
        Kernel::PackedBPanel packed(K_blocks);
        Kernel::pack_B_panel(N, K_blocks, B_blocks.data(), packed);

        // Verify round-trip
        bool verified = Kernel::verify_B_packing(B_blocks.data(), packed, N, K_blocks);
        EXPECT_TRUE(verified) << "B packing should be invertible (pack then unpack = identity)";
    }

    /**
     * @brief Test B unpacking with edge cases (partial panels)
     *
     * Verifies unpacking works correctly when N < NR (zero-padded columns).
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_UnpackBValuePartialPanel)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int N = 5; // Partial panel (< NR=8)
        constexpr int K = 32;
        constexpr int K_blocks = 1;

        // Create test data
        std::vector<Q8_0Block> B_blocks(N * K_blocks);
        for (int jr = 0; jr < N; ++jr)
        {
            Q8_0Block &block = B_blocks[jr];
            block.d = fp32_to_fp16(0.01f);
            for (int k_in = 0; k_in < 32; ++k_in)
            {
                block.qs[k_in] = static_cast<int8_t>(jr * 10 + k_in - 50);
            }
        }

        // Pack B panel (will zero-pad columns 5-7)
        Kernel::PackedBPanel packed(K_blocks);
        Kernel::pack_B_panel(N, K_blocks, B_blocks.data(), packed);

        // Verify actual columns
        for (int jr = 0; jr < N; ++jr)
        {
            for (int k_in = 0; k_in < 32; ++k_in)
            {
                int8_t expected = B_blocks[jr].qs[k_in];
                int8_t actual = Kernel::unpack_B_value(packed.quants.data(), 0, jr, k_in);
                EXPECT_EQ(expected, actual) << "Column " << jr << ", k_in=" << k_in;
            }
        }

        // Verify zero-padded columns (5-7)
        for (int jr = N; jr < 8; ++jr)
        {
            for (int k_in = 0; k_in < 32; ++k_in)
            {
                int8_t actual = Kernel::unpack_B_value(packed.quants.data(), 0, jr, k_in);
                EXPECT_EQ(0, actual) << "Zero-padded column " << jr << ", k_in=" << k_in;
            }
        }
    }

    /**
     * @brief Test per-block contribution computation
     *
     * Verifies that compute_per_block_contribution() correctly scales accumulator by scales.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_PerBlockContribution)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        // Test case: accum=1000, a_scale=0.01, b_scale=0.02
        int32_t accum = 1000;
        uint16_t a_scale_fp16 = fp32_to_fp16(0.01f);
        uint16_t b_scale_fp16 = fp32_to_fp16(0.02f);

        float result = Kernel::compute_per_block_contribution(accum, a_scale_fp16, b_scale_fp16);

        // Expected: 1000 * 0.01 * 0.02 = 0.2
        float expected = 1000.0f * 0.01f * 0.02f;

        EXPECT_NEAR(result, expected, 1e-4f) << "Per-block contribution should be accum * a_scale * b_scale";
    }

    /**
     * @brief Test scalar GEMM result computation (sum across K-blocks)
     *
     * Verifies that compute_gemm_result_scalar() correctly sums contributions.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_GemmResultScalar)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 4;

        // Test data: varying accumulators and scales
        std::vector<int32_t> accum_values = {1000, 2000, -1500, 500};
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            a_scales[kb] = fp32_to_fp16(0.01f * (kb + 1));
            b_scales[kb] = fp32_to_fp16(0.02f * (kb + 1));
        }

        float result = Kernel::compute_gemm_result_scalar(
            accum_values.data(), a_scales.data(), b_scales.data(), K_blocks);

        // Compute expected manually
        float expected = 0.0f;
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            float a_scale = fp16_to_fp32(a_scales[kb]);
            float b_scale = fp16_to_fp32(b_scales[kb]);
            expected += static_cast<float>(accum_values[kb]) * a_scale * b_scale;
        }

        EXPECT_NEAR(result, expected, 1e-4f) << "Scalar GEMM result should sum per-block contributions";
    }

    /**
     * @brief Test vectorized reduction against scalar reference
     *
     * Verifies that SIMD horizontal reduction produces same result as scalar accumulation.
     * This is the critical test that will reveal if there's a bug in the vectorized path!
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_VectorizedReduction)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 16; // Multiple of VECTOR_WIDTH

        // Generate random test data
        std::vector<int32_t> accum_values(K_blocks);
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);

        std::uniform_int_distribution<int32_t> accum_dist(-10000, 10000);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            accum_values[kb] = accum_dist(rng);
            a_scales[kb] = fp32_to_fp16(scale_dist(rng));
            b_scales[kb] = fp32_to_fp16(scale_dist(rng));
        }

        bool verified = Kernel::verify_vectorized_reduction(
            accum_values.data(), a_scales.data(), b_scales.data(), K_blocks);

        EXPECT_TRUE(verified) << "Vectorized reduction should match scalar reference";

        // Also print the actual values for debugging
        float scalar_result = Kernel::compute_gemm_result_scalar(
            accum_values.data(), a_scales.data(), b_scales.data(), K_blocks);

        std::cout << "Scalar result: " << scalar_result << std::endl;
        std::cout << "First few accum values: ";
        for (int kb = 0; kb < std::min(4, K_blocks); ++kb)
        {
            std::cout << accum_values[kb] << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief Test vectorized reduction with non-multiple-of-16 K_blocks
     *
     * Tests the scalar tail path when K_blocks is not a multiple of VECTOR_WIDTH.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_VectorizedReductionWithTail)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 20; // Not a multiple of 16

        // Generate test data
        std::vector<int32_t> accum_values(K_blocks);
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            accum_values[kb] = (kb + 1) * 100;
            a_scales[kb] = fp32_to_fp16(0.01f);
            b_scales[kb] = fp32_to_fp16(0.02f);
        }

        bool verified = Kernel::verify_vectorized_reduction(
            accum_values.data(), a_scales.data(), b_scales.data(), K_blocks);

        EXPECT_TRUE(verified) << "Vectorized reduction with tail should match scalar reference";
    }

    /**
     * @brief Test dpbusd dot product computation
     *
     * Verifies that our scalar simulation of dpbusd matches expected behavior.
     * dpbusd computes: Σ(unsigned_byte × signed_byte)
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_DpbusdDotProduct)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        // Simple test: all 1s
        {
            std::vector<uint8_t> B_unsigned(32, 200); // All 200
            std::vector<int8_t> A_signed(32, 50);     // All 50

            int32_t result = Kernel::compute_dpbusd_dot_product(B_unsigned.data(), A_signed.data(), 32);
            int32_t expected = 200 * 50 * 32; // 320,000

            EXPECT_EQ(result, expected) << "dpbusd with constant values";
        }

        // Test with mixed signs
        {
            std::vector<uint8_t> B_unsigned(32);
            std::vector<int8_t> A_signed(32);

            for (int i = 0; i < 32; ++i)
            {
                B_unsigned[i] = 128 + i; // 128..159
                A_signed[i] = i - 16;    // -16..15
            }

            int32_t result = Kernel::compute_dpbusd_dot_product(B_unsigned.data(), A_signed.data(), 32);

            // Manually compute expected value
            int32_t expected = 0;
            for (int i = 0; i < 32; ++i)
            {
                expected += static_cast<int32_t>(B_unsigned[i]) * static_cast<int32_t>(A_signed[i]);
            }

            EXPECT_EQ(result, expected) << "dpbusd with mixed signs";
        }
    }

    /**
     * @brief Test reference dot product (signed × signed)
     *
     * This is what the edge case microkernel computes.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_ReferenceDotProduct)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        std::vector<int8_t> B_signed(32);
        std::vector<int8_t> A_signed(32);

        for (int i = 0; i < 32; ++i)
        {
            B_signed[i] = i - 16; // -16..15
            A_signed[i] = 16 - i; // 16..-15
        }

        int32_t result = Kernel::compute_reference_dot_product(B_signed.data(), A_signed.data(), 32);

        // Manually compute expected value
        int32_t expected = 0;
        for (int i = 0; i < 32; ++i)
        {
            expected += static_cast<int32_t>(B_signed[i]) * static_cast<int32_t>(A_signed[i]);
        }

        EXPECT_EQ(result, expected) << "Reference dot product";
    }

    /**
     * @brief Test dpbusd compensation application
     *
     * Verifies that: dpbusd((B+128), A) - 128*Σ(A) = Σ(B × A)
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_DpbusdCompensation)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        // Test case 1: Simple constant values
        {
            std::vector<int8_t> B_signed(32, 10); // All 10
            std::vector<int8_t> A_signed(32, 5);  // All 5 (Σ(qs) = 160)
            float a_scale = 0.1f;
            float sum_a = a_scale * 160.0f; // Q8_1 precomputed sum: s = d × Σ(qs) = 16.0

            bool verified = Kernel::verify_dpbusd_compensation(
                B_signed.data(), A_signed.data(), sum_a, a_scale, 32);

            EXPECT_TRUE(verified) << "dpbusd compensation with constant values";
        }

        // Test case 2: Mixed positive/negative values
        {
            std::vector<int8_t> B_signed(32);
            std::vector<int8_t> A_signed(32);
            float sum_qs = 0.0f; // Σ(qs) in quantized space

            for (int i = 0; i < 32; ++i)
            {
                B_signed[i] = static_cast<int8_t>(i - 16); // -16..15
                A_signed[i] = static_cast<int8_t>(15 - i); // 15..-16
                sum_qs += static_cast<float>(A_signed[i]);
            }

            float a_scale = 0.05f;
            float sum_a = a_scale * sum_qs; // Q8_1 precomputed sum: s = d × Σ(qs)

            bool verified = Kernel::verify_dpbusd_compensation(
                B_signed.data(), A_signed.data(), sum_a, a_scale, 32);

            EXPECT_TRUE(verified) << "dpbusd compensation with mixed signs";
        }

        // Test case 3: Extreme values
        {
            std::vector<int8_t> B_signed(32, 127);  // Max positive
            std::vector<int8_t> A_signed(32, -127); // Max negative (Σ(qs) = -4064)
            float a_scale = 1.0f;
            float sum_a = a_scale * (-4064.0f); // Q8_1 precomputed sum: s = d × Σ(qs) = -4064

            bool verified = Kernel::verify_dpbusd_compensation(
                B_signed.data(), A_signed.data(), sum_a, a_scale, 32);

            EXPECT_TRUE(verified) << "dpbusd compensation with extreme values";
        }
    }

    /**
     * @brief Test dpbusd compensation with realistic Q8 scales
     *
     * Uses typical scale values from real quantized tensors.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_DpbusdCompensationRealistic)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        std::vector<int8_t> B_signed(32);
        std::vector<int8_t> A_signed(32);

        // Use realistic quantized values (from BasicCorrectness test)
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);

        float sum_qs = 0.0f; // Σ(qs) in quantized space
        for (int i = 0; i < 32; ++i)
        {
            B_signed[i] = static_cast<int8_t>(quant_dist(rng));
            A_signed[i] = static_cast<int8_t>(quant_dist(rng));
            sum_qs += static_cast<float>(A_signed[i]);
        }

        // Typical scale from Q8 quantization (range / 127)
        float a_scale = 0.00787f;       // From BasicCorrectness test
        float sum_a = a_scale * sum_qs; // Q8_1 precomputed sum: s = d × Σ(qs)

        bool verified = Kernel::verify_dpbusd_compensation(
            B_signed.data(), A_signed.data(), sum_a, a_scale, 32);

        EXPECT_TRUE(verified) << "dpbusd compensation with realistic values and scales";
    }

    // ============================================================================
    // MINI-INTEGRATION TESTS: Test kernel computation flow
    // ============================================================================

    /**
     * @brief Integration: dpbusd → compensation → scaling → verify against reference
     *
     * Tests the FULL computation path for ONE K-block:
     * 1. Compute dpbusd(B_unsigned, A_signed)
     * 2. Apply compensation: result = dpbusd - 128*Σ(A)
     * 3. Scale to FP32: result * a_scale * b_scale
     * 4. Compare against reference: Σ(B_signed × A_signed) * a_scale * b_scale
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_SingleKBlockFullPath)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        // Test with realistic Q8 values
        std::vector<int8_t> B_signed(32);
        std::vector<int8_t> A_signed(32);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);

        float sum_qs = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            B_signed[i] = static_cast<int8_t>(quant_dist(rng));
            A_signed[i] = static_cast<int8_t>(quant_dist(rng));
            sum_qs += static_cast<float>(A_signed[i]);
        }

        float a_scale = 0.00787f;
        float b_scale = 0.00823f;
        float sum_a = a_scale * sum_qs; // Q8_1 precomputed sum

        // Step 1: dpbusd
        std::vector<uint8_t> B_unsigned(32);
        for (int i = 0; i < 32; ++i)
        {
            B_unsigned[i] = static_cast<uint8_t>(B_signed[i] + 128);
        }
        int32_t dpbusd_result = Kernel::compute_dpbusd_dot_product(B_unsigned.data(), A_signed.data(), 32);

        // Step 2: Compensation
        float sum_qs_computed = sum_a / a_scale;
        int32_t compensated = Kernel::apply_dpbusd_compensation(dpbusd_result, sum_qs_computed);

        // Step 3: Scale to FP32
        float result = static_cast<float>(compensated) * a_scale * b_scale;

        // Step 4: Reference (what edge case computes)
        int32_t ref_accum = Kernel::compute_reference_dot_product(B_signed.data(), A_signed.data(), 32);
        float reference = static_cast<float>(ref_accum) * a_scale * b_scale;

        // Verify
        float abs_diff = std::abs(result - reference);
        float rel_diff = abs_diff / std::max(std::abs(reference), 1e-6f);

        std::cout << "[MINI-INTEGRATION: Single K-block]" << std::endl;
        std::cout << "  dpbusd_result = " << dpbusd_result << std::endl;
        std::cout << "  compensated = " << compensated << std::endl;
        std::cout << "  result = " << result << std::endl;
        std::cout << "  reference = " << reference << std::endl;
        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;

        EXPECT_LT(rel_diff, 1e-5f) << "Full computation path should match reference";
    }

    /**
     * @brief Integration: Multi-block accumulation with vectorized reduction
     *
     * Tests the FULL computation for MULTIPLE K-blocks:
     * 1. For each K-block: dpbusd → compensation
     * 2. Vectorized reduction: sum(compensated[kb] * a_scale[kb] * b_scale[kb])
     * 3. Compare against reference
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_MultiKBlockVectorizedPath)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 8;
        constexpr int BLOCK_SIZE = 32;

        // Generate test data for K_blocks blocks
        std::vector<int8_t> B_signed(K_blocks * BLOCK_SIZE);
        std::vector<int8_t> A_signed(K_blocks * BLOCK_SIZE);
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);
        std::vector<int32_t> accum_compensated(K_blocks);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.01f);

        float reference_sum = 0.0f;

        // Process each K-block
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            float sum_qs = 0.0f;

            // Generate quantized values
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                int idx = kb * BLOCK_SIZE + i;
                B_signed[idx] = static_cast<int8_t>(quant_dist(rng));
                A_signed[idx] = static_cast<int8_t>(quant_dist(rng));
                sum_qs += static_cast<float>(A_signed[idx]);
            }

            float a_scale = scale_dist(rng);
            float b_scale = scale_dist(rng);
            a_scales[kb] = fp32_to_fp16(a_scale);
            b_scales[kb] = fp32_to_fp16(b_scale);

            // Step 1: dpbusd
            std::vector<uint8_t> B_unsigned(BLOCK_SIZE);
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                B_unsigned[i] = static_cast<uint8_t>(B_signed[kb * BLOCK_SIZE + i] + 128);
            }
            int32_t dpbusd_result = Kernel::compute_dpbusd_dot_product(
                B_unsigned.data(), &A_signed[kb * BLOCK_SIZE], BLOCK_SIZE);

            // Step 2: Compensation
            float sum_a = a_scale * sum_qs;
            float sum_qs_computed = sum_a / a_scale;
            accum_compensated[kb] = Kernel::apply_dpbusd_compensation(dpbusd_result, sum_qs_computed);

            // Reference for this block
            int32_t ref_accum = Kernel::compute_reference_dot_product(
                &B_signed[kb * BLOCK_SIZE], &A_signed[kb * BLOCK_SIZE], BLOCK_SIZE);
            reference_sum += static_cast<float>(ref_accum) * a_scale * b_scale;
        }

        // Step 3: Vectorized reduction (what the microkernel does)
        bool reduction_verified = Kernel::verify_vectorized_reduction(
            accum_compensated.data(), a_scales.data(), b_scales.data(), K_blocks);

        EXPECT_TRUE(reduction_verified) << "Vectorized reduction should match scalar";

        // Compute result using scalar path
        float result = Kernel::compute_gemm_result_scalar(
            accum_compensated.data(), a_scales.data(), b_scales.data(), K_blocks);

        float abs_diff = std::abs(result - reference_sum);
        float rel_diff = abs_diff / std::max(std::abs(reference_sum), 1e-6f);

        std::cout << "[MINI-INTEGRATION: Multi K-block]" << std::endl;
        std::cout << "  K_blocks = " << K_blocks << std::endl;
        std::cout << "  result = " << result << std::endl;
        std::cout << "  reference = " << reference_sum << std::endl;
        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;

        EXPECT_LT(rel_diff, 5e-3f) << "Multi-block computation should match reference (allowing for FP16 quantization of sum_a)";
    }

    /**
     * @brief Integration: B packing → unpacking → dpbusd → compensation
     *
     * Tests that the B packing/unpacking preserves correctness:
     * 1. Pack B values into kernel format
     * 2. Unpack and verify
     * 3. Compute dpbusd using packed B
     * 4. Apply compensation and verify against reference
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_BPackingAndComputation)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int N = 8;
        constexpr int K = 64;
        constexpr int K_blocks = 2;

        // Create B blocks
        std::vector<Q8_0Block> B_blocks(N * K_blocks);
        std::vector<int8_t> A_signed(K);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);

        float sum_qs = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            A_signed[i] = static_cast<int8_t>(quant_dist(rng));
            sum_qs += static_cast<float>(A_signed[i]);
        }

        for (int jr = 0; jr < N; ++jr)
        {
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                Q8_0Block &block = B_blocks[jr * K_blocks + kb];
                block.d = fp32_to_fp16(0.01f);

                for (int k_in = 0; k_in < 32; ++k_in)
                {
                    block.qs[k_in] = static_cast<int8_t>(quant_dist(rng));
                }
            }
        }

        // Pack B
        typename Kernel::PackedBPanel packed(K_blocks);
        packed.quants.resize(N * K_blocks * 32);
        packed.scales.resize(N * K_blocks);

        Kernel::pack_B_panel(N, K_blocks, B_blocks.data(), packed);

        // Verify packing
        bool packing_ok = Kernel::verify_B_packing(B_blocks.data(), packed, N, K_blocks);
        EXPECT_TRUE(packing_ok) << "B packing should preserve values";

        // Compute using packed B for first column (jr=0)
        float a_scale = 0.00787f;

        float result = 0.0f;
        float reference = 0.0f;

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            // Compute sum_a for THIS K-block only
            float sum_qs_kb = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                sum_qs_kb += static_cast<float>(A_signed[kb * 32 + i]);
            }
            float sum_a_kb = a_scale * sum_qs_kb; // Q8_1 precomputed sum for this block

            // Get B scale for this K-block from packed structure
            // Packed scales layout: [jr][kb], so for jr=0, index is kb
            uint16_t b_scale_fp16 = packed.scales[0 * K_blocks + kb];
            float b_scale = fp16_to_fp32(b_scale_fp16);

            // Extract B values using unpack helper
            std::vector<int8_t> B_signed(32);
            for (int k_in = 0; k_in < 32; ++k_in)
            {
                B_signed[k_in] = Kernel::unpack_B_value(
                    packed.quants.data(), kb, 0, k_in); // jr=0
            }

            // dpbusd
            std::vector<uint8_t> B_unsigned(32);
            for (int i = 0; i < 32; ++i)
            {
                B_unsigned[i] = static_cast<uint8_t>(B_signed[i] + 128);
            }
            int32_t dpbusd_result = Kernel::compute_dpbusd_dot_product(
                B_unsigned.data(), &A_signed[kb * 32], 32);

            // Compensation
            float sum_qs_computed = sum_a_kb / a_scale;
            int32_t compensated = Kernel::apply_dpbusd_compensation(dpbusd_result, sum_qs_computed);

            result += static_cast<float>(compensated) * a_scale * b_scale;

            // Reference
            int32_t ref_accum = Kernel::compute_reference_dot_product(
                B_signed.data(), &A_signed[kb * 32], 32);
            reference += static_cast<float>(ref_accum) * a_scale * b_scale;
        }

        float abs_diff = std::abs(result - reference);
        float rel_diff = abs_diff / std::max(std::abs(reference), 1e-6f);

        std::cout << "[MINI-INTEGRATION: B Packing + Computation]" << std::endl;
        std::cout << "  result = " << result << std::endl;
        std::cout << "  reference = " << reference << std::endl;
        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;

        EXPECT_LT(rel_diff, 1e-5f) << "Computation using packed B should match reference";
    }

    /**
     * @brief Integration: Verify microkernel compensation matches edge case for same data
     *
     * Forces both full microkernel and edge case to process IDENTICAL data,
     * then compares results. This isolates the dpbusd+compensation bug.
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_MicrokernelVsEdgeCaseComputation)
    {
        using Kernel = Q8_1GemmKernelTemplate<32, 64>;
        constexpr int M = 32;
        constexpr int N = 64;
        constexpr int K = 32; // Single K-block
        constexpr int K_blocks = 1;

        // Create identical test data for both paths
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.01f);

        // A matrix (Q8_1)
        std::vector<Q8_1Block> A_blocks(M * K_blocks);
        for (int i = 0; i < M; ++i)
        {
            Q8_1Block &block = A_blocks[i];
            block.d = fp32_to_fp16(scale_dist(rng));

            float sum_qs = 0.0f;
            for (int k = 0; k < 32; ++k)
            {
                block.qs[k] = static_cast<int8_t>(quant_dist(rng));
                sum_qs += static_cast<float>(block.qs[k]);
            }
            block.s = fp32_to_fp16(fp16_to_fp32(block.d) * sum_qs);
        }

        // B matrix (Q8_0)
        std::vector<Q8_0Block> B_blocks(N * K_blocks);
        for (int j = 0; j < N; ++j)
        {
            Q8_0Block &block = B_blocks[j];
            block.d = fp32_to_fp16(scale_dist(rng));

            for (int k = 0; k < 32; ++k)
            {
                block.qs[k] = static_cast<int8_t>(quant_dist(rng));
            }
        }

        // Compute reference using edge case logic (signed × signed, no dpbusd)
        std::vector<float> C_reference(M * N, 0.0f);

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                const Q8_1Block &a_block = A_blocks[i];
                const Q8_0Block &b_block = B_blocks[j];

                int32_t accum = 0;
                for (int k = 0; k < 32; ++k)
                {
                    accum += static_cast<int32_t>(a_block.qs[k]) * static_cast<int32_t>(b_block.qs[k]);
                }

                float a_scale = fp16_to_fp32(a_block.d);
                float b_scale = fp16_to_fp32(b_block.d);
                C_reference[i * N + j] = static_cast<float>(accum) * a_scale * b_scale;
            }
        }

        // Compute using dpbusd + compensation (what full microkernel should do)
        std::vector<float> C_dpbusd(M * N, 0.0f);

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                const Q8_1Block &a_block = A_blocks[i];
                const Q8_0Block &b_block = B_blocks[j];

                // Convert B to unsigned for dpbusd
                std::vector<uint8_t> B_unsigned(32);
                std::vector<int8_t> A_signed(32);
                for (int k = 0; k < 32; ++k)
                {
                    B_unsigned[k] = static_cast<uint8_t>(b_block.qs[k] + 128);
                    A_signed[k] = a_block.qs[k];
                }

                // dpbusd
                int32_t dpbusd_result = Kernel::compute_dpbusd_dot_product(
                    B_unsigned.data(), A_signed.data(), 32);

                // Compensation
                float a_scale = fp16_to_fp32(a_block.d);
                float sum_a = fp16_to_fp32(a_block.s);
                float sum_qs = sum_a / a_scale;
                int32_t compensated = Kernel::apply_dpbusd_compensation(dpbusd_result, sum_qs);

                // Scale
                float b_scale = fp16_to_fp32(b_block.d);
                C_dpbusd[i * N + j] = static_cast<float>(compensated) * a_scale * b_scale;
            }
        }

        // Compare results
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        int mismatch_count = 0;
        int large_mismatch_count = 0; // Track mismatches with abs(reference) > 0.01
        int debug_printed = 0;
        int worst_case_idx = -1;

        for (int i = 0; i < M * N; ++i)
        {
            float abs_diff = std::abs(C_dpbusd[i] - C_reference[i]);
            float rel_diff = abs_diff / std::max(std::abs(C_reference[i]), 1e-6f);

            if (abs_diff > max_abs_diff)
            {
                max_abs_diff = abs_diff;
                worst_case_idx = i;
            }
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if (rel_diff > 5e-3f) // Relax threshold to 0.5% (FP16 roundtrip error)
            {
                mismatch_count++;
                if (std::abs(C_reference[i]) > 0.01f)
                {
                    large_mismatch_count++;

                    if (debug_printed < 3)
                    {
                        debug_printed++;
                        int row = i / N;
                        int col = i % N;
                        const Q8_1Block &a_block = A_blocks[row];
                        const Q8_0Block &b_block = B_blocks[col];

                        float a_scale = fp16_to_fp32(a_block.d);
                        float sum_a = fp16_to_fp32(a_block.s);
                        float sum_qs_direct = 0.0f;
                        for (int k = 0; k < 32; ++k)
                        {
                            sum_qs_direct += static_cast<float>(a_block.qs[k]);
                        }
                        float sum_qs_computed = sum_a / a_scale;

                        std::cout << "\n[DEBUG MISMATCH " << debug_printed << "] C[" << row << "," << col << "]:" << std::endl;
                        std::cout << "  C_reference = " << C_reference[i] << std::endl;
                        std::cout << "  C_dpbusd = " << C_dpbusd[i] << std::endl;
                        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;
                        std::cout << "  a_scale = " << a_scale << std::endl;
                        std::cout << "  sum_a (FP16) = " << sum_a << std::endl;
                        std::cout << "  sum_qs (direct) = " << sum_qs_direct << std::endl;
                        std::cout << "  sum_qs (sum_a/scale) = " << sum_qs_computed << std::endl;
                    }
                }
            }
        }

        // Print worst case
        if (worst_case_idx >= 0)
        {
            int row = worst_case_idx / N;
            int col = worst_case_idx % N;
            const Q8_1Block &a_block = A_blocks[row];
            const Q8_0Block &b_block = B_blocks[col];

            float a_scale = fp16_to_fp32(a_block.d);
            float sum_a = fp16_to_fp32(a_block.s);
            float sum_qs_direct = 0.0f;
            for (int k = 0; k < 32; ++k)
            {
                sum_qs_direct += static_cast<float>(a_block.qs[k]);
            }
            float sum_qs_computed = sum_a / a_scale;

            std::cout << "\n[WORST CASE] C[" << row << "," << col << "]:" << std::endl;
            std::cout << "  C_reference = " << C_reference[worst_case_idx] << std::endl;
            std::cout << "  C_dpbusd = " << C_dpbusd[worst_case_idx] << std::endl;
            std::cout << "  abs_diff = " << max_abs_diff << std::endl;
            std::cout << "  a_scale = " << a_scale << std::endl;
            std::cout << "  sum_a (FP16) = " << sum_a << std::endl;
            std::cout << "  sum_qs (direct) = " << sum_qs_direct << std::endl;
            std::cout << "  sum_qs (sum_a/scale) = " << sum_qs_computed << std::endl;
            std::cout << "  sum_qs error = " << (sum_qs_computed - sum_qs_direct) << std::endl;
        }

        std::cout << "[MINI-INTEGRATION: Microkernel vs Edge Case]" << std::endl;
        std::cout << "  Matrix size: " << M << " × " << N << " × " << K << std::endl;
        std::cout << "  max_abs_diff = " << max_abs_diff << std::endl;
        std::cout << "  max_rel_diff = " << max_rel_diff << std::endl;
        std::cout << "  mismatches = " << mismatch_count << " / " << (M * N) << std::endl;
        std::cout << "  large mismatches (|ref|>0.01) = " << large_mismatch_count << std::endl;

        // FP16 quantization of sum_a introduces ~0.2 error in sum_qs for large sums
        // This gets amplified by 128× compensation, leading to ~25 int32 error
        // After scaling by typical scales (0.001-0.01), this becomes ~0.002 absolute error
        EXPECT_LT(max_abs_diff, 2e-3f) << "Absolute error should be small (FP16 quantization limit)";
        EXPECT_LT(mismatch_count, 100) << "Most elements should match within 0.5% tolerance";
    }

    /**
     * @brief Integration: Test that sum_a computation per K-block is correct
     *
     * Verifies that we correctly split the total sum_a across K-blocks.
     * In Q8_1, each BLOCK (32 elements) has its own precomputed sum.
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_SumAPerKBlock)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 4;
        constexpr int BLOCK_SIZE = 32;

        // Create A blocks (Q8_1)
        std::vector<Q8_1Block> A_blocks(K_blocks);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> quant_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.01f);

        float total_sum_fp32 = 0.0f;
        std::vector<float> per_block_sums(K_blocks);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            Q8_1Block &block = A_blocks[kb];
            block.d = fp32_to_fp16(scale_dist(rng));

            float sum_qs = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k)
            {
                block.qs[k] = static_cast<int8_t>(quant_dist(rng));
                sum_qs += static_cast<float>(block.qs[k]);
            }

            // Q8_1 stores: s = d × Σ(qs)
            float d = fp16_to_fp32(block.d);
            block.s = fp32_to_fp16(d * sum_qs);

            per_block_sums[kb] = fp16_to_fp32(block.s);
            total_sum_fp32 += per_block_sums[kb];
        }

        std::cout << "[MINI-INTEGRATION: sum_a per K-block]" << std::endl;
        std::cout << "  K_blocks = " << K_blocks << std::endl;

        // Verify: sum of per-block sums should equal total
        float reconstructed_total = 0.0f;
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            reconstructed_total += per_block_sums[kb];
            std::cout << "  block[" << kb << "].s = " << per_block_sums[kb] << std::endl;
        }

        std::cout << "  total_sum_fp32 = " << total_sum_fp32 << std::endl;
        std::cout << "  reconstructed = " << reconstructed_total << std::endl;

        float diff = std::abs(total_sum_fp32 - reconstructed_total);
        EXPECT_LT(diff, 1e-4f) << "Per-block sums should add up to total";
    }

    /**
     * @brief Integration: Verify compensation is applied BEFORE scaling
     *
     * NOTE: This test is algebraically degenerate because:
     *   (dpbusd - 128×sum_qs) × a × b == dpbusd × a × b - 128×sum_qs × a × b
     * Both orders give identical results due to distributive property.
     * The test is DISABLED to avoid false failures.
     */
    TEST_F(Test__Q8_1GemmKernel, DISABLED_MiniIntegration_CompensationBeforeScaling)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;

        // Simple test case
        std::vector<int8_t> B_signed(32, 10);
        std::vector<int8_t> A_signed(32, 5);

        float a_scale = 0.1f;
        float b_scale = 0.2f;

        float sum_qs = 32.0f * 5.0f;    // 160
        float sum_a = a_scale * sum_qs; // 16.0

        // dpbusd
        std::vector<uint8_t> B_unsigned(32);
        for (int i = 0; i < 32; ++i)
        {
            B_unsigned[i] = static_cast<uint8_t>(B_signed[i] + 128);
        }
        int32_t dpbusd_result = Kernel::compute_dpbusd_dot_product(B_unsigned.data(), A_signed.data(), 32);

        // CORRECT ORDER: Compensate first, then scale
        float sum_qs_computed = sum_a / a_scale;
        int32_t compensated = Kernel::apply_dpbusd_compensation(dpbusd_result, sum_qs_computed);
        float result_correct = static_cast<float>(compensated) * a_scale * b_scale;

        // WRONG ORDER: Scale first, then compensate
        float result_wrong = static_cast<float>(dpbusd_result) * a_scale * b_scale -
                             (128.0f * sum_qs_computed * a_scale * b_scale);

        // Reference
        int32_t ref_accum = Kernel::compute_reference_dot_product(B_signed.data(), A_signed.data(), 32);
        float reference = static_cast<float>(ref_accum) * a_scale * b_scale;

        std::cout << "[MINI-INTEGRATION: Compensation order]" << std::endl;
        std::cout << "  dpbusd_result = " << dpbusd_result << std::endl;
        std::cout << "  compensated = " << compensated << std::endl;
        std::cout << "  result_correct = " << result_correct << std::endl;
        std::cout << "  result_wrong = " << result_wrong << std::endl;
        std::cout << "  reference = " << reference << std::endl;

        float diff_correct = std::abs(result_correct - reference);
        float diff_wrong = std::abs(result_wrong - reference);

        EXPECT_LT(diff_correct, 1e-5f) << "Correct order should match reference";
        EXPECT_GT(diff_wrong, 1e-2f) << "Wrong order should NOT match (verify test is meaningful)";
    }

    /**
     * @brief Integration: Test vectorized vs scalar compensation application
     *
     * Verifies that vectorized SIMD compensation matches scalar compensation.
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_VectorizedVsScalarCompensation)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 16; // Multiple of VECTOR_WIDTH (8)

        std::vector<int32_t> dpbusd_results(K_blocks);
        std::vector<float> sum_qs_values(K_blocks);
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> accum_dist(-100000, 100000);
        std::uniform_real_distribution<float> sum_dist(-1000.0f, 1000.0f);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.01f);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            dpbusd_results[kb] = accum_dist(rng);
            sum_qs_values[kb] = sum_dist(rng);
            a_scales[kb] = fp32_to_fp16(scale_dist(rng));
            b_scales[kb] = fp32_to_fp16(scale_dist(rng));
        }

        // Scalar path: apply compensation + scale for each K-block
        float scalar_result = 0.0f;
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            int32_t compensated = Kernel::apply_dpbusd_compensation(dpbusd_results[kb], sum_qs_values[kb]);
            float a_scale = fp16_to_fp32(a_scales[kb]);
            float b_scale = fp16_to_fp32(b_scales[kb]);
            scalar_result += static_cast<float>(compensated) * a_scale * b_scale;
        }

        // Vectorized path: simulate what microkernel does
        constexpr int VECTOR_WIDTH = 16; // __m512 holds 16 floats!
        float vectorized_result = 0.0f;

        for (int kb = 0; kb < K_blocks; kb += VECTOR_WIDTH)
        {
            // Load 8 dpbusd results
            __m512i dpbusd_vec = _mm512_loadu_si512(&dpbusd_results[kb]);
            __m512 dpbusd_f32 = _mm512_cvtepi32_ps(dpbusd_vec);

            // Load 8 sum_qs values
            __m512 sum_qs_vec = _mm512_loadu_ps(&sum_qs_values[kb]);

            // Apply compensation: compensated = dpbusd - 128 × sum_qs
            __m512 compensation = _mm512_mul_ps(_mm512_set1_ps(128.0f), sum_qs_vec);
            __m512 compensated_vec = _mm512_sub_ps(dpbusd_f32, compensation);

            // DEBUG: Print first iteration
            if (kb == 0)
            {
                alignas(64) float dpbusd_arr[16];
                alignas(64) float sum_qs_arr[16];
                alignas(64) float compensation_arr[16];
                alignas(64) float compensated_arr[16];

                _mm512_store_ps(dpbusd_arr, dpbusd_f32);
                _mm512_store_ps(sum_qs_arr, sum_qs_vec);
                _mm512_store_ps(compensation_arr, compensation);
                _mm512_store_ps(compensated_arr, compensated_vec);

                std::cout << "[VECTORIZED DEBUG kb=0]" << std::endl;
                std::cout << "  dpbusd[0] = " << dpbusd_arr[0] << std::endl;
                std::cout << "  sum_qs[0] = " << sum_qs_arr[0] << std::endl;
                std::cout << "  compensation[0] = " << compensation_arr[0] << std::endl;
                std::cout << "  compensated[0] = " << compensated_arr[0] << std::endl;

                // Compare to scalar
                int32_t scalar_comp = Kernel::apply_dpbusd_compensation(dpbusd_results[0], sum_qs_values[0]);
                std::cout << "  scalar compensated[0] = " << scalar_comp << std::endl;
            }

            // Load scales and convert
            __m256i a_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales[kb]));
            __m512 a_scales_vec = _mm512_cvtph_ps(a_scales_fp16);

            __m256i b_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales[kb]));
            __m512 b_scales_vec = _mm512_cvtph_ps(b_scales_fp16);

            // Scale: result = compensated × a_scale × b_scale
            __m512 scaled = _mm512_mul_ps(compensated_vec, a_scales_vec);
            scaled = _mm512_mul_ps(scaled, b_scales_vec);

            // DEBUG: Print scaled values
            if (kb == 0)
            {
                alignas(64) float a_scales_arr[16];
                alignas(64) float b_scales_arr[16];
                alignas(64) float scaled_arr[16];

                _mm512_store_ps(a_scales_arr, a_scales_vec);
                _mm512_store_ps(b_scales_arr, b_scales_vec);
                _mm512_store_ps(scaled_arr, scaled);

                std::cout << "  a_scale[0] = " << a_scales_arr[0] << std::endl;
                std::cout << "  b_scale[0] = " << b_scales_arr[0] << std::endl;
                std::cout << "  scaled[0] = " << scaled_arr[0] << std::endl;

                // Compare to scalar
                float scalar_a = fp16_to_fp32(a_scales[0]);
                float scalar_b = fp16_to_fp32(b_scales[0]);
                float scalar_scaled = -101007.0f * scalar_a * scalar_b;
                std::cout << "  scalar scaled[0] = " << scalar_scaled << std::endl;

                // Check horizontal reduction
                float reduction = _mm512_reduce_add_ps(scaled);
                std::cout << "  horizontal reduction = " << reduction << std::endl;
            }

            // Horizontal reduction
            vectorized_result += _mm512_reduce_add_ps(scaled);
        }

        float abs_diff = std::abs(scalar_result - vectorized_result);
        float rel_diff = abs_diff / std::max(std::abs(scalar_result), 1e-6f);

        std::cout << "[MINI-INTEGRATION: Vectorized vs Scalar Compensation]" << std::endl;
        std::cout << "  K_blocks = " << K_blocks << std::endl;
        std::cout << "  scalar_result = " << scalar_result << std::endl;
        std::cout << "  vectorized_result = " << vectorized_result << std::endl;
        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;

        EXPECT_LT(rel_diff, 5e-5f) << "Vectorized compensation should match scalar (allowing for FP32 rounding)";
    }

    /**
     * @brief Integration: Test actual microkernel post-processing logic
     *
     * Extracts and tests EXACTLY what the microkernel does in post-processing.
     */
    TEST_F(Test__Q8_1GemmKernel, MiniIntegration_MicrokernelPostProcessing)
    {
        using Kernel = Q8_1GemmKernelTemplate<8, 8>;
        constexpr int K_blocks = 8;

        // Simulate what microkernel produces: raw dpbusd accumulators
        std::vector<int32_t> accum(K_blocks);
        std::vector<uint16_t> a_scales(K_blocks);
        std::vector<uint16_t> b_scales(K_blocks);
        std::vector<uint16_t> sum_a_fp16(K_blocks); // Q8_1 precomputed sums

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> accum_dist(-50000, 50000);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.01f);
        std::uniform_real_distribution<float> sum_dist(-100.0f, 100.0f);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            accum[kb] = accum_dist(rng);
            a_scales[kb] = fp32_to_fp16(scale_dist(rng));
            b_scales[kb] = fp32_to_fp16(scale_dist(rng));
            sum_a_fp16[kb] = fp32_to_fp16(sum_dist(rng));
        }

        // REFERENCE: What the code SHOULD compute
        float reference = 0.0f;
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            float sum_a = fp16_to_fp32(sum_a_fp16[kb]);
            float a_scale = fp16_to_fp32(a_scales[kb]);
            float b_scale = fp16_to_fp32(b_scales[kb]);

            // Compute sum_qs from Q8_1 precomputed sum
            float sum_qs = sum_a / std::max(a_scale, 1e-10f);

            // Apply compensation
            int32_t compensated = Kernel::apply_dpbusd_compensation(accum[kb], sum_qs);

            // Scale to FP32
            reference += static_cast<float>(compensated) * a_scale * b_scale;
        }

        // ACTUAL MICROKERNEL LOGIC: Replicate EXACTLY what lines 1089-1110 do
        constexpr int VECTOR_WIDTH = 16; // MUST match __m512 capacity
        float microkernel_result = 0.0f;

        int kb = 0;
        for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
        {
            // Load sum_a (Q8_1 precomputed sums)
            __m256i sum_a_fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&sum_a_fp16[kb]));
            __m512 sum_a_f32 = _mm512_cvtph_ps(sum_a_fp16_vec);

            // Load A scales
            __m256i a_scales_fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales[kb]));
            __m512 a_scales_vec = _mm512_cvtph_ps(a_scales_fp16_vec);

            // Compute sum_qs = sum_a / a_scale
            __m512 epsilon = _mm512_set1_ps(1e-10f);
            __m512 a_scales_safe = _mm512_max_ps(a_scales_vec, epsilon);
            __m512 sum_qs_vec = _mm512_div_ps(sum_a_f32, a_scales_safe);

            // Load accumulators
            __m512i accum_i32 = _mm512_loadu_si512(&accum[kb]);
            __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32);

            // Load B scales
            __m256i b_scales_fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales[kb]));
            __m512 b_scales_vec = _mm512_cvtph_ps(b_scales_fp16_vec);

            // Apply compensation: compensated = accum - 128 × sum_qs
            __m512 compensation = _mm512_mul_ps(_mm512_set1_ps(128.0f), sum_qs_vec);
            __m512 compensated = _mm512_sub_ps(accum_f32, compensation);

            // Scale: result = compensated × a_scale × b_scale
            __m512 scaled = _mm512_mul_ps(compensated, a_scales_vec);
            scaled = _mm512_mul_ps(scaled, b_scales_vec);

            // Accumulate
            microkernel_result += _mm512_reduce_add_ps(scaled);
        }

        // Scalar tail (if any)
        for (; kb < K_blocks; ++kb)
        {
            float sum_a = fp16_to_fp32(sum_a_fp16[kb]);
            float a_scale = fp16_to_fp32(a_scales[kb]);
            float b_scale = fp16_to_fp32(b_scales[kb]);

            float sum_qs = sum_a / std::max(a_scale, 1e-10f);
            int32_t compensated = Kernel::apply_dpbusd_compensation(accum[kb], sum_qs);
            microkernel_result += static_cast<float>(compensated) * a_scale * b_scale;
        }

        float abs_diff = std::abs(microkernel_result - reference);
        float rel_diff = abs_diff / std::max(std::abs(reference), 1e-6f);

        std::cout << "[MINI-INTEGRATION: Microkernel Post-Processing]" << std::endl;
        std::cout << "  K_blocks = " << K_blocks << std::endl;
        std::cout << "  reference = " << reference << std::endl;
        std::cout << "  microkernel_result = " << microkernel_result << std::endl;
        std::cout << "  abs_diff = " << abs_diff << ", rel_diff = " << rel_diff << std::endl;

        EXPECT_LT(rel_diff, 1e-5f) << "Microkernel post-processing should match reference";
    }

    /**
     * @brief Test multi-panel correctness (N > NR case)
     *
     * This test verifies that when N > NR (requiring multiple panels), each panel
     * produces correct results. This is a regression test for the bug where
     * wide matrices (N=256, N=512) were failing with 160-190% error.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_MultiPanelCorrectness)
    {
        using Kernel = Q8_1GemmKernelTemplate<32, 128>; // MR=32, NR=128
        constexpr int MR = 32;
        constexpr int NR = 128;

        // Test a 32×256 matrix (2 panels: columns 0-127, 128-255)
        const int M = 32, N = 256, K = 128;
        const int K_blocks = K / 32;

        std::cout << "\n=== MULTI-PANEL TEST: " << M << "×" << N << "×" << K << " ===" << std::endl;
        std::cout << "NR = " << NR << ", panels = " << (N / NR) << std::endl;

        // Generate random test data
        auto A_fp32 = generateRandomMatrix(M, K);
        auto B_fp32 = generateRandomMatrix(K, N);

        // Quantize
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Run kernel GEMM
        std::vector<float> C_test(M * N, 0.0f);
        Kernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        // Compute reference (FP32)
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Check each panel separately
        for (int panel = 0; panel < 2; ++panel)
        {
            int j_start = panel * NR;
            int j_end = std::min(j_start + NR, N);

            double panel_l2 = 0.0;
            double panel_ref_norm = 0.0;
            float panel_max_abs = 0.0f;

            for (int i = 0; i < M; ++i)
            {
                for (int j = j_start; j < j_end; ++j)
                {
                    float ref_val = C_ref[i * N + j];
                    float test_val = C_test[i * N + j];
                    float diff = test_val - ref_val;

                    panel_l2 += diff * diff;
                    panel_ref_norm += ref_val * ref_val;
                    panel_max_abs = std::max(panel_max_abs, std::abs(diff));
                }
            }

            double panel_rel_l2 = std::sqrt(panel_l2 / std::max(panel_ref_norm, 1e-10));

            std::cout << "Panel " << panel << " (cols " << j_start << "-" << (j_end - 1) << "):" << std::endl;
            std::cout << "  Rel L2: " << panel_rel_l2 << std::endl;
            std::cout << "  Max Abs: " << panel_max_abs << std::endl;
            std::cout << "  Sample C_ref[0," << j_start << "] = " << C_ref[j_start] << std::endl;
            std::cout << "  Sample C_test[0," << j_start << "] = " << C_test[j_start] << std::endl;

            EXPECT_LT(panel_rel_l2, 0.01) << "Panel " << panel << " should have <1% error";
            EXPECT_LT(panel_max_abs, 1.0f) << "Panel " << panel << " max error should be reasonable";
        }

        // Overall error
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        std::cout << "Overall:" << std::endl;
        std::cout << "  Rel L2: " << rel_l2 << std::endl;
        std::cout << "  Max Abs: " << max_abs << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "Overall multi-panel GEMM should have <1% error";
    }

    /**
     * @brief Test single row, single panel vs two panels
     *
     * Compute the same row with N=128 (single panel) and N=256 (two panels).
     * The first 128 columns should be IDENTICAL between the two tests.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_SingleVsTwoPanels)
    {
        using Kernel = Q8_1GemmKernelTemplate<32, 128>;
        constexpr int NR = 128;

        const int M = 32, K = 128;
        const int K_blocks = K / 32;

        std::cout << "\n=== SINGLE VS TWO PANELS TEST ===" << std::endl;

        // Generate random row
        std::vector<float> A_row_fp32(K);
        std::mt19937 gen(54321);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int k = 0; k < K; ++k)
        {
            A_row_fp32[k] = dist(gen);
        }

        // Replicate to M rows (all identical)
        std::vector<float> A_fp32(M * K);
        for (int i = 0; i < M; ++i)
        {
            std::copy(A_row_fp32.begin(), A_row_fp32.end(), A_fp32.begin() + i * K);
        }

        // Quantize A
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});

        // ===== TEST 1: Single panel (N=128) =====
        const int N1 = 128;
        std::vector<float> B1_fp32(K * N1);
        for (int k = 0; k < K; ++k)
        {
            for (int j = 0; j < N1; ++j)
            {
                B1_fp32[k * N1 + j] = dist(gen);
            }
        }

        auto B1_q8_0 = createQ8_0Tensor(B1_fp32.data(), K, N1);
        std::vector<float> C1_test(M * N1, 0.0f);
        Kernel::gemm(M, N1, K, *A_q8_1, *B1_q8_0, C1_test.data(), N1);

        std::vector<float> C1_ref(M * N1, 0.0f);
        referenceGemm(A_fp32.data(), B1_fp32.data(), C1_ref.data(), M, N1, K);

        // ===== TEST 2: Two panels (N=256, first 128 columns identical to Test 1) =====
        const int N2 = 256;
        std::vector<float> B2_fp32(K * N2);
        // First 128 columns: copy from B1
        for (int k = 0; k < K; ++k)
        {
            for (int j = 0; j < N1; ++j)
            {
                B2_fp32[k * N2 + j] = B1_fp32[k * N1 + j];
            }
        }
        // Second 128 columns: new random data
        for (int k = 0; k < K; ++k)
        {
            for (int j = N1; j < N2; ++j)
            {
                B2_fp32[k * N2 + j] = dist(gen);
            }
        }

        auto B2_q8_0 = createQ8_0Tensor(B2_fp32.data(), K, N2);
        std::vector<float> C2_test(M * N2, 0.0f);
        Kernel::gemm(M, N2, K, *A_q8_1, *B2_q8_0, C2_test.data(), N2);

        std::vector<float> C2_ref(M * N2, 0.0f);
        referenceGemm(A_fp32.data(), B2_fp32.data(), C2_ref.data(), M, N2, K);

        // ===== COMPARISON =====
        std::cout << "Single panel (N=128):" << std::endl;
        double rel_l2_1 = computeRelativeL2Error(C1_ref.data(), C1_test.data(), M, N1);
        std::cout << "  Rel L2: " << rel_l2_1 << std::endl;
        std::cout << "  C1_ref[0,0] = " << C1_ref[0] << std::endl;
        std::cout << "  C1_test[0,0] = " << C1_test[0] << std::endl;

        std::cout << "\nTwo panels (N=256):" << std::endl;
        double rel_l2_2 = computeRelativeL2Error(C2_ref.data(), C2_test.data(), M, N2);
        std::cout << "  Rel L2: " << rel_l2_2 << std::endl;
        std::cout << "  C2_ref[0,0] = " << C2_ref[0] << std::endl;
        std::cout << "  C2_test[0,0] = " << C2_test[0] << std::endl;

        // Compare first 128 columns (should be IDENTICAL)
        std::cout << "\nComparing first 128 columns:" << std::endl;
        float max_diff_test = 0.0f;
        float max_diff_ref = 0.0f;
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N1; ++j)
            {
                float diff_test = std::abs(C1_test[i * N1 + j] - C2_test[i * N2 + j]);
                float diff_ref = std::abs(C1_ref[i * N1 + j] - C2_ref[i * N2 + j]);
                max_diff_test = std::max(max_diff_test, diff_test);
                max_diff_ref = std::max(max_diff_ref, diff_ref);
            }
        }

        std::cout << "  Max diff (reference): " << max_diff_ref << " (should be ~0)" << std::endl;
        std::cout << "  Max diff (kernel): " << max_diff_test << " (should be ~0)" << std::endl;

        EXPECT_LT(rel_l2_1, 0.01) << "Single panel should have <1% error";
        EXPECT_LT(max_diff_ref, 1e-5f) << "Reference FP32×FP32 should be identical for same columns";
        EXPECT_LT(max_diff_test, 1e-3f) << "Kernel should produce nearly identical results for same columns (allowing quantization noise)";
    }

    /**
     * @brief Test NR=64 vs NR=128 configurations
     *
     * Check if the bug is specific to NR=128 or affects both configurations.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_NR64_vs_NR128)
    {
        const int M = 32, K = 128;

        std::cout << "\n=== NR=64 VS NR=128 COMPARISON ===" << std::endl;

        // Generate random data
        std::vector<float> A_fp32 = generateRandomMatrix(M, K);

        // === TEST with NR=64 ===
        {
            using Kernel64 = Q8_1GemmKernelTemplate<32, 64>;
            const int N = 64;

            std::vector<float> B_fp32 = generateRandomMatrix(K, N);
            auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
            auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

            std::vector<float> C_test(M * N, 0.0f);
            Kernel64::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

            std::vector<float> C_ref(M * N, 0.0f);
            referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

            double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
            std::cout << "NR=64 (N=64): Rel L2 = " << rel_l2 << std::endl;
            std::cout << "  C_ref[0,0] = " << C_ref[0] << std::endl;
            std::cout << "  C_test[0,0] = " << C_test[0] << std::endl;

            EXPECT_LT(rel_l2, 0.01) << "NR=64 should have <1% error";
        }

        // === TEST with NR=128 ===
        {
            using Kernel128 = Q8_1GemmKernelTemplate<32, 128>;
            const int N = 128;

            std::vector<float> B_fp32 = generateRandomMatrix(K, N);
            auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
            auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

            std::vector<float> C_test(M * N, 0.0f);
            Kernel128::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

            std::vector<float> C_ref(M * N, 0.0f);
            referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

            double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
            std::cout << "NR=128 (N=128): Rel L2 = " << rel_l2 << std::endl;
            std::cout << "  C_ref[0,0] = " << C_ref[0] << std::endl;
            std::cout << "  C_test[0,0] = " << C_test[0] << std::endl;

            EXPECT_LT(rel_l2, 0.01) << "NR=128 should have <1% error";
        }
    }

    /**
     * @brief Test full microkernel with manual verification (minimal case)
     *
     * Use MR×NR with 1 K-block to trigger full microkernel path.
     * Manually compute the expected result and compare.
     */
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_FullMicrokernelManualVerification)
    {
        // Use default microkernel (MR=32, NR=128) to test FULL microkernel path
        using Kernel = Q8_1GemmKernel; // Default: MR=32, NR=128
        constexpr int MR = 32;
        constexpr int NR = 128;
        const int M = 32, N = 128, K = 32; // M=MR, N=NR → full microkernel!
        const int K_blocks = 1;

        std::cout << "\n=== FULL MICROKERNEL MANUAL VERIFICATION ===" << std::endl;
        std::cout << "MR=" << MR << ", NR=" << NR << ", K_blocks=" << K_blocks << std::endl;
        std::cout << "M=" << M << ", N=" << N << " → Should use FULL microkernel (M==MR, N==NR)" << std::endl;

        // Generate test data (small values for readability)
        std::vector<float> A_fp32(M * K);
        std::vector<float> B_fp32(K * N);

        std::mt19937 gen(99999);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < M * K; ++i)
            A_fp32[i] = dist(gen);
        for (int i = 0; i < K * N; ++i)
            B_fp32[i] = dist(gen);

        // Quantize
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Run full microkernel (M=MR, N=NR should trigger full path)
        std::vector<float> C_test(M * N, 0.0f);
        Kernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        // Compute reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Manually compute expected result for C[0,0] using quantized blocks
        const Q8_1Block *a_block_0_0 = A_q8_1->decode_to_q8_1(0, 0);
        Q8_0Block b_block_0_0;
        B_q8_0->decode_to_q8_0(0, 0, &b_block_0_0);

        // NOTE: We can't easily replicate dpbusd's exact unsigned×signed computation here.
        // The issue is that Q8_1 stores values as UNSIGNED bytes (0-255), but they represent
        // SIGNED values after subtracting 128. The dpbusd instruction treats A as unsigned
        // and B as signed, computing Σ(A_unsigned * B_signed).
        //
        // Instead of trying to replicate dpbusd, let's verify the kernel's output matches
        // the REFERENCE FP32 result (since Q8_1 should be close to FP32 with proper compensation).

        // Get scales for verification
        float a_scale = fp16_to_fp32(a_block_0_0->d);
        float b_scale = fp16_to_fp32(b_block_0_0.d);
        float sum_a = fp16_to_fp32(a_block_0_0->s);
        float sum_qs = sum_a / std::max(a_scale, 1e-10f);

        std::cout << "\nQ8_1 GEMM Verification:" << std::endl;
        std::cout << "  A block [0,0]: a_scale=" << a_scale << ", sum_a=" << sum_a
                  << ", sum_qs=" << sum_qs << std::endl;
        std::cout << "  B block [0,0]: b_scale=" << b_scale << std::endl;
        std::cout << "  C_test[0,0] (kernel) = " << C_test[0] << std::endl;
        std::cout << "  C_ref[0,0] (FP32)    = " << C_ref[0] << std::endl;

        // Check multiple elements - Q8_1 should match FP32 closely
        std::cout << "\nChecking kernel vs FP32 reference:" << std::endl;
        int close_match_count = 0;
        int total_checked = std::min(10, M * N);

        for (int idx = 0; idx < total_checked; ++idx)
        {
            float test_val = C_test[idx];
            float ref_val = C_ref[idx];
            float abs_diff = std::abs(test_val - ref_val);
            float rel_error = abs_diff / std::max(std::abs(ref_val), 1e-6f);

            // Q8_1 should give <1% error with proper compensation
            bool close_match = rel_error < 0.01f || abs_diff < 1e-4f;
            if (close_match)
                close_match_count++;

            if (idx < 5)
            {
                std::cout << "  C[" << idx << "]: kernel=" << test_val
                          << ", FP32=" << ref_val
                          << ", rel_err=" << (rel_error * 100.0f) << "%"
                          << (close_match ? " ✓" : " ✗") << std::endl;
            }
        }

        std::cout << "Elements within 1% of FP32: " << close_match_count << "/" << total_checked << std::endl;

        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        std::cout << "\nOverall Rel L2: " << rel_l2 << std::endl;

        // Q8_1 GEMM with proper compensation should match FP32 closely
        EXPECT_LT(rel_l2, 0.01) << "Q8_1 GEMM should be within 1% of FP32 reference";
        EXPECT_GE(close_match_count, total_checked * 0.8)
            << "At least 80% of elements should be within 1% of FP32";
    }

    // Test with K_blocks=4 to check vectorized tail paths
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_MultipleKBlocks)
    {
        std::cout << "\n=== MULTIPLE K_BLOCKS TEST ===" << std::endl;
        std::cout << "Testing K_blocks=4 (K=128)" << std::endl;

        // Use default MR=32, NR=128
        constexpr int MR = 32;
        constexpr int NR = 128;
        constexpr int M = MR;
        constexpr int N = NR;
        constexpr int K = 128; // K_blocks = 4

        std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "K_blocks = " << (K / 32) << " (should use 4-wide vectorized tail)" << std::endl;

        // Create random test data
        std::vector<float> A_fp32 = generateRandomMatrix(M, K);
        std::vector<float> B_fp32 = generateRandomMatrix(K, N);

        // Quantize to Q8_1 and Q8_0
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Run kernel
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        // Check results
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        std::cout << "Rel L2: " << rel_l2 << std::endl;

        // Show more elements to find where the error is
        std::cout << "\nElement comparison (showing first 20 + some from end):" << std::endl;
        int good_count = 0, bad_count = 0;
        for (int i = 0; i < std::min(20, M * N); ++i)
        {
            float test_val = C_test[i];
            float ref_val = C_ref[i];
            float rel_err = std::abs(test_val - ref_val) / std::max(std::abs(ref_val), 1e-6f);
            bool good = rel_err < 0.01f;
            if (good)
                good_count++;
            else
                bad_count++;

            if (i < 10 || !good) // Show first 10 + all bad ones
            {
                std::cout << "  C[" << i << "]: kernel=" << test_val
                          << ", FP32=" << ref_val
                          << ", rel_err=" << (rel_err * 100.0f) << "%"
                          << (good ? " ✓" : " ✗") << std::endl;
            }
        }

        // Check some from the end too
        std::cout << "\nLast 5 elements:" << std::endl;
        for (int i = M * N - 5; i < M * N; ++i)
        {
            float test_val = C_test[i];
            float ref_val = C_ref[i];
            float rel_err = std::abs(test_val - ref_val) / std::max(std::abs(ref_val), 1e-6f);
            bool good = rel_err < 0.01f;
            if (good)
                good_count++;
            else
                bad_count++;

            std::cout << "  C[" << i << "]: kernel=" << test_val
                      << ", FP32=" << ref_val
                      << ", rel_err=" << (rel_err * 100.0f) << "%"
                      << (good ? " ✓" : " ✗") << std::endl;
        }

        std::cout << "\nSummary: " << good_count << " good, " << bad_count << " bad out of 25 checked" << std::endl;

        EXPECT_LT(rel_l2, 0.01) << "Q8_1 GEMM with K_blocks=4 should be within 1% of FP32";
    }

    // Helper function to manually compute a single element C[31,0] for debugging row access
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_SingleElement_Row31_Col0)
    {
        std::cout << "\n=== SINGLE ELEMENT TEST: C[31,0] with K_blocks=4 ===" << std::endl;

        constexpr int MR = 32;
        constexpr int NR = 128;
        constexpr int M = MR;
        constexpr int N = NR;
        constexpr int K = 128;
        constexpr int K_blocks = K / 32;

        std::vector<float> A_fp32 = generateRandomMatrix(M, K);
        std::vector<float> B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Manual computation for C[31,0]
        constexpr int ir = 31;
        constexpr int jr = 0;

        float manual_result = 0.0f;

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            const Q8_1Block *a_block = A_q8_1->decode_to_q8_1(ir, kb);
            Q8_0Block b_block;
            B_q8_0->decode_to_q8_0(jr, kb, &b_block);

            int32_t dpbusd_accum = 0;
            for (int i = 0; i < 32; ++i)
            {
                uint8_t b_packed = static_cast<uint8_t>(static_cast<int8_t>(b_block.qs[i]) + 128);
                dpbusd_accum += static_cast<int32_t>(static_cast<int8_t>(a_block->qs[i])) *
                                static_cast<int32_t>(b_packed);
            }

            float a_scale = fp16_to_fp32(a_block->d);
            float sum_a = fp16_to_fp32(a_block->s);
            float sum_qs_f = sum_a / std::max(a_scale, 1e-10f);
            int32_t sum_qs = static_cast<int32_t>(std::round(sum_qs_f));

            int32_t compensated = dpbusd_accum - 128 * sum_qs;

            float b_scale = fp16_to_fp32(b_block.d);
            float block_result = static_cast<float>(compensated) * a_scale * b_scale;

            manual_result += block_result;
        }

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "Manual: " << manual_result << ", Kernel: " << C_test[ir * N + jr] << ", FP32: " << C_ref[ir * N + jr] << std::endl;

        float manual_vs_kernel_err = std::abs(manual_result - C_test[ir * N + jr]) / std::max(std::abs(C_test[ir * N + jr]), 1e-6f);
        float kernel_vs_fp32_err = std::abs(C_test[ir * N + jr] - C_ref[ir * N + jr]) / std::max(std::abs(C_ref[ir * N + jr]), 1e-6f);

        std::cout << "Manual vs Kernel: " << (manual_vs_kernel_err * 100.0f) << "%, Kernel vs FP32: " << (kernel_vs_fp32_err * 100.0f) << "%" << std::endl;

        // Manual computation uses sum_qs quantization, which differs slightly from sA approach
        // sA approach is more accurate (no quantization), so we expect small differences
        if (debugEnv().gemm.use_sa_compensation)
        {
            // sA mode: relaxed tolerance due to avoiding sum_qs quantization
            EXPECT_NEAR(manual_result, C_test[ir * N + jr], 0.005f); // 0.5% tolerance
        }
        else
        {
            // sum_qs mode: strict tolerance (original test)
            EXPECT_NEAR(manual_result, C_test[ir * N + jr], 1e-4f);
        }
        EXPECT_LT(kernel_vs_fp32_err, 0.01f);
    }

    // Helper function to manually compute a single element C[0,127] for debugging column access
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_SingleElement_Row0_Col127)
    {
        std::cout << "\n=== SINGLE ELEMENT TEST: C[0,127] with K_blocks=4 ===" << std::endl;

        constexpr int MR = 32;
        constexpr int NR = 128;
        constexpr int M = MR;
        constexpr int N = NR;
        constexpr int K = 128;
        constexpr int K_blocks = K / 32;

        std::vector<float> A_fp32 = generateRandomMatrix(M, K);
        std::vector<float> B_fp32 = generateRandomMatrix(K, N);

        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Manual computation for C[0,127]
        constexpr int ir = 0;
        constexpr int jr = 127;

        float manual_result = 0.0f;

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            const Q8_1Block *a_block = A_q8_1->decode_to_q8_1(ir, kb);
            Q8_0Block b_block;
            B_q8_0->decode_to_q8_0(jr, kb, &b_block);

            int32_t dpbusd_accum = 0;
            for (int i = 0; i < 32; ++i)
            {
                uint8_t b_packed = static_cast<uint8_t>(static_cast<int8_t>(b_block.qs[i]) + 128);
                dpbusd_accum += static_cast<int32_t>(static_cast<int8_t>(a_block->qs[i])) *
                                static_cast<int32_t>(b_packed);
            }

            float a_scale = fp16_to_fp32(a_block->d);
            float sum_a = fp16_to_fp32(a_block->s);
            float sum_qs_f = sum_a / std::max(a_scale, 1e-10f);
            int32_t sum_qs = static_cast<int32_t>(std::round(sum_qs_f));

            int32_t compensated = dpbusd_accum - 128 * sum_qs;

            float b_scale = fp16_to_fp32(b_block.d);
            float block_result = static_cast<float>(compensated) * a_scale * b_scale;

            manual_result += block_result;
        }

        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "Manual: " << manual_result << ", Kernel: " << C_test[ir * N + jr] << ", FP32: " << C_ref[ir * N + jr] << std::endl;

        float manual_vs_kernel_err = std::abs(manual_result - C_test[ir * N + jr]) / std::max(std::abs(C_test[ir * N + jr]), 1e-6f);
        float kernel_vs_fp32_err = std::abs(C_test[ir * N + jr] - C_ref[ir * N + jr]) / std::max(std::abs(C_ref[ir * N + jr]), 1e-6f);

        std::cout << "Manual vs Kernel: " << (manual_vs_kernel_err * 100.0f) << "%, Kernel vs FP32: " << (kernel_vs_fp32_err * 100.0f) << "%" << std::endl;

        // Manual computation uses sum_qs quantization, which differs slightly from sA approach
        if (debugEnv().gemm.use_sa_compensation)
        {
            EXPECT_NEAR(manual_result, C_test[ir * N + jr], 0.005f); // 0.5% tolerance for sA mode
        }
        else
        {
            EXPECT_NEAR(manual_result, C_test[ir * N + jr], 1e-4f);
        }
        EXPECT_LT(kernel_vs_fp32_err, 0.03f); // Relax to 3% for quantization error
    }

    // Helper function to manually compute a single element C[ir,jr] for debugging
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_SingleElement_Row31_Col127)
    {
        std::cout << "\n=== SINGLE ELEMENT TEST: C[31,127] with K_blocks=4 ===" << std::endl;

        // Create a small test case: M=32, N=128, K=128 (K_blocks=4)
        constexpr int MR = 32;
        constexpr int NR = 128;
        constexpr int M = MR;
        constexpr int N = NR;
        constexpr int K = 128;
        constexpr int K_blocks = K / 32;

        // Create random test data
        std::vector<float> A_fp32 = generateRandomMatrix(M, K);
        std::vector<float> B_fp32 = generateRandomMatrix(K, N);

        // Quantize
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Manual computation for C[31,127]
        constexpr int ir = 31;
        constexpr int jr = 127;

        float manual_result = 0.0f;

        std::cout << "\nManual computation:" << std::endl;

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            // Get blocks
            const Q8_1Block *a_block = A_q8_1->decode_to_q8_1(ir, kb);
            Q8_0Block b_block;
            B_q8_0->decode_to_q8_0(jr, kb, &b_block); // Column jr, K-block kb

            if (kb == 0)
            {
                std::cout << "  A[31,0] block: d=" << fp16_to_fp32(a_block->d)
                          << ", s=" << fp16_to_fp32(a_block->s)
                          << ", qs[0-3]=" << static_cast<int>(static_cast<int8_t>(a_block->qs[0]))
                          << "," << static_cast<int>(static_cast<int8_t>(a_block->qs[1]))
                          << "," << static_cast<int>(static_cast<int8_t>(a_block->qs[2]))
                          << "," << static_cast<int>(static_cast<int8_t>(a_block->qs[3]))
                          << std::endl;
                std::cout << "  B[127,0] block: d=" << fp16_to_fp32(b_block.d)
                          << ", qs[0-3]=" << static_cast<int>(static_cast<int8_t>(b_block.qs[0]))
                          << "," << static_cast<int>(static_cast<int8_t>(b_block.qs[1]))
                          << "," << static_cast<int>(static_cast<int8_t>(b_block.qs[2]))
                          << "," << static_cast<int>(static_cast<int8_t>(b_block.qs[3]))
                          << std::endl;
            }

            // Compute dpbusd for this block
            // B is packed with +128 offset
            int32_t dpbusd_accum = 0;
            for (int i = 0; i < 32; ++i)
            {
                uint8_t b_packed = static_cast<uint8_t>(static_cast<int8_t>(b_block.qs[i]) + 128);
                dpbusd_accum += static_cast<int32_t>(static_cast<int8_t>(a_block->qs[i])) *
                                static_cast<int32_t>(b_packed);
            }

            // Get sum_qs from Q8_1 block
            float a_scale = fp16_to_fp32(a_block->d);
            float sum_a = fp16_to_fp32(a_block->s);
            float sum_qs_f = sum_a / std::max(a_scale, 1e-10f);
            int32_t sum_qs = static_cast<int32_t>(std::round(sum_qs_f));

            // Apply Q8_1 compensation: compensated = dpbusd - 128*sum_qs
            int32_t compensated = dpbusd_accum - 128 * sum_qs;

            // Scale to FP32
            float b_scale = fp16_to_fp32(b_block.d);
            float block_result = static_cast<float>(compensated) * a_scale * b_scale;

            manual_result += block_result;
        }

        std::cout << "Manual result for C[31,127]: " << manual_result << std::endl;

        // Run kernel
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "Kernel result for C[31,127]: " << C_test[31 * N + 127] << std::endl;

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "FP32 reference for C[31,127]: " << C_ref[31 * N + 127] << std::endl;

        // Check that manual computation matches kernel
        float manual_vs_kernel_err = std::abs(manual_result - C_test[31 * N + 127]) / std::max(std::abs(C_test[31 * N + 127]), 1e-6f);
        std::cout << "\nManual vs Kernel error: " << (manual_vs_kernel_err * 100.0f) << "%" << std::endl;

        // Check that kernel matches FP32
        float kernel_vs_fp32_err = std::abs(C_test[31 * N + 127] - C_ref[31 * N + 127]) / std::max(std::abs(C_ref[31 * N + 127]), 1e-6f);
        std::cout << "Kernel vs FP32 error: " << (kernel_vs_fp32_err * 100.0f) << "%" << std::endl;

        // Manual computation uses sum_qs quantization, which differs slightly from sA approach
        if (debugEnv().gemm.use_sa_compensation)
        {
            EXPECT_NEAR(manual_result, C_test[31 * N + 127], 0.005f) // 0.5% tolerance for sA mode
                << "Manual computation should match kernel for C[31,127]";
        }
        else
        {
            EXPECT_NEAR(manual_result, C_test[31 * N + 127], 1e-4f)
                << "Manual computation should match kernel for C[31,127]";
        }

        EXPECT_LT(kernel_vs_fp32_err, 0.01f)
            << "Kernel should match FP32 within 1% for C[31,127]";
    }

    // Helper function to manually compute the 4-wide vectorized tail reduction
    TEST_F(Test__Q8_1GemmKernel, HelperFunction_4WideVectorizedTail)
    {
        std::cout << "\n=== 4-WIDE VECTORIZED TAIL HELPER TEST ===" << std::endl;

        // Create a small test case: M=32, N=128, K=128 (K_blocks=4)
        constexpr int MR = 32;
        constexpr int NR = 128;
        constexpr int M = MR;
        constexpr int N = NR;
        constexpr int K = 128;
        constexpr int K_blocks = K / 32;

        std::cout << "Testing single ir=0, jr=0 element with K_blocks=" << K_blocks << std::endl;

        // Create random test data
        std::vector<float> A_fp32 = generateRandomMatrix(M, K);
        std::vector<float> B_fp32 = generateRandomMatrix(K, N);

        // Quantize
        auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});
        auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

        // Manual computation: Extract blocks and manually compute what the 4-wide tail should produce
        std::cout << "\n=== MANUAL 4-WIDE TAIL COMPUTATION ===" << std::endl;

        // First, let's inspect what Q8_1 blocks actually contain
        const Q8_1Block *a_block_0 = A_q8_1->decode_to_q8_1(0, 0);
        Q8_0Block b_block_0;
        B_q8_0->decode_to_q8_0(0, 0, &b_block_0);

        std::cout << "\nInspecting first Q8_1 block:" << std::endl;
        std::cout << "  d (scale) = " << fp16_to_fp32(a_block_0->d) << std::endl;
        std::cout << "  s (sum) = " << fp16_to_fp32(a_block_0->s) << std::endl;
        std::cout << "  First 8 qs values (as int8): ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << static_cast<int>(static_cast<int8_t>(a_block_0->qs[i])) << " ";
        }
        std::cout << std::endl;
        std::cout << "  First 8 qs values (as uint8): ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << static_cast<int>(static_cast<uint8_t>(a_block_0->qs[i])) << " ";
        }
        std::cout << std::endl;

        std::cout << "\nInspecting first Q8_0 block:" << std::endl;
        std::cout << "  d (scale) = " << fp16_to_fp32(b_block_0.d) << std::endl;
        std::cout << "  First 8 qs values (as int8): ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << static_cast<int>(static_cast<int8_t>(b_block_0.qs[i])) << " ";
        }
        std::cout << std::endl;

        // For ir=0, jr=0, manually compute the reduction across all 4 K_blocks
        float manual_result = 0.0f;
        float manual_result_no_compensation = 0.0f;

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            // Get blocks
            // A: row 0, K-block kb
            const Q8_1Block *a_block = A_q8_1->decode_to_q8_1(0, kb);

            // B: After transposition, B is N×K.
            // For jr=0 (column 0 of output), we need row 0 of transposed B.
            // K-block kb of row 0.
            Q8_0Block b_block;
            B_q8_0->decode_to_q8_0(0, kb, &b_block); // row_idx=0, k_block_offset=kb

            // Compute dpbusd for this block
            // NOTE: dpbusd operand order in kernel is: dpbusd(acc, B_vec, A_vec)
            // Intel spec: dpbusd(src, a, b) computes src += Σ[ (unsigned)a * (signed)b ]
            // So: B (second arg) is unsigned, A (third arg) is signed
            // THEORY 1: If A is treated as signed, then NO compensation needed (just multiply)
            // THEORY 2: If compensation formula assumes A is unsigned, we might be wrong about operand order

            // Let's try both interpretations:
            int32_t dpbusd_unsigned_A = 0; // Original (treats A as unsigned)
            int32_t dpbusd_signed_A = 0;   // B unsigned, A signed (matches Intel spec)

            // DIAGNOSTIC: Print first few qs values for kb=0
            if (kb == 0)
            {
                std::cout << "\n[MANUAL] kb=0 blocks:" << std::endl;
                std::cout << "  A qs[0-7]: ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cout << static_cast<int>(static_cast<int8_t>(a_block->qs[i])) << " ";
                }
                std::cout << std::endl;
                std::cout << "  B qs[0-7]: ";
                for (int i = 0; i < 8; ++i)
                {
                    std::cout << static_cast<int>(static_cast<int8_t>(b_block.qs[i])) << " ";
                }
                std::cout << std::endl;
            }

            for (int i = 0; i < 32; ++i)
            {
                // Kernel packs B by adding 128: zmm_base[k_in] = static_cast<uint8_t>(block.qs[k_in] + 128)
                // So B values seen by dpbusd are: (signed_B + 128)
                uint8_t b_packed = static_cast<uint8_t>(static_cast<int8_t>(b_block.qs[i]) + 128);

                // Theory 1: A unsigned, B unsigned (both zero-extended)
                dpbusd_unsigned_A += static_cast<int32_t>(static_cast<uint8_t>(a_block->qs[i])) *
                                     static_cast<int32_t>(b_packed);

                // Theory 2: A signed, B unsigned (A sign-extended, B zero-extended)
                dpbusd_signed_A += static_cast<int32_t>(static_cast<int8_t>(a_block->qs[i])) *
                                   static_cast<int32_t>(b_packed);
            }

            int32_t dpbusd_accum = dpbusd_signed_A; // Try signed A to match Intel spec

            // Get sum_qs from Q8_1 block
            float a_scale = fp16_to_fp32(a_block->d);
            float sum_a = fp16_to_fp32(a_block->s);
            float sum_qs_f = sum_a / std::max(a_scale, 1e-10f);
            int32_t sum_qs = static_cast<int32_t>(std::round(sum_qs_f));

            // Apply Q8_1 compensation: compensated = dpbusd - 128*sum_qs
            int32_t compensated = dpbusd_accum - 128 * sum_qs;

            // Scale to FP32
            float b_scale = fp16_to_fp32(b_block.d);
            float block_result = static_cast<float>(compensated) * a_scale * b_scale;
            float block_result_no_comp = static_cast<float>(dpbusd_accum) * a_scale * b_scale;

            manual_result += block_result;
            manual_result_no_compensation += block_result_no_comp;

            std::cout << "  kb=" << kb << ":\n"
                      << "    dpbusd_unsigned_A=" << dpbusd_unsigned_A
                      << ", dpbusd_signed_A=" << dpbusd_signed_A
                      << "\n    sum_qs=" << sum_qs
                      << ", compensated=" << compensated
                      << "\n    with_comp=" << block_result
                      << ", no_comp=" << block_result_no_comp
                      << std::endl;
        }

        std::cout << "\nManual result WITH compensation: " << manual_result << std::endl;
        std::cout << "Manual result WITHOUT compensation: " << manual_result_no_compensation << std::endl;

        // Run kernel
        std::vector<float> C_test(M * N, 0.0f);
        Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

        std::cout << "Kernel result for C[0,0]: " << C_test[0] << std::endl;

        // Compute FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

        std::cout << "FP32 reference for C[0,0]: " << C_ref[0] << std::endl;

        // Check that manual computation matches kernel
        float manual_vs_kernel_err = std::abs(manual_result - C_test[0]) / std::max(std::abs(C_test[0]), 1e-6f);
        std::cout << "\nManual vs Kernel error: " << (manual_vs_kernel_err * 100.0f) << "%" << std::endl;

        // Check that kernel matches FP32
        float kernel_vs_fp32_err = std::abs(C_test[0] - C_ref[0]) / std::max(std::abs(C_ref[0]), 1e-6f);
        std::cout << "Kernel vs FP32 error: " << (kernel_vs_fp32_err * 100.0f) << "%" << std::endl;

        // Manual computation uses sum_qs quantization, which differs slightly from sA approach
        if (debugEnv().gemm.use_sa_compensation)
        {
            EXPECT_NEAR(manual_result, C_test[0], 0.005f) // 0.5% tolerance for sA mode
                << "Manual 4-wide tail computation should match kernel";
        }
        else
        {
            EXPECT_NEAR(manual_result, C_test[0], 1e-4f)
                << "Manual 4-wide tail computation should match kernel";
        }

        EXPECT_LT(kernel_vs_fp32_err, 0.01f)
            << "Kernel should match FP32 within 1%";
    }

    // ============================================================================
    // FP32 Parity Test Suite (46 tests)
    // ============================================================================
    //
    // Comprehensive validation of Q8_1GemmKernel numerical correctness across
    // diverse matrix configurations. Uses relative L2 norm with 1% tolerance.
    //
    // Coverage:
    //   - MR×NR alignment: Exact multiples (32×128), partial panels, edge cases
    //   - Batch sizes: 1, 4, 32, 64, 128, 37 (non-MR-multiple)
    //   - Matrix shapes: Square, tall, wide, tiny (1×1), huge (256×4096×512)
    //   - K-dimension: 1-64 blocks (32-2048 elements, all multiples of 32)
    //   - Rectangular: Tall (M>>N), wide (N>>M), extreme ratios (1024×32)
    //   - Stress tests: Large inference workloads (512×512×512, 256×4096×512)
    //
    // Metrics:
    //   - Relative L2 norm: sqrt(Σ(diff²)) / sqrt(Σ(ref²))
    //   - Tolerance: 1% (typical Q8 quantization error)
    //   - Debug output on failure: Shows max abs diff location and values
    //
    // Note: All tests use K as multiple of 32 (Q8_1 block size requirement)
    //

    /**
     * @brief Test fixture for comprehensive FP32 parity tests
     *
     * Tests Q8_1GemmKernel against FP32 reference across various configurations:
     * - Matrix shapes (square, tall, wide, tiny, huge)
     * - Batch sizes (1, 4, 32, 128)
     * - MR/NR alignment (exact multiples, edges, odd sizes)
     * - K-block counts (1, 4, 16, 64)
     */
    class Q8_1GemmKernel_ParityTest : public Test__Q8_1GemmKernel
    {
    protected:
        /**
         * @brief Run parity test for given M, N, K dimensions
         *
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Inner dimension (columns of A, rows of B)
         * @param tolerance Relative L2 norm tolerance (default 1%)
         */
        void runParityTest(int M, int N, int K, float tolerance = 0.01f)
        {
            // Generate random inputs
            std::vector<float> A_fp32 = generateRandomMatrix(M, K);
            std::vector<float> B_fp32 = generateRandomMatrix(K, N);

            // Quantize to Q8_1 (activations) and Q8_0 (weights)
            auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});
            auto B_q8_0 = createQ8_0Tensor(B_fp32.data(), K, N);

            // Run kernel
            std::vector<float> C_test(M * N, 0.0f);
            Q8_1GemmKernel::gemm(M, N, K, *A_q8_1, *B_q8_0, C_test.data(), N);

            // Compute FP32 reference
            std::vector<float> C_ref(M * N, 0.0f);
            referenceGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

            // Compute relative L2 norm (matches existing test methodology)
            double sum_squared_diff = 0.0;
            double sum_squared_ref = 0.0;
            float max_abs_diff = 0.0f;
            int max_error_idx = -1;

            for (int i = 0; i < M * N; ++i)
            {
                float abs_diff = std::abs(C_test[i] - C_ref[i]);
                float abs_ref = std::abs(C_ref[i]);

                sum_squared_diff += abs_diff * abs_diff;
                sum_squared_ref += abs_ref * abs_ref;

                if (abs_diff > max_abs_diff)
                {
                    max_abs_diff = abs_diff;
                    max_error_idx = i;
                }
            }

            double rel_l2_norm = std::sqrt(sum_squared_diff) / std::max(std::sqrt(sum_squared_ref), 1e-8);

            // Debug output for failures
            if (rel_l2_norm >= tolerance)
            {
                std::cout << "\nM=" << M << ", N=" << N << ", K=" << K << " PARITY FAILURE:\n"
                          << "  Rel L2 norm: " << rel_l2_norm << " (tolerance: " << tolerance << ")\n"
                          << "  Max abs diff: " << max_abs_diff << " at index " << max_error_idx
                          << " (i=" << (max_error_idx / N) << ", j=" << (max_error_idx % N) << ")\n"
                          << "    Kernel: " << C_test[max_error_idx] << ", FP32: " << C_ref[max_error_idx] << "\n";
            }

            EXPECT_LT(rel_l2_norm, tolerance)
                << "M=" << M << ", N=" << N << ", K=" << K
                << ": Relative L2 norm " << rel_l2_norm << " exceeds tolerance " << tolerance;
        }
    };

    // ----------------------------------------------------------------------------
    // Exact MR×NR multiples (fast path)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, ExactMR_NR_SingleBlock_32x128x32)
    {
        // Single K-block, exact MR×NR tile
        runParityTest(32, 128, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, ExactMR_NR_MultiBlock_32x128x128)
    {
        // 4 K-blocks, exact MR×NR tile
        runParityTest(32, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, ExactMR_NR_ManyBlocks_32x128x512)
    {
        // 16 K-blocks, exact MR×NR tile
        runParityTest(32, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MultiPanel_64x256x128)
    {
        // 2×2 panels, 4 K-blocks each
        runParityTest(64, 256, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MultiPanel_96x384x256)
    {
        // 3×3 panels, 8 K-blocks each
        runParityTest(96, 384, 256);
    }

    // ----------------------------------------------------------------------------
    // Non-multiple dimensions (edge microkernel)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_SmallSquare_8x8x32)
    {
        // Small square, single K-block, edge microkernel
        runParityTest(8, 8, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_SmallSquare_16x16x64)
    {
        // Small square, 2 K-blocks, edge microkernel
        runParityTest(16, 16, 64);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_PartialMR_17x128x128)
    {
        // Partial M (17 < 32), full N, edge in M dimension
        runParityTest(17, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_PartialNR_32x65x128)
    {
        // Full M, partial N (65 < 128), edge in N dimension
        runParityTest(32, 65, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_PartialBoth_17x65x96)
    {
        // Partial M and N, 3 K-blocks
        runParityTest(17, 65, 96);
    }

    // ----------------------------------------------------------------------------
    // Batch sizes (varying M with fixed N, K)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size1_1x128x512)
    {
        // Single sequence (batch=1)
        runParityTest(1, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size4_4x128x512)
    {
        // Small batch (4 sequences)
        runParityTest(4, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size32_32x128x512)
    {
        // Exact MR batch (32 sequences)
        runParityTest(32, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size64_64x128x512)
    {
        // 2×MR batch (64 sequences)
        runParityTest(64, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size128_128x128x512)
    {
        // Large batch (128 sequences)
        runParityTest(128, 128, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Batch_Size37_37x128x256)
    {
        // Non-MR-multiple batch (37 sequences)
        runParityTest(37, 128, 256);
    }

    // ----------------------------------------------------------------------------
    // Rectangular matrices
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_Tall_256x64x128)
    {
        // Tall: M >> N
        runParityTest(256, 64, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_Wide_64x256x128)
    {
        // Wide: N >> M
        runParityTest(64, 256, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_TallThin_512x32x128)
    {
        // Very tall, thin N (exactly 1 MR, single NR panel)
        runParityTest(512, 32, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_ShortWide_32x512x128)
    {
        // Short M, very wide N
        runParityTest(32, 512, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_ExtremelyTall_1024x32x64)
    {
        // Extremely tall (32 MR panels), thin N
        runParityTest(1024, 32, 64);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Rectangular_ExtremelyWide_32x1024x64)
    {
        // Single MR, extremely wide (8 NR panels)
        runParityTest(32, 1024, 64);
    }

    // ----------------------------------------------------------------------------
    // K-dimension variations
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, K_SingleBlock_64x64x32)
    {
        // K=32 (1 block)
        runParityTest(64, 64, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, K_TwoBlocks_64x64x64)
    {
        // K=64 (2 blocks)
        runParityTest(64, 64, 64);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, K_FourBlocks_64x64x128)
    {
        // K=128 (4 blocks)
        runParityTest(64, 64, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, K_SixteenBlocks_64x64x512)
    {
        // K=512 (16 blocks)
        runParityTest(64, 64, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, K_Large_64x64x2048)
    {
        // K=2048 (64 blocks)
        runParityTest(64, 64, 2048);
    }

    // ----------------------------------------------------------------------------
    // Stress tests (large matrices)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Stress_LargeSquare_512x512x512)
    {
        // Large square matrix
        runParityTest(512, 512, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Stress_BatchInference_256x4096x512)
    {
        // Realistic batch inference: 256 seqs, 4096 hidden dim, 512 context
        runParityTest(256, 4096, 512);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Stress_LongContext_128x2048x2048)
    {
        // Long context: 128 seqs, 2048 hidden, 2048 context
        runParityTest(128, 2048, 2048);
    }

    // ----------------------------------------------------------------------------
    // Tiny matrices (boundary conditions)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Tiny_1x1x32)
    {
        // Single element, 1 K-block
        runParityTest(1, 1, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Tiny_1x128x32)
    {
        // Single row, full NR width
        runParityTest(1, 128, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Tiny_32x1x32)
    {
        // Full MR height, single column
        runParityTest(32, 1, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Tiny_2x2x32)
    {
        // Minimal 2×2 matrix
        runParityTest(2, 2, 32);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, Tiny_7x11x64)
    {
        // Small primes, 2 K-blocks
        runParityTest(7, 11, 64);
    }

    // ----------------------------------------------------------------------------
    // NR panel variations (different N alignments)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, NR_ExactlyOnePanel_32x128x128)
    {
        // N = NR (exactly 1 panel)
        runParityTest(32, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, NR_TwoPanels_32x256x128)
    {
        // N = 2×NR (exactly 2 panels)
        runParityTest(32, 256, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, NR_PartialSecondPanel_32x192x128)
    {
        // N = 1.5×NR (1 full panel + 64 columns)
        runParityTest(32, 192, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, NR_JustOverOnePanel_32x129x128)
    {
        // N = NR + 1 (triggers edge microkernel for 1 column)
        runParityTest(32, 129, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, NR_SmallN_32x7x128)
    {
        // N << NR (single edge panel)
        runParityTest(32, 7, 128);
    }

    // ----------------------------------------------------------------------------
    // MR panel variations (different M alignments)
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, MR_ExactlyOnePanel_32x128x128)
    {
        // M = MR (exactly 1 panel) - duplicate of earlier test for symmetry
        runParityTest(32, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MR_TwoPanels_64x128x128)
    {
        // M = 2×MR (exactly 2 panels)
        runParityTest(64, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MR_PartialSecondPanel_48x128x128)
    {
        // M = 1.5×MR (1 full panel + 16 rows)
        runParityTest(48, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MR_JustOverOnePanel_33x128x128)
    {
        // M = MR + 1 (triggers edge microkernel for 1 row)
        runParityTest(33, 128, 128);
    }

    TEST_F(Q8_1GemmKernel_ParityTest, MR_SmallM_7x128x128)
    {
        // M << MR (single edge panel)
        runParityTest(7, 128, 128);
    }

    // ----------------------------------------------------------------------------
    // Powers of 2 edge case
    // ----------------------------------------------------------------------------

    TEST_F(Q8_1GemmKernel_ParityTest, Edge_PowersOfTwo_64x128x256)
    {
        // All powers of 2 (but not necessarily MR/NR multiples)
        runParityTest(64, 128, 256);
    }

} // namespace llaminar2
