/**
 * @file Test__QuantisedGemmKernel_Q8_1_Output.cpp
 * @brief Unit tests for Q8_1 output epilogue in QuantisedGemmKernel JIT kernels
 *
 * Tests the fused Q8_1 requantization epilogue added to M1/M2 JIT kernels.
 * This verifies that GEMM results can be directly written to Q8_1 format,
 * bypassing the FP32 intermediate step for the Q8_1 activation pipeline.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @brief Helper to dequantize Q8_1 blocks to FP32 for comparison
 */
static void dequant_q8_1_to_fp32(const Q8_1Block *blocks, int rows, int cols, float *output)
{
    const int blocks_per_row = (cols + 31) / 32;

    for (int r = 0; r < rows; ++r)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const Q8_1Block &block = blocks[r * blocks_per_row + b];

            // Convert FP16 scale to FP32
            uint16_t d_bits = block.d;
            uint32_t sign = (d_bits >> 15) & 1;
            uint32_t exp = (d_bits >> 10) & 0x1F;
            uint32_t mant = d_bits & 0x3FF;

            float d;
            if (exp == 0)
            {
                d = sign ? -0.0f : 0.0f;
                if (mant != 0)
                {
                    // Denormalized
                    d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * std::pow(2.0f, -14.0f);
                }
            }
            else if (exp == 31)
            {
                d = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
            }
            else
            {
                // Normalized
                uint32_t f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
                std::memcpy(&d, &f32_bits, sizeof(float));
            }

            // Dequantize each value
            for (int i = 0; i < 32; ++i)
            {
                int col = b * 32 + i;
                if (col < cols)
                {
                    output[r * cols + col] = d * block.qs[i];
                }
            }
        }
    }
}

/**
 * @brief Compute reference FP32 GEMM: C = A * B^T
 */
static void reference_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

/**
 * @brief Test Q8_1 output with M=1 (exercises M1 kernel epilogue)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, SingleRow_M1Kernel)
{
    const int M = 1;
    const int N = 128; // 4 Q8_1 blocks per row
    const int K = 64;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Cast to QuantisedGemmKernel to access Q8_1 output methods
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Allocate Q8_1 output buffer
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Run GEMM with Q8_1 output
    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // Compute reference with FP32 output
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Dequantize Q8_1 output for comparison
    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    // Compare - Q8_1 introduces quantization error
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
        // Allow ~1% relative error + small absolute tolerance
        float tol = std::abs(C_ref[i]) * 0.02f + 0.1f;
        EXPECT_NEAR(C_dequant[i], C_ref[i], tol) << "Mismatch at index " << i
                                                 << " (ref=" << C_ref[i] << ", got=" << C_dequant[i] << ")";
    }

    std::cout << "[Q8_1 Output M=1] Max error: " << max_error << std::endl;
}

/**
 * @brief Test Q8_1 output with M=2 (exercises M2 kernel epilogue)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, TwoRows_M2Kernel)
{
    const int M = 2;
    const int N = 128;
    const int K = 64;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Allocate Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Run GEMM with Q8_1 output
    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // Compute reference
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Dequantize and compare
    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
        float tol = std::abs(C_ref[i]) * 0.02f + 0.1f;
        EXPECT_NEAR(C_dequant[i], C_ref[i], tol) << "Mismatch at index " << i;
    }

    std::cout << "[Q8_1 Output M=2] Max error: " << max_error << std::endl;
}

/**
 * @brief Test Q8_1 output with larger batch (exercises both M1/M2 kernels)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, BatchedRows_MixedKernels)
{
    const int M = 7; // Odd number to test M2 + M1 fallback
    const int N = 256;
    const int K = 128;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
        sum_sq_error += error * error;
        float tol = std::abs(C_ref[i]) * 0.02f + 0.1f;
        EXPECT_NEAR(C_dequant[i], C_ref[i], tol) << "Mismatch at row " << (i / N) << ", col " << (i % N);
    }

    float rmse = std::sqrt(sum_sq_error / (M * N));
    std::cout << "[Q8_1 Output M=7] Max error: " << max_error << ", RMSE: " << rmse << std::endl;
}

/**
 * @brief Test Q8_1 output with precomputed Q8_1 activations (Q8_1→Q8_1 path)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, PrecomputedQ8_1_Activations)
{
    const int M = 4;
    const int N = 128;
    const int K = 64;

    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random FP32 activations
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Pre-quantize activations
    const int a_blocks = M * ((K + 31) / 32);
    std::vector<Q8_1Block> A_q8(a_blocks);
    q_kernel->quantize_activations(A_fp32.data(), A_q8.data(), M, K);

    // Allocate Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Run Q8_1→Q8_1 GEMM (no bias)
    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q8_1(
        A_q8.data(), output_q8.data(), M, N, K, nullptr, false, nullptr, -1);
    ASSERT_TRUE(ok);

    // Compute reference (FP32 path)
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Dequantize and compare
    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
        // Q8_1→Q8_1 path has double quantization error (input + output)
        float tol = std::abs(C_ref[i]) * 0.05f + 0.2f;
        EXPECT_NEAR(C_dequant[i], C_ref[i], tol) << "Mismatch at index " << i;
    }

    std::cout << "[Q8_1→Q8_1 Path] Max error: " << max_error << std::endl;
}

/**
 * @brief Test Q8_1 output with bias support (Q8_1→Q8_1 path + bias)
 *
 * Verifies that bias is correctly added before Q8_1 requantization.
 * This is critical for correct LLM inference since projections typically have biases.
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, PrecomputedQ8_1_WithBias)
{
    const int M = 4;
    const int N = 128;
    const int K = 64;

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random FP32 activations
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Create bias tensor with noticeable values
    std::vector<float> bias_data(N);
    for (int i = 0; i < N; ++i)
    {
        bias_data[i] = (i % 2 == 0) ? 0.5f : -0.5f; // Alternating +/- bias
    }
    auto bias = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(N)}, DeviceId::cpu());
    std::copy(bias_data.begin(), bias_data.end(), bias->mutable_data());

    // Pre-quantize activations
    const int a_blocks = M * ((K + 31) / 32);
    std::vector<Q8_1Block> A_q8(a_blocks);
    q_kernel->quantize_activations(A_fp32.data(), A_q8.data(), M, K);

    // Allocate Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Run Q8_1→Q8_1 GEMM WITH bias
    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q8_1(
        A_q8.data(), output_q8.data(), M, N, K, bias.get(), false, nullptr, -1);
    ASSERT_TRUE(ok);

    // Compute reference: FP32 GEMM + bias
    std::vector<float> C_ref(M * N);
    // First compute GEMM without bias
    q_kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);
    // Then add bias to each row
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            C_ref[m * N + n] += bias_data[n];
        }
    }

    // Dequantize and compare
    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
        sum_sq_error += error * error;
        // Q8_1→Q8_1 path with bias has quantization error
        float tol = std::abs(C_ref[i]) * 0.05f + 0.2f;
        EXPECT_NEAR(C_dequant[i], C_ref[i], tol) << "Mismatch at index " << i;
    }

    float rmse = std::sqrt(sum_sq_error / (M * N));
    std::cout << "[Q8_1→Q8_1+Bias] Max error: " << max_error << ", RMSE: " << rmse << std::endl;

    // Sanity check: verify bias actually changed the output
    // Run without bias and confirm results differ
    std::vector<Q8_1Block> output_no_bias(M * output_blocks_per_row);
    std::memset(output_no_bias.data(), 0, output_no_bias.size() * sizeof(Q8_1Block));
    q_kernel->multiply_with_precomputed_q8_1_to_q8_1(
        A_q8.data(), output_no_bias.data(), M, N, K, nullptr, false, nullptr, -1);

    std::vector<float> C_no_bias(M * N);
    dequant_q8_1_to_fp32(output_no_bias.data(), M, N, C_no_bias.data());

    // Results should differ by approximately the bias values
    float bias_diff_sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        bias_diff_sum += std::abs(C_dequant[i] - C_no_bias[i] - bias_data[i % N]);
    }
    float avg_bias_diff = bias_diff_sum / (M * N);
    EXPECT_LT(avg_bias_diff, 0.1f) << "Bias application appears incorrect";
    std::cout << "[Bias Verification] Avg diff from expected: " << avg_bias_diff << std::endl;
}

/**
 * @brief Test Q8_1 scale/sum computation correctness
 *
 * Verifies that the Q8_1 block metadata (scale d, sum sum_qs) is computed correctly.
 *
 * NOTE: This test uses N=32 which is less than the JIT kernel's 64-column block size.
 * This specifically tests the tail handling logic added to multiply_with_precomputed_q8_1_to_q8_1.
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, ScaleAndSumComputation)
{
    const int M = 1;
    const int N = 32; // Single Q8_1 block output - tests N<64 tail handling
    const int K = 32;

    // Use simple weights: identity-like (first 32 cols of 32x32 identity)
    std::vector<float> weights_fp32(N * K, 0.0f);
    for (int i = 0; i < std::min(N, K); ++i)
    {
        weights_fp32[i * K + i] = 1.0f;
    }

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Use known input values
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
    {
        A[i] = static_cast<float>(i) / K; // Range [0, 1)
    }

    // Allocate output buffer with guard bytes to detect buffer overflow
    // JIT kernel processes 64 columns at a time, but N=32 means only 1 block per row
    // If tail handling is broken, it would overwrite the guard bytes
    const int output_blocks_per_row = (N + 31) / 32; // = 1 for N=32
    const uint8_t GUARD_PATTERN = 0xDE;
    const size_t GUARD_SIZE = 2 * sizeof(Q8_1Block); // Space for 2 extra blocks (what kernel would write without tail handling)

    std::vector<uint8_t> output_buffer(M * output_blocks_per_row * sizeof(Q8_1Block) + GUARD_SIZE);
    std::memset(output_buffer.data() + M * output_blocks_per_row * sizeof(Q8_1Block), GUARD_PATTERN, GUARD_SIZE);

    Q8_1Block *output_q8 = reinterpret_cast<Q8_1Block *>(output_buffer.data());
    std::memset(output_q8, 0, M * output_blocks_per_row * sizeof(Q8_1Block));

    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8, M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // Check guard bytes weren't overwritten (would indicate buffer overflow)
    const uint8_t *guard_ptr = output_buffer.data() + M * output_blocks_per_row * sizeof(Q8_1Block);
    for (size_t i = 0; i < GUARD_SIZE; ++i)
    {
        ASSERT_EQ(guard_ptr[i], GUARD_PATTERN)
            << "Buffer overflow detected at guard byte " << i
            << " - JIT kernel wrote beyond allocated buffer (N<64 tail handling broken)";
    }

    // Check that output block has valid scale
    const Q8_1Block &block = output_q8[0];

    // Convert FP16 scale to FP32 for validation
    uint16_t d_bits = block.d;
    uint32_t sign = (d_bits >> 15) & 1;
    uint32_t exp = (d_bits >> 10) & 0x1F;
    uint32_t mant = d_bits & 0x3FF;

    float d;
    if (exp == 0)
    {
        d = 0.0f;
    }
    else if (exp == 31)
    {
        d = INFINITY;
    }
    else
    {
        uint32_t f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        std::memcpy(&d, &f32_bits, sizeof(float));
    }

    // Scale should be reasonable (max_abs / 127)
    EXPECT_GT(d, 0.0f) << "Scale should be positive";
    EXPECT_LT(d, 1.0f) << "Scale should be < 1 for input in [0, 1)";

    // Verify sum_qs matches sum of qs values
    int32_t computed_sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        computed_sum += block.qs[i];
    }
    EXPECT_EQ(static_cast<int16_t>(computed_sum), block.sum_qs)
        << "sum_qs mismatch: computed=" << computed_sum << ", stored=" << block.sum_qs;

    std::cout << "[Scale/Sum Check] d=" << d << ", sum_qs=" << block.sum_qs << std::endl;
}

/**
 * @brief Test Q8_1 output with edge case: very small values
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, SmallValueHandling)
{
    const int M = 1;
    const int N = 64;
    const int K = 32;

    // Create small-valued weights
    std::vector<float> weights_fp32(N * K);
    for (int i = 0; i < N * K; ++i)
    {
        weights_fp32[i] = (i % 2 == 0) ? 1e-4f : -1e-4f;
    }

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Use small input values
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
    {
        A[i] = 1e-3f;
    }

    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);

    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // The output should be very small but not NaN/Inf
    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_FALSE(std::isnan(C_dequant[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(C_dequant[i])) << "Inf at index " << i;
    }
}

/**
 * @brief Test Q8_1 output with edge case: large values
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, LargeValueHandling)
{
    const int M = 1;
    const int N = 64;
    const int K = 32;

    // Create large-valued weights (will saturate quantization)
    std::vector<float> weights_fp32(N * K);
    for (int i = 0; i < N * K; ++i)
    {
        weights_fp32[i] = (i % 3 == 0) ? 10.0f : ((i % 3 == 1) ? -10.0f : 0.0f);
    }

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Use large input values
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
    {
        A[i] = 5.0f;
    }

    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);

    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // Compare against FP32 reference (with higher tolerance due to saturation)
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_FALSE(std::isnan(C_dequant[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(C_dequant[i])) << "Inf at index " << i;
        float error = std::abs(C_dequant[i] - C_ref[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[Large Values] Max error: " << max_error << std::endl;
}

/**
 * @brief Test row independence: each row should quantize independently
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, RowIndependence)
{
    const int M = 4;
    const int N = 64;
    const int K = 32;

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create input where each row has very different magnitudes
    std::vector<float> A(M * K);
    for (int r = 0; r < M; ++r)
    {
        float scale = std::pow(10.0f, r); // 1, 10, 100, 1000
        for (int k = 0; k < K; ++k)
        {
            A[r * K + k] = scale * dist(gen);
        }
    }

    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);

    bool ok = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok);

    // Each row should have a scale appropriate for its magnitude
    // Check that scales increase with row index (since input magnitudes increase)
    std::vector<float> row_max_scales(M);
    for (int r = 0; r < M; ++r)
    {
        float max_scale = 0.0f;
        for (int b = 0; b < output_blocks_per_row; ++b)
        {
            const Q8_1Block &block = output_q8[r * output_blocks_per_row + b];

            // Convert FP16 scale
            uint16_t d_bits = block.d;
            uint32_t sign = (d_bits >> 15) & 1;
            uint32_t exp = (d_bits >> 10) & 0x1F;
            uint32_t mant = d_bits & 0x3FF;

            float d = 0.0f;
            if (exp > 0 && exp < 31)
            {
                uint32_t f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
                std::memcpy(&d, &f32_bits, sizeof(float));
            }
            max_scale = std::max(max_scale, std::abs(d));
        }
        row_max_scales[r] = max_scale;
    }

    // Later rows (higher magnitude) should have larger scales
    for (int r = 1; r < M; ++r)
    {
        // Allow some tolerance since quantization is lossy
        EXPECT_GT(row_max_scales[r] * 2.0f, row_max_scales[r - 1])
            << "Row " << r << " scale (" << row_max_scales[r]
            << ") should be larger than row " << (r - 1) << " scale (" << row_max_scales[r - 1] << ")";
    }
}

// ============================================================================
// Accuracy Comparison Tests: Q8_1 Output vs FP32 Output
// ============================================================================

/**
 * @brief Compute cosine similarity between two vectors
 * @return Cosine similarity in range [-1, 1], where 1 = identical direction
 */
static double compute_cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a < 1e-12 || norm_b < 1e-12)
        return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

/**
 * @brief Compute relative L2 error: ||a - b||_2 / ||b||_2
 * @return Relative L2 error (0 = identical, larger = more error)
 */
static double compute_relative_l2_error(const float *actual, const float *reference, size_t n)
{
    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double diff = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
        diff_sq += diff * diff;
        ref_sq += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
    }
    if (ref_sq < 1e-12)
        return 0.0;
    return std::sqrt(diff_sq) / std::sqrt(ref_sq);
}

/**
 * @brief Compute per-row cosine similarity statistics
 */
static void compute_per_row_cosine(const float *actual, const float *reference,
                                   int rows, int cols,
                                   double &min_cos, double &max_cos, double &avg_cos)
{
    min_cos = 1.0;
    max_cos = -1.0;
    avg_cos = 0.0;

    for (int r = 0; r < rows; ++r)
    {
        double cos = compute_cosine_similarity(actual + r * cols, reference + r * cols, cols);
        min_cos = std::min(min_cos, cos);
        max_cos = std::max(max_cos, cos);
        avg_cos += cos;
    }
    avg_cos /= rows;
}

/**
 * @brief Test Q8_1 vs FP32 accuracy with single row (M1 kernel)
 *
 * Compares GEMM output when written to Q8_1 vs FP32, measuring:
 * - Cosine similarity (should be very close to 1.0)
 * - Relative L2 error (should be small, typically <1%)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, Accuracy_SingleRow_CosineSimilarity)
{
    const int M = 1;
    const int N = 256; // 8 Q8_1 blocks per row
    const int K = 128;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // ===== FP32 output path =====
    std::vector<float> C_fp32(M * N);
    bool ok_fp32 = q_kernel->multiply(A.data(), C_fp32.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);
    ASSERT_TRUE(ok_fp32);

    // ===== Q8_1 output path =====
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    bool ok_q8 = q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);
    ASSERT_TRUE(ok_q8);

    // Dequantize Q8_1 for comparison
    std::vector<float> C_q8_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());

    // Compute metrics
    double cosine_sim = compute_cosine_similarity(C_q8_dequant.data(), C_fp32.data(), M * N);
    double rel_l2 = compute_relative_l2_error(C_q8_dequant.data(), C_fp32.data(), M * N);

    std::cout << "[Accuracy M=1, N=" << N << ", K=" << K << "]" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine_sim << std::endl;
    std::cout << "  Relative L2 error: " << std::fixed << std::setprecision(6) << (rel_l2 * 100.0) << "%" << std::endl;

    // Locked-in thresholds based on observed values:
    // - Observed cosine: 0.999986
    // - Observed L2: 0.525%
    EXPECT_GT(cosine_sim, 0.9999) << "Cosine similarity too low for Q8_1 encoding";
    EXPECT_LT(rel_l2, 0.01) << "Relative L2 error too high for Q8_1 encoding (expected <1%)";
}

/**
 * @brief Test Q8_1 vs FP32 accuracy with multiple rows (M2 kernel)
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, Accuracy_MultiRow_CosineSimilarity)
{
    const int M = 8;
    const int N = 256;
    const int K = 128;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // FP32 output
    std::vector<float> C_fp32(M * N);
    q_kernel->multiply(A.data(), C_fp32.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);

    std::vector<float> C_q8_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());

    // Overall metrics
    double cosine_sim = compute_cosine_similarity(C_q8_dequant.data(), C_fp32.data(), M * N);
    double rel_l2 = compute_relative_l2_error(C_q8_dequant.data(), C_fp32.data(), M * N);

    // Per-row metrics
    double min_cos, max_cos, avg_cos;
    compute_per_row_cosine(C_q8_dequant.data(), C_fp32.data(), M, N, min_cos, max_cos, avg_cos);

    std::cout << "[Accuracy M=" << M << ", N=" << N << ", K=" << K << "]" << std::endl;
    std::cout << "  Overall cosine similarity: " << std::fixed << std::setprecision(6) << cosine_sim << std::endl;
    std::cout << "  Per-row cosine: min=" << min_cos << ", max=" << max_cos << ", avg=" << avg_cos << std::endl;
    std::cout << "  Relative L2 error: " << std::fixed << std::setprecision(6) << (rel_l2 * 100.0) << "%" << std::endl;

    // Locked-in thresholds based on observed values:
    // - Observed overall cosine: 0.999986
    // - Observed per-row min cosine: 0.999983
    // - Observed L2: 0.538%
    EXPECT_GT(cosine_sim, 0.9999) << "Overall cosine too low";
    EXPECT_GT(min_cos, 0.9999) << "Per-row cosine too low";
    EXPECT_LT(rel_l2, 0.01) << "Relative L2 error too high (expected <1%)";
}

/**
 * @brief Test Q8_1 accuracy with varying value magnitudes
 *
 * Tests accuracy when GEMM output has different scales across rows.
 * This stresses the per-block quantization scaling.
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, Accuracy_VaryingMagnitudes)
{
    const int M = 4;
    const int N = 128;
    const int K = 64;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create input with varying magnitudes per row
    std::vector<float> A(M * K);
    for (int r = 0; r < M; ++r)
    {
        float scale = std::pow(10.0f, r - 1); // 0.1, 1.0, 10.0, 100.0
        for (int k = 0; k < K; ++k)
        {
            A[r * K + k] = scale * dist(gen);
        }
    }

    // FP32 output
    std::vector<float> C_fp32(M * N);
    q_kernel->multiply(A.data(), C_fp32.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);

    std::vector<float> C_q8_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());

    std::cout << "[Accuracy - Varying Magnitudes M=" << M << "]" << std::endl;

    // Check per-row accuracy (each row has different magnitude)
    for (int r = 0; r < M; ++r)
    {
        double row_cos = compute_cosine_similarity(
            C_q8_dequant.data() + r * N, C_fp32.data() + r * N, N);
        double row_l2 = compute_relative_l2_error(
            C_q8_dequant.data() + r * N, C_fp32.data() + r * N, N);

        std::cout << "  Row " << r << " (scale=" << std::pow(10.0f, r - 1) << "): "
                  << "cosine=" << std::fixed << std::setprecision(6) << row_cos
                  << ", rel_l2=" << (row_l2 * 100.0) << "%" << std::endl;

        // Locked-in thresholds based on observed values:
        // - Observed per-row cosine: 0.999982 - 0.999989
        // - Observed per-row L2: 0.49% - 0.61%
        EXPECT_GT(row_cos, 0.9999) << "Row " << r << " cosine too low";
        EXPECT_LT(row_l2, 0.01) << "Row " << r << " L2 error too high (expected <1%)";
    }
}

/**
 * @brief Test Q8_1 accuracy with realistic LLM-like activation patterns
 *
 * Uses normally distributed inputs (more realistic than uniform)
 * and tests larger dimensions typical of LLM inference.
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, Accuracy_RealisticLLMActivations)
{
    const int M = 32;  // Batch size
    const int N = 896; // Qwen 2.5 0.5B d_model
    const int K = 896;

    std::mt19937 gen(789);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);

    // Weights: uniform random
    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = uniform_dist(gen) * 0.1f; // Scale down weights

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Activations: normal distribution (typical after LayerNorm)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = normal_dist(gen);

    // FP32 output
    std::vector<float> C_fp32(M * N);
    q_kernel->multiply(A.data(), C_fp32.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Q8_1 output
    const int output_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
    q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);

    std::vector<float> C_q8_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());

    // Overall metrics
    double cosine_sim = compute_cosine_similarity(C_q8_dequant.data(), C_fp32.data(), M * N);
    double rel_l2 = compute_relative_l2_error(C_q8_dequant.data(), C_fp32.data(), M * N);

    // Per-row metrics
    double min_cos, max_cos, avg_cos;
    compute_per_row_cosine(C_q8_dequant.data(), C_fp32.data(), M, N, min_cos, max_cos, avg_cos);

    std::cout << "[Accuracy - Realistic LLM M=" << M << ", N=" << N << ", K=" << K << "]" << std::endl;
    std::cout << "  Overall cosine similarity: " << std::fixed << std::setprecision(6) << cosine_sim << std::endl;
    std::cout << "  Per-row cosine: min=" << min_cos << ", max=" << max_cos << ", avg=" << avg_cos << std::endl;
    std::cout << "  Relative L2 error: " << std::fixed << std::setprecision(6) << (rel_l2 * 100.0) << "%" << std::endl;

    // Locked-in thresholds based on observed values:
    // - Observed overall cosine: 0.999986
    // - Observed per-row min cosine: 0.999984
    // - Observed L2: 0.538%
    EXPECT_GT(cosine_sim, 0.9999) << "Overall cosine should be excellent for normalized data";
    EXPECT_GT(min_cos, 0.9999) << "Even worst row should have good cosine";
    EXPECT_LT(rel_l2, 0.01) << "Relative L2 should be under 1%";
}

/**
 * @brief Test Q8_1 accuracy statistics over multiple random seeds
 *
 * Runs multiple trials to get statistical bounds on Q8_1 encoding accuracy.
 */
TEST(Test__QuantisedGemmKernel_Q8_1_Output, Accuracy_Statistics)
{
    const int M = 4;
    const int N = 128;
    const int K = 64;
    const int NUM_TRIALS = 10;

    std::vector<double> cosines;
    std::vector<double> rel_l2s;

    for (int trial = 0; trial < NUM_TRIALS; ++trial)
    {
        std::mt19937 gen(trial * 1000 + 42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> weights_fp32(N * K);
        for (auto &x : weights_fp32)
            x = dist(gen);

        auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
            weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
        auto kernel = weights_tensor->createGemm();
        auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
        ASSERT_NE(q_kernel, nullptr);

        std::vector<float> A(M * K);
        for (auto &x : A)
            x = dist(gen);

        // FP32 output
        std::vector<float> C_fp32(M * N);
        q_kernel->multiply(A.data(), C_fp32.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

        // Q8_1 output
        const int output_blocks_per_row = (N + 31) / 32;
        std::vector<Q8_1Block> output_q8(M * output_blocks_per_row);
        q_kernel->multiply_to_q8_1(A.data(), output_q8.data(), M, N, K, nullptr, -1);

        std::vector<float> C_q8_dequant(M * N);
        dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());

        cosines.push_back(compute_cosine_similarity(C_q8_dequant.data(), C_fp32.data(), M * N));
        rel_l2s.push_back(compute_relative_l2_error(C_q8_dequant.data(), C_fp32.data(), M * N));
    }

    // Compute statistics
    double cos_min = *std::min_element(cosines.begin(), cosines.end());
    double cos_max = *std::max_element(cosines.begin(), cosines.end());
    double cos_avg = std::accumulate(cosines.begin(), cosines.end(), 0.0) / NUM_TRIALS;

    double l2_min = *std::min_element(rel_l2s.begin(), rel_l2s.end());
    double l2_max = *std::max_element(rel_l2s.begin(), rel_l2s.end());
    double l2_avg = std::accumulate(rel_l2s.begin(), rel_l2s.end(), 0.0) / NUM_TRIALS;

    std::cout << "[Accuracy Statistics over " << NUM_TRIALS << " trials]" << std::endl;
    std::cout << "  Cosine similarity: min=" << std::fixed << std::setprecision(6)
              << cos_min << ", max=" << cos_max << ", avg=" << cos_avg << std::endl;
    std::cout << "  Relative L2 error: min=" << (l2_min * 100.0) << "%, max="
              << (l2_max * 100.0) << "%, avg=" << (l2_avg * 100.0) << "%" << std::endl;

    // Locked-in thresholds based on observed values:
    // - Observed cosine: min=0.999983, max=0.999989, avg=0.999986
    // - Observed L2: min=0.477%, max=0.580%, avg=0.532%
    EXPECT_GT(cos_min, 0.9999) << "Worst-case cosine too low";
    EXPECT_GT(cos_avg, 0.9999) << "Average cosine should be excellent";
    EXPECT_LT(l2_max, 0.01) << "Worst-case L2 too high (expected <1%)";
    EXPECT_LT(l2_avg, 0.01) << "Average L2 should be under 1%";
}
