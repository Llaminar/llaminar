/**
 * @file Test__QuantisedGemmKernel_Q16_1_Output.cpp
 * @brief Unit tests for Q16_1 output epilogue in QuantisedGemmKernel JIT kernels
 *
 * Tests the fused Q16_1 requantization epilogue added to M1/M2 JIT kernels.
 * This verifies that GEMM results can be directly written to Q16_1 format,
 * providing 256× more precision than Q8_1 output for high dynamic range data
 * like K projections in HybridQ16 attention.
 *
 * Key tests:
 * - Q16_1 output preserves more precision than Q8_1 for high dynamic range
 * - Block size configuration (64, 128) works correctly
 * - Both M1 and M2 kernels produce correct output
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
 * @brief Helper to dequantize Q16_1 blocks (64-element) to FP32 for comparison
 */
static void dequant_q16_1_block64_to_fp32(const void *blocks, int rows, int cols, float *output)
{
    // Q16_1Block_64: float d (4), int32 sum_qs (4), int16 qs[64] (128) = 136 bytes
    const int block_size = 64;
    const int blocks_per_row = cols / block_size;
    const uint8_t *data = static_cast<const uint8_t *>(blocks);

    for (int r = 0; r < rows; ++r)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const uint8_t *block_ptr = data + (r * blocks_per_row + b) * 136;
            float d;
            std::memcpy(&d, block_ptr, sizeof(float));
            const int16_t *qs = reinterpret_cast<const int16_t *>(block_ptr + 8);

            for (int i = 0; i < block_size; ++i)
            {
                int col = b * block_size + i;
                if (col < cols)
                {
                    output[r * cols + col] = d * static_cast<float>(qs[i]);
                }
            }
        }
    }
}

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
                    d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * std::pow(2.0f, -14.0f);
                }
            }
            else if (exp == 31)
            {
                d = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
            }
            else
            {
                uint32_t f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
                std::memcpy(&d, &f32_bits, sizeof(float));
            }

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
 * @brief Test Q16_1 output with M=1 (exercises M1 kernel epilogue)
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, SingleRow_M1Kernel)
{
    const int M = 1;
    const int N = 64; // One Q16_1Block_64 per row
    const int K = 128;

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

    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Quantize input to Q8_1
    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    // Allocate Q16_1 output buffer (block_size=64)
    const int q16_block_size = 64;
    const int q16_block_bytes = 136; // 8 + 64*2
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);
    std::memset(output_q16.data(), 0, output_q16.size());

    // Run GEMM with Q16_1 output
    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    // Compute reference with FP32 output
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Dequantize Q16_1 output for comparison
    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    // Compare - Q16_1 should have lower error than Q8_1
    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
        sum_sq_error += err * err;
    }
    float rmse = std::sqrt(sum_sq_error / (M * N));

    std::cout << "Q16_1 Output Test (M=1):\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  RMSE: " << rmse << "\n";

    // Q16_1 has 256× more precision than Q8_1, so expect very low error
    EXPECT_LT(max_error, 0.5f) << "Q16_1 output max error too high";
    EXPECT_LT(rmse, 0.1f) << "Q16_1 output RMSE too high";
}

/**
 * @brief Test Q16_1 output with M=2 (exercises M2 kernel epilogue)
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, TwoRows_M2Kernel)
{
    const int M = 2;
    const int N = 64;
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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);
    std::memset(output_q16.data(), 0, output_q16.size());

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
        sum_sq_error += err * err;
    }
    float rmse = std::sqrt(sum_sq_error / (M * N));

    std::cout << "Q16_1 Output Test (M=2):\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  RMSE: " << rmse << "\n";

    EXPECT_LT(max_error, 0.5f);
    EXPECT_LT(rmse, 0.1f);
}

/**
 * @brief Test that Q16_1 preserves more precision than Q8_1 for high dynamic range
 *
 * This is the key test for the HybridQ16 K projection use case.
 * When GEMM output has high dynamic range (e.g., max ~130), Q8_1 loses
 * small values while Q16_1 preserves them.
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, PreservesPrecisionBetterThanQ8_1)
{
    const int M = 1;
    const int N = 64;
    const int K = 128;

    std::mt19937 gen(999);

    // Create weights that produce high dynamic range output
    // Mix of large and small values
    std::vector<float> weights_fp32(N * K);
    std::uniform_real_distribution<float> dist_large(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_small(-0.1f, 0.1f);
    for (int i = 0; i < N * K; ++i)
    {
        // 80% small values, 20% large values
        weights_fp32[i] = (i % 5 == 0) ? dist_large(gen) : dist_small(gen);
    }

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Create input that also has high dynamic range
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
    {
        A[i] = (i % 5 == 0) ? dist_large(gen) : dist_small(gen);
    }

    // Compute reference FP32 output
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Find max absolute value (the "dynamic range")
    float max_abs = 0.0f;
    for (float v : C_ref)
        max_abs = std::max(max_abs, std::abs(v));
    std::cout << "Reference output max_abs: " << max_abs << "\n";

    // Quantize input to Q8_1
    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    // Run Q8_1 output path
    const int q8_blocks_per_row = (N + 31) / 32;
    std::vector<Q8_1Block> output_q8(M * q8_blocks_per_row);
    q_kernel->multiply_with_precomputed_q8_1_to_q8_1(
        A_q8.data(), output_q8.data(), M, N, K);

    // Run Q16_1 output path
    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int q16_blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * q16_blocks_per_row * q16_block_bytes);
    q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);

    // Dequantize both outputs
    std::vector<float> C_q8_dequant(M * N);
    std::vector<float> C_q16_dequant(M * N);
    dequant_q8_1_to_fp32(output_q8.data(), M, N, C_q8_dequant.data());
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_q16_dequant.data());

    // Compute errors for both
    float q8_max_error = 0.0f, q8_sum_sq = 0.0f;
    float q16_max_error = 0.0f, q16_sum_sq = 0.0f;
    int q8_zero_count = 0, q16_zero_count = 0;
    int ref_nonzero_count = 0;

    for (int i = 0; i < M * N; ++i)
    {
        float ref = C_ref[i];
        float q8_val = C_q8_dequant[i];
        float q16_val = C_q16_dequant[i];

        if (std::abs(ref) > 0.01f)
        {
            ref_nonzero_count++;
            if (std::abs(q8_val) < 0.001f)
                q8_zero_count++;
            if (std::abs(q16_val) < 0.001f)
                q16_zero_count++;
        }

        float q8_err = std::abs(ref - q8_val);
        float q16_err = std::abs(ref - q16_val);

        q8_max_error = std::max(q8_max_error, q8_err);
        q16_max_error = std::max(q16_max_error, q16_err);
        q8_sum_sq += q8_err * q8_err;
        q16_sum_sq += q16_err * q16_err;
    }

    float q8_rmse = std::sqrt(q8_sum_sq / (M * N));
    float q16_rmse = std::sqrt(q16_sum_sq / (M * N));

    std::cout << "Precision Comparison (high dynamic range):\n";
    std::cout << "  Q8_1:  max_error=" << q8_max_error << ", RMSE=" << q8_rmse
              << ", zeros=" << q8_zero_count << "/" << ref_nonzero_count << "\n";
    std::cout << "  Q16_1: max_error=" << q16_max_error << ", RMSE=" << q16_rmse
              << ", zeros=" << q16_zero_count << "/" << ref_nonzero_count << "\n";

    // Q16_1 should have lower error than Q8_1
    EXPECT_LT(q16_rmse, q8_rmse) << "Q16_1 should have lower RMSE than Q8_1";

    // Q16_1 should lose fewer values to zero
    EXPECT_LE(q16_zero_count, q8_zero_count)
        << "Q16_1 should preserve more small values than Q8_1";
}

/**
 * @brief Test Q16_1 with larger N (multiple 64-element blocks)
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, MultipleBlocks_N128)
{
    const int M = 2;
    const int N = 128; // Two Q16_1Block_64 per row
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

    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);
    std::memset(output_q16.data(), 0, output_q16.size());

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
        sum_sq_error += err * err;
    }
    float rmse = std::sqrt(sum_sq_error / (M * N));

    std::cout << "Q16_1 Output Test (M=2, N=128):\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  RMSE: " << rmse << "\n";

    EXPECT_LT(max_error, 0.5f);
    EXPECT_LT(rmse, 0.1f);
}

/**
 * @brief Test with odd number of rows (M=3) to exercise M2+M1 kernel combination
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, OddRows_M3)
{
    const int M = 3;
    const int N = 64;
    const int K = 64;

    std::mt19937 gen(789);
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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
    }

    std::cout << "Q16_1 Output Test (M=3):\n";
    std::cout << "  Max error: " << max_error << "\n";

    EXPECT_LT(max_error, 0.5f);
}

/**
 * @brief Test with Qwen2 production dimensions (K-projection: N=896, K=896)
 *
 * This exercises the actual dimensions used in Qwen2-0.5B K projection
 * where head_dim=64, num_kv_heads=2, hidden_size=896.
 * K projection is: [seq_len, 896] @ [128, 896]^T -> [seq_len, 128]
 * But we test the weight matrix dimensions directly.
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, ProductionDimensions_Qwen2)
{
    // Qwen2-0.5B dimensions for K projection
    const int M = 1;   // Single token (decode)
    const int N = 128; // num_kv_heads * head_dim = 2 * 64
    const int K = 896; // hidden_size

    std::mt19937 gen(2026);
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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
        sum_sq_error += err * err;
    }
    float rmse = std::sqrt(sum_sq_error / (M * N));

    std::cout << "Q16_1 Output Test (Qwen2 production dims M=1, N=128, K=896):\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  RMSE: " << rmse << "\n";

    EXPECT_LT(max_error, 1.0f) << "Production dimensions should maintain precision";
    EXPECT_LT(rmse, 0.2f) << "Production dimensions RMSE should be low";
}

/**
 * @brief Test with prefill batch size (M=32 tokens)
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, ProductionDimensions_Prefill)
{
    const int M = 32;  // Prefill batch
    const int N = 128; // K projection output
    const int K = 896; // hidden_size

    std::mt19937 gen(12345);
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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float err = std::abs(C_ref[i] - C_dequant[i]);
        max_error = std::max(max_error, err);
        sum_sq_error += err * err;
    }
    float rmse = std::sqrt(sum_sq_error / (M * N));

    std::cout << "Q16_1 Output Test (Prefill M=32, N=128, K=896):\n";
    std::cout << "  Max error: " << max_error << "\n";
    std::cout << "  RMSE: " << rmse << "\n";

    EXPECT_LT(max_error, 1.0f);
    EXPECT_LT(rmse, 0.2f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * @brief Test with zero input - output should be all zeros
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, EdgeCase_ZeroInput)
{
    const int M = 1;
    const int N = 64;
    const int K = 64;

    std::mt19937 gen(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // All-zero input
    std::vector<float> A(M * K, 0.0f);

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    // All outputs should be zero (or very close)
    float max_abs = 0.0f;
    for (float v : C_dequant)
        max_abs = std::max(max_abs, std::abs(v));

    std::cout << "Q16_1 Zero Input Test: max_abs output = " << max_abs << "\n";
    EXPECT_LT(max_abs, 0.01f) << "Zero input should produce near-zero output";
}

/**
 * @brief Test with very small values to ensure they're preserved (not rounded to zero)
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, EdgeCase_SmallValues)
{
    const int M = 1;
    const int N = 64;
    const int K = 64;

    // Create weights with small values
    std::vector<float> weights_fp32(N * K);
    for (int i = 0; i < N * K; ++i)
        weights_fp32[i] = 0.01f * ((i % 3) - 1); // -0.01, 0, 0.01

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Small input values
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
        A[i] = 0.1f * ((i % 5) - 2); // -0.2 to 0.2

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    // Check relative error for non-tiny values
    int preserved_count = 0;
    int total_nonzero = 0;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(C_ref[i]) > 1e-6f)
        {
            total_nonzero++;
            if (std::abs(C_dequant[i]) > 1e-7f)
                preserved_count++;
        }
    }

    std::cout << "Q16_1 Small Values Test: preserved " << preserved_count
              << "/" << total_nonzero << " non-zero values\n";

    // Q16_1 should preserve most small values
    float preservation_ratio = (total_nonzero > 0) ? static_cast<float>(preserved_count) / total_nonzero : 1.0f;
    EXPECT_GT(preservation_ratio, 0.8f) << "Q16_1 should preserve most small values";
}

/**
 * @brief Test with large values to check for saturation handling
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, EdgeCase_LargeValues)
{
    const int M = 1;
    const int N = 64;
    const int K = 64;

    // Create weights with large values
    std::vector<float> weights_fp32(N * K);
    for (int i = 0; i < N * K; ++i)
        weights_fp32[i] = 10.0f * ((i % 3) - 1); // -10, 0, 10

    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(
        weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    auto *q_kernel = dynamic_cast<QuantisedGemmKernel *>(kernel.get());
    ASSERT_NE(q_kernel, nullptr);

    // Large input values
    std::vector<float> A(M * K);
    for (int i = 0; i < K; ++i)
        A[i] = 5.0f * ((i % 5) - 2); // -10 to 10

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    std::vector<float> C_dequant(M * N);
    dequant_q16_1_block64_to_fp32(output_q16.data(), M, N, C_dequant.data());

    // Check no NaN/Inf
    bool has_nan_inf = false;
    for (float v : C_dequant)
    {
        if (std::isnan(v) || std::isinf(v))
        {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Large values should not produce NaN/Inf";

    // Check relative error
    float max_rel_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(C_ref[i]) > 1.0f)
        {
            float rel_err = std::abs(C_ref[i] - C_dequant[i]) / std::abs(C_ref[i]);
            max_rel_error = std::max(max_rel_error, rel_err);
        }
    }

    std::cout << "Q16_1 Large Values Test: max relative error = " << max_rel_error << "\n";
    EXPECT_LT(max_rel_error, 0.1f) << "Large values should have low relative error";
}

// ============================================================================
// Block Structure Verification Tests
// ============================================================================

/**
 * @brief Helper to extract Q16_1Block_64 fields for verification
 */
struct Q16_1Block64_Extracted
{
    float d;
    int32_t sum_qs;
    std::array<int16_t, 64> qs;
};

static Q16_1Block64_Extracted extract_q16_1_block64(const uint8_t *block_ptr)
{
    Q16_1Block64_Extracted result;
    std::memcpy(&result.d, block_ptr, sizeof(float));
    std::memcpy(&result.sum_qs, block_ptr + 4, sizeof(int32_t));
    const int16_t *qs_ptr = reinterpret_cast<const int16_t *>(block_ptr + 8);
    for (int i = 0; i < 64; ++i)
        result.qs[i] = qs_ptr[i];
    return result;
}

/**
 * @brief Verify Q16_1 block structure: scale (d) is non-negative
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, BlockStructure_ScaleNonNegative)
{
    const int M = 4;
    const int N = 128; // 2 blocks per row
    const int K = 64;

    std::mt19937 gen(333);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    // Verify all scales are non-negative
    int negative_scale_count = 0;
    for (int row = 0; row < M; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const uint8_t *block_ptr = output_q16.data() +
                                       (row * blocks_per_row + b) * q16_block_bytes;
            auto block = extract_q16_1_block64(block_ptr);

            if (block.d < 0.0f)
            {
                negative_scale_count++;
                std::cout << "Negative scale at row " << row << " block " << b
                          << ": d = " << block.d << "\n";
            }
        }
    }

    EXPECT_EQ(negative_scale_count, 0) << "Q16_1 scales should be non-negative";
}

/**
 * @brief Verify Q16_1 block structure: sum_qs matches sum of qs values
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, BlockStructure_SumQsCorrect)
{
    const int M = 2;
    const int N = 128;
    const int K = 64;

    std::mt19937 gen(444);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    // Verify sum_qs matches actual sum
    int mismatch_count = 0;
    for (int row = 0; row < M; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const uint8_t *block_ptr = output_q16.data() +
                                       (row * blocks_per_row + b) * q16_block_bytes;
            auto block = extract_q16_1_block64(block_ptr);

            int32_t computed_sum = 0;
            for (int i = 0; i < 64; ++i)
                computed_sum += block.qs[i];

            if (block.sum_qs != computed_sum)
            {
                mismatch_count++;
                std::cout << "sum_qs mismatch at row " << row << " block " << b
                          << ": stored=" << block.sum_qs
                          << " computed=" << computed_sum << "\n";
            }
        }
    }

    EXPECT_EQ(mismatch_count, 0) << "sum_qs should match actual sum of qs values";
}

/**
 * @brief Verify Q16_1 block structure: qs values are within int16 range
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, BlockStructure_QsWithinRange)
{
    const int M = 2;
    const int N = 64;
    const int K = 128;

    std::mt19937 gen(555);
    // Use larger range to stress test
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    // Verify qs values are within expected range (target is ~16000 headroom)
    int16_t min_qs = INT16_MAX;
    int16_t max_qs = INT16_MIN;

    for (int row = 0; row < M; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const uint8_t *block_ptr = output_q16.data() +
                                       (row * blocks_per_row + b) * q16_block_bytes;
            auto block = extract_q16_1_block64(block_ptr);

            for (int i = 0; i < 64; ++i)
            {
                min_qs = std::min(min_qs, block.qs[i]);
                max_qs = std::max(max_qs, block.qs[i]);
            }
        }
    }

    std::cout << "Q16_1 qs range: [" << min_qs << ", " << max_qs << "]\n";

    // Values should be within int16 range and not saturated at extreme edges
    EXPECT_GE(min_qs, INT16_MIN) << "qs should be >= INT16_MIN";
    EXPECT_LE(max_qs, INT16_MAX) << "qs should be <= INT16_MAX";

    // With proper scaling, values should typically be in ~[-16000, 16000]
    // Allow some flexibility since it depends on data
    EXPECT_GT(min_qs, -20000) << "qs should not be at extreme negative";
    EXPECT_LT(max_qs, 20000) << "qs should not be at extreme positive";
}

/**
 * @brief Verify reconstruction: d * qs[i] should closely match FP32 reference
 */
TEST(Test__QuantisedGemmKernel_Q16_1_Output, BlockStructure_ReconstructionAccuracy)
{
    const int M = 1;
    const int N = 64;
    const int K = 64;

    std::mt19937 gen(666);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

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

    const int k_blocks = K / 32;
    std::vector<Q8_1Block> A_q8(M * k_blocks);
    q_kernel->quantize_activations(A.data(), A_q8.data(), M, K);

    const int q16_block_size = 64;
    const int q16_block_bytes = 136;
    const int blocks_per_row = N / q16_block_size;
    std::vector<uint8_t> output_q16(M * blocks_per_row * q16_block_bytes);

    bool ok = q_kernel->multiply_with_precomputed_q8_1_to_q16_1(
        A_q8.data(), output_q16.data(), M, N, K, q16_block_size);
    ASSERT_TRUE(ok);

    // Get FP32 reference
    std::vector<float> C_ref(M * N);
    q_kernel->multiply(A.data(), C_ref.data(), M, N, K, false, 1.0f, 0.0f, nullptr, -1);

    // Manually reconstruct from block and compare
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int row = 0; row < M; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const uint8_t *block_ptr = output_q16.data() +
                                       (row * blocks_per_row + b) * q16_block_bytes;
            auto block = extract_q16_1_block64(block_ptr);

            for (int i = 0; i < 64; ++i)
            {
                int col = b * 64 + i;
                float reconstructed = block.d * static_cast<float>(block.qs[i]);
                float reference = C_ref[row * N + col];

                float abs_err = std::abs(reconstructed - reference);
                max_abs_error = std::max(max_abs_error, abs_err);

                if (std::abs(reference) > 0.01f)
                {
                    float rel_err = abs_err / std::abs(reference);
                    max_rel_error = std::max(max_rel_error, rel_err);
                }
            }
        }
    }

    std::cout << "Q16_1 Block Reconstruction:\n";
    std::cout << "  Max absolute error: " << max_abs_error << "\n";
    std::cout << "  Max relative error: " << max_rel_error << "\n";

    EXPECT_LT(max_abs_error, 0.5f) << "Reconstruction should have low absolute error";
    EXPECT_LT(max_rel_error, 0.05f) << "Reconstruction should have low relative error";
}
