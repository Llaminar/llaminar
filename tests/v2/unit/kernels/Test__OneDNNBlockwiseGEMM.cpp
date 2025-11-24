/**
 * @file Test__OneDNNBlockwiseGEMM.cpp
 * @brief Unit tests for OneDNN blockwise Q8_1 × quantized weight GEMM microkernel
 * @author David Sanftenberg
 * @date November 24, 2025
 *
 * Unit tests for the blockwise GEMM microkernel with controlled inputs to isolate
 * numerical correctness issues.
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

class Test__OneDNNBlockwiseGEMM : public ::testing::Test
{
protected:
    /**
     * @brief Create simple Q8_1 activation blocks for testing
     *
     * Creates M rows × K cols with K/32 blocks per row.
     * Uses simple integer values for easy verification.
     */
    std::vector<Q8_1Block> createSimpleQ8_1Activations(int M, int K)
    {
        EXPECT_EQ(K % 32, 0) << "K must be multiple of 32";

        size_t k_blocks = K / 32;
        std::vector<Q8_1Block> blocks(M * k_blocks);

        for (int m = 0; m < M; ++m)
        {
            for (size_t kb = 0; kb < k_blocks; ++kb)
            {
                Q8_1Block &block = blocks[m * k_blocks + kb];

                // Simple pattern: fill with small integers
                int32_t sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t val = static_cast<int8_t>((m * 32 + i) % 127);
                    block.qs[i] = val;
                    sum += val;
                }

                // Scale = 1.0 for simplicity
                block.d = fp32_to_fp16(1.0f);

                // sum_qs should be sum × d (but d=1.0, so just sum)
                block.sum_qs = fp32_to_fp16(static_cast<float>(sum));
            }
        }

        return blocks;
    }

    /**
     * @brief Create simple Q4_0 weight tensor for testing
     *
     * Creates N rows × K cols quantized to Q4_0 format.
     */
    std::shared_ptr<Q4_0Tensor> createSimpleQ4_0Weights(int N, int K)
    {
        EXPECT_EQ(K % 32, 0) << "K must be multiple of 32";

        // Create FP32 weights first
        std::vector<float> weights_fp32(N * K);
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K; ++k)
            {
                // Simple pattern: small integers
                weights_fp32[n * K + k] = static_cast<float>((n + k) % 16);
            }
        }

        // Quantize to Q4_0 manually
        size_t k_blocks = K / 32;
        size_t total_blocks = N * k_blocks;
        size_t bytes_per_block = sizeof(uint16_t) + 16; // FP16 scale + 16 bytes (32 × 4-bit)
        std::vector<uint8_t> raw_data(total_blocks * bytes_per_block);

        for (int n = 0; n < N; ++n)
        {
            for (size_t kb = 0; kb < k_blocks; ++kb)
            {
                size_t block_idx = n * k_blocks + kb;
                uint8_t *block_ptr = raw_data.data() + block_idx * bytes_per_block;

                // Find max absolute value
                const float *src = weights_fp32.data() + n * K + kb * 32;
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(src[i]));
                }

                // Compute scale
                float d = max_abs / 7.0f; // 4-bit: [-7, 7]
                if (d < 1e-8f)
                    d = 1.0f;
                float id = 1.0f / d;

                // Store scale (FP16)
                uint16_t d_fp16 = fp32_to_fp16(d);
                std::memcpy(block_ptr, &d_fp16, sizeof(uint16_t));

                // Quantize and pack 4-bit values
                uint8_t *quants = block_ptr + sizeof(uint16_t);
                for (int i = 0; i < 32; i += 2)
                {
                    float val0 = src[i] * id;
                    float val1 = src[i + 1] * id;

                    int8_t q0 = static_cast<int8_t>(std::round(std::clamp(val0, -7.0f, 7.0f))) + 8;
                    int8_t q1 = static_cast<int8_t>(std::round(std::clamp(val1, -7.0f, 7.0f))) + 8;

                    quants[i / 2] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                }
            }
        }

        std::vector<size_t> shape = {static_cast<size_t>(N), static_cast<size_t>(K)};
        return std::make_shared<Q4_0Tensor>(shape, raw_data);
    }

    /**
     * @brief Reference FP32 GEMM: C = A @ B^T
     */
    void referenceGEMM(const float *A, const float *B, float *C, int M, int N, int K)
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
     * @brief Print matrix for debugging
     */
    void printMatrix(const char *name, const float *mat, int rows, int cols, int max_rows = 4, int max_cols = 8)
    {
        std::cout << name << " (" << rows << "×" << cols << "):\n";
        for (int i = 0; i < std::min(rows, max_rows); ++i)
        {
            std::cout << "  [";
            for (int j = 0; j < std::min(cols, max_cols); ++j)
            {
                std::cout << mat[i * cols + j];
                if (j < std::min(cols, max_cols) - 1)
                    std::cout << ", ";
            }
            if (cols > max_cols)
                std::cout << ", ...";
            std::cout << "]\n";
        }
        if (rows > max_rows)
            std::cout << "  ...\n";
    }
};

// =============================================================================
// UNIT TESTS
// =============================================================================

/**
 * @brief Test with minimal dimensions (2×2×64)
 *
 * Smallest reasonable test case to verify basic functionality.
 */
TEST_F(Test__OneDNNBlockwiseGEMM, MinimalDimensions_2x2x64)
{
    const int M = 2;  // Activation rows
    const int N = 2;  // Weight rows (output cols)
    const int K = 64; // Inner dimension (2 blocks)

    std::cout << "\n=== Test: MinimalDimensions_2x2x64 ===\n";

    // Create Q8_1 activations
    auto q8_1_blocks = createSimpleQ8_1Activations(M, K);

    // Create Q4_0 weights
    auto q4_0_weights = createSimpleQ4_0Weights(N, K);

    // Dequantize for reference
    std::vector<float> activations_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        size_t k_blocks = K / 32;
        for (size_t kb = 0; kb < k_blocks; ++kb)
        {
            const Q8_1Block &block = q8_1_blocks[m * k_blocks + kb];
            float d = fp16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                activations_fp32[m * K + kb * 32 + i] = static_cast<float>(block.qs[i]) * d;
            }
        }
    }

    std::vector<float> weights_fp32(N * K);
    q4_0_weights->to_fp32(weights_fp32.data());

    std::cout << "Created Q8_1 activations: " << M << "×" << K << " (" << q8_1_blocks.size() << " blocks)\n";
    std::cout << "Created Q4_0 weights: " << N << "×" << K << "\n";

    printMatrix("Activations (FP32)", activations_fp32.data(), M, K, 2, 8);
    printMatrix("Weights (FP32)", weights_fp32.data(), N, K, 2, 8);

    // Reference path: FP32 GEMM
    std::vector<float> reference_output(M * N, 0.0f);
    referenceGEMM(activations_fp32.data(), weights_fp32.data(), reference_output.data(), M, N, K);

    std::cout << "\nReference output:\n";
    printMatrix("C_ref", reference_output.data(), M, N, M, N);

    // Microkernel path
    OneDNNGemmKernel gemm_kernel(q4_0_weights.get());

    // Pack weights
    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value()) << "Failed to pack weights";

    std::cout << "\nWeight pack created:\n";
    std::cout << "  unpacked_s8 size: " << weight_pack->unpacked_s8.size() << " (expected " << K * N << ")\n";
    std::cout << "  block_scales size: " << weight_pack->block_scales.size() << " (expected " << (K / 32) * N << ")\n";

    // Execute microkernel
    std::vector<float> microkernel_output(M * N, 0.0f);
    bool success = gemm_kernel.execute_blockwise_gemm_test(
        q8_1_blocks.data(),
        weight_pack.value(),
        microkernel_output.data(),
        M, N, K,
        nullptr, // No bias
        1.0f,    // alpha
        0.0f     // beta
    );
    ASSERT_TRUE(success) << "Blockwise GEMM execution failed";

    std::cout << "\nMicrokernel output:\n";
    printMatrix("C_micro", microkernel_output.data(), M, N, M, N);

    // Compare outputs
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int max_abs_idx = -1, max_rel_idx = -1;

    for (int i = 0; i < M * N; ++i)
    {
        float abs_diff = std::abs(reference_output[i] - microkernel_output[i]);
        float rel_diff = abs_diff / (std::abs(reference_output[i]) + 1e-6f);

        if (abs_diff > max_abs_diff)
        {
            max_abs_diff = abs_diff;
            max_abs_idx = i;
        }
        if (rel_diff > max_rel_diff)
        {
            max_rel_diff = rel_diff;
            max_rel_idx = i;
        }
    }

    std::cout << "\nError analysis:\n";
    std::cout << "  Max absolute diff: " << max_abs_diff << " at index " << max_abs_idx
              << " (ref=" << reference_output[max_abs_idx] << ", micro=" << microkernel_output[max_abs_idx] << ")\n";
    std::cout << "  Max relative diff: " << max_rel_diff << " at index " << max_rel_idx
              << " (ref=" << reference_output[max_rel_idx] << ", micro=" << microkernel_output[max_rel_idx] << ")\n";

    // Quantization error tolerance (Q4_0 is 4-bit, so expect some error)
    // For small test values, absolute error is more meaningful
    EXPECT_LT(max_abs_diff, K * 0.5f) << "Max absolute difference too large";
    EXPECT_LT(max_rel_diff, 0.15f) << "Max relative difference too large (15%)";
}

/**
 * @brief Test with slightly larger dimensions (4×8×128)
 */
TEST_F(Test__OneDNNBlockwiseGEMM, SmallBatch_4x8x128)
{
    const int M = 4;   // Activation rows
    const int N = 8;   // Weight rows (output cols)
    const int K = 128; // Inner dimension (4 blocks)

    std::cout << "\n=== Test: SmallBatch_4x8x128 ===\n";

    // Create inputs
    auto q8_1_blocks = createSimpleQ8_1Activations(M, K);
    auto q4_0_weights = createSimpleQ4_0Weights(N, K);

    // Dequantize for reference
    std::vector<float> activations_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        size_t k_blocks = K / 32;
        for (size_t kb = 0; kb < k_blocks; ++kb)
        {
            const Q8_1Block &block = q8_1_blocks[m * k_blocks + kb];
            float d = fp16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                activations_fp32[m * K + kb * 32 + i] = static_cast<float>(block.qs[i]) * d;
            }
        }
    }

    std::vector<float> weights_fp32(N * K);
    q4_0_weights->to_fp32(weights_fp32.data());

    // Reference GEMM
    std::vector<float> reference_output(M * N, 0.0f);
    referenceGEMM(activations_fp32.data(), weights_fp32.data(), reference_output.data(), M, N, K);

    // Microkernel GEMM
    OneDNNGemmKernel gemm_kernel(q4_0_weights.get());
    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value()) << "Failed to pack weights";

    std::vector<float> microkernel_output(M * N, 0.0f);
    bool success = gemm_kernel.execute_blockwise_gemm_test(
        q8_1_blocks.data(),
        weight_pack.value(),
        microkernel_output.data(),
        M, N, K,
        nullptr, 1.0f, 0.0f);
    ASSERT_TRUE(success) << "Blockwise GEMM execution failed";

    // Compare
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float abs_diff = std::abs(reference_output[i] - microkernel_output[i]);
        float rel_diff = abs_diff / (std::abs(reference_output[i]) + 1e-6f);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }

    std::cout << "Max absolute diff: " << max_abs_diff << "\n";
    std::cout << "Max relative diff: " << max_rel_diff << "\n";

    EXPECT_LT(max_abs_diff, K * 0.5f) << "Max absolute difference too large";
    EXPECT_LT(max_rel_diff, 0.15f) << "Max relative difference too large (15%)";
}
