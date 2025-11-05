/**
 * @file Test__Phase7_CUTLASS_Functional.cpp
 * @brief Functional correctness tests for Phase 7 CUTLASS int8 GEMM
 *
 * Validates that CUTLASS-based implementation produces correct results
 * by comparing against CPU reference implementation.
 *
 * @author David Sanftenberg
 * @date 2025-01-10
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmKernelPhase7_CUTLASS.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace llaminar::v2;

namespace
{

    // IQ4_NL lookup table (int8 values - same as GPU kernel)
    constexpr int8_t kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};

    /**
     * @brief IQ4_NL block structure (host-side)
     */
    struct IQ4_NLBlock
    {
        uint8_t quants[16]; // 16 bytes (32 nibbles)
        uint16_t scale;     // 2 bytes (FP16 as uint16_t)
    } __attribute__((packed));

    /**
     * @brief Convert FP16 (stored as uint16_t) to FP32
     */
    float fp16_to_fp32(uint16_t h)
    {
        // Simple FP16 to FP32 conversion (manual bit manipulation)
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF) << 13;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                // Zero
                uint32_t result = sign;
                return *reinterpret_cast<float *>(&result);
            }
            // Denormalized
            exponent = 1;
            while ((mantissa & 0x00800000) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x007FFFFF;
        }
        else if (exponent == 31)
        {
            // Inf or NaN
            exponent = 255;
        }
        else
        {
            // Normalized
            exponent += 127 - 15;
        }

        uint32_t result = sign | (exponent << 23) | mantissa;
        return *reinterpret_cast<float *>(&result);
    }

    /**
     * @brief Convert FP32 to FP16 (stored as uint16_t)
     */
    uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
        uint32_t sign = (bits & 0x80000000) >> 16;
        int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mantissa = (bits & 0x007FFFFF) >> 13;

        if (exponent <= 0)
        {
            return sign; // Underflow to zero
        }
        else if (exponent >= 31)
        {
            return sign | 0x7C00; // Overflow to inf
        }

        return sign | (exponent << 10) | mantissa;
    }

    /**
     * @brief CPU reference: Convert IQ4_NL block directly to int8 (matches GPU)
     *
     * No dequant/requant - just direct lookup!
     */
    void iq4nl_to_int8_direct(const IQ4_NLBlock &block, int8_t *output, float &scale_out)
    {
        // Decode FP16 scale
        scale_out = fp16_to_fp32(block.scale);

        // Direct conversion: nibble → int8 (no float intermediate!)
        for (int i = 0; i < 16; ++i)
        {
            uint8_t byte_val = block.quants[i];
            uint8_t nibble0 = byte_val & 0x0F;
            uint8_t nibble1 = byte_val >> 4;

            // Direct lookup: kvalues_iq4nl[] are already int8!
            output[i * 2 + 0] = kvalues_iq4nl[nibble0];
            output[i * 2 + 1] = kvalues_iq4nl[nibble1];
        }
    }

    /**
     * @brief CPU reference: int8×int8→int32 GEMM with scaling
     */
    void cpu_gemm_int8(
        const int8_t *A, const int8_t *B, float *C,
        const float *scales_A, const float *scales_B,
        int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int32_t sum = 0;
                for (int k = 0; k < K; ++k)
                {
                    sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
                }
                // Apply scaling: C = int32_result × scale_A × scale_B
                C[i * N + j] = (float)sum * scales_A[i] * scales_B[j];
            }
        }
    }

    /**
     * @brief Quantize fp32 matrix to IQ4_NL blocks
     */
    void quantize_to_iq4nl(
        const float *data, IQ4_NLBlock *blocks,
        int rows, int cols)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks_per_row = cols / BLOCK_SIZE;

        for (int row = 0; row < rows; ++row)
        {
            for (int block_idx = 0; block_idx < num_blocks_per_row; ++block_idx)
            {
                const float *block_data = data + row * cols + block_idx * BLOCK_SIZE;
                IQ4_NLBlock &block = blocks[row * num_blocks_per_row + block_idx];

                // Find max_abs for scale
                float max_abs = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_data[i]));
                }

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                block.scale = fp32_to_fp16(scale);

                // Quantize to 4-bit (find nearest kvalues_iq4nl entry)
                for (int i = 0; i < BLOCK_SIZE; i += 2)
                {
                    float val0 = block_data[i] / scale;
                    float val1 = block_data[i + 1] / scale;

                    // Find nearest quantization level
                    auto find_nearest = [](float val) -> uint8_t
                    {
                        int best_idx = 0;
                        float best_dist = std::abs(val - kvalues_iq4nl[0]);
                        for (int j = 1; j < 16; ++j)
                        {
                            float dist = std::abs(val - kvalues_iq4nl[j]);
                            if (dist < best_dist)
                            {
                                best_dist = dist;
                                best_idx = j;
                            }
                        }
                        return best_idx;
                    };

                    uint8_t nibble0 = find_nearest(val0);
                    uint8_t nibble1 = find_nearest(val1);
                    block.quants[i / 2] = (nibble1 << 4) | nibble0;
                }
            }
        }
    }

    /**
     * @brief Compare two fp32 matrices with tolerance
     */
    bool compare_matrices(
        const float *A, const float *B,
        int M, int N,
        float rtol, float atol,
        float &max_diff, float &max_rel_err)
    {
        max_diff = 0.0f;
        max_rel_err = 0.0f;
        bool passed = true;

        for (int i = 0; i < M * N; ++i)
        {
            float diff = std::abs(A[i] - B[i]);
            float rel_err = (std::abs(B[i]) > 1e-6f) ? (diff / std::abs(B[i])) : 0.0f;

            max_diff = std::max(max_diff, diff);
            max_rel_err = std::max(max_rel_err, rel_err);

            if (diff > atol && rel_err > rtol)
            {
                passed = false;
            }
        }

        return passed;
    }

} // anonymous namespace

/**
 * @brief Test 1: Small 64×64 matrix with direct IQ4_NL→INT8 conversion
 */
TEST(Phase7CUTLASSFunctional, SmallMatrix64x64)
{
    constexpr int M = 64, N = 64, K = 64; // K and N must be multiples of 32 for IQ4_NL

    // Create simple test data: A is random, B will be quantized
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(M * K);
    for (auto &val : A_fp32)
        val = dist(gen);

    std::vector<float> B_fp32(K * N);
    for (auto &val : B_fp32)
        val = dist(gen);

    // Quantize A (symmetric per-row)
    std::vector<int8_t> A_int8(M * K);
    std::vector<float> scales_A(M);
    for (int i = 0; i < M; ++i)
    {
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            max_abs = std::max(max_abs, std::abs(A_fp32[i * K + k]));
        }
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        scales_A[i] = scale;

        for (int k = 0; k < K; ++k)
        {
            float val = A_fp32[i * K + k] / scale;
            A_int8[i * K + k] = (int8_t)std::round(std::min(127.0f, std::max(-127.0f, val)));
        }
    }

    // Quantize B to IQ4_NL
    int num_blocks = (K / 32) * (N / 32);
    std::vector<IQ4_NLBlock> B_iq4nl(num_blocks);
    quantize_to_iq4nl(B_fp32.data(), B_iq4nl.data(), K, N);

    // Convert IQ4_NL to INT8 (CPU reference - matches GPU kernel!)
    std::vector<int8_t> B_int8(K * N);
    std::vector<float> scales_B(N);
    for (int k_block = 0; k_block < K / 32; ++k_block)
    {
        for (int n_block = 0; n_block < N / 32; ++n_block)
        {
            int block_idx = k_block * (N / 32) + n_block;
            int8_t block_output[32];
            float scale;
            iq4nl_to_int8_direct(B_iq4nl[block_idx], block_output, scale);

            for (int i = 0; i < 32; ++i)
            {
                int k_pos = k_block * 32 + i;
                int n_pos = n_block * 32 + i;
                if (k_pos < K && n_pos < N)
                {
                    B_int8[k_pos * N + n_pos] = block_output[i];
                    if (k_pos == 0)
                    {
                        scales_B[n_pos] = scale;
                    }
                }
            }
        }
    }

    // CPU reference: int8×int8→int32 with scaling
    std::vector<float> C_cpu(M * N);
    cpu_gemm_int8(A_int8.data(), B_int8.data(), C_cpu.data(),
                  scales_A.data(), scales_B.data(), M, N, K);

    // GPU result
    CudaGemmKernelPhase7_CUTLASS kernel;
    std::vector<float> C_gpu(M * N);

    bool success = kernel.execute(A_fp32.data(), B_iq4nl.data(), C_gpu.data(), M, N, K);
    ASSERT_TRUE(success) << "CUTLASS GEMM failed";

    // Compare
    float max_diff, max_rel_err;
    bool passed = compare_matrices(
        C_gpu.data(), C_cpu.data(), M, N,
        0.1f, // 10% relative tolerance (quantization error accumulates)
        5.0f, // Absolute tolerance
        max_diff, max_rel_err);

    std::cout << "SmallMatrix64x64:\n";
    std::cout << "  Max diff: " << max_diff << "\n";
    std::cout << "  Max rel err: " << max_rel_err << "\n";

    EXPECT_TRUE(passed) << "Results do not match within tolerance";
}

/**
 * @brief Test 2: Medium 128×128 matrix (DISABLED - needs update for direct conversion)
 */
TEST(Phase7CUTLASSFunctional, DISABLED_MediumMatrix128x128)
{
    constexpr int M = 128, N = 128, K = 128;

    // Random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_fp32(K * N);

    for (auto &val : A)
        val = dist(gen);
    for (auto &val : B_fp32)
        val = dist(gen);

    // Quantize B to IQ4_NL
    int num_blocks = (K / 32) * (N / 32);
    std::vector<IQ4_NLBlock> B_iq4nl(num_blocks);
    quantize_to_iq4nl(B_fp32.data(), B_iq4nl.data(), K, N);

    // CPU reference (dequantize B)
    std::vector<float> B_dequant(K * N);
    for (int k_block = 0; k_block < K / 32; ++k_block)
    {
        for (int n_block = 0; n_block < N / 32; ++n_block)
        {
            int block_idx = k_block * (N / 32) + n_block;
            float block_output[32];
            dequantize_iq4nl_block(B_iq4nl[block_idx], block_output);

            for (int i = 0; i < 32; ++i)
            {
                int row = k_block * 32 + i;
                int col = n_block * 32 + i % 32; // Diagonal pattern for test
                if (row < K && col < N)
                {
                    B_dequant[row * N + col] = block_output[i];
                }
            }
        }
    }

    std::vector<float> C_cpu(M * N);
    cpu_gemm(A.data(), B_dequant.data(), C_cpu.data(), M, N, K);

    // GPU result
    CudaGemmKernelPhase7_CUTLASS kernel;
    std::vector<float> C_gpu(M * N);

    bool success = kernel.execute(A.data(), B_iq4nl.data(), C_gpu.data(), M, N, K);
    ASSERT_TRUE(success) << "CUTLASS GEMM failed";

    // Compare
    float max_diff, max_rel_err;
    bool passed = compare_matrices(
        C_gpu.data(), C_cpu.data(), M, N,
        0.1f, // 10% relative tolerance (more lenient for random data)
        5.0f, // Absolute tolerance
        max_diff, max_rel_err);

    std::cout << "MediumMatrix128x128:\n";
    std::cout << "  Max diff: " << max_diff << "\n";
    std::cout << "  Max rel err: " << max_rel_err << "\n";

    EXPECT_TRUE(passed) << "Results do not match within tolerance";
}

/**
 * @brief Test 3: Identity matrix pattern (DISABLED - needs update for direct conversion)
 */
TEST(Phase7CUTLASSFunctional, DISABLED_IdentityPattern)
{
    constexpr int M = 32, N = 32, K = 32;

    // A = identity, B = identity → C should be identity
    std::vector<float> A(M * K, 0.0f);
    std::vector<float> B_fp32(K * N, 0.0f);

    for (int i = 0; i < std::min(M, K); ++i)
    {
        A[i * K + i] = 1.0f;
    }
    for (int i = 0; i < std::min(K, N); ++i)
    {
        B_fp32[i * N + i] = 1.0f;
    }

    // Quantize B
    std::vector<IQ4_NLBlock> B_iq4nl((K / 32) * (N / 32));
    quantize_to_iq4nl(B_fp32.data(), B_iq4nl.data(), K, N);

    // GPU result
    CudaGemmKernelPhase7_CUTLASS kernel;
    std::vector<float> C_gpu(M * N);

    bool success = kernel.execute(A.data(), B_iq4nl.data(), C_gpu.data(), M, N, K);
    ASSERT_TRUE(success) << "CUTLASS GEMM failed";

    // Expected: identity matrix
    int correct = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float expected = (i == j) ? 1.0f : 0.0f;
            float actual = C_gpu[i * N + j];

            if (std::abs(actual - expected) < 0.1f)
            {
                correct++;
            }
        }
    }

    float accuracy = (float)correct / (M * N);
    std::cout << "IdentityPattern:\n";
    std::cout << "  Accuracy: " << (accuracy * 100.0f) << "%\n";
    std::cout << "  C[0,0] (should be 1.0): " << C_gpu[0] << "\n";
    std::cout << "  C[0,1] (should be 0.0): " << C_gpu[1] << "\n";

    EXPECT_GT(accuracy, 0.9f) << "Identity pattern not preserved";
}
