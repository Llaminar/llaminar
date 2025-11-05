/**
 * @file Test__Phase7_CUTLASS_Functional_Simple.cpp
 * @brief Simplified Phase 7 test with direct IQ4_NL→INT8 conversion
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

    struct IQ4_NLBlock
    {
        uint8_t quants[16];
        uint16_t scale;
    } __attribute__((packed));

    float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF) << 13;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                uint32_t result = sign;
                return *reinterpret_cast<float *>(&result);
            }
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
            exponent = 255;
        }
        else
        {
            exponent += 127 - 15;
        }

        uint32_t result = sign | (exponent << 23) | mantissa;
        return *reinterpret_cast<float *>(&result);
    }

    uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
        uint32_t sign = (bits & 0x80000000) >> 16;
        int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mantissa = (bits & 0x007FFFFF) >> 13;

        if (exponent <= 0)
        {
            return sign;
        }
        else if (exponent >= 31)
        {
            return sign | 0x7C00;
        }

        return sign | (exponent << 10) | mantissa;
    }

    void quantize_to_iq4nl(const float *data, IQ4_NLBlock *blocks, int rows, int cols)
    {
        constexpr int BLOCK_SIZE = 32;
        int num_blocks_per_row = cols / BLOCK_SIZE;

        for (int row = 0; row < rows; ++row)
        {
            for (int block_idx = 0; block_idx < num_blocks_per_row; ++block_idx)
            {
                const float *block_data = data + row * cols + block_idx * BLOCK_SIZE;
                IQ4_NLBlock &block = blocks[row * num_blocks_per_row + block_idx];

                float max_abs = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_data[i]));
                }

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                block.scale = fp32_to_fp16(scale);

                for (int i = 0; i < BLOCK_SIZE; i += 2)
                {
                    float val0 = block_data[i] / scale;
                    float val1 = block_data[i + 1] / scale;

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

} // namespace

TEST(Phase7CUTLASS, DirectIQ4NLConversion64x64)
{
    constexpr int M = 64, N = 64, K = 64;

    // Random input data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(M * K);
    for (auto &val : A_fp32)
        val = dist(gen);

    std::vector<float> B_fp32(K * N);
    for (auto &val : B_fp32)
        val = dist(gen);

    // Quantize B to IQ4_NL
    // B is K×N = 64×64
    // Each block is 32 elements, arranged as 32-element groups along columns
    // For a K×N matrix with 32-element blocks: K rows, N/32 blocks per row
    int num_blocks = (K) * (N / 32);
    std::vector<IQ4_NLBlock> B_iq4nl(num_blocks);

    quantize_to_iq4nl(B_fp32.data(), B_iq4nl.data(), K, N);

    // Run GPU kernel
    CudaGemmKernelPhase7_CUTLASS kernel;
    std::vector<float> C_gpu(M * N);

    bool success = kernel.execute(A_fp32.data(), B_iq4nl.data(), C_gpu.data(), M, N, K);
    ASSERT_TRUE(success) << "CUTLASS GEMM failed";

    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0;
    for (auto val : C_gpu)
    {
        if (std::isnan(val))
            nan_count++;
        if (std::isinf(val))
            inf_count++;
    }

    std::cout << "DirectIQ4NLConversion64x64:\n";
    std::cout << "  NaN count: " << nan_count << " / " << (M * N) << "\n";
    std::cout << "  Inf count: " << inf_count << " / " << (M * N) << "\n";
    std::cout << "  C[0,0]: " << C_gpu[0] << "\n";
    std::cout << "  C[M-1,N-1]: " << C_gpu[M * N - 1] << "\n";

    EXPECT_EQ(nan_count, 0) << "Output contains NaN values";
    EXPECT_EQ(inf_count, 0) << "Output contains Inf values";
}
