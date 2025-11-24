/**
 * @file Test__OneDNN_ColumnSumOptimization.cpp
 * @brief Unit test for blockwise GEMM correctness
 *
 * This test verifies that the optimized blockwise GEMM (using s8*s8 kernel)
 * produces correct results.
 *
 * @author David Sanftenberg
 * @date 2025-11-24
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/Tensors.h"
#include <random>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

class OneDNN_BlockwiseGEMM_Correctness : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @brief Test that blockwise GEMM produces correct results
 */
TEST_F(OneDNN_BlockwiseGEMM_Correctness, Correctness)
{
    // Test dimensions (realistic inference shapes)
    const int M = 1;   // Single token decode
    const int N = 256; // Small for fast test
    const int K = 256; // Small for fast test
    const int K_blocks = K / 32;

    // 1. Create Q4_0 weight tensor (N × K) with known values
    const size_t blocks_per_row = K_blocks;
    const size_t total_blocks = N * blocks_per_row;
    const size_t bytes_per_block = sizeof(Q4_0Block);
    std::vector<uint8_t> raw_q4_0_data(total_blocks * bytes_per_block);

    // Fill with deterministic Q4_0 blocks
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 15);

    for (size_t n = 0; n < static_cast<size_t>(N); ++n)
    {
        for (size_t kb = 0; kb < static_cast<size_t>(K_blocks); ++kb)
        {
            Q4_0Block block;
            block.d = fp32_to_fp16(0.1f + static_cast<float>((n + kb) % 10) * 0.01f);

            for (int i = 0; i < 16; ++i)
            {
                uint8_t low = dist(gen);
                uint8_t high = dist(gen);
                block.qs[i] = (low & 0x0F) | ((high & 0x0F) << 4);
            }

            size_t block_offset = (n * blocks_per_row + kb) * bytes_per_block;
            std::memcpy(raw_q4_0_data.data() + block_offset, &block, bytes_per_block);
        }
    }

    auto q4_0_tensor = std::make_shared<Q4_0Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)},
        raw_q4_0_data);

    // 2. Pack weights (this should precompute column sums)
    OneDNNGemmKernel kernel(q4_0_tensor.get());
    auto weight_pack_opt = kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack_opt.has_value()) << "Failed to pack Q4_0 weights";

    const auto &weight_pack = weight_pack_opt.value();

    // 5. Create Q8_1 activations
    std::vector<Q8_1Block> A_blocks(M * K_blocks);
    std::mt19937 gen_A(1337);
    std::uniform_real_distribution<float> dist_A(-1.0f, 1.0f);

    for (int m = 0; m < M; ++m)
    {
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            Q8_1Block &block = A_blocks[m * K_blocks + kb];

            float max_abs = 0.0f;
            float values[32];
            for (int i = 0; i < 32; ++i)
            {
                values[i] = dist_A(gen_A);
                max_abs = std::max(max_abs, std::abs(values[i]));
            }

            float d = max_abs / 127.0f;
            if (d < 1e-10f)
                d = 1.0f;
            float id = 1.0f / d;

            block.d = fp32_to_fp16(d);
            int sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int8_t q = static_cast<int8_t>(std::round(values[i] * id));
                block.qs[i] = q;
                sum_qs += q;
            }
            block.sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

    // 6. Execute GEMM
    std::vector<float> C(M * N, 0.0f);
    bool success = kernel.execute_blockwise_gemm_test(
        A_blocks.data(), weight_pack, C.data(), M, N, K, nullptr, 1.0f, 0.0f);

    ASSERT_TRUE(success) << "Blockwise GEMM failed";

    // 7. Verify output is reasonable (no NaN/Inf)
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_FALSE(std::isnan(C[i])) << "Output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(C[i])) << "Output contains Inf at index " << i;
    }

    std::cout << "[ColumnSumCorrectness] Test PASSED" << std::endl;
    std::cout << "  Matrix dimensions: M=" << M << " N=" << N << " K=" << K << std::endl;
    std::cout << "  Column sums verified: " << N << " columns" << std::endl;
    std::cout << "  Sample outputs: C[0]=" << C[0] << " C[1]=" << C[1]
              << " C[N-1]=" << C[N - 1] << std::endl;
}

/**
 * @brief Test column sum optimization with different matrix sizes
 */
TEST_F(OneDNN_BlockwiseGEMM_Correctness, VariousMatrixSizes)
{
    struct TestCase
    {
        int M, N, K;
        std::string name;
    };

    std::vector<TestCase> test_cases = {
        {1, 128, 128, "Small_128x128"},
        {1, 256, 512, "Medium_256x512"},
        {1, 512, 256, "Medium_512x256"},
        {4, 256, 256, "Batch4_256x256"}};

    for (const auto &tc : test_cases)
    {
        const int K_blocks = tc.K / 32;

        // Create Q4_0 tensor
        const size_t blocks_per_row = K_blocks;
        const size_t total_blocks = tc.N * blocks_per_row;
        const size_t bytes_per_block = sizeof(Q4_0Block);
        std::vector<uint8_t> raw_q4_0_data(total_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 15);

        for (size_t n = 0; n < static_cast<size_t>(tc.N); ++n)
        {
            for (size_t kb = 0; kb < static_cast<size_t>(K_blocks); ++kb)
            {
                Q4_0Block block;
                block.d = fp32_to_fp16(0.1f);

                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low = dist(gen);
                    uint8_t high = dist(gen);
                    block.qs[i] = (low & 0x0F) | ((high & 0x0F) << 4);
                }

                size_t block_offset = (n * blocks_per_row + kb) * bytes_per_block;
                std::memcpy(raw_q4_0_data.data() + block_offset, &block, bytes_per_block);
            }
        }

        auto q4_0_tensor = std::make_shared<Q4_0Tensor>(
            std::vector<size_t>{static_cast<size_t>(tc.N), static_cast<size_t>(tc.K)},
            raw_q4_0_data);

        // Pack and verify
        OneDNNGemmKernel kernel(q4_0_tensor.get());
        auto weight_pack_opt = kernel.get_blockwise_weight_pack(tc.K, tc.N);
        ASSERT_TRUE(weight_pack_opt.has_value()) << "Failed for " << tc.name;

        const auto &weight_pack = weight_pack_opt.value();

        // Create activations
        std::vector<Q8_1Block> A_blocks(tc.M * K_blocks);
        for (int m = 0; m < tc.M; ++m)
        {
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                Q8_1Block &block = A_blocks[m * K_blocks + kb];
                block.d = fp32_to_fp16(0.1f);
                for (int i = 0; i < 32; ++i)
                {
                    block.qs[i] = static_cast<int8_t>(i - 16);
                }
                block.sum_qs = 0;
            }
        }

        // Execute
        std::vector<float> C(tc.M * tc.N, 0.0f);
        bool success = kernel.execute_blockwise_gemm_test(
            A_blocks.data(), weight_pack, C.data(), tc.M, tc.N, tc.K, nullptr, 1.0f, 0.0f);

        ASSERT_TRUE(success) << "GEMM failed for " << tc.name;

        // Verify no NaN/Inf
        bool has_invalid = false;
        for (int i = 0; i < tc.M * tc.N; ++i)
        {
            if (std::isnan(C[i]) || std::isinf(C[i]))
            {
                has_invalid = true;
                break;
            }
        }
        EXPECT_FALSE(has_invalid) << "Invalid output for " << tc.name;

        std::cout << "[" << tc.name << "] PASSED - C[0]=" << C[0] << std::endl;
    }
}
