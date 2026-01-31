/**
 * @file Test__Q16MPI_Allreduce.cpp
 * @brief Integration tests for Q16 variable block size MPI allreduce operations
 *
 * Tests the templated allreduce_q16_inplace<BlockType>() for all supported
 * Q16 block sizes (32, 64, 128, 192) across 2 MPI ranks.
 *
 * Key Behavior: q16_sum_n performs proper dequantize-accumulate-requantize:
 * 1. Dequantizes each input: value = scale * qs[i]
 * 2. Accumulates in FP32
 * 3. Requantizes with new scale based on max_abs
 *
 * Tests verify that the DEQUANTIZED output matches the expected FP32 sum
 * (with small tolerance for requantization error).
 *
 * Requires: MPI with 2 ranks (run via mpirun -np 2)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>

#include "v2/utils/MPIContext.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/TensorFactory.h"

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Helper to dequantize a Q16 block element
         */
        template <typename BlockType>
        float dequantize_element(const BlockType &block, size_t idx)
        {
            return block.d * static_cast<float>(block.qs[idx]);
        }

        /**
         * @brief Helper to compute expected FP32 sum for verification
         * Uses relative tolerance since requantization introduces small errors
         */
        constexpr float Q16_TOLERANCE = 1e-3f;

        class Q16MPIAllreduceTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                int initialized;
                MPI_Initialized(&initialized);
                if (!initialized)
                {
                    GTEST_SKIP() << "MPI not initialized";
                }

                MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

                if (world_size_ != 2)
                {
                    GTEST_SKIP() << "Test requires exactly 2 MPI ranks, got " << world_size_;
                }

                mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            }

            int rank_ = 0;
            int world_size_ = 1;
            std::shared_ptr<MPIContext> mpi_ctx_;
        };

        // =============================================================================
        // Q16_1Block (32 elements) Tests
        // =============================================================================

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block32_SingleBlock)
        {
            constexpr size_t n_blocks = 1;
            Q16_1Block block;

            // Initialize with rank-specific values
            // Rank 0: scale=1.0, qs[i] = 100 + i → FP32 value = 100 + i
            // Rank 1: scale=1.0, qs[i] = 200 + i → FP32 value = 200 + i
            // Expected sum: 300 + 2*i
            const float scale = 1.0f;
            block.d = scale;
            block.sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                block.qs[i] = static_cast<int16_t>((rank_ + 1) * 100 + i);
                block.sum_qs += block.qs[i];
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block>(&block, n_blocks);

            // Verify dequantized output matches expected sum
            for (int i = 0; i < 32; ++i)
            {
                float expected_sum = static_cast<float>(300 + 2 * i); // (100+i) + (200+i)
                float actual = dequantize_element(block, i);
                EXPECT_NEAR(actual, expected_sum, Q16_TOLERANCE * std::abs(expected_sum) + 0.5f)
                    << "Mismatch at element " << i;
            }

            // Verify sum_qs is consistent with qs values
            int32_t computed_sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                computed_sum += block.qs[i];
            }
            EXPECT_EQ(block.sum_qs, computed_sum) << "sum_qs inconsistent";
        }

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block32_MultipleBlocks)
        {
            constexpr size_t n_blocks = 4;
            std::vector<Q16_1Block> blocks(n_blocks);

            // Store expected FP32 values per block for verification
            std::vector<std::vector<float>> expected_sums(n_blocks, std::vector<float>(32));

            // Initialize blocks with different patterns per rank
            for (size_t b = 0; b < n_blocks; ++b)
            {
                const float scale = static_cast<float>(b + 1);
                blocks[b].d = scale;
                blocks[b].sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int16_t val = static_cast<int16_t>((rank_ + 1) * 100 + b * 10 + i);
                    blocks[b].qs[i] = val;
                    blocks[b].sum_qs += val;

                    // Expected sum = scale * rank0_val + scale * rank1_val
                    // = scale * ((100 + b*10 + i) + (200 + b*10 + i))
                    // = scale * (300 + 2*b*10 + 2*i)
                    expected_sums[b][i] = scale * (300.0f + 2.0f * b * 10.0f + 2.0f * i);
                }
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block>(blocks.data(), n_blocks);

            // Verify each block's dequantized values
            for (size_t b = 0; b < n_blocks; ++b)
            {
                for (int i = 0; i < 32; ++i)
                {
                    float actual = dequantize_element(blocks[b], i);
                    float expected = expected_sums[b][i];
                    EXPECT_NEAR(actual, expected, Q16_TOLERANCE * std::abs(expected) + 0.5f)
                        << "Block " << b << " element " << i << " mismatch";
                }
            }
        }

        // =============================================================================
        // Q16_1Block_64 (64 elements) Tests
        // =============================================================================

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block64_SingleBlock)
        {
            constexpr size_t n_blocks = 1;
            Q16_1Block_64 block;

            // Different scales per rank to test proper dequant-requant
            // Rank 0: scale=0.5, qs[i] = 100 → FP32 value = 50
            // Rank 1: scale=1.0, qs[i] = 100 → FP32 value = 100
            // Expected sum: 150
            const float scale = 0.5f * (rank_ + 1); // Rank 0: 0.5, Rank 1: 1.0
            block.d = scale;
            block.sum_qs = 0;
            for (int i = 0; i < 64; ++i)
            {
                block.qs[i] = static_cast<int16_t>(100 + (i % 16));
                block.sum_qs += block.qs[i];
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_64>(&block, n_blocks);

            // Verify dequantized output
            for (int i = 0; i < 64; ++i)
            {
                // Rank 0: 0.5 * (100 + i%16), Rank 1: 1.0 * (100 + i%16)
                // Sum: 1.5 * (100 + i%16)
                float expected = 1.5f * static_cast<float>(100 + (i % 16));
                float actual = dequantize_element(block, i);
                EXPECT_NEAR(actual, expected, Q16_TOLERANCE * std::abs(expected) + 0.5f)
                    << "Mismatch at element " << i;
            }
        }

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block64_HeadDimAligned)
        {
            // Test with n_blocks matching typical attention head count
            // Qwen2.5-0.5B: 14 heads with head_dim=64 → 14 blocks
            constexpr size_t n_heads = 14;
            std::vector<Q16_1Block_64> blocks(n_heads);
            std::vector<std::vector<float>> expected(n_heads, std::vector<float>(64));

            for (size_t h = 0; h < n_heads; ++h)
            {
                blocks[h].d = 1.0f;
                blocks[h].sum_qs = 0;
                for (int i = 0; i < 64; ++i)
                {
                    // Each head gets unique values, rank-differentiated
                    int16_t val = static_cast<int16_t>(rank_ * 500 + h * 10 + (i % 10));
                    blocks[h].qs[i] = val;
                    blocks[h].sum_qs += val;

                    // Sum: (0 + h*10 + i%10) + (500 + h*10 + i%10) = 500 + 2*h*10 + 2*(i%10)
                    expected[h][i] = 500.0f + 2.0f * h * 10.0f + 2.0f * (i % 10);
                }
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_64>(blocks.data(), n_heads);

            for (size_t h = 0; h < n_heads; ++h)
            {
                for (int i = 0; i < 64; ++i)
                {
                    float actual = dequantize_element(blocks[h], i);
                    EXPECT_NEAR(actual, expected[h][i], Q16_TOLERANCE * std::abs(expected[h][i]) + 0.5f)
                        << "Head " << h << " elem " << i << " mismatch";
                }
            }
        }

        // =============================================================================
        // Q16_1Block_128 (128 elements) Tests
        // =============================================================================

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block128_SingleBlock)
        {
            constexpr size_t n_blocks = 1;
            Q16_1Block_128 block;

            // Both ranks use scale=2.0
            block.d = 2.0f;
            block.sum_qs = 0;
            for (int i = 0; i < 128; ++i)
            {
                block.qs[i] = static_cast<int16_t>(rank_ * 100 + i);
                block.sum_qs += block.qs[i];
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_128>(&block, n_blocks);

            // Verify dequantized values
            for (int i = 0; i < 128; ++i)
            {
                // Rank 0: 2.0 * (0 + i), Rank 1: 2.0 * (100 + i)
                // Sum: 2.0 * (100 + 2*i)
                float expected = 2.0f * (100.0f + 2.0f * i);
                float actual = dequantize_element(block, i);
                EXPECT_NEAR(actual, expected, Q16_TOLERANCE * std::abs(expected) + 0.5f)
                    << "Mismatch at element " << i;
            }
        }

        TEST_F(Q16MPIAllreduceTest, AllreduceQ16Block128_Llama3Config)
        {
            // Llama3: 32 heads with head_dim=128 → 32 blocks
            constexpr size_t n_heads = 32;
            std::vector<Q16_1Block_128> blocks(n_heads);
            std::vector<std::vector<float>> expected(n_heads, std::vector<float>(128));

            for (size_t h = 0; h < n_heads; ++h)
            {
                blocks[h].d = 1.0f;
                blocks[h].sum_qs = 0;
                for (int i = 0; i < 128; ++i)
                {
                    int16_t val = static_cast<int16_t>(rank_ * 1000 + h);
                    blocks[h].qs[i] = val;
                    blocks[h].sum_qs += val;

                    // All elements in head h: 0*1000 + h + 1*1000 + h = 1000 + 2*h
                    expected[h][i] = 1000.0f + 2.0f * h;
                }
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_128>(blocks.data(), n_heads);

            for (size_t h = 0; h < n_heads; ++h)
            {
                for (int i = 0; i < 128; ++i)
                {
                    float actual = dequantize_element(blocks[h], i);
                    EXPECT_NEAR(actual, expected[h][i], Q16_TOLERANCE * std::abs(expected[h][i]) + 0.5f)
                        << "Head " << h << " elem " << i;
                }
            }
        }

        // Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
        // with independent scales, not a combined 192-dim block.
        // See PROJECT_Q16_INTEGER_ATTENTION_V2.md MLA Architecture section.

        // =============================================================================
        // Edge Case Tests
        // =============================================================================

        TEST_F(Q16MPIAllreduceTest, AllreduceWithNegativeValues)
        {
            constexpr size_t n_blocks = 2;
            std::vector<Q16_1Block_64> blocks(n_blocks);
            std::vector<std::vector<float>> expected(n_blocks, std::vector<float>(64));

            for (size_t b = 0; b < n_blocks; ++b)
            {
                blocks[b].d = 1.0f;
                blocks[b].sum_qs = 0;
                for (int i = 0; i < 64; ++i)
                {
                    // Rank 0: negative, Rank 1: positive (should partially cancel)
                    if (rank_ == 0)
                    {
                        blocks[b].qs[i] = static_cast<int16_t>(-100 - i);
                    }
                    else
                    {
                        blocks[b].qs[i] = static_cast<int16_t>(150 + i);
                    }
                    blocks[b].sum_qs += blocks[b].qs[i];

                    // Sum: (-100 - i) + (150 + i) = 50
                    expected[b][i] = 50.0f;
                }
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_64>(blocks.data(), n_blocks);

            for (size_t b = 0; b < n_blocks; ++b)
            {
                for (int i = 0; i < 64; ++i)
                {
                    float actual = dequantize_element(blocks[b], i);
                    EXPECT_NEAR(actual, expected[b][i], Q16_TOLERANCE * std::abs(expected[b][i]) + 0.5f)
                        << "Block " << b << " elem " << i;
                }
            }
        }

        TEST_F(Q16MPIAllreduceTest, AllreduceWithZeroScale)
        {
            constexpr size_t n_blocks = 1;
            Q16_1Block_128 block;

            // One rank has zero scale (effectively zero contribution)
            block.d = (rank_ == 0) ? 0.0f : 1.0f;
            block.sum_qs = 0;
            for (int i = 0; i < 128; ++i)
            {
                block.qs[i] = static_cast<int16_t>(100 + i);
                block.sum_qs += block.qs[i];
            }

            mpi_ctx_->allreduce_q16_inplace<Q16_1Block_128>(&block, n_blocks);

            // Rank 0 contributes 0.0 * (100+i) = 0, Rank 1 contributes 1.0 * (100+i)
            // Sum: just rank 1's values
            for (int i = 0; i < 128; ++i)
            {
                float expected = static_cast<float>(100 + i);
                float actual = dequantize_element(block, i);
                EXPECT_NEAR(actual, expected, Q16_TOLERANCE * std::abs(expected) + 0.5f)
                    << "Mismatch at element " << i;
            }
        }

    } // namespace test
} // namespace llaminar2

// =============================================================================
// Main with MPI initialization
// =============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
