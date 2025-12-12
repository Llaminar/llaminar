/**
 * @file Test__MPI_PrecisionAwareAllreduce.cpp
 * @brief MPI integration tests for precision-aware allreduce operations
 *
 * Tests the actual MPI allreduce methods in MPIContext with real multi-rank
 * communication. Verifies correctness of Q8_1, FP16, BF16, and FP32 allreduce
 * across MPI ranks.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Test fixture for MPI precision-aware allreduce integration tests
     */
    class Test__MPI_PrecisionAwareAllreduce : public ::testing::Test
    {
    protected:
        std::shared_ptr<MPIContext> mpi_ctx_;
        int rank_ = 0;
        int world_size_ = 1;
        std::mt19937 rng_;

        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            rng_.seed(42 + rank_); // Different seed per rank for unique data
        }

        void TearDown() override
        {
            mpi_ctx_->barrier();
        }

        // Generate random FP32 values
        std::vector<float> generate_random_fp32(size_t count, float min_val = -1.0f, float max_val = 1.0f)
        {
            std::uniform_real_distribution<float> dist(min_val, max_val);
            std::vector<float> result(count);
            for (auto &v : result)
            {
                v = dist(rng_);
            }
            return result;
        }

        // Convert FP32 to FP16
        std::vector<uint16_t> fp32_to_fp16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> fp16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                fp16[i] = simd::fp32_to_fp16(fp32[i]);
            }
            return fp16;
        }

        // Convert FP32 to BF16
        std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> bf16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                bf16[i] = simd::fp32_to_bf16(fp32[i]);
            }
            return bf16;
        }

        // Convert FP16 to FP32
        std::vector<float> fp16_to_fp32_vec(const std::vector<uint16_t> &fp16)
        {
            std::vector<float> fp32(fp16.size());
            for (size_t i = 0; i < fp16.size(); ++i)
            {
                fp32[i] = simd::fp16_to_fp32(fp16[i]);
            }
            return fp32;
        }

        // Convert BF16 to FP32
        std::vector<float> bf16_to_fp32_vec(const std::vector<uint16_t> &bf16)
        {
            std::vector<float> fp32(bf16.size());
            for (size_t i = 0; i < bf16.size(); ++i)
            {
                fp32[i] = simd::bf16_to_fp32(bf16[i]);
            }
            return fp32;
        }

        // Quantize FP32 to Q8_1 blocks
        std::vector<Q8_1Block> fp32_to_q8_1_blocks(const std::vector<float> &fp32)
        {
            const size_t n_blocks = (fp32.size() + 31) / 32;
            std::vector<Q8_1Block> blocks(n_blocks);

            std::vector<float> padded = fp32;
            padded.resize(n_blocks * 32, 0.0f);

            simd::quantize_fp32_to_q8_1_blocks(padded.data(), blocks.data(), padded.size());
            return blocks;
        }

        // Dequantize Q8_1 blocks to FP32
        std::vector<float> q8_1_blocks_to_fp32(const std::vector<Q8_1Block> &blocks, size_t count)
        {
            std::vector<float> fp32(blocks.size() * 32);
            simd::dequantize_q8_1_to_fp32(blocks.data(), fp32.data(), blocks.size() * 32);
            fp32.resize(count);
            return fp32;
        }

        // Compute cosine similarity
        float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
                return 0.0f;

            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a < 1e-10f || norm_b < 1e-10f)
                return 1.0f;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }
    };

    // =============================================================================
    // FP32 MPI Allreduce Tests
    // =============================================================================

    TEST_F(Test__MPI_PrecisionAwareAllreduce, FP32_Allreduce_TwoRanks)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        const size_t count = 128;

        // Each rank generates unique data
        auto local_data = generate_random_fp32(count);

        // Gather all data to rank 0 for computing reference
        std::vector<float> all_data(count * world_size_);
        MPI_Allgather(local_data.data(), count, MPI_FLOAT,
                      all_data.data(), count, MPI_FLOAT, MPI_COMM_WORLD);

        // Compute reference sum
        std::vector<float> reference(count, 0.0f);
        for (int r = 0; r < world_size_; ++r)
        {
            for (size_t i = 0; i < count; ++i)
            {
                reference[i] += all_data[r * count + i];
            }
        }

        // Perform allreduce
        std::vector<float> result = local_data;
        mpi_ctx_->allreduce_sum_inplace(result.data(), count);

        // Verify
        float sim = cosine_similarity(result, reference);
        EXPECT_GT(sim, 0.9999f) << "FP32 allreduce should match reference exactly";
    }

    // =============================================================================
    // FP16 MPI Allreduce Tests
    // =============================================================================

    TEST_F(Test__MPI_PrecisionAwareAllreduce, FP16_Allreduce_TwoRanks)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        const size_t count = 128;

        // Each rank generates unique FP32 data, converts to FP16
        auto local_fp32 = generate_random_fp32(count);
        auto local_fp16 = fp32_to_fp16_vec(local_fp32);

        // Gather all FP16 data for computing reference (in FP32)
        std::vector<uint16_t> all_fp16(count * world_size_);
        MPI_Allgather(local_fp16.data(), count * sizeof(uint16_t), MPI_BYTE,
                      all_fp16.data(), count * sizeof(uint16_t), MPI_BYTE, MPI_COMM_WORLD);

        // Compute reference sum in FP32
        std::vector<float> reference(count, 0.0f);
        for (int r = 0; r < world_size_; ++r)
        {
            for (size_t i = 0; i < count; ++i)
            {
                reference[i] += simd::fp16_to_fp32(all_fp16[r * count + i]);
            }
        }

        // Perform FP16 allreduce
        std::vector<uint16_t> result = local_fp16;
        mpi_ctx_->allreduce_fp16_inplace(result.data(), count);

        // Convert result to FP32 for comparison
        auto result_fp32 = fp16_to_fp32_vec(result);

        // Verify (allow for FP16 precision loss)
        float sim = cosine_similarity(result_fp32, reference);
        EXPECT_GT(sim, 0.998f) << "FP16 allreduce should have high similarity to reference";
    }

    // =============================================================================
    // BF16 MPI Allreduce Tests
    // =============================================================================

    TEST_F(Test__MPI_PrecisionAwareAllreduce, BF16_Allreduce_TwoRanks)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        const size_t count = 128;

        // Each rank generates unique FP32 data, converts to BF16
        auto local_fp32 = generate_random_fp32(count);
        auto local_bf16 = fp32_to_bf16_vec(local_fp32);

        // Gather all BF16 data for computing reference
        std::vector<uint16_t> all_bf16(count * world_size_);
        MPI_Allgather(local_bf16.data(), count * sizeof(uint16_t), MPI_BYTE,
                      all_bf16.data(), count * sizeof(uint16_t), MPI_BYTE, MPI_COMM_WORLD);

        // Compute reference sum in FP32
        std::vector<float> reference(count, 0.0f);
        for (int r = 0; r < world_size_; ++r)
        {
            for (size_t i = 0; i < count; ++i)
            {
                reference[i] += simd::bf16_to_fp32(all_bf16[r * count + i]);
            }
        }

        // Perform BF16 allreduce
        std::vector<uint16_t> result = local_bf16;
        mpi_ctx_->allreduce_bf16_inplace(result.data(), count);

        // Convert result to FP32 for comparison
        auto result_fp32 = bf16_to_fp32_vec(result);

        // Verify (allow for BF16 precision loss)
        float sim = cosine_similarity(result_fp32, reference);
        EXPECT_GT(sim, 0.997f) << "BF16 allreduce should have high similarity to reference";
    }

    // =============================================================================
    // Q8_1 MPI Allreduce Tests
    // =============================================================================

    TEST_F(Test__MPI_PrecisionAwareAllreduce, Q8_1_Allreduce_TwoRanks)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        const size_t count = 128; // 4 Q8_1 blocks
        const size_t n_blocks = (count + 31) / 32;

        // Each rank generates unique FP32 data, quantizes to Q8_1
        auto local_fp32 = generate_random_fp32(count);
        auto local_blocks = fp32_to_q8_1_blocks(local_fp32);

        // Gather all Q8_1 blocks for computing reference
        std::vector<Q8_1Block> all_blocks(n_blocks * world_size_);
        MPI_Allgather(local_blocks.data(), n_blocks * sizeof(Q8_1Block), MPI_BYTE,
                      all_blocks.data(), n_blocks * sizeof(Q8_1Block), MPI_BYTE, MPI_COMM_WORLD);

        // Compute reference sum (dequant all, sum, requant)
        std::vector<float> reference(count, 0.0f);
        for (int r = 0; r < world_size_; ++r)
        {
            auto rank_fp32 = q8_1_blocks_to_fp32(
                std::vector<Q8_1Block>(all_blocks.begin() + r * n_blocks,
                                       all_blocks.begin() + (r + 1) * n_blocks),
                count);
            for (size_t i = 0; i < count; ++i)
            {
                reference[i] += rank_fp32[i];
            }
        }

        // Perform Q8_1 allreduce
        std::vector<Q8_1Block> result = local_blocks;
        mpi_ctx_->allreduce_q8_1_inplace(result.data(), n_blocks);

        // Dequantize result for comparison
        auto result_fp32 = q8_1_blocks_to_fp32(result, count);

        // Verify (allow for Q8_1 quantization noise)
        float sim = cosine_similarity(result_fp32, reference);
        EXPECT_GT(sim, 0.99f) << "Q8_1 allreduce should have high similarity to reference";
    }

    TEST_F(Test__MPI_PrecisionAwareAllreduce, Q8_1_Allreduce_LargerData)
    {
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 MPI ranks";
        }

        const size_t count = 1024; // 32 Q8_1 blocks
        const size_t n_blocks = (count + 31) / 32;

        auto local_fp32 = generate_random_fp32(count);
        auto local_blocks = fp32_to_q8_1_blocks(local_fp32);

        // Gather and compute reference
        std::vector<Q8_1Block> all_blocks(n_blocks * world_size_);
        MPI_Allgather(local_blocks.data(), n_blocks * sizeof(Q8_1Block), MPI_BYTE,
                      all_blocks.data(), n_blocks * sizeof(Q8_1Block), MPI_BYTE, MPI_COMM_WORLD);

        std::vector<float> reference(count, 0.0f);
        for (int r = 0; r < world_size_; ++r)
        {
            auto rank_fp32 = q8_1_blocks_to_fp32(
                std::vector<Q8_1Block>(all_blocks.begin() + r * n_blocks,
                                       all_blocks.begin() + (r + 1) * n_blocks),
                count);
            for (size_t i = 0; i < count; ++i)
            {
                reference[i] += rank_fp32[i];
            }
        }

        // Perform Q8_1 allreduce
        std::vector<Q8_1Block> result = local_blocks;
        mpi_ctx_->allreduce_q8_1_inplace(result.data(), n_blocks);

        auto result_fp32 = q8_1_blocks_to_fp32(result, count);

        float sim = cosine_similarity(result_fp32, reference);
        EXPECT_GT(sim, 0.99f) << "Q8_1 allreduce on larger data should have high similarity";
    }

    // =============================================================================
    // Bandwidth Comparison Test (informational)
    // =============================================================================

    TEST_F(Test__MPI_PrecisionAwareAllreduce, Bandwidth_Comparison)
    {
        // Informational test: compare data sizes for same element count
        const size_t count = 1024;

        size_t fp32_bytes = count * sizeof(float);                   // 4096 bytes
        size_t fp16_bytes = count * sizeof(uint16_t);                // 2048 bytes
        size_t bf16_bytes = count * sizeof(uint16_t);                // 2048 bytes
        size_t q8_1_bytes = ((count + 31) / 32) * sizeof(Q8_1Block); // 1152 bytes (32 blocks * 36 bytes)

        if (rank_ == 0)
        {
            LOG_INFO("[Bandwidth Comparison] For " << count << " elements:");
            LOG_INFO("  FP32: " << fp32_bytes << " bytes (1.0x baseline)");
            LOG_INFO("  FP16: " << fp16_bytes << " bytes (" << (float)fp32_bytes / fp16_bytes << "x better)");
            LOG_INFO("  BF16: " << bf16_bytes << " bytes (" << (float)fp32_bytes / bf16_bytes << "x better)");
            LOG_INFO("  Q8_1: " << q8_1_bytes << " bytes (" << (float)fp32_bytes / q8_1_bytes << "x better)");
        }

        // Just verify the math
        EXPECT_EQ(fp32_bytes, 4096u);
        EXPECT_EQ(fp16_bytes, 2048u);
        EXPECT_EQ(bf16_bytes, 2048u);
        EXPECT_EQ(q8_1_bytes, 1152u); // 32 blocks * 36 bytes
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}