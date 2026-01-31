/**
 * @file Test__MPI_ShardedKVCache.cpp
 * @brief MPI integration test for sharded KV cache (tensor parallelism)
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the sharded KV cache in a real 2-rank MPI environment.
 * Validates that:
 * - Each rank creates its own local KV cache with correct sharding
 * - No MPI communication is needed for KV cache operations (each rank independent)
 * - Memory is reduced proportionally across ranks
 * - Append/gather operations work correctly on each rank's local heads
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <iostream>

#include "backends/DeviceId.h"
#include "kernels/cpu/CPUKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

namespace llaminar2
{
    namespace
    {

        /**
         * @brief MPI test fixture for sharded KV cache
         */
        class Test__MPI_ShardedKVCache : public ::testing::Test
        {
        protected:
            // Test parameters (Qwen2.5-0.5B style)
            static constexpr int kNumLayers = 24;
            static constexpr int kBatchSize = 1;
            static constexpr int kMaxSeqLen = 256;
            static constexpr int kNKVHeads = 2; // Total KV heads (GQA)
            static constexpr int kHeadDim = 64;
            static constexpr int kKVDim = kNKVHeads * kHeadDim; // 128

            void SetUp() override
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

                // This test requires exactly 2 MPI ranks
                if (world_size_ != 2)
                {
                    GTEST_SKIP() << "Test requires exactly 2 MPI ranks, got " << world_size_;
                }

                // Create MPI context for this rank
                mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

                // Calculate local sharding parameters
                local_n_kv_heads_ = kNKVHeads / world_size_;  // 1 head per rank
                kv_head_start_ = rank_ * local_n_kv_heads_;   // Rank 0: head 0, Rank 1: head 1
                local_kv_dim_ = local_n_kv_heads_ * kHeadDim; // 64
            }

            int rank_;
            int world_size_;
            std::shared_ptr<MPIContext> mpi_ctx_;

            // Sharding parameters computed in SetUp
            int local_n_kv_heads_;
            int kv_head_start_;
            int local_kv_dim_;
        };

        // =========================================================================
        // Test: Each rank creates its own sharded cache
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, EachRankCreatesLocalShardedCache)
        {
            // Each rank creates its own sharded cache
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Verify sharding metadata is correct for this rank
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
            EXPECT_EQ(cache->local_n_kv_heads(), local_n_kv_heads_);
            EXPECT_EQ(cache->kv_head_start(), kv_head_start_);
            EXPECT_EQ(cache->local_kv_dim(), local_kv_dim_);

            // Verify rank-specific offset
            if (rank_ == 0)
            {
                EXPECT_EQ(cache->kv_head_start(), 0);
            }
            else if (rank_ == 1)
            {
                EXPECT_EQ(cache->kv_head_start(), 1);
            }

            // Barrier to ensure all ranks complete before proceeding
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: Tensor shapes are local on each rank
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, TensorShapesAreLocalOnEachRank)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Check K/V tensor shapes on each rank
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                const ITensor *k = cache->get_k(layer, 0);
                const ITensor *v = cache->get_v(layer, 0);

                ASSERT_NE(k, nullptr) << "K tensor null at layer " << layer << " on rank " << rank_;
                ASSERT_NE(v, nullptr) << "V tensor null at layer " << layer << " on rank " << rank_;

                // Shape should be [max_seq_len, local_kv_dim]
                EXPECT_EQ(k->shape().size(), 2);
                EXPECT_EQ(k->shape()[0], static_cast<size_t>(kMaxSeqLen));
                EXPECT_EQ(k->shape()[1], static_cast<size_t>(local_kv_dim_))
                    << "Wrong K shape at layer " << layer << " on rank " << rank_;

                EXPECT_EQ(v->shape().size(), 2);
                EXPECT_EQ(v->shape()[0], static_cast<size_t>(kMaxSeqLen));
                EXPECT_EQ(v->shape()[1], static_cast<size_t>(local_kv_dim_))
                    << "Wrong V shape at layer " << layer << " on rank " << rank_;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: Append works independently on each rank
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, AppendWorksIndependentlyOnEachRank)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Create local K/V tensors (different per rank for verification)
            int num_tokens = 4;
            std::vector<size_t> shape = {static_cast<size_t>(num_tokens), static_cast<size_t>(local_kv_dim_)};
            auto new_k = std::make_unique<FP32Tensor>(shape);
            auto new_v = std::make_unique<FP32Tensor>(shape);

            // Fill with rank-specific data
            float *k_data = new_k->mutable_data();
            float *v_data = new_v->mutable_data();
            for (size_t i = 0; i < new_k->numel(); ++i)
            {
                k_data[i] = static_cast<float>(rank_ * 1000 + i) * 0.01f;
                v_data[i] = static_cast<float>(rank_ * 1000 + i) * 0.02f;
            }

            // Append to all layers on this rank (no communication with other rank)
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                bool success = cache->append_kv(layer, 0, new_k.get(), new_v.get(), num_tokens);
                EXPECT_TRUE(success) << "Append failed at layer " << layer << " on rank " << rank_;
            }

            // Verify token counts on this rank
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                EXPECT_EQ(cache->get_cached_tokens(layer, 0), num_tokens)
                    << "Token count wrong at layer " << layer << " on rank " << rank_;
            }

            // Barrier - no data exchange needed, just synchronization for test ordering
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: Clear works independently on each rank
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, ClearWorksIndependentlyOnEachRank)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            // Add some data
            std::vector<size_t> shape = {2, static_cast<size_t>(local_kv_dim_)};
            auto new_k = std::make_unique<FP32Tensor>(shape);
            auto new_v = std::make_unique<FP32Tensor>(shape);

            cache->append_kv(0, 0, new_k.get(), new_v.get(), 2);
            EXPECT_EQ(cache->get_cached_tokens(0, 0), 2);

            // Clear on this rank only (no communication)
            cache->clear();

            // Verify cleared
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                EXPECT_EQ(cache->get_cached_tokens(layer, 0), 0)
                    << "Cache not cleared at layer " << layer << " on rank " << rank_;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: Memory usage is reduced on each rank
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, MemoryUsageReducedOnEachRank)
        {
            // Calculate expected per-rank memory
            size_t full_memory_per_rank = 2 * kNumLayers * kMaxSeqLen * kKVDim * sizeof(float);
            size_t sharded_memory_per_rank = 2 * kNumLayers * kMaxSeqLen * local_kv_dim_ * sizeof(float);

            // With 2-way sharding, each rank should use half the full memory
            EXPECT_EQ(sharded_memory_per_rank, full_memory_per_rank / world_size_);

            // Create sharded cache and verify tensor sizes
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            const ITensor *k = cache->get_k(0, 0);
            EXPECT_EQ(k->numel(), static_cast<size_t>(kMaxSeqLen * local_kv_dim_));

            // Gather total memory usage across ranks (for logging)
            size_t local_tensor_bytes = k->numel() * sizeof(float);
            size_t total_tensor_bytes = 0;
            MPI_Allreduce(&local_tensor_bytes, &total_tensor_bytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

            // Total should equal full K tensor size (both ranks combined)
            size_t expected_total = static_cast<size_t>(kMaxSeqLen) * kKVDim * sizeof(float);
            EXPECT_EQ(total_tensor_bytes, expected_total);

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: All precisions work in MPI environment
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, AllPrecisionsWorkInMPIEnvironment)
        {
            std::vector<ActivationPrecision> precisions = {
                ActivationPrecision::FP32,
                ActivationPrecision::BF16,
                ActivationPrecision::FP16,
                ActivationPrecision::Q8_1};

            for (auto precision : precisions)
            {
                auto cache = createShardedCPUKVCache(
                    precision,
                    *mpi_ctx_,
                    kNumLayers, kBatchSize, kMaxSeqLen,
                    kNKVHeads, local_n_kv_heads_, kv_head_start_,
                    kHeadDim,
                    DeviceId::cpu());

                ASSERT_NE(cache, nullptr) << "Failed to create cache with precision "
                                          << static_cast<int>(precision) << " on rank " << rank_;
                EXPECT_EQ(cache->precision(), precision);
                EXPECT_TRUE(cache->is_sharded());
                EXPECT_EQ(cache->local_kv_dim(), local_kv_dim_);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // =========================================================================
        // Test: Sequential decode simulation (each rank handles its local heads)
        // =========================================================================

        TEST_F(Test__MPI_ShardedKVCache, SequentialDecodeSimulation)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, local_n_kv_heads_, kv_head_start_,
                kHeadDim,
                DeviceId::cpu());

            // Simulate prefill: add 10 tokens to all layers
            int prefill_tokens = 10;
            {
                std::vector<size_t> shape = {static_cast<size_t>(prefill_tokens), static_cast<size_t>(local_kv_dim_)};
                auto k = std::make_unique<FP32Tensor>(shape);
                auto v = std::make_unique<FP32Tensor>(shape);

                // Fill with prefill data (rank-specific)
                float *k_data = k->mutable_data();
                float *v_data = v->mutable_data();
                for (size_t i = 0; i < k->numel(); ++i)
                {
                    k_data[i] = static_cast<float>(rank_) + static_cast<float>(i) * 0.001f;
                    v_data[i] = static_cast<float>(rank_) + static_cast<float>(i) * 0.002f;
                }

                for (int layer = 0; layer < kNumLayers; ++layer)
                {
                    bool success = cache->append_kv(layer, 0, k.get(), v.get(), prefill_tokens);
                    EXPECT_TRUE(success);
                }
            }

            // Simulate decode: add 1 token at a time
            int decode_steps = 20;
            for (int step = 0; step < decode_steps; ++step)
            {
                std::vector<size_t> shape = {1, static_cast<size_t>(local_kv_dim_)};
                auto k = std::make_unique<FP32Tensor>(shape);
                auto v = std::make_unique<FP32Tensor>(shape);

                float *k_data = k->mutable_data();
                float *v_data = v->mutable_data();
                for (int i = 0; i < local_kv_dim_; ++i)
                {
                    k_data[i] = static_cast<float>(rank_ * 100 + step) + static_cast<float>(i) * 0.01f;
                    v_data[i] = static_cast<float>(rank_ * 100 + step) + static_cast<float>(i) * 0.02f;
                }

                for (int layer = 0; layer < kNumLayers; ++layer)
                {
                    bool success = cache->append_kv(layer, 0, k.get(), v.get(), 1);
                    EXPECT_TRUE(success) << "Decode append failed at step " << step << " layer " << layer;
                }

                // Verify token count increases
                EXPECT_EQ(cache->get_cached_tokens(0, 0), prefill_tokens + step + 1);
            }

            // Final token count should be prefill + decode
            int expected_tokens = prefill_tokens + decode_steps;
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                EXPECT_EQ(cache->get_cached_tokens(layer, 0), expected_tokens)
                    << "Final token count wrong at layer " << layer << " on rank " << rank_;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

    } // namespace
} // namespace llaminar2

/**
 * @brief Main entry point for MPI test
 */
int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
