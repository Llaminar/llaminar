/**
 * @file Test__ShardedKVCache.cpp
 * @brief Unit tests for sharded KV cache (tensor parallelism support)
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests the sharded KV cache functionality introduced in Phase 6 of the
 * distributed architecture. Verifies that:
 * - Sharded caches store only local KV heads
 * - Memory usage is reduced proportionally to the number of shards
 * - Append/gather operations work correctly with local heads
 * - Non-sharded caches behave as before (backward compatibility)
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "tensors/CPUKVCache.h"
#include "utils/MPIContext.h"

namespace llaminar2
{
    namespace
    {

        /**
         * @brief Test fixture for sharded KV cache tests
         */
        class Test__ShardedKVCache : public ::testing::Test
        {
        protected:
            // Test parameters matching Qwen2.5-0.5B
            static constexpr int kNumLayers = 24;
            static constexpr int kBatchSize = 1;
            static constexpr int kMaxSeqLen = 256;
            static constexpr int kNKVHeads = 2; // Total KV heads (GQA)
            static constexpr int kHeadDim = 64;
            static constexpr int kKVDim = kNKVHeads * kHeadDim; // 128

            // Simulated 2-rank tensor parallelism
            static constexpr int kWorldSize = 2;
            static constexpr int kLocalKVHeads = kNKVHeads / kWorldSize; // 1 head per rank
            static constexpr int kLocalKVDim = kLocalKVHeads * kHeadDim; // 64

            void SetUp() override
            {
                // Create a mock MPI context (rank 0)
                mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_SELF);
            }

            std::shared_ptr<MPIContext> mpi_ctx_;
        };

        // =========================================================================
        // Test: Non-sharded cache (backward compatibility)
        // =========================================================================

        TEST_F(Test__ShardedKVCache, NonShardedCache_HasFullKVDim)
        {
            // Create non-sharded cache (standard API)
            auto cache = createCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Verify non-sharded metadata
            EXPECT_FALSE(cache->is_sharded());
            EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
            EXPECT_EQ(cache->local_n_kv_heads(), kNKVHeads); // Same as total
            EXPECT_EQ(cache->kv_head_start(), 0);
            EXPECT_EQ(cache->local_kv_dim(), kKVDim); // Full KV dim
        }

        TEST_F(Test__ShardedKVCache, NonShardedCache_TensorShapeIsFullKVDim)
        {
            auto cache = createCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // K/V tensors should have shape [max_seq_len, kv_dim]
            const ITensor *k = cache->get_k(0, 0);
            const ITensor *v = cache->get_v(0, 0);

            ASSERT_NE(k, nullptr);
            ASSERT_NE(v, nullptr);

            EXPECT_EQ(k->shape().size(), 2);
            EXPECT_EQ(k->shape()[0], static_cast<size_t>(kMaxSeqLen));
            EXPECT_EQ(k->shape()[1], static_cast<size_t>(kKVDim));

            EXPECT_EQ(v->shape().size(), 2);
            EXPECT_EQ(v->shape()[0], static_cast<size_t>(kMaxSeqLen));
            EXPECT_EQ(v->shape()[1], static_cast<size_t>(kKVDim));
        }

        // =========================================================================
        // Test: Sharded cache (tensor parallelism)
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_Rank0_HasLocalKVDim)
        {
            // Create sharded cache for rank 0
            int rank = 0;
            int kv_head_start = rank * kLocalKVHeads;

            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, kv_head_start,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Verify sharded metadata
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);           // Total heads
            EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads); // Local heads
            EXPECT_EQ(cache->kv_head_start(), 0);                // Rank 0 starts at head 0
            EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);       // Reduced KV dim
        }

        TEST_F(Test__ShardedKVCache, ShardedCache_Rank1_HasCorrectOffset)
        {
            // Create sharded cache for rank 1
            int rank = 1;
            int kv_head_start = rank * kLocalKVHeads;

            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, kv_head_start,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Verify rank 1 metadata
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
            EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);
            EXPECT_EQ(cache->kv_head_start(), kLocalKVHeads); // Rank 1 starts at head 1
            EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
        }

        TEST_F(Test__ShardedKVCache, ShardedCache_TensorShapeIsLocalKVDim)
        {
            int rank = 0;
            int kv_head_start = rank * kLocalKVHeads;

            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32,
                *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, kv_head_start,
                kHeadDim,
                DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // K/V tensors should have reduced shape [max_seq_len, local_kv_dim]
            const ITensor *k = cache->get_k(0, 0);
            const ITensor *v = cache->get_v(0, 0);

            ASSERT_NE(k, nullptr);
            ASSERT_NE(v, nullptr);

            EXPECT_EQ(k->shape().size(), 2);
            EXPECT_EQ(k->shape()[0], static_cast<size_t>(kMaxSeqLen));
            EXPECT_EQ(k->shape()[1], static_cast<size_t>(kLocalKVDim)); // Reduced!

            EXPECT_EQ(v->shape().size(), 2);
            EXPECT_EQ(v->shape()[0], static_cast<size_t>(kMaxSeqLen));
            EXPECT_EQ(v->shape()[1], static_cast<size_t>(kLocalKVDim)); // Reduced!
        }

        // =========================================================================
        // Test: Memory savings from sharding
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_MemorySavings)
        {
            // Calculate expected memory usage
            // Non-sharded: 2 (K+V) * layers * max_seq_len * kv_dim * sizeof(float)
            size_t full_memory = 2 * kNumLayers * kMaxSeqLen * kKVDim * sizeof(float);

            // Sharded: same formula but with local_kv_dim
            size_t sharded_memory = 2 * kNumLayers * kMaxSeqLen * kLocalKVDim * sizeof(float);

            // With 2-way sharding, memory should be halved
            EXPECT_EQ(sharded_memory, full_memory / kWorldSize);

            // Verify by creating caches and checking tensor sizes
            auto full_cache = createCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kHeadDim, DeviceId::cpu());

            auto sharded_cache = createShardedCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            // Get actual tensor sizes
            const ITensor *full_k = full_cache->get_k(0, 0);
            const ITensor *sharded_k = sharded_cache->get_k(0, 0);

            EXPECT_EQ(sharded_k->numel(), full_k->numel() / kWorldSize);
        }

        // =========================================================================
        // Test: Append/Get operations work with sharded cache
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_AppendKV_Works)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            ASSERT_NE(cache, nullptr);

            // Create input K/V tensors with local KV dim
            std::vector<size_t> shape = {1, static_cast<size_t>(kLocalKVDim)};
            auto new_k = std::make_unique<FP32Tensor>(shape);
            auto new_v = std::make_unique<FP32Tensor>(shape);

            // Fill with test data
            float *k_data = new_k->mutable_data();
            float *v_data = new_v->mutable_data();
            for (int i = 0; i < kLocalKVDim; ++i)
            {
                k_data[i] = static_cast<float>(i) * 0.1f;
                v_data[i] = static_cast<float>(i) * 0.2f;
            }

            // Append to layer 0
            bool success = cache->append_kv(0, 0, new_k.get(), new_v.get(), 1);
            EXPECT_TRUE(success);

            // Verify token count increased
            EXPECT_EQ(cache->get_cached_tokens(0, 0), 1);
        }

        TEST_F(Test__ShardedKVCache, ShardedCache_MultipleAppends_Work)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            // Create batch input (4 tokens)
            int num_tokens = 4;
            std::vector<size_t> shape = {static_cast<size_t>(num_tokens), static_cast<size_t>(kLocalKVDim)};
            auto new_k = std::make_unique<FP32Tensor>(shape);
            auto new_v = std::make_unique<FP32Tensor>(shape);

            // Fill with test data
            float *k_data = new_k->mutable_data();
            float *v_data = new_v->mutable_data();
            for (size_t i = 0; i < new_k->numel(); ++i)
            {
                k_data[i] = static_cast<float>(i) * 0.01f;
                v_data[i] = static_cast<float>(i) * 0.02f;
            }

            // Append to all layers
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                bool success = cache->append_kv(layer, 0, new_k.get(), new_v.get(), num_tokens);
                EXPECT_TRUE(success) << "Append failed at layer " << layer;
            }

            // Verify all layers have correct token count
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                EXPECT_EQ(cache->get_cached_tokens(layer, 0), num_tokens)
                    << "Token count wrong at layer " << layer;
            }
        }

        // =========================================================================
        // Test: Clear operations work with sharded cache
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_Clear_Works)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            // Create and append some data
            std::vector<size_t> shape = {2, static_cast<size_t>(kLocalKVDim)};
            auto new_k = std::make_unique<FP32Tensor>(shape);
            auto new_v = std::make_unique<FP32Tensor>(shape);

            cache->append_kv(0, 0, new_k.get(), new_v.get(), 2);
            EXPECT_EQ(cache->get_cached_tokens(0, 0), 2);

            // Clear
            cache->clear();

            // Verify cleared
            EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
        }

        // =========================================================================
        // Test: Multiple precisions work with sharded cache
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_BF16_Works)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::BF16, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            ASSERT_NE(cache, nullptr);
            EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
        }

        TEST_F(Test__ShardedKVCache, ShardedCache_FP16_Works)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP16, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            ASSERT_NE(cache, nullptr);
            EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
        }

        TEST_F(Test__ShardedKVCache, ShardedCache_Q8_1_Works)
        {
            auto cache = createShardedCPUKVCache(
                ActivationPrecision::Q8_1, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim, DeviceId::cpu());

            ASSERT_NE(cache, nullptr);
            EXPECT_EQ(cache->precision(), ActivationPrecision::Q8_1);
            EXPECT_TRUE(cache->is_sharded());
            EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
        }

        // =========================================================================
        // Test: Per-layer device placement works with sharded cache
        // =========================================================================

        TEST_F(Test__ShardedKVCache, ShardedCache_PerLayerDevices_Works)
        {
            // Create device list (alternating 0 and 1 for simulation)
            std::vector<int> devices(kNumLayers);
            for (int i = 0; i < kNumLayers; ++i)
            {
                devices[i] = i % 2; // Alternating 0, 1, 0, 1, ...
            }

            auto cache = createShardedCPUKVCache(
                ActivationPrecision::FP32, *mpi_ctx_,
                kNumLayers, kBatchSize, kMaxSeqLen,
                kNKVHeads, kLocalKVHeads, 0, kHeadDim,
                devices);

            ASSERT_NE(cache, nullptr);
            EXPECT_TRUE(cache->is_sharded());

            // Verify per-layer device placement
            for (int layer = 0; layer < kNumLayers; ++layer)
            {
                EXPECT_EQ(cache->get_layer_device(layer), DeviceId::fromLegacyIndex(layer % 2))
                    << "Wrong device at layer " << layer;
            }
        }

    } // namespace
} // namespace llaminar2

/**
 * @brief Main entry point for GTest
 */
int main(int argc, char **argv)
{
    // Initialize MPI for any tests that need it
    MPI_Init(&argc, &argv);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
