/**
 * @file Test__UnifiedKVCache.cpp
 * @brief Unit tests for unified KV cache
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests both single-sequence and batched modes across all precision types.
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/UnifiedKVCache.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include <vector>
#include <memory>
#include <cmath>

using namespace llaminar2;

// Single-rank MPI context for unit tests
static MPIContext getTestMPIContext()
{
    return MPIContext(0, 1, MPI_COMM_WORLD);
}

class Test__UnifiedKVCache : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test: Basic Construction - FP32
// =============================================================================

TEST_F(Test__UnifiedKVCache, ConstructionFP32_SingleSequence)
{
    int n_layers = 24;
    int batch_size = 1; // Single sequence
    int max_seq_len = 2048;
    int n_kv_heads = 2;
    int head_dim = 64;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    EXPECT_EQ(cache.num_layers(), n_layers);
    EXPECT_EQ(cache.batch_size(), batch_size);
    EXPECT_EQ(cache.max_seq_len(), max_seq_len);
    EXPECT_EQ(cache.precision(), ActivationPrecision::FP32);

    // Initially, all cached token counts should be 0
    for (int layer = 0; layer < n_layers; ++layer)
    {
        EXPECT_EQ(cache.get_cached_tokens(layer), 0);
    }
}

TEST_F(Test__UnifiedKVCache, ConstructionFP32_Batched)
{
    int n_layers = 4;
    int batch_size = 4;
    int max_seq_len = 128;
    int n_kv_heads = 2;
    int head_dim = 32;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    EXPECT_EQ(cache.batch_size(), batch_size);

    // All sequences should have 0 cached tokens initially
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache.get_cached_tokens(layer, seq), 0);
        }
    }
}

// =============================================================================
// Test: Multi-Precision Construction
// =============================================================================

TEST_F(Test__UnifiedKVCache, ConstructionBF16)
{
    UnifiedKVCacheBF16 cache(getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    EXPECT_EQ(cache.precision(), ActivationPrecision::BF16);
}

TEST_F(Test__UnifiedKVCache, ConstructionFP16)
{
    UnifiedKVCacheFP16 cache(getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    EXPECT_EQ(cache.precision(), ActivationPrecision::FP16);
}

TEST_F(Test__UnifiedKVCache, ConstructionQ8_1)
{
    UnifiedKVCacheQ8_1 cache(getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    EXPECT_EQ(cache.precision(), ActivationPrecision::Q8_1);
}

TEST_F(Test__UnifiedKVCache, ConstructionQ16_1)
{
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    EXPECT_EQ(cache.precision(), ActivationPrecision::Q16_1);
}

// =============================================================================
// Test: Factory Function
// =============================================================================

TEST_F(Test__UnifiedKVCache, FactoryFP32)
{
    auto cache = createUnifiedKVCache(ActivationPrecision::FP32, getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP32);
}

TEST_F(Test__UnifiedKVCache, FactoryQ8_1)
{
    auto cache = createUnifiedKVCache(ActivationPrecision::Q8_1, getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q8_1);
}

TEST_F(Test__UnifiedKVCache, FactoryQ16_1)
{
    auto cache = createUnifiedKVCache(ActivationPrecision::Q16_1, getTestMPIContext(), 4, 2, 64, 2, 32, -1);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
}

TEST_F(Test__UnifiedKVCache, ShardedFactoryQ16_1)
{
    // Test sharded factory for tensor parallelism
    int n_layers = 4;
    int batch_size = 1;
    int max_seq_len = 64;
    int n_kv_heads = 4;       // Total across all ranks
    int local_n_kv_heads = 2; // This rank's portion
    int kv_head_start = 0;    // Starting head index
    int head_dim = 32;

    auto cache = createShardedKVCache(
        ActivationPrecision::Q16_1, getTestMPIContext(),
        n_layers, batch_size, max_seq_len,
        n_kv_heads, local_n_kv_heads, kv_head_start,
        head_dim, -1);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache->local_n_kv_heads(), local_n_kv_heads);
    EXPECT_EQ(cache->kv_head_start(), kv_head_start);
}

// =============================================================================
// Test: Single-Sequence Append (batch_size=1)
// =============================================================================

TEST_F(Test__UnifiedKVCache, AppendSingleSequence_FP32)
{
    int n_layers = 2;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Append 5 tokens to layer 0
    auto k1 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);

    // Fill with test pattern
    for (size_t i = 0; i < 5 * kv_dim; ++i)
    {
        k1->mutable_data()[i] = static_cast<float>(i);
        v1->mutable_data()[i] = static_cast<float>(i + 100);
    }

    ASSERT_TRUE(cache.append_kv(0, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 5);

    // Append 3 more tokens
    auto k2 = factory.createFP32({3, static_cast<size_t>(kv_dim)}, -1);
    auto v2 = factory.createFP32({3, static_cast<size_t>(kv_dim)}, -1);

    for (size_t i = 0; i < 3 * kv_dim; ++i)
    {
        k2->mutable_data()[i] = static_cast<float>(i + 1000);
        v2->mutable_data()[i] = static_cast<float>(i + 2000);
    }

    ASSERT_TRUE(cache.append_kv(0, k2.get(), v2.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 8);

    // Verify data integrity
    auto cached_k = cache.get_k_typed(0);
    ASSERT_NE(cached_k, nullptr);

    // First 5 tokens
    EXPECT_FLOAT_EQ(cached_k->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(cached_k->data()[4 * kv_dim], static_cast<float>(4 * kv_dim));

    // Next 3 tokens (starting at position 5)
    EXPECT_FLOAT_EQ(cached_k->data()[5 * kv_dim], 1000.0f);
}

// =============================================================================
// Test: Batched Append
// =============================================================================

TEST_F(Test__UnifiedKVCache, AppendBatched_FP32)
{
    int n_layers = 2;
    int batch_size = 3;
    int max_seq_len = 10;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Append different lengths to different sequences in layer 0
    // Sequence 0: 3 tokens
    auto k0 = factory.createFP32({3, static_cast<size_t>(kv_dim)}, -1);
    auto v0 = factory.createFP32({3, static_cast<size_t>(kv_dim)}, -1);
    std::fill_n(k0->mutable_data(), 3 * kv_dim, 1.0f);
    std::fill_n(v0->mutable_data(), 3 * kv_dim, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get()));

    // Sequence 1: 5 tokens
    auto k1 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);
    std::fill_n(k1->mutable_data(), 5 * kv_dim, 2.0f);
    std::fill_n(v1->mutable_data(), 5 * kv_dim, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 1, k1.get(), v1.get()));

    // Sequence 2: 7 tokens
    auto k2 = factory.createFP32({7, static_cast<size_t>(kv_dim)}, -1);
    auto v2 = factory.createFP32({7, static_cast<size_t>(kv_dim)}, -1);
    std::fill_n(k2->mutable_data(), 7 * kv_dim, 3.0f);
    std::fill_n(v2->mutable_data(), 7 * kv_dim, 3.0f);
    ASSERT_TRUE(cache.append_kv(0, 2, k2.get(), v2.get()));

    // Verify per-sequence counts
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 3);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 5);
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 7);

    // Verify data isolation
    auto cached_k0 = cache.get_k_typed(0, 0);
    auto cached_k1 = cache.get_k_typed(0, 1);
    auto cached_k2 = cache.get_k_typed(0, 2);

    EXPECT_FLOAT_EQ(cached_k0->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(cached_k1->data()[0], 2.0f);
    EXPECT_FLOAT_EQ(cached_k2->data()[0], 3.0f);
}

// =============================================================================
// Test: Q8_1 Precision
// =============================================================================

TEST_F(Test__UnifiedKVCache, AppendQ8_1)
{
    int n_layers = 2;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 32; // Must be multiple of 32 for Q8_1 blocks
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheQ8_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Create Q8_1 tensors
    auto k = factory.createQ8_1({4, static_cast<size_t>(kv_dim)});
    auto v = factory.createQ8_1({4, static_cast<size_t>(kv_dim)});

    // Initialize Q8_1 blocks with test data
    size_t blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            auto &k_block = k->mutable_q8_1_blocks()[row * blocks_per_row + b];
            auto &v_block = v->mutable_q8_1_blocks()[row * blocks_per_row + b];
            // Q8_1Block has: uint16_t d (FP16 scale), int16_t sum_qs, int8_t qs[32]
            k_block.d = 0x3C00; // 1.0 in FP16
            v_block.d = 0x3C00;
            k_block.sum_qs = 0;
            v_block.sum_qs = 0;
            for (int i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
            {
                k_block.qs[i] = static_cast<int8_t>(i);
                v_block.qs[i] = static_cast<int8_t>(-i);
            }
        }
    }

    ASSERT_TRUE(cache.append_kv_typed(0, 0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 4);

    // Verify cached data
    auto cached_k = cache.get_k_typed(0);
    ASSERT_NE(cached_k, nullptr);
    // d is stored as FP16 (uint16_t) - 0x3C00 is 1.0 in FP16
    EXPECT_EQ(cached_k->q8_1_blocks()[0].d, 0x3C00);
}

// =============================================================================
// Test: Q16_1 Precision
// =============================================================================

TEST_F(Test__UnifiedKVCache, AppendQ16_1_SingleSequence)
{
    int n_layers = 2;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 64;                  // Use 64 so optimal_q16_block_size returns BLOCK_64
    int kv_dim = n_kv_heads * head_dim; // 128

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Create Q16_1 tensors for 4 tokens with BLOCK_64 to match cache
    auto k = factory.createQ16_1({4, static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64);
    auto v = factory.createQ16_1({4, static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64);

    // Initialize Q16_1Block_64 blocks with test data
    const size_t block_elements = 64;
    size_t blocks_per_row = (kv_dim + block_elements - 1) / block_elements;
    auto *k_blocks_init = k->mutable_as_block_64();
    auto *v_blocks_init = v->mutable_as_block_64();
    ASSERT_NE(k_blocks_init, nullptr);
    ASSERT_NE(v_blocks_init, nullptr);

    for (size_t row = 0; row < 4; ++row)
    {
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            auto &k_block = k_blocks_init[row * blocks_per_row + b];
            auto &v_block = v_blocks_init[row * blocks_per_row + b];
            // Q16_1Block_64 has: float d (scale), int32_t sum_qs, int16_t qs[64]
            k_block.d = 1.0f;
            v_block.d = 2.0f;
            k_block.sum_qs = 0;
            v_block.sum_qs = 0;
            for (int i = 0; i < 64; ++i)
            {
                k_block.qs[i] = static_cast<int16_t>(row * 100 + i);
                v_block.qs[i] = static_cast<int16_t>(-(row * 100 + i));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv_typed(0, 0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 4);

    // Verify cached data
    auto cached_k = cache.get_k_typed(0);
    ASSERT_NE(cached_k, nullptr);
    const auto *k_blocks = cached_k->as_block_64();
    ASSERT_NE(k_blocks, nullptr) << "Cache should use BLOCK_64 for head_dim=64";
    EXPECT_FLOAT_EQ(k_blocks[0].d, 1.0f);
    EXPECT_EQ(k_blocks[0].qs[0], 0); // Row 0, element 0
    EXPECT_EQ(k_blocks[0].qs[1], 1); // Row 0, element 1

    auto cached_v = cache.get_v_typed(0);
    ASSERT_NE(cached_v, nullptr);
    const auto *v_blocks = cached_v->as_block_64();
    ASSERT_NE(v_blocks, nullptr) << "Cache should use BLOCK_64 for head_dim=64";
    EXPECT_FLOAT_EQ(v_blocks[0].d, 2.0f);
    EXPECT_EQ(v_blocks[0].qs[0], 0);  // Row 0, element 0
    EXPECT_EQ(v_blocks[0].qs[1], -1); // Row 0, element 1
}

TEST_F(Test__UnifiedKVCache, AppendQ16_1_MultipleAppends)
{
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 32;
    int n_kv_heads = 2;
    int head_dim = 64;                  // Use head_dim=64 so cache auto-selects BLOCK_64
    int kv_dim = n_kv_heads * head_dim; // 128

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Cache uses BLOCK_64 for head_dim=64, so blocks_per_row = kv_dim / 64 = 2
    const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
    size_t block_elements = q16_block_size_elements(block_size);
    size_t blocks_per_row = (kv_dim + block_elements - 1) / block_elements;

    // First append: 5 tokens - create with matching block size
    auto k1 = factory.createQ16_1({5, static_cast<size_t>(kv_dim)}, block_size);
    auto v1 = factory.createQ16_1({5, static_cast<size_t>(kv_dim)}, block_size);

    // Access as Q16_1Block_64 since we're using BLOCK_64
    auto *k1_blocks = reinterpret_cast<Q16_1Block_64 *>(k1->raw_mutable_data());
    auto *v1_blocks = reinterpret_cast<Q16_1Block_64 *>(v1->raw_mutable_data());
    for (size_t row = 0; row < 5; ++row)
    {
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            k1_blocks[row * blocks_per_row + b].d = 1.0f;
            k1_blocks[row * blocks_per_row + b].qs[0] = static_cast<int16_t>(row);
            v1_blocks[row * blocks_per_row + b].d = 1.0f;
        }
    }
    ASSERT_TRUE(cache.append_kv_typed(0, 0, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 5);

    // Second append: 3 tokens
    auto k2 = factory.createQ16_1({3, static_cast<size_t>(kv_dim)}, block_size);
    auto v2 = factory.createQ16_1({3, static_cast<size_t>(kv_dim)}, block_size);

    auto *k2_blocks = reinterpret_cast<Q16_1Block_64 *>(k2->raw_mutable_data());
    auto *v2_blocks = reinterpret_cast<Q16_1Block_64 *>(v2->raw_mutable_data());
    for (size_t row = 0; row < 3; ++row)
    {
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            k2_blocks[row * blocks_per_row + b].d = 2.0f; // Different scale
            k2_blocks[row * blocks_per_row + b].qs[0] = static_cast<int16_t>(100 + row);
            v2_blocks[row * blocks_per_row + b].d = 2.0f;
        }
    }
    ASSERT_TRUE(cache.append_kv_typed(0, 0, k2.get(), v2.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 8);

    // Verify data integrity
    auto cached_k = cache.get_k_typed(0);
    ASSERT_NE(cached_k, nullptr);

    auto *cached_blocks = reinterpret_cast<const Q16_1Block_64 *>(cached_k->raw_data());

    // First batch (rows 0-4): scale=1.0, qs[0]=row
    EXPECT_FLOAT_EQ(cached_blocks[0].d, 1.0f);
    EXPECT_EQ(cached_blocks[0].qs[0], 0);
    EXPECT_EQ(cached_blocks[4 * blocks_per_row].qs[0], 4);

    // Second batch (rows 5-7): scale=2.0, qs[0]=100+row
    EXPECT_FLOAT_EQ(cached_blocks[5 * blocks_per_row].d, 2.0f);
    EXPECT_EQ(cached_blocks[5 * blocks_per_row].qs[0], 100);
    EXPECT_EQ(cached_blocks[7 * blocks_per_row].qs[0], 102);
}

TEST_F(Test__UnifiedKVCache, AppendQ16_1_Batched)
{
    int n_layers = 2;
    int batch_size = 3;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 64;                  // Use 64 so optimal_q16_block_size returns BLOCK_64
    int kv_dim = n_kv_heads * head_dim; // 128

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());
    const size_t block_elements = 64;
    size_t blocks_per_row = (kv_dim + block_elements - 1) / block_elements;

    // Append different lengths to different sequences
    for (int seq = 0; seq < batch_size; ++seq)
    {
        int tokens = (seq + 1) * 2; // 2, 4, 6 tokens
        auto k = factory.createQ16_1({static_cast<size_t>(tokens), static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64);
        auto v = factory.createQ16_1({static_cast<size_t>(tokens), static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64);

        auto *k_blocks = k->mutable_as_block_64();
        auto *v_blocks = v->mutable_as_block_64();
        ASSERT_NE(k_blocks, nullptr);
        ASSERT_NE(v_blocks, nullptr);

        for (int row = 0; row < tokens; ++row)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                k_blocks[row * blocks_per_row + b].d = static_cast<float>(seq + 1);
                k_blocks[row * blocks_per_row + b].qs[0] = static_cast<int16_t>(seq * 1000 + row);
                v_blocks[row * blocks_per_row + b].d = static_cast<float>(seq + 1);
            }
        }

        ASSERT_TRUE(cache.append_kv_typed(0, seq, k.get(), v.get()));
    }

    // Verify per-sequence counts
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 2);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 6);

    // Verify data isolation
    auto k0 = cache.get_k_typed(0, 0);
    auto k1 = cache.get_k_typed(0, 1);
    auto k2 = cache.get_k_typed(0, 2);

    const auto *k0_blocks = k0->as_block_64();
    const auto *k1_blocks = k1->as_block_64();
    const auto *k2_blocks = k2->as_block_64();
    ASSERT_NE(k0_blocks, nullptr);
    ASSERT_NE(k1_blocks, nullptr);
    ASSERT_NE(k2_blocks, nullptr);

    EXPECT_FLOAT_EQ(k0_blocks[0].d, 1.0f);
    EXPECT_EQ(k0_blocks[0].qs[0], 0);

    EXPECT_FLOAT_EQ(k1_blocks[0].d, 2.0f);
    EXPECT_EQ(k1_blocks[0].qs[0], 1000);

    EXPECT_FLOAT_EQ(k2_blocks[0].d, 3.0f);
    EXPECT_EQ(k2_blocks[0].qs[0], 2000);
}

TEST_F(Test__UnifiedKVCache, EvictQ16_1)
{
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 32;
    int n_kv_heads = 2;
    int head_dim = 64;                  // Use head_dim=64 so cache auto-selects BLOCK_64
    int kv_dim = n_kv_heads * head_dim; // 128

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Cache uses BLOCK_64 for head_dim=64
    const Q16BlockSize block_size = Q16BlockSize::BLOCK_64;
    size_t block_elements = q16_block_size_elements(block_size);
    size_t blocks_per_row = (kv_dim + block_elements - 1) / block_elements;

    // Append 10 tokens
    auto k = factory.createQ16_1({10, static_cast<size_t>(kv_dim)}, block_size);
    auto v = factory.createQ16_1({10, static_cast<size_t>(kv_dim)}, block_size);

    auto *k_blocks = reinterpret_cast<Q16_1Block_64 *>(k->raw_mutable_data());
    auto *v_blocks = reinterpret_cast<Q16_1Block_64 *>(v->raw_mutable_data());
    for (int row = 0; row < 10; ++row)
    {
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            k_blocks[row * blocks_per_row + b].d = 1.0f;
            k_blocks[row * blocks_per_row + b].qs[0] = static_cast<int16_t>(row * 100);
            v_blocks[row * blocks_per_row + b].d = 1.0f;
        }
    }
    ASSERT_TRUE(cache.append_kv_typed(0, 0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 10);

    // Evict 4 oldest tokens
    cache.evict_oldest(4);
    EXPECT_EQ(cache.get_cached_tokens(0), 6);

    // Verify data shifted correctly
    auto cached_k = cache.get_k_typed(0);
    ASSERT_NE(cached_k, nullptr);

    auto *cached_blocks = reinterpret_cast<const Q16_1Block_64 *>(cached_k->raw_data());

    // Position 0 should now have what was at position 4 (qs[0] = 400)
    EXPECT_EQ(cached_blocks[0].qs[0], 400);
    // Position 5 should have what was at position 9 (qs[0] = 900)
    EXPECT_EQ(cached_blocks[5 * blocks_per_row].qs[0], 900);
}

TEST_F(Test__UnifiedKVCache, ClearQ16_1)
{
    int kv_dim = 64;
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 2, 2, 16, 2, 32, -1);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createQ16_1({4, static_cast<size_t>(kv_dim)});
    auto v = factory.createQ16_1({4, static_cast<size_t>(kv_dim)});

    // Append to multiple layers and sequences
    cache.append_kv_typed(0, 0, k.get(), v.get());
    cache.append_kv_typed(0, 1, k.get(), v.get());
    cache.append_kv_typed(1, 0, k.get(), v.get());

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 4);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 4);

    // Clear only sequence 1
    cache.clear_sequence(1);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 0);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(1, 1), 0);

    // Clear all
    cache.clear();
    for (int layer = 0; layer < 2; ++layer)
    {
        for (int seq = 0; seq < 2; ++seq)
        {
            EXPECT_EQ(cache.get_cached_tokens(layer, seq), 0);
        }
    }
}

TEST_F(Test__UnifiedKVCache, Q16_1_CapacityExceeded)
{
    int n_kv_heads = 2;
    int head_dim = 32;
    int kv_dim = n_kv_heads * head_dim;
    int max_seq_len = 8;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 1, 1, max_seq_len, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Fill to near capacity (6 tokens)
    auto k1 = factory.createQ16_1({6, static_cast<size_t>(kv_dim)});
    auto v1 = factory.createQ16_1({6, static_cast<size_t>(kv_dim)});
    ASSERT_TRUE(cache.append_kv_typed(0, 0, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 6);

    // Try to exceed capacity (6 + 5 = 11 > 8)
    auto k2 = factory.createQ16_1({5, static_cast<size_t>(kv_dim)});
    auto v2 = factory.createQ16_1({5, static_cast<size_t>(kv_dim)});
    EXPECT_FALSE(cache.append_kv_typed(0, 0, k2.get(), v2.get()));

    // Cache should be unchanged
    EXPECT_EQ(cache.get_cached_tokens(0), 6);
}

TEST_F(Test__UnifiedKVCache, Q16_1_PolymorphicInterface)
{
    auto cache = createUnifiedKVCache(ActivationPrecision::Q16_1, getTestMPIContext(), 2, 1, 16, 2, 32, -1);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
    EXPECT_EQ(cache->num_layers(), 2);
    EXPECT_EQ(cache->batch_size(), 1);
    EXPECT_EQ(cache->max_seq_len(), 16);

    int kv_dim = 2 * 32;
    TensorFactory factory(getTestMPIContext());
    auto k = factory.createQ16_1({4, static_cast<size_t>(kv_dim)});
    auto v = factory.createQ16_1({4, static_cast<size_t>(kv_dim)});

    // Use base interface methods
    ASSERT_TRUE(cache->append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache->get_cached_tokens(0), 4);

    // Verify we can get base pointers
    auto k_base = cache->get_k_base(0);
    auto v_base = cache->get_v_base(0);
    ASSERT_NE(k_base, nullptr);
    ASSERT_NE(v_base, nullptr);
    EXPECT_EQ(k_base->native_type(), TensorType::Q16_1);
    EXPECT_EQ(v_base->native_type(), TensorType::Q16_1);

    cache->clear();
    EXPECT_EQ(cache->get_cached_tokens(0), 0);
}

// =============================================================================
// Test: Clear Operations
// =============================================================================

TEST_F(Test__UnifiedKVCache, ClearAll)
{
    UnifiedKVCacheFP32 cache(getTestMPIContext(), 2, 3, 10, 2, 4, -1);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createFP32({4, 8}, -1);
    auto v = factory.createFP32({4, 8}, -1);

    // Append to multiple layers and sequences
    cache.append_kv(0, 0, k.get(), v.get());
    cache.append_kv(0, 1, k.get(), v.get());
    cache.append_kv(1, 0, k.get(), v.get());

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 4);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 4);

    cache.clear();

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 0);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 0);
}

TEST_F(Test__UnifiedKVCache, ClearSequence)
{
    UnifiedKVCacheFP32 cache(getTestMPIContext(), 2, 3, 10, 2, 4, -1);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createFP32({4, 8}, -1);
    auto v = factory.createFP32({4, 8}, -1);

    // Append to all sequences
    for (int seq = 0; seq < 3; ++seq)
    {
        cache.append_kv(0, seq, k.get(), v.get());
        cache.append_kv(1, seq, k.get(), v.get());
    }

    // Clear only sequence 1
    cache.clear_sequence(1);

    // Sequence 0 and 2 should still have data
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 0); // Cleared
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 4);

    EXPECT_EQ(cache.get_cached_tokens(1, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(1, 1), 0); // Cleared
    EXPECT_EQ(cache.get_cached_tokens(1, 2), 4);
}

// =============================================================================
// Test: Eviction
// =============================================================================

TEST_F(Test__UnifiedKVCache, EvictOldest)
{
    int kv_dim = 8;
    UnifiedKVCacheFP32 cache(getTestMPIContext(), 1, 2, 20, 2, 4, -1);

    TensorFactory factory(getTestMPIContext());

    // Seq 0: 10 tokens
    auto k0 = factory.createFP32({10, static_cast<size_t>(kv_dim)}, -1);
    auto v0 = factory.createFP32({10, static_cast<size_t>(kv_dim)}, -1);
    for (int i = 0; i < 10 * kv_dim; ++i)
    {
        k0->mutable_data()[i] = static_cast<float>(i);
    }
    cache.append_kv(0, 0, k0.get(), v0.get());

    // Seq 1: 8 tokens
    auto k1 = factory.createFP32({8, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = factory.createFP32({8, static_cast<size_t>(kv_dim)}, -1);
    cache.append_kv(0, 1, k1.get(), v1.get());

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 10);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 8);

    // Evict 3 oldest from all sequences
    cache.evict_oldest(3);

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 7); // 10 - 3
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 5); // 8 - 3

    // Verify data shifted correctly (seq 0 should now start with what was token 3)
    auto cached_k0 = cache.get_k_typed(0, 0);
    // Position 0 should now have data that was at position 3 (value = 3 * kv_dim)
    EXPECT_FLOAT_EQ(cached_k0->data()[0], static_cast<float>(3 * kv_dim));
}

// =============================================================================
// Test: Per-Layer Device Placement
// =============================================================================

TEST_F(Test__UnifiedKVCache, PerLayerDevicePlacement)
{
    std::vector<int> devices = {-1, -1, 0, 0}; // Layers 0-1 CPU, 2-3 GPU

    UnifiedKVCacheFP32 cache(getTestMPIContext(), 4, 1, 64, 2, 32, devices);

    EXPECT_EQ(cache.get_layer_device(0), -1);
    EXPECT_EQ(cache.get_layer_device(1), -1);
    EXPECT_EQ(cache.get_layer_device(2), 0);
    EXPECT_EQ(cache.get_layer_device(3), 0);
}

// =============================================================================
// Test: Capacity Exceeded
// =============================================================================

TEST_F(Test__UnifiedKVCache, CapacityExceeded)
{
    int kv_dim = 8;
    int max_seq_len = 10;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), 1, 1, max_seq_len, 2, 4, -1);

    TensorFactory factory(getTestMPIContext());

    // Fill to capacity
    auto k1 = factory.createFP32({8, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = factory.createFP32({8, static_cast<size_t>(kv_dim)}, -1);
    ASSERT_TRUE(cache.append_kv(0, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 8);

    // Try to exceed capacity
    auto k2 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);
    auto v2 = factory.createFP32({5, static_cast<size_t>(kv_dim)}, -1);
    EXPECT_FALSE(cache.append_kv(0, k2.get(), v2.get())); // 8 + 5 = 13 > 10

    // Cache should be unchanged
    EXPECT_EQ(cache.get_cached_tokens(0), 8);
}

// =============================================================================
// Test: IUnifiedKVCache Interface
// =============================================================================

TEST_F(Test__UnifiedKVCache, PolymorphicInterface)
{
    auto cache = createUnifiedKVCache(ActivationPrecision::FP32, getTestMPIContext(), 2, 1, 16, 2, 4, -1);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP32);
    EXPECT_EQ(cache->num_layers(), 2);
    EXPECT_EQ(cache->batch_size(), 1);
    EXPECT_EQ(cache->max_seq_len(), 16);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createFP32({4, 8}, -1);
    auto v = factory.createFP32({4, 8}, -1);

    // Use interface methods (default seq_idx=0)
    ASSERT_TRUE(cache->append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache->get_cached_tokens(0), 4);

    cache->clear();
    EXPECT_EQ(cache->get_cached_tokens(0), 0);
}

// =============================================================================
// Test: Backward Compatibility API (batch_size=1)
// =============================================================================

TEST_F(Test__UnifiedKVCache, BackwardCompatSingleSequence)
{
    // Test that batch_size=1 works identically to the old KVCache API
    UnifiedKVCacheFP32 cache(getTestMPIContext(), 2, 1, 16, 2, 4, -1);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createFP32({4, 8}, -1);
    auto v = factory.createFP32({4, 8}, -1);

    // These should all work with default seq_idx=0
    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get())); // layer, k, v
    EXPECT_EQ(cache.get_cached_tokens(0), 4);          // layer only
    EXPECT_NE(cache.get_k(0), nullptr);                // layer only
    EXPECT_NE(cache.get_v(0), nullptr);                // layer only
    EXPECT_NE(cache.get_k_base(0), nullptr);           // layer only
    EXPECT_NE(cache.get_v_base(0), nullptr);           // layer only

    cache.clear();
    EXPECT_EQ(cache.get_cached_tokens(0), 0);
}
// =============================================================================
// Test: Q16_1 Variable Block Size Support (Phase 3)
// =============================================================================

TEST_F(Test__UnifiedKVCache, Q16_1_AutoSelectBlockSize_HeadDim64)
{
    // Qwen2.5-0.5B: head_dim=64 should auto-select BLOCK_64
    int head_dim = 64;
    int n_kv_heads = 4;
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 2, 1, 16, n_kv_heads, head_dim, -1);

    // Get the K tensor for layer 0 and verify block size
    auto k_base = cache.get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->native_type(), TensorType::Q16_1);

    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_base);
    ASSERT_NE(k_q16, nullptr);
    EXPECT_EQ(k_q16->block_size(), 64) << "head_dim=64 should use BLOCK_64";
    EXPECT_EQ(k_q16->q16_block_size(), Q16BlockSize::BLOCK_64);
}

TEST_F(Test__UnifiedKVCache, Q16_1_AutoSelectBlockSize_HeadDim128)
{
    // Llama3: head_dim=128 should auto-select BLOCK_128
    int head_dim = 128;
    int n_kv_heads = 4;
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 2, 1, 16, n_kv_heads, head_dim, -1);

    auto k_base = cache.get_k_base(0);
    ASSERT_NE(k_base, nullptr);

    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_base);
    ASSERT_NE(k_q16, nullptr);
    EXPECT_EQ(k_q16->block_size(), 128) << "head_dim=128 should use BLOCK_128";
    EXPECT_EQ(k_q16->q16_block_size(), Q16BlockSize::BLOCK_128);
}

// Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
// with independent scales, not a combined 192-dim block.

TEST_F(Test__UnifiedKVCache, Q16_1_VariableBlockAppend_HeadDim64)
{
    // Test append with BLOCK_64 tensors
    int head_dim = 64;
    int n_kv_heads = 2;
    int kv_dim = n_kv_heads * head_dim; // 128
    int seq_len = 8;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 1, 1, 32, n_kv_heads, head_dim, -1);

    // Create Q16_1 tensors with matching block size
    TensorFactory factory(getTestMPIContext());
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_64, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_64, -1);

    EXPECT_EQ(k->block_size(), 64);
    EXPECT_EQ(v->block_size(), 64);

    // Append should work with matching block sizes
    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);
}

TEST_F(Test__UnifiedKVCache, Q16_1_VariableBlockAppend_HeadDim128)
{
    // Test append with BLOCK_128 tensors
    int head_dim = 128;
    int n_kv_heads = 2;
    int kv_dim = n_kv_heads * head_dim; // 256
    int seq_len = 8;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 1, 1, 32, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_128, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_128, -1);

    EXPECT_EQ(k->block_size(), 128);
    EXPECT_EQ(v->block_size(), 128);

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);
}

TEST_F(Test__UnifiedKVCache, Q16_1_VariableBlockEvict_HeadDim64)
{
    // Test eviction with BLOCK_64 preserves data integrity
    int head_dim = 64;
    int n_kv_heads = 2;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 1, 1, 20, n_kv_heads, head_dim, -1);

    TensorFactory factory(getTestMPIContext());

    // Append 10 tokens
    auto k = factory.createQ16_1({10, static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64, -1);
    auto v = factory.createQ16_1({10, static_cast<size_t>(kv_dim)}, Q16BlockSize::BLOCK_64, -1);

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 10);

    // Evict 3 oldest tokens
    cache.evict_oldest(3);
    EXPECT_EQ(cache.get_cached_tokens(0), 7);

    // Verify tensor still has correct block size
    auto k_base = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_base);
    ASSERT_NE(k_q16, nullptr);
    EXPECT_EQ(k_q16->block_size(), 64);
}

TEST_F(Test__UnifiedKVCache, Q16_1_OneBlockPerHead)
{
    // Verify 1-block-per-head optimization: each head dimension = 1 Q16 block
    int head_dim = 128;
    int n_kv_heads = 4;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), 1, 1, 16, n_kv_heads, head_dim, -1);

    auto k_base = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_base);
    ASSERT_NE(k_q16, nullptr);

    // blocks_per_row should equal n_kv_heads (one block per head)
    // kv_dim = n_kv_heads * head_dim = 4 * 128 = 512
    // With BLOCK_128: blocks_per_row = 512 / 128 = 4 = n_kv_heads
    EXPECT_EQ(k_q16->blocks_per_row(), static_cast<size_t>(n_kv_heads))
        << "With head-aligned block size, blocks_per_row should equal n_kv_heads";
}