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

// =============================================================================
// Test: Factory Layout Mode Selection
// =============================================================================

TEST_F(Test__UnifiedKVCache, Factory_DefaultLayoutMode_IsPositionMajor)
{
    // All precisions should default to POSITION_MAJOR when layout_mode not specified
    auto fp32_cache = createUnifiedKVCache(ActivationPrecision::FP32, getTestMPIContext(), 2, 1, 16, 2, 32, -1);
    ASSERT_NE(fp32_cache, nullptr);
    EXPECT_EQ(fp32_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
    EXPECT_EQ(fp32_cache->kv_layout(), TensorLayout::KV_POS_HEAD_DIM);

    auto bf16_cache = createUnifiedKVCache(ActivationPrecision::BF16, getTestMPIContext(), 2, 1, 16, 2, 32, -1);
    ASSERT_NE(bf16_cache, nullptr);
    EXPECT_EQ(bf16_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);

    auto fp16_cache = createUnifiedKVCache(ActivationPrecision::FP16, getTestMPIContext(), 2, 1, 16, 2, 32, -1);
    ASSERT_NE(fp16_cache, nullptr);
    EXPECT_EQ(fp16_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);

    auto q8_1_cache = createUnifiedKVCache(ActivationPrecision::Q8_1, getTestMPIContext(), 2, 1, 16, 2, 32, -1);
    ASSERT_NE(q8_1_cache, nullptr);
    EXPECT_EQ(q8_1_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);

    auto q16_1_cache = createUnifiedKVCache(ActivationPrecision::Q16_1, getTestMPIContext(), 2, 1, 16, 2, 32, -1);
    ASSERT_NE(q16_1_cache, nullptr);
    EXPECT_EQ(q16_1_cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

TEST_F(Test__UnifiedKVCache, Factory_ExplicitHeadMajor_Q16_1)
{
    // Q16_1 with explicit HEAD_MAJOR should use HEAD_MAJOR layout
    // This is required for Q16IntegerAttention kernel compatibility
    auto cache = createUnifiedKVCache(
        ActivationPrecision::Q16_1, getTestMPIContext(),
        2, 1, 16, 2, 64, -1,
        KVCacheLayoutMode::HEAD_MAJOR);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
    EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_HEAD_POS_DIM);

    // Verify tensor shape matches HEAD_MAJOR: [n_kv_heads * max_seq_len, head_dim]
    auto k_base = cache->get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), 2u * 16u);  // n_kv_heads * max_seq_len
    EXPECT_EQ(k_base->cols(), 64u);       // head_dim
}

TEST_F(Test__UnifiedKVCache, Factory_ExplicitHeadMajor_FP32)
{
    // FP32 with explicit HEAD_MAJOR should also work
    auto cache = createUnifiedKVCache(
        ActivationPrecision::FP32, getTestMPIContext(),
        2, 1, 16, 4, 32, -1,
        KVCacheLayoutMode::HEAD_MAJOR);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP32);
    EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_HEAD_POS_DIM);

    // Verify tensor shape
    auto k_base = cache->get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), 4u * 16u);  // n_kv_heads * max_seq_len
    EXPECT_EQ(k_base->cols(), 32u);       // head_dim
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
    // Default layout mode should be POSITION_MAJOR
    EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
}

TEST_F(Test__UnifiedKVCache, ShardedFactory_ExplicitHeadMajor_Q16_1)
{
    // Test sharded factory with explicit HEAD_MAJOR layout
    // This is what GraphOrchestrator should use for HybridQ16 mode
    int n_layers = 4;
    int batch_size = 1;
    int max_seq_len = 64;
    int n_kv_heads = 4;       // Total across all ranks
    int local_n_kv_heads = 2; // This rank's portion
    int kv_head_start = 0;    // Starting head index
    int head_dim = 64;        // 1 block per head with BLOCK_64

    auto cache = createShardedKVCache(
        ActivationPrecision::Q16_1, getTestMPIContext(),
        n_layers, batch_size, max_seq_len,
        n_kv_heads, local_n_kv_heads, kv_head_start,
        head_dim, -1,
        KVCacheLayoutMode::HEAD_MAJOR);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_HEAD_POS_DIM);

    // Verify tensor shape for sharded HEAD_MAJOR: [local_n_kv_heads * max_seq_len, head_dim]
    auto k_base = cache->get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), 2u * 64u);  // local_n_kv_heads * max_seq_len
    EXPECT_EQ(k_base->cols(), 64u);       // head_dim
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

// =============================================================================
// Test: HEAD_MAJOR Layout Mode
// =============================================================================

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_Construction_FP32)
{
    int n_layers = 2;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 4;
    int head_dim = 32;

    // Create cache with HEAD_MAJOR layout
    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                             n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache.kv_layout(), TensorLayout::KV_HEAD_POS_DIM);

    // Verify tensor allocation shape: [n_kv_heads * max_seq_len, head_dim]
    auto k_base = cache.get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), static_cast<size_t>(n_kv_heads * max_seq_len));
    EXPECT_EQ(k_base->cols(), static_cast<size_t>(head_dim));
}

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_Construction_Q16_1)
{
    int n_layers = 2;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 64;

    // Create cache with HEAD_MAJOR layout for Q16 (optimal for Q16IntegerAttention)
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache.kv_layout(), TensorLayout::KV_HEAD_POS_DIM);

    // Verify tensor allocation shape: [n_kv_heads * max_seq_len, head_dim]
    auto k_base = cache.get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), static_cast<size_t>(n_kv_heads * max_seq_len));
    EXPECT_EQ(k_base->cols(), static_cast<size_t>(head_dim));
}

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_AppendScatterCopy_FP32)
{
    // Test that HEAD_MAJOR scatter-copies input from POSITION_MAJOR to HEAD_MAJOR storage
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                             n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Create input in POSITION_MAJOR format: [seq_len, n_kv_heads * head_dim]
    // Layout: [t=0][h=0,d=0..3], [t=0][h=1,d=0..3], [t=1][h=0,d=0..3], ...
    int seq_len = 3;
    auto k_in = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);
    auto v_in = factory.createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, -1);

    // Fill with identifiable pattern: value = t * 100 + h * 10 + d
    float *k_data = k_in->mutable_data();
    float *v_data = v_in->mutable_data();
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                float val = static_cast<float>(t * 100 + h * 10 + d);
                k_data[t * kv_dim + h * head_dim + d] = val;
                v_data[t * kv_dim + h * head_dim + d] = val + 1000.0f;
            }
        }
    }

    // Append to cache
    ASSERT_TRUE(cache.append_kv(0, k_in.get(), v_in.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);

    // Verify HEAD_MAJOR storage: [n_kv_heads * max_seq_len, head_dim]
    // Expected layout: [h=0][t=0..2], [h=1][t=0..2], ...
    auto k_cached = cache.get_k_typed(0);
    auto v_cached = cache.get_v_typed(0);
    ASSERT_NE(k_cached, nullptr);
    ASSERT_NE(v_cached, nullptr);

    const float *k_cache = k_cached->data();
    const float *v_cache = v_cached->data();

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                // HEAD_MAJOR index: (h * max_seq_len + t) * head_dim + d
                size_t cache_idx = (h * max_seq_len + t) * head_dim + d;
                float expected_k = static_cast<float>(t * 100 + h * 10 + d);
                float expected_v = expected_k + 1000.0f;

                EXPECT_FLOAT_EQ(k_cache[cache_idx], expected_k)
                    << "K mismatch at h=" << h << " t=" << t << " d=" << d;
                EXPECT_FLOAT_EQ(v_cache[cache_idx], expected_v)
                    << "V mismatch at h=" << h << " t=" << t << " d=" << d;
            }
        }
    }
}

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_IncrementalAppend_FP32)
{
    // Test incremental append (decode-style, 1 token at a time)
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                             n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Append 5 tokens one at a time
    for (int t = 0; t < 5; ++t)
    {
        auto k = factory.createFP32({1, static_cast<size_t>(kv_dim)}, -1);
        auto v = factory.createFP32({1, static_cast<size_t>(kv_dim)}, -1);

        // Fill with pattern: position * 100 + head * 10 + dim
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                float val = static_cast<float>(t * 100 + h * 10 + d);
                k->mutable_data()[h * head_dim + d] = val;
                v->mutable_data()[h * head_dim + d] = val;
            }
        }

        ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
        EXPECT_EQ(cache.get_cached_tokens(0), t + 1);
    }

    // Verify all tokens are correctly placed in HEAD_MAJOR layout
    auto k_cached = cache.get_k_typed(0);
    const float *k_cache = k_cached->data();

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < 5; ++t)
        {
            size_t cache_idx = (h * max_seq_len + t) * head_dim;
            float expected = static_cast<float>(t * 100 + h * 10); // First element of each position
            EXPECT_FLOAT_EQ(k_cache[cache_idx], expected)
                << "Incremental append mismatch at h=" << h << " t=" << t;
        }
    }
}

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_Eviction_FP32)
{
    // Test that eviction shifts data correctly in HEAD_MAJOR layout
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheFP32 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                             n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Append 6 tokens
    auto k = factory.createFP32({6, static_cast<size_t>(kv_dim)}, -1);
    auto v = factory.createFP32({6, static_cast<size_t>(kv_dim)}, -1);

    for (int t = 0; t < 6; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                float val = static_cast<float>(t * 100 + h * 10 + d);
                k->mutable_data()[t * kv_dim + h * head_dim + d] = val;
                v->mutable_data()[t * kv_dim + h * head_dim + d] = val;
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 6);

    // Evict 2 oldest tokens
    cache.evict_oldest(2);
    EXPECT_EQ(cache.get_cached_tokens(0), 4);

    // Verify remaining tokens are positions 2-5 (originally t=2,3,4,5)
    auto k_cached = cache.get_k_typed(0);
    const float *k_cache = k_cached->data();

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < 4; ++t)
        {
            size_t cache_idx = (h * max_seq_len + t) * head_dim;
            // After evicting 2, position t in cache = original position t+2
            float expected = static_cast<float>((t + 2) * 100 + h * 10);
            EXPECT_FLOAT_EQ(k_cache[cache_idx], expected)
                << "Eviction mismatch at h=" << h << " t=" << t;
        }
    }
}

TEST_F(Test__UnifiedKVCache, HeadMajorLayout_Q16_1_DataIntegrity)
{
    // Test HEAD_MAJOR with Q16_1 quantized data
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = 64; // 1 block per head with BLOCK_64
    int kv_dim = n_kv_heads * head_dim;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Create Q16_1 input in POSITION_MAJOR format
    int seq_len = 4;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_64, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 Q16BlockSize::BLOCK_64, -1);

    // Initialize with identifiable pattern
    size_t blocks_per_row = n_kv_heads; // 1 block per head
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            size_t block_idx = t * blocks_per_row + h;
            auto *k_block = reinterpret_cast<Q16_1Block_64 *>(k->raw_mutable_data()) + block_idx;
            auto *v_block = reinterpret_cast<Q16_1Block_64 *>(v->raw_mutable_data()) + block_idx;

            k_block->d = 0.01f * (t + 1);     // Different scale per position
            v_block->d = 0.01f * (t + 1);
            k_block->qs[0] = static_cast<int16_t>(t * 100 + h * 10); // Identifiable pattern
            v_block->qs[0] = static_cast<int16_t>(t * 100 + h * 10 + 1000);
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);

    // Verify HEAD_MAJOR storage
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    // blocks_per_head = 1 (head_dim=64, BLOCK_64)
    // HEAD_MAJOR index: (h * max_seq_len + t) * blocks_per_head
    const auto *k_blocks = reinterpret_cast<const Q16_1Block_64 *>(k_q16->raw_data());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            size_t block_idx = (h * max_seq_len + t);
            const auto &blk = k_blocks[block_idx];

            float expected_d = 0.01f * (t + 1);
            int16_t expected_qs0 = static_cast<int16_t>(t * 100 + h * 10);

            EXPECT_FLOAT_EQ(blk.d, expected_d)
                << "Q16 scale mismatch at h=" << h << " t=" << t;
            EXPECT_EQ(blk.qs[0], expected_qs0)
                << "Q16 qs[0] mismatch at h=" << h << " t=" << t;
        }
    }
}

TEST_F(Test__UnifiedKVCache, PositionMajorLayout_Default)
{
    // Verify default layout is POSITION_MAJOR
    UnifiedKVCacheFP32 cache(getTestMPIContext(), 2, 1, 16, 4, 32, -1);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
    EXPECT_EQ(cache.kv_layout(), TensorLayout::KV_POS_HEAD_DIM);

    // Verify tensor shape: [max_seq_len, n_kv_heads * head_dim]
    auto k_base = cache.get_k_base(0);
    ASSERT_NE(k_base, nullptr);
    EXPECT_EQ(k_base->rows(), 16u);          // max_seq_len
    EXPECT_EQ(k_base->cols(), 4u * 32u);     // n_kv_heads * head_dim
}

// =============================================================================
// Parameterized Tests: Q16 Block Size Variants for HEAD_MAJOR Layout
// =============================================================================

struct Q16BlockSizeTestParams
{
    Q16BlockSize block_size;
    int head_dim;
    std::string name;
};

class Test__UnifiedKVCache_Q16BlockSizes : public ::testing::TestWithParam<Q16BlockSizeTestParams>
{
  protected:
    MPIContext getTestMPIContext()
    {
        return MPIContext(0, 1, MPI_COMM_WORLD);
    }
};

TEST_P(Test__UnifiedKVCache_Q16BlockSizes, HeadMajor_AppendAndVerify)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // Get block element count for this block size
    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = head_dim / block_elements;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Create Q16_1 input in POSITION_MAJOR format
    int seq_len = 4;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    // Initialize with identifiable pattern per block
    size_t blocks_per_row = n_kv_heads * blocks_per_head;
    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    uint8_t *v_raw = static_cast<uint8_t *>(v->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(k->q16_block_size());

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t block_idx = t * blocks_per_row + h * blocks_per_head + b;

                // Set scale (first 4 bytes of each block structure)
                float scale_k = 0.01f * (t + 1) * (h + 1) * (b + 1);
                float scale_v = 0.02f * (t + 1) * (h + 1) * (b + 1);
                std::memcpy(k_raw + block_idx * block_bytes, &scale_k, sizeof(float));
                std::memcpy(v_raw + block_idx * block_bytes, &scale_v, sizeof(float));

                // Set first quantized value (after 4-byte scale)
                int16_t qs_k = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                int16_t qs_v = static_cast<int16_t>(t * 1000 + h * 100 + b * 10 + 5000);
                std::memcpy(k_raw + block_idx * block_bytes + sizeof(float), &qs_k, sizeof(int16_t));
                std::memcpy(v_raw + block_idx * block_bytes + sizeof(float), &qs_v, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);

    // Verify HEAD_MAJOR storage
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                // HEAD_MAJOR index: (h * max_seq_len + t) * blocks_per_head + b
                size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head + b;

                float cached_scale;
                int16_t cached_qs;
                std::memcpy(&cached_scale, k_cache_raw + cache_block_idx * block_bytes, sizeof(float));
                std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + sizeof(float), sizeof(int16_t));

                float expected_scale = 0.01f * (t + 1) * (h + 1) * (b + 1);
                int16_t expected_qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);

                EXPECT_FLOAT_EQ(cached_scale, expected_scale)
                    << "Scale mismatch at h=" << h << " t=" << t << " b=" << b
                    << " block_size=" << params.name;
                EXPECT_EQ(cached_qs, expected_qs)
                    << "qs[0] mismatch at h=" << h << " t=" << t << " b=" << b
                    << " block_size=" << params.name;
            }
        }
    }
}

TEST_P(Test__UnifiedKVCache_Q16BlockSizes, HeadMajor_IncrementalDecode)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = head_dim / block_elements;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Append 5 tokens one at a time (decode style)
    for (int t = 0; t < 5; ++t)
    {
        auto k = factory.createQ16_1({1, static_cast<size_t>(kv_dim)}, params.block_size, -1);
        auto v = factory.createQ16_1({1, static_cast<size_t>(kv_dim)}, params.block_size, -1);

        uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
        size_t block_bytes = q16_block_size_bytes(k->q16_block_size());
        size_t blocks_per_row = n_kv_heads * blocks_per_head;

        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t block_idx = h * blocks_per_head + b;
                float scale = 0.01f * (t + 1);
                int16_t qs = static_cast<int16_t>(t * 100 + h * 10 + b);
                std::memcpy(k_raw + block_idx * block_bytes, &scale, sizeof(float));
                std::memcpy(k_raw + block_idx * block_bytes + sizeof(float), &qs, sizeof(int16_t));
            }
        }

        ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
        EXPECT_EQ(cache.get_cached_tokens(0), t + 1);
    }

    // Verify all tokens
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());
    size_t block_bytes = q16_block_size_bytes(k_q16->q16_block_size());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < 5; ++t)
        {
            size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head;
            int16_t cached_qs;
            std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + sizeof(float), sizeof(int16_t));

            int16_t expected_qs = static_cast<int16_t>(t * 100 + h * 10);
            EXPECT_EQ(cached_qs, expected_qs)
                << "Incremental decode mismatch at h=" << h << " t=" << t
                << " block_size=" << params.name;
        }
    }
}

TEST_P(Test__UnifiedKVCache_Q16BlockSizes, HeadMajor_EvictionPreservesData)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 16;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = head_dim / block_elements;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Append 6 tokens
    int seq_len = 6;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(k->q16_block_size());
    size_t blocks_per_row = n_kv_heads * blocks_per_head;

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t block_idx = t * blocks_per_row + h * blocks_per_head + b;
                float scale = 0.01f * (t + 1);
                int16_t qs = static_cast<int16_t>(t * 100 + h * 10 + b);
                std::memcpy(k_raw + block_idx * block_bytes, &scale, sizeof(float));
                std::memcpy(k_raw + block_idx * block_bytes + sizeof(float), &qs, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 6);

    // Evict 2 oldest tokens
    cache.evict_oldest(2);
    EXPECT_EQ(cache.get_cached_tokens(0), 4);

    // Verify remaining tokens are positions 2-5 (originally t=2,3,4,5)
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());
    block_bytes = q16_block_size_bytes(k_q16->q16_block_size());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < 4; ++t)
        {
            size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head;
            int16_t cached_qs;
            std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + sizeof(float), sizeof(int16_t));

            // After evicting 2, position t in cache = original position t+2
            int16_t expected_qs = static_cast<int16_t>((t + 2) * 100 + h * 10);
            EXPECT_EQ(cached_qs, expected_qs)
                << "Eviction mismatch at h=" << h << " t=" << t
                << " block_size=" << params.name;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Q16BlockSizeVariants,
    Test__UnifiedKVCache_Q16BlockSizes,
    ::testing::Values(
        Q16BlockSizeTestParams{Q16BlockSize::BLOCK_32, 32, "BLOCK_32_head32"},
        Q16BlockSizeTestParams{Q16BlockSize::BLOCK_64, 64, "BLOCK_64_head64"},
        Q16BlockSizeTestParams{Q16BlockSize::BLOCK_128, 128, "BLOCK_128_head128"}),
    [](const ::testing::TestParamInfo<Q16BlockSizeTestParams> &info)
    {
        return info.param.name;
    });
// =============================================================================
// Extended Q16_1 Block Size Tests: Multi-Blocks-Per-Head
// =============================================================================

// These tests exercise cases where head_dim > block_size, resulting in
// multiple blocks per head. This is important to catch pointer arithmetic bugs.

struct Q16MultiBlockTestParams
{
    Q16BlockSize block_size;
    int head_dim;
    int expected_blocks_per_head;
    std::string name;
};

class Test__UnifiedKVCache_Q16MultiBlock : public ::testing::TestWithParam<Q16MultiBlockTestParams>
{
  protected:
    MPIContext getTestMPIContext()
    {
        return MPIContext(0, 1, MPI_COMM_WORLD);
    }
};

TEST_P(Test__UnifiedKVCache_Q16MultiBlock, HeadMajor_AppendMultiBlockPerHead)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 8;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = params.expected_blocks_per_head;
    ASSERT_EQ(head_dim / block_elements, blocks_per_head) << "Test param mismatch";

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Create input with 3 tokens
    int seq_len = 3;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    // Fill with unique pattern per block: encode (token, head, block_within_head)
    size_t blocks_per_row = n_kv_heads * blocks_per_head;
    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    uint8_t *v_raw = static_cast<uint8_t *>(v->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(params.block_size);

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t src_block_idx = t * blocks_per_row + h * blocks_per_head + b;

                // Use distinctive values to detect any reordering issues
                float scale_k = 0.001f * (t * 100 + h * 10 + b);
                float scale_v = 0.002f * (t * 100 + h * 10 + b);
                
                // Pattern encodes position: t*1000 + h*100 + b*10 + element_offset
                int16_t qs_k_0 = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                int16_t qs_k_1 = static_cast<int16_t>(t * 1000 + h * 100 + b * 10 + 1);
                int16_t qs_v_0 = static_cast<int16_t>(t * 1000 + h * 100 + b * 10 + 5000);

                std::memcpy(k_raw + src_block_idx * block_bytes, &scale_k, sizeof(float));
                std::memcpy(v_raw + src_block_idx * block_bytes, &scale_v, sizeof(float));
                
                // Write first two int16 elements (after scale and sum_qs fields)
                // Block layout: float d (4) + int32_t sum_qs (4) + int16_t qs[N]
                size_t qs_offset = 8; // 4 bytes scale + 4 bytes sum_qs
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset, &qs_k_0, sizeof(int16_t));
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset + 2, &qs_k_1, sizeof(int16_t));
                std::memcpy(v_raw + src_block_idx * block_bytes + qs_offset, &qs_v_0, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);

    // Verify HEAD_MAJOR storage - check all blocks are in correct positions
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                // HEAD_MAJOR: (h * max_seq_len + t) * blocks_per_head + b
                size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head + b;

                float cached_scale;
                int16_t cached_qs_0, cached_qs_1;
                
                std::memcpy(&cached_scale, k_cache_raw + cache_block_idx * block_bytes, sizeof(float));
                size_t qs_offset = 8;
                std::memcpy(&cached_qs_0, k_cache_raw + cache_block_idx * block_bytes + qs_offset, sizeof(int16_t));
                std::memcpy(&cached_qs_1, k_cache_raw + cache_block_idx * block_bytes + qs_offset + 2, sizeof(int16_t));

                float expected_scale = 0.001f * (t * 100 + h * 10 + b);
                int16_t expected_qs_0 = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                int16_t expected_qs_1 = static_cast<int16_t>(t * 1000 + h * 100 + b * 10 + 1);

                EXPECT_NEAR(cached_scale, expected_scale, 1e-6f)
                    << "Scale mismatch at h=" << h << " t=" << t << " b=" << b
                    << " (multi-block: " << params.name << ")";
                EXPECT_EQ(cached_qs_0, expected_qs_0)
                    << "qs[0] mismatch at h=" << h << " t=" << t << " b=" << b
                    << " (multi-block: " << params.name << ")";
                EXPECT_EQ(cached_qs_1, expected_qs_1)
                    << "qs[1] mismatch at h=" << h << " t=" << t << " b=" << b
                    << " (multi-block: " << params.name << ")";
            }
        }
    }
}

TEST_P(Test__UnifiedKVCache_Q16MultiBlock, HeadMajor_EvictionMultiBlockPerHead)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 8;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = params.expected_blocks_per_head;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    // Append 5 tokens
    int seq_len = 5;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    size_t blocks_per_row = n_kv_heads * blocks_per_head;
    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(params.block_size);
    size_t qs_offset = 8;

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t src_block_idx = t * blocks_per_row + h * blocks_per_head + b;
                int16_t qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset, &qs, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), 5);

    // Evict 2 oldest tokens
    cache.evict_oldest(2);
    EXPECT_EQ(cache.get_cached_tokens(0), 3);

    // Verify remaining tokens (originally t=2,3,4) are correctly shifted
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < 3; ++t)  // 3 remaining tokens
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head + b;

                int16_t cached_qs;
                std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + qs_offset, sizeof(int16_t));

                // After evicting 2, cache[t] = original[t+2]
                int16_t expected_qs = static_cast<int16_t>((t + 2) * 1000 + h * 100 + b * 10);
                EXPECT_EQ(cached_qs, expected_qs)
                    << "Eviction mismatch at h=" << h << " t=" << t << " b=" << b
                    << " (multi-block: " << params.name << ")";
            }
        }
    }
}

TEST_P(Test__UnifiedKVCache_Q16MultiBlock, PositionMajor_AppendMultiBlockPerHead)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 8;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = params.expected_blocks_per_head;
    size_t blocks_per_row = n_kv_heads * blocks_per_head;

    // Use POSITION_MAJOR layout
    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::POSITION_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);

    TensorFactory factory(getTestMPIContext());

    int seq_len = 3;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(params.block_size);
    size_t qs_offset = 8;

    // Fill with pattern
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t src_block_idx = t * blocks_per_row + h * blocks_per_head + b;
                int16_t qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset, &qs, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    EXPECT_EQ(cache.get_cached_tokens(0), seq_len);

    // Verify POSITION_MAJOR storage: data should be contiguous by position
    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                // POSITION_MAJOR: t * blocks_per_row + h * blocks_per_head + b
                size_t cache_block_idx = t * blocks_per_row + h * blocks_per_head + b;

                int16_t cached_qs;
                std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + qs_offset, sizeof(int16_t));

                int16_t expected_qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                EXPECT_EQ(cached_qs, expected_qs)
                    << "POSITION_MAJOR mismatch at t=" << t << " h=" << h << " b=" << b
                    << " (multi-block: " << params.name << ")";
            }
        }
    }
}

TEST_P(Test__UnifiedKVCache_Q16MultiBlock, PositionMajor_EvictionMultiBlockPerHead)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 8;
    int n_kv_heads = 2;
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int block_elements = static_cast<int>(params.block_size);
    int blocks_per_head = params.expected_blocks_per_head;
    size_t blocks_per_row = n_kv_heads * blocks_per_head;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::POSITION_MAJOR);

    TensorFactory factory(getTestMPIContext());

    int seq_len = 5;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(params.block_size);
    size_t qs_offset = 8;

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t src_block_idx = t * blocks_per_row + h * blocks_per_head + b;
                int16_t qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset, &qs, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));
    cache.evict_oldest(2);
    EXPECT_EQ(cache.get_cached_tokens(0), 3);

    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int t = 0; t < 3; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t cache_block_idx = t * blocks_per_row + h * blocks_per_head + b;

                int16_t cached_qs;
                std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + qs_offset, sizeof(int16_t));

                // After evicting 2, cache[t] = original[t+2]
                int16_t expected_qs = static_cast<int16_t>((t + 2) * 1000 + h * 100 + b * 10);
                EXPECT_EQ(cached_qs, expected_qs)
                    << "POSITION_MAJOR eviction mismatch at t=" << t << " h=" << h << " b=" << b
                    << " (multi-block: " << params.name << ")";
            }
        }
    }
}

// Test with 4 KV heads to exercise more complex scatter patterns
TEST_P(Test__UnifiedKVCache_Q16MultiBlock, HeadMajor_FourKVHeads)
{
    const auto &params = GetParam();
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 8;
    int n_kv_heads = 4;  // More heads
    int head_dim = params.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int blocks_per_head = params.expected_blocks_per_head;
    size_t blocks_per_row = n_kv_heads * blocks_per_head;

    UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, -1, KVCacheLayoutMode::HEAD_MAJOR);

    TensorFactory factory(getTestMPIContext());

    int seq_len = 3;
    auto k = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);
    auto v = factory.createQ16_1({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)},
                                 params.block_size, -1);

    uint8_t *k_raw = static_cast<uint8_t *>(k->raw_mutable_data());
    size_t block_bytes = q16_block_size_bytes(params.block_size);
    size_t qs_offset = 8;

    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t src_block_idx = t * blocks_per_row + h * blocks_per_head + b;
                int16_t qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                std::memcpy(k_raw + src_block_idx * block_bytes + qs_offset, &qs, sizeof(int16_t));
            }
        }
    }

    ASSERT_TRUE(cache.append_kv(0, k.get(), v.get()));

    auto k_cached = cache.get_k_base(0);
    auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_cached);
    ASSERT_NE(k_q16, nullptr);

    const uint8_t *k_cache_raw = static_cast<const uint8_t *>(k_q16->raw_data());

    for (int h = 0; h < n_kv_heads; ++h)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                size_t cache_block_idx = (h * max_seq_len + t) * blocks_per_head + b;

                int16_t cached_qs;
                std::memcpy(&cached_qs, k_cache_raw + cache_block_idx * block_bytes + qs_offset, sizeof(int16_t));

                int16_t expected_qs = static_cast<int16_t>(t * 1000 + h * 100 + b * 10);
                EXPECT_EQ(cached_qs, expected_qs)
                    << "4-head mismatch at h=" << h << " t=" << t << " b=" << b
                    << " (multi-block: " << params.name << ")";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Q16MultiBlockVariants,
    Test__UnifiedKVCache_Q16MultiBlock,
    ::testing::Values(
        // Multi-block scenarios using head_dim values where optimal_q16_block_size
        // returns a smaller block, naturally creating multiple blocks per head.
        //
        // head_dim=96: optimal returns BLOCK_32 (96 % 32 == 0) -> 3 blocks
        Q16MultiBlockTestParams{Q16BlockSize::BLOCK_32, 96, 3, "head96_BLOCK_32_3blocks"},
        // head_dim=256: optimal returns BLOCK_128 (256 % 128 == 0) -> 2 blocks
        Q16MultiBlockTestParams{Q16BlockSize::BLOCK_128, 256, 2, "head256_BLOCK_128_2blocks"},
        // head_dim=384: optimal returns BLOCK_128 (384 % 128 == 0) -> 3 blocks
        Q16MultiBlockTestParams{Q16BlockSize::BLOCK_128, 384, 3, "head384_BLOCK_128_3blocks"},
        // head_dim=512: optimal returns BLOCK_128 (512 % 128 == 0) -> 4 blocks
        Q16MultiBlockTestParams{Q16BlockSize::BLOCK_128, 512, 4, "head512_BLOCK_128_4blocks"},
        // head_dim=192: optimal returns BLOCK_64 (192 % 64 == 0) -> 3 blocks
        Q16MultiBlockTestParams{Q16BlockSize::BLOCK_64, 192, 3, "head192_BLOCK_64_3blocks"}),
    [](const ::testing::TestParamInfo<Q16MultiBlockTestParams> &info)
    {
        return info.param.name;
    });
