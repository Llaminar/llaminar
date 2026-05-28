/**
 * @file Test__ROCmRingKVCache_Sharding.cpp
 * @brief Unit tests for ROCm ring buffer KV cache sharding (tensor parallelism)
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the sharded ROCm KV cache functionality for tensor parallelism.
 * Verifies that:
 * - Sharded caches store only local KV heads
 * - Memory allocation uses local_kv_dim (not full kv_dim)
 * - Metadata correctly reports sharding state
 * - Non-sharded caches behave as before (backward compatibility)
 * - Both factory functions work correctly
 *
 * NOTE: ROCm KVCache sharding is not yet implemented. These tests currently
 * skip with GTEST_SKIP() until the sharding API is added to ROCmRingKVCache.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/kvcache/ROCmRingKVCache.h"
#include "kernels/rocm/kvcache/ROCmRingKVCacheFactory.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Helper: Check ROCm availability
    // =========================================================================

    bool hasROCm()
    {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
    }

    // =========================================================================
    // Helper: Check if ROCm KV cache supports sharding
    // =========================================================================

    bool supportsSharding()
    {
        // ROCm KV cache sharding is now implemented
        return true;
    }

} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ROCmRingKVCache_Sharding : public ::testing::Test
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
};

// =============================================================================
// Test: Non-sharded cache (backward compatibility)
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, NonShardedCache_HasFullKVDim)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Create non-sharded cache using concrete type (tests need ROCm-specific methods)
    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kHeadDim,
        0 // device_id
    );

    ASSERT_NE(cache, nullptr);

    // Verify non-sharded metadata via IKVCache interface
    // Non-sharded caches use default implementations that return false/0
    EXPECT_FALSE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
    EXPECT_EQ(cache->kv_dim(), kKVDim); // Full KV dim

    LOG_INFO("[NonShardedCache_HasFullKVDim] PASSED");
}

TEST_F(Test__ROCmRingKVCache_Sharding, NonShardedCache_MetadataConsistency)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);

    // For non-sharded ROCm cache, is_sharded() returns false
    // local_n_kv_heads() and local_kv_dim() return 0 (not overridden)
    EXPECT_FALSE(cache->is_sharded());
    EXPECT_EQ(cache->kv_head_start(), 0);
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
    EXPECT_EQ(cache->kv_dim(), kKVDim);

    LOG_INFO("[NonShardedCache_MetadataConsistency] PASSED");
}

// =============================================================================
// Test: Sharded cache (tensor parallelism) - NOT YET IMPLEMENTED
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_Rank0_HasLocalKVDim)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    int rank = 0;
    int kv_head_start = rank * kLocalKVHeads;
    // Use sharded constructor (n_kv_heads, local_n_kv_heads, kv_head_start variant)
    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, kv_head_start,
        kHeadDim,
        0 // device_id
    );

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);
    EXPECT_EQ(cache->kv_head_start(), 0);
    EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);

    LOG_INFO("[ShardedCache_Rank0_HasLocalKVDim] PASSED");
}

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_Rank1_HasCorrectOffset)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    int rank = 1;
    int kv_head_start = rank * kLocalKVHeads;
    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, kv_head_start,
        kHeadDim, 0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->kv_head_start(), kLocalKVHeads);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);

    LOG_INFO("[ShardedCache_Rank1_HasCorrectOffset] PASSED");
}

// =============================================================================
// Test: Multi-precision sharded caches - NOT YET IMPLEMENTED
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_FP16_Works)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP16>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);

    LOG_INFO("[ShardedCache_FP16_Works] PASSED");
}

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_BF16_Works)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::BF16>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);

    LOG_INFO("[ShardedCache_BF16_Works] PASSED");
}

// =============================================================================
// Test: Sharding state detection
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, IsSharded_FalseWhenLocalEqualsTotal)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kNKVHeads, 0, // local == total
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->is_sharded()) << "Should not be sharded when local_n_kv_heads == n_kv_heads";

    LOG_INFO("[IsSharded_FalseWhenLocalEqualsTotal] PASSED");
}

TEST_F(Test__ROCmRingKVCache_Sharding, IsSharded_TrueWhenLocalLessThanTotal)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0, // local < total
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded()) << "Should be sharded when local_n_kv_heads < n_kv_heads";

    LOG_INFO("[IsSharded_TrueWhenLocalLessThanTotal] SKIPPED - sharding not implemented");
}

// =============================================================================
// Test: Sharded cache with larger models (Qwen2.5-7B-like) - NOT YET IMPLEMENTED
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_Qwen7BLike_4RankTP)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    // Test 4-rank tensor parallelism with Qwen2.5-7B-like config
    const int n_layers = 28;
    const int n_kv_heads = 4;
    const int head_dim = 128;
    const int world_size = 4;
    const int local_kv_heads = n_kv_heads / world_size; // 1 per rank

    for (int rank = 0; rank < world_size; ++rank)
    {
        int kv_head_start = rank * local_kv_heads;
        auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP16>>(
            n_layers, 1, 128,
            n_kv_heads, local_kv_heads, kv_head_start,
            head_dim,
            0);

        ASSERT_NE(cache, nullptr);
        EXPECT_TRUE(cache->is_sharded());
        EXPECT_EQ(cache->kv_head_start(), kv_head_start);
        EXPECT_EQ(cache->local_n_kv_heads(), local_kv_heads);
        EXPECT_EQ(cache->local_kv_dim(), static_cast<size_t>(local_kv_heads * head_dim));
    }

    LOG_INFO("[ShardedCache_Qwen7BLike_4RankTP] PASSED");
}

// =============================================================================
// Test: Append to sharded cache works correctly - NOT YET IMPLEMENTED
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, ShardedCache_AppendAndRetrieve)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    // Test append/retrieve on sharded cache
    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);

    const int num_tokens = 10;
    const size_t local_kv_dim = kLocalKVDim;
    std::vector<float> h_K(num_tokens * local_kv_dim);
    std::vector<float> h_V(num_tokens * local_kv_dim);

    for (size_t i = 0; i < h_K.size(); ++i)
    {
        h_K[i] = static_cast<float>(i) * 0.01f;
        h_V[i] = static_cast<float>(i) * -0.01f;
    }

    float *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * local_kv_dim * sizeof(float));
    hipMalloc(&d_V, num_tokens * local_kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), num_tokens * local_kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(h_V.data(), h_V.data(), num_tokens * local_kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * local_kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens, 0));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, num_tokens);

    std::vector<float> h_K_out(num_tokens * local_kv_dim);
    std::vector<float> h_V_out(num_tokens * local_kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, num_tokens * local_kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, num_tokens * local_kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    for (size_t i = 0; i < h_K.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_K_out[i], h_K[i]);
        EXPECT_FLOAT_EQ(h_V_out[i], h_V[i]);
    }

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[ShardedCache_AppendAndRetrieve] PASSED");
}

// =============================================================================
// Test: IKVCache interface compliance
// =============================================================================

TEST_F(Test__ROCmRingKVCache_Sharding, IKVCacheInterfaceCompliance)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }
    if (!supportsSharding())
    {
        GTEST_SKIP() << "ROCm KVCache sharding not yet implemented";
    }

    // Test IKVCache interface compliance for sharded cache
    auto rocm_cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(rocm_cache, nullptr);

    // Access via concrete type - provides all ROCm-specific methods
    auto *cache = rocm_cache.get();

    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);
    EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
    EXPECT_EQ(cache->kv_head_start(), 0);

    EXPECT_EQ(cache->n_layers(), kNumLayers);
    EXPECT_EQ(cache->max_seq_len(), kMaxSeqLen);
    EXPECT_EQ(cache->get_cached_tokens(0), 0);

    LOG_INFO("[IKVCacheInterfaceCompliance] PASSED");
}

#else // !HAVE_ROCM

// Provide a dummy test when ROCm is not available
TEST(Test__ROCmRingKVCache_Sharding, NoROCm_Skipped)
{
    GTEST_SKIP() << "ROCm not enabled at build time";
}

#endif // HAVE_ROCM
