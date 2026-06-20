/**
 * @file Test__CUDARingKVCache_Sharding.cpp
 * @brief Unit tests for CUDA ring buffer KV cache sharding (tensor parallelism)
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the sharded CUDA KV cache functionality for tensor parallelism.
 * Verifies that:
 * - Sharded caches store only local KV heads
 * - Memory allocation uses local_kv_dim (not full kv_dim)
 * - Metadata correctly reports sharding state
 * - Non-sharded caches behave as before (backward compatibility)
 * - Both factory functions work correctly
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Helper: Check CUDA availability
    // =========================================================================

    bool hasCUDA()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class Test__CUDARingKVCache_Sharding : public ::testing::Test
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

TEST_F(Test__CUDARingKVCache_Sharding, NonShardedCache_HasFullKVDim)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create non-sharded cache using standard factory
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kHeadDim,
        0 // device_id
    );

    ASSERT_NE(cache, nullptr);

    // Verify non-sharded metadata
    EXPECT_FALSE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
    EXPECT_EQ(cache->local_n_kv_heads(), kNKVHeads); // Same as total
    EXPECT_EQ(cache->kv_head_start(), 0);
    EXPECT_EQ(cache->local_kv_dim(), kKVDim); // Full KV dim
    EXPECT_EQ(cache->kv_dim(), kKVDim);       // Full KV dim

    LOG_INFO("[NonShardedCache_HasFullKVDim] PASSED");
}

TEST_F(Test__CUDARingKVCache_Sharding, NonShardedCache_MetadataConsistency)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);

    // For non-sharded cache, local and total should be identical
    EXPECT_EQ(cache->local_n_kv_heads(), cache->n_kv_heads());
    EXPECT_EQ(cache->local_kv_dim(), cache->kv_dim());
    EXPECT_EQ(cache->kv_head_start(), 0);
    EXPECT_FALSE(cache->is_sharded());

    LOG_INFO("[NonShardedCache_MetadataConsistency] PASSED");
}

// =============================================================================
// Test: Sharded cache (tensor parallelism)
// =============================================================================

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_Rank0_HasLocalKVDim)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create sharded cache for rank 0
    int rank = 0;
    int kv_head_start = rank * kLocalKVHeads;

    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, kv_head_start,
        kHeadDim,
        0 // device_id
    );

    ASSERT_NE(cache, nullptr);

    // Verify sharded metadata
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);           // Total heads
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads); // Local heads
    EXPECT_EQ(cache->kv_head_start(), 0);                // Rank 0 starts at head 0
    EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);       // Reduced KV dim
    EXPECT_EQ(cache->kv_dim(), kLocalKVDim);             // Storage uses local dim

    LOG_INFO("[ShardedCache_Rank0_HasLocalKVDim] PASSED");
}

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_Rank1_HasCorrectOffset)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create sharded cache for rank 1
    int rank = 1;
    int kv_head_start = rank * kLocalKVHeads;

    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, kv_head_start,
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);

    // Verify rank 1 metadata
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->n_kv_heads(), kNKVHeads);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);
    EXPECT_EQ(cache->kv_head_start(), kLocalKVHeads); // Rank 1 starts at head 1
    EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);

    LOG_INFO("[ShardedCache_Rank1_HasCorrectOffset] PASSED");
}

// =============================================================================
// Test: Multi-precision sharded caches
// =============================================================================

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_FP16_Works)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP16,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0, // rank 0
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);

    LOG_INFO("[ShardedCache_FP16_Works] PASSED");
}

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_BF16_Works)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::BF16,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0, // rank 0
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

TEST_F(Test__CUDARingKVCache_Sharding, IsSharded_FalseWhenLocalEqualsTotal)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create with local_n_kv_heads == n_kv_heads (not actually sharded)
    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kNKVHeads, 0, // local == total
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->is_sharded()) << "Should not be sharded when local_n_kv_heads == n_kv_heads";

    LOG_INFO("[IsSharded_FalseWhenLocalEqualsTotal] PASSED");
}

TEST_F(Test__CUDARingKVCache_Sharding, IsSharded_TrueWhenLocalLessThanTotal)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create with local_n_kv_heads < n_kv_heads (actually sharded)
    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0, // local < total
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded()) << "Should be sharded when local_n_kv_heads < n_kv_heads";

    LOG_INFO("[IsSharded_TrueWhenLocalLessThanTotal] PASSED");
}

// =============================================================================
// Test: Sharded cache with larger models (Qwen2.5-7B-like)
// =============================================================================

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_Qwen7BLike_4RankTP)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Parameters similar to Qwen2.5-7B
    const int n_layers = 28;
    const int n_kv_heads = 4; // GQA with 4 KV heads
    const int head_dim = 128;
    const int world_size = 4;
    const int local_kv_heads = n_kv_heads / world_size; // 1 per rank

    for (int rank = 0; rank < world_size; ++rank)
    {
        int kv_head_start = rank * local_kv_heads;

        auto cache = createShardedCUDARingKVCache(
            ActivationPrecision::FP16,
            n_layers, 1, 128, // batch=1, max_seq=128
            n_kv_heads, local_kv_heads, kv_head_start,
            head_dim,
            0);

        ASSERT_NE(cache, nullptr) << "Failed to create cache for rank " << rank;
        EXPECT_TRUE(cache->is_sharded());
        EXPECT_EQ(cache->kv_head_start(), kv_head_start);
        EXPECT_EQ(cache->local_n_kv_heads(), local_kv_heads);
        EXPECT_EQ(cache->local_kv_dim(), local_kv_heads * head_dim);
    }

    LOG_INFO("[ShardedCache_Qwen7BLike_4RankTP] PASSED");
}

// =============================================================================
// Test: Append to sharded cache works correctly
// =============================================================================

TEST_F(Test__CUDARingKVCache_Sharding, ShardedCache_AppendAndRetrieve)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create sharded cache for rank 0
    auto cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(cache, nullptr);

    // Generate test data with LOCAL kv_dim
    const int num_tokens = 10;
    const int local_kv_dim = kLocalKVDim;
    std::vector<float> h_K(num_tokens * local_kv_dim);
    std::vector<float> h_V(num_tokens * local_kv_dim);

    for (size_t i = 0; i < h_K.size(); ++i)
    {
        h_K[i] = static_cast<float>(i) * 0.01f;
        h_V[i] = static_cast<float>(i) * -0.01f;
    }

    // Allocate device memory
    float *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * local_kv_dim * sizeof(float));
    cudaMalloc(&d_V, num_tokens * local_kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), num_tokens * local_kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), num_tokens * local_kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Append to cache (layer 0)
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens, 0));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve K/V
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and verify
    std::vector<float> h_K_out(num_tokens * local_kv_dim);
    std::vector<float> h_V_out(num_tokens * local_kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, num_tokens * local_kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, num_tokens * local_kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify data matches
    for (size_t i = 0; i < h_K.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_K_out[i], h_K[i]) << "K mismatch at index " << i;
        EXPECT_FLOAT_EQ(h_V_out[i], h_V[i]) << "V mismatch at index " << i;
    }

    // Cleanup
    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[ShardedCache_AppendAndRetrieve] PASSED");
}

// =============================================================================
// Test: IKVCache interface compliance
// =============================================================================

TEST_F(Test__CUDARingKVCache_Sharding, IKVCacheInterfaceCompliance)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create sharded cache and access through IKVCache interface
    std::unique_ptr<ICUDARingKVCache> cuda_cache = createShardedCUDARingKVCache(
        ActivationPrecision::FP32,
        kNumLayers, kBatchSize, kMaxSeqLen,
        kNKVHeads, kLocalKVHeads, 0,
        kHeadDim,
        0);

    ASSERT_NE(cuda_cache, nullptr);

    // Access through IKVCache pointer
    IKVCache *cache = cuda_cache.get();

    // Verify IKVCache sharding methods work
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->local_n_kv_heads(), kLocalKVHeads);
    EXPECT_EQ(cache->local_kv_dim(), kLocalKVDim);
    EXPECT_EQ(cache->kv_head_start(), 0);

    // Verify IKVCache core methods
    EXPECT_EQ(cache->n_layers(), kNumLayers);
    EXPECT_EQ(cache->max_seq_len(), kMaxSeqLen);
    EXPECT_EQ(cache->get_cached_tokens(0), 0);

    LOG_INFO("[IKVCacheInterfaceCompliance] PASSED");
}

#else // !HAVE_CUDA

// Provide a dummy test when CUDA is not available
TEST(Test__CUDARingKVCache_Sharding, NoCUDA_Skipped)
{
    GTEST_SKIP() << "CUDA not enabled at build time";
}

#endif // HAVE_CUDA
