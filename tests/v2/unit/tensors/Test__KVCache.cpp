/**
 * @file Test__KVCache.cpp
 * @brief Unit tests for KV cache functionality
 * @author David Sanftenberg
 */

#include "../../../../src/v2/tensors/KVCache.h"
#include "../../../../src/v2/utils/Logger.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

// Single-rank MPI context for unit tests
static MPIContext getTestMPIContext()
{
    return MPIContext(0, 1, MPI_COMM_WORLD);
}

class KVCacheTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        n_layers_ = 4;
        max_seq_len_ = 128;
        n_kv_heads_ = 2;
        head_dim_ = 64;

        cache_ = std::make_shared<KVCache>(getTestMPIContext(), n_layers_, max_seq_len_, n_kv_heads_, head_dim_);
    }

    int n_layers_;
    int max_seq_len_;
    int n_kv_heads_;
    int head_dim_;
    std::shared_ptr<KVCache> cache_;
};

/**
 * @brief Test basic cache initialization
 */
TEST_F(KVCacheTest, Initialization)
{
    EXPECT_EQ(cache_->num_layers(), n_layers_);
    EXPECT_EQ(cache_->max_seq_len(), max_seq_len_);

    // All layers should start empty
    for (int i = 0; i < n_layers_; ++i)
    {
        EXPECT_EQ(cache_->get_cached_tokens(i), 0);
        EXPECT_EQ(cache_->get_k(i), nullptr); // Empty cache returns nullptr
        EXPECT_EQ(cache_->get_v(i), nullptr);
    }
}

/**
 * @brief Test appending K/V to cache
 */
TEST_F(KVCacheTest, AppendKV)
{
    int layer = 0;
    int seq_len = 8;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Create sample K/V tensors
    auto K = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, -1);
    auto V = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, -1);

    // Fill with test data
    for (size_t i = 0; i < seq_len * kv_dim; ++i)
    {
        K->mutable_data()[i] = static_cast<float>(i);
        V->mutable_data()[i] = static_cast<float>(i + 1000);
    }

    // Append to cache
    EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
    EXPECT_EQ(cache_->get_cached_tokens(layer), seq_len);

    // Verify data
    auto cached_K = cache_->get_k(layer);
    auto cached_V = cache_->get_v(layer);
    ASSERT_NE(cached_K, nullptr);
    ASSERT_NE(cached_V, nullptr);

    for (size_t i = 0; i < seq_len * kv_dim; ++i)
    {
        EXPECT_FLOAT_EQ(cached_K->data()[i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(cached_V->data()[i], static_cast<float>(i + 1000));
    }
}

/**
 * @brief Test incremental append (prefill + decode)
 */
TEST_F(KVCacheTest, IncrementalAppend)
{
    int layer = 1;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Prefill: 16 tokens
    int prefill_len = 16;
    auto K_prefill = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(prefill_len), kv_dim}, -1);
    auto V_prefill = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(prefill_len), kv_dim}, -1);

    for (size_t i = 0; i < prefill_len * kv_dim; ++i)
    {
        K_prefill->mutable_data()[i] = 1.0f;
        V_prefill->mutable_data()[i] = 2.0f;
    }

    EXPECT_TRUE(cache_->append_kv(layer, K_prefill.get(), V_prefill.get()));
    EXPECT_EQ(cache_->get_cached_tokens(layer), prefill_len);

    // Decode: append 1 token at a time (3 steps)
    for (int step = 0; step < 3; ++step)
    {
        auto K_decode = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, kv_dim}, -1);
        auto V_decode = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, kv_dim}, -1);

        for (size_t i = 0; i < kv_dim; ++i)
        {
            K_decode->mutable_data()[i] = 10.0f + step;
            V_decode->mutable_data()[i] = 20.0f + step;
        }

        EXPECT_TRUE(cache_->append_kv(layer, K_decode.get(), V_decode.get()));
        EXPECT_EQ(cache_->get_cached_tokens(layer), prefill_len + step + 1);
    }

    // Verify total tokens
    EXPECT_EQ(cache_->get_cached_tokens(layer), prefill_len + 3);

    // Verify cached data (spot check)
    auto cached_K = cache_->get_k(layer);
    // First token from prefill
    EXPECT_FLOAT_EQ(cached_K->data()[0], 1.0f);
    // First token from decode
    EXPECT_FLOAT_EQ(cached_K->data()[prefill_len * kv_dim], 10.0f);
}

/**
 * @brief Test cache capacity overflow with automatic eviction (sliding window)
 */
TEST_F(KVCacheTest, CapacityOverflow)
{
    int layer = 0;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Fill cache to capacity with known values
    auto K_fill = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len_), kv_dim}, -1);
    auto V_fill = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len_), kv_dim}, -1);

    // Initialize with position-based values so we can verify eviction
    for (int i = 0; i < max_seq_len_; ++i)
    {
        for (size_t j = 0; j < kv_dim; ++j)
        {
            K_fill->mutable_data()[i * kv_dim + j] = static_cast<float>(i);
            V_fill->mutable_data()[i * kv_dim + j] = static_cast<float>(i + 1000);
        }
    }

    EXPECT_TRUE(cache_->append_kv(layer, K_fill.get(), V_fill.get()));
    EXPECT_EQ(cache_->get_cached_tokens(layer), max_seq_len_);

    // Append one more token - should auto-evict oldest and succeed
    auto K_extra = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);
    auto V_extra = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);

    // Fill with distinctive value
    for (size_t j = 0; j < kv_dim; ++j)
    {
        K_extra->mutable_data()[j] = 999.0f;
        V_extra->mutable_data()[j] = 1999.0f;
    }

    // With sliding window eviction, this should succeed
    EXPECT_TRUE(cache_->append_kv(layer, K_extra.get(), V_extra.get()));
    EXPECT_EQ(cache_->get_cached_tokens(layer), max_seq_len_); // Still at max (evicted 1, added 1)

    // Verify the oldest token was evicted (position 0 data should be gone)
    // The new first position should have the old position 1 data
    auto cached_K = cache_->get_k(layer);
    EXPECT_FLOAT_EQ(cached_K->data()[0], 1.0f); // Was position 1, now position 0

    // Verify the newest token is at the end
    size_t last_offset = (max_seq_len_ - 1) * kv_dim;
    EXPECT_FLOAT_EQ(cached_K->data()[last_offset], 999.0f);
}

/**
 * @brief Test cache clearing
 */
TEST_F(KVCacheTest, CacheClear)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Append data to multiple layers
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);
        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
    }

    // Verify all layers have data
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        EXPECT_EQ(cache_->get_cached_tokens(layer), 8);
    }

    // Clear all
    cache_->clear();

    // Verify all layers are empty
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        EXPECT_EQ(cache_->get_cached_tokens(layer), 0);
    }
}

/**
 * @brief Test layer-specific clearing
 */
TEST_F(KVCacheTest, LayerSpecificClear)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Append data to all layers
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);
        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
    }

    // Clear layer 1 only
    cache_->clear_layer(1);

    // Verify layer 1 is empty, others unchanged
    EXPECT_EQ(cache_->get_cached_tokens(0), 8);
    EXPECT_EQ(cache_->get_cached_tokens(1), 0); // Cleared
    EXPECT_EQ(cache_->get_cached_tokens(2), 8);
    EXPECT_EQ(cache_->get_cached_tokens(3), 8);
}

/**
 * @brief Test heterogeneous device placement (CPU + GPU split)
 *
 * Simulates a model with layers split across devices. The cache follows
 * where attention computation happens (not necessarily where FFN/experts live).
 */
TEST(KVCacheHeterogeneousTest, MultiDevicePlacement)
{
    int n_layers = 8;
    int max_seq_len = 64;
    int n_kv_heads = 2;
    int head_dim = 32;

    // Split layers across devices (attention device per layer):
    // Layers 0-3: Attention on CPU
    // Layers 4-7: Attention on GPU 0
    // (Note: For MoE, FFN/experts might be on different devices)
    std::vector<int> attention_devices = {-1, -1, -1, -1, 0, 0, 0, 0};

    auto cache = std::make_shared<KVCache>(getTestMPIContext(), n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);

    // Verify attention device affinity for each layer
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(cache->get_attention_device(i), -1) << "Layer " << i << " attention should be on CPU";
        EXPECT_EQ(cache->get_layer_device(i), -1) << "Backward compat: get_layer_device works";
    }
    for (int i = 4; i < 8; ++i)
    {
        EXPECT_EQ(cache->get_attention_device(i), 0) << "Layer " << i << " attention should be on GPU 0";
    }

    // Test appending K/V to different devices
    int seq_len = 8;
    size_t kv_dim = n_kv_heads * head_dim;

    for (int layer = 0; layer < n_layers; ++layer)
    {
        int expected_device = (layer < 4) ? -1 : 0;

        // Create K/V on the correct device for this layer
        auto new_k = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, expected_device);
        auto new_v = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, expected_device);

        // Fill with layer-specific pattern
        float *k_data = const_cast<float *>(new_k->data()); // Safe: we own this tensor
        float *v_data = const_cast<float *>(new_v->data());
        for (size_t i = 0; i < seq_len * kv_dim; ++i)
        {
            k_data[i] = static_cast<float>(layer * 100 + i);
            v_data[i] = static_cast<float>(layer * 100 + i + 0.5f);
        }

        // Append to cache
        EXPECT_TRUE(cache->append_kv(layer, new_k.get(), new_v.get()))
            << "Failed to append K/V to layer " << layer << " on device " << expected_device;

        // Verify cached data
        EXPECT_EQ(cache->get_cached_tokens(layer), seq_len);

        auto cached_k = cache->get_k(layer);
        auto cached_v = cache->get_v(layer);
        ASSERT_NE(cached_k, nullptr);
        ASSERT_NE(cached_v, nullptr);

        // Verify device placement
        EXPECT_EQ(cached_k->device_index(), expected_device);
        EXPECT_EQ(cached_v->device_index(), expected_device);

        // Spot check data integrity
        EXPECT_FLOAT_EQ(cached_k->data()[0], static_cast<float>(layer * 100));
        EXPECT_FLOAT_EQ(cached_v->data()[0], static_cast<float>(layer * 100 + 0.5f));
    }
}

/**
 * @brief Test heterogeneous with 3 devices (CPU + 2 GPUs)
 *
 * Demonstrates cache placement for multi-GPU scenarios. Each layer's
 * cache resides where its attention computation occurs.
 */
TEST(KVCacheHeterogeneousTest, ThreeDevicePlacement)
{
    int n_layers = 12;
    int max_seq_len = 32;
    int n_kv_heads = 4;
    int head_dim = 16;

    // Split across 3 devices (attention device per layer):
    // Layers 0-3: Attention on CPU
    // Layers 4-7: Attention on GPU 0
    // Layers 8-11: Attention on GPU 1
    std::vector<int> attention_devices = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};

    auto cache = std::make_shared<KVCache>(getTestMPIContext(), n_layers, max_seq_len, n_kv_heads, head_dim, attention_devices);

    // Verify each layer's attention device
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(cache->get_attention_device(i), -1);
    for (int i = 4; i < 8; ++i)
        EXPECT_EQ(cache->get_attention_device(i), 0);
    for (int i = 8; i < 12; ++i)
        EXPECT_EQ(cache->get_attention_device(i), 1);

    // Test clearing specific layers doesn't affect other devices
    int seq_len = 4;
    size_t kv_dim = n_kv_heads * head_dim;

    // Add data to all layers
    for (int layer = 0; layer < n_layers; ++layer)
    {
        int attn_device = attention_devices[layer];
        auto new_k = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, attn_device);
        auto new_v = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim}, attn_device);
        EXPECT_TRUE(cache->append_kv(layer, new_k.get(), new_v.get()));
    }

    // Clear GPU 0 layers only (4-7)
    for (int i = 4; i < 8; ++i)
    {
        cache->clear_layer(i);
    }

    // Verify GPU 0 layers cleared
    for (int i = 4; i < 8; ++i)
    {
        EXPECT_EQ(cache->get_cached_tokens(i), 0);
    }

    // Verify CPU and GPU 1 layers untouched
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(cache->get_cached_tokens(i), seq_len);
    }
    for (int i = 8; i < 12; ++i)
    {
        EXPECT_EQ(cache->get_cached_tokens(i), seq_len);
    }
}

// =============================================================================
// Edge Case and Boundary Condition Tests
// =============================================================================

/**
 * @brief Test appending data to cache in interleaved order across layers
 *
 * Validates that appending to different layers in non-sequential order
 * doesn't corrupt data.
 */
TEST_F(KVCacheTest, InterleavedLayerAppends)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Append to layers in random order: 2, 0, 3, 1
    std::vector<int> layer_order = {2, 0, 3, 1};

    for (int layer : layer_order)
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);

        // Fill with layer-specific pattern
        for (size_t i = 0; i < 4 * kv_dim; ++i)
        {
            K->mutable_data()[i] = static_cast<float>(layer * 100 + i);
            V->mutable_data()[i] = static_cast<float>(layer * 100 + i + 0.5f);
        }

        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
    }

    // Verify each layer has correct data
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        EXPECT_EQ(cache_->get_cached_tokens(layer), 4);

        auto K = cache_->get_k(layer);
        ASSERT_NE(K, nullptr);

        // Verify first element matches layer pattern
        EXPECT_FLOAT_EQ(K->data()[0], static_cast<float>(layer * 100));
    }
}

/**
 * @brief Test multiple small appends followed by verification
 *
 * Simulates realistic decode: many single-token appends
 */
TEST_F(KVCacheTest, ManySmallAppends)
{
    int layer = 0;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Append 50 single tokens
    int num_tokens = 50;
    for (int t = 0; t < num_tokens; ++t)
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);

        for (size_t i = 0; i < kv_dim; ++i)
        {
            K->mutable_data()[i] = static_cast<float>(t);
            V->mutable_data()[i] = static_cast<float>(t + 1000);
        }

        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
        EXPECT_EQ(cache_->get_cached_tokens(layer), t + 1);
    }

    // Verify all data
    auto cached_K = cache_->get_k(layer);
    ASSERT_NE(cached_K, nullptr);

    for (int t = 0; t < num_tokens; ++t)
    {
        // Check first element of each token's kv_dim block
        EXPECT_FLOAT_EQ(cached_K->data()[t * kv_dim], static_cast<float>(t))
            << "Token " << t << " has wrong value";
    }
}

/**
 * @brief Test boundary: append exactly max_seq_len tokens
 */
TEST_F(KVCacheTest, ExactCapacityFill)
{
    int layer = 0;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // Fill to exactly max_seq_len
    auto K = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len_), kv_dim}, -1);
    auto V = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(max_seq_len_), kv_dim}, -1);

    for (size_t i = 0; i < max_seq_len_ * kv_dim; ++i)
    {
        K->mutable_data()[i] = static_cast<float>(i);
        V->mutable_data()[i] = static_cast<float>(i);
    }

    EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
    EXPECT_EQ(cache_->get_cached_tokens(layer), max_seq_len_);

    // Verify first and last elements
    auto cached_K = cache_->get_k(layer);
    EXPECT_FLOAT_EQ(cached_K->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(cached_K->data()[max_seq_len_ * kv_dim - 1],
                    static_cast<float>(max_seq_len_ * kv_dim - 1));
}

/**
 * @brief Test data integrity after clear and refill
 */
TEST_F(KVCacheTest, ClearAndRefill)
{
    int layer = 0;
    size_t kv_dim = n_kv_heads_ * head_dim_;

    // First fill with pattern A
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{8, kv_dim}, -1);

        for (size_t i = 0; i < 8 * kv_dim; ++i)
        {
            K->mutable_data()[i] = 111.0f;
            V->mutable_data()[i] = 222.0f;
        }

        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
        EXPECT_EQ(cache_->get_cached_tokens(layer), 8);
    }

    // Clear
    cache_->clear_layer(layer);
    EXPECT_EQ(cache_->get_cached_tokens(layer), 0);

    // Refill with pattern B
    {
        auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);
        auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);

        for (size_t i = 0; i < 4 * kv_dim; ++i)
        {
            K->mutable_data()[i] = 333.0f;
            V->mutable_data()[i] = 444.0f;
        }

        EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
        EXPECT_EQ(cache_->get_cached_tokens(layer), 4);
    }

    // Verify pattern B (not A)
    auto cached_K = cache_->get_k(layer);
    auto cached_V = cache_->get_v(layer);

    for (size_t i = 0; i < 4 * kv_dim; ++i)
    {
        EXPECT_FLOAT_EQ(cached_K->data()[i], 333.0f);
        EXPECT_FLOAT_EQ(cached_V->data()[i], 444.0f);
    }
}

/**
 * @brief Test invalid layer indices are handled gracefully
 */
TEST_F(KVCacheTest, InvalidLayerHandling)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;

    auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);
    auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);

    // Negative layer
    EXPECT_FALSE(cache_->append_kv(-1, K.get(), V.get()));

    // Layer >= n_layers
    EXPECT_FALSE(cache_->append_kv(n_layers_, K.get(), V.get()));
    EXPECT_FALSE(cache_->append_kv(n_layers_ + 10, K.get(), V.get()));

    // get_cached_tokens for invalid layers returns 0
    EXPECT_EQ(cache_->get_cached_tokens(-1), 0);
    EXPECT_EQ(cache_->get_cached_tokens(n_layers_), 0);

    // get_k/get_v for invalid layers returns nullptr
    EXPECT_EQ(cache_->get_k(-1), nullptr);
    EXPECT_EQ(cache_->get_v(n_layers_), nullptr);
}

/**
 * @brief Test null tensor handling
 */
TEST_F(KVCacheTest, NullTensorHandling)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;
    auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);
    auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{4, kv_dim}, -1);

    // Null K
    EXPECT_FALSE(cache_->append_kv(0, nullptr, V.get()));

    // Null V
    EXPECT_FALSE(cache_->append_kv(0, K.get(), nullptr));

    // Both null
    EXPECT_FALSE(cache_->append_kv(0, nullptr, nullptr));

    // Cache should still be empty
    EXPECT_EQ(cache_->get_cached_tokens(0), 0);
}

/**
 * @brief Test prefill + multiple decode pattern matching real pipeline usage
 *
 * Simulates: prefill with N tokens, then 3 decode steps with 1 token each
 */
TEST_F(KVCacheTest, RealisticPrefillDecodePattern)
{
    size_t kv_dim = n_kv_heads_ * head_dim_;
    int prefill_len = 32;  // Typical prompt length
    int decode_steps = 10; // Typical decode steps

    // Test across all layers to simulate real transformer
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        // Prefill: batch of tokens
        {
            auto K = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(prefill_len), kv_dim}, -1);
            auto V = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(prefill_len), kv_dim}, -1);

            // Fill with identifiable pattern
            for (int t = 0; t < prefill_len; ++t)
            {
                for (size_t d = 0; d < kv_dim; ++d)
                {
                    K->mutable_data()[t * kv_dim + d] = static_cast<float>(layer * 10000 + t);
                    V->mutable_data()[t * kv_dim + d] = static_cast<float>(layer * 10000 + t + 0.5f);
                }
            }

            EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
            EXPECT_EQ(cache_->get_cached_tokens(layer), prefill_len);
        }

        // Decode: single tokens
        for (int step = 0; step < decode_steps; ++step)
        {
            auto K = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);
            auto V = std::make_shared<FP32Tensor>(std::vector<size_t>{1, kv_dim}, -1);

            int token_id = prefill_len + step;
            for (size_t d = 0; d < kv_dim; ++d)
            {
                K->mutable_data()[d] = static_cast<float>(layer * 10000 + token_id);
                V->mutable_data()[d] = static_cast<float>(layer * 10000 + token_id + 0.5f);
            }

            EXPECT_TRUE(cache_->append_kv(layer, K.get(), V.get()));
            EXPECT_EQ(cache_->get_cached_tokens(layer), prefill_len + step + 1);
        }
    }

    // Final verification: all layers should have same token count
    int expected_tokens = prefill_len + decode_steps;
    for (int layer = 0; layer < n_layers_; ++layer)
    {
        EXPECT_EQ(cache_->get_cached_tokens(layer), expected_tokens);

        // Verify data integrity
        auto K = cache_->get_k(layer);

        // First token from prefill
        EXPECT_FLOAT_EQ(K->data()[0], static_cast<float>(layer * 10000));

        // First token from decode
        EXPECT_FLOAT_EQ(K->data()[prefill_len * kv_dim],
                        static_cast<float>(layer * 10000 + prefill_len));

        // Last token from decode
        int last_token_id = expected_tokens - 1;
        EXPECT_FLOAT_EQ(K->data()[last_token_id * kv_dim],
                        static_cast<float>(layer * 10000 + last_token_id));
    }
}
