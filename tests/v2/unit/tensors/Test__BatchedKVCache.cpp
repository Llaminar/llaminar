/**
 * @file Test__BatchedKVCache.cpp
 * @brief Unit tests for batched KV cache
 * @author David Sanftenberg
 * @date October 26, 2025
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/BatchedKVCache.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include <vector>
#include <memory>

using namespace llaminar2;

class Test__BatchedKVCache : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test: Basic Construction and Metadata
// =============================================================================

TEST_F(Test__BatchedKVCache, ConstructionUniformDevice)
{
    int n_layers = 24;
    int batch_size = 4;
    int max_seq_len = 2048;
    int n_kv_heads = 2;
    int head_dim = 64;
    int device_idx = -1; // CPU

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_idx);

    EXPECT_EQ(cache.num_layers(), n_layers);
    EXPECT_EQ(cache.batch_size(), batch_size);
    EXPECT_EQ(cache.max_seq_len(), max_seq_len);

    // Initially, all cached token counts should be 0
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache.get_cached_tokens(layer, seq), 0);
        }
    }

    // All layers should be on same device
    for (int layer = 0; layer < n_layers; ++layer)
    {
        EXPECT_EQ(cache.get_layer_device(layer), device_idx);
    }
}

TEST_F(Test__BatchedKVCache, ConstructionPerLayerDevices)
{
    int n_layers = 4;
    int batch_size = 2;
    int max_seq_len = 128;
    int n_kv_heads = 2;
    int head_dim = 32;

    // Heterogeneous placement: layers 0-1 on CPU, layers 2-3 on device 0
    std::vector<int> attention_devices = {-1, -1, 0, 0};

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices);

    EXPECT_EQ(cache.get_layer_device(0), -1);
    EXPECT_EQ(cache.get_layer_device(1), -1);
    EXPECT_EQ(cache.get_layer_device(2), 0);
    EXPECT_EQ(cache.get_layer_device(3), 0);
}

// =============================================================================
// Test: Multi-Sequence Append
// =============================================================================

TEST_F(Test__BatchedKVCache, MultiSequenceAppend)
{
    int n_layers = 2;
    int batch_size = 3;
    int max_seq_len = 10;
    int n_kv_heads = 2;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim; // 8

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append K/V for different sequences in layer 0
    // Sequence 0: 3 tokens
    auto k0 = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    auto v0 = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    float *k0_data = k0->mutable_data();
    float *v0_data = v0->mutable_data();
    for (int i = 0; i < 3 * kv_dim; ++i)
    {
        k0_data[i] = 1.0f + i;
        v0_data[i] = 100.0f + i;
    }

    EXPECT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get()));
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 3);

    // Sequence 1: 2 tokens
    auto k1 = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
    float *k1_data = k1->mutable_data();
    float *v1_data = v1->mutable_data();
    for (int i = 0; i < 2 * kv_dim; ++i)
    {
        k1_data[i] = 2.0f + i;
        v1_data[i] = 200.0f + i;
    }

    EXPECT_TRUE(cache.append_kv(0, 1, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 2);

    // Sequence 2: 4 tokens
    auto k2 = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(kv_dim)}, -1);
    auto v2 = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(kv_dim)}, -1);
    float *k2_data = k2->mutable_data();
    float *v2_data = v2->mutable_data();
    for (int i = 0; i < 4 * kv_dim; ++i)
    {
        k2_data[i] = 3.0f + i;
        v2_data[i] = 300.0f + i;
    }

    EXPECT_TRUE(cache.append_kv(0, 2, k2.get(), v2.get()));
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 4);

    // Other sequences in layer 0 should still be 0
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 0);
    EXPECT_EQ(cache.get_cached_tokens(1, 1), 0);
    EXPECT_EQ(cache.get_cached_tokens(1, 2), 0);
}

// =============================================================================
// Test: Per-Sequence Retrieval
// =============================================================================

TEST_F(Test__BatchedKVCache, PerSequenceRetrieval)
{
    int n_layers = 1;
    int batch_size = 2;
    int max_seq_len = 8;
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append different data to each sequence
    auto k0 = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    auto v0 = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    float *k0_data = k0->mutable_data();
    float *v0_data = v0->mutable_data();
    for (int i = 0; i < 3 * kv_dim; ++i)
    {
        k0_data[i] = 10.0f;
        v0_data[i] = 20.0f;
    }
    cache.append_kv(0, 0, k0.get(), v0.get());

    auto k1 = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
    float *k1_data = k1->mutable_data();
    float *v1_data = v1->mutable_data();
    for (int i = 0; i < 2 * kv_dim; ++i)
    {
        k1_data[i] = 30.0f;
        v1_data[i] = 40.0f;
    }
    cache.append_kv(0, 1, k1.get(), v1.get());

    // Retrieve sequence 0
    auto retrieved_k0 = cache.get_k(0, 0);
    auto retrieved_v0 = cache.get_v(0, 0);
    ASSERT_NE(retrieved_k0, nullptr);
    ASSERT_NE(retrieved_v0, nullptr);

    const auto &k0_shape = retrieved_k0->shape();
    EXPECT_EQ(k0_shape[0], 3); // cached_tokens
    EXPECT_EQ(k0_shape[1], kv_dim);

    const float *retrieved_k0_data = retrieved_k0->data();
    const float *retrieved_v0_data = retrieved_v0->data();
    for (int i = 0; i < 3 * kv_dim; ++i)
    {
        EXPECT_EQ(retrieved_k0_data[i], 10.0f);
        EXPECT_EQ(retrieved_v0_data[i], 20.0f);
    }

    // Retrieve sequence 1
    auto retrieved_k1 = cache.get_k(0, 1);
    auto retrieved_v1 = cache.get_v(0, 1);
    ASSERT_NE(retrieved_k1, nullptr);
    ASSERT_NE(retrieved_v1, nullptr);

    const auto &k1_shape = retrieved_k1->shape();
    EXPECT_EQ(k1_shape[0], 2); // cached_tokens
    EXPECT_EQ(k1_shape[1], kv_dim);

    const float *retrieved_k1_data = retrieved_k1->data();
    const float *retrieved_v1_data = retrieved_v1->data();
    for (int i = 0; i < 2 * kv_dim; ++i)
    {
        EXPECT_EQ(retrieved_k1_data[i], 30.0f);
        EXPECT_EQ(retrieved_v1_data[i], 40.0f);
    }
}

// =============================================================================
// Test: Incremental Append (Decode)
// =============================================================================

TEST_F(Test__BatchedKVCache, IncrementalAppend)
{
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 10;
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Prefill: Append 3 tokens
    auto k_prefill = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    auto v_prefill = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
    float *k_prefill_data = k_prefill->mutable_data();
    float *v_prefill_data = v_prefill->mutable_data();
    for (int i = 0; i < 3 * kv_dim; ++i)
    {
        k_prefill_data[i] = 1.0f;
        v_prefill_data[i] = 2.0f;
    }
    cache.append_kv(0, 0, k_prefill.get(), v_prefill.get());
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 3);

    // Decode step 1: Append 1 token
    auto k_decode1 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    auto v_decode1 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    float *k_decode1_data = k_decode1->mutable_data();
    float *v_decode1_data = v_decode1->mutable_data();
    for (int i = 0; i < kv_dim; ++i)
    {
        k_decode1_data[i] = 3.0f;
        v_decode1_data[i] = 4.0f;
    }
    cache.append_kv(0, 0, k_decode1.get(), v_decode1.get());
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);

    // Decode step 2: Append 1 more token
    auto k_decode2 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    auto v_decode2 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    float *k_decode2_data = k_decode2->mutable_data();
    float *v_decode2_data = v_decode2->mutable_data();
    for (int i = 0; i < kv_dim; ++i)
    {
        k_decode2_data[i] = 5.0f;
        v_decode2_data[i] = 6.0f;
    }
    cache.append_kv(0, 0, k_decode2.get(), v_decode2.get());
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 5);

    // Retrieve and verify concatenation
    auto k = cache.get_k(0, 0);
    auto v = cache.get_v(0, 0);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);

    EXPECT_EQ(k->shape()[0], 5);
    const float *k_data = k->data();
    const float *v_data = v->data();

    // First 3 tokens (prefill)
    for (int i = 0; i < 3 * kv_dim; ++i)
    {
        EXPECT_EQ(k_data[i], 1.0f);
        EXPECT_EQ(v_data[i], 2.0f);
    }

    // 4th token (decode step 1)
    for (int i = 0; i < kv_dim; ++i)
    {
        EXPECT_EQ(k_data[3 * kv_dim + i], 3.0f);
        EXPECT_EQ(v_data[3 * kv_dim + i], 4.0f);
    }

    // 5th token (decode step 2)
    for (int i = 0; i < kv_dim; ++i)
    {
        EXPECT_EQ(k_data[4 * kv_dim + i], 5.0f);
        EXPECT_EQ(v_data[4 * kv_dim + i], 6.0f);
    }
}

// =============================================================================
// Test: Capacity Management
// =============================================================================

TEST_F(Test__BatchedKVCache, CapacityExceeded)
{
    int n_layers = 1;
    int batch_size = 1;
    int max_seq_len = 5; // Small capacity
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append 5 tokens (exactly at capacity)
    auto k1 = std::make_shared<FP32Tensor>(std::vector<size_t>{5, static_cast<size_t>(kv_dim)}, -1);
    auto v1 = std::make_shared<FP32Tensor>(std::vector<size_t>{5, static_cast<size_t>(kv_dim)}, -1);
    EXPECT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get()));
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 5);

    // Try to append 1 more token (should fail - capacity exceeded)
    auto k2 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    auto v2 = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, -1);
    EXPECT_FALSE(cache.append_kv(0, 0, k2.get(), v2.get()));

    // Cache should remain at 5 tokens
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 5);
}

// =============================================================================
// Test: Clear Operations
// =============================================================================

TEST_F(Test__BatchedKVCache, ClearSequence)
{
    int n_layers = 1;
    int batch_size = 3;
    int max_seq_len = 10;
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append to all sequences
    for (int seq = 0; seq < batch_size; ++seq)
    {
        auto k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
        auto v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, static_cast<size_t>(kv_dim)}, -1);
        cache.append_kv(0, seq, k.get(), v.get());
    }

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 3);
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 3);
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 3);

    // Clear sequence 1
    cache.clear_sequence(1);

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 3); // unchanged
    EXPECT_EQ(cache.get_cached_tokens(0, 1), 0); // cleared
    EXPECT_EQ(cache.get_cached_tokens(0, 2), 3); // unchanged
}

TEST_F(Test__BatchedKVCache, ClearAll)
{
    int n_layers = 2;
    int batch_size = 2;
    int max_seq_len = 10;
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append to all layers and sequences
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            auto k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
            auto v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
            cache.append_kv(layer, seq, k.get(), v.get());
        }
    }

    // Verify all have cached tokens
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache.get_cached_tokens(layer, seq), 2);
        }
    }

    // Clear all
    cache.clear();

    // Verify all are cleared
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache.get_cached_tokens(layer, seq), 0);
        }
    }
}

TEST_F(Test__BatchedKVCache, ClearLayer)
{
    int n_layers = 3;
    int batch_size = 2;
    int max_seq_len = 10;
    int n_kv_heads = 1;
    int head_dim = 4;
    int kv_dim = n_kv_heads * head_dim;

    BatchedKVCache cache(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, -1);

    // Append to all layers and sequences
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            auto k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
            auto v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, static_cast<size_t>(kv_dim)}, -1);
            cache.append_kv(layer, seq, k.get(), v.get());
        }
    }

    // Clear layer 1
    cache.clear_layer(1);

    // Verify layer 1 is cleared, others unchanged
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(cache.get_cached_tokens(0, seq), 2); // unchanged
        EXPECT_EQ(cache.get_cached_tokens(1, seq), 0); // cleared
        EXPECT_EQ(cache.get_cached_tokens(2, seq), 2); // unchanged
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
