/**
 * @file Test__CPURingKVCache.cpp
 * @brief Phase 1 tests for CPU ring KV cache metadata and append/evict semantics
 */

#include <gtest/gtest.h>

#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

static MPIContext getTestMPIContext()
{
    return MPIContext(0, 1, MPI_COMM_WORLD);
}

class Test__CPURingKVCache : public ::testing::Test
{
};

TEST_F(Test__CPURingKVCache, Construction_InitialState)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 2, 1, 4, 1, 2, DeviceId::cpu());

    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);
    EXPECT_EQ(cache.ring_head(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
}

TEST_F(Test__CPURingKVCache, AppendWithinCapacity_TracksRingState)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});

    float *k = in_k->mutable_data();
    float *v = in_v->mutable_data();
    k[0] = 1.0f;
    k[1] = 2.0f;
    k[2] = 3.0f;
    k[3] = 4.0f;
    v[0] = 10.0f;
    v[1] = 20.0f;
    v[2] = 30.0f;
    v[3] = 40.0f;

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 0), 2);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 2);
}

TEST_F(Test__CPURingKVCache, AppendOverflow_OverwritesOldest)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in1_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in1_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in2_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in2_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});

    std::fill(in1_k->mutable_data(), in1_k->mutable_data() + 6, 1.0f);
    std::fill(in1_v->mutable_data(), in1_v->mutable_data() + 6, 2.0f);
    std::fill(in2_k->mutable_data(), in2_k->mutable_data() + 6, 3.0f);
    std::fill(in2_v->mutable_data(), in2_v->mutable_data() + 6, 4.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in1_k.get(), in1_v.get(), 3));
    ASSERT_TRUE(cache.append_kv(0, 0, in2_k.get(), in2_v.get(), 3));

    EXPECT_EQ(cache.ring_size(0, 0), 4);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 4);
    EXPECT_EQ(cache.ring_head(0, 0), 2);
}

TEST_F(Test__CPURingKVCache, EvictOldest_AdvancesHead)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 8, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 8, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 4));
    cache.evict_oldest_from_sequence(0, 2);

    EXPECT_EQ(cache.ring_size(0, 0), 2);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 2);
    EXPECT_EQ(cache.ring_head(0, 0), 2);
}

TEST_F(Test__CPURingKVCache, Clear_ResetsState)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 2, 2, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 4, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 4, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
    ASSERT_TRUE(cache.append_kv(1, 1, in_k.get(), in_v.get(), 2));

    cache.clear();

    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_head(0, 0), 0);
    EXPECT_EQ(cache.ring_size(1, 1), 0);
    EXPECT_EQ(cache.ring_head(1, 1), 0);
}

TEST_F(Test__CPURingKVCache, HeadMajor_AppendWithinCapacity_TracksRingState)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 2, 2, DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 8, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 8, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 0), 2);
}

TEST_F(Test__CPURingKVCache, HeadMajor_DataMappedByHead)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 2, 2, DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4});

    float *k = in_k->mutable_data();
    float *v = in_v->mutable_data();

    // token 0: h0=[1,2], h1=[10,20]
    // token 1: h0=[3,4], h1=[30,40]
    k[0] = 1.0f;
    k[1] = 2.0f;
    k[2] = 10.0f;
    k[3] = 20.0f;
    k[4] = 3.0f;
    k[5] = 4.0f;
    k[6] = 30.0f;
    k[7] = 40.0f;

    std::fill(v, v + 8, 0.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));

    auto *k_cache = dynamic_cast<FP32Tensor *>(cache.get_k(0, 0));
    ASSERT_NE(k_cache, nullptr);
    const float *kc = k_cache->data();

    // HEAD_MAJOR layout rows: [h0,pos0], [h0,pos1], ... [h1,pos0], [h1,pos1], ...
    // row width = head_dim = 2
    EXPECT_FLOAT_EQ(kc[0], 1.0f); // h0, pos0, d0
    EXPECT_FLOAT_EQ(kc[1], 2.0f); // h0, pos0, d1
    EXPECT_FLOAT_EQ(kc[2], 3.0f); // h0, pos1, d0
    EXPECT_FLOAT_EQ(kc[3], 4.0f); // h0, pos1, d1

    const size_t h1_row0 = static_cast<size_t>(4) * 2; // h=1 starts at row 4, 2 cols each
    EXPECT_FLOAT_EQ(kc[h1_row0 + 0], 10.0f);
    EXPECT_FLOAT_EQ(kc[h1_row0 + 1], 20.0f);
    EXPECT_FLOAT_EQ(kc[h1_row0 + 2], 30.0f);
    EXPECT_FLOAT_EQ(kc[h1_row0 + 3], 40.0f);
}

TEST_F(Test__CPURingKVCache, PositionMajor_GatherLogicalOrder_WithWrap)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu(), KVCacheLayoutMode::POSITION_MAJOR);

    auto in1_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in1_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in2_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in2_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});

    float *k1 = in1_k->mutable_data();
    float *v1 = in1_v->mutable_data();
    float *k2 = in2_k->mutable_data();
    float *v2 = in2_v->mutable_data();

    // Tokens t0,t1,t2
    for (int t = 0; t < 3; ++t)
    {
        k1[t * 2 + 0] = 100.0f + t;
        k1[t * 2 + 1] = 200.0f + t;
        v1[t * 2 + 0] = 300.0f + t;
        v1[t * 2 + 1] = 400.0f + t;
    }
    // Tokens t3,t4,t5
    for (int t = 0; t < 3; ++t)
    {
        k2[t * 2 + 0] = 103.0f + t;
        k2[t * 2 + 1] = 203.0f + t;
        v2[t * 2 + 0] = 303.0f + t;
        v2[t * 2 + 1] = 403.0f + t;
    }

    ASSERT_TRUE(cache.append_kv(0, 0, in1_k.get(), in1_v.get(), 3));
    ASSERT_TRUE(cache.append_kv(0, 0, in2_k.get(), in2_v.get(), 3));

    EXPECT_EQ(cache.ring_head(0, 0), 2);
    EXPECT_EQ(cache.ring_size(0, 0), 4);

    auto out_k = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    auto out_v = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, 4);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], 4);

    const float *ok = out_k->data();
    const float *ov = out_v->data();

    // Expect logical order: t2,t3,t4,t5
    for (int row = 0; row < 4; ++row)
    {
        const float tok = 102.0f + row;
        EXPECT_FLOAT_EQ(ok[row * 2 + 0], tok);
        EXPECT_FLOAT_EQ(ok[row * 2 + 1], tok + 100.0f);
        EXPECT_FLOAT_EQ(ov[row * 2 + 0], tok + 200.0f);
        EXPECT_FLOAT_EQ(ov[row * 2 + 1], tok + 300.0f);
    }
}

TEST_F(Test__CPURingKVCache, HeadMajor_GatherLogicalOrder_WithWrap)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 2, 2, DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{5, 4});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{5, 4});

    float *k = in_k->mutable_data();
    float *v = in_v->mutable_data();

    // token t: h0=[10+t,20+t], h1=[30+t,40+t]
    for (int t = 0; t < 5; ++t)
    {
        k[t * 4 + 0] = 10.0f + t;
        k[t * 4 + 1] = 20.0f + t;
        k[t * 4 + 2] = 30.0f + t;
        k[t * 4 + 3] = 40.0f + t;

        v[t * 4 + 0] = 110.0f + t;
        v[t * 4 + 1] = 120.0f + t;
        v[t * 4 + 2] = 130.0f + t;
        v[t * 4 + 3] = 140.0f + t;
    }

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 5));
    EXPECT_EQ(cache.ring_size(0, 0), 4);

    auto out_k = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
    auto out_v = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, 4);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], 4);

    const float *ok = out_k->data();
    const float *ov = out_v->data();

    // Ring should keep t1,t2,t3,t4 in logical order
    for (int row = 0; row < 4; ++row)
    {
        const float t = 1.0f + row;
        EXPECT_FLOAT_EQ(ok[row * 4 + 0], 10.0f + t);
        EXPECT_FLOAT_EQ(ok[row * 4 + 1], 20.0f + t);
        EXPECT_FLOAT_EQ(ok[row * 4 + 2], 30.0f + t);
        EXPECT_FLOAT_EQ(ok[row * 4 + 3], 40.0f + t);

        EXPECT_FLOAT_EQ(ov[row * 4 + 0], 110.0f + t);
        EXPECT_FLOAT_EQ(ov[row * 4 + 1], 120.0f + t);
        EXPECT_FLOAT_EQ(ov[row * 4 + 2], 130.0f + t);
        EXPECT_FLOAT_EQ(ov[row * 4 + 3], 140.0f + t);
    }
}

// =============================================================================
// Ring buffer wrap + clear cycle tests
// =============================================================================

TEST_F(Test__CPURingKVCache, WrapThenClear_ResetsFullState)
{
    // Cache with small capacity (4 tokens)
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 6, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 6, 2.0f);

    // Fill to capacity (3 tokens) then overflow (3 more → wraps with 2 overwritten)
    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 3));
    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 3));
    EXPECT_EQ(cache.ring_size(0, 0), 4); // Full, wrapped

    // Clear resets everything including wrap warning state
    cache.clear();

    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    // Should be able to fill again from scratch
    auto in2_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    auto in2_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    float *k = in2_k->mutable_data();
    float *v = in2_v->mutable_data();
    k[0] = 10.0f; k[1] = 20.0f; k[2] = 30.0f; k[3] = 40.0f;
    v[0] = 50.0f; v[1] = 60.0f; v[2] = 70.0f; v[3] = 80.0f;

    ASSERT_TRUE(cache.append_kv(0, 0, in2_k.get(), in2_v.get(), 2));
    EXPECT_EQ(cache.ring_size(0, 0), 2);
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    // Verify data integrity after clear+refill
    auto *k_cache = dynamic_cast<FP32Tensor *>(cache.get_k(0, 0));
    ASSERT_NE(k_cache, nullptr);
    const float *kc = k_cache->data();
    EXPECT_FLOAT_EQ(kc[0], 10.0f);
    EXPECT_FLOAT_EQ(kc[1], 20.0f);
    EXPECT_FLOAT_EQ(kc[2], 30.0f);
    EXPECT_FLOAT_EQ(kc[3], 40.0f);
}

TEST_F(Test__CPURingKVCache, MultipleWrapCycles_DataIntegrity)
{
    // Small cache: 3 token capacity
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 3, 1, 2, DeviceId::cpu());

    for (int cycle = 0; cycle < 3; ++cycle)
    {
        auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
        auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
        float base = static_cast<float>(cycle * 100);
        float *k = in_k->mutable_data();
        float *v = in_v->mutable_data();
        k[0] = base + 1; k[1] = base + 2; k[2] = base + 3; k[3] = base + 4;
        v[0] = base + 5; v[1] = base + 6; v[2] = base + 7; v[3] = base + 8;

        ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
        EXPECT_EQ(cache.ring_size(0, 0), 2);

        // Overflow: add 2 more tokens to a 3-capacity cache that already has 2
        auto in2_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
        auto in2_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
        std::fill(in2_k->mutable_data(), in2_k->mutable_data() + 4, base + 10);
        std::fill(in2_v->mutable_data(), in2_v->mutable_data() + 4, base + 20);

        ASSERT_TRUE(cache.append_kv(0, 0, in2_k.get(), in2_v.get(), 2));
        EXPECT_EQ(cache.ring_size(0, 0), 3); // Capped at capacity

        // Clear for next cycle
        cache.clear();
        EXPECT_EQ(cache.ring_size(0, 0), 0);
        EXPECT_EQ(cache.ring_head(0, 0), 0);
    }
}
