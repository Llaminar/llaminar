#include <gtest/gtest.h>

#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <algorithm>

using namespace llaminar2;

namespace
{
    MPIContext getTestMPIContext()
    {
        return MPIContext(0, 1, MPI_COMM_WORLD);
    }
} // namespace

TEST(Test__PrefixCacheStateProbe, FloatHashAndZeroDetection)
{
    const std::vector<float> zeros(8, 0.0f);
    const std::vector<float> values = {0.0f, 1.0f, -2.0f, 3.5f};

    EXPECT_TRUE(floatBufferAllZeroForPrefixProbe(zeros.data(), zeros.size()));
    EXPECT_FALSE(floatBufferAllZeroForPrefixProbe(values.data(), values.size()));
    EXPECT_NE(hashFloatBufferForPrefixProbe(zeros.data(), zeros.size()),
              hashFloatBufferForPrefixProbe(values.data(), values.size()));
}

TEST(Test__PrefixCacheStateProbe, CapturesCPURingKVInventory)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 2, 1, 4, 2, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 4});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 4});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 12, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 12, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 0, in_k.get(), in_v.get(), 3));

    const auto probe = inspectKVCacheForPrefixProbe(cache, "unit", DeviceId::cpu());
    ASSERT_EQ(probe.layers.size(), 2u);
    EXPECT_EQ(probe.owner, "unit");
    EXPECT_EQ(probe.n_layers, 2);
    EXPECT_EQ(probe.max_seq_len, 4);
    EXPECT_EQ(probe.n_kv_heads, 2);
    EXPECT_EQ(probe.local_n_kv_heads, 2);
    EXPECT_EQ(probe.kv_head_start, 0);
    EXPECT_EQ(probe.layers[0].cached_tokens, 3);
    EXPECT_EQ(probe.layers[0].ring_head, 0);
    EXPECT_EQ(probe.layers[1].cached_tokens, 3);
}

TEST(Test__PrefixCacheStateProbe, CapturesClearedCPURingKVInventory)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 4, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 4, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
    cache.clear();

    PrefixRuntimeStateSnapshot snapshot;
    snapshot.kv_caches.push_back(inspectKVCacheForPrefixProbe(cache, "unit", DeviceId::cpu()));

    ASSERT_EQ(snapshot.kv_caches[0].layers.size(), 1u);
    EXPECT_EQ(snapshot.kv_caches[0].layers[0].cached_tokens, 0);
    EXPECT_EQ(snapshot.kv_caches[0].layers[0].ring_head, 0);
    EXPECT_EQ(snapshot.totalCachedTokens(), 0);
    EXPECT_FALSE(snapshot.hasAnyKVState());
}
