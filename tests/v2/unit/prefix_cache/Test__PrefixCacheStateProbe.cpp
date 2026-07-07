/**
 * @file Test__PrefixCacheStateProbe.cpp
 * @brief Unit coverage for prefix-cache runtime-state probes and KV segment hashing.
 *
 * These tests keep the diagnostic surface honest without requiring a GPU.  The
 * CPU ring cache still exercises the same logical-block export contract that
 * GPU prefix-restore parity diagnostics use after captured graph execution.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

using namespace llaminar2;

namespace
{
    MPIContext getTestMPIContext()
    {
        return MPIContext(0, 1, MPI_COMM_WORLD);
    }

    /**
     * @brief Restore one environment variable at the end of a probe test.
     *
     * Prefix probe diagnostics are enabled with process-wide environment
     * variables.  Keeping changes scoped prevents one diagnostic test from
     * changing the contract exercised by later tests in the same binary.
     */
    class ScopedEnvVar
    {
    public:
        ScopedEnvVar(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
                old_value_ = std::string(old_value);
            if (value)
                ::setenv(name, value, 1);
            else
                ::unsetenv(name);
        }

        ~ScopedEnvVar()
        {
            if (old_value_)
                ::setenv(name_.c_str(), old_value_->c_str(), 1);
            else
                ::unsetenv(name_.c_str());
        }

        ScopedEnvVar(const ScopedEnvVar &) = delete;
        ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

    private:
        std::string name_;
        std::optional<std::string> old_value_;
    };
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

TEST(Test__PrefixCacheStateProbe, CapturesNamedKVPayloadSegments)
{
    ScopedEnvVar hash_payloads("LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1");
    ScopedEnvVar hash_segments("LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", "1");
    ScopedEnvVar requested_segments(
        "LLAMINAR_PREFIX_PROBE_KV_SEGMENTS",
        "prefix=0:2,suffix=2:2");

    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
    for (size_t i = 0; i < 8; ++i)
    {
        in_k->mutable_data()[i] = static_cast<float>(10 + i);
        in_v->mutable_data()[i] = static_cast<float>(100 + i);
    }

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 4));

    const auto probe = inspectKVCacheForPrefixProbe(cache, "unit", DeviceId::cpu());
    ASSERT_EQ(probe.layers.size(), 1u);
    const PrefixKVLayerProbe &layer = probe.layers.front();
    ASSERT_TRUE(layer.payload_hash_available);
    ASSERT_EQ(layer.segments.size(), 2u);
    EXPECT_EQ(layer.segments[0].name, "prefix");
    EXPECT_EQ(layer.segments[0].token_start, 0);
    EXPECT_EQ(layer.segments[0].token_count, 2);
    EXPECT_TRUE(layer.segments[0].hash_available);
    EXPECT_EQ(layer.segments[1].name, "suffix");
    EXPECT_EQ(layer.segments[1].token_start, 2);
    EXPECT_EQ(layer.segments[1].token_count, 2);
    EXPECT_TRUE(layer.segments[1].hash_available);

    const auto layout = cache.logicalBlockLayout(/*global_layer=*/0, /*token_count=*/2);
    ASSERT_GT(layout.k_bytes, 0u);
    ASSERT_GT(layout.v_bytes, 0u);

    std::vector<uint8_t> suffix_k(layout.k_bytes);
    std::vector<uint8_t> suffix_v(layout.v_bytes);
    IKVCache::KVCacheLogicalBlockDescriptor desc;
    desc.layer = 0;
    desc.seq_idx = 0;
    desc.logical_token_start = 2;
    desc.token_count = 2;
    ASSERT_TRUE(cache.exportLogicalBlock(desc, suffix_k.data(), suffix_v.data()));

    EXPECT_EQ(layer.segments[1].k_payload_bytes, layout.k_bytes);
    EXPECT_EQ(layer.segments[1].v_payload_bytes, layout.v_bytes);
    EXPECT_EQ(layer.segments[1].k_payload_hash,
              hashByteBufferForPrefixProbe(suffix_k.data(), suffix_k.size()));
    EXPECT_EQ(layer.segments[1].v_payload_hash,
              hashByteBufferForPrefixProbe(suffix_v.data(), suffix_v.size()));
}
