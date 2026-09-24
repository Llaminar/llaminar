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

TEST(Test__PrefixCacheStateProbe,
     TerminalValueCaptureAlsoSelectsMatchingDigest)
{
    ScopedEnvVar hash_terminal(
        "LLAMINAR_PREFIX_PROBE_HASH_TERMINAL_STATE",
        nullptr);
    ScopedEnvVar capture_values(
        "LLAMINAR_PREFIX_PROBE_CAPTURE_TERMINAL_HIDDEN_VALUES",
        "1");
    ScopedEnvVar capture_logits_values(
        "LLAMINAR_PREFIX_PROBE_CAPTURE_TERMINAL_LOGITS_VALUES",
        "1");

    const PrefixProbeCapturePolicy policy =
        PrefixProbeCapturePolicy::fromEnvironment();
    EXPECT_TRUE(policy.capture_terminal_hidden_values);
    EXPECT_TRUE(policy.capture_terminal_logits_values);
    EXPECT_TRUE(policy.hash_terminal_state)
        << "Raw terminal values and their digest must come from one materialization";
}

TEST(Test__PrefixCacheStateProbe,
     GDNValueCaptureAlsoSelectsAuthoritativeDeviceDigest)
{
    ScopedEnvVar hash_gdn(
        "LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE",
        nullptr);
    ScopedEnvVar capture_gdn(
        "LLAMINAR_PREFIX_PROBE_CAPTURE_GDN_VALUES",
        "1");

    const PrefixProbeCapturePolicy policy =
        PrefixProbeCapturePolicy::fromEnvironment();
    EXPECT_TRUE(policy.capture_gdn_values);
    EXPECT_TRUE(policy.hash_gdn_device_state)
        << "Raw GPU GDN values and their digest must share one ordered export";
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

/** @brief Each cache's seed length owns its exact-prefix and complete-suffix export. */
TEST(Test__PrefixCacheStateProbe, ContinuationPartitionsUseActualCacheSeeds)
{
    for (const int copied_tokens : {0, 1, 2, 3})
    {
        CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 8, 1, 2, DeviceId::cpu());
        auto k = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
        auto v = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 2});
        std::fill(k->mutable_data(), k->mutable_data() + 8, 1.f);
        std::fill(v->mutable_data(), v->mutable_data() + 8, 2.f);
        PrefixProbeCapturePolicy base;
        base.hash_full_kv_payloads = true;
        base.capture_requested_kv_segment_payloads = true;
        if (copied_tokens)
            ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), copied_tokens));
        PrefixRuntimeStateSnapshot seed;
        seed.initialized = true;
        seed.mtp_kv_caches.push_back(inspectKVCacheForPrefixProbe(
            cache, "mtp:0", DeviceId::cpu(), 1, nullptr, base));
        auto policy = base.forKVContinuationOf(seed);
        ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 4 - copied_tokens));
        const auto probe = inspectKVCacheForPrefixProbe(
            cache, "mtp:0", DeviceId::cpu(), 1, nullptr, policy);
        const auto &layer = probe.layers.front();
        EXPECT_EQ(layer.leading_segment_tokens, copied_tokens);
        if (copied_tokens)
        {
            EXPECT_EQ(layer.leading_k_payload_hash, seed.mtp_kv_caches.front().layers.front().k_payload_hash);
            EXPECT_EQ(layer.leading_v_payload_hash, seed.mtp_kv_caches.front().layers.front().v_payload_hash);
        }
        ASSERT_EQ(layer.segments.size(), 1u);
        EXPECT_EQ(layer.segments.front().token_start, copied_tokens);
        EXPECT_EQ(layer.segments.front().token_count, 4 - copied_tokens);
        EXPECT_EQ(layer.segments.front().k_payload.size(), (4 - copied_tokens) * 2u * sizeof(float));
        EXPECT_THROW(inspectKVCacheForPrefixProbe(
            cache, "wrong-owner", DeviceId::cpu(), 1, nullptr, policy), std::invalid_argument);
        policy.kv_continuation_partitions.push_back(policy.kv_continuation_partitions.front());
        EXPECT_THROW(inspectKVCacheForPrefixProbe(
            cache, "mtp:0", DeviceId::cpu(), 1, nullptr, policy), std::invalid_argument);
        seed.kv_caches = seed.mtp_kv_caches;
        EXPECT_THROW(base.forKVContinuationOf(seed), std::invalid_argument);
    }
}

TEST(Test__PrefixCacheStateProbe, CapturesClearedCPURingKVInventory)
{
    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 4, 1, 2, DeviceId::cpu());

    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    std::fill(in_k->mutable_data(), in_k->mutable_data() + 4, 1.0f);
    std::fill(in_v->mutable_data(), in_v->mutable_data() + 4, 2.0f);

    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 2));
    ASSERT_TRUE(cache.resetRequestState(
        IKVCache::StateResetContext::testReinitialization(nullptr)));

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
    ScopedEnvVar capture_segment_payloads(
        "LLAMINAR_PREFIX_PROBE_CAPTURE_KV_SEGMENT_PAYLOADS",
        "1");
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
    EXPECT_EQ(layer.segments[1].k_payload, suffix_k);
    EXPECT_EQ(layer.segments[1].v_payload, suffix_v);
}

/**
 * @brief Proves invocation-scoped diagnostics do not depend on global env state.
 *
 * Failed mirrored-MTP reporting must be able to request a small logical tail
 * after an error without enabling expensive full-payload hashing for every
 * normal request in the process.  This test disables the compatibility env
 * controls, selects a two-token tail explicitly, and verifies both the full
 * payload and tail hashes come only from the typed policy.
 */
TEST(Test__PrefixCacheStateProbe, ExplicitCapturePolicySelectsBoundedKVPayloads)
{
    ScopedEnvVar hash_payloads("LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", nullptr);
    ScopedEnvVar hash_segments("LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", nullptr);

    CPURingKVCacheFP32 cache(getTestMPIContext(), 1, 1, 8, 1, 2, DeviceId::cpu());
    auto in_k = std::make_shared<FP32Tensor>(std::vector<size_t>{6, 2});
    auto in_v = std::make_shared<FP32Tensor>(std::vector<size_t>{6, 2});
    for (size_t i = 0; i < 12; ++i)
    {
        in_k->mutable_data()[i] = static_cast<float>(20 + i);
        in_v->mutable_data()[i] = static_cast<float>(200 + i);
    }
    ASSERT_TRUE(cache.append_kv(0, 0, in_k.get(), in_v.get(), 6));

    PrefixProbeCapturePolicy capture_policy;
    capture_policy.hash_full_kv_payloads = true;
    capture_policy.trailing_kv_tokens = 2;
    const PrefixKVCacheProbe probe = inspectKVCacheForPrefixProbe(
        cache,
        "explicit",
        DeviceId::cpu(),
        /*sequence_count=*/1,
        /*stream=*/nullptr,
        capture_policy);

    ASSERT_EQ(probe.layers.size(), 1u);
    const PrefixKVLayerProbe &layer = probe.layers.front();
    EXPECT_TRUE(layer.payload_hash_available);
    ASSERT_EQ(layer.segments.size(), 1u);
    EXPECT_EQ(layer.segments.front().name, "diagnostic_tail");
    EXPECT_EQ(layer.segments.front().token_start, 4);
    EXPECT_EQ(layer.segments.front().token_count, 2);
    EXPECT_TRUE(layer.segments.front().hash_available);

    const auto layout = cache.logicalBlockLayout(
        /*global_layer=*/0,
        /*token_count=*/2);
    std::vector<uint8_t> tail_k(layout.k_bytes);
    std::vector<uint8_t> tail_v(layout.v_bytes);
    IKVCache::KVCacheLogicalBlockDescriptor desc;
    desc.layer = 0;
    desc.seq_idx = 0;
    desc.logical_token_start = 4;
    desc.token_count = 2;
    ASSERT_TRUE(cache.exportLogicalBlock(desc, tail_k.data(), tail_v.data()));
    EXPECT_EQ(
        layer.segments.front().k_payload_hash,
        hashByteBufferForPrefixProbe(tail_k.data(), tail_k.size()));
    EXPECT_EQ(
        layer.segments.front().v_payload_hash,
        hashByteBufferForPrefixProbe(tail_v.data(), tail_v.size()));
}
