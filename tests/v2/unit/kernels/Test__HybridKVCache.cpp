/**
 * @file Test__HybridKVCache.cpp
 * @brief Unit tests for hybrid KV cache: layer mapping, GDN state, and factory creation
 *
 * Regression tests for:
 * - HybridLayerMap global→KV and global→GDN index mapping
 * - CPUHybridRingKVCache layer remapping (GDN layers return 0 tokens, no-op append)
 * - IHybridKVCache interface correctness (isGDNLayer, getGDNState, kernel access)
 * - KernelFactory::createHybridKVCache for all CPU precisions (break fallthrough bug)
 * - clear() resetting both KV and GDN state
 */

#include <gtest/gtest.h>
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/HybridGDNStateGeometry.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUHybridRingKVCache.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include "../../utils/TestTensorFactory.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <bit>
#include <limits>
#include <cmath>
#include "utils/OpenMPUtils.h"
#include <utility>
#include <vector>

using namespace llaminar::v2::kernels;
using namespace llaminar2;

namespace llaminar2::test
{

    static const IMPIContext &getTestMPIContext()
    {
        static MPIContext ctx(0, 1, MPI_COMM_WORLD);
        return ctx;
    }

    // =============================================================================
    // Qwen 3.5 0.8B: 24 layers, full_attention_interval=4
    // FA layers: 3, 7, 11, 15, 19, 23 (6 FA, 18 GDN)
    // =============================================================================

    static HybridKVCacheConfig makeQwen35_08B_Config()
    {
        HybridKVCacheConfig config;
        config.layer_types.resize(24);
        for (int i = 0; i < 24; ++i)
        {
            config.layer_types[i] = ((i + 1) % 4 == 0) ? "full_attention" : "gdn";
        }
        config.gdn_conv_kernel_size = 4;
        config.gdn_state_size = 64; // d_k = d_v = 64
        config.gdn_inner_size = 0;
        config.gdn_group_count = 4;     // n_k_heads
        config.gdn_time_step_rank = 16; // n_v_heads
        config.n_heads = 32;
        config.local_n_heads = 0;
        return config;
    }

    static HybridKVCacheConfig makeQwen35_08B_StageConfig(int first_layer, int layer_count)
    {
        HybridKVCacheConfig config = makeQwen35_08B_Config();
        std::vector<std::string> stage_layer_types;
        stage_layer_types.reserve(static_cast<size_t>(layer_count));
        for (int layer = first_layer; layer < first_layer + layer_count; ++layer)
        {
            stage_layer_types.push_back(config.layer_types.at(static_cast<size_t>(layer)));
        }
        config.layer_types = std::move(stage_layer_types);
        config.first_layer_index = first_layer;
        return config;
    }

    // =============================================================================
    // Test: HybridLayerMap basic mapping for Qwen 3.5 0.8B layout
    // =============================================================================

    class Test__HybridLayerMap : public ::testing::Test
    {
    };

    TEST_F(Test__HybridLayerMap, Qwen35_0_8B_LayerMapping)
    {
        auto config = makeQwen35_08B_Config();
        HybridLayerMap map;
        map.build(config.layer_types);

        EXPECT_EQ(map.totalLayers(), 24);
        EXPECT_EQ(map.kvLayerCount(), 6);
        EXPECT_EQ(map.gdnLayerCount(), 18);

        // FA layers at indices 3, 7, 11, 15, 19, 23
        int fa_layers[] = {3, 7, 11, 15, 19, 23};
        for (int fa_idx = 0; fa_idx < 6; ++fa_idx)
        {
            int fa_layer = fa_layers[fa_idx];
            EXPECT_TRUE(map.isFullAttention(fa_layer)) << "Layer " << fa_layer;
            EXPECT_EQ(map.toKVIndex(fa_layer), fa_idx) << "Layer " << fa_layer;
            EXPECT_EQ(map.toGDNIndex(fa_layer), -1) << "Layer " << fa_layer;
        }

        // First 3 GDN layers (0, 1, 2)
        for (int gdn_idx = 0; gdn_idx < 3; ++gdn_idx)
        {
            EXPECT_FALSE(map.isFullAttention(gdn_idx)) << "Layer " << gdn_idx;
            EXPECT_EQ(map.toKVIndex(gdn_idx), -1) << "Layer " << gdn_idx;
            EXPECT_EQ(map.toGDNIndex(gdn_idx), gdn_idx) << "Layer " << gdn_idx;
        }
    }

    TEST_F(Test__HybridLayerMap, OutOfBoundsReturnsNegative)
    {
        auto config = makeQwen35_08B_Config();
        HybridLayerMap map;
        map.build(config.layer_types);

        EXPECT_EQ(map.toKVIndex(-1), -1);
        EXPECT_EQ(map.toKVIndex(24), -1);
        EXPECT_EQ(map.toGDNIndex(-1), -1);
        EXPECT_EQ(map.toGDNIndex(24), -1);
        EXPECT_FALSE(map.isFullAttention(-1));
        EXPECT_FALSE(map.isFullAttention(24));
    }

    TEST_F(Test__HybridLayerMap, AllFAModel_NoGDNLayers)
    {
        HybridLayerMap map;
        std::vector<std::string> all_fa(8, "full_attention");
        map.build(all_fa);

        EXPECT_EQ(map.kvLayerCount(), 8);
        EXPECT_EQ(map.gdnLayerCount(), 0);
        for (int i = 0; i < 8; ++i)
        {
            EXPECT_TRUE(map.isFullAttention(i));
            EXPECT_EQ(map.toKVIndex(i), i);
            EXPECT_EQ(map.toGDNIndex(i), -1);
        }
    }

    TEST_F(Test__HybridLayerMap, AllGDNModel_NoFALayers)
    {
        HybridLayerMap map;
        std::vector<std::string> all_gdn(8, "gdn");
        map.build(all_gdn);

        EXPECT_EQ(map.kvLayerCount(), 0);
        EXPECT_EQ(map.gdnLayerCount(), 8);
        for (int i = 0; i < 8; ++i)
        {
            EXPECT_FALSE(map.isFullAttention(i));
            EXPECT_EQ(map.toKVIndex(i), -1);
            EXPECT_EQ(map.toGDNIndex(i), i);
        }
    }

    // =============================================================================
    // Test: canonical GDN geometry and physical byte formulas
    // =============================================================================

    /** @brief Equal local/full geometry owns one request-backed live bank. */
    TEST(Test__HybridGDNStateGeometry, ZeroLayersOwnNoCPUCheckpointHistory)
    {
        const HybridGDNStateGeometry absent;
        EXPECT_EQ(absent.localPayloadBytes(0), 0u);
        EXPECT_THROW((void)absent.localPayloadBytes(-1), std::invalid_argument);
        EXPECT_THROW((void)absent.localPayloadBytes(1), std::invalid_argument);
        const auto present = HybridGDNStateGeometry::resolve(4, 0, 4, 2, 4, 2, 8, 4);
        EXPECT_EQ(present.localPayloadBytes(0), 0u);
        EXPECT_GT(present.localPayloadBytes(1), 0u);
    }

    TEST(Test__HybridGDNStateGeometry,
         UnshardedArenaDoesNotPriceAnImaginaryPrimaryBank)
    {
        const HybridGDNStateGeometry geometry =
            HybridGDNStateGeometry::resolve(
                /*attention_heads=*/4,
                /*local_head_start=*/0,
                /*local_attention_heads=*/0,
                /*group_count=*/2,
                /*time_step_rank=*/4,
                /*state_size=*/2,
                /*inner_size=*/8,
                /*conv_kernel_size=*/3);

        EXPECT_TRUE(geometry.hasSingleBankGeometry());
        EXPECT_EQ(geometry.full_qkv_dim, 16);
        EXPECT_EQ(geometry.local_qkv_dim, 16);
        EXPECT_EQ(geometry.local_conv_state_floats, 32);
        EXPECT_EQ(geometry.local_recurrence_state_floats, 16);

        /*
         * One 128-byte convolution request bank begins at zero. The 64-byte
         * recurrence request bank begins at the next 256-byte boundary.
         */
        EXPECT_EQ(
            geometry.deviceArenaBytes(
                /*gdn_layers=*/1,
                /*request_capacity=*/1),
            320u);
        EXPECT_EQ(geometry.deviceSerializedPayloadBytes(1), 192u);
        // Four heads: phase + 2x2 checkpoint + eight (2+2+2) raw records.
        EXPECT_EQ(geometry.cpuLocalRecurrenceStateFloats(), 212);
        EXPECT_EQ(geometry.localPayloadBytes(1), 976u);
    }

    /** @brief Modulo-linked K/V sharding uses exact weight/runtime ownership. */
    TEST(Test__HybridGDNStateGeometry,
         TensorParallelPartitionKeepsLinkedHeadCountsAndBytesExact)
    {
        const HybridGDNStateGeometry geometry =
            HybridGDNStateGeometry::resolve(
                /*attention_heads=*/64,
                /*local_head_start=*/16,
                /*local_attention_heads=*/16,
                /*group_count=*/16,
                /*time_step_rank=*/32,
                /*state_size=*/128,
                /*inner_size=*/4096,
                /*conv_kernel_size=*/4);

        EXPECT_EQ(geometry.local_key_heads, 4);
        EXPECT_EQ(geometry.local_value_heads, 8);
        EXPECT_EQ(geometry.local_qkv_dim, 2048);
        EXPECT_EQ(geometry.full_qkv_dim, 8192);
        EXPECT_EQ(geometry.local_conv_state_floats, 6144);
        EXPECT_EQ(geometry.full_conv_state_floats, 24576);
        EXPECT_EQ(geometry.local_recurrence_state_floats, 131072);
        EXPECT_EQ(geometry.full_recurrence_state_floats, 524288);
        EXPECT_FALSE(geometry.hasSingleBankGeometry());

        const size_t one_layer = geometry.deviceArenaBytes(1, 2);
        EXPECT_EQ(geometry.deviceArenaBytes(3, 2), 3u * one_layer);
        EXPECT_GT(one_layer, geometry.localPayloadBytes(1));
    }

    /** @brief Non-integral TP boundaries fail before planning or allocation. */
    TEST(Test__HybridGDNStateGeometry,
         RejectsAHeadPartitionThatWouldRequireRounding)
    {
        EXPECT_THROW(
            (void)HybridGDNStateGeometry::resolve(
                /*attention_heads=*/64,
                /*local_head_start=*/1,
                /*local_attention_heads=*/16,
                /*group_count=*/16,
                /*time_step_rank=*/32,
                /*state_size=*/128,
                /*inner_size=*/4096,
                /*conv_kernel_size=*/4),
            std::invalid_argument);
    }

    // =============================================================================
    // Test: CPUHybridRingKVCache — IHybridKVCache interface
    // =============================================================================

    class Test__CPUHybridKVCache : public ::testing::Test
    {
    protected:
        static constexpr int N_LAYERS = 24;
        static constexpr int BATCH_SIZE = 1;
        static constexpr int MAX_SEQ_LEN = 128;
        static constexpr int N_KV_HEADS = 4;
        static constexpr int HEAD_DIM = 64;

        std::unique_ptr<CPUHybridRingKVCacheFP32> createCache()
        {
            auto config = makeQwen35_08B_Config();
            return std::make_unique<CPUHybridRingKVCacheFP32>(
                config, getTestMPIContext(), N_LAYERS, BATCH_SIZE,
                MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM);
        }
    };

    TEST_F(Test__CPUHybridKVCache, ReportsCorrectTotalLayers)
    {
        auto cache = createCache();
        EXPECT_EQ(cache->n_layers(), 24);
        EXPECT_EQ(cache->num_layers(), 24);
    }

    TEST_F(Test__CPUHybridKVCache, IsGDNLayer_Correct)
    {
        auto cache = createCache();

        // FA layers: 3, 7, 11, 15, 19, 23
        for (int fa : {3, 7, 11, 15, 19, 23})
        {
            EXPECT_FALSE(cache->isGDNLayer(fa)) << "Layer " << fa;
            EXPECT_TRUE(cache->isFullAttentionLayer(fa)) << "Layer " << fa;
        }

        // GDN layers: 0, 1, 2, 4, 5, 6, ...
        for (int gdn : {0, 1, 2, 4, 5, 6, 8, 9, 10})
        {
            EXPECT_TRUE(cache->isGDNLayer(gdn)) << "Layer " << gdn;
            EXPECT_FALSE(cache->isFullAttentionLayer(gdn)) << "Layer " << gdn;
        }
    }

    TEST_F(Test__CPUHybridKVCache, GetGDNState_NonNullForGDN_NullForFA)
    {
        auto cache = createCache();

        // GDN layers should have state
        for (int gdn : {0, 1, 2, 4, 5, 6})
        {
            EXPECT_NE(cache->getGDNState(gdn), nullptr) << "Layer " << gdn;
            EXPECT_NE(cache->getRecurrenceState(gdn), nullptr) << "Layer " << gdn;
            EXPECT_NE(cache->getConvState(gdn), nullptr) << "Layer " << gdn;
        }

        // FA layers should NOT have GDN state
        for (int fa : {3, 7, 11, 15, 19, 23})
        {
            EXPECT_EQ(cache->getGDNState(fa), nullptr) << "Layer " << fa;
            EXPECT_EQ(cache->getRecurrenceState(fa), nullptr) << "Layer " << fa;
            EXPECT_EQ(cache->getConvState(fa), nullptr) << "Layer " << fa;
        }
    }

    TEST_F(Test__CPUHybridKVCache, GDNStateInitializedToZero)
    {
        auto cache = createCache();

        auto *state = cache->getGDNState(0); // First GDN layer
        ASSERT_NE(state, nullptr);

        // Recurrence state should be all zeros
        for (float val : state->recurrence_state)
        {
            EXPECT_EQ(val, 0.0f);
        }

        // Conv state should be all zeros
        for (float val : state->conv_state)
        {
            EXPECT_EQ(val, 0.0f);
        }
    }

    TEST_F(Test__CPUHybridKVCache, GDNState_Dimensions)
    {
        auto cache = createCache();

        auto *state = cache->getGDNState(0);
        ASSERT_NE(state, nullptr);

        EXPECT_EQ(state->n_k_heads, 4);
        EXPECT_EQ(state->n_v_heads, 16);
        EXPECT_EQ(state->d_k, 64);
        EXPECT_EQ(state->d_v, 64);
        EXPECT_EQ(state->conv_kernel_size, 4);

        // Sixteen complete heads: phase + 64x64 checkpoint + eight 130-float records.
        EXPECT_EQ(state->recurrence_state.size(), 82192u);

        // Conv state: qkv_dim * (conv_kernel - 1)
        // qkv_dim = 2 * n_k_heads * d_k + n_v_heads * d_v = 2*4*64 + 16*64 = 512 + 1024 = 1536
        // conv state = 1536 * 3 = 4608
        EXPECT_EQ(state->conv_state.size(), 4608u);
    }

    TEST_F(Test__CPUHybridKVCache, GetCachedTokens_ZeroForGDN)
    {
        auto cache = createCache();

        // GDN layers should report 0 cached tokens
        for (int gdn : {0, 1, 2, 4, 5, 6})
        {
            EXPECT_EQ(cache->get_cached_tokens(gdn), 0) << "Layer " << gdn;
        }

        // FA layers should also start at 0 (nothing appended yet)
        for (int fa : {3, 7, 11, 15, 19, 23})
        {
            EXPECT_EQ(cache->get_cached_tokens(fa), 0) << "Layer " << fa;
        }
    }

    TEST_F(Test__CPUHybridKVCache, AppendKV_NoopForGDN)
    {
        auto cache = createCache();

        // Create dummy K, V tensors (1 token, n_kv_heads * head_dim)
        const int kv_dim = N_KV_HEADS * HEAD_DIM; // 256
        auto k_tensor = TestTensorFactory::createFP32Zeros({1, static_cast<size_t>(kv_dim)});
        auto v_tensor = TestTensorFactory::createFP32Zeros({1, static_cast<size_t>(kv_dim)});

        // Append to a GDN layer — should no-op, return true
        EXPECT_TRUE(cache->append_kv(0, 0, k_tensor.get(), v_tensor.get()));
        EXPECT_EQ(cache->get_cached_tokens(0), 0); // Still 0 — GDN layer ignores append
    }

    TEST_F(Test__CPUHybridKVCache, GetKV_ReturnsFalseForGDN_TrueForFA)
    {
        auto cache = createCache();

        ITensor *k = nullptr, *v = nullptr;
        int kv_len = -1;

        // GDN layer should return false
        EXPECT_FALSE(cache->get_kv(0, 0, &k, &v, &kv_len));
        EXPECT_EQ(k, nullptr);
        EXPECT_EQ(v, nullptr);
        EXPECT_EQ(kv_len, 0);

        // FA layer should return true (K/V tensors exist even if empty)
        k = nullptr;
        v = nullptr;
        kv_len = -1;
        EXPECT_TRUE(cache->get_kv(3, 0, &k, &v, &kv_len));
        EXPECT_NE(k, nullptr);
        EXPECT_NE(v, nullptr);
        EXPECT_EQ(kv_len, 0); // Nothing appended yet
    }

    TEST_F(Test__CPUHybridKVCache, ClearResetsGDNState)
    {
        auto cache = createCache();

        // Modify GDN state
        auto *state = cache->getGDNState(0);
        ASSERT_NE(state, nullptr);
        state->recurrence_state[0] = 42.0f;
        state->conv_state[0] = 99.0f;

        // Clear should reset
        ASSERT_TRUE(cache->resetRequestState(
            IKVCache::StateResetContext::testReinitialization(nullptr)));

        EXPECT_EQ(state->recurrence_state[0], 0.0f);
        EXPECT_EQ(state->conv_state[0], 0.0f);
    }

    TEST_F(Test__CPUHybridKVCache, ClearLayerResetsGDN)
    {
        auto cache = createCache();

        auto *state = cache->getGDNState(0);
        ASSERT_NE(state, nullptr);
        state->recurrence_state[0] = 42.0f;

        ASSERT_TRUE(cache->resetLayerState(
            0,
            IKVCache::StateResetContext::testReinitialization(nullptr)));

        EXPECT_EQ(state->recurrence_state[0], 0.0f);
    }

    TEST_F(Test__CPUHybridKVCache, KVLayerAndGDNLayerCounts)
    {
        auto cache = createCache();

        EXPECT_EQ(cache->kvLayerCount(), 6);
        EXPECT_EQ(cache->gdnLayerCount(), 18);
    }

    TEST_F(Test__CPUHybridKVCache, GDNMemoryBytesNonZero)
    {
        auto cache = createCache();

        // 18 GDN layers, each with recurrence + conv state
        EXPECT_GT(cache->gdnMemoryBytes(), 0u);
    }

    TEST_F(Test__CPUHybridKVCache, FALayersHaveIndependentKVTensors)
    {
        auto cache = createCache();

        // FA layers at different global indices should map to different KV indices
        // and have independent K/V tensor pointers
        ITensor *k3 = nullptr, *v3 = nullptr;
        ITensor *k7 = nullptr, *v7 = nullptr;

        EXPECT_TRUE(cache->get_kv(3, 0, &k3, &v3));
        EXPECT_TRUE(cache->get_kv(7, 0, &k7, &v7));

        EXPECT_NE(k3, nullptr);
        EXPECT_NE(k7, nullptr);
        EXPECT_NE(k3, k7) << "Different FA layers should have different K tensors";
        EXPECT_NE(v3, v7) << "Different FA layers should have different V tensors";
    }

    TEST_F(Test__CPUHybridKVCache, LogicalBlockIOUsesGlobalFALayerIds)
    {
        auto cache = createCache();

        const int kv_dim = N_KV_HEADS * HEAD_DIM;
        auto k_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        auto v_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        for (int i = 0; i < 2 * kv_dim; ++i)
        {
            k_tensor->mutable_data()[i] = static_cast<float>(1000 + i);
            v_tensor->mutable_data()[i] = static_cast<float>(2000 + i);
        }

        ASSERT_TRUE(cache->append_kv(3, 0, k_tensor.get(), v_tensor.get(), 2));
        EXPECT_EQ(cache->get_cached_tokens(3), 2);
        EXPECT_EQ(cache->get_cached_tokens(0), 0);

        const auto gdn_layout = cache->logicalBlockLayout(0, 2);
        EXPECT_EQ(gdn_layout.k_bytes, 0u);
        EXPECT_EQ(gdn_layout.v_bytes, 0u);

        const auto fa_layout = cache->logicalBlockLayout(3, 2);
        ASSERT_GT(fa_layout.k_bytes, 0u);
        ASSERT_GT(fa_layout.v_bytes, 0u);

        std::vector<uint8_t> k_payload(fa_layout.k_bytes);
        std::vector<uint8_t> v_payload(fa_layout.v_bytes);
        IKVCache::KVCacheLogicalBlockDescriptor desc;
        desc.layer = 3;
        desc.seq_idx = 0;
        desc.logical_token_start = 0;
        desc.token_count = 2;
        ASSERT_TRUE(cache->exportLogicalBlock(desc, k_payload.data(), v_payload.data()));

        auto restored = createCache();
        ASSERT_TRUE(restored->importLogicalBlock(desc, k_payload.data(), v_payload.data()));
        EXPECT_EQ(restored->get_cached_tokens(3), 2);
        EXPECT_EQ(restored->get_cached_tokens(0), 0);

        std::vector<uint8_t> restored_k(fa_layout.k_bytes);
        std::vector<uint8_t> restored_v(fa_layout.v_bytes);
        ASSERT_TRUE(restored->exportLogicalBlock(desc, restored_k.data(), restored_v.data()));
        EXPECT_EQ(restored_k, k_payload);
        EXPECT_EQ(restored_v, v_payload);
    }

    TEST_F(Test__CPUHybridKVCache, PrefixCachedTokensProbeUsesFirstFullAttentionLayer)
    {
        auto cache = createCache();

        const int kv_dim = N_KV_HEADS * HEAD_DIM;
        auto k_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        auto v_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        ASSERT_TRUE(cache->append_kv(3, 0, k_tensor.get(), v_tensor.get(), 2));

        EXPECT_EQ(cache->get_cached_tokens(cache->first_layer_index(), 0), 0);
        EXPECT_EQ(firstRestorablePrefixLayer(*cache), 3);
        EXPECT_EQ(restorablePrefixCachedTokens(*cache, 0), 2);
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateMetadataCountsHostBytes)
    {
        auto cache = createCache();

        const HybridPrefixStateMetadata metadata = cache->hybridPrefixStateMetadata();
        EXPECT_EQ(metadata.total_layers, N_LAYERS);
        EXPECT_EQ(metadata.gdn_layers, 18);
        EXPECT_EQ(metadata.host_bytes, cache->gdnMemoryBytes());
        EXPECT_GT(metadata.host_bytes, 0u);
        EXPECT_EQ(metadata.device_bytes, 0u);
        EXPECT_FALSE(metadata.has_device_kernel_state);
        EXPECT_EQ(metadata.host_encoding, HybridPrefixHostEncoding::CPUFixedEpochPackedV2);
        EXPECT_GT(metadata.host_serialized_capacity_bytes, 0u);
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateDescriptorDefaultsToSynchronousCompletion)
    {
        HybridPrefixStateDescriptor desc;
        EXPECT_TRUE(desc.synchronize);
        EXPECT_TRUE(desc.include_host_state);
        EXPECT_TRUE(desc.include_device_state);
        EXPECT_EQ(desc.purpose, HybridPrefixStatePurpose::DurablePrefix);
        EXPECT_EQ(desc.host_encoding, HybridPrefixHostEncoding::None);
        EXPECT_EQ(desc.host_payload_bytes, 0u);
    }

    TEST_F(Test__CPUHybridKVCache, PrefixPayloadLayoutCountsFALayersAndHybridState)
    {
        auto cache = createCache();

        const PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
            *cache,
            DeviceId::cpu(),
            2);

        EXPECT_EQ(layout.total_layers, N_LAYERS);
        EXPECT_EQ(layout.fa_layers, 6);
        EXPECT_EQ(layout.gdn_layers, 18);
        EXPECT_GT(layout.bytes_per_fa_layer_k, 0u);
        EXPECT_GT(layout.bytes_per_fa_layer_v, 0u);
        EXPECT_TRUE(layout.includes_hybrid_state);
        EXPECT_EQ(layout.hybrid_host_state_bytes, cache->hybridPrefixStateMetadata().host_serialized_capacity_bytes);
        EXPECT_EQ(layout.hybrid_host_state_capacity_bytes, layout.hybrid_host_state_bytes);
        EXPECT_EQ(layout.hybrid_host_encoding, HybridPrefixHostEncoding::CPUFixedEpochPackedV2);
        EXPECT_EQ(layout.hybrid_device_state_bytes, 0u);
        EXPECT_EQ(layout.hybrid_state_bytes, layout.hybrid_host_state_bytes);
        EXPECT_EQ(layout.totalBytes(), layout.faKVBytes() + layout.hybrid_host_state_bytes);
    }

    TEST_F(Test__CPUHybridKVCache, PrefixPayloadLayoutAndLogicalIOUsePPStageGlobalLayerIds)
    {
        constexpr int FIRST_LAYER = 8;
        constexpr int STAGE_LAYERS = 8;
        auto stage_config = makeQwen35_08B_StageConfig(FIRST_LAYER, STAGE_LAYERS);
        auto cache = std::make_unique<CPUHybridRingKVCacheFP32>(
            stage_config, getTestMPIContext(), STAGE_LAYERS, BATCH_SIZE,
            MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM);

        EXPECT_EQ(cache->first_layer_index(), FIRST_LAYER);
        EXPECT_TRUE(cache->isFullAttentionLayer(3));
        EXPECT_TRUE(cache->isFullAttentionLayer(11));
        EXPECT_FALSE(cache->isFullAttentionLayer(8));
        EXPECT_EQ(firstRestorablePrefixLayer(*cache), 11);

        const PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
            *cache,
            DeviceId::cpu(),
            2);
        EXPECT_EQ(layout.first_layer_index, FIRST_LAYER);
        EXPECT_EQ(layout.total_layers, STAGE_LAYERS);
        EXPECT_EQ(layout.fa_layers, 2);
        ASSERT_GT(layout.bytes_per_fa_layer_k, 0u);
        ASSERT_GT(layout.bytes_per_fa_layer_v, 0u);

        const int kv_dim = N_KV_HEADS * HEAD_DIM;
        auto k_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        auto v_tensor = TestTensorFactory::createFP32Zeros({2, static_cast<size_t>(kv_dim)});
        for (int i = 0; i < 2 * kv_dim; ++i)
        {
            k_tensor->mutable_data()[i] = static_cast<float>(3000 + i);
            v_tensor->mutable_data()[i] = static_cast<float>(4000 + i);
        }

        ASSERT_TRUE(cache->append_kv(11, 0, k_tensor.get(), v_tensor.get(), 2));
        EXPECT_EQ(cache->get_cached_tokens(11), 2);
        EXPECT_EQ(cache->get_cached_tokens(3), 2);
        EXPECT_EQ(restorablePrefixCachedTokens(*cache, 0), 2);

        const auto local_layout = cache->logicalBlockLayout(3, 2);
        const auto global_layout = cache->logicalBlockLayout(11, 2);
        EXPECT_EQ(global_layout.k_bytes, local_layout.k_bytes);
        EXPECT_EQ(global_layout.v_bytes, local_layout.v_bytes);
        ASSERT_GT(global_layout.k_bytes, 0u);
        ASSERT_GT(global_layout.v_bytes, 0u);

        std::vector<uint8_t> k_payload(global_layout.k_bytes);
        std::vector<uint8_t> v_payload(global_layout.v_bytes);
        IKVCache::KVCacheLogicalBlockDescriptor desc;
        desc.layer = 11;
        desc.seq_idx = 0;
        desc.logical_token_start = 0;
        desc.token_count = 2;
        ASSERT_TRUE(cache->exportLogicalBlock(desc, k_payload.data(), v_payload.data()));

        auto restored = std::make_unique<CPUHybridRingKVCacheFP32>(
            stage_config, getTestMPIContext(), STAGE_LAYERS, BATCH_SIZE,
            MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM);
        ASSERT_TRUE(restored->importLogicalBlock(desc, k_payload.data(), v_payload.data()));
        EXPECT_EQ(restored->sequenceState(11, 0).cached_tokens, 2);

        std::vector<uint8_t> restored_k(global_layout.k_bytes);
        std::vector<uint8_t> restored_v(global_layout.v_bytes);
        ASSERT_TRUE(restored->exportLogicalBlock(desc, restored_k.data(), restored_v.data()));
        EXPECT_EQ(restored_k, k_payload);
        EXPECT_EQ(restored_v, v_payload);
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateRoundTripUsesLocalLayerIndicesForStageCache)
    {
        constexpr int FIRST_LAYER = 1;
        constexpr int STAGE_LAYERS = 8;
        auto stage_config = makeQwen35_08B_StageConfig(FIRST_LAYER, STAGE_LAYERS);
        auto cache = std::make_unique<CPUHybridRingKVCacheFP32>(
            stage_config, getTestMPIContext(), STAGE_LAYERS, BATCH_SIZE,
            MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM);
        auto restored = std::make_unique<CPUHybridRingKVCacheFP32>(
            stage_config, getTestMPIContext(), STAGE_LAYERS, BATCH_SIZE,
            MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM);

        std::vector<int> gdn_layers;
        std::vector<std::vector<float>> expected_recurrence;
        std::vector<std::vector<float>> expected_conv;
        for (int global_layer = FIRST_LAYER; global_layer < FIRST_LAYER + STAGE_LAYERS; ++global_layer)
        {
            auto *state = cache->getGDNState(global_layer);
            if (!state)
                continue;

            for (size_t i = 0; i < state->recurrence_state.size(); ++i)
            {
                state->recurrence_state[i] =
                    static_cast<float>(global_layer * 10000 + static_cast<int>(i % 251));
            }
            for (size_t i = 0; i < state->conv_state.size(); ++i)
            {
                state->conv_state[i] =
                    static_cast<float>(global_layer * 20000 + static_cast<int>(i % 127));
            }
            gdn_layers.push_back(global_layer);
            expected_recurrence.push_back(state->recurrence_state);
            expected_conv.push_back(state->conv_state);
        }
        ASSERT_GT(gdn_layers.size(), 2u);

        const HybridPrefixStateMetadata metadata = cache->hybridPrefixStateMetadata();
        ASSERT_EQ(metadata.host_bytes, cache->gdnMemoryBytes());
        ASSERT_GT(metadata.host_bytes, 0u);

        constexpr uint8_t kGuard = 0xCD;
        std::vector<uint8_t> payload(metadata.host_bytes + 4096u, kGuard);
        HybridPrefixStateDescriptor desc;
        desc.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        desc.host_payload_bytes = metadata.host_bytes;
        desc.seq_idx = 0;
        desc.logical_token_count = 13;
        ASSERT_TRUE(cache->exportHybridPrefixState(desc, payload.data(), nullptr));
        for (size_t i = metadata.host_bytes; i < payload.size(); ++i)
        {
            ASSERT_EQ(payload[i], kGuard)
                << "hybrid prefix export wrote past metadata.host_bytes at guard offset "
                << (i - metadata.host_bytes);
        }

        ASSERT_TRUE(restored->importHybridPrefixState(desc, payload.data(), nullptr));
        for (size_t i = 0; i < gdn_layers.size(); ++i)
        {
            auto *state = restored->getGDNState(gdn_layers[i]);
            ASSERT_NE(state, nullptr) << "missing restored GDN layer " << gdn_layers[i];
            EXPECT_EQ(state->recurrence_state, expected_recurrence[i])
                << "recurrence state mismatch for global layer " << gdn_layers[i];
            EXPECT_EQ(state->conv_state, expected_conv[i])
                << "conv state mismatch for global layer " << gdn_layers[i];
        }
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateRoundTripRestoresGDNState)
    {
        auto cache = createCache();

        auto *layer0 = cache->getGDNState(0);
        auto *layer4 = cache->getGDNState(4);
        ASSERT_NE(layer0, nullptr);
        ASSERT_NE(layer4, nullptr);
        ASSERT_FALSE(layer0->recurrence_state.empty());
        ASSERT_FALSE(layer0->conv_state.empty());
        ASSERT_FALSE(layer4->recurrence_state.empty());
        ASSERT_FALSE(layer4->conv_state.empty());

        layer0->recurrence_state[0] = 11.0f;
        layer0->recurrence_state[1] = 12.0f;
        layer0->conv_state[0] = 13.0f;
        layer4->recurrence_state[0] = 41.0f;
        layer4->conv_state[0] = 43.0f;

        const HybridPrefixStateMetadata metadata = cache->hybridPrefixStateMetadata();
        std::vector<uint8_t> payload(metadata.host_bytes);
        HybridPrefixStateDescriptor desc;
        desc.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        desc.host_payload_bytes = metadata.host_bytes;
        desc.seq_idx = 0;
        desc.logical_token_count = 7;
        ASSERT_TRUE(cache->exportHybridPrefixState(desc, payload.data(), nullptr));

        const float *payload_floats = reinterpret_cast<const float *>(payload.data());
        EXPECT_FLOAT_EQ(payload_floats[0], 11.0f);
        EXPECT_FLOAT_EQ(payload_floats[1], 12.0f);
        const size_t layer0_conv_offset = layer0->recurrence_state.size();
        EXPECT_FLOAT_EQ(payload_floats[layer0_conv_offset], 13.0f);
        const size_t floats_per_gdn_layer = layer0->recurrence_state.size() + layer0->conv_state.size();
        const size_t layer4_offset = 3 * floats_per_gdn_layer;
        EXPECT_FLOAT_EQ(payload_floats[layer4_offset], 41.0f);
        EXPECT_FLOAT_EQ(payload_floats[layer4_offset + layer4->recurrence_state.size()], 43.0f);

        ASSERT_TRUE(cache->resetRequestState(
            IKVCache::StateResetContext::testReinitialization(nullptr)));
        EXPECT_FLOAT_EQ(layer0->recurrence_state[0], 0.0f);
        EXPECT_FLOAT_EQ(layer0->conv_state[0], 0.0f);
        EXPECT_FLOAT_EQ(layer4->recurrence_state[0], 0.0f);
        EXPECT_FLOAT_EQ(layer4->conv_state[0], 0.0f);

        ASSERT_TRUE(cache->importHybridPrefixState(desc, payload.data(), nullptr));
        EXPECT_FLOAT_EQ(layer0->recurrence_state[0], 11.0f);
        EXPECT_FLOAT_EQ(layer0->recurrence_state[1], 12.0f);
        EXPECT_FLOAT_EQ(layer0->conv_state[0], 13.0f);
        EXPECT_FLOAT_EQ(layer4->recurrence_state[0], 41.0f);
        EXPECT_FLOAT_EQ(layer4->conv_state[0], 43.0f);
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateParallelHostCopyPreservesAllGDNState)
    {
        auto cache = createCache();
        auto restored = createCache();

        std::vector<std::vector<float>> expected_recurrence;
        std::vector<std::vector<float>> expected_conv;
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            auto *state = cache->getGDNState(layer);
            if (!state)
                continue;

            for (size_t i = 0; i < state->recurrence_state.size(); ++i)
            {
                state->recurrence_state[i] =
                    static_cast<float>(layer * 100000 + static_cast<int>(i % 997));
            }
            for (size_t i = 0; i < state->conv_state.size(); ++i)
            {
                state->conv_state[i] =
                    static_cast<float>(layer * 200000 + static_cast<int>(i % 389));
            }
            expected_recurrence.push_back(state->recurrence_state);
            expected_conv.push_back(state->conv_state);
        }

        const HybridPrefixStateMetadata metadata = cache->hybridPrefixStateMetadata();
        ASSERT_GT(metadata.host_bytes, 1u << 20)
            << "fixture should exercise the parallel host-state copy threshold";
        std::vector<uint8_t> payload(metadata.host_bytes);
        HybridPrefixStateDescriptor desc;
        desc.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        desc.host_payload_bytes = metadata.host_bytes;
        desc.seq_idx = 0;
        desc.logical_token_count = 11;
        ASSERT_TRUE(cache->exportHybridPrefixState(desc, payload.data(), nullptr));
        ASSERT_TRUE(restored->importHybridPrefixState(desc, payload.data(), nullptr));

        size_t gdn_index = 0;
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            auto *state = restored->getGDNState(layer);
            if (!state)
                continue;

            ASSERT_LT(gdn_index, expected_recurrence.size());
            EXPECT_EQ(state->recurrence_state, expected_recurrence[gdn_index])
                << "recurrence state mismatch for layer " << layer;
            EXPECT_EQ(state->conv_state, expected_conv[gdn_index])
                << "conv state mismatch for layer " << layer;
            ++gdn_index;
        }
        EXPECT_EQ(gdn_index, expected_recurrence.size());
    }

    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateRejectsMissingHostBuffers)
    {
        auto cache = createCache();

        const HybridPrefixStateMetadata metadata = cache->hybridPrefixStateMetadata();
        ASSERT_GT(metadata.host_bytes, 0u);

        HybridPrefixStateDescriptor desc;
        desc.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        desc.host_payload_bytes = metadata.host_bytes;
        desc.seq_idx = 0;
        desc.logical_token_count = 7;

        EXPECT_FALSE(cache->exportHybridPrefixState(desc, nullptr, nullptr));
        EXPECT_FALSE(cache->importHybridPrefixState(desc, nullptr, nullptr));

        std::vector<uint8_t> payload(metadata.host_bytes);
        desc.seq_idx = -1;
        EXPECT_FALSE(cache->exportHybridPrefixState(desc, payload.data(), nullptr));
        EXPECT_FALSE(cache->importHybridPrefixState(desc, payload.data(), nullptr));
    }

    // =============================================================================
    // REGRESSION TEST: createHybridKVCache must not throw for any supported precision
    //
    // This test catches the missing-break bug in switch/case blocks that caused
    // all precision cases to fall through to `default: throw "Unsupported precision"`.
    // =============================================================================

    class Test__KernelFactory_HybridKVCache : public ::testing::Test
    {
    };

    TEST_F(Test__KernelFactory_HybridKVCache, CreateCPUHybrid_AllPrecisions_NoThrow)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        for (auto precision : {
                 ActivationPrecision::FP32,
                 ActivationPrecision::BF16,
                 ActivationPrecision::FP16,
                 ActivationPrecision::Q8_1,
                 ActivationPrecision::Q16_1})
        {
            KVCacheConfig config;
            config.precision = precision;
            config.device = DeviceId::cpu();
            config.num_layers = 24;
            config.batch_size = 1;
            config.max_seq_len = 128;
            config.n_kv_heads = 4;
            config.head_dim = 64;
            config.mpi_ctx = &getTestMPIContext();
            config.hybrid_config = &hybrid_config;

            EXPECT_NO_THROW({
                auto cache = KernelFactory::createHybridKVCache(config);
                EXPECT_NE(cache, nullptr);
            }) << "Failed for precision: "
               << activationPrecisionToString(precision);
        }
    }

    TEST_F(Test__KernelFactory_HybridKVCache, CreateCPUHybrid_CorrectPrecision)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = &hybrid_config;

        auto cache = KernelFactory::createHybridKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->k_precision(), ActivationPrecision::FP32);
        EXPECT_EQ(cache->n_layers(), 24); // Total layers (not just FA)
    }

    TEST_F(Test__KernelFactory_HybridKVCache, CreateCPUHybrid_IHybridKVCacheInterface)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = &hybrid_config;

        auto cache = KernelFactory::createHybridKVCache(config);

        auto *hybrid = dynamic_cast<IHybridKVCache *>(cache.get());
        ASSERT_NE(hybrid, nullptr) << "Cache must implement IHybridKVCache";
        EXPECT_EQ(hybrid->kvLayerCount(), 6);
        EXPECT_EQ(hybrid->gdnLayerCount(), 18);
        EXPECT_TRUE(hybrid->isGDNLayer(0));
        EXPECT_FALSE(hybrid->isGDNLayer(3));
        EXPECT_NE(hybrid->getGDNState(0), nullptr);
        EXPECT_EQ(hybrid->getGDNState(3), nullptr);
    }

    TEST_F(Test__KernelFactory_HybridKVCache, HybridPrefixStateClearImportPreservesKernelObjects)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = &hybrid_config;

        auto cache = KernelFactory::createHybridKVCache(config);
        auto *hybrid = dynamic_cast<IHybridKVCache *>(cache.get());
        ASSERT_NE(hybrid, nullptr);

        auto *layer0 = hybrid->getGDNState(0);
        ASSERT_NE(layer0, nullptr);
        layer0->recurrence_state[0] = 77.0f;
        layer0->conv_state[0] = 88.0f;

        auto *conv0 = hybrid->getConvKernel(0);
        auto *rec0 = hybrid->getRecurrenceKernel(0);
        ASSERT_NE(conv0, nullptr);
        ASSERT_NE(rec0, nullptr);

        const HybridPrefixStateMetadata metadata = hybrid->hybridPrefixStateMetadata();
        std::vector<uint8_t> payload(metadata.host_bytes);
        HybridPrefixStateDescriptor desc;
        desc.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        desc.host_payload_bytes = metadata.host_bytes;
        desc.seq_idx = 0;
        desc.logical_token_count = 5;
        ASSERT_TRUE(hybrid->exportHybridPrefixState(desc, payload.data(), nullptr));

        ASSERT_TRUE(cache->resetRequestState(
            IKVCache::StateResetContext::testReinitialization(nullptr)));
        EXPECT_EQ(hybrid->getConvKernel(0), conv0);
        EXPECT_EQ(hybrid->getRecurrenceKernel(0), rec0);
        EXPECT_FLOAT_EQ(layer0->recurrence_state[0], 0.0f);
        EXPECT_FLOAT_EQ(layer0->conv_state[0], 0.0f);

        ASSERT_TRUE(hybrid->importHybridPrefixState(desc, payload.data(), nullptr));
        EXPECT_EQ(hybrid->getConvKernel(0), conv0);
        EXPECT_EQ(hybrid->getRecurrenceKernel(0), rec0);
        EXPECT_FLOAT_EQ(layer0->recurrence_state[0], 77.0f);
        EXPECT_FLOAT_EQ(layer0->conv_state[0], 88.0f);
    }

    TEST_F(Test__KernelFactory_HybridKVCache, CreateCPUHybrid_Sharded_AllPrecisions)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        for (auto precision : {
                 ActivationPrecision::FP32,
                 ActivationPrecision::BF16,
                 ActivationPrecision::FP16,
                 ActivationPrecision::Q8_1,
                 ActivationPrecision::Q16_1})
        {
            KVCacheConfig config;
            config.precision = precision;
            config.device = DeviceId::cpu();
            config.num_layers = 24;
            config.batch_size = 1;
            config.max_seq_len = 128;
            config.n_kv_heads = 8;
            config.local_n_kv_heads = 4;
            config.kv_head_start = 0;
            config.head_dim = 64;
            config.mpi_ctx = &getTestMPIContext();
            config.hybrid_config = &hybrid_config;

            EXPECT_NO_THROW({
                auto cache = KernelFactory::createHybridKVCache(config);
                EXPECT_NE(cache, nullptr);
            }) << "Failed for sharded precision: "
               << activationPrecisionToString(precision);
        }
    }

    TEST_F(Test__KernelFactory_HybridKVCache, CreateDispatchesToHybrid_WhenHybridConfigSet)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = &hybrid_config;

        EXPECT_TRUE(config.is_hybrid());

        // createKVCache should dispatch to createHybridKVCache
        auto cache = KernelFactory::createKVCache(config);
        ASSERT_NE(cache, nullptr);

        auto *hybrid = dynamic_cast<IHybridKVCache *>(cache.get());
        EXPECT_NE(hybrid, nullptr) << "createKVCache with hybrid_config should return IHybridKVCache";
    }

    TEST_F(Test__KernelFactory_HybridKVCache, NullHybridConfig_Throws)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = nullptr;

        EXPECT_THROW(KernelFactory::createHybridKVCache(config), std::runtime_error);
    }

    TEST_F(Test__KernelFactory_HybridKVCache, GDNKernelsCreatedForEachGDNLayer)
    {
        auto hybrid_config = makeQwen35_08B_Config();

        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();
        config.hybrid_config = &hybrid_config;

        auto cache = KernelFactory::createHybridKVCache(config);
        auto *hybrid = dynamic_cast<IHybridKVCache *>(cache.get());
        ASSERT_NE(hybrid, nullptr);

        // Each GDN layer should have both conv and recurrence kernels
        for (int i = 0; i < 24; ++i)
        {
            if (hybrid->isGDNLayer(i))
            {
                EXPECT_NE(hybrid->getConvKernel(i), nullptr)
                    << "GDN layer " << i << " missing conv kernel";
                EXPECT_NE(hybrid->getRecurrenceKernel(i), nullptr)
                    << "GDN layer " << i << " missing recurrence kernel";
            }
            else
            {
                EXPECT_EQ(hybrid->getConvKernel(i), nullptr)
                    << "FA layer " << i << " should not have conv kernel";
                EXPECT_EQ(hybrid->getRecurrenceKernel(i), nullptr)
                    << "FA layer " << i << " should not have recurrence kernel";
            }
        }
    }

    // =============================================================================
    // Test: HybridKVCacheConfig helper methods
    // =============================================================================

    TEST_F(Test__KernelFactory_HybridKVCache, KVCacheConfig_IsHybrid)
    {
        KVCacheConfig config;
        config.hybrid_config = nullptr;
        EXPECT_FALSE(config.is_hybrid());

        HybridKVCacheConfig hc;
        config.hybrid_config = &hc;
        EXPECT_TRUE(config.is_hybrid());
    }

    TEST_F(Test__KernelFactory_HybridKVCache, HybridKVCacheConfig_CountKVLayers)
    {
        auto config = makeQwen35_08B_Config();
        EXPECT_EQ(config.countKVLayers(), 6);
        EXPECT_TRUE(config.isHybrid());
    }

    TEST_F(Test__KernelFactory_HybridKVCache, EmptyConfig_NotHybrid)
    {
        HybridKVCacheConfig config;
        EXPECT_FALSE(config.isHybrid());
        EXPECT_EQ(config.countKVLayers(), 0);
    }

    /** @brief Format changes preserve byte ownership and reject unknown metadata. */
    TEST(Test__HybridGDNStateGeometry, RecurrenceLayoutPreservesOwnerFootprintAndRejectsUnknown)
    {
        using L = GDNRecurrenceStateLayout;
        const auto original = HybridGDNStateGeometry::resolve(4, 0, 0, 2, 4, 2, 8, 3);
        const auto transposed =
            HybridGDNStateGeometry::resolve(4, 0, 0, 2, 4, 2, 8, 3, L::ValueMajorKeyContiguous);
        auto config = makeQwen35_08B_Config();
        EXPECT_EQ(config.gdnStateGeometry().recurrence_layout, L::KeyMajorValueContiguous);
        config.recurrence_layout = L::ValueMajorKeyContiguous;
        EXPECT_EQ(config.gdnStateGeometry().recurrence_layout, L::ValueMajorKeyContiguous);
        EXPECT_EQ(original.recurrence_layout, L::KeyMajorValueContiguous);
        EXPECT_EQ(transposed.recurrence_layout, L::ValueMajorKeyContiguous);
        EXPECT_EQ(original.local_recurrence_state_floats, transposed.local_recurrence_state_floats);
        EXPECT_EQ(original.full_recurrence_state_floats, transposed.full_recurrence_state_floats);
        EXPECT_EQ(original.deviceSerializedPayloadBytes(3),
                  transposed.deviceSerializedPayloadBytes(3));
        std::array<float, 16> storage{};
        GDNDeviceStateBinding binding;
        binding.primary_state = storage.data();
        binding.primary_state_floats = 16;
        binding.request_state_bank = storage.data();
        binding.request_state_bank_floats = storage.size();
        binding.request_capacity = 1;
        ASSERT_TRUE(binding.valid());
        EXPECT_EQ(binding.recurrence_layout, L::KeyMajorValueContiguous);
        binding.recurrence_layout = L::ValueMajorKeyContiguous;
        EXPECT_TRUE(binding.valid()); // Metadata represents both implemented native orientations.
        binding.recurrence_layout = static_cast<L>(255);
        EXPECT_FALSE(binding.valid());
        EXPECT_THROW(HybridGDNStateGeometry::resolve(4, 0, 0, 2, 4, 2, 8, 3, static_cast<L>(255)),
                     std::invalid_argument);
    }

    /** @brief Same-sized foreign physical state must fail before any payload read/write. */
    TEST_F(Test__CPUHybridKVCache, HybridPrefixStateRejectsSameSizeForeignLayoutBeforeImport)
    {
        using L = GDNRecurrenceStateLayout;
        auto cache = createCache();
        auto *state = cache->getGDNState(0);
        ASSERT_NE(state, nullptr);
        ASSERT_FALSE(state->recurrence_state.empty());
        state->recurrence_state[0] = 0.125f;
        const auto original = state->recurrence_state;
        const auto metadata = cache->hybridPrefixStateMetadata();
        ASSERT_GT(metadata.host_bytes, 0u);
        HybridPrefixStateDescriptor request;
        request.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        request.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        request.host_payload_bytes = metadata.host_bytes;
        request.seq_idx = 0;
        request.include_device_state = false;
        request.host_recurrence_layout = L::ValueMajorKeyContiguous;
        EXPECT_FALSE(hybridPrefixStateLayoutsMatch(metadata, request));
        // This non-null deliberately unreadable address proves admission runs
        // before copying a payload whose capacity alone would look compatible.
        const auto *unreadable = reinterpret_cast<const void *>(std::uintptr_t{1});
        EXPECT_FALSE(cache->importHybridPrefixState(request, unreadable, nullptr));
        EXPECT_FALSE(
            cache->exportHybridPrefixState(request, const_cast<void *>(unreadable), nullptr));
        EXPECT_EQ(state->recurrence_state, original);
        request.host_recurrence_layout = L::KeyMajorValueContiguous;
        EXPECT_TRUE(hybridPrefixStateLayoutsMatch(metadata, request));
        request.host_recurrence_layout = static_cast<L>(255);
        EXPECT_FALSE(hybridPrefixStateLayoutsMatch(metadata, request));
        HybridPrefixStateMetadata device_metadata;
        device_metadata.device_bytes = metadata.host_bytes; // Identical physical capacity.
        device_metadata.device_recurrence_layout = L::ValueMajorKeyContiguous;
        request = {};
        request.include_host_state = false;
        EXPECT_FALSE(hybridPrefixStateLayoutsMatch(device_metadata, request));
        request.device_recurrence_layout = L::ValueMajorKeyContiguous;
        EXPECT_TRUE(hybridPrefixStateLayoutsMatch(device_metadata, request));
    }

    /** @brief Same-capacity speculative banks cannot be rebound using foreign inner axes. */
    TEST(Test__HybridGDNStateGeometry, SnapshotBindingsRejectForeignLayoutBeforeMutation)
    {
        CPUGatedDeltaNet kernel;
        using L = GDNRecurrenceStateLayout;
        float storage[6]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        EXPECT_EQ(kernel.recurrenceStateLayout(), L::KeyMajorValueContiguous);
        EXPECT_NO_THROW(
            kernel.bindRecurrenceStateCaptureWorkspace(storage, 1, 6, L::KeyMajorValueContiguous));
        EXPECT_NO_THROW(
            kernel.bindRecurrenceSpeculativeWorkspace(storage, 6, L::KeyMajorValueContiguous));
        EXPECT_THROW(
            kernel.bindRecurrenceStateCaptureWorkspace(storage, 1, 6, L::ValueMajorKeyContiguous),
            std::invalid_argument);
        EXPECT_THROW(kernel.bindRecurrenceSpeculativeWorkspace(storage, 6, static_cast<L>(255)),
                     std::invalid_argument);
        float restored[6]{};
        ASSERT_TRUE(kernel.restoreVerifierStateCaptureRow(restored, 0, nullptr));
        for (int i = 0; i < 6; ++i)
            EXPECT_EQ(restored[i], storage[i]);
        EXPECT_NO_THROW(
            kernel.bindRecurrenceStateCaptureWorkspace(nullptr, 0, 6, L::KeyMajorValueContiguous));
        EXPECT_NO_THROW(
            kernel.bindRecurrenceSpeculativeWorkspace(nullptr, 6, L::KeyMajorValueContiguous));
    }

    /** @brief Rectangular diagnostic values preserve every logical coordinate and signed-zero bit.
     */
    TEST(Test__HybridGDNStateGeometry, DiagnosticCoordinatesPreserveRectangularStateBytes)
    {
        using L = GDNRecurrenceStateLayout;
        std::vector<float> physical{
            0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f, -0.0f, 9.0f, 7.0f, 10.0f, 8.0f, 11.0f};
        const std::vector<float> expected{
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, -0.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
        canonicalizeRecurrenceValuesForPrefixProbe(physical, 2, 3, L::ValueMajorKeyContiguous);
        ASSERT_EQ(physical.size(), expected.size());
        EXPECT_EQ(std::memcmp(physical.data(), expected.data(), physical.size() * sizeof(float)),
                  0);
        canonicalizeRecurrenceValuesForPrefixProbe(physical, 2, 3, L::KeyMajorValueContiguous);
        EXPECT_EQ(std::memcmp(physical.data(), expected.data(), physical.size() * sizeof(float)),
                  0);
        EXPECT_THROW(
            canonicalizeRecurrenceValuesForPrefixProbe(physical, 2, 5, L::ValueMajorKeyContiguous),
            std::invalid_argument);
        EXPECT_THROW(
            canonicalizeRecurrenceValuesForPrefixProbe(physical, 2, 3, static_cast<L>(255)),
            std::invalid_argument);
    }

    /** @brief Construct the real admitted square-head CPU cache on a nonzero PP interval. */
    static std::unique_ptr<CPUHybridRingKVCacheFP32> packedEpochCache(int width)
    {
        HybridKVCacheConfig config;
        config.first_layer_index = 5;
        config.layer_types = {"gdn", "full_attention", "gdn"};
        config.gdn_conv_kernel_size = 4;
        config.gdn_state_size = width;
        config.gdn_group_count = config.gdn_time_step_rank = config.n_heads = 2;
        return std::make_unique<CPUHybridRingKVCacheFP32>(config, getTestMPIContext(),
            3, 1, 16, 2, width, DeviceId::cpu());
    }

    /**
     * @brief Seed original checkpoint/live history through actual scalar recurrence.
     * @param cache Original CPU state owner on the admitted PP interval.
     * @param incoming Base phase; heads intentionally retain unequal valid phases.
     * @param poison_inactive True authors defensive fixture-only dead-history poison.
     */
    static void seedPackedEpochCache(CPUHybridRingKVCacheFP32 &cache, int incoming, bool poison_inactive = true)
    {
        for (int layer : {5, 7})
        {
            auto *state = cache.getGDNState(layer);
            if (!state) throw std::logic_error("Missing actual CPU prefix state owner");
            const CPUGDNStateLayout one(1, state->d_k, state->d_v);
            const CPUGDNStateLayout full(state->n_v_heads, state->d_k, state->d_v);
            for (int head = 0; head < state->n_v_heads; ++head)
            {
                CPUGatedDeltaNet kernel;
                std::vector<float> bank(one.floats(), 0.0f), q(state->d_k), k(state->d_k),
                    v(state->d_v), out(state->d_v);
                const float alpha = 0.13f, beta = -0.21f, a = -0.3f, bias = 0.07f;
                const int phase = (incoming + head * 3 + layer - 5) % CPUGDNStateLayout::kPeriod;
                for (int row = 0; row < 8 + phase; ++row)
                {
                    for (int key = 0; key < state->d_k; ++key)
                    {
                        q[key] = std::sin(float(key * 3 + row + head + 1)) * 0.2f;
                        k[key] = std::cos(float(key + row * 2 + layer)) * 0.3f;
                    }
                    for (int value = 0; value < state->d_v; ++value)
                        v[value] = float((value * 7 + row + layer) % 13 - 6) / 32.0f;
                    if (!kernel.recurrent_step(q.data(), k.data(), v.data(), &alpha, &beta, &a, &bias,
                            out.data(), bank.data(), 1, state->d_k, state->d_v, true))
                        throw std::runtime_error("Actual scalar CPU epoch seed failed");
                }
                // Dead history is deliberately dirty. Durable wire must omit it;
                // internal rollback must retain those exact original bytes.
                if (poison_inactive)
                    std::fill(bank.begin() + one.packedHeadBytes(phase) / sizeof(float), bank.end(), -777.25f);
                std::memcpy(state->recurrence_state.data() + static_cast<size_t>(head) * full.headFloats(),
                            bank.data(), bank.size() * sizeof(float));
            }
            for (size_t value = 0; value < state->conv_state.size(); ++value)
                state->conv_state[value] = float(static_cast<int>((value + layer) % 19) - 9) / 64.0f;
        }
    }

    /** @return Immutable full physical bytes for an atomicity/local-image oracle only. */
    static std::vector<uint8_t> packedEpochPhysicalBytes(const CPUHybridRingKVCacheFP32 &cache)
    {
        std::vector<uint8_t> bytes;
        for (int layer : {5, 7})
        {
            const auto *state = cache.getGDNState(layer);
            for (const auto *bank : {&state->recurrence_state, &state->conv_state})
            {
                const auto *first = reinterpret_cast<const uint8_t *>(bank->data());
                bytes.insert(bytes.end(), first, first + bank->size() * sizeof(float));
            }
        }
        return bytes;
    }

    /**
     * @brief Compare exact live epoch spans while allowing only dead-history representation changes.
     * @param left Original complete state before durable packing.
     * @param right Restored complete bank with canonical inactive zeros.
     * @param layout Original immutable head geometry.
     * @return True for exact phase/checkpoint/live history, without materializing an epoch.
     */
    static bool packedEpochLogicalEqual(const std::vector<float> &left, const std::vector<float> &right,
                                         const CPUGDNStateLayout &layout)
    {
        if (left.size() != static_cast<size_t>(layout.floats()) || right.size() != left.size() ||
            !layout.valid(left.data()) || !layout.valid(right.data())) return false;
        for (int head = 0; head < layout.heads(); ++head)
        {
            const auto first = static_cast<size_t>(head) * layout.headFloats();
            if (std::bit_cast<uint32_t>(left[first]) != std::bit_cast<uint32_t>(right[first])) return false;
            const int phase = CPUGDNStateLayout::prefixPhase(std::bit_cast<uint32_t>(left[first]));
            if (std::memcmp(left.data() + first, right.data() + first, layout.packedHeadBytes(phase))) return false;
        }
        return true;
    }

    /** @brief Real cache round trips keep stable identity, compact live bytes and subsequent grouped output. */
    TEST(Test__CPUFixedEpochPrefix, PackedPhasesAndContinuationUseStableCapacity)
    {
        llaminar::v2::ThreadCountGuard threads(1);
        for (int width : {1, 3, 64, 128})
        {
            auto source = packedEpochCache(width), restored = packedEpochCache(width);
            const auto stable = buildDensePrefixPayloadLayout(*source, DeviceId::cpu(), 4);
            const auto metadata = source->hybridPrefixStateMetadata();
            size_t smallest = std::numeric_limits<size_t>::max(), largest = 0;
            for (int phase = 0; phase < CPUGDNStateLayout::kPeriod; ++phase)
            {
                SCOPED_TRACE(::testing::Message() << "DK=DV=" << width << " phase=" << phase);
                seedPackedEpochCache(*source, phase);
                EXPECT_EQ(buildDensePrefixPayloadLayout(*source, DeviceId::cpu(), 4), stable);
                EXPECT_EQ(source->hybridPrefixStateMetadata().host_bytes, metadata.host_bytes);
                EXPECT_EQ(source->hybridPrefixStateMetadata().host_serialized_capacity_bytes,
                          metadata.host_serialized_capacity_bytes);
                auto live = stable;
                applyHybridPrefixPayloadExtent(live, *source, 0);
                EXPECT_TRUE(live.compatiblePayloadShape(stable));
                EXPECT_TRUE(stable.compatiblePayloadShape(live));
                smallest = std::min(smallest, live.hybrid_host_state_bytes);
                largest = std::max(largest, live.hybrid_host_state_bytes);
                HybridPrefixStateDescriptor desc;
                desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
                desc.host_payload_bytes = live.hybrid_host_state_bytes;
                std::vector<uint8_t> wire(desc.host_payload_bytes + 16, 0x5a);
                ASSERT_TRUE(source->exportHybridPrefixState(desc, wire.data(), nullptr));
                for (size_t tail = desc.host_payload_bytes; tail < wire.size(); ++tail) EXPECT_EQ(wire[tail], 0x5a);
                ASSERT_TRUE(restored->importHybridPrefixState(desc, wire.data(), nullptr));
                for (int layer : {5, 7})
                {
                    const auto *original = source->getGDNState(layer);
                    const auto *imported = restored->getGDNState(layer);
                    const CPUGDNStateLayout layout(original->n_v_heads, original->d_k, original->d_v);
                    EXPECT_TRUE(packedEpochLogicalEqual(original->recurrence_state, imported->recurrence_state, layout));
                    EXPECT_EQ(std::memcmp(original->conv_state.data(), imported->conv_state.data(),
                                          original->conv_state.size() * sizeof(float)), 0);
                    for (int head = 0; head < layout.heads(); ++head)
                    {
                        const size_t first = static_cast<size_t>(head) * layout.headFloats();
                        const int incoming = CPUGDNStateLayout::prefixPhase(std::bit_cast<uint32_t>(imported->recurrence_state[first]));
                        for (size_t index = first + layout.packedHeadBytes(incoming) / sizeof(float);
                             index < first + layout.headFloats(); ++index)
                            EXPECT_EQ(std::bit_cast<uint32_t>(imported->recurrence_state[index]), 0u);
                    }
                    constexpr int rows = 3;
                    const int heads = layout.heads(), dk = layout.keyWidth(), dv = layout.valueWidth();
                    std::vector<float> q(rows * heads * dk), k(q.size()), v(rows * heads * dv),
                        alpha(rows * heads, 0.11f), beta(rows * heads, -0.2f), a(heads, -0.3f), bias(heads, 0.07f),
                        expected(v.size()), actual(v.size()), serial = original->recurrence_state,
                        grouped = imported->recurrence_state;
                    for (size_t i = 0; i < q.size(); ++i) { q[i] = float(static_cast<int>(i % 17) - 8) / 32.0f; k[i] = q[i] * 0.5f; }
                    for (size_t i = 0; i < v.size(); ++i) v[i] = float(static_cast<int>(i % 19) - 9) / 64.0f;
                    CPUGatedDeltaNet serial_kernel, grouped_kernel;
                    for (int row = 0; row < rows; ++row)
                        ASSERT_TRUE(serial_kernel.recurrent_step(q.data() + row * heads * dk, k.data() + row * heads * dk,
                            v.data() + row * heads * dv, alpha.data() + row * heads, beta.data() + row * heads,
                            a.data(), bias.data(), expected.data() + row * heads * dv, serial.data(), heads, dk, dv, true));
                    ASSERT_TRUE(grouped_kernel.chunk_forward(q.data(), k.data(), v.data(), alpha.data(), beta.data(),
                        a.data(), bias.data(), actual.data(), grouped.data(), rows, heads, dk, dv, 8, true));
                    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
                    EXPECT_TRUE(packedEpochLogicalEqual(serial, grouped, layout));
                }
            }
            EXPECT_LT(smallest, largest);
        }
    }

    /** @brief Every malformed durable source is rejected before any layer or convolution mutates. */
    TEST(Test__CPUFixedEpochPrefix, PackedImportIsBoundedAtomicAndRejectsForeignEncoding)
    {
        llaminar::v2::ThreadCountGuard threads(1);
        auto source = packedEpochCache(3), restored = packedEpochCache(3);
        seedPackedEpochCache(*source, 2);
        seedPackedEpochCache(*restored, 5);
        auto layout = buildDensePrefixPayloadLayout(*source, DeviceId::cpu(), 4);
        applyHybridPrefixPayloadExtent(layout, *source, 0);
        HybridPrefixStateDescriptor desc;
        desc.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
        desc.host_payload_bytes = layout.hybrid_host_state_bytes;
        std::vector<uint8_t> wire(desc.host_payload_bytes);
        ASSERT_TRUE(source->exportHybridPrefixState(desc, wire.data(), nullptr));
        const auto before = packedEpochPhysicalBytes(*restored);
        const auto reject = [&](const void *payload, HybridPrefixStateDescriptor request) {
            EXPECT_FALSE(restored->importHybridPrefixState(request, payload, nullptr));
            EXPECT_EQ(packedEpochPhysicalBytes(*restored), before);
        };
        for (auto encoding : {HybridPrefixHostEncoding::None, HybridPrefixHostEncoding::CPUFixedEpochLocalImage,
                              static_cast<HybridPrefixHostEncoding>(255)})
        {
            auto wrong = desc; wrong.host_encoding = encoding;
            reject(reinterpret_cast<const void *>(std::uintptr_t{1}), wrong);
        }
        for (size_t bytes : {size_t{0}, wire.size() - 1, wire.size() + 1, std::numeric_limits<size_t>::max()})
        { auto wrong = desc; wrong.host_payload_bytes = bytes; reject(wire.data(), wrong); }
        auto damaged = wire;
        CPUGDNPrefixSectionHeader section;
        std::memcpy(&section, wire.data(), sizeof(section));
        section.version = 1;
        std::memcpy(damaged.data(), &section, sizeof(section)); reject(damaged.data(), desc);
        damaged = wire;
        CPUGDNPrefixLayerHeader first;
        std::memcpy(&first, wire.data() + sizeof(section), sizeof(first));
        const size_t second = sizeof(section) + sizeof(first) + first.recurrence_bytes + first.conv_bytes;
        CPUGDNPrefixLayerHeader last;
        std::memcpy(&last, wire.data() + second, sizeof(last));
        ++last.key_width;
        std::memcpy(damaged.data() + second, &last, sizeof(last)); reject(damaged.data(), desc);
        damaged = wire;
        const uint32_t invalid_phase = std::bit_cast<uint32_t>(0.5f);
        std::memcpy(damaged.data() + second + sizeof(last), &invalid_phase, sizeof(invalid_phase));
        reject(damaged.data(), desc);
        damaged = wire;
        std::memcpy(&last, wire.data() + second, sizeof(last));
        last.recurrence_bytes = std::numeric_limits<uint64_t>::max();
        std::memcpy(damaged.data() + second, &last, sizeof(last)); reject(damaged.data(), desc);
        auto extra = wire; extra.resize(wire.size() + 4, 0);
        auto trailing = desc; trailing.host_payload_bytes = extra.size();
        std::memcpy(&section, extra.data(), sizeof(section)); section.bytes = extra.size();
        std::memcpy(extra.data(), &section, sizeof(section)); reject(extra.data(), trailing);
        reject(restored->getGDNState(5)->recurrence_state.data(), desc);
        reject(restored->getGDNState(7)->conv_state.data(), desc);
        EXPECT_FALSE(source->exportHybridPrefixState(desc, source->getGDNState(7)->conv_state.data(), nullptr));
        std::vector<uint8_t> poison(desc.host_payload_bytes, 0x6d);
        source->getGDNState(7)->recurrence_state[0] = 8.0f;
        EXPECT_FALSE(source->exportHybridPrefixState(desc, poison.data(), nullptr));
        EXPECT_TRUE(std::all_of(poison.begin(), poison.end(), [](uint8_t value) { return value == 0x6d; }));
    }

    /** @brief Local rollback is a bounded complete image, including deliberately dirty inactive slots. */
    TEST(Test__CPUFixedEpochPrefix, LocalRollbackImageRemainsByteExactAndNeverDurable)
    {
        llaminar::v2::ThreadCountGuard threads(1);
        auto source = packedEpochCache(3), restored = packedEpochCache(3);
        seedPackedEpochCache(*source, 6);
        const auto metadata = source->hybridPrefixStateMetadata();
        HybridPrefixStateDescriptor local;
        local.purpose = HybridPrefixStatePurpose::LocalCheckpoint;
        local.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochLocalImage;
        local.host_payload_bytes = metadata.host_bytes;
        std::vector<uint8_t> full(local.host_payload_bytes);
        const auto before = packedEpochPhysicalBytes(*source);
        ASSERT_TRUE(source->exportHybridPrefixState(local, full.data(), nullptr));
        EXPECT_EQ(full, before);
        ASSERT_TRUE(restored->importHybridPrefixState(local, full.data(), nullptr));
        EXPECT_EQ(packedEpochPhysicalBytes(*restored), before);
        const auto unchanged = packedEpochPhysicalBytes(*restored);
        auto durable = local; durable.purpose = HybridPrefixStatePurpose::DurablePrefix;
        EXPECT_FALSE(restored->importHybridPrefixState(durable, full.data(), nullptr));
        EXPECT_EQ(packedEpochPhysicalBytes(*restored), unchanged);
        auto packed_local = local; packed_local.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
        EXPECT_FALSE(restored->importHybridPrefixState(packed_local, full.data(), nullptr));
        auto truncated = local; --truncated.host_payload_bytes;
        EXPECT_FALSE(restored->importHybridPrefixState(truncated, full.data(), nullptr));
        EXPECT_EQ(packedEpochPhysicalBytes(*restored), unchanged);
    }

    /** @brief Rectangular kernel-storage spans remain checked without inventing rectangular cache admission. */
    TEST(Test__CPUFixedEpochPrefix, RectangularPackedHeadSpansAreChecked)
    {
        for (const auto [dk, dv] : {std::pair{3, 5}, std::pair{5, 3}, std::pair{1, 511}})
        {
            const CPUGDNStateLayout layout(2, dk, dv);
            for (int phase = 0; phase < 8; ++phase)
            {
                const int other = (phase + 3) % 8;
                std::vector<float> state(layout.floats(), -7.0f);
                state[0] = static_cast<float>(phase); state[layout.headFloats()] = static_cast<float>(other);
                const size_t expected = (2u * (1u + dk * dv) + (phase + other) * (dk + dv + 2u)) * sizeof(float);
                EXPECT_EQ(layout.packedPrefixBytes(state.data()), expected);
                std::vector<std::byte> wire(expected);
                const size_t first = layout.packedHeadBytes(phase);
                std::memcpy(wire.data(), state.data(), first);
                std::memcpy(wire.data() + first, state.data() + layout.headFloats(), expected - first);
                EXPECT_TRUE(layout.validPackedPrefix(wire));
                EXPECT_FALSE(layout.validPackedPrefix(std::span<const std::byte>(wire.data(), wire.size() - 1)));
                const uint32_t negative_zero = 0x80000000u;
                std::memcpy(wire.data() + first, &negative_zero, sizeof(negative_zero));
                EXPECT_FALSE(layout.validPackedPrefix(wire));
            }
        }
        EXPECT_THROW((void)CPUGDNStateLayout(std::numeric_limits<int>::max(), 512, 512), std::overflow_error);
    }

    /** @test Durable format identity is explicit and independent of live phase or local-image length. */
    TEST(Test__CPUFixedEpochPrefix, DurableFingerprintRejectsLocalAndDuplicateAuthorities)
    {
        std::vector<PrefixFingerprintField> fields;
        for (const auto encoding : {HybridPrefixHostEncoding::None,
             HybridPrefixHostEncoding::CPUFixedEpochLocalImage, static_cast<HybridPrefixHostEncoding>(77)})
            EXPECT_THROW(appendCPUHostPrefixEncodingFingerprintFields(fields, encoding), std::invalid_argument);
        EXPECT_TRUE(fields.empty());
        appendCPUHostPrefixEncodingFingerprintFields(fields, HybridPrefixHostEncoding::CPUFixedEpochPackedV2);
        ASSERT_EQ(fields.size(), 1u);
        EXPECT_EQ(fields[0].name, "cpu_gdn_host_prefix_encoding");
        EXPECT_EQ(fields[0].value, "cpu-fixed-epoch-packed-v2");
        EXPECT_THROW(appendCPUHostPrefixEncodingFingerprintFields(fields,
            HybridPrefixHostEncoding::CPUFixedEpochPackedV2), std::invalid_argument);
        EXPECT_EQ(fields.size(), 1u);
    }

    /**
     * @test Every real producer/publication preserves canonical inactive history.
     *
     * Dirty inactive bytes in the defensive packing tests are authored fixture
     * poison. Cache reset, ordinary/grouped recurrence, explicit dense oracle
     * snapshots and production compact accepted-state publication all retain
     * exact +0 tails, so the existing physical prefix probe remains byte-valid.
     */
    TEST(Test__CPUFixedEpochPrefix, ProductionCanonicalTailsKeepPhysicalProbeExact)
    {
        llaminar::v2::ThreadCountGuard threads(1);
        for (int width : {3, 64})
            for (int phase = 0; phase < 8; ++phase)
            {
                SCOPED_TRACE(::testing::Message() << "width=" << width << " phase=" << phase);
                auto cache = packedEpochCache(width), restored = packedEpochCache(width);
                seedPackedEpochCache(*cache, phase, false);
                const auto *source = cache->getGDNState(5);
                const CPUGDNStateLayout layout(source->n_v_heads, source->d_k, source->d_v);
                const auto canonical = [&](const float *bank) {
                    if (!layout.valid(bank)) return false;
                    for (int head = 0; head < layout.heads(); ++head)
                    {
                        const auto first = static_cast<size_t>(head) * layout.headFloats();
                        const auto p = CPUGDNStateLayout::prefixPhase(std::bit_cast<uint32_t>(bank[first]));
                        for (size_t i = first + layout.packedHeadBytes(p) / sizeof(float);
                             i < first + layout.headFloats(); ++i)
                            if (std::bit_cast<uint32_t>(bank[i]) != 0u) return false;
                    }
                    return true;
                };
                ASSERT_TRUE(canonical(source->recurrence_state.data()));
                auto wire_layout = buildDensePrefixPayloadLayout(*cache, DeviceId::cpu(), 4);
                applyHybridPrefixPayloadExtent(wire_layout, *cache, 0);
                HybridPrefixStateDescriptor descriptor;
                descriptor.host_encoding = HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
                descriptor.host_payload_bytes = wire_layout.hybrid_host_state_bytes;
                std::vector<uint8_t> wire(descriptor.host_payload_bytes);
                ASSERT_TRUE(cache->exportHybridPrefixState(descriptor, wire.data(), nullptr));
                ASSERT_TRUE(restored->importHybridPrefixState(descriptor, wire.data(), nullptr));
                EXPECT_EQ(packedEpochPhysicalBytes(*cache), packedEpochPhysicalBytes(*restored));
                constexpr int rows = 3;
                const int heads=layout.heads(), dk=layout.keyWidth(), dv=layout.valueWidth();
                std::vector<float> q(rows*heads*dk), k(q.size()), v(rows*heads*dv), alpha(rows*heads,0.12f),
                    beta(rows*heads,-0.2f), alog(heads,-0.3f), bias(heads,0.04f);
                for (size_t i=0;i<q.size();++i) { q[i]=float(static_cast<int>(i%13)-6)/32.0f; k[i]=q[i]*0.75f; }
                for (size_t i=0;i<v.size();++i) v[i]=float(static_cast<int>(i%17)-8)/64.0f;
                auto ordinary=source->recurrence_state, grouped=ordinary, dense=ordinary, compact=ordinary;
                const auto initial=ordinary;
                std::vector<float> serial_out(v.size()), grouped_out(v.size()), dense_out(v.size()), compact_out(v.size()),
                    serial_states(rows*layout.floats()), snapshots(serial_states.size(),-19.0f);
                CPUGatedDeltaNet serial_kernel, grouped_kernel, dense_kernel, compact_kernel;
                for (int row=0;row<rows;++row)
                {
                    ASSERT_TRUE(serial_kernel.recurrent_step(q.data()+row*heads*dk,k.data()+row*heads*dk,
                        v.data()+row*heads*dv,alpha.data()+row*heads,beta.data()+row*heads,alog.data(),bias.data(),
                        serial_out.data()+row*heads*dv,ordinary.data(),heads,dk,dv,true));
                    ASSERT_TRUE(canonical(ordinary.data()));
                    std::copy(ordinary.begin(),ordinary.end(),serial_states.begin()+row*layout.floats());
                }
                ASSERT_TRUE(grouped_kernel.chunk_forward(q.data(),k.data(),v.data(),alpha.data(),beta.data(),
                    alog.data(),bias.data(),grouped_out.data(),grouped.data(),rows,heads,dk,dv,8,true));
                ASSERT_TRUE(canonical(grouped.data()));
                EXPECT_EQ(std::memcmp(grouped.data(),ordinary.data(),ordinary.size()*sizeof(float)),0);
                ASSERT_TRUE(dense_kernel.chunkForwardWithStateSnapshots(q.data(),k.data(),v.data(),alpha.data(),beta.data(),
                    alog.data(),bias.data(),dense_out.data(),dense.data(),rows,heads,dk,dv,8,true,
                    snapshots.data(),layout.floats(),rows));
                EXPECT_EQ(std::memcmp(snapshots.data(),serial_states.data(),serial_states.size()*sizeof(float)),0);
                std::vector<float> slots(CPUSharedGDNLogicalCapacity(rows,1).slotFloats(layout),-23.0f);
                compact_kernel.bindVerifierStateCaptureWorkspace(slots.data(),rows,layout.floats());
                compact_kernel.bindCPUSharedStateCapture(slots.data(),slots.size(),rows);
                ASSERT_TRUE(compact_kernel.chunk_forward(q.data(),k.data(),v.data(),alpha.data(),beta.data(),
                    alog.data(),bias.data(),compact_out.data(),compact.data(),rows,heads,dk,dv,8,true));
                EXPECT_EQ(std::memcmp(compact.data(),initial.data(),initial.size()*sizeof(float)),0)
                    << "Candidates never mutate the canonical owner before publication";
                for (int row=0;row<rows;++row)
                {
                    ASSERT_TRUE(compact_kernel.restoreVerifierStateCaptureRow(compact.data(),row,nullptr));
                    ASSERT_TRUE(canonical(compact.data()));
                    EXPECT_EQ(std::memcmp(compact.data(),serial_states.data()+row*layout.floats(),
                                          compact.size()*sizeof(float)),0);
                }
                for (const auto *out : {&grouped_out,&dense_out,&compact_out})
                    EXPECT_EQ(std::memcmp(serial_out.data(),out->data(),out->size()*sizeof(float)),0);
            }
    }

} // namespace llaminar2::test
