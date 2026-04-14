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
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUHybridRingKVCache.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include "../../utils/TestTensorFactory.h"

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

        // Recurrence state: n_v_heads * d_k * d_v = 16 * 64 * 64 = 65536
        EXPECT_EQ(state->recurrence_state.size(), 65536u);

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
        cache->clear();

        EXPECT_EQ(state->recurrence_state[0], 0.0f);
        EXPECT_EQ(state->conv_state[0], 0.0f);
    }

    TEST_F(Test__CPUHybridKVCache, ClearLayerResetsGDN)
    {
        auto cache = createCache();

        auto *state = cache->getGDNState(0);
        ASSERT_NE(state, nullptr);
        state->recurrence_state[0] = 42.0f;

        cache->clear_layer(0);

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

} // namespace llaminar2::test
