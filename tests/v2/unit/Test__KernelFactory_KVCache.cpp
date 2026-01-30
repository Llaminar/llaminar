/**
 * @file Test__KernelFactory_KVCache.cpp
 * @brief Unit tests for KernelFactory KVCache creation methods
 * @author Generated from task specification
 * @date January 2026
 *
 * Tests the KernelFactory::createKVCache() and createCPUKVCache() methods
 * which provide a unified interface for KV cache creation across different
 * device types and precision modes.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/KernelFactory.h"
#include "../../../src/v2/kernels/cpu/CPUKVCache.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/backends/DeviceId.h"
#include "../../../src/v2/execution/config/RuntimeConfig.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;

namespace llaminar2::test
{

    // Single-rank MPI context for unit tests
    static const MPIContext &getTestMPIContext()
    {
        static MPIContext ctx(0, 1, MPI_COMM_WORLD);
        return ctx;
    }

    class Test__KernelFactory_KVCache : public ::testing::Test
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    // =============================================================================
    // Test: CPU KVCache Creation with FP32 precision
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_FP32)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::FP32);
        EXPECT_EQ(cache->num_layers(), 4);
        EXPECT_EQ(cache->batch_size(), 1);
        EXPECT_EQ(cache->max_seq_len(), 128);
        EXPECT_EQ(cache->n_kv_heads(), 4);
        EXPECT_EQ(cache->local_n_kv_heads(), 4); // Non-sharded: local == total
        EXPECT_FALSE(cache->is_sharded());
    }

    // =============================================================================
    // Test: CPU KVCache Creation with BF16 precision
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_BF16)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::BF16;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 2;
        config.max_seq_len = 256;
        config.n_kv_heads = 8;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);
        EXPECT_EQ(cache->num_layers(), 4);
        EXPECT_EQ(cache->batch_size(), 2);
        EXPECT_EQ(cache->max_seq_len(), 256);
        EXPECT_EQ(cache->n_kv_heads(), 8);
        EXPECT_FALSE(cache->is_sharded());
    }

    // =============================================================================
    // Test: CPU KVCache Creation with FP16 precision
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_FP16)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP16;
        config.device = DeviceId::cpu();
        config.num_layers = 6;
        config.batch_size = 1;
        config.max_seq_len = 512;
        config.n_kv_heads = 4;
        config.head_dim = 128;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);
        EXPECT_EQ(cache->num_layers(), 6);
        EXPECT_EQ(cache->batch_size(), 1);
        EXPECT_EQ(cache->max_seq_len(), 512);
        EXPECT_EQ(cache->n_kv_heads(), 4);
        EXPECT_FALSE(cache->is_sharded());
    }

    // =============================================================================
    // Test: CPU KVCache Creation with Q8_1 precision
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_Q8_1)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::Q8_1;
        config.device = DeviceId::cpu();
        config.num_layers = 24;
        config.batch_size = 1;
        config.max_seq_len = 2048;
        config.n_kv_heads = 2;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::Q8_1);
        EXPECT_EQ(cache->num_layers(), 24);
        EXPECT_EQ(cache->max_seq_len(), 2048);
        EXPECT_EQ(cache->n_kv_heads(), 2);
        EXPECT_FALSE(cache->is_sharded());
    }

    // =============================================================================
    // Test: CPU KVCache Creation with Q16_1 precision
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_Q16_1)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::Q16_1;
        config.device = DeviceId::cpu();
        config.num_layers = 12;
        config.batch_size = 1;
        config.max_seq_len = 1024;
        config.n_kv_heads = 4;
        config.head_dim = 64; // Q16_1 typically uses 64-element blocks
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::Q16_1);
        EXPECT_EQ(cache->num_layers(), 12);
        EXPECT_EQ(cache->max_seq_len(), 1024);
        EXPECT_EQ(cache->n_kv_heads(), 4);
        EXPECT_FALSE(cache->is_sharded());
    }

    // =============================================================================
    // Test: Sharded CPU KVCache Creation
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateShardedCPUKVCache)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 8;       // Total heads across all ranks
        config.local_n_kv_heads = 4; // This rank gets half the heads
        config.kv_head_start = 0;    // Starting from head 0
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_TRUE(cache->is_sharded());
        EXPECT_EQ(cache->n_kv_heads(), 8);       // Total heads
        EXPECT_EQ(cache->local_n_kv_heads(), 4); // Local heads for this rank
        EXPECT_EQ(cache->kv_head_start(), 0);
        EXPECT_EQ(cache->local_kv_dim(), 4 * 64); // local_n_kv_heads * head_dim
    }

    // =============================================================================
    // Test: Sharded KVCache Second Rank
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateShardedCPUKVCache_SecondRank)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::BF16;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 8;       // Total heads
        config.local_n_kv_heads = 4; // Half the heads
        config.kv_head_start = 4;    // Rank 1 starts at head 4
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_TRUE(cache->is_sharded());
        EXPECT_EQ(cache->kv_head_start(), 4);
        EXPECT_EQ(cache->local_n_kv_heads(), 4);
    }

    // =============================================================================
    // Test: Invalid Config - Null MPI Context throws
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, InvalidConfig_NullMPIContext)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = nullptr; // Invalid: null MPI context

        EXPECT_THROW(KernelFactory::createCPUKVCache(config), std::runtime_error);
    }

    // =============================================================================
    // Test: Invalid Config - Zero Layers throws
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, InvalidConfig_ZeroLayers)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 0; // Invalid: zero layers
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        EXPECT_THROW(KernelFactory::createCPUKVCache(config), std::runtime_error);
    }

    // =============================================================================
    // Test: Invalid Config - Zero KV Heads throws
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, InvalidConfig_ZeroKVHeads)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 0; // Invalid: zero KV heads
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        EXPECT_THROW(KernelFactory::createCPUKVCache(config), std::runtime_error);
    }

    // =============================================================================
    // Test: KVCacheConfig Helper Methods
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, KVCacheConfig_HelperMethods)
    {
        // Test is_sharded() with non-sharded config
        {
            KVCacheConfig config;
            config.n_kv_heads = 8;
            config.local_n_kv_heads = 0; // 0 means use all heads (not sharded)
            EXPECT_FALSE(config.is_sharded());
        }

        // Test is_sharded() with full heads (local == total)
        {
            KVCacheConfig config;
            config.n_kv_heads = 8;
            config.local_n_kv_heads = 8; // Same as total = not sharded
            EXPECT_FALSE(config.is_sharded());
        }

        // Test is_sharded() with sharded config
        {
            KVCacheConfig config;
            config.n_kv_heads = 8;
            config.local_n_kv_heads = 4; // Half = sharded
            EXPECT_TRUE(config.is_sharded());
        }

        // Test is_cuda() with CPU device
        {
            KVCacheConfig config;
            config.device = DeviceId::cpu();
            EXPECT_FALSE(config.is_cuda());
        }

        // Test is_cuda() with CUDA device
        {
            KVCacheConfig config;
            config.device = DeviceId::cuda(0);
            EXPECT_TRUE(config.is_cuda());
        }

        // Test is_cuda() with multiple CUDA devices
        {
            KVCacheConfig config;
            config.device = DeviceId::cuda(3);
            EXPECT_TRUE(config.is_cuda());
        }
    }

    // =============================================================================
    // Test: createKVCache dispatches to createCPUKVCache for CPU devices
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateKVCache_DispatchesToCPU)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 1;
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        // createKVCache should dispatch to createCPUKVCache for CPU devices
        auto cache = KernelFactory::createKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->precision(), ActivationPrecision::FP32);
        EXPECT_EQ(cache->n_layers(), 4);
    }

    // =============================================================================
    // Test: Layout Mode Configuration
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_PositionMajorLayout)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 2;
        config.batch_size = 1;
        config.max_seq_len = 64;
        config.n_kv_heads = 2;
        config.head_dim = 32;
        config.mpi_ctx = &getTestMPIContext();
        config.layout_mode = KVCacheLayoutMode::POSITION_MAJOR;

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
        EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_POS_HEAD_DIM);
    }

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_HeadMajorLayout)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::Q16_1; // Q16_1 often uses HEAD_MAJOR
        config.device = DeviceId::cpu();
        config.num_layers = 2;
        config.batch_size = 1;
        config.max_seq_len = 64;
        config.n_kv_heads = 2;
        config.head_dim = 64; // head_dim=64 for Q16_1
        config.mpi_ctx = &getTestMPIContext();
        config.layout_mode = KVCacheLayoutMode::HEAD_MAJOR;

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
        EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_HEAD_POS_DIM);
    }

    // =============================================================================
    // Test: Cache Tensor Access After Creation
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_TensorAccess)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 2;
        config.batch_size = 1;
        config.max_seq_len = 64;
        config.n_kv_heads = 2;
        config.head_dim = 32;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);
        ASSERT_NE(cache, nullptr);

        // Verify we can access K/V tensors for each layer
        for (int layer = 0; layer < 2; ++layer)
        {
            ITensor *k = cache->get_k(layer);
            ITensor *v = cache->get_v(layer);
            ASSERT_NE(k, nullptr) << "K tensor null for layer " << layer;
            ASSERT_NE(v, nullptr) << "V tensor null for layer " << layer;
        }

        // Initially, all cached token counts should be 0
        for (int layer = 0; layer < 2; ++layer)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer), 0);
        }
    }

    // =============================================================================
    // Test: Batched Mode Creation
    // =============================================================================

    TEST_F(Test__KernelFactory_KVCache, CreateCPUKVCache_BatchedMode)
    {
        KVCacheConfig config;
        config.precision = ActivationPrecision::FP32;
        config.device = DeviceId::cpu();
        config.num_layers = 4;
        config.batch_size = 8; // Batched mode
        config.max_seq_len = 128;
        config.n_kv_heads = 4;
        config.head_dim = 64;
        config.mpi_ctx = &getTestMPIContext();

        auto cache = KernelFactory::createCPUKVCache(config);

        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->batch_size(), 8);

        // Verify all sequences have 0 cached tokens initially
        for (int layer = 0; layer < 4; ++layer)
        {
            for (int seq = 0; seq < 8; ++seq)
            {
                EXPECT_EQ(cache->get_cached_tokens(layer, seq), 0);
            }
        }
    }

} // namespace llaminar2::test
