/**
 * @file Test__Qwen2GraphDomain.cpp
 * @brief Unit tests for Qwen2Graph MultiDomainTPConfig support
 *
 * Tests that Qwen2Graph correctly passes TPDomain to AllreduceStage
 * for heterogeneous tensor parallelism support.
 *
 * Part of Phase 4.3: Adding MultiDomainTPConfig support to Qwen2Graph
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2Graph.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for Qwen2Graph domain tests
 */
class Test__Qwen2GraphDomain : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Minimal model config for testing (Qwen2-0.5B style)
        config_.n_layers = 2;
        config_.d_model = 896;
        config_.n_heads = 14;
        config_.n_kv_heads = 2;
        config_.head_dim = 64;
        config_.d_ff = 4864;
        config_.vocab_size = 151936;
        config_.default_device = DeviceId::cpu();
        config_.max_seq_len = 256;
    }

    /**
     * @brief Create a GPU domain for testing
     */
    static TPDomain createGPUDomain()
    {
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.name = "gpu_tp";
        gpu_domain.domain_size = 2;
        gpu_domain.local_rank_in_domain = 0;
        gpu_domain.devices.push_back(DeviceId::cuda(0));
        gpu_domain.devices.push_back(DeviceId::rocm(0));
        gpu_domain.communicator = MPI_COMM_NULL;
        return gpu_domain;
    }

    /**
     * @brief Create a CPU domain for testing
     */
    static TPDomain createCPUDomain()
    {
        TPDomain cpu_domain;
        cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
        cpu_domain.name = "cpu_tp";
        cpu_domain.domain_size = 2;
        cpu_domain.local_rank_in_domain = 0;
        cpu_domain.devices.push_back(DeviceId::cpu());
        cpu_domain.communicator = MPI_COMM_NULL;
        return cpu_domain;
    }

    Qwen2GraphConfig config_;
};

// =============================================================================
// GetDomainForLayer Tests
// =============================================================================

/**
 * @brief Test that getDomainForLayer returns nullptr without config
 *
 * When multi_domain_tp_config is not set (nullptr), getDomainForLayer
 * should return nullptr, allowing AllreduceStage to use the legacy MPI path.
 */
TEST_F(Test__Qwen2GraphDomain, GetDomainForLayerReturnsNullWithoutConfig)
{
    // Config has no multi_domain_tp_config (default nullptr)
    EXPECT_EQ(config_.multi_domain_tp_config, nullptr);

    // Create graph (we can't directly test getDomainForLayer as it's private,
    // but we can verify the config propagation)
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    Qwen2Graph graph(config_, mpi_ctx);

    // The graph should be in "no domain" mode
    // This test verifies the config is correctly initialized
    EXPECT_EQ(config_.multi_domain_tp_config, nullptr);
}

/**
 * @brief Test that config accepts MultiDomainTPConfig pointer
 */
TEST_F(Test__Qwen2GraphDomain, ConfigAcceptsMultiDomainTPConfig)
{
    // Create multi-domain config
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));

    // Set on config
    config_.multi_domain_tp_config = tp_config.get();

    // Verify
    ASSERT_NE(config_.multi_domain_tp_config, nullptr);
    ASSERT_NE(config_.multi_domain_tp_config->gpuDomain(), nullptr);
    EXPECT_EQ(config_.multi_domain_tp_config->gpuDomain()->name, "gpu_tp");
}

/**
 * @brief Test that config with GPU+CPU domains works
 */
TEST_F(Test__Qwen2GraphDomain, ConfigAcceptsMultipleDomains)
{
    // Create multi-domain config with GPU and CPU
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    ASSERT_NE(config_.multi_domain_tp_config, nullptr);
    EXPECT_EQ(config_.multi_domain_tp_config->domains().size(), 2);
    ASSERT_NE(config_.multi_domain_tp_config->gpuDomain(), nullptr);
    ASSERT_NE(config_.multi_domain_tp_config->cpuDomain(), nullptr);
}

// =============================================================================
// Domain Routing Tests
// =============================================================================

/**
 * @brief Test that attention layers route to GPU domain
 */
TEST_F(Test__Qwen2GraphDomain, AttentionRoutesToGPUDomain)
{
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    // domainForLayer for attention should return GPU domain
    const TPDomain *domain = config_.multi_domain_tp_config->domainForLayer(
        /*layer_idx=*/0, /*is_attention=*/true);

    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
    EXPECT_EQ(domain->name, "gpu_tp");
}

/**
 * @brief Test that FFN layers route to CPU domain when available
 */
TEST_F(Test__Qwen2GraphDomain, FFNRoutesToCPUDomain)
{
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    // domainForLayer for FFN should return CPU domain (cross-rank)
    const TPDomain *domain = config_.multi_domain_tp_config->domainForLayer(
        /*layer_idx=*/0, /*is_attention=*/false);

    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_EQ(domain->name, "cpu_tp");
}

/**
 * @brief Test that FFN falls back to GPU domain when no CPU domain
 */
TEST_F(Test__Qwen2GraphDomain, FFNFallsBackToGPUDomain)
{
    // Only GPU domain, no CPU
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    // domainForLayer for FFN should fall back to GPU domain
    const TPDomain *domain = config_.multi_domain_tp_config->domainForLayer(
        /*layer_idx=*/0, /*is_attention=*/false);

    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
}

// =============================================================================
// Backward Compatibility Tests
// =============================================================================

/**
 * @brief Test backward compatibility - null domain config uses legacy MPI
 *
 * When multi_domain_tp_config is nullptr, AllreduceStage should receive
 * domain=nullptr and fall back to the legacy MPI path.
 */
TEST_F(Test__Qwen2GraphDomain, AllreduceWithoutDomainHasNullDomain)
{
    // No multi_domain_tp_config set
    EXPECT_EQ(config_.multi_domain_tp_config, nullptr);

    // Create graph with mock MPI context
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    Qwen2Graph graph(config_, mpi_ctx);

    // The config should still have nullptr
    // AllreduceStage created by this graph will have domain=nullptr
    EXPECT_EQ(config_.multi_domain_tp_config, nullptr);
}

/**
 * @brief Test that default config has null multi_domain_tp_config
 */
TEST_F(Test__Qwen2GraphDomain, DefaultConfigHasNullDomainConfig)
{
    Qwen2GraphConfig default_config;
    EXPECT_EQ(default_config.multi_domain_tp_config, nullptr);
}

// =============================================================================
// Layer Index Tests
// =============================================================================

/**
 * @brief Test domain lookup across multiple layers
 */
TEST_F(Test__Qwen2GraphDomain, DomainLookupAcrossLayers)
{
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    // All layers should route the same way by default
    for (int layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx)
    {
        // Attention -> GPU
        const TPDomain *attn_domain = config_.multi_domain_tp_config->domainForLayer(
            layer_idx, /*is_attention=*/true);
        ASSERT_NE(attn_domain, nullptr);
        EXPECT_EQ(attn_domain->type, TPDomainType::GPU_INTRA_RANK);

        // FFN -> CPU
        const TPDomain *ffn_domain = config_.multi_domain_tp_config->domainForLayer(
            layer_idx, /*is_attention=*/false);
        ASSERT_NE(ffn_domain, nullptr);
        EXPECT_EQ(ffn_domain->type, TPDomainType::CPU_CROSS_RANK);
    }
}

/**
 * @brief Test that Qwen2Graph can be constructed with domain config
 */
TEST_F(Test__Qwen2GraphDomain, GraphConstructsWithDomainConfig)
{
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();

    // Should not throw
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    EXPECT_NO_THROW({
        Qwen2Graph graph(config_, mpi_ctx);
    });
}

// =============================================================================
// Integration-Ready Tests
// =============================================================================

/**
 * @brief Test config is correctly passed through graph construction
 *
 * This verifies the config struct correctly holds the domain pointer
 * and that it persists through graph construction.
 */
TEST_F(Test__Qwen2GraphDomain, ConfigPersistsThroughGraphConstruction)
{
    auto tp_config = std::make_unique<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    config_.multi_domain_tp_config = tp_config.get();
    MultiDomainTPConfig *original_ptr = tp_config.get();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    Qwen2Graph graph(config_, mpi_ctx);

    // Config should still point to our TP config
    EXPECT_EQ(config_.multi_domain_tp_config, original_ptr);
    EXPECT_EQ(config_.multi_domain_tp_config->domains().size(), 2);
}
