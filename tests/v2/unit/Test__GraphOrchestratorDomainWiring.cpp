/**
 * @file Test__GraphOrchestratorDomainWiring.cpp
 * @brief Unit tests for GraphOrchestrator MultiDomainTPConfig wiring (Phase 6.3)
 *
 * Tests that GraphOrchestrator correctly wires domain configuration for
 * heterogeneous tensor parallelism support.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/GraphOrchestrator.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for GraphOrchestrator domain wiring tests
 */
class Test__GraphOrchestratorDomainWiring : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager (required for orchestrator construction)
        DeviceManager::instance().initialize(-1);

        // Minimal model config for testing (Qwen2-0.5B style)
        config_.n_layers = 4;
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
// setDomainConfig Tests
// =============================================================================

/**
 * @brief Test that domain config can be set on orchestrator
 */
TEST_F(Test__GraphOrchestratorDomainWiring, SetDomainConfigWorks)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Initially no domain config
    EXPECT_EQ(orchestrator->domainConfig(), nullptr);

    // Create and set domain config
    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));

    orchestrator->setDomainConfig(domain_config);

    // Verify config was set
    ASSERT_NE(orchestrator->domainConfig(), nullptr);
    EXPECT_EQ(orchestrator->domainConfig()->domains().size(), 1);
}

/**
 * @brief Test that setting nullptr clears domain config
 */
TEST_F(Test__GraphOrchestratorDomainWiring, SetDomainConfigToNullClears)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Set domain config
    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));
    orchestrator->setDomainConfig(domain_config);
    ASSERT_NE(orchestrator->domainConfig(), nullptr);

    // Clear by setting nullptr
    orchestrator->setDomainConfig(nullptr);
    EXPECT_EQ(orchestrator->domainConfig(), nullptr);
}

/**
 * @brief Test that domain config with multiple domains works
 */
TEST_F(Test__GraphOrchestratorDomainWiring, SetDomainConfigWithMultipleDomains)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Create config with GPU and CPU domains
    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));

    orchestrator->setDomainConfig(domain_config);

    ASSERT_NE(orchestrator->domainConfig(), nullptr);
    EXPECT_EQ(orchestrator->domainConfig()->domains().size(), 2);
    EXPECT_NE(orchestrator->domainConfig()->gpuDomain(), nullptr);
    EXPECT_NE(orchestrator->domainConfig()->cpuDomain(), nullptr);
}

// =============================================================================
// getDomainForLayer Tests
// =============================================================================

/**
 * @brief Test that getDomainForLayer returns nullptr without config
 */
TEST_F(Test__GraphOrchestratorDomainWiring, GetDomainForLayerReturnsNullWithoutConfig)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // No domain config set
    EXPECT_EQ(orchestrator->domainConfig(), nullptr);

    // getDomainForLayer should return nullptr (legacy MPI path)
    EXPECT_EQ(orchestrator->getDomainForLayer(0, true), nullptr);
    EXPECT_EQ(orchestrator->getDomainForLayer(0, false), nullptr);
    EXPECT_EQ(orchestrator->getDomainForLayer(3, true), nullptr);
}

/**
 * @brief Test that getDomainForLayer returns GPU domain for attention
 */
TEST_F(Test__GraphOrchestratorDomainWiring, GetDomainForLayerReturnsGPUDomainForAttention)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));
    orchestrator->setDomainConfig(domain_config);

    // Attention should use GPU domain
    const TPDomain *domain = orchestrator->getDomainForLayer(0, /*is_attention=*/true);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
    EXPECT_EQ(domain->name, "gpu_tp");
}

/**
 * @brief Test that getDomainForLayer returns CPU domain for FFN
 */
TEST_F(Test__GraphOrchestratorDomainWiring, GetDomainForLayerReturnsCPUDomainForFFN)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));
    orchestrator->setDomainConfig(domain_config);

    // FFN should use CPU domain
    const TPDomain *domain = orchestrator->getDomainForLayer(0, /*is_attention=*/false);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_EQ(domain->name, "cpu_tp");
}

/**
 * @brief Test that getDomainForLayer returns same domain for all layers
 */
TEST_F(Test__GraphOrchestratorDomainWiring, GetDomainForLayerConsistentAcrossLayers)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));
    orchestrator->setDomainConfig(domain_config);

    // All attention layers should use same domain
    for (int layer = 0; layer < config_.n_layers; ++layer)
    {
        const TPDomain *domain = orchestrator->getDomainForLayer(layer, /*is_attention=*/true);
        ASSERT_NE(domain, nullptr) << "Layer " << layer << " attention domain should not be null";
        EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK)
            << "Layer " << layer << " attention should use GPU domain";
    }

    // All FFN layers should use same domain
    for (int layer = 0; layer < config_.n_layers; ++layer)
    {
        const TPDomain *domain = orchestrator->getDomainForLayer(layer, /*is_attention=*/false);
        ASSERT_NE(domain, nullptr) << "Layer " << layer << " FFN domain should not be null";
        EXPECT_EQ(domain->type, TPDomainType::CPU_CROSS_RANK)
            << "Layer " << layer << " FFN should use CPU domain";
    }
}

/**
 * @brief Test that FFN falls back to GPU domain when no CPU domain
 */
TEST_F(Test__GraphOrchestratorDomainWiring, GetDomainForLayerFFNFallbackToGPU)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Only GPU domain, no CPU
    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));
    orchestrator->setDomainConfig(domain_config);

    // FFN should fall back to GPU domain
    const TPDomain *domain = orchestrator->getDomainForLayer(0, /*is_attention=*/false);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
}

// =============================================================================
// Domain Config Persistence Tests
// =============================================================================

/**
 * @brief Test that domain config persists through multiple orchestrator operations
 */
TEST_F(Test__GraphOrchestratorDomainWiring, DomainConfigPersistsAcrossOperations)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    auto domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));
    orchestrator->setDomainConfig(domain_config);

    // Perform some operations
    orchestrator->clearCache();
    orchestrator->initializeGraphCache(config_.n_layers);

    // Domain config should still be set
    ASSERT_NE(orchestrator->domainConfig(), nullptr);
    EXPECT_EQ(orchestrator->domainConfig()->domains().size(), 1);

    // getDomainForLayer should still work
    const TPDomain *domain = orchestrator->getDomainForLayer(0, true);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->name, "gpu_tp");
}

/**
 * @brief Test that domain config can be updated
 */
TEST_F(Test__GraphOrchestratorDomainWiring, DomainConfigCanBeUpdated)
{
    auto orchestrator = std::make_unique<GraphOrchestrator>(config_, nullptr);

    // Set first config (GPU only)
    auto config1 = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain()}));
    orchestrator->setDomainConfig(config1);
    EXPECT_EQ(orchestrator->domainConfig()->domains().size(), 1);

    // Update to second config (GPU + CPU)
    auto config2 = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({createGPUDomain(), createCPUDomain()}));
    orchestrator->setDomainConfig(config2);

    // Should have new config
    ASSERT_NE(orchestrator->domainConfig(), nullptr);
    EXPECT_EQ(orchestrator->domainConfig()->domains().size(), 2);
    EXPECT_NE(orchestrator->domainConfig()->cpuDomain(), nullptr);
}
