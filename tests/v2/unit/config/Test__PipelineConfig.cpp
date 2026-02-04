/**
 * @file Test__PipelineConfig.cpp
 * @brief Unit tests for PipelineConfig
 *
 * Tests the PipelineConfig structure which combines TP domains and PP stages
 * into a complete pipeline specification for the unified PP graph architecture.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "config/PipelineConfig.h"
#include "config/TPDomainConfig.h"
#include "config/PPStageConfig.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Helper Functions
// =============================================================================

namespace
{

/// Create a simple valid config for tests that need a baseline
PipelineConfig createValidTwoStageConfig()
{
    PipelineConfig config;
    config.total_layers = 24;

    // Two domains
    TPDomainConfig domain0;
    domain0.name = "gpu0";
    domain0.devices.push_back(DeviceId::cuda(0));
    domain0.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "gpu1";
    domain1.devices.push_back(DeviceId::cuda(1));
    domain1.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain1));

    // Two stages
    config.pp_stages.push_back(PPStageConfig::firstStage(0, "gpu0", 0, 12));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "gpu1", 12, 24));

    config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::PCIE_BAR;

    return config;
}

/// Create a valid config with 2-way TP
PipelineConfig createValidTPConfig()
{
    return PipelineConfig::tensorParallel(
        24,
        {DeviceId::cuda(0), DeviceId::cuda(1)},
        CollectiveBackendType::NCCL);
}

} // anonymous namespace

// =============================================================================
// Lookup Tests - getDomain
// =============================================================================

TEST(Test__PipelineConfig, GetDomain_FindsExistingDomain)
{
    PipelineConfig config = createValidTwoStageConfig();

    const TPDomainConfig *domain = config.getDomain("gpu0");

    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->name, "gpu0");
    EXPECT_EQ(domain->devices.size(), 1);
    EXPECT_TRUE(domain->devices[0].is_cuda());
    EXPECT_EQ(domain->devices[0].ordinal, 0);
}

TEST(Test__PipelineConfig, GetDomain_ReturnsNullForMissing)
{
    PipelineConfig config = createValidTwoStageConfig();

    const TPDomainConfig *domain = config.getDomain("nonexistent");

    EXPECT_EQ(domain, nullptr);
}

TEST(Test__PipelineConfig, GetDomain_EmptyConfig)
{
    PipelineConfig config;

    const TPDomainConfig *domain = config.getDomain("any");

    EXPECT_EQ(domain, nullptr);
}

// =============================================================================
// Lookup Tests - getDomainForStage
// =============================================================================

TEST(Test__PipelineConfig, GetDomainForStage_FindsCorrectDomain)
{
    PipelineConfig config = createValidTwoStageConfig();

    const TPDomainConfig *domain0 = config.getDomainForStage(0);
    const TPDomainConfig *domain1 = config.getDomainForStage(1);

    ASSERT_NE(domain0, nullptr);
    ASSERT_NE(domain1, nullptr);
    EXPECT_EQ(domain0->name, "gpu0");
    EXPECT_EQ(domain1->name, "gpu1");
}

TEST(Test__PipelineConfig, GetDomainForStage_ReturnsNullForMissingStage)
{
    PipelineConfig config = createValidTwoStageConfig();

    const TPDomainConfig *domain = config.getDomainForStage(999);

    EXPECT_EQ(domain, nullptr);
}

// =============================================================================
// Lookup Tests - getStageForLayer
// =============================================================================

TEST(Test__PipelineConfig, GetStageForLayer_FindsCorrectStage)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Layer 0 should be in stage 0
    const PPStageConfig *stage0 = config.getStageForLayer(0);
    ASSERT_NE(stage0, nullptr);
    EXPECT_EQ(stage0->stage_id, 0);

    // Layer 11 should be in stage 0 (last layer before split)
    const PPStageConfig *stage0b = config.getStageForLayer(11);
    ASSERT_NE(stage0b, nullptr);
    EXPECT_EQ(stage0b->stage_id, 0);

    // Layer 12 should be in stage 1 (first layer after split)
    const PPStageConfig *stage1 = config.getStageForLayer(12);
    ASSERT_NE(stage1, nullptr);
    EXPECT_EQ(stage1->stage_id, 1);

    // Layer 23 should be in stage 1
    const PPStageConfig *stage1b = config.getStageForLayer(23);
    ASSERT_NE(stage1b, nullptr);
    EXPECT_EQ(stage1b->stage_id, 1);
}

TEST(Test__PipelineConfig, GetStageForLayer_ReturnsNullForInvalidLayer)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Negative layer
    EXPECT_EQ(config.getStageForLayer(-1), nullptr);

    // Layer beyond total_layers
    EXPECT_EQ(config.getStageForLayer(24), nullptr);
    EXPECT_EQ(config.getStageForLayer(100), nullptr);
}

// =============================================================================
// Lookup Tests - getDeviceForLayer
// =============================================================================

TEST(Test__PipelineConfig, GetDeviceForLayer_ReturnsCorrectDevice)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Layer 5 -> stage 0 -> gpu0 domain -> cuda:0
    DeviceId device0 = config.getDeviceForLayer(5);
    EXPECT_TRUE(device0.is_cuda());
    EXPECT_EQ(device0.ordinal, 0);

    // Layer 15 -> stage 1 -> gpu1 domain -> cuda:1
    DeviceId device1 = config.getDeviceForLayer(15);
    EXPECT_TRUE(device1.is_cuda());
    EXPECT_EQ(device1.ordinal, 1);
}

TEST(Test__PipelineConfig, GetDeviceForLayer_ReturnsCPUForInvalidLayer)
{
    PipelineConfig config = createValidTwoStageConfig();

    DeviceId device = config.getDeviceForLayer(100);

    EXPECT_TRUE(device.is_cpu());
}

// =============================================================================
// Lookup Tests - needsPPTransfer
// =============================================================================

TEST(Test__PipelineConfig, NeedsPPTransfer_TrueAcrossStageBoundary)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Layer 11 (stage 0) -> Layer 12 (stage 1)
    EXPECT_TRUE(config.needsPPTransfer(11, 12));

    // Reverse direction
    EXPECT_TRUE(config.needsPPTransfer(12, 11));
}

TEST(Test__PipelineConfig, NeedsPPTransfer_FalseWithinStage)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Within stage 0
    EXPECT_FALSE(config.needsPPTransfer(0, 5));
    EXPECT_FALSE(config.needsPPTransfer(5, 11));

    // Within stage 1
    EXPECT_FALSE(config.needsPPTransfer(12, 20));
    EXPECT_FALSE(config.needsPPTransfer(15, 23));
}

TEST(Test__PipelineConfig, NeedsPPTransfer_FalseForInvalidLayers)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Invalid layers should return false (not trigger transfer)
    EXPECT_FALSE(config.needsPPTransfer(-1, 5));
    EXPECT_FALSE(config.needsPPTransfer(5, 100));
}

// =============================================================================
// Lookup Tests - getTransferBackend
// =============================================================================

TEST(Test__PipelineConfig, GetTransferBackend_ReturnsConfiguredBackend)
{
    PipelineConfig config = createValidTwoStageConfig();

    CollectiveBackendType backend = config.getTransferBackend(0, 1);

    EXPECT_EQ(backend, CollectiveBackendType::PCIE_BAR);
}

TEST(Test__PipelineConfig, GetTransferBackend_ReturnsAutoForUnconfigured)
{
    PipelineConfig config = createValidTwoStageConfig();

    // Reverse direction not configured
    CollectiveBackendType backend = config.getTransferBackend(1, 0);

    EXPECT_EQ(backend, CollectiveBackendType::AUTO);
}

TEST(Test__PipelineConfig, GetTransferBackend_ReturnsAutoForNonexistentStages)
{
    PipelineConfig config = createValidTwoStageConfig();

    CollectiveBackendType backend = config.getTransferBackend(0, 99);

    EXPECT_EQ(backend, CollectiveBackendType::AUTO);
}

// =============================================================================
// Lookup Tests - getStageIdForLayer
// =============================================================================

TEST(Test__PipelineConfig, GetStageIdForLayer_ReturnsCorrectId)
{
    PipelineConfig config = createValidTwoStageConfig();

    EXPECT_EQ(config.getStageIdForLayer(0), 0);
    EXPECT_EQ(config.getStageIdForLayer(11), 0);
    EXPECT_EQ(config.getStageIdForLayer(12), 1);
    EXPECT_EQ(config.getStageIdForLayer(23), 1);
}

TEST(Test__PipelineConfig, GetStageIdForLayer_ReturnsNegativeForInvalid)
{
    PipelineConfig config = createValidTwoStageConfig();

    EXPECT_EQ(config.getStageIdForLayer(-1), -1);
    EXPECT_EQ(config.getStageIdForLayer(24), -1);
    EXPECT_EQ(config.getStageIdForLayer(100), -1);
}

// =============================================================================
// Query Tests - numStages
// =============================================================================

TEST(Test__PipelineConfig, NumStages_ReturnsCorrectCount)
{
    // Two-stage config
    PipelineConfig config2 = createValidTwoStageConfig();
    EXPECT_EQ(config2.numStages(), 2);

    // Single-stage config
    PipelineConfig config1 = PipelineConfig::singleDevice(24, DeviceId::cuda(0));
    EXPECT_EQ(config1.numStages(), 1);

    // Empty config
    PipelineConfig empty;
    EXPECT_EQ(empty.numStages(), 0);
}

// =============================================================================
// Query Tests - maxTPDegree
// =============================================================================

TEST(Test__PipelineConfig, MaxTPDegree_ReturnsLargestDomain)
{
    PipelineConfig config;
    config.total_layers = 24;

    // Domain with 2 devices
    TPDomainConfig domain0;
    domain0.name = "small";
    domain0.devices.push_back(DeviceId::cuda(0));
    domain0.devices.push_back(DeviceId::cuda(1));
    config.tp_domains.push_back(std::move(domain0));

    // Domain with 4 devices
    TPDomainConfig domain1;
    domain1.name = "large";
    domain1.devices.push_back(DeviceId::rocm(0));
    domain1.devices.push_back(DeviceId::rocm(1));
    domain1.devices.push_back(DeviceId::rocm(2));
    domain1.devices.push_back(DeviceId::rocm(3));
    config.tp_domains.push_back(std::move(domain1));

    EXPECT_EQ(config.maxTPDegree(), 4);
}

TEST(Test__PipelineConfig, MaxTPDegree_ReturnsZeroForEmpty)
{
    PipelineConfig config;

    EXPECT_EQ(config.maxTPDegree(), 0);
}

// =============================================================================
// Query Tests - isSingleStage
// =============================================================================

TEST(Test__PipelineConfig, IsSingleStage_TrueForOneStage)
{
    PipelineConfig config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));

    EXPECT_TRUE(config.isSingleStage());
}

TEST(Test__PipelineConfig, IsSingleStage_FalseForMultipleStages)
{
    PipelineConfig config = createValidTwoStageConfig();

    EXPECT_FALSE(config.isSingleStage());
}

TEST(Test__PipelineConfig, IsSingleStage_FalseForEmpty)
{
    PipelineConfig config;

    EXPECT_FALSE(config.isSingleStage());
}

// =============================================================================
// Query Tests - hasTP
// =============================================================================

TEST(Test__PipelineConfig, HasTP_TrueWhenDomainHasMultipleDevices)
{
    PipelineConfig config = createValidTPConfig();

    EXPECT_TRUE(config.hasTP());
}

TEST(Test__PipelineConfig, HasTP_FalseWhenAllDomainsHaveOneDevice)
{
    PipelineConfig config = createValidTwoStageConfig();

    EXPECT_FALSE(config.hasTP());
}

TEST(Test__PipelineConfig, HasTP_FalseForEmpty)
{
    PipelineConfig config;

    EXPECT_FALSE(config.hasTP());
}

// =============================================================================
// Query Tests - hasPP
// =============================================================================

TEST(Test__PipelineConfig, HasPP_TrueWhenMultipleStages)
{
    PipelineConfig config = createValidTwoStageConfig();

    EXPECT_TRUE(config.hasPP());
}

TEST(Test__PipelineConfig, HasPP_FalseWhenSingleStage)
{
    PipelineConfig config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));

    EXPECT_FALSE(config.hasPP());
}

// =============================================================================
// Query Tests - getAllDevices
// =============================================================================

TEST(Test__PipelineConfig, GetAllDevices_ReturnsAllUniqueDevices)
{
    PipelineConfig config = createValidTwoStageConfig();

    std::vector<DeviceId> devices = config.getAllDevices();

    ASSERT_EQ(devices.size(), 2);
    EXPECT_TRUE(devices[0].is_cuda());
    EXPECT_EQ(devices[0].ordinal, 0);
    EXPECT_TRUE(devices[1].is_cuda());
    EXPECT_EQ(devices[1].ordinal, 1);
}

TEST(Test__PipelineConfig, GetAllDevices_DeduplicatesSharedDevices)
{
    PipelineConfig config;
    config.total_layers = 24;

    // Two domains sharing cuda:0
    TPDomainConfig domain0;
    domain0.name = "d0";
    domain0.devices.push_back(DeviceId::cuda(0));
    domain0.devices.push_back(DeviceId::cuda(1));
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "d1";
    domain1.devices.push_back(DeviceId::cuda(0)); // Shared
    domain1.devices.push_back(DeviceId::cuda(2));
    config.tp_domains.push_back(std::move(domain1));

    std::vector<DeviceId> devices = config.getAllDevices();

    // Should have 3 unique devices: cuda:0, cuda:1, cuda:2
    ASSERT_EQ(devices.size(), 3);
}

TEST(Test__PipelineConfig, GetAllDevices_EmptyForEmptyConfig)
{
    PipelineConfig config;

    std::vector<DeviceId> devices = config.getAllDevices();

    EXPECT_TRUE(devices.empty());
}

// =============================================================================
// Validation Tests - Domain Requirements
// =============================================================================

TEST(Test__PipelineConfig, Validate_FailsOnNoDomains)
{
    PipelineConfig config;
    config.total_layers = 24;
    config.pp_stages.push_back(PPStageConfig::fullModel(24, "default"));

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("No TP domains") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_FailsOnNoStages)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("No PP stages") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_FailsOnMissingDomainReference)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "actual_domain";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    // Stage references nonexistent domain
    PPStageConfig stage;
    stage.stage_id = 0;
    stage.domain_name = "nonexistent";
    stage.first_layer = 0;
    stage.last_layer = 24;
    stage.has_embedding = true;
    stage.has_lm_head = true;
    config.pp_stages.push_back(stage);

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("unknown domain") != std::string::npos);
}

// =============================================================================
// Validation Tests - Layer Coverage
// =============================================================================

TEST(Test__PipelineConfig, Validate_FailsOnLayerGap)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    // Gap between layers 10-12
    PPStageConfig stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "default";
    stage0.first_layer = 0;
    stage0.last_layer = 10;
    stage0.has_embedding = true;
    config.pp_stages.push_back(stage0);

    PPStageConfig stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "default";
    stage1.first_layer = 12; // Gap! Should be 10
    stage1.last_layer = 24;
    stage1.has_lm_head = true;
    config.pp_stages.push_back(stage1);

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("gap") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_FailsOnLayerOverlap)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    // Overlap between stages
    PPStageConfig stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "default";
    stage0.first_layer = 0;
    stage0.last_layer = 14; // Overlaps with stage1
    stage0.has_embedding = true;
    config.pp_stages.push_back(stage0);

    PPStageConfig stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "default";
    stage1.first_layer = 10; // Overlap! Should be 14
    stage1.last_layer = 24;
    stage1.has_lm_head = true;
    config.pp_stages.push_back(stage1);

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("overlap") != std::string::npos);
}

// =============================================================================
// Validation Tests - Embedding/LM Head
// =============================================================================

TEST(Test__PipelineConfig, Validate_FailsOnNoEmbeddingStage)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    PPStageConfig stage;
    stage.stage_id = 0;
    stage.domain_name = "default";
    stage.first_layer = 0;
    stage.last_layer = 24;
    stage.has_embedding = false; // Missing!
    stage.has_lm_head = true;
    config.pp_stages.push_back(stage);

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("has_embedding") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_FailsOnNoLMHeadStage)
{
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    PPStageConfig stage;
    stage.stage_id = 0;
    stage.domain_name = "default";
    stage.first_layer = 0;
    stage.last_layer = 24;
    stage.has_embedding = true;
    stage.has_lm_head = false; // Missing!
    config.pp_stages.push_back(stage);

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("has_lm_head") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_FailsOnZeroTotalLayers)
{
    PipelineConfig config;
    config.total_layers = 0;

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_TRUE(error.find("total_layers") != std::string::npos);
}

TEST(Test__PipelineConfig, Validate_SucceedsForValidConfig)
{
    PipelineConfig config = createValidTwoStageConfig();

    std::string error;
    EXPECT_TRUE(config.validate(&error));
    EXPECT_TRUE(error.empty());
}

// =============================================================================
// Factory Tests - singleDevice
// =============================================================================

TEST(Test__PipelineConfig, SingleDevice_CreatesValidConfig)
{
    PipelineConfig config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_EQ(config.total_layers, 24);
    EXPECT_EQ(config.tp_domains.size(), 1);
    EXPECT_EQ(config.pp_stages.size(), 1);
    EXPECT_EQ(config.tp_domains[0].degree(), 1);
    EXPECT_TRUE(config.tp_domains[0].devices[0].is_cuda());
    EXPECT_TRUE(config.isSingleStage());
    EXPECT_FALSE(config.hasTP());
    EXPECT_FALSE(config.hasPP());
}

TEST(Test__PipelineConfig, SingleDevice_WorksWithCPU)
{
    PipelineConfig config = PipelineConfig::singleDevice(16, DeviceId::cpu());

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_TRUE(config.tp_domains[0].devices[0].is_cpu());
}

// =============================================================================
// Factory Tests - tensorParallel
// =============================================================================

TEST(Test__PipelineConfig, TensorParallel_CreatesValidConfig)
{
    std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    PipelineConfig config = PipelineConfig::tensorParallel(24, devices, CollectiveBackendType::NCCL);

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_EQ(config.total_layers, 24);
    EXPECT_EQ(config.tp_domains.size(), 1);
    EXPECT_EQ(config.pp_stages.size(), 1);
    EXPECT_EQ(config.tp_domains[0].degree(), 2);
    EXPECT_EQ(config.tp_domains[0].tp_backend, CollectiveBackendType::NCCL);
    EXPECT_TRUE(config.isSingleStage());
    EXPECT_TRUE(config.hasTP());
    EXPECT_FALSE(config.hasPP());
    EXPECT_EQ(config.maxTPDegree(), 2);
}

TEST(Test__PipelineConfig, TensorParallel_WorksWithMixedGPUs)
{
    std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    PipelineConfig config = PipelineConfig::tensorParallel(24, devices, CollectiveBackendType::PCIE_BAR);

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_FALSE(config.tp_domains[0].isHomogeneous());
}

// =============================================================================
// Factory Tests - pipelineParallel2Stage
// =============================================================================

TEST(Test__PipelineConfig, PipelineParallel2Stage_CreatesValidConfig)
{
    PipelineConfig config = PipelineConfig::pipelineParallel2Stage(
        24,
        DeviceId::cuda(0), 12,
        DeviceId::cuda(1),
        CollectiveBackendType::PCIE_BAR);

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_EQ(config.total_layers, 24);
    EXPECT_EQ(config.tp_domains.size(), 2);
    EXPECT_EQ(config.pp_stages.size(), 2);
    EXPECT_FALSE(config.isSingleStage());
    EXPECT_FALSE(config.hasTP());
    EXPECT_TRUE(config.hasPP());

    // Check stage layer ranges
    EXPECT_EQ(config.pp_stages[0].first_layer, 0);
    EXPECT_EQ(config.pp_stages[0].last_layer, 12);
    EXPECT_EQ(config.pp_stages[1].first_layer, 12);
    EXPECT_EQ(config.pp_stages[1].last_layer, 24);

    // Check transfer backend
    EXPECT_EQ(config.getTransferBackend(0, 1), CollectiveBackendType::PCIE_BAR);
}

TEST(Test__PipelineConfig, PipelineParallel2Stage_WorksWithHeterogeneousDevices)
{
    PipelineConfig config = PipelineConfig::pipelineParallel2Stage(
        28,
        DeviceId::cuda(0), 14,
        DeviceId::rocm(0),
        CollectiveBackendType::HOST);

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    // Stage 0 device
    DeviceId dev0 = config.getDeviceForLayer(5);
    EXPECT_TRUE(dev0.is_cuda());
    EXPECT_EQ(dev0.ordinal, 0);

    // Stage 1 device
    DeviceId dev1 = config.getDeviceForLayer(20);
    EXPECT_TRUE(dev1.is_rocm());
    EXPECT_EQ(dev1.ordinal, 0);
}

// =============================================================================
// Integration Tests - Complex Configurations
// =============================================================================

TEST(Test__PipelineConfig, ComplexConfig_PPWithTP_Validates)
{
    // PP with 2 stages, each having 2-way TP
    PipelineConfig config;
    config.total_layers = 28;

    // Two TP domains
    TPDomainConfig domain0;
    domain0.name = "nvidia_tp";
    domain0.devices.push_back(DeviceId::cuda(0));
    domain0.devices.push_back(DeviceId::cuda(1));
    domain0.tp_backend = CollectiveBackendType::NCCL;
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "amd_tp";
    domain1.devices.push_back(DeviceId::rocm(0));
    domain1.devices.push_back(DeviceId::rocm(1));
    domain1.tp_backend = CollectiveBackendType::RCCL;
    config.tp_domains.push_back(std::move(domain1));

    // Two stages
    config.pp_stages.push_back(PPStageConfig::firstStage(0, "nvidia_tp", 0, 14));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "amd_tp", 14, 28));

    config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_TRUE(config.hasTP());
    EXPECT_TRUE(config.hasPP());
    EXPECT_EQ(config.maxTPDegree(), 2);
    EXPECT_EQ(config.getAllDevices().size(), 4);
}

TEST(Test__PipelineConfig, ComplexConfig_ThreeStages_Validates)
{
    // 3-stage PP across different device types
    PipelineConfig config;
    config.total_layers = 27;

    // Three domains
    TPDomainConfig domain0;
    domain0.name = "nvidia";
    domain0.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "amd";
    domain1.devices.push_back(DeviceId::rocm(0));
    config.tp_domains.push_back(std::move(domain1));

    TPDomainConfig domain2;
    domain2.name = "cpu";
    domain2.devices.push_back(DeviceId::cpu());
    config.tp_domains.push_back(std::move(domain2));

    // Three stages (9 layers each)
    config.pp_stages.push_back(PPStageConfig::firstStage(0, "nvidia", 0, 9));
    config.pp_stages.push_back(PPStageConfig::middleStage(1, "amd", 9, 18));
    config.pp_stages.push_back(PPStageConfig::lastStage(2, "cpu", 18, 27));

    config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::PCIE_BAR;
    config.pp_transfer_backends[{1, 2}] = CollectiveBackendType::HOST;

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_EQ(config.numStages(), 3);
    EXPECT_TRUE(config.hasPP());
    EXPECT_FALSE(config.hasTP());
    EXPECT_EQ(config.getAllDevices().size(), 3);

    // Verify transfer needs
    EXPECT_TRUE(config.needsPPTransfer(8, 9));
    EXPECT_TRUE(config.needsPPTransfer(17, 18));
    EXPECT_FALSE(config.needsPPTransfer(5, 8));
}

TEST(Test__PipelineConfig, ComplexConfig_UnevenLayerSplit_Validates)
{
    // Uneven layer split: 10 + 8 + 6 = 24
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "gpu";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    config.pp_stages.push_back(PPStageConfig::firstStage(0, "gpu", 0, 10));
    config.pp_stages.push_back(PPStageConfig::middleStage(1, "gpu", 10, 18));
    config.pp_stages.push_back(PPStageConfig::lastStage(2, "gpu", 18, 24));

    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;

    EXPECT_EQ(config.pp_stages[0].numLayers(), 10);
    EXPECT_EQ(config.pp_stages[1].numLayers(), 8);
    EXPECT_EQ(config.pp_stages[2].numLayers(), 6);
}

// =============================================================================
// Auto-Selection Tests - autoSelectBackends
// =============================================================================

TEST(Test__PipelineConfig, AutoSelectBackends_ResolvesTPDomainAuto)
{
    // Create config with a TP domain where tp_backend = AUTO
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "tp_domain";
    domain.devices.push_back(DeviceId::cuda(0));
    domain.devices.push_back(DeviceId::cuda(1));
    domain.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain));

    config.pp_stages.push_back(PPStageConfig::fullModel(24, "tp_domain"));

    // Initially AUTO
    EXPECT_EQ(config.tp_domains[0].tp_backend, CollectiveBackendType::AUTO);

    config.autoSelectBackends();

    // Should no longer be AUTO - for all-CUDA, expect NCCL
    EXPECT_NE(config.tp_domains[0].tp_backend, CollectiveBackendType::AUTO);
    EXPECT_EQ(config.tp_domains[0].tp_backend, CollectiveBackendType::NCCL);
}

TEST(Test__PipelineConfig, AutoSelectBackends_FillsMissingPPTransfers)
{
    // Create 2-stage PP config with empty pp_transfer_backends
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain0;
    domain0.name = "gpu0";
    domain0.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "gpu1";
    domain1.devices.push_back(DeviceId::cuda(1));
    config.tp_domains.push_back(std::move(domain1));

    config.pp_stages.push_back(PPStageConfig::firstStage(0, "gpu0", 0, 12));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "gpu1", 12, 24));

    // Initially empty
    EXPECT_TRUE(config.pp_transfer_backends.empty());

    config.autoSelectBackends();

    // Should now have entry for {0, 1}
    EXPECT_FALSE(config.pp_transfer_backends.empty());
    auto it = config.pp_transfer_backends.find({0, 1});
    ASSERT_NE(it, config.pp_transfer_backends.end());
    EXPECT_NE(it->second, CollectiveBackendType::AUTO);
}

TEST(Test__PipelineConfig, AutoSelectBackends_PreservesExplicitBackends)
{
    // Create config where pp_transfer_backends[{0,1}] = HOST explicitly
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain0;
    domain0.name = "gpu0";
    domain0.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "gpu1";
    domain1.devices.push_back(DeviceId::cuda(1));
    config.tp_domains.push_back(std::move(domain1));

    config.pp_stages.push_back(PPStageConfig::firstStage(0, "gpu0", 0, 12));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "gpu1", 12, 24));

    // Explicitly set to HOST
    config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::HOST;

    config.autoSelectBackends();

    // Should still be HOST (not overwritten to NCCL)
    auto backend = config.pp_transfer_backends[{0, 1}];
    EXPECT_EQ(backend, CollectiveBackendType::HOST);
}

TEST(Test__PipelineConfig, AutoSelectBackends_CUDAToCUDAGetsNCCL)
{
    // Create 2 stages, both with CUDA devices
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain0;
    domain0.name = "cuda0";
    domain0.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "cuda1";
    domain1.devices.push_back(DeviceId::cuda(1));
    config.tp_domains.push_back(std::move(domain1));

    config.pp_stages.push_back(PPStageConfig::firstStage(0, "cuda0", 0, 12));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "cuda1", 12, 24));

    config.autoSelectBackends();

    // CUDA to CUDA should get NCCL
    auto backend = config.pp_transfer_backends[{0, 1}];
    EXPECT_EQ(backend, CollectiveBackendType::NCCL);
}

TEST(Test__PipelineConfig, AutoSelectBackends_CPUToCPUGetsHOST)
{
    // Create 2 stages, both with CPU devices
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain0;
    domain0.name = "cpu0";
    domain0.devices.push_back(DeviceId::cpu());
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "cpu1";
    domain1.devices.push_back(DeviceId::cpu());
    config.tp_domains.push_back(std::move(domain1));

    config.pp_stages.push_back(PPStageConfig::firstStage(0, "cpu0", 0, 12));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "cpu1", 12, 24));

    config.autoSelectBackends();

    // CPU to CPU should get HOST
    auto backend = config.pp_transfer_backends[{0, 1}];
    EXPECT_EQ(backend, CollectiveBackendType::HOST);
}

// =============================================================================
// Auto-Completion Tests - completeAndValidate
// =============================================================================

TEST(Test__PipelineConfig, CompleteAndValidate_Success)
{
    // Create a valid config with AUTO backends
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "tp_domain";
    domain.devices.push_back(DeviceId::cuda(0));
    domain.devices.push_back(DeviceId::cuda(1));
    domain.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain));

    config.pp_stages.push_back(PPStageConfig::fullModel(24, "tp_domain"));

    std::string error_msg;
    EXPECT_TRUE(config.completeAndValidate(&error_msg));
    EXPECT_TRUE(error_msg.empty());

    // Backend should now be resolved
    EXPECT_NE(config.tp_domains[0].tp_backend, CollectiveBackendType::AUTO);
}

TEST(Test__PipelineConfig, CompleteAndValidate_InvalidConfigFails)
{
    // Create an invalid config (overlapping layer ranges)
    PipelineConfig config;
    config.total_layers = 24;

    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(DeviceId::cuda(0));
    config.tp_domains.push_back(std::move(domain));

    // Overlapping layer ranges
    PPStageConfig stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "default";
    stage0.first_layer = 0;
    stage0.last_layer = 14;  // Overlaps with stage1
    stage0.has_embedding = true;
    config.pp_stages.push_back(stage0);

    PPStageConfig stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "default";
    stage1.first_layer = 10;  // Overlap! Should be 14
    stage1.last_layer = 24;
    stage1.has_lm_head = true;
    config.pp_stages.push_back(stage1);

    std::string error_msg;
    EXPECT_FALSE(config.completeAndValidate(&error_msg));
    EXPECT_FALSE(error_msg.empty());
    EXPECT_TRUE(error_msg.find("overlap") != std::string::npos);
}
