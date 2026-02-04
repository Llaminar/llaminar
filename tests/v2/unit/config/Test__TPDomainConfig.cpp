/**
 * @file Test__TPDomainConfig.cpp
 * @brief Unit tests for TPDomainConfig
 *
 * Tests the TPDomainConfig structure which defines tensor parallel domains
 * for the unified PP graph architecture.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "config/TPDomainConfig.h"
#include "collective/ILocalTPContext.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Default Construction Tests
// =============================================================================

TEST(Test__TPDomainConfig, DefaultConstruction)
{
    TPDomainConfig config;

    // Default values
    EXPECT_TRUE(config.name.empty());
    EXPECT_TRUE(config.devices.empty());
    EXPECT_EQ(config.tp_backend, CollectiveBackendType::AUTO);
    EXPECT_TRUE(config.device_weights.empty());

    // Degree of empty config is 0
    EXPECT_EQ(config.degree(), 0);
}

// =============================================================================
// Primary Device Tests
// =============================================================================

TEST(Test__TPDomainConfig, PrimaryDevice_ReturnsFirstDevice)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));

    DeviceId primary = config.primaryDevice();

    EXPECT_TRUE(primary.is_cuda());
    EXPECT_EQ(primary.ordinal, 0);
}

TEST(Test__TPDomainConfig, PrimaryDevice_ReturnsCPUWhenEmpty)
{
    TPDomainConfig config;
    config.name = "empty_domain";

    DeviceId primary = config.primaryDevice();

    EXPECT_TRUE(primary.is_cpu());
}

// =============================================================================
// Degree Tests
// =============================================================================

TEST(Test__TPDomainConfig, Degree_ReturnsDeviceCount)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.devices.push_back(DeviceId::rocm(0));

    EXPECT_EQ(config.degree(), 3);
}

// =============================================================================
// Homogeneous Tests
// =============================================================================

TEST(Test__TPDomainConfig, IsHomogeneous_TrueForSameDeviceType)
{
    TPDomainConfig config;
    config.name = "cuda_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.devices.push_back(DeviceId::cuda(2));

    EXPECT_TRUE(config.isHomogeneous());
}

TEST(Test__TPDomainConfig, IsHomogeneous_FalseForMixedTypes)
{
    TPDomainConfig config;
    config.name = "mixed_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::rocm(0));

    EXPECT_FALSE(config.isHomogeneous());
}

TEST(Test__TPDomainConfig, IsHomogeneous_TrueForSingleDevice)
{
    TPDomainConfig config;
    config.name = "single_device";
    config.devices.push_back(DeviceId::cuda(0));

    EXPECT_TRUE(config.isHomogeneous());
}

TEST(Test__TPDomainConfig, IsHomogeneous_TrueForEmptyDomain)
{
    TPDomainConfig config;
    config.name = "empty";

    EXPECT_TRUE(config.isHomogeneous());
}

// =============================================================================
// GPU Domain Tests
// =============================================================================

TEST(Test__TPDomainConfig, IsGPUDomain_TrueForAllGPUs)
{
    TPDomainConfig config;
    config.name = "gpu_only";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::rocm(0));

    EXPECT_TRUE(config.isGPUDomain());
}

TEST(Test__TPDomainConfig, IsGPUDomain_FalseForMixed)
{
    TPDomainConfig config;
    config.name = "mixed";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cpu());

    EXPECT_FALSE(config.isGPUDomain());
}

TEST(Test__TPDomainConfig, IsGPUDomain_FalseForEmpty)
{
    TPDomainConfig config;
    config.name = "empty";

    EXPECT_FALSE(config.isGPUDomain());
}

// =============================================================================
// CPU Domain Tests
// =============================================================================

TEST(Test__TPDomainConfig, IsCPUDomain_TrueForAllCPUs)
{
    TPDomainConfig config;
    config.name = "cpu_only";
    config.devices.push_back(DeviceId::cpu());

    EXPECT_TRUE(config.isCPUDomain());
}

TEST(Test__TPDomainConfig, IsCPUDomain_FalseForGPUs)
{
    TPDomainConfig config;
    config.name = "gpu_domain";
    config.devices.push_back(DeviceId::cuda(0));

    EXPECT_FALSE(config.isCPUDomain());
}

TEST(Test__TPDomainConfig, IsCPUDomain_FalseForEmpty)
{
    TPDomainConfig config;
    config.name = "empty";

    EXPECT_FALSE(config.isCPUDomain());
}

// =============================================================================
// Cross-Vendor Tests
// =============================================================================

TEST(Test__TPDomainConfig, IsCrossVendor_TrueForCUDAAndROCm)
{
    TPDomainConfig config;
    config.name = "cross_vendor";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::rocm(0));

    EXPECT_TRUE(config.isCrossVendor());
}

TEST(Test__TPDomainConfig, IsCrossVendor_FalseForSameVendor)
{
    TPDomainConfig config;
    config.name = "same_vendor";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));

    EXPECT_FALSE(config.isCrossVendor());
}

TEST(Test__TPDomainConfig, IsCrossVendor_FalseForCPUOnly)
{
    TPDomainConfig config;
    config.name = "cpu_only";
    config.devices.push_back(DeviceId::cpu());

    EXPECT_FALSE(config.isCrossVendor());
}

TEST(Test__TPDomainConfig, IsCrossVendor_FalseForROCmOnly)
{
    TPDomainConfig config;
    config.name = "rocm_only";
    config.devices.push_back(DeviceId::rocm(0));
    config.devices.push_back(DeviceId::rocm(1));

    EXPECT_FALSE(config.isCrossVendor());
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST(Test__TPDomainConfig, Validate_FailsOnEmptyName)
{
    TPDomainConfig config;
    config.name = "";
    config.devices.push_back(DeviceId::cuda(0));

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("name"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnNoDevices)
{
    TPDomainConfig config;
    config.name = "test_domain";

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("device"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnWeightCountMismatch)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.device_weights = {0.5f}; // Only one weight for two devices

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("weight count"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnWeightsSumNotOne)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.device_weights = {0.5f, 0.3f}; // Sum = 0.8, not 1.0

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("sum to 1.0"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnNegativeWeights)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.device_weights = {1.5f, -0.5f}; // Negative weight

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("negative"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_SucceedsForValidConfig)
{
    TPDomainConfig config;
    config.name = "valid_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));
    config.device_weights = {0.6f, 0.4f};
    config.tp_backend = CollectiveBackendType::NCCL;

    std::string error;
    EXPECT_TRUE(config.validate(&error));
    EXPECT_TRUE(error.empty());
}

TEST(Test__TPDomainConfig, Validate_SucceedsWithoutWeights)
{
    TPDomainConfig config;
    config.name = "no_weights";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::rocm(0));
    config.tp_backend = CollectiveBackendType::PCIE_BAR;

    std::string error;
    EXPECT_TRUE(config.validate(&error));
}

TEST(Test__TPDomainConfig, Validate_FailsOnNCCLWithNonCUDA)
{
    TPDomainConfig config;
    config.name = "nccl_with_rocm";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::rocm(0));
    config.tp_backend = CollectiveBackendType::NCCL;

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("NCCL"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnRCCLWithNonROCm)
{
    TPDomainConfig config;
    config.name = "rccl_with_cuda";
    config.devices.push_back(DeviceId::rocm(0));
    config.devices.push_back(DeviceId::cuda(0));
    config.tp_backend = CollectiveBackendType::RCCL;

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("RCCL"), std::string::npos);
}

TEST(Test__TPDomainConfig, Validate_FailsOnInvalidDevice)
{
    TPDomainConfig config;
    config.name = "invalid_device";
    config.devices.push_back(DeviceId::invalid());

    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_NE(error.find("invalid"), std::string::npos);
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST(Test__TPDomainConfig, CreateTPContext_ReturnsNullptrForNow)
{
    TPDomainConfig config;
    config.name = "test_domain";
    config.devices.push_back(DeviceId::cuda(0));
    config.devices.push_back(DeviceId::cuda(1));

    auto context = config.createTPContext();

    // Currently unimplemented - should return nullptr
    EXPECT_EQ(context, nullptr);
}
