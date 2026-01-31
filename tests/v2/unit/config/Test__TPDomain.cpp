/**
 * @file Test__TPDomain.cpp
 * @brief Unit tests for TPDomain and MultiDomainTPConfig
 *
 * Tests the tensor parallel domain system WITHOUT requiring MPI.
 * Uses createForTest() factory method to test domain logic in isolation.
 *
 * For MPI integration tests, see Test__TPDomainMPI.cpp
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/TPDomain.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// TPDomainType Tests
// =============================================================================

TEST(Test__TPDomain, TypeToString)
{
    EXPECT_STREQ(tpDomainTypeToString(TPDomainType::GPU_INTRA_RANK), "GPU_INTRA_RANK");
    EXPECT_STREQ(tpDomainTypeToString(TPDomainType::CPU_CROSS_RANK), "CPU_CROSS_RANK");
}

// =============================================================================
// TPDomain Basic Tests
// =============================================================================

TEST(Test__TPDomain, DomainIsTrivial)
{
    // Trivial domain (size <= 1)
    TPDomain trivial_domain;
    trivial_domain.domain_size = 1;
    EXPECT_TRUE(trivial_domain.isTrivial());

    // Non-trivial domain (size > 1)
    TPDomain multi_domain;
    multi_domain.domain_size = 2;
    EXPECT_FALSE(multi_domain.isTrivial());

    // Empty domain (size 0) is also trivial
    TPDomain empty_domain;
    empty_domain.domain_size = 0;
    EXPECT_TRUE(empty_domain.isTrivial());
}

TEST(Test__TPDomain, DomainIsCrossRank)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    EXPECT_FALSE(gpu_domain.isCrossRank());

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    EXPECT_TRUE(cpu_domain.isCrossRank());
}

TEST(Test__TPDomain, DomainToString)
{
    TPDomain domain;
    domain.type = TPDomainType::GPU_INTRA_RANK;
    domain.name = "gpu_test";
    domain.domain_size = 2;
    domain.local_rank_in_domain = 0;
    domain.devices.push_back(DeviceId::cuda(0));
    domain.devices.push_back(DeviceId::rocm(0));
    domain.communicator = MPI_COMM_NULL;

    std::string str = domain.toString();

    // Verify key information is present
    EXPECT_NE(str.find("gpu_test"), std::string::npos);
    EXPECT_NE(str.find("GPU_INTRA_RANK"), std::string::npos);
    EXPECT_NE(str.find("size=2"), std::string::npos);
    EXPECT_NE(str.find("CUDA:0"), std::string::npos); // DeviceId::to_string() uses uppercase
    EXPECT_NE(str.find("ROCm:0"), std::string::npos); // DeviceId::to_string() uses ROCm (mixed case)
}

TEST(Test__TPDomain, DomainIsValid)
{
    // Invalid: no devices
    TPDomain empty_domain;
    empty_domain.domain_size = 1;
    EXPECT_FALSE(empty_domain.isValid());

    // Invalid: size is 0
    TPDomain zero_size;
    zero_size.domain_size = 0;
    zero_size.devices.push_back(DeviceId::cuda(0));
    EXPECT_FALSE(zero_size.isValid());

    // Valid: has devices and size > 0
    TPDomain valid_domain;
    valid_domain.domain_size = 1;
    valid_domain.devices.push_back(DeviceId::cuda(0));
    EXPECT_TRUE(valid_domain.isValid());
}

// =============================================================================
// MultiDomainTPConfig::createForTest Tests
// =============================================================================

TEST(Test__TPDomain, CreateForTestSingleGPUDomain)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu_single";
    gpu_domain.domain_size = 1;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices.push_back(DeviceId::cuda(0));
    gpu_domain.communicator = MPI_COMM_NULL;

    auto config = MultiDomainTPConfig::createForTest({gpu_domain});

    EXPECT_EQ(config.domains().size(), 1);
    ASSERT_NE(config.gpuDomain(), nullptr);
    EXPECT_EQ(config.gpuDomain()->name, "gpu_single");
    EXPECT_EQ(config.cpuDomain(), nullptr);
}

TEST(Test__TPDomain, CreateForTestDualGPUDomain)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu_dual";
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices.push_back(DeviceId::cuda(0));
    gpu_domain.devices.push_back(DeviceId::rocm(0));
    gpu_domain.communicator = MPI_COMM_NULL;

    auto config = MultiDomainTPConfig::createForTest({gpu_domain});

    EXPECT_EQ(config.domains().size(), 1);
    ASSERT_NE(config.gpuDomain(), nullptr);
    EXPECT_EQ(config.gpuDomain()->domain_size, 2);
    EXPECT_EQ(config.gpuDomain()->devices.size(), 2);
    EXPECT_FALSE(config.gpuDomain()->isTrivial());
}

TEST(Test__TPDomain, CreateForTestCPUDomain)
{
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "cpu_cross";
    cpu_domain.domain_size = 2;
    cpu_domain.local_rank_in_domain = 0;
    cpu_domain.devices.push_back(DeviceId::cpu());
    cpu_domain.communicator = MPI_COMM_NULL; // Would be real comm in actual use

    auto config = MultiDomainTPConfig::createForTest({cpu_domain});

    EXPECT_EQ(config.domains().size(), 1);
    EXPECT_EQ(config.gpuDomain(), nullptr);
    ASSERT_NE(config.cpuDomain(), nullptr);
    EXPECT_EQ(config.cpuDomain()->name, "cpu_cross");
    EXPECT_TRUE(config.cpuDomain()->isCrossRank());
}

TEST(Test__TPDomain, CreateForTestMultipleDomains)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu_tp";
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices.push_back(DeviceId::cuda(0));
    gpu_domain.devices.push_back(DeviceId::rocm(0));
    gpu_domain.communicator = MPI_COMM_NULL;

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "cpu_tp";
    cpu_domain.domain_size = 2;
    cpu_domain.local_rank_in_domain = 0;
    cpu_domain.devices.push_back(DeviceId::cpu());
    cpu_domain.communicator = MPI_COMM_NULL;

    auto config = MultiDomainTPConfig::createForTest({gpu_domain, cpu_domain});

    EXPECT_EQ(config.domains().size(), 2);
    ASSERT_NE(config.gpuDomain(), nullptr);
    ASSERT_NE(config.cpuDomain(), nullptr);
    EXPECT_EQ(config.gpuDomain()->name, "gpu_tp");
    EXPECT_EQ(config.cpuDomain()->name, "cpu_tp");
}

// =============================================================================
// MultiDomainTPConfig Access Tests
// =============================================================================

TEST(Test__TPDomain, ConfigGpuDomainAccess)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "my_gpu";
    gpu_domain.domain_size = 1;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    auto config = MultiDomainTPConfig::createForTest({gpu_domain});

    const TPDomain *gpu = config.gpuDomain();
    ASSERT_NE(gpu, nullptr);
    EXPECT_EQ(gpu->name, "my_gpu");
    EXPECT_EQ(gpu->type, TPDomainType::GPU_INTRA_RANK);
}

TEST(Test__TPDomain, ConfigCpuDomainAccess)
{
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "my_cpu";
    cpu_domain.domain_size = 2;
    cpu_domain.devices.push_back(DeviceId::cpu());

    auto config = MultiDomainTPConfig::createForTest({cpu_domain});

    const TPDomain *cpu = config.cpuDomain();
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(cpu->name, "my_cpu");
    EXPECT_EQ(cpu->type, TPDomainType::CPU_CROSS_RANK);
}

// =============================================================================
// DomainForLayer Tests
// =============================================================================

TEST(Test__TPDomain, ConfigDomainForLayerAttention)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "cpu";
    cpu_domain.domain_size = 2;
    cpu_domain.devices.push_back(DeviceId::cpu());

    auto config = MultiDomainTPConfig::createForTest({gpu_domain, cpu_domain});

    // Default: attention uses GPU domain
    const TPDomain *domain = config.domainForLayer(0, /*is_attention=*/true);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
}

TEST(Test__TPDomain, ConfigDomainForLayerFFN)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "cpu";
    cpu_domain.domain_size = 2;
    cpu_domain.devices.push_back(DeviceId::cpu());

    auto config = MultiDomainTPConfig::createForTest({gpu_domain, cpu_domain});

    // Default: FFN uses CPU domain if available
    const TPDomain *domain = config.domainForLayer(0, /*is_attention=*/false);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::CPU_CROSS_RANK);
}

TEST(Test__TPDomain, ConfigDomainForLayerFFNFallsBackToGPU)
{
    // Only GPU domain available
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    auto config = MultiDomainTPConfig::createForTest({gpu_domain});

    // No CPU domain, so FFN falls back to GPU
    const TPDomain *domain = config.domainForLayer(0, /*is_attention=*/false);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->type, TPDomainType::GPU_INTRA_RANK);
}

// =============================================================================
// HasCrossRankTP Tests
// =============================================================================

TEST(Test__TPDomain, ConfigHasCrossRankTP)
{
    // Case 1: Only GPU domain (no cross-rank)
    {
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.name = "gpu";
        gpu_domain.domain_size = 2;
        gpu_domain.devices.push_back(DeviceId::cuda(0));

        auto config = MultiDomainTPConfig::createForTest({gpu_domain});
        EXPECT_FALSE(config.hasCrossRankTP());
    }

    // Case 2: Trivial CPU domain (size 1)
    {
        TPDomain cpu_domain;
        cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
        cpu_domain.name = "cpu";
        cpu_domain.domain_size = 1; // Trivial
        cpu_domain.devices.push_back(DeviceId::cpu());

        auto config = MultiDomainTPConfig::createForTest({cpu_domain});
        EXPECT_FALSE(config.hasCrossRankTP());
    }

    // Case 3: Non-trivial CPU domain (cross-rank)
    {
        TPDomain cpu_domain;
        cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
        cpu_domain.name = "cpu";
        cpu_domain.domain_size = 2; // Non-trivial
        cpu_domain.devices.push_back(DeviceId::cpu());

        auto config = MultiDomainTPConfig::createForTest({cpu_domain});
        EXPECT_TRUE(config.hasCrossRankTP());
    }
}

// =============================================================================
// ToString Tests
// =============================================================================

TEST(Test__TPDomain, ConfigToStringNotEmpty)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu_test";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));
    gpu_domain.devices.push_back(DeviceId::rocm(0));

    auto config = MultiDomainTPConfig::createForTest({gpu_domain});

    std::string str = config.toString();

    // Should contain useful information
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("MultiDomainTPConfig"), std::string::npos);
    EXPECT_NE(str.find("gpu_test"), std::string::npos);
}

// =============================================================================
// SetLayerDomainMapping Tests
// =============================================================================

TEST(Test__TPDomain, SetLayerDomainMapping)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.name = "cpu";
    cpu_domain.domain_size = 2;
    cpu_domain.devices.push_back(DeviceId::cpu());

    auto config = MultiDomainTPConfig::createForTest({gpu_domain, cpu_domain});

    // Map specific layers
    std::unordered_map<int, TPDomainType> attention_map;
    attention_map[5] = TPDomainType::CPU_CROSS_RANK; // Layer 5 attention on CPU

    std::unordered_map<int, TPDomainType> ffn_map;
    ffn_map[10] = TPDomainType::GPU_INTRA_RANK; // Layer 10 FFN on GPU

    config.setLayerDomainMapping(attention_map, ffn_map);

    // Verify mapped layers use custom domain
    const TPDomain *layer5_attn = config.domainForLayer(5, true);
    ASSERT_NE(layer5_attn, nullptr);
    EXPECT_EQ(layer5_attn->type, TPDomainType::CPU_CROSS_RANK);

    const TPDomain *layer10_ffn = config.domainForLayer(10, false);
    ASSERT_NE(layer10_ffn, nullptr);
    EXPECT_EQ(layer10_ffn->type, TPDomainType::GPU_INTRA_RANK);

    // Non-mapped layers use default
    const TPDomain *layer0_attn = config.domainForLayer(0, true);
    ASSERT_NE(layer0_attn, nullptr);
    EXPECT_EQ(layer0_attn->type, TPDomainType::GPU_INTRA_RANK); // Default for attention
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(Test__TPDomain, ConfigMoveConstruction)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    auto config1 = MultiDomainTPConfig::createForTest({gpu_domain});
    ASSERT_NE(config1.gpuDomain(), nullptr);

    // Move construct
    auto config2 = std::move(config1);

    // config2 should have the data
    ASSERT_NE(config2.gpuDomain(), nullptr);
    EXPECT_EQ(config2.gpuDomain()->name, "gpu");
    EXPECT_EQ(config2.domains().size(), 1);
}

TEST(Test__TPDomain, ConfigMoveAssignment)
{
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu";
    gpu_domain.domain_size = 2;
    gpu_domain.devices.push_back(DeviceId::cuda(0));

    auto config1 = MultiDomainTPConfig::createForTest({gpu_domain});
    MultiDomainTPConfig config2;

    // Move assign
    config2 = std::move(config1);

    // config2 should have the data
    ASSERT_NE(config2.gpuDomain(), nullptr);
    EXPECT_EQ(config2.gpuDomain()->name, "gpu");
}

// =============================================================================
// Empty Config Tests
// =============================================================================

TEST(Test__TPDomain, EmptyConfig)
{
    auto config = MultiDomainTPConfig::createForTest({});

    EXPECT_TRUE(config.domains().empty());
    EXPECT_EQ(config.gpuDomain(), nullptr);
    EXPECT_EQ(config.cpuDomain(), nullptr);
    EXPECT_FALSE(config.hasCrossRankTP());

    // domainForLayer returns nullptr for empty config
    EXPECT_EQ(config.domainForLayer(0, true), nullptr);
    EXPECT_EQ(config.domainForLayer(0, false), nullptr);
}
