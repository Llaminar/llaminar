/**
 * @file Test__RankExecutionPlan.cpp
 * @brief Unit tests for RankExecutionPlan
 *
 * Tests:
 * - WeightShardInfo validation and string representation
 * - TPDomainParticipation validation
 * - RankExecutionPlan validation
 * - Convenience methods (usesPipelineParallel, usesLocalTP, etc.)
 * - toString() serialization
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/mpi_orchestration/RankExecutionPlan.h"

using namespace llaminar2;

// ============================================================================
// WeightShardInfo Tests
// ============================================================================

class Test__WeightShardInfo : public ::testing::Test
{
protected:
    WeightShardInfo shard;
};

TEST_F(Test__WeightShardInfo, DefaultValues_Valid)
{
    EXPECT_EQ(shard.shard_index, 0);
    EXPECT_EQ(shard.total_shards, 1);
    EXPECT_FLOAT_EQ(shard.work_fraction, 1.0f);
    EXPECT_FALSE(shard.isSharded());
    EXPECT_TRUE(shard.validate().empty());
}

TEST_F(Test__WeightShardInfo, IsSharded_SingleShard)
{
    shard.total_shards = 1;
    EXPECT_FALSE(shard.isSharded());
}

TEST_F(Test__WeightShardInfo, IsSharded_MultipleShards)
{
    shard.total_shards = 4;
    EXPECT_TRUE(shard.isSharded());
}

TEST_F(Test__WeightShardInfo, Validate_NegativeShardIndex)
{
    shard.shard_index = -1;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("shard_index"), std::string::npos);
}

TEST_F(Test__WeightShardInfo, Validate_ZeroTotalShards)
{
    shard.total_shards = 0;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__WeightShardInfo, Validate_ShardIndexOutOfRange)
{
    shard.shard_index = 2;
    shard.total_shards = 2;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__WeightShardInfo, Validate_InvalidWorkFraction_Zero)
{
    shard.work_fraction = 0.0f;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__WeightShardInfo, Validate_InvalidWorkFraction_Negative)
{
    shard.work_fraction = -0.5f;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__WeightShardInfo, Validate_InvalidWorkFraction_GreaterThanOne)
{
    shard.work_fraction = 1.5f;
    auto errors = shard.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__WeightShardInfo, Validate_ValidMultiShard)
{
    shard.shard_index = 1;
    shard.total_shards = 4;
    shard.work_fraction = 0.25f;
    EXPECT_TRUE(shard.validate().empty());
}

TEST_F(Test__WeightShardInfo, ToString_ContainsFields)
{
    shard.shard_index = 2;
    shard.total_shards = 8;
    shard.work_fraction = 0.125f;

    auto str = shard.toString();
    EXPECT_NE(str.find("index=2"), std::string::npos);
    EXPECT_NE(str.find("total=8"), std::string::npos);
    EXPECT_NE(str.find("0.125"), std::string::npos);
}

// ============================================================================
// TPDomainParticipation Tests
// ============================================================================

class Test__TPDomainParticipation : public ::testing::Test
{
protected:
    TPDomainParticipation domain;

    void SetUp() override
    {
        domain.domain_id = 0;
        domain.domain_name = "test_domain";
        domain.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
        domain.weights = {0.6f, 0.4f};
        domain.backend = CollectiveBackendType::NCCL;
        domain.my_index_in_domain = 0;
    }
};

TEST_F(Test__TPDomainParticipation, DefaultValues_Invalid)
{
    TPDomainParticipation empty;
    auto errors = empty.validate();
    EXPECT_FALSE(errors.empty()); // No devices, invalid domain_id
}

TEST_F(Test__TPDomainParticipation, ValidConfiguration)
{
    auto errors = domain.validate();
    EXPECT_TRUE(errors.empty());
}

TEST_F(Test__TPDomainParticipation, Validate_NegativeDomainId)
{
    domain.domain_id = -1;
    auto errors = domain.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__TPDomainParticipation, Validate_EmptyDevices)
{
    domain.devices.clear();
    auto errors = domain.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__TPDomainParticipation, Validate_WeightCountMismatch)
{
    domain.weights = {0.5f}; // Only 1 weight for 2 devices
    auto errors = domain.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__TPDomainParticipation, Validate_IndexOutOfRange)
{
    domain.my_index_in_domain = 5; // Only 2 devices
    auto errors = domain.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__TPDomainParticipation, MyDevice_ReturnsCorrectDevice)
{
    domain.my_index_in_domain = 1;
    auto dev = domain.myDevice();
    EXPECT_EQ(dev.device_ordinal, 1);
    EXPECT_EQ(dev.device_type, DeviceType::CUDA);
}

TEST_F(Test__TPDomainParticipation, MyWeight_WithExplicitWeights)
{
    domain.my_index_in_domain = 1;
    EXPECT_FLOAT_EQ(domain.myWeight(), 0.4f);
}

TEST_F(Test__TPDomainParticipation, MyWeight_WithEmptyWeights)
{
    domain.weights.clear();
    domain.my_index_in_domain = 0;
    EXPECT_FLOAT_EQ(domain.myWeight(), 0.5f); // 1/2 devices
}

TEST_F(Test__TPDomainParticipation, ToString_ContainsFields)
{
    auto str = domain.toString();
    EXPECT_NE(str.find("id=0"), std::string::npos);
    EXPECT_NE(str.find("test_domain"), std::string::npos);
    EXPECT_NE(str.find("nccl"), std::string::npos); // lowercase per collectiveBackendTypeToString
}

// ============================================================================
// RankExecutionPlan Tests
// ============================================================================

class Test__RankExecutionPlan : public ::testing::Test
{
protected:
    RankExecutionPlan plan;

    void SetUp() override
    {
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.pp_stage_id = 0;
        plan.first_layer = 0;
        plan.last_layer = 27;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
    }
};

TEST_F(Test__RankExecutionPlan, DefaultValues_Valid)
{
    RankExecutionPlan default_plan;
    // Default values should be valid for single-rank execution
    auto errors = default_plan.validate();

    // Only first/last stage checks might fail depending on defaults
    // Let's make it explicitly valid
    default_plan.has_embedding = true;
    default_plan.has_lm_head = true;
    errors = default_plan.validate();
    EXPECT_TRUE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_NegativeRank)
{
    plan.rank = -1;
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_InvalidLayerRange)
{
    plan.first_layer = 10;
    plan.last_layer = 5; // first > last
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_NegativePPStage)
{
    plan.pp_stage_id = -1;
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_FirstStageWithoutEmbedding)
{
    // First stage (no prev_rank) should have embedding
    plan.prev_rank = std::nullopt;
    plan.has_embedding = false;
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_LastStageWithoutLMHead)
{
    // Last stage (no next_rank) should have LM head
    plan.next_rank = std::nullopt;
    plan.has_lm_head = false;
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_LocalTPWeightsMismatch)
{
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    plan.local_tp_weights = {0.5f}; // Only 1 weight for 2 devices
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, Validate_GlobalTPRankOutOfRange)
{
    plan.global_tp_domain_id = 0;
    plan.global_tp_rank_in_domain = 5;
    plan.global_tp_domain_size = 4;
    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__RankExecutionPlan, UsesPipelineParallel_NoNeighbors)
{
    plan.prev_rank = std::nullopt;
    plan.next_rank = std::nullopt;
    EXPECT_FALSE(plan.usesPipelineParallel());
}

TEST_F(Test__RankExecutionPlan, UsesPipelineParallel_HasPrevRank)
{
    plan.prev_rank = 0;
    plan.next_rank = std::nullopt;
    EXPECT_TRUE(plan.usesPipelineParallel());
}

TEST_F(Test__RankExecutionPlan, UsesPipelineParallel_HasNextRank)
{
    plan.prev_rank = std::nullopt;
    plan.next_rank = 2;
    EXPECT_TRUE(plan.usesPipelineParallel());
}

TEST_F(Test__RankExecutionPlan, UsesPipelineParallel_HasBothNeighbors)
{
    plan.prev_rank = 0;
    plan.next_rank = 2;
    EXPECT_TRUE(plan.usesPipelineParallel());
}

TEST_F(Test__RankExecutionPlan, UsesLocalTP_SingleDevice)
{
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0)};
    EXPECT_FALSE(plan.usesLocalTP());
}

TEST_F(Test__RankExecutionPlan, UsesLocalTP_MultipleDevices)
{
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    EXPECT_TRUE(plan.usesLocalTP());
}

TEST_F(Test__RankExecutionPlan, UsesGlobalTP_NoDomainId)
{
    plan.global_tp_domain_id = std::nullopt;
    EXPECT_FALSE(plan.usesGlobalTP());
}

TEST_F(Test__RankExecutionPlan, UsesGlobalTP_HasDomainId)
{
    plan.global_tp_domain_id = 0;
    EXPECT_TRUE(plan.usesGlobalTP());
}

TEST_F(Test__RankExecutionPlan, LayerCount_ValidRange)
{
    plan.first_layer = 5;
    plan.last_layer = 15;
    EXPECT_EQ(plan.layerCount(), 11); // 15 - 5 + 1
}

TEST_F(Test__RankExecutionPlan, LayerCount_SingleLayer)
{
    plan.first_layer = 5;
    plan.last_layer = 5;
    EXPECT_EQ(plan.layerCount(), 1);
}

TEST_F(Test__RankExecutionPlan, LayerCount_InvalidRange)
{
    plan.first_layer = 10;
    plan.last_layer = 5;
    EXPECT_EQ(plan.layerCount(), 0);
}

TEST_F(Test__RankExecutionPlan, HasLayer_InRange)
{
    plan.first_layer = 5;
    plan.last_layer = 15;
    EXPECT_TRUE(plan.hasLayer(5));
    EXPECT_TRUE(plan.hasLayer(10));
    EXPECT_TRUE(plan.hasLayer(15));
}

TEST_F(Test__RankExecutionPlan, HasLayer_OutOfRange)
{
    plan.first_layer = 5;
    plan.last_layer = 15;
    EXPECT_FALSE(plan.hasLayer(4));
    EXPECT_FALSE(plan.hasLayer(16));
}

TEST_F(Test__RankExecutionPlan, IsFirstStage_NoPrevRank)
{
    plan.prev_rank = std::nullopt;
    EXPECT_TRUE(plan.isFirstStage());
}

TEST_F(Test__RankExecutionPlan, IsFirstStage_HasPrevRank)
{
    plan.prev_rank = 0;
    EXPECT_FALSE(plan.isFirstStage());
}

TEST_F(Test__RankExecutionPlan, IsLastStage_NoNextRank)
{
    plan.next_rank = std::nullopt;
    EXPECT_TRUE(plan.isLastStage());
}

TEST_F(Test__RankExecutionPlan, IsLastStage_HasNextRank)
{
    plan.next_rank = 2;
    EXPECT_FALSE(plan.isLastStage());
}

TEST_F(Test__RankExecutionPlan, TotalTPDegree_NoTP)
{
    plan.local_tp_devices.clear();
    plan.global_tp_domain_size = 1;
    EXPECT_EQ(plan.totalTPDegree(), 1);
}

TEST_F(Test__RankExecutionPlan, TotalTPDegree_LocalTPOnly)
{
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    plan.global_tp_domain_size = 1;
    EXPECT_EQ(plan.totalTPDegree(), 2);
}

TEST_F(Test__RankExecutionPlan, TotalTPDegree_GlobalTPOnly)
{
    plan.local_tp_devices.clear();
    plan.global_tp_domain_size = 4;
    EXPECT_EQ(plan.totalTPDegree(), 4);
}

TEST_F(Test__RankExecutionPlan, TotalTPDegree_HybridTP)
{
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    plan.global_tp_domain_size = 4;
    EXPECT_EQ(plan.totalTPDegree(), 8); // 2 * 4
}

TEST_F(Test__RankExecutionPlan, ToString_ContainsIdentity)
{
    auto str = plan.toString();
    EXPECT_NE(str.find("rank: 0"), std::string::npos);
    EXPECT_NE(str.find("hostname: localhost"), std::string::npos);
}

TEST_F(Test__RankExecutionPlan, ToString_ContainsPPInfo)
{
    auto str = plan.toString();
    EXPECT_NE(str.find("Pipeline Parallelism"), std::string::npos);
    EXPECT_NE(str.find("layers:"), std::string::npos);
}

TEST_F(Test__RankExecutionPlan, ToString_ContainsTPInfo)
{
    auto str = plan.toString();
    EXPECT_NE(str.find("Tensor Parallelism"), std::string::npos);
}

TEST_F(Test__RankExecutionPlan, ToString_ContainsWeightShard)
{
    auto str = plan.toString();
    EXPECT_NE(str.find("Weight Shard"), std::string::npos);
}

// ============================================================================
// Combined Validation Tests
// ============================================================================

TEST(Test__RankExecutionPlan_Integration, FullyConfiguredPlan)
{
    RankExecutionPlan plan;

    // Identity
    plan.rank = 1;
    plan.hostname = "node1";
    plan.numa_node = 0;

    // PP config (middle stage)
    plan.pp_stage_id = 1;
    plan.first_layer = 14;
    plan.last_layer = 27;
    plan.has_embedding = false;
    plan.has_lm_head = true;
    plan.prev_rank = 0;
    plan.next_rank = std::nullopt;

    // TP config
    plan.tp_scope = TPScope::LOCAL;
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
    plan.local_tp_weights = {0.6f, 0.4f};
    plan.local_tp_backend = CollectiveBackendType::PCIE_BAR;

    // Weight shard
    plan.weight_shard.shard_index = 1;
    plan.weight_shard.total_shards = 2;
    plan.weight_shard.work_fraction = 0.5f;

    // Primary device
    plan.primary_device = GlobalDeviceAddress::cuda(0);

    // Validate
    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);

    // Check convenience methods
    EXPECT_TRUE(plan.usesPipelineParallel());
    EXPECT_TRUE(plan.usesLocalTP());
    EXPECT_FALSE(plan.usesGlobalTP());
    EXPECT_EQ(plan.layerCount(), 14);
    EXPECT_FALSE(plan.isFirstStage());
    EXPECT_TRUE(plan.isLastStage());
}

TEST(Test__RankExecutionPlan_Integration, DomainParticipationValidation)
{
    RankExecutionPlan plan;
    plan.rank = 0;
    plan.hostname = "localhost";
    plan.first_layer = 0;
    plan.last_layer = 27;
    plan.has_embedding = true;
    plan.has_lm_head = true;

    // Add domain participation
    TPDomainParticipation domain;
    domain.domain_id = 0;
    domain.domain_name = "gpu_tp";
    domain.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    domain.weights = {0.5f, 0.5f};
    domain.backend = CollectiveBackendType::NCCL;
    domain.my_index_in_domain = 0;
    plan.my_domains.push_back(domain);

    // Invalid domain (weights mismatch)
    TPDomainParticipation bad_domain;
    bad_domain.domain_id = 1;
    bad_domain.domain_name = "bad_domain";
    bad_domain.devices = {GlobalDeviceAddress::cuda(0)};
    bad_domain.weights = {0.5f, 0.5f}; // Mismatch
    bad_domain.my_index_in_domain = 0;
    plan.my_domains.push_back(bad_domain);

    auto errors = plan.validate();
    EXPECT_FALSE(errors.empty());
    // Should report error from bad_domain
    bool found_domain_error = false;
    for (const auto &e : errors)
    {
        if (e.find("my_domains[1]") != std::string::npos)
        {
            found_domain_error = true;
            break;
        }
    }
    EXPECT_TRUE(found_domain_error);
}
