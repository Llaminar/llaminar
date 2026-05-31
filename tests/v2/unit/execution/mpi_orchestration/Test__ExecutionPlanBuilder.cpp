/**
 * @file Test__ExecutionPlanBuilder.cpp
 * @brief Unit tests for ExecutionPlanBuilder
 *
 * Tests:
 * - Building plans for single device (no TP, no PP)
 * - Building plans for LOCAL TP (multiple devices on one rank)
 * - Building plans for GLOBAL TP (multiple ranks)
 * - Building plans for PP (multiple stages)
 * - Building plans with named domains (Scenario 7 style)
 * - Weight shard calculation
 * - Backend selection
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "execution/global_pp/GlobalPPTopology.h"

using namespace llaminar2;

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Helper to create a minimal ClusterInventory for testing
 */
class ClusterInventoryBuilder
{
public:
    ClusterInventoryBuilder &addRank(
        int rank,
        const std::string &hostname,
        int numa_node,
        std::vector<std::pair<DeviceType, int>> gpus = {})
    {
        RankInventory rank_inv;
        rank_inv.rank = rank;
        rank_inv.hostname = hostname;
        rank_inv.node_id = rank; // Simple: one rank per node
        rank_inv.local_rank = 0;
        rank_inv.numa_nodes = 2;

        for (const auto &[type, ordinal] : gpus)
        {
            DeviceInfo gpu;
            gpu.type = type;
            gpu.local_device_id = ordinal;
            gpu.numa_node = numa_node;
            gpu.memory_bytes = 16ULL * 1024 * 1024 * 1024; // 16GB
            gpu.compute_units = 108;
            gpu.tflops_fp16 = 100.0f;
            rank_inv.gpus.push_back(gpu);
        }

        ranks_.push_back(std::move(rank_inv));
        return *this;
    }

    ClusterInventory build()
    {
        ClusterInventory cluster;
        cluster.world_size = static_cast<int>(ranks_.size());
        cluster.ranks = std::move(ranks_);
        cluster.buildNodeAggregations();
        return cluster;
    }

private:
    std::vector<RankInventory> ranks_;
};

class Test__ExecutionPlanBuilder : public ::testing::Test
{
protected:
    std::unique_ptr<IExecutionPlanBuilder> builder;
    ModelConfig model;
    OrchestrationConfig config;

    void SetUp() override
    {
        builder = createExecutionPlanBuilder();
        model = ModelConfig::qwen2_7b();
        config = OrchestrationConfig::defaults();
    }
};

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, CreateBuilder_ReturnsNonNull)
{
    EXPECT_NE(builder, nullptr);
}

// ============================================================================
// Single Device Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_SingleRank_SingleGPU)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.rank, 0);
    EXPECT_EQ(plan.hostname, "localhost");
    EXPECT_EQ(plan.first_layer, 0);
    EXPECT_EQ(plan.last_layer, model.n_layers - 1);
    EXPECT_TRUE(plan.has_embedding);
    EXPECT_TRUE(plan.has_lm_head);
    EXPECT_FALSE(plan.usesPipelineParallel());
    EXPECT_FALSE(plan.usesGlobalTP());

    // Validate the plan
    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_SingleRank_NoGPU_UsesCPU)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {}) // No GPUs
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_TRUE(plan.primary_device.isCPU());

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Local TP Tests (Multiple Devices on One Rank)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_LocalTP_TwoGPUs)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::LOCAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.local_tp_devices.size(), 2);
    EXPECT_TRUE(plan.usesLocalTP());
    EXPECT_FALSE(plan.usesGlobalTP());
    EXPECT_EQ(plan.local_tp_backend, CollectiveBackendType::NCCL);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_LocalTP_MixedGPUs_UsesPCIeBAR)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::LOCAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_EQ(plan.local_tp_devices.size(), 2);
    EXPECT_EQ(plan.local_tp_backend, CollectiveBackendType::HETEROGENEOUS);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Global TP Tests (Multiple Ranks)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_GlobalTP_TwoRanks)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::GLOBAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Both ranks should have all layers (TP shards work, not layers)
    for (int r = 0; r < 2; ++r)
    {
        const auto &plan = plans[r];
        EXPECT_EQ(plan.rank, r);
        EXPECT_EQ(plan.first_layer, 0);
        EXPECT_EQ(plan.last_layer, model.n_layers - 1);
        EXPECT_TRUE(plan.usesGlobalTP());
        EXPECT_EQ(plan.global_tp_rank_in_domain, r % 2);

        auto errors = plan.validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NodeLocalTP_TwoRanksUsesCrossRankDomain)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {})
                       .addRank(1, "localhost", 1, {})
                       .build();

    config.tp_degree = 2;
    config.tp_scope = TPScope::NODE_LOCAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);
    for (int r = 0; r < 2; ++r)
    {
        const auto &plan = plans[r];
        EXPECT_TRUE(plan.usesGlobalTP());
        EXPECT_EQ(plan.tp_scope, TPScope::NODE_LOCAL);
        EXPECT_EQ(plan.global_tp_domain_size, 2);
        EXPECT_EQ(plan.global_tp_rank_in_domain, r);
        EXPECT_EQ(plan.weight_shard.total_shards, 2);
        EXPECT_EQ(plan.weight_shard.shard_index, r);

        auto errors = plan.validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

// ============================================================================
// Pipeline Parallelism Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_PP_TwoStages)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 2;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Stage 0: first half of layers
    const auto &plan0 = plans[0];
    EXPECT_EQ(plan0.pp_stage_id, 0);
    EXPECT_EQ(plan0.first_layer, 0);
    EXPECT_TRUE(plan0.has_embedding);
    EXPECT_FALSE(plan0.has_lm_head);
    EXPECT_FALSE(plan0.prev_rank.has_value());
    EXPECT_TRUE(plan0.next_rank.has_value());
    EXPECT_EQ(*plan0.next_rank, 1);

    // Stage 1: second half of layers
    const auto &plan1 = plans[1];
    EXPECT_EQ(plan1.pp_stage_id, 1);
    EXPECT_EQ(plan1.last_layer, model.n_layers - 1);
    EXPECT_FALSE(plan1.has_embedding);
    EXPECT_TRUE(plan1.has_lm_head);
    EXPECT_TRUE(plan1.prev_rank.has_value());
    EXPECT_EQ(*plan1.prev_rank, 0);
    EXPECT_FALSE(plan1.next_rank.has_value());

    // Verify layers are split evenly (28 layers / 2 = 14 each)
    EXPECT_EQ(plan0.layerCount() + plan1.layerCount(), model.n_layers);

    for (int r = 0; r < 2; ++r)
    {
        auto errors = plans[r].validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_PP_FourStages)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "node0", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "node0", 1, {{DeviceType::CUDA, 0}})
                       .addRank(2, "node1", 0, {{DeviceType::CUDA, 0}})
                       .addRank(3, "node1", 1, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 4;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 4);

    // Verify layer coverage is complete
    int total_layers = 0;
    for (const auto &plan : plans)
    {
        total_layers += plan.layerCount();

        auto errors = plan.validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << plan.rank << " errors: " << (errors.empty() ? "" : errors[0]);
    }
    EXPECT_EQ(total_layers, model.n_layers);

    // Verify PP chain
    EXPECT_FALSE(plans[0].prev_rank.has_value());
    EXPECT_TRUE(plans[0].next_rank.has_value());

    EXPECT_TRUE(plans[1].prev_rank.has_value());
    EXPECT_TRUE(plans[1].next_rank.has_value());

    EXPECT_TRUE(plans[2].prev_rank.has_value());
    EXPECT_TRUE(plans[2].next_rank.has_value());

    EXPECT_TRUE(plans[3].prev_rank.has_value());
    EXPECT_FALSE(plans[3].next_rank.has_value());
}

// ============================================================================
// Named Domain Tests (Scenario 7 Style)
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_TwoDomainsPP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    // Define two domains
    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};
    gpu0.backend = CollectiveBackendType::NCCL;

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(0, 1)};
    gpu1.backend = CollectiveBackendType::NCCL;

    config.domain_definitions = {gpu0, gpu1};

    // Define PP stages
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Verify domain-based assignment
    EXPECT_EQ(plans[0].first_layer, 0);
    EXPECT_EQ(plans[0].last_layer, 13);
    EXPECT_TRUE(plans[0].has_embedding);

    EXPECT_EQ(plans[1].first_layer, 14);
    EXPECT_EQ(plans[1].last_layer, 27);
    EXPECT_TRUE(plans[1].has_lm_head);

    for (int r = 0; r < 2; ++r)
    {
        auto errors = plans[r].validate();
        EXPECT_TRUE(errors.empty()) << "Rank " << r << " errors: " << (errors.empty() ? "" : errors[0]);
    }
}

// ============================================================================
// LOCAL PP with TP-in-PP Composition Tests
//
// When a single rank owns multiple PP stages where each stage has a multi-
// device TP domain, the builder must populate local_pp_stage_tp_info
// and intentionally leave local_tp_devices EMPTY. This prevents the
// graph builder from dispatching to the TP path instead of the PP path.
// ============================================================================

/**
 * @brief LOCAL PP with TP-in-PP: verify local_pp_stage_tp_info is populated
 *        and local_tp_devices is intentionally empty.
 *
 * Scenario: Single rank owns 2 PP stages, each with 2-device TP domain.
 * The TP-in-PP guard must prevent global local_tp_devices from being set.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_LocalPP_TPInPP_GuardPreventsGlobalTP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}, {DeviceType::ROCm, 0}, {DeviceType::ROCm, 1}})
                       .build();

    // Two TP domains, both on rank 0
    DomainDefinition cuda_tp;
    cuda_tp.name = "cuda_tp";
    cuda_tp.devices = {GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(1, 0)};
    cuda_tp.backend = CollectiveBackendType::NCCL;

    DomainDefinition rocm_tp;
    rocm_tp.name = "rocm_tp";
    rocm_tp.devices = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(1, 0)};
    rocm_tp.backend = CollectiveBackendType::RCCL;

    config.domain_definitions = {cuda_tp, rocm_tp};

    // Two PP stages on the same rank
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "cuda_tp";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "rocm_tp";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Critical: local_tp_devices must be EMPTY (TP-in-PP guard)
    EXPECT_TRUE(plan.local_tp_devices.empty())
        << "local_tp_devices must be empty when TP-in-PP is active. "
           "Non-empty would cause buildComputeGraph() to dispatch to TP mode "
           "instead of PP mode.";

    // local_pp_stage_tp_info must be populated with per-stage TP domains
    ASSERT_EQ(plan.local_pp_stage_tp_info.size(), 2);

    // Stage 0: CUDA TP domain
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].devices.size(), 2);
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].tp_backend, CollectiveBackendType::NCCL);

    // Stage 1: ROCm TP domain
    EXPECT_EQ(plan.local_pp_stage_tp_info[1].devices.size(), 2);
    EXPECT_EQ(plan.local_pp_stage_tp_info[1].tp_backend, CollectiveBackendType::RCCL);

    // LOCAL PP devices should be populated (primary device per stage)
    EXPECT_EQ(plan.local_pp_devices.size(), 2);

    // Must be identified as LOCAL PP
    EXPECT_TRUE(plan.usesLocalPP());

    // Must NOT be identified as LOCAL TP (TP is per-stage, not global)
    EXPECT_FALSE(plan.usesLocalTP());
}

/**
 * @brief LOCAL PP with single-device stages: local_tp_devices IS populated
 *        because no TP-in-PP guard fires.
 *
 * Scenario: Single rank owns 2 PP stages, each with 1-device domain.
 * No TP composition, so no TP-in-PP guard.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_LocalPP_SingleDeviceStages_NoTPGuard)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .build();

    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};
    gpu0.backend = CollectiveBackendType::NCCL;

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(1, 0)};
    gpu1.backend = CollectiveBackendType::NCCL;

    config.domain_definitions = {gpu0, gpu1};

    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Single-device stages: local_pp_stage_tp_info has entries but each has 1 device
    ASSERT_EQ(plan.local_pp_stage_tp_info.size(), 2);
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].devices.size(), 1);
    EXPECT_EQ(plan.local_pp_stage_tp_info[1].devices.size(), 1);

    // LOCAL PP still active (2 stages on 1 rank)
    EXPECT_TRUE(plan.usesLocalPP());
    EXPECT_EQ(plan.local_pp_devices.size(), 2);
}

/**
 * @brief LOCAL PP with TP-in-PP: verify per-stage layer boundaries.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_LocalPP_LayerBoundaries)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .build();

    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(1, 0)};

    config.domain_definitions = {gpu0, gpu1};

    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 13; // inclusive

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 14;
    stage1.last_layer = 27; // inclusive

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Layer boundaries: [stage0_first, stage1_first, total_layers]
    ASSERT_EQ(plan.local_pp_layer_boundaries.size(), 3)
        << "Expected 2 stage starts + 1 sentinel";
    EXPECT_EQ(plan.local_pp_layer_boundaries[0], 0);
    EXPECT_EQ(plan.local_pp_layer_boundaries[1], 14);
    EXPECT_EQ(plan.local_pp_layer_boundaries[2], model.n_layers);

    // Global layer range spans all stages
    EXPECT_EQ(plan.first_layer, 0);
    EXPECT_EQ(plan.last_layer, 27);
}

/**
 * @brief LOCAL PP with TP-in-PP: verify TP weights propagate per-stage.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_LocalPP_TPWeightsPropagate)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}, {DeviceType::ROCm, 0}})
                       .build();

    DomainDefinition mixed;
    mixed.name = "mixed_tp";
    mixed.devices = {GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::rocm(0, 0)};
    mixed.weights = {0.7f, 0.3f};
    mixed.backend = CollectiveBackendType::HETEROGENEOUS;

    DomainDefinition single;
    single.name = "single_gpu";
    single.devices = {GlobalDeviceAddress::cuda(1, 0)};

    config.domain_definitions = {mixed, single};

    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "mixed_tp";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "single_gpu";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    ASSERT_EQ(plan.local_pp_stage_tp_info.size(), 2);

    // Stage 0: mixed TP with weights
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].devices.size(), 2);
    ASSERT_EQ(plan.local_pp_stage_tp_info[0].tp_weights.size(), 2);
    EXPECT_FLOAT_EQ(plan.local_pp_stage_tp_info[0].tp_weights[0], 0.7f);
    EXPECT_FLOAT_EQ(plan.local_pp_stage_tp_info[0].tp_weights[1], 0.3f);
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].tp_backend, CollectiveBackendType::HETEROGENEOUS);

    // Stage 1: single device (no TP weights)
    EXPECT_EQ(plan.local_pp_stage_tp_info[1].devices.size(), 1);
    EXPECT_TRUE(plan.local_pp_stage_tp_info[1].tp_weights.empty());

    // TP-in-PP guard: stage 0 has 2 devices → local_tp_devices must be empty
    EXPECT_TRUE(plan.local_tp_devices.empty());
}

/**
 * @brief LOCAL PP with TP-in-PP: 3-stage split with heterogeneous TP degrees.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_LocalPP_ThreeStage_HeterogeneousTP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}, {DeviceType::CUDA, 2}, {DeviceType::ROCm, 0}})
                       .build();

    // Stage 0: 2-way CUDA TP
    DomainDefinition cuda_tp;
    cuda_tp.name = "cuda_tp";
    cuda_tp.devices = {GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(1, 0)};
    cuda_tp.backend = CollectiveBackendType::NCCL;

    // Stage 1: single ROCm device (no TP)
    DomainDefinition rocm_single;
    rocm_single.name = "rocm_single";
    rocm_single.devices = {GlobalDeviceAddress::rocm(0, 0)};

    // Stage 2: single CUDA device (no TP)
    DomainDefinition cuda_single;
    cuda_single.name = "cuda_single";
    cuda_single.devices = {GlobalDeviceAddress::cuda(2, 0)};

    config.domain_definitions = {cuda_tp, rocm_single, cuda_single};

    // 28 layers → 10 + 9 + 9
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "cuda_tp";
    stage0.first_layer = 0;
    stage0.last_layer = 9;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "rocm_single";
    stage1.first_layer = 10;
    stage1.last_layer = 18;

    PPStageDefinition stage2;
    stage2.stage_id = 2;
    stage2.domain_name = "cuda_single";
    stage2.first_layer = 19;
    stage2.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1, stage2};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // 3 LOCAL PP stages
    EXPECT_TRUE(plan.usesLocalPP());
    EXPECT_EQ(plan.local_pp_devices.size(), 3);

    // 3 stage TP info entries
    ASSERT_EQ(plan.local_pp_stage_tp_info.size(), 3);
    EXPECT_EQ(plan.local_pp_stage_tp_info[0].devices.size(), 2); // CUDA TP
    EXPECT_EQ(plan.local_pp_stage_tp_info[1].devices.size(), 1); // ROCm single
    EXPECT_EQ(plan.local_pp_stage_tp_info[2].devices.size(), 1); // CUDA single

    // Layer boundaries
    ASSERT_EQ(plan.local_pp_layer_boundaries.size(), 4);
    EXPECT_EQ(plan.local_pp_layer_boundaries[0], 0);
    EXPECT_EQ(plan.local_pp_layer_boundaries[1], 10);
    EXPECT_EQ(plan.local_pp_layer_boundaries[2], 19);
    EXPECT_EQ(plan.local_pp_layer_boundaries[3], model.n_layers);

    // TP-in-PP guard fires (stage 0 has 2 devices)
    EXPECT_TRUE(plan.local_tp_devices.empty());

    // Embedding/LM head ownership
    EXPECT_TRUE(plan.has_embedding);
    EXPECT_TRUE(plan.has_lm_head);
}

/**
 * @brief Cross-rank PP does NOT populate local_pp_stage_tp_info.
 *
 * When 2 stages are on 2 different ranks, each rank has only 1 stage,
 * so my_stages.size() == 1 and no LOCAL PP info is populated.
 */
TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_CrossRankPP_NoLocalPPInfo)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .addRank(1, "localhost", 1, {{DeviceType::ROCm, 0}, {DeviceType::ROCm, 1}})
                       .build();

    DomainDefinition cuda_tp;
    cuda_tp.name = "cuda_tp";
    cuda_tp.devices = {GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(1, 0)};
    cuda_tp.backend = CollectiveBackendType::NCCL;

    DomainDefinition rocm_tp;
    rocm_tp.name = "rocm_tp";
    rocm_tp.devices = {GlobalDeviceAddress::rocm(0, 1), GlobalDeviceAddress::rocm(1, 1)};
    rocm_tp.backend = CollectiveBackendType::RCCL;

    config.domain_definitions = {cuda_tp, rocm_tp};

    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "cuda_tp";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "rocm_tp";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 2);

    // Rank 0: single stage → no LOCAL PP
    EXPECT_FALSE(plans[0].usesLocalPP());
    EXPECT_TRUE(plans[0].local_pp_devices.empty());
    EXPECT_TRUE(plans[0].local_pp_stage_tp_info.empty());
    EXPECT_TRUE(plans[0].local_pp_layer_boundaries.empty());

    // Rank 0: cross-rank TP IS populated (single stage, no TP-in-PP guard)
    EXPECT_EQ(plans[0].local_tp_devices.size(), 2);
    EXPECT_TRUE(plans[0].usesLocalTP());

    // Rank 1: same — no LOCAL PP, has LOCAL TP
    EXPECT_FALSE(plans[1].usesLocalPP());
    EXPECT_TRUE(plans[1].local_pp_devices.empty());
    EXPECT_EQ(plans[1].local_tp_devices.size(), 2);
    EXPECT_TRUE(plans[1].usesLocalTP());

    // PP chain
    EXPECT_TRUE(plans[0].has_embedding);
    EXPECT_FALSE(plans[0].has_lm_head);
    EXPECT_FALSE(plans[1].has_embedding);
    EXPECT_TRUE(plans[1].has_lm_head);
}

// ============================================================================
// Named Domain Tests (Scenario 7 Style) - Original Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_NamedDomains_MixedVendorTP)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    // Define mixed-vendor domain
    DomainDefinition mixed;
    mixed.name = "mixed_tp";
    mixed.devices = {
        GlobalDeviceAddress::cuda(0, 0),
        GlobalDeviceAddress::rocm(0, 0)};
    mixed.weights = {0.6f, 0.4f}; // CUDA is faster
    mixed.backend = CollectiveBackendType::HETEROGENEOUS;

    config.domain_definitions = {mixed};

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Verify domain participation
    ASSERT_FALSE(plan.my_domains.empty());
    const auto &domain = plan.my_domains[0];
    EXPECT_EQ(domain.domain_name, "mixed_tp");
    EXPECT_EQ(domain.devices.size(), 2);
    EXPECT_EQ(domain.weights.size(), 2);
    EXPECT_FLOAT_EQ(domain.weights[0], 0.6f);
    EXPECT_EQ(domain.backend, CollectiveBackendType::HETEROGENEOUS);

    auto errors = plan.validate();
    EXPECT_TRUE(errors.empty()) << "Errors: " << (errors.empty() ? "" : errors[0]);
}

// ============================================================================
// Weight Shard Calculation Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, WeightShard_SingleRank_NoSharding)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &shard = plans[0].weight_shard;

    EXPECT_EQ(shard.shard_index, 0);
    EXPECT_EQ(shard.total_shards, 1);
    EXPECT_FLOAT_EQ(shard.work_fraction, 1.0f);
    EXPECT_FALSE(shard.isSharded());
}

TEST_F(Test__ExecutionPlanBuilder, WeightShard_GlobalTP_EqualSharding)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .addRank(2, "localhost", 2, {{DeviceType::CUDA, 0}})
                       .addRank(3, "localhost", 3, {{DeviceType::CUDA, 0}})
                       .build();

    config.tp_degree = 4;
    config.tp_scope = TPScope::GLOBAL;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 4);

    for (int r = 0; r < 4; ++r)
    {
        const auto &shard = plans[r].weight_shard;
        EXPECT_TRUE(shard.isSharded());
        EXPECT_EQ(shard.total_shards, 4);
        EXPECT_FLOAT_EQ(shard.work_fraction, 0.25f);
    }
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_ValidSimpleConfig)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_TRUE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_PPDegreeExceedsRanks)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    config.pp_degree = 4; // Only 1 rank

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_NamedDomainLocalPPAllowsMoreStagesThanRanks)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::ROCm, 0}, {DeviceType::ROCm, 1}})
                       .build();

    DomainDefinition stage0_domain;
    stage0_domain.name = "rocm0";
    stage0_domain.devices = {GlobalDeviceAddress::rocm(0, 0)};

    DomainDefinition stage1_domain;
    stage1_domain.name = "rocm1";
    stage1_domain.devices = {GlobalDeviceAddress::rocm(1, 0)};

    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "rocm0";
    stage0.first_layer = 0;
    stage0.last_layer = 13;

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "rocm1";
    stage1.first_layer = 14;
    stage1.last_layer = 27;

    config.pp_degree = 2;
    config.pp_split = PPSplitMode::MANUAL;
    config.domain_definitions = {stage0_domain, stage1_domain};
    config.pp_stage_definitions = {stage0, stage1};

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_TRUE(errors.empty()) << "Unexpected validation error: "
                                << (errors.empty() ? "" : errors.front());

    auto plans = builder->buildAllPlans(config, model, cluster);
    ASSERT_EQ(plans.size(), 1);
    EXPECT_TRUE(plans[0].usesLocalPP());
    EXPECT_EQ(plans[0].local_pp_devices.size(), 2);
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_PPDegreeExceedsLayers)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    ModelConfig tiny_model;
    tiny_model.n_layers = 1;
    tiny_model.n_heads = 1;
    tiny_model.n_kv_heads = 1;
    tiny_model.hidden_size = 64;

    config.pp_degree = 2; // Only 1 layer

    auto errors = builder->validateConfig(config, tiny_model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_UndefinedDomain)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    // Reference domain that doesn't exist
    PPStageDefinition stage;
    stage.stage_id = 0;
    stage.domain_name = "nonexistent";
    stage.first_layer = 0;
    stage.last_layer = 27;

    config.pp_stage_definitions = {stage};

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

TEST_F(Test__ExecutionPlanBuilder, ValidateConfig_OverlappingLayers)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    DomainDefinition gpu0;
    gpu0.name = "gpu0";
    gpu0.devices = {GlobalDeviceAddress::cuda(0, 0)};

    DomainDefinition gpu1;
    gpu1.name = "gpu1";
    gpu1.devices = {GlobalDeviceAddress::cuda(0, 1)};

    config.domain_definitions = {gpu0, gpu1};

    // Overlapping layer ranges
    PPStageDefinition stage0;
    stage0.stage_id = 0;
    stage0.domain_name = "gpu0";
    stage0.first_layer = 0;
    stage0.last_layer = 15; // Overlaps with stage1

    PPStageDefinition stage1;
    stage1.stage_id = 1;
    stage1.domain_name = "gpu1";
    stage1.first_layer = 10; // Overlap!
    stage1.last_layer = 27;

    config.pp_stage_definitions = {stage0, stage1};

    auto errors = builder->validateConfig(config, model, cluster);
    EXPECT_FALSE(errors.empty());
}

// ============================================================================
// buildPlanForRank Tests
// ============================================================================

TEST_F(Test__ExecutionPlanBuilder, BuildPlanForRank_ValidRank)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .addRank(1, "localhost", 1, {{DeviceType::CUDA, 0}})
                       .build();

    auto plan = builder->buildPlanForRank(config, model, cluster, 1);

    EXPECT_EQ(plan.rank, 1);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlanForRank_InvalidRank_ReturnsEmpty)
{
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plan = builder->buildPlanForRank(config, model, cluster, 5);

    // Invalid rank should return empty/default plan
    EXPECT_EQ(plan.rank, 0); // Default
}

// ============================================================================
// ModelConfig Tests
// ============================================================================

TEST(Test__ModelConfig, QWen2_7B_HasCorrectValues)
{
    auto config = ModelConfig::qwen2_7b();

    EXPECT_EQ(config.n_layers, 28);
    EXPECT_EQ(config.n_heads, 28);
    EXPECT_EQ(config.n_kv_heads, 4);
    EXPECT_EQ(config.hidden_size, 3584);
}

TEST(Test__ModelConfig, Validate_ValidConfig)
{
    auto config = ModelConfig::qwen2_7b();
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__ModelConfig, Validate_ZeroLayers)
{
    ModelConfig config;
    config.n_layers = 0;
    config.n_heads = 28;
    config.n_kv_heads = 4;
    config.hidden_size = 3584;

    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__ModelConfig, ToString_ContainsName)
{
    auto config = ModelConfig::qwen2_7b();
    auto str = config.toString();
    EXPECT_NE(str.find("Qwen2-7B"), std::string::npos);
}

// ============================================================================
// GPU Selection Priority Tests
// ============================================================================
// These tests verify the critical fix where GPU is selected as primary_device
// when GPUs are available in ClusterInventory, rather than defaulting to CPU.

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_WithGPUs_SelectsGPUAsPrimary)
{
    // Regression test: Ensure GPU is selected when available in inventory
    // This was broken when gatherClusterInventory() didn't enumerate GPUs
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // GPU must be selected as primary device, not CPU
    EXPECT_FALSE(plan.primary_device.isCPU())
        << "GPU should be selected as primary_device when available in ClusterInventory";
    EXPECT_EQ(plan.primary_device.device_type, DeviceType::CUDA);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_WithMultipleGPUs_NoTPConfig_SelectsFirstGPU)
{
    // When multiple GPUs exist but no TP is configured, first GPU should be primary
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}, {DeviceType::ROCm, 0}})
                       .build();

    // No TP configuration - single device mode
    config.tp_degree = 1;

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_FALSE(plan.primary_device.isCPU());
    EXPECT_EQ(plan.primary_device.device_type, DeviceType::CUDA);
    EXPECT_EQ(plan.primary_device.device_ordinal, 0);
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_EmptyInventory_FallsBackToCPU)
{
    // When ClusterInventory has no GPUs, CPU must be selected
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {}) // No GPUs
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    EXPECT_TRUE(plan.primary_device.isCPU())
        << "CPU fallback should be used when no GPUs in ClusterInventory";
}

TEST_F(Test__ExecutionPlanBuilder, BuildPlan_MixedVendorGPUs_SelectsFirstAvailable)
{
    // Mixed vendor scenario (CUDA + ROCm) - should select first GPU
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::ROCm, 0}})
                       .build();

    auto plans = builder->buildAllPlans(config, model, cluster);

    ASSERT_EQ(plans.size(), 1);
    const auto &plan = plans[0];

    // Primary device should be the first GPU (CUDA:0)
    EXPECT_FALSE(plan.primary_device.isCPU());
    // Either vendor is acceptable as long as it's a GPU
    EXPECT_TRUE(plan.primary_device.device_type == DeviceType::CUDA ||
                plan.primary_device.device_type == DeviceType::ROCm);
}

TEST_F(Test__ExecutionPlanBuilder, ClusterInventory_TotalGPUs_MatchesAddedDevices)
{
    // Verify ClusterInventory correctly counts total GPUs
    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0,
                                {{DeviceType::CUDA, 0}, {DeviceType::CUDA, 1}})
                       .addRank(1, "remotehost", 1,
                                {{DeviceType::ROCm, 0}})
                       .build();

    EXPECT_EQ(cluster.world_size, 2);
    EXPECT_EQ(cluster.ranks[0].gpus.size(), 2);
    EXPECT_EQ(cluster.ranks[1].gpus.size(), 1);

    // Total GPUs across cluster
    int total_gpus = 0;
    for (const auto &rank : cluster.ranks)
    {
        total_gpus += static_cast<int>(rank.gpus.size());
    }
    EXPECT_EQ(total_gpus, 3);
}

// =============================================================================
// Phase 5: buildGlobalPPTopology tests
// =============================================================================

/**
 * @brief Build a LocalTP-then-NodeLocalTP topology:
 *        Stage 0: "rocm_domain" — two ROCm GPUs on rank 0 (scope=local)
 *        Stage 1: "cpu_domain"  — one CPU per rank on ranks 0+1 (scope=node_local)
 *        n_layers = 28
 */
TEST_F(Test__ExecutionPlanBuilder, BuildGlobalPPTopology_LocalThenNodeLocal)
{
    OrchestrationConfig cfg;

    // Domain 0: two ROCm GPUs on rank 0 (local)
    cfg.domain_definitions.push_back(DomainDefinition::parse(
        "rocm_domain=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0"));

    // Domain 1: one CPU per rank, two ranks (node_local)
    cfg.domain_definitions.push_back(DomainDefinition::parse(
        "cpu_domain=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"));

    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=rocm_domain:0-13"));
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("1=cpu_domain:14-27"));

    ModelConfig mc;
    mc.n_layers = 28;

    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "host0", 0, {{DeviceType::ROCm, 0}, {DeviceType::ROCm, 1}})
                       .addRank(1, "host1", 0, {})
                       .build();

    ExecutionPlanBuilder concrete_builder;
    auto topo = concrete_builder.buildGlobalPPTopology(cfg, mc, cluster);

    ASSERT_EQ(topo.stages.size(), 2u);

    // Stage 0 should be local (owning rank 0, LOCAL_TP since 2 devices)
    const auto &stage0 = topo.stages[0];
    EXPECT_EQ(stage0.stage_id, 0);
    EXPECT_FALSE(stage0.is_global_tp);
    EXPECT_EQ(stage0.owning_rank, 0);
    EXPECT_EQ(stage0.inner_mode, InnerParallelism::LOCAL_TP);
    EXPECT_EQ(stage0.backend, CollectiveBackendType::RCCL);
    EXPECT_TRUE(stage0.has_embedding);

    // Stage 1 should be global TP with ranks 0 and 1
    const auto &stage1 = topo.stages[1];
    EXPECT_EQ(stage1.stage_id, 1);
    EXPECT_TRUE(stage1.is_global_tp);
    ASSERT_EQ(stage1.participating_ranks.size(), 2u);
    EXPECT_EQ(stage1.participating_ranks[0], 0);
    EXPECT_EQ(stage1.participating_ranks[1], 1);
    EXPECT_EQ(stage1.backend, CollectiveBackendType::UPI);
    ASSERT_EQ(stage1.per_rank_devices.size(), 2u);
    EXPECT_EQ(stage1.per_rank_devices[0].numa_node, 0);
    EXPECT_EQ(stage1.per_rank_devices[1].numa_node, 1);
    EXPECT_TRUE(stage1.has_lm_head);

    auto rank0_plan = GlobalPPRankPlanBuilder::build(topo, 0);
    ASSERT_EQ(rank0_plan.steps.size(), 4u);
    ASSERT_EQ(rank0_plan.steps[0].type, GlobalPPRankPlan::Step::Type::EXECUTE_STAGE);
    EXPECT_EQ(rank0_plan.steps[0].stage_action.stage_id, 0);
    EXPECT_EQ(rank0_plan.steps[0].stage_action.backend, CollectiveBackendType::RCCL);
    ASSERT_EQ(rank0_plan.steps[1].type, GlobalPPRankPlan::Step::Type::TRANSFER);
    EXPECT_EQ(rank0_plan.steps[1].transfer_action.direction, RankTransferAction::Direction::LOCAL_HANDOFF);
    ASSERT_EQ(rank0_plan.steps[2].type, GlobalPPRankPlan::Step::Type::TRANSFER);
    EXPECT_EQ(rank0_plan.steps[2].transfer_action.direction, RankTransferAction::Direction::SEND);
    EXPECT_EQ(rank0_plan.steps[2].transfer_action.peer_rank, 1);
    ASSERT_EQ(rank0_plan.steps[3].type, GlobalPPRankPlan::Step::Type::EXECUTE_STAGE);
    EXPECT_EQ(rank0_plan.steps[3].stage_action.stage_id, 1);
    EXPECT_EQ(rank0_plan.steps[3].stage_action.backend, CollectiveBackendType::UPI);
    EXPECT_EQ(rank0_plan.steps[3].stage_action.device.numa_node, 0);

    auto rank1_plan = GlobalPPRankPlanBuilder::build(topo, 1);
    ASSERT_EQ(rank1_plan.steps.size(), 2u);
    ASSERT_EQ(rank1_plan.steps[0].type, GlobalPPRankPlan::Step::Type::TRANSFER);
    EXPECT_EQ(rank1_plan.steps[0].transfer_action.direction, RankTransferAction::Direction::RECV);
    EXPECT_EQ(rank1_plan.steps[0].transfer_action.peer_rank, 0);
    ASSERT_EQ(rank1_plan.steps[1].type, GlobalPPRankPlan::Step::Type::EXECUTE_STAGE);
    EXPECT_EQ(rank1_plan.steps[1].stage_action.stage_id, 1);
    EXPECT_EQ(rank1_plan.steps[1].stage_action.backend, CollectiveBackendType::UPI);
    EXPECT_EQ(rank1_plan.steps[1].stage_action.device.numa_node, 1);
}

TEST_F(Test__ExecutionPlanBuilder, BuildGlobalPPTopology_SingleStage_FallbackToLocal)
{
    // Single stage, AUTO scope, single-rank resolved → should be local
    OrchestrationConfig cfg;
    cfg.domain_definitions.push_back(DomainDefinition::parse("gpu=0:cuda:0"));
    cfg.pp_stage_definitions.push_back(PPStageDefinition::parse("0=gpu:0-11"));

    ModelConfig mc;
    mc.n_layers = 12;

    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {{DeviceType::CUDA, 0}})
                       .build();

    ExecutionPlanBuilder concrete_builder;
    auto topo = concrete_builder.buildGlobalPPTopology(cfg, mc, cluster);

    ASSERT_EQ(topo.stages.size(), 1u);
    EXPECT_FALSE(topo.stages[0].is_global_tp);
    EXPECT_EQ(topo.stages[0].owning_rank, 0);
    EXPECT_TRUE(topo.stages[0].has_embedding);
    EXPECT_TRUE(topo.stages[0].has_lm_head);
}

TEST_F(Test__ExecutionPlanBuilder, BuildGlobalPPTopology_EmptyPPStages_ReturnsSingleFallback)
{
    // No pp_stage_definitions → should return a single-stage fallback
    OrchestrationConfig cfg;

    ModelConfig mc;
    mc.n_layers = 24;

    auto cluster = ClusterInventoryBuilder()
                       .addRank(0, "localhost", 0, {})
                       .build();

    ExecutionPlanBuilder concrete_builder;
    auto topo = concrete_builder.buildGlobalPPTopology(cfg, mc, cluster);

    // Should return a degenerate single-stage topology
    ASSERT_EQ(topo.stages.size(), 1u);
    EXPECT_EQ(topo.stages[0].first_layer, 0);
    EXPECT_EQ(topo.stages[0].last_layer, mc.n_layers - 1);
}

// =============================================================================
// Phase 5: renderMultiDomainTopologyInfo test
// =============================================================================

TEST(Test__RenderTopologyInfo, ContainsDomainNamesAndRankCounts)
{
    // Build a minimal 2-stage topology with known stages
    GlobalPPStageSpec s0;
    s0.stage_id = 0;
    s0.domain_name = "rocm_domain";
    s0.first_layer = 0;
    s0.last_layer = 13;
    s0.has_embedding = true;
    s0.is_global_tp = false;
    s0.owning_rank = 0;
    s0.inner_mode = InnerParallelism::LOCAL_TP;

    GlobalPPStageSpec s1;
    s1.stage_id = 1;
    s1.domain_name = "cpu_domain";
    s1.first_layer = 14;
    s1.last_layer = 27;
    s1.has_lm_head = true;
    s1.is_global_tp = true;
    s1.participating_ranks = {0, 1};
    s1.per_rank_device = GlobalDeviceAddress::cpu();

    std::vector<GlobalPPStageSpec> specs = {std::move(s0), std::move(s1)};
    auto topo = GlobalPPTopology::build(std::move(specs), 28, 2);

    std::string info = renderMultiDomainTopologyInfo(topo, 2);

    // Must mention both domain names
    EXPECT_NE(info.find("rocm_domain"), std::string::npos);
    EXPECT_NE(info.find("cpu_domain"), std::string::npos);
    // Must mention per-rank counts
    EXPECT_NE(info.find("rank 0"), std::string::npos);
    EXPECT_NE(info.find("rank 1"), std::string::npos);
}
