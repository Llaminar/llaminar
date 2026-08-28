/**
 * @file Test__WeightManagerMoEExpertOverlayPreparation.cpp
 * @brief Device-free ownership and preparation tests for ExpertOverlay weights.
 *
 * These cases validate that the preparation request is exact in both logical
 * expert identity and MPI ownership before any device packing occurs.  The
 * real-weight integration campaigns rely on this layer to prevent a rank from
 * loading a remote CPU/accelerator participant's expert slice.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "loaders/WeightManager.h"
#include "mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
namespace
{
    using Role = ExpertGemmRegistry::WeightRole;

    RoutedExpertDomain domainWith(
        const std::string &name,
        ExecutionDomainScope kind,
        std::vector<GlobalDeviceAddress> participants,
        RoutedExpertComputePolicy routed_compute_policy,
        CollectiveBackendType backend)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = kind;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.owner_rank = 0;
        domain.routed_compute_policy = routed_compute_policy;
        return domain;
    }

    RoutedExpertTier tier(
        std::string name,
        std::string domain,
        int priority,
        bool fallback = false,
        int max_experts_per_layer = 0,
        size_t memory_budget_bytes = 0)
    {
        RoutedExpertTier result;
        result.name = std::move(name);
        result.domain = std::move(domain);
        result.priority = priority;
        result.fallback = fallback;
        result.max_experts_per_layer = max_experts_per_layer;
        result.memory_budget_bytes = memory_budget_bytes;
        return result;
    }

    MoERoutedExpertPlacementPlan threeTierPlan(std::vector<RoutedExpertLayerPlacement> placements)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "cuda_fast";
        plan.shared_expert_domain = "cuda_fast";
        plan.residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
        plan.domains = {
            domainWith("cuda_fast", ExecutionDomainScope::SINGLE,
                       {GlobalDeviceAddress::cuda(0)},
                       RoutedExpertComputePolicy::Apportioned,
                       CollectiveBackendType::NCCL),
            domainWith("rocm_warm", ExecutionDomainScope::RANK_LOCAL,
                       {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                       RoutedExpertComputePolicy::Apportioned,
                       CollectiveBackendType::RCCL),
            domainWith("cpu_cold", ExecutionDomainScope::NODE_LOCAL,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       RoutedExpertComputePolicy::Apportioned,
                       CollectiveBackendType::UPI),
        };
        plan.routed_tiers = {
            tier("hottest", "cuda_fast", 0, false, 2, 4096),
            tier("warm", "rocm_warm", 1, false, 2, 8192),
            tier("cold", "cpu_cold", 2, true),
        };
        plan.placements = std::move(placements);
        return plan;
    }

    MoERoutedExpertModelMetadata metadata(int layers, int experts)
    {
        MoERoutedExpertModelMetadata model;
        model.num_layers = layers;
        model.num_experts = experts;
        model.d_model = 16;
        model.routed_intermediate_size = 32;
        model.has_shared_expert = true;
        model.shared_intermediate_size = 32;
        model.routed_quant_type = "Q4_0";
        model.shared_quant_type = "Q4_0";
        return model;
    }

    std::shared_ptr<Q4_0Tensor> createQ4_0WithData(const std::vector<size_t> &shape, uint32_t seed)
    {
        size_t total_elements = 1;
        for (auto dim : shape)
            total_elements *= dim;

        const size_t cols = (shape.size() == 3) ? shape[0] : shape.back();
        const size_t rows = total_elements / cols;
        const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        const size_t total_blocks = rows * blocks_per_row;

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_0Block));
        auto *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            float values[Q4_0Block::BLOCK_SIZE];
            float max_abs = 0.0f;
            for (size_t i = 0; i < Q4_0Block::BLOCK_SIZE; ++i)
            {
                values[i] = dist(rng);
                max_abs = std::max(max_abs, std::abs(values[i]));
            }

            const float scale = max_abs / 7.0f;
            uint32_t bits = 0;
            std::memcpy(&bits, &scale, sizeof(bits));
            const uint32_t sign = (bits >> 31) & 0x1u;
            const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
            const uint32_t mant = (bits >> 13) & 0x3ffu;
            if (exp <= 0)
                blocks[block_idx].d = static_cast<uint16_t>(sign << 15);
            else if (exp >= 31)
                blocks[block_idx].d = static_cast<uint16_t>((sign << 15) | 0x7c00u);
            else
                blocks[block_idx].d = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);

            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (size_t i = 0; i < Q4_0Block::BLOCK_SIZE / 2; ++i)
            {
                int q0 = std::clamp(static_cast<int>(std::round(values[2 * i] * inv_scale)) + 8, 0, 15);
                int q1 = std::clamp(static_cast<int>(std::round(values[2 * i + 1] * inv_scale)) + 8, 0, 15);
                blocks[block_idx].qs[i] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }

        return std::make_shared<Q4_0Tensor>(shape, raw_data);
    }

    void addSingleLayerExpertParents(
        const std::shared_ptr<MockModelLoader> &loader,
        size_t d_model,
        size_t intermediate,
        size_t num_experts)
    {
        loader->addTensor("blk.0.ffn_gate_exps.weight",
                          createQ4_0WithData({d_model, intermediate, num_experts}, 101));
        loader->addTensor("blk.0.ffn_up_exps.weight",
                          createQ4_0WithData({d_model, intermediate, num_experts}, 102));
        loader->addTensor("blk.0.ffn_down_exps.weight",
                          createQ4_0WithData({intermediate, d_model, num_experts}, 103));
    }

    /**
     * @brief Build a one-layer distributed CPU cold tier for registry tests.
     *
     * The production NodeTP arrangement gives each MPI rank its own
     * WeightManager and one exact half of the expert range.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan> singleLayerCpuColdPlan(size_t num_experts)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "cpu_cold";
        plan->shared_expert_domain = "cpu_cold";
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::TensorParallel);
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->domains = {
            domainWith("cpu_cold", ExecutionDomainScope::NODE_LOCAL,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       RoutedExpertComputePolicy::Apportioned,
                       CollectiveBackendType::UPI),
        };
        plan->domains[0].world_ranks = {0, 1};
        plan->routed_tiers = {
            tier("cold", "cpu_cold", 0, true),
        };
        plan->placements = {
            RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = std::vector<int>(num_experts, 0)},
        };
        return plan;
    }

    /**
     * @brief Build a host-qualified NodeTP plan with one CPU endpoint per rank.
     *
     * Concrete host names emulate inventory binding.  They ensure the runtime
     * resolver exposes only the endpoint owned by the current rank rather than
     * treating both portable `localhost` addresses as process-local.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan>
    distributedCpuColdPlan(int num_experts)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "cpu_cold";
        plan->shared_expert_domain = "cpu_cold";
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::TensorParallel);
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->domains = {
            domainWith(
                "cpu_cold",
                ExecutionDomainScope::NODE_LOCAL,
                {
                    GlobalDeviceAddress::cpu(/*numa=*/0, "bound-node"),
                    GlobalDeviceAddress::cpu(/*numa=*/1, "bound-node"),
                },
                RoutedExpertComputePolicy::Apportioned,
                CollectiveBackendType::UPI),
        };
        plan->domains.front().world_ranks = {0, 1};
        plan->routed_tiers = {
            tier("cold", "cpu_cold", 0, true),
        };
        plan->placements = {
            RoutedExpertLayerPlacement{
                .layer = 0,
                .routed_expert_tier = std::vector<int>(
                    static_cast<size_t>(num_experts), 0)},
        };
        return plan;
    }
} // namespace

TEST(Test__WeightManagerMoEExpertOverlayPreparation, BuildsTierAwareRequestsAndDiagnostics)
{
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(threeTierPlan({
        RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 0, 1, 2}},
        RoutedExpertLayerPlacement{.layer = 1, .routed_expert_tier = {1, 2, 0, 1, 2, 0}},
    }));
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 1024);

    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cuda(0), 0, 3, Role::DOWN));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::cuda(0), 0, 1, Role::GATE));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::UP));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::rocm(1), 0, 1, Role::UP));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::rocm(0), 0, 4, Role::UP));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(1), 0, 4, Role::UP));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cpu(), 0, 2, Role::GATE));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::cpu(), 0, 5, Role::GATE));
    EXPECT_TRUE(prep.hasCpuRoutedAssignments());

    const auto devices = prep.acceleratorDevices();
    EXPECT_EQ(devices, (std::vector<DeviceId>{DeviceId::cuda(0), DeviceId::rocm(0), DeviceId::rocm(1)}));

    const auto *cuda_stats = prep.diagnostics().domainStats("cuda_fast", DeviceId::cuda(0));
    ASSERT_NE(cuda_stats, nullptr);
    EXPECT_TRUE(cuda_stats->accelerator);
    EXPECT_FALSE(cuda_stats->fallback);
    EXPECT_EQ(cuda_stats->assigned_routed_experts, 4u);
    EXPECT_EQ(cuda_stats->planned_engine_count, 12u);
    EXPECT_EQ(cuda_stats->estimated_routed_bytes, 4096u);
    EXPECT_EQ(cuda_stats->memory_budget_bytes, 4096u);

    const auto *rocm0_stats = prep.diagnostics().domainStats("rocm_warm", DeviceId::rocm(0));
    const auto *rocm1_stats = prep.diagnostics().domainStats("rocm_warm", DeviceId::rocm(1));
    ASSERT_NE(rocm0_stats, nullptr);
    ASSERT_NE(rocm1_stats, nullptr);
    EXPECT_TRUE(rocm0_stats->accelerator);
    EXPECT_TRUE(rocm1_stats->accelerator);
    EXPECT_EQ(rocm0_stats->assigned_routed_experts, rocm1_stats->assigned_routed_experts);
    EXPECT_EQ(rocm0_stats->planned_engine_count, rocm1_stats->planned_engine_count);

    const auto *cpu_stats = prep.diagnostics().domainStats("cpu_cold", DeviceId::cpu());
    ASSERT_NE(cpu_stats, nullptr);
    EXPECT_FALSE(cpu_stats->accelerator);
    EXPECT_TRUE(cpu_stats->fallback);
    EXPECT_EQ(cpu_stats->assigned_routed_experts, 2u);

    const auto *cpu_rank0_stats = prep.diagnostics().domainStats("cpu_cold", DeviceId::cpu(), 0, 0);
    const auto *cpu_rank1_stats = prep.diagnostics().domainStats("cpu_cold", DeviceId::cpu(), 1, 1);
    ASSERT_NE(cpu_rank0_stats, nullptr);
    EXPECT_EQ(cpu_rank0_stats->residency_category, WeightResidencyCategory::CpuFallbackExpert);
    EXPECT_EQ(cpu_rank1_stats, nullptr);
    EXPECT_NE(prep.diagnostics().render().find("memory_by_role"), std::string::npos);
    EXPECT_NE(prep.diagnostics().render().find("routed_tier="), std::string::npos);
    EXPECT_NE(prep.diagnostics().render().find("fallback="), std::string::npos);
}

/**
 * @brief Verify rank filtering preserves the already-local CPU request set.
 *
 * Each MPI process builds its own preparation plan.  Rank filtering may change
 * the worker residency category, but it must not require a root process to
 * synthesize a remote rank's expert requests.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     FiltersRankLocalRequestsByOverlayRankRoleAndParticipant)
{
    const auto plan = distributedCpuColdPlan(/*num_experts=*/8);
    const auto root_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
        });
    const auto worker_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 1,
            .validate_mvp_root_reachability = false,
        });
    const auto root_prep = MoEExpertOverlayPreparationPlan::build(*root_runtime, 2048);
    const auto worker_prep = MoEExpertOverlayPreparationPlan::build(*worker_runtime, 2048);
    const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
        plan,
        MoEExpertOverlayExecutionPlanResolverOptions{
            .current_world_rank = 0,
            .world_size = 2,
        });

    const auto *rank0 = execution_plan.rankPlanFor(0);
    const auto *rank1 = execution_plan.rankPlanFor(1);
    ASSERT_NE(rank0, nullptr);
    ASSERT_NE(rank1, nullptr);

    const auto root_filtered = root_prep.filteredForRank(*rank0);
    EXPECT_FALSE(root_filtered.hasAcceleratorRequests());
    EXPECT_TRUE(root_filtered.hasCpuRoutedAssignments());
    EXPECT_TRUE(root_filtered.shouldPrepare(DeviceId::cpu(), 0, 3, Role::GATE));
    EXPECT_FALSE(root_filtered.shouldPrepare(DeviceId::cpu(), 0, 4, Role::GATE));
    const auto *root_request = root_filtered.requestForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, 3, Role::GATE);
    ASSERT_NE(root_request, nullptr);
    EXPECT_EQ(root_request->residency_category, WeightResidencyCategory::CpuFallbackExpert);

    const auto worker_filtered = worker_prep.filteredForRank(*rank1);
    EXPECT_FALSE(worker_filtered.hasAcceleratorRequests());
    EXPECT_TRUE(worker_filtered.hasCpuRoutedAssignments());
    EXPECT_FALSE(worker_filtered.shouldPrepare(DeviceId::cpu(), 0, 3, Role::GATE));
    EXPECT_TRUE(worker_filtered.shouldPrepare(DeviceId::cpu(), 0, 4, Role::GATE));
    const auto *worker_request = worker_filtered.requestForParticipant(
        "cpu_cold", DeviceId::cpu(), 1, 1, 0, 4, Role::GATE);
    ASSERT_NE(worker_request, nullptr);
    EXPECT_TRUE(rank1->hasRole(OverlayRankRole::ContinuationParticipant));
    EXPECT_EQ(worker_request->residency_category,
              WeightResidencyCategory::CpuFallbackExpert);
    EXPECT_NE(worker_filtered.diagnostics().render().find("fallback="),
              std::string::npos);
    EXPECT_EQ(worker_filtered.diagnostics().render().find("AcceleratorRoutedExpert"), std::string::npos);
}

/**
 * @brief Prove each MPI rank prepares only its apportioned CPU NodeTP slice.
 *
 * This regression uses inventory-bound hostnames so a rank cannot accidentally
 * see another rank's CPU endpoint as a local `localhost` device.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     DistributedNodeTPPreparesOnlyExpertsOwnedByThisRank)
{
    constexpr int kExperts = 8;
    const auto plan = distributedCpuColdPlan(kExperts);

    const auto rank_zero_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
        });
    const auto rank_one_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 1,
            .validate_mvp_root_reachability = false,
        });

    const auto rank_zero_prep =
        MoEExpertOverlayPreparationPlan::build(*rank_zero_runtime, 128);
    const auto rank_one_prep =
        MoEExpertOverlayPreparationPlan::build(*rank_one_runtime, 128);

    std::vector<int> rank_zero_experts;
    std::vector<int> rank_one_experts;
    for (const auto &request : rank_zero_prep.requests())
    {
        if (request.role == Role::GATE)
            rank_zero_experts.push_back(request.expert_id);
        EXPECT_EQ(request.participant_world_rank, 0);
        EXPECT_EQ(request.participant_index, 0);
    }
    for (const auto &request : rank_one_prep.requests())
    {
        if (request.role == Role::GATE)
            rank_one_experts.push_back(request.expert_id);
        EXPECT_EQ(request.participant_world_rank, 1);
        EXPECT_EQ(request.participant_index, 1);
    }

    EXPECT_EQ(rank_zero_experts, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(rank_one_experts, (std::vector<int>{4, 5, 6, 7}));
    EXPECT_FALSE(rank_zero_prep.shouldPrepare(DeviceId::cpu(), 0, 4, Role::GATE));
    EXPECT_FALSE(rank_one_prep.shouldPrepare(DeviceId::cpu(), 0, 3, Role::GATE));

    const auto *rank_zero_stats = rank_zero_prep.diagnostics().domainStats(
        "cpu_cold", DeviceId::cpu(), 0, 0);
    const auto *rank_one_stats = rank_one_prep.diagnostics().domainStats(
        "cpu_cold", DeviceId::cpu(), 1, 1);
    ASSERT_NE(rank_zero_stats, nullptr);
    ASSERT_NE(rank_one_stats, nullptr);
    EXPECT_EQ(rank_zero_stats->assigned_routed_experts, 4u);
    EXPECT_EQ(rank_one_stats->assigned_routed_experts, 4u);
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, FiltersRequestsToOneGraphParticipantDevice)
{
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(threeTierPlan({
        RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 0, 1, 2}},
    }));
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    const auto rank_wide = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 2048);

    const auto rocm0 = rank_wide.filteredForDevice(DeviceId::rocm(0));
    EXPECT_TRUE(rocm0.hasAcceleratorRequests());
    EXPECT_EQ(rocm0.acceleratorDevices(),
              (std::vector<DeviceId>{DeviceId::rocm(0)}));
    EXPECT_TRUE(rocm0.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::GATE));
    EXPECT_FALSE(rocm0.shouldPrepare(DeviceId::rocm(1), 0, 4, Role::GATE));
    EXPECT_FALSE(rocm0.shouldPrepare(DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_FALSE(rocm0.hasCpuRoutedAssignments());
    ASSERT_EQ(rocm0.diagnostics().domains.size(), 1u);
    EXPECT_EQ(rocm0.diagnostics().domains.front().device, DeviceId::rocm(0));

    const auto cpu = rank_wide.filteredForDevice(DeviceId::cpu());
    EXPECT_FALSE(cpu.hasAcceleratorRequests());
    EXPECT_TRUE(cpu.hasCpuRoutedAssignments());
    EXPECT_TRUE(cpu.shouldPrepare(DeviceId::cpu(), 0, 2, Role::DOWN));
    EXPECT_FALSE(cpu.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::DOWN));

    const auto heterogeneous_root = rank_wide.filteredForDevices(
        {DeviceId::cuda(0), DeviceId::cpu()});
    EXPECT_TRUE(heterogeneous_root.hasAcceleratorRequests());
    EXPECT_TRUE(heterogeneous_root.hasCpuRoutedAssignments());
    EXPECT_EQ(
        heterogeneous_root.acceleratorDevices(),
        (std::vector<DeviceId>{DeviceId::cuda(0)}));
    EXPECT_TRUE(heterogeneous_root.shouldPrepare(
        DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_TRUE(heterogeneous_root.shouldPrepare(
        DeviceId::cpu(), 0, 2, Role::DOWN));
    EXPECT_FALSE(heterogeneous_root.shouldPrepare(
        DeviceId::rocm(0), 0, 1, Role::GATE));
    EXPECT_THROW(
        rank_wide.filteredForDevices({}),
        std::invalid_argument);
    EXPECT_THROW(
        rank_wide.filteredForDevices(
            {DeviceId::cuda(0), DeviceId::cuda(0)}),
        std::invalid_argument);
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, SmallModelBudgetKeepsCudaPartialUnlessUncapped)
{
    MoERoutedExpertPlacementPlan capped;
    capped.enabled = true;
    capped.topology = RoutedExpertPlacementTopology::TieredOverlay;
    capped.continuation_domain = "cuda_fast";
    capped.shared_expert_domain = "cuda_fast";
    capped.residency_policy = RoutedExpertResidencyPolicy::StaticById;
    capped.domains = {
        domainWith("cuda_fast", ExecutionDomainScope::SINGLE,
                   {GlobalDeviceAddress::cuda(0)},
                   RoutedExpertComputePolicy::Apportioned,
                   CollectiveBackendType::NCCL),
        domainWith("cpu_cold", ExecutionDomainScope::NODE_LOCAL,
                   {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                   RoutedExpertComputePolicy::Apportioned,
                   CollectiveBackendType::UPI),
    };
    capped.routed_tiers = {
        tier("hottest", "cuda_fast", 0, false, 1),
        tier("cold", "cpu_cold", 1, true),
    };

    const auto capped_result = MoERoutedExpertPlacementPlanner::plan(capped, metadata(1, 2));
    ASSERT_EQ(capped_result.planned_plan.placements.size(), 1u);
    EXPECT_EQ(capped_result.planned_plan.placements[0].routed_expert_tier,
              (std::vector<int>{0, 1}));

    auto capped_runtime = resolveMoEExpertOverlayRuntimePlan(
        std::make_shared<MoERoutedExpertPlacementPlan>(capped_result.planned_plan));
    const auto capped_prep = MoEExpertOverlayPreparationPlan::build(*capped_runtime, 128);
    const auto *capped_cuda = capped_prep.diagnostics().domainStats("cuda_fast", DeviceId::cuda(0));
    ASSERT_NE(capped_cuda, nullptr);
    EXPECT_EQ(capped_cuda->assigned_routed_experts, 1u);

    capped.routed_tiers[0].max_experts_per_layer = 0;
    const auto uncapped_result = MoERoutedExpertPlacementPlanner::plan(capped, metadata(1, 2));
    EXPECT_EQ(uncapped_result.planned_plan.placements[0].routed_expert_tier,
              (std::vector<int>{0, 0}));
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, KeepsSameDeviceDomainsSeparateInPreparationRequests)
{
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
    plan->continuation_domain = "cuda_fast";
    plan->shared_expert_domain = "cuda_fast";
    plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;
    plan->domains = {
        domainWith("cuda_fast", ExecutionDomainScope::SINGLE,
                   {GlobalDeviceAddress::cuda(0)},
                   RoutedExpertComputePolicy::Apportioned,
                   CollectiveBackendType::NCCL),
        domainWith("cuda_warm", ExecutionDomainScope::SINGLE,
                   {GlobalDeviceAddress::cuda(0)},
                   RoutedExpertComputePolicy::Apportioned,
                   CollectiveBackendType::NCCL),
        domainWith("cpu_cold", ExecutionDomainScope::NODE_LOCAL,
                   {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                   RoutedExpertComputePolicy::Apportioned,
                   CollectiveBackendType::UPI),
    };
    plan->routed_tiers = {
        tier("fast", "cuda_fast", 0),
        tier("warm", "cuda_warm", 1),
        tier("cold", "cpu_cold", 2, true),
    };
    plan->placements = {
        RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 2}},
    };

    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 128);

    const auto *fast_stats = prep.diagnostics().domainStats("cuda_fast", DeviceId::cuda(0));
    const auto *warm_stats = prep.diagnostics().domainStats("cuda_warm", DeviceId::cuda(0));
    ASSERT_NE(fast_stats, nullptr);
    ASSERT_NE(warm_stats, nullptr);
    EXPECT_EQ(fast_stats->assigned_routed_experts, 1u);
    EXPECT_EQ(warm_stats->assigned_routed_experts, 1u);

    auto count_requests = [&](const std::string &domain_name, int expert_id) {
        return static_cast<int>(std::count_if(prep.requests().begin(), prep.requests().end(),
                                             [&](const auto &request) {
                                                 return request.domain_name == domain_name &&
                                                        request.device == DeviceId::cuda(0) &&
                                                        request.layer == 0 &&
                                                        request.expert_id == expert_id;
                                             }));
    };

    EXPECT_EQ(count_requests("cuda_fast", 0), 3);
    EXPECT_EQ(count_requests("cuda_warm", 1), 3);
    EXPECT_EQ(count_requests("cuda_fast", 1), 0);
    EXPECT_EQ(count_requests("cuda_warm", 0), 0);
    EXPECT_EQ(prep.expertsForDomainDeviceLayerRole("cuda_fast", DeviceId::cuda(0), 0, Role::GATE),
              (std::vector<int>{0}));
    EXPECT_EQ(prep.expertsForDomainDeviceLayerRole("cuda_warm", DeviceId::cuda(0), 0, Role::GATE),
              (std::vector<int>{1}));
    EXPECT_EQ(prep.domainsForDeviceLayerRole(DeviceId::cuda(0), 0, Role::GATE),
              (std::vector<std::string>{"cuda_fast", "cuda_warm"}));
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, AcceleratorPreparationRejectsMutableCacheSource)
{
    /**
     * A LocalTP runner freezes the exact weight slice or replica that its graph
     * owns. Accelerator overlay preparation must use that immutable identity;
     * accepting a null frozen set would silently return to WeightManager's
     * process-wide cache, where a different participant's expert slice may live.
     * This regression stops before touching CUDA, so it remains a true unit test.
     */
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(threeTierPlan({
        RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2}},
    }));
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(MoEExpertOverlayPreparationPlan::build(*runtime_plan, 128)
                    .hasAcceleratorRequests());

    auto loader = MockModelLoader::createMinimal();
    WeightManager manager(*loader);

    EXPECT_FALSE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cuda(0)));
}

/**
 * @brief Prove independent MPI-rank registries contain only their owned experts.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     PreparesCpuFallbackExpertsIntoRankLocalRegistry)
{
    const size_t d_model = 64;
    const size_t intermediate = 32;
    const size_t num_experts = 4;
    const auto plan = singleLayerCpuColdPlan(num_experts);
    const auto rank_zero_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 0,
        });
    const auto rank_one_runtime = resolveMoEExpertOverlayRuntimePlan(
        plan,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 1,
            .validate_mvp_root_reachability = false,
        });

    auto rank_zero_loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(rank_zero_loader, d_model, intermediate, num_experts);
    WeightManager rank_zero_manager(*rank_zero_loader);
    ASSERT_NE(rank_zero_manager.getWeightForDevice("blk.0.ffn_gate_exps.weight"), nullptr);
    ASSERT_NE(rank_zero_manager.getWeightForDevice("blk.0.ffn_up_exps.weight"), nullptr);
    ASSERT_NE(rank_zero_manager.getWeightForDevice("blk.0.ffn_down_exps.weight"), nullptr);
    ASSERT_TRUE(rank_zero_manager.prepareMoEExpertOverlayWeights(
        *rank_zero_runtime, DeviceId::cpu()));

    std::vector<ITensorGemm *> gate;
    std::vector<ITensorGemm *> up;
    std::vector<ITensorGemm *> down;
    EXPECT_FALSE(rank_zero_manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const bool owned_by_participant = expert < 2;
        EXPECT_EQ(gate[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(up[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(down[expert] != nullptr, owned_by_participant) << "expert=" << expert;
    }

    EXPECT_FALSE(rank_zero_manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 1, 1, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        EXPECT_EQ(gate[expert], nullptr) << "expert=" << expert;
        EXPECT_EQ(up[expert], nullptr) << "expert=" << expert;
        EXPECT_EQ(down[expert], nullptr) << "expert=" << expert;
    }

    auto rank_one_loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(rank_one_loader, d_model, intermediate, num_experts);
    WeightManager rank_one_manager(*rank_one_loader);
    ASSERT_TRUE(rank_one_manager.prepareMoEExpertOverlayWeights(
        *rank_one_runtime, DeviceId::cpu()));

    EXPECT_FALSE(rank_one_manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        EXPECT_EQ(gate[expert], nullptr) << "expert=" << expert;
        EXPECT_EQ(up[expert], nullptr) << "expert=" << expert;
        EXPECT_EQ(down[expert], nullptr) << "expert=" << expert;
    }

    EXPECT_FALSE(rank_one_manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 1, 1, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const bool owned_by_participant = expert >= 2;
        EXPECT_EQ(gate[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(up[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(down[expert] != nullptr, owned_by_participant) << "expert=" << expert;
    }
}

/**
 * @brief A sealed reusable context must retain its physical CPU slot aliases.
 *
 * Swapping the two complete triplets models a migration followed by terminal
 * logical-placement restoration: logical experts are back in their original
 * participant, but their bytes now occupy recycled physical slots. Re-entering
 * weight preparation must adopt the atomically rebound registry and must not
 * reinterpret loader tensor slot zero as logical expert zero again.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     ReusesSealedCpuParticipantBankWithoutRebuildingLoaderOffsets)
{
    constexpr size_t kModel = 64;
    constexpr size_t kIntermediate = 32;
    constexpr size_t kExperts = 4;
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, kModel, kIntermediate, kExperts);
    WeightManager manager(*loader);
    const auto plan = singleLayerCpuColdPlan(kExperts);
    const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));

    auto &registry = manager.expertGemmRegistry();
    const auto acquire = [&](int expert, Role role)
    {
        return registry.getEngineLifetimeForParticipant(
            "cpu_cold", DeviceId::cpu(), 0, 0, 0, expert, role);
    };
    const auto gate0 = acquire(0, Role::GATE);
    const auto up0 = acquire(0, Role::UP);
    const auto down0 = acquire(0, Role::DOWN);
    const auto gate1 = acquire(1, Role::GATE);
    const auto up1 = acquire(1, Role::UP);
    const auto down1 = acquire(1, Role::DOWN);
    ASSERT_NE(gate0, nullptr);
    ASSERT_NE(up0, nullptr);
    ASSERT_NE(down0, nullptr);
    ASSERT_NE(gate1, nullptr);
    ASSERT_NE(up1, nullptr);
    ASSERT_NE(down1, nullptr);

    const ExpertGemmRegistry::ParticipantLayerScope scope{
        .domain_name = "cpu_cold",
        .device = DeviceId::cpu(),
        .participant_world_rank = 0,
        .participant_index = 0,
        .layer = 0,
    };
    const std::vector<ExpertGemmRegistry::ParticipantExpertBinding> rebound{
        {
            .scope = scope,
            .expert = 0,
            .gate = gate1,
            .up = up1,
            .down = down1,
        },
        {
            .scope = scope,
            .expert = 1,
            .gate = gate0,
            .up = up0,
            .down = down0,
        },
    };
    std::string replacement_error;
    ASSERT_TRUE(registry.replaceParticipantResidency(
        std::span<const ExpertGemmRegistry::ParticipantLayerScope>(&scope, 1u),
        rebound,
        &replacement_error))
        << replacement_error;

    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));
    EXPECT_EQ(acquire(0, Role::GATE).get(), gate1.get());
    EXPECT_EQ(acquire(0, Role::UP).get(), up1.get());
    EXPECT_EQ(acquire(0, Role::DOWN).get(), down1.get());
    EXPECT_EQ(acquire(1, Role::GATE).get(), gate0.get());
    EXPECT_EQ(acquire(1, Role::UP).get(), up0.get());
    EXPECT_EQ(acquire(1, Role::DOWN).get(), down0.get());
    EXPECT_EQ(
        registry.getEngineForDomain(
            "cpu_cold", DeviceId::cpu(), 0, 0, Role::GATE),
        gate1.get());
    EXPECT_EQ(
        registry.getEngineForDomain(
            "cpu_cold", DeviceId::cpu(), 0, 1, Role::GATE),
        gate0.get());
}

/**
 * @brief Certified reuse consumes one complete sealed registry or fails.
 *
 * Registry population alone is deliberately insufficient: the production
 * model-context lifecycle must first certify preparation and graph completion.
 * Once admitted, deleting one domain alias models a torn terminal seal. The
 * reuse path must reject it without consulting the still-live loader parents
 * or repairing the missing entry.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     CertifiedReuseRequiresLifecycleAndRejectsTornRegistryWithoutFallback)
{
    constexpr size_t kExperts = 4;
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, kExperts);
    WeightManager manager(*loader);
    const auto plan = singleLayerCpuColdPlan(kExperts);
    const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));

    EXPECT_FALSE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan,
        DeviceId::cpu(),
        nullptr,
        nullptr,
        PreparedWeightAdmission::ReuseCertifiedCompleteSet));

    manager.markMaterializationComplete();
    manager.markDevicePreparationComplete();
    manager.markGraphMaterializationComplete();
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan,
        DeviceId::cpu(),
        nullptr,
        nullptr,
        PreparedWeightAdmission::ReuseCertifiedCompleteSet));

    auto &registry = manager.expertGemmRegistry();
    ASSERT_TRUE(registry.removeEngineForDomain(
        "cpu_cold", DeviceId::cpu(), 0, 0, Role::GATE));
    ASSERT_NE(
        registry.getEngineForParticipant(
            "cpu_cold", DeviceId::cpu(), 0, 0, 0, 0, Role::GATE),
        nullptr);

    EXPECT_FALSE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan,
        DeviceId::cpu(),
        nullptr,
        nullptr,
        PreparedWeightAdmission::ReuseCertifiedCompleteSet));
    EXPECT_EQ(
        registry.getEngineForDomain(
            "cpu_cold", DeviceId::cpu(), 0, 0, Role::GATE),
        nullptr);
}

/**
 * @brief Reject a torn scoped bank instead of repairing it from stale slots.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     RejectsPartialCpuParticipantBankOnPreparationReentry)
{
    constexpr size_t kExperts = 4;
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, kExperts);
    WeightManager manager(*loader);
    const auto plan = singleLayerCpuColdPlan(kExperts);
    const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));

    auto &registry = manager.expertGemmRegistry();
    ASSERT_TRUE(registry.removeEngineForDomain(
        "cpu_cold", DeviceId::cpu(), 0, 0, Role::GATE));
    EXPECT_FALSE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));
    EXPECT_NE(
        registry.getEngineForParticipant(
            "cpu_cold", DeviceId::cpu(), 0, 0, 0, 0, Role::GATE),
        nullptr);
    EXPECT_EQ(
        registry.getEngineForDomain(
            "cpu_cold", DeviceId::cpu(), 0, 0, Role::GATE),
        nullptr);
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, HydratesCpuFallbackParentsWhenParentsWereNotPreloaded)
{
    auto loader = MockModelLoader::createMinimal();
    const size_t d_model = 64;
    const size_t intermediate = 32;
    const size_t num_experts = 4;
    addSingleLayerExpertParents(loader, d_model, intermediate, num_experts);

    WeightManager manager(*loader);

    auto plan = singleLayerCpuColdPlan(num_experts);
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu()));

    std::vector<ITensorGemm *> gate;
    std::vector<ITensorGemm *> up;
    std::vector<ITensorGemm *> down;
    EXPECT_FALSE(manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, static_cast<int>(num_experts), gate, up, down));
    EXPECT_NE(gate[0], nullptr);
    EXPECT_NE(up[0], nullptr);
    EXPECT_NE(down[0], nullptr);
    EXPECT_EQ(gate[2], nullptr);
    EXPECT_EQ(up[2], nullptr);
    EXPECT_EQ(down[2], nullptr);
}

} // namespace llaminar2::test
