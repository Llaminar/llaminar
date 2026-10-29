/**
 * @file Test__WeightManagerMoEExpertOverlayPreparation.cpp
 * @brief Device-free ownership and preparation tests for ExpertOverlay weights.
 *
 * These cases validate that the preparation request is exact in both logical
 * expert identity and MPI ownership before any device packing occurs.  The
 * real-weight integration campaigns rely on this layer to prevent a rank from
 * loading a remote CPU/accelerator participant's expert slice. Canonical parent
 * readiness also controls retirement of independent typed-cache expert copies.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEOverlayCpuPreparationGate.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "loaders/WeightManager.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "kernels/PackedWeightsSerialization.h"
#include "loaders/PreparedWeightStore.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "models/GraphTypes.h"
#include "execution/moe/MoEExpertOverlayGraphScope.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "mocks/MockModelLoader.h"
#include "mocks/MockLocalTPContext.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <future>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace llaminar2::test
{
namespace
{
    using Role = ExpertGemmRegistry::WeightRole;

    /**
     * @brief Admit an explicit fixture owner envelope before native packing.
     * @param manager Actual production preparation owner.
     * @param bytes Test's admitted maximum; live assertions read PMA leases.
     * @param rank Rank whose independent host allocator is under test.
     * @return Sole materialization authority retained by the manager and storage.
     */
    std::shared_ptr<PhysicalMemoryAuthority> admitRoutedFixture(
        WeightManager &manager, size_t bytes = 1u << 20, int rank = 0)
    {
        PhysicalMemoryPlanBuilder plan;
        plan.add({rank, DeviceId::cpu(), 1u << 24, 1u << 24},
                 PhysicalMemoryOwner::RoutedExpertWeights, bytes);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), rank);
        manager.installPhysicalMemoryAuthority(authority);
        return authority;
    }

    /** @return Exact live routed bytes observed under the authority's ledger lock. */
    size_t liveRoutedBytes(const PhysicalMemoryAuthority &authority)
    {
        size_t bytes = 0;
        for (const auto &line : authority.rankAttestation())
            if (line.owner == PhysicalMemoryOwner::RoutedExpertWeights)
                bytes += line.materialized_new_bytes;
        return bytes;
    }

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
    admitRoutedFixture(manager);

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
    admitRoutedFixture(rank_zero_manager);
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
    admitRoutedFixture(rank_one_manager, 1u << 20, 1);
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
 * @brief Select the exact frozen parent for two colocated CPU expert endpoints.
 *
 * The auto planner can place both NUMA endpoints on one MPI rank. Their
 * bindings then have the same canonical name and generic CPU DeviceId, but
 * disjoint logical expert slices. Selecting the first name/device match for
 * both endpoints drops the upper-half experts during server startup.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     SelectsFrozenParentByColocatedCpuParticipantIdentity)
{
    constexpr size_t kModel = 64;
    constexpr size_t kIntermediate = 32;
    constexpr size_t kExperts = 4;
    auto plan = singleLayerCpuColdPlan(kExperts);
    plan->domains[0].scope = ExecutionDomainScope::RANK_LOCAL;
    plan->domains[0].world_ranks.clear();
    const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, kModel, kIntermediate, kExperts);
    WeightManager manager(*loader);
    admitRoutedFixture(manager);

    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.model_id = ModelContextId{8123};
    strategy.devices = {DeviceId::cpu()};
    ModelWeightSetBuilder builder(strategy);
    for (int participant = 0; participant < 2; ++participant)
    {
        for (const auto &[name, role, rows, cols] : {
                 std::tuple{"blk.0.ffn_gate_exps.weight", WeightRole::MoEExpertGate, kModel, kIntermediate},
                 std::tuple{"blk.0.ffn_up_exps.weight", WeightRole::MoEExpertUp, kModel, kIntermediate},
                 std::tuple{"blk.0.ffn_down_exps.weight", WeightRole::MoEExpertDown, kIntermediate, kModel},
             })
        {
            WeightBinding binding;
            binding.identity.canonical_name = name;
            binding.identity.role = role;
            binding.identity.derivation = WeightDerivationKind::ExpertSlice;
            binding.identity.layer = 0;
            binding.identity.overlay_domain = "cpu_cold";
            binding.identity.overlay_participant_index = participant;
            binding.identity.overlay_participant_world_rank = 0;
            binding.residency.home_device = DeviceId::cpu();
            binding.residency.resident_device = DeviceId::cpu();
            binding.slice.expert_start = static_cast<size_t>(participant) * 2u;
            binding.slice.expert_count = 2u;
            binding.slice.expert_ids = {participant * 2, participant * 2 + 1};
            binding.tensor_owner = createQ4_0WithData(
                {rows, cols, 2u},
                static_cast<uint32_t>(101 + participant * 3 +
                                      (role == WeightRole::MoEExpertUp ? 1 :
                                       role == WeightRole::MoEExpertDown ? 2 : 0)));
            binding.tensor = binding.tensor_owner.get();
            builder.addBinding(std::move(binding));
        }
    }
    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());

    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
        *runtime_plan, DeviceId::cpu(), &frozen));
    for (int participant = 0; participant < 2; ++participant)
    {
        for (int expert = participant * 2; expert < participant * 2 + 2; ++expert)
        {
            for (const auto role : {Role::GATE, Role::UP, Role::DOWN})
            {
                EXPECT_NE(
                    manager.expertGemmRegistry().getEngineForParticipant(
                        "cpu_cold", DeviceId::cpu(), 0, participant,
                        0, expert, role),
                    nullptr) << "participant=" << participant
                             << " expert=" << expert;
            }
        }
    }
}

/**
 * @brief Dependent graph construction joins only the CPU bank's startup edge.
 */
TEST(Test__MoEOverlayCpuPreparationGate,
     WaitsForExactPublisherAndRejectsDuplicatePublication)
{
    auto gate = std::make_shared<MoEOverlayCpuPreparationGate>();
    const auto publisher =
        MoEOverlayCpuPreparationParticipation::publisher(gate);
    const auto dependent =
        MoEOverlayCpuPreparationParticipation::dependent(gate);
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    auto graph_builder = std::async(std::launch::async, [&]
    {
        entered.set_value();
        dependent.afterLocalPreparation();
        return true;
    });
    entered_future.wait();
    EXPECT_EQ(graph_builder.wait_for(std::chrono::milliseconds(0)),
              std::future_status::timeout);
    publisher.afterLocalPreparation();
    EXPECT_TRUE(graph_builder.get());
    EXPECT_THROW(publisher.afterLocalPreparation(), std::logic_error);
}

/**
 * @brief Producer failure releases dependents with the original setup error.
 */
TEST(Test__MoEOverlayCpuPreparationGate,
     PropagatesProducerFailureWithoutASecondSetupPath)
{
    auto gate = std::make_shared<MoEOverlayCpuPreparationGate>();
    const auto publisher =
        MoEOverlayCpuPreparationParticipation::publisher(gate);
    const auto dependent =
        MoEOverlayCpuPreparationParticipation::dependent(gate);
    publisher.publishFailureIfProducer("frozen CPU parent missing");
    try
    {
        dependent.afterLocalPreparation();
        FAIL() << "dependent graph ignored producer failure";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("frozen CPU parent missing"),
                  std::string::npos);
    }
    EXPECT_THROW(publisher.afterLocalPreparation(), std::logic_error);
    EXPECT_THROW(
        MoEOverlayCpuPreparationParticipation::dependent(nullptr),
        std::invalid_argument);
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
    admitRoutedFixture(manager);
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
    admitRoutedFixture(manager);
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
    admitRoutedFixture(manager);
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

    admitRoutedFixture(manager);

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

/** @test Typed CPU cache identities resolve canonical parent readiness while preserving CPU retention. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, CanonicalParentReadinessReleasesOnlyReleasableCachedCopies)
{
    constexpr size_t experts=4;
    auto loader=MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader,64,32,experts);
    WeightManager manager(*loader);
    admitRoutedFixture(manager);
    WeightShardingConfig sharding;
    sharding.default_mode=WeightShardingMode::Replicate;
    manager.setWeightShardingConfig(sharding);
    const auto assignments=std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(
        2,4,2,64,128,std::vector<DeviceId>{DeviceId::cpu(),DeviceId::cpu()}));
    manager.setTensorParallelConfig(assignments);
    std::vector<std::shared_ptr<TensorBase>> copies;
    for(const auto *name:{"blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"})
    {
        const auto source=loader->loadTensor(name);
        for(int rank=0;rank<2;++rank)
        {
            auto copy=manager.getShardedWeightForAssignment(name,DeviceId::cpu(),assignments->forRank(rank),0);
            ASSERT_TRUE(copy);ASSERT_NE(copy,source);
            ASSERT_NE(copy->raw_data(),source->raw_data());
            ASSERT_FALSE(copy->hasPreparedDeviceState());
            copies.push_back(std::move(copy));
        }
    }
    // Retention remains tensor-owned: readiness of another consumer cannot
    // remove a declared CPU source, even when both use the same parent name.
    auto *metadata=manager.weightMetadataRegistry();
    auto retained_residency=metadata->residency(copies.front().get());
    ASSERT_TRUE(retained_residency);
    retained_residency->host_policy=WeightHostPolicy::RequiredForCPUExecution;
    metadata->updateResidency(copies.front().get(),*retained_residency);
    manager.markMaterializationComplete();
    manager.markDevicePreparationComplete();
    manager.markGraphMaterializationComplete();
    EXPECT_EQ(manager.releaseAllHostWeightData(),0u);
    for(const auto &copy:copies)EXPECT_FALSE(copy->is_raw_data_released());

    // The actual CPU preparation publishes canonical parent READY tickets.
    // These independent cache copies do not acquire prepared-device metadata;
    // only the canonical parent readiness branch can authorize retirement.
    const auto runtime=resolveMoEExpertOverlayRuntimePlan(singleLayerCpuColdPlan(experts));
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu()));
    for(const auto &copy:copies)ASSERT_FALSE(copy->hasPreparedDeviceState());
    manager.releaseAllHostWeightData();
    EXPECT_FALSE(copies.front()->is_raw_data_released());
    for(size_t i=1;i<copies.size();++i)EXPECT_TRUE(copies[i]->is_raw_data_released());
    EXPECT_EQ(manager.releaseAllHostWeightData(),0u);
    // The second broad sweep also preserves tensor-owned CPU retention.
    copies.front()->setHostResident();
    manager.releaseHostResidentWeightData();
    EXPECT_FALSE(copies.front()->is_raw_data_released());
}

/** @test Source and real preparation select the same logical CPU owner despite shared storage/reordering. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, ExactCPUGraphScopeOwnsOnlyItsSourcesAndPreparation)
{
    for(const bool reverse:{false,true})
    {
        SCOPED_TRACE(reverse);
        auto source=singleLayerCpuColdPlan(4);
        source->domains[0].scope=ExecutionDomainScope::RANK_LOCAL;
        source->domains[0].world_ranks.clear();
        if(reverse)std::reverse(source->domains[0].participants.begin(),source->domains[0].participants.end());
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(source);
        auto loader=MockModelLoader::createMinimal();
        // MockModelLoader's exact expert-selection API supports FP32. Use
        // that supported source so real materialization and CPU preparation
        // operate on compact two-expert tensors, never a full-parent substitute.
        loader->addFP32RandomTensor("blk.0.ffn_gate_exps.weight", {64, 32, 4}, -1.0f, 1.0f, 101);
        loader->addFP32RandomTensor("blk.0.ffn_up_exps.weight", {64, 32, 4}, -1.0f, 1.0f, 102);
        loader->addFP32RandomTensor("blk.0.ffn_down_exps.weight", {32, 64, 4}, -1.0f, 1.0f, 103);
        const auto all=MoEExpertOverlayPreparationPlan::build(*runtime,*loader);

        // Original device-only production selection retains both NUMA owners.
        // This concrete control must reproduce the graph-facing ambiguity.
        WeightManager legacy_manager(*loader);
        admitRoutedFixture(legacy_manager);
        const auto legacy_plan=all.filteredForDevice(DeviceId::cpu()).sourceWeightPlan(*loader,ModelContextId{920});
        ASSERT_EQ(legacy_plan.size(),6u);
        const auto legacy=legacy_manager.materialize(legacy_plan);
        const auto legacy_bindings=makeModelWeightBindings(legacy,DeviceId::cpu());
        EXPECT_THROW(legacy_bindings.get_layer_weights(0),std::runtime_error);

        WeightManager manager(*loader);

        admitRoutedFixture(manager);
        MockLocalTPContext local;
        local.setDevices({GlobalDeviceAddress::cpu(1),GlobalDeviceAddress::cpu(0)});
        std::vector<int> seen(4);
        // Visit by the TP context's exact address, independently of domain order.
        for(int graph_index=0;graph_index<2;++graph_index)
        {
            const auto &endpoint=local.devices()[graph_index];
            const auto found=std::find(source->domains[0].participants.begin(),source->domains[0].participants.end(),endpoint);
            ASSERT_NE(found,source->domains[0].participants.end());
            const int participant=static_cast<int>(found-source->domains[0].participants.begin());
            const auto scope=MoEExpertOverlayGraphScope::resolveForTPParticipant(
                *runtime,DeviceId::cpu(),&local,graph_index);
            const auto scoped=all.filteredForGraphScope(*runtime,scope);
            const auto plan=scoped.sourceWeightPlan(*loader,ModelContextId{921});
            ASSERT_EQ(plan.size(),3u);
            for(const auto &requirement:plan.requirements())
            {
                EXPECT_EQ(requirement.overlay_domain,"cpu_cold");
                EXPECT_EQ(requirement.overlay_participant_world_rank,0);
                EXPECT_EQ(requirement.overlay_participant_index,participant);
                EXPECT_EQ(requirement.slice.expert_ids,(std::vector<int>{participant*2,participant*2+1}));
            }
            const auto frozen=manager.materialize(plan);
            for (const auto &binding : frozen.bindings())
            {
                ASSERT_TRUE(binding.tensor);
                ASSERT_EQ(binding.tensor->shape().size(), 3u);
                EXPECT_EQ(binding.tensor->shape().back(), 2u);
                EXPECT_EQ(binding.slice.expert_count, 2u);
                EXPECT_EQ(binding.slice.expert_ids, (std::vector<int>{participant * 2, participant * 2 + 1}));
            }
            const auto bindings=makeModelWeightBindings(frozen,DeviceId::cpu());
            const auto layer=bindings.get_layer_weights(0);
            ASSERT_TRUE(layer.moe_gate_exps);ASSERT_TRUE(layer.moe_up_exps);ASSERT_TRUE(layer.moe_down_exps);
            EXPECT_EQ(layer.moe_gate_exps->identity.overlay_participant_index,participant);
            ASSERT_TRUE(manager.prepareMoEExpertOverlayWeightsForGraph(*runtime,scope,&frozen,nullptr,
                PreparedWeightAdmission::AllocateCompleteSet));
            for(const auto &request:scoped.requests())
            {
                EXPECT_EQ(request.participant_index,participant);
                EXPECT_NE(manager.expertGemmRegistry().getEngineForParticipant(
                    request.domain_name,request.device,request.participant_world_rank,request.participant_index,
                    request.layer,request.expert_id,request.role),nullptr);
                // No engine may be published under a sibling's ownership tuple.
                EXPECT_EQ(manager.expertGemmRegistry().getEngineForParticipant(
                    request.domain_name,request.device,request.participant_world_rank,1-participant,
                    request.layer,request.expert_id,request.role),nullptr);
                if(request.role==Role::GATE)++seen[request.expert_id];
            }
        }
        EXPECT_EQ(seen,(std::vector<int>{1,1,1,1}));
    }
}

/** @test Exact scopes fail closed on missing, ambiguous, stale and foreign logical identity. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, ExactGraphScopeRejectsAmbiguousAndForeignIdentity)
{
    auto source=singleLayerCpuColdPlan(4);
    source->domains[0].scope=ExecutionDomainScope::RANK_LOCAL;
    source->domains[0].world_ranks.clear();
    const auto runtime=resolveMoEExpertOverlayRuntimePlan(source);
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cpu(),GlobalDeviceAddress::cpu(7)),std::invalid_argument);
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cuda(0),GlobalDeviceAddress::cpu(0)),std::invalid_argument);
    MockLocalTPContext local;
    local.setDevices({GlobalDeviceAddress::cpu(1),GlobalDeviceAddress::cpu(0)});
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolveForTPParticipant(*runtime,DeviceId::cpu(),&local,-1),std::invalid_argument);
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolveForTPParticipant(*runtime,DeviceId::cpu(),&local,2),std::invalid_argument);
    const auto scope=MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cpu(),GlobalDeviceAddress::cpu(1));
    EXPECT_TRUE(scope.contains("cpu_cold",DeviceId::cpu(),0,1));
    EXPECT_FALSE(scope.contains("wrong_domain",DeviceId::cpu(),0,1));
    EXPECT_FALSE(scope.contains("cpu_cold",DeviceId::cpu(),1,1));
    EXPECT_FALSE(scope.contains("cpu_cold",DeviceId::cpu(),0,0));
    EXPECT_FALSE(scope.contains("cpu_cold",DeviceId::cuda(1),0,1));
    const auto other_runtime=resolveMoEExpertOverlayRuntimePlan(source);
    EXPECT_THROW(scope.validate(*other_runtime,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(scope.validate(*runtime,DeviceId::cuda(0)),std::invalid_argument);
    const auto prep=MoEExpertOverlayPreparationPlan::build(*runtime,128);
    EXPECT_THROW(prep.filteredForGraphScope(*other_runtime,scope),std::invalid_argument);
    // Existing distributed CPU owners remain device-unambiguous in each MPI rank.
    const auto distributed=resolveMoEExpertOverlayRuntimePlan(singleLayerCpuColdPlan(4));
    const auto single=MoEExpertOverlayGraphScope::resolve(*distributed,DeviceId::cpu());
    EXPECT_TRUE(single.contains("cpu_cold",DeviceId::cpu(),0,0));
}

/** @test Only routed owners and selected endpoints require rank proof at graph selection. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, ExactGraphScopePreservesUnusedDomainsAndRejectsUnboundOwners)
{
    auto source=singleLayerCpuColdPlan(4);
    RoutedExpertDomain unbound;
    unbound.name="unbound_remote";
    unbound.scope=ExecutionDomainScope::SINGLE;
    unbound.backend=CollectiveBackendType::NCCL;
    unbound.participants={GlobalDeviceAddress::cuda(0,NUMA_NODE_UNKNOWN,"unbound-host")};
    source->domains.push_back(unbound);
    // A domain not named by any routed tier is not an expert owner. Existing
    // runtime metadata may retain it unbound without changing this graph scope.
    const auto runtime=resolveMoEExpertOverlayRuntimePlan(source);
    const auto *foreign=runtime->domainForName("unbound_remote");
    ASSERT_TRUE(foreign);ASSERT_EQ(foreign->participants.size(),1u);
    ASSERT_FALSE(foreign->participants.front().world_rank_known);
    ASSERT_EQ(foreign->routed_tier_count,0u);
    const auto scope=MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cpu(),GlobalDeviceAddress::cpu(0));
    EXPECT_TRUE(scope.contains("cpu_cold",DeviceId::cpu(),0,0));
    EXPECT_FALSE(scope.contains("unbound_remote",DeviceId::cuda(0),0,0));
    // Selecting that endpoint itself still requires an authenticated MPI owner.
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolve(*runtime,DeviceId::cuda(0),unbound.participants.front()),
        std::invalid_argument);

    // Adding a routed tier makes its participants actual owner-map entries,
    // even before any expert is assigned there. Match the original factory's
    // fail-fast owner-map boundary, rather than broadening runtime validation.
    auto routed_source=std::make_shared<MoERoutedExpertPlacementPlan>(*source);
    routed_source->routed_tiers.front().priority=2;
    routed_source->routed_tiers.push_back(tier("unbound","unbound_remote",1));
    const auto owner_runtime=resolveMoEExpertOverlayRuntimePlan(routed_source);
    ASSERT_EQ(owner_runtime->domainForName("unbound_remote")->routed_tier_count,1u);
    const auto owners=MoEExpertOwnerMap::build(*routed_source);
    const auto unbound_owner=std::find_if(owners.participants().begin(),owners.participants().end(),
        [](const auto &owner){return owner.domain_name=="unbound_remote";});
    ASSERT_NE(unbound_owner,owners.participants().end());
    ASSERT_FALSE(unbound_owner->world_rank_known);
    EXPECT_THROW(MoEExpertOverlayGraphScope::resolve(*owner_runtime,DeviceId::cpu(),GlobalDeviceAddress::cpu(0)),
        std::invalid_argument);
}

/** @test Canonical host/NUMA/device identities are copied unchanged for CPU and both GPU backends. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, ExactGraphScopePreservesCanonicalCPUAndGPUAddresses)
{
    const std::vector<GlobalDeviceAddress> endpoints{
        GlobalDeviceAddress::cpu(1,"canonical-node"),
        GlobalDeviceAddress::cuda(3,1,"canonical-node"),
        GlobalDeviceAddress::rocm(2,1,"canonical-node")};
    for(const auto &endpoint:endpoints)
    {
        SCOPED_TRACE(endpoint.toString());
        auto source=singleLayerCpuColdPlan(4);
        source->domains[0].scope=ExecutionDomainScope::SINGLE;
        // This fixture projects one endpoint, so its inherited NodeTP dense
        // continuation and shared-expert policy must also become singleton.
        source->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::Replicated);
        source->domains[0].participants={endpoint};
        source->domains[0].world_ranks={0};
        source->domains[0].backend=endpoint.isCPU()?CollectiveBackendType::UPI:
            endpoint.device_type==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(source);
        ASSERT_EQ(runtime->continuationDomain().participants.front().address,endpoint);
        MockLocalTPContext local;
        local.setDevices(source->domains[0].participants);
        const auto scope=MoEExpertOverlayGraphScope::resolveForTPParticipant(
            *runtime,endpoint.toLocalDeviceId(),&local,0);
        EXPECT_TRUE(scope.contains("cpu_cold",endpoint.toLocalDeviceId(),0,0));
        // Host aliases are not reinterpreted at this post-normalization boundary.
        auto unnormalized=endpoint;unnormalized.hostname="localhost";
        local.setDevices({unnormalized});
        EXPECT_THROW(MoEExpertOverlayGraphScope::resolveForTPParticipant(
            *runtime,endpoint.toLocalDeviceId(),&local,0),std::invalid_argument);
    }
}

/**
 * @test GPU continuation scope owns both colocated CPU groups while its sibling owns only itself.
 * @details Only production request/source metadata is constructed; no GPU backend is initialized.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, GraphScopeSeparatesGPUContinuationCPUOwnersFromSibling)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        const auto gpu_address = [backend](int ordinal)
        {
            return backend == DeviceType::CUDA ? GlobalDeviceAddress::cuda(ordinal) :
                                                 GlobalDeviceAddress::rocm(ordinal);
        };
        auto source = std::make_shared<MoERoutedExpertPlacementPlan>();
        source->enabled = true;
        source->topology = RoutedExpertPlacementTopology::TieredOverlay;
        source->continuation_domain = "gpu_main";
        source->shared_expert_domain = "gpu_main";
        source->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        source->domains = {
            domainWith("gpu_main", ExecutionDomainScope::RANK_LOCAL,
                       {gpu_address(0), gpu_address(1)}, RoutedExpertComputePolicy::Apportioned,
                       backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL),
            domainWith("cpu_cold", ExecutionDomainScope::RANK_LOCAL,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       RoutedExpertComputePolicy::Apportioned, CollectiveBackendType::UPI),
        };
        source->routed_tiers = {tier("hot", "gpu_main", 0), tier("cold", "cpu_cold", 1, true)};
        source->placements = {
            RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 0, 0, 0, 1, 1, 1, 1}},
        };
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(source);
        auto loader = MockModelLoader::createMinimal();
        addSingleLayerExpertParents(loader, 64, 32, 8);
        const auto all = MoEExpertOverlayPreparationPlan::build(*runtime, *loader);
        ASSERT_EQ(all.requests().size(), 24u);

        // TP ordering is deliberately independent of the domain declaration.
        MockLocalTPContext local;
        local.setDevices({gpu_address(1), gpu_address(0)});
        const auto root_device = gpu_address(0).toLocalDeviceId();
        const auto sibling_device = gpu_address(1).toLocalDeviceId();
        const auto root = MoEExpertOverlayGraphScope::resolveForTPParticipant(
            *runtime, root_device, &local, 1, MoEOverlayColocatedCPUOwnership::CurrentRank);
        const auto sibling = MoEExpertOverlayGraphScope::resolveForTPParticipant(
            *runtime, sibling_device, &local, 0);
        EXPECT_TRUE(root.contains("gpu_main", root_device, 0, 0));
        EXPECT_TRUE(root.contains("cpu_cold", DeviceId::cpu(), 0, 0));
        EXPECT_TRUE(root.contains("cpu_cold", DeviceId::cpu(), 0, 1));
        EXPECT_FALSE(root.contains("gpu_main", sibling_device, 0, 1));
        EXPECT_TRUE(sibling.contains("gpu_main", sibling_device, 0, 1));
        EXPECT_FALSE(sibling.contains("cpu_cold", DeviceId::cpu(), 0, 0));
        EXPECT_FALSE(sibling.contains("cpu_cold", DeviceId::cpu(), 0, 1));
        EXPECT_THROW(MoEExpertOverlayGraphScope::resolveForTPParticipant(
            *runtime, sibling_device, &local, 0, MoEOverlayColocatedCPUOwnership::CurrentRank),
            std::invalid_argument);

        const auto root_preparation = all.filteredForGraphScope(*runtime, root);
        const auto sibling_preparation = all.filteredForGraphScope(*runtime, sibling);
        EXPECT_EQ(root_preparation.requests().size(), 18u);
        EXPECT_EQ(sibling_preparation.requests().size(), 6u);
        const auto root_sources = root_preparation.sourceWeightPlan(*loader, ModelContextId{930});
        const auto sibling_sources = sibling_preparation.sourceWeightPlan(*loader, ModelContextId{930});
        EXPECT_EQ(root_sources.size(), 9u);
        EXPECT_EQ(sibling_sources.size(), 3u);
        std::set<int> root_cpu_participants;
        for (const auto &requirement : root_sources.requirements())
        {
            EXPECT_TRUE(root.contains(requirement.overlay_domain, requirement.target_device,
                requirement.overlay_participant_world_rank, requirement.overlay_participant_index));
            EXPECT_EQ(requirement.slice.expert_ids.size(), 2u);
            if (requirement.target_device.is_cpu())
                root_cpu_participants.insert(requirement.overlay_participant_index);
            else
                EXPECT_EQ(requirement.target_device, root_device);
        }
        EXPECT_EQ(root_cpu_participants, (std::set<int>{0, 1}));
        for (const auto &requirement : sibling_sources.requirements())
        {
            EXPECT_EQ(requirement.target_device, sibling_device);
            EXPECT_EQ(requirement.overlay_domain, "gpu_main");
            EXPECT_EQ(requirement.overlay_participant_world_rank, 0);
            EXPECT_EQ(requirement.overlay_participant_index, 1);
            EXPECT_EQ(requirement.slice.expert_ids, (std::vector<int>{2, 3}));
        }
    }
}

/** @test Distributed rank one retains its exact CPU source tuple and excludes rank zero. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation, GraphScopeSelectsDistributedRankOneSources)
{
    const auto source = singleLayerCpuColdPlan(4);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(source,
        MoEExpertOverlayRuntimeResolverOptions{
            .current_world_rank = 1,
            .validate_mvp_root_reachability = false,
        });
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, 4);
    const auto scope = MoEExpertOverlayGraphScope::resolveForTPParticipant(
        *runtime, DeviceId::cpu(), nullptr, 0);
    EXPECT_TRUE(scope.contains("cpu_cold", DeviceId::cpu(), 1, 1));
    EXPECT_FALSE(scope.contains("cpu_cold", DeviceId::cpu(), 0, 0));
    const auto preparation = MoEExpertOverlayPreparationPlan::build(*runtime, *loader)
        .filteredForGraphScope(*runtime, scope);
    ASSERT_EQ(preparation.requests().size(), 6u);
    for (const auto &request : preparation.requests())
    {
        EXPECT_TRUE(request.participant_world_rank_known);
        EXPECT_EQ(request.participant_world_rank, 1);
        EXPECT_EQ(request.participant_index, 1);
        EXPECT_GE(request.expert_id, 2);
        EXPECT_LT(request.expert_id, 4);
    }
    const auto sources = preparation.sourceWeightPlan(*loader, ModelContextId{931});
    ASSERT_EQ(sources.size(), 3u);
    for (const auto &requirement : sources.requirements())
    {
        EXPECT_EQ(requirement.target_device, DeviceId::cpu());
        EXPECT_EQ(requirement.overlay_domain, "cpu_cold");
        EXPECT_EQ(requirement.overlay_participant_world_rank, 1);
        EXPECT_EQ(requirement.overlay_participant_index, 1);
        EXPECT_EQ(requirement.slice.expert_ids, (std::vector<int>{2, 3}));
    }
}

/**
 * @test Initial CPU packs retain one physical claim across aliases and graph reuse.
 *
 * Exercise real quantized and floating owning engines with one/two local CPU
 * participants and reversed ownership order. Registry clearing retires only
 * aliases; a retained graph consumer keeps its exact allocation lease alive.
 */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     InitialCpuPhysicalClaimsFollowStorageAcrossAliasesAndFamilies)
{
    for (const bool floating : {false, true})
    for (const int participants : {1, 2})
    for (const bool reverse : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "floating=" << floating
            << " participants=" << participants << " reverse=" << reverse);
        auto loader = MockModelLoader::createMinimal();
        if (floating)
        {
            loader->addFP32RandomTensor("blk.0.ffn_gate_exps.weight", {64, 32, 4}, -1.0f, 1.0f, 101);
            loader->addFP32RandomTensor("blk.0.ffn_up_exps.weight", {64, 32, 4}, -1.0f, 1.0f, 102);
            loader->addFP32RandomTensor("blk.0.ffn_down_exps.weight", {32, 64, 4}, -1.0f, 1.0f, 103);
        }
        else
            addSingleLayerExpertParents(loader, 64, 32, 4);
        auto plan = singleLayerCpuColdPlan(4);
        plan->domains[0].scope = ExecutionDomainScope::RANK_LOCAL;
        plan->domains[0].world_ranks.clear();
        plan->domains[0].participants.resize(participants);
        if (participants == 1)
            plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::Replicated);
        if (reverse)
            std::reverse(plan->domains[0].participants.begin(), plan->domains[0].participants.end());
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(plan);
        WeightManager manager(*loader);
        const auto authority = admitRoutedFixture(manager);
        EXPECT_EQ(liveRoutedBytes(*authority), 0u);
        ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime, DeviceId::cpu()));
        const size_t live = liveRoutedBytes(*authority);
        ASSERT_GT(live, 0u);
        std::shared_ptr<ITensorGemm> consumer;
        for (int participant = 0; participant < participants; ++participant)
        {
            const int expert = participant * (4 / participants);
            auto exact = manager.expertGemmRegistry().getEngineLifetimeForParticipant(
                "cpu_cold", DeviceId::cpu(), 0, participant, 0, expert, Role::GATE);
            auto alias = manager.expertGemmRegistry().getEngineLifetimeForDomain(
                "cpu_cold", DeviceId::cpu(), 0, expert, Role::GATE);
            ASSERT_TRUE(exact);
            ASSERT_EQ(exact.get(), alias.get());
            EXPECT_FALSE(exact.owner_before(alias));
            EXPECT_FALSE(alias.owner_before(exact));
            consumer = std::move(exact);
        }
        // A second graph family reuses the certified canonical engines and
        // cannot re-charge storage or choose another physical allocation.
        manager.markMaterializationComplete();
        manager.markDevicePreparationComplete();
        manager.markGraphMaterializationComplete();
        ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(
            *runtime, DeviceId::cpu(), nullptr, nullptr,
            PreparedWeightAdmission::ReuseCertifiedCompleteSet));
        EXPECT_EQ(liveRoutedBytes(*authority), live);
        manager.expertGemmRegistry().clear();
        const auto retained = liveRoutedBytes(*authority);
        EXPECT_GT(retained, 0u);
        EXPECT_LT(retained, live);
        consumer.reset();
        EXPECT_EQ(liveRoutedBytes(*authority), 0u);
    }
}

/** @test Missing admission and insufficient owner capacity never publish partial CPU engines. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     InitialCpuPhysicalClaimsRejectMissingAndInsufficientAdmission)
{
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, 4);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(singleLayerCpuColdPlan(4));
    WeightManager missing(*loader);
    EXPECT_THROW(missing.prepareMoEExpertOverlayWeights(*runtime, DeviceId::cpu()), std::logic_error);
    EXPECT_EQ(missing.expertGemmRegistry().size(), 0u);
    WeightManager insufficient(*loader);
    const auto authority = admitRoutedFixture(insufficient, 1);
    EXPECT_FALSE(insufficient.prepareMoEExpertOverlayWeights(*runtime, DeviceId::cpu()));
    EXPECT_EQ(insufficient.expertGemmRegistry().size(), 0u);
    EXPECT_EQ(liveRoutedBytes(*authority), 0u);
}

/** @test A full durable owner can export and receive through separately admitted staging. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     FullPersistentBudgetExportsReceivesAndRetiresStagingWithoutRechargingWeights)
{
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, 4);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(singleLayerCpuColdPlan(4));
    // Record exact physical storage from an actual native pack; this test's
    // second authority admits precisely that recorded durable geometry.
    size_t durable_bytes = 0;
    {
        WeightManager probe(*loader);
        const auto authority = admitRoutedFixture(probe);
        ASSERT_TRUE(probe.prepareMoEExpertOverlayWeights(*runtime, DeviceId::cpu()));
        durable_bytes = liveRoutedBytes(*authority);
    }
    ASSERT_GT(durable_bytes, 0u);
    PhysicalMemoryPlanBuilder plan;
    const PhysicalMemoryResource host{0, DeviceId::cpu(), 1u << 24, 1u << 24};
    plan.add(host, PhysicalMemoryOwner::RoutedExpertWeights, durable_bytes);
    plan.add(host, PhysicalMemoryOwner::ExpertMigrationStaging, 1u << 20);
    const auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    WeightManager manager(*loader);
    manager.installPhysicalMemoryAuthority(authority);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime, DeviceId::cpu()));
    ASSERT_EQ(liveRoutedBytes(*authority), durable_bytes);
    auto engine = manager.expertGemmRegistry().getEngineLifetimeForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, 0, Role::GATE);
    ASSERT_TRUE(engine);
    const auto *source = engine->exportCPUNativeVNNIPackedWeights();
    ASSERT_NE(source, nullptr);
    const auto staging = CPUWeightStoragePlacement::accounted(authority, DeviceId::cpu(),
        PhysicalMemoryOwner::ExpertMigrationStaging, CPUWeightStoragePlacement::local());
    auto live_staging = [&]()
    {
        for (const auto &line : authority->rankAttestation())
            if (line.owner == PhysicalMemoryOwner::ExpertMigrationStaging)
                return line.materialized_new_bytes;
        return size_t{0};
    };
    for (int replay = 0; replay < 3; ++replay)
    {
        EXPECT_EQ(live_staging(), 0u);
        {
            auto clone = engine->cloneWeightsTo(staging);
            ASSERT_TRUE(clone);
            ExpertTransferBuffer wire;
            ASSERT_TRUE(packed_weights_serialization::serializeInto(*source, wire, staging));
            auto received = packed_weights_serialization::deserialize(wire.data(), wire.size(), staging);
            ASSERT_TRUE(received);
            packed_weights_serialization::PackedWeightsTransferDescriptor descriptor;
            packed_weights_serialization::PackedWeightsConstSectionViews sections;
            ASSERT_TRUE(packed_weights_serialization::describeTransfer(*received, descriptor, sections));
            auto direct = packed_weights_serialization::allocateTransferTarget(descriptor, staging);
            for (size_t section = 0; section < sections.sizes.size(); ++section)
                if (sections.sizes[section])
                    std::memcpy(direct.data[section], sections.data[section], sections.sizes[section]);
            ExpertTransferBuffer replayed;
            ASSERT_TRUE(packed_weights_serialization::serializeInto(*direct.weights, replayed, staging));
            ASSERT_EQ(wire.size(), replayed.size());
            EXPECT_EQ(std::memcmp(wire.data(), replayed.data(), wire.size()), 0);
            EXPECT_GT(live_staging(), 0u);
            EXPECT_EQ(liveRoutedBytes(*authority), durable_bytes);
        }
        EXPECT_EQ(live_staging(), 0u);
        EXPECT_EQ(liveRoutedBytes(*authority), durable_bytes);
    }
    manager.expertGemmRegistry().clear();
    engine.reset();
    EXPECT_EQ(liveRoutedBytes(*authority), 0u);
}

/** @test Standalone CPU MoE retains the frozen plan's additional-model owner. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     StandaloneCpuOwnerComesFromFrozenBindingsAndRejectsMissingOwner)
{
    const FrozenModelWeightSet frozen(InferenceStrategy{}, {}, PhysicalMemoryOwner::AdditionalModelWeights);
    const auto bindings = makeModelWeightBindings(frozen, DeviceId::cpu());
    EXPECT_EQ(bindings.physical_memory_owner, PhysicalMemoryOwner::AdditionalModelWeights);
    auto loader = MockModelLoader::createMinimal();
    addSingleLayerExpertParents(loader, 64, 32, 4);
    const auto gate = loader->loadTensor("blk.0.ffn_gate_exps.weight");
    const auto up = loader->loadTensor("blk.0.ffn_up_exps.weight");
    const auto down = loader->loadTensor("blk.0.ffn_down_exps.weight");
    PhysicalMemoryPlanBuilder plan;
    plan.add({0, DeviceId::cpu(), 1u << 24, 1u << 24},
        bindings.physical_memory_owner, 1u << 20);
    const auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    {
        PreparedWeightStore store;
        store.installPhysicalMemoryAuthority(authority);
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.num_experts = 4;
        params.expert_intermediate = 32;
        params.d_model = 64;
        params.layer_idx = 0;
        params.gate_exps = gate.get();
        params.up_exps = up.get();
        params.down_exps = down.get();
        params.prepared_store = &store;
        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
        EXPECT_THROW(MoEExpertComputeStage::prepareExpertGemmEngines(params), std::invalid_argument);
        params.persistent_weight_owner = bindings.physical_memory_owner;
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));
        bool additional_live = false;
        for (const auto &line : authority->rankAttestation())
        {
            if (line.owner == PhysicalMemoryOwner::PrimaryModelWeights)
            {
                EXPECT_EQ(line.materialized_new_bytes, 0u);
                EXPECT_EQ(line.committed_new_bytes, 0u);
            }
            if (line.owner == PhysicalMemoryOwner::AdditionalModelWeights)
                additional_live = line.materialized_new_bytes > 0;
        }
        EXPECT_TRUE(additional_live);
    }
    for (const auto &line : authority->rankAttestation())
        EXPECT_EQ(line.materialized_new_bytes, 0u);
}

/** @test Cache-owned CPU prepacking retains the actual primary pack across source release and replay. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     PrimaryCpuCachePrepackRetainsClaimsAcrossRawReleaseReplayAndFinalStoreRetirement)
{
    for (const auto device : {DeviceId::cpu(), DeviceId{DeviceType::CPU, 1}})
    {
        SCOPED_TRACE(device.toString());
        auto loader = MockModelLoader::createMinimal();
        loader->addTensor("blk.0.attn_q.weight", createQ4_0WithData({32, 64}, 777));
        PhysicalMemoryPlanBuilder plan;
        plan.add({0, device, 1u << 24, 1u << 24}, PhysicalMemoryOwner::PrimaryModelWeights, 1u << 20);
        const auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        auto live = [&]()
        {
            size_t bytes = 0;
            for (const auto &line : authority->rankAttestation())
                bytes += line.materialized_new_bytes;
            return bytes;
        };
        std::shared_ptr<PreparedWeightStore> retained_store;
        size_t packed_bytes = 0;
        {
            WeightManager manager(*loader);
            WeightShardingConfig sharding;
            sharding.default_mode = WeightShardingMode::Replicate;
            manager.setWeightShardingConfig(sharding);
            manager.installPhysicalMemoryAuthority(authority);
            const auto source = manager.getWeightForDevice("blk.0.attn_q.weight", device);
            ASSERT_TRUE(source);
            ASSERT_TRUE(source->raw_data());
            ASSERT_TRUE(manager.packGemmWeights(device, nullptr, true));
            EXPECT_TRUE(source->is_raw_data_released());
            packed_bytes = live();
            ASSERT_GT(packed_bytes, 0u);
            retained_store = manager.preparedWeightStore();
            for (int replay = 0; replay < 3; ++replay)
            {
                ASSERT_TRUE(manager.packGemmWeights(device, nullptr, true));
                EXPECT_EQ(live(), packed_bytes);
            }
        }
        EXPECT_EQ(live(), packed_bytes);
        retained_store.reset();
        EXPECT_EQ(live(), 0u);
    }
}

/** @test Cache-owned prepacking rejects unadmitted physical bytes before source retirement. */
TEST(Test__WeightManagerMoEExpertOverlayPreparation,
     PrimaryCpuCachePrepackRejectsMissingAndInsufficientAdmission)
{
    for (const bool admitted : {false, true})
    {
        auto loader = MockModelLoader::createMinimal();
        loader->addTensor("blk.0.attn_q.weight", createQ4_0WithData({32, 64}, 777));
        WeightManager manager(*loader);
        WeightShardingConfig sharding;
        sharding.default_mode = WeightShardingMode::Replicate;
        manager.setWeightShardingConfig(sharding);
        std::shared_ptr<PhysicalMemoryAuthority> authority;
        if (admitted)
        {
            PhysicalMemoryPlanBuilder plan;
            plan.add({0, DeviceId::cpu(), 1u << 24, 1u << 24}, PhysicalMemoryOwner::PrimaryModelWeights, 1);
            authority = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            manager.installPhysicalMemoryAuthority(authority);
        }
        const auto source = manager.getWeightForDevice("blk.0.attn_q.weight");
        ASSERT_TRUE(source);
        EXPECT_FALSE(manager.packGemmWeights(DeviceId::cpu(), nullptr, true));
        EXPECT_FALSE(source->is_raw_data_released());
        if (authority)
            for (const auto &line : authority->rankAttestation())
                EXPECT_EQ(line.materialized_new_bytes, 0u);
    }
}

} // namespace llaminar2::test
