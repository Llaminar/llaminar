#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace llaminar2::test
{
namespace
{
    using Role = ExpertGemmRegistry::WeightRole;

    ExpertComputeDomain domainWith(
        const std::string &name,
        ExpertDomainKind kind,
        std::vector<GlobalDeviceAddress> participants,
        ExpertDomainComputeKind compute_kind,
        CollectiveBackendType backend)
    {
        ExpertComputeDomain domain;
        domain.name = name;
        domain.kind = kind;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.owner_rank = 0;
        domain.compute_kind = compute_kind;
        return domain;
    }

    ExpertRoutedTier tier(
        std::string name,
        std::string domain,
        int priority,
        bool fallback = false,
        int max_experts_per_layer = 0,
        size_t memory_budget_bytes = 0)
    {
        ExpertRoutedTier result;
        result.name = std::move(name);
        result.domain = std::move(domain);
        result.priority = priority;
        result.fallback = fallback;
        result.max_experts_per_layer = max_experts_per_layer;
        result.memory_budget_bytes = memory_budget_bytes;
        return result;
    }

    MoEExpertParallelPlan threeTierPlan(std::vector<ExpertLayerPlacement> placements)
    {
        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = "cuda_fast";
        plan.shared_expert_domain = "cuda_fast";
        plan.residency_policy = ExpertResidencyPolicy::ExplicitMasks;
        plan.domains = {
            domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                       {GlobalDeviceAddress::cuda(0)},
                       ExpertDomainComputeKind::ReplicatedExperts,
                       CollectiveBackendType::NCCL),
            domainWith("rocm_warm", ExpertDomainKind::LocalTP,
                       {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
                       CollectiveBackendType::RCCL),
            domainWith("cpu_cold", ExpertDomainKind::NodeLocalTP,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       ExpertDomainComputeKind::TensorParallelExperts,
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

    MoEExpertModelMetadata metadata(int layers, int experts)
    {
        MoEExpertModelMetadata model;
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
} // namespace

TEST(Test__WeightManagerMoEExpertOverlayPreparation, BuildsTierAwareRequestsAndDiagnostics)
{
    auto plan = std::make_shared<MoEExpertParallelPlan>(threeTierPlan({
        ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 0, 1, 2}},
        ExpertLayerPlacement{.layer = 1, .routed_expert_tier = {1, 2, 0, 1, 2, 0}},
    }));
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 1024);

    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cuda(0), 0, 3, Role::DOWN));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::cuda(0), 0, 1, Role::GATE));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::UP));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(1), 0, 1, Role::UP));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::cpu(), 0, 2, Role::GATE));
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
    EXPECT_EQ(cpu_stats->assigned_routed_experts, 4u);
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, SmallModelBudgetKeepsCudaPartialUnlessUncapped)
{
    MoEExpertParallelPlan capped;
    capped.enabled = true;
    capped.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    capped.continuation_domain = "cuda_fast";
    capped.shared_expert_domain = "cuda_fast";
    capped.residency_policy = ExpertResidencyPolicy::StaticById;
    capped.domains = {
        domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                   {GlobalDeviceAddress::cuda(0)},
                   ExpertDomainComputeKind::ReplicatedExperts,
                   CollectiveBackendType::NCCL),
        domainWith("cpu_cold", ExpertDomainKind::NodeLocalTP,
                   {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                   ExpertDomainComputeKind::TensorParallelExperts,
                   CollectiveBackendType::UPI),
    };
    capped.routed_tiers = {
        tier("hottest", "cuda_fast", 0, false, 1),
        tier("cold", "cpu_cold", 1, true),
    };

    const auto capped_result = MoEExpertParallelPlanner::plan(capped, metadata(1, 2));
    ASSERT_EQ(capped_result.planned_plan.placements.size(), 1u);
    EXPECT_EQ(capped_result.planned_plan.placements[0].routed_expert_tier,
              (std::vector<int>{0, 1}));

    auto capped_runtime = resolveMoEExpertOverlayRuntimePlan(
        std::make_shared<MoEExpertParallelPlan>(capped_result.planned_plan));
    const auto capped_prep = MoEExpertOverlayPreparationPlan::build(*capped_runtime, 128);
    const auto *capped_cuda = capped_prep.diagnostics().domainStats("cuda_fast", DeviceId::cuda(0));
    ASSERT_NE(capped_cuda, nullptr);
    EXPECT_EQ(capped_cuda->assigned_routed_experts, 1u);

    capped.routed_tiers[0].max_experts_per_layer = 0;
    const auto uncapped_result = MoEExpertParallelPlanner::plan(capped, metadata(1, 2));
    EXPECT_EQ(uncapped_result.planned_plan.placements[0].routed_expert_tier,
              (std::vector<int>{0, 0}));
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, KeepsSameDeviceDomainsSeparateInPreparationRequests)
{
    auto plan = std::make_shared<MoEExpertParallelPlan>();
    plan->enabled = true;
    plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    plan->continuation_domain = "cuda_fast";
    plan->shared_expert_domain = "cuda_fast";
    plan->residency_policy = ExpertResidencyPolicy::ExplicitMasks;
    plan->domains = {
        domainWith("cuda_fast", ExpertDomainKind::SingleDevice,
                   {GlobalDeviceAddress::cuda(0)},
                   ExpertDomainComputeKind::ReplicatedExperts,
                   CollectiveBackendType::NCCL),
        domainWith("cuda_warm", ExpertDomainKind::SingleDevice,
                   {GlobalDeviceAddress::cuda(0)},
                   ExpertDomainComputeKind::ReplicatedExperts,
                   CollectiveBackendType::NCCL),
        domainWith("cpu_cold", ExpertDomainKind::NodeLocalTP,
                   {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                   ExpertDomainComputeKind::TensorParallelExperts,
                   CollectiveBackendType::UPI),
    };
    plan->routed_tiers = {
        tier("fast", "cuda_fast", 0),
        tier("warm", "cuda_warm", 1),
        tier("cold", "cpu_cold", 2, true),
    };
    plan->placements = {
        ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 2}},
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

} // namespace llaminar2::test
