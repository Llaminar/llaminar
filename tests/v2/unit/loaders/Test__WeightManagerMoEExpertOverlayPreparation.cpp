#include <gtest/gtest.h>

#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertParallelPlanner.h"
#include "loaders/WeightManager.h"
#include "mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
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
                       ExpertDomainComputeKind::ReplicatedExperts,
                       CollectiveBackendType::RCCL),
            domainWith("cpu_cold", ExpertDomainKind::NodeLocalTP,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       ExpertDomainComputeKind::ReplicatedExperts,
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

    std::shared_ptr<MoEExpertParallelPlan> singleLayerCpuColdReplicatedPlan(size_t num_experts)
    {
        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->continuation_domain = "cpu_cold";
        plan->shared_expert_domain = "cpu_cold";
        plan->residency_policy = ExpertResidencyPolicy::StaticById;
        plan->domains = {
            domainWith("cpu_cold", ExpertDomainKind::NodeLocalTP,
                       {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                       ExpertDomainComputeKind::ReplicatedExperts,
                       CollectiveBackendType::UPI),
        };
        plan->domains[0].world_ranks = {0, 1};
        plan->routed_tiers = {
            tier("cold", "cpu_cold", 0, true),
        };
        plan->placements = {
            ExpertLayerPlacement{.layer = 0, .routed_expert_tier = std::vector<int>(num_experts, 0)},
        };
        return plan;
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
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::rocm(1), 0, 1, Role::UP));
    EXPECT_FALSE(prep.shouldPrepare(DeviceId::rocm(0), 0, 4, Role::UP));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::rocm(1), 0, 4, Role::UP));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cpu(), 0, 2, Role::GATE));
    EXPECT_TRUE(prep.shouldPrepare(DeviceId::cpu(), 0, 5, Role::GATE));
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
    ASSERT_NE(cpu_rank1_stats, nullptr);
    EXPECT_EQ(cpu_rank0_stats->residency_category, WeightResidencyCategory::CpuFallbackExpert);
    EXPECT_EQ(cpu_rank1_stats->residency_category, WeightResidencyCategory::CpuFallbackExpert);
    EXPECT_EQ(cpu_rank0_stats->assigned_routed_experts, cpu_rank1_stats->assigned_routed_experts);
    EXPECT_NE(prep.diagnostics().render().find("memory_by_role"), std::string::npos);
    EXPECT_NE(prep.diagnostics().render().find("routed_tier="), std::string::npos);
    EXPECT_NE(prep.diagnostics().render().find("fallback="), std::string::npos);
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, FiltersRequestsByOverlayRankRoleAndParticipant)
{
    auto plan = std::make_shared<MoEExpertParallelPlan>(threeTierPlan({
        ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 0, 1, 2}},
    }));
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime_plan, 2048);
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

    const auto root_filtered = prep.filteredForRank(*rank0);
    EXPECT_TRUE(root_filtered.hasAcceleratorRequests());
    EXPECT_TRUE(root_filtered.shouldPrepare(DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_TRUE(root_filtered.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::GATE));
    EXPECT_FALSE(root_filtered.shouldPrepare(DeviceId::rocm(1), 0, 1, Role::GATE));
    EXPECT_FALSE(root_filtered.shouldPrepare(DeviceId::rocm(0), 0, 4, Role::GATE));
    EXPECT_TRUE(root_filtered.shouldPrepare(DeviceId::rocm(1), 0, 4, Role::GATE));
    EXPECT_TRUE(root_filtered.shouldPrepare(DeviceId::cpu(), 0, 2, Role::GATE));
    EXPECT_FALSE(root_filtered.shouldPrepare(DeviceId::cpu(), 0, 5, Role::GATE));
    EXPECT_NE(root_filtered.requestForParticipant("cpu_cold", DeviceId::cpu(), 0, 0, 0, 2, Role::GATE), nullptr);
    EXPECT_EQ(root_filtered.requestForParticipant("cpu_cold", DeviceId::cpu(), 1, 1, 0, 2, Role::GATE), nullptr);

    const auto worker_filtered = prep.filteredForRank(*rank1);
    EXPECT_FALSE(worker_filtered.hasAcceleratorRequests());
    EXPECT_TRUE(worker_filtered.hasCpuRoutedAssignments());
    EXPECT_FALSE(worker_filtered.shouldPrepare(DeviceId::cuda(0), 0, 0, Role::GATE));
    EXPECT_FALSE(worker_filtered.shouldPrepare(DeviceId::rocm(0), 0, 1, Role::GATE));
    EXPECT_FALSE(worker_filtered.shouldPrepare(DeviceId::cpu(), 0, 2, Role::GATE));
    EXPECT_TRUE(worker_filtered.shouldPrepare(DeviceId::cpu(), 0, 5, Role::GATE));
    const auto *worker_request = worker_filtered.requestForParticipant("cpu_cold", DeviceId::cpu(), 1, 1, 0, 5, Role::GATE);
    ASSERT_NE(worker_request, nullptr);
    EXPECT_EQ(worker_request->residency_category, WeightResidencyCategory::WorkerFallbackExpert);
    EXPECT_NE(worker_filtered.diagnostics().render().find("worker="), std::string::npos);
    EXPECT_EQ(worker_filtered.diagnostics().render().find("AcceleratorRoutedExpert"), std::string::npos);
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
                   ExpertDomainComputeKind::ReplicatedExperts,
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
                   ExpertDomainComputeKind::ReplicatedExperts,
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

TEST(Test__WeightManagerMoEExpertOverlayPreparation, PreparesCpuFallbackExpertsIntoRegistry)
{
    auto loader = MockModelLoader::createMinimal();
    const size_t d_model = 64;
    const size_t intermediate = 32;
    const size_t num_experts = 4;
    addSingleLayerExpertParents(loader, d_model, intermediate, num_experts);

    WeightManager manager(*loader);
    ASSERT_NE(manager.getWeightForDevice("blk.0.ffn_gate_exps.weight"), nullptr);
    ASSERT_NE(manager.getWeightForDevice("blk.0.ffn_up_exps.weight"), nullptr);
    ASSERT_NE(manager.getWeightForDevice("blk.0.ffn_down_exps.weight"), nullptr);

    auto plan = singleLayerCpuColdReplicatedPlan(num_experts);

    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime_plan));

    std::vector<ITensorGemm *> gate;
    std::vector<ITensorGemm *> up;
    std::vector<ITensorGemm *> down;
    EXPECT_FALSE(manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 0, 0, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const bool owned_by_participant = expert < 2;
        EXPECT_EQ(gate[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(up[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(down[expert] != nullptr, owned_by_participant) << "expert=" << expert;
    }

    EXPECT_FALSE(manager.expertGemmRegistry().populateExpertEnginesForParticipant(
        "cpu_cold", DeviceId::cpu(), 1, 1, 0, static_cast<int>(num_experts), gate, up, down));
    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const bool owned_by_participant = expert >= 2;
        EXPECT_EQ(gate[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(up[expert] != nullptr, owned_by_participant) << "expert=" << expert;
        EXPECT_EQ(down[expert] != nullptr, owned_by_participant) << "expert=" << expert;
    }
}

TEST(Test__WeightManagerMoEExpertOverlayPreparation, HydratesCpuFallbackParentsWhenParentsWereNotPreloaded)
{
    auto loader = MockModelLoader::createMinimal();
    const size_t d_model = 64;
    const size_t intermediate = 32;
    const size_t num_experts = 4;
    addSingleLayerExpertParents(loader, d_model, intermediate, num_experts);

    WeightManager manager(*loader);

    auto plan = singleLayerCpuColdReplicatedPlan(num_experts);
    auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);
    ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime_plan));

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
