#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {

        constexpr int kLayers = 1;
        constexpr int kExperts = 4;
        constexpr int kDModel = 8;
        constexpr int kIntermediate = 4;

        ExpertComputeDomain cudaDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(0, 1)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.max_experts_per_layer = max_experts_per_layer;
            result.fallback = fallback;
            return result;
        }

        MoEExpertModelMetadata metadata()
        {
            MoEExpertModelMetadata model;
            model.num_layers = kLayers;
            model.num_experts = kExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        MoEExpertParallelPlan baseGpuOnlyPlan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "zeta_cuda_domain";
            plan.shared_expert_domain = "zeta_cuda_domain";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {
                cudaDomain("zeta_cuda_domain"),
                rocmDomain("alpha_rocm_domain"),
            };
            plan.routed_tiers = {
                tier("zebra_user_label", "zeta_cuda_domain", 10, 2),
                tier("alpha_user_label", "alpha_rocm_domain", 0, 2),
            };
            return plan;
        }

        const MoEExpertTierMemoryEstimate *memoryTier(
            const MoEExpertParallelMemoryEstimate &memory,
            int tier_index)
        {
            auto found = std::find_if(memory.tiers.begin(), memory.tiers.end(), [&](const auto &entry)
                                      { return entry.tier_index == tier_index; });
            return found == memory.tiers.end() ? nullptr : &(*found);
        }

        void expectExactlyOneOwnerPerExpert(const MoEExpertOwnerMap &owner_map)
        {
            for (int expert = 0; expert < kExperts; ++expert)
            {
                EXPECT_EQ(owner_map.ownerCountForExpert(0, expert), 1u) << "expert=" << expert;
                EXPECT_NE(owner_map.ownerFor(0, expert), nullptr) << "expert=" << expert;
            }
        }

    } // namespace

    TEST(Test__MoERoutedTierPlanner, FlexibleNamesAndBudgets)
    {
        auto gpu_only = baseGpuOnlyPlan();
        const auto gpu_only_result = MoEExpertParallelPlanner::plan(gpu_only, metadata());

        ASSERT_EQ(gpu_only_result.planned_plan.placements.size(), 1u);
        EXPECT_EQ(gpu_only_result.planned_plan.placements.front().routed_expert_tier,
                  (std::vector<int>{1, 1, 0, 0}));
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        gpu_only_result.planned_plan,
                        {.layer_count = kLayers, .routed_expert_count = kExperts})
                        .ok());

        ASSERT_EQ(gpu_only_result.planned_plan.routed_tiers.size(), 2u);
        EXPECT_EQ(gpu_only_result.planned_plan.routed_tiers[0].name, "zebra_user_label");
        EXPECT_EQ(gpu_only_result.planned_plan.routed_tiers[1].name, "alpha_user_label");
        EXPECT_FALSE(gpu_only_result.planned_plan.routed_tiers[0].fallback);
        EXPECT_FALSE(gpu_only_result.planned_plan.routed_tiers[1].fallback);

        const auto *cuda_memory = memoryTier(gpu_only_result.memory, 0);
        const auto *rocm_memory = memoryTier(gpu_only_result.memory, 1);
        ASSERT_NE(cuda_memory, nullptr);
        ASSERT_NE(rocm_memory, nullptr);
        EXPECT_EQ(cuda_memory->tier_name, "zebra_user_label");
        EXPECT_EQ(rocm_memory->tier_name, "alpha_user_label");
        EXPECT_EQ(cuda_memory->routed_expert_count, 2u);
        EXPECT_EQ(rocm_memory->routed_expert_count, 2u);

        const auto gpu_only_owner_map = MoEExpertOwnerMap::build(gpu_only_result.planned_plan);
        expectExactlyOneOwnerPerExpert(gpu_only_owner_map);
        for (int expert = 0; expert < kExperts; ++expert)
        {
            const auto *owner = gpu_only_owner_map.ownerFor(0, expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_FALSE(owner->device.is_cpu());
            if (expert < 2)
            {
                EXPECT_EQ(owner->tier_idx, 1);
                EXPECT_EQ(owner->tier_name, "alpha_user_label");
                EXPECT_EQ(owner->domain_name, "alpha_rocm_domain");
                EXPECT_TRUE(owner->device.is_rocm());
            }
            else
            {
                EXPECT_EQ(owner->tier_idx, 0);
                EXPECT_EQ(owner->tier_name, "zebra_user_label");
                EXPECT_EQ(owner->domain_name, "zeta_cuda_domain");
                EXPECT_TRUE(owner->device.is_cuda());
            }
        }

        auto insufficient = baseGpuOnlyPlan();
        insufficient.routed_tiers[0].max_experts_per_layer = 1;
        insufficient.routed_tiers[1].max_experts_per_layer = 1;
        try
        {
            (void)MoEExpertParallelPlanner::plan(insufficient, metadata());
            FAIL() << "Expected insufficient no-fallback capacity to throw";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("no-fallback routed tier capacity cannot cover every expert"),
                      std::string::npos);
        }

        auto with_fallback = insufficient;
        with_fallback.domains.push_back(cpuDomain("omega_cpu_domain"));
        with_fallback.routed_tiers.push_back(tier("omega_user_label", "omega_cpu_domain", 20, 0, true));
        const auto fallback_result = MoEExpertParallelPlanner::plan(with_fallback, metadata());
        ASSERT_EQ(fallback_result.planned_plan.placements.size(), 1u);
        EXPECT_EQ(fallback_result.planned_plan.placements.front().routed_expert_tier,
                  (std::vector<int>{1, 0, 2, 2}));
        EXPECT_TRUE(fallback_result.planned_plan.routed_tiers[2].fallback);

        const auto fallback_owner_map = MoEExpertOwnerMap::build(fallback_result.planned_plan);
        expectExactlyOneOwnerPerExpert(fallback_owner_map);
        const auto *fallback_owner = fallback_owner_map.ownerFor(0, 2);
        ASSERT_NE(fallback_owner, nullptr);
        EXPECT_EQ(fallback_owner->tier_name, "omega_user_label");
        EXPECT_EQ(fallback_owner->domain_name, "omega_cpu_domain");
        EXPECT_TRUE(fallback_owner->device.is_cpu());

        auto tensor_parallel_routed = gpu_only_result.planned_plan;
        tensor_parallel_routed.domains[1].compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
        EXPECT_THROW((void)MoEExpertOwnerMap::build(tensor_parallel_routed), std::invalid_argument);

        auto explicit_over_capacity = gpu_only_result.planned_plan;
        explicit_over_capacity.routed_tiers[1].max_experts_per_layer = 1;
        const auto validation = validateMoEExpertParallelPlan(
            explicit_over_capacity,
            {.layer_count = kLayers, .routed_expert_count = kExperts});
        EXPECT_FALSE(validation.ok());
        EXPECT_TRUE(std::any_of(validation.errors.begin(), validation.errors.end(), [](const std::string &error)
                                { return error.find("max_experts_per_layer") != std::string::npos; }));
    }

} // namespace llaminar2::test