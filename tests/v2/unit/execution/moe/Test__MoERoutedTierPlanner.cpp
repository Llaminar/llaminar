#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"

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

        RoutedExpertDomain cudaDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain rocmDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(0, 1)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain cpuDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.max_experts_per_layer = max_experts_per_layer;
            result.fallback = fallback;
            return result;
        }

        MoERoutedExpertModelMetadata metadata()
        {
            MoERoutedExpertModelMetadata model;
            model.num_layers = kLayers;
            model.num_experts = kExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        MoERoutedExpertPlacementPlan baseGpuOnlyPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "zeta_cuda_domain";
            plan.shared_expert_domain = "zeta_cuda_domain";
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
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

        const MoERoutedExpertTierMemoryEstimate *memoryTier(
            const MoERoutedExpertPlacementMemoryEstimate &memory,
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
        const auto gpu_only_result = MoERoutedExpertPlacementPlanner::plan(gpu_only, metadata());

        ASSERT_EQ(gpu_only_result.planned_plan.placements.size(), 1u);
        EXPECT_EQ(gpu_only_result.planned_plan.placements.front().routed_expert_tier,
                  (std::vector<int>{1, 1, 0, 0}));
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(
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
            (void)MoERoutedExpertPlacementPlanner::plan(insufficient, metadata());
            FAIL() << "Expected insufficient no-fallback capacity to throw";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("no-fallback tier capacity cannot cover every expert"),
                      std::string::npos);
        }

        auto with_fallback = insufficient;
        with_fallback.domains.push_back(cpuDomain("omega_cpu_domain"));
        with_fallback.routed_tiers.push_back(tier("omega_user_label", "omega_cpu_domain", 20, 0, true));
        const auto fallback_result = MoERoutedExpertPlacementPlanner::plan(with_fallback, metadata());
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
        tensor_parallel_routed.domains[1].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        EXPECT_THROW((void)MoEExpertOwnerMap::build(tensor_parallel_routed), std::invalid_argument);

        auto explicit_over_capacity = gpu_only_result.planned_plan;
        explicit_over_capacity.routed_tiers[1].max_experts_per_layer = 1;
        const auto validation = validateMoERoutedExpertPlacementPlan(
            explicit_over_capacity,
            {.layer_count = kLayers, .routed_expert_count = kExperts});
        EXPECT_FALSE(validation.ok());
        EXPECT_TRUE(std::any_of(validation.errors.begin(), validation.errors.end(), [](const std::string &error)
                                { return error.find("max_experts_per_layer") != std::string::npos; }));
    }

} // namespace llaminar2::test
