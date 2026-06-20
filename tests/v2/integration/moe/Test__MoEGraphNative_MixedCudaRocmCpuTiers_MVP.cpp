/**
 * @file Test__MoEGraphNative_MixedCudaRocmCpuTiers_MVP.cpp
 * @brief Mixed CUDA, ROCm, and CPU routed tiers for graph-native sparse MoE.
 */

#include <gtest/gtest.h>

#include "backends/HardwareInventory.h"
#include "execution/moe/MoEExpertParallelPlanner.h"
#include "integration/moe/MoEGraphNativeRoutedTierTestUtils.h"
#include "mocks/MockComputeStage.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace moe_graph_native_routed_tier;

        constexpr int kDomain = 9302;
        constexpr uint64_t kGeneration = 93;
        constexpr uint64_t kStep = 2;
        constexpr int kParticipantCount = 3;
        constexpr int kFallbackTier = 2;

        std::string mixedHardwareSkipReason()
        {
#if !defined(HAVE_CUDA)
            return "Built without CUDA support (-DHAVE_CUDA=OFF)";
#elif !defined(HAVE_ROCM)
            return "Built without ROCm support (-DHAVE_ROCM=OFF)";
#else
            const auto hardware = HardwareInventory::detect();
            if (hardware.cuda_device_count() < 1)
                return "Mixed routed-tier MVP requires at least one visible CUDA device";
            if (hardware.rocm_device_count() < 1)
                return "Mixed routed-tier MVP requires at least one visible ROCm device";
            return {};
#endif
        }

        ExpertComputeDomain cudaDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "hot_cuda_domain";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "warm_rocm_domain";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cold_cpu_domain";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int capacity,
            bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.max_experts_per_layer = capacity;
            tier.fallback = fallback;
            return tier;
        }

        MoEExpertModelMetadata metadata()
        {
            MoEExpertModelMetadata model;
            model.num_layers = 1;
            model.num_experts = kNumExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        MoEExpertParallelPlan makeMixedPlan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "hot_cuda_domain";
            plan.shared_expert_domain = "hot_cuda_domain";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {cudaDomain(), rocmDomain(), cpuDomain()};
            plan.routed_tiers = {
                routedTier("hot_cuda", "hot_cuda_domain", 0, 1),
                routedTier("warm_rocm", "warm_rocm_domain", 1, 1),
                routedTier("cold_cpu", "cold_cpu_domain", 99, 0, true),
            };
            return MoEExpertParallelPlanner::plan(plan, metadata()).planned_plan;
        }

        void expectOwnerMapHasMixedCanonicalOwners(const MoEExpertParallelPlan &plan,
                                                   const MoEExpertOwnerMap &owner_map)
        {
            bool saw_cuda = false;
            bool saw_rocm = false;
            bool saw_cpu = false;
            std::set<std::pair<int, int>> seen;

            ASSERT_EQ(plan.routed_tiers.size(), 3u);
            ASSERT_TRUE(plan.routed_tiers[kFallbackTier].fallback);
            for (size_t tier = 0; tier < plan.routed_tiers.size(); ++tier)
            {
                if (tier != static_cast<size_t>(kFallbackTier))
                    EXPECT_FALSE(plan.routed_tiers[tier].fallback);
            }

            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                EXPECT_EQ(owner_map.ownerCountForExpert(kLayer, expert), 1u) << "expert=" << expert;
                const auto *owner = owner_map.ownerFor(kLayer, expert);
                ASSERT_NE(owner, nullptr);
                EXPECT_TRUE(seen.insert({owner->layer_idx, owner->expert_id}).second);
                EXPECT_EQ(owner->tier_name, plan.routed_tiers[static_cast<size_t>(owner->tier_idx)].name);
                EXPECT_EQ(owner->domain_name, plan.routed_tiers[static_cast<size_t>(owner->tier_idx)].domain);

                saw_cuda = saw_cuda || owner->device.is_cuda();
                saw_rocm = saw_rocm || owner->device.is_rocm();
                saw_cpu = saw_cpu || owner->device.is_cpu();

                if (plan.routed_tiers[static_cast<size_t>(owner->tier_idx)].fallback)
                {
                    EXPECT_EQ(owner->tier_idx, kFallbackTier);
                    EXPECT_TRUE(owner->device.is_cpu());
                }
                else
                {
                    EXPECT_NE(owner->tier_idx, kFallbackTier);
                    EXPECT_FALSE(owner->device.is_cpu());
                }
            }

            EXPECT_EQ(seen.size(), static_cast<size_t>(kNumExperts));
            EXPECT_TRUE(saw_cuda);
            EXPECT_TRUE(saw_rocm);
            EXPECT_TRUE(saw_cpu);
        }

    } // namespace

    TEST(Test__MoEGraphNative_MixedCudaRocmCpuTiers_MVP,
         SparseDispatchLocalExpertReturnReduceMatchesFullReference)
    {
        const std::string skip_reason = mixedHardwareSkipReason();
        if (!skip_reason.empty())
            GTEST_SKIP() << skip_reason;

        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = kParticipantCount, .slot_count = 16});
        std::vector<ParticipantState> participants(static_cast<size_t>(kParticipantCount));
        for (auto &participant : participants)
            initializeParticipant(&participant, kGeneration, kStep);

        const auto plan = makeMixedPlan();
        EXPECT_EQ(plan.placements.front().routed_expert_tier, (std::vector<int>{0, 1, 2, 2}));
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        plan,
                        {.layer_count = 1, .routed_expert_count = kNumExperts})
                        .ok());

        const auto owner_map = MoEExpertOwnerMap::build(plan);
        expectOwnerMapHasMixedCanonicalOwners(plan, owner_map);

        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        auto reference_output = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runReference(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights, reference_output.get()));

        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan);
        ASSERT_EQ(dispatch.tiers.size(), plan.routed_tiers.size());

        for (int tier = 0; tier < static_cast<int>(plan.routed_tiers.size()); ++tier)
        {
            const auto participant_ids = owner_map.participantIdsForTier(tier);
            ASSERT_EQ(participant_ids.size(), 1u);
            runTierThroughLocalSparseStages(
                cpu_ctx.get(),
                &collective,
                &participants,
                kParticipantCount,
                kDomain,
                kGeneration,
                kStep,
                tier,
                participant_ids.front(),
                dispatch,
                owner_map,
                hidden.get(),
                routing_indices.get(),
                routing_weights.get(),
                weights);
        }

        expectTensorNear(participants[kRootParticipant].combined_output.get(), reference_output.get());
    }

} // namespace llaminar2::test