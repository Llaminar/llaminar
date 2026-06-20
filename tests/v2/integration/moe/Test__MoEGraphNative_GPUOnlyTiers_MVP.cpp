/**
 * @file Test__MoEGraphNative_GPUOnlyTiers_MVP.cpp
 * @brief GPU-only routed tiers without CPU fallback for graph-native sparse MoE.
 */

#include <gtest/gtest.h>

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

        constexpr int kDomain = 9301;
        constexpr uint64_t kGeneration = 93;
        constexpr uint64_t kStep = 1;
        constexpr int kParticipantCount = 2;

        ExpertComputeDomain cudaTierDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "orchid_cuda_domain";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmTierDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "saffron_rocm_domain";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, int priority, int capacity)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.max_experts_per_layer = capacity;
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

        MoEExpertParallelPlan makeGpuOnlyPlan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "orchid_cuda_domain";
            plan.shared_expert_domain = "orchid_cuda_domain";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {cudaTierDomain(), rocmTierDomain()};
            plan.routed_tiers = {
                routedTier("orchid_user_tier", "orchid_cuda_domain", 1, 2),
                routedTier("saffron_user_tier", "saffron_rocm_domain", 0, 2),
            };
            return MoEExpertParallelPlanner::plan(plan, metadata()).planned_plan;
        }

        void expectOwnerMapHasGpuOnlyCanonicalOwners(const MoEExpertParallelPlan &plan,
                                                     const MoEExpertOwnerMap &owner_map)
        {
            ASSERT_EQ(plan.routed_tiers.size(), 2u);
            for (const auto &tier : plan.routed_tiers)
                EXPECT_FALSE(tier.fallback);

            std::set<std::pair<int, int>> seen;
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                EXPECT_EQ(owner_map.ownerCountForExpert(kLayer, expert), 1u) << "expert=" << expert;
                const auto *owner = owner_map.ownerFor(kLayer, expert);
                ASSERT_NE(owner, nullptr);
                EXPECT_TRUE(seen.insert({owner->layer_idx, owner->expert_id}).second);
                EXPECT_FALSE(owner->device.is_cpu());
                EXPECT_TRUE(owner->device.is_cuda() || owner->device.is_rocm());
                EXPECT_EQ(owner->tier_name, plan.routed_tiers[static_cast<size_t>(owner->tier_idx)].name);
                EXPECT_EQ(owner->domain_name, plan.routed_tiers[static_cast<size_t>(owner->tier_idx)].domain);
            }
            EXPECT_EQ(seen.size(), static_cast<size_t>(kNumExperts));
        }

    } // namespace

    TEST(Test__MoEGraphNative_GPUOnlyTiers_MVP, SparseDispatchLocalExpertReturnReduceMatchesFullReference)
    {
        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = kParticipantCount, .slot_count = 16});
        std::vector<ParticipantState> participants(static_cast<size_t>(kParticipantCount));
        for (auto &participant : participants)
            initializeParticipant(&participant, kGeneration, kStep);

        const auto plan = makeGpuOnlyPlan();
        EXPECT_EQ(plan.placements.front().routed_expert_tier, (std::vector<int>{1, 1, 0, 0}));
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        plan,
                        {.layer_count = 1, .routed_expert_count = kNumExperts})
                        .ok());

        const auto owner_map = MoEExpertOwnerMap::build(plan);
        expectOwnerMapHasGpuOnlyCanonicalOwners(plan, owner_map);

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