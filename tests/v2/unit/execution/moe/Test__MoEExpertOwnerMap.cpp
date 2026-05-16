#include "execution/moe/MoEExpertOwnerMap.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        ExpertComputeDomain rocmWarmDomain(ExpertDomainComputeKind compute_kind)
        {
            ExpertComputeDomain domain;
            domain.name = "rocm_warm";
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0),
                                   GlobalDeviceAddress::rocm(1, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = compute_kind;
            return domain;
        }

        ExpertComputeDomain cpuColdDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {2};
            domain.owner_rank = 2;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        MoEExpertParallelPlan disjointRocmPlan(ExpertDomainComputeKind compute_kind)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "rocm_warm";
            plan.shared_expert_domain = "rocm_warm";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {rocmWarmDomain(compute_kind), cpuColdDomain()};
            plan.routed_tiers = {
                tier("warm", "rocm_warm", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            plan.placements = {
                ExpertLayerPlacement{.layer = 0,
                                     .routed_expert_tier = {0, 0, 0, 0, 0, 0}},
            };
            return plan;
        }

        bool allFalseOverlap(const std::vector<bool> &lhs, const std::vector<bool> &rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (lhs[i] && rhs[i])
                    return false;
            }
            return true;
        }

    } // namespace

    TEST(Test__MoEExpertOwnerMap, DisjointAcceleratorParticipantsOwnWholeExperts)
    {
        const auto plan = disjointRocmPlan(ExpertDomainComputeKind::ReplicatedExperts);
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        ASSERT_EQ(owner_map.participants().size(), 3u);
        const auto warm_participants = owner_map.participantIdsForTier(0);
        ASSERT_EQ(warm_participants.size(), 2u);

        for (int expert = 0; expert < 6; ++expert)
        {
            EXPECT_EQ(owner_map.ownerCountForExpert(0, expert), 1u) << "expert=" << expert;
            const auto *owner = owner_map.ownerFor(0, expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_EQ(owner->tier_idx, 0);
            EXPECT_TRUE(owner->resident);
            EXPECT_TRUE(owner->device.is_rocm());
            EXPECT_TRUE(std::find(warm_participants.begin(), warm_participants.end(), owner->owner_participant) != warm_participants.end());

            const auto *participant = owner_map.participantForId(owner->owner_participant);
            ASSERT_NE(participant, nullptr);
            EXPECT_EQ(owner->address, participant->address);
            EXPECT_EQ(owner->domain_participant_index, participant->domain_participant_index);
            EXPECT_EQ(owner->device, participant->device);
        }

        const auto first_mask = owner_map.expertMaskForParticipant(0, warm_participants[0], 6);
        const auto second_mask = owner_map.expertMaskForParticipant(0, warm_participants[1], 6);
        ASSERT_EQ(first_mask.size(), 6u);
        ASSERT_EQ(second_mask.size(), 6u);
        EXPECT_TRUE(allFalseOverlap(first_mask, second_mask));

        for (size_t expert = 0; expert < first_mask.size(); ++expert)
            EXPECT_TRUE(first_mask[expert] || second_mask[expert]) << "expert=" << expert;
    }

    TEST(Test__MoEExpertOwnerMap, RejectsTensorParallelExpertsForGraphNativeRoutedTiers)
    {
        const auto plan = disjointRocmPlan(ExpertDomainComputeKind::TensorParallelExperts);
        EXPECT_THROW((void)MoEExpertOwnerMap::build(plan), std::invalid_argument);
    }

} // namespace llaminar2::test