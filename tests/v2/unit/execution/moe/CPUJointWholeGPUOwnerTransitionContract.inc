/**
 * @file CPUJointWholeGPUOwnerTransitionContract.inc
 * @brief Keep fixed CPU fragment targets while committing complete whole-owner peer cycles.
 *
 * Original tiny GGUF metadata owns source precision and CPU cohort cuts. CUDA
 * and ROCm endpoints below are inert declarations; these model-free contracts
 * never allocate native storage or claim physical migration readiness.
 */

TEST(CPUJointWholeGPUOwnerTransitionContract, TierAndPeerChangesRetainSourceCohortsAndExactSlots)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    // Immutable source metadata is shared across the topology product. Each
    // case still constructs its own actual canonical cohort/cuts and live map.
    JointPlacementOriginal source(2, 2, 11, 2, 3);
    std::vector<MoEOverlayLayerWeightManifest> manifest;
    for (int layer = 3; layer < 5; ++layer)
        manifest.push_back(source.preparation->cpuJointOwnerMap()->placementFor(layer, 8)->shard_cohort->source());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int members : {2, 4, 8})
    for (const int peers : {2, 4, 8})
    {
        SCOPED_TRACE(::testing::Message() << int(backend) << '/' << members << '/' << peers);
        auto plan = source.plan;
        auto &gpu = plan.domains.front();
        gpu.participants.clear();
        for (int peer = 0; peer < peers; ++peer)
            gpu.participants.push_back(GlobalDeviceAddress::fromLocalDeviceId({backend, peer}));
        gpu.backend = backend == DeviceType::CUDA
            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        auto &cpu = plan.domains.back();
        cpu.participants.clear();
        cpu.world_ranks.clear();
        for (int member = 0; member < members; ++member)
        {
            cpu.participants.push_back(GlobalDeviceAddress::cpu(member));
            cpu.world_ranks.push_back(member);
        }
        const auto original = MoEExpertOwnerMap::buildShardPlacement(plan, manifest);
        const auto initial = original.owners();
        auto tiers = plan;
        for (auto &placement : tiers.placements)
            std::swap(placement.routed_expert_tier[0], placement.routed_expert_tier[8]);
        const auto base = MoEExpertOwnerMap::buildShardTransition(tiers, original, original);
        EXPECT_EQ(MoEExpertOwnerMap::buildShardTransition(tiers, original, original,
                      std::span<const MoELayeredExpertOwnershipChange>{}).owners(), base.owners());
        std::vector<MoELayeredExpertOwnershipChange> changes;
        for (int layer = 3; layer < 5; ++layer)
        {
            const auto incoming_owner = base.placementFor(layer, 8)->owner_participant;
            for (int owner = 0; owner < peers; ++owner)
                changes.push_back({layer, owner == incoming_owner ? 8 :
                    base.expertsForParticipant(layer, owner).front(), owner, (owner + 1) % peers});
        }
        const auto attempts = MoEExpertOwnerMap::initialBuildAttempts();
        const auto candidate = MoEExpertOwnerMap::buildShardTransition(tiers, original, original, changes);
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            const auto replay = MoEExpertOwnerMap::buildShardTransition(tiers, original, original, changes);
            EXPECT_EQ(replay.owners(), candidate.owners());
        }
        auto reversed = changes;
        std::reverse(reversed.begin(), reversed.end());
        EXPECT_EQ(MoEExpertOwnerMap::buildShardTransition(tiers, original, original, reversed).owners(),
                  candidate.owners());
        EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts(), attempts);
        EXPECT_EQ(original.owners(), initial);
        for (const auto &participant : original.participants())
            for (int layer = 3; layer < 5; ++layer)
                EXPECT_EQ(candidate.expertsForParticipant(layer, participant.participant_id).size(),
                          original.expertsForParticipant(layer, participant.participant_id).size());
        for (const auto &row : candidate.owners())
        {
            if (row.tier_idx == 1)
            {
                const auto cohort = original.placementFor(row.layer_idx, 8)->shard_cohort;
                ASSERT_TRUE(row.shard_cohort);
                EXPECT_EQ(row.shard_cohort.get(), cohort.get());
                EXPECT_FALSE(row.shard_cohort.owner_before(cohort));
                EXPECT_FALSE(cohort.owner_before(row.shard_cohort));
                EXPECT_EQ(row.owner_participant, -1);
                EXPECT_FALSE(row.device.is_valid());
            }
        }
        const auto retained = MoEExpertOwnerMap::buildShardTransition(tiers, candidate, original);
        EXPECT_EQ(retained.owners(), candidate.owners());
    }
}

TEST(CPUJointWholeGPUOwnerTransitionContract, RejectsStaleForeignCohortAndUnclosedParticipantChanges)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal source(2, 2, 7, 2, 3);
    const auto &plan = source.plan;
    const auto &original = *source.preparation->cpuJointOwnerMap();
    const int first = original.expertsForParticipant(3, 0).front();
    const int second = original.expertsForParticipant(3, 1).front();
    const std::vector<MoELayeredExpertOwnershipChange> valid{{3, first, 0, 1}, {3, second, 1, 0}};
    ASSERT_NO_THROW((void)MoEExpertOwnerMap::buildShardTransition(plan, original, original, valid));
    for (int failure = 0; failure < 9; ++failure)
    {
        SCOPED_TRACE(failure);
        auto bad = valid;
        switch (failure)
        {
        case 0: bad.pop_back(); break; // An open edge changes physical slot capacity.
        case 1: bad.push_back(bad.front()); break;
        case 2: bad.front().previous_participant = 1; break;
        case 3: bad.front().current_participant = 9999; break;
        case 4: bad.front().current_participant = original.participants().back().participant_id; break;
        case 5: bad.front().expert_id = 6; break; // A CPU cohort is never a whole GPU source.
        case 6: bad.front().layer_idx = 2; break;
        case 7: bad.back().layer_idx = 4; break; // Layers cannot exchange each other's slots.
        case 8: bad.front().current_participant = bad.front().previous_participant; break;
        }
        EXPECT_THROW((void)MoEExpertOwnerMap::buildShardTransition(plan, original, original, bad),
                     std::invalid_argument);
    }
    EXPECT_EQ(original.placementFor(3, first)->owner_participant, 0);
    EXPECT_EQ(original.placementFor(3, second)->owner_participant, 1);
}

/** @test Whole CPU and GPU peers retain independent capacities beside a fixed CPU cohort. */
TEST(CPUJointWholeGPUOwnerTransitionContract, MixedWholeCPUAndGPUPeersRetainCohortsAndReplay)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal source(2, 2, 15, 2, 3);
    std::vector<MoEOverlayLayerWeightManifest> manifests;
    for (int layer = 3; layer < 5; ++layer)
        manifests.push_back(source.preparation->cpuJointOwnerMap()
            ->placementFor(layer, 12)->shard_cohort->source());

    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(int(backend));
        auto plan = source.plan;
        plan.domains.front().backend = backend == DeviceType::CUDA
            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        plan.domains.front().participants = {
            GlobalDeviceAddress::fromLocalDeviceId({backend, 0}),
            GlobalDeviceAddress::fromLocalDeviceId({backend, 1})};
        // These are inert declared endpoints, not native device observations.
        // The existing cohort remains tensor-sharded; the extra domain owns
        // complete experts and therefore enters the ordinary peer interface.
        auto whole_cpu = plan.domains.back();
        whole_cpu.name = "whole_cpu";
        whole_cpu.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        whole_cpu.participants = {GlobalDeviceAddress::cpu(2), GlobalDeviceAddress::cpu(3)};
        whole_cpu.world_ranks = {2, 3};
        plan.domains.push_back(std::move(whole_cpu));
        plan.routed_tiers.front().max_experts_per_layer = 8;
        auto cohort_tier = plan.routed_tiers.back();
        cohort_tier.priority = 2;
        plan.routed_tiers = {plan.routed_tiers.front(),
            {.name = "whole_cpu", .domain = "whole_cpu", .priority = 1, .max_experts_per_layer = 4}, cohort_tier};
        for (auto &placement : plan.placements)
        {
            for (auto &tier : placement.routed_expert_tier)
                if (tier == 1) tier = 2;
            for (int expert = 8; expert < 12; ++expert)
                placement.routed_expert_tier[expert] = 1;
        }
        const auto original = MoEExpertOwnerMap::buildShardPlacement(plan, manifests);
        std::vector<MoELayeredExpertOwnershipChange> changes;
        for (int layer = 3; layer < 5; ++layer)
            for (const int tier : {0, 1})
            {
                const auto participants = original.participantIdsForTier(tier);
                ASSERT_EQ(participants.size(), 2u);
                for (std::size_t index = 0; index < participants.size(); ++index)
                {
                    const int participant = participants[index];
                    const auto experts = original.expertsForParticipant(layer, participant);
                    ASSERT_FALSE(experts.empty());
                    changes.push_back({layer, experts.front(), participant,
                        participants[(index + 1) % participants.size()]});
                }
            }
        const auto candidate = MoEExpertOwnerMap::buildShardTransition(plan, original, original, changes);
        for (const auto &change : changes)
            EXPECT_EQ(candidate.placementFor(change.layer_idx, change.expert_id)->owner_participant,
                      change.current_participant);
        for (const auto &participant : original.participants())
            for (int layer = 3; layer < 5; ++layer)
                EXPECT_EQ(candidate.expertsForParticipant(layer, participant.participant_id).size(),
                          original.expertsForParticipant(layer, participant.participant_id).size());
        for (int layer = 3; layer < 5; ++layer)
            for (int expert = 12; expert < 15; ++expert)
            {
                const auto &before = original.placementFor(layer, expert)->shard_cohort;
                const auto &after = candidate.placementFor(layer, expert)->shard_cohort;
                ASSERT_TRUE(after);
                EXPECT_EQ(after.get(), before.get());
                EXPECT_FALSE(after.owner_before(before));
                EXPECT_FALSE(before.owner_before(after));
            }
        EXPECT_EQ(MoEExpertOwnerMap::buildShardTransition(plan, candidate, original).owners(),
                  candidate.owners());
        std::reverse(changes.begin(), changes.end());
        EXPECT_EQ(MoEExpertOwnerMap::buildShardTransition(plan, original, original, changes).owners(),
                  candidate.owners());
        for (auto &change : changes)
            std::swap(change.previous_participant, change.current_participant);
        EXPECT_EQ(MoEExpertOwnerMap::buildShardTransition(plan, candidate, original, changes).owners(),
                  original.owners());

        auto open = changes;
        open.pop_back();
        EXPECT_THROW((void)MoEExpertOwnerMap::buildShardTransition(plan, candidate, original, open),
                     std::invalid_argument);
        const auto cohort_members = original.participantIdsForTier(2);
        ASSERT_FALSE(cohort_members.empty());
        const std::vector<MoELayeredExpertOwnershipChange> member_spoof{
            {3, 12, cohort_members.front(), original.placementFor(3, 8)->owner_participant}};
        EXPECT_THROW((void)MoEExpertOwnerMap::buildShardTransition(plan, original, original, member_spoof),
                     std::invalid_argument);
    }
}
