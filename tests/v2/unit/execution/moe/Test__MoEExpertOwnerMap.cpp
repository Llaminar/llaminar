/**
 * @file Test__MoEExpertOwnerMap.cpp
 * @brief Regression tests for routed-expert ownership policy and owner maps.
 *
 * These tests prove that static whole-expert assignment is deterministic,
 * balanced, disjoint, complete, and represented identically as sorted packed
 * expert IDs and graph-facing masks. They are deliberately device-free so the
 * physical-weight and graph-policy contract remains part of the fast unit gate.
 * Pipeline-stage cases retain model-global layer IDs in compact ownership,
 * prove exact mask extents, and reject incomplete or cross-stage epoch maps.
 */

#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/RoutedExpertOwnerAssignment.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        RoutedExpertDomain rocmWarmDomain(RoutedExpertComputePolicy routed_compute_policy)
        {
            RoutedExpertDomain domain;
            domain.name = "rocm_warm";
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0),
                                   GlobalDeviceAddress::rocm(1, 0)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = routed_compute_policy;
            return domain;
        }

        RoutedExpertDomain cpuColdDomain()
        {
            RoutedExpertDomain domain;
            domain.name = "cpu_cold";
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {2};
            domain.owner_rank = 2;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        MoERoutedExpertPlacementPlan disjointRocmPlan(RoutedExpertComputePolicy routed_compute_policy)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "rocm_warm";
            plan.shared_expert_domain = "rocm_warm";
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan.domains = {rocmWarmDomain(routed_compute_policy), cpuColdDomain()};
            plan.routed_tiers = {
                tier("warm", "rocm_warm", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            plan.placements = {
                RoutedExpertLayerPlacement{.layer = 0,
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
        const auto plan = disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
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

    /**
     * @brief Canonical lookup must not mistake an adjacent owner for a miss.
     *
     * Production dispatch performs this query for every routed activation.
     * The logarithmic implementation relies on canonical owner order, so the
     * boundary and cross-layer misses below lock down its exact equality check.
     */
    TEST(Test__MoEExpertOwnerMap,
         CanonicalCoordinateLookupIsTotalAndRejectsAdjacentMisses)
    {
        auto plan =
            disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
        plan.placements.push_back(RoutedExpertLayerPlacement{
            .layer = 1,
            .routed_expert_tier = {0, 0, 0, 0, 0, 0},
        });
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        for (int layer = 0; layer < 2; ++layer)
        {
            for (int expert = 0; expert < 6; ++expert)
            {
                const auto *const owner = owner_map.ownerFor(layer, expert);
                ASSERT_NE(owner, nullptr)
                    << "layer=" << layer << " expert=" << expert;
                EXPECT_EQ(owner->layer_idx, layer);
                EXPECT_EQ(owner->expert_id, expert);
                EXPECT_EQ(owner_map.ownerCountForExpert(layer, expert), 1u);
            }
        }

        for (const auto [layer, expert] : {
                 std::pair{-1, 0},
                 std::pair{0, -1},
                 std::pair{0, 6},
                 std::pair{1, 6},
                 std::pair{2, 0}})
        {
            EXPECT_EQ(owner_map.ownerFor(layer, expert), nullptr)
                << "layer=" << layer << " expert=" << expert;
            EXPECT_EQ(owner_map.ownerCountForExpert(layer, expert), 0u);
        }
    }

    /**
     * @brief Explicit epoch ownership preserves a same-tier skew correction.
     *
     * The tier vector is unchanged: only the exact ROCm participant owners are
     * swapped. Rebuilding from cold-start order would silently discard this
     * decision, so the explicit factory must retain it exactly.
     */
    TEST(Test__MoEExpertOwnerMap,
         ExplicitOwnershipPreservesSameTierParticipantSwap)
    {
        const auto plan =
            disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
        const auto initial = MoEExpertOwnerMap::build(plan);
        auto ownership = initial.layeredOwnership(1, 6);
        const int first_owner = ownership.owner(0, 0);
        const int fourth_owner = ownership.owner(0, 3);
        ASSERT_NE(first_owner, fourth_owner);

        ownership.assignOwner(0, 0, fourth_owner);
        ownership.assignOwner(0, 3, first_owner);
        const auto candidate =
            MoEExpertOwnerMap::buildExplicit(plan, ownership);

        EXPECT_EQ(candidate.ownerFor(0, 0)->owner_participant, fourth_owner);
        EXPECT_EQ(candidate.ownerFor(0, 3)->owner_participant, first_owner);
        for (int expert = 0; expert < 6; ++expert)
        {
            EXPECT_EQ(candidate.ownerFor(0, expert)->tier_idx, 0);
            EXPECT_EQ(candidate.ownerCountForExpert(0, expert), 1u);
        }
    }

    /** @brief An explicit owner cannot escape the expert's selected tier. */
    TEST(Test__MoEExpertOwnerMap,
         ExplicitOwnershipRejectsParticipantFromAnotherTier)
    {
        const auto plan =
            disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
        const auto initial = MoEExpertOwnerMap::build(plan);
        auto ownership = initial.layeredOwnership(1, 6);
        const auto cold_participants = initial.participantIdsForTier(1);
        ASSERT_EQ(cold_participants.size(), 1u);
        ownership.assignOwner(0, 0, cold_participants.front());

        EXPECT_THROW(
            (void)MoEExpertOwnerMap::buildExplicit(plan, ownership),
            std::invalid_argument);
    }

    /**
     * @brief Stage storage scales with owned layers while all public IDs remain global.
     *
     * CPU, CUDA and ROCm use the same metadata-only ownership authority. Vary
     * the origin, retained tail and owner order to catch accidental zero-origin
     * indexing without initializing any backend or allocating model storage.
     */
    TEST(Test__MoEExpertOwnerMap, PipelineStageOwnershipUsesCompactGlobalLayerRows)
    {
        for (const auto backend : {CollectiveBackendType::MPI,
                                  CollectiveBackendType::NCCL,
                                  CollectiveBackendType::RCCL})
        for (const auto order : {RoutedExpertOwnerOrder::Ordinal,
                                RoutedExpertOwnerOrder::Random})
        for (const int first : {0, 20, 40})
        for (const int count : {1, 2, 21})
        {
            SCOPED_TRACE(::testing::Message() << "first=" << first << " count=" << count);
            auto plan = disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
            plan.owner_order = order;
            auto &domain = plan.domains.front();
            domain.backend = backend;
            if (backend == CollectiveBackendType::MPI)
                domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            else if (backend == CollectiveBackendType::NCCL)
                domain.participants = {GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(1, 0)};
            const auto first_placement = plan.placements.front();
            plan.placements.clear();
            for (int layer = first; layer < first + count; ++layer)
            {
                auto placement = first_placement;
                placement.layer = layer;
                plan.placements.push_back(std::move(placement));
            }
            const auto owner_map = MoEExpertOwnerMap::build(plan);
            const auto ownership = owner_map.layeredOwnership(count, 6, first);
            EXPECT_EQ(ownership.firstModelLayer(), first);
            EXPECT_EQ(ownership.endModelLayer(), first + count);
            EXPECT_EQ(ownership.layerCount(), count);
            EXPECT_EQ(owner_map.owners().size(), static_cast<size_t>(count * 6));
            const auto explicit_map = MoEExpertOwnerMap::buildExplicit(plan, ownership);
            for (int participant = 0; participant < ownership.participantCount(); ++participant)
            {
                const auto masks = ownership.masksForParticipant(participant);
                ASSERT_EQ(masks.size(), static_cast<size_t>(count));
                for (int row = 0; row < count; ++row)
                {
                    const int layer = ownership.modelLayerForStorageIndex(row);
                    EXPECT_EQ(layer, first + row);
                    EXPECT_EQ(ownership.storageIndexForModelLayer(layer), static_cast<size_t>(row));
                    ASSERT_EQ(masks[row].size(), 6u);
                    for (int expert = 0; expert < 6; ++expert)
                    {
                        const auto *owner = explicit_map.ownerFor(layer, expert);
                        ASSERT_NE(owner, nullptr);
                        EXPECT_EQ(owner->owner_participant, ownership.owner(layer, expert));
                        EXPECT_EQ(masks[row][expert], owner->owner_participant == participant);
                    }
                }
            }
            EXPECT_THROW((void)ownership.owner(first - 1, 0), std::out_of_range);
            EXPECT_THROW((void)ownership.owner(first + count, 0), std::out_of_range);
            EXPECT_THROW((void)ownership.modelLayerForStorageIndex(count), std::out_of_range);
        }
    }

    /** @brief A stage-local swap publishes global IDs and preserves exact capacities. */
    TEST(Test__MoEExpertOwnerMap, PipelineStageOwnershipChangesKeepGlobalIdentities)
    {
        const auto previous = MoELayeredExpertOwnership::uniform(2, 2, {0, 0, 1, 1}, 20);
        auto candidate = previous;
        candidate.assignOwner(21, 0, 1);
        candidate.assignOwner(21, 2, 0);
        EXPECT_EQ(candidate.changesFrom(previous),
                  (std::vector<MoELayeredExpertOwnershipChange>{
                      {21, 0, 0, 1}, {21, 2, 1, 0}}));
        EXPECT_TRUE(candidate.hasSameLayerCapacitiesAs(previous));
        EXPECT_EQ(candidate.ownersForLayer(20), previous.ownersForLayer(20));
        EXPECT_THROW(candidate.assignOwner(0, 0, 1), std::out_of_range);
        candidate.assignOwner(20, 0, 1);
        EXPECT_FALSE(candidate.hasSameLayerCapacitiesAs(previous));
    }

    /** @brief Equal dimensions do not make distinct pipeline stages interchangeable. */
    TEST(Test__MoEExpertOwnerMap, PipelineStageOwnershipRejectsCrossStageEpochComparison)
    {
        const auto head = MoELayeredExpertOwnership::uniform(20, 2, {0, 1}, 0);
        const auto tail = MoELayeredExpertOwnership::uniform(20, 2, {0, 1}, 20);
        EXPECT_NE(head, tail);
        EXPECT_THROW((void)tail.changesFrom(head), std::invalid_argument);
        EXPECT_THROW((void)tail.hasSameLayerCapacitiesAs(head), std::invalid_argument);
        EXPECT_THROW((void)head.storageIndexForModelLayer(20), std::out_of_range);
        EXPECT_THROW((void)tail.storageIndexForModelLayer(0), std::out_of_range);
    }

    /** @brief Missing stage rows and foreign rows must fail before ownership publication. */
    TEST(Test__MoEExpertOwnerMap, PipelineStageOwnershipRequiresExactLayerCoverage)
    {
        auto plan = disjointRocmPlan(RoutedExpertComputePolicy::Apportioned);
        const auto row = plan.placements.front();
        plan.placements.clear();
        for (int layer = 20; layer < 23; ++layer)
        {
            auto placement = row;
            placement.layer = layer;
            plan.placements.push_back(std::move(placement));
        }
        const auto complete = MoEExpertOwnerMap::build(plan);
        for (size_t missing = 0; missing < plan.placements.size(); ++missing)
        {
            auto incomplete = plan;
            incomplete.placements.erase(incomplete.placements.begin() + missing);
            const auto owner_map = MoEExpertOwnerMap::build(incomplete);
            EXPECT_THROW((void)owner_map.layeredOwnership(3, 6, 20), std::logic_error);
            EXPECT_THROW((void)MoEExpertOwnerMap::buildExplicit(
                incomplete, complete.layeredOwnership(3, 6, 20)), std::invalid_argument);
        }
        EXPECT_THROW((void)complete.layeredOwnership(2, 6, 20), std::logic_error);
        EXPECT_THROW((void)complete.layeredOwnership(3, 6, 19), std::logic_error);
        EXPECT_THROW((void)complete.layeredOwnership(3, 6, 21), std::logic_error);
        const auto other_stage = MoELayeredExpertOwnership::uniform(3, 3, {0, 0, 0, 1, 1, 1}, 0);
        EXPECT_THROW((void)MoEExpertOwnerMap::buildExplicit(plan, other_stage), std::invalid_argument);
    }

    /** @brief Layer interval validation precedes large allocations and integer arithmetic. */
    TEST(Test__MoEExpertOwnerMap, PipelineStageOwnershipRejectsInvalidLayerIntervals)
    {
        constexpr int max = std::numeric_limits<int>::max();
        EXPECT_THROW((MoELayeredExpertOwnership{2, {{0, 1}}, -1}), std::invalid_argument);
        EXPECT_THROW((MoELayeredExpertOwnership{2, {{0, 1}}, max}), std::invalid_argument);
        EXPECT_THROW((void)MoELayeredExpertOwnership::uniform(max, 2, {0, 1}, 1), std::invalid_argument);
        const auto edge = MoELayeredExpertOwnership::uniform(2, 2, {0, 1}, max - 2);
        EXPECT_EQ(edge.endModelLayer(), max);
        EXPECT_EQ(edge.modelLayerForStorageIndex(1), max - 1);
        EXPECT_EQ(edge.owner(max - 1, 1), 1);
        const auto owner_map = MoEExpertOwnerMap::build(
            disjointRocmPlan(RoutedExpertComputePolicy::Apportioned));
        EXPECT_THROW((void)owner_map.layeredOwnership(1, 6, -1), std::invalid_argument);
        EXPECT_THROW((void)owner_map.layeredOwnership(max, 6, 1), std::invalid_argument);
    }

    TEST(Test__MoEExpertOwnerMap, RejectsTensorShardedComputeForWholeExpertOwnerMap)
    {
        const auto plan = disjointRocmPlan(RoutedExpertComputePolicy::TensorSharded);
        EXPECT_THROW((void)MoEExpertOwnerMap::build(plan), std::invalid_argument);
    }

    TEST(
        Test__MoEExpertOwnerMap,
        TransitionPreservesRetainedParticipantsAndRejectsTopologyChanges)
    {
        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            SCOPED_TRACE(routedExpertOwnerOrderToString(owner_order));
            auto previous_plan = disjointRocmPlan(
                RoutedExpertComputePolicy::Apportioned);
            previous_plan.owner_order = owner_order;
            previous_plan.placements.front().routed_expert_tier = {
                0, 0, 0, 0, 1, 1};
            const auto previous = MoEExpertOwnerMap::build(previous_plan);

            auto candidate_plan = previous_plan;
            candidate_plan.placements.front().routed_expert_tier = {
                0, 1, 0, 1, 0, 0};
            const auto candidate = MoEExpertOwnerMap::buildTransition(
                candidate_plan,
                previous);

            /* e0 and e2 stay in the two-participant accelerator tier. */
            for (const int retained_expert : {0, 2})
            {
                ASSERT_NE(previous.ownerFor(0, retained_expert), nullptr);
                ASSERT_NE(candidate.ownerFor(0, retained_expert), nullptr);
                EXPECT_EQ(
                    candidate.ownerFor(0, retained_expert)
                        ->owner_participant,
                    previous.ownerFor(0, retained_expert)
                        ->owner_participant);
            }

            const auto physical_moves = std::count_if(
                candidate.owners().begin(),
                candidate.owners().end(),
                [&](const auto &owner)
                {
                    const auto *old = previous.ownerFor(
                        owner.layer_idx,
                        owner.expert_id);
                    return old && old->owner_participant !=
                                      owner.owner_participant;
                });
            EXPECT_EQ(physical_moves, 4)
                << "Only the four experts changing logical tiers may move";

            auto changed_topology = candidate_plan;
            std::swap(
                changed_topology.domains.front().participants[0],
                changed_topology.domains.front().participants[1]);
            EXPECT_THROW(
                (void)MoEExpertOwnerMap::buildTransition(
                    changed_topology,
                    previous),
                std::invalid_argument);
        }
    }

    TEST(Test__MoEExpertOwnerMap, OrdinalAssignmentPreservesBalancedSourceOrder)
    {
        using routed_expert_ownership::expertIdsForParticipant;

        EXPECT_EQ(
            expertIdsForParticipant(10, 3, 0, 7, RoutedExpertOwnerOrder::Ordinal),
            (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(
            expertIdsForParticipant(10, 3, 1, 7, RoutedExpertOwnerOrder::Ordinal),
            (std::vector<int>{4, 5, 6}));
        EXPECT_EQ(
            expertIdsForParticipant(10, 3, 2, 7, RoutedExpertOwnerOrder::Ordinal),
            (std::vector<int>{7, 8, 9}));
    }

    TEST(Test__MoEExpertOwnerMap,
         ContiguousSelectionClassificationIsSharedAcrossPublicationPaths)
    {
        using routed_expert_ownership::expertIdsFormContiguousSpan;

        EXPECT_TRUE(expertIdsFormContiguousSpan({}));
        EXPECT_TRUE(expertIdsFormContiguousSpan({7}));
        EXPECT_TRUE(expertIdsFormContiguousSpan({4, 5, 6, 7}));
        EXPECT_FALSE(expertIdsFormContiguousSpan({4, 5, 7}));
        EXPECT_FALSE(expertIdsFormContiguousSpan({4, 6, 7}));
    }

    TEST(Test__MoEExpertOwnerMap, RandomAssignmentIsDeterministicBalancedDisjointAndMaskEquivalent)
    {
        using routed_expert_ownership::expertIdsForParticipant;
        using routed_expert_ownership::expertMaskForParticipant;

        constexpr int kNumExperts = 23;
        for (int participant_count = 2; participant_count <= 8; ++participant_count)
        {
            std::vector<int> owner_count(static_cast<size_t>(kNumExperts), 0);
            size_t smallest_assignment = static_cast<size_t>(kNumExperts);
            size_t largest_assignment = 0u;

            for (int participant_index = 0;
                 participant_index < participant_count;
                 ++participant_index)
            {
                const std::vector<int> first = expertIdsForParticipant(
                    kNumExperts,
                    participant_count,
                    participant_index,
                    11,
                    RoutedExpertOwnerOrder::Random);
                const std::vector<int> second = expertIdsForParticipant(
                    kNumExperts,
                    participant_count,
                    participant_index,
                    11,
                    RoutedExpertOwnerOrder::Random);
                const std::vector<bool> mask = expertMaskForParticipant(
                    kNumExperts,
                    participant_count,
                    participant_index,
                    11,
                    RoutedExpertOwnerOrder::Random);

                EXPECT_EQ(first, second);
                EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
                ASSERT_EQ(mask.size(), static_cast<size_t>(kNumExperts));
                EXPECT_EQ(
                    static_cast<size_t>(std::count(mask.begin(), mask.end(), true)),
                    first.size());

                smallest_assignment = std::min(smallest_assignment, first.size());
                largest_assignment = std::max(largest_assignment, first.size());
                for (const int expert_id : first)
                {
                    ASSERT_GE(expert_id, 0);
                    ASSERT_LT(expert_id, kNumExperts);
                    EXPECT_TRUE(mask[static_cast<size_t>(expert_id)]);
                    ++owner_count[static_cast<size_t>(expert_id)];
                }
            }

            EXPECT_LE(largest_assignment - smallest_assignment, 1u);
            for (int expert_id = 0; expert_id < kNumExperts; ++expert_id)
            {
                EXPECT_EQ(owner_count[static_cast<size_t>(expert_id)], 1)
                    << "participant_count=" << participant_count
                    << " expert_id=" << expert_id;
            }
        }

        const auto layer_zero = expertIdsForParticipant(
            kNumExperts, 4, 0, 0, RoutedExpertOwnerOrder::Random);
        const auto layer_one = expertIdsForParticipant(
            kNumExperts, 4, 0, 1, RoutedExpertOwnerOrder::Random);
        const auto ordinal = expertIdsForParticipant(
            kNumExperts, 4, 0, 0, RoutedExpertOwnerOrder::Ordinal);
        EXPECT_NE(layer_zero, layer_one);
        EXPECT_NE(layer_zero, ordinal);
    }

    TEST(Test__MoEExpertOwnerMap, DenseOwnerMapMatchesEveryIdListAndMask)
    {
        using routed_expert_ownership::expertIdsForParticipant;
        using routed_expert_ownership::expertMaskForParticipant;
        using routed_expert_ownership::ownerParticipantByExpert;

        constexpr int kNumExperts = 23;
        constexpr int kParticipantCount = 4;
        constexpr int kLayer = 11;
        for (const RoutedExpertOwnerOrder order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            const std::vector<int> owners = ownerParticipantByExpert(
                kNumExperts, kParticipantCount, kLayer, order);
            ASSERT_EQ(owners.size(), static_cast<size_t>(kNumExperts));

            for (int participant = 0;
                 participant < kParticipantCount;
                 ++participant)
            {
                const std::vector<int> ids = expertIdsForParticipant(
                    kNumExperts,
                    kParticipantCount,
                    participant,
                    kLayer,
                    order);
                const std::vector<bool> mask = expertMaskForParticipant(
                    kNumExperts,
                    kParticipantCount,
                    participant,
                    kLayer,
                    order);
                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    const bool expected = owners[static_cast<size_t>(expert)] == participant;
                    EXPECT_EQ(mask[static_cast<size_t>(expert)], expected)
                        << "participant=" << participant << " expert=" << expert;
                    EXPECT_EQ(
                        std::binary_search(ids.begin(), ids.end(), expert),
                        expected)
                        << "participant=" << participant << " expert=" << expert;
                }
            }
        }
    }

    TEST(Test__MoEExpertOwnerMap, AssignmentRejectsIncompleteOwnershipGeometry)
    {
        using routed_expert_ownership::expertIdsForParticipant;

        EXPECT_THROW(
            (void)expertIdsForParticipant(
                3, 4, 0, 0, RoutedExpertOwnerOrder::Random),
            std::invalid_argument);
        EXPECT_THROW(
            (void)expertIdsForParticipant(
                4, 2, 2, 0, RoutedExpertOwnerOrder::Random),
            std::invalid_argument);
        EXPECT_THROW(
            (void)expertIdsForParticipant(
                4, 2, 0, -1, RoutedExpertOwnerOrder::Random),
            std::invalid_argument);
    }

} // namespace llaminar2::test
