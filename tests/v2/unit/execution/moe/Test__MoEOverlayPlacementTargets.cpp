/**
 * @file Test__MoEOverlayPlacementTargets.cpp
 * @brief Device-free replay of the actual whole/cohort migration cycle builders.
 *
 * The traces preserve original projection geometry, ordered physical membership,
 * logical expert identity and movement axes. They prove slot conservation and
 * shadow obligations without claiming native transfer, publication or throughput.
 * Widths beyond this host's two CPU sockets are metadata tests, never fake ranks.
 */
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @return A real whole-expert placement shape, with no fabricated cohort alias. */
MoEExpertOwner whole(int expert, int participant, int tier)
{
    return {.layer_idx = 7, .expert_id = expert, .tier_idx = tier,
        .owner_participant = participant, .device = DeviceId::rocm(participant), .resident = true};
}

/**
 * @brief Construct checked source geometry; no physical socket execution is asserted.
 * @param expert Logical expert coordinate within the original eight-expert source.
 * @param first First canonical participant ID, independent of its NUMA coordinate.
 * @param width Metadata cohort width, two through eight.
 * @param tier Original logical tier coordinate.
 * @param intermediate Original full F extent, independent of local cuts.
 * @param storage Original floating weight precision.
 * @return Complete immutable cohort with original floating projection dimensions.
 */
MoEExpertOwner cohort(int expert, int first, int width, int tier,
    int intermediate = 544, TensorType storage = TensorType::FP16)
{
    const auto format = ExpertWeightFormat::floating(storage);
    MoEOverlayLayerWeightManifest source{.layer_idx = 7, .projections = {{
        {.projection = ExpertTierWeightProjection::Gate, .N = intermediate, .K = 64, .format = format},
        {.projection = ExpertTierWeightProjection::Up, .N = intermediate, .K = 64, .format = format},
        {.projection = ExpertTierWeightProjection::Down, .N = 64, .K = intermediate, .format = format}}}};
    std::vector<MoEExpertShardCohort::Participant> members;
    for (int i = 0; i < width; ++i)
        members.push_back({first + i, first + i, GlobalDeviceAddress::cpu(i)});
    auto checked = MoEExpertShardCohort::checked(source, 8, std::move(members),
        MoEExpertShardCohort::balancedIntervals(source, width));
    return {.layer_idx = 7, .expert_id = expert, .tier_idx = tier,
        .owner_participant = -1, .device = DeviceId::invalid(), .resident = true,
        .shard_cohort = std::make_shared<const MoEExpertShardCohort>(std::move(checked))};
}

/** @return One immutable logical movement whose endpoints retain the same expert. */
MoEOverlayTierMigration edge(MoEExpertOwner source, MoEExpertOwner destination)
{
    destination.expert_id = source.expert_id;
    return {.layer_idx = source.layer_idx, .expert_id = source.expert_id,
        .activation_count = 11, .estimated_weight_bytes = 4096,
        .source = std::move(source), .destination = std::move(destination)};
}
}

TEST(MoEOverlayPlacementTargets, AlternativesPreserveCompleteMembershipAndRejectAliases)
{
    const auto one = MoEExpertPlacementTarget::fromPlacement(whole(0, 3, 0));
    ASSERT_TRUE(one);
    EXPECT_FALSE(one->isCohort());
    EXPECT_EQ(one->participantCount(), 1u);
    EXPECT_EQ(one->participantId(0), 3);
    EXPECT_THROW((void)one->participantId(1), std::out_of_range);
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        auto original = cohort(0, 10, width, 2);
        const auto target = MoEExpertPlacementTarget::fromPlacement(original);
        ASSERT_TRUE(target);
        EXPECT_TRUE(target->isCohort());
        EXPECT_EQ(target->participantCount(), static_cast<size_t>(width));
        for (int member = 0; member < width; ++member)
            EXPECT_EQ(target->participantId(member), 10 + member);
        EXPECT_THROW((void)target->participantId(width), std::out_of_range);
        EXPECT_EQ(*target, *MoEExpertPlacementTarget::fromPlacement(cohort(1, 10, width, 2)));
        EXPECT_NE(*target, *MoEExpertPlacementTarget::fromPlacement(cohort(0, 20, width, 2)));
        original.owner_participant = 10;
        EXPECT_FALSE(MoEExpertPlacementTarget::fromPlacement(original));
        original.owner_participant = -1;
        original.owner_world_rank_known = true;
        EXPECT_FALSE(MoEExpertPlacementTarget::fromPlacement(original));
        original.owner_world_rank_known = false;
        original.layer_idx = 8;
        EXPECT_FALSE(MoEExpertPlacementTarget::fromPlacement(original));
    }
}

TEST(MoEOverlayPlacementCycleReplay, WholeAndCohortMovesReserveEveryMemberOnce)
{
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        const std::vector<MoEOverlayTierMigration> recorded{
            edge(whole(0, 0, 0), cohort(0, 10, width, 2)),
            edge(cohort(1, 10, width, 2), whole(1, 1, 1)),
            edge(whole(2, 1, 1), whole(2, 0, 0))};
        for (int replay = 0; replay < 3; ++replay)
        {
            auto work = recorded;
            const auto evidence = analyzeMoEOverlayMigrationCapacity(work);
            ASSERT_TRUE(evidence.capacityPreserved());
            EXPECT_EQ(evidence.edges_checked, 3u);
            EXPECT_EQ(evidence.tier_coordinates_checked, 3u);
            EXPECT_EQ(evidence.participant_coordinates_checked, static_cast<size_t>(width + 2));
            const auto cycles = buildMoEOverlayMigrationCycles(work);
            ASSERT_EQ(cycles.size(), 1u);
            EXPECT_TRUE(cycles.front().valid(work));
            EXPECT_EQ(cycles.front().migration_indices, (std::vector<size_t>{0, 1, 2}));
            const auto slots = buildMoEOverlayShadowRequirements(work);
            ASSERT_EQ(slots.size(), static_cast<size_t>(width + 2));
            for (const auto &slot : slots)
                EXPECT_EQ(slot.slot_count, 1u);
            for (int member = 0; member < width; ++member)
                EXPECT_NE(std::find(slots.begin(), slots.end(), MoEOverlayTierShadowRequirement{
                    .layer_idx = 7, .tier_idx = 2, .destination_participant = 10 + member, .slot_count = 1}), slots.end());
            // Reset/empty and smaller waves must not retain any prior obligation.
            std::vector<MoEOverlayTierMigration> empty;
            EXPECT_TRUE(buildMoEOverlayMigrationCycles(empty).empty());
            EXPECT_TRUE(buildMoEOverlayShadowRequirements(empty).empty());
            std::vector<MoEOverlayTierMigration> small{edge(whole(3, 0, 0), whole(3, 0, 0))};
            EXPECT_EQ(buildMoEOverlayMigrationCycles(small).size(), 1u);
            EXPECT_EQ(buildMoEOverlayShadowRequirements(small).size(), 1u);
        }
    }
}

TEST(MoEOverlayPlacementCycleReplay, DistinctCohortsAreNotTheSameAbsentScalarOwner)
{
    for (const int width : {2, 4, 8})
    {
        std::vector<MoEOverlayTierMigration> work{
            edge(cohort(0, 10, width, 1), cohort(0, 20, width, 2)),
            edge(cohort(1, 20, width, 2), cohort(1, 10, width, 1))};
        EXPECT_TRUE(analyzeMoEOverlayMigrationCapacity(work).capacityPreserved());
        const auto cycles = buildMoEOverlayMigrationCycles(work);
        ASSERT_EQ(cycles.size(), 1u);
        EXPECT_EQ(cycles.front().migration_indices, (std::vector<size_t>{0, 1}));
        EXPECT_EQ(buildMoEOverlayShadowRequirements(work).size(), static_cast<size_t>(2 * width));
        work.pop_back();
        EXPECT_FALSE(analyzeMoEOverlayMigrationCapacity(work).capacityPreserved());
        EXPECT_FALSE(cycles.front().valid(work));
        EXPECT_THROW((void)buildMoEOverlayMigrationCycles(work), std::logic_error);
    }
}

TEST(MoEOverlayPlacementCycleReplay, MissingFragmentAndForeignSourceReject)
{
    const std::vector<MoEOverlayTierMigration> original{
        edge(whole(0, 0, 0), cohort(0, 10, 4, 1)),
        edge(cohort(1, 10, 4, 1), whole(1, 0, 0))};
    auto partial = original;
    partial[1].source = cohort(1, 10, 3, 1);
    const auto evidence = analyzeMoEOverlayMigrationCapacity(partial);
    EXPECT_FALSE(evidence.capacityPreserved());
    EXPECT_EQ(evidence.participant_flow_violations, 1u);
    EXPECT_EQ(evidence.tier_flow_violations, 0u);
    EXPECT_THROW((void)buildMoEOverlayMigrationCycles(partial), std::logic_error);
    for (int invalid = 0; invalid < 3; ++invalid)
    {
        auto broken = original;
        if (invalid == 0) broken[0].destination.owner_participant = 10;
        if (invalid == 1) broken[0].destination.expert_id = 7;
        if (invalid == 2) broken[0].destination.layer_idx = 8;
        EXPECT_EQ(analyzeMoEOverlayMigrationCapacity(broken).malformed_edges, 1u);
        EXPECT_THROW((void)buildMoEOverlayMigrationCycles(broken), std::logic_error);
        EXPECT_THROW((void)buildMoEOverlayShadowRequirements(broken), std::logic_error);
    }
}

TEST(MoEOverlayPlacementCycleReplay, EqualSlotCountsCannotHideChangedSourceCuts)
{
    auto changed = cohort(1, 10, 4, 1);
    const auto &original = *changed.shard_cohort;
    const auto slices = original.ownership(0).intermediateIntervals();
    std::vector<MoEExpertShardCohort::Interval> intervals(slices.begin(), slices.end());
    ++intervals[0].columns;
    ++intervals[1].first_column;
    --intervals[1].columns;
    const auto members = original.participants();
    changed.shard_cohort = std::make_shared<const MoEExpertShardCohort>(
        MoEExpertShardCohort::checked(original.source(), 8,
            std::vector<MoEExpertShardCohort::Participant>(members.begin(), members.end()),
            std::move(intervals)));
    std::vector<MoEOverlayTierMigration> work{
        edge(whole(0, 0, 0), cohort(0, 10, 4, 1)),
        edge(changed, whole(1, 0, 0))};
    EXPECT_TRUE(analyzeMoEOverlayMigrationCapacity(work).capacityPreserved());
    const MoEOverlayTierMigrationCycle cycle{.layer_idx = 7, .migration_indices = {0, 1}};
    EXPECT_FALSE(cycle.valid(work));
    EXPECT_THROW((void)buildMoEOverlayMigrationCycles(work), std::logic_error);

    // Different fixed admitted cuts in different tiers are valid redistribution,
    // provided each cycle returns to the complete original target at each tier.
    changed.tier_idx = 2;
    work = {edge(cohort(0, 20, 4, 1), changed), edge(changed, cohort(1, 20, 4, 1))};
    EXPECT_TRUE(analyzeMoEOverlayMigrationCapacity(work).capacityPreserved());
    const auto cycles = buildMoEOverlayMigrationCycles(work);
    ASSERT_EQ(cycles.size(), 1u);
    EXPECT_TRUE(cycles.front().valid(work));
}

TEST(MoEOverlayPlacementCycleReplay, ReciprocalForeignMatricesAndDuplicateExpertsReject)
{
    for (const auto &[f, type] : {std::pair{1056, TensorType::FP16}, std::pair{544, TensorType::BF16}})
    {
        std::vector<MoEOverlayTierMigration> work{
            edge(cohort(0, 10, 2, 1), cohort(0, 20, 4, 2, f, type)),
            edge(cohort(1, 20, 4, 2, f, type), cohort(1, 10, 2, 1))};
        EXPECT_EQ(analyzeMoEOverlayMigrationCapacity(work).malformed_edges, 2u);
        EXPECT_THROW((void)buildMoEOverlayMigrationCycles(work), std::logic_error);
        EXPECT_THROW((void)buildMoEOverlayShadowRequirements(work), std::logic_error);
    }
    const auto duplicate = edge(whole(0, 0, 0), whole(0, 0, 0));
    std::vector<MoEOverlayTierMigration> work{duplicate, duplicate};
    EXPECT_EQ(analyzeMoEOverlayMigrationCapacity(work).malformed_edges, 1u);
    const MoEOverlayTierMigrationCycle cycle{.layer_idx = 7, .migration_indices = {0, 1}};
    EXPECT_FALSE(cycle.valid(work));
    EXPECT_THROW((void)buildMoEOverlayMigrationCycles(work), std::logic_error);
    EXPECT_THROW((void)buildMoEOverlayShadowRequirements(work), std::logic_error);
}

TEST(MoEOverlayPlacementTargets, MovedFromCohortRejectsWithoutTermination)
{
    auto row = cohort(0, 10, 2, 1);
    auto original = *row.shard_cohort;
    auto retained = std::move(original);
    row.shard_cohort = std::make_shared<const MoEExpertShardCohort>(std::move(original));
    EXPECT_FALSE(MoEExpertPlacementTarget::fromPlacement(row));
    row.shard_cohort = std::make_shared<const MoEExpertShardCohort>(std::move(retained));
    EXPECT_TRUE(MoEExpertPlacementTarget::fromPlacement(row));
}
}
