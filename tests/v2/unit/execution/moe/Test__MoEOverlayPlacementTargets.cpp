/**
 * @file Test__MoEOverlayPlacementTargets.cpp
 * @brief Device-free replay of the actual whole/cohort migration cycle builders.
 *
 * The traces preserve original projection geometry, ordered physical membership,
 * logical expert identity and movement axes. They prove slot conservation and
 * shadow obligations without claiming native transfer, publication or throughput.
 * Widths beyond this host's two CPU sockets are metadata tests, never fake ranks.
 * Ordered-adjacency equivalence retains the original global-scan DFS as a test oracle
 * for exact edge choices, active-path backtracking and malformed diagnostics.
 */
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
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

/**
 * @brief Replay the original checked endpoint resolver, including foreign cuts.
 * @param migration Original metadata edge, with no physical allocation.
 * @return Complete typed endpoints or no value for an invalid source contract.
 */
std::optional<std::pair<MoEExpertPlacementTarget, MoEExpertPlacementTarget>>
replayTargets(const MoEOverlayTierMigration &migration)
{
    if (migration.layer_idx < 0 || migration.expert_id < 0 ||
        migration.source.layer_idx != migration.layer_idx ||
        migration.destination.layer_idx != migration.layer_idx ||
        migration.source.expert_id != migration.expert_id ||
        migration.destination.expert_id != migration.expert_id)
        return std::nullopt;
    auto source = MoEExpertPlacementTarget::fromPlacement(migration.source);
    auto destination = MoEExpertPlacementTarget::fromPlacement(migration.destination);
    if (!source || !destination) return std::nullopt;
    if (migration.source.shard_cohort && migration.destination.shard_cohort &&
        (migration.source.shard_cohort->source() != migration.destination.shard_cohort->source() ||
         migration.source.shard_cohort->ownership(0).geometry() != migration.destination.shard_cohort->ownership(0).geometry()))
        return std::nullopt;
    return std::pair{std::move(*source), std::move(*destination)};
}

/**
 * @brief Retain the pre-optimization cycle validator as an independent oracle.
 * @param cycle Authored candidate indices, including malformed cases.
 * @param migrations Complete original edge vector.
 * @return Original validity, including duplicate-index and expert checks.
 */
bool originalCycleValid(const MoEOverlayTierMigrationCycle &cycle,
    const std::vector<MoEOverlayTierMigration> &migrations)
{
    if (cycle.layer_idx < 0 || cycle.migration_indices.empty())
        return false;
    std::vector<bool> seen(migrations.size(), false);
    std::set<std::pair<int, int>> logical_edges;
    for (size_t offset = 0; offset < cycle.migration_indices.size(); ++offset)
    {
        const size_t index = cycle.migration_indices[offset];
        const size_t next_index = cycle.migration_indices[(offset + 1) % cycle.migration_indices.size()];
        if (index >= migrations.size() || next_index >= migrations.size() || seen[index])
            return false;
        seen[index] = true;
        const auto &edge = migrations[index];
        const auto &next = migrations[next_index];
        if (!logical_edges.emplace(edge.layer_idx, edge.expert_id).second)
            return false;
        const auto targets = replayTargets(edge);
        const auto next_targets = replayTargets(next);
        if (edge.layer_idx != cycle.layer_idx || next.layer_idx != cycle.layer_idx ||
            !targets || !next_targets || targets->second != next_targets->first)
            return false;
    }
    return true;
}

/**
 * @brief Retain the original global-scan DFS, not a second production mode.
 * @param migrations Authored complete metadata population, in original order.
 * @return Original ordered cycles; original exception text is retained.
 */
std::vector<MoEOverlayTierMigrationCycle> originalGlobalScanCycles(
    const std::vector<MoEOverlayTierMigration> &migrations)
{
    if (!analyzeMoEOverlayMigrationCapacity(migrations).capacityPreserved())
        throw std::logic_error("MoE migration requires complete capacity-preserving placement targets");
    std::vector<std::pair<MoEExpertPlacementTarget, MoEExpertPlacementTarget>> targets;
    targets.reserve(migrations.size());
    for (const auto &edge : migrations)
        targets.push_back(*replayTargets(edge));
    std::vector<MoEOverlayTierMigrationCycle> cycles;
    std::vector<bool> consumed(migrations.size(), false);
    for (size_t seed_index = 0; seed_index < migrations.size(); ++seed_index)
    {
        if (consumed[seed_index])
            continue;
        const auto &seed = migrations[seed_index];
        consumed[seed_index] = true;
        MoEOverlayTierMigrationCycle cycle{.layer_idx = seed.layer_idx,
            .migration_indices = {seed_index}};
        if (targets[seed_index].first != targets[seed_index].second)
        {
            std::vector<size_t> return_path;
            std::vector<MoEExpertPlacementTarget> visited;
            // A cohort is one vertex, never one vertex per fragment or the
            // absent scalar owner -1. Closing that vertex preserves every
            // admitted member's capacity without splitting a logical move.
            const auto find_return_path = [&](auto &&self,
                const MoEExpertPlacementTarget &current) -> bool
            {
                if (current == targets[seed_index].first)
                    return true;
                if (std::find(visited.begin(), visited.end(), current) != visited.end())
                    return false;
                visited.push_back(current);
                for (size_t index = 0; index < migrations.size(); ++index)
                {
                    if (consumed[index] || migrations[index].layer_idx != seed.layer_idx ||
                        targets[index].first != current)
                        continue;
                    consumed[index] = true;
                    return_path.push_back(index);
                    if (self(self, targets[index].second))
                        return true;
                    return_path.pop_back();
                    consumed[index] = false;
                }
                visited.pop_back();
                return false;
            };
            if (!find_return_path(find_return_path, targets[seed_index].second))
                throw std::logic_error("MoE placement delta cannot form complete whole/cohort cycles");
            cycle.migration_indices.insert(cycle.migration_indices.end(),
                return_path.begin(), return_path.end());
        }
        if (!originalCycleValid(cycle, migrations))
            throw std::logic_error("MoE overlay produced an invalid placement migration cycle");
        cycles.push_back(std::move(cycle));
    }
    return cycles;
}

/**
 * @brief Compare exact cycle order and immutable inputs through the public builder.
 * @param recorded Complete authored edge population; never a model route claim.
 */
void expectOriginalCycleOrder(const std::vector<MoEOverlayTierMigration> &recorded)
{
    const auto expected = originalGlobalScanCycles(recorded);
    auto actual_input = recorded;
    const auto actual = buildMoEOverlayMigrationCycles(actual_input);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index)
    {
        EXPECT_EQ(actual[index].layer_idx, expected[index].layer_idx);
        EXPECT_EQ(actual[index].migration_indices, expected[index].migration_indices);
        EXPECT_EQ(actual[index].valid(recorded), originalCycleValid(actual[index], recorded));
    }
    ASSERT_EQ(actual_input.size(), recorded.size());
    for (size_t index = 0; index < recorded.size(); ++index)
    {
        EXPECT_EQ(actual_input[index].source, recorded[index].source);
        EXPECT_EQ(actual_input[index].destination, recorded[index].destination);
        EXPECT_EQ(actual_input[index].axis, recorded[index].axis);
    }
}

/**
 * @brief Compare the precise original logic-error diagnostic, not just rejection.
 * @param recorded Incomplete or malformed original metadata edge population.
 */
void expectOriginalCycleFailure(const std::vector<MoEOverlayTierMigration> &recorded)
{
    std::string expected;
    try { (void)originalGlobalScanCycles(recorded); }
    catch (const std::logic_error &error) { expected = error.what(); }
    ASSERT_FALSE(expected.empty());
    try
    {
        (void)decomposeMoEOverlayMigrationCycles(recorded);
        FAIL() << "Invalid trace unexpectedly decomposed";
    }
    catch (const std::logic_error &error) { EXPECT_EQ(error.what(), expected); }
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

TEST(MoEOverlayOrderedCycleEquivalence, OrderedAdjacencyMatchesOriginalAcrossLiveCountsAndInputOrders)
{
    std::mt19937 random(42);
    for (const int targets : {1, 2, 3, 4, 5, 6, 7, 8, 17, 65})
    {
        SCOPED_TRACE(targets);
        std::vector<MoEOverlayTierMigration> recorded;
        for (const int layer : {103, 7, 4096})
        {
            int expert = 0;
            const auto add = [&](int from, int to)
            {
                auto source = whole(expert, 100 + from, from % 5);
                auto destination = whole(expert, 100 + to, to % 5);
                source.layer_idx = destination.layer_idx = layer;
                recorded.push_back(edge(std::move(source), std::move(destination)));
                ++expert;
            };
            // Long forward/reverse rings, skewed repeated hub exchanges and
            // self loops exercise arbitrary target counts without fake devices.
            for (int target = 0; target < targets; ++target)
                add(target, (target + 1) % targets);
            for (int target = targets - 1; target >= 0; --target)
                add(target, (target + targets - 1) % targets);
            for (int repeat = 0; repeat < 9; ++repeat)
            {
                const int peer = targets == 1 ? 0 : 1 + repeat % (targets - 1);
                add(0, peer);
                add(peer, 0);
            }
            add(0, 0);
        }
        for (int ordering = 0; ordering < 4; ++ordering)
        {
            if (ordering != 0) std::shuffle(recorded.begin(), recorded.end(), random);
            for (int replay = 0; replay < 3; ++replay)
            {
                expectOriginalCycleOrder(recorded);
                expectOriginalCycleOrder({});
                expectOriginalCycleOrder({edge(whole(0, 0, 0), whole(0, 0, 0))});
            }
        }
    }
}

TEST(MoEOverlayOrderedCycleEquivalence, OrderedAdjacencyBacktracksAndExpiresActiveMarks)
{
    // B->C->B is encountered before B->A. It must fail the current path,
    // restore both consumed edges, then close A->B->A in original edge order.
    std::vector<MoEOverlayTierMigration> recorded{
        edge(whole(0, 0, 0), whole(0, 1, 1)),
        edge(whole(1, 1, 1), whole(1, 2, 2)),
        edge(whole(2, 2, 2), whole(2, 1, 1)),
        edge(whole(3, 1, 1), whole(3, 0, 0))};
    for (int replay = 0; replay < 3; ++replay)
    {
        expectOriginalCycleOrder(recorded);
        const auto actual = decomposeMoEOverlayMigrationCycles(recorded);
        ASSERT_EQ(actual.size(), 2u);
        EXPECT_EQ(actual[0].migration_indices, (std::vector<size_t>{0, 3}));
        EXPECT_EQ(actual[1].migration_indices, (std::vector<size_t>{1, 2}));
    }
    for (const auto &indices : std::vector<std::vector<size_t>>{
            {}, {0}, {0, 0}, {0, 3}, {3, 0}, {0, 1, 2, 3}, {0, 4},
            {0, std::numeric_limits<size_t>::max()}})
    {
        const MoEOverlayTierMigrationCycle cycle{.layer_idx = 7, .migration_indices = indices};
        EXPECT_EQ(cycle.valid(recorded), originalCycleValid(cycle, recorded));
    }
}

TEST(MoEOverlayOrderedCycleEquivalence, OrderedAdjacencyKeepsExactCohortCutsAndOriginalDiagnostics)
{
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        auto original = cohort(0, 10, width, 1);
        auto changed = original;
        const auto &source = *original.shard_cohort;
        const auto cuts = source.ownership(0).intermediateIntervals();
        std::vector<MoEExpertShardCohort::Interval> intervals(cuts.begin(), cuts.end());
        ++intervals[0].columns;
        ++intervals[1].first_column;
        --intervals[1].columns;
        const auto members = source.participants();
        changed.shard_cohort = std::make_shared<const MoEExpertShardCohort>(
            MoEExpertShardCohort::checked(source.source(), 8,
                std::vector<MoEExpertShardCohort::Participant>(members.begin(), members.end()), std::move(intervals)));
        const auto at = [](MoEExpertOwner owner, int expert) { owner.expert_id = expert; return owner; };
        std::vector<MoEOverlayTierMigration> recorded{
            edge(whole(0, 0, 0), at(original, 0)),
            edge(at(changed, 1), whole(1, 0, 0)),
            edge(at(original, 2), whole(2, 0, 0)),
            edge(whole(3, 0, 0), at(changed, 3))};
        for (int replay = 0; replay < 3; ++replay)
            expectOriginalCycleOrder(recorded);
        const auto actual = decomposeMoEOverlayMigrationCycles(recorded);
        ASSERT_EQ(actual.size(), 2u);
        EXPECT_EQ(actual[0].migration_indices, (std::vector<size_t>{0, 2}));
        EXPECT_EQ(actual[1].migration_indices, (std::vector<size_t>{1, 3}));
        // Member/slot counts balance, but different source cuts cannot close
        // one another. They deliberately collide in the scalar bucket key.
        expectOriginalCycleFailure({recorded[0], recorded[1]});
        expectOriginalCycleFailure({recorded[0]});
        auto malformed = recorded;
        malformed[0].destination.expert_id = 7;
        expectOriginalCycleFailure(malformed);
        malformed = recorded;
        malformed[1].expert_id = malformed[0].expert_id;
        malformed[1].source.expert_id = malformed[1].destination.expert_id = malformed[0].expert_id;
        expectOriginalCycleFailure(malformed);
    }
}

}
