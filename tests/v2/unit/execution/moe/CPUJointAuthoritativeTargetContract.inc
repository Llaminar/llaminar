/**
 * @file CPUJointAuthoritativeTargetContract.inc
 * @brief Dense original-source proposal reconstruction and exact typed-target wire semantics.
 *
 * These metadata contracts use the actual authority and tiny original GGUF directory.
 * They grant neither physical readiness nor dynamic maintenance. Cohort widths beyond
 * local sockets are declarations, never claims of additional native participants.
 */
TEST(CPUJointAuthoritativeTargetContract, CompleteSourceTargetsRoundTripWithoutScalarAliases)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        JointPlacementOriginal original(width);
        MoEOverlayResidencyAuthority authority(original.config());
        auto placements = original.plan.placements;
        std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
        auto tx = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(authority, placements);
        tx.histogram_window = frozenWindow(17, {7, 11, 3, 5});
        tx.histogram_generation = tx.histogram_window->generation;
        for (auto &edge : tx.migrations)
            edge.activation_count = tx.histogram_window->activationCount(edge.layer_idx, edge.expert_id);
        ASSERT_TRUE(tx.valid());
        const auto plan = authority.exportAuthoritativeResidencyPlan(tx);
        ASSERT_TRUE(plan.valid());
        EXPECT_TRUE(plan.entries[0].candidate_target.isCPUCohort());
        EXPECT_TRUE(plan.entries[1].candidate_target.isWhole());
        EXPECT_TRUE(plan.entries[2].candidate_target.isCPUCohort()); // unchanged also authenticated
        const auto proposal = makeMoEOverlayDistributedResidencyProposal(plan, tx);
        ASSERT_TRUE(proposal.valid());
        std::vector<uint8_t> wire(moeOverlayDistributedResidencyProposalWireBytes(proposal));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedResidencyProposal(proposal, wire, &error)) << error;
        EXPECT_EQ(wire.size(), moeOverlayDistributedResidencyProposalWireBytes(1, 4));
        EXPECT_EQ(wire.size(), size_t(96) + moeOverlayDistributedHistogramWireBytes(1, 4) +
                                  size_t(4) * (4 + 4 + 4 + 8 + 8));
        MoEOverlayDistributedResidencyProposal decoded;
        ASSERT_TRUE(decodeMoEOverlayDistributedResidencyProposal(wire, 1, 4, &decoded, &error)) << error;
        EXPECT_EQ(decoded.plan.entries, plan.entries);
        const auto adopted = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, decoded.plan);
        ASSERT_TRUE(adopted.valid());
        EXPECT_EQ(fingerprintMoEOverlayResidencyExecutionPlan(adopted), proposal.execution_fingerprint);
        EXPECT_EQ(authority.snapshot(), tx.previous); // reconstruction never publishes
        EXPECT_THROW(authority.adoptAuthoritativeResidencyPlan(plan), std::logic_error);
        // ABI2 interpreted this same-sized word as an unconditional whole participant.
        auto old = wire;
        old[4] = 2; old[5] = old[6] = old[7] = 0;
        EXPECT_FALSE(decodeMoEOverlayDistributedResidencyProposal(old, 1, 4, &decoded, &error));
        for (size_t index : {size_t(0), size_t(2)})
            for (const auto bad_target : {MoEOverlayRouteTarget(), MoEOverlayRouteTarget::whole(1),
                                          MoEOverlayRouteTarget::cpuCohort(7)})
            {
                auto bad = plan;
                bad.entries[index].candidate_target = bad_target;
                EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, bad),
                             std::exception);
            }
        auto partial = plan;
        partial.entries.pop_back();
        EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, partial),
                     std::invalid_argument);
        // Equal-sized target words do not erase a receiver's original cut/member
        // definition: its independent execution fingerprint must disagree.
        JointPlacementOriginal different_cuts(width == 8 ? 2 : width + 1);
        MoEOverlayResidencyAuthority cut_receiver(different_cuts.config());
        const auto cut_tx = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(cut_receiver, plan);
        EXPECT_NE(fingerprintMoEOverlayResidencyExecutionPlan(cut_tx), proposal.execution_fingerprint);
        JointPlacementOriginal different_members(width);
        for (auto &endpoint : different_members.plan.domains[1].participants)
            endpoint.numa_node += 16;
        const auto changed_source = std::make_shared<const MoERoutedExpertPlacementPlan>(different_members.plan);
        const auto changed_runtime = resolveMoEExpertOverlayRuntimePlan(changed_source,
            {.current_world_rank = 0, .validate_mvp_root_reachability = false});
        different_members.preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
            MoEExpertOverlayPreparationPlan::build(*changed_runtime, different_members.loader));
        MoEOverlayResidencyAuthority member_receiver(different_members.config());
        const auto member_tx = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(member_receiver, plan);
        EXPECT_NE(fingerprintMoEOverlayResidencyExecutionPlan(member_tx), proposal.execution_fingerprint);
        MoEOverlayResidencyAuthority foreign(original.config());
        EXPECT_THROW(foreign.exportAuthoritativeResidencyPlan(tx), std::logic_error);
    }
}

#include "execution/moe/MoEOverlayPreparedTargetService.h"
#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEOverlayEconomyEvidenceCodec.h"
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"
#include "execution/moe/MoEOverlayPreparedTransactionCost.h"

/** @brief Replay original root samples and the production objective without granting Off maintenance. */
TEST(CPUJointAuthoritativeTargetContract, PreparedTargetsReplaySourceCodecObjectiveAndOffOwnership)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    using namespace llaminar2::moe_overlay_economy;
    JointPlacementOriginal original(2);
    MoEOverlayResidencyAuthority authority(original.config());
    const auto snapshot = authority.snapshot();
    const auto cohort = snapshot->originalOwnerMap().placementFor(0, 1)->shard_cohort;
    const auto catalog = std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(
        std::vector<MoEOverlayLayerWeightManifest>{cohort->source()});
    const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
    const auto plan = MoEOverlayPreparedTargetServicePlan::checked(snapshot, catalog, topology);
    ASSERT_EQ(plan->entries().size(), 2u);
    std::vector<MoEOverlayPreparedTargetServiceTotals> samples;
    for (const auto &entry : plan->entries())
    {
        MoEOverlayPreparedTargetServiceTotals row{.coordinate = entry.coordinate};
        for (size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
            if (entry.sample_layers[phase] >= 0)
            {
                row.sample_count[phase] = entry.measured_invocations;
                row.activation_count[phase] = entry.sample_rows[phase] * entry.measured_invocations;
                row.total_nanoseconds[phase] = row.activation_count[phase] *
                    (entry.coordinate.target.isWhole() ? 10 : 30);
                row.expert_ids[phase] = plan->sampleExpert(entry, phase);
                const auto *resident = snapshot->placementOwnerMap().placementFor(
                    entry.sample_layers[phase], row.expert_ids[phase]);
                ASSERT_NE(resident, nullptr);
                EXPECT_EQ(bool(resident->shard_cohort), entry.coordinate.target.isCPUCohort());
            }
        samples.push_back(row);
    }
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        std::vector<MoEOverlayPreparedTargetServiceTotals> decoded;
        for (int rank : {0, 1})
        {
            std::vector<MoEOverlayPreparedTargetServiceTotals> local;
            for (const auto &row : samples)
                if (plan->entry(row.coordinate).root_world_rank == rank) local.push_back(row);
            const auto bytes = MoEOverlayEconomyEvidenceCodec::encodePreparedService(*plan, local, rank);
            auto restored = MoEOverlayEconomyEvidenceCodec::decodePreparedService(*plan, bytes, rank);
            decoded.insert(decoded.end(), restored.begin(), restored.end());
            if (!bytes.empty())
            {
                auto truncated = bytes; truncated.pop_back();
                EXPECT_THROW(MoEOverlayEconomyEvidenceCodec::decodePreparedService(*plan, truncated, rank), std::invalid_argument);
                auto foreign = local; foreign.front().expert_ids[0] =
                    foreign.front().coordinate.target.isWhole() ? 1 : 0;
                EXPECT_THROW(MoEOverlayEconomyEvidenceCodec::encodePreparedService(*plan, foreign, rank), std::invalid_argument);
            }
        }
        const auto profile = MoEOverlayEconomyProfileComposer::composePreparedService(plan, decoded);
        ASSERT_EQ(profile->target_costs.size(), 2u);
        EXPECT_THROW(profile->participantCosts(), std::invalid_argument);
        // Resolve complete route alternatives into the same compact vertices used
        // by production. A cohort receives ONE full-output price, independent of width.
        std::array<uint64_t, 2> prices{};
        std::array<int32_t, 4> before{}, after{};
        for (size_t vertex = 0; vertex < profile->target_costs.size(); ++vertex)
        {
            const auto &price = profile->target_costs[vertex];
            prices[vertex] = price.nanoseconds_per_activation[0];
            for (int expert = 0; expert < 4; ++expert)
            {
                const auto *owner = snapshot->placementOwnerMap().placementFor(0, expert);
                const auto target = owner->shard_cohort ? MoEOverlayRouteTarget::cpuCohort(owner->tier_idx) :
                    MoEOverlayRouteTarget::whole(owner->owner_participant);
                if (target.word() == price.target.word()) before[expert] = int32_t(vertex);
            }
        }
        after = before; std::swap(after[0], after[1]);
        const std::array<int32_t, 4> routes{0, 1, 1, 1};
        PreparedTransactionCost transaction({routes.data(), routes.size(), 4, 1, 1}, 4);
        std::array<ServiceCostPair, 2> scratch{};
        const auto scored = transaction.score({before.data(), after.data(), prices.data(), 4, 2}, scratch.data(), 2);
        PreparedTransactionSwapSearch search(transaction, before, prices);
        const auto decision = search.scoreSwap(0, 1);
        ASSERT_EQ(scored.status, TransactionCostStatus::Complete);
        EXPECT_EQ(decision.status, scored.status);
        EXPECT_EQ(decision.cost.before_ns, scored.cost.before_ns);
        EXPECT_EQ(decision.cost.after_ns, scored.cost.after_ns);
        EXPECT_EQ(scored.cost.before_ns, 90u);
        EXPECT_EQ(scored.cost.after_ns, 30u);
        EXPECT_EQ(authority.snapshot(), snapshot);
        EXPECT_THROW(authority.proposeFromHistogram(), std::logic_error);
        EXPECT_EQ(authority.snapshot(), snapshot);
    }
}

/** @brief Contradictory NUMA labels cannot price one physical accelerator as independent vertices. */
TEST(CPUJointAuthoritativeTargetContract, PreparedTargetsRejectOverlappingPhysicalVertices)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2, 2, 5);
    // Both complete target declarations name the same device on rank zero;
    // distinct NUMA labels do not manufacture independent accelerator hardware.
    original.plan.domains[0].participants = {
        GlobalDeviceAddress::cuda(0, 0), GlobalDeviceAddress::cuda(0, 1)};
    const auto source = std::make_shared<const MoERoutedExpertPlacementPlan>(original.plan);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(source,
        {.current_world_rank = 0, .validate_mvp_root_reachability = false});
    original.preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
        MoEExpertOverlayPreparationPlan::build(*runtime, original.loader));
    MoEOverlayResidencyAuthority authority(original.config());
    const auto snapshot = authority.snapshot();
    const auto cohort = snapshot->originalOwnerMap().placementFor(0, 2)->shard_cohort;
    ASSERT_TRUE(cohort);
    const auto catalog = std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(
        std::vector<MoEOverlayLayerWeightManifest>{cohort->source()});
    const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
    EXPECT_THROW(MoEOverlayPreparedTargetServicePlan::checked(snapshot, catalog, topology), std::invalid_argument);
    EXPECT_EQ(authority.snapshot(), snapshot);
}

/** @test The SAME production host authority accepts original complete-target prices and proposes typed swaps. */
TEST(CPUJointAuthoritativeTargetContract, DynamicUsesOrdinaryAuthorityAndTypedDemandWithoutOffPermission)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2);
    MoEOverlayResidencyAuthority cold(original.config());
    const auto snapshot = cold.snapshot();
    const auto catalog = std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(
        std::vector<MoEOverlayLayerWeightManifest>{snapshot->placementOwnerMap().placementFor(0, 1)->shard_cohort->source()});
    const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
    const auto declarations = MoEOverlayPreparedTargetServicePlan::checked(snapshot, catalog, topology);
    std::vector<MoEOverlayPreparedTargetServiceTotals> recorded;
    for (const auto &entry : declarations->entries())
    {
        MoEOverlayPreparedTargetServiceTotals row{.coordinate = entry.coordinate};
        for (std::size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
        {
            row.sample_count[phase] = entry.measured_invocations;
            row.activation_count[phase] = entry.sample_rows[phase] * entry.measured_invocations;
            row.total_nanoseconds[phase] = row.activation_count[phase] * (entry.coordinate.target.isWhole() ? 10 : 40);
            row.expert_ids[phase] = declarations->sampleExpert(entry, phase);
        }
        recorded.push_back(row);
    }
    // Explicit recorded policy inputs certify this device-free controller test
    // only. Native service/transport gates provide actual observation evidence.
    const auto service = MoEOverlayEconomyProfileComposer::composePreparedService(declarations, recorded);
    auto movement = std::make_shared<MoEOverlayMigrationCostProfile>();
    movement->identity = "recorded-original-complete-target-policy";
    MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot = snapshot,
        .complete_expert_bytes_per_layer = catalog->completeExpertBytesPerLayer()});
    for (const auto &coordinate : planner.requiredTargetCoordinates())
        movement->costs.push_back({coordinate, 1, 1});
    DecodeExpertHistogram histogram({.num_layers = 1, .num_experts = 4, .top_k = 2,
        .window_size = 1, .placement_ownership = original.preparation->cpuJointOwnerMap()});
    auto config = original.config();
    config.maintenance_mode = MoERebalanceRuntimeMode::Dynamic;
    config.histogram = &histogram;
    config.phase_service_profile = service;
    config.migration_cost_profile = movement;
    config.migration_economy_policy = MoEOverlayMigrationEconomyPolicy{
        .historical_window_weight = 0, .current_window_weight = 1,
        .payoff_horizon_tokens = 1000, .minimum_net_benefit_ns = 0, .minimum_residency_generations = 0};
    config.economy_layer_catalog = catalog;
    config.shadow_slots_per_endpoint_layer = 1;
    config.max_concurrent_cycles = 1;
    MoEOverlayResidencyAuthority authority(config);
    EXPECT_TRUE(authority.migrationEnabled());
    EXPECT_TRUE(authority.hasEconomyCertification());
    const auto window = observedLayerBatches({{{ExpertHistogramSource::DecodeToken, {1, 2}}}}, 4, 2, 17);
    const auto proposal = authority.proposeFromFrozenHistogramWindow(window);
    ASSERT_TRUE(proposal.valid());
    ASSERT_EQ(proposal.migrations.size(), 2u);
    EXPECT_GT(proposal.economy.projected_net_benefit_ns, 0u);
    EXPECT_TRUE(proposal.migrations[0].source.shard_cohort || proposal.migrations[1].source.shard_cohort);
    EXPECT_TRUE(proposal.migrations[0].destination.shard_cohort || proposal.migrations[1].destination.shard_cohort);
    EXPECT_EQ(authority.snapshot()->epoch, 1u); // Policy cannot publish physical readiness.
    auto forbidden_off = config; forbidden_off.maintenance_mode = MoERebalanceRuntimeMode::Off;
    EXPECT_THROW((void)MoEOverlayResidencyAuthority(forbidden_off), std::invalid_argument);
}

/** @test Movement telemetry counts only live original intersections and actual backends. */
TEST(CPUJointAuthoritativeTargetContract, CohortRankAndBackendFactsHaveNoScalarSentinelAlias)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2);
    MoEOverlayResidencyAuthority authority(original.config());
    const auto &map = authority.snapshot()->placementOwnerMap();
    const auto &whole = *map.placementFor(0, 0);
    const auto &joint = *map.placementFor(0, 1);
    MoEOverlayTierMigration edge{.source = whole, .destination = joint};
    EXPECT_TRUE(edge.crossesWorldRank());
    EXPECT_TRUE(edge.crossesBackend());
    edge.source = joint; edge.destination = whole;
    EXPECT_TRUE(edge.crossesWorldRank());
    EXPECT_TRUE(edge.crossesBackend());
    // Identical fixed cuts have only same-member intersections. Full Cartesian
    // member counts would falsely claim cross-rank bytes for this empty change.
    edge.destination = joint;
    EXPECT_FALSE(edge.crossesWorldRank());
    EXPECT_FALSE(edge.crossesBackend());
}

/** @test Observe counts the same logical routes but proves immobility through the ordinary typed API. */
TEST(CPUJointAuthoritativeTargetContract, ObserveCountsLogicalDemandWithoutPublicationPermission)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2);
    DecodeExpertHistogram histogram({.num_layers = 1, .num_experts = 4, .top_k = 1,
        .window_size = 1, .placement_ownership = original.preparation->cpuJointOwnerMap()});
    auto config = original.config();
    config.maintenance_mode = MoERebalanceRuntimeMode::Observe;
    config.histogram = &histogram;
    MoEOverlayResidencyAuthority authority(config);
    EXPECT_TRUE(authority.observesHistogram());
    EXPECT_FALSE(authority.migrationEnabled());
    const auto initial = authority.snapshot();
    const auto noop = authority.proposeFromHistogram();
    ASSERT_TRUE(noop.valid());
    EXPECT_TRUE(noop.empty());
    RecordingTransport no_movement(&authority);
    EXPECT_EQ(authority.beginApply(noop, no_movement).status, MoEOverlayResidencyApplyStatus::StaticNoMovement);
    EXPECT_TRUE(no_movement.calls.empty());
    EXPECT_EQ(authority.snapshot(), initial);
    EXPECT_FALSE(authority.hasEconomyCertification());
}
