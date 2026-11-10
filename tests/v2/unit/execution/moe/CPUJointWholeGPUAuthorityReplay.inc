/**
 * @file CPUJointWholeGPUAuthorityReplay.inc
 * @brief Replay normal host maintenance with original CPU cohorts and GPU peer placement.
 *
 * Recorded inputs exercise the real constructor, histogram planner, canonical
 * candidate/cycle composer, recording-transport commit and reuse restoration.
 * Original tiny GGUF metadata seals CPU cuts and preparation. Declared backend
 * and participant populations are metadata; these tests allocate no GPU or
 * fabricate native service/transfer qualification.
 */
#include "execution/moe/MoEOptimizationMovementTopology.h"
#include "execution/moe/MoEOverlayPreparedTransactionCost.h"

namespace llaminar2::test
{
    namespace
    {
        /** @brief Publish a checked controller transaction through its real state machine. */
        void publishJointControllerTransaction(MoEOverlayResidencyAuthority &authority,
            RecordingTransport &transport, const MoEOverlayResidencyTransaction &transaction)
        {
            ASSERT_TRUE(transaction.valid());
            ASSERT_FALSE(transaction.empty());
            ASSERT_EQ(authority.beginApply(transaction, transport).status, MoEOverlayResidencyApplyStatus::Started);
            ASSERT_EQ(authority.advanceBackground().status, MoEOverlayResidencyApplyStatus::Preparing);
            ASSERT_EQ(authority.advanceBackground().status, MoEOverlayResidencyApplyStatus::Publishing);
            ASSERT_EQ(authority.advanceBackground().status, MoEOverlayResidencyApplyStatus::Published);
            ASSERT_EQ(authority.advanceBackground().status, MoEOverlayResidencyApplyStatus::Idle);
        }

        /** @brief Prove complete logical/physical slot counts and original cohort control blocks. */
        void expectJointSourcePreserved(const MoEOverlayResidencySnapshot &before,
            const MoEOverlayResidencySnapshot &after)
        {
            EXPECT_TRUE(after.valid());
            EXPECT_TRUE(after.layered_ownership.empty());
            EXPECT_EQ(after.originalProjectionPreparation(), before.originalProjectionPreparation());
            EXPECT_EQ(&after.originalOwnerMap(), &before.originalOwnerMap());
            for (const auto &participant : before.placementOwnerMap().participants())
                EXPECT_EQ(before.placementOwnerMap().expertsForParticipant(0, participant.participant_id).size(),
                          after.placementOwnerMap().expertsForParticipant(0, participant.participant_id).size());
            for (const auto &row : after.placementOwnerMap().owners())
                if (row.shard_cohort)
                {
                    const auto *source = before.originalOwnerMap().placementFor(row.layer_idx, row.expert_id);
                    // All original cohort rows in one tier/layer share this SAME source/cut owner.
                    ASSERT_NE(source, nullptr);
                    if (!source->shard_cohort)
                    {
                        const auto found = std::find_if(before.originalOwnerMap().owners().begin(),
                            before.originalOwnerMap().owners().end(), [&](const auto &candidate) {
                                return candidate.layer_idx == row.layer_idx && candidate.tier_idx == row.tier_idx && candidate.shard_cohort;
                            });
                        ASSERT_NE(found, before.originalOwnerMap().owners().end());
                        source = &*found;
                    }
                    EXPECT_EQ(row.shard_cohort.get(), source->shard_cohort.get());
                    EXPECT_FALSE(row.shard_cohort.owner_before(source->shard_cohort));
                    EXPECT_FALSE(source->shard_cohort.owner_before(row.shard_cohort));
                    EXPECT_EQ(row.owner_participant, -1);
                    EXPECT_FALSE(row.device.is_valid());
                }
        }
    }
}

/** @test Both declared axes retain CPU source identity through peer swap, no-op and repeated reset/replay. */
TEST(CPUJointWholeGPUAuthorityReplay, RuntimeConstructorPeerSwapCommitResetAndReplay)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (int members : {2, 4, 8})
    for (int gpus : {2, 4})
    {
        SCOPED_TRACE(::testing::Message() << int(backend) << '/' << members << '/' << gpus);
        const int experts = 3 * gpus + 3;
        JointPlacementOriginal original(members, gpus, experts, 1, 0, false, backend);
        DecodeExpertHistogram histogram({.num_layers = 1, .num_experts = experts, .top_k = 1,
            .window_size = 1, .placement_ownership = original.preparation->cpuJointOwnerMap()});
        auto config = original.config();
        config.maintenance_mode = MoERebalanceRuntimeMode::Dynamic;
        config.histogram = &histogram;
        config.participant_rebalance_policy = {.enabled = true, .maximum_swaps_per_layer = 1,
            .maximum_plan_entries_per_wave = 2, .minimum_window_activations = 1};
        config.max_concurrent_cycles = 1;
        config.shadow_slots_per_endpoint_layer = 1;
        ASSERT_EQ(availableMoEOptimizationMovementAxes(config.initial_plan), MoEOptimizationMovementAxes::Both);
        MoEOverlayResidencyAuthority authority(config);
        RecordingTransport transport(&authority);
        const auto initial = authority.snapshot();
        std::vector<int> routes(50, 0);
        routes.insert(routes.end(), 50, 1);
        std::vector<std::tuple<int, int, int>> recorded_edges;
        for (int replay = 0; replay < 2; ++replay)
        {
            const auto window = observedLayerBatches({{{ExpertHistogramSource::PrefillChunk, routes}}}, experts, 1, 17 + replay);
            const auto proposal = authority.proposeFromFrozenHistogramWindow(window);
            ASSERT_TRUE(proposal.valid());
            ASSERT_EQ(proposal.migrations.size(), 2u);
            ASSERT_EQ(proposal.migration_cycles.size(), 1u);
            EXPECT_TRUE(std::all_of(proposal.migrations.begin(), proposal.migrations.end(), [](const auto &edge) {
                return edge.axis == MoEOptimizationMovementAxis::ParticipantPlacement;
            }));
            std::vector<std::tuple<int, int, int>> edges;
            for (const auto &move : proposal.migrations)
            {
                EXPECT_FALSE(move.crossesTier());
                EXPECT_FALSE(move.source.shard_cohort);
                EXPECT_FALSE(move.destination.shard_cohort);
                EXPECT_TRUE(move.source.device.is_gpu());
                EXPECT_TRUE(move.destination.device.is_gpu());
                edges.emplace_back(move.expert_id, move.source.owner_participant, move.destination.owner_participant);
            }
            if (replay == 0) recorded_edges = edges;
            else EXPECT_EQ(edges, recorded_edges);
            EXPECT_TRUE(analyzeMoEOverlayMigrationCapacity(proposal.migrations).capacityPreserved());
            expectJointSourcePreserved(*initial, *proposal.candidate);
            publishJointControllerTransaction(authority, transport, proposal);
            const auto noop = authority.proposeFromFrozenHistogramWindow(window);
            EXPECT_TRUE(noop.valid());
            EXPECT_TRUE(noop.empty());
            const auto restoration = authority.proposeInitialPreparedPlacementRestoration();
            ASSERT_EQ(restoration.migrations.size(), 2u);
            publishJointControllerTransaction(authority, transport, restoration);
            EXPECT_TRUE(authority.initialPreparedPlacementPublished());
            EXPECT_EQ(authority.snapshot()->placementOwnerMap().owners(), initial->placementOwnerMap().owners());
            expectJointSourcePreserved(*initial, *authority.snapshot());
        }
        EXPECT_EQ(authority.activeTicketCount(), 0u);
        EXPECT_EQ(authority.pendingRetirementCount(), 0u);
    }
}

/** @test Dense target-price indices include CPU cohorts and compose tier/GPU moves into one profitable cycle. */
TEST(CPUJointWholeGPUAuthorityReplay, PreparedTargetEconomyComposesPromotedPeerMoveAndAuthoritativeReplay)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (int members : {2, 4, 8})
    {
        SCOPED_TRACE(::testing::Message() << int(backend) << '/' << members);
        JointPlacementOriginal original(members, 2, 9, 1, 0, false, backend);
        MoEOverlayResidencyAuthority cold(original.config());
        const auto source = cold.snapshot();
        const auto catalog = std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(
            std::vector<MoEOverlayLayerWeightManifest>{source->placementOwnerMap().placementFor(0, 6)->shard_cohort->source()});
        const auto topology = ExpertHistogramProductionTopology::uniform(1, kAllExpertHistogramProductionSources);
        const auto declarations = MoEOverlayPreparedTargetServicePlan::checked(source, catalog, topology);
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
        auto service = MoEOverlayEconomyProfileComposer::composePreparedService(declarations, recorded);
        auto movement = std::make_shared<MoEOverlayMigrationCostProfile>();
        movement->identity = "recorded-whole-gpu-plus-original-cpu-cohort";
        MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot = source,
            .complete_expert_bytes_per_layer = catalog->completeExpertBytesPerLayer()});
        for (const auto &coordinate : planner.requiredTargetCoordinates())
            movement->costs.push_back({coordinate, 1, 1});
        DecodeExpertHistogram histogram({.num_layers = 1, .num_experts = 9, .top_k = 1,
            .window_size = 1, .placement_ownership = original.preparation->cpuJointOwnerMap()});
        auto config = original.config();
        config.maintenance_mode = MoERebalanceRuntimeMode::Dynamic;
        config.histogram = &histogram;
        config.participant_rebalance_policy = {.enabled = true, .maximum_swaps_per_layer = 1,
            .maximum_plan_entries_per_wave = 2, .minimum_window_activations = 1};
        config.phase_service_profile = service;
        config.migration_cost_profile = movement;
        config.migration_economy_policy = MoEOverlayMigrationEconomyPolicy{
            .historical_window_weight = 0, .current_window_weight = 1, .payoff_horizon_tokens = 1000,
            .minimum_net_benefit_ns = 0, .minimum_residency_generations = 0};
        config.economy_layer_catalog = catalog;
        config.max_concurrent_cycles = 2;
        config.shadow_slots_per_endpoint_layer = 1;
        MoEOverlayResidencyAuthority authority(config);
        // Each rank/authority owns its own histogram admission lifecycle. The
        // first pre-certified constructor has already activated its histogram.
        DecodeExpertHistogram peer_histogram({.num_layers = 1, .num_experts = 9, .top_k = 1,
            .window_size = 1, .placement_ownership = original.preparation->cpuJointOwnerMap()});
        auto peer_config = config;
        peer_config.histogram = &peer_histogram;
        MoEOverlayResidencyAuthority peer(peer_config);
        std::vector<int> routes(20, 3);
        routes.insert(routes.end(), 20, 4);
        routes.insert(routes.end(), 80, 6);
        const auto observed = observedLayerBatches({{{ExpertHistogramSource::PrefillChunk, routes}}}, 9, 1, 17);
        // Use the ACTUAL source-bound CPU cohort and apportioned GPU targets,
        // including its negative target word (never the absent scalar owner).
        // Compare every pair to the original complete scorer before exercising
        // normal authority selection/three-edge final coherent admission below.
        std::vector<MoEOverlayRouteTarget> targets;
        for (const auto &price : service->target_costs)
            if (price.layer == 0) targets.push_back(price.target);
        std::sort(targets.begin(), targets.end(), [](auto a, auto b) { return a.word() < b.word(); });
        ASSERT_EQ(targets.size(), 3u);
        ASSERT_TRUE(targets.front().isCPUCohort());
        std::vector<int32_t> owners(9);
        for (const auto &row : source->placementOwnerMap().placementsForLayer(0))
        {
            const auto target = row.shard_cohort ? MoEOverlayRouteTarget::cpuCohort(row.tier_idx)
                                                : MoEOverlayRouteTarget::whole(row.owner_participant);
            const auto found = std::find_if(targets.begin(), targets.end(),
                [&](auto candidate) { return candidate.word() == target.word(); });
            ASSERT_NE(found, targets.end());
            owners.at(row.expert_id) = int32_t(found - targets.begin());
            if (row.shard_cohort) EXPECT_EQ(row.owner_participant, -1);
        }
        for (const uint32_t rows : {1u, 3u, 64u, 448u})
        {
            std::vector<int32_t> ids(rows);
            for (uint32_t row = 0; row < rows; ++row) ids[row] = (row * 7) % 8; // expert8 has zero demand
            const moe_overlay_economy::PreparedTransactionCost prepared({ids.data(), ids.size(), rows, 1, 1}, 9);
            for (size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
            {
                std::vector<uint64_t> prices;
                for (const auto target : targets)
                {
                    const auto found = std::find_if(service->target_costs.begin(), service->target_costs.end(),
                        [&](const auto &price) { return price.layer == 0 && price.target.word() == target.word(); });
                    ASSERT_NE(found, service->target_costs.end());
                    prices.push_back(found->nanoseconds_per_activation[phase]);
                }
                const moe_overlay_economy::PreparedTransactionSwapSearch exact(prepared, owners, prices);
                std::vector<moe_overlay_economy::ServiceCostPair> scratch(targets.size());
                for (uint32_t first = 0; first < 9; ++first)
                    for (uint32_t second = 0; second < 9; ++second)
                    {
                        auto after = owners;
                        std::swap(after[first], after[second]);
                        const auto full = prepared.score({owners.data(), after.data(), prices.data(), 9,
                            uint32_t(prices.size())}, scratch.data(), uint32_t(scratch.size()));
                        const auto actual = exact.scoreSwap(first, second);
                        ASSERT_EQ(actual.status, full.status);
                        ASSERT_TRUE(actual.complete());
                        EXPECT_EQ(actual.cost.before_ns, full.cost.before_ns);
                        EXPECT_EQ(actual.cost.after_ns, full.cost.after_ns);
                    }
            }
        }
        // Prove the PRODUCTION pre-cut branch domain, not merely the final
        // three-edge composition. These are the same original planner/current
        // source and candidate constructor, before participant correction.
        auto tier_plan = *source->placement_plan;
        tier_plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        MoERoutedExpertPlacementPlannerOptions tier_options;
        tier_options.decode_histogram_window = observed.get();
        tier_options.phase_service_profile = service.get();
        tier_options.rebalancer.enabled = true;
        tier_options.rebalancer.previous_placements = source->placement_plan->placements;
        const auto tier_result = MoERoutedExpertPlacementPlanner::plan(tier_plan, config.model_metadata, tier_options);
        const auto tier_transaction = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(
            authority, tier_result.planned_plan.placements);
        ASSERT_TRUE(tier_transaction.valid());
        ASSERT_EQ(tier_transaction.migrations.size(), 2u);
        ASSERT_EQ(tier_transaction.migration_cycles.size(), 1u);
        ASSERT_EQ(tier_transaction.migration_cycles.front().migration_indices.size(), 2u);
        std::vector<std::pair<const MoEExpertOwner *, const MoEExpertOwner *>> changed;
        for (const auto &after : tier_transaction.candidate->placementOwnerMap().placementsForLayer(0))
        {
            const auto *before = source->placementOwnerMap().placementFor(0, after.expert_id);
            ASSERT_NE(before, nullptr);
            if (!(*before == after)) changed.push_back({before, &after});
        }
        ASSERT_EQ(changed.size(), 2u);
        EXPECT_NE(changed[0].first->expert_id, changed[1].first->expert_id);
        const auto target_word = [](const MoEExpertOwner &row)
        {
            return (row.shard_cohort ? MoEOverlayRouteTarget::cpuCohort(row.tier_idx)
                                     : MoEOverlayRouteTarget::whole(row.owner_participant)).word();
        };
        EXPECT_EQ(target_word(*changed[0].first), target_word(*changed[1].second));
        EXPECT_EQ(target_word(*changed[1].first), target_word(*changed[0].second));
        EXPECT_TRUE(changed[0].first->shard_cohort || changed[1].first->shard_cohort);
        const auto proposal = authority.proposeFromFrozenHistogramWindow(observed);
        ASSERT_TRUE(proposal.valid());
        ASSERT_EQ(proposal.migration_cycles.size(), 1u);
        ASSERT_EQ(proposal.migrations.size(), 3u);
        EXPECT_TRUE(std::any_of(proposal.migrations.begin(), proposal.migrations.end(), [](const auto &edge) {
            return edge.axis == MoEOptimizationMovementAxis::ParticipantPlacement;
        }));
        EXPECT_GT(proposal.economy.projected_net_benefit_ns, 0u);
        EXPECT_TRUE(std::any_of(proposal.migrations.begin(), proposal.migrations.end(), [](const auto &edge) {
            return edge.expert_id == 6 && edge.source.shard_cohort && edge.destination.device.is_gpu() &&
                edge.axis == MoEOptimizationMovementAxis::Combined;
        }));
        expectJointSourcePreserved(*source, *proposal.candidate);
        const auto encoded = authority.exportAuthoritativeResidencyPlan(proposal);
        const auto decoded = peer.adoptAuthoritativeResidencyPlan(encoded);
        ASSERT_TRUE(decoded.valid());
        EXPECT_EQ(fingerprintMoEOverlayResidencyExecutionPlan(proposal), fingerprintMoEOverlayResidencyExecutionPlan(decoded));
        EXPECT_EQ(proposal.candidate->placementOwnerMap().owners(), decoded.candidate->placementOwnerMap().owners());
        RecordingTransport transport(&authority);
        publishJointControllerTransaction(authority, transport, proposal);
        expectJointSourcePreserved(*source, *authority.snapshot());
        const auto restore = authority.proposeInitialPreparedPlacementRestoration();
        publishJointControllerTransaction(authority, transport, restore);
        EXPECT_TRUE(authority.initialPreparedPlacementPublished());
        EXPECT_EQ(authority.snapshot()->placementOwnerMap().owners(), source->placementOwnerMap().owners());
    }
}
