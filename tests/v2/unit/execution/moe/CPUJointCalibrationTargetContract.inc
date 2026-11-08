/**
 * @file CPUJointCalibrationTargetContract.inc
 * @brief Genuine source-bound whole/cohort shadow calibration without maintenance permission.
 *
 * Actual tiny GGUF metadata and the same source preparation seal the initial
 * placement. Widths2..8 are metadata declarations, never fabricated sockets,
 * engines, ready banks or native transfer observations. The scalar downstream
 * economy consumers remain explicitly unsupported for cohort coordinates.
 */
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"

TEST(CPUJointCalibrationTargetContract, CompleteOriginalTargetsProduceUnpublishableShadowWaves)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        JointPlacementOriginal original(width);
        MoEOverlayResidencyAuthority authority(original.config());
        const auto live = authority.snapshot();
        const auto cohort_source = live->originalOwnerMap().placementFor(0, 1)->shard_cohort;
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({cohort_source->source()});
        MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot=live,
            .complete_expert_bytes_per_layer=catalog.completeExpertBytesPerLayer()});
        const auto &coordinates = planner.requiredTargetCoordinates();
        ASSERT_EQ(coordinates.size(), 2u); // One whole plus one complete cohort, not width+1 owners.
        EXPECT_TRUE(std::is_sorted(coordinates.begin(), coordinates.end()));
        EXPECT_THROW((void)planner.requiredCoordinates(), std::logic_error);
        const auto whole = MoEOverlayRouteTarget::whole(0);
        const auto cohort = MoEOverlayRouteTarget::cpuCohort(1);
        for (const auto &coordinate : coordinates)
        {
            ASSERT_TRUE(coordinate.valid());
            auto wave = planner.buildPairSwap(coordinate.source_target, coordinate.destination_target,
                                              coordinate.layer, 17);
            ASSERT_TRUE(wave.valid());
            EXPECT_EQ(wave.purpose, MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
            EXPECT_EQ(wave.calibration_sequence, 17u);
            EXPECT_EQ(wave.previous, live);
            EXPECT_EQ(wave.candidate->epoch, live->epoch + 1);
            EXPECT_EQ(wave.candidate->placement_plan, live->placement_plan);
            EXPECT_EQ(wave.candidate->jointOwnerMap(), live->jointOwnerMap());
            EXPECT_EQ(wave.candidate->originalProjectionPreparation(), live->originalProjectionPreparation());
            ASSERT_EQ(wave.migrations.size(), 2u);
            ASSERT_EQ(wave.migration_cycles.size(), 1u);
            EXPECT_EQ(wave.shadow_requirements.size(), size_t(width + 1));
            for (const auto &requirement : wave.shadow_requirements) EXPECT_EQ(requirement.slot_count, 1u);
            EXPECT_TRUE(analyzeMoEOverlayMigrationCapacity(wave.migrations).capacityPreserved());
            EXPECT_EQ(authority.snapshot(), live);
            EXPECT_FALSE(wave.histogram_window);
            EXPECT_FALSE(wave.host_admission);
            EXPECT_FALSE(wave.economy.enabled);
            EXPECT_THROW((void)authority.exportAuthoritativeResidencyPlan(wave), std::invalid_argument);
            RecordingTransport no_staging(&authority);
            EXPECT_THROW((void)authority.beginApply(wave, no_staging), std::logic_error);
            EXPECT_TRUE(no_staging.calls.empty());
            auto publish = wave;
            publish.purpose = MoEOverlayResidencyTransactionPurpose::PlacementChange;
            publish.calibration_sequence = 0;
            EXPECT_FALSE(publish.valid()); // A shadow wave is not a changed live placement.
            auto bad = wave;
            bad.shadow_requirements.pop_back();
            EXPECT_FALSE(bad.valid());
            bad = wave;
            bad.calibration_sequence = 0;
            EXPECT_FALSE(bad.valid());
            bad = wave;
            bad.migrations.clear();
            bad.migration_cycles.clear();
            bad.shadow_requirements.clear();
            EXPECT_FALSE(bad.valid());
            bad = wave;
            bad.migrations.front().activation_count = 9;
            // Calibration carries no routing evidence or hotness observations.
            EXPECT_FALSE(bad.valid());
            bad = wave;
            for (auto &edge : bad.migrations)
                if (edge.destination.shard_cohort)
                {
                    edge.destination.shard_cohort = std::shared_ptr<const MoEExpertShardCohort>(
                        edge.destination.shard_cohort.get(), [](auto *) {});
                    break;
                }
            EXPECT_FALSE(bad.valid()); // Identical address without original control block is foreign.
        }
        EXPECT_THROW((void)planner.buildPairSwap(0, 1, 0, 3), std::invalid_argument);
        EXPECT_THROW((void)planner.buildPairSwap(whole, whole, 0, 3), std::invalid_argument);
        EXPECT_THROW((void)planner.buildPairSwap(whole, cohort, 0, 0), std::invalid_argument);
        EXPECT_THROW((void)planner.buildPairSwap(whole, MoEOverlayRouteTarget::cpuCohort(7), 0, 3), std::invalid_argument);
        EXPECT_THROW((void)planner.buildPairSwap(whole, cohort, 1, 3), std::invalid_argument);
        // Ordinary public joint maintenance remains unsupported despite valid metadata.
        EXPECT_THROW((void)authority.proposeFromHistogram(), std::logic_error);
    }
}

TEST(CPUJointCalibrationTargetContract, WholeAdapterIsOneCheckedProjectionOfCanonicalTargets)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    const auto plan = oneTierTwoParticipantPlan(RoutedExpertResidencyPolicy::StaticById);
    MoEOverlayResidencyAuthority authority({.initial_plan=plan,
        .model_metadata={.num_layers=1,.num_experts=4,.d_model=64,.routed_intermediate_size=128}});
    MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot=authority.snapshot(),
        .complete_expert_bytes_per_layer={4096}});
    const auto scalar = planner.requiredCoordinates();
    const auto &typed = planner.requiredTargetCoordinates();
    ASSERT_EQ(scalar.size(), typed.size());
    ASSERT_EQ(typed.size(), 2u);
    for (size_t i = 0; i < typed.size(); ++i)
    {
        EXPECT_TRUE(typed[i].source_target.isWhole());
        EXPECT_TRUE(typed[i].destination_target.isWhole());
        EXPECT_EQ(typed[i].source_target.word(), scalar[i].source_participant);
        EXPECT_EQ(typed[i].destination_target.word(), scalar[i].destination_participant);
        const auto by_int = planner.buildPairSwap(scalar[i].source_participant,
            scalar[i].destination_participant, scalar[i].layer, 4);
        const auto by_target = planner.buildPairSwap(typed[i].source_target,
            typed[i].destination_target, typed[i].layer, 4);
        ASSERT_TRUE(by_int.valid());
        ASSERT_TRUE(by_target.valid());
        EXPECT_EQ(fingerprintMoEOverlayResidencyExecutionPlan(by_int),
                  fingerprintMoEOverlayResidencyExecutionPlan(by_target));
    }
}

TEST(CPUJointCalibrationTargetContract, EmptyOriginalQuotaCannotBecomeAShadowSource)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2, 1, 3); // All original experts belong to the CPU target.
    MoEOverlayResidencyAuthority authority(original.config());
    const MoEOverlayEconomyCalibrationLayerCatalog catalog({
        authority.snapshot()->originalOwnerMap().placementFor(0, 0)->shard_cohort->source()});
    EXPECT_THROW((void)MoEOverlayEconomyCalibrationPlanner({.live_snapshot=authority.snapshot(),
        .complete_expert_bytes_per_layer=catalog.completeExpertBytesPerLayer()}), std::invalid_argument);
}

TEST(CPUJointCalibrationTargetContract, UnequalOriginalCohortsAreCompleteTargetsNotPhysicalEdges)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    JointPlacementOriginal original(2, 1, 7);
    original.plan.routed_tiers[0].max_experts_per_layer = 1;
    auto second_domain = original.plan.domains[1];
    second_domain.name = "second_original_cpu_cohort";
    second_domain.participants.clear();
    second_domain.world_ranks.clear();
    for (int member = 0; member < 3; ++member)
    {
        second_domain.participants.push_back(GlobalDeviceAddress::cpu(2 + member));
        second_domain.world_ranks.push_back(2 + member);
    }
    original.plan.domains.push_back(second_domain);
    auto second_tier = original.plan.routed_tiers[1];
    second_tier.name = "second_cpu_tier";
    second_tier.domain = second_domain.name;
    second_tier.priority = 2;
    // Only the final capacity tier is fallback; the earlier CPU cohort keeps its three-expert quota.
    original.plan.routed_tiers[1].fallback = false;
    original.plan.routed_tiers.push_back(second_tier);
    original.plan.placements[0].routed_expert_tier = {0,1,1,1,2,2,2};
    const auto declared = std::make_shared<const MoERoutedExpertPlacementPlan>(original.plan);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(declared,
        {.current_world_rank=0,.validate_mvp_root_reachability=false});
    original.preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
        MoEExpertOverlayPreparationPlan::build(*runtime, original.loader));
    MoEOverlayResidencyAuthority authority(original.config());
    const auto live = authority.snapshot();
    const auto source = live->originalOwnerMap().placementFor(0, 1)->shard_cohort;
    const MoEOverlayEconomyCalibrationLayerCatalog catalog({source->source()});
    MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot=live,
        .complete_expert_bytes_per_layer=catalog.completeExpertBytesPerLayer()});
    ASSERT_EQ(planner.requiredTargetCoordinates().size(), 6u); // Three targets, not six physical members.
    const auto wave = planner.buildPairSwap(MoEOverlayRouteTarget::cpuCohort(1),
        MoEOverlayRouteTarget::cpuCohort(2), 0, 1);
    ASSERT_TRUE(wave.valid());
    ASSERT_EQ(wave.migrations.size(), 2u);
    EXPECT_EQ(wave.migrations[0].source.shard_cohort->participants().size(), 2u);
    EXPECT_EQ(wave.migrations[0].destination.shard_cohort->participants().size(), 3u);
    EXPECT_EQ(wave.shadow_requirements.size(), 5u);
    EXPECT_EQ(wave.previous->jointOwnerMap(), wave.candidate->jointOwnerMap());
    auto missing = wave;
    missing.migrations.pop_back();
    missing.migration_cycles.clear();
    missing.shadow_requirements = buildMoEOverlayShadowRequirements(missing.migrations);
    EXPECT_FALSE(missing.valid());
}

/** @test Recorded geometry observations price only exact original source-equivalent layers. */
TEST(CPUJointCalibrationTargetContract, SourceBoundExpansionPreservesMeasuredLayerAndRejectsForeignGeometry)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (const int first : {0, 40})
    {
        SCOPED_TRACE(first);
        JointPlacementOriginal original(2, 1, 4, 2, first);
        MoEOverlayResidencyAuthority authority(original.config());
        const auto snapshot = authority.snapshot();
        std::vector<MoEOverlayLayerWeightManifest> manifests;
        for (int layer = first; layer < first + 2; ++layer)
            manifests.push_back(snapshot->placementOwnerMap().placementFor(layer, 1)->shard_cohort->source());
        const MoEOverlayEconomyCalibrationLayerCatalog catalog(manifests);
        // Prove the actual source builder's rule, not a test-authored split:
        // one plan-level member order and balanced source-block intervals must
        // produce equal Gate/Up F rows and Down K columns in each class layer.
        for (int layer = first; layer < first + 2; ++layer)
        {
            const auto cohort = snapshot->placementOwnerMap().placementFor(layer, 1)->shard_cohort;
            const auto intervals = MoEExpertShardCohort::balancedIntervals(cohort->source(), 2);
            for (std::size_t member = 0; member < intervals.size(); ++member)
            {
                EXPECT_EQ(cohort->participants()[member].participant_id,
                          snapshot->placementOwnerMap().placementFor(first, 1)->shard_cohort->participants()[member].participant_id);
                for (const auto role : {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp})
                {
                    const auto cut = cohort->ownership(member).projection(role);
                    EXPECT_EQ(cut.first_row, intervals[member].first_column);
                    EXPECT_EQ(cut.rows, intervals[member].columns);
                }
                const auto down = cohort->ownership(member).projection(WeightRole::MoEExpertDown);
                EXPECT_EQ(down.first_column, intervals[member].first_column);
                EXPECT_EQ(down.columns, intervals[member].columns);
            }
        }
        MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot = snapshot,
            .complete_expert_bytes_per_layer = catalog.completeExpertBytesPerLayer()});
        // These are explicitly recorded protocol inputs, not a native timing
        // claim. The separate two-socket controller test supplies actual times.
        const auto recorded = [&](const MoEOverlayEconomyCalibrationPlanner &source_planner,
                                  const MoEOverlayResidencySnapshot &source_snapshot) {
            MoEOverlaySealedMigrationMeasurements result{.identity = "recorded-original-target-expansion"};
            for (const auto &coordinate : source_planner.requiredTargetCoordinates())
            {
                if (coordinate.layer != first) continue;
                const auto wave = source_planner.buildPairSwap(coordinate.source_target,
                    coordinate.destination_target, coordinate.layer, 1);
                auto geometry = makePhysicalMoEOverlayMeasurementGeometry(wave.migrations.front(),
                    manifests.front(), source_snapshot.expertCount(), source_snapshot.placementOwnerMap());
                MoEOverlayTargetLayerMigrationMeasurement row{.coordinate = coordinate, .geometry = geometry,
                    .wave_wall_nanoseconds = 100, .wave_sample_count = 3};
                for (std::size_t piece = 0; piece < geometry->pieces().size(); ++piece)
                    row.pieces.push_back({128 + piece, {.sequence = 1, .bytes = 128 + piece,
                        .wall_nanoseconds = 100, .host_nanoseconds = 100}});
                result.rows.push_back(std::move(row));
            }
            return result;
        };
        const auto measured = recorded(planner, *snapshot);
        ASSERT_TRUE(measured.valid());
        const auto expanded = catalog.expand(measured, snapshot);
        ASSERT_TRUE(expanded.valid());
        ASSERT_EQ(expanded.rows.size(), 4u);
        for (const auto &row : expanded.rows)
        {
            const auto input = std::find_if(measured.rows.begin(), measured.rows.end(), [&](const auto &candidate) {
                return candidate.coordinate.source_target.word() == row.coordinate.source_target.word();
            });
            ASSERT_NE(input, measured.rows.end());
            EXPECT_EQ(row.geometry, input->geometry); // SAME physically measured geometry owner.
            EXPECT_EQ(row.wave_wall_nanoseconds, input->wave_wall_nanoseconds);
            EXPECT_EQ(row.wave_sample_count, input->wave_sample_count);
            EXPECT_EQ(bool(row.measured_coordinate), row.coordinate.layer != first);
            if (row.measured_coordinate) EXPECT_EQ(*row.measured_coordinate, input->coordinate);
        }
        EXPECT_EQ(MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(expanded.rows).size(), 4u);
        EXPECT_THROW((void)catalog.expand(expanded, snapshot), std::invalid_argument); // No projection chaining.
        EXPECT_THROW((void)catalog.expand(measured), std::invalid_argument); // Whole adapter remains strict.
        auto missing = measured; missing.rows.pop_back();
        EXPECT_THROW((void)catalog.expand(missing, snapshot), std::invalid_argument);
        auto duplicate = measured; duplicate.rows.push_back(duplicate.rows.front());
        EXPECT_THROW((void)catalog.expand(duplicate, snapshot), std::invalid_argument);
        auto wrong_format = measured;
        const auto &geometry = *wrong_format.rows.front().geometry;
        auto formats = geometry.formats(); formats[0] = ExpertWeightFormat::floating(TensorType::FP16);
        wrong_format.rows.front().geometry = MoEOverlayMigrationMeasurementGeometry::checked(geometry.targets(), formats);
        ASSERT_TRUE(wrong_format.valid());
        EXPECT_THROW((void)catalog.expand(wrong_format, snapshot), std::invalid_argument);
        auto different_cuts = measured;
        auto targets = measured.rows.front().geometry->targets();
        auto &cohort = targets.first.members.size() == 2 ? targets.first : targets.second;
        const auto down_index = static_cast<std::size_t>(ExpertTierWeightProjection::Down);
        const int full_f = cohort.members.front().projections[down_index].source_columns;
        for (std::size_t member = 0; member < cohort.members.size(); ++member)
        {
            const int begin = member == 0 ? 0 : full_f / 4;
            const int width = member == 0 ? full_f / 4 : full_f - full_f / 4;
            for (const auto role : {ExpertTierWeightProjection::Gate, ExpertTierWeightProjection::Up})
            {
                auto &cut = cohort.members[member].projections[static_cast<std::size_t>(role)];
                cut.first_row = begin; cut.rows = width;
            }
            auto &down = cohort.members[member].projections[down_index];
            down.first_column = begin; down.columns = width;
        }
        // This unequal passive partition is well-formed geometry, but it was
        // never the original source builder's admitted balanced partition.
        different_cuts.rows.front().geometry = MoEOverlayMigrationMeasurementGeometry::checked(
            std::move(targets), measured.rows.front().geometry->formats());
        ASSERT_TRUE(different_cuts.valid());
        EXPECT_THROW((void)catalog.expand(different_cuts, snapshot), std::invalid_argument);
        EXPECT_EQ(authority.snapshot(), snapshot);
        EXPECT_FALSE(authority.migrationEnabled());
    }
}
