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
