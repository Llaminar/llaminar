/**
 * @file CPUJointCalibrationControllerContract.inc
 * @brief Actual two-socket source/fabric/controller/distributed-evidence/ledger lifecycle.
 *
 * Original setup geometry is declared before any sample. The existing
 * controller alone chooses warmup/sample waves and owns every real staging,
 * abort/drain and evidence exchange. No fake-ready factory, source permit,
 * synthetic inference pair or policy loop supplies measurements.
 */
#include "execution/moe/MoEOverlayEconomyCalibrationController.h"
#include "execution/moe/MoEOverlayDistributedResidencyTransport.h"
#include "execution/moe/MoEOverlayMPIResidencyConsensus.h"
#include "execution/moe/MoEOverlayMPIEconomyEvidenceExchange.h"

namespace llaminar2::test {
namespace {
/** @brief Drive the original complete-target calibration policy over real physical/MPI owners. */
void proveCPUJointCalibrationController()
{
    FabricOriginal original;
    const auto initial=original.authority->snapshot();
    const auto footprint=MoEOverlayCapacityResolver::preparedFootprints(original.manifests).front();
    auto planner=std::make_shared<MoEOverlayEconomyCalibrationPlanner>(
        MoEOverlayEconomyCalibrationPlanner::Config{.live_snapshot=initial,
            .complete_expert_bytes_per_layer={footprint.cpu_live_bytes},.calibration_layers={0}});
    const auto& coordinates=planner->requiredTargetCoordinates();
    ASSERT_EQ(coordinates.size(),2u);
    std::vector<MoEOverlayRequiredTargetMigrationMeasurement> declarations;
    for(const auto& coordinate:coordinates) {
        const auto original_wave=planner->buildPairSwap(coordinate.source_target,
            coordinate.destination_target,coordinate.layer,91);
        ASSERT_TRUE(original_wave.valid());ASSERT_EQ(original_wave.migrations.size(),2u);
        // Source-authored passive geometry is fixed BEFORE evidence arrives.
        // No setup byte quote is guessed: the actual Ledger binds invariant
        // operation quotes on its first fully merged stored measured sample.
        declarations.push_back({.coordinate=coordinate,
            .geometry=makePhysicalMoEOverlayMeasurementGeometry(original_wave.migrations.front(),
                original.manifests.front(),kExperts,initial->placementOwnerMap())});
    }
    {
        auto ledger=std::make_shared<MoEOverlayMigrationMeasurementLedger>(
            MoEOverlayMigrationMeasurementLedger::Config{.required_target_measurements=declarations,
                .warmup_samples_per_coordinate=1,.measured_samples_per_coordinate=3,
                .measurement_identity="native-cpu-joint-controller-original-targets"});
        auto journal=std::make_shared<MoEOverlayMigrationMeasurementJournal>(
            MoEOverlayMigrationMeasurementJournal::Config{.maximum_migrations_per_wave=2});
        MoEOverlayParticipantPreparedWaveFactory factory({.registry=original.registry,
            .transfer_provider=original.fabric,.perf_device="native-cpu-joint-controller"});
        auto local=std::make_shared<MoEOverlayTierMigrationTransport>(
            MoEOverlayTierMigrationTransport::Config{.factory=&factory,.transfer_progress_authority=original.fabric,
                .projections_per_expert=3,.measurement_sink=journal,.require_complete_local_measurements=true,
                .perf_device="native-cpu-joint-controller"});
        auto consensus=std::make_shared<MoEOverlayMPIResidencyConsensus>(
            MoEOverlayMPIResidencyConsensus::Config{.mpi_context=original.mpi,
                .perf_device="native-cpu-joint-controller"});
        auto distributed=std::make_shared<MoEOverlayDistributedResidencyTransport>(
            MoEOverlayDistributedResidencyTransport::Config{.local_transport=local.get(),
                .consensus=consensus,.perf_device="native-cpu-joint-controller"});
        auto evidence=std::make_shared<MoEOverlayMPIEconomyEvidenceExchange>(
            MoEOverlayMPIEconomyEvidenceExchange::Config{.mpi_context=original.mpi,
                .owner_map=initial->placementOwnerMap(),
                .production_topology=ExpertHistogramProductionTopology::uniform(1,kAllExpertHistogramProductionSources),
                .perf_device="native-cpu-joint-controller"});
        MoEOverlayEconomyCalibrationController controller({.planner=planner,.ledger=ledger,
            .journal=journal,.transport=distributed,.evidence_exchange=evidence,
            .perf_device="native-cpu-joint-controller"});
        // Fatal errors wake the peer while every actual owner is still alive.
        // Successful completion is the controller's normal abort/drain path,
        // never requestStop or a fabricated terminal state.
        try {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            for(;;) {
                controller.poll();
                if(!controller.healthy())throw std::runtime_error(controller.failureMessage());
                // Each original controller progresses independently. A test-side
                // collective here would hide rank skew in the real protocol.
                if(controller.state()==MoEOverlayEconomyCalibrationState::Complete)break;
                if(std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("CPU joint calibration controller exceeded original test deadline");
                std::this_thread::yield();
            }
            // Only terminal assertions/retirement synchronize the test ranks;
            // a lagging controller retains its own deadline and fatal abort.
            if(MPI_Barrier(MPI_COMM_WORLD)!=MPI_SUCCESS)
                throw std::runtime_error("CPU joint controller terminal barrier failed");
        } catch(const std::exception& failure) {
            ADD_FAILURE()<<failure.what();MPI_Abort(MPI_COMM_WORLD,2);throw;
        }
        ASSERT_EQ(controller.state(),MoEOverlayEconomyCalibrationState::Complete);
        EXPECT_TRUE(controller.healthy());EXPECT_EQ(controller.expectedAcceptedPairs(),4u);
        const auto stats=controller.stats();
        EXPECT_EQ(stats.waves_started,4u);EXPECT_EQ(stats.waves_completed,4u);
        EXPECT_EQ(stats.waves_aborted,4u);EXPECT_EQ(stats.accepted_pairs,4u);
        EXPECT_EQ(stats.evidence_exchanges_started,4u);EXPECT_EQ(stats.evidence_exchanges_completed,4u);
        EXPECT_EQ(stats.fatal_failures,0u);EXPECT_EQ(stats.reused_sealed_profiles,0u);
        EXPECT_EQ(stats.baseline_arms,0u);EXPECT_EQ(stats.concurrent_arms,0u);
        const auto ledger_stats=ledger->stats();
        EXPECT_EQ(ledger_stats.completed_waves_observed,4u);
        EXPECT_EQ(ledger_stats.migration_observations_seen,8u);
        EXPECT_EQ(ledger_stats.warmup_migration_observations,2u);
        EXPECT_EQ(ledger_stats.stored_migration_observations,6u);
        EXPECT_EQ(ledger_stats.rejected_observations,0u);EXPECT_FALSE(ledger->ready()); // Sealed ledgers admit no further samples.
        const auto* sealed=controller.sealedMeasurements();ASSERT_TRUE(sealed);
        ASSERT_TRUE(sealed->valid());ASSERT_EQ(sealed->rows.size(),2u);
        for(size_t index=0;index<sealed->rows.size();++index) {
            const auto& row=sealed->rows[index];EXPECT_TRUE(row.valid());
            EXPECT_EQ(row.coordinate,coordinates[index]);ASSERT_TRUE(row.geometry);
            EXPECT_TRUE(row.geometry->sameIdentity(*declarations[index].geometry));
            EXPECT_EQ(row.wave_sample_count,3u);EXPECT_GT(row.wave_wall_nanoseconds,0u);
            EXPECT_EQ(row.interference_sample_count,0u);EXPECT_EQ(row.inference_interference_nanoseconds,0u);
            ASSERT_EQ(row.pieces.size(),6u);
            for(const auto& piece:row.pieces) {
                EXPECT_TRUE(piece.valid());EXPECT_EQ(piece.quoted_payload_bytes,49152u);
                EXPECT_EQ(piece.measurement.bytes,49152u);
            }
        }
        EXPECT_THROW((void)sealed->wholeRows(),std::invalid_argument);
        std::string reuse_error;
        EXPECT_FALSE(ledger->matchesDeclarations(*sealed,&reuse_error)); // No setup-quoted presealed reuse claim.
        EXPECT_NE(reuse_error.find("setup-bound"),std::string::npos)<<reuse_error;
        EXPECT_EQ(local->stats().waves_started,4u);EXPECT_EQ(local->stats().transfer_operations_started,48u);
        EXPECT_EQ(local->stats().transfer_operations_completed,48u);EXPECT_EQ(local->stats().abort_cleanups_completed,4u);
        EXPECT_EQ(local->stats().commits_started,0u);EXPECT_EQ(distributed->stats().waves_published,0u);
        EXPECT_TRUE(consensus->idle());EXPECT_TRUE(evidence->idle());EXPECT_FALSE(journal->hasUnreadWave());
        EXPECT_EQ(journal->stats().waves_recorded,4u);EXPECT_EQ(journal->stats().waves_consumed,4u);
        EXPECT_EQ(evidence->stats().migration_profile_exchanges_completed,4u);
        EXPECT_EQ(evidence->stats().evidence_payload_collectives_started,4u);
        const uint64_t remote_bytes=4u*uint64_t(3)*kH*kF*sizeof(float)/2;
        EXPECT_EQ(original.remote->stats().bytes_sent,remote_bytes);
        EXPECT_EQ(original.remote->stats().bytes_received,remote_bytes);
        EXPECT_EQ(original.remote->stats().active_mpi_requests,0u);
        EXPECT_EQ(original.authority->snapshot(),initial);
        for(int id:original.registry->localParticipantIds())EXPECT_FALSE(original.registry->endpoint(id)->acquire(initial->epoch+1));
    }
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(initial->epoch);
        EXPECT_FALSE(endpoint->retiredEpochReaders(initial->epoch));
    }
    original.fabric.reset();original.registry.reset();original.remote.reset();original.context.reset();
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
} // namespace
/** @test Existing controller alone warms, measures, drains, exchanges and seals genuine CPU targets. */
TEST(MoEOverlayCPUJointCalibration, RealTwoSocketControllerSealsCompleteTargetMeasurements)
{
    try { proveCPUJointCalibrationController(); }
    catch(const std::exception& error) {ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}
} // namespace llaminar2::test
