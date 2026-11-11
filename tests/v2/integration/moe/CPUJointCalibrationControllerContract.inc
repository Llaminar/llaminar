/**
 * @file CPUJointCalibrationControllerContract.inc
 * @brief Actual two-socket source/fabric/controller/distributed-evidence/ledger lifecycle.
 *
 * Original setup geometry is declared before any sample. The existing
 * controller alone chooses warmup/sample waves and owns every real staging,
 * abort/drain and evidence exchange. No fake-ready factory, source permit,
 * synthetic inference pair or policy loop supplies measurements. GPUwhole+CPU2
 * variants join actual prepared service and physical movement prices under the
 * same original source/PMA; calibration never publishes a candidate bank.
 */
#include "execution/moe/MoEOverlayEconomyCalibrationController.h"
#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEOverlayEconomyEvidenceCodec.h"
#include "execution/moe/MoEOverlayDistributedResidencyTransport.h"
#include "execution/moe/MoEOverlayMPIResidencyConsensus.h"
#include "execution/moe/MoEOverlayMPIEconomyEvidenceExchange.h"

namespace llaminar2::test {
namespace {
/**
 * @brief Join real original target service into the same existing evidence lane.
 * @param original Same prepared model/engines/cuts/PMA as the movement controller.
 * @param catalog Original source-bound layer equivalence authority.
 * @param topology Same retained phase declaration as the movement evidence.
 * @param evidence Existing exact-live MPI lane, idle before and after this call.
 * @return Complete source-bound target prices; no installation/publication permit.
 */
std::shared_ptr<const MoERoutedTierServiceProfile> measureJoinedGPUCPUService(
    FabricOriginal& original,std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog,
    const ExpertHistogramProductionTopology& topology,
    const std::shared_ptr<MoEOverlayMPIEconomyEvidenceExchange>& evidence)
{
    using Service=MoEOverlayCPUServiceMeasurement;
    const auto initial=original.authority->snapshot();
    const auto declaration=MoEOverlayPreparedTargetServicePlan::checked(initial,std::move(catalog),topology);
    if(declaration->entries().size()!=2 || !original.service_domain)
        throw std::logic_error("Joined calibration requires original GPUwhole and complete CPUcohort");
    bool saw_whole=false,saw_cohort=false;
    for(const auto& entry:declaration->entries()) {
        if(entry.coordinate.target.isWhole()) {
            if(entry.root_world_rank!=original.whole_owner)throw std::logic_error("Joined original GPU target root changed");
            saw_whole=true;
            EXPECT_EQ(entry.coordinate.target.word(),original.whole);
        } else {
            saw_cohort=true;
            const auto cohort=declaration->cohort(entry);
            const auto* owner=initial->placementOwnerMap().placementFor(0,2);
            if(!owner || !owner->shard_cohort)throw std::logic_error("Joined original expert lost cohort ownership");
            EXPECT_EQ(cohort,owner->shard_cohort);
            if(!cohort || cohort->participants().size()!=2)
                throw std::logic_error("Joined service lost complete original cohort");
            if(entry.root_world_rank!=cohort->participants().front().world_rank)
                throw std::logic_error("Joined original CPU target root changed");
            for(size_t member=0;member<2;++member) {
                EXPECT_EQ(cohort->participants()[member].world_rank,int(member));
                EXPECT_EQ(cohort->ownership(member).projection(WeightRole::MoEExpertGate).rows,kF/2);
                EXPECT_EQ(cohort->ownership(member).projection(WeightRole::MoEExpertDown).columns,kF/2);
            }
        }
    }
    if(!saw_whole || !saw_cohort)throw std::logic_error("Joined target vector aliased an original member");
    const int rank=original.mpi->rank();
    if(rank==original.whole_owner) {
        const auto endpoint=original.registry->endpoint(original.whole);
        const auto bank=endpoint?endpoint->acquire(initial->epoch):MoEOverlayParticipantBankLease{};
        if(!bank || bank->device!=original.whole_device || !original.service_native_source ||
            original.service_native_source->memory()!=original.memory ||
            original.context->nativeContextPreparation()!=original.service_native_source)
            throw std::logic_error("Joined GPU service lost original prepared bank/source/PMA");
    }
    const auto before=evidence->stats();
    const auto local=Service::measurePreparedTargets(*original.registry,*declaration,{kH,kF},
        original.service_domain,original.memory,&original.service_profile,&original.service_inventory);
    const size_t local_targets=std::count_if(declaration->entries().begin(),declaration->entries().end(),
        [&](const auto& entry){return entry.root_world_rank==rank;});
    EXPECT_EQ(local.size(),local_targets);
    declaration->requireLocalSamples(local,rank);
    const auto wire=MoEOverlayEconomyEvidenceCodec::encodePreparedService(*declaration,local,rank);
    const size_t local_bytes=MoEOverlayEconomyEvidenceCodec::preparedServiceBytes(*declaration,rank);
    size_t root_bytes=0;
    for(int original_rank=0;original_rank<original.mpi->world_size();++original_rank)
        root_bytes+=MoEOverlayEconomyEvidenceCodec::preparedServiceBytes(*declaration,original_rank);
    EXPECT_EQ(root_bytes,2u*40u*kExpertHistogramProductionSourceCount);
    EXPECT_EQ(wire.size(),local_bytes);
    std::string error;
    if(!evidence->beginPreparedService(declaration,local,&error))throw std::runtime_error(error);
    std::vector<MoEOverlayPreparedTargetServiceTotals> complete;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        const auto progress=evidence->pollPreparedService(&complete,&error);
        if(progress==Progress::Ready)break;
        if(progress!=Progress::Pending || std::chrono::steady_clock::now()>=deadline)
            throw std::runtime_error(error.empty()?"Joined service evidence exceeded original deadline":error);
        std::this_thread::yield();
    }
    if(complete.size()!=2)throw std::logic_error("Joined service omitted an original target");
    for(const auto& row:complete) {
        const auto& entry=declaration->entry(row.coordinate);
        for(size_t phase=0;phase<kExpertHistogramProductionSourceCount;++phase) {
            EXPECT_GT(row.total_nanoseconds[phase],0u);
            EXPECT_EQ(row.sample_count[phase],Service::kMeasuredSamples);
            EXPECT_EQ(row.activation_count[phase],entry.sample_rows[phase]*Service::kMeasuredSamples);
            EXPECT_EQ(row.expert_ids[phase],declaration->sampleExpert(entry,phase));
        }
        if(entry.root_world_rank==rank) {
            const auto found=std::find_if(local.begin(),local.end(),[&](const auto& measured){return measured.coordinate==row.coordinate;});
            if(found==local.end())throw std::logic_error("Joined root lost actual local measurement");
            EXPECT_EQ(row.total_nanoseconds,found->total_nanoseconds);
            EXPECT_EQ(row.activation_count,found->activation_count);
        }
    }
    const auto service=MoEOverlayEconomyProfileComposer::composePreparedService(declaration,complete);
    EXPECT_EQ(service->prepared_plan,declaration);
    EXPECT_EQ(service->target_costs.size(),2u);
    for(const auto& target:service->target_costs)
        for(const auto cost:target.nanoseconds_per_activation)EXPECT_GT(cost,0u);
    const auto after=evidence->stats();
    EXPECT_EQ(after.evidence_payload_bytes_sent-before.evidence_payload_bytes_sent,local_bytes);
    EXPECT_EQ(after.evidence_payload_bytes_received-before.evidence_payload_bytes_received,root_bytes);
    EXPECT_EQ(after.evidence_payload_collectives_started-before.evidence_payload_collectives_started,1u);
    EXPECT_TRUE(evidence->idle());
    EXPECT_EQ(original.authority->snapshot(),initial);
    return service;
}

/** @brief Drive the original complete-target calibration policy over real physical/MPI owners.
 * @param whole_device Original whole endpoint; CPU retains existing physical-only test.
 * @param joined_service Join native measured service for a disjoint whole GPU and CPU2.
 */
void proveCPUJointCalibrationController(DeviceId whole_device=DeviceId::cpu(),bool joined_service=false)
{
    FabricOriginal original(whole_device,joined_service);
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
        std::shared_ptr<const MoERoutedTierServiceProfile> service;
        if(joined_service) {
            try {
                service=measureJoinedGPUCPUService(original,
                    std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(original.manifests),
                    ExpertHistogramProductionTopology::uniform(1,kAllExpertHistogramProductionSources),evidence);
            } catch(const std::exception& failure) {
                // This also covers checked declarations before the producer's
                // own abort boundary. Preserve the reason before fatal wake.
                LOG_ERROR("[CPUJointCalibration] Original joined service/evidence failed rank="
                    << original.mpi->rank() << ": " << failure.what());
                original.service_domain->requestAbort();
                MPI_Abort(MPI_COMM_WORLD,2);
                throw;
            } catch(...) {
                LOG_ERROR("[CPUJointCalibration] Original joined service/evidence failed rank="
                    << original.mpi->rank() << ": unknown exception");
                // A nonroot may already be waiting in the original MPI evidence
                // lane after its cohort work. Abort while domain/source/evidence
                // owners still live, before stack unwinding can wait on a peer.
                original.service_domain->requestAbort();
                MPI_Abort(MPI_COMM_WORLD,2);
                throw;
            }
        }
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
        // Feed the actual sealed native corpus through production source-bound
        // expansion and the SAME typed Composer used by certification. No new
        // timing, readiness or physical observation is authored by this check.
        const MoEOverlayEconomyCalibrationLayerCatalog catalog(original.manifests);
        const auto expanded=catalog.expand(*sealed,initial);
        const auto costs=MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(expanded.rows);
        ASSERT_EQ(costs.size(),coordinates.size());
        for(size_t index=0;index<costs.size();++index) {
            EXPECT_EQ(costs[index].coordinate,coordinates[index]);
            EXPECT_GT(costs[index].transfer_and_repack_ns,0u);
            EXPECT_FALSE(expanded.rows[index].measured_coordinate);
            EXPECT_EQ(expanded.rows[index].geometry,sealed->rows[index].geometry);
        }

        if(joined_service) {
            // Identical completion shape to production prepared-target Composer:
            // real service rows plus the actual source-bound migration corpus.
            auto movement=std::make_shared<MoEOverlayMigrationCostProfile>();
            movement->identity=expanded.identity;
            movement->costs=costs;
            const MoEOverlayCertifiedEconomyProfiles profiles{service,movement,{}};
            EXPECT_TRUE(profiles.valid());
            EXPECT_EQ(profiles.service->prepared_plan->snapshot(),initial);
            EXPECT_EQ(profiles.migration->costs.size(),2u);
            EXPECT_EQ(profiles.migration->costs[0].coordinate.source_target.word(),
                profiles.migration->costs[1].coordinate.destination_target.word());
            EXPECT_EQ(profiles.migration->costs[0].coordinate.destination_target.word(),
                profiles.migration->costs[1].coordinate.source_target.word());
            EXPECT_FALSE(original.authority->migrationEnabled());
        }
        std::string reuse_error;
        EXPECT_FALSE(ledger->matchesDeclarations(*sealed,&reuse_error)); // No setup-quoted presealed reuse claim.
        EXPECT_NE(reuse_error.find("setup-bound"),std::string::npos)<<reuse_error;
        EXPECT_EQ(local->stats().waves_started,4u);EXPECT_EQ(local->stats().transfer_operations_started,48u);
        EXPECT_EQ(local->stats().transfer_operations_completed,48u);EXPECT_EQ(local->stats().abort_cleanups_completed,4u);
        EXPECT_EQ(local->stats().commits_started,0u);EXPECT_EQ(distributed->stats().waves_published,0u);
        EXPECT_TRUE(consensus->idle());EXPECT_TRUE(evidence->idle());EXPECT_FALSE(journal->hasUnreadWave());
        EXPECT_EQ(journal->stats().waves_recorded,4u);EXPECT_EQ(journal->stats().waves_consumed,4u);
        EXPECT_EQ(evidence->stats().migration_profile_exchanges_completed,4u);
        EXPECT_EQ(evidence->stats().evidence_payload_collectives_started,joined_service?5u:4u);
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
    if(original.service_domain) {
        EXPECT_TRUE(original.service_domain->retireOriginalCPUExpertDomain());
        original.service_domain.reset();
        EXPECT_EQ(original.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
    }
    original.fabric.reset();original.registry.reset();original.remote.reset();original.context.reset();
    if(original.service_native_source) {
        original.service_native_use.reset();
        EXPECT_EQ(original.service_native_source->retireReleasedContexts(),1u);
        original.service_native_source.reset();
    }
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    if(joined_service && original.mpi->rank()==original.whole_owner)
        EXPECT_EQ(original.memory->claimedBytes(whole_device,PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
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
#ifdef HAVE_CUDA
/** @test Native CUDA whole and original CPU2 service/movement form one complete measured profile. */
TEST(MoEOverlayGPUCPUJointCalibration, CUDAJoinedOriginalPreparedServiceAndMovement)
{
    try { proveCPUJointCalibrationController(DeviceId::cuda(0),true); }
    catch(const std::exception& error) {ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}
#endif
#ifdef HAVE_ROCM
/** @test One native ROCm whole owner uses no RCCL communicator for original CPU2 measurements. */
TEST(MoEOverlayGPUCPUJointCalibration, ROCmJoinedOriginalPreparedServiceAndMovement)
{
    try { proveCPUJointCalibrationController(DeviceId::rocm(0),true); }
    catch(const std::exception& error) {ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}
#endif
} // namespace llaminar2::test
