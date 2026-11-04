/**
 * @file CPUJointShadowCalibrationContract.inc
 * @brief Genuine two-socket complete-target shadow calibration through public owners.
 *
 * The planner issues EconomyCalibration directly against original admitted
 * targets. Real Fabric operations complete twelve original intersections into
 * bounded shadow roots, and the composite/journal owns every observation.
 * Neither a placement relabel nor candidate publication substitutes for work.
 */
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"

namespace llaminar2::test {
namespace {
/** @brief Execute one genuine source-authenticated, unpublishable calibration wave. */
void proveCPUJointShadowCalibration()
{
    FabricOriginal original;
    const auto initial=original.authority->snapshot();
    const auto footprint=MoEOverlayCapacityResolver::preparedFootprints(original.manifests).front();
    MoEOverlayEconomyCalibrationPlanner planner({.live_snapshot=initial,
        .complete_expert_bytes_per_layer={footprint.cpu_live_bytes},.calibration_layers={0}});
    const auto& coordinates=planner.requiredTargetCoordinates();
    ASSERT_EQ(coordinates.size(),2u);
    const auto& coordinate=coordinates.front();
    const auto transaction=planner.buildPairSwap(coordinate.source_target,
        coordinate.destination_target,coordinate.layer,17);
    ASSERT_TRUE(transaction.valid());ASSERT_EQ(transaction.migrations.size(),2u);
    ASSERT_EQ(transaction.purpose,MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
    ASSERT_EQ(transaction.calibration_sequence,17u);
    ASSERT_EQ(transaction.previous,initial);
    ASSERT_EQ(&transaction.candidate->placementOwnerMap(),&initial->placementOwnerMap());
    ASSERT_EQ(transaction.candidate->placement_plan,initial->placement_plan);
    ASSERT_EQ(transaction.candidate->originalProjectionPreparation(),original.preparation);
    ASSERT_EQ(transaction.candidate->epoch,initial->epoch+1);
    EXPECT_EQ(transaction.histogram_generation,0u);EXPECT_FALSE(transaction.histogram_window);

    // The normal candidate-placement producer must still reject an incoming
    // expert that calibration deliberately does not place in its live map.
    for(size_t index=0;index<transaction.migrations.size();++index) {
        const auto& edge=transaction.migrations[index];
        if(!edge.destination.shard_cohort)continue;
        for(int id:original.registry->localParticipantIds()) {
            const auto members=edge.destination.shard_cohort->participants();
            if(std::none_of(members.begin(),members.end(),[id](const auto& member){return member.participant_id==id;}))continue;
            std::string error;
            EXPECT_FALSE(original.registry->prepareCPUJointArrival(*transaction.candidate,id,
                edge.layer_idx,edge.expert_id,&error));
            auto arrival=original.registry->prepareCPUJointCalibrationArrival(transaction,index,id,&error);
            ASSERT_TRUE(arrival)<<error;
            MoEOverlayPreparedCPUFragmentPayload incomplete;
            EXPECT_FALSE(arrival->completeCPUFragment(incomplete,&error));
            EXPECT_FALSE(original.registry->prepareCPUJointCalibrationArrival(transaction,index,original.whole,&error));
            auto relabelled=transaction;
            relabelled.purpose=MoEOverlayResidencyTransactionPurpose::PlacementChange;
            EXPECT_FALSE(relabelled.valid());
            EXPECT_FALSE(original.registry->prepareCPUJointCalibrationArrival(relabelled,index,id,&error));
        }
    }
    std::string error;
    EXPECT_FALSE(original.registry->prepareCPUJointCalibrationArrival(transaction,
        transaction.migrations.size(),original.last_shard,&error));
    {
        ObservedCPUFabricFactory factory(original);
        auto journal=std::make_shared<MoEOverlayMigrationMeasurementJournal>(
            MoEOverlayMigrationMeasurementJournal::Config{.maximum_migrations_per_wave=2});
        MoEOverlayTierMigrationTransport transport({.factory=&factory,.transfer_progress_authority=original.fabric,
            .projections_per_expert=3,.measurement_sink=journal,.require_complete_local_measurements=true,
            .perf_device="native-cpu-joint-shadow-calibration"});
        OwnedCPUFabricMeasuredWave owned(transport,transaction);
        try {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            for(;;) {
                const auto progress=owned.wave->pollStage(&error);
                if(progress==Progress::Failed)throw std::runtime_error(error);
                int local_ready=progress==Progress::Ready,all_ready=0;
                if(MPI_Allreduce(&local_ready,&all_ready,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)!=MPI_SUCCESS)
                    throw std::runtime_error("CPU joint shadow calibration progress vote failed");
                if(all_ready)break;
                if(std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("CPU joint shadow calibration exceeded original deadline");
                std::this_thread::yield();
            }
        } catch(const std::exception& failure) {
            ADD_FAILURE()<<failure.what();MPI_Abort(MPI_COMM_WORLD,2);throw;
        }
        ASSERT_EQ(factory.operation_count,12u);ASSERT_EQ(factory.mapping.size(),12u);
        ASSERT_EQ(factory.geometries.size(),2u);
        std::vector<MoEOverlayCompletedMigrationMeasurement> rows;
        ASSERT_TRUE(journal->consume(&rows,&error))<<error;ASSERT_EQ(rows.size(),2u);
        size_t samples=0;
        for(size_t migration=0;migration<2;++migration) {
            const auto& row=rows[migration];EXPECT_TRUE(row.valid());
            EXPECT_EQ(row.geometry,factory.geometries[migration]);
            EXPECT_EQ(row.expert,transaction.migrations[migration].expert_id);
            ASSERT_EQ(row.piece_observations.size(),6u);
            for(size_t piece=0;piece<6;++piece) {
                const auto& mapping=factory.mapping[migration*6+piece];
                EXPECT_EQ(mapping.migration,migration);EXPECT_EQ(mapping.piece,piece);
                const auto& sample=row.piece_observations[piece];
                EXPECT_EQ(bool(sample.observation),mapping.requires_local_observation);
                if(mapping.requires_local_observation) {
                    ASSERT_TRUE(sample.quoted_payload_bytes);ASSERT_TRUE(sample.observation);
                    EXPECT_EQ(*sample.quoted_payload_bytes,49152u);
                    EXPECT_EQ(sample.observation->bytes,49152u);EXPECT_TRUE(sample.observation->valid());++samples;
                } else EXPECT_FALSE(sample.quoted_payload_bytes);
            }
        }
        EXPECT_EQ(samples,original.mpi->rank()==0?12u:6u);
        EXPECT_FALSE(owned.wave->beginPrepare(&error));
        EXPECT_NE(error.find("calibration"),std::string::npos);
        EXPECT_FALSE(owned.wave->beginPublication(&error));
        EXPECT_EQ(transport.stats().commits_started,0u);
        EXPECT_EQ(journal->stats().waves_recorded,1u);
        EXPECT_EQ(original.authority->snapshot(),initial);
        const uint64_t bytes=uint64_t(3)*kH*kF*sizeof(float)/2;
        EXPECT_EQ(original.remote->stats().bytes_sent,bytes);
        EXPECT_EQ(original.remote->stats().bytes_received,bytes);
        owned.drain();owned.wave.reset();
        EXPECT_EQ(original.remote->stats().active_mpi_requests,0u);
        EXPECT_EQ(transport.stats().abort_cleanups_completed,1u);
    }
    // A second genuine sample uses the same production Fabric and original
    // reusable slots after public composite cleanup. This supplemental physical
    // byte oracle does not replace the public factory/journal assertions above.
    const auto byte_transaction=planner.buildPairSwap(coordinate.source_target,
        coordinate.destination_target,coordinate.layer,18);
    {
        ComponentWave bytes(original,byte_transaction);
        pollComponent(bytes);
        for(size_t index=0;index<bytes.prepared.migrations.size();++index)
            for(const auto& destination:bytes.prepared.migrations[index].destination_arrivals)
                expectDestination(original,byte_transaction,index,destination);
        bytes.abortAndDrain();
        expectStableTerminalQueries(bytes,*original.remote,ComponentTerminalQuery::AbortDrained);
    }
    // Source model expiry forbids new arrivals, while actual installed bank
    // engines remain independently owned until normal RCU retirement.
    original.context.reset();
    for(size_t index=0;index<transaction.migrations.size();++index) {
        if(!transaction.migrations[index].destination.shard_cohort)continue;
        const auto members=transaction.migrations[index].destination.shard_cohort->participants();
        for(int id:original.registry->localParticipantIds()) {
            if(std::none_of(members.begin(),members.end(),[id](const auto& member){return member.participant_id==id;}))continue;
            EXPECT_FALSE(original.registry->prepareCPUJointCalibrationArrival(transaction,index,id,&error));
            EXPECT_NE(error.find("expired"),std::string::npos)<<error;
        }
    }
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);
        EXPECT_TRUE(endpoint->acquire(initial->epoch));EXPECT_FALSE(endpoint->acquire(transaction.candidate->epoch));
        (void)endpoint->retire(initial->epoch);EXPECT_FALSE(endpoint->retiredEpochReaders(initial->epoch));
    }
    original.fabric.reset();original.registry.reset();original.remote.reset();
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.authority->snapshot(),initial);
}
} // namespace
/** @test Genuine complete CPU targets calibrate only bounded physical shadows and always drain. */
TEST(MoEOverlayCPUJointCalibration, RealTwoSocketShadowWaveNeverPublishesPlacement)
{
    try { proveCPUJointShadowCalibration(); }
    catch(const std::exception& error) {ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}
} // namespace llaminar2::test
