/**
 * @file CPUJointFabricMeasurementContract.inc
 * @brief Genuine two-socket public prepared-wave/composite/journal attribution.
 *
 * The original source-bound factory supplies all twelve actual operations.
 * A passive observer retains their checked mapping, never substitutes work,
 * source readiness, samples, or bank publication. The composite owns polling
 * and transfers its completed evidence into the original bounded journal.
 */
#include "execution/moe/MoEOverlayMigrationMeasurementExchange.h"

namespace llaminar2::test {
namespace {
/** @brief Observe the public factory's original metadata without copying operations. */
class ObservedCPUFabricFactory final : public IMoEOverlayTierPreparedWaveFactory {
public:
    /** @param original Genuine context-bound source, registry and physical fabric. */
    explicit ObservedCPUFabricFactory(FabricOriginal& original)
        : factory_({.registry=original.registry,.transfer_provider=original.fabric,
                    .perf_device="native-cpu-cohort-measurement"}) {}

    /**
     * @brief Delegate once and borrow only immutable checked geometry/mapping.
     * @param transaction Original authority-authored two-migration wave.
     * @return Unchanged real operations and inactive-bank transaction.
     */
    MoEOverlayTierPreparedWave prepare(const MoEOverlayResidencyTransaction& transaction) override {
        auto result=factory_.prepare(transaction);
        geometries=result.measurement_geometries;
        mapping=result.measurement_operations;
        operation_count=result.transfers.size();
        return result;
    }
    std::vector<std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry>> geometries;
    std::vector<MoEOverlayMigrationOperationMeasurement> mapping;
    size_t operation_count=0;
private:
    MoEOverlayParticipantPreparedWaveFactory factory_;
};

/** @brief Keep actual composite resources alive until successful asynchronous abort/drain. */
class OwnedCPUFabricMeasuredWave final {
public:
    /**
     * @brief Start through the public composite transport, retaining failed prefixes too.
     * @param transport Actual transport bound to public source-authenticated factory.
     * @param transaction Original authority-authored placement candidate.
     */
    OwnedCPUFabricMeasuredWave(MoEOverlayTierMigrationTransport& transport,
        const MoEOverlayResidencyTransaction& transaction) {
        auto start=transport.beginStage(transaction);
        wave=std::move(start.wave);
        if(start.status!=MoEOverlayResidencyStageStartStatus::Started || !wave) {
            wave=std::move(start.cleanup_wave);
            drain();
            throw std::runtime_error(start.error.empty()?"Public CPU cohort measured wave did not start":start.error);
        }
    }
    /** @brief Assertions cannot destroy a still-running original operation. */
    ~OwnedCPUFabricMeasuredWave() {if(!drained_)try{drain();}catch(...){std::terminate();}}
    /** @brief Consume the actual unpublished-wave cancellation and terminal cleanup boundary. */
    void drain() {
        if(drained_)return;
        if(!wave){drained_=true;return;}
        wave->abortStaged();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            std::string error;const auto progress=wave->pollAbort(&error);
            if(progress==Progress::Ready){drained_=true;return;}
            if(progress==Progress::Failed)throw std::runtime_error(error);
            if(std::chrono::steady_clock::now()>=deadline)
                throw std::runtime_error("CPU cohort measured wave abort exceeded original deadline");
            std::this_thread::yield();
        }
    }
    std::unique_ptr<IMoEOverlayResidencyWave> wave;
private:
    bool drained_=false;
};

/**
 * @brief Execute the actual public composite and inspect its owned logical journal.
 *
 * Staging stops before inactive-bank preparation/global publication. Two logical
 * migrations each have six canonical physical pieces; rank-zero observes all,
 * rank-one observes only its original member edges. No absent sample is invented.
 */
void proveCPUFabricMeasurementAttribution() {
    FabricOriginal original;
    const auto transaction=original.transaction();
    ASSERT_TRUE(transaction.valid());ASSERT_EQ(transaction.migrations.size(),2u);
    { // All factory/transport owners retire before final physical-byte assertions.
        ObservedCPUFabricFactory factory(original);
        auto journal=std::make_shared<MoEOverlayMigrationMeasurementJournal>(
            MoEOverlayMigrationMeasurementJournal::Config{.maximum_migrations_per_wave=2});
        MoEOverlayTierMigrationTransport transport({.factory=&factory,
            .transfer_progress_authority=original.fabric,.projections_per_expert=3,
            .measurement_sink=journal,.require_complete_local_measurements=true,
            .perf_device="native-cpu-cohort-measurement"});
        OwnedCPUFabricMeasuredWave owned(transport,transaction);
        const int rank=original.mpi->rank();
        try {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            for(;;) {
                std::string error;const auto progress=owned.wave->pollStage(&error);
                if(progress==Progress::Failed)throw std::runtime_error(error);
                int local_ready=progress==Progress::Ready,all_ready=0;
                if(MPI_Allreduce(&local_ready,&all_ready,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)!=MPI_SUCCESS)
                    throw std::runtime_error("CPU cohort measured stage progress vote failed");
                if(all_ready)break;
                if(std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("CPU cohort measured stage exceeded original deadline");
                std::this_thread::yield();
            }
        } catch(const std::exception& error) {
            ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;
        }
        ASSERT_EQ(factory.operation_count,12u);ASSERT_EQ(factory.mapping.size(),12u);
        ASSERT_EQ(factory.geometries.size(),2u);
        const auto& topology=transaction.previous->placementOwnerMap();
        size_t local_operations=0;
        for(size_t index=0;index<factory.mapping.size();++index) {
            const auto& mapping=factory.mapping[index];
            EXPECT_EQ(mapping.migration,index/6);EXPECT_EQ(mapping.piece,index%6);
            const auto& geometry=factory.geometries.at(mapping.migration);
            ASSERT_TRUE(geometry);ASSERT_EQ(geometry->pieces().size(),6u);
            const auto& edge=transaction.migrations.at(mapping.migration);
            const auto from=MoEExpertPlacementTarget::fromPlacement(edge.source);
            const auto to=MoEExpertPlacementTarget::fromPlacement(edge.destination);
            ASSERT_TRUE(from);ASSERT_TRUE(to);
            EXPECT_EQ(geometry->targets().first.tier_idx,from->tierIndex());
            EXPECT_EQ(geometry->targets().second.tier_idx,to->tierIndex());
            ASSERT_EQ(geometry->targets().first.members.size(),from->participantCount());
            ASSERT_EQ(geometry->targets().second.members.size(),to->participantCount());
            for(size_t member=0;member<from->participantCount();++member)
                EXPECT_EQ(geometry->targets().first.members[member].participant_id,from->participantId(member));
            for(size_t member=0;member<to->participantCount();++member)
                EXPECT_EQ(geometry->targets().second.members[member].participant_id,to->participantId(member));
            const auto& piece=geometry->pieces().at(mapping.piece);
            ASSERT_TRUE(piece.valid());ASSERT_TRUE(piece.intersection);
            const auto* source=topology.participantForId(piece.source_participant);
            const auto* destination=topology.participantForId(piece.destination_participant);
            ASSERT_TRUE(source);ASSERT_TRUE(destination);
            ASSERT_TRUE(source->world_rank_known);ASSERT_TRUE(destination->world_rank_known);
            const bool local=source->world_rank==rank || destination->world_rank==rank;
            EXPECT_EQ(mapping.requires_local_observation,local);
            local_operations+=local;
            EXPECT_EQ(unsigned(piece.projection),mapping.piece/2);
            const size_t member=mapping.piece%2;
            if(edge.source.shard_cohort)ASSERT_EQ(edge.source.shard_cohort->participants().size(),2u);
            if(edge.destination.shard_cohort)ASSERT_EQ(edge.destination.shard_cohort->participants().size(),2u);
            EXPECT_EQ(piece.source_participant,edge.source.shard_cohort?
                edge.source.shard_cohort->participants()[member].participant_id:edge.source.owner_participant);
            EXPECT_EQ(piece.destination_participant,edge.destination.shard_cohort?
                edge.destination.shard_cohort->participants()[member].participant_id:edge.destination.owner_participant);
            EXPECT_EQ(piece.format,ExpertWeightFormat::floating(TensorType::FP32));
            const auto& overlap=*piece.intersection;
            const bool down=piece.projection==ExpertTierWeightProjection::Down;
            EXPECT_EQ(overlap.rows(),down?kH:kF/2);
            EXPECT_EQ(overlap.columns(),down?kF/2:kH);
            EXPECT_EQ(overlap.firstRow(),down?0:int(mapping.piece%2)*kF/2);
            EXPECT_EQ(overlap.firstColumn(),down?int(mapping.piece%2)*kF/2:0);
            EXPECT_EQ(overlap.source().source_rows,down?kH:kF);
            EXPECT_EQ(overlap.source().source_columns,down?kF:kH);
            EXPECT_EQ(overlap.destination().source_rows,down?kH:kF);
            EXPECT_EQ(overlap.destination().source_columns,down?kF:kH);
            for(const auto& [projection,joint]:{std::pair{overlap.source(),bool(edge.source.shard_cohort)},
                                               std::pair{overlap.destination(),bool(edge.destination.shard_cohort)}}) {
                EXPECT_EQ(projection.rows,down?kH:(joint?kF/2:kF));
                EXPECT_EQ(projection.columns,down?(joint?kF/2:kF):kH);
                EXPECT_EQ(projection.first_row,!down && joint?int(member)*kF/2:0);
                EXPECT_EQ(projection.first_column,down && joint?int(member)*kF/2:0);
            }
        }
        EXPECT_EQ(local_operations,rank==0?12u:6u);
        ASSERT_TRUE(journal->hasUnreadWave());
        std::vector<MoEOverlayCompletedMigrationMeasurement> rows;
        std::string error;ASSERT_TRUE(journal->consume(&rows,&error))<<error;
        ASSERT_EQ(rows.size(),2u); // Logical rows, never index/3 over twelve pieces.
        size_t samples=0;
        for(size_t migration=0;migration<rows.size();++migration) {
            const auto& row=rows[migration];const auto& edge=transaction.migrations[migration];
            EXPECT_TRUE(row.valid());EXPECT_EQ(row.expected_epoch,transaction.expected_epoch);
            EXPECT_EQ(row.candidate_epoch,transaction.candidate->epoch);
            EXPECT_EQ(row.layer,edge.layer_idx);EXPECT_EQ(row.expert,edge.expert_id);
            EXPECT_EQ(row.source_participant,edge.source.owner_participant);
            EXPECT_EQ(row.destination_participant,edge.destination.owner_participant);
            EXPECT_EQ(row.geometry,factory.geometries[migration]);
            ASSERT_EQ(row.piece_observations.size(),6u);EXPECT_GT(row.wave_wall_nanoseconds,0u);
            for(const auto& legacy:row.projections)EXPECT_FALSE(legacy);
            for(size_t piece=0;piece<6;++piece) {
                const auto& observation=row.piece_observations[piece];
                const bool local=factory.mapping[migration*6+piece].requires_local_observation;
                EXPECT_TRUE(observation.valid());EXPECT_EQ(bool(observation.observation),local);
                if(local) {
                    ASSERT_TRUE(observation.quoted_payload_bytes);ASSERT_TRUE(observation.observation);
                    EXPECT_EQ(*observation.quoted_payload_bytes,49152u);
                    EXPECT_EQ(observation.observation->bytes,49152u);
                    EXPECT_TRUE(observation.observation->valid());++samples;
                } else EXPECT_FALSE(observation.quoted_payload_bytes);
            }
        }
        EXPECT_EQ(samples,local_operations);
        EXPECT_EQ(rows[0].wave_wall_nanoseconds,rows[1].wave_wall_nanoseconds);
        EXPECT_EQ(journal->stats().waves_recorded,1u);EXPECT_EQ(journal->stats().waves_consumed,1u);
        EXPECT_EQ(journal->stats().rejected_waves,0u);EXPECT_FALSE(journal->hasUnreadWave());
        for(int repeat=0;repeat<20;++repeat)EXPECT_EQ(owned.wave->pollStage(&error),Progress::Ready)<<error;
        EXPECT_EQ(journal->stats().waves_recorded,1u);
        EXPECT_EQ(transport.stats().transfer_operations_started,12u);
        EXPECT_EQ(transport.stats().transfer_operations_completed,12u);
        owned.drain();owned.wave.reset();
        EXPECT_EQ(transport.stats().abort_cleanups_completed,1u);
        EXPECT_EQ(original.remote->stats().active_mpi_requests,0u);
        EXPECT_EQ(original.authority->snapshot()->epoch,1u);
        for(int id:original.registry->localParticipantIds()) {
            auto endpoint=original.registry->endpoint(id);EXPECT_FALSE(endpoint->acquire(2));
            (void)endpoint->retire(1);EXPECT_FALSE(endpoint->retiredEpochReaders(1));
        }
    }
    original.fabric.reset();original.registry.reset();original.remote.reset();
    original.context.reset();
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
} // namespace
/** @test Actual public source factory/composite preserves all-piece attribution in two logical rows. */
TEST(MoEOverlayCPUCohortMeasurement, PublicCompositeOwnsTwoLogicalRowsForTwelvePhysicalPieces) {
    try { proveCPUFabricMeasurementAttribution(); }
    catch(const std::exception& error) {
        ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;
    }
}
} // namespace llaminar2::test
