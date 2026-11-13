/**
 * @file PhysicalMeasurementQuoteDrainContract.inc
 * @brief Actual composite progress rejects replayed sample corruption against retained command bytes.
 *
 * Existing scripted operations replay controller states only. They are not
 * device timing evidence or native transfer qualification.
 */
namespace llaminar2::test {
namespace {
/** @brief Retain a recorded command byte extent independently of the returned sample. */
class RecordedQuotedOperation final:public IMoEOverlayTierTransferOperation {
public:
    /** @param operation Original replay operation. @param bytes Recorded submitted command extent.
     * @param corrupt Change only observed bytes, preserving the original command quote. */
    RecordedQuotedOperation(std::unique_ptr<IMoEOverlayTierTransferOperation> operation,std::uint64_t bytes,bool corrupt,std::shared_ptr<int> lease={})
        : operation_(std::move(operation)),bytes_(bytes),corrupt_(corrupt),lease_(std::move(lease)) {}
    /** @brief Preserve original asynchronous progress. */
    MoEOverlayResidencyWaveProgress poll(std::string *error)noexcept override{return operation_->poll(error);}
    /** @brief Preserve original cancellation. */
    void abort()noexcept override{operation_->abort();}
    /** @brief Preserve original drain before retirement. */
    MoEOverlayResidencyWaveProgress pollAbort(std::string *error)noexcept override{return operation_->pollAbort(error);}
    /** @return Original recorded command quote, independent of observations. */
    std::optional<std::uint64_t> quotedPayloadBytes()const noexcept override{return bytes_;}
    /** @return Replayed event sample, optionally corrupting only its returned byte count. */
    std::optional<ExpertTierProjectionTransferMeasurement> completedMeasurement()const noexcept override {
        auto value=operation_->completedMeasurement();if(value&&corrupt_)++value->bytes;return value;
    }
private:
    std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
    std::uint64_t bytes_;bool corrupt_;
    /** Recorded owner witness: the operation must retain it through pending abort. */
    std::shared_ptr<int> lease_;
};
/** @brief Decorate the existing whole-operation replay with its exact sealed identity mapping. */
class RecordedQuotedFactory final:public IMoEOverlayTierPreparedWaveFactory {
public:
    explicit RecordedQuotedFactory(bool corrupt,bool unresolved_rank=false,bool timing=true):original(unresolved_rank?ScriptedWaveFactory::Mode::TransferFailureWithPendingAbort:ScriptedWaveFactory::Mode::Exact,timing),corrupt_(corrupt),unresolved_rank_(unresolved_rank){}
    MoEOverlayTierPreparedWave prepare(const MoEOverlayResidencyTransaction &transaction)override {
        auto wave=original.prepare(transaction);
        auto lease=std::make_shared<int>(1);lease_witness=lease;
        for(std::size_t migration=0;migration<transaction.migrations.size();++migration) {
            const auto &edge=transaction.migrations[migration];
            MoEOverlayPhysicalProjectionMember source,destination;
            source.participant_id=edge.source.owner_participant;destination.participant_id=edge.destination.owner_participant;
            source.device=destination.device=DeviceId::cpu();
            source.world_rank=destination.world_rank=unresolved_rank_?-1:0;
            const MoEExpertProjectionOwnership::Projection whole{MoEProjectionResidence::ExpertOwner,1,1,0,1,0,1};
            source.projections.fill(whole);destination.projections.fill(whole);
            const auto format=ExpertWeightFormat::floating(TensorType::FP32);
            wave.measurement_geometries.push_back(MoEOverlayMigrationMeasurementGeometry::checked(
                {MoEOverlayPhysicalProjectionTarget{edge.source.tier_idx,{source}},
                 MoEOverlayPhysicalProjectionTarget{edge.destination.tier_idx,{destination}}},{format,format,format}));
            for(std::size_t role=0;role<3;++role) {
                const auto index=migration*3+role;
                wave.measurement_operations.push_back({migration,role,true,unresolved_rank_?-1:0});
                wave.transfers[index]=std::make_unique<RecordedQuotedOperation>(std::move(wave.transfers[index]),1024+index,corrupt_,lease);
            }
        }
        return wave;
    }
    ScriptedWaveFactory original;
    std::weak_ptr<int> lease_witness;
private:bool corrupt_;bool unresolved_rank_;
};
}
TEST(MoEOverlayTierMigrationTransport, BoundQuoteRejectsWrongSamplesBeforePrepareAndDrains)
{
    for(bool corrupt:{false,true}) {
        auto fixture=makeAuthority();const auto transaction=fixture.authority->proposeFromHistogram();ASSERT_TRUE(transaction.valid());
        RecordedQuotedFactory factory(corrupt);
        auto sink=std::make_shared<RecordingMeasurementSink>(factory.original.observations);
        MoEOverlayTierMigrationTransport transport({.factory=&factory,.measurement_sink=sink,.require_complete_local_measurements=true});
        auto started=transport.beginStage(transaction);ASSERT_EQ(started.status,MoEOverlayResidencyStageStartStatus::Started)<<started.error;
        ASSERT_TRUE(started.wave);std::string error;
        EXPECT_EQ(started.wave->pollStage(&error),MoEOverlayResidencyWaveProgress::Pending);
        EXPECT_EQ(started.wave->pollStage(&error),corrupt?MoEOverlayResidencyWaveProgress::Failed:MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_EQ(factory.original.observations->measurement_sink_calls,corrupt?0u:1u);
        EXPECT_EQ(factory.original.observations->bank_begin_prepares,0u);
        if(corrupt)EXPECT_NE(error.find("quote"),std::string::npos);
        started.wave->abortStaged();
        auto progress=started.wave->pollAbort(&error);
        for(int poll=0;poll<4&&progress==MoEOverlayResidencyWaveProgress::Pending;++poll)progress=started.wave->pollAbort(&error);
        EXPECT_EQ(progress,MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_GT(factory.original.observations->transfer_aborts,0u);
        EXPECT_EQ(factory.original.observations->bank_begin_publications,0u);
    }
}
}

namespace llaminar2::test {
TEST(MoEOverlayTierMigrationTransport, InvalidMeasurementContextRetainsActualAbortDrain) {
    auto fixture=makeAuthority();const auto transaction=fixture.authority->proposeFromHistogram();ASSERT_TRUE(transaction.valid());
    RecordedQuotedFactory factory(false,true);
    auto sink=std::make_shared<RecordingMeasurementSink>(factory.original.observations);
    MoEOverlayTierMigrationTransport transport({.factory=&factory,.measurement_sink=sink,.require_complete_local_measurements=true});
    auto started=transport.beginStage(transaction);
    ASSERT_EQ(started.status,MoEOverlayResidencyStageStartStatus::Failed);EXPECT_FALSE(started.wave);
    EXPECT_EQ(transport.stats().waves_started,0u);EXPECT_EQ(transport.stats().waves_failed_to_prepare,1u);
    ASSERT_TRUE(started.cleanup_wave);EXPECT_FALSE(factory.lease_witness.expired());
    EXPECT_EQ(factory.original.observations->transfers_destroyed,0u);
    EXPECT_NE(started.error.find("observer ranks"),std::string::npos);
    EXPECT_EQ(factory.original.observations->measurement_sink_calls,0u);
    EXPECT_EQ(factory.original.observations->bank_begin_prepares,0u);
    started.cleanup_wave->abortStaged();std::string error;
    auto progress=started.cleanup_wave->pollAbort(&error);
    EXPECT_EQ(progress,MoEOverlayResidencyWaveProgress::Pending);EXPECT_FALSE(factory.lease_witness.expired());
    EXPECT_EQ(factory.original.observations->transfers_destroyed,0u);
    for(int poll=0;poll<4&&progress==MoEOverlayResidencyWaveProgress::Pending;++poll)progress=started.cleanup_wave->pollAbort(&error);
    EXPECT_EQ(progress,MoEOverlayResidencyWaveProgress::Ready);
    EXPECT_GT(factory.original.observations->transfer_aborts,0u);
    EXPECT_EQ(factory.original.observations->bank_begin_publications,0u);
    started.cleanup_wave.reset();EXPECT_TRUE(factory.lease_witness.expired());
    EXPECT_GT(factory.original.observations->transfers_destroyed,0u);
}
}

namespace llaminar2::test {
/** @test Recorded actual-progress replay emits each exact piece once with timing collection disabled. */
TEST(MoEOverlayTierMigrationTransport, TerminalPiecesRetainOriginalQuotesWithoutTimingSamples)
{
    ScopedMovementEvidence evidence;
    auto fixture = makeAuthority();
    const auto transaction = fixture.authority->proposeFromHistogram();
    ASSERT_TRUE(transaction.valid());
    RecordedQuotedFactory factory(false, false, false);
    MoEOverlayTierMigrationTransport transport({.factory = &factory});
    auto started = transport.beginStage(transaction);
    ASSERT_EQ(started.status, MoEOverlayResidencyStageStartStatus::Started);
    ASSERT_TRUE(started.wave);
    ASSERT_TRUE(started.measurement_context);
    std::string error;
    EXPECT_EQ(started.wave->pollStage(&error), MoEOverlayResidencyWaveProgress::Pending);
    for (const auto &record : PerfStatsCollector::snapshot({"moe_overlay_residency"}))
        EXPECT_NE(record.name, "physical_migration_pieces_completed");
    ASSERT_EQ(started.wave->pollStage(&error), MoEOverlayResidencyWaveProgress::Ready) << error;
    EXPECT_EQ(started.wave->pollStage(&error), MoEOverlayResidencyWaveProgress::Ready);
    std::size_t count = 0;
    uint64_t actual_bytes = 0;
    for (const auto &record : PerfStatsCollector::snapshot({"moe_overlay_residency"}))
    {
        if (record.name != "physical_migration_pieces_completed") continue;
        ++count;
        EXPECT_EQ(record.value, 1.0);
        EXPECT_EQ(record.tags.at("schema_version"), "1");
        EXPECT_EQ(record.tags.at("observer_world_rank"), "0");
        EXPECT_EQ(record.tags.at("source_world_rank"), "0");
        EXPECT_EQ(record.tags.at("destination_world_rank"), "0");
        EXPECT_EQ(record.tags.at("intersection_rows"), "1");
        EXPECT_EQ(record.tags.at("intersection_columns"), "1");
        const auto migration = std::stoull(record.tags.at("migration_index"));
        const auto piece = std::stoull(record.tags.at("piece_index"));
        const auto quote = 1024 + migration * 3 + piece;
        EXPECT_EQ(std::stoull(record.tags.at("payload_bytes")), quote);
        actual_bytes += quote;
    }
    EXPECT_EQ(count, transaction.migrations.size() * 3);
    EXPECT_EQ(transport.stats().placement_transfer_payload_bytes_completed, actual_bytes);
    EXPECT_EQ(factory.original.observations->measurement_sink_calls, 0u);
    EXPECT_EQ(factory.original.observations->bank_begin_publications, 0u);
    started.wave->abortStaged();
    EXPECT_EQ(started.wave->pollAbort(&error), MoEOverlayResidencyWaveProgress::Ready);
}
} // namespace llaminar2::test
