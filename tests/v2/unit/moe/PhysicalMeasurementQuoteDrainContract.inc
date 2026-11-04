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
    RecordedQuotedOperation(std::unique_ptr<IMoEOverlayTierTransferOperation> operation,std::uint64_t bytes,bool corrupt)
        : operation_(std::move(operation)),bytes_(bytes),corrupt_(corrupt) {}
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
};
/** @brief Decorate the existing whole-operation replay with its exact sealed identity mapping. */
class RecordedQuotedFactory final:public IMoEOverlayTierPreparedWaveFactory {
public:
    explicit RecordedQuotedFactory(bool corrupt):original(ScriptedWaveFactory::Mode::Exact,true),corrupt_(corrupt){}
    MoEOverlayTierPreparedWave prepare(const MoEOverlayResidencyTransaction &transaction)override {
        auto wave=original.prepare(transaction);
        for(std::size_t migration=0;migration<transaction.migrations.size();++migration) {
            const auto &edge=transaction.migrations[migration];
            MoEOverlayPhysicalProjectionMember source,destination;
            source.participant_id=edge.source.owner_participant;destination.participant_id=edge.destination.owner_participant;
            source.device=destination.device=DeviceId::cpu();
            const MoEExpertProjectionOwnership::Projection whole{MoEProjectionResidence::ExpertOwner,1,1,0,1,0,1};
            source.projections.fill(whole);destination.projections.fill(whole);
            const auto format=ExpertWeightFormat::floating(TensorType::FP32);
            wave.measurement_geometries.push_back(MoEOverlayMigrationMeasurementGeometry::checked(
                {MoEOverlayPhysicalProjectionTarget{edge.source.tier_idx,{source}},
                 MoEOverlayPhysicalProjectionTarget{edge.destination.tier_idx,{destination}}},{format,format,format}));
            for(std::size_t role=0;role<3;++role) {
                const auto index=migration*3+role;
                wave.measurement_operations.push_back({migration,role,true});
                wave.transfers[index]=std::make_unique<RecordedQuotedOperation>(std::move(wave.transfers[index]),1024+index,corrupt_);
            }
        }
        return wave;
    }
    ScriptedWaveFactory original;
private:bool corrupt_;
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
