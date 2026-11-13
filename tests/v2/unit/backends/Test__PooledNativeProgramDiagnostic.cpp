/**
 * @file Test__PooledNativeProgramDiagnostic.cpp
 * @brief Actual typed raw/certified boundary and SDK-prefix cleanup without hardware.
 */
#include "fixtures/backends/PooledNativeProgramDiagnosticAccess.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "backends/GPUGraphMemoryContract.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "execution/runner/ModelContextRetirement.h"
#include "mocks/MockBackend.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <type_traits>
namespace llaminar2::test {
std::shared_ptr<RetainedParticipantGraphExecutable> NativeContextPreparationReplayAccess::resumePreparedCalibrationLookup(
    const std::shared_ptr<PreparedServiceCalibrationSlot>& slot,const PreparedGPUContextUsePermit& original,void* stream){
    return slot->acquire(original,stream);
}
static_assert(!std::is_copy_constructible_v<PreparedServiceCalibrationSlot>);
static_assert(!std::is_move_constructible_v<PreparedServiceCalibrationSlot>);
static_assert(!std::is_copy_assignable_v<PreparedServiceCalibrationSlot>);
static_assert(!std::is_move_assignable_v<PreparedServiceCalibrationSlot>);
struct RetainedParticipantGraphExecutableReplayAccess {
    static auto create(const PreparedGPUContextUsePermit& original,void* stream){return RetainedParticipantGraphExecutable::createDeviceBound(original,stream);}
    /** @return Actual closed calibration-purpose factory, with no invented source permit. */
    static auto calibration(const PreparedGPUContextUsePermit& original,void* stream){return RetainedParticipantGraphExecutable::createPreparedCalibration(original,stream);}
    /** @brief Exercise actual cold reset; it cannot alter ordinary participant purposes. */
    static void resetCalibration(RetainedParticipantGraphExecutable& owner){owner.resetPreparedCalibration();}
    /** @brief Restore only recorded stale-generation test input before final original teardown. */
    static void restoreGeneration(RetainedParticipantGraphExecutable& owner){--owner.generation_;}
    static IGPUGraphCapture& recording(RetainedParticipantGraphExecutable& owner){return owner.capture();}
    static bool submit(RetainedParticipantGraphExecutable& owner,std::uint64_t ordinal){return owner.submit(ordinal);}
    static void ready(RetainedParticipantGraphExecutable& owner){owner.worker_->submitAndWait([&]{auto& recording=owner.capture();
        if(!recording.beginCapture() || !recording.endCapture() || !recording.instantiate())throw std::runtime_error("Recorded original graph did not seal");});}
    static bool observe(RetainedParticipantGraphExecutable& owner){return owner.observeTerminal();}
    static void staleGeneration(RetainedParticipantGraphExecutable& owner){++owner.generation_;}
};
namespace {
template<class T>concept PublicRaw=requires(const PreparedGPUContextUsePermit& permit){T::materializeDiagnostic(permit);typename T::UnqualifiedDiagnostic;};
static_assert(!PublicRaw<ReadAheadNativeResourceFamily> && !PublicRaw<ReadAheadNativeParentTerminal>);
/** @brief Return/exception injection surrounds actual facade calls, never its state machine. */
class FaultCapture final:public IGPUGraphCapture {
public:
    FaultCapture(void* stream,int& fault,int& calls,std::optional<std::size_t>& nodes,bool& flatKinds,int& inspections,std::function<void()> destroyed)
        :inner_(stream,std::move(destroyed)),fault_(fault),calls_(calls),nodes_(nodes),flatKinds_(flatKinds),inspections_(inspections){}
    bool beginCapture()override{check(1);return inner_.beginCapture();}
    bool endCapture()override{check(2);return inner_.endCapture();}
    bool instantiate()override{check(3);if(fault_==5)return true;return inner_.instantiate();}
    bool prepareRuntimeContextStorage(const std::shared_ptr<PhysicalMemoryAuthority>&)override{check(4);return fault_!=6;}
    bool launch()override{return inner_.launch();}
    void* executionStream()const noexcept override{return inner_.executionStream();}
    GraphUpdateResult tryUpdate()override{return inner_.tryUpdate();}
    bool supportsExecutableUpdate()const noexcept override{return false;}
    bool hasExecutable()const override{return inner_.hasExecutable();}
    std::size_t residentMemoryBytes()const noexcept override{return 0;}
    std::size_t nodeCount()const override{return nodes_.value_or(inner_.nodeCount());}
    void reset()override{check(7);inner_.reset();}
    const char* backendName()const override{return "recorded-fault";}
protected:
    /** @brief Only native SDK kind inspection is doubled; class/count policy stays production-owned. */
    bool validateFlatHelperNodeKinds(std::string* error)const override{
        ++inspections_;
        if(!flatKinds_ && error)*error="recorded native graph contains a forbidden conditional node";
        return flatKinds_;
    }
private:
    void check(int step){++calls_;if(fault_==step)throw std::runtime_error("original native recording failure");}
    testing::MockGPUGraphCapture inner_;int& fault_;int& calls_;
    std::optional<std::size_t>& nodes_;bool& flatKinds_;int& inspections_;
};
/** @brief Recorded creator return codes; no native runtime or worker thread is started. */
class ProgramWorker final:public testing::MockWorkerGPUContext {
public:
    ProgramWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
    int calls=0,fail=0,streamsDestroyed=0,eventsDestroyed=0,capturesDestroyed=0;
    int timingEventsCreated=0;
    int queueClosures=0;
    int graphFault=0,graphCalls=0;
    std::optional<std::size_t> graphNodes;bool graphFlatKinds=true;int graphKindInspections=0;
    bool ready=true,failSubmit=false;GPUStreamExecutionState streamState=GPUStreamExecutionState::Complete;
    void submitAndWait(std::function<void()> work)override{if(failSubmit)throw std::runtime_error("recorded enqueue failure before native creator");++queueClosures;MockWorkerGPUContext::submitAndWait(std::move(work));}
    void* createStream()override{if(++calls==fail)return nullptr;return MockWorkerGPUContext::createStream();}
    void* createEvent(GPUEventPurpose purpose)override{if(++calls==fail)return nullptr;if(purpose==GPUEventPurpose::Timing)++timingEventsCreated;return MockWorkerGPUContext::createEvent(purpose);}
    void destroyStream(void* value)override{++streamsDestroyed;MockWorkerGPUContext::destroyStream(value);}
    void destroyEvent(void* value)override{++eventsDestroyed;MockWorkerGPUContext::destroyEvent(value);}
    bool queryEventChecked(void* value,bool& observed)override{observed=ready;return value!=nullptr;}
    GPUStreamExecutionState queryStreamExecutionState(void*,std::string_view)override{return streamState;}
    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void* stream)override{
        if(++calls==fail)return nullptr;
        return std::make_unique<FaultCapture>(stream,graphFault,graphCalls,graphNodes,graphFlatKinds,graphKindInspections,[this]{++capturesDestroyed;});
    }
};
/** @brief Own actual original mapped object/pool/PMA and canonical recorded reset. */
struct Source {
    std::shared_ptr<NativeContextPreparation> source;
    std::optional<PreparedGPUContextUsePermit> original;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> certificate;
    Source(DeviceId device,std::size_t graphs=0,bool admit=true,std::size_t gpuCapacity=64u<<20,bool originalGeneral=false,std::size_t calibrationSlots=0){
        NativeContextPreparationReplayAccess::seed(device,[device](auto stack){return NativeContextPreparationReplayAccess::construct<ProgramWorker>(std::move(stack),device.ordinal);});
        if(originalGeneral){
            const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),4ULL<<30,4ULL<<30},PhysicalMemoryResource{0,device,gpuCapacity,gpuCapacity}};
            source=NativeContextPreparationReplayAccess::source<ProgramWorker>(device,observed);
        }else source=NativeContextPreparationReplayAccess::source<ProgramWorker>(device,gpuCapacity);
        original.emplace(source->acquire(device));memory=source->memory();
        source->freezeForPlanning();
        PhysicalMemoryPlanBuilder extra(source->mandatoryBOM());
        if(graphs){const auto& observed=source->originalResource(device);
            if(originalGeneral){const auto general=RetainedParticipantGraphExecutable::generalBOM(source->originalResource(DeviceId::cpu()),observed,graphs);for(const auto& row:general.resources())extra.add(row);}
            else extra.add(observed,PhysicalMemoryOwner::NativeGraphExecutable,
                graphs*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));}
        for(std::size_t slot=0;slot<calibrationSlots;++slot){
            const auto calibration=RetainedParticipantGraphExecutable::calibrationBOM(
                source->originalResource(DeviceId::cpu()),source->originalResource(device));
            for(const auto& row:calibration.resources())extra.add(row);
        }
        certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(extra.build());
        if(admit)source->admitFinal(certificate);
    }
    ProgramWorker& worker(){return static_cast<ProgramWorker&>(*original->worker());}
    void reset(){const auto device=original->device();original.reset();auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextPreparationReplayAccess::physicalReset(scope);}
};
}
TEST(PooledNativeProgramDiagnostic, RawFamilyParentHaveNoInventedCommitmentAndRetireActualTerminal){
    for(const auto device:{DeviceId::cuda(210),DeviceId::rocm(210)}){
        Source source(device);auto& worker=source.worker();
        auto parent=PooledNativeProgramDiagnosticAccess::parent(*source.original);
        auto family=PooledNativeProgramDiagnosticAccess::side(*source.original,worker.defaultStream());
        EXPECT_EQ(parent->ownershipKind(),NativeAuxiliaryOwnershipKind::UnqualifiedDiagnostic);
        EXPECT_EQ(family->ownershipKind(),NativeAuxiliaryOwnershipKind::UnqualifiedDiagnostic);
        EXPECT_NE(family->captureIdentity(),0u);
        EXPECT_EQ(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),0u);
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
        family->bindTerminal(parent->eventOwner());
        for(int epoch=0;epoch<3;++epoch){family->beginSubmission();EXPECT_THROW(family->beginSubmission(),std::logic_error);
            worker.ready=false;family->recordTerminal();EXPECT_FALSE(family->observeTerminal());
            EXPECT_THROW(family->beginSubmission(),std::logic_error);worker.ready=true;EXPECT_TRUE(family->observeTerminal());}
        family.reset();EXPECT_EQ(worker.eventsDestroyed,2);EXPECT_EQ(worker.streamsDestroyed,1);
        parent.reset();EXPECT_EQ(worker.eventsDestroyed,3);source.reset();
    }
}
TEST(PooledNativeProgramDiagnostic, ActualCreatorFailureReturnsDestroyOnlySuccessfulPrefix){
    for(const auto device:{DeviceId::cuda(211),DeviceId::rocm(211)}){
        Source source(device);auto& worker=source.worker();
        for(int failure:{1,2,3}){worker.calls=worker.eventsDestroyed=worker.streamsDestroyed=0;worker.fail=failure;
            EXPECT_THROW(PooledNativeProgramDiagnosticAccess::side(*source.original,worker.defaultStream()),std::runtime_error);
            EXPECT_EQ(worker.streamsDestroyed,failure>1?1:0);EXPECT_EQ(worker.eventsDestroyed,failure>2?1:0);}
        worker.calls=0;worker.fail=1;EXPECT_THROW(PooledNativeProgramDiagnosticAccess::parent(*source.original),std::runtime_error);
        worker.fail=0;source.reset();
    }
}
TEST(PooledNativeProgramDiagnostic, InvalidPrimaryAndPreparingSourceRejectBeforeAnyCreator){
    Source source(DeviceId::cuda(212));auto& worker=source.worker();
    EXPECT_THROW(PooledNativeProgramDiagnosticAccess::side(*source.original,nullptr),std::invalid_argument);
    EXPECT_THROW(PooledNativeProgramDiagnosticAccess::side(*source.original,reinterpret_cast<void*>(0x1234)),std::invalid_argument);
    EXPECT_EQ(worker.calls,0);source.reset();
    Source preparing(DeviceId::cuda(216),0,false);
    EXPECT_THROW(PooledNativeProgramDiagnosticAccess::side(*preparing.original,preparing.worker().defaultStream()),std::invalid_argument);
    EXPECT_EQ(preparing.worker().calls,0);
    preparing.source->admitFinal(preparing.certificate);preparing.reset();
}
/** @test Public original General admission uses planned slots, rejects unplanned creation and preserves process retirement. */
TEST(PooledNativeProgramDiagnostic, GeneralOriginalAllowanceEnforcesPlannedSlotsAndZeroAttemptCleanup){
    for(const auto device:{DeviceId::cuda(213),DeviceId::rocm(213)}){
        EXPECT_EXIT({
            Source source(device,1,true,64u<<20,true);auto& worker=source.worker();
            worker.failSubmit=true;
            try{(void)RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream());std::_Exit(2);}catch(const std::runtime_error&){}
            worker.failSubmit=false;
            if(worker.calls || source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable) || source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable))std::_Exit(3);
            auto owner=RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream());
            if(owner->hostQualification()!=GraphHostQualification::GeneralProcessAllowance || worker.calls!=1)std::_Exit(4);
            const auto calls=worker.calls;
            try{(void)RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream());std::_Exit(5);}catch(const std::exception&){}
            if(worker.calls!=calls)std::_Exit(6); // PMA capacity rejects before the unplanned native creator.
            RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
            if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1) ||
               !RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(7);
            owner.reset();if(worker.capturesDestroyed!=1)std::_Exit(8);
            const auto allowance=GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General);
            if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=allowance)std::_Exit(9);
            const auto hostAllowance=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(device);
            if(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable)!=hostAllowance)std::_Exit(10);
            source.reset();
            if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=allowance ||
               source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable)!=hostAllowance)std::_Exit(11);
            std::_Exit(0);
        },::testing::ExitedWithCode(0),"");
    }
}
/** @test A device-only declaration cannot create a public General host owner; arithmetic and resource identities stay checked. */
TEST(PooledNativeProgramDiagnostic,GeneralHostBOMRejectsMissingForeignAndOverflowBeforeNative){
    Source source(DeviceId::cuda(220),1);auto& worker=source.worker();
    EXPECT_THROW((void)RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream()),std::exception);
    EXPECT_EQ(worker.calls,0);EXPECT_EQ(source.memory->reservedBytes(DeviceId::cuda(220),PhysicalMemoryOwner::NativeGraphExecutable),0u);
    const PhysicalMemoryResource host{0,DeviceId::cpu(),4ULL<<30,4ULL<<30},gpu{0,DeviceId::cuda(220),64u<<20,64u<<20};
    const auto one=RetainedParticipantGraphExecutable::generalBOM(host,gpu,1),two=RetainedParticipantGraphExecutable::generalBOM(host,gpu,2);
    EXPECT_EQ(two.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),2*one.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable));
    EXPECT_EQ(two.find({0,gpu.device})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),2*GPUGraphMemoryContract::reservationBytesPerExecutable(gpu.device));
    EXPECT_THROW((void)RetainedParticipantGraphExecutable::generalBOM(host,gpu,std::numeric_limits<std::size_t>::max()),std::overflow_error);
    EXPECT_THROW((void)RetainedParticipantGraphExecutable::generalBOM(gpu,host),std::invalid_argument);
    const auto none=RetainedParticipantGraphExecutable::generalBOM(host,gpu,0);EXPECT_TRUE(none.resources().empty());source.reset();
}
/** @test A native creator failure retains attempted host/device backing through the same original physical reset. */
TEST(PooledNativeProgramDiagnostic,GeneralAttemptedHostAndDeviceBackingSurviveOriginalReset){
    for(const auto device:{DeviceId::cuda(219),DeviceId::rocm(219)}){
        EXPECT_EXIT({
            Source source(device,1,true,64u<<20,true);auto& worker=source.worker();worker.fail=1;
            try{(void)RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream());std::_Exit(2);}catch(const std::runtime_error&){}
            if(worker.calls!=1 || worker.capturesDestroyed)std::_Exit(3);
            const auto host=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(device);
            const auto gpu=GPUGraphMemoryContract::reservationBytesPerExecutable(device);
            if(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable)!=host ||
               source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=gpu)std::_Exit(4);
            source.reset();
            if(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable)!=host ||
               source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=gpu)std::_Exit(5);
            std::_Exit(0);
        },::testing::ExitedWithCode(0),"");
    }
}
TEST(PooledNativeProgramDiagnostic, GeneralAttemptedBackingRemainsProcessRetainedAcrossActualReset){
    EXPECT_EXIT({
        Source source(DeviceId::cuda(214),2);auto& worker=source.worker();const auto device=source.original->device();
        worker.fail=1;
        try{(void)RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());std::_Exit(2);}catch(const std::runtime_error&){}
        worker.fail=0;
        auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
        RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
        if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1))std::_Exit(8);
        worker.streamState=GPUStreamExecutionState::Pending;
        if(RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(3);
        worker.streamState=GPUStreamExecutionState::Complete;
        if(!RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(4);
        owner.reset();if(worker.capturesDestroyed!=1)std::_Exit(5);
        const auto expected=2*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General);
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=expected)std::_Exit(6);
        source.reset();
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=expected)std::_Exit(7);
        std::_Exit(0); // OS reclamation is the physical lifetime; no fabricated reset release.
    },::testing::ExitedWithCode(0),"");
}
TEST(PooledNativeProgramDiagnostic, GeneralLiveReadersRejectTeardownBeforeActualCaptureDestruction){
    EXPECT_EXIT({
        Source source(DeviceId::cuda(215),1);auto& worker=source.worker();
        auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
        RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
        if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1))std::_Exit(8);
        static ProgramWorker* observed=nullptr;observed=&worker;
        std::set_terminate([]{std::_Exit(observed->capturesDestroyed==0?77:78);});
        owner.reset();std::_Exit(2);
    },::testing::ExitedWithCode(77),"");
}

TEST(PooledNativeProgramDiagnostic, RecordingFacadeRejectsUntrackedDuplicateAndResetLaunches){
    EXPECT_EXIT({
        Source source(DeviceId::rocm(217),1);auto& worker=source.worker();
        auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
        auto& recording=RetainedParticipantGraphExecutableReplayAccess::recording(*owner);
        try{(void)recording.launch();std::_Exit(2);}catch(const std::logic_error&){}
        try{(void)RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1);std::_Exit(3);}catch(const std::logic_error&){}
        worker.submitAndWait([&]{if(!recording.beginCapture() || !recording.endCapture() || !recording.instantiate())std::_Exit(4);});
        try{(void)recording.launch();std::_Exit(5);}catch(const std::logic_error&){}
        try{(void)recording.launchOnStream(worker.defaultStream());std::_Exit(6);}catch(const std::logic_error&){}
        try{recording.reset();std::_Exit(7);}catch(const std::logic_error&){}
        try{(void)recording.beginCapture();std::_Exit(8);}catch(const std::logic_error&){}
        try{(void)RetainedParticipantGraphExecutableReplayAccess::recording(*owner);std::_Exit(9);}catch(const std::logic_error&){}
        if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1))std::_Exit(10);
        try{(void)RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1);std::_Exit(11);}catch(const std::logic_error&){}
        try{(void)RetainedParticipantGraphExecutableReplayAccess::submit(*owner,3);std::_Exit(12);}catch(const std::logic_error&){}
        if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,2))std::_Exit(13);
        worker.streamState=GPUStreamExecutionState::Pending;
        if(RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(14);
        worker.streamState=GPUStreamExecutionState::Complete;
        if(!RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(15);
        if(!RetainedParticipantGraphExecutableReplayAccess::submit(*owner,3))std::_Exit(16);
        if(!RetainedParticipantGraphExecutableReplayAccess::observe(*owner))std::_Exit(17);
        owner.reset();source.reset();std::_Exit(0);
    },::testing::ExitedWithCode(0),"");
}

TEST(PooledNativeProgramDiagnostic, RecordingNativeFailuresIrreversiblyRejectEveryRetry){
    EXPECT_EXIT({
        // One original authority must retain ALL process-lifetime graph slots.
        // Independent per-fault PMAs would conflict with the first retained root.
        const auto device=DeviceId::cuda(220);
        constexpr std::size_t faults=6;
        const auto graphBytes=faults*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General);
        Source source(device,faults,true,(64u<<20)+graphBytes);auto& worker=source.worker();
        for(int fault=1;fault<=6;++fault){
            worker.graphFault=fault;
            auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
            auto& facade=RetainedParticipantGraphExecutableReplayAccess::recording(*owner);
            worker.submitAndWait([&]{
                bool failed=false;
                try{
                    if(!facade.beginCapture())std::_Exit(2);
                    if(!facade.endCapture())std::_Exit(3);
                    if(fault==4 || fault==6){if(!facade.prepareRuntimeContextStorage(source.memory))failed=true;}
                    else if(!facade.instantiate())failed=true;
                }catch(const std::exception&){failed=true;}
                if(!failed)std::_Exit(4);
                const auto nativeCalls=worker.graphCalls;
                try{(void)facade.beginCapture();std::_Exit(5);}catch(const std::logic_error&){}
                try{(void)facade.endCapture();std::_Exit(6);}catch(const std::logic_error&){}
                try{(void)facade.instantiate();std::_Exit(7);}catch(const std::logic_error&){}
                try{(void)facade.prepareRuntimeContextStorage(source.memory);std::_Exit(8);}catch(const std::logic_error&){}
                if(worker.graphCalls!=nativeCalls)std::_Exit(9);
            });
            owner.reset();
            if(worker.capturesDestroyed!=fault)std::_Exit(10);
        }
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=graphBytes)std::_Exit(11);
        source.reset();
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=graphBytes)std::_Exit(12);
        std::_Exit(0); // Every original fault ran; only OS teardown releases graph backing.
    },::testing::ExitedWithCode(0),"");
}
TEST(PooledNativeProgramDiagnostic, RecordingRejectsForeignControlBlockAndWrongWorkerBeforeNative){
    EXPECT_EXIT({
        const auto device=DeviceId::rocm(221);
        Source source(device,4);auto& worker=source.worker();
        for(int wrong=0;wrong<4;++wrong){
            auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
            auto& facade=RetainedParticipantGraphExecutableReplayAccess::recording(*owner);
            if(wrong==0){
                try{(void)facade.beginCapture();std::_Exit(2);}catch(const std::logic_error&){}
                if(worker.graphCalls)std::_Exit(3);
            }else worker.submitAndWait([&]{
                if(!facade.beginCapture() || !facade.endCapture())std::_Exit(4);
                const auto before=worker.graphCalls;
                if(wrong==3)RetainedParticipantGraphExecutableReplayAccess::staleGeneration(*owner);
                auto foreign=wrong==1?std::shared_ptr<PhysicalMemoryAuthority>{}:wrong==3?source.memory:std::shared_ptr<PhysicalMemoryAuthority>(source.memory.get(),[](auto*){});
                try{(void)facade.prepareRuntimeContextStorage(foreign);std::_Exit(5);}catch(const std::invalid_argument&){}
                if(worker.graphCalls!=before)std::_Exit(6);
                try{(void)facade.instantiate();std::_Exit(7);}catch(const std::logic_error&){}
            });
            owner.reset();
            if(worker.capturesDestroyed!=wrong+1)std::_Exit(8);
        }
        const auto expected=4*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General);
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=expected)std::_Exit(9);
        source.reset();
        if(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable)!=expected)std::_Exit(10);
        std::_Exit(0);
    },::testing::ExitedWithCode(0),"");
}


/** @test The actual retained facade delegates native flat-kind checks and preserves exact class bounds. */
TEST(PooledNativeProgramDiagnostic, RecordingDelegatesBoundedFlatNativeValidation){
    for(const auto device:{DeviceId::cuda(230),DeviceId::rocm(230)}){
        EXPECT_EXIT({
            Source source(device,1);
            auto& worker=source.worker();
            auto owner=RetainedParticipantGraphExecutableReplayAccess::create(*source.original,worker.defaultStream());
            auto& facade=RetainedParticipantGraphExecutableReplayAccess::recording(*owner);
            worker.submitAndWait([&]{
                if(!facade.beginCapture() || !facade.endCapture())std::_Exit(1);
                std::string error;
                if(!facade.validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper,&error) ||
                   !error.empty() || worker.graphKindInspections!=1)std::_Exit(2);
                worker.graphFlatKinds=false;
                if(facade.validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper,&error) ||
                   error!="recorded native graph contains a forbidden conditional node" ||
                   worker.graphKindInspections!=2)std::_Exit(3);
                worker.graphFlatKinds=true;
                worker.graphNodes=GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes;
                if(!facade.validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper,&error) ||
                   !error.empty() || worker.graphKindInspections!=3)std::_Exit(4);
                for(const auto nodes:{std::size_t{0},GPUGraphMemoryContract::kBoundedFlatHelperMaxNodes+1}){
                    worker.graphNodes=nodes;
                    if(facade.validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper,&error) ||
                       error.find("exceeds its native-node contract")==std::string::npos ||
                       worker.graphKindInspections!=3)std::_Exit(5);
                }
                if(!facade.validateExecutableMemoryClass(GPUGraphExecutableClass::General,&error) ||
                   !error.empty() || worker.graphKindInspections!=3)std::_Exit(6);
            });
            owner.reset();source.reset();std::_Exit(0);
        },::testing::ExitedWithCode(0),"");
    }
}

/** @test Fully committed serving graph capacity cannot consume the separate admitted calibration slot. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationBOMIsolatedFromServingReservation)
{
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
        const PhysicalMemoryResource host{0,DeviceId::cpu(),4ULL<<30,4ULL<<30};
        const PhysicalMemoryResource gpu{0,device,256u<<20,256u<<20};
        const auto serving=RetainedParticipantGraphExecutable::generalBOM(host,gpu);
        const auto calibration=RetainedParticipantGraphExecutable::calibrationBOM(host,gpu);
        PhysicalMemoryPlanBuilder plan;
        for(const auto& row:serving.resources())plan.add(row);
        for(const auto& row:calibration.resources())plan.add(row);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        for(const auto resource:{host,gpu}){
            const auto key=PhysicalMemoryAllocatorIdentity{resource.world_rank,resource.device};
            const auto servingBytes=serving.find(key)->bytes(PhysicalMemoryOwner::NativeGraphExecutable);
            const auto calibrationBytes=calibration.find(key)->bytes(PhysicalMemoryOwner::NativePreparationGraphExecutable);
            const auto knownHolder=resource.device.is_cpu()?PreparedServiceCalibrationSlot::physicalBytes():0u;
            EXPECT_EQ(servingBytes+knownHolder,calibrationBytes);
            EXPECT_EQ(serving.find(key)->totalBytes()+knownHolder,calibration.find(key)->totalBytes());
            EXPECT_EQ(calibration.find(key)->bytes(PhysicalMemoryOwner::NativeGraphExecutable),0u);
            auto committed=memory->reserveNewAllocations(resource.device,PhysicalMemoryOwner::NativeGraphExecutable,servingBytes);
            // Original-owner control: the former calibration charge collides
            // after the model has committed all its own named graph budget.
            EXPECT_THROW((void)memory->reserveNewAllocations(resource.device,
                PhysicalMemoryOwner::NativeGraphExecutable,calibrationBytes),std::logic_error);
            auto preparation=memory->reserveNewAllocations(resource.device,
                PhysicalMemoryOwner::NativePreparationGraphExecutable,calibrationBytes);
            EXPECT_EQ(memory->reservedBytes(resource.device,PhysicalMemoryOwner::NativeGraphExecutable),servingBytes);
            EXPECT_EQ(memory->reservedBytes(resource.device,PhysicalMemoryOwner::NativePreparationGraphExecutable),calibrationBytes);
        }
        EXPECT_EQ(toString(PhysicalMemoryOwner::NativePreparationGraphExecutable),"native_preparation_graph_executable");
    }
}

namespace {
/** @brief Recorded reset/submit behavior through one actual original General owner. */
void sequentialPreparedCalibration(DeviceId device)
{
    Source source(device,1,true,256u<<20,true,1);auto& worker=source.worker();
    const auto servingBOM=RetainedParticipantGraphExecutable::generalBOM(
        source.source->originalResource(DeviceId::cpu()),source.source->originalResource(device));
    const auto servingHost=servingBOM.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable);
    const auto servingGPU=servingBOM.find({0,device})->bytes(PhysicalMemoryOwner::NativeGraphExecutable);
    auto modelHost=source.memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,servingHost);
    auto modelGPU=source.memory->reserveNewAllocations(device,PhysicalMemoryOwner::NativeGraphExecutable,servingGPU);
    auto owner=RetainedParticipantGraphExecutableReplayAccess::calibration(*source.original,worker.defaultStream());
    EXPECT_EQ(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable),servingGPU);
    const auto calls=worker.calls;
    const auto destroyedEvents=worker.eventsDestroyed;
    EXPECT_EQ(worker.timingEventsCreated,2);
    const auto host=source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable);
    const auto gpu=source.memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable);
    for(std::uint64_t definition=0;definition<12;++definition){
        RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
        EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::logic_error);
        if(definition)EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::submit(*owner,definition),std::logic_error);
        ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::submit(*owner,definition+1));
        EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::logic_error);
        worker.streamState=GPUStreamExecutionState::Pending;
        EXPECT_FALSE(RetainedParticipantGraphExecutableReplayAccess::observe(*owner));
        EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::logic_error);
        worker.streamState=GPUStreamExecutionState::Complete;
        ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::observe(*owner));
        RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner);
        EXPECT_EQ(worker.calls,calls); // Same physical wrapper; no hidden creator/token per class.
        EXPECT_EQ(worker.capturesDestroyed,0);
        EXPECT_EQ(worker.timingEventsCreated,2);
        EXPECT_EQ(worker.eventsDestroyed,destroyedEvents);
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable),host);
        EXPECT_EQ(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable),gpu);
        EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::submit(*owner,definition+1),std::logic_error);
    }
    owner.reset();EXPECT_EQ(worker.capturesDestroyed,1);
    EXPECT_EQ(worker.eventsDestroyed,destroyedEvents+2);
    EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable),
        GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(device));
    modelHost={};modelGPU={};
    source.reset();
}
/** @brief Native timing doubles require every operation on the original worker. */
class CalibrationTimingBackend final:public MockBackend {
public:
    CalibrationTimingBackend(DeviceType type,ProgramWorker& worker):MockBackend(type),worker_(worker){}
    void* createTimingEvent(int)override{ADD_FAILURE()<<"Measurement replaced original owner timing pair";return nullptr;}
    void destroyEvent(void*,int)override{ADD_FAILURE()<<"Measurement destroyed original owner timing pair";}
    bool recordEvent(void*,int,void* stream)override{EXPECT_TRUE(worker_.ownsCurrentThread());EXPECT_EQ(stream,worker_.defaultStream());recordClosures.push_back(worker_.queueClosures);++records;return true;}
    bool queryEvent(void*,int,bool* ready)override{EXPECT_TRUE(worker_.ownsCurrentThread());*ready=true;return true;}
    bool eventElapsedTimeMs(void*,void*,int,float* elapsed)override{EXPECT_TRUE(worker_.ownsCurrentThread());*elapsed=3;return true;}
    int records=0;
    std::vector<int> recordClosures;
private:ProgramWorker& worker_;
};
/** @brief Actual timing owner borrows one persistent pair and marshals only native operations. */
void measuredPreparedCalibration(DeviceId device)
{
    Source source(device,0,true,256u<<20,true,1);auto& worker=source.worker();
    auto owner=RetainedParticipantGraphExecutableReplayAccess::calibration(*source.original,worker.defaultStream());
    CalibrationTimingBackend backend(device.type,worker);
    const PlanningMeasurementWork work(PlanningWorkUnit::ArithmeticOperations,64,"original recorded prepared phase");
    for(int phase=0;phase<3;++phase){
        RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
        const auto before=worker.queueClosures;
        const auto observed=PlanningExecutionMeasurement::gpu(work,backend,device,*owner);
        EXPECT_EQ(worker.queueClosures-before,6); // warmup, query, timed batch, query, elapsed, terminal.
        ASSERT_EQ(backend.recordClosures.size(),size_t((phase+1)*3));
        EXPECT_EQ(backend.recordClosures[phase*3],before+1);
        EXPECT_EQ(backend.recordClosures[phase*3+1],before+3);
        EXPECT_EQ(backend.recordClosures[phase*3+2],before+3);
        EXPECT_EQ(observed.elapsedSeconds(),0.003);
        EXPECT_EQ(worker.timingEventsCreated,2);
        RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner);
    }
    EXPECT_EQ(backend.records,9);
    owner.reset();source.reset();
}
/** @brief Ordinary immutability, stale generation and failed reset stay absorbing. */
void rejectedPreparedCalibration(DeviceId device)
{
    Source source(device,1,true,256u<<20,true,2);auto& worker=source.worker();
    auto ordinary=RetainedParticipantGraphExecutable::create(*source.original,worker.defaultStream());
    RetainedParticipantGraphExecutableReplayAccess::ready(*ordinary);
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::submit(*ordinary,1));
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::observe(*ordinary));
    const auto prior=worker.graphCalls;
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*ordinary),std::logic_error);
    EXPECT_EQ(worker.graphCalls,prior);
    ordinary.reset();
    auto owner=RetainedParticipantGraphExecutableReplayAccess::calibration(*source.original,worker.defaultStream());
    RetainedParticipantGraphExecutableReplayAccess::ready(*owner);
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::submit(*owner,1));
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::observe(*owner));
    RetainedParticipantGraphExecutableReplayAccess::staleGeneration(*owner);
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::invalid_argument);
    EXPECT_EQ(worker.graphCalls,prior+3);
    RetainedParticipantGraphExecutableReplayAccess::restoreGeneration(*owner);
    worker.graphFault=7;
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::runtime_error);
    worker.graphFault=0;
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*owner),std::logic_error);
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::ready(*owner),std::logic_error);
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::submit(*owner,2),std::logic_error);
    owner.reset();EXPECT_EQ(worker.capturesDestroyed,2);
    const auto retiredEvents=worker.eventsDestroyed;
    const auto retiredCaptures=worker.capturesDestroyed;
    worker.fail=worker.calls+3; // Wrapper and first timing event succeed; second event fails.
    EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::calibration(*source.original,worker.defaultStream()),std::runtime_error);
    worker.fail=0;
    EXPECT_EQ(worker.eventsDestroyed,retiredEvents+1);
    EXPECT_EQ(worker.capturesDestroyed,retiredCaptures+1);
    source.reset();
}
}
/** @test Same original slot across every bounded class/phase, with exact terminal proof and monotonic submit identity. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationCUDASequential){EXPECT_EXIT({sequentialPreparedCalibration(DeviceId::cuda(281));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationROCmSequential){EXPECT_EXIT({sequentialPreparedCalibration(DeviceId::rocm(281));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}
/** @test Actual event loop keeps native timing operations on the original worker and reuses its pair. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationCUDATimingOwner){EXPECT_EXIT({measuredPreparedCalibration(DeviceId::cuda(283));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationROCmTimingOwner){EXPECT_EXIT({measuredPreparedCalibration(DeviceId::rocm(283));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}
/** @test Normal graph definitions, stale provenance and failed native reset never gain a re-record capability. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationCUDARejectsUnproved){EXPECT_EXIT({rejectedPreparedCalibration(DeviceId::cuda(282));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationROCmRejectsUnproved){EXPECT_EXIT({rejectedPreparedCalibration(DeviceId::rocm(282));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");}

namespace {
/** @brief Preserve precise assertion diagnostics when a death child exits without normal listeners. */
void reportCalibrationChildFailures(){
    const auto* result=::testing::UnitTest::GetInstance()->current_test_info()->result();
    for(int i=0;i<result->total_part_count();++i){const auto& part=result->GetTestPartResult(i);
        if(part.failed())std::fprintf(stderr,"%s:%d: %s\n",part.file_name(),part.line_number(),part.message());}
}
/** @brief Replay twenty runner-local borrows against one actual original prepared source. */
void sourceCalibrationRecreation(DeviceId device){
    Source source(device,0,true,256u<<20,true,1);auto& worker=source.worker();
    RetainedParticipantGraphExecutable* originalOwner=nullptr;
    std::uint64_t ordinal=0,host=0,gpu=0;
    for(int runner=0;runner<20;++runner){
#ifdef LLAMINAR_OLD_PER_RUNNER_CALIBRATION
        auto loan=RetainedParticipantGraphExecutableReplayAccess::calibration(*source.original,worker.defaultStream());
#else
        auto loan=source.original->borrowPreparedCalibration(worker.defaultStream());
        EXPECT_THROW(source.original->borrowPreparedCalibration(worker.defaultStream()),std::logic_error);
        EXPECT_THROW(source.original->borrowPreparedCalibration(reinterpret_cast<void*>(0x1234)),std::invalid_argument);
        if(!runner)originalOwner=loan.get();else EXPECT_EQ(loan.get(),originalOwner);
#endif
        for(int phase=0;phase<3;++phase){
            RetainedParticipantGraphExecutableReplayAccess::ready(*loan);
            if(ordinal)EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::submit(*loan,ordinal),std::logic_error);
            ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::submit(*loan,++ordinal));
            worker.streamState=GPUStreamExecutionState::Pending;
            EXPECT_FALSE(RetainedParticipantGraphExecutableReplayAccess::observe(*loan));
            EXPECT_THROW(RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*loan),std::logic_error);
            worker.streamState=GPUStreamExecutionState::Complete;
            ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::observe(*loan));
            RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*loan);
        }
#ifndef LLAMINAR_OLD_PER_RUNNER_CALIBRATION
        if(runner==19)EXPECT_THROW(NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source),std::logic_error);
#endif
        loan.reset();
        if(!runner){
            host=source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable);
            gpu=source.memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable);
        }
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable),host);
        EXPECT_EQ(source.memory->reservedBytes(device,PhysicalMemoryOwner::NativePreparationGraphExecutable),gpu);
#ifndef LLAMINAR_OLD_PER_RUNNER_CALIBRATION
        // This recorded counter includes one wrapper plus its two timing creators.
        EXPECT_EQ(worker.calls,3);EXPECT_EQ(worker.capturesDestroyed,0);
        EXPECT_EQ(worker.timingEventsCreated,2);EXPECT_EQ(worker.eventsDestroyed,0);
#endif
    }
    NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source);
    EXPECT_EQ(worker.capturesDestroyed,1);EXPECT_EQ(worker.eventsDestroyed,2);
    EXPECT_THROW(source.original->borrowPreparedCalibration(worker.defaultStream()),std::logic_error);
    EXPECT_EQ(source.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable),host);
    source.reset();
}
}
/** @test Original source lifetime retains the same physical calibration across runner replacement. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceCUDARecreation){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationRecreation(DeviceId::cuda(284));reportCalibrationChildFailures();std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceROCmRecreation){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationRecreation(DeviceId::rocm(284));reportCalibrationChildFailures();std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}

namespace {
/** @brief Deterministically pause the same production lookup before its off-lock acquire. */
void sourceCalibrationLookupRetirement(DeviceId device){
    Source source(device,0,true,256u<<20,true,1);auto& worker=source.worker();
    auto pending=NativeContextPreparationReplayAccess::pausePreparedCalibrationLookup(*source.original,worker.defaultStream());
    EXPECT_EQ(worker.calls,0);EXPECT_EQ(worker.timingEventsCreated,0);
    const auto claimed=source.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        PhysicalMemoryMaterializationKind::NewAllocation);
    EXPECT_EQ(claimed,PreparedServiceCalibrationSlot::physicalBytes());
    try{NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source);FAIL()<<"Pending holder loan must reject final retirement";}
    catch(const std::logic_error& failure){EXPECT_EQ(std::string(failure.what()),
        "Original prepared calibration retirement has an in-flight holder borrower");}
    EXPECT_EQ(source.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        PhysicalMemoryMaterializationKind::NewAllocation),claimed);
    try{auto denied=NativeContextPreparationReplayAccess::resumePreparedCalibrationLookup(pending,*source.original,worker.defaultStream());
        FAIL()<<"An already-issued lookup cannot create after source close";}
    catch(const std::logic_error& failure){EXPECT_EQ(std::string(failure.what()),"Original prepared calibration source slot is retiring");}
    EXPECT_EQ(worker.calls,0);EXPECT_EQ(worker.timingEventsCreated,0);
    EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable),0u);
    pending.reset();NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source);
    EXPECT_EQ(source.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_THROW(source.original->borrowPreparedCalibration(worker.defaultStream()),std::logic_error);
    source.reset();
}
}
/** @test Off-pool holder loans fail precisely before native teardown/zero-allocation retirement. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceCUDAHolderRetirement){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationLookupRetirement(DeviceId::cuda(285));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceROCmHolderRetirement){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationLookupRetirement(DeviceId::rocm(285));std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}

namespace {
/** @brief Exercise the same pending completion used by both actual final-model callers. */
void sourceCalibrationModelTicketOrder(DeviceId device){
    Source source(device,0,true,256u<<20,true,1);auto& worker=source.worker();
    auto loan=source.original->borrowPreparedCalibration(worker.defaultStream());
    RetainedParticipantGraphExecutableReplayAccess::ready(*loan);
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::submit(*loan,1));
    ASSERT_TRUE(RetainedParticipantGraphExecutableReplayAccess::observe(*loan));
    RetainedParticipantGraphExecutableReplayAccess::resetCalibration(*loan);
    // Storage-only native workers need no fabricated model-weight retention row.
    // The overload closes source loans before the SAME completion dispatch for
    // either HostOnly or DeviceBatch, including its one-shot consumption flag.
    auto pending=PendingExclusiveModelRetirement::begin({});
    const auto complete=[&]{
#ifdef LLAMINAR_OLD_MODEL_TICKET_CALIBRATION_ORDER
        auto result=pending.complete();
        source.source->retireReleasedContexts(); // Literal former caller order.
        return result;
#else
        return pending.complete(source.source);
#endif
    };
    EXPECT_THROW((void)complete(),std::logic_error);
    EXPECT_EQ(worker.capturesDestroyed,0);EXPECT_EQ(worker.eventsDestroyed,0);
    loan.reset();
    EXPECT_NO_THROW({const auto receipt=complete();EXPECT_EQ(receipt.retiredDeviceCount(),0u);});
    // Identical pending state is consumed only by successful completion.
    EXPECT_THROW((void)pending.complete(source.source),std::logic_error);
    NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source);
    NativeContextPreparationReplayAccess::closePreparedCalibrations(source.source);
    EXPECT_EQ(worker.capturesDestroyed,1);EXPECT_EQ(worker.eventsDestroyed,2);
    EXPECT_EQ(source.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    source.reset();
}
}
/** @test Source loan preflight precedes actual pending model-ticket consumption. */
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceCUDAModelTicketOrder){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationModelTicketOrder(DeviceId::cuda(286));reportCalibrationChildFailures();
        std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}
TEST(PooledNativeProgramDiagnostic,PreparedCalibrationSourceROCmModelTicketOrder){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT({const rlimit limit{};if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("cannot disable child core dumps\n",stderr);_exit(2);}
        sourceCalibrationModelTicketOrder(DeviceId::rocm(286));reportCalibrationChildFailures();
        std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
}

}
