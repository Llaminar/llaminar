/**
 * @file Test__ReadAheadSnapshotOrdering.cpp
 * @brief Replay actual event completion with exactly one immutable snapshot recorder.
 *
 * Native event-closed producers copy before join. Unrelated manual/CPU ticket
 * boundaries keep controller ownership, even in the same graph invocation.
 */
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/SnapshotRecordingOwnership.h"
#include <gtest/gtest.h>
#include <tuple>
namespace llaminar2 { namespace {
using Authority=StageSnapshotRecordingAuthority;
using Boundary=SnapshotPublicationBoundary;
using Ledger=GraphCaptureDependencyLedger;
/** @brief Native record/wait sink proves exact publication order without a device. */
struct Recorder final:Ledger::EventRecorder {
    std::vector<std::tuple<char,void*,void*>> trace;
    void record(void* event,void* stream)override{trace.emplace_back('r',event,stream);}
    void wait(void* event,void* stream)override{trace.emplace_back('w',event,stream);}
};
TEST(ReadAheadSnapshotOrdering, SideCopyPrecedesJoinAndMixedManualStageRetainsExactlyOneControllerCopy)
{
    for(int epoch=0;epoch<20;++epoch){
        int handles[7]{};auto* main=&handles[0];auto* side=&handles[1];auto* fork=&handles[2];auto* join=&handles[3];
        Recorder native;
        Ledger ledger(DeviceId::cuda(0),main,{
            {.stage_identity=&handles[4],.stage_name="hash",.completion_event=fork},
            {.stage_identity=&handles[5],.stage_name="gather",.recording_stream=side,.completion_event=join,.wait_for_stage=0},
            {.stage_identity=&handles[6],.stage_name="native_unit_terminal",.wait_for_stage=1}},"snapshot",Ledger::ExternalInputAuthority::RequireReadyBytes,&native);
        ledger.beginCapture();ledger.beginStage(&handles[4],main);ledger.completeStage(&handles[4]);
        int copies=0;
        const auto sideOwner=snapshotRecordingAuthorityForBoundary(Authority::GraphCaptureController,Boundary::NativeEventClosedProducer);
        ledger.beginStage(&handles[5],side);
        if(sideOwner==Authority::StageExecutor){native.trace.emplace_back('s',nullptr,side);++copies;}
        ledger.completeStage(&handles[5]);
        if(sideOwner==Authority::GraphCaptureController){native.trace.emplace_back('s',nullptr,main);++copies;}
        ledger.beginStage(&handles[6],main);ledger.completeStage(&handles[6]);ledger.endCapture();
        ASSERT_EQ(native.trace.size(),5u);
        EXPECT_EQ(native.trace[2],std::make_tuple('s',static_cast<void*>(nullptr),static_cast<void*>(side)));
        EXPECT_EQ(native.trace[3],std::make_tuple('r',static_cast<void*>(join),static_cast<void*>(side)));
        EXPECT_EQ(native.trace[4],std::make_tuple('w',static_cast<void*>(join),static_cast<void*>(main)));
        EXPECT_EQ(copies,1);
        // Later declared manual/CPU ticket service is outside native capture.
        const auto manualOwner=snapshotRecordingAuthorityForBoundary(Authority::GraphCaptureController,Boundary::ControllerStage);
        int stageCopies=0,controllerCopies=0;
        if(manualOwner==Authority::StageExecutor)++stageCopies;
        if(manualOwner==Authority::GraphCaptureController)++controllerCopies;
        EXPECT_EQ(stageCopies,0);EXPECT_EQ(controllerCopies,1);
    }
}
TEST(ReadAheadSnapshotOrdering, FailedSideSnapshotCannotPublishJoinOrAuthorizeConsumer)
{
    int handles[7]{};auto* main=&handles[0];auto* side=&handles[1];auto* fork=&handles[2];auto* join=&handles[3];
    Recorder native;
    Ledger ledger(DeviceId::rocm(0),main,{
        {.stage_identity=&handles[4],.stage_name="hash",.completion_event=fork},
        {.stage_identity=&handles[5],.stage_name="gather",.recording_stream=side,.completion_event=join,.wait_for_stage=0},
        {.stage_identity=&handles[6],.stage_name="consume",.wait_for_stage=1}},
        "failed snapshot",Ledger::ExternalInputAuthority::RequireReadyBytes,&native);
    ledger.beginCapture();ledger.beginStage(&handles[4],main);ledger.completeStage(&handles[4]);
    ledger.beginStage(&handles[5],side);
    // runStage's exceptional/false completion takes this production abort edge.
    ledger.abortCurrentStage();
    EXPECT_THROW(ledger.beginStage(&handles[6],main),std::logic_error);
    EXPECT_FALSE(ledger.allStagesRecorded());
    ASSERT_EQ(native.trace.size(),2u);
    EXPECT_EQ(std::get<0>(native.trace[0]),'r');EXPECT_EQ(std::get<0>(native.trace[1]),'w');
}
TEST(ReadAheadSnapshotOrdering, ResetAndUnknownOwnershipNeverAuthorizeAnotherRecorder)
{
    EXPECT_EQ(snapshotRecordingAuthorityForBoundary(Authority::StageExecutor,Boundary::ControllerStage),Authority::StageExecutor);
    EXPECT_THROW(snapshotRecordingAuthorityForBoundary(Authority::GraphCaptureController,static_cast<Boundary>(255)),std::invalid_argument);
    EXPECT_THROW(snapshotRecordingAuthorityForBoundary(static_cast<Authority>(255),Boundary::NativeEventClosedProducer),std::invalid_argument);
}
}} // namespace llaminar2
