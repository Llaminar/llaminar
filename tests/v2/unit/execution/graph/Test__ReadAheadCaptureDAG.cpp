/**
 * @file Test__ReadAheadCaptureDAG.cpp
 * @brief Device-free replay of the production capture ledger's two-stream DAG.
 *
 * Fake handles identify exact streams/events; the production ledger performs
 * the same record/wait lifecycle used while recording CUDA/HIP parents. Tests
 * reject host enqueue order as a substitute for a native dependency.
 */
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/ReadAheadScheduling.h"
#include "execution/local_execution/graph/ReadAheadParticipantContract.h"
#include <gtest/gtest.h>
#include <tuple>

namespace llaminar2 { namespace {
using Ledger = GraphCaptureDependencyLedger;
using Plan = Ledger::StagePlan;
struct Recorder final : Ledger::EventRecorder
{
    std::vector<std::tuple<char,void*,void*>> trace;
    bool fail_record=false, fail_wait=false;
    /** @brief Record the exact edge or reproduce native rejection. */
    void record(void* e,void* s) override
    {if(fail_record)throw std::runtime_error("record failed");trace.emplace_back('r',e,s);}
    /** @brief Queue the exact wait or reproduce native rejection. */
    void wait(void* e,void* s) override
    {if(fail_wait)throw std::runtime_error("wait failed");trace.emplace_back('w',e,s);}
};
struct Fixture : ::testing::Test
{
    int ids[10]{};
    void* main=&ids[0];void* side=&ids[1];void* fork=&ids[2];void* join=&ids[3];
    const TensorBase* query=reinterpret_cast<const TensorBase*>(&ids[4]);
    const TensorBase* output=reinterpret_cast<const TensorBase*>(&ids[5]);
    Recorder recorder;
    /** @brief Freeze Hash -> side gather alongside independent trunk -> Consume. */
    std::vector<Plan> plan()
    {
        return {
            {.stage_identity=&ids[6],.stage_name="hash",.outputs={query},.completion_event=fork},
            {.stage_identity=&ids[7],.stage_name="gather",.internal_inputs={{query,0}},.outputs={output},.recording_stream=side,.completion_event=join,.wait_for_stage=0},
            {.stage_identity=&ids[8],.stage_name="trunk"},
            {.stage_identity=&ids[9],.stage_name="consume",.internal_inputs={{output,1}},.wait_for_stage=1}
        };
    }
    /** @brief Replay recording through the actual production stage lifecycle. */
    void replay(Ledger& ledger)
    {ledger.beginCapture();for(int i=6;i<10;++i){ledger.beginStage(&ids[i],i==7?side:main);ledger.completeStage(&ids[i]);}EXPECT_TRUE(ledger.allStagesRecorded());ledger.endCapture();}
};
TEST(ReadAheadParticipantShape, DegreeOneContextAuthenticatesExactFullOriginalAssignment)
{
    const ReadAheadFullGeometry full{16,4,128,256};
    const std::array<GlobalDeviceAddress,1> members{{{.device_type=DeviceType::CUDA,.device_ordinal=0}}};
    std::array<DeviceShardingAssignment,1> assignment{{{.device=DeviceId::cuda(0),.head_start=0,.head_count=16,
        .kv_head_start=0,.kv_head_count=4,.d_ff_start=0,.d_ff_count=128,.vocab_start=0,.vocab_count=256,.work_fraction=1,.local_rank=0}}};
    EXPECT_NO_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),{}, {},full));
    EXPECT_NO_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),members,assignment,full));
    auto partial=assignment;partial[0].head_count=8;
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),members,partial,full),std::invalid_argument);
    partial=assignment;partial[0].vocab_start=16;
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),members,partial,full),std::invalid_argument);
    partial=assignment;partial[0].work_fraction=.5f;
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),members,partial,full),std::invalid_argument);
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::rocm(0),members,assignment,full),std::invalid_argument);
}
TEST(ReadAheadParticipantShape, MultipleForeignOrIncompleteParticipantsCannotEnterLocalReadAhead)
{
    const ReadAheadFullGeometry full{16,4,128,256};
    const std::array<GlobalDeviceAddress,2> multiple{{{.device_type=DeviceType::CUDA,.device_ordinal=0},{.device_type=DeviceType::CUDA,.device_ordinal=1}}};
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),multiple,{},full),std::invalid_argument);
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cuda(0),{}, {},{16,0,128,256}),std::invalid_argument);
    EXPECT_THROW(requireReadAheadSingleParticipantShape(DeviceId::cpu(),{}, {},full),std::invalid_argument);
}
TEST(ReadAheadJoinPlanning, EarliestDeclaredHostBoundaryClosesSideBeforeLaterPLE)
{
    using Node=ReadAheadJoinFrontier;using Kind=Node::Kind;
    const std::array<Node,7> order{{{"hash"},{"gather"},{"attention"},{"dispatch",Kind::BeforeDeclaredHostBoundary},{"cpu_service"},{"return"},{"key",Kind::TransactionTerminal}}};
    for(int replay=0;replay<20;++replay)
        EXPECT_EQ(resolveReadAheadJoinAnchor(order,"hash","gather","key",ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier),"dispatch");
    EXPECT_THROW((void)resolveReadAheadJoinAnchor(order,"hash","gather","key",ReadAheadJoinPolicy::AtConsumer),std::invalid_argument);
    EXPECT_THROW((void)resolveReadAheadJoinAnchor(order,"hash","gather","missing",ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier),std::invalid_argument);
}
TEST(ReadAheadJoinPlanning, CompleteLocalGraphJoinsAtActualConsumerAndRejectsBrokenCrossing)
{
    using Node=ReadAheadJoinFrontier;using Kind=Node::Kind;
    const std::array<Node,4> local{{{"hash"},{"gather"},{"attention"},{"key"}}};
    EXPECT_EQ(resolveReadAheadJoinAnchor(local,"hash","gather","key",ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier),"key");
    EXPECT_THROW((void)resolveReadAheadJoinAnchor(local,"hash","gather","key",static_cast<ReadAheadJoinPolicy>(255)),std::invalid_argument);
    auto unknown=local;unknown[2].kind=static_cast<Kind>(255);
    EXPECT_THROW((void)resolveReadAheadJoinAnchor(unknown,"hash","gather","key",ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier),std::invalid_argument);
    const std::array<Node,4> broken{{{"hash"},{"dispatch",Kind::BeforeDeclaredHostBoundary},{"gather"},{"key"}}};
    EXPECT_THROW((void)resolveReadAheadJoinAnchor(broken,"hash","gather","key",ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier),std::invalid_argument);
}
TEST(ReadAheadRecordingAuthority, DistinctRecordingAndSubmissionRemainStableBeforePreparationAndAfterSeal)
{
    int handles[4]{};ReadAheadRecordingBinding binding;
    EXPECT_THROW(binding.seal(&handles[0]),std::logic_error);
    EXPECT_TRUE(binding.plan(&handles[0],&handles[1]));
    const auto identity=binding.identity();EXPECT_NE(identity,0u);
    EXPECT_NO_THROW(binding.require(&handles[0]));
    EXPECT_THROW(binding.requireSealed(&handles[0]),std::logic_error);
    EXPECT_THROW(binding.require(&handles[1]),std::logic_error);
    EXPECT_THROW(binding.seal(&handles[2]),std::logic_error);
    EXPECT_NO_THROW(binding.seal(&handles[0]));
    EXPECT_EQ(binding.identity(),identity);EXPECT_FALSE(binding.plan(&handles[0],&handles[1]));
    for(int replay=0;replay<20;++replay){EXPECT_NO_THROW(binding.requireSealed(&handles[0]));EXPECT_EQ(binding.identity(),identity);}
    EXPECT_THROW(binding.plan(&handles[2],&handles[1]),std::logic_error);
    EXPECT_THROW(binding.plan(&handles[0],&handles[3]),std::logic_error);
    EXPECT_THROW(binding.plan(nullptr,&handles[1]),std::invalid_argument);
}
TEST(ReadAheadRecordingAuthority, SerializedVariantsHaveOwnRecordingIdentitiesAndOneSubmissionAuthority)
{
    int handles[3]{};ReadAheadRecordingBinding first,second;
    EXPECT_TRUE(first.plan(&handles[0],&handles[2]));EXPECT_TRUE(second.plan(&handles[1],&handles[2]));
    EXPECT_NE(first.identity(),second.identity());first.seal(&handles[0]);second.seal(&handles[1]);
    EXPECT_THROW(first.requireSealed(&handles[1]),std::logic_error);
    EXPECT_THROW(second.requireSealed(&handles[0]),std::logic_error);
}
TEST(ReadAheadPhasePolicy, ResidentCandidateDeliberatelySeparatesPrefillAndDecode)
{
    using Policy=ImmutableLookupSchedule;using Selected=ImmutableLookupInvocationSchedule;
    EXPECT_EQ(resolveImmutableLookupInvocationSchedule(Policy::EarlyCurrentResidentDecodeGPUCandidate,ForwardExecutionPhase::Prefill),Selected::ConsumerScheduled);
    EXPECT_EQ(resolveImmutableLookupInvocationSchedule(Policy::EarlyCurrentResidentDecodeGPUCandidate,ForwardExecutionPhase::Decode),Selected::ResidentSideLane);
    for(const auto phase:{ForwardExecutionPhase::Prefill,ForwardExecutionPhase::Decode})
    {
        EXPECT_EQ(resolveImmutableLookupInvocationSchedule(Policy::ConsumerScheduled,phase),Selected::ConsumerScheduled);
        EXPECT_EQ(resolveImmutableLookupInvocationSchedule(Policy::EarlyCurrentHostedCandidate,phase),Selected::HostedPublishConsume);
    }
    EXPECT_THROW((void)resolveImmutableLookupInvocationSchedule(static_cast<Policy>(255),ForwardExecutionPhase::Decode),std::invalid_argument);
    EXPECT_THROW((void)resolveImmutableLookupInvocationSchedule(Policy::EarlyCurrentResidentDecodeGPUCandidate,static_cast<ForwardExecutionPhase>(255)),std::invalid_argument);
}
TEST(ReadAheadPhasePolicy, CaptureIdentityIncludesExplicitPhaseAndLaneIndependentOfGeometry)
{
    using Selected=ImmutableLookupInvocationSchedule;
    const auto prefill=immutableLookupPhaseIdentity(ForwardExecutionPhase::Prefill,Selected::ConsumerScheduled);
    const auto decode=immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,Selected::ConsumerScheduled);
    const auto side=immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,Selected::ResidentSideLane);
    EXPECT_NE(prefill,decode);EXPECT_NE(decode,side);EXPECT_NE(prefill,side);
    for(int replay=0;replay<20;++replay)EXPECT_EQ(side,immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,Selected::ResidentSideLane));
    EXPECT_THROW((void)immutableLookupPhaseIdentity(static_cast<ForwardExecutionPhase>(255),Selected::ResidentSideLane),std::invalid_argument);
    EXPECT_THROW((void)immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,static_cast<Selected>(255)),std::invalid_argument);
}
TEST_F(Fixture, ExactForkJoinRepeatedReplayAndReset)
{
    Ledger ledger(DeviceId::cuda(0),main,plan(),"read-ahead",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder);
    for(int replay_index=0;replay_index<4;++replay_index)
    {
        recorder.trace.clear();replay(ledger);
        ASSERT_EQ(recorder.trace.size(),4u);
        EXPECT_EQ(recorder.trace[0],std::make_tuple('r',fork,main));
        EXPECT_EQ(recorder.trace[1],std::make_tuple('w',fork,side));
        EXPECT_EQ(recorder.trace[2],std::make_tuple('r',join,side));
        EXPECT_EQ(recorder.trace[3],std::make_tuple('w',join,main));
    }
}
TEST_F(Fixture, JoinedFirstUnitPublishesOriginalResultToOrderedLaterPLEUnit)
{
    auto first=plan();first.resize(3);first[2].wait_for_stage=1;
    Ledger source(DeviceId::cuda(0),main,first,"before original host cutpoint",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder);
    for(int epoch=0;epoch<20;++epoch)
    {
        source.beginCapture();for(int i=6;i<9;++i){source.beginStage(&ids[i],i==7?side:main);source.completeStage(&ids[i]);}
        EXPECT_TRUE(source.allStagesRecorded());source.endCapture();
        Ledger later(DeviceId::cuda(0),main,{{.stage_identity=&ids[9],.stage_name="PLE key",.retained_parent_inputs={output}}},"ordered later retained unit");
        later.beginCapture();later.beginStage(&ids[9]);
        EXPECT_EQ(later.classifyInput(output,DeviceId::cuda(0),main),Ledger::InputDisposition::RetainedParentRecorded);
        EXPECT_EQ(later.classifyInput(query,DeviceId::cuda(0),main),Ledger::InputDisposition::StrictExternal);
        later.completeStage(&ids[9]);EXPECT_TRUE(later.allStagesRecorded());later.endCapture();
    }
}
TEST_F(Fixture, HostSubmissionOrderCannotProveCrossStreamInput)
{
    auto p=plan();p[1].wait_for_stage.reset();
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"missing fork",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
    p=plan();p[3].wait_for_stage.reset();
    EXPECT_THROW((Ledger(DeviceId::rocm(0),main,p,"missing join",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
}
TEST_F(Fixture, ClosureRejectsUnjoinedBranchEvenWithoutTensorRead)
{
    auto p=plan();p[3].internal_inputs.clear();p[3].wait_for_stage.reset();
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"open branch",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
}
TEST_F(Fixture, RejectsUnknownEventFutureWaitDuplicateProducerAndThirdLane)
{
    auto p=plan();p[1].wait_for_stage=3;
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"future",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
    p=plan();p[0].completion_event=nullptr;
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"no event",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
    p=plan();p[1].completion_event=fork;
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"alias",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
    p=plan();p[2].recording_stream=&ids[4];
    EXPECT_THROW((Ledger(DeviceId::cuda(0),main,p,"third lane",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder)),std::invalid_argument);
}
TEST_F(Fixture, FailureAndPartialCaptureCannotPublishCompletion)
{
    Ledger ledger(DeviceId::cuda(0),main,plan(),"failure",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder);
    ledger.beginCapture();ledger.beginStage(&ids[6]);recorder.fail_record=true;
    EXPECT_THROW(ledger.completeStage(&ids[6]),std::runtime_error);
    EXPECT_TRUE(ledger.failed());EXPECT_FALSE(ledger.allStagesRecorded());ledger.endCapture();
    recorder.fail_record=false;ledger.beginCapture();ledger.beginStage(&ids[6]);ledger.completeStage(&ids[6]);recorder.fail_wait=true;
    EXPECT_THROW(ledger.beginStage(&ids[7],side),std::runtime_error);
    EXPECT_TRUE(ledger.failed());EXPECT_FALSE(ledger.allStagesRecorded());ledger.endCapture();
    recorder.fail_wait=false;replay(ledger);
}
TEST_F(Fixture, ExactStageStreamAuthorityAndLateCompletionAreRejected)
{
    Ledger ledger(DeviceId::cuda(0),main,plan(),"stream",Ledger::ExternalInputAuthority::RequireReadyBytes,&recorder);
    ledger.beginCapture();ledger.beginStage(&ids[6]);ledger.completeStage(&ids[6]);
    EXPECT_THROW(ledger.beginStage(&ids[7],main),std::logic_error);
    EXPECT_THROW(ledger.beginStage(&ids[7]),std::logic_error);
    ledger.endCapture();ledger.beginCapture();ledger.beginStage(&ids[6]);ledger.completeStage(&ids[6]);ledger.beginStage(&ids[7],side);
    EXPECT_THROW(ledger.classifyInput(query,DeviceId::cuda(0),main),std::logic_error);
    EXPECT_EQ(ledger.classifyInput(query,DeviceId::cuda(0),side),Ledger::InputDisposition::InternalRecorded);
    ledger.abortCurrentStage();EXPECT_THROW(ledger.completeStage(&ids[7]),std::logic_error);ledger.endCapture();
    EXPECT_THROW(ledger.completeStage(&ids[7]),std::logic_error);replay(ledger);
}
}}
