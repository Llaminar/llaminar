/**
 * @file Test__RetainedGraphParticipantEnqueueGroup.cpp
 * @brief Bounded replay of actual recurring host enqueue ordering and fatal partial ownership.
 *
 * Workers execute CPU-only jobs. The source gate models native enqueue
 * backpressure, not GPU completion or model state. No vendor API is loaded.
 */
#include "execution/moe/MoEOverlayFollowerRowGeometry.h"

#include <gtest/gtest.h>
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include "execution/local_execution/graph/RetainedGraphParticipantEnqueueGroup.h"
#include "mocks/MockWorkerGPUContext.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockBackend.h"
#include "tensors/TensorClasses.h"
#include "memory/BufferArena.h"
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "fixtures/execution/RetainedGraphParticipantsReplayAccess.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include "collective/LocalTPContext.h"
#include <array>
namespace llaminar2::test {
/** @brief Genuine asynchronous CPU executor; synchronous submit emulation would mask the regression. */
class EnqueueReplayWorker final:public testing::MockWorkerGPUContext {
public:
    explicit EnqueueReplayWorker(bool reject=false):reject_(reject){}
    std::future<void> submitAsync(std::function<void()> work)override{
        if(reject_)throw std::runtime_error("recorded queue rejection before enqueue");
        return std::async(std::launch::async,[this,work=std::move(work)]{submitAndWait(work);});
    }
private:bool reject_;
};
TEST(RetainedGraphParticipantEnqueueGroup, SourceBeforeRemoteHostJoinResetAndRepeat){
    auto worker=std::make_shared<EnqueueReplayWorker>();
    std::shared_future<void> source;
    std::atomic<unsigned> timedOut=0,returned=0;
    auto group=RetainedGraphParticipantsReplayAccess::group({{worker,[&]{
        if(source.wait_for(std::chrono::milliseconds(100))!=std::future_status::ready)++timedOut;
        ++returned;
    }}});
    // Original wait-first ordering is a bounded red control: its remote cannot
    // complete until the source call starts. It times out instead of hanging.
    {std::promise<void> ready;source=ready.get_future().share();group->dispatch();
     group->completeOriginalEnqueue();ready.set_value();EXPECT_EQ(timedOut.load(),1u);}
    for(unsigned epoch=0;epoch<20;++epoch){
        std::promise<void> ready;source=ready.get_future().share();
        group->dispatch(); // Must return without awaiting the remote SDK call.
        ready.set_value(); // The exact original source enqueue starts here.
        group->completeOriginalEnqueue();
        EXPECT_EQ(timedOut.load(),1u);EXPECT_EQ(returned.load(),epoch+2u);
    }
}
TEST(RetainedGraphParticipantEnqueueGroup, PreDispatchRejectionOwnsNoNativeReader){
    auto worker=std::make_shared<EnqueueReplayWorker>(true);
    auto group=RetainedGraphParticipantsReplayAccess::group({{worker,[]{FAIL()<<"Rejected queue ran native body";}}});
    for(unsigned repeat=0;repeat<20;++repeat)EXPECT_THROW(group->dispatch(),std::runtime_error);
    // Actual destructor is safe: no remote job escaped the original queue.
    group.reset();
}
TEST(RetainedGraphParticipantEnqueueGroup, PartialDispatchAndMissingSourceCannotUnwindReaders){
    for(const bool partial:{false,true}){
        EXPECT_EXIT(( [partial]{
            std::set_terminate([]{std::_Exit(71);});
            struct Payload{~Payload(){std::_Exit(99);}} payload;
            auto worker=std::make_shared<EnqueueReplayWorker>();
            std::promise<void> neverSource;auto source=neverSource.get_future().share();
            std::vector<RetainedGraphParticipantEnqueueGroup::Job> jobs{{worker,[source]{source.wait_for(std::chrono::milliseconds(200));}}};
            if(partial)jobs.push_back({std::make_shared<EnqueueReplayWorker>(true),[]{}});
            auto group=RetainedGraphParticipantsReplayAccess::group(std::move(jobs));
            try{group->dispatch();throw std::runtime_error("recorded source launch failure");}
            catch(...){group.reset();}
            std::_Exit(98);
        }()),::testing::ExitedWithCode(71),"");
    }
}
/** @brief Recorded worker construction uses the actual original mapped object/root but executes no native API. */
class PreparationReplayWorker final:public testing::MockWorkerGPUContext {
public:PreparationReplayWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
};
TEST(RetainedGraphParticipantEnqueueGroup, PreparationStreamRejectsForeignAndRetiredSource){
    EXPECT_EXIT(([]{
        {
            using Access=NativeContextPreparationReplayAccess;
            const std::array devices{DeviceId::cuda(0),DeviceId::rocm(0)};
            const std::array<PhysicalMemoryResource,3> observed{{{0,DeviceId::cpu(),32ull<<30,30ull<<30},
                {0,devices[0],32ull<<30,30ull<<30},{0,devices[1],32ull<<30,30ull<<30}}};
            for(const auto device:devices)Access::seed(device,[ordinal=device.ordinal](auto stack){
                return Access::construct<PreparationReplayWorker>(std::move(stack),ordinal);
            });
            auto source=Access::preparedSource<PreparationReplayWorker>(devices,observed);
            {
                auto original=source->acquire(devices[0]),foreign=source->acquire(devices[1]);
                void* exact=nullptr;original.worker()->submitAndWait([&]{exact=original.worker()->defaultStream();});
                auto token=RetainedGraphParticipantsReplayAccess::preparation(original,exact);
                EXPECT_EQ(token->stream(),exact);EXPECT_NO_THROW(token->requireLive(original));
                EXPECT_THROW(token->requireLive(foreign),std::logic_error);
                RetainedGraphParticipantsReplayAccess::retire(*token);
                EXPECT_THROW(token->requireLive(original),std::logic_error);
                auto replacement=RetainedGraphParticipantsReplayAccess::preparation(original,exact);
                EXPECT_NO_THROW(replacement->requireLive(original));
                EXPECT_THROW(token->requireLive(original),std::logic_error);
                RetainedGraphParticipantsReplayAccess::retire(*replacement);
            }
            for(const auto device:devices){auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(reset);}
            source->retireUnadmitted();
        }
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

TEST(RetainedGraphParticipantEnqueueGroup, FaultedRemoteCannotLeaveSourceWaiting){
    EXPECT_EXIT(([]{
        std::set_terminate([]{std::_Exit(71);});
        auto worker=std::make_shared<EnqueueReplayWorker>();
        auto group=RetainedGraphParticipantsReplayAccess::group({{worker,[]{throw std::runtime_error("recorded peer SDK failure");}}});
        group->dispatch();
        std::promise<void> absentPeer;
        // The actual source call could be backpressured on this failed peer.
        // A latent future exception would reach this bounded red-control exit.
        absentPeer.get_future().wait_for(std::chrono::milliseconds(200));
        std::_Exit(98);
    }()),::testing::ExitedWithCode(71),"");
}

/** @brief Native work is absent; callbacks prove actual cold binding enters the selected worker capability. */
class ColdPreparationPlan final:public IRetainedGraphParticipantSourcePlan {
public:
 explicit ColdPreparationPlan(PreparedGPUContextUsePermit original):original_(std::move(original)){}
 const PreparedGPUContextUsePermit& originalSource()const noexcept override{return original_;}
 void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream> stream)const override{
  EXPECT_TRUE(original_.worker()->ownsCurrentThread());stream->requireLive(original_);prepared=std::move(stream);++preparations;
 }
 std::uint64_t captureIdentity()const noexcept override{return 19;}
 void authenticateRecordingOwner(const IGPUGraphCapture& actual)const override{
  EXPECT_TRUE(original_.worker()->ownsCurrentThread());const auto current=prepared.lock();if(!current)throw std::logic_error("Cold recorder lost its original cache token");EXPECT_EQ(actual.executionStream(),current->stream());++recordings;
 }
 std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(const RetainedGraphParticipantSourceSeal&)const override{throw std::logic_error("Cold caller replay never submits a model");}
 mutable unsigned preparations=0,recordings=0;
 // Production retains preparation observers weakly: each token owns its source plan.
 mutable std::weak_ptr<const RetainedGraphParticipantPreparationStream> prepared;
private:PreparedGPUContextUsePermit original_;
};
/** @brief Recorded dispatch uses the same worker entry convention; broken queue tests cannot forge ownership. */
class ColdPreparationWorker final:public testing::MockWorkerGPUContext {
public:
 ColdPreparationWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
 void submitAndWait(std::function<void()> work)override{++submissions;if(brokenDispatch)work();else MockWorkerGPUContext::submitAndWait(std::move(work));}
 unsigned submissions=0;bool brokenDispatch=false;
};
/** @brief Original test source retirement also runs when the old cold guard throws. */
struct ColdPreparationRetirement {
 DeviceId device;
 ~ColdPreparationRetirement(){auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextPreparationReplayAccess::physicalReset(reset);}
};
/** @test The actual first-capture preparation and cold parent seal accept control and worker callers on the SAME original stream. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdOriginalPreparationMarshalsOffWorkerAndRunsInlineOnWorker){
 using A=NativeContextPreparationReplayAccess;
 for(const auto device:{DeviceId::cuda(71),DeviceId::rocm(71)}){
  A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
  auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
  {
   const auto original=source->acquire(device);auto* worker=static_cast<ColdPreparationWorker*>(original.worker().get());
   auto plan=std::make_shared<ColdPreparationPlan>(original);ComputeGraph graph;graph.setRetainedParticipantSourcePlan(plan);
   DeviceGraphExecutor executor;BufferArena bindingArena;executor.setArena(&bindingArena);DeviceGraphExecutor::GraphSegmentCache cache;void* const exact=reinterpret_cast<void*>(0x8170);
   ASSERT_TRUE(cache.bindBorrowedCaptureStream(worker,exact,device,false));
   cache.retained_parent_capture=std::make_unique<testing::MockGPUGraphCapture>(exact);ASSERT_TRUE(cache.retained_parent_capture->beginCapture());ASSERT_TRUE(cache.retained_parent_capture->endCapture());ASSERT_TRUE(cache.retained_parent_capture->instantiate());
   EXPECT_FALSE(worker->ownsCurrentThread());const auto before=worker->submissions;
   RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,worker);
   RetainedGraphParticipantsReplayAccess::authenticateRecording(executor,graph,cache,worker);
   EXPECT_EQ(worker->submissions,before+2);EXPECT_EQ(plan->preparations,1u);EXPECT_EQ(plan->recordings,1u);EXPECT_EQ(plan->prepared.lock()->stream(),exact);
   auto first=plan->prepared.lock();
   worker->submitAndWait([&]{const auto inlineBefore=worker->submissions;
    RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,worker);
    RetainedGraphParticipantsReplayAccess::authenticateRecording(executor,graph,cache,worker);
    EXPECT_EQ(worker->submissions,inlineBefore);
   });
   EXPECT_EQ(first,plan->prepared.lock());EXPECT_EQ(plan->preparations,2u);EXPECT_EQ(plan->recordings,2u);
   cache.reset();EXPECT_THROW(first->requireLive(original),std::logic_error);
   first.reset();EXPECT_TRUE(plan->prepared.expired())<<"Only the original cache/scoped borrow may retain its token";
  }
 }
}
/** @test Null/foreign owners or cache streams reject before enqueue, and a broken worker queue cannot satisfy the inside-worker guard. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdOriginalPreparationRejectsForeignNullAndBrokenDispatch){
 using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(72);
 A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
 {
  const auto original=source->acquire(device);auto* worker=static_cast<ColdPreparationWorker*>(original.worker().get());auto plan=std::make_shared<ColdPreparationPlan>(original);
  ComputeGraph graph;graph.setRetainedParticipantSourcePlan(plan);DeviceGraphExecutor executor;BufferArena bindingArena;executor.setArena(&bindingArena);DeviceGraphExecutor::GraphSegmentCache cache;testing::MockWorkerGPUContext foreign(72);
  const auto before=worker->submissions;
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,nullptr),std::invalid_argument);
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,&foreign),std::invalid_argument);
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,worker),std::invalid_argument);
  EXPECT_EQ(worker->submissions,before);EXPECT_EQ(plan->preparations,0u);
  ASSERT_TRUE(cache.bindBorrowedCaptureStream(worker,reinterpret_cast<void*>(0x8171),device,false));
  worker->brokenDispatch=true;
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,worker),std::logic_error);
  EXPECT_FALSE(cache.participant_preparation_stream);EXPECT_EQ(plan->preparations,0u);worker->brokenDispatch=false;
  RetainedGraphParticipantsReplayAccess::bindPreparation(executor,graph,cache,worker);
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::authenticateRecording(executor,graph,cache,worker),std::invalid_argument);
  cache.retained_parent_capture=std::make_unique<testing::MockGPUGraphCapture>(reinterpret_cast<void*>(0x8172));
  ASSERT_TRUE(cache.retained_parent_capture->beginCapture());ASSERT_TRUE(cache.retained_parent_capture->endCapture());ASSERT_TRUE(cache.retained_parent_capture->instantiate());
  EXPECT_THROW(RetainedGraphParticipantsReplayAccess::authenticateRecording(executor,graph,cache,worker),std::invalid_argument);
  EXPECT_EQ(plan->recordings,0u);cache.reset();EXPECT_TRUE(plan->prepared.expired());
 }
}

/** @brief The stage observes actual recorder ordering without native SDK work. */
class SetupRecorderStage final : public testing::MockComputeStage {
public:
 SetupRecorderStage(DeviceId device,IWorkerGPUContext& worker,std::vector<std::string>& order,bool fail=false)
  :MockComputeStage(ComputeStageType::COPY,"setup",device),worker_(worker),order_(order),fail_(fail){}
 GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
 bool prepareGraphLaunch(IDeviceContext*,void* stream)override{
  EXPECT_TRUE(worker_.ownsCurrentThread());EXPECT_FALSE(worker_.isDeviceGraphCaptureActive());
  exact_=stream;order_.push_back("prepare");++preparations;return true;
 }
 bool execute(IDeviceContext*)override{
  EXPECT_TRUE(worker_.ownsCurrentThread());EXPECT_TRUE(worker_.isDeviceGraphCaptureActive());
  EXPECT_TRUE(isGraphCaptureActive());EXPECT_NE(currentGraphCaptureDependencyLedger(),nullptr);
  EXPECT_EQ(gpuStream(),exact_);order_.push_back("record");++records;return !fail_;
 }
 bool transitionGraphCaptureActivity(IDeviceContext*,void* stream,GraphCaptureActivityTransition edge)override{
  EXPECT_EQ(stream,exact_);order_.push_back(edge==GraphCaptureActivityTransition::Entering?"enter":
   edge==GraphCaptureActivityTransition::Completed?"finish":"abort");return true;
 }
 unsigned preparations=0,records=0;
private:IWorkerGPUContext& worker_;std::vector<std::string>& order_;bool fail_;void* exact_=nullptr;
};
/** @test One cold pass records exact setup addresses without executing or publishing request completion. */
TEST(RetainedGraphParticipantEnqueueGroup, SetupRecorderUsesOriginalWorkerAndNeverPublishesExecution){
 using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(81);
 A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
 auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
 {
  auto original=source->acquire(device);auto& worker=*original.worker();
  testing::MockDeviceContext ctx(device,ComputeBackendType::GPU_CUDA);
  // Real TensorBase/arena contracts expose accidental runtime coherence or
  // dirty publication. MockBackend owns only host-backed simulated storage.
  test::MockBackend backend(DeviceType::CUDA);
  FP32Tensor input(std::vector<size_t>{1,4},DeviceId::cpu());
  FP32Tensor output(std::vector<size_t>{1,4},DeviceId::cpu());
  input.setBackendForTesting(&backend);output.setBackendForTesting(&backend);
  ASSERT_TRUE(input.allocateOnDevice(device));ASSERT_TRUE(output.allocateOnDevice(device));
  ASSERT_FALSE(input.isDeviceValid());ASSERT_FALSE(output.isDeviceValid());
  const auto input_owner=input.getAuthoritativeDevice(),output_owner=output.getAuthoritativeDevice();
  BufferArena arena;DeviceGraphExecutor executor;executor.setArena(&arena);
  ASSERT_TRUE(arena.registerExternalBuffer(BufferId::HIDDEN_STATE,&input));
  ASSERT_TRUE(arena.registerExternalBuffer(BufferId::NORMALIZED,&output));
  std::vector<std::string> order;ComputeGraph graph;auto stage=std::make_unique<SetupRecorderStage>(device,worker,order);auto* observed=stage.get();
  stage->setBufferContract(StageBufferContract::build().addInput(BufferId::HIDDEN_STATE).addOutput(BufferId::NORMALIZED));
  graph.addNode("setup",std::move(stage),device);void* const exact=reinterpret_cast<void*>(0x8181);testing::MockGPUGraphCapture capture(exact);
  testing::MockDeviceContext foreign(DeviceId::rocm(81),ComputeBackendType::GPU_ROCM);
  testing::MockGPUGraphCapture missing(nullptr);
  EXPECT_THROW(executor.captureSetupGraphFragment(graph,&foreign,&capture,original),std::invalid_argument);
  EXPECT_THROW(executor.captureSetupGraphFragment(graph,&ctx,&missing,original),std::invalid_argument);
  for(const auto sentinel:{std::uintptr_t{1},std::uintptr_t{2}}){
   testing::MockGPUGraphCapture implicit(reinterpret_cast<void*>(sentinel));
   EXPECT_THROW(executor.captureSetupGraphFragment(graph,&ctx,&implicit,original),std::invalid_argument);
  }
  EXPECT_THROW(executor.captureSetupGraphFragment(graph,&ctx,&capture,original),std::logic_error);
  worker.submitAndWait([&]{ASSERT_TRUE(executor.captureSetupGraphFragment(graph,&ctx,&capture,original));});
  EXPECT_EQ(order,(std::vector<std::string>{"prepare","enter","record","finish"}));
  EXPECT_EQ(observed->preparations,1u);EXPECT_EQ(observed->records,1u);EXPECT_FALSE(graph.getNode("setup")->completed);
  EXPECT_FALSE(worker.isDeviceGraphCaptureActive());EXPECT_FALSE(capture.hasExecutable());EXPECT_EQ(capture.launchCount(),0);
  EXPECT_FALSE(input.isDeviceValid());EXPECT_FALSE(output.isDeviceValid());
  EXPECT_EQ(input.getAuthoritativeDevice(),input_owner);EXPECT_EQ(output.getAuthoritativeDevice(),output_owner);
  const auto transfers=backend.getTransferStats();
  EXPECT_EQ(transfers.h2d_count,0u);EXPECT_EQ(transfers.d2h_count,0u);EXPECT_EQ(transfers.d2d_count,0u);
  EXPECT_EQ(backend.getEventRecordCount(),0u);
  worker.submitAndWait([&]{EXPECT_FALSE(executor.captureSetupGraphFragment(graph,&ctx,&capture,original));});
  EXPECT_EQ(observed->records,1u);
 }
}
/** @test Failed setup closes native/software capture and aborts activity without completion or launch. */
TEST(RetainedGraphParticipantEnqueueGroup, SetupRecorderFailureAbortsWithoutCompletion){
 using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(81);
 A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
 auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
 {
  auto original=source->acquire(device);auto& worker=*original.worker();testing::MockDeviceContext ctx(device,ComputeBackendType::GPU_ROCM);
  BufferArena arena;DeviceGraphExecutor executor;executor.setArena(&arena);std::vector<std::string> order;ComputeGraph graph;
  graph.addNode("setup",std::make_unique<SetupRecorderStage>(device,worker,order,true),device);testing::MockGPUGraphCapture capture(reinterpret_cast<void*>(0x8182));
  worker.submitAndWait([&]{EXPECT_FALSE(executor.captureSetupGraphFragment(graph,&ctx,&capture,original));});
  EXPECT_EQ(order,(std::vector<std::string>{"prepare","enter","record","abort"}));EXPECT_FALSE(graph.getNode("setup")->completed);
  EXPECT_FALSE(worker.isDeviceGraphCaptureActive());EXPECT_FALSE(isGraphCaptureActive());EXPECT_FALSE(capture.hasExecutable());EXPECT_EQ(capture.nodeCount(),0u);EXPECT_EQ(capture.launchCount(),0);
 }
}

}

#include "LocalTPCaptureSourceDefinitionReplay.inc"
