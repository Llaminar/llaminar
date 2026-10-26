/**
 * @file Test__PLERetainedDevice.cpp
 * @brief Captured native PLE stages with immutable repeated-prefix publication.
 *
 * Two retained graphs model the ordered snapshot/retain transaction and its
 * accepted-prefix publication. A capture dependency ledger names every real
 * producer and defers event publication while recording. This is an explicit
 * native integration test, not a Unit or full model/controller lifetime proof.
 */
#include <gtest/gtest.h>
#include "QFNOuterPublicationTestSupport.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLERetainedStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include <array>
#include <cstring>
#include <stdexcept>
using namespace llaminar2;
namespace {
/** @brief Abort a native diagnostic on any failed infrastructure operation. */
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
/** @brief Retire both graphs before any captured tensor/stage owner can unwind. */
struct Retirement {
 IBackend*backend;DeviceId device;void*stream;
 std::unique_ptr<IGPUGraphCapture>transaction,publication;
 /** @brief Failed native completion is fatal; pointers cannot be released safely. */
 ~Retirement()noexcept {if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();publication.reset();transaction.reset();}
};
/** @brief Download only the requested diagnostic state interval after its exact stream edge. */
template<class T>std::vector<T>observe(IBackend&backend,DeviceId device,void*stream,const void*source,std::size_t elements){
 std::vector<T>values(elements);require(backend.deviceToHostOnStream(values.data(),source,elements*sizeof(T),device.ordinal,stream),"PLE native terminal copy failed");require(backend.synchronizeStream(stream,device.ordinal),"PLE native terminal completion failed");return values;
}
/** @brief Prove native snapshot-before-publication, repeated acceptance and exact payload counters. */
void prove(DeviceId device, bool outerLedger = false){
 auto*backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
 auto context=IDeviceContext::create(device,1);require(bool(context),"PLE native context failed");
 auto&worker=GPUDeviceContextPool::instance().getContext(device);
 worker.submitAsync([&]{
  void*stream=ExplicitGPUStream(worker.defaultStream()).get();
  for(int m:{1,15,512,1024})for(int channels:{8,10240}){
   using Stage=Qwen4ExpPLERetainedStage;
   INT32Tensor acceptedTokens({2}),priorTokens({2}),frozen({1}),tokens({m}),retainedTokens({m+1}),counters({4}),controls({2});
   FP32Tensor acceptedConv({9,channels}),priorConv({9,channels}),normalized({m,channels}),retainedNorm({m+1,channels});
   const std::array<int,2>originalTokens{100,101};std::memcpy(acceptedTokens.raw_mutable_data(),originalTokens.data(),8);
   for(int i=0;i<9*channels;++i)acceptedConv.mutable_data()[i]=float(-10000-i);
   const std::vector<float>originalConv(acceptedConv.data(),acceptedConv.data()+9*channels);
   for(int row=0;row<m;++row){static_cast<int*>(tokens.raw_mutable_data())[row]=300+row;for(int c=0;c<channels;++c)normalized.mutable_data()[std::size_t(row)*channels+c]=float(row*16+c);}
   std::memset(priorTokens.raw_mutable_data(),0xcd,priorTokens.size_bytes());std::memset(priorConv.raw_mutable_data(),0xcd,priorConv.size_bytes());
   auto*hostControls=static_cast<int*>(controls.raw_mutable_data());hostControls[0]=m;hostControls[1]=m-1;
   for(TensorBase*t:std::array<TensorBase*,11>{&acceptedTokens,&priorTokens,&frozen,&tokens,&retainedTokens,&counters,&controls,&acceptedConv,&priorConv,&normalized,&retainedNorm}){TransferEngine::prepareDeviceInput(t,device,stream);TransferEngine::requireDeviceInput(t,device,stream);}
   auto*deviceControls=static_cast<int*>(controls.gpu_data_ptr());
   Stage::Params params{.device_id=device,.operation=Stage::Operation::SnapshotPrior,.requests=1,.requestIndex=0,.rowCapacity=m,.channels=channels,
    .liveCounts=Qwen4ExpNativeIntInput{device,deviceControls,1},.transactionLiveCounts={frozen,BufferId::QFN_PLE_TRANSACTION_LIVE},
    .acceptedTokens={acceptedTokens,BufferId::QFN_PLE_TOKEN_HISTORY},.priorTokens={priorTokens,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY},.retainedTokens={retainedTokens,BufferId::QFN_PLE_RETAINED_TOKENS},
    .tokens=Qwen4ExpNativeIntInput{device,static_cast<int*>(tokens.gpu_data_ptr()),std::size_t(m)},
    .acceptedConvolution={acceptedConv,BufferId::QFN_PLE_CONV_HISTORY},.priorConvolution={priorConv,BufferId::QFN_PLE_PRIOR_CONV_HISTORY},
    .normalized={normalized,BufferId::QFN_PLE_NORMALIZED},.retainedNormalized={retainedNorm,BufferId::QFN_PLE_RETAINED_NORMALIZED},
    .copiedBytes={counters,BufferId::QFN_PLE_TRANSACTION_COPY_BYTES},.verifierOrigins=RequestRowOrigins::fixedStride(0,m)};
   Stage snapshot(params);params.operation=Stage::Operation::RetainCandidates;params.copiedBytes.byteOffset=8;Stage retain(params);
   snapshot.setGPUStream(stream);retain.setGPUStream(stream);
   const auto beforePrepare=snapshot.graphCaptureVariantSignature();
   require(snapshot.prepareGraphLaunch(context.get(),stream)&&retain.prepareGraphLaunch(context.get(),stream),"PLE prepared storage seal failed");
   EXPECT_EQ(beforePrepare,snapshot.graphCaptureVariantSignature());
   Retirement retirement{backend,device,stream,worker.createGraphCapture(stream),worker.createGraphCapture(stream)};
   using Plan=GraphCaptureDependencyLedger::StagePlan;
   GraphCaptureDependencyLedger transactionLedger(device,stream,{
    Plan{.stage_identity=&snapshot,.stage_name="snapshot_prior",.external_inputs={&acceptedTokens,&acceptedConv},.outputs={&priorTokens,&priorConv,&frozen,&counters}},
    Plan{.stage_identity=&retain,.stage_name="retain_candidates",.external_inputs={&normalized},.internal_inputs={{&priorTokens,0},{&priorConv,0},{&frozen,0}},.outputs={&retainedTokens,&retainedNorm,&counters}}},"PLE native transaction");
   {ScopedBackendGraphCapture capture(worker,*retirement.transaction,"PLE native transaction",&transactionLedger);require(capture.begin(),"PLE snapshot capture begin failed");
    {ScopedGraphCaptureStage scope(&snapshot);require(snapshot.execute(context.get()),"PLE snapshot capture failed");for(TensorBase*t:std::array<TensorBase*,4>{&priorTokens,&priorConv,&frozen,&counters})TransferEngine::publishDeviceWrite(t,device,stream);scope.complete();}
    {ScopedGraphCaptureStage scope(&retain);require(retain.execute(context.get()),"PLE retain capture failed");for(TensorBase*t:std::array<TensorBase*,3>{&retainedTokens,&retainedNorm,&counters})TransferEngine::publishDeviceWrite(t,device,stream);scope.complete();}
    capture.finish();}
   MTPSpeculativeStatePublicationStage::Params outerParams;
   outerParams.device_id = device;
   outerParams.authority = MTPSpeculativeStatePublicationStage::Authority::PipelineFollower;
   outerParams.verifier_state_stages = {&retain};
   MTPSpeculativeStatePublicationStage outer(outerParams);
   auto outerPlan = qfn_outer_publication_test::plan(outer, {
       {BufferId::QFN_PLE_TOKEN_HISTORY, &acceptedTokens},
       {BufferId::QFN_PLE_CONV_HISTORY, &acceptedConv}});
   if (outerLedger)
   {
       ASSERT_EQ(outerPlan.outputs.size(), 2u);
       outerPlan.retained_parent_inputs = {&priorTokens, &priorConv, &frozen,
           &retainedTokens, &retainedNorm};
       qfn_outer_publication_test::rejectMissingWrites(worker, device, stream, outerPlan, [&] {
           require(retain.restoreVerifierStateCaptureRowFromDeviceIndex(std::span<const int>(deviceControls, controls.numel()).subspan(1), stream),
               "PLE negative outer hook failed without a ledger diagnostic");
       });
   }
   GraphCaptureDependencyLedger outerPublicationLedger(device, stream, {outerPlan},
       "PLE outer MTP publication");
   GraphCaptureDependencyLedger publicationLedger(device,stream,{Plan{.stage_identity=&retain,.stage_name="publish_accepted",.external_inputs={&acceptedTokens,&acceptedConv,&normalized},.retained_parent_inputs={&priorTokens,&priorConv,&frozen,&retainedTokens,&retainedNorm,&counters},.outputs={&acceptedTokens,&acceptedConv}}},"PLE native publication");
   {ScopedBackendGraphCapture capture(worker,*retirement.publication,"PLE native publication",outerLedger ? &outerPublicationLedger : &publicationLedger);require(capture.begin(),"PLE publication capture begin failed");ScopedGraphCaptureStage scope(outerLedger ? static_cast<const void *>(&outer) : static_cast<const void *>(&retain));require(retain.restoreVerifierStateCaptureRowFromDeviceIndex(std::span<const int>(deviceControls, controls.numel()).subspan(1),stream),"PLE publication capture failed");scope.complete();capture.finish();}
   for(auto*graph:{retirement.transaction.get(),retirement.publication.get()}){std::vector<GPUGraphKernelNodeInfo>nodes;std::string error;require(graph->inspectKernelNodes(nodes,&error),error.c_str());ASSERT_EQ(nodes.size(),2u);for(const auto&node:nodes)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);require(graph->instantiate(),"PLE native graph instantiation failed");}
   require(retirement.transaction->launch(),"PLE native transaction replay failed");
   auto bytes=observe<std::uint64_t>(*backend,device,stream,counters.gpu_data_ptr(),2);EXPECT_EQ(bytes[0],8+std::size_t(channels)*36);EXPECT_EQ(bytes[1],std::size_t(m)*(channels+1)*4);
   const int nextLiveInput=0;require(backend->hostToDeviceOnStream(deviceControls,&nextLiveInput,4,device.ordinal,stream),"PLE mutable next-input count publication failed");
   const auto snapshotSignature=snapshot.graphCaptureVariantSignature(),retainSignature=retain.graphCaptureVariantSignature();
   for(int accepted:{m,m>1?2:0,0,m>7?7:m,0}){
    const int flat=accepted-1;require(backend->hostToDeviceOnStream(deviceControls+1,&flat,4,device.ordinal,stream),"PLE native decision publication failed");require(retirement.publication->launch(),"PLE accepted replay failed");
    const auto actualTokens=observe<int>(*backend,device,stream,acceptedTokens.gpu_data_ptr(),2);const auto actualConv=observe<float>(*backend,device,stream,acceptedConv.gpu_data_ptr(),std::size_t(channels)*9);
    for(int i=0;i<2;++i){const int source=accepted-2+i;EXPECT_EQ(actualTokens[i],source<0?originalTokens[source+2]:300+source);}
    for(int row=0;row<9;++row){const int source=accepted-9+row;const float*expected=source<0?originalConv.data()+std::size_t(source+9)*channels:normalized.data()+std::size_t(source)*channels;EXPECT_EQ(std::memcmp(actualConv.data()+std::size_t(row)*channels,expected,std::size_t(channels)*4),0);}
    auto immutable=observe<float>(*backend,device,stream,priorConv.gpu_data_ptr(),std::size_t(channels)*9);EXPECT_EQ(std::memcmp(immutable.data(),originalConv.data(),immutable.size()*4),0);
    EXPECT_EQ(snapshotSignature,snapshot.graphCaptureVariantSignature());EXPECT_EQ(retainSignature,retain.graphCaptureVariantSignature());
   }
   // A new empty transaction follows the completed prior reader/publication
   // stream. It reports no payload and neither snapshots nor publishes history.
   const int empty=0;require(backend->hostToDeviceOnStream(deviceControls,&empty,4,device.ordinal,stream),"PLE empty count publication failed");
   require(retirement.transaction->launch(),"PLE empty transaction failed");require(retirement.publication->launch(),"PLE empty publication failed");
   bytes=observe<std::uint64_t>(*backend,device,stream,counters.gpu_data_ptr(),2);EXPECT_EQ(bytes[0],0u);EXPECT_EQ(bytes[1],0u);
   const auto unchanged=observe<float>(*backend,device,stream,acceptedConv.gpu_data_ptr(),std::size_t(channels)*9);EXPECT_EQ(std::memcmp(unchanged.data(),originalConv.data(),unchanged.size()*4),0);
   const int small=m>3?3:m;const std::array<int,2>smallControls{small,small-1};
   auto*tokenGuard=static_cast<int*>(retainedTokens.gpu_data_ptr())+small;
   auto*normGuard=static_cast<float*>(retainedNorm.gpu_data_ptr())+std::size_t(small)*channels;
   const int tokenPoison=-7777;const float normPoison=-7777.0f;
   require(backend->hostToDeviceOnStream(tokenGuard,&tokenPoison,4,device.ordinal,stream),"PLE token guard publication failed");
   require(backend->hostToDeviceOnStream(normGuard,&normPoison,4,device.ordinal,stream),"PLE normalized guard publication failed");
   require(backend->hostToDeviceOnStream(deviceControls,smallControls.data(),8,device.ordinal,stream),"PLE small controls publication failed");
   require(retirement.transaction->launch(),"PLE small transaction failed");require(retirement.publication->launch(),"PLE small publication failed");
   bytes=observe<std::uint64_t>(*backend,device,stream,counters.gpu_data_ptr(),2);EXPECT_EQ(bytes[0],8+std::size_t(channels)*36);EXPECT_EQ(bytes[1],std::size_t(small)*(channels+1)*4);
   EXPECT_EQ(observe<int>(*backend,device,stream,tokenGuard,1)[0],tokenPoison);EXPECT_EQ(observe<float>(*backend,device,stream,normGuard,1)[0],normPoison);
   const auto smallTail=observe<float>(*backend,device,stream,acceptedConv.gpu_data_ptr(),std::size_t(channels)*9);
   for(int row=0;row<9;++row){const int source=small-9+row;const float*expected=source<0?originalConv.data()+std::size_t(source+9)*channels:normalized.data()+std::size_t(source)*channels;EXPECT_EQ(std::memcmp(smallTail.data()+std::size_t(row)*channels,expected,std::size_t(channels)*4),0);}

  }
 }).get();
}
}
#ifdef HAVE_CUDA
TEST(PLERetainedDevice, CUDACapturedPublication){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(PLERetainedDevice, ROCmCapturedPublication){prove(DeviceId::rocm(0));}
#endif

#ifdef HAVE_CUDA
/** @test Actual outer MTP write authority preserves captured accepted-prefix bytes. */
TEST(PLERetainedDevice, CUDAOuterMTPPublicationLedger)
{
    prove(DeviceId::cuda(0), true);
}
#endif
#ifdef HAVE_ROCM
/** @test Actual outer MTP write authority preserves captured accepted-prefix bytes. */
TEST(PLERetainedDevice, ROCmOuterMTPPublicationLedger)
{
    prove(DeviceId::rocm(0), true);
}
#endif
