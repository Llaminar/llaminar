/**
 * @file Test__Qwen4ExpColdStagePreparation.cpp
 * @brief Actual cold executor allocation for write-only QFN primitive and PLE roots.
 *
 * Arena tensors initially own only host storage. The ordinary executor must
 * materialize exact declared output owners before storage sealing and capture;
 * tests never pre-upload output roots or substitute a setup allocation path.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLERetainedStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "../../../utils/GraphArenaTestHarness.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_set>
#include <array>
#include <cstring>
#include <functional>
#include <stdexcept>
using namespace llaminar2;
namespace {
/** @brief Preserve the exact infrastructure failure at this diagnostic boundary. */
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
/** @brief Complete captured readers before destroying executable/arena owners. */
struct Retirement {
 IBackend&backend;DeviceId device;void*stream;DeviceGraphExecutor::GraphSegmentCache&cache;
 /** @brief Failure to retire native work is fatal, never permission to release roots. */
 ~Retirement()noexcept{if(!backend.synchronizeStream(cache.capture_stream?cache.capture_stream:stream,device.ordinal))std::terminate();cache.reset();}
};
/** @brief Read a diagnostic result only after the exact native capture stream completes. */
template<class T>std::vector<T>observe(IBackend&backend,DeviceId device,void*stream,TensorBase&tensor){
 std::vector<T> result(tensor.size_bytes()/sizeof(T));require(backend.deviceToHostOnStream(result.data(),tensor.gpu_data_ptr(),tensor.size_bytes(),device.ordinal,stream),"Cold stage terminal copy failed");require(backend.synchronizeStream(stream,device.ordinal),"Cold stage terminal completion failed");return result;
}
/** @brief Run the same actual executor executable repeatedly without output setup or rebinding. */
void coldGraph(DeviceId device,IDeviceContext&context,IWorkerGPUContext&worker,test::GraphArenaTestHarness&arena,
 std::unique_ptr<IComputeStage>owned,const std::vector<TensorBase*>&roots,const std::function<void(void*)>&verify){
 auto*backend=getBackendFor(device);const auto declared=owned->graphCaptureVariantSignature();auto*stage=owned.get();
 for(auto*root:roots)ASSERT_EQ(root->gpu_data_ptr(),nullptr);
 ComputeGraph graph;graph.addNode("cold_qfn_stage",std::move(owned),device);
 GraphExecutorConfig config;config.enable_validation=false;DeviceGraphExecutor executor(config);arena.bindExecutor(executor);
 DeviceGraphExecutor::GraphSegmentCache cache;void*stream=ExplicitGPUStream(worker.defaultStream()).get();Retirement retirement{*backend,device,stream,cache};
 // Only true external reads have a host producer in this isolated graph.
 // Ordinary admission uploads them; distinct write-only outputs stay cold.
 std::unordered_set<ITensor*>admitted;
 for(const auto&binding:stage->bufferContract().allArenaReads()){
  auto*t=arena.tensor(binding.id);if(admitted.insert(t).second)TransferEngine::prepareDeviceInput(t,device,stream);}
 for(const auto&binding:stage->bufferContract().outputs){auto*t=arena.tensor(binding.id);
  if(!admitted.contains(t))ASSERT_EQ(t->gpu_data_ptr(),nullptr);}
 require(executor.executeWithCachedGraphReplay(graph,&context,cache,stream,&worker,nullptr),"Cold QFN executor capture failed");
 ASSERT_TRUE(cache.initialized);ASSERT_FALSE(cache.needs_capture);ASSERT_EQ(cache.segments.size(),1u);EXPECT_EQ(cache.variant_recapture_count,0u);
 std::vector<const void*>addresses;for(auto*root:roots){ASSERT_NE(root->gpu_data_ptr(),nullptr);EXPECT_EQ(root->current_device(),std::optional<DeviceId>(device));addresses.push_back(root->gpu_data_ptr());}
 const auto*executable=cache.segments.front().capture.get();verify(cache.capture_stream);
 for(int replay=0;replay<3;++replay){graph.reset();require(executor.executeWithCachedGraphReplay(graph,&context,cache,stream,&worker,nullptr),"Cold QFN retained replay failed");
  EXPECT_EQ(cache.variant_recapture_count,0u);EXPECT_EQ(cache.segments.front().capture.get(),executable);EXPECT_EQ(stage->graphCaptureVariantSignature(),declared);
  for(std::size_t i=0;i<roots.size();++i)EXPECT_EQ(roots[i]->gpu_data_ptr(),addresses[i]);verify(cache.capture_stream);}
}
/** @brief A primitive's distinct write-only wide output has no downstream reader to allocate it. */
void provePrimitive(DeviceId device){
 auto*backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
 auto context=IDeviceContext::create(device,1);auto&worker=GPUDeviceContextPool::instance().getContext(device);
 worker.submitAsync([&]{constexpr int m=3,h=7;test::GraphArenaTestHarness arena;
  auto*input=arena.createPersistentTensor<FP32Tensor>(BufferId::QFN_EMBEDDING_ROWS,std::vector<std::size_t>{m,h});
  auto*output=arena.createPersistentTensor<FP32Tensor>(BufferId::HIDDEN_STATE,std::vector<std::size_t>{m,4*h});
  for(int i=0;i<m*h;++i)input->mutable_data()[i]=float(i-9);std::fill_n(output->mutable_data(),output->numel(),-777.f);
  auto stage=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{.device_id=device,.rows={m,{}},
   .operation=Qwen4ExpPrimitiveStage::ResidualExpand{h,{*input,BufferId::QFN_EMBEDDING_ROWS},{*output,BufferId::HIDDEN_STATE}}});
  coldGraph(device,*context,worker,arena,std::move(stage),{input,output},[&](void*stream){const auto actual=observe<float>(*backend,device,stream,*output);
   for(int row=0;row<m;++row)for(int branch=0;branch<4;++branch)EXPECT_EQ(std::memcmp(actual.data()+(row*4+branch)*h,input->data()+row*h,h*4),0);});
 }).get();
}
/** @brief Snapshot and retention separately materialize every cold output before their physical seals. */
void provePLE(DeviceId device,Qwen4ExpPLERetainedStage::Operation operation){
 auto*backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
 auto context=IDeviceContext::create(device,1);auto&worker=GPUDeviceContextPool::instance().getContext(device);
 worker.submitAsync([&]{constexpr int m=3,c=7;test::GraphArenaTestHarness arena;
  const auto integer=[&](BufferId id,int n){return arena.createPersistentTensor<INT32Tensor>(id,std::vector<std::size_t>{std::size_t(n)});};
  const auto floating=[&](BufferId id,int rows){return arena.createPersistentTensor<FP32Tensor>(id,std::vector<std::size_t>{std::size_t(rows),c});};
  auto*live=integer(BufferId::QFN_PLE_TRANSACTION_LIVE,1);auto*acceptedT=integer(BufferId::QFN_PLE_TOKEN_HISTORY,2);
  auto*priorT=integer(BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY,2);auto*tokens=integer(BufferId::REQUEST_TOKEN_IDS,m);
  auto*retainedT=integer(BufferId::QFN_PLE_RETAINED_TOKENS,m);auto*bytes=integer(BufferId::QFN_PLE_TRANSACTION_COPY_BYTES,2);
  auto*acceptedC=floating(BufferId::QFN_PLE_CONV_HISTORY,9);auto*priorC=floating(BufferId::QFN_PLE_PRIOR_CONV_HISTORY,9);
  auto*normalized=floating(BufferId::QFN_PLE_NORMALIZED,m);auto*retainedC=floating(BufferId::QFN_PLE_RETAINED_NORMALIZED,m);
  *static_cast<int*>(live->raw_mutable_data())=m;auto*accepted=static_cast<int*>(acceptedT->raw_mutable_data());accepted[0]=17;accepted[1]=19;
  for(int i=0;i<m;++i)static_cast<int*>(tokens->raw_mutable_data())[i]=100+i;
  for(int i=0;i<9*c;++i){acceptedC->mutable_data()[i]=float(-1000-i);priorC->mutable_data()[i]=float(-2000-i);}
  for(int i=0;i<m*c;++i)normalized->mutable_data()[i]=float(1000+i);
  using Stage=Qwen4ExpPLERetainedStage;
  auto stage=std::make_unique<Stage>(Stage::Params{.device_id=device,.operation=operation,.requests=1,.requestIndex=0,.rowCapacity=m,.channels=c,
   .transactionLiveCounts={*live,BufferId::QFN_PLE_TRANSACTION_LIVE},.acceptedTokens={*acceptedT,BufferId::QFN_PLE_TOKEN_HISTORY},
   .priorTokens={*priorT,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY},.retainedTokens={*retainedT,BufferId::QFN_PLE_RETAINED_TOKENS},
   .tokens={*tokens,BufferId::REQUEST_TOKEN_IDS},.acceptedConvolution={*acceptedC,BufferId::QFN_PLE_CONV_HISTORY},
   .priorConvolution={*priorC,BufferId::QFN_PLE_PRIOR_CONV_HISTORY},.normalized={*normalized,BufferId::QFN_PLE_NORMALIZED},
   .retainedNormalized={*retainedC,BufferId::QFN_PLE_RETAINED_NORMALIZED},.copiedBytes={*bytes,BufferId::QFN_PLE_TRANSACTION_COPY_BYTES},
   .verifierOrigins=RequestRowOrigins::fixedStride(0,m)});
  std::vector<TensorBase*>roots;
  const auto appendRoot=[&](const ITensor*tensor){auto*root=dynamic_cast<const TensorBase*>(tensor);
   require(root!=nullptr,"Cold fixture lost its canonical TensorBase owner");roots.push_back(const_cast<TensorBase*>(root));};
  const auto dump=stage->getDumpInfoSnapshot();for(const auto&b:dump.inputs)appendRoot(b.tensor);
  for(const auto&b:dump.outputs)appendRoot(b.tensor);
  coldGraph(device,*context,worker,arena,std::move(stage),roots,[&](void*stream){
   EXPECT_EQ(observe<std::uint64_t>(*backend,device,stream,*bytes)[0],operation==Stage::Operation::SnapshotPrior?8u+36u*c:std::uint64_t(m)*(c+1)*4);
   if(operation==Stage::Operation::SnapshotPrior){EXPECT_EQ(observe<int>(*backend,device,stream,*live)[0],m);
    const auto t=observe<int>(*backend,device,stream,*priorT);EXPECT_EQ(t[0],17);EXPECT_EQ(t[1],19);
    const auto values=observe<float>(*backend,device,stream,*priorC);EXPECT_EQ(std::memcmp(values.data(),acceptedC->data(),values.size()*4),0);
   }else{const auto t=observe<int>(*backend,device,stream,*retainedT);for(int i=0;i<m;++i)EXPECT_EQ(t[i],100+i);
    const auto values=observe<float>(*backend,device,stream,*retainedC);EXPECT_EQ(std::memcmp(values.data(),normalized->data(),values.size()*4),0);}
  });
 }).get();
}
}
#ifdef HAVE_CUDA
TEST(Qwen4ExpColdStagePreparation, CUDAPrimitive){provePrimitive(DeviceId::cuda(0));}
TEST(Qwen4ExpColdStagePreparation, CUDAPLESnapshot){provePLE(DeviceId::cuda(0),Qwen4ExpPLERetainedStage::Operation::SnapshotPrior);}
TEST(Qwen4ExpColdStagePreparation, CUDAPLERetain){provePLE(DeviceId::cuda(0),Qwen4ExpPLERetainedStage::Operation::RetainCandidates);}
#endif
#ifdef HAVE_ROCM
TEST(Qwen4ExpColdStagePreparation, ROCmPrimitive){provePrimitive(DeviceId::rocm(0));}
TEST(Qwen4ExpColdStagePreparation, ROCmPLESnapshot){provePLE(DeviceId::rocm(0),Qwen4ExpPLERetainedStage::Operation::SnapshotPrior);}
TEST(Qwen4ExpColdStagePreparation, ROCmPLERetain){provePLE(DeviceId::rocm(0),Qwen4ExpPLERetainedStage::Operation::RetainCandidates);}
#endif
