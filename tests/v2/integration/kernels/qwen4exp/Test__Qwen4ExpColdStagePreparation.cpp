/**
 * @file Test__Qwen4ExpColdStagePreparation.cpp
 * @brief Actual cold executor allocation for write-only QFN primitive and PLE roots.
 *
 * Arena tensors initially own only host storage. The ordinary executor must
 * materialize exact declared output owners before storage sealing and capture;
 * tests never pre-upload output roots or substitute a setup allocation path.
 * The hash replay also changes tokens, accepted history and live rows while
 * retaining the original executable, prepared addresses and capture identity.
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
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/RequestInputPositionAdmission.h"
#include <numeric>
#include "kernels/KernelFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <unordered_set>
#include <array>
#include <bit>
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
/**
 * @brief Seal a genuinely cold hash output through the production executor and replay changing input data.
 * @param device Exact CUDA or ROCm participant selected by its native registration.
 *
 * The first full replay establishes every output byte without pre-uploading the
 * write-only result. Later partial/empty replays poison that same storage at the
 * explicit test-ingress boundary. Changed tokens, accepted history and live
 * count must affect output without rebinding roots or reconstructing the graph.
 */
void proveHash(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
        GTEST_SKIP() << "Requested backend unavailable";
    auto context = IDeviceContext::create(device, 1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    auto cpu = llaminar::v2::kernels::KernelFactory::createQwen4Exp(DeviceId::cpu());
    worker.submitAsync([&] {
        constexpr int capacity = 4;
        constexpr std::array<int, 4> liveRows{capacity, 1, 0, 3};
        constexpr auto poison = std::bit_cast<std::int64_t>(std::uint64_t{0xa5a5a5a5a5a5a5a5});
        test::GraphArenaTestHarness arena;
        const auto integer = [&](BufferId id, std::size_t count) {
            return arena.createPersistentTensor<INT32Tensor>(id, std::vector<std::size_t>{count});
        };
        auto *tokens = integer(BufferId::REQUEST_TOKEN_IDS, capacity);
        auto *history = integer(BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY, 2);
        auto *count = integer(BufferId::QFN_PLE_TRANSACTION_LIVE, 1);
        auto *ids = integer(BufferId::QFN_NGRAM_ROW_IDS, capacity * 32);
        qwen4exp::NgramHashPolicy policy;
        policy.multipliers = {3, 5, 7};
        for (unsigned head = 0; head < 16; ++head) {
            policy.moduli[head] = 17;
            policy.offsets[head] = (std::uint64_t{1} << 33) + head * 64;
        }
        std::array<std::int32_t, capacity> tokenValues{};
        std::array<std::int32_t, 2> priorValues{};
        const auto setHostInputs = [&](std::size_t replay) {
            for (int row = 0; row < capacity; ++row)
                tokenValues[row] = row == 2 ? static_cast<std::int32_t>(policy.eosToken)
                                           : static_cast<std::int32_t>(17 + row + 31 * replay);
            priorValues = {static_cast<std::int32_t>(7 + replay), static_cast<std::int32_t>(11 + replay)};
            std::copy(tokenValues.begin(), tokenValues.end(), static_cast<std::int32_t *>(tokens->raw_mutable_data()));
            std::copy(priorValues.begin(), priorValues.end(), static_cast<std::int32_t *>(history->raw_mutable_data()));
            *static_cast<std::int32_t *>(count->raw_mutable_data()) = liveRows[replay];
        };
        setHostInputs(0);
        using Stage = Qwen4ExpPrimitiveStage;
        using Int = Qwen4ExpIntBinding;
        auto stage = std::make_unique<Stage>(Stage::Params{
            .device_id = device,
            .rows = {capacity, Int(*count, BufferId::QFN_PLE_TRANSACTION_LIVE)},
            .operation = Stage::NgramHash{
                Int(*tokens, BufferId::REQUEST_TOKEN_IDS),
                Int(*history, BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY),
                Int(*ids, BufferId::QFN_NGRAM_ROW_IDS), policy}});
        ASSERT_NE(stage->graphCaptureVariantSignature(), 0u);
        EXPECT_EQ(stage->graphLaunchPreparationPolicy(), GraphLaunchPreparationPolicy::CaptureOnly);
        std::size_t replay = 0;
        coldGraph(device, *context, worker, arena, std::move(stage), {tokens, history, count, ids},
            [&](void *stream) {
                ASSERT_LT(replay, liveRows.size());
                std::vector<std::int64_t> expected(capacity * 16, poison);
                cpu->ngramHash({{capacity, &liveRows[replay]}, tokenValues.data(), priorValues.data(),
                    expected.data(), policy}, nullptr);
                EXPECT_EQ(observe<std::int64_t>(*backend, device, stream, *ids), expected);
                if (++replay == liveRows.size()) return;
                setHostInputs(replay);
                for (auto *input : {tokens, history, count})
                    TransferEngine::prepareDeviceInput(input, device, stream);
                // Explicit fixture ingress poisons existing output bytes only.
                // It neither reallocates the sealed root nor runs in capture.
                require(backend->memset(ids->gpu_data_ptr(), 0xa5, ids->size_bytes(), device.ordinal, stream),
                    "Hash replay output poison failed");
            });
        EXPECT_EQ(replay, liveRows.size());
    }).get();
}
/** @brief Same production Hash stage reuses a borrowed original request view through partial/empty/reset data. */
void proveNextWindowHash(DeviceId device)
{
    auto* backend=getBackendFor(device);
    require(backend && backend->deviceCount()>device.ordinal,"Requested next-window native backend unavailable");
    auto context=IDeviceContext::create(device,1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        constexpr int capacity=12,bucket=4;
        test::GraphArenaTestHarness arena;
        const auto integer=[&](BufferId id,std::size_t count){return arena.createPersistentTensor<INT32Tensor>(id,std::vector<std::size_t>{count});};
        auto* tokens=integer(BufferId::REQUEST_TOKEN_IDS,capacity);
        auto* positions=integer(BufferId::REQUEST_POSITION_IDS,capacity);
        auto* total=integer(BufferId::REQUEST_BATCH_GEOMETRY,1);
        auto* current=integer(BufferId::PREFILL_CHUNK_POSITION_IDS,bucket);
        auto* count=integer(BufferId::PREFILL_CHUNK_GEOMETRY,1);
        auto* history=integer(BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY,2);
        auto* ids=integer(BufferId::QFN_NGRAM_ROW_IDS,bucket*32);
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        std::array<std::int32_t,capacity> hostTokens{},hostPositions{};
        std::iota(hostPositions.begin(),hostPositions.end(),100);
        for(int i=0;i<capacity;++i)hostTokens[i]=i%3==0?248044:31+i;
        const std::array<std::int32_t,2> prior{11,248044};
        std::copy(hostTokens.begin(),hostTokens.end(),static_cast<int*>(tokens->raw_mutable_data()));
        std::copy(hostPositions.begin(),hostPositions.end(),static_cast<int*>(positions->raw_mutable_data()));
        std::copy(prior.begin(),prior.end(),static_cast<int*>(history->raw_mutable_data()));
        *static_cast<int*>(total->raw_mutable_data())=capacity;*static_cast<int*>(current->raw_mutable_data())=100;
        *static_cast<int*>(count->raw_mutable_data())=bucket;
        for(auto* input:{tokens,positions,total,current,count,history}){
            TransferEngine::prepareDeviceInput(input,device,stream);TransferEngine::requireDeviceInput(input,device,stream);
        }
        DevicePrefillChunkGraphBinding original{.backend=backend,
            .request_token_ids_device=static_cast<const int*>(tokens->gpu_data_ptr()),
            .request_position_ids_device=static_cast<const int*>(positions->gpu_data_ptr()),
            .request_total_rows_device=static_cast<const int*>(total->gpu_data_ptr()),
            .cached_tokens_device=static_cast<const int*>(current->gpu_data_ptr()),
            .chunk_token_ids_device=static_cast<int*>(tokens->gpu_data_ptr()),
            .chunk_position_ids_device=static_cast<int*>(current->gpu_data_ptr()),
            .chunk_real_rows_device=static_cast<int*>(count->gpu_data_ptr()),
            .chunk_row_stride_device=static_cast<int*>(count->gpu_data_ptr()),
            .request_row_capacity=capacity,.bucket_seq_len=bucket,.capture_identity=0x4e57494e};
        const auto window=qwen4exp::Qwen4ExpPrefillNextWindowBinding::fromOriginalChunk(original,device,bucket,
            RequestInputPositionLayoutRequirement::ContiguousSingleRequestPrefill);
        using Stage=Qwen4ExpPrimitiveStage;using Int=Qwen4ExpIntBinding;
        qwen4exp::NgramHashPolicy policy;policy.multipliers={3,5,7};
        for(unsigned head=0;head<16;++head){policy.moduli[head]=17;policy.offsets[head]=(std::uint64_t{1}<<33)+head*64;}
        auto stage=std::make_unique<Stage>(Stage::Params{.device_id=device,.rows={bucket,{}},
            .operation=Stage::NgramHash{Qwen4ExpNativeIntInput{device,static_cast<const int*>(tokens->gpu_data_ptr()),capacity},
                Int(*history,BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY),Int(*ids,BufferId::QFN_NGRAM_ROW_IDS),policy,window}});
        auto* raw=stage.get();const auto identity=raw->graphCaptureVariantSignature();
        ComputeGraph graph;graph.addNode("actual_next_window_hash",std::move(stage),device);
        GraphExecutorConfig config;config.enable_validation=false;DeviceGraphExecutor executor(config);arena.bindExecutor(executor);
        DeviceGraphExecutor::GraphSegmentCache cache;Retirement retirement{*backend,device,stream,cache};
        auto cpu=qwen4exp::createCPUQwen4ExpKernels();
        for(unsigned replay=0;replay<20;++replay)for(const auto values:std::array{
            std::array{0,12,4},std::array{4,10,4},std::array{8,10,2},std::array{0,4,1}}){
            const int first=100+values[0],logical=values[1],live=values[2];
            *static_cast<int*>(current->raw_mutable_data())=first;
            *static_cast<int*>(total->raw_mutable_data())=logical;*static_cast<int*>(count->raw_mutable_data())=live;
            // The retained graph executes on its own exact capture stream.
            // Later fixture publications and output poisoning must precede
            // that replay there, rather than race it on the setup stream.
            const auto replayStream=cache.initialized?cache.capture_stream:stream;
            for(auto* input:{current,total,count})TransferEngine::prepareDeviceInput(input,device,replayStream);
            if(ids->gpu_data_ptr())require(backend->memset(ids->gpu_data_ptr(),0xa5,ids->size_bytes(),device.ordinal,replayStream),"Next-window poison failed");
            graph.reset();require(executor.executeWithCachedGraphReplay(graph,context.get(),cache,stream,&worker,nullptr),"Next-window captured replay failed");
            EXPECT_EQ(cache.variant_recapture_count,0u);EXPECT_EQ(raw->graphCaptureVariantSignature(),identity);
            auto actual=observe<std::int64_t>(*backend,device,cache.capture_stream,*ids);
            const int next=values[0]+live,nextRows=std::min(bucket,logical-next);
            const auto poison=std::bit_cast<std::int64_t>(std::uint64_t{0xa5a5a5a5a5a5a5a5});
            std::vector<std::int64_t> expected(bucket*16,poison);
            // First capture has no prior poison publication; only active rows
            // are compared there. Every later suffix is explicitly protected.
            if(nextRows)cpu->ngramHash({{nextRows},hostTokens.data()+next,prior.data(),expected.data(),policy},nullptr);
            EXPECT_EQ(std::memcmp(actual.data(),expected.data(),nextRows*16*8),0)
                <<"replay="<<replay<<" current_first="<<first<<" current_rows="<<live
                <<" next_first="<<next<<" next_rows="<<nextRows
                <<" first_actual="<<actual.front()<<" first_expected="<<expected.front();
            if(replay || values[0])for(std::size_t i=nextRows*16;i<actual.size();++i)EXPECT_EQ(actual[i],poison);
            EXPECT_EQ(observe<std::int32_t>(*backend,device,cache.capture_stream,*count).front(),live);
        }
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
TEST(Qwen4ExpColdStagePreparation, CUDANgramHash){proveHash(DeviceId::cuda(0));}
TEST(Qwen4ExpColdStagePreparation, CUDAPLESnapshot){provePLE(DeviceId::cuda(0),Qwen4ExpPLERetainedStage::Operation::SnapshotPrior);}
TEST(Qwen4ExpColdStagePreparation, CUDAPLERetain){provePLE(DeviceId::cuda(0),Qwen4ExpPLERetainedStage::Operation::RetainCandidates);}
#endif
#ifdef HAVE_ROCM
TEST(Qwen4ExpColdStagePreparation, ROCmPrimitive){provePrimitive(DeviceId::rocm(0));}
TEST(Qwen4ExpColdStagePreparation, ROCmNgramHash){proveHash(DeviceId::rocm(0));}
TEST(Qwen4ExpColdStagePreparation, ROCmPLESnapshot){provePLE(DeviceId::rocm(0),Qwen4ExpPLERetainedStage::Operation::SnapshotPrior);}
TEST(Qwen4ExpColdStagePreparation, ROCmPLERetain){provePLE(DeviceId::rocm(0),Qwen4ExpPLERetainedStage::Operation::RetainCandidates);}
#endif

#ifdef HAVE_CUDA
TEST(Qwen4ExpColdStagePreparation,CUDANgramHashNextWindow){proveNextWindowHash(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(Qwen4ExpColdStagePreparation,ROCmNgramHashNextWindow){proveNextWindowHash(DeviceId::rocm(0));}
#endif
