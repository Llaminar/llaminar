/** @file Test__QwenSparseRetainedDevice.cpp @brief Captured actual retained-stage publication with native request holes. */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <cstring>
#include <memory>
#include <stdexcept>
using namespace llaminar2;
namespace {
using Stage=QwenSparseAttentionStage;using Float=Qwen4ExpFloatBinding;using Int=Qwen4ExpIntBinding;
void require(bool valid,const char *message){if(!valid)throw std::runtime_error(message);}
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        constexpr int requests=3,capacity=4,width=128;
        constexpr int priorWords=sizeof(qwen4exp::QwenSparseTransactionPrior)/sizeof(std::int32_t);
        auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        FP32Tensor source({requests*capacity,width}),retained({requests*capacity,width}),partial({requests,4*width});
        INT32Tensor replay({requests*3}),observed({requests*2}),decisions({requests*2}),
            prior({requests*priorWords}),priorTraffic({requests*2});
        std::vector<TensorBase *> tensors{&source,&retained,&partial,&replay,&observed,&decisions,&prior,&priorTraffic};
        for(auto *tensor:tensors) {
            std::memset(tensor->raw_mutable_data(),0,tensor->size_bytes());
            TransferEngine::prepareDeviceInput(tensor,device,stream);
            TransferEngine::requireDeviceInput(tensor,device,stream);
        }
        std::vector<std::unique_ptr<Stage>> snapshots,stages;
        for(int r=0;r<requests;++r) {
            auto offset=std::size_t(r)*capacity*width*sizeof(float);
            const QwenSparseTransactionPriorBinding original{
                Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR,r*sizeof(qwen4exp::QwenSparseTransactionPrior))};
            snapshots.push_back(std::make_unique<Stage>(Stage::Params{
                .device_id=device,.rowCapacity=capacity,.tokenCapacity=32,
                .replay=Int(replay,BufferId::QFN_QSA_REPLAY,r*3*sizeof(int)),.operation=Stage::SnapshotPrior{
                    Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS,r*4*width*sizeof(float)),original,
                    Int(priorTraffic,BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC,r*sizeof(qwen4exp::QwenSparsePriorTraffic))}}));
            snapshots.back()->setGPUStream(stream);
            stages.push_back(std::make_unique<Stage>(Stage::Params{.device_id=device,.rowCapacity=capacity,.tokenCapacity=32,
                .replay=original.storage,.operation=Stage::RetainCandidates{
                    Float(source,BufferId::QFN_QSA_RAW_KEY,offset),Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,offset),
                    Int(observed,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,r*sizeof(std::uint64_t)),
                    Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS,r*4*width*sizeof(float)),r,requests,
                    RequestRowOrigins::fixedStride(0,capacity),original}}));
            stages.back()->setGPUStream(stream);
        }
        require(backend->synchronizeStream(stream,device.ordinal),"GPU setup fence failed");
        auto graph=worker.createGraphCapture(stream);
        {
            ScopedBackendGraphCapture capture(worker,*graph,"QSA retained accepted-prefix hooks");
            require(capture.begin(),"GPU capture begin failed");
            for(auto &stage:snapshots)require(stage->execute(context.get()),"GPU immutable prior snapshot capture failed");
            for(auto &stage:stages)require(stage->execute(context.get()),"GPU retention capture failed");
            for(auto &stage:stages)require(stage->restoreVerifierStateCaptureRowsFromDeviceIndices(
                static_cast<const int *>(decisions.gpu_data_ptr()),requests,2,stream),"GPU accepted publication capture failed");
            capture.finish();
        }
        EXPECT_EQ(graph->nodeCount(),requests*3u);
        require(graph->instantiate(),"GPU retained executable failed");
        // A later accepted-prefix revision must never resnapshot the now-mutated
        // accepted tail. Capture the publication-only consumer of original storage.
        auto publication=worker.createGraphCapture(stream);
        {
            ScopedBackendGraphCapture capture(worker,*publication,"QSA immutable prior repeated publication");
            require(capture.begin(),"GPU repeat-publication capture begin failed");
            for(auto &stage:stages)require(stage->restoreVerifierStateCaptureRowsFromDeviceIndices(
                static_cast<const int *>(decisions.gpu_data_ptr()),requests,2,stream),"GPU repeat-publication capture failed");
            capture.finish();
        }
        EXPECT_EQ(publication->nodeCount(),requests*1u);
        require(publication->instantiate(),"GPU repeat-publication executable failed");
        const std::array<std::array<int,requests>,4> live{{{4,3,0},{0,2,4},{1,0,3},{0,0,0}}};
        const std::array<std::array<int,requests>,4> accepted{{{1,3,0},{0,1,2},{1,0,3},{0,0,0}}};
        std::vector<float> raw(source.numel()),tail(partial.numel()),actualTail(tail.size()),
            expectedRetained(retained.numel(),0.0f),actualRetained(retained.numel());
        for(std::size_t transaction=0;transaction<live.size();++transaction) {
            std::array<int,requests*3> state{};std::array<int,requests*2> flat{};std::array<std::uint64_t,requests> bytes{};
            for(std::size_t i=0;i<raw.size();++i)raw[i]=float(10000*transaction+i+1);
            for(std::size_t i=0;i<tail.size();++i)tail[i]=-float(10000*transaction+i+1);
            auto expectedTail=tail;
            for(int r=0;r<requests;++r) {
                const int prefix=r+1,count=accepted[transaction][r],total=prefix+count;
                state[r*3]=prefix;state[r*3+1]=live[transaction][r];state[r*3+2]=17;
                flat[r*2]=count ? r*capacity+count-1 : -1;flat[r*2+1]=9999;
                std::copy_n(raw.data()+r*capacity*width,live[transaction][r]*width,
                    expectedRetained.data()+r*capacity*width);
                if(live[transaction][r])for(int row=0;row<total%4;++row) {
                    const int token=total/4*4+row;
                    const float *from=token<prefix ? tail.data()+(r*4+token)*width :
                        raw.data()+(r*capacity+token-prefix)*width;
                    std::copy_n(from,width,expectedTail.data()+(r*4+row)*width);
                }
            }
            for(const auto &[tensor,host]:std::array<std::pair<TensorBase *,const void *>,4>{{
                {&source,raw.data()},{&partial,tail.data()},{&replay,state.data()},{&decisions,flat.data()}}})
                require(backend->hostToDeviceOnStream(tensor->gpu_data_ptr(),host,tensor->size_bytes(),device.ordinal,stream),"GPU transaction input publication failed");
            require(graph->launch(),"GPU retained transaction replay failed");
            require(backend->deviceToHostOnStream(actualTail.data(),partial.gpu_data_ptr(),partial.size_bytes(),device.ordinal,stream),"GPU tail observation failed");
            require(backend->deviceToHostOnStream(actualRetained.data(),retained.gpu_data_ptr(),retained.size_bytes(),device.ordinal,stream),"GPU candidate observation failed");
            require(backend->deviceToHostOnStream(bytes.data(),observed.gpu_data_ptr(),observed.size_bytes(),device.ordinal,stream),"GPU byte observation failed");
            require(backend->synchronizeStream(stream,device.ordinal),"GPU terminal observation fence failed");
            EXPECT_EQ(std::memcmp(actualTail.data(),expectedTail.data(),partial.size_bytes()),0)<<transaction;
            EXPECT_EQ(std::memcmp(actualRetained.data(),expectedRetained.data(),retained.size_bytes()),0)<<transaction;
            for(int r=0;r<requests;++r)EXPECT_EQ(bytes[r],std::uint64_t(live[transaction][r])*width*sizeof(float));
            std::array<qwen4exp::QwenSparsePriorTraffic,requests> traffic{};
            std::vector<std::int32_t> frozen(prior.numel()),after(prior.numel());
            require(backend->deviceToHostOnStream(traffic.data(),priorTraffic.gpu_data_ptr(),priorTraffic.size_bytes(),device.ordinal,stream),"GPU prior traffic observation failed");
            require(backend->deviceToHostOnStream(frozen.data(),prior.gpu_data_ptr(),prior.size_bytes(),device.ordinal,stream),"GPU original prior observation failed");
            require(backend->synchronizeStream(stream,device.ordinal),"GPU prior terminal observation fence failed");
            for(int r=0;r<requests;++r) {
                EXPECT_EQ(traffic[r].metadata_bytes,16u);
                EXPECT_EQ(traffic[r].payload_bytes,live[transaction][r] ? std::uint32_t((r+1)%4*width*sizeof(float)) : 0u);
            }
            for(int revision:{0,1,0}) {
                expectedTail=actualTail;
                for(int r=0;r<requests;++r) {
                    const int count=std::min(revision,live[transaction][r]),prefix=r+1,total=prefix+count;
                    flat[r*2]=count ? r*capacity+count-1 : -1;
                    if(live[transaction][r])for(int row=0;row<total%4;++row) {
                        const int token=total/4*4+row;
                        const float* from=token<prefix ? tail.data()+(r*4+token)*width : raw.data()+(r*capacity+token-prefix)*width;
                        std::copy_n(from,width,expectedTail.data()+(r*4+row)*width);
                    }
                }
                require(backend->hostToDeviceOnStream(decisions.gpu_data_ptr(),flat.data(),decisions.size_bytes(),device.ordinal,stream),"GPU revised decision publication failed");
                require(publication->launch(),"GPU revised acceptance replay failed");
                require(backend->deviceToHostOnStream(actualTail.data(),partial.gpu_data_ptr(),partial.size_bytes(),device.ordinal,stream),"GPU revised tail observation failed");
                require(backend->deviceToHostOnStream(after.data(),prior.gpu_data_ptr(),prior.size_bytes(),device.ordinal,stream),"GPU prior immutability observation failed");
                require(backend->synchronizeStream(stream,device.ordinal),"GPU revised terminal observation fence failed");
                EXPECT_EQ(actualTail,expectedTail)<<transaction<<" revision "<<revision;
                EXPECT_EQ(after,frozen)<<"An acceptance revision changed its original transaction prior";
            }
        }
        publication.reset();graph.reset();
    }).get();
}
}
#ifdef HAVE_CUDA
TEST(QwenSparseRetainedDevice, CUDA){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(QwenSparseRetainedDevice, ROCm){prove(DeviceId::rocm(0));}
#endif
