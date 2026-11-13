/**
 * @file QwenSparseTerminalCounterDeviceTest.h
 * @brief Real native atomic writers, complete-parent frontier and exact terminal observation.
 *
 * Backend wrappers bind production capture owners and native QSA factories.
 * Storage/copies/synchronization below are explicit fixture boundaries. The
 * negative control removes one native dependency only in its private graph;
 * a device handshake makes a missing delayed branch deterministic.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/common/SamplingMath.h"
#include "backends/IGPUGraphCapture.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace llaminar2::qwen4exp::QSC_SUITE {
/** @brief Fatal native fixture error, retaining the exact backend diagnostic. */
template<class T>void check(T status){if(status!=QSC_RT(Success))throw std::runtime_error(QSC_RT(GetErrorString)(status));}
/** @brief Fixture-admitted persistent device bytes, retired after terminal completion. */
template<class T>struct Buffer {
    T*data=nullptr;std::size_t count;
    explicit Buffer(std::size_t n):count(n){check(QSC_RT(Malloc)(reinterpret_cast<void**>(&data),n*sizeof(T)));}
    ~Buffer(){(void)QSC_RT(Free)(data);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    /** @brief Explicit pre-replay input publication. */
    void upload(const T*input,QSC_STREAM stream){check(QSC_RT(MemcpyAsync)(data,input,count*sizeof(T),QSC_RT(MemcpyHostToDevice),stream));}
    /** @brief Existing terminal-only observation, outside captured execution. */
    std::vector<T> download(QSC_STREAM stream){std::vector<T>out(count);check(QSC_RT(MemcpyAsync)(out.data(),data,count*sizeof(T),QSC_RT(MemcpyDeviceToHost),stream));check(QSC_RT(StreamSynchronize)(stream));return out;}
};
/** @brief Device-only assertion, supported by both native assemblers. */
__device__ __forceinline__ void failCounter(){
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#else
    asm("trap;");
#endif
}
/** @brief Distinct real independent fast writer; native live work never exceeds4096 rows. */
__global__ void fastWriter(QwenSparseWorkCounterRow row,const int*live,int*completed){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<*live){if(!addNativeCounter64(row.metric(QwenSparseWorkCounter::CandidatePayload),1))failCounter();
        __threadfence();atomicAdd(completed,1);}
}
/** @brief Independent single-block delayed branch, with deterministic negative-control handshake. */
__global__ void delayedWriter(QwenSparseWorkCounterRow row,const int*live,int*release,bool negative){
    if(threadIdx.x || blockIdx.x || *live==0)return;
    if(negative){while(atomicAdd(release,0)==0){}}
    else {const auto start=clock64();while(clock64()-start<1000000ull){}}
    if(!addNativeCounter64(row.metric(QwenSparseWorkCounter::CandidatePayload),0x100000000ull))failCounter();
}
/** @brief Required protocol-only open/close marker with no counter contribution. */
__global__ void marker(){}
/** @brief Immutable terminal device observation and deterministic negative release. */
__global__ void terminalObserver(QwenSparseWorkCounterRow row,std::uint64_t*out,int*release,const int*live){
    if(!threadIdx.x){while(atomicAdd(release+1,0)<*live){}
        *out=readNativeCounter64(row.metric(QwenSparseWorkCounter::CandidatePayload).words);__threadfence();atomicExch(release,1);}
}
/** @brief Return which exact native kernels occur recursively in one child graph. */
unsigned contents(QSC_GRAPH graph){
    std::size_t count=0;check(QSC_RT(GraphGetNodes)(graph,nullptr,&count));std::vector<QSC_NODE>nodes(count);
    check(QSC_RT(GraphGetNodes)(graph,nodes.data(),&count));unsigned result=0;
    for(auto node:nodes){QSC_NODE_TYPE type{};check(QSC_RT(GraphNodeGetType)(node,&type));
        if(type==QSC_RT(GraphNodeTypeKernel)){QSC_KERNEL_PARAMS params{};check(QSC_RT(GraphKernelNodeGetParams)(node,&params));
            if(params.func==reinterpret_cast<void*>(fastWriter))result|=1;
            if(params.func==reinterpret_cast<void*>(delayedWriter))result|=2;
        }else if(type==QSC_RT(GraphNodeTypeGraph)){QSC_GRAPH child{};check(QSC_RT(GraphChildGraphNodeGetGraph)(node,&child));result|=contents(child);}
    }return result;
}
/** @brief Negative control removes a terminal completion edge; a device handshake retains every fast writer. */
bool omitDelayedCompletion(QSC_GRAPH graph){
    std::size_t count=0;check(QSC_RT(GraphGetNodes)(graph,nullptr,&count));std::vector<QSC_NODE>nodes(count);
    check(QSC_RT(GraphGetNodes)(graph,nodes.data(),&count));
    for(auto node:nodes){QSC_NODE_TYPE type{};check(QSC_RT(GraphNodeGetType)(node,&type));unsigned found=0;QSC_GRAPH child{};
        if(type==QSC_RT(GraphNodeTypeKernel)){QSC_KERNEL_PARAMS params{};check(QSC_RT(GraphKernelNodeGetParams)(node,&params));if(params.func==reinterpret_cast<void*>(delayedWriter))found=2;}
        else if(type==QSC_RT(GraphNodeTypeGraph)){check(QSC_RT(GraphChildGraphNodeGetGraph)(node,&child));found=contents(child);}
        if(found&2){std::size_t n=0;check(QSC_DEPENDENTS(node,nullptr,&n));
            if(n){std::vector<QSC_NODE>after(n);check(QSC_DEPENDENTS(node,after.data(),&n));
                for(auto destination:after)check(QSC_REMOVE(graph,&node,&destination,1));return true;}}
        if(child && omitDelayedCompletion(child))return true;
    }return false;
}
/** @brief Delayed independent writers must reach the existing parent terminal; deleting their edge misses exact bytes. */
TEST(QSC_SUITE, CompleteParentFrontierIncludesDelayedIndependentAtomicWriters){
    check(QSC_RT(SetDevice)(0));QSC_STREAM stream{};check(QSC_RT(StreamCreateWithFlags)(&stream,QSC_RT(StreamNonBlocking)));
    {
        Buffer<int>control(sampling_math::kDeviceGenerationControlCount),live(1),release(2);Buffer<std::uint64_t>observed(1);
        for(bool negative:{false,true}){
            const QwenSparseWorkCounterRow row{control.data,sampling_math::kDeviceGenerationControlCount};
            auto body=std::make_unique<QSC_CAPTURE>(stream,0),open=std::make_unique<QSC_CAPTURE>(stream,0),
                worker=std::make_unique<QSC_CAPTURE>(stream,0),close=std::make_unique<QSC_CAPTURE>(stream,0),
                terminal=std::make_unique<QSC_CAPTURE>(stream,0),parent=std::make_unique<QSC_CAPTURE>(stream,0);
            ASSERT_TRUE(body->beginCapture());fastWriter<<<32,128,0,stream>>>(row,live.data,release.data+1);ASSERT_TRUE(body->endCapture());
            ASSERT_TRUE(open->beginCapture());marker<<<1,1,0,stream>>>();ASSERT_TRUE(open->endCapture());
            ASSERT_TRUE(worker->beginCapture());delayedWriter<<<1,32,0,stream>>>(row,live.data,release.data,negative);ASSERT_TRUE(worker->endCapture());
            ASSERT_TRUE(close->beginCapture());marker<<<1,1,0,stream>>>();ASSERT_TRUE(close->endCapture());
            ASSERT_TRUE(body->appendParallelBranch({*open,*worker,*close}));
            ASSERT_TRUE(terminal->beginCapture());terminalObserver<<<1,1,0,stream>>>(row,observed.data,release.data,live.data);ASSERT_TRUE(terminal->endCapture());
            const GPUOrderedTimelineStep steps[]{{.name="complete_verifier",.capture=body.get()},{.name="existing_terminal",.capture=terminal.get()}};
            ASSERT_TRUE(parent->buildOrderedTimelineTransaction(steps));
            if(negative)ASSERT_TRUE(omitDelayedCompletion(parent->graph()));
            ASSERT_TRUE(parent->instantiate());
            for(int active:{4096,1,0,4095,0}){
                std::array<int,sampling_math::kDeviceGenerationControlCount>host;host.fill(-19);
                ASSERT_TRUE(sampling_math::initialize_device_generation_control(4,4,sampling_math::DeviceGenerationPolicy::fixed(1),host.data()));
                const int zero[2]{};std::uint64_t poison=~std::uint64_t(0);control.upload(host.data(),stream);live.upload(&active,stream);release.upload(zero,stream);observed.upload(&poison,stream);
                ASSERT_TRUE(parent->launch());const auto terminalValue=observed.download(stream).front();
                const auto final=control.download(stream);const std::uint64_t expected=active ? 0x100000000ull+active : 0;
                EXPECT_EQ(readNativeCounter64(final.data()+sampling_math::kDeviceGenerationControlQsaCandidatePayloadBytesLow),expected);
                if(negative && active)EXPECT_EQ(terminalValue,std::uint64_t(active));else EXPECT_EQ(terminalValue,expected);
                EXPECT_EQ(qwenSparseCounterPair(final.data(),sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow),0u);
            }
        }
    }check(QSC_RT(StreamDestroy)(stream));
}
/** @brief Production raw operations expose exact immutable-prior and revised-acceptance totals in one retained graph. */
TEST(QSC_SUITE, CapturedActualSnapshotRetentionAndRevisedAcceptanceReportLiveBytes){
    check(QSC_RT(SetDevice)(0));QSC_STREAM stream{};check(QSC_RT(StreamCreateWithFlags)(&stream,QSC_RT(StreamNonBlocking)));
    {
        Buffer<int>control(sampling_math::kDeviceGenerationControlCount),decision(1);Buffer<QwenSparseReplay>replay(1);
        Buffer<float>partial(512),raw(5*128),retained(5*128);Buffer<QwenSparseTransactionPrior>prior(1);
        Buffer<QwenSparsePriorTraffic>traffic(1);Buffer<std::uint64_t>copied(1);
        auto kernels=QSC_FACTORY();QwenSparseWorkCounterRow row{control.data,sampling_math::kDeviceGenerationControlCount};
        auto prepare=std::make_unique<QSC_CAPTURE>(stream,0),publication=std::make_unique<QSC_CAPTURE>(stream,0);
        ASSERT_TRUE(prepare->beginCapture());kernels->snapshotPrior({replay.data,partial.data,prior.data,traffic.data,5,128,row},stream);
        kernels->retainCandidates({raw.data,retained.data,&prior.data->replay.live_rows,copied.data,5,row},stream);ASSERT_TRUE(prepare->endCapture());ASSERT_TRUE(prepare->instantiate());
        ASSERT_TRUE(publication->beginCapture());kernels->commitPartial({&prior.data->replay,prior.data->partial,retained.data,
            AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,5,&prior.data->replay.live_rows),decision.data),
            partial.data,prior.data,128,row},stream);ASSERT_TRUE(publication->endCapture());ASSERT_TRUE(publication->instantiate());
        for(int active:{5,0,1,0}){
            std::array<int,sampling_math::kDeviceGenerationControlCount>host;host.fill(-71);
            ASSERT_TRUE(sampling_math::initialize_device_generation_control(4,4,sampling_math::DeviceGenerationPolicy::fixed(1),host.data()));
            QwenSparseReplay base{7,active,91};std::array<float,512>original;std::array<float,640>source;
            for(int i=0;i<512;++i)original[i]=i+0.5f;for(int i=0;i<640;++i)source[i]=1000+i;
            control.upload(host.data(),stream);replay.upload(&base,stream);partial.upload(original.data(),stream);raw.upload(source.data(),stream);
            ASSERT_TRUE(prepare->launch());std::uint64_t expectedAccepted=0;
            const auto immutable=prior.download(stream);
            for(int accepted:{active,0,active ? 1:0,0}){
                decision.upload(&accepted,stream);ASSERT_TRUE(publication->launch());const auto result=control.download(stream);
                QwenSparseCounterTotals total;ASSERT_TRUE(readQwenSparseTerminalCounters(result.data(),sampling_math::kDeviceGenerationControlCount,total));
                if(active)expectedAccepted+=std::uint64_t((7+accepted)%4)*512;
                EXPECT_EQ(total.prior_header,16u);EXPECT_EQ(total.prior_payload,active ? 1536u:0u);
                EXPECT_EQ(total.candidate_payload,std::uint64_t(active)*512);EXPECT_EQ(total.accepted_payload,expectedAccepted);
                const auto frozen=prior.download(stream);EXPECT_EQ(std::memcmp(frozen.data(),immutable.data(),sizeof(QwenSparseTransactionPrior)),0);
            }
            const auto recovered=partial.download(stream);EXPECT_EQ(std::memcmp(recovered.data(),original.data(),(active ? 3*128:512)*sizeof(float)),0);
        }
    }check(QSC_RT(StreamDestroy)(stream));
}
} // namespace llaminar2::qwen4exp::QSC_SUITE
