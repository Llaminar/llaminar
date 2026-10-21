/**
 * @file Test__ImmutableFixedHeadLookupDevice.cpp
 * @brief Captured native fixed-head primitives with original source/row authority.
 *
 * This is deliberately a primitive proof, not a distributed serving witness.
 * Peer headers and rank-major receive bytes are immutable oracle inputs staged
 * before replay; no collective is simulated or certified. The captured local
 * header is copied exactly8B into its actual receive slot. NativeAllGatherStage,
 * replicated-query admission and graph/PMA integration have separate gates.
 */
#include <gtest/gtest.h>
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "kernels/common/SamplingMath.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/ExplicitGPUStream.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/EmbeddingVerifierFormats.h"
#include <algorithm>
#include <cstring>
#include <string_view>
#include <vector>
using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Fail infrastructure operations without abandoning graph retirement. */
void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
/** @brief Own all captured buffers and retire recordings before releasing any borrowed pointer. */
struct Storage
{
    DeviceId device;IBackend *backend;void *stream;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
    std::unique_ptr<IImmutableFixedHeadLookupKernels> kernels;
    std::unique_ptr<IGPUGraphCapture> graph;
    /** @brief Completion/retirement remains mandatory during assertion/exception unwind. */
    ~Storage() noexcept
    {
        if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();
        graph.reset();
    }
    /** @brief Allocate only before recording, through the public transfer authority. */
    template<class T>T *allocate(std::size_t count)
    {
        auto owner=TransferEngine::instance().allocateDeviceTransferBuffer(count*sizeof(T),device);
        require(owner&&owner->isBound(),"Fixed-head fixture allocation failed");
        auto *pointer=static_cast<T *>(owner->mutableDeviceData());owners.push_back(std::move(owner));return pointer;
    }
    /** @brief Publish fixture admission/oracle data outside capture. */
    template<class T>void upload(T *pointer,const std::vector<T> &values)
    {
        require(backend->hostToDeviceOnStream(pointer,values.data(),values.size()*sizeof(T),device.ordinal,stream),"Fixed-head upload failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Fixed-head setup completion failed");
    }
    /** @brief Observe terminal diagnostic data after native work; never feeds production inference. */
    template<class T>std::vector<T> download(const T *pointer,std::size_t count)
    {
        std::vector<T> values(count);
        require(backend->deviceToHostOnStream(values.data(),pointer,count*sizeof(T),device.ordinal,stream),"Fixed-head download failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Fixed-head terminal completion failed");return values;
    }
};
/** @brief Unequal source ranges include an original two-row gap and eight trailing rows. */
NgramHashPolicy policy()
{
    NgramHashPolicy p;p.multipliers={1,3,7};std::uint64_t end=0;
    for(int h=0;h<16;++h){if(h==8)end+=2;p.offsets[h]=end;p.moduli[h]=h+3;end+=h+3;}
    return p;
}
/** @brief Assert every FP32 bit, including NaN payloads and untouched suffix poison. */
void sameBytes(const std::vector<float> &a,const std::vector<float> &b)
{
    ASSERT_EQ(a.size(),b.size());EXPECT_EQ(std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),0);
}
/**
 * @brief Exercise an actual captured owner projection, original codec and frozen capacities.
 * @param device Exact native device, never a synthetic multi-device cohort.
 * @param backend Actual backend owner of the fixture allocations.
 * @param worker Context retaining capture stream and graph resource inspection.
 * @param format Canonical native-format fixture factory.
 * @param width Original decoded row width, including production IQ4_NL width160.
 * @param capacity Original token capacity; every physicalM1 retains its count pointer.
 * @param degree Number of oracle source banks, two/four/eight.
 * @param rank Exact owner coordinate, including nonzero global source origins.
 */
void proveFormat(DeviceId device,IBackend *backend,IWorkerGPUContext &worker,
    const test::EmbeddingVerifierFormatCase &format,int width,int capacity,int degree,int rank)
{
    SCOPED_TRACE(::testing::Message()<<device.toString()<<'/'<<format.label<<" width="<<width<<" M="<<capacity<<" P="<<degree<<" rank="<<rank);
    const auto metadata=policy();const int heads=16/degree;
    auto tensor=format.create({178,static_cast<std::size_t>(width)},123);
    const auto type=tensor->native_type();
    const auto stride=type==TensorType::FP32?width*4u:type==TensorType::FP16||type==TensorType::BF16?width*2u:
        NativeEmbeddingLayout::forSource(type,1,width).row_stride_bytes;
    ASSERT_EQ(tensor->size_bytes(),178*stride);
    const auto *data=static_cast<const std::byte *>(tensor->raw_data());
    const std::vector<std::byte> original(data,data+tensor->size_bytes());
    const auto first=rank?metadata.offsets[rank*heads]:0;
    const auto end=rank+1==degree?178:metadata.offsets[(rank+1)*heads];
    const std::vector<std::byte> localSource(original.begin()+first*stride,original.begin()+end*stride);
    Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{},{},{}};
    auto *source=s.allocate<std::byte>(localSource.size());s.upload(source,localSource);
    const auto view=ImmutableResidentTableRangeView::bind(device,type,source,178,first,end-first,width,stride,localSource.size(),9);
    const auto plan=ImmutableFixedHeadLookupPlan::bind(view,metadata,degree,rank);
#ifdef HAVE_CUDA
    if(device.is_cuda())s.kernels=createCUDAImmutableFixedHeadLookupKernels(plan);
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())s.kernels=createROCmImmutableFixedHeadLookupKernels(plan);
#endif
    require(bool(s.kernels),"Fixed-head backend factory absent");
    const auto idsCount=static_cast<std::size_t>(capacity)*16;
    const auto packetBytes=static_cast<std::size_t>(capacity)*heads*stride;
    auto *ids=s.allocate<std::int64_t>(idsCount);
    auto *header=s.allocate<FixedHeadValidationHeader>(1),*headers=s.allocate<FixedHeadValidationHeader>(degree);
    auto *transport=s.allocate<int>(1),*live=s.allocate<int>(1),*healthy=s.allocate<int>(1),*error=s.allocate<int>(1);
    auto *packet=s.allocate<std::byte>(packetBytes),*gathered=s.allocate<std::byte>(packetBytes*degree);
    auto *output=s.allocate<float>(idsCount*width);
    ImmutableFixedHeadLookupArgs a{DeviceRowRange::deviceCounted(capacity,live),ids,idsCount,header,headers,
        static_cast<std::size_t>(degree),transport,packet,packetBytes,gathered,packetBytes*degree,output,idsCount*width,healthy,error};
    const auto identity=s.kernels->captureIdentity(a,s.stream);
    const std::vector<std::byte> packetPoison(packetBytes,std::byte{0xa7}),receivePoison(packetBytes*degree,std::byte{0x6d});
    const std::vector<float> outputPoison(idsCount*width,-12345.F);
    s.upload(live,std::vector{capacity});s.upload(error,std::vector{0});s.upload(healthy,std::vector{0});
    // This exact fixed-size self-slot copy proves header publication, not a
    // stand-in collective. Other peer slots are test inputs before replay.
    s.graph=worker.createGraphCapture(s.stream);
    {
        ScopedBackendGraphCapture capture(worker,*s.graph,"fixed-head original-native primitives");
        require(capture.begin(),"Fixed-head capture begin failed");
        s.kernels->validateLocal(a,s.stream);
        require(backend->deviceCopyAsync(headers+rank,header,sizeof(FixedHeadValidationHeader),device.ordinal,s.stream),"Fixed8B header publication failed");
        s.kernels->validateCohort(a,s.stream);s.kernels->pack(a,s.stream);s.kernels->decode(a,s.stream);
        capture.finish();
    }
    std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
    require(s.graph->inspectKernelNodes(nodes,&detail),detail.c_str());ASSERT_EQ(nodes.size(),5u);
    for(const auto &node:nodes)
    {
        EXPECT_EQ(node.local_memory_bytes_per_thread,0u);EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);
        EXPECT_TRUE(node.block_x==1u||node.block_x==128u||node.block_x==256u);
    }
    require(s.graph->instantiate(),"Fixed-head instantiate failed");const auto owners=s.owners.size();
    std::vector<std::int64_t> hostIds(idsCount);
    std::vector<int> lives{capacity,1,0,std::min(2,capacity),capacity};
    if(capacity==15)for(int rows=0;rows<=15;++rows)lives.push_back(rows);
    const int storageFailure=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
    // Original count and peer headers are independently changed only outside
    // replay. Every bad peer must zero a previously successful derived extent.
    const auto replay=[&](int rows,int badKind,int initialHealth=0)
    {
        SCOPED_TRACE(::testing::Message()<<"live="<<rows<<" bad="<<badKind);
        const int valid=std::clamp(rows,0,capacity);
        auto received=receivePoison;auto expectedPacket=packetPoison;auto expected=outputPoison;
        std::vector<std::byte> decodedNative(static_cast<std::size_t>(valid)*16*stride);
        for(int token=0;token<capacity;++token)for(int h=0;h<16;++h)
        {
            const auto id=metadata.offsets[h]+(static_cast<unsigned>(token*7+h)%metadata.moduli[h]);
            hostIds[static_cast<std::size_t>(token)*16+h]=token<valid?static_cast<std::int64_t>(id):-1;
            if(token>=valid)continue;
            const auto *row=original.data()+id*stride;
            std::memcpy(received.data()+static_cast<std::size_t>(h/heads)*packetBytes+
                (static_cast<std::size_t>(token)*heads+h%heads)*stride,row,stride);
            std::memcpy(decodedNative.data()+(static_cast<std::size_t>(token)*16+h)*stride,row,stride);
            if(h/heads==rank)std::memcpy(expectedPacket.data()+(static_cast<std::size_t>(token)*heads+h%heads)*stride,row,stride);
        }
        if(valid)decodeCPUImmutableLookupRows(type,width,stride,decodedNative.data(),static_cast<std::size_t>(valid)*16,expected.data());
        std::vector<FixedHeadValidationHeader> peerHeaders(degree,{rows,0});
        const int lastOwnedHead=(rank+1)*heads-1;
        if(badKind==1)hostIds[(static_cast<std::size_t>(valid)-1)*16+lastOwnedHead]=
            static_cast<std::int64_t>(metadata.offsets[lastOwnedHead]+metadata.moduli[lastOwnedHead]);
        if(badKind==2)peerHeaders[(rank+1)%degree].firstError=73;
        if(badKind==3)peerHeaders[(rank+1)%degree].originalLiveRows=rows?rows-1:1;
        if(badKind==4)peerHeaders[(rank+1)%degree].originalLiveRows=capacity+1;
        // Foreign-head poison must not make a local owner inspect peer IDs.
        if(badKind==0)for(int token=0;token<valid;++token)for(int h=0;h<16;++h)
            if(h/heads!=rank)hostIds[static_cast<std::size_t>(token)*16+h]=-1;
        s.upload(ids,hostIds);s.upload(live,std::vector{rows});s.upload(headers,peerHeaders);
        s.upload(gathered,received);s.upload(packet,packetPoison);s.upload(output,outputPoison);
        s.upload(healthy,std::vector{initialHealth});s.upload(error,std::vector{badKind==5?79:0});
        s.upload(transport,std::vector{capacity}); // Dirty previous-success metadata.
        require(s.graph->launch(),"Fixed-head retained replay failed");
        const bool failed=badKind!=0||rows<0||rows>capacity;
        EXPECT_EQ(s.download(transport,1).front(),failed?0:rows);
        EXPECT_EQ(s.download(error,1).front(),failed?(badKind==2?73:badKind==5?79:storageFailure):0);
        EXPECT_EQ(s.download(healthy,1).front(),failed?0:initialHealth); // Never seed generation health.
        const auto published=s.download(header,1).front();EXPECT_EQ(published.originalLiveRows,rows);
        EXPECT_EQ(published.firstError,badKind==5?79:badKind==1||rows<0||rows>capacity?storageFailure:0);
        peerHeaders[rank]=published;EXPECT_EQ(s.download(headers,degree),peerHeaders);
        EXPECT_EQ(s.download(packet,packetBytes),failed?packetPoison:expectedPacket);
        sameBytes(s.download(output,outputPoison.size()),failed?outputPoison:expected);
        EXPECT_EQ(s.download(gathered,received.size()),received);
        EXPECT_EQ(s.download(ids,hostIds.size()),hostIds);
        EXPECT_EQ(s.kernels->captureIdentity(a,s.stream),identity);EXPECT_EQ(s.owners.size(),owners);
    };
    for(int rows:lives)replay(rows,0);
    for(int bad:{1,2,3,4,5})replay(capacity,bad);
    replay(-1,0);replay(capacity+1,0);replay(0,2);replay(capacity,0);
    replay(capacity,0,1);replay(capacity,1,1);replay(capacity,0);
    // Do not reset previous-success packet/output bytes: a zero replay must
    // preserve real stale results rather than merely restore a poison pattern.
    const auto successfulPacket=s.download(packet,packetBytes);
    const auto successfulOutput=s.download(output,outputPoison.size());
    for(const int status:{0,79})
    {
        s.upload(live,std::vector{0});s.upload(headers,std::vector<FixedHeadValidationHeader>(degree,{0,0}));
        s.upload(error,std::vector{status});s.upload(transport,std::vector{capacity});
        require(s.graph->launch(),"Fixed-head stale successful suffix replay failed");
        EXPECT_EQ(s.download(transport,1).front(),0);EXPECT_EQ(s.download(error,1).front(),status);
        EXPECT_EQ(s.download(packet,packetBytes),successfulPacket);
        sameBytes(s.download(output,outputPoison.size()),successfulOutput);
    }
    EXPECT_EQ(s.download(source,localSource.size()),localSource);
}
/** @brief Canonical decoder inventory plus production IQ4_NL width and all supported head degrees. */
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        for(const auto &format:test::embeddingVerifierFormats())
        {
            proveFormat(device,backend,worker,format,256,1,2,1);
            proveFormat(device,backend,worker,format,256,15,4,2);
            if(std::string_view(format.label)=="IQ4_NL")
                for(int degree:{2,4,8})for(int rank=0;rank<degree;++rank)
                    for(int rows:{1,17,512})proveFormat(device,backend,worker,format,160,rows,degree,rank);
        }
    }).get();
}
}
#ifdef HAVE_CUDA
/** @test CUDA captured primitives preserve every original codec, exact count and suffix. */
TEST(ImmutableFixedHeadLookupDevice, CUDAOriginalFormatsAndProtocol){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm captured primitives have the same native-byte/header/count contract. */
TEST(ImmutableFixedHeadLookupDevice, ROCmOriginalFormatsAndProtocol){prove(DeviceId::rocm(0));}
#endif
