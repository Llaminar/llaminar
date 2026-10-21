/**
 * @file Test__ImmutableResidentLookupDevice.cpp
 * @brief Captured direct-VRAM native gather proof without service pages or host tickets.
 *
 * One source-native table allocation and one captured two-kernel lookup survive
 * every live-count replay. CPU decoding and serial GPU rows are diagnostic
 * oracles only. Poisoned inactive IDs/output and failed-ID transactions prove
 * live extents, failure isolation and immutable source storage.
 */
#include <gtest/gtest.h>
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
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
#include <array>
#include <bit>
#include <sstream>
#include <span>
#include <string_view>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Treat test infrastructure failure as fatal, retaining graph-storage RAII. */
void require(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
/** @brief Retain every captured pointer until explicit completion and graph retirement. */
struct Storage
{
    DeviceId device;
    IBackend *backend;
    void *stream;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
    std::unique_ptr<IImmutableResidentLookupKernels> kernels;
    std::unique_ptr<IGPUGraphCapture> graph;
    /** @brief Retire submitted native work and recordings before any source/output owner. */
    ~Storage() noexcept
    {
        if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();
        graph.reset();
    }
    /** @brief Allocate persistent test storage before graph recording. */
    template<class T>T *allocate(std::size_t count)
    {
        auto owner=TransferEngine::instance().allocateDeviceTransferBuffer(count*sizeof(T),device);
        require(owner&&owner->isBound(),"Resident lookup test allocation failed");
        auto *pointer=static_cast<T *>(owner->mutableDeviceData());owners.push_back(std::move(owner));return pointer;
    }
    /** @brief Publish test admission bytes outside capture, on the exact stream. */
    template<class T>void upload(T *pointer,const std::vector<T> &values)
    {
        require(backend->hostToDeviceOnStream(pointer,values.data(),values.size()*sizeof(T),device.ordinal,stream),"Resident lookup upload failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Resident lookup admission observation failed");
    }
    /** @brief Copy only test terminal evidence after native completion. */
    template<class T>std::vector<T> download(const T *pointer,std::size_t count)
    {
        std::vector<T> values(count);
        require(backend->deviceToHostOnStream(values.data(),pointer,count*sizeof(T),device.ordinal,stream),"Resident lookup terminal download failed");
        require(backend->synchronizeStream(stream,device.ordinal),"Resident lookup terminal observation failed");return values;
    }
};
/** @brief Original native source coordinates for bit-exact mismatch evidence. */
struct SourceEvidence
{
    TensorType type;
    int width;
    std::size_t stride;
    std::span<const std::int64_t> ids;
    std::span<const std::byte> source;
    std::size_t liveIds;
};
/**
 * @brief Compare every output bit and report the first source/result mismatch.
 * @param actual Native result, including poisoned inactive capacity.
 * @param expected Independent CPU or native serial reference.
 * @param evidence Optional immutable original-source coordinates.
 * @param comparison Explicit oracle label, preventing CPU/native conflation.
 *
 * NaNs remain strict byte comparisons. Classification reports whether all
 * differences are NaN-payload changes; it never suppresses a failed assertion.
 */
void sameBytes(const std::vector<float> &actual,const std::vector<float> &expected,
    const SourceEvidence *evidence=nullptr,const char *comparison="complete payload")
{
    ASSERT_EQ(actual.size(),expected.size());
    if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0)return;
    const auto isNaN=[](std::uint32_t bits) {
        return (bits&0x7f800000u)==0x7f800000u&&(bits&0x007fffffu)!=0;
    };
    std::size_t first=actual.size(),nanPayloadDifferences=0,otherDifferences=0;
    for(std::size_t i=0;i<actual.size();++i)
    {
        const auto a=std::bit_cast<std::uint32_t>(actual[i]);
        const auto e=std::bit_cast<std::uint32_t>(expected[i]);
        if(a==e)continue;
        if(first==actual.size())first=i;
        if(isNaN(a)&&isNaN(e))++nanPayloadDifferences;
        else ++otherDifferences;
    }
    std::ostringstream detail;
    detail<<comparison<<" first_index="<<first
        <<" actual_bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(actual[first])
        <<" expected_bits=0x"<<std::bit_cast<std::uint32_t>(expected[first])<<std::dec
        <<" nan_payload_differences="<<nanPayloadDifferences
        <<" other_differences="<<otherDifferences;
    if(evidence)
    {
        const auto gatheredRow=first/static_cast<std::size_t>(evidence->width);
        const auto column=first%static_cast<std::size_t>(evidence->width);
        detail<<" gathered_id_index="<<gatheredRow<<" column="<<column
            <<" live_ids="<<evidence->liveIds;
        if(gatheredRow<evidence->liveIds)
        {
            const auto id=evidence->ids[gatheredRow];
            detail<<" source_row="<<id;
            const auto *row=evidence->source.data()+static_cast<std::size_t>(id)*evidence->stride;
            if(evidence->type==TensorType::FP16||evidence->type==TensorType::BF16)
            {
                std::uint16_t bits;std::memcpy(&bits,row+column*sizeof(bits),sizeof(bits));
                detail<<" source_bits=0x"<<std::hex<<bits<<std::dec;
            }
            else if(evidence->type==TensorType::FP32)
            {
                std::uint32_t bits;std::memcpy(&bits,row+column*sizeof(bits),sizeof(bits));
                detail<<" source_bits=0x"<<std::hex<<bits<<std::dec;
            }
        }
        else detail<<" inactive_poison=true";
    }
    ADD_FAILURE()<<detail.str();
}
/** @brief Retain exact source zeros, subnormals, infinities and quiet-NaN payload controls. */
void addSourceEdgeCases(TensorBase &source)
{
    auto *raw=source.raw_mutable_data();
    if(source.native_type()==TensorType::FP32)
    {
        constexpr std::array<std::uint32_t,8> bits{0,0x80000000u,1,0x80000001u,0x7f7fffffu,0x7f800000u,0xff800000u,0x7fc12345u};
        std::memcpy(raw,bits.data(),sizeof(bits));
    }
    else if(source.native_type()==TensorType::FP16)
    {
        constexpr std::array<std::uint16_t,16> bits{
            0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,
            0x7bff,0xfbff,0x7c00,0xfc00,0x7e55,0xfe55,0x7c01,0xfc01};
        std::memcpy(raw,bits.data(),sizeof(bits));
    }
    else if(source.native_type()==TensorType::BF16)
    {
        constexpr std::array<std::uint16_t,8> bits{0,0x8000,1,0x8001,0x7f7f,0x7f80,0xff80,0x7fc5};
        std::memcpy(raw,bits.data(),sizeof(bits));
    }
    else if(source.native_type()==TensorType::Q8_K)
    {
        auto *blocks=static_cast<Q8_KBlock *>(raw);
        constexpr std::array<std::uint32_t,4> bits{1,0x80000001u,0x00800000u,0x80800000u};
        for(std::size_t i=0;i<bits.size();++i)
        {
            std::memcpy(&blocks[i].d,&bits[i],sizeof(float));
            std::fill_n(blocks[i].qs,Q8_KBlock::BLOCK_SIZE,std::int8_t{1});
        }
    }
}
/** @brief Capture and replay one unchanged source format and admitted geometry. */
void proveFormat(DeviceId device,IBackend *backend,IWorkerGPUContext &worker,
    const test::EmbeddingVerifierFormatCase &format,int width,int capacity)
{
    SCOPED_TRACE(::testing::Message()<<device.toString()<<'/'<<format.label<<" H="<<width<<" M="<<capacity);
    constexpr int tableRows=37;
    auto source=format.create({tableRows,static_cast<std::size_t>(width)},1701);
    addSourceEdgeCases(*source);
    const auto type=source->native_type();
    const auto stride=type==TensorType::FP32?static_cast<std::size_t>(width)*4:
        type==TensorType::FP16||type==TensorType::BF16?static_cast<std::size_t>(width)*2:
        NativeEmbeddingLayout::forSource(type,1,width).row_stride_bytes;
    ASSERT_EQ(source->size_bytes(),tableRows*stride);
    const auto *raw=static_cast<const std::byte *>(source->raw_data());
    const std::vector<std::byte> original(raw,raw+tableRows*stride);
    Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{},{},{}};
    auto *table=storage.allocate<std::byte>(original.size());storage.upload(table,original);
    const auto view=ImmutableResidentTableView::bind(device,type,table,tableRows,width,stride,original.size(),1);
#ifdef HAVE_CUDA
    if(device.is_cuda())storage.kernels=createCUDAImmutableResidentLookupKernels(view);
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())storage.kernels=createROCmImmutableResidentLookupKernels(view);
#endif
    require(bool(storage.kernels),"Resident lookup backend factory unavailable");
    EXPECT_EQ(storage.kernels->table(),view);
    const auto idCapacity=static_cast<std::size_t>(capacity)*16,outputElements=idCapacity*width;
    auto *ids=storage.allocate<std::int64_t>(idCapacity);
    auto *output=storage.allocate<float>(outputElements),*serial=storage.allocate<float>(outputElements);
    auto *live=storage.allocate<int>(1),*healthy=storage.allocate<int>(1),*error=storage.allocate<int>(1);
    std::vector<std::int64_t> hostIds(idCapacity);
    for(std::size_t i=0;i<hostIds.size();++i)hostIds[i]=(i*17+i/7)%tableRows;
    storage.upload(ids,hostIds);storage.upload(live,std::vector{capacity});
    storage.upload(healthy,std::vector{1});storage.upload(error,std::vector{0});
    const std::vector<float> poison(outputElements,-12345.0F);
    storage.upload(output,poison);storage.upload(serial,poison);
    ImmutableResidentGatherArgs args{{capacity,live},ids,idCapacity,output,outputElements,healthy,error};
    storage.graph=worker.createGraphCapture(storage.stream);
    {
        ScopedBackendGraphCapture capture(worker,*storage.graph,"native resident immutable lookup");
        require(capture.begin(),"Resident lookup capture begin failed");
        storage.kernels->gather(args,storage.stream);capture.finish();
    }
    EXPECT_EQ(storage.graph->nodeCount(),2u);
    std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
    require(storage.graph->inspectKernelNodes(nodes,&detail),detail.c_str());
    ASSERT_EQ(nodes.size(),2u);
    for(const auto &node:nodes)
    {
        EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
        EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);
        EXPECT_TRUE(node.block_x==128u||node.block_x==256u);
    }
    require(storage.graph->instantiate(),"Resident lookup instantiation failed");
    const auto owners=storage.owners.size();
    std::vector<int> lives{capacity,1,0,std::max(1,capacity/2),capacity};
    if(capacity==15)for(int rows=0;rows<=15;++rows)lives.push_back(rows);
    for(const int rows:lives)
    {
        SCOPED_TRACE(::testing::Message()<<"live_token_rows="<<rows);
        const auto count=static_cast<std::size_t>(rows)*16;
        for(std::size_t i=0;i<hostIds.size();++i)hostIds[i]=i<count?static_cast<std::int64_t>((i*17+i/7)%tableRows):-1;
        storage.upload(ids,hostIds);storage.upload(live,std::vector{rows});storage.upload(output,poison);
        require(storage.graph->launch(),"Resident lookup captured replay failed");
        std::vector<std::byte> gathered(count*stride);
        for(std::size_t row=0;row<count;++row)
            std::memcpy(gathered.data()+row*stride,original.data()+hostIds[row]*stride,stride);
        auto expected=poison;
        decodeCPUImmutableLookupRows(type,width,stride,gathered.data(),count,expected.data());
        const SourceEvidence evidence{type,width,stride,hostIds,original,count};
        const auto grouped=storage.download(output,outputElements);
        sameBytes(grouped,expected,&evidence,"captured GPU versus CPU");
        EXPECT_EQ(storage.download(healthy,1).front(),1);EXPECT_EQ(storage.download(error,1).front(),0);
        EXPECT_EQ(storage.owners.size(),owners);EXPECT_EQ(storage.kernels->table(),view);
        // Native serial rows are an oracle only. Small complete sweeps also
        // prove identical real GPU publication independent of grouped width.
        if(capacity<=15)
        {
            storage.upload(serial,poison);
            for(int row=0;row<rows;++row)
            {
                auto one=args;one.tokenRows={1,nullptr};one.ids=ids+row*16;one.idCapacity=16;
                one.output=serial+static_cast<std::size_t>(row)*16*width;one.outputElements=16*width;
                storage.kernels->gather(one,storage.stream);
            }
            const auto serialResult=storage.download(serial,outputElements);
            sameBytes(serialResult,expected,&evidence,"serial GPU versus CPU");
            sameBytes(grouped,serialResult,&evidence,"captured GPU versus serial GPU");
        }
    }
    // Invalid IDs anywhere in the live prefix suppress every output write,
    // while an invalid inactive ID was already proved safe above.
    for(const auto bad:std::array<std::int64_t,2>{-1,tableRows})
    {
        std::fill(hostIds.begin(),hostIds.end(),0);hostIds.back()=bad;
        storage.upload(ids,hostIds);storage.upload(live,std::vector{capacity});storage.upload(output,poison);
        storage.upload(healthy,std::vector{1});storage.upload(error,std::vector{0});
        require(storage.graph->launch(),"Resident lookup invalid-ID graph submission failed");
        sameBytes(storage.download(output,outputElements),poison);
        EXPECT_EQ(storage.download(healthy,1).front(),0);
        EXPECT_EQ(storage.download(error,1).front(),static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    }
    for(const int invalid:{-1,capacity+1})
    {
        storage.upload(live,std::vector{invalid});storage.upload(output,poison);
        storage.upload(healthy,std::vector{1});storage.upload(error,std::vector{79});
        require(storage.graph->launch(),"Resident lookup invalid-count graph submission failed");
        sameBytes(storage.download(output,outputElements),poison);
        EXPECT_EQ(storage.download(healthy,1).front(),0);EXPECT_EQ(storage.download(error,1).front(),79);
    }
    EXPECT_EQ(storage.download(table,original.size()),original);
    EXPECT_EQ(storage.graph->nodeCount(),2u);EXPECT_EQ(storage.owners.size(),owners);
}
/**
 * @brief Isolate canonical binary16 widening through actual captured gather.
 * @param device Exact native backend under test.
 *
 * Positive/negative quiet and signaling NaNs retain their source payload bits;
 * infinities, signed zeros and both subnormal boundaries remain exact. The
 * same strict helper compares captured output with CPU and actual serial GPU
 * rows, so no backend-specific NaN exception can hide publication differences.
 */
void proveHalfPayload(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    const auto &formats=test::embeddingVerifierFormats();
    const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &entry) {
        return std::string_view(entry.label)=="FP16";
    });
    ASSERT_NE(format,formats.end());
    worker.submitAsync([&] {
        for(const int rows:{1,15})proveFormat(device,backend,worker,*format,256,rows);
    }).get();
}
/** @brief Exercise the canonical format inventory and actual IQ4_NL table width on one backend. */
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        for(const auto &format:test::embeddingVerifierFormats())
        {
            proveFormat(device,backend,worker,format,256,1);
            proveFormat(device,backend,worker,format,256,15);
            if(std::string_view(format.label)=="IQ4_NL")
                for(const int rows:{1,15,512,1024})proveFormat(device,backend,worker,format,160,rows);
        }
    }).get();
}
}
#ifdef HAVE_CUDA
/** @test CUDA retains two native kernels and exact original-source/live-row semantics. */
TEST(ImmutableResidentLookupDevice, CUDAOriginalFormatsAndCapturedLiveRows){prove(DeviceId::cuda(0));}
/** @test CUDA canonical half widening preserves source sign/payload and serial bytes. */
TEST(ImmutableResidentLookupDevice, CUDAHalfSourcePayloadPreserved){proveHalfPayload(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm retains two native kernels and exact original-source/live-row semantics. */
TEST(ImmutableResidentLookupDevice, ROCmOriginalFormatsAndCapturedLiveRows){prove(DeviceId::rocm(0));}
/** @test ROCm shares the exact binary16 source-bit contract and live-prefix controls. */
TEST(ImmutableResidentLookupDevice, ROCmHalfSourcePayloadPreserved){proveHalfPayload(DeviceId::rocm(0));}
#endif
