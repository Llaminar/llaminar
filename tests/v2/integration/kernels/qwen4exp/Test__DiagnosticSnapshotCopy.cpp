/**
 * @file Test__DiagnosticSnapshotCopy.cpp
 * @brief Native captured authored-copy and production terminal-observer extent proof.
 *
 * Persistent diagnostic slots retain maximum capacity. Real CUDA/HIP replay
 * changes only native counts; the production executor downloads headers, exact
 * offsets and authored bytes, leaving poisoned host capacity untouched. The
 * diagnostic callback reconstructs canonical selection padding after validation.
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "snapshots/SnapshotCapture.h"
#include "tensors/Tensors.h"
#include "tensors/GPUTensorVerification.h"
#include <array>
#include <limits>
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace llaminar2
{
/** @brief Exercise the exact terminal acquisition/publication infrastructure without model wiring. */
struct GraphSnapshotSemanticExtentTestAccess
{
    /** @brief Observe only already-produced native diagnostic slots, then publish their callbacks. */
    static bool observe(DeviceGraphExecutor &executor,DeviceGraphExecutor::GraphSnapshotManifest &manifest,void *stream,
        const std::string &stage="layer1_qsa_indexer")
    {
        return executor.acquireGraphSnapshotArenaOnHost(stream,manifest)&&executor.publishGraphSnapshotCopies(
            stage,stream,manifest,nullptr,
            DeviceGraphExecutor::GraphSnapshotPublicationMode::ArenaAlreadyAcquired);
    }
};
}
using namespace llaminar2;
namespace
{
/** @brief Fail a native test immediately when publication or graph recording fails. */
void check(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
/** @brief Prove real retained capture and exact terminal D2H extents on one assigned GPU. */
void proveDiagnosticObservation(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        auto *stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(const int rows:{1,15,512,1024})
        {
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<rows);
            constexpr int columns=2051;
            INT32Tensor source({static_cast<size_t>(rows),columns}),controls({static_cast<size_t>(rows)+2});
            FP32Tensor pools({static_cast<size_t>(rows),128});
            auto *ids=static_cast<std::int32_t *>(source.raw_mutable_data());
            auto *counts=static_cast<std::int32_t *>(controls.raw_mutable_data());
            for(int row=0;row<rows;++row)for(int col=0;col<columns;++col)ids[size_t(row)*columns+col]=col;
            std::iota(pools.mutable_data(),pools.mutable_data()+pools.numel(),1.f);
            const size_t selectionBytes=size_t(rows)*columns*4,metadataOffset=(selectionBytes+7)&~size_t(7);
            const size_t selectionMetadata=16+(size_t(rows)+1)*8;
            const size_t stateOffset=(metadataOffset+selectionMetadata+63)&~size_t(63);
            const size_t stateBytes=size_t(rows)*128*4,totalBytes=stateOffset+stateBytes+16;
            auto arena=std::make_shared<FP32Tensor>(std::vector<size_t>{1,(totalBytes+3)/4},DeviceId::cpu());
            std::fill_n(arena->mutable_data(),arena->numel(),0.f);
            std::fill_n(counts,rows+2,0);
            for(TensorBase *tensor:std::vector<TensorBase *>{&source,&controls,&pools,arena.get()}) {
                TransferEngine::prepareDeviceInput(tensor,device,stream);
                TransferEngine::requireDeviceInput(tensor,device,stream);
            }
            auto *gpuArena=static_cast<std::byte *>(arena->gpu_data_ptr());
            auto *gpuCounts=static_cast<std::int32_t *>(controls.gpu_data_ptr());
            const auto copier=DiagnosticSnapshotCopy::prepare(device);
            DiagnosticSnapshotCopyArgs selection{{DiagnosticSnapshotCopyKind::CountedSelectionRows,rows,columns,4},
                source.gpu_data_ptr(),gpuArena,reinterpret_cast<SnapshotObservedHeader *>(gpuArena+metadataOffset),nullptr,gpuCounts+1,gpuCounts+2};
            DiagnosticSnapshotCopyArgs state{{DiagnosticSnapshotCopyKind::StatePrefixRows,rows,128,4,rows*4,4},
                pools.gpu_data_ptr(),gpuArena+stateOffset,reinterpret_cast<SnapshotObservedHeader *>(gpuArena+stateOffset+stateBytes),gpuCounts};
            DeviceGraphExecutor executor;SnapshotCapture captured;
            executor.setSnapshotCallback([&](const std::string &name,const StageDumpInfo &dump){captured.captureStage(name,dump);});
            DeviceGraphExecutor::GraphSnapshotManifest manifest;
            manifest.storage_arena=arena;manifest.storage_device=device;manifest.storage_capacity_bytes=arena->size_bytes();
            manifest.storage_required_bytes=totalBytes;manifest.bound_slot_count=2;
            manifest.storage_state=DeviceGraphExecutor::GraphSnapshotManifest::StorageState::StorageBound;
            manifest.semantic_copier=copier;
            DeviceGraphExecutor::GraphSnapshotOutputCopy a;
            a.name="selected";a.dtype="INT32";a.rows=rows;a.cols=columns;a.element_size=4;a.byte_size=selectionBytes;
            a.storage_bytes=metadataOffset+selectionMetadata;a.device=device;a.source_ptr=source.gpu_data_ptr();
            a.semantic_extent=SnapshotCountedSelectionRows{};a.metadata_offset_bytes=metadataOffset;
            auto b=a;b.name="pooled_keys";b.dtype="FP32";b.cols=128;b.byte_size=stateBytes;b.storage_bytes=stateBytes+16;
            b.storage_offset_bytes=stateOffset;b.metadata_offset_bytes=stateBytes;b.source_ptr=pools.gpu_data_ptr();
            b.semantic_extent=SnapshotStatePrefixRows{{},rows*4,4};
            manifest.stage_copies["layer1_qsa_indexer"].outputs={a,b};
            auto graph=worker.createGraphCapture(stream);check(bool(graph),"Missing diagnostic graph owner");
            {ScopedBackendGraphCapture capture(worker,*graph,"Authored diagnostic snapshots");check(capture.begin(),"Diagnostic capture failed");
                copier.capture(selection,stream);copier.capture(state,stream);capture.finish();}
            std::vector<GPUGraphKernelNodeInfo> resources;std::string error;
            check(graph->inspectKernelNodes(resources,&error),error.c_str());ASSERT_EQ(resources.size(),4u);
            for(const auto &node:resources)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
            check(graph->instantiate(),"Diagnostic native graph instantiation failed");
            for(const int live:{rows,std::min(rows,2),0,rows})
            {
                counts[0]=0;counts[1]=live;size_t authored=0;
                for(int row=0;row<rows;++row){counts[row+2]=row<live?(row%4==0?0:row%4==1?1:row%4==2?columns:17):-99; if(row<live)authored+=counts[row+2];}
                check(backend->hostToDeviceOnStream(gpuCounts,counts,controls.size_bytes(),device.ordinal,stream),"Diagnostic live-count upload failed");
                auto *hostArena=static_cast<std::byte *>(const_cast<void *>(arena->raw_data()));std::memset(hostArena,0xc3,arena->size_bytes());
                check(graph->launch(),"Diagnostic retained replay failed");
                const auto before=manifest.observed_download_bytes;
                check(GraphSnapshotSemanticExtentTestAccess::observe(executor,manifest,stream),"Diagnostic terminal observation failed");
                const size_t stateLiveBytes=size_t(live/4)*128*4;
                EXPECT_EQ(manifest.observed_download_bytes-before,32+(size_t(live)+1)*8+authored*4+stateLiveBytes);
                EXPECT_TRUE(std::all_of(hostArena+authored*4,hostArena+selectionBytes,[](auto x){return x==std::byte{0xc3};}));
                EXPECT_TRUE(std::all_of(hostArena+stateOffset+stateLiveBytes,hostArena+stateOffset+stateBytes,[](auto x){return x==std::byte{0xc3};}));
                const auto selected=captured.getShared("layer1_QSA_SELECTED_INDICES"),pool=captured.getShared("layer1_QSA_INDEXER_POOLED_KEYS");
                ASSERT_TRUE(selected);ASSERT_TRUE(pool);EXPECT_EQ(selected->rows,size_t(live));EXPECT_EQ(pool->rows,size_t(live/4));
                for(int row=0;row<live;++row)for(int col=0;col<columns;++col)
                    EXPECT_EQ(selected->data[size_t(row)*columns+col],col<counts[row+2]?float(col):-1.f);
            }
            graph.reset();
        }
    }).get();
}
}
#ifdef HAVE_CUDA
/** @test CUDA retained capture copies and downloads only authored diagnostic extents. */
TEST(DiagnosticSnapshotCopy,CUDA){proveDiagnosticObservation(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm retained capture copies and downloads only authored diagnostic extents. */
TEST(DiagnosticSnapshotCopy,ROCm){proveDiagnosticObservation(DeviceId::rocm(0));}
#endif

namespace
{
/**
 * @brief Retained native transaction proves original holes, exact bytes and strict input scans.
 * @param device Exact CUDA/ROCm participant selected by the explicit integration test.
 * @details Setup owns all storage; native metadata/copy nodes retain original
 * publication addresses. Test-only ordered replay publication changes values,
 * never topology. Only the existing terminal observer downloads authored bytes.
 */
void proveAuthoredQueryObservation(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        auto *stream=ExplicitGPUStream(worker.defaultStream()).get();
        constexpr int requests=2,rows=8,columns=4;
        FP32Tensor source({rows,columns});INT32Tensor controls({4});
        auto *words=static_cast<std::int32_t *>(controls.raw_mutable_data());std::fill_n(words,4,0);
        std::fill_n(source.mutable_data(),rows*columns,1.f);
        constexpr std::size_t bytes=rows*columns*4,metadataOffset=bytes;
        constexpr std::size_t totalBytes=metadataOffset+sizeof(SnapshotObservedHeader)+requests*sizeof(SnapshotAuthoredTokenRowRecord);
        auto arena=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,totalBytes/4},DeviceId::cpu());
        std::fill_n(arena->mutable_data(),arena->numel(),0.f);
        for(TensorBase *tensor:std::vector<TensorBase *>{&source,&controls,arena.get()}) {
            TransferEngine::prepareDeviceInput(tensor,device,stream);TransferEngine::requireDeviceInput(tensor,device,stream);
        }
        auto *gpuWords=static_cast<std::int32_t *>(controls.gpu_data_ptr());
        auto *gpuArena=static_cast<std::byte *>(arena->gpu_data_ptr());
        const auto work=ProjectionRowWork::causalStrided(requests,4,gpuWords,gpuWords+1,2,8192,2051);
        const SnapshotAuthoredTokenRows authored{work,{{{&controls,0},{&controls,4},{}},{{&controls,8},{&controls,12},{}}}};
        StageDumpInfo::OutputBuffer output{.name="C",.data=source.gpu_data_ptr(),.rows=rows,.cols=columns,
            .dtype="FP32",.element_size=4,.byte_size=bytes,.tensor=&source,.semantic_extent=authored};
        const auto args=snapshotCopyArgs(output,gpuArena,reinterpret_cast<SnapshotObservedHeader *>(gpuArena+metadataOffset),true,device);
        const auto copier=DiagnosticSnapshotCopy::prepare(device);
        DeviceGraphExecutor executor;SnapshotCapture captured;
        executor.setSnapshotCallback([&](const std::string &name,const StageDumpInfo &dump){captured.captureStage(name,dump);});
        DeviceGraphExecutor::GraphSnapshotManifest manifest;
        manifest.storage_arena=arena;manifest.storage_device=device;manifest.storage_capacity_bytes=arena->size_bytes();
        manifest.storage_required_bytes=totalBytes;manifest.bound_slot_count=1;
        manifest.storage_state=DeviceGraphExecutor::GraphSnapshotManifest::StorageState::StorageBound;manifest.semantic_copier=copier;
        DeviceGraphExecutor::GraphSnapshotOutputCopy copy;
        copy.name="C";copy.dtype="FP32";copy.rows=rows;copy.cols=columns;copy.element_size=4;
        copy.byte_size=bytes;copy.storage_bytes=totalBytes;copy.device=device;copy.source_ptr=source.gpu_data_ptr();
        copy.semantic_extent=authored;copy.metadata_offset_bytes=metadataOffset;
        manifest.stage_copies["layer1_qsa_query_proj"].outputs={copy};
        auto graph=worker.createGraphCapture(stream);check(bool(graph),"Missing authored-query graph owner");
        {ScopedBackendGraphCapture capture(worker,*graph,"Original request authored query observation");
            check(capture.begin(),"Authored query capture failed");copier.capture(args,stream);capture.finish();}
        std::vector<GPUGraphKernelNodeInfo> resources;std::string error;
        check(graph->inspectKernelNodes(resources,&error),error.c_str());ASSERT_EQ(resources.size(),2u);
        for(const auto &node:resources)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
        check(graph->instantiate(),"Authored query graph instantiation failed");
        auto *validator=getTensorValidator(device.type,device.ordinal);ASSERT_NE(validator,nullptr);
        TensorValidationExtent validation{{SnapshotContiguousRowKind::Physical,rows},columns,nullptr,TensorValidationRowSelection::ProjectionWork,work};
        for(const int phase:{0,1,2,0}) {
            const std::array<std::int32_t,4> values=phase==0?std::array<std::int32_t,4>{2049,4,2051,2}:
                phase==1?std::array<std::int32_t,4>{0,4,2048,3}:std::array<std::int32_t,4>{2051,1,2050,3};
            const std::vector<int> expected=phase==0?std::vector<int>{2,3,4,5}:
                phase==1?std::vector<int>{}:std::vector<int>{0,5,6};
            std::copy(values.begin(),values.end(),words);
            auto *payload=source.mutable_data();std::fill_n(payload,rows*columns,std::numeric_limits<float>::quiet_NaN());
            for(const int row:expected)for(int col=0;col<columns;++col)payload[row*columns+col]=float(100+row*columns+col);
            check(backend->hostToDeviceOnStream(gpuWords,words,controls.size_bytes(),device.ordinal,stream),"Original request publication failed");
            check(backend->hostToDeviceOnStream(source.gpu_data_ptr(),payload,source.size_bytes(),device.ordinal,stream),"Authored query payload publication failed");
            auto *hostArena=static_cast<std::byte *>(const_cast<void *>(arena->raw_data()));std::memset(hostArena,0xc3,arena->size_bytes());
            check(graph->launch(),"Retained authored query replay failed");const auto before=manifest.observed_download_bytes;
            check(GraphSnapshotSemanticExtentTestAccess::observe(executor,manifest,stream,"layer1_qsa_query_proj"),"Authored query terminal observation failed");
            EXPECT_EQ(manifest.observed_download_bytes-before,sizeof(SnapshotObservedHeader)+requests*sizeof(SnapshotAuthoredTokenRowRecord)+expected.size()*columns*4);
            EXPECT_TRUE(std::all_of(hostArena+expected.size()*columns*4,hostArena+bytes,[](auto x){return x==std::byte{0xc3};}));
            const auto observed=captured.getShared("layer1_QSA_INDEXER_RAW_QUERY");ASSERT_TRUE(observed);
            EXPECT_EQ(observed->rows,expected.size());EXPECT_EQ(observed->cols,columns);
            ASSERT_TRUE(observed->authored_token_rows);ASSERT_EQ(observed->authored_token_rows->requests.size(),2u);
            for(std::size_t packed=0;packed<expected.size();++packed)for(int col=0;col<columns;++col)
                EXPECT_EQ(observed->data[packed*columns+col],float(100+expected[packed]*columns+col));
            auto result=validator->validateObserved(source.gpu_data_ptr(),validation,TensorValidationDataType::FP32,ExplicitGPUStream{stream});
            EXPECT_TRUE(result.passed());EXPECT_EQ(result.total_checked,expected.size()*columns);
            if(!expected.empty()) {
                payload[expected.front()*columns]=std::numeric_limits<float>::quiet_NaN();
                check(backend->hostToDeviceOnStream(source.gpu_data_ptr(),payload,source.size_bytes(),device.ordinal,stream),"Useful-query poison publication failed");
                result=validator->validateObserved(source.gpu_data_ptr(),validation,TensorValidationDataType::FP32,ExplicitGPUStream{stream});
                EXPECT_FALSE(result.passed());EXPECT_EQ(result.nan_count,1u);
            }
        }
        graph.reset();
    }).get();
}
}
#ifdef HAVE_CUDA
/** @test CUDA captures exact authored intervals and scans only useful original input rows. */
TEST(DiagnosticSnapshotCopy,AuthoredQueryCUDA){proveAuthoredQueryObservation(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm captures exact authored intervals and scans only useful original input rows. */
TEST(DiagnosticSnapshotCopy,AuthoredQueryROCm){proveAuthoredQueryObservation(DeviceId::rocm(0));}
#endif
