/**
 * @file Test__BranchConcatDevice.cpp
 * @brief Captured CUDA/HIP Dense0 stage proof with exact original token counts.
 *
 * This explicit native fixture is never a Unit test. Terminal copies and stream
 * joins are diagnostic boundaries, outside capture. All captured tensor and
 * control owners outlive graph retirement on success, assertion and exception.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpBranchConcatStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "utils/GraphArenaTestHarness.h"
#include <bit>
#include <array>
#include <cstring>
#include <stdexcept>
using namespace llaminar2;
namespace {
/** @brief Stop the native fixture on an infrastructure failure. */
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
/** @brief Complete and retire captured work before any borrowed owner can unwind. */
struct Retirement {
    IBackend *backend;DeviceId device;void *stream;std::unique_ptr<IGPUGraphCapture>graph;
    /** @brief Failed native retirement is fatal, never permission to destroy live storage. */
    ~Retirement()noexcept {if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();graph.reset();}
};
/** @brief Prove actual prepared stage capture, serial bytes and dirty inactive extents. */
void prove(DeviceId device){
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    auto context=IDeviceContext::create(device,1);require(bool(context),"BranchConcat native context failed");
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        void *stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(int m:{1,15,512,1024})for(int h:{7,257,2560}){
            FP32Tensor hidden({m,h*4}),embedding({m,h}),output({m,h*8}),serial({m,h*8});
            INT32Tensor control({1});
            for(std::size_t i=0;i<hidden.numel();++i)hidden.mutable_data()[i]=std::bit_cast<float>(std::uint32_t(i%7==0?0x80000000u:0x3f000000u+std::uint32_t(i%1000)));
            for(std::size_t i=0;i<embedding.numel();++i)embedding.mutable_data()[i]=std::bit_cast<float>(std::uint32_t(i%7==0?0x7fc01234u:0x40000000u+std::uint32_t(i%1000)));
            std::memset(output.raw_mutable_data(),0xcd,output.size_bytes());std::memset(serial.raw_mutable_data(),0xcd,serial.size_bytes());
            *static_cast<int*>(control.raw_mutable_data())=m;
            TransferEngine::prepareDeviceInput(&control,device,stream);TransferEngine::requireDeviceInput(&control,device,stream);
            auto *count=static_cast<int*>(control.gpu_data_ptr());
            Qwen4ExpBranchConcatStage stage({.device_id=device,.rows=DeviceRowRange::deviceCounted(m,count),.hidden_dim=h,
                .hidden={hidden,BufferId::MTP_NORM_HIDDEN},.embedding={embedding,BufferId::MTP_NORM_EMBEDDING},.output={output,BufferId::MTP_CONCAT}});
            stage.setGPUStream(stream);const auto signature=stage.graphCaptureVariantSignature();
            EXPECT_EQ(hidden.gpu_data_ptr(),nullptr);
 EXPECT_EQ(output.gpu_data_ptr(),nullptr);
            for(TensorBase *t:std::array<TensorBase*,4>{&hidden,&embedding,&output,&serial}){
                TransferEngine::prepareDeviceInput(t,device,stream);TransferEngine::requireDeviceInput(t,device,stream);
            }
            require(stage.prepareGraphLaunch(context.get(),stream),"BranchConcat prepared storage seal failed");
            EXPECT_EQ(signature,stage.graphCaptureVariantSignature());
            // Declared after every captured owner and stage: all unwind paths
            // complete graph work before those owners are released.
            Retirement retirement{backend,device,stream,worker.createGraphCapture(stream)};
            {ScopedBackendGraphCapture capture(worker,*retirement.graph,"Dense0 original-token BranchConcat");
                require(capture.begin(),"BranchConcat capture begin failed");
                require(stage.execute(context.get()),"BranchConcat stage capture failed");
                capture.finish();}
            EXPECT_EQ(retirement.graph->nodeCount(),1u);
            std::vector<GPUGraphKernelNodeInfo>nodes;std::string error;
            require(retirement.graph->inspectKernelNodes(nodes,&error),error.c_str());
            ASSERT_EQ(nodes.size(),1u);
            EXPECT_EQ(nodes[0].local_memory_bytes_per_thread,0u);
            require(retirement.graph->instantiate(),"BranchConcat graph instantiation failed");
            const auto operation=qwen4exp::createBranchConcatOperation(device);
            for(int row=0;row<m;++row)operation({DeviceRowRange::fullyActive(1),h,
                static_cast<float*>(hidden.gpu_data_ptr())+std::size_t(row)*h*4,
                static_cast<float*>(embedding.gpu_data_ptr())+std::size_t(row)*h,
                static_cast<float*>(serial.gpu_data_ptr())+std::size_t(row)*h*8},stream);
            std::vector<float>serialHost(std::size_t(m)*h*8),actual(serialHost.size()),poison(serialHost.size());
            std::memset(poison.data(),0xcd,poison.size()*4);
            require(backend->deviceToHostOnStream(serialHost.data(),serial.gpu_data_ptr(),serial.size_bytes(),device.ordinal,stream),"BranchConcat serial terminal copy failed");
            require(backend->synchronizeStream(stream,device.ordinal),"BranchConcat serial terminal completion failed");
            for(int row=0;row<m;++row)for(int branch=0;branch<4;++branch){
                const auto *p=serialHost.data()+(std::size_t(row)*4+branch)*h*2;
                EXPECT_EQ(std::memcmp(p,embedding.data()+std::size_t(row)*h,std::size_t(h)*4),0);
                EXPECT_EQ(std::memcmp(p+h,hidden.data()+(std::size_t(row)*4+branch)*h,std::size_t(h)*4),0);
            }
            for(int live:{m,m>1?m-1:0,0,m>2?2:m}){
                require(backend->hostToDeviceOnStream(count,&live,4,device.ordinal,stream),"BranchConcat count publication failed");
                require(backend->hostToDeviceOnStream(output.gpu_data_ptr(),poison.data(),output.size_bytes(),device.ordinal,stream),"BranchConcat diagnostic poison publication failed");
                require(retirement.graph->launch(),"BranchConcat retained replay failed");
                require(backend->deviceToHostOnStream(actual.data(),output.gpu_data_ptr(),output.size_bytes(),device.ordinal,stream),"BranchConcat terminal observation failed");
                require(backend->synchronizeStream(stream,device.ordinal),"BranchConcat terminal completion failed");
                EXPECT_EQ(std::memcmp(actual.data(),serialHost.data(),std::size_t(live)*h*32),0);
                EXPECT_EQ(std::memcmp(actual.data()+std::size_t(live)*h*8,poison.data()+std::size_t(live)*h*8,std::size_t(m-live)*h*32),0);
                EXPECT_EQ(signature,stage.graphCaptureVariantSignature());
            }
            // Retirement owns completion before tensor/stage destruction.
        }
    }).get();
}
/** @brief Retire executor cache before its stage/arena roots on every unwind path. */
struct CacheRetirement {
    IBackend*backend;DeviceId device;void*stream;DeviceGraphExecutor::GraphSegmentCache&cache;
    /** @brief Observe the real capture stream before destroying its executable. */
    ~CacheRetirement()noexcept {
        void*completion=cache.capture_stream?cache.capture_stream:stream;
        if(!backend->synchronizeStream(completion,device.ordinal))std::terminate();cache.reset();
    }
};
/** @brief Real executor first-use/capture/replay proof with initially absent GPU addresses. */
void proveExecutorIdentity(DeviceId device){
    auto*backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested backend unavailable";
    auto context=IDeviceContext::create(device,1);require(bool(context),"BranchConcat executor context failed");
    auto&worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        constexpr int m=15,h=257;test::GraphArenaTestHarness arena;
        auto*hidden=arena.createPersistentTensor<FP32Tensor>(BufferId::MTP_NORM_HIDDEN,std::vector<std::size_t>{m,h*4});
        auto*embedding=arena.createPersistentTensor<FP32Tensor>(BufferId::MTP_NORM_EMBEDDING,std::vector<std::size_t>{m,h});
        auto*output=arena.createPersistentTensor<FP32Tensor>(BufferId::MTP_CONCAT,std::vector<std::size_t>{m,h*8});
        for(std::size_t i=0;i<hidden->numel();++i)hidden->mutable_data()[i]=float(i%67);
        for(std::size_t i=0;i<embedding->numel();++i)embedding->mutable_data()[i]=float(-int(i%71));
        auto owned=std::make_unique<Qwen4ExpBranchConcatStage>(Qwen4ExpBranchConcatStage::Params{.device_id=device,.rows=DeviceRowRange::fullyActive(m),.hidden_dim=h,
            .hidden={*hidden,BufferId::MTP_NORM_HIDDEN},.embedding={*embedding,BufferId::MTP_NORM_EMBEDDING},.output={*output,BufferId::MTP_CONCAT}});
        auto*stage=owned.get();const auto declared=stage->graphCaptureVariantSignature();
        ASSERT_EQ(hidden->gpu_data_ptr(),nullptr);
        ASSERT_EQ(embedding->gpu_data_ptr(),nullptr);
        ASSERT_EQ(output->gpu_data_ptr(),nullptr);
        ComputeGraph graph;graph.addNode("branch_concat",std::move(owned),device);
        GraphExecutorConfig config;config.enable_validation=false;
        DeviceGraphExecutor executor(config);arena.bindExecutor(executor);
        DeviceGraphExecutor::GraphSegmentCache cache;void*stream=ExplicitGPUStream(worker.defaultStream()).get();
        CacheRetirement retirement{backend,device,stream,cache};
        // This isolated graph has no preceding native input producer. Admit
        // its external input bytes normally; leave the output entirely cold.
        TransferEngine::prepareDeviceInput(hidden,device,stream);
        TransferEngine::prepareDeviceInput(embedding,device,stream);
        ASSERT_EQ(output->gpu_data_ptr(),nullptr);
        require(executor.executeWithCachedGraphReplay(graph,context.get(),cache,stream,&worker,nullptr),"BranchConcat executor cold capture failed");
        std::array<const void*,3> originalAddresses{};
        const std::array<TensorBase*,3> roots{hidden,embedding,output};
        for(std::size_t i=0;i<roots.size();++i){ASSERT_NE(roots[i]->gpu_data_ptr(),nullptr);
            EXPECT_EQ(roots[i]->current_device(),std::optional<DeviceId>(device));originalAddresses[i]=roots[i]->gpu_data_ptr();}
        ASSERT_TRUE(cache.initialized);
 ASSERT_FALSE(cache.needs_capture);
 EXPECT_EQ(cache.variant_recapture_count,0u);
        EXPECT_EQ(declared,stage->graphCaptureVariantSignature());
 ASSERT_EQ(cache.segments.size(),1u);
        const auto*executable=cache.segments[0].capture.get();
        for(int replay=0;replay<3;++replay){graph.reset();require(executor.executeWithCachedGraphReplay(graph,context.get(),cache,stream,&worker,nullptr),"BranchConcat executor cached replay failed");
 EXPECT_EQ(cache.variant_recapture_count,0u);
 EXPECT_EQ(executable,cache.segments[0].capture.get());
 EXPECT_EQ(declared,stage->graphCaptureVariantSignature());
            for(std::size_t i=0;i<roots.size();++i)EXPECT_EQ(roots[i]->gpu_data_ptr(),originalAddresses[i]);}
        // Arena replay intentionally leaves completion with its graph owner.
        // This terminal host observer publishes that exact producer stream
        // through TransferEngine before reading or retiring the output root.
        TransferEngine::publishDeviceWrite(output,device,cache.capture_stream);
        TransferEngine::prepareHostInput(output);
        const float*actual=output->data();
        for(int row=0;row<m;++row)for(int branch=0;branch<4;++branch){const auto*p=actual+(row*4+branch)*h*2;
 EXPECT_EQ(std::memcmp(p,embedding->data()+row*h,h*4),0);
 EXPECT_EQ(std::memcmp(p+h,hidden->data()+(row*4+branch)*h,h*4),0);}
        // Deliberate stale-storage negative control after completed readers:
        // the next executor identity check must reject before submitting work.
        ASSERT_TRUE(output->releaseDeviceMemory());
 EXPECT_THROW(stage->graphCaptureVariantSignature(),std::logic_error);
        EXPECT_THROW(executor.executeWithCachedGraphReplay(graph,context.get(),cache,stream,&worker,nullptr),std::logic_error);
    }).get();
}

}
#ifdef HAVE_CUDA
TEST(BranchConcatDevice, CUDAOriginalTokenCapturedStage){prove(DeviceId::cuda(0));}
TEST(BranchConcatDevice, CUDAExecutorPreparedIdentity){proveExecutorIdentity(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(BranchConcatDevice, ROCmOriginalTokenCapturedStage){prove(DeviceId::rocm(0));}
TEST(BranchConcatDevice, ROCmExecutorPreparedIdentity){proveExecutorIdentity(DeviceId::rocm(0));}
#endif
