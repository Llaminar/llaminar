/**
 * @file Test__Qwen4ExpAuxiliaryAdmission.cpp
 * @brief Prove real schema-backed GPU state materializes before first cache reset.
 *
 * No model weights are loaded. The graph's actual schema and resolver create
 * host-only arena tensors, matching early DGO cache setup before general GPU
 * scratch materialization. The production cache factory and auxiliary adapter
 * must materialize their exact borrowed owners, publish the initializing stream,
 * and let a different consumer stream observe reset bytes without host repair.
 * Protocol/replay roles are registered too, proving their distinct setup lifetime.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../../utils/CPUActivationArenaFixture.h"
#include "../../../unit/models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "../../../unit/models/qwen4exp/Qwen4ExpEmptyAuxiliaryFixture.h"
#include "kernels/IHybridKVCache.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <limits>
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "memory/BufferArena.h"
#include "../../../mocks/MockMPIContext.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "../../../utils/Qwen4ExpPlacementFixture.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Fail fixture infrastructure rather than continuing with incomplete native setup. */
void require(bool value,const char *message)
{
    if(!value)throw std::runtime_error(message);
}
/** @brief Own the distinct setup/consumer streams until terminal observations finish. */
struct Streams
{
    IBackend &backend;
    DeviceId device;
    void *producer, *consumer;
    /** @brief Create both explicit streams before model-state initialization. */
    Streams(IBackend &native,DeviceId target):backend(native),device(target),
        producer(native.createStream(target.ordinal)),consumer(native.createStream(target.ordinal))
    {
        require(producer && consumer && producer!=consumer,"QFN admission fixture requires distinct native streams");
    }
    /** @brief Fixture-only terminal retirement joins outstanding work before releasing streams. */
    ~Streams()
    {
        (void)backend.synchronizeStream(producer,device.ordinal);
        (void)backend.synchronizeStream(consumer,device.ordinal);
        backend.destroyStream(consumer,device.ordinal);
        backend.destroyStream(producer,device.ordinal);
    }
    /** @brief Observe reset bytes only after joining their exact published producer event. */
    template<class T> std::vector<T> read(TensorBase &tensor)
    {
        TransferEngine::requireDeviceInput(&tensor,device,consumer);
        std::vector<T> result(tensor.size_bytes()/sizeof(T));
        require(backend.deviceToHostOnStream(result.data(),tensor.gpu_data_ptr(),tensor.size_bytes(),
            device.ordinal,consumer),"QFN admission terminal observation copy failed");
        require(backend.synchronizeStream(consumer,device.ordinal),"QFN admission terminal observation failed");
        return result;
    }
};
/**
 * @brief Capture exact pool archive export/import without a host completion wait.
 * @param owner Initialized schema-backed accepted state with one sparse layer.
 * @param backend Native backend owning all source/destination addresses.
 * @param worker Exact native graph recorder for this device.
 * @param streams Distinct producer and diagnostic consumer streams.
 *
 * Capture rejects the old blocking deviceToDevice compatibility API. Poisoned
 * trailing archive rows also prove that one completed block copies 512 bytes,
 * independent of the admitted destination capacity. Observation joins an event
 * after graph launch; archive submission itself cannot observe completion.
 */
void proveCapturedArchive(Qwen4ExpAuxiliaryState &owner,IBackend &backend,
    IWorkerGPUContext &worker,Streams &streams)
{
    const auto device=owner.device();
    const auto &entry=owner.bindings().sparseLayers.front();
    auto &pools=*entry.pooledKeys.tensor;
    const CacheAuxiliaryBlockRange first{entry.layer,entry.sequence,0,4};
    const CacheAuxiliaryBlockRange second{entry.layer,entry.sequence,4,4};
    FP32Tensor archive({3,128});
    std::vector<float> poison(archive.numel(),-71.25f);
    std::copy(poison.begin(),poison.end(),static_cast<float *>(archive.raw_mutable_data()));
    TransferEngine::prepareDeviceInput(&archive,device,streams.producer);
    owner.requireReady(streams.producer);
    TransferEngine::requireDeviceInput(&archive,device,streams.producer);
    ASSERT_TRUE(backend.synchronizeStream(streams.producer,device.ordinal));
    auto *archivePointer=archive.gpu_data_ptr();
    ASSERT_NE(archivePointer,nullptr);
    const CacheAuxiliaryMemory payload{archivePointer,archive.size_bytes(),CacheAuxiliaryMemoryDomain::Device};
    EXPECT_THROW(owner.exportBlock(first,payload,nullptr),std::invalid_argument);
    EXPECT_THROW(owner.importBlock(second,payload,nullptr),std::invalid_argument);
    // Empty logical ranges submit no work and need no synthetic pointer/event.
    ASSERT_NO_THROW(owner.exportBlock({entry.layer,entry.sequence,0,0},
        {nullptr,0,CacheAuxiliaryMemoryDomain::Device},nullptr));
    auto graph=worker.createGraphCapture(streams.producer);
    {
        ScopedBackendGraphCapture capture(worker,*graph,"QFN exact asynchronous device archive");
        ASSERT_TRUE(capture.begin());
        owner.exportBlock(first,payload,streams.producer);
        owner.importBlock(second,payload,streams.producer);
        capture.finish();
    }
    ASSERT_GE(graph->nodeCount(),2u);
    ASSERT_TRUE(graph->instantiate());
    for(int replay=0;replay<2;++replay) {
        std::vector<float> poolBytes(pools.size_bytes()/sizeof(float));
        for(std::size_t i=0;i<poolBytes.size();++i)poolBytes[i]=float(1000*replay+i+1);
        ASSERT_TRUE(backend.hostToDeviceOnStream(pools.gpu_data_ptr(),poolBytes.data(),
            pools.size_bytes(),device.ordinal,streams.producer));
        TransferEngine::publishDeviceWrite(&pools,device,streams.producer);
        ASSERT_TRUE(graph->launch());
        // The archive is a raw caller-retained payload; its owner publishes the
        // aggregate completion after all section copies, just as prefix I/O does.
        TransferEngine::publishDeviceWrite(&archive,device,streams.producer);
        TransferEngine::publishDeviceWrite(&pools,device,streams.producer);
        auto expectedArchive=poison;
        const auto offset=entry.pooledKeys.byteOffset/sizeof(float);
        std::copy_n(poolBytes.data()+offset,128,expectedArchive.data());
        std::copy_n(poolBytes.data()+offset,128,poolBytes.data()+offset+128);
        EXPECT_EQ(streams.read<float>(archive),expectedArchive)<<replay;
        EXPECT_EQ(streams.read<float>(pools),poolBytes)<<replay;
    }
}
/**
 * @brief Prove a real GDN-only GPU cache keeps its native frontier without QSA/PLE ownership.
 *
 * Both the first and a nonzero-origin interval come from authenticated metadata.
 * Native cache reset keeps its existing explicit-stream lifecycle; no arena,
 * dummy auxiliary storage, or host copy is introduced to make setup succeed.
 */
void proveAbsentAdmission(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested GPU backend unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        Streams streams(*backend,device);MockMPIContext mpi;
        const auto layers=emptyAuxiliaryLayers();
        ASSERT_GT(layers.size(),1u);
        for(int layer:{layers.front(),layers.back()}) {
            SCOPED_TRACE(layer);
            const auto config=emptyAuxiliaryConfig(device,layer);
            Qwen4ExpGraph graph(config);
            ASSERT_EQ(graph.auxiliaryContent(),Qwen4ExpGraph::AuxiliaryContent::Absent);
            const auto hybridConfig=emptyAuxiliaryHybridConfig(config);
            auto cacheConfig=emptyAuxiliaryKVConfig(config,mpi,hybridConfig);
            const auto kvBytes=cacheConfig.estimateBytes();
            const auto recurrentBytes=hybridConfig.gdnStateGeometry().deviceArenaBytes(
                hybridConfig.countGDNLayers(),cacheConfig.batch_size);
            ASSERT_GT(kvBytes,0u);ASSERT_GT(recurrentBytes,0u);
            ASSERT_LE(kvBytes,std::numeric_limits<std::size_t>::max()-recurrentBytes);
            const PhysicalMemoryAllocatorIdentity identity{mpi.rank(),device};
            // These are the same canonical geometry functions used by the real
            // factory and persistent-state planner. No extra capacity is admitted.
            PhysicalMemoryPlanBuilder plan;
            const PhysicalMemoryResource resource{.world_rank=mpi.rank(),.device=device,
                .total_bytes=kvBytes+recurrentBytes,.admission_available_bytes=kvBytes+recurrentBytes};
            plan.add(resource,PhysicalMemoryOwner::KVCache,kvBytes);
            plan.add(resource,PhysicalMemoryOwner::RecurrentLiveState,recurrentBytes);
            auto admission=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build());
            auto authority=std::make_shared<PhysicalMemoryAuthority>(admission,mpi.rank());
            cacheConfig.physical_memory_authority=authority;
            auto cache=llaminar::v2::kernels::KernelFactory::createKVCache(cacheConfig);
            ASSERT_TRUE(cache->hasPhysicalMemoryLease());
            ASSERT_TRUE(authority->rankComplete());
            EXPECT_EQ(authority->claimedBytes(device,PhysicalMemoryOwner::KVCache,
                PhysicalMemoryMaterializationKind::NewAllocation),kvBytes);
            EXPECT_EQ(authority->claimedBytes(device,PhysicalMemoryOwner::RecurrentLiveState,
                PhysicalMemoryMaterializationKind::NewAllocation),recurrentBytes);
            auto *hybrid=dynamic_cast<IHybridKVCache *>(cache.get());ASSERT_NE(hybrid,nullptr);
            EXPECT_EQ(hybrid->gdnMemoryBytes(),recurrentBytes);
            ASSERT_EQ(hybrid->kvLayerCount(),0);ASSERT_EQ(hybrid->gdnLayerCount(),1);
            ASSERT_TRUE(cache->supportsDeviceResidentSequenceStatePublication());
            auto *state=hybrid->getGDNState(layer);ASSERT_NE(state,nullptr);
            ASSERT_TRUE(state->recurrence_state.empty());ASSERT_TRUE(state->conv_state.empty());
            auto *recurrence=state->recurrence_device_state.request_state_bank;
            auto *conv=state->conv_device_state.request_state_bank;
            ASSERT_NE(recurrence,nullptr);ASSERT_NE(conv,nullptr);
            EXPECT_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr),std::invalid_argument);
            ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,streams.producer));
            ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,streams.consumer));
            EXPECT_EQ(cache->auxiliaryState(),nullptr);
            EXPECT_EQ(state->recurrence_device_state.request_state_bank,recurrence);
            EXPECT_EQ(state->conv_device_state.request_state_bank,conv);
            // The actual cache, not an absent extension, still resets native GDN.
            ASSERT_TRUE(cache->resetRequestState(IKVCache::StateResetContext::testReinitialization(streams.consumer)));
            EXPECT_EQ(cache->auxiliaryState(),nullptr);
            EXPECT_EQ(state->recurrence_device_state.request_state_bank,recurrence);
            EXPECT_EQ(state->conv_device_state.request_state_bank,conv);
            // Terminal fixture retirement completes native reset before its cache owner dies.
            ASSERT_TRUE(backend->synchronizeStream(streams.consumer,device.ordinal));
            // No auxiliary banks or owners are admitted/materialized by the hook.
            ASSERT_TRUE(authority->rankComplete());
            for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i) {
                const auto owner=PhysicalMemoryBOM::ownerAt(i);
                if(owner==PhysicalMemoryOwner::KVCache || owner==PhysicalMemoryOwner::RecurrentLiveState)continue;
                EXPECT_EQ(admission->plan().find(identity)->bytes(owner),0u)<<toString(owner);
                EXPECT_EQ(authority->claimedBytes(device,owner,
                    PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
            }
            cache.reset();
            EXPECT_EQ(authority->claimedBytes(device,PhysicalMemoryOwner::KVCache,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(authority->claimedBytes(device,PhysicalMemoryOwner::RecurrentLiveState,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
    }).get();
}
/** @brief Resolve real model declarations and prove native early initialization for one backend. */
void proveAdmission(DeviceId device,bool proveArchive=false)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested GPU backend unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        Streams streams(*backend,device);
        MockMPIContext mpi;
        TensorFactory factory(mpi);
        GraphConfig config;
        config.default_device=device;config.n_layers=4;config.max_seq_len=8;
        config.max_request_count=1;config.max_activation_rows=1;
        config.local_n_heads=24;config.local_n_kv_heads=2;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
        admitQwen4ExpTestTable(config);
        Qwen4ExpGraph graph(config);
        const auto schema=graph.getSchema();
        const auto requirements=BufferAllocator::resolveLayerBuffers(schema,graph.getResolverConfig(1));
        BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
        constexpr std::array owners{BufferId::QFN_PLE_TOKEN_HISTORY,BufferId::QFN_PLE_CONV_HISTORY,
            BufferId::QFN_QSA_POOLED_KEYS,BufferId::QFN_QSA_PARTIAL_KEYS};
        constexpr std::array protocols{BufferId::QFN_NGRAM_LOOKUP_STATE,BufferId::QFN_QSA_REPLAY};
        for(const auto role:{owners[0],owners[1],owners[2],owners[3],protocols[0],protocols[1]}) {
            const auto declarations=qwen4ExpBufferDeclarations();
            const auto declaration=std::find_if(declarations.begin(),declarations.end(),
                [role](const auto &entry){return entry.id==role;});
            ASSERT_NE(declaration,declarations.end());
            EXPECT_EQ(declaration->spec.semantic,BufferSemantic::InOut);
            const auto descriptor=std::find_if(requirements.buffers.begin(),requirements.buffers.end(),
                [&](const auto &entry){return entry.name==declaration->spec.name;});
            ASSERT_NE(descriptor,requirements.buffers.end());
            ASSERT_TRUE(arena.registerBuffer(role,*descriptor));
        }
        ASSERT_TRUE(arena.allocate());
        const auto tensor=[&](BufferId role)->TensorBase & {
            auto *value=dynamic_cast<TensorBase *>(arena.getTensor(role));
            require(value,"QFN admission schema did not create a native tensor");return *value;
        };
        for(const auto role:{owners[0],owners[1],owners[2],owners[3],protocols[0],protocols[1]}) {
            EXPECT_NE(tensor(role).raw_data(),nullptr);
            EXPECT_EQ(tensor(role).gpu_data_ptr(),nullptr);
        }
        graph.setArena(&arena);
        llaminar::v2::kernels::KVCacheConfig cacheConfig;
        cacheConfig.precision=ActivationPrecision::FP16;cacheConfig.device=device;
        cacheConfig.num_layers=4;cacheConfig.batch_size=1;cacheConfig.max_seq_len=8;
        cacheConfig.n_kv_heads=2;cacheConfig.head_dim=256;cacheConfig.mpi_ctx=&mpi;
        auto cache=llaminar::v2::kernels::KernelFactory::createKVCache(cacheConfig);
        ASSERT_NE(cache,nullptr);
        // Rejection happens before the first arena device allocation, so a bad
        // initialization stream cannot leave an attached or half-reset owner.
        EXPECT_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr),std::invalid_argument);
        EXPECT_EQ(cache->auxiliaryState(),nullptr);
        for(const auto role:owners)EXPECT_EQ(tensor(role).gpu_data_ptr(),nullptr);
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,ExplicitGPUStream(streams.producer).get()));
        auto *owner=dynamic_cast<Qwen4ExpAuxiliaryState *>(cache->auxiliaryState());
        ASSERT_NE(owner,nullptr);
        ASSERT_EQ(owner->bindings().sparseLayers.size(),1u);
        ASSERT_EQ(owner->bindings().pleHistories.size(),1u);
        std::array<void *,owners.size()> addresses{};
        for(std::size_t index=0;index<owners.size();++index) {
            addresses[index]=tensor(owners[index]).gpu_data_ptr();
            EXPECT_NE(addresses[index],nullptr);
            EXPECT_TRUE(tensor(owners[index]).deviceValid());
        }
        // Ready on a distinct stream must join the initializing producer; no
        // upload of unspecified host bytes is allowed to establish validity.
        ASSERT_NO_THROW(owner->requireReady(ExplicitGPUStream(streams.consumer).get()));
        const auto tokens=streams.read<std::int32_t>(tensor(owners[0]));
        for(const auto value:tokens)EXPECT_EQ(value,config.flash_next->ngrams().hash_reset_token);
        const auto history=streams.read<float>(tensor(owners[1]));
        for(const auto value:history)EXPECT_EQ(value,0.0f);
        const auto partial=streams.read<float>(tensor(owners[3]));
        for(const auto value:partial)EXPECT_EQ(value,0.0f);
        // Replay and lookup protocol have their own ordinary scratch/admission
        // owners. Auxiliary reset neither initializes nor implicitly adopts them.
        for(const auto role:protocols)EXPECT_EQ(tensor(role).gpu_data_ptr(),nullptr);
        std::vector<float> retained(partial.size(),17.25f);
        ASSERT_TRUE(backend->hostToDeviceOnStream(tensor(owners[3]).gpu_data_ptr(),retained.data(),
            retained.size()*sizeof(float),device.ordinal,streams.consumer));
        TransferEngine::publishDeviceWrite(&tensor(owners[3]),device,streams.consumer);
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,streams.producer));
        EXPECT_EQ(cache->auxiliaryState(),owner);
        EXPECT_EQ(streams.read<float>(tensor(owners[3])),retained);
        EXPECT_THROW(owner->reset({0,4,0,0},streams.consumer),std::invalid_argument);
        for(std::size_t index=0;index<owners.size();++index)
            EXPECT_EQ(tensor(owners[index]).gpu_data_ptr(),addresses[index]);
        if(proveArchive)proveCapturedArchive(*owner,*backend,worker,streams);
        for(const auto role:protocols) {
            ASSERT_TRUE(arena.allocateDeviceStorage(role,device));
            EXPECT_NE(tensor(role).gpu_data_ptr(),nullptr);
        }
    }).get();
}
} // namespace
#ifdef HAVE_CUDA
/** @test CUDA cache setup materializes only admitted auxiliary owners and publishes exact initialization events. */
TEST(Qwen4ExpAuxiliaryAdmission, EmptyParticipantCUDA) { proveAbsentAdmission(DeviceId::cuda(0)); }
TEST(Qwen4ExpAuxiliaryAdmission, CUDA) { proveAdmission(DeviceId::cuda(0)); }
/** @test CUDA device-hot pool archive copies remain asynchronous and preserve unused capacity. */
TEST(Qwen4ExpAuxiliaryAdmission, CapturedDeviceArchiveCUDA) { proveAdmission(DeviceId::cuda(0),true); }
#endif
#ifdef HAVE_ROCM
/** @test HIP cache setup materializes only admitted auxiliary owners and publishes exact initialization events. */
TEST(Qwen4ExpAuxiliaryAdmission, EmptyParticipantROCm) { proveAbsentAdmission(DeviceId::rocm(0)); }
TEST(Qwen4ExpAuxiliaryAdmission, ROCm) { proveAdmission(DeviceId::rocm(0)); }
/** @test HIP device-hot pool archive copies remain asynchronous and preserve unused capacity. */
TEST(Qwen4ExpAuxiliaryAdmission, CapturedDeviceArchiveROCm) { proveAdmission(DeviceId::rocm(0),true); }
#endif
} // namespace llaminar2::test
