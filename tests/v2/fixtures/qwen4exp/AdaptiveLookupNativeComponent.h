/**
 * @file AdaptiveLookupNativeComponent.h
 * @brief Actual admitted protocol2 adaptive component shared by native tests and timing.
 *
 * This harness owns a small original-native source, the real CPU subordinate
 * provider/service, one PMA and the canonical GPU arena controller. It captures
 * the production component bundle directly; it does not impersonate a model,
 * selection policy, graph stage, source service or generation loop. Host I/O and
 * synchronization below are diagnostic setup/terminal boundaries only.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include "kernels/qwen4exp/AdaptiveLookupPreparedStorage.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheGeometry.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheOps.h"
#include "loaders/HostImmutableRowProvider.h"
#include "memory/BufferArena.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "mocks/MockMPIContext.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace llaminar2::test::adaptive_native {
using namespace qwen4exp;
namespace policy=::llaminar2::qwen4exp::adaptive_lookup;
/** @brief Fail infrastructure without continuing into an invalid native address.
 * @param ok Required infrastructure invariant. @param why Precise diagnostic. */
inline void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
/** @return Checked extent before file/buffer allocation.
 * @param a First nonnegative factor. @param b Second nonnegative factor.
 * @throws std::overflow_error Before any allocation for unrepresentable extents. */
inline std::size_t product(std::size_t a,std::size_t b){if(b&&a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("Adaptive component fixture extent");return a*b;}
/** @brief Own only the synthetic original-native file, never a downloaded/model payload. */
class Source final {
public:
    /** @brief Persist an exact complete original-native table without conversion.
     * @param bytes Original row-major codebook bytes authored by the caller.
     * @param rows Exact complete original source row count.
     * @param columns Original logical width.
     * @param type Exact GGUF source discriminator.
     */
    Source(std::span<const std::byte> bytes,std::size_t rows,int columns,GGUFTensorType type)
        :native(bytes.begin(),bytes.end()),rows(rows),columns(columns),type(type){
        char pattern[]="/tmp/llaminar-adaptive-native-XXXXXX";fd=::mkstemp(pattern);require(fd>=0,"Adaptive fixture source create failed");path=pattern;
        try{std::size_t done=0;while(done<bytes.size()){const auto n=::write(fd,bytes.data()+done,bytes.size()-done);require(n>0,"Adaptive fixture source write failed");done+=static_cast<std::size_t>(n);}require(descriptor().tensorBytes()==native.size(),"Adaptive fixture source byte extent differs");}
        catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief Keep file alive through every provider/source-read lease. */
    ~Source(){::close(fd);::unlink(path.c_str());}
    Source(const Source&)=delete;Source&operator=(const Source&)=delete;
    /** @return Original exact source descriptor, used by the real loaders. */
    ImmutableRowTableSource descriptor()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=0,.rows=rows,.row_elements=static_cast<std::uint64_t>(columns),.format=type};}
    /** @brief Deterministic source-failure injection only after native/component preparation. */
    void removePayload(){require(::ftruncate(fd,0)==0,"Adaptive source failure injection failed");}
    std::vector<std::byte> native;std::size_t rows;int columns;GGUFTensorType type;int fd=-1;std::string path;
};
/** @return Exact native IQ4_NL bytes; no floating-weight requantization.
 * @param rows Original source row count. @param columns Exact complete-block logical width. */
inline std::vector<std::byte> iq4Source(std::size_t rows,int columns){
    require(columns>0&&columns%32==0,"IQ4_NL component width must contain complete original blocks");
    const auto n=product(rows,static_cast<std::size_t>(columns)/32);std::vector<IQ4_NLBlock> blocks(n);
    for(std::size_t i=0;i<n;++i){blocks[i].d=0x3c00;for(unsigned j=0;j<16;++j)blocks[i].qs[j]=static_cast<std::uint8_t>((i+j)%16|(((i*3+j)%16)<<4));}
    std::vector<std::byte> result(product(n,sizeof(IQ4_NLBlock)));std::memcpy(result.data(),blocks.data(),result.size());return result;
}
/** @brief Exact executable family materialized by the actual fixture caller. */
enum class AdaptiveFixtureGraphFamily
{
    None,
    LookupAndTerminal,
    LookupTerminalAndWarm
};
/**
 * @brief Resolve the caller's complete immutable executable family before admission.
 * @param family Actual graphs instantiated by this caller, including stage-owned graphs.
 * @return Exact executable count; a never-instantiated rejection capture is not an executable.
 * @throws std::invalid_argument If the family is unknown.
 */
inline std::size_t fixtureGraphCount(AdaptiveFixtureGraphFamily family)
{
    switch (family)
    {
    case AdaptiveFixtureGraphFamily::None:
        return 0;
    case AdaptiveFixtureGraphFamily::LookupAndTerminal:
        return 2;
    case AdaptiveFixtureGraphFamily::LookupTerminalAndWarm:
        return 3;
    }
    throw std::invalid_argument("Unknown adaptive fixture executable family");
}
/** @brief Original device/query/source/cache lifetime for one captured component instance. */
class Component final {
public:
    /**
     * @brief Admit and prepare actual component/service ownership before capture.
     * @param source Original native file retained outside this component's lifetime.
     * @param device Exact CUDA/HIP endpoint selected by the explicit lane.
     * @param worker Existing canonical native worker.
     * @param rows Positive physical token capacity, never communication extent.
     * @param slots Exact explicit GPU optional-cache slots (including zero).
     * @param graphs Exact immutable family owned by this caller (including external stage captures).
     * @param hostSlots Explicit CPU subordinate cache slots.
     * @param observations Optional frozen actual GPU/CPU startup observations for timing; pure fixtures use explicit synthetic capacity.
     */
    Component(Source &source,DeviceId device,IWorkerGPUContext &worker,int rows,std::uint32_t slots,AdaptiveFixtureGraphFamily graphs,std::uint32_t hostSlots=4,std::optional<std::array<PhysicalMemoryResource,2>> observations=std::nullopt)
        :source(source),device(device),worker(worker),backend(getBackendFor(device)),stream(ExplicitGPUStream(worker.defaultStream()).get()),
         rows(rows),maximumIds(product(static_cast<std::size_t>(rows),16)),factory(mpi),arena(ArenaConfig{.factory=&factory}){
        require(rows>0&&backend,"Adaptive component requires actual positive native endpoint");
        struct ConstructionRetirement {IBackend *backend;void *stream;DeviceId device;~ConstructionRetirement()noexcept{if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();}} construction{backend,stream,device};
        // Native capture tracks residency but does not acquire this model/test
        // executable owner. Price the actual caller family once; the retained
        // lease below outlives both local and externally stage-owned captures.
        const auto graphBytes = product(fixtureGraphCount(graphs),
                                        GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        const auto descriptor=source.descriptor();const PhysicalMemoryAllocatorIdentity gpu{0,device},host{0,DeviceId::cpu()};
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(descriptor.format,descriptor.rows,descriptor.rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{gpu,ImmutableRowStorage::VRAM},0,slots,{}},{{host,ImmutableRowStorage::RAM},0,hostSlots,{}}},{gpu},{},
            {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},
            slots?std::optional<PhysicalMemoryAllocatorIdentity>(gpu):std::nullopt,maximumIds);
        geometry.emplace(AdaptiveImmutableRowCacheGeometry::fromLayout(layout,gpu,descriptor,rows,{4096,3,4}));
        const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,host,gpu);
        const auto observed=[&](DeviceId id){if(observations){const auto &r=(*observations)[id.is_gpu()?0:1];require(r.world_rank==0&&r.device==id,"Adaptive observed resource differs from its canonical allocator");return r;}return PhysicalMemoryResource{0,id,std::size_t{1}<<34,std::size_t{1}<<34};};
        PhysicalMemoryPlanBuilder plan;
        for(const auto &p:{cpu.bom(observed(DeviceId::cpu())),geometry->cacheBOM(observed(device)),geometry->mandatoryBOM(observed(device))})for(const auto &b:p.resources())plan.add(b);
        plan.add(observed(DeviceId::cpu()),PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(geometry->channelLayout()));
        activationBytes=(sampling_math::kDeviceGenerationControlCount+1)*sizeof(int)+maximumIds*sizeof(std::int64_t)+product(maximumIds,source.columns)*sizeof(float);
        plan.add(observed(device),PhysicalMemoryOwner::ActivationArena,activationBytes);
        if (graphBytes)
            plan.add(observed(device), PhysicalMemoryOwner::NativeGraphExecutable, graphBytes);
        const auto hostRoots=activationBytes+geometry->mandatoryDeviceBytes()+geometry->cache().bytes;
        plan.add(observed(DeviceId::cpu()),PhysicalMemoryOwner::ActivationArena,hostRoots);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        // forSubordinateGPU seals the exact requester; the independent host
        // provider factory deliberately rejects that protocol2 source owner.
        provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(descriptor,cpu,memory));
        service=ImmutableLookupService::create(provider,geometry->channelLayout(),std::array{device},memory,DeviceId::cpu());
        sourceBinding.emplace(service->bindSubordinate(device));
        activationLease.emplace(memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,activationBytes));
        if (graphBytes)
        {
            graphLease.emplace(memory->claimNewAllocation(device, PhysicalMemoryOwner::NativeGraphExecutable,
                                                           graphBytes));
            require(graphLease->bytes() == graphBytes, "Adaptive fixture executable lease differs from its family");
        }
        hostRootLease.emplace(memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,hostRoots));
        frameLease.emplace(memory->claimNewAllocation(device,PhysicalMemoryOwner::ImmutableLookupStaging,geometry->mandatoryDeviceBytes()));
        if(slots)cacheLease.emplace(memory->claimNewAllocation(device,PhysicalMemoryOwner::ImmutableTableStorage,geometry->cache().bytes));
        require(arena.registerBuffer(BufferId::MTP_GENERATION_CONTROL,1,sampling_math::kDeviceGenerationControlCount,"INT32",device),"Adaptive canonical controller declaration failed");
        require(arena.registerBuffer(BufferId::PREFILL_CHUNK_GEOMETRY,1,1,"INT32",device),"Adaptive original row declaration failed");
        require(arena.registerBuffer(BufferId::QFN_NGRAM_ROW_IDS,rows,32,"INT32",device),"Adaptive ID declaration failed");
        require(arena.registerBuffer(BufferId::QFN_PLE_EMBEDDING,rows,16*source.columns,"FP32",device),"Adaptive output declaration failed");require(arena.allocate(),"Adaptive arena allocation failed");
        frame=std::make_shared<INT8Tensor>(std::vector<std::size_t>{geometry->frame().bytes});
        protocol=std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
        if(slots)cache=std::make_shared<INT8Tensor>(std::vector<std::size_t>{geometry->cache().bytes});
        for (auto id : {BufferId::MTP_GENERATION_CONTROL, BufferId::PREFILL_CHUNK_GEOMETRY,
                        BufferId::QFN_NGRAM_ROW_IDS, BufferId::QFN_PLE_EMBEDDING})
        {
            // Borrow the arena's actual typed root; ITensor alone does not
            // expose the canonical TensorBase preparation contract.
            auto root = arena.getSharedTensor(id);
            require(bool(root), "Adaptive fixture canonical arena root is missing");
            prepare(*root);
        }
        prepare(*frame);prepare(*protocol);if(cache)prepare(*cache);
        control.emplace(DeviceGenerationControlBinding::admit(arena,device,0));
        query.emplace(AdaptiveLookupQuery{DeviceRowRange::deviceCounted(rows,live()),ids(),maximumIds,output(),product(maximumIds,source.columns)});
        controlIdentity=control->captureIdentity();rowIdentity=query->token_rows.captureIdentity();
    }
    /**
     * @brief Retire every native reader and the real source worker before any borrowed owner.
     * @details The failure path fences again after a possibly partial close submission.
     * A successful fence precedes worker cancellation and storage retirement;
     * an unknowable native lifetime terminates instead of releasing live roots.
     */
    ~Component() noexcept
    {
        // The worker must keep serving until all native response readers retire.
        if (backend && !backend->synchronizeStream(stream, device.ordinal))
            std::terminate();
        graph.reset();
        terminal.reset();
        if (armed)
        {
            try
            {
                kernels->close(*storage, stream);
                if (!backend->synchronizeStream(stream, device.ordinal))
                    std::terminate();
                (void)service->await(*ticket);
            }
            catch (...)
            {
                // closeNative may enqueue before its checked launch status
                // throws. The earlier fence cannot retire that later work.
                // Fence while the real worker, mapped ACK and protocol owner
                // are still live, then cancel/await the CPU service.
                if (!backend->synchronizeStream(stream, device.ordinal))
                    std::terminate();
                (void)service->cancel(*ticket);
                try
                {
                    (void)service->await(*ticket);
                }
                catch (...)
                {
                    std::terminate();
                }
            }
            armed = false;
        }
        kernels.reset();
        sourceBinding.reset();
        storage.reset();
        service.reset();
        provider.reset();
    }
    Component(const Component&)=delete;Component&operator=(const Component&)=delete;
    /** @brief Declare this source's one exact original storage owner; no native work occurs. */
    void declareStorage(){storage=sourceBinding->declareStorage(geometry->nativePlan(),cache,frame,protocol);}
    /** @return Actual production factory result; callers must first declare and seal the source-owned roots. */
    std::unique_ptr<IImmutableAdaptiveLookupKernels> createBundle(){
#ifdef HAVE_CUDA
        if(device.is_cuda())return createCUDAImmutableAdaptiveLookupKernels(geometry->nativePlan(),*sourceBinding);
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())return createROCmImmutableAdaptiveLookupKernels(geometry->nativePlan(),*sourceBinding);
#endif
        throw std::logic_error("Adaptive exact backend factory unavailable");
    }
    /** @brief Complete the ordinary source-issued declaration/seal/factory sequence before initialization. */
    void prepareBundle(){declareStorage();storage->seal();kernels=createBundle();require(bool(kernels),"Adaptive native bundle unavailable");identity=kernels->captureIdentity(*storage);}
    /** @brief Submit actual once-only metadata initialization before admission or capture. */
    void initializeMetadata(){kernels->initialize(*storage,stream);join();}
    /** @brief Ordinary setup materialization/publication; never called inside capture.
     * @param tensor Original canonical root to initialize and prepare. */
    void prepare(TensorBase &tensor){std::memset(tensor.raw_mutable_data(),0,tensor.size_bytes());TransferEngine::prepareDeviceInput(&tensor,device,stream);TransferEngine::requireDeviceInput(&tensor,device,stream);}
    /** @brief Join only a fixture setup/terminal observation, never a production-stage dependency. */
    void join(){require(backend->synchronizeStream(stream,device.ordinal),"Adaptive native fixture terminal join failed");}
    /** @brief Fence possible partial diagnostic submissions before borrowed host backing unwinds. */
    struct HostCopyRetirement {Component &c;bool complete=false;~HostCopyRetirement()noexcept{if(!complete&&!c.backend->synchronizeStream(c.stream,c.device.ordinal))std::terminate();}};
    /** @brief Publish test admission data on its exact stream, outside native replay.
     * @tparam T Original scalar/record type. @param ptr Exact persistent destination.
     * @param values Borrowed host span; the guard retires partial submissions before it unwinds. */
    template<class T>void upload(T *ptr,std::span<const T> values){HostCopyRetirement retirement{*this};require(backend->hostToDeviceOnStream(ptr,values.data(),values.size_bytes(),device.ordinal,stream),"Adaptive fixture upload failed");join();retirement.complete=true;}
    /** @brief Observe bounded terminal native records only after actual submission completion.
     * @tparam T Exact scalar/record type. @param ptr Original source address.
     * @param count Exact observed element extent. @return Owned terminal values, empty without a copy for count0. */
    template<class T>std::vector<T> read(const T *ptr,std::size_t count){if(!count)return {};std::vector<T> v(count);HostCopyRetirement retirement{*this};require(backend->deviceToHostOnStream(v.data(),ptr,product(count,sizeof(T)),device.ordinal,stream),"Adaptive terminal copy failed");join();retirement.complete=true;return v;}
    /** @brief Reject a failed source before native admission, then start a newer actual service interval.
     * Healthy reset preserves warmed optional state and the original protocol sequence. */
    void admit(){require(!armed,"Adaptive fixture interval still armed");service->requireHealthy();++generation;kernels->admit(*storage,generation,generation==1?ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce:ImmutableLookupAdmissionArgs::ProtocolInitialization::Preserve,stream);join();ticket.emplace(service->arm(generation));armed=true;}
    /** @brief Capture the complete actual component and conditional terminal as separate retained graphs. */
    void capture(){require(armed,"Adaptive capture requires initialized admission");graph=worker.createGraphCapture(stream);terminal=worker.createGraphCapture(stream);require(graph&&terminal,"Adaptive graph owner unavailable");{ScopedBackendGraphCapture scope(worker,*graph,"adaptive native component");require(scope.begin(),"Adaptive graph begin failed");kernels->gather(*storage,*query,*control,stream);kernels->closeWhenTerminal(*storage,*control,stream);scope.finish();}{ScopedBackendGraphCapture scope(worker,*terminal,"adaptive native terminal");require(scope.begin(),"Adaptive terminal begin failed");kernels->closeWhenTerminal(*storage,*control,stream);scope.finish();}require(graph->instantiate()&&terminal->instantiate(),"Adaptive graph instantiation failed");}
    /** @brief Retain a diagnostic workload outside hot replay; original count remains the only logical authority.
     * @param active Original logical live token rows. @param liveIds Exact active*16 signed original IDs. */
    void workload(int active,std::span<const std::int64_t> liveIds){require(active>=0&&active<=rows&&liveIds.size()==static_cast<std::size_t>(active)*16,"Adaptive workload rows/IDs disagree");std::vector<std::int64_t> all(maximumIds,-1);std::copy(liveIds.begin(),liveIds.end(),all.begin());upload(ids(),std::span<const std::int64_t>(all));upload(live(),std::span<const int>(&active,1));std::vector<float> poison(product(maximumIds,source.columns),-12345.f);upload(output(),std::span<const float>(poison));std::vector<std::byte> nativePoison(product(maximumIds,geometry->rowBytes()),std::byte{0xa5});upload(static_cast<std::byte*>(frame->gpu_data_ptr())+geometry->frame().native_response,std::span<const std::byte>(nativePoison));std::vector<int> words(sampling_math::kDeviceGenerationControlCount);words[sampling_math::kDeviceGenerationControlOk]=1;upload(control->row(),std::span<const int>(words));}
    /** @brief Submit retained transaction; no service re-arm or count readback occurs here. */
    void launch(){require(graph&&graph->launch(),"Adaptive retained native component launch failed");}
    /** @brief Close the current actual service interval only through canonical controller completion.
     * @return Actual source worker outcome after native terminal publication and worker retirement. */
    bool close(){const int done=1;upload(control->row()+sampling_math::kDeviceGenerationControlRequestComplete,std::span<const int>(&done,1));require(terminal&&terminal->launch(),"Adaptive terminal graph launch failed");join();const bool ok=service->await(*ticket);armed=false;ticket.reset();return ok;}
    /** @return Exact canonical original live row owner. */
    int *live()const{return static_cast<int*>(arena.getTensor(BufferId::PREFILL_CHUNK_GEOMETRY)->gpu_data_ptr());}
    /** @return Exact original ID source; signed64 rows are represented by native INT32 storage. */
    std::int64_t *ids()const{return static_cast<std::int64_t*>(arena.getTensor(BufferId::QFN_NGRAM_ROW_IDS)->gpu_data_ptr());}
    /** @return Direct native output; no compact-response scratch replaces this result. */
    float *output()const{return static_cast<float*>(arena.getTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr());}
    /** @return Original immutable source-native protocol root for terminal evidence only. */
    ImmutableLookupDeviceState protocolState(){return read(static_cast<ImmutableLookupDeviceState*>(protocol->gpu_data_ptr()),1).front();}
    /** @return Actual last-frame common metadata after its pins retired. */
    policy::FrameControl frameState(){return read(reinterpret_cast<const policy::FrameControl*>(static_cast<const std::byte*>(frame->gpu_data_ptr())+geometry->frame().control),1).front();}

    Source &source;DeviceId device;IWorkerGPUContext &worker;IBackend *backend;void *stream;int rows;std::size_t maximumIds,activationBytes=0;
    std::shared_ptr<PhysicalMemoryAuthority> memory;std::optional<PhysicalMemoryAllocationLease> activationLease,frameLease,cacheLease,hostRootLease,graphLease;
    std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<ImmutableLookupService> service;std::optional<ImmutableSubordinateLookupBinding> sourceBinding;
    std::optional<AdaptiveImmutableRowCacheGeometry> geometry;
    MockMPIContext mpi;TensorFactory factory;BufferArena arena;
    std::shared_ptr<INT8Tensor> cache,frame;std::shared_ptr<INT32Tensor> protocol;
    std::optional<DeviceGenerationControlBinding> control;std::optional<AdaptiveLookupQuery> query;
    std::shared_ptr<AdaptiveLookupPreparedStorage> storage;std::unique_ptr<IImmutableAdaptiveLookupKernels> kernels;
    std::unique_ptr<IGPUGraphCapture> graph,terminal;std::optional<ImmutableLookupService::Ticket> ticket;bool armed=false;
    std::uint64_t generation=0,identity=0,controlIdentity=0;DeviceRowRange::CaptureIdentity rowIdentity;
};
} // namespace llaminar2::test::adaptive_native
