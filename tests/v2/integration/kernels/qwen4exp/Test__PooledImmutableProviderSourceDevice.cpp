/**
 * @file Test__PooledImmutableProviderSourceDevice.cpp
 * @brief Retained real-provider pooled graph for the full loadable native-format inventory and sparse high-row IQ4_NL.
 *
 * Original canonical-format rows and sparse IQ4_NL high rows use the real subordinate service. The
 * production graph builder records every owner/coordinator/source operation;
 * no diagnostic delivery producer substitutes bytes or ACKs. Input uploads and
 * terminal readbacks below are test boundaries, never inference fallbacks.
 * Cold-declaration replay leaves original ID/output storage unprepared through
 * production graph creation and identity/contract queries; the actual executor
 * then materializes storage before capture-only seals retain its allocations.
 * Scope is one explicit local edge; remote/cross-vendor transport and serving
 * remain separately unqualified.
 */
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "PooledACKReadyCancellation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/NativeTensorExtent.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <optional>
#include <map>
#include <type_traits>
#include <unistd.h>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
/** @brief Fail the exact requested native cell without a skip or alternate path. */
void requireProvider(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
/** @brief Author original rows at their full source coordinates; sparse holes are file layout, never transmitted padding. */
class ProviderFile final {
public:
    static constexpr std::uint64_t High=0x100000007ULL,Rows=0x100000100ULL;
    static constexpr std::size_t Columns=160,RowBytes=90,Offset=37;
    /** @brief Keep the original sparse IQ4_NL high-row diagnostic geometry. */
    ProviderFile(){
        open();
        try{
            requireProvider(::ftruncate(fd,static_cast<off_t>(Offset+rows*row_bytes))==0,"Cannot declare exact sparse provider extent");
            for(auto row:std::array<std::uint64_t,7>{7,101,102,700,701,900,High})author(row,legacy(row));
            for(std::uint64_t row=2000;row<2016;++row)author(row,legacy(row));
        }catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief Author canonical original-native fixture tensor bytes, never a conversion in the production path.
     * @param native Canonical test inventory's original native tensor. @param sourceType Actual loader GGUF discriminator.
     * @param width Original complete native row width. */
    ProviderFile(const TensorBase& native,GGUFTensorType sourceType,std::size_t width)
        :rows(97),columns(width),format(sourceType),type(native.native_type()){
        row_bytes=descriptor().rowBytes();requireProvider(native.size_bytes()==rows*row_bytes,"Canonical source row geometry differs from original native tensor");open();
        try{
            requireProvider(::ftruncate(fd,static_cast<off_t>(Offset+rows*row_bytes))==0,"Cannot declare canonical provider extent");
            const auto* bytes=static_cast<const std::byte*>(native.raw_data());
            for(std::uint64_t row=0;row<rows;++row)author(row,{bytes+row*row_bytes,bytes+(row+1)*row_bytes});
        }catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief The actual source owner is gone before its original file unlinks. */
    ~ProviderFile(){::close(fd);::unlink(path.c_str());}
    /** @return Exact file descriptor consumed by the existing production provider. */
    ImmutableRowTableSource descriptor()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=Offset,.rows=rows,.row_elements=columns,.format=format};}
    /** @return Authored immutable source bytes for the terminal independent CPU decoder oracle. */
    std::span<const std::byte> original(std::uint64_t row)const{return authored.at(row);}
    std::uint64_t rows=Rows;std::size_t columns=Columns,row_bytes=RowBytes;GGUFTensorType format=GGUFTensorType::IQ4_NL;TensorType type=TensorType::IQ4_NL;
    int fd=-1;std::string path;
private:
    /** @brief Create only fixture source I/O, never an alternate delivery publisher. */
    void open(){char name[]="/tmp/qfn-real-provider-XXXXXX";fd=::mkstemp(name);requireProvider(fd>=0,"Cannot create original provider file");path=name;}
    /** @brief Write one complete original native row at its real declared source coordinate. */
    void author(std::uint64_t row,std::vector<std::byte> bytes){
        requireProvider(row<rows&&bytes.size()==row_bytes,"Invalid authored original source row");
        requireProvider(::pwrite(fd,bytes.data(),bytes.size(),static_cast<off_t>(Offset+row*row_bytes))==static_cast<ssize_t>(bytes.size()),"Cannot author original provider bytes");authored.emplace(row,std::move(bytes));
    }
    /** @return Original sparse IQ4_NL source pattern, preserving the prior high-row test oracle. */
    static std::vector<std::byte> legacy(std::uint64_t row){
        std::vector<std::byte> result(RowBytes);
        const auto signature=row^(row>>32); // High source bits must change bytes, even against the same low32 row.
        for(unsigned block=0;block<5;++block){result[block*18]=std::byte{0};result[block*18+1]=std::byte{0x3c};
            for(unsigned code=0;code<16;++code)result[block*18+2+code]=std::byte((signature+block+code)%16|(((signature*3+block+code)%16)<<4));}
        return result;
    }
    std::map<std::uint64_t,std::vector<std::byte>> authored;
};
/** @brief Explicit recorded setup fault, never a production selection or retry policy. */
enum class OwnerAdmission {Original,RejectForeignAuthority};
/** @brief Immutable diagnostic capture edit; never a runtime serving selector. */
enum class ACKInterleaving {None,Cancel900BeforeReady,OriginalUnchecked900};
/** @brief Recorded activation lifecycle; both modes execute the same production stages and native code. */
enum class ActivationPreparation {BeforeGraph,ColdGraph};
/** @brief One actual admitted service and retained fixed graph; no mutable device state is mirrored on the host. */
class ProviderReplay final {
public:
    ProviderReplay(ProviderFile& file,DeviceId device,IWorkerGPUContext& worker,std::uint32_t maximum,std::uint32_t slots,
                   std::uint32_t hostSlots,n::OriginalIdWidth width,bool tokenQueries=false,OwnerAdmission ownership=OwnerAdmission::Original,ACKInterleaving interleaving=ACKInterleaving::None,ActivationPreparation activationPreparation=ActivationPreparation::BeforeGraph)
        :file(file),device(device),worker(worker),backend(getBackendFor(device)),stream(ExplicitGPUStream(worker.defaultStream()).get()),
         maximum(maximum),slots(slots),width(width),tokenQueries(tokenQueries),interleaving(interleaving),activationPreparation(activationPreparation){
        requireProvider(backend && maximum>=16,"Real provider fixture requires its exact native participant");
        requireProvider(!tokenQueries || (!(maximum%16) && width==n::OriginalIdWidth::UInt64),
            "Production PLE replay requires the complete original INT64 sixteen-head token envelope");
        struct Construction {ProviderReplay& self;~Construction(){self.join();}} construction{*this};
        const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};const auto source=file.descriptor();
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{gpuId,ImmutableRowStorage::VRAM},0,slots,{}},{{hostId,ImmutableRowStorage::RAM},0,hostSlots,{}}},{gpuId},{},
            {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},slots?std::optional(gpuId):std::nullopt,16);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},maximum);
        const auto cpuGeometry=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
        const PhysicalMemoryResource gpu{0,device,1ull<<34,1ull<<34},host{0,DeviceId::cpu(),1ull<<34,1ull<<34};
        const auto channel=q::ImmutableLookupLayout::createSubordinate(16,source.rowBytes());
        PhysicalMemoryPlanBuilder plan;
        for(const auto& bom:{cpuGeometry.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
            q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
            q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,0,source,gpu)})
            for(const auto& resource:bom.resources())plan.add(resource);
        plan.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const std::array declarations{
            CPUTensorStorageDeclaration(1,std::size_t{maximum}*(width==n::OriginalIdWidth::UInt32?1:2),BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(tokenQueries?maximum/16:1,tokenQueries?file.columns*16:std::size_t{maximum}*file.columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        const auto activation=std::size_t{maximum}*(width==n::OriginalIdWidth::UInt32?4:8)+sizeof(int)+sampling_math::kDeviceGenerationControlCount*sizeof(int)+std::size_t{maximum}*file.columns*sizeof(float);
        plan.add(gpu,PhysicalMemoryOwner::ActivationArena,activation);
        for(const auto& declaration:declarations)plan.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
        const auto executableBytes=(tokenQueries?4:3)*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General);
        plan.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,executableBytes);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        activationLease=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,activation);
        executableLease=memory->claimNewAllocation(device,PhysicalMemoryOwner::NativeGraphExecutable,executableBytes);
        provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpuGeometry,memory));
        service=q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu());
        const auto sourceBinding=service->bindSubordinate(device);
        auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
        auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
        const auto sourceGeometry=q::PooledImmutableSourceGeometry::fromPlans(ownerPlan,coordinatorPlan,source);
        if(ownership==OwnerAdmission::RejectForeignAuthority){
            const auto foreign=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
            auto wrong=q::PooledImmutableOwnerPreparedStorage::create(ownerPlan,foreign,stream);
            auto actualCoordinator=q::PooledImmutableCoordinatorPreparedStorage::create(coordinatorPlan,memory,stream);
            const auto before=memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation);
            bool rejected=false;
            try{(void)q::ImmutableLookupExecutionBoundary::createPooled(service,sourceGeometry,wrong,actualCoordinator,memory,stream);}
            catch(const std::logic_error& error){rejected=std::string(error.what()).find("original PMA")!=std::string::npos;}
            requireProvider(rejected,"A foreign equal-sized owner PMA must reject before source roots or registration publish");
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),before);
            // No partial source registration survived. These negative-control
            // roots retire before the actual original declarations below.
        }
        owner=q::PooledImmutableOwnerPreparedStorage::create(std::move(ownerPlan),memory,stream);
        coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(std::move(coordinatorPlan),memory,stream);
#ifdef HAVE_CUDA
        if(device.is_cuda())code=q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())code=q::createROCmPooledImmutableNativeKernels();
#endif
        requireProvider(bool(code),"Real provider native phase bundle unavailable");code->initialize(*owner,stream);code->initialize(*coordinator,stream);
        boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourceGeometry,owner,coordinator,memory,stream);
        arena=std::make_unique<BufferArena>();
        const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};
        for(std::size_t index=0;index<declarations.size();++index){
            std::shared_ptr<TensorBase> root(declarations[index].materialize(memory));
            if(index==3)std::fill_n(static_cast<float*>(root->raw_mutable_data()),std::size_t{maximum}*file.columns,-12345.f);
            if(activationPreparation==ActivationPreparation::BeforeGraph || (index!=0 && index!=3)){
                TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);
            }
            requireProvider(arena->registerExternalBuffer(roles[index],root),"Real provider original activation registration failed");
        }
        bindPooledNativeRoots(*arena,*owner);bindPooledNativeRoots(*arena,*coordinator);bindPooledSourceRoots(*arena,*boundary);
        control.emplace(DeviceGenerationControlBinding::admit(*arena,device,0));boundary->bindPooledGenerationControl(*control);
        expected.assign(std::size_t{maximum}*file.columns,-12345.f);
        // Poison only setup payload banks. No production copy may clear padding
        // or inactive rows merely because those bytes were admitted.
        for(const auto& root:{owner->roots().cache,boundary->pooledStorage()->roots().frame})if(root){
            const auto offset=root==owner->roots().cache?owner->plan().parameters().cache.payload:0;
            std::vector<std::byte> poison(root->size_bytes()-offset,std::byte{0xa5});
            requireProvider(backend->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+offset,poison.data(),poison.size(),device.ordinal,stream),"Real provider payload poison failed");join();
            TransferEngine::publishDeviceWrite(root.get(),device,stream);
        }
        context=IDeviceContext::create(device,1);build();if(tokenQueries)buildSmallProductionGraph();identity=boundary->captureIdentity();
    }
    /** @brief Retire actual submissions and worker before any graph/source/backing owner releases. */
    ~ProviderReplay(){
        join();
        if(boundary && boundary->hasActiveInterval()){
            try{boundary->recordTerminalClose(stream);join();boundary->awaitTerminal();}
            catch(...){/* The original failed service remains retained through joined graph teardown. */}
        }
    }
    /** @brief Setup/terminal diagnostic wait only, never called by a production stage. */
    void join()const{if(!backend || !backend->synchronizeStream(stream,device.ordinal))std::terminate();}
    /** @brief Build once through actual reusable stages and capture dependency planning. */
    void build(){
        q::PooledNativeOperandSpans begin{};const auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        begin[static_cast<unsigned>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,std::size_t{maximum}*(width==n::OriginalIdWidth::UInt32?4:8)};
        begin[static_cast<unsigned>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,4};
        begin[static_cast<unsigned>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,
            offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),8};
        const auto output=arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto tokens=DeviceRowRange::deviceCounted(tokenQueries?maximum/16:maximum,
            static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        const auto original=tokenQueries?tokens.repeatEachRow(16):tokens;
        overboundInputs.emplace(n::CoordinatorAction::Begin,coordinator->plan(),n::OriginalRows{original,width},begin);
        if(tokenQueries)
            appendPooledImmutablePLELookupGraph(compute,*arena,boundary,code,tokens,
                {ids,BufferId::QFN_NGRAM_ROW_IDS,0,std::size_t{maximum}*sizeof(std::uint64_t)},
                q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(std::int32_t)},*control,
                {output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"production_ple","ple_gather");
        else appendPooledImmutableLocalLookupGraph(compute,*arena,boundary,code,{original,width},std::move(begin),*control,
            {output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"real_provider");
        executor.setArena(arena.get());
        std::vector<std::uint64_t> declarations;
        for(const auto& name:compute.getExecutionOrder()){
            const auto& stage=*compute.getNode(name)->stage;
            declarations.push_back(stage.graphCaptureVariantSignature());
            (void)stage.bufferContract();
        }
        if(activationPreparation==ActivationPreparation::ColdGraph){
            // The old production constructor seal fails here before the graph
            // exists. Exact original declarations must survive without native addresses.
            EXPECT_EQ(ids->gpu_data_ptr(),nullptr);EXPECT_EQ(output->gpu_data_ptr(),nullptr);
            auto* begin=compute.getNode(compute.getExecutionOrder().front())->stage.get();
            auto* reader=compute.getNode("ple_gather")->stage.get();
            EXPECT_THROW(begin->execute(context.get()),std::logic_error);
            EXPECT_THROW(reader->execute(context.get()),std::logic_error);
            EXPECT_THROW(begin->prepareGraphLaunch(context.get(),stream),std::logic_error);
            EXPECT_THROW(reader->prepareGraphLaunch(context.get(),stream),std::logic_error);
            EXPECT_EQ(ids->gpu_data_ptr(),nullptr);EXPECT_EQ(output->gpu_data_ptr(),nullptr);
            requireProvider(executor.allocateGraphStorageForCapture(compute,context.get(),"real provider cold declarations"),
                "Real provider cold arena materialization failed");
            EXPECT_NE(ids->gpu_data_ptr(),nullptr);EXPECT_NE(output->gpu_data_ptr(),nullptr);
            // Initial fixture I/O supplies the same poison and input bytes used
            // by warm setup, after allocation and before the captured transaction.
            for(const auto& root:{ids,output}){
                TransferEngine::prepareDeviceInput(root.get(),device,stream);
                TransferEngine::requireDeviceInput(root.get(),device,stream);
            }
        }
        for(const auto& name:compute.getExecutionOrder()){
            auto& stage=*compute.getNode(name)->stage;
            requireProvider(stage.prepareGraphLaunch(context.get(),stream),"Real provider capture-only stage preparation failed");
            requireProvider(stage.prepareGraphLaunch(context.get(),stream),"Real provider repeated capture-only preparation failed");
        }
        for(std::size_t index=0;const auto& name:compute.getExecutionOrder())
            EXPECT_EQ(compute.getNode(name)->stage->graphCaptureVariantSignature(),declarations[index++]);
        requireProvider(executor.prepareGraphStorageForCapture(compute,context.get(),stream,"real provider complete native lookup"),"Real provider graph preparation failed");
        const auto& order=compute.getExecutionOrder();auto dependencies=executor.planGraphCaptureDependencies(compute,order,device,stream,"real provider source/owner/control edges");
        requireProvider(bool(dependencies),"Real provider capture dependency plan missing");graph=worker.createGraphCapture(stream);terminal=worker.createGraphCapture(stream);overbound=worker.createGraphCapture(stream);
        requireProvider(graph && terminal && overbound,"Real provider retained graph creation failed");
        overboundInputs->seal();
        {
            ScopedBackendGraphCapture capture(worker,*graph,"real provider complete local lookup",dependencies.get());requireProvider(capture.begin(),"Real provider capture begin failed");
            requireProvider(executor.recordSetupGraphMaterialization(compute,context.get()),"Real provider actual stage recording failed");capture.finish();
        }
        {
            ScopedBackendGraphCapture capture(worker,*terminal,"real provider complete parent terminal");requireProvider(capture.begin(),"Real provider terminal capture failed");
            boundary->recordTerminalClose(stream,control->healthy(),control->row()+sampling_math::kDeviceGenerationControlRequestComplete);capture.finish();
        }
        {
            ScopedBackendGraphCapture capture(worker,*overbound,"real provider unique-bound rejection");requireProvider(capture.begin(),"Real provider over-bound capture failed");
            code->enqueue(n::CoordinatorAction::Begin,*coordinator,*overboundInputs,stream);capture.finish();
        }
        std::string detail;requireProvider(graph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireProvider(terminal->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireProvider(overbound->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        if(interleaving!=ACKInterleaving::None){
            const auto original=sourceArguments();
            const auto replay=interleaving==ACKInterleaving::OriginalUnchecked900?
                pooled_ack::AuthorizationReplay::OriginalUnchecked:pooled_ack::AuthorizationReplay::Production;
#ifdef HAVE_CUDA
            if(device.is_cuda())pooled_ack::spliceCUDA(*graph,original,stream,replay);
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())pooled_ack::spliceROCm(*graph,original,stream,replay);
#endif
        }
        requireProvider(graph->instantiate() && terminal->instantiate() && overbound->instantiate(),"Real provider retained graph instantiation failed");
        std::vector<GPUGraphKernelNodeInfo> nodes;requireProvider(graph->inspectKernelNodes(nodes,&detail),detail.c_str());
        for(const auto& node:nodes){requireProvider(node.valid(),"Real provider native resource inventory incomplete");
            const auto scope=node.name.find("pooled_source")!=std::string::npos?"source":node.name.find("coordinator")!=std::string::npos?"coordinator":"owner";
            std::cout<<"POOLED_PROVIDER_NODE scope="<<scope<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread
                <<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes
                <<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';
        }
    }
    /** @return Exact original source launch operands for setup-only graph authentication. */
    q::pooled_source::DeviceArgs sourceArguments()const{
        const auto storage=boundary->pooledStorage();const auto own=owner->roots(),coord=coordinator->roots(),root=storage->roots();
        const auto address=[](const std::shared_ptr<TensorBase>& value){return value?static_cast<std::byte*>(value->gpu_data_ptr()):nullptr;};
        const auto& p=storage->geometry().parameters();
        return {p,{p.owner,address(own.cache),address(own.frame),address(own.native),{}},
            {p.coordinator,address(coord.cache),address(coord.frame),address(coord.native),{}},
            address(root.frame),address(root.native),boundary->pooledKernels().source().deviceView(),control->healthy(),control->error(),control->row()};
    }
    /** @brief Capture a genuine smaller PLE variant against the same original source roots and token count. */
    void buildSmallProductionGraph(){
        const auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto output=arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto token=DeviceRowRange::deviceCounted(1,static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        appendPooledImmutablePLELookupGraph(smallCompute,*arena,boundary,code,token,
            {ids,BufferId::QFN_NGRAM_ROW_IDS,0,16*sizeof(std::uint64_t)},
            q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(std::int32_t)},*control,
            {output,BufferId::QFN_PLE_EMBEDDING,0,16*file.columns*sizeof(float)},"production_ple_small","ple_gather_small");
        smallExecutor.setArena(arena.get());
        for(const auto& name:smallCompute.getExecutionOrder())
            requireProvider(smallCompute.getNode(name)->stage->prepareGraphLaunch(context.get(),stream),
                "Small production PLE capture-only stage preparation failed");
        requireProvider(smallExecutor.prepareGraphStorageForCapture(smallCompute,context.get(),stream,"production PLE smaller variant"),"Small production PLE preparation failed");
        const auto& order=smallCompute.getExecutionOrder();
        auto dependencies=smallExecutor.planGraphCaptureDependencies(smallCompute,order,device,stream,"production PLE smaller native frame");
        requireProvider(bool(dependencies),"Small production PLE capture dependency plan missing");
        smallGraph=worker.createGraphCapture(stream);requireProvider(bool(smallGraph),"Small production PLE retained graph missing");
        {ScopedBackendGraphCapture capture(worker,*smallGraph,"production PLE smaller variant",dependencies.get());
            requireProvider(capture.begin(),"Small PLE capture begin failed");
            requireProvider(smallExecutor.recordSetupGraphMaterialization(smallCompute,context.get()),"Small PLE recording failed");capture.finish();}
        std::string error;requireProvider(smallGraph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&error),error.c_str());
        requireProvider(smallGraph->instantiate(),"Small production PLE retained graph instantiation failed");
        const auto* reader=smallCompute.getNode("ple_gather_small");requireProvider(bool(reader),"Small PLE semantic reader absent");
        const auto& dump=reader->stage->getDumpInfo();requireProvider(dump.outputs.size()==1 && dump.outputs.front().rows==1,
            "Small production PLE output span does not preserve actual graph geometry");
    }
    /** @brief Initial input/control publication; native output/cache state is never downloaded and re-uploaded. */
    void input(std::span<const std::uint64_t> original){
        requireProvider(original.size()<=maximum && (!tokenQueries || !(original.size()%16)),"Real provider original input exceeds its exact token/ID bound");auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS);
        std::memset(ids->raw_mutable_data(),0xff,ids->size_bytes());
        if(width==n::OriginalIdWidth::UInt64){if(!original.empty())std::memcpy(ids->raw_mutable_data(),original.data(),original.size_bytes());}
        else for(std::size_t i=0;i<original.size();++i){requireProvider(original[i]<=~std::uint32_t{0},"Real UInt32 producer overflow");static_cast<std::uint32_t*>(ids->raw_mutable_data())[i]=original[i];}
        auto count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);*static_cast<int*>(count->raw_mutable_data())=tokenQueries?original.size()/16:original.size();
        auto controller=arena->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);std::memset(controller->raw_mutable_data(),0,controller->size_bytes());static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=1;
        for(const auto& root:{ids,count,controller}){TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);}join();
    }
    /** @brief Read terminal evidence only; no observation below can select a production branch or repair native state. */
    template<class T>std::vector<T> read(const void* source,std::size_t count){
        std::vector<T> result(count);if(count)requireProvider(backend->deviceToHostOnStream(result.data(),source,count*sizeof(T),device.ordinal,stream),"Real provider terminal read failed");join();return result;
    }
    /** @brief Arm the actual existing boundary once per request while retaining proven captured topology. */
    void admit(std::uint64_t generation){boundary->beginAdmission(generation,stream);join();}
    /** @brief Replay actual provider delivery and compare unchanged native rows, original permutation and final pin retirement. */
    void replay(std::span<const std::uint64_t> original){
        input(original);void* start=backend->createTimingEvent(device.ordinal);void* stop=backend->createTimingEvent(device.ordinal);
        requireProvider(start && stop,"Real provider timing events missing");requireProvider(backend->recordEvent(start,device.ordinal,stream),"Real provider timing start failed");
        requireProvider((tokenQueries && original.size()<=16?smallGraph:graph)->launch(),"Real provider captured replay failed");
        for(const auto& storage:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{storage.cache,storage.frame,storage.native})if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        TransferEngine::publishDeviceWrite(control->owner(),device,stream);TransferEngine::publishDeviceWrite(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get(),device,stream);
        requireProvider(backend->recordEvent(stop,device.ordinal,stream) && backend->waitForEvent(stop,device.ordinal),"Real provider replay did not retire");
        float elapsed=0;requireProvider(backend->eventElapsedTimeMs(start,stop,device.ordinal,&elapsed),"Real provider timing read failed");
        backend->destroyEvent(stop,device.ordinal);backend->destroyEvent(start,device.ordinal);
        std::cout<<"POOLED_PROVIDER_TIMING original="<<original.size()<<" maximum="<<maximum<<" slots="<<slots<<" format="<<static_cast<unsigned>(file.type)<<" columns="<<file.columns<<" row_bytes="<<file.row_bytes<<" captured_us="<<elapsed*1000<<" uploads_terminal_observations_timed=0 profile=unqualified\n";
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::LookupPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.original_ids,original.size());
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],1);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],0);
        q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observed));
        EXPECT_EQ(observed.calls,1u);EXPECT_EQ(observed.completed_live_ids,original.size());EXPECT_EQ(observed.failed_frames,0u);
        EXPECT_EQ(observed.source_bytes,observed.source_rows*file.row_bytes);
        if(tokenQueries)EXPECT_TRUE(q::validQwenAdaptiveCounters(observed,file.row_bytes,maximum));
        if(!slots){EXPECT_EQ(observed.hit_rows,0u);EXPECT_EQ(observed.promotions,0u);EXPECT_EQ(observed.evictions,0u);}
        if(!original.empty()){
            std::vector<std::byte> selected(original.size()*file.row_bytes);
            for(std::size_t position=0;position<original.size();++position){const auto row=file.original(original[position]);std::memcpy(selected.data()+position*file.row_bytes,row.data(),row.size());}
            q::decodeCPUImmutableLookupRows(file.type,file.columns,file.row_bytes,selected.data(),original.size(),expected.data());
        }
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
        const auto io=read<q::ImmutableLookupDeviceState>(source.native->gpu_data_ptr(),1).front();
        const auto outcome=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(outcome.disposition,io.idCount?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished);
        EXPECT_EQ(outcome.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(io.issuedSequence,io.consumedSequence);
        const auto recordBytes=read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.record_bytes,1).front();
        EXPECT_EQ(recordBytes%sizeof(n::Fill),0u);
        const auto delivered=read<n::Fill>(static_cast<const std::byte*>(source.frame->gpu_data_ptr())+offsets.fills,recordBytes/sizeof(n::Fill));
        for(const auto& fill:delivered){EXPECT_EQ(fill.acknowledgement.destination_token,fill.original.destination_token);EXPECT_EQ(fill.acknowledgement.source.sequence,outcome.source.sequence);}
        if(slots){if(op.owner.rowWidth()==q::adaptive_lookup::RowKeyWidth::UInt32)verifySlots<q::adaptive_lookup::Slot32>();else verifySlots<q::adaptive_lookup::Slot64>();}
        expectedRequests+=io.idCount!=0;expectedIds+=io.idCount;EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Check unchanged original native cache bytes and padding with the actual source-row metadata width. */
    template<class Slot>void verifySlots(){
        const auto roots=owner->roots();const auto& op=owner->plan().parameters();
        const auto metadata=read<Slot>(static_cast<const std::byte*>(roots.cache->gpu_data_ptr())+op.cache.slots,slots);
        const auto payload=read<std::byte>(static_cast<const std::byte*>(roots.cache->gpu_data_ptr())+op.cache.payload,slots*op.slot_stride);
        for(std::size_t slot=0;slot<slots;++slot){
            for(std::size_t byte=file.row_bytes;byte<op.slot_stride;++byte)EXPECT_EQ(payload[slot*op.slot_stride+byte],std::byte{0xa5});
            if(q::adaptive_lookup::PhysicalSlot<Slot>::state(metadata[slot])==q::adaptive_lookup::SlotState::Ready){const auto row=file.original(metadata[slot].row);EXPECT_EQ(std::memcmp(row.data(),payload.data()+slot*op.slot_stride,row.size()),0);}
        }
    }
    /** @brief Prove acknowledged cancellation cannot promote its reservation, while warm pins and consumed witness retire exactly. */
    void cancelledACK(std::span<const std::uint64_t> original){
        requireProvider(interleaving!=ACKInterleaving::None && !original.empty() && original.back()==900,
            "ACK trace requires the original retained cancellation graph and target row");
        const bool originalUnchecked=interleaving==ACKInterleaving::OriginalUnchecked900;
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();
        const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        std::uint64_t warmGeneration=0;
        if(slots){const auto before=read<q::adaptive_lookup::Slot64>(static_cast<const std::byte*>(own.cache->gpu_data_ptr())+op.cache.slots,slots);
            for(const auto& row:before)if(row.row==7 && q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot64>::state(row)==q::adaptive_lookup::SlotState::Ready)warmGeneration=row.generation;
            requireProvider(bool(warmGeneration),"ACK trace requires its actual warmed original row7");}
        input(original);requireProvider(graph->launch(),"Pooled ACK cancellation replay failed");join();
        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],0);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],static_cast<int>(sampling_math::DeviceGenerationError::InvalidController));
        q::QwenAdaptiveCounterTotals observation;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observation));
        EXPECT_EQ(observation.calls,1u);EXPECT_EQ(observation.failed_frames,1u);EXPECT_EQ(observation.completed_live_ids,0u);EXPECT_EQ(observation.promotions,originalUnchecked?1u:0u);
        const auto frame=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
        const auto lookup=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(lookup.phase,p::LookupPhase::Idle);
        if(slots){const auto after=read<q::adaptive_lookup::Slot64>(static_cast<const std::byte*>(own.cache->gpu_data_ptr())+op.cache.slots,slots);
            bool preserved=false,promoted=false;for(const auto& row:after){const auto state=q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot64>::state(row);
                promoted|=row.row==900 && state==q::adaptive_lookup::SlotState::Ready;
                if(row.row==7 && state==q::adaptive_lookup::SlotState::Ready){preserved=true;EXPECT_EQ(row.generation,warmGeneration);}}
            EXPECT_TRUE(preserved);EXPECT_EQ(promoted,originalUnchecked);}
        const auto io=read<q::ImmutableLookupDeviceState>(source.native->gpu_data_ptr(),1).front();
        const auto frontier=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(frontier.disposition,n::SourceDisposition::Consumed);EXPECT_EQ(frontier.status,q::adaptive_lookup::Status::Cancelled);
        EXPECT_EQ(io.issuedSequence,io.consumedSequence);EXPECT_GT(io.nativeBytes,0u);EXPECT_EQ(io.nativeBytes,io.idCount*file.row_bytes);
        EXPECT_EQ(observation.source_bytes,io.nativeBytes);EXPECT_EQ(observation.source_rows,io.idCount);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
        expectedRequests+=io.idCount!=0;expectedIds+=io.idCount;
        EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Exercise the entire actual source capacity while free slots still guarantee mixed reservations/source-only. */
    void fullSourceCapacity(std::uint64_t first=2000){
        std::array<std::uint64_t,16> full;for(std::size_t i=0;i<full.size();++i)full[i]=first+i;replay(full);
        const auto roots=boundary->pooledStorage()->roots();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto io=read<q::ImmutableLookupDeviceState>(roots.native->gpu_data_ptr(),1).front();
        EXPECT_EQ(io.idCount,16u);EXPECT_EQ(io.nativeBytes,16*file.row_bytes);
        const auto fills=read<std::uint64_t>(static_cast<const std::byte*>(roots.native->gpu_data_ptr())+offsets.record_bytes,1).front()/sizeof(n::Fill);
        if(slots){EXPECT_GE(fills,slots-2u);EXPECT_LE(fills,slots);EXPECT_LT(fills,16u);}else EXPECT_EQ(fills,0u);
    }
    /** @brief Replay the exact cooperative production Begin alone; overflow must reject before any owner/source scope opens. */
    void rejectUniqueOverflow(){
        std::array<std::uint64_t,17> original;for(std::size_t i=0;i<original.size();++i)original[i]=3000+i;input(original);
        requireProvider(!boundary->hasActiveInterval(),"Bound negative must not leave a worker waiting on an invalid transaction");
        const auto beforeIO=service->stats();const auto roots=owner->roots();const std::array ownerRoots{roots.cache,roots.frame,roots.native};
        std::array<std::vector<std::byte>,3> originalOwner;
        for(std::size_t i=0;i<ownerRoots.size();++i)if(ownerRoots[i])originalOwner[i]=read<std::byte>(ownerRoots[i]->gpu_data_ptr(),ownerRoots[i]->size_bytes());
        const auto sourceNative=boundary->pooledStorage()->roots().native;
        const auto originalSource=read<std::byte>(sourceNative->gpu_data_ptr(),sourceNative->size_bytes());
        requireProvider(overbound->launch(),"Real provider bound-negative Begin launch failed");join();
        for(std::size_t i=0;i<ownerRoots.size();++i)if(ownerRoots[i]){
            const auto after=read<std::byte>(ownerRoots[i]->gpu_data_ptr(),ownerRoots[i]->size_bytes());EXPECT_EQ(after,originalOwner[i]);
        }
        EXPECT_EQ(read<std::byte>(sourceNative->gpu_data_ptr(),sourceNative->size_bytes()),originalSource);
        const auto delta=RetainedStorageTerminalObservation::between(1,service->mode(),beforeIO,service->stats()).counters;
        static_assert(std::has_unique_object_representations_v<RetainedStorageCounters>);
        const RetainedStorageCounters zero{};EXPECT_EQ(std::memcmp(&delta,&zero,sizeof(delta)),0);EXPECT_FALSE(boundary->hasActiveInterval());
        const auto& cp=coordinator->plan().parameters();const auto& op=owner->plan().parameters();
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coordinator->roots().frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_NE(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.unique_rows,17u);
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
    }
    /** @brief Fail the actual retained source parent before request publication or through its original provider.
     * @param original Original live producer IDs that still create the real two allocation waves.
     * @param abortBeforePublication Whether canonical request failure bars I/O before any ACK; false truncates the real original source.
     * @note No source bytes/completion/ACK are authored by this test. The existing worker owns every source outcome.
     */
    void replayFailure(std::span<const std::uint64_t> original,bool abortBeforePublication){
        input(original);
        if(abortBeforePublication){
            const int unhealthy=0;
            requireProvider(backend->hostToDeviceOnStream(control->healthy(),&unhealthy,sizeof(unhealthy),device.ordinal,stream),"Real source request cancellation input failed");
            TransferEngine::publishDeviceWrite(control->owner(),device,stream);join();
        }else requireProvider(::ftruncate(file.fd,0)==0,"Cannot fail the original admitted provider source");
        void* start=backend->createTimingEvent(device.ordinal);void* stop=backend->createTimingEvent(device.ordinal);
        requireProvider(start && stop,"Real provider failure timing events missing");
        requireProvider(backend->recordEvent(start,device.ordinal,stream),"Real provider failure timing start failed");
        requireProvider(graph->launch(),"Real provider failed parent launch failed");
        for(const auto& storage:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{storage.cache,storage.frame,storage.native})if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        TransferEngine::publishDeviceWrite(control->owner(),device,stream);
        TransferEngine::publishDeviceWrite(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get(),device,stream);
        requireProvider(backend->recordEvent(stop,device.ordinal,stream) && backend->waitForEvent(stop,device.ordinal),"Real provider failure did not retire");
        float elapsed=0;requireProvider(backend->eventElapsedTimeMs(start,stop,device.ordinal,&elapsed),"Real provider failure timing read failed");
        backend->destroyEvent(stop,device.ordinal);backend->destroyEvent(start,device.ordinal);
        std::cout<<"POOLED_PROVIDER_FAILURE_TIMING original="<<original.size()<<" maximum="<<maximum<<" slots="<<slots
            <<" path="<<(abortBeforePublication?"before_publication_abort":"provider_storage_failure")<<" captured_us="<<elapsed*1000
            <<" uploads_terminal_observations_timed=0 profile=unqualified\n";
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::LookupPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success); // Retired frame, not a successful source outcome.
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto allocation=read<p::AllocationFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.allocation_control,1).front();
        const auto expectedGrantCount=slots?3u:0u; // Independent fixture oracle: two free grants plus one victim.
        const auto grants=read<p::Grant>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.replies,expectedGrantCount);
        const auto accepted=read<p::AllocationDecision>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.accepted_plan,expectedGrantCount);
        EXPECT_EQ(allocation.count,0u);EXPECT_EQ(allocation.phase,p::AllocationPhase::Idle);
        for(std::size_t index=0;index<grants.size();++index){
            EXPECT_EQ(grants[index].row,accepted[index].row);EXPECT_EQ(grants[index].unique,accepted[index].unique);
            EXPECT_EQ(grants[index].destination_token,std::uint64_t{grants[index].unique}+1);
            EXPECT_NE(grants[index].generation,0u);EXPECT_EQ(grants[index].kind,p::GrantKind::Reservation);
        }

        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],0);EXPECT_NE(controller[sampling_math::kDeviceGenerationControlErrorCode],0);
        q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observed));
        EXPECT_EQ(observed.calls,1u);EXPECT_EQ(observed.failed_frames,1u);EXPECT_EQ(observed.completed_live_ids,0u);
        EXPECT_EQ(observed.source_rows,0u);EXPECT_EQ(observed.source_bytes,0u);EXPECT_EQ(observed.promotions,0u);
        EXPECT_EQ(observed.evictions,slots?1u:0u);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0); // No failed row/output escapes.
        const auto outcome=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(outcome.disposition,abortBeforePublication?n::SourceDisposition::Aborted:n::SourceDisposition::Consumed);
        EXPECT_NE(outcome.status,q::adaptive_lookup::Status::Success);
        EXPECT_EQ(read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.record_bytes,1).front(),0u);
        EXPECT_EQ(read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.payload_bytes,1).front(),0u);
        requireProvider(terminal->launch(),"Real provider failed complete-parent terminal launch failed");join();
        EXPECT_THROW(boundary->awaitTerminal(),std::runtime_error);
        const auto stats=boundary->terminalObservation();EXPECT_EQ(stats.counters.requests,abortBeforePublication?0u:1u);
        EXPECT_EQ(stats.counters.nativeBytes,0u);EXPECT_EQ(stats.counters.ids,0u);EXPECT_EQ(stats.counters.idBytes,0u);
        EXPECT_EQ(stats.counters.metadataBytes,abortBeforePublication?0u:q::kLookupRequestLiveBytes+q::kLookupCompletionLiveBytes+3*sizeof(ImmutableRowSubordinateIntent));
        EXPECT_EQ(stats.counters.terminalBytes,abortBeforePublication?32u:16u);
        EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Publish only canonical terminal completion and join the actual worker through its existing boundary. */
    void close(){
        const int complete=1;requireProvider(backend->hostToDeviceOnStream(control->row()+sampling_math::kDeviceGenerationControlRequestComplete,&complete,sizeof(complete),device.ordinal,stream),"Real provider terminal input failed");join();
        requireProvider(terminal->launch(),"Real provider retained terminal launch failed");join();boundary->awaitTerminal();
        const auto stats=boundary->terminalObservation();EXPECT_EQ(stats.transport,RetainedStorageTransport::PooledSubordinate);
        EXPECT_EQ(stats.counters.requests,expectedRequests);EXPECT_EQ(stats.counters.ids,expectedIds);EXPECT_EQ(stats.counters.idBytes,expectedIds*sizeof(std::int64_t));
        EXPECT_EQ(stats.counters.nativeBytes,expectedIds*file.row_bytes);EXPECT_EQ(stats.counters.terminalBytes,24u);
        expectedRequests=expectedIds=0;
    }
    ProviderFile& file;DeviceId device;IWorkerGPUContext& worker;IBackend* backend;void* stream;std::uint32_t maximum,slots;n::OriginalIdWidth width;bool tokenQueries;ACKInterleaving interleaving;ActivationPreparation activationPreparation;
    std::shared_ptr<PhysicalMemoryAuthority> memory;PhysicalMemoryAllocationLease activationLease,executableLease;
    std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<q::ImmutableLookupService> service;
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code;std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;
    std::unique_ptr<BufferArena> arena;std::optional<DeviceGenerationControlBinding> control;std::unique_ptr<IDeviceContext> context;
    std::vector<float> expected;ComputeGraph compute,smallCompute;DeviceGraphExecutor executor,smallExecutor;
    std::optional<q::PooledImmutableCoordinatorOperands> overboundInputs;std::unique_ptr<IGPUGraphCapture> graph,terminal,overbound,smallGraph;
    std::uint64_t identity=0,expectedRequests=0,expectedIds=0;
};
/** @brief Actual native source-only/positive-owner successful cycle and request reset, serially on one explicit participant. */
void providerReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit real-provider GPU is unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(auto slots:{0u,4u})for(auto maximum:{32u,513u}){
        std::shared_ptr<PhysicalMemoryAuthority> retiredAuthority;
        {
        ProviderFile file;ProviderReplay replay(file,device,worker,maximum,slots,slots?2:0,maximum<=512?n::OriginalIdWidth::UInt32:n::OriginalIdWidth::UInt64);
        retiredAuthority=replay.memory;
        replay.admit(1);replay.replay(std::array<std::uint64_t,2>{101,102});replay.fullSourceCapacity();replay.replay(std::array<std::uint64_t,2>{101,102});
        const auto hot=maximum<=512?900:ProviderFile::High;std::vector<std::uint64_t> large(maximum,hot);large[0]=700;large[1]=701;
        replay.replay(large);replay.replay(std::array<std::uint64_t,1>{hot});replay.replay({});replay.close();
        replay.admit(2);replay.replay(std::array<std::uint64_t,1>{hot});replay.replay(large);replay.close();replay.rejectUniqueOverflow();
        }
        // The outside authority observes retirement after every graph, stage,
        // boundary, source pin, mapped channel and activation owner is gone.
        // Driver-context memory belongs to its independent native authority.
        for(const auto participant:{device,DeviceId::cpu()})
            for(unsigned index=0;index<static_cast<unsigned>(PhysicalMemoryOwner::Count);++index){
                const auto owner=static_cast<PhysicalMemoryOwner>(index);
                EXPECT_EQ(retiredAuthority->claimedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
                EXPECT_EQ(retiredAuthority->reservedBytes(participant,owner),0u)<<toString(owner);
                EXPECT_EQ(retiredAuthority->committedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
            }

    }});
}
/** @brief Use the same original retained graph for real provider failure and GPU-owned exact I/O abort. */
void providerFailureReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit real-provider failure GPU is unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto slots:{0u,4u})for(const bool abortBeforePublication:{false,true}){
        std::shared_ptr<PhysicalMemoryAuthority> retiredAuthority;
        {
            ProviderFile file;ProviderReplay replay(file,device,worker,513,slots,slots?2:0,n::OriginalIdWidth::UInt64);
            retiredAuthority=replay.memory;replay.admit(1);replay.replay(std::array<std::uint64_t,2>{101,102});replay.close();
            replay.admit(2);std::vector<std::uint64_t> original(513,ProviderFile::High);original[0]=700;original[1]=701;
            replay.replayFailure(original,abortBeforePublication);
        }
        for(const auto participant:{device,DeviceId::cpu()})
            for(unsigned index=0;index<static_cast<unsigned>(PhysicalMemoryOwner::Count);++index){
                const auto owner=static_cast<PhysicalMemoryOwner>(index);
                EXPECT_EQ(retiredAuthority->claimedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
                EXPECT_EQ(retiredAuthority->reservedBytes(participant,owner),0u)<<toString(owner);
                EXPECT_EQ(retiredAuthority->committedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
            }
    }});
}
/** @return Existing loader discriminator, derived from its supported code range rather than a backend-local format list.
 * @note A newly inventoried format outside the current GGUF enum range fails the sweep; it cannot disappear silently. */
std::optional<GGUFTensorType> providerFormat(TensorType type){
    for(std::uint32_t code=0;code<=static_cast<std::uint32_t>(GGUFTensorType::BF16);++code){
        try{const auto format=static_cast<GGUFTensorType>(code);if(ggufToTensorType(format)==type)return format;}
        catch(const std::invalid_argument&){}
    }
    return std::nullopt;
}
/** @brief Cross the canonical original-native inventory through the new actual pooled source consumer on one backend. */
void providerAllFormatsReplay(DeviceId device,bool productionPLE=false){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit all-format provider GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        std::size_t supported=0,unsupported=0;
        for(const auto& entry:embeddingVerifierFormats()){
            const std::size_t columns=entry.prepared_native_quantized?768:257; // Three K blocks retain multi-block and two-byte padding cases.
            auto original=entry.create({97,columns},1701);const auto format=providerFormat(original->native_type());
            if(!format){
                requireProvider(original->native_type()==TensorType::Q8_1,"Canonical embedding format has no supported original GGUF provider descriptor");
                EXPECT_THROW((void)ggufToTensorType(static_cast<GGUFTensorType>(9)),std::invalid_argument);
                std::cout<<"POOLED_PROVIDER_FORMAT_UNSUPPORTED format="<<entry.label<<" reason=no_original_GGUF_provider_descriptor\n";++unsupported;continue;
            }
            ++supported;
            for(const auto slots:{0u,3u}){
                SCOPED_TRACE(::testing::Message()<<device.toString()<<" format="<<entry.label<<" columns="<<columns<<" slots="<<slots);
                std::shared_ptr<PhysicalMemoryAuthority> retiredAuthority;
                {
                    ProviderFile file(*original,*format,columns);
                    ProviderReplay replay(file,device,worker,productionPLE?64:33,slots,slots?2:0,
                        productionPLE?n::OriginalIdWidth::UInt64:n::OriginalIdWidth::UInt32,productionPLE);
                    retiredAuthority=replay.memory;
                    std::cout<<"POOLED_PROVIDER_FORMAT format="<<entry.label<<" columns="<<columns<<" row_bytes="<<file.row_bytes<<" slots="<<slots
                        <<" source_row_key_bits=32 original_id_bits="<<(productionPLE?64:32)<<" production_ple="<<productionPLE<<'\n';
                    if(productionPLE){
                        std::vector<std::uint64_t> large(64,81),small(16,81),first(16,1);
                        large[0]=79;large[1]=80;first[1]=2;
                        replay.admit(1);replay.replay(first);replay.fullSourceCapacity(16);
                        replay.replay(large);replay.replay(small);replay.replay({});replay.close();
                        replay.admit(2);replay.replay(first);replay.replay(large);replay.replay({});replay.close();
                    }else{
                        replay.admit(1);replay.replay(std::array<std::uint64_t,2>{1,2});replay.fullSourceCapacity(16);
                        std::vector<std::uint64_t> large(33,81);large[0]=79;large[1]=80;
                        replay.replay(large);replay.replay(std::array<std::uint64_t,1>{81});replay.replay({});replay.close();
                        replay.admit(2);replay.replay(std::array<std::uint64_t,3>{1,2,1});replay.replay(large);
                        std::array<std::uint64_t,16> fresh;for(std::size_t i=0;i<fresh.size();++i)fresh[i]=48+i;replay.replay(fresh);replay.replay({});replay.close();
                    }
                }
                for(const auto participant:{device,DeviceId::cpu()})
                    for(unsigned index=0;index<static_cast<unsigned>(PhysicalMemoryOwner::Count);++index){
                        const auto owner=static_cast<PhysicalMemoryOwner>(index);
                        EXPECT_EQ(retiredAuthority->claimedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
                        EXPECT_EQ(retiredAuthority->reservedBytes(participant,owner),0u)<<toString(owner);
                        EXPECT_EQ(retiredAuthority->committedBytes(participant,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u)<<toString(owner);
                    }
            }
        }
        EXPECT_EQ(unsupported,1u);EXPECT_EQ(supported+unsupported,embeddingVerifierFormats().size());
        std::cout<<"POOLED_PROVIDER_FORMAT_INVENTORY supported="<<supported<<" unsupported="<<unsupported<<" source_only_and_positive_owner=1 capture_reset_empty_tail=1\n";
    });
}
/** @brief Execute the model builder's exact INT64/token-count lookup helper through retained native capture. */
void productionPLELookupReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit production PLE GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto slots:{0u,4u}){
        ProviderFile file;ProviderReplay replay(file,device,worker,64,slots,slots?2:0,n::OriginalIdWidth::UInt64,true);
        const auto signature=replay.boundary->captureIdentity();
        const auto ids=replay.arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=replay.arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto output=replay.arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto tokens=DeviceRowRange::deviceCounted(4,static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        ComputeGraph narrowed,missing;
        EXPECT_THROW(appendPooledImmutablePLELookupGraph(narrowed,*replay.arena,replay.boundary,replay.code,tokens,
            (q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,64*sizeof(std::uint32_t)}),
            (q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,4}),*replay.control,
            (q::PooledNativeTensorSpan{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()}),"narrowed","narrowed_gather"),std::invalid_argument);
        EXPECT_THROW(appendPooledImmutablePLELookupGraph(missing,*replay.arena,replay.boundary,replay.code,tokens,
            (q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,64*sizeof(std::uint64_t)}),std::nullopt,*replay.control,
            (q::PooledNativeTensorSpan{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()}),"missing","missing_gather"),std::invalid_argument);
        EXPECT_TRUE(narrowed.getExecutionOrder().empty());EXPECT_TRUE(missing.getExecutionOrder().empty());
        const auto* reader=replay.compute.getNode("ple_gather");ASSERT_TRUE(reader);
        EXPECT_EQ(reader->stage->type(),ComputeStageType::QWEN4EXP_NGRAM_LOOKUP);
        const auto dump=reader->stage->getDumpInfo();ASSERT_EQ(dump.outputs.size(),1u);
        EXPECT_EQ(dump.outputs.front().rows,4u);EXPECT_EQ(dump.outputs.front().cols,16*file.columns);
        std::vector<std::uint64_t> large(64,ProviderFile::High),small(16,900);
        for(std::size_t i=0;i<large.size();++i){if(i%4==1)large[i]=7;else if(i%4==2)large[i]=700;}
        EXPECT_NE(std::memcmp(file.original(ProviderFile::High).data(),file.original(7).data(),file.row_bytes),0);
        for(std::size_t i=0;i<small.size();++i)if(i%2)small[i]=ProviderFile::High;
        replay.admit(1);replay.replay(large);replay.replay(small);replay.replay({});replay.close();
        replay.admit(2);replay.replay(large);replay.replay(small);replay.close();
        EXPECT_EQ(replay.boundary->captureIdentity(),signature);
        const auto observation=replay.boundary->pooledStorage()->retainedOwnerObservation();
        EXPECT_EQ(observation.authority().get(),replay.memory.get());
        EXPECT_EQ(observation.lookupStagingBytes(),replay.memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation));
    }});
}
/** @brief Replay full/partial/empty and request reset through one actual ACK/fill/cancellation frontier twenty times. */
void pooledACKReadyCancellation(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit pooled ACK GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto slots:{0u,2u}){
        ProviderFile file;ProviderReplay replay(file,device,worker,64,slots,0,n::OriginalIdWidth::UInt64,false,
            OwnerAdmission::Original,ACKInterleaving::Cancel900BeforeReady);
        auto* retained=replay.graph.get();const auto identity=replay.identity;std::uint64_t epoch=0;
        for(unsigned iteration=0;iteration<20;++iteration){
            const std::vector<std::uint64_t> warm(64,7);
            replay.admit(++epoch);replay.replay(warm);replay.replay({});replay.close();
            const auto count=iteration%3==0?64u:iteration%3==1?17u:1u;
            std::vector<std::uint64_t> cancelled(count,7);cancelled.back()=900;
            replay.admit(++epoch);replay.cancelledACK(cancelled);replay.close();
            replay.admit(++epoch);replay.replay(warm);replay.replay({});replay.close();
            EXPECT_EQ(replay.graph.get(),retained);EXPECT_EQ(replay.boundary->captureIdentity(),identity);
        }
    }
    {
        ProviderFile file;ProviderReplay original(file,device,worker,64,2,0,n::OriginalIdWidth::UInt64,false,
            OwnerAdmission::Original,ACKInterleaving::OriginalUnchecked900);
        const std::vector<std::uint64_t> warm(64,7);original.admit(1);original.replay(warm);original.close();
        std::vector<std::uint64_t> cancelled(17,7);cancelled.back()=900;
        original.admit(2);original.cancelledACK(cancelled);original.close();
        std::cout<<"POOLED_ACK_ORIGINAL_UNCHECKED_CONTROL expected_ready900=1 expected_promotions=1 expected_consumed_witness=1\n";
    }});
}
/**
 * @brief Replay original cold declarations, exact arena materialization and strict post-capture root seals.
 * @param device Explicit CUDA or ROCm participant; no backend substitution is admitted.
 * @note C0 and positive cache ownership share the real RAM/SSD provider. Full,
 * partial, empty and repeated request resets reuse the same retained graph.
 */
void productionColdArenaSealReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit cold arena GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto slots:{0u,4u})for(const auto maximum:{16u,64u,256u,1024u}){
        ProviderFile file;ProviderReplay replay(file,device,worker,maximum,slots,slots?2:0,n::OriginalIdWidth::UInt64,true,
            OwnerAdmission::Original,ACKInterleaving::None,ActivationPreparation::ColdGraph);
        std::vector<std::uint64_t> large(maximum,ProviderFile::High),small(16,700);
        for(std::uint64_t generation=1;generation<=3;++generation){
            replay.admit(generation);replay.replay(large);replay.replay(small);replay.replay({});replay.close();
        }
        const auto ids=replay.arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS);
        const auto output=replay.arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        // Retained shared arena roles reject substitution before a stage could
        // acquire another owner's pointer, including rebinding the same root.
        EXPECT_FALSE(replay.arena->bindExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids.get()));
        EXPECT_FALSE(replay.arena->bindExternalBuffer(BufferId::QFN_PLE_EMBEDDING,output.get()));
        auto* reader=replay.compute.getNode("ple_gather")->stage.get();
        EXPECT_NO_THROW(reader->graphCaptureVariantSignature());
        replay.join();
        // Real allocator leases reject retirement while the retained executable
        // owns these original activation pointers. No fake pointer or runtime
        // generation is installed to manufacture a physically invalid state.
        EXPECT_THROW(replay.backend->free(output->gpu_data_ptr(),device.ordinal),std::logic_error);
        EXPECT_THROW(replay.backend->free(ids->gpu_data_ptr(),device.ordinal),std::logic_error);
        EXPECT_NO_THROW(reader->graphCaptureVariantSignature());
        auto* begin=replay.compute.getNode(replay.compute.getExecutionOrder().front())->stage.get();
        EXPECT_NO_THROW(begin->graphCaptureVariantSignature());
    }});
}
/** @brief Reject actual foreign equal-sized owner leases before source roots publish. */
void productionSourceAuthorityRejection(DeviceId device){
    auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Explicit pooled preparation GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto slots:{0u,4u}){
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        {ProviderFile file;ProviderReplay replay(file,device,worker,64,slots,slots?2:0,n::OriginalIdWidth::UInt64,true,
                OwnerAdmission::RejectForeignAuthority);
            memory=replay.memory;}
        for(const auto participant:{device,DeviceId::cpu()})for(unsigned index=0;index<static_cast<unsigned>(PhysicalMemoryOwner::Count);++index)
            EXPECT_EQ(memory->claimedBytes(participant,static_cast<PhysicalMemoryOwner>(index),PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }});
}
}
#ifdef HAVE_CUDA
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderAllFormatsReplay){providerAllFormatsReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderFailureReplay){providerFailureReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDACapturedRealProviderResetReplay){providerReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionPLEInt64RepeatedTokenReplay){productionPLELookupReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionPLEAllFormatsReplay){providerAllFormatsReplay(DeviceId::cuda(0),true);}
TEST(PooledImmutableProviderSourceDevice,CUDAACKReadyCancellationReplay){pooledACKReadyCancellation(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionColdArenaSealReplay){productionColdArenaSealReplay(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionSourceAuthorityRejection){productionSourceAuthorityRejection(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderAllFormatsReplay){providerAllFormatsReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderFailureReplay){providerFailureReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmCapturedRealProviderResetReplay){providerReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionPLEInt64RepeatedTokenReplay){productionPLELookupReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionPLEAllFormatsReplay){providerAllFormatsReplay(DeviceId::rocm(0),true);}
TEST(PooledImmutableProviderSourceDevice,ROCmACKReadyCancellationReplay){pooledACKReadyCancellation(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionColdArenaSealReplay){productionColdArenaSealReplay(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionSourceAuthorityRejection){productionSourceAuthorityRejection(DeviceId::rocm(0));}
#endif
}
