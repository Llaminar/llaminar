/** @file Test__PooledLocalHotNativeDevice.cpp
 * @brief Actual production L1 adapters on original source/PMA roots with immutable recorded responses.
 *
 * This component replays native cache policy, byte copies and decoding. Only
 * response inputs are recorded: it proves no L2 delivery, Rank cohort or model
 * inference. No source seal or physical retirement authority is synthesized.
 */
#include "fixtures/backends/NativeGraphTimingEvents.h"
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/PooledImmutableSourceKernels.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotOperands.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "backends/NativeOrderingEventGroup.h"
#include "fixtures/qwen4exp/PooledOriginalSourceSetupPrograms.h"
#include "backends/ExplicitGPUStream.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "loaders/HostImmutableRowProvider.h"
#include "loaders/ImmutableLookupSetupObservation.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PooledNgramReplayTrace.h"
#include "LocalHotRecordedResponse.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/RequestInputPositionAdmission.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <thread>
#include <tuple>
#include <unistd.h>
namespace llaminar2::test { namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;namespace h=q::local_hot;namespace t=q::local_hot_transport;
#include "PooledImmutableProviderFile.inc"
/** @brief Actual supported loader format, never a separate backend inventory. */
std::optional<GGUFTensorType> originalFormat(TensorType type)
{
    for(unsigned code=0;code<=unsigned(GGUFTensorType::BF16);++code)try {
        if(ggufToTensorType(GGUFTensorType(code))==type)return GGUFTensorType(code);
    }catch(const std::invalid_argument&){}
    return std::nullopt;
}
/** @brief Explicit source fixture and complete original native row bank, retained until every graph retires. */
struct ReplayCase {
    std::string label;
    std::unique_ptr<ProviderFile> file;
    std::vector<std::uint64_t> keys;
    std::vector<std::byte> native;
};
/** @return Same original packet staging plus an aligned separately observed terminal output, priced before admission. */
std::size_t nativeReplayStagingBytes(const ReplayCase& c)
{
    const auto packet=std::size_t{256}+c.keys.size()*8+c.native.size();
    return ((packet+63)&~std::size_t{63})+h::terminalObservationBytes(1,sampling_math::kDeviceGenerationControlCount);
}
/**
 * @brief Keep admission and materialization on one original query-capacity declaration.
 * @param c Original native source file and recorded row roster.
 * @param device Exact consumer allocator.
 * @param slots Optional native local cache population.
 * @param maximumOriginal Full sixteen-head ID capacity, independent of compact misses.
 * @return Original geometry and arena declarations, with no live capacity decision.
 */
auto geometry(const ReplayCase& c,DeviceId device,std::uint32_t slots,std::uint32_t maximumOriginal=64,bool nextWindow=false)
{
    const auto source=c.file->descriptor();const PhysicalMemoryAllocatorIdentity gpu{0,device},host{0,DeviceId::cpu()};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{gpu,ImmutableRowStorage::VRAM},0,0,{}},{{host,ImmutableRowStorage::RAM},0,0,{}}},{gpu},{},
        {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},{},16);
    auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpu},gpu);
    auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},maximumOriginal);
    auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,host,gpu);
    auto channel=q::ImmutableLookupLayout::createSubordinate(16,source.rowBytes());
    auto local=q::PooledImmutableLocalHotGeometry::declare(source,gpu,maximumOriginal,slots,{65536,3,4});
    std::vector declarations{
        CPUTensorStorageDeclaration(1,maximumOriginal*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(maximumOriginal/16,16*c.file->columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
    if(nextWindow)declarations.emplace_back(1,2*(maximumOriginal/16+1)+4,BufferTensorType::INT32,
        PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
    return std::tuple{std::move(layout),std::move(physical),std::move(policy),std::move(cpu),std::move(channel),std::move(local),std::move(declarations)};
}
/** @brief Existing inventory, low-key scan pressure, or full-width large-radix correctness. */
enum class ReplayShape { SmallInventory, LargeRadixScan, LargeRadixWideKey, NextWindowCount, LargeFutureAdoption, SmallProbeBoundary, SmallProbeTiming };
/**
 * @brief Run the exact selected native backend; absence is an error, never a skip.
 * @param device Original requested native participant.
 * @param slots Admitted local slots; the small-boundary sweep includes zero capacity.
 * @param wide Select the original sparse high-key-only inventory.
 * @param shape Immutable original query capacity and finite replay workload.
 */
void replayLocal(DeviceId device,std::uint32_t slots,bool wide,
    ReplayShape shape=ReplayShape::SmallInventory)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    const bool nextWindow=shape==ReplayShape::NextWindowCount;
    const bool scanPressure = shape == ReplayShape::LargeRadixScan;
    const bool largeWideKey = shape == ReplayShape::LargeRadixWideKey;
    const bool timingMode = shape == ReplayShape::SmallProbeTiming;
    const bool smallBoundary = shape == ReplayShape::SmallProbeBoundary || timingMode;
    const bool large = scanPressure || largeWideKey || shape == ReplayShape::LargeFutureAdoption || smallBoundary;
    const std::uint32_t maximumOriginal = smallBoundary ? 512 : large ? 1024 : 64;
    requireProvider(!large || ((slots == 4 || (smallBoundary && slots == 0)) &&
        (wide == largeWideKey || (smallBoundary && wide))),
        "Large native trace requires its exact original four-slot format inventory");
    std::vector<ReplayCase> cases;
    if(wide)cases.push_back({"UInt64_IQ4_NL",std::make_unique<ProviderFile>(),{7,101,102,ProviderFile::High,900},{}});
    else for(const auto& entry:embeddingVerifierFormats()) {
        const auto columns=entry.prepared_native_quantized?768u:257u;
        auto native=entry.create({97,columns},1701);const auto format=originalFormat(native->native_type());
        if(!format){requireProvider(native->native_type()==TensorType::Q8_1,"Canonical format has no original GGUF declaration");continue;}
        if (large && *format != GGUFTensorType::F32 && *format != GGUFTensorType::IQ4_NL) continue;
        std::vector<std::uint64_t> keys{1,2,3,4,5};
        if (large) { keys.resize(64); std::iota(keys.begin(), keys.end(), 1); }
        cases.push_back({entry.label,std::make_unique<ProviderFile>(*native,*format,columns),std::move(keys),{}});
    }
    if(timingMode) {
        // Match the real-model observer's IQ4_NL160/90-byte row and table
        // extent. Scattered authored rows exercise the production eight-pass
        // radix range without allocating the table's sparse file holes.
        // Responses are still recorded component inputs, not real I/O timing.
        constexpr std::uint64_t qfnRows=320001536;
        std::vector<std::uint64_t> keys(64);
        for(std::size_t i=0;i<keys.size();++i)keys[i]=7+i*((qfnRows-16)/keys.size());
        cases.push_back({"QFN_IQ4_NL160",std::make_unique<ProviderFile>(qfnRows,keys),std::move(keys),{}});
    }
    requireProvider(!scanPressure || cases.size()==2,
        "Large native trace lost its original FP32/IQ4_NL representative formats");
    requireProvider(!largeWideKey || cases.size()==1,
        "Wide large-radix trace lost its original sparse IQ4_NL source");
    for(auto& c:cases)for(auto key:c.keys){const auto row=c.file->original(key);c.native.insert(c.native.end(),row.begin(),row.end());}
    const std::array devices{device};auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto memory=session->memory();const auto gpu=session->resource(device),host=session->resource(DeviceId::cpu());
    const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
    constexpr std::size_t AdmissionOffset=0,RequestOffset=64,LaneOffset=192,KeyBytesOffset=224,ControlBytes=256;
    PhysicalMemoryPlanBuilder bom;
    for(const auto& c:cases) {
        const auto [layout,physical,policy,cpu,channel,local,declarations]=geometry(c,device,slots,maximumOriginal,nextWindow);
        for(const auto& contribution:{cpu.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
            q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
            q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,c.file->descriptor(),gpu),local.mandatoryBOM(gpu,host),local.slotDependentBOM(gpu,host),io.bom(host),
            RetainedParticipantGraphExecutable::generalBOM(host,gpu,5)})for(const auto& row:contribution.resources())bom.add(row);
        bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const auto activation=maximumOriginal*8+4+sampling_math::kDeviceGenerationControlCount*4+maximumOriginal*c.file->columns*4+
            (nextWindow?(2*(maximumOriginal/16+1)+4)*sizeof(std::int32_t):0);
        bom.add(gpu,PhysicalMemoryOwner::ActivationArena,activation);
        for(const auto& declaration:declarations)bom.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
        bom.add(gpu,PhysicalMemoryOwner::ActivationTransportStaging,nativeReplayStagingBytes(c));
        for(const auto program:originalPooledSourceSetupOrderingPrograms(physical,policy,c.file->descriptor()))
            {const auto contribution=NativeOrderingEventGroup::bom(program,host,gpu);for(const auto& row:contribution.resources())bom.add(row);}
    }
    session->admit(bom.build());const auto& permit=session->permit(device);auto worker=permit.worker();auto* backend=getBackendFor(device);
    const auto stream=ExplicitGPUStream(worker->defaultStream()).get();
    std::size_t ordinal=0;
    for(const auto& c:cases) {
        const auto source=c.file->descriptor();const auto [layout,physical,policy,cpu,channel,local,declarations]=geometry(c,device,slots,maximumOriginal,nextWindow);
        const RetainedHostServiceInstance ioKey{0,device,RetainedHostServiceRole::ImmutableLookupIO,ordinal++};
        RetainedHostServiceDeclarations::Builder builder(0);builder.add(ioKey,io);auto declared=std::move(builder).freeze();
        auto binding=declared->bind(memory,std::span{&ioKey,std::size_t{1}});
        std::shared_ptr<HostImmutableRowProvider> provider;
        std::shared_ptr<q::ImmutableLookupService> service;
        std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;
        std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
        std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;
        std::shared_ptr<q::PooledImmutableLocalHotPreparedStorage> storage;
        std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode;
        std::unique_ptr<q::IPooledImmutableLocalHotKernels> code;
        PhysicalMemoryAllocationLease activationLease,stagingLease;
        BufferArena arena;
        std::optional<DeviceGenerationControlBinding> controllerValue;
        std::shared_ptr<DeviceTransferBuffer> staging;
        std::shared_ptr<TensorBase> hashes,count,controlRoot,output,windowRoot;
        std::optional<q::PooledImmutableLocalHotOperands> queryValue,futureValue,adoptionValue,readValue,resultValue,resetValue,frameValue;
        local_hot_recorded::Args args{};
        std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> graphs;
        for(unsigned profile=0;profile<5;++profile)graphs.push_back(RetainedParticipantGraphExecutableFixtureAccess::original(permit,stream));
        const auto onWorker=[&](auto function){if(worker->ownsCurrentThread())function();else worker->submitAndWait(function);};
        const auto launch=[&](local_hot_recorded::Action action,const local_hot_recorded::Args& operands){onWorker([&]{
#ifdef HAVE_CUDA
            if(device.is_cuda())local_hot_recorded::launchCUDA(action,operands,stream);
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())local_hot_recorded::launchROCm(action,operands,stream);
#endif
        });};
        worker->submitAndWait([&] {
            provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpu,memory));
            service=q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu(),binding->take(ioKey));
            auto sourceBinding=service->bindSubordinate(device);
            const auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
            const auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
            const auto sourcePlan=q::PooledImmutableSourceGeometry::fromPlans(ownerPlan,coordinatorPlan,source,1);
            owner=q::PooledImmutableOwnerPreparedStorage::create(ownerPlan,memory,stream);
            coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(coordinatorPlan,memory,stream);
#ifdef HAVE_CUDA
            if(device.is_cuda()){poolCode=q::createCUDAPooledImmutableNativeKernels();code=q::createCUDAPooledImmutableLocalHotKernels();}
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm()){poolCode=q::createROCmPooledImmutableNativeKernels();code=q::createROCmPooledImmutableLocalHotKernels();}
#endif
            requireProvider(poolCode && code,"Exact native L1 adapters are unavailable");
            poolCode->initialize(*owner,stream);poolCode->initialize(*coordinator,stream);
            boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourcePlan,owner,coordinator,memory,stream);
            storage=q::PooledImmutableLocalHotPreparedStorage::create(local,boundary->pooledStorage(),memory,stream);code->initialize(*storage,stream);
            storage->requireFor(local,boundary->pooledStorage(),memory);
            // These are the actual materialized roots, not configured ceilings
            // or a recreated equal-geometry source. GPU and CPU ready leases
            // must remain disjoint in the production setup projection.
            const std::array<std::shared_ptr<const q::PooledImmutableSourcePreparedStorage>,1> sourceOwners{boundary->pooledStorage()};
            const std::array<std::shared_ptr<const q::PooledImmutableLocalHotPreparedStorage>,1> localOwners{storage};
            const auto setup=ImmutableLookupSetupObservation::captureAdaptive(layout,cpu,service,sourceOwners,localOwners);
            std::size_t localRows=0,readyRows=0;
            for (const auto& row : setup.owners()) {
                if (row.role==ImmutableLookupSetupObservation::Role::LocalGPUCache) {
                    ++localRows;EXPECT_EQ(row.allocator,local.allocator());EXPECT_EQ(row.admitted_slots,slots);
                    EXPECT_EQ(row.payload_bytes,std::size_t(slots)*source.rowBytes());
                    EXPECT_EQ(row.metadata_bytes,local.cacheOffsets().bytes-row.payload_bytes);
                    EXPECT_EQ(row.table_storage_bytes,local.cacheOffsets().bytes);
                    EXPECT_EQ(row.lookup_staging_bytes,local.frameOffsets().bytes+local.scratchOffsets().bytes);
                    EXPECT_EQ(row.source_storage_bytes,0u);
                } else if (row.role==ImmutableLookupSetupObservation::Role::LocalReady) {
                    ++readyRows;EXPECT_EQ(row.allocator,(PhysicalMemoryAllocatorIdentity{0,DeviceId::cpu()}));
                    EXPECT_EQ(row.lookup_staging_bytes,TransferEngine::mappedHostRegionAllocationBytes(8));
                    EXPECT_EQ(row.table_storage_bytes,0u);EXPECT_EQ(row.source_storage_bytes,0u);
                    EXPECT_EQ(row.payload_bytes,0u);EXPECT_EQ(row.metadata_bytes,0u);EXPECT_EQ(row.admitted_slots,0u);
                }
            }
            EXPECT_EQ(localRows,1u);EXPECT_EQ(readyRows,1u);
            const std::array<std::shared_ptr<const q::PooledImmutableLocalHotPreparedStorage>,2> duplicateLocals{storage,storage};
            EXPECT_THROW((void)ImmutableLookupSetupObservation::captureAdaptive(layout,cpu,service,sourceOwners,duplicateLocals),std::invalid_argument);

            const auto activation=maximumOriginal*8+4+sampling_math::kDeviceGenerationControlCount*4+maximumOriginal*c.file->columns*4+
            (nextWindow?(2*(maximumOriginal/16+1)+4)*sizeof(std::int32_t):0);
            activationLease=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,activation);
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};
            for(unsigned i=0;i<4;++i){std::shared_ptr<TensorBase> root(declarations[i].materialize(memory));std::memset(root->raw_mutable_data(),0,root->size_bytes());
                TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);
                requireProvider(arena.registerExternalBuffer(roles[i],root),"Original L1 arena declaration failed");}
            if(nextWindow){
                windowRoot=std::shared_ptr<TensorBase>(declarations[4].materialize(memory));
                auto* words=static_cast<std::int32_t*>(windowRoot->raw_mutable_data());
                const int requestCapacity=maximumOriginal/16+1;
                std::fill_n(words,windowRoot->size_bytes()/4,1);
                std::iota(words+requestCapacity,words+2*requestCapacity,100);
                words[2*requestCapacity]=requestCapacity;words[2*requestCapacity+1]=100;
                words[2*requestCapacity+2]=1;words[2*requestCapacity+3]=1;
                TransferEngine::prepareDeviceInput(windowRoot.get(),device,stream);
                TransferEngine::requireDeviceInput(windowRoot.get(),device,stream);
            }
            controllerValue.emplace(DeviceGenerationControlBinding::admit(arena,device,0));
            const auto& controller=*controllerValue;
            const auto controlBytes=nativeReplayStagingBytes(c);
            stagingLease=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationTransportStaging,controlBytes);
            TransferEngine transfers;staging=transfers.allocateDeviceTransferBuffer(controlBytes,device);
            auto ptr=[&](std::size_t offset){return static_cast<std::byte*>(staging->mutableDeviceData(offset));};
            const t::ConsumerGeometry consumer{0,1,local.maximumOriginalIds(),local.maximumCompactMissRows(),local.extent().rows(),local.rowBytes(),std::uint32_t(local.format()),storage->consumerStorageIdentity()};
            args={static_cast<std::byte*>(storage->roots().frame->gpu_data_ptr()),local.frameOffsets(),consumer,
                reinterpret_cast<t::SourceAdmissionHeader*>(ptr(AdmissionOffset)),reinterpret_cast<t::RequestHeader*>(ptr(RequestOffset)),
                reinterpret_cast<t::RequestLaneIdentity*>(ptr(LaneOffset)),reinterpret_cast<std::uint64_t*>(ptr(KeyBytesOffset)),
                reinterpret_cast<const std::uint64_t*>(static_cast<const std::byte*>(boundary->pooledStorage()->roots().native->gpu_data_ptr())+sourcePlan.parameters().offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration)),
                reinterpret_cast<const std::uint64_t*>(static_cast<const std::byte*>(boundary->pooledStorage()->roots().native->gpu_data_ptr())+sourcePlan.parameters().offsets.invocation),
                storage->plan().table(),storage->plan().sourceStorageGeneration(),reinterpret_cast<const std::uint64_t*>(ptr(ControlBytes)),ptr(ControlBytes+c.keys.size()*8),unsigned(c.keys.size()),c.file->row_bytes};
            requireProvider(backend->hostToDeviceOnStream(ptr(ControlBytes),c.keys.data(),c.keys.size()*8,device.ordinal,stream),"Recorded source keys upload failed");
            requireProvider(backend->hostToDeviceOnStream(ptr(ControlBytes+c.keys.size()*8),c.native.data(),c.native.size(),device.ordinal,stream),"Recorded native rows upload failed");
            hashes=arena.getSharedTensor(roles[0]);count=arena.getSharedTensor(roles[1]);controlRoot=arena.getSharedTensor(roles[2]);output=arena.getSharedTensor(roles[3]);
            queryValue.emplace(q::PooledImmutableLocalHotOperands::query(*storage,consumer,DeviceRowRange::deviceCounted(maximumOriginal/16,static_cast<const int*>(count->gpu_data_ptr())),
                {hashes,roles[0],0,hashes->size_bytes()},q::PooledNativeTensorSpan{count,roles[1],0,4},controller,q::PooledImmutableLocalHotAccessIntent::CurrentInvocation));
            futureValue.emplace(q::PooledImmutableLocalHotOperands::query(*storage,consumer,DeviceRowRange::deviceCounted(maximumOriginal/16,static_cast<const int*>(count->gpu_data_ptr())),
                {hashes,roles[0],0,hashes->size_bytes()},q::PooledNativeTensorSpan{count,roles[1],0,4},controller,q::PooledImmutableLocalHotAccessIntent::FutureReadAhead));
            if(nextWindow){
                const int capacity=maximumOriginal/16+1;
                const auto* words=static_cast<const std::int32_t*>(windowRoot->gpu_data_ptr());
                DevicePrefillChunkGraphBinding original{.backend=backend,
                    .request_token_ids_device=words,.request_position_ids_device=words+capacity,
                    .request_total_rows_device=words+2*capacity,.cached_tokens_device=words+2*capacity+1,
                    .chunk_token_ids_device=const_cast<std::int32_t*>(words),.chunk_position_ids_device=const_cast<std::int32_t*>(words+2*capacity+1),
                    .chunk_real_rows_device=const_cast<std::int32_t*>(words+2*capacity+2),
                    .chunk_row_stride_device=const_cast<std::int32_t*>(words+2*capacity+3),
                    .request_row_capacity=capacity,.bucket_seq_len=1,.capture_identity=storage->captureIdentity()};
                auto window=q::Qwen4ExpPrefillNextWindowBinding::fromOriginalChunk(original,device,maximumOriginal/16,
                    RequestInputPositionLayoutRequirement::ContiguousSingleRequestPrefill);
                futureValue.emplace(q::PooledImmutableLocalHotOperands::nextWindowQuery(*storage,consumer,window,
                    q::PooledNativeTensorSpan{hashes,roles[0],0,hashes->size_bytes()},controller));
                EXPECT_TRUE(futureValue->hasNextWindow());
                EXPECT_NE(futureValue->declarationIdentity(),queryValue->declarationIdentity());
            }
            adoptionValue.emplace(q::PooledImmutableLocalHotOperands::adoption(*storage,consumer,
                DeviceRowRange::deviceCounted(maximumOriginal/16,static_cast<const int*>(count->gpu_data_ptr())),
                {hashes,roles[0],0,hashes->size_bytes()},q::PooledNativeTensorSpan{count,roles[1],0,4},
                {staging,AdmissionOffset,sizeof(t::SourceAdmissionHeader)},controller));
            readValue.emplace(q::PooledImmutableLocalHotOperands::read(*storage,consumer,{output,roles[3],0,output->size_bytes()},controller));
            resultValue.emplace(q::PooledImmutableLocalHotOperands::result(*storage,consumer,{staging,RequestOffset,sizeof(t::RequestHeader)},controller));
            resetValue.emplace(q::PooledImmutableLocalHotOperands::requestReset(*storage,consumer,{staging,AdmissionOffset,sizeof(t::SourceAdmissionHeader)},controller));
            frameValue.emplace(q::PooledImmutableLocalHotOperands::frame(*storage,consumer,controller));
            for(auto* operand:{&*queryValue,&*futureValue,&*adoptionValue,&*readValue,&*resultValue,&*resetValue,&*frameValue})operand->seal();
            EXPECT_THROW(frameValue->require(q::PooledImmutableLocalHotAction::Adopt,*storage),std::logic_error);
            EXPECT_NE(adoptionValue->declarationIdentity(),frameValue->declarationIdentity());
            const auto& query=*queryValue;const auto& future=*futureValue;const auto& read=*readValue;const auto& result=*resultValue;const auto& frame=*frameValue;
            for(unsigned profile=0;profile<5;++profile){
                auto& capture=RetainedParticipantGraphExecutableFixtureAccess::capture(graphs[profile]);
                ScopedBackendGraphCapture scope(*worker,capture,"actual L1 immutable recorded response replay");requireProvider(scope.begin(),"L1 graph recording failed");
                const auto phase=[&](q::PooledImmutableLocalHotAction action,const q::PooledImmutableLocalHotOperands& operands){code->enqueue(action,*storage,operands,stream);};
                phase(q::PooledImmutableLocalHotAction::Probe,profile==1 || profile==2 || (profile==4 && nextWindow)?future:query);
                if(profile==0 || profile==3){launch(local_hot_recorded::Action::RecordedResponse,args);phase(q::PooledImmutableLocalHotAction::AcceptResult,result);}
                if(profile==4) {scope.finish();requireProvider(capture.prepareRuntimeContextStorage(memory)&&capture.instantiate(),"L1 negative guard graph instantiation failed");continue;}
                if(profile==2)phase(q::PooledImmutableLocalHotAction::Discard,frame);
                else {
                    if(profile==1)phase(q::PooledImmutableLocalHotAction::Adopt,*adoptionValue);
                    storage->enqueueReadyWait(stream);phase(q::PooledImmutableLocalHotAction::LocalRead,read);storage->enqueueLocalReadCompleteWait(stream);
                    phase(q::PooledImmutableLocalHotAction::ConsumeDemand,frame);phase(q::PooledImmutableLocalHotAction::CopyFills,frame);
                    phase(profile==3?q::PooledImmutableLocalHotAction::Cancel:q::PooledImmutableLocalHotAction::CompleteFills,frame);
                }
                phase(q::PooledImmutableLocalHotAction::Retire,frame);scope.finish();
                requireProvider(capture.prepareRuntimeContextStorage(memory)&&capture.instantiate(),"L1 graph instantiation failed");
            }
        });
            const auto& controller=*controllerValue;const auto& reset=*resetValue;
            const auto captureIdentity=storage->captureIdentity();std::array<std::uint64_t,5> graphOrdinals{};std::uint64_t sequence=0;
            const auto join=[&]{onWorker([&]{requireProvider(backend->synchronizeStream(stream,device.ordinal),"L1 diagnostic native join failed");});};join();
            // The explicit diagnostic uses the existing complete retained programs.
            // Neither input upload nor result comparison is inside these events.
            std::unique_ptr<NativeGraphTimingEvents> timing;
            if(timingMode)timing=std::make_unique<NativeGraphTimingEvents>(*worker,device);
            const auto observe=[&]<class T>(std::size_t offset){T value{};onWorker([&]{requireProvider(backend->deviceToHostOnStream(&value,args.frame+offset,sizeof(T),device.ordinal,stream),"L1 terminal metadata copy failed");});join();return value;};
            /**
             * @brief Replay actual production dispatch, then inspect only its completed native frame.
             * @param profile Original retained graph's current/discard/cancel action sequence.
             * @param ids Authored original row IDs; capacity never changes their live extent.
             * @return Genuine terminal frame, used by the finite scan's hot-row oracle.
             */
            const auto run=[&](unsigned profile,std::span<const std::uint64_t> ids){
                requireProvider(ids.size()%16==0 && ids.size()<=maximumOriginal,"L1 recorded original token extent is invalid");
                std::vector<std::uint64_t> full(maximumOriginal,0);std::copy(ids.begin(),ids.end(),full.begin());
                const float poison=-12345.f;std::vector<float> pristine(maximumOriginal*c.file->columns,poison);
                const auto before = large ? std::optional{observe.template operator()<h::Counters>(local.frameOffsets().counters)} : std::nullopt;
                const int tokens=int(ids.size()/16);onWorker([&]{requireProvider(backend->hostToDeviceOnStream(hashes->gpu_data_ptr(),full.data(),maximumOriginal*sizeof(std::uint64_t),device.ordinal,stream),"L1 original recorded hashes upload failed");
                requireProvider(backend->hostToDeviceOnStream(count->gpu_data_ptr(),&tokens,4,device.ordinal,stream),"L1 original token count upload failed");
                requireProvider(backend->hostToDeviceOnStream(output->gpu_data_ptr(),pristine.data(),pristine.size()*4,device.ordinal,stream),"L1 output poison upload failed");});
                if(nextWindow){
                    const int total=tokens+1,offset=2*(maximumOriginal/16+1);
                    onWorker([&]{requireProvider(backend->hostToDeviceOnStream(
                        static_cast<std::int32_t*>(windowRoot->gpu_data_ptr())+offset,&total,4,device.ordinal,stream),
                        "Original next-window request count upload failed");});
                }
                const auto submit=[&]{requireProvider(RetainedParticipantGraphExecutableFixtureAccess::submit(graphs[profile],++graphOrdinals[profile]),"L1 original graph submit failed");};
                if(timing)onWorker([&]{timing->begin(stream);submit();timing->end(stream);});
                else submit();
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(graphs[profile])){requireProvider(std::chrono::steady_clock::now()<deadline,"L1 exact original terminal deadline");std::this_thread::yield();}
                if(nextWindow){
                    std::int32_t currentRows=0;
                    onWorker([&]{requireProvider(backend->deviceToHostOnStream(&currentRows,
                        static_cast<const std::int32_t*>(windowRoot->gpu_data_ptr())+2*(maximumOriginal/16+1)+2,
                        4,device.ordinal,stream),"Original current count preservation observation failed");});join();
                    EXPECT_EQ(currentRows,1);
                }
                const auto identity=observe.template operator()<h::FrameIdentity>(local.frameOffsets().identity);
                const auto native=observe.template operator()<q::adaptive_lookup::FrameControl>(local.frameOffsets().control);
                EXPECT_EQ(identity.query.sequence,++sequence);EXPECT_EQ(identity.disposition,h::Disposition::Idle);EXPECT_EQ(identity.terminal_error,q::adaptive_lookup::Status::Success);
                EXPECT_EQ(native.phase,q::adaptive_lookup::Phase::Idle);EXPECT_EQ(native.held_pins,0u);
                EXPECT_EQ(native.status,profile==2 || profile==3?q::adaptive_lookup::Status::Cancelled:q::adaptive_lookup::Status::Success);
                std::vector<float> actual(pristine.size());onWorker([&]{requireProvider(backend->deviceToHostOnStream(actual.data(),output->gpu_data_ptr(),actual.size()*4,device.ordinal,stream),"L1 output terminal copy failed");});join();
                if(profile!=2){std::vector<std::byte> rows;for(auto id:ids){const auto row=c.file->original(id);rows.insert(rows.end(),row.begin(),row.end());}
                    std::vector<float> expected(ids.size()*c.file->columns);q::decodeCPUImmutableLookupRows(c.file->type,int(c.file->columns),c.file->row_bytes,rows.data(),ids.size(),expected.data());
                    EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*4),0)<<c.label;}
                for(std::size_t i=profile==2?0:ids.size()*c.file->columns;i<actual.size();++i)EXPECT_EQ(actual[i],poison);
                if (large) {
                    EXPECT_EQ(native.original_ids,ids.size());
                    requireProvider(native.unique_rows<=ids.size() && native.source_rows<=native.unique_rows,
                        "Large native terminal row counts exceed their original live bounds");
                    // The independent first-occurrence oracle checks the real
                    // GPU radix/inverse output, including stale scratch reuse.
                    std::vector<std::uint64_t> expectedUnique;
                    std::vector<std::uint32_t> expectedInverse;
                    for (const auto id : ids) {
                        auto found=std::find(expectedUnique.begin(),expectedUnique.end(),id);
                        if (found==expectedUnique.end()) { expectedUnique.push_back(id); found=expectedUnique.end()-1; }
                        expectedInverse.push_back(static_cast<std::uint32_t>(found-expectedUnique.begin()));
                    }
                    EXPECT_EQ(native.unique_rows,expectedUnique.size());
                    std::vector<q::adaptive_lookup::UniqueRecord> unique(expectedUnique.size());
                    std::vector<std::uint32_t> inverse(ids.size());
                    std::vector<std::uint64_t> misses(native.source_rows);
                    const auto copy=[&](void* destination,std::size_t offset,std::size_t bytes) {
                        if (!bytes) return;
                        onWorker([&]{requireProvider(backend->deviceToHostOnStream(destination,args.frame+offset,bytes,device.ordinal,stream),
                            "Large native terminal live-prefix observation failed");});
                    };
                    copy(unique.data(),local.frameOffsets().unique,unique.size()*sizeof(unique.front()));
                    copy(inverse.data(),local.frameOffsets().original_to_unique,inverse.size()*sizeof(inverse.front()));
                    copy(misses.data(),local.frameOffsets().miss_keys,misses.size()*sizeof(misses.front()));join();
                    EXPECT_EQ(inverse,expectedInverse);
                    std::vector<std::uint64_t> expectedMisses;
                    std::uint64_t hitOriginal=0;
                    for (std::size_t u=0; u<unique.size(); ++u) {
                        EXPECT_EQ(unique[u].row,expectedUnique[u]);
                        EXPECT_EQ(unique[u].first_original_coordinate,
                            std::find(ids.begin(),ids.end(),expectedUnique[u])-ids.begin());
                        if (!(unique[u].flags&q::adaptive_lookup::InitialHit)) expectedMisses.push_back(expectedUnique[u]);
                    }
                    for (const auto u : expectedInverse) if (unique[u].flags&q::adaptive_lookup::InitialHit) ++hitOriginal;
                    EXPECT_EQ(misses,expectedMisses);
                    const auto after=observe.template operator()<h::Counters>(local.frameOffsets().counters);
                    EXPECT_EQ(after.read_original_ids-before->read_original_ids,profile==2?0u:ids.size());
                    EXPECT_EQ(after.local_hit_original_ids-before->local_hit_original_ids,profile==2?0u:hitOriginal);
                    EXPECT_EQ(after.consumed_original_ids-before->consumed_original_ids,profile==0 || profile==1?ids.size():0u);
                    EXPECT_EQ(after.promotions-before->promotions,native.promotions);
                    EXPECT_EQ(after.evictions-before->evictions,native.evictions);
                    EXPECT_EQ(after.future_on_time_native_rows,0u);EXPECT_EQ(after.future_late_native_rows,0u);
                    EXPECT_EQ(after.future_useful_native_rows,0u);EXPECT_EQ(after.future_wasted_native_rows,0u);
                    if (native.source_rows) {
                        const auto result=observe.template operator()<t::ResultHeader>(local.frameOffsets().result_header);
                        EXPECT_EQ(result.miss_rows,native.source_rows);
                        EXPECT_EQ(result.native_bytes,native.source_rows*c.file->row_bytes);
                        EXPECT_EQ(observe.template operator()<std::uint64_t>(local.frameOffsets().received_payload_bytes),result.native_bytes);
                        EXPECT_EQ(observe.template operator()<std::uint64_t>(local.frameOffsets().received_header_bytes),sizeof(t::ResultHeader));
                    }
                }
                EXPECT_EQ(storage->captureIdentity(),captureIdentity);
                if(timing)std::cout<<"LOCAL_HOT_NATIVE_TIMING {\"device\":\""<<device.toString()
                    <<"\",\"format\":\""<<c.label<<"\",\"profile\":"<<profile
                    <<",\"ordinal\":"<<sequence<<",\"original_ids\":"<<ids.size()
                    <<",\"physical_original_capacity\":"<<maximumOriginal
                    <<",\"source_rows\":"<<native.source_rows<<",\"native_us\":"<<timing->microseconds()
                    <<",\"scope\":\"complete_captured_local_graph_recorded_response\"}\n";
                return native;
            };
            h::TerminalObservation previousTerminal{};
            const unsigned requestLimit=timingMode?20:large?2:wide || c.label=="FP32"?20:2;
            for(unsigned request=1;request<=requestLimit;++request) {
                std::vector<int> healthy(sampling_math::kDeviceGenerationControlCount);healthy[sampling_math::kDeviceGenerationControlOk]=request%2?0:1;
                onWorker([&]{requireProvider(backend->hostToDeviceOnStream(controller.row(),healthy.data(),healthy.size()*4,device.ordinal,stream),"Original request controller initialization failed");});
                const auto ticket=service->arm(request);onWorker([&]{boundary->pooledKernels().admit(*boundary->pooledStorage(),ticket.generation,ticket.invocation,stream);
                launch(local_hot_recorded::Action::ForwardOriginalAdmission,args);code->enqueue(q::PooledImmutableLocalHotAction::ResetRequest,*storage,reset,stream);});join();
                if (large) {
                    // Exercise the exact single-CTA ceiling and dirty shorter replays
                    // through the original retained production graph.
                    const std::array extents = smallBoundary ? std::array{496u,512u,32u,16u,0u} :
                        std::array{512u,528u,1024u,16u,0u};
                    for (const auto live : extents) {
                        std::vector<std::uint64_t> input(live);
                        for (std::size_t i=0;i<input.size();++i) input[i]=c.keys[i%c.keys.size()];
                        run(0,input);
                    }
                    if(smallBoundary && !timingMode) {
                        // Coalescing is format-independent, but its live shape
                        // and key width are not. Sweep every supported token
                        // count in the small-query range on the actual retained
                        // kernels, including C0 and wide-key registrations.
                        for(unsigned tokens=0;tokens<=maximumOriginal/16;++tokens) {
                            std::vector<std::uint64_t> input(tokens*16);
                            for(std::size_t i=0;i<input.size();++i)
                                input[i]=c.keys[(i*3+tokens)%c.keys.size()];
                            run(0,input);
                        }
                    }
                }
                if (scanPressure) {
                    // Hot demand has ample real frequency before a finite
                    // disjoint scan crosses this same four-slot native cache.
                    std::vector<std::uint64_t> hot(16,c.keys.front());
                    for (unsigned repeat=0;repeat<20;++repeat) run(0,hot);
                    const auto beforeScan=observe.template operator()<h::Counters>(local.frameOffsets().counters);
                    std::vector<std::uint64_t> scan(32);
                    for (std::size_t i=0;i<scan.size();++i) scan[i]=c.keys[4+i%(c.keys.size()-4)];
                    for (unsigned repeat=0;repeat<20;++repeat) {
                        run(0,scan);
                        EXPECT_EQ(run(0,hot).source_rows,0u);
                    }
                    const auto afterScan=observe.template operator()<h::Counters>(local.frameOffsets().counters);
                    EXPECT_GT(afterScan.evictions,beforeScan.evictions);
                    EXPECT_GT(afterScan.promotions,beforeScan.promotions);
                }
                std::vector<std::uint64_t> warm(16,c.keys.front());run(0,warm);run(0,warm);
                std::vector<std::uint64_t> mixed(64,c.keys.front());for(unsigned i=0;i<mixed.size();++i)mixed[i]=c.keys[i%4];run(0,mixed);run(0,std::span<const std::uint64_t>{});
                const std::vector<std::uint64_t> futureIds=slots?warm:std::vector<std::uint64_t>{};
                if (!large || !slots) {
                    run(1,futureIds);run(2,futureIds);
                    if(nextWindow){
                        // All four original rows are warm. Derive two/four/zero
                        // token counts from the SAME immutable request window,
                        // without overwriting the current chunk count bank.
                        run(1,std::span<const std::uint64_t>(mixed.data(),32));
                        run(1,mixed);
                        run(2,std::span<const std::uint64_t>{});
                    }
                }
                else {
                    // Demand exceeding one block's lane count still uses the
                    // retained cooperative adoption kernel. Repeated IDs keep
                    // every future lookup local, isolating adoption from the
                    // separate original-source delivery fixture. Large-to-small
                    // and empty replays must not inspect poisoned capacity tails.
                    const std::array extents = smallBoundary ? std::array{496u,512u,16u,0u} :
                        std::array{528u,1024u,16u,0u};
                    for (const auto live : extents) {
                        const std::vector<std::uint64_t> demand(live,c.keys.front());
                        EXPECT_EQ(run(1,demand).source_rows,0u);
                        EXPECT_EQ(run(2,demand).source_rows,0u);
                    }
                }
                // Warm first row is guaranteed pinned; cancellation after actual
                // reservation/native copy drains both retained and Reserved pins.
                std::vector<std::uint64_t> partial(16,c.keys.front());partial.back()=c.keys.back();run(3,partial);
                onWorker([&]{boundary->pooledKernels().closeForReset(*boundary->pooledStorage(),controller,stream);});join();requireProvider(service->await(ticket),"Original source worker did not retire");
                // The original component aggregate has observed every actual
                // native graph and the source worker before terminal packing.
                const auto bytes=h::terminalObservationBytes(1,sampling_math::kDeviceGenerationControlCount);
                const auto offset=nativeReplayStagingBytes(c)-bytes;
                std::vector<std::byte> packed(bytes);std::array<int,sampling_math::kDeviceGenerationControlCount> originalControl{};
                onWorker([&]{
                    code->packTerminal(*storage,controller,1,q::PooledNativeTransferSpan(staging,offset,bytes),stream);
                    requireProvider(backend->deviceToHostOnStream(packed.data(),static_cast<const std::byte*>(staging->deviceData())+offset,bytes,device.ordinal,stream),"Actual local terminal packed observation failed");
                    requireProvider(backend->deviceToHostOnStream(originalControl.data(),controller.row(),sizeof(originalControl),device.ordinal,stream),"Original terminal controller byte oracle failed");
                });join();
                h::TerminalObservation actual;std::memcpy(&actual,packed.data(),sizeof(actual));
                EXPECT_EQ(std::memcmp(packed.data()+sizeof(actual),originalControl.data(),sizeof(originalControl)),0);
                const auto live=observe.template operator()<h::Counters>(local.frameOffsets().counters);
                const auto identity=observe.template operator()<h::FrameIdentity>(local.frameOffsets().identity);
                const auto frame=observe.template operator()<q::adaptive_lookup::FrameControl>(local.frameOffsets().control);
                const auto expected=h::snapshotTerminalCounters(live,identity,0,frame.status);
                EXPECT_EQ(std::memcmp(&actual,&expected,sizeof(actual)),0);
                h::TerminalObservation delta;ASSERT_TRUE(h::terminalObservationDelta(actual,previousTerminal,delta));
                EXPECT_GT(delta.values[0],0u);EXPECT_EQ(actual.status,h::TerminalObservationStatus::Complete);
                previousTerminal=actual;
            }
            if(nextWindow){
                // Actual malformed publication fails before opening another
                // source/frame; observe only the original terminal controller.
                std::vector<int> healthy(sampling_math::kDeviceGenerationControlCount);
                healthy[sampling_math::kDeviceGenerationControlOk]=1;
                const int invalidTotal=0,offset=2*(maximumOriginal/16+1);
                onWorker([&]{
                    requireProvider(backend->hostToDeviceOnStream(controller.row(),healthy.data(),healthy.size()*4,device.ordinal,stream),"Next-window guard controller publication failed");
                    requireProvider(backend->hostToDeviceOnStream(static_cast<int*>(windowRoot->gpu_data_ptr())+offset,&invalidTotal,4,device.ordinal,stream),"Malformed next-window publication failed");
                });
                requireProvider(RetainedParticipantGraphExecutableFixtureAccess::submit(graphs[4],++graphOrdinals[4]),"Next-window malformed graph submission failed");
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(graphs[4])){
                    requireProvider(std::chrono::steady_clock::now()<deadline,"Malformed next-window terminal deadline");std::this_thread::yield();
                }
                std::vector<int> observed(healthy.size());
                onWorker([&]{requireProvider(backend->deviceToHostOnStream(observed.data(),controller.row(),observed.size()*4,device.ordinal,stream),"Next-window error observation failed");});join();
                EXPECT_EQ(observed[sampling_math::kDeviceGenerationControlOk],0);
                EXPECT_EQ(observed[sampling_math::kDeviceGenerationControlErrorCode],int(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
                EXPECT_EQ(observe.template operator()<q::adaptive_lookup::FrameControl>(local.frameOffsets().control).held_pins,0u);
            }
            for (const auto bad : std::array<std::array<int,2>,3>{{{{2,0}},{{0,77}},{{1,77}}}}) {
                std::vector<int> state(sampling_math::kDeviceGenerationControlCount);
                state[sampling_math::kDeviceGenerationControlOk]=bad[0];state[sampling_math::kDeviceGenerationControlErrorCode]=bad[1];
                onWorker([&]{requireProvider(backend->hostToDeviceOnStream(controller.row(),state.data(),state.size()*4,device.ordinal,stream),"Recorded invalid controller upload failed");});
                requireProvider(RetainedParticipantGraphExecutableFixtureAccess::submit(graphs[4],++graphOrdinals[4]),"L1 guard replay submit failed");
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(graphs[4])){requireProvider(std::chrono::steady_clock::now()<deadline,"L1 guard terminal deadline");std::this_thread::yield();}
                const auto ready=std::atomic_ref(*static_cast<std::uint64_t*>(storage->readyRegion()->mutableHostData())).load(std::memory_order_acquire);
                EXPECT_EQ(ready,UINT64_MAX);
                onWorker([&]{requireProvider(backend->deviceToHostOnStream(state.data(),controller.row(),state.size()*4,device.ordinal,stream),"L1 original error observation failed");});join();EXPECT_EQ(state[sampling_math::kDeviceGenerationControlErrorCode],bad[1]);
            }
            {
                const auto bytes=h::terminalObservationBytes(1,sampling_math::kDeviceGenerationControlCount);
                const auto offset=nativeReplayStagingBytes(c)-bytes;std::vector<std::byte> packed(bytes);
                onWorker([&]{
                    code->packTerminal(*storage,controller,1,q::PooledNativeTransferSpan(staging,offset,bytes),stream);
                    requireProvider(backend->deviceToHostOnStream(packed.data(),static_cast<const std::byte*>(staging->deviceData())+offset,bytes,device.ordinal,stream),"Failed-controller terminal observation did not drain");
                });join();h::TerminalObservation failed;std::memcpy(&failed,packed.data(),sizeof(failed));
                EXPECT_EQ(failed.status,h::TerminalObservationStatus::Failed);h::TerminalObservation unused;
                EXPECT_FALSE(h::terminalObservationDelta(failed,previousTerminal,unused));
            }
            graphs.clear();join();
            const auto counters=observe.template operator()<h::Counters>(local.frameOffsets().counters);
            EXPECT_GT(counters.read_original_ids,0u);EXPECT_GT(counters.consumed_original_ids,0u);
            if(slots){EXPECT_GT(counters.local_hit_original_ids,0u);EXPECT_GT(counters.all_hit_wait_avoided_original_ids,0u);}else EXPECT_EQ(counters.local_hit_original_ids,0u);
            std::cout<<"LOCAL_HOT_NATIVE_KERNEL_REPLAY device="<<device.toString()<<" format="<<c.label<<" slots="<<slots<<" requests="<<requestLimit<<" ordinal="<<sequence
                <<" original_key_bits="<<(c.file->rows>UINT32_MAX?64:32)<<" physical_original_capacity="<<maximumOriginal
                <<" small_probe_boundary="<<smallBoundary<<" next_window_count="<<nextWindow<<" large_radix_scan="<<scanPressure<<" large_radix_wide_key="<<largeWideKey
                <<" l2_transport_proven=0 model_executed=0\n";
    }
    session->requirePayloadRetired();worker.reset();session->retire();session->requireExactRetired();
}
#define LOCAL_HOT_NATIVE_TESTS(BACKEND,DEVICE) \
TEST(PooledLocalHotNative,BACKEND##_AllFormatsC0){replayLocal(DEVICE,0,false);} \
TEST(PooledLocalHotNative,BACKEND##_AllFormatsPositive){replayLocal(DEVICE,4,false);} \
TEST(PooledLocalHotNative,BACKEND##_WideKeyPositive){replayLocal(DEVICE,4,true);} \
TEST(PooledLocalHotNative,BACKEND##_TerminalObservation){replayLocal(DEVICE,4,false);} \
TEST(PooledLocalHotNative,BACKEND##_LargeRadixScan){replayLocal(DEVICE,4,false,ReplayShape::LargeRadixScan);} \
TEST(PooledLocalHotNative,BACKEND##_LargeFutureAdoption){replayLocal(DEVICE,4,false,ReplayShape::LargeFutureAdoption);} \
TEST(PooledLocalHotNative,BACKEND##_SmallProbeBoundary){replayLocal(DEVICE,4,false,ReplayShape::SmallProbeBoundary);} \
TEST(PooledLocalHotNative,BACKEND##_SmallProbeBoundaryC0){replayLocal(DEVICE,0,false,ReplayShape::SmallProbeBoundary);} \
TEST(PooledLocalHotNative,BACKEND##_SmallProbeBoundaryWideKey){replayLocal(DEVICE,4,true,ReplayShape::SmallProbeBoundary);} \
TEST(PooledLocalHotNative,BACKEND##_SmallProbeBoundaryWideKeyC0){replayLocal(DEVICE,0,true,ReplayShape::SmallProbeBoundary);} \
TEST(PooledLocalHotNative,BACKEND##_SmallProbeTiming){replayLocal(DEVICE,4,false,ReplayShape::SmallProbeTiming);} \
TEST(PooledLocalHotNative,BACKEND##_LargeRadixWideKey){replayLocal(DEVICE,4,true,ReplayShape::LargeRadixWideKey);} \
TEST(PooledLocalHotNative,BACKEND##_PrefillNextWindowCount){replayLocal(DEVICE,4,false,ReplayShape::NextWindowCount);}
#ifdef HAVE_CUDA
LOCAL_HOT_NATIVE_TESTS(CUDA,DeviceId::cuda(0))
#endif
#ifdef HAVE_ROCM
LOCAL_HOT_NATIVE_TESTS(ROCm,DeviceId::rocm(0))
#endif
#undef LOCAL_HOT_NATIVE_TESTS
}}
