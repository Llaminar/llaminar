/**
 * @file Test__PooledImmutableNativeDevice.cpp
 * @brief Captured native cancellation and successful local original-byte/ACK/Ready/reader lifecycle replays.
 *
 * Original inputs are immutable diagnostic traces. Native owner/coordinator
 * phases produce actual decisions, grants and retirement; the separate test
 * source delivers admitted original bytes and ACKs on device. This exercises
 * the local diagnostic lifecycle. Production RAM/SSD source binding and
 * cross-device transport remain unqualified. Readback/timing are diagnostics.
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "loaders/ModelLoader.h"
#include "kernels/qwen4exp/AdaptiveImmutablePhysicalSlot.h"
#include "memory/BufferArena.h"
#include "fixtures/qwen4exp/PooledImmutableNativeSourceFixture.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=q::adaptive_lookup::pooled;
/** @brief Hard-fail the explicitly requested native path rather than silently skipping it. */
void requireNative(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
namespace f=pooled_source_fixture;
/** @brief Test-only original source/directed-observation node in the same real production capture DAG. */
class SourceFixtureStage final:public IComputeStage {
public:
    /** @brief Retain and seal original diagnostic source/staging roots before capture; execution observes no host values. */
    SourceFixtureStage(BufferArena& arena,f::Action action,f::Args args,std::vector<std::shared_ptr<TensorBase>> retained,
        std::shared_ptr<TensorBase> source):IComputeStage(args.owner.device),action_(action),args_(args),retained_(std::move(retained)),source_(std::move(source))
    {
        std::vector<TensorBase*> pointers;for(const auto& root:retained_)if(root)pointers.push_back(root.get());
        prepared_.emplace(device(),pointers);prepared_->seal();
        for(const auto& root:retained_)if(root)captured_.push_back(TransferEngine::instance().retainCapturedDeviceStorage(*root,device(),root->size_bytes()));
        const auto contract=bufferContract();
        for(const auto& binding:contract.allArenaReads())if(auto* root=dynamic_cast<TensorBase*>(arena.getTensor(binding.id)))reads_.push_back(root);
        for(const auto& bindings:{contract.outputs,contract.inouts})for(const auto& binding:bindings)if(auto* root=dynamic_cast<TensorBase*>(arena.getTensor(binding.id)))writes_.push_back(root);
    }
    /** @brief Bind only the exact capture stream; source/upload storage already exists and no mutable bytes are inspected. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Native fixture capture requires its exact stream/participant");
        prepared_->validate();setGPUStream(stream);return true;
    }
    /** @return Immutable metadata is prepared once before retained capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @brief Resolve exact original kernel resources; no allocation, synchronization or mutable host observation occurs. */
    bool execute(IDeviceContext* context)override{
        if(!context || context->deviceId()!=device())throw std::invalid_argument("Native diagnostic source has a foreign participant");
        prepared_->validate();auto execution=gpuExecution();for(auto* root:reads_)execution.requirePreparedInput(root);
        if(action_==f::Action::DeliverSource)execution.requirePreparedInput(source_.get());
        for(auto* root:writes_)execution.requirePreparedOutput(root);
#ifdef HAVE_CUDA
        if(device().is_cuda())f::launchCUDA(action_,args_,execution.nativeStream());
#endif
#ifdef HAVE_ROCM
        if(device().is_rocm())f::launchROCm(action_,args_,execution.nativeStream());
#endif
        for(auto* root:writes_)execution.publish(root);return true;
    }
    /** @return Complete fixture-only source geometry/address/runtime/producer operation identity. */
    std::uint64_t graphCaptureVariantSignature()const override{
        prepared_->validate();std::uint64_t h=1469598103934665603ULL;const auto mix=[&](std::uint64_t x){h=(h^x)*1099511628211ULL;};
        mix(1);mix(0x736f75726365ULL);mix(static_cast<unsigned>(action_));mix(prepared_->preparedIdentity());mix(args_.owner.identity);mix(args_.coordinator.identity);
        for(const auto value:{args_.offsets.header,args_.offsets.record_bytes,args_.offsets.payload_bytes,args_.offsets.records,args_.offsets.payload,args_.offsets.bytes})mix(value);
        mix(args_.source_rows);mix(args_.maximum_original);return h?h:1;
    }
    /** @return The fixture is explicitly a native phase, never a selectable source implementation. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    /** @return Only the requested original native backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return (device().is_cuda() && backend==ComputeBackendType::GPU_CUDA) || (device().is_rocm() && backend==ComputeBackendType::GPU_ROCM);}
    /** @return Exact actual source/directed-input dependencies and separately admitted output roles. */
    StageBufferContract bufferContract()const override{
        auto contract=StageBufferContract::build().addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_FRAME,"INT8");
        if(action_==f::Action::DeliverPins || action_==f::Action::DeliverFree || action_==f::Action::DeliverVictims || action_==f::Action::SourceHeader)
            contract.addInput(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
        if(action_==f::Action::DeliverPins || action_==f::Action::DeliverFree || action_==f::Action::DeliverVictims || action_==f::Action::DeliverSource)
            contract.addPreallocatedOutput(BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,"INT8");
        if(action_==f::Action::SourceHeader || action_==f::Action::DeliverSource)contract.addPreallocatedInOut(BufferId::QFN_NGRAM_LOOKUP_STATE,"INT8");
        if(action_==f::Action::DeliverSource){contract.addInput(BufferId::QFN_POOLED_OWNER_FRAME,"INT8");contract.addWeight(source_.get());}
        if(action_==f::Action::ReadRows)contract.addInput(BufferId::QFN_POOLED_OWNER_CACHE,"INT8").addPreallocatedOutput(BufferId::QFN_NGRAM_NATIVE_ROWS,"INT8");
        return contract;
    }
    /** @return No live device state is dumped through this fixture stage. */
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
private:
    f::Action action_;
    f::Args args_;
    std::vector<std::shared_ptr<TensorBase>> retained_;
    std::shared_ptr<TensorBase> source_;
    std::vector<TensorBase*> reads_,writes_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::vector<std::shared_ptr<const CapturedDeviceStorageLease>> captured_;
};
/** @brief The two component gates retain distinct coverage and qualification scopes. */
enum class RecordedTrace {PlanCancellation,SuccessfulSource};
/** @brief Actual one-GPU diagnostic family; independent placement is covered by device-free protocol replay. */
class NativeReplay {
public:
    /** @brief Declare exact PMA roots and one general executable before any materialization. */
    NativeReplay(DeviceId device,IWorkerGPUContext& worker,std::uint32_t maximum,std::uint32_t slots,RecordedTrace mode=RecordedTrace::PlanCancellation,n::OriginalIdWidth width=n::OriginalIdWidth::UInt64):device(device),worker(worker),
        backend(getBackendFor(device)),stream(ExplicitGPUStream(worker.defaultStream()).get()),maximum(maximum),slots(slots),mode(mode),originalWidth(width)
    {
        // A constructor exception must retire diagnostic uploads before host input backing unwinds.
        struct ConstructionRetirement {IBackend* backend;DeviceId device;void* stream;
            ~ConstructionRetirement(){if(!backend || !backend->synchronizeStream(stream,device.ordinal))std::terminate();}} retirement{backend,device,stream};
        const PhysicalMemoryAllocatorIdentity participant{0,device},host{0,DeviceId::cpu()};
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(
            ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,mode==RecordedTrace::SuccessfulSource?1024:0x100000100ULL,90),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{participant,ImmutableRowStorage::VRAM},0,slots,{}}},{participant},{},
            {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},slots?std::optional(participant):std::nullopt,16);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{participant},participant);
        const auto geometry=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},maximum);
        auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);
        auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        const PhysicalMemoryResource gpu{0,device,std::size_t{1}<<34,std::size_t{1}<<34},cpu{0,DeviceId::cpu(),std::size_t{1}<<34,std::size_t{1}<<34};
        PhysicalMemoryPlanBuilder bom;
        for(const auto& plan:{physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),geometry.policyBOM(gpu),geometry.mandatoryBOM(gpu),ownerPlan.mandatoryBOM(gpu),coordinatorPlan.mandatoryBOM(gpu)})
            for(const auto& resource:plan.resources())bom.add(resource);
        if(mode==RecordedTrace::SuccessfulSource){
            ioOffsets.payload=((ioOffsets.records+slots*std::max({sizeof(p::OwnerCommand),sizeof(n::Fill),sizeof(p::AllocationDecision)})+63)/64)*64;
            ioOffsets.bytes=((ioOffsets.payload+slots*90+63)/64)*64;
        }
        const auto idBytes=width==n::OriginalIdWidth::UInt32?4u:8u;
        const auto inputBytes=std::size_t{maximum}*idBytes+4+128+ioOffsets.bytes+(mode==RecordedTrace::SuccessfulSource?std::size_t{maximum}*90:0);
        const auto sourceBytes=mode==RecordedTrace::SuccessfulSource?std::size_t{1024}*90:0;
        graphClass=mode==RecordedTrace::SuccessfulSource && maximum>512?GPUGraphExecutableClass::General:GPUGraphExecutableClass::BoundedFlatHelper;
        const auto graphBytes=GPUGraphMemoryContract::reservationBytesPerExecutable(device,graphClass);
        bom.add(gpu,PhysicalMemoryOwner::ActivationArena,inputBytes);bom.add(cpu,PhysicalMemoryOwner::ActivationArena,inputBytes);
        bom.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,graphBytes);
        if(sourceBytes){bom.add(gpu,PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);bom.add(cpu,PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);}
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bom.build()),0);
        gpuInputs=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,inputBytes);
        hostInputs=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,inputBytes);
        executable=memory->claimNewAllocation(device,PhysicalMemoryOwner::NativeGraphExecutable,graphBytes);
        owner=q::PooledImmutableOwnerPreparedStorage::create(std::move(ownerPlan),memory,stream);
        coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(std::move(coordinatorPlan),memory,stream);
        ids=std::make_shared<INT32Tensor>(std::vector<std::size_t>{std::size_t{maximum}*(width==n::OriginalIdWidth::UInt32?1u:2u)});
        count=std::make_shared<INT32Tensor>(std::vector<std::size_t>{1});
        trace=std::make_shared<INT8Tensor>(std::vector<std::size_t>{128});
        if(sourceBytes){
            sourceGPU=memory->claimNewAllocation(device,PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);
            sourceHost=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);
            source=std::make_shared<INT8Tensor>(std::vector<std::size_t>{sourceBytes});
            io=std::make_shared<INT8Tensor>(std::vector<std::size_t>{ioOffsets.bytes});
            output=std::make_shared<INT8Tensor>(std::vector<std::size_t>{std::size_t{maximum}*90});
            auto* original=static_cast<std::byte*>(source->raw_mutable_data());for(std::size_t i=0;i<sourceBytes;++i)original[i]=std::byte((i*7+i/90)%256);
            for(std::size_t row=0;row<1024;++row)for(std::size_t block=0;block<5;++block){original[row*90+block*18]=std::byte{0};original[row*90+block*18+1]=std::byte{0x3c};}
            std::memset(io->raw_mutable_data(),0xa5,io->size_bytes());std::memset(output->raw_mutable_data(),0xa5,output->size_bytes());
            expectedOutput.assign(output->size_bytes(),std::byte{0xa5});
            for(const auto& root:{source,io,output})TransferEngine::prepareDeviceInput(root.get(),device,stream);
        }
        for(const auto& root:{ids,count,trace}){std::memset(root->raw_mutable_data(),0,root->size_bytes());TransferEngine::prepareDeviceInput(root.get(),device,stream);}
#ifdef HAVE_CUDA
        if(device.is_cuda())code=q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())code=q::createROCmPooledImmutableNativeKernels();
#endif
        requireNative(bool(code),"Requested pooled native backend has no compiled production bundle");
        code->initialize(*owner,stream);code->initialize(*coordinator,stream);
        bindPooledNativeRoots(arena,*owner);bindPooledNativeRoots(arena,*coordinator);
        requireNative(arena.registerExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids) &&
            arena.registerExternalBuffer(BufferId::PREFILL_CHUNK_GEOMETRY,count) &&
            arena.registerExternalBuffer(BufferId::QFN_NGRAM_LOOKUP_STATE,trace),"Native diagnostic original input registration failed");
        if(mode==RecordedTrace::SuccessfulSource){
            requireNative(arena.registerExternalBuffer(BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,io) && arena.registerExternalBuffer(BufferId::QFN_NGRAM_NATIVE_ROWS,output),"Native source diagnostic persistent role registration failed");
            const auto args=fixtureArgs();
#ifdef HAVE_CUDA
            if(device.is_cuda())f::launchCUDA(f::Action::PoisonCache,args,stream);
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())f::launchROCm(f::Action::PoisonCache,args,stream);
#endif
            TransferEngine::publishDeviceWrite(owner->roots().cache.get(),device,stream);
        }
        context=IDeviceContext::create(device,1);
        build();identity=coordinator->captureIdentity();
    }
    /** @brief Retire the diagnostic stream before graph/input/root/claim destruction, including failed setup. */
    ~NativeReplay(){if(backend && !backend->synchronizeStream(stream,device.ordinal))std::terminate();}
    /** @brief Seal one declared original input with its exact nominal role and byte interval. */
    static void input(q::PooledNativeOperandSpans& spans,q::PooledNativeOperand role,std::shared_ptr<TensorBase> root,BufferId id,std::size_t offset,std::size_t bytes){
        spans[static_cast<unsigned>(role)]=q::PooledNativeTensorSpan{std::move(root),id,offset,bytes};
    }
    /** @brief Add one actual checked owner stage; C0 never declares payload or record dummy roots. */
    void add(n::OwnerAction action,std::optional<std::size_t> header_bytes=std::nullopt){
        q::PooledNativeOperandSpans spans{};const auto cr=coordinator->roots();const auto oroot=owner->roots();
        input(spans,q::PooledNativeOperand::Invocation,cr.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator->plan().parameters().native.invocation,sizeof(n::Invocation));
        if(header_bytes){
            input(spans,q::PooledNativeOperand::Frontier,oroot.native,BufferId::QFN_POOLED_OWNER_NATIVE,owner->plan().parameters().native.header,*header_bytes);}
        if(mode==RecordedTrace::SuccessfulSource){const auto& plan=owner->plan().parameters();
            if(action==n::OwnerAction::Pin){input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,slots*sizeof(p::OwnerCommand));
                input(spans,q::PooledNativeOperand::RecordBytes,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.record_bytes,8);}
            if(action==n::OwnerAction::AcceptFree || action==n::OwnerAction::AcceptVictim){
                input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,slots*sizeof(p::AllocationDecision));
                input(spans,q::PooledNativeOperand::Frontier,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.header,action==n::OwnerAction::AcceptFree?sizeof(n::FreePlanHeader):sizeof(n::VictimPlanHeader));}
            if(action==n::OwnerAction::SealSource || action==n::OwnerAction::ApplySourceDelivery)
                input(spans,q::PooledNativeOperand::Frontier,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,64,sizeof(n::SourceFrontier));
            if(action==n::OwnerAction::ApplySourceDelivery){
                input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,slots*sizeof(n::Fill));
                input(spans,q::PooledNativeOperand::RecordBytes,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.record_bytes,8);
                input(spans,q::PooledNativeOperand::Payload,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.payload,slots*plan.row_bytes);
                input(spans,q::PooledNativeOperand::PayloadBytes,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.payload_bytes,8);}
        }
        auto operands=q::PooledImmutableOwnerOperands(action,owner->plan(),std::move(spans));
        stages.push_back(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,std::move(operands)}));
    }
    /** @brief Add one actual independent coordinator phase with only its required original producer roles. */
    void add(n::CoordinatorAction action){
        q::PooledNativeOperandSpans spans{};std::optional<n::OriginalRows> rows;
        if(action==n::CoordinatorAction::Begin){rows=n::OriginalRows{DeviceRowRange::deviceCounted(maximum,static_cast<const std::int32_t*>(count->gpu_data_ptr())),originalWidth};
            input(spans,q::PooledNativeOperand::Ids,ids,BufferId::QFN_NGRAM_ROW_IDS,0,maximum*(originalWidth==n::OriginalIdWidth::UInt32?4u:8u));
            input(spans,q::PooledNativeOperand::Admission,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,0,8);
            input(spans,q::PooledNativeOperand::OriginalCount,count,BufferId::PREFILL_CHUNK_GEOMETRY,0,4);}
        if(action==n::CoordinatorAction::SealPins || action==n::CoordinatorAction::Finish || action==n::CoordinatorAction::PlanFree || action==n::CoordinatorAction::CompleteFree || action==n::CoordinatorAction::CompleteVictims || action==n::CoordinatorAction::Publish)
            input(spans,q::PooledNativeOperand::Frontier,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,owner->plan().parameters().native.header,
                action==n::CoordinatorAction::SealPins?sizeof(p::PinHeader):action==n::CoordinatorAction::Finish?sizeof(p::RetiredHeader):
                action==n::CoordinatorAction::PlanFree?sizeof(n::Availability):action==n::CoordinatorAction::CompleteFree?sizeof(p::FreeAllocationReceipt):
                action==n::CoordinatorAction::CompleteVictims?sizeof(p::VictimAllocationReceipt):sizeof(p::ReadyHeader));
        if(action==n::CoordinatorAction::Cancel){input(spans,q::PooledNativeOperand::Source,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,64,sizeof(n::SourceFrontier));
            if(slots){const auto& plan=owner->plan().parameters();
                input(spans,q::PooledNativeOperand::Records,owner->roots().frame,BufferId::QFN_POOLED_OWNER_FRAME,plan.frame.release_receipts,slots*sizeof(p::ReleaseReceipt));
                input(spans,q::PooledNativeOperand::RecordBytes,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.progress+offsetof(n::OwnerProgress,release_bytes),8);}}
        if(action==n::CoordinatorAction::SealSource || action==n::CoordinatorAction::Publish)
            input(spans,q::PooledNativeOperand::Source,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,64,sizeof(n::SourceFrontier));
        if(action==n::CoordinatorAction::AcquirePins || action==n::CoordinatorAction::AcquireFree || action==n::CoordinatorAction::AcquireVictims){const auto& plan=owner->plan().parameters();
            input(spans,q::PooledNativeOperand::Records,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.replies,slots*sizeof(p::Grant));
            input(spans,q::PooledNativeOperand::RecordBytes,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.progress+offsetof(n::OwnerProgress,reply_bytes),8);}
        auto operands=q::PooledImmutableCoordinatorOperands(action,coordinator->plan(),rows,std::move(spans));
        stages.push_back(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,coordinator,std::move(operands)}));
    }
    /** @return Exact actual native addresses plus the separately admitted diagnostic source/IO roots. */
    f::Args fixtureArgs()const{
        const auto oroot=owner->roots(),croot=coordinator->roots();
        return {owner->plan().parameters(),coordinator->plan().parameters(),static_cast<std::byte*>(oroot.cache->gpu_data_ptr()),static_cast<std::byte*>(oroot.frame->gpu_data_ptr()),static_cast<std::byte*>(oroot.native->gpu_data_ptr()),
            static_cast<std::byte*>(croot.frame->gpu_data_ptr()),static_cast<std::byte*>(croot.native->gpu_data_ptr()),static_cast<std::byte*>(io->gpu_data_ptr()),static_cast<std::byte*>(trace->gpu_data_ptr()),
            static_cast<std::byte*>(output->gpu_data_ptr()),static_cast<const std::byte*>(source->gpu_data_ptr()),ioOffsets,1024,maximum};
    }
    /** @brief Add a diagnostic producer, distinct in code/resource identity from every production owner/coordinator operation. */
    void add(f::Action action){
        const auto oroot=owner->roots(),croot=coordinator->roots();
        stages.push_back(std::make_unique<SourceFixtureStage>(arena,action,fixtureArgs(),
            std::vector<std::shared_ptr<TensorBase>>{oroot.cache,oroot.frame,oroot.native,croot.cache,croot.frame,croot.native,io,trace,source,output},source));
    }
    /** @brief Retain one complete planned/unaccepted-cancellation graph; all retired headers are actual native producers. */
    void build(){
        add(n::CoordinatorAction::Begin);add(n::OwnerAction::Begin);
        if(mode==RecordedTrace::SuccessfulSource){add(n::CoordinatorAction::PinCommands);add(f::Action::DeliverPins);add(n::OwnerAction::Pin);}
        add(n::OwnerAction::SealPins);
        if(mode==RecordedTrace::SuccessfulSource)add(n::CoordinatorAction::AcquirePins);
        add(n::CoordinatorAction::SealPins);add(n::OwnerAction::AcquirePins,sizeof(p::PinHeader));
        add(n::OwnerAction::Availability);add(n::CoordinatorAction::PlanFree);
        if(mode==RecordedTrace::PlanCancellation){
            add(n::OwnerAction::CancelFree);add(n::CoordinatorAction::CompleteFree);add(n::OwnerAction::Release);add(n::OwnerAction::Retire);add(n::CoordinatorAction::Cancel);
        }else{
            add(f::Action::DeliverFree);add(n::OwnerAction::AcceptFree);add(n::OwnerAction::ExecuteFree);add(n::CoordinatorAction::AcquireFree);add(n::OwnerAction::FreeReceipt);
            add(n::CoordinatorAction::CompleteFree);add(n::OwnerAction::AcquireFree,sizeof(p::FreeAllocationReceipt));add(n::CoordinatorAction::PlanVictims);
            add(f::Action::DeliverVictims);add(n::OwnerAction::AcceptVictim);add(n::OwnerAction::ExecuteVictim);add(n::CoordinatorAction::AcquireVictims);add(n::OwnerAction::VictimReceipt);
            add(n::CoordinatorAction::CompleteVictims);add(f::Action::SourceHeader);add(n::CoordinatorAction::SealSource);add(n::OwnerAction::SealSource);
            add(f::Action::DeliverSource);add(n::OwnerAction::ApplySourceDelivery);add(n::OwnerAction::ValidateFills);add(n::OwnerAction::AcquirePublication,sizeof(p::PublishHeader));
            add(n::OwnerAction::MakeReady);add(n::OwnerAction::ReadyReceipt);add(n::CoordinatorAction::Publish);add(f::Action::ReadRows);add(n::OwnerAction::Release);add(n::OwnerAction::Retire);
        }
        add(n::CoordinatorAction::Finish);
        std::vector<std::string> order;
        for(std::size_t i=0;i<stages.size();++i){
            requireNative(stages[i]->prepareGraphLaunch(context.get(),stream),"Native diagnostic stage preparation failed");
            order.push_back("pooled_phase_"+std::to_string(i));compute.addNode(order.back(),std::move(stages[i]),device);
            if(i)compute.addDependency(order[i],order[i-1]);
        }
        stages.clear();executor.setArena(&arena);
        requireNative(executor.prepareGraphStorageForCapture(compute,context.get(),stream,"pooled native recorded frontier"),"Native diagnostic capture frontier failed");
        auto dependency=executor.planGraphCaptureDependencies(compute,order,device,stream,"pooled native exact phase dependencies");
        requireNative(bool(dependency),"Native diagnostic capture dependency plan unavailable");
        graph=worker.createGraphCapture(stream);requireNative(bool(graph),"Native diagnostic graph unavailable");
        ScopedBackendGraphCapture capture(worker,*graph,mode==RecordedTrace::PlanCancellation?"pooled native planned cancellation replay":"pooled native successful diagnostic source replay",dependency.get());requireNative(capture.begin(),"Native diagnostic capture begin failed");
        requireNative(executor.recordSetupGraphMaterialization(compute,context.get()),"Native diagnostic production stage recording failed");capture.finish();
        std::string detail;requireNative(graph->validateExecutableMemoryClass(graphClass,&detail),detail.c_str());
        requireNative(graph->instantiate(),"Native diagnostic captured graph instantiation failed");
    }
    /** @brief Upload a complete immutable recorded input outside capture; live state is never downloaded/re-uploaded. */
    void inputTrace(std::uint64_t sequence,std::span<const std::uint64_t> original){
        requireNative(original.size()<=maximum,"Native recorded original input exceeds admitted bound");
        if(originalWidth==n::OriginalIdWidth::UInt32){
            auto* data=static_cast<std::uint32_t*>(ids->raw_mutable_data());std::fill_n(data,maximum,~std::uint32_t{0});
            for(std::size_t i=0;i<original.size();++i){requireNative(original[i]<=~std::uint32_t{0},"Recorded UInt32 producer value does not fit");data[i]=static_cast<std::uint32_t>(original[i]);}
        }else{auto* data=ids->raw_mutable_data();std::memset(data,0xff,ids->size_bytes());
            if(!original.empty())std::memcpy(data,original.data(),original.size_bytes());}
        *static_cast<std::int32_t*>(count->raw_mutable_data())=static_cast<std::int32_t>(original.size());
        std::memset(trace->raw_mutable_data(),0,trace->size_bytes());*static_cast<std::uint64_t*>(trace->raw_mutable_data())=mode==RecordedTrace::SuccessfulSource && sequence>=3?2:1;
        *reinterpret_cast<n::SourceFrontier*>(static_cast<std::byte*>(trace->raw_mutable_data())+64)=
            {{701,mode==RecordedTrace::SuccessfulSource && sequence>=3?2u:1u,sequence},{},n::SourceDisposition::Unpublished,q::adaptive_lookup::Status::Success,n::Version};
        for(const auto& root:{ids,count,trace}){TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);}
    }
    /** @return Bounded diagnostic terminal state, never used to choose a production branch. */
    template<class T> std::vector<T> read(const std::byte* root,std::size_t offset,std::size_t count){
        std::vector<T> result(count);if(!count)return result;
        requireNative(backend->deviceToHostOnStream(result.data(),root+offset,count*sizeof(T),device.ordinal,stream),"Native diagnostic terminal read failed");
        requireNative(backend->synchronizeStream(stream,device.ordinal),"Native diagnostic terminal read did not retire");return result;
    }
    /** @brief Replay large/small/empty traces and byte-check actual stable coalescing/permutation and retired C0 state. */
    void replay(std::uint64_t sequence,std::span<const std::uint64_t> original){
        inputTrace(sequence,original);
        struct Timing {IBackend* backend;DeviceId device;void* stream;void* start=nullptr;void* stop=nullptr;
            ~Timing(){if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();if(stop)backend->destroyEvent(stop,device.ordinal);if(start)backend->destroyEvent(start,device.ordinal);}} timing{backend,device,stream};
        timing.start=backend->createTimingEvent(device.ordinal);timing.stop=backend->createTimingEvent(device.ordinal);
        requireNative(timing.start && timing.stop,"Native diagnostic timing events unavailable");
        requireNative(backend->recordEvent(timing.start,device.ordinal,stream),"Native diagnostic start event failed");
        requireNative(graph->launch(),"Native diagnostic retained replay failed");
        // The test's native graph-launch boundary publishes actual authority once, after submission.
        for(const auto& storage:{owner->roots(),coordinator->roots()})for(const auto& root:{storage.cache,storage.frame,storage.native})
            if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        if(mode==RecordedTrace::SuccessfulSource)for(const auto& root:{io,trace,output})TransferEngine::publishDeviceWrite(root.get(),device,stream);
        requireNative(backend->recordEvent(timing.stop,device.ordinal,stream) && backend->waitForEvent(timing.stop,device.ordinal),"Native diagnostic stop event failed");
        float elapsed=0;requireNative(backend->eventElapsedTimeMs(timing.start,timing.stop,device.ordinal,&elapsed),"Native diagnostic elapsed event unavailable");
        std::cout<<"POOLED_NATIVE_TIMING trace="<<(mode==RecordedTrace::PlanCancellation?"plan_cancel":"successful_source")<<" original_bits="<<(originalWidth==n::OriginalIdWidth::UInt32?32:64)<<" maximum_original="<<maximum<<" original="<<original.size()<<" owner_slots="<<slots<<" captured_transaction_us="<<elapsed*1000<<" input_upload_and_terminal_observation_timed=0 profile=unqualified\n";
        const auto root=coordinator->roots();const auto& plan=coordinator->plan().parameters();const auto* frame=static_cast<const std::byte*>(root.frame->gpu_data_ptr());
        const auto state=read<p::LookupFrame>(frame,plan.frame.control,1).front();
        ASSERT_EQ(state.phase,p::LookupPhase::Idle);ASSERT_EQ(state.status,q::adaptive_lookup::Status::Success);ASSERT_EQ(state.interval.sequence,sequence);
        std::vector<std::uint64_t> unique;std::vector<std::uint32_t> expected;
        for(const auto row:original){const auto found=std::find(unique.begin(),unique.end(),row);if(found==unique.end()){expected.push_back(unique.size());unique.push_back(row);}else expected.push_back(found-unique.begin());}
        ASSERT_EQ(state.unique_rows,unique.size());ASSERT_EQ(state.original_ids,original.size());
        const auto actual=read<p::CoordinatorUnique>(frame,plan.frame.unique,unique.size());
        for(std::size_t i=0;i<unique.size();++i)EXPECT_EQ(actual[i].row,unique[i]);
        const auto permutation=read<std::uint32_t>(frame,plan.frame.original_to_unique,original.size());EXPECT_EQ(permutation,expected);
        EXPECT_EQ(bool(owner->roots().cache),slots!=0);EXPECT_EQ(owner->plan().parameters().native.bytes,128u+64u*slots);EXPECT_EQ(coordinator->captureIdentity(),identity);
        const auto& ownerPlan=owner->plan().parameters();
        const auto ownerFrame=read<p::OwnerFrame>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.control,1).front();
        EXPECT_EQ(ownerFrame.phase,p::OwnerPhase::Idle);EXPECT_EQ(ownerFrame.status,mode==RecordedTrace::PlanCancellation?q::adaptive_lookup::Status::Cancelled:q::adaptive_lookup::Status::Success);
        EXPECT_EQ(ownerFrame.held_pins,0u);EXPECT_EQ(ownerFrame.reservations,0u);
        if(slots){const auto actualOwner=read<p::PhysicalOwnerControl>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.control,1).front();
            if(mode==RecordedTrace::PlanCancellation){EXPECT_EQ(actualOwner.filled,0u);EXPECT_EQ(actualOwner.next_slot_generation,0u);}
            else{EXPECT_EQ(actualOwner.filled,sequence==1?2u:4u);EXPECT_EQ(actualOwner.next_slot_generation,sequence==1?2u:5u);}}
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),owner->plan().parameters().cache.bytes+coordinator->plan().parameters().cache.bytes+(source?source->size_bytes():0));
        if(mode==RecordedTrace::SuccessfulSource){
            for(std::size_t i=0;i<original.size();++i)std::copy_n(static_cast<const std::byte*>(source->raw_data())+original[i]*90,90,expectedOutput.data()+i*90);
            const auto actualOutput=read<std::byte>(static_cast<const std::byte*>(output->gpu_data_ptr()),0,output->size_bytes());EXPECT_EQ(actualOutput,expectedOutput);
            const auto nativePayload=read<std::byte>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.payload,slots*90);
            for(std::size_t i=(sequence==1?2:3)*90;i<nativePayload.size();++i)EXPECT_EQ(nativePayload[i],std::byte{0xa5});
            const auto slotsState=read<q::adaptive_lookup::Slot32>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.slots,slots);
            const auto cachePayload=read<std::byte>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.payload,slots*ownerPlan.slot_stride);
            for(std::size_t slot=0;slot<slots;++slot){
                for(std::size_t i=90;i<ownerPlan.slot_stride;++i)EXPECT_EQ(cachePayload[slot*ownerPlan.slot_stride+i],std::byte{0xa5});
                if(q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot32>::state(slotsState[slot])==q::adaptive_lookup::SlotState::Ready)
                    EXPECT_EQ(std::memcmp(cachePayload.data()+slot*ownerPlan.slot_stride,static_cast<const std::byte*>(source->raw_data())+slotsState[slot].row*90,90),0);
            }
            if(sequence>=2){for(const auto& slot:slotsState)if(slot.row==900 && q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot32>::state(slot)==q::adaptive_lookup::SlotState::Ready){
                if(sequence==2)hotGeneration=slot.generation;else EXPECT_EQ(slot.generation,hotGeneration);}}
            if(sequence==2){EXPECT_EQ(state.free_planned,2u);EXPECT_EQ(state.victim_planned,1u);
                const auto grants=read<p::Grant>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.replies,3);
                const auto accepted=read<p::AllocationDecision>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.accepted_plan,3);
                const std::array<std::uint64_t,3> rows{700,701,900};
                for(unsigned i=0;i<3;++i){EXPECT_EQ(grants[i].row,rows[i]);EXPECT_EQ(grants[i].generation,i+3);EXPECT_EQ(grants[i].destination_token,i+1);EXPECT_EQ(accepted[i].row,rows[i]);
                    EXPECT_EQ(accepted[i].kind,i<2?p::CommandKind::ReserveFree:p::CommandKind::ReserveVictim);}
            }
            if(sequence>=3){EXPECT_EQ(state.free_planned,0u);EXPECT_EQ(state.victim_planned,0u);}
            const auto sourceFrontier=read<n::SourceFrontier>(static_cast<const std::byte*>(trace->gpu_data_ptr()),64,1).front();
            EXPECT_EQ(sourceFrontier.disposition,sequence<=2?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished);
            const auto records=read<std::uint64_t>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.record_bytes,1).front();
            const auto payloadBytes=read<std::uint64_t>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.payload_bytes,1).front();
            const auto produced=sequence==1?2u:sequence==2?3u:0u;EXPECT_EQ(records,produced*sizeof(n::Fill));EXPECT_EQ(payloadBytes,produced*90u);
            EXPECT_EQ(state.interval.admission,sequence>=3?2u:1u);
        }
    }
    DeviceId device;IWorkerGPUContext& worker;IBackend* backend;void* stream;std::uint32_t maximum,slots;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryAllocationLease gpuInputs,hostInputs,executable,sourceGPU,sourceHost;
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<TensorBase> ids,count,trace,source,io,output;
    std::vector<std::byte> expectedOutput;
    f::IO ioOffsets;
    RecordedTrace mode;
    n::OriginalIdWidth originalWidth;
    BufferArena arena;
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code;
    std::unique_ptr<IDeviceContext> context;
    std::vector<std::unique_ptr<IComputeStage>> stages;
    ComputeGraph compute;
    DeviceGraphExecutor executor;
    std::unique_ptr<IGPUGraphCapture> graph;
    std::uint64_t identity=0,hotGeneration=0;
    GPUGraphExecutableClass graphClass=GPUGraphExecutableClass::General;
};
/** @brief Exercise both cooperative Begin paths and original N>>U under actual native graph replay. */
void nativeReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto maximum:{32u,32768u})for(const auto slots:{0u,2u}){
        NativeReplay replay(device,worker,maximum,slots);std::vector<std::uint64_t> large(maximum);
        for(std::size_t i=0;i<large.size();++i)large[i]=(i%3==0)?0x100000007ULL:((i%3==1)?9:3);
        replay.replay(1,large);replay.replay(2,std::array<std::uint64_t,1>{0x100000007ULL});replay.replay(3,{});replay.replay(4,large);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;requireNative(replay.graph->inspectKernelNodes(nodes,&detail),detail.c_str());
        EXPECT_EQ(nodes.size(),(maximum<=512?16u:63u)+(slots?1u:0u));
        for(const auto& node:nodes){requireNative(node.valid(),"Native diagnostic kernel resource inventory incomplete");
            std::cout<<"POOLED_NATIVE_NODE name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
        std::cout<<"POOLED_NATIVE_REPLAY maximum_original="<<maximum<<" unique=3 owner_slots="<<slots<<" graph_nodes="<<nodes.size()<<" serving=unqualified source_transport=decoded_fixture\n";
    }});
}
/** @brief Actual unsigned UInt32 originals exercise both sides of the 512/513 dispatch boundary. */
void narrowBoundaryReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled UInt32 native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto maximum:{512u,513u})for(const auto slots:{0u,2u}){
        NativeReplay replay(device,worker,maximum,slots,RecordedTrace::PlanCancellation,n::OriginalIdWidth::UInt32);
        std::vector<std::uint64_t> original(maximum);for(std::size_t i=0;i<original.size();++i)original[i]=i%3==0?0xffff0007u:i%3==1?9u:3u;
        replay.replay(1,original);replay.replay(2,std::array<std::uint64_t,1>{0xffff0007u});replay.replay(3,{});replay.replay(4,original);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;requireNative(replay.graph->inspectKernelNodes(nodes,&detail),detail.c_str());
        EXPECT_EQ(nodes.size(),(maximum<=512?16u:63u)+(slots?1u:0u));
        for(const auto& node:nodes){requireNative(node.valid(),"Native UInt32 boundary resource inventory incomplete");
            std::cout<<"POOLED_NATIVE_UINT32_NODE maximum_original="<<maximum<<" original_bits=32 owner_slots="<<slots<<" name="<<node.name<<" registers="<<node.registers_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" local_bytes="<<node.local_memory_bytes_per_thread<<'\n';}
    }});
}
/** @brief Actual admitted original rows and ACKs drive successful free+victim, Ready, reader and retained-hit frames. */
void successfulNativeReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled successful native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{for(const auto maximum:{32u,32768u})for(const auto width:{n::OriginalIdWidth::UInt32,n::OriginalIdWidth::UInt64}){
        NativeReplay replay(device,worker,maximum,4,RecordedTrace::SuccessfulSource,width);
        std::vector<std::uint64_t> original(maximum,900);original[0]=700;original[1]=701;
        replay.replay(1,std::array<std::uint64_t,2>{101,102});replay.replay(2,original);replay.replay(3,std::array<std::uint64_t,1>{900});replay.replay(4,{});replay.replay(5,original);
        // Resource inspection follows all unprofiled timing records; source/diagnostic kernels have their own class.
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;requireNative(replay.graph->inspectKernelNodes(nodes,&detail),detail.c_str());
        EXPECT_EQ(nodes.size(),maximum<=512?47u:76u);
        for(const auto& node:nodes){requireNative(node.valid(),"Native successful kernel resource inventory incomplete");
            const std::array coordinatorNames{"coordinator","Original","radix","coalesceSmall","seedKeys","markGroups","scanGroups","materializeGroups","materializeUnique"};
            const auto scope=node.name.find("pooled_source_fixture")!=std::string::npos?"diagnostic_source":
                std::any_of(coordinatorNames.begin(),coordinatorNames.end(),[&](const auto* name){return node.name.find(name)!=std::string::npos;})?"coordinator":"owner";
            std::cout<<"POOLED_NATIVE_SOURCE_NODE maximum_original="<<maximum<<" original_bits="<<(width==n::OriginalIdWidth::UInt32?32:64)<<" scope="<<scope<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
        std::cout<<"POOLED_NATIVE_SUCCESS maximum_original="<<maximum<<" original_bits="<<(width==n::OriginalIdWidth::UInt32?32:64)<<" owner_slots=4 graph_nodes="<<nodes.size()<<" source=admitted_diagnostic_original_IQ4_NL production_RAM_SSD_transport=unqualified serving=unqualified\n";
    }});
}
}
#ifdef HAVE_CUDA
TEST(PooledImmutableNativeDevice,CUDAUInt32DispatchBoundaryReplay){narrowBoundaryReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,CUDACapturedSuccessfulSourceReplay){successfulNativeReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,CUDACapturedPlanCancelLargeSmallEmptyReplay){nativeReplay(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(PooledImmutableNativeDevice,ROCmUInt32DispatchBoundaryReplay){narrowBoundaryReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,ROCmCapturedSuccessfulSourceReplay){successfulNativeReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,ROCmCapturedPlanCancelLargeSmallEmptyReplay){nativeReplay(DeviceId::rocm(0));}
#endif
}
