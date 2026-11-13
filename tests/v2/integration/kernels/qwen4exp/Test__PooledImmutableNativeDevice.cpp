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
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventBudgetProjection.h"
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
#include "fixtures/qwen4exp/PooledImmutableScalarPolicyReplay.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <cmath>
#include <nlohmann/json.hpp>
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
        mix(args_.source_rows);mix(args_.maximum_original);
        mix(reinterpret_cast<std::uintptr_t>(args_.coordinator_cache));mix(args_.frequency_fault_cell);return h?h:1;
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
        if(action_==f::Action::CorruptFrequencyEpoch){
            contract.addInput(BufferId::QFN_POOLED_OWNER_NATIVE,"INT8");
            contract.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_POLICY,"INT8");
            contract.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
        }
        if(action_==f::Action::DeliverFailedFree){
            contract.addInput(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
            contract.addPreallocatedOutput(BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,"INT8");
        }
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
/** @brief Contract certification and opt-in transaction timing remain distinct diagnostic uses. */
enum class NativeReplayIntent {ContractReplay,TransactionTiming,FailedEpochReplay};
/** @brief Explicit input-history authority; fixed free/victim witnesses are never inferred from sequence alone. */
enum class NativeReplayHistory {OriginalFreeVictim,AuthoredPolicy};
/** @brief Raw event observation becomes valid only after the original scalar/terminal checks pass. */
struct NativeTransactionEventSample {
    float elapsed_ms=0;
    bool valid=false;
    std::uint32_t unique_rows=0,initial_hit_rows=0,source_rows=0,free_rows=0,victim_rows=0;
    std::uint64_t source_native_bytes=0;
};
/** @brief Actual one-GPU diagnostic family; independent placement is covered by device-free protocol replay. */
class NativeReplay {
public:
    /**
     * @brief Declare exact PMA roots and one general executable before any materialization.
     * @param device Original native participant.
     * @param session Existing prepared/admitted fixture owner.
     * @param maximum Original query bound retained by every replay.
     * @param slots Original physical owner capacity.
     * @param mode Recorded cancellation or successful-source transaction.
     * @param width Actual original input producer's UInt32/UInt64 width.
     * @param declarationOnly Declare the same BOM without native materialization.
     * @param policyProof Retain the independent serial-position policy oracle for terminal comparisons.
     * @param aging Exact immutable arithmetic policy period, identical in native and oracle declarations.
     * @param intent Explicit contract replay or separately selected opt-in transaction timing.
     * @param history Original fixed-history witness or independent scalar-authenticated arbitrary policy history.
     */
    NativeReplay(DeviceId device,PreparedNativeLookupFixture& session,std::uint32_t maximum,std::uint32_t slots,RecordedTrace mode=RecordedTrace::PlanCancellation,n::OriginalIdWidth width=n::OriginalIdWidth::UInt64,bool declarationOnly=false,bool policyProof=false,std::uint64_t aging=65536,NativeReplayIntent intent=NativeReplayIntent::ContractReplay,NativeReplayHistory history=NativeReplayHistory::OriginalFreeVictim):device(device),worker(*session.permit(device).worker()),
        backend(getBackendFor(device)),stream(ExplicitGPUStream(worker.defaultStream()).get()),maximum(maximum),slots(slots),mode(mode),originalWidth(width),intent(intent),history(history)
    {
        requireNative(history!=NativeReplayHistory::AuthoredPolicy || policyProof,
            "Authored policy history requires the independent original scalar oracle");
        // Ownership/admission stays on the original control thread. Only native
        // storage preparation and recording marshal onto the exact worker.
        if(!declarationOnly)
            graphOwner=RetainedParticipantGraphExecutableFixtureAccess::original(session.permit(device),stream);
        worker.submitAndWait([&]{
        // A constructor exception must retire diagnostic uploads before host input backing unwinds.
        struct ConstructionRetirement {IBackend* backend;DeviceId device;void* stream;
            ~ConstructionRetirement(){if(backend && !backend->synchronizeStream(stream,device.ordinal))std::terminate();}} retirement{declarationOnly?nullptr:backend,device,stream};
        const PhysicalMemoryAllocatorIdentity participant{0,device},host{0,DeviceId::cpu()};
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(
            ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,mode==RecordedTrace::SuccessfulSource?1024:0x100000100ULL,90),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{participant,ImmutableRowStorage::VRAM},0,slots,{}}},{participant},{},
            {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},slots?std::optional(participant):std::nullopt,16);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{participant},participant);
        const auto geometry=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{aging,1,2},maximum);
        auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);
        auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        const auto& gpu=session.resource(device);const auto& cpu=session.resource(DeviceId::cpu());
        PhysicalMemoryPlanBuilder bom;
        for(const auto& plan:{physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),geometry.policyBOM(gpu),geometry.mandatoryBOM(gpu),ownerPlan.mandatoryBOM(gpu),coordinatorPlan.mandatoryBOM(gpu)})
            for(const auto& resource:plan.resources())bom.add(resource);
        if(mode==RecordedTrace::SuccessfulSource){
            const auto replies=ownerPlan.parameters().native.reply_capacity;
            ioOffsets.payload=((ioOffsets.records+std::size_t{replies}*std::max({sizeof(p::OwnerCommand),sizeof(n::Fill),sizeof(p::AllocationDecision)})+63)/64)*64;
            ioOffsets.bytes=((ioOffsets.payload+std::size_t{replies}*ownerPlan.parameters().row_bytes+63)/64)*64;
        }
        const auto idBytes=width==n::OriginalIdWidth::UInt32?4u:8u;
        const auto inputBytes=std::size_t{maximum}*idBytes+4+128+ioOffsets.bytes+(mode==RecordedTrace::SuccessfulSource?std::size_t{maximum}*90:0);
        const auto sourceBytes=mode==RecordedTrace::SuccessfulSource?std::size_t{1024}*90:0;
        graphClass=mode==RecordedTrace::SuccessfulSource && maximum>512?GPUGraphExecutableClass::General:GPUGraphExecutableClass::BoundedFlatHelper;
        bom.add(gpu,PhysicalMemoryOwner::ActivationArena,inputBytes);bom.add(cpu,PhysicalMemoryOwner::ActivationArena,inputBytes);
        const auto general=RetainedParticipantGraphExecutable::generalBOM(cpu,gpu,1);
        for(const auto& row:general.resources())bom.add(row);
        for(const auto program:{q::PooledImmutableOwnerNativePlan::orderingProgram(physical,geometry,0),
                                q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,geometry)})
        {
            const auto ordering=NativeOrderingEventBudgetProjection::installed().bom(program,cpu,gpu);
            for(const auto& row:ordering.resources())bom.add(row);
        }
        if(sourceBytes){bom.add(gpu,PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);bom.add(cpu,PhysicalMemoryOwner::ImmutableTableStorage,sourceBytes);}
        declaredBOM=bom.build();
        if(declarationOnly)return;
        if(policyProof){
            requireNative(mode==RecordedTrace::SuccessfulSource ||
                (intent==NativeReplayIntent::TransactionTiming && slots==0),
                "Policy oracle requires successful source replay or explicit zero-capacity timing control");
            scalarPolicy=std::make_unique<pooled_scalar_replay::Pool>(physical,maximum,q::adaptive_lookup::Policy{aging,1,2});
            scalarPolicy->frequency_mode=pooled_scalar_replay::FrequencyObservationMode::ScalarPositions;
            scalarPolicy->original_queue_order=true; // Actual original queue body, independent of native batching.
        }
        if(intent==NativeReplayIntent::FailedEpochReplay){
            requireNative(policyProof && slots==4 && aging==17,"Fatal epoch replay has incompatible immutable geometry");
            pooled_scalar_replay::Pool probe(physical,maximum,q::adaptive_lookup::Policy{aging,1,2});
            probe.frequency_mode=pooled_scalar_replay::FrequencyObservationMode::ScalarPositions;
            probe.begin(std::array<std::uint64_t,1>{35});
            probe.all<n::OwnerAction::Availability>();
            const auto receipts=probe.headers<n::Availability>();probe.args.input.frontier=receipts.data();
            probe.run<n::CoordinatorAction::PlanFree>();
            const auto view=probe.view();
            for(std::uint32_t at=3*view.parameters.laneWidth();at<4*view.parameters.laneWidth();++at){
                if(view.frequency_counts[at])frequencyFaultCell=at;
            }
            requireNative(frequencyFaultCell!=q::adaptive_lookup::None,"Scalar fault author did not identify its exact live cell");
        }
        memory=session.memory();materialized=true;
        gpuInputs=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,inputBytes);
        hostInputs=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,inputBytes);
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
        });
    }
    /** @brief Retire the diagnostic stream before graph/input/root/claim destruction, including failed setup. */
    ~NativeReplay(){
        if(!materialized)return;
        try{
            worker.submitAndWait([&]{if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();});
            if(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(graphOwner))std::terminate();
            graphOwner.reset();
        }catch(...){std::terminate();}
    }
    /** @return Exact declaration consumed by the same original session before any payload creator. */
    const PhysicalMemoryPlan& declaration()const noexcept{return declaredBOM;}
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
            const auto replies=plan.native.reply_capacity;
            if(action==n::OwnerAction::Pin){input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,std::size_t{replies}*sizeof(p::OwnerCommand));
                input(spans,q::PooledNativeOperand::RecordBytes,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.record_bytes,8);}
            if(action==n::OwnerAction::AcceptFree || action==n::OwnerAction::AcceptVictim){
                input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,std::size_t{replies}*sizeof(p::AllocationDecision));
                input(spans,q::PooledNativeOperand::Frontier,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.header,action==n::OwnerAction::AcceptFree?sizeof(n::FreePlanHeader):sizeof(n::VictimPlanHeader));}
            if(action==n::OwnerAction::SealSource || action==n::OwnerAction::ApplySourceDelivery)
                input(spans,q::PooledNativeOperand::Frontier,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,64,sizeof(n::SourceFrontier));
            if(action==n::OwnerAction::ApplySourceDelivery){
                input(spans,q::PooledNativeOperand::Records,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.records,std::size_t{replies}*sizeof(n::Fill));
                input(spans,q::PooledNativeOperand::RecordBytes,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.record_bytes,8);
                input(spans,q::PooledNativeOperand::Payload,io,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,ioOffsets.payload,std::size_t{replies}*plan.row_bytes);
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
            const auto replies=coordinator->plan().parameters().native.reply_capacity;
            if(replies){const auto& plan=owner->plan().parameters();
                input(spans,q::PooledNativeOperand::Records,owner->roots().frame,BufferId::QFN_POOLED_OWNER_FRAME,plan.frame.release_receipts,std::size_t{replies}*sizeof(p::ReleaseReceipt));
                input(spans,q::PooledNativeOperand::RecordBytes,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.progress+offsetof(n::OwnerProgress,release_bytes),8);}}
        if(action==n::CoordinatorAction::SealSource || action==n::CoordinatorAction::Publish)
            input(spans,q::PooledNativeOperand::Source,trace,BufferId::QFN_NGRAM_LOOKUP_STATE,64,sizeof(n::SourceFrontier));
        if(action==n::CoordinatorAction::AcquirePins || action==n::CoordinatorAction::AcquireFree || action==n::CoordinatorAction::AcquireVictims){const auto& plan=owner->plan().parameters();
            input(spans,q::PooledNativeOperand::Records,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.replies,
                std::size_t{coordinator->plan().parameters().native.reply_capacity}*sizeof(p::Grant));
            input(spans,q::PooledNativeOperand::RecordBytes,owner->roots().native,BufferId::QFN_POOLED_OWNER_NATIVE,plan.native.progress+offsetof(n::OwnerProgress,reply_bytes),8);}
        auto operands=q::PooledImmutableCoordinatorOperands(action,coordinator->plan(),rows,std::move(spans));
        stages.push_back(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{action,coordinator,std::move(operands)}));
    }
    /** @return Exact actual native addresses plus the separately admitted diagnostic source/IO roots. */
    f::Args fixtureArgs()const{
        const auto oroot=owner->roots(),croot=coordinator->roots();
        f::Args args{owner->plan().parameters(),coordinator->plan().parameters(),static_cast<std::byte*>(oroot.cache->gpu_data_ptr()),static_cast<std::byte*>(oroot.frame->gpu_data_ptr()),static_cast<std::byte*>(oroot.native->gpu_data_ptr()),
            static_cast<std::byte*>(croot.frame->gpu_data_ptr()),static_cast<std::byte*>(croot.native->gpu_data_ptr()),static_cast<std::byte*>(io->gpu_data_ptr()),static_cast<std::byte*>(trace->gpu_data_ptr()),
            static_cast<std::byte*>(output->gpu_data_ptr()),static_cast<const std::byte*>(source->gpu_data_ptr()),ioOffsets,1024,maximum};
        args.coordinator_cache=croot.cache?static_cast<std::byte*>(croot.cache->gpu_data_ptr()):nullptr;
        args.frequency_fault_cell=frequencyFaultCell;return args;
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
        add(n::OwnerAction::Availability);
        if(intent==NativeReplayIntent::FailedEpochReplay)add(f::Action::CorruptFrequencyEpoch);
        add(n::CoordinatorAction::PlanFree);
        if(intent==NativeReplayIntent::FailedEpochReplay){
            add(f::Action::DeliverFailedFree);add(n::OwnerAction::AcceptFree);
            add(n::CoordinatorAction::Cancel); // Ordinary cancellation must retain the first fatal error.
        }else if(mode==RecordedTrace::PlanCancellation){
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
        graph=&RetainedParticipantGraphExecutableFixtureAccess::capture(graphOwner);requireNative(bool(graph),"Native diagnostic graph unavailable");
        ScopedBackendGraphCapture capture(worker,*graph,mode==RecordedTrace::PlanCancellation?"pooled native planned cancellation replay":"pooled native successful diagnostic source replay",dependency.get());requireNative(capture.begin(),"Native diagnostic capture begin failed");
        requireNative(executor.recordSetupGraphMaterialization(compute,context.get()),"Native diagnostic production stage recording failed");capture.finish();
        std::string detail;
        const bool validClass=graph->validateExecutableMemoryClass(graphClass,&detail);
        requireNative(validClass,detail.c_str());
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
    /**
     * @brief Replay one original captured transaction and validate its terminal output/state outside event timing.
     * @param sequence Original monotonically increasing lookup sequence.
     * @param original Exact live ordered inputs, including partial/empty work.
     * @param sample Optional raw transaction event output, valid only after scalar/terminal checks pass.
     * @note Opt-in timing validates/advances the independent scalar oracle before recording its start event.
     */
    void replay(std::uint64_t sequence,std::span<const std::uint64_t> original,NativeTransactionEventSample* sample=nullptr){
        if(sample)sample->valid=false;
        const bool originalHistory=mode==RecordedTrace::SuccessfulSource && history==NativeReplayHistory::OriginalFreeVictim;
        if(originalHistory && sequence==1)requireNative(original.size()==2 && original[0]==101 && original[1]==102,
            "Original free/victim witness lost its exact first input");
        if(originalHistory && sequence==2)requireNative(original.size()>=3 && original[0]==700 && original[1]==701 &&
            std::all_of(original.begin()+2,original.end(),[](auto row){return row==900;}),
            "Original free/victim witness lost its exact second input");
        if(intent==NativeReplayIntent::TransactionTiming){
            requireNative(original.size()<=maximum,"Timing input count exceeds its admitted original bound");
            const auto sourceRows=coordinator->plan().parameters().policy.sourceRows();
            for(const auto row:original)requireNative(row<sourceRows,"Timing input row exceeds its declared original source extent");
            requireNative(bool(scalarPolicy),"Opt-in timing requires its independent original scalar oracle");
            if(mode==RecordedTrace::SuccessfulSource){
                scalarPolicy->complete(original,sequence>=3?2u:1u);
                const auto expected=scalarPolicy->view();
                const auto& ownerArgs=scalarPolicy->owners.front()->args;
                const auto pins=n::at<p::OwnerFrame>(ownerArgs.frame,ownerArgs.plan.frame.control)->initial_pin_count;
                requireNative(pins+expected.frame->free_planned+expected.frame->victim_planned==expected.frame->unique_rows,
                    "Authored timing history requires unsupported source-only payload; refusing before event timing");
            }else{
                advanceZeroCapacityScalarPolicy(original);
            }
        }
        inputTrace(sequence,original);
        struct Timing {IBackend* backend;DeviceId device;void* stream;void* start=nullptr;void* stop=nullptr;
            ~Timing(){if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();if(stop)backend->destroyEvent(stop,device.ordinal);if(start)backend->destroyEvent(start,device.ordinal);}} timing{backend,device,stream};
        timing.start=backend->createTimingEvent(device.ordinal);timing.stop=backend->createTimingEvent(device.ordinal);
        requireNative(timing.start && timing.stop,"Native diagnostic timing events unavailable");
        requireNative(backend->recordEvent(timing.start,device.ordinal,stream),"Native diagnostic start event failed");
        requireNative(RetainedParticipantGraphExecutableFixtureAccess::submit(graphOwner,++submissionOrdinal),"Native diagnostic retained replay failed");
        // The test's native graph-launch boundary publishes actual authority once, after submission.
        for(const auto& storage:{owner->roots(),coordinator->roots()})for(const auto& root:{storage.cache,storage.frame,storage.native})
            if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        if(mode==RecordedTrace::SuccessfulSource)for(const auto& root:{io,trace,output})TransferEngine::publishDeviceWrite(root.get(),device,stream);
        requireNative(backend->recordEvent(timing.stop,device.ordinal,stream) && backend->waitForEvent(timing.stop,device.ordinal),"Native diagnostic stop event failed");
        float elapsed=0;requireNative(backend->eventElapsedTimeMs(timing.start,timing.stop,device.ordinal,&elapsed),"Native diagnostic elapsed event unavailable");
        if(intent==NativeReplayIntent::ContractReplay)std::cout<<"POOLED_NATIVE_TIMING trace="<<(mode==RecordedTrace::PlanCancellation?"plan_cancel":"successful_source")<<" original_bits="<<(originalWidth==n::OriginalIdWidth::UInt32?32:64)<<" maximum_original="<<maximum<<" original="<<original.size()<<" owner_slots="<<slots<<" captured_transaction_us="<<elapsed*1000<<" input_upload_and_terminal_observation_timed=0 profile=unqualified\n";
        if(scalarPolicy){
            if(intent==NativeReplayIntent::ContractReplay)scalarPolicy->complete(original,sequence>=3?2u:1u);
            compareScalarPolicy();
        }
        const auto root=coordinator->roots();const auto& plan=coordinator->plan().parameters();const auto* frame=static_cast<const std::byte*>(root.frame->gpu_data_ptr());
        const auto state=read<p::LookupFrame>(frame,plan.frame.control,1).front();
        ASSERT_EQ(state.phase,p::LookupPhase::Idle);ASSERT_EQ(state.status,q::adaptive_lookup::Status::Success);ASSERT_EQ(state.interval.sequence,sequence);
        std::vector<std::uint64_t> unique;std::vector<std::uint32_t> expected;
        for(const auto row:original){const auto found=std::find(unique.begin(),unique.end(),row);if(found==unique.end()){expected.push_back(unique.size());unique.push_back(row);}else expected.push_back(found-unique.begin());}
        ASSERT_EQ(state.unique_rows,unique.size());ASSERT_EQ(state.original_ids,original.size());
        const auto actual=read<p::CoordinatorUnique>(frame,plan.frame.unique,unique.size());
        for(std::size_t i=0;i<unique.size();++i)EXPECT_EQ(actual[i].row,unique[i]);
        const auto permutation=read<std::uint32_t>(frame,plan.frame.original_to_unique,original.size());EXPECT_EQ(permutation,expected);
        const auto& ownerPlan=owner->plan().parameters();
        EXPECT_EQ(ownerPlan.owner.capacity(),slots);EXPECT_EQ(bool(owner->roots().cache),ownerPlan.owner.capacity()!=0);
        EXPECT_EQ(owner->roots().native->size_bytes(),ownerPlan.native.bytes);EXPECT_EQ(coordinator->captureIdentity(),identity);
        const auto ownerFrame=read<p::OwnerFrame>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.control,1).front();
        EXPECT_EQ(ownerFrame.phase,p::OwnerPhase::Idle);EXPECT_EQ(ownerFrame.status,mode==RecordedTrace::PlanCancellation?q::adaptive_lookup::Status::Cancelled:q::adaptive_lookup::Status::Success);
        EXPECT_EQ(ownerFrame.held_pins,0u);EXPECT_EQ(ownerFrame.reservations,0u);
        if(slots){const auto actualOwner=read<p::PhysicalOwnerControl>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.control,1).front();
            if(mode==RecordedTrace::PlanCancellation){EXPECT_EQ(actualOwner.filled,0u);EXPECT_EQ(actualOwner.next_slot_generation,0u);}
            else if(scalarPolicy){
                const auto& expectedOwner=scalarPolicy->owners.front()->args;
                const auto& expected=*n::at<p::PhysicalOwnerControl>(expectedOwner.cache,expectedOwner.plan.cache.control);
                EXPECT_EQ(actualOwner.filled,expected.filled);
                EXPECT_EQ(actualOwner.next_slot_generation,expected.next_slot_generation);
            }else{EXPECT_EQ(actualOwner.filled,sequence==1?2u:4u);EXPECT_EQ(actualOwner.next_slot_generation,sequence==1?2u:5u);}}
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation),owner->plan().parameters().cache.bytes+coordinator->plan().parameters().cache.bytes+(source?source->size_bytes():0));
        if(mode==RecordedTrace::SuccessfulSource){
            const auto expectedProduced=scalarPolicy?scalarPolicy->view().frame->free_planned+scalarPolicy->view().frame->victim_planned:
                sequence==1?2u:sequence==2?3u:0u;
            requireNative(expectedProduced<=ownerPlan.native.reply_capacity,"Scalar source frontier exceeds original reply capacity");
            // IO is poisoned once, not per replay. Retain the actual scalar-authored high-water prefix;
            // previous larger source replies may remain in its inactive tail during later all-hit frames.
            payloadWrittenRows=std::max(payloadWrittenRows,expectedProduced);
            for(std::size_t i=0;i<original.size();++i)std::copy_n(static_cast<const std::byte*>(source->raw_data())+original[i]*90,90,expectedOutput.data()+i*90);
            const auto actualOutput=read<std::byte>(static_cast<const std::byte*>(output->gpu_data_ptr()),0,output->size_bytes());EXPECT_EQ(actualOutput,expectedOutput);
            const auto nativePayload=read<std::byte>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.payload,std::size_t{ownerPlan.native.reply_capacity}*ownerPlan.row_bytes);
            for(std::size_t i=std::size_t{payloadWrittenRows}*ownerPlan.row_bytes;i<nativePayload.size();++i)EXPECT_EQ(nativePayload[i],std::byte{0xa5});
            const auto slotsState=read<q::adaptive_lookup::Slot32>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.slots,ownerPlan.owner.capacity());
            const auto cachePayload=read<std::byte>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr()),ownerPlan.cache.payload,std::size_t{ownerPlan.owner.capacity()}*ownerPlan.slot_stride);
            for(std::size_t slot=0;slot<ownerPlan.owner.capacity();++slot){
                for(std::size_t i=90;i<ownerPlan.slot_stride;++i)EXPECT_EQ(cachePayload[slot*ownerPlan.slot_stride+i],std::byte{0xa5});
                if(q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot32>::state(slotsState[slot])==q::adaptive_lookup::SlotState::Ready)
                    EXPECT_EQ(std::memcmp(cachePayload.data()+slot*ownerPlan.slot_stride,static_cast<const std::byte*>(source->raw_data())+slotsState[slot].row*90,90),0);
            }
            if(originalHistory && sequence>=2){for(const auto& slot:slotsState)if(slot.row==900 && q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot32>::state(slot)==q::adaptive_lookup::SlotState::Ready){
                if(!hotGeneration)hotGeneration=slot.generation;else EXPECT_EQ(slot.generation,hotGeneration);}}
            if(originalHistory && sequence==2){EXPECT_EQ(state.free_planned,2u);EXPECT_EQ(state.victim_planned,1u);
                const auto grants=read<p::Grant>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.replies,3);
                const auto accepted=read<p::AllocationDecision>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr()),ownerPlan.frame.accepted_plan,3);
                const std::array<std::uint64_t,3> rows{700,701,900};
                for(unsigned i=0;i<3;++i){EXPECT_EQ(grants[i].row,rows[i]);EXPECT_EQ(grants[i].generation,i+3);EXPECT_EQ(grants[i].destination_token,i+1);EXPECT_EQ(accepted[i].row,rows[i]);
                    EXPECT_EQ(accepted[i].kind,i<2?p::CommandKind::ReserveFree:p::CommandKind::ReserveVictim);}
            }
            if(originalHistory && sequence>=3 && !scalarPolicy){EXPECT_EQ(state.free_planned,0u);EXPECT_EQ(state.victim_planned,0u);}
            const auto sourceFrontier=read<n::SourceFrontier>(static_cast<const std::byte*>(trace->gpu_data_ptr()),64,1).front();
            EXPECT_EQ(sourceFrontier.disposition,expectedProduced?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished);
            const auto records=read<std::uint64_t>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.record_bytes,1).front();
            const auto payloadBytes=read<std::uint64_t>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.payload_bytes,1).front();
            const auto produced=expectedProduced;EXPECT_EQ(records,produced*sizeof(n::Fill));EXPECT_EQ(payloadBytes,produced*90u);
            EXPECT_EQ(state.interval.admission,sequence>=3?2u:1u);
        }
        if(sample){
            sample->elapsed_ms=elapsed;
            sample->unique_rows=state.unique_rows;
            sample->initial_hit_rows=ownerFrame.initial_pin_count;
            sample->free_rows=state.free_planned;
            sample->victim_rows=state.victim_planned;
            sample->source_rows=mode==RecordedTrace::SuccessfulSource?state.free_planned+state.victim_planned:0;
            sample->source_native_bytes=std::uint64_t{sample->source_rows}*ownerPlan.row_bytes;
            sample->valid=std::isfinite(elapsed) && elapsed>=0 && !::testing::Test::HasFailure();
        }
    }
    /** @brief Prove actual captured failure headers, sticky poison and rejection of retained failed reuse.
     * @note Authored late corruption has no rollback contract. Terminal reads are outside captured execution.
     */
    void replayFailedEpoch(){
        requireNative(intent==NativeReplayIntent::FailedEpochReplay && scalarPolicy,"Missing fatal-epoch scalar/native owner");
        const std::array<std::uint64_t,3> original{31,35,31};
        auto& oracle=*scalarPolicy;
        oracle.begin(original);oracle.all<n::OwnerAction::Availability>();
        const auto receipts=oracle.headers<n::Availability>();oracle.args.input.frontier=receipts.data();
        oracle.view().frequency_epochs[frequencyFaultCell]=UINT32_MAX;
        EXPECT_THROW(oracle.run<n::CoordinatorAction::PlanFree>(),std::logic_error);
        EXPECT_EQ(oracle.view().frame->status,q::adaptive_lookup::Status::Invariant);
        ASSERT_EQ(oracle.view().policy->logical_clock,2u); // Row31 really completed before the bad row35 cell.
        const auto& plan=coordinator->plan().parameters();const auto cr=coordinator->roots();
        const auto& ownerPlan=owner->plan().parameters();const auto oroot=owner->roots();
        for(std::uint64_t attempt=1;attempt<=2;++attempt){
            inputTrace(attempt,original);
            requireNative(RetainedParticipantGraphExecutableFixtureAccess::submit(graphOwner,++submissionOrdinal),"Fatal-epoch retained native replay failed");
            for(const auto& storage:{oroot,cr})for(const auto& root:{storage.cache,storage.frame,storage.native}){
                if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
            }
            TransferEngine::publishDeviceWrite(io.get(),device,stream);
            const auto state=read<p::LookupFrame>(static_cast<const std::byte*>(cr.frame->gpu_data_ptr()),plan.frame.control,1).front();
            EXPECT_EQ(state.status,q::adaptive_lookup::Status::Invariant);EXPECT_EQ(state.phase,p::LookupPhase::Availability);
            EXPECT_EQ(state.ordinal,1u);EXPECT_EQ(state.interval.sequence,1u);EXPECT_FALSE(state.observed);
            EXPECT_EQ(state.free_planned,0u);EXPECT_EQ(state.outstanding,0u);
            const auto policy=read<p::CoordinatorPolicyControl>(static_cast<const std::byte*>(cr.cache->gpu_data_ptr()),plan.cache.control,1).front();
            EXPECT_NE(policy.flags&q::adaptive_lookup::OwnerFailed,0u);
            const auto counts=read<std::uint16_t>(static_cast<const std::byte*>(cr.cache->gpu_data_ptr()),plan.cache.counts,3*plan.policy.laneWidth());
            EXPECT_TRUE(std::any_of(counts.begin(),counts.end(),[](auto value){return value!=0;}));
            const auto headers=read<n::FreePlanHeader>(static_cast<const std::byte*>(cr.native->gpu_data_ptr()),plan.native.free_headers,plan.policy.participants());
            const auto slices=read<n::DirectedSlice>(static_cast<const std::byte*>(cr.native->gpu_data_ptr()),plan.native.free_slices,plan.policy.participants());
            for(unsigned ownerIndex=0;ownerIndex<headers.size();++ownerIndex){
                EXPECT_EQ(headers[ownerIndex].version,n::Version);EXPECT_EQ(headers[ownerIndex].status,q::adaptive_lookup::Status::Invariant);
                const auto expected=receipts[ownerIndex].lease;
                EXPECT_EQ(headers[ownerIndex].lease.owner,expected.owner);
                EXPECT_EQ(headers[ownerIndex].lease.free_slots,expected.free_slots);
                EXPECT_EQ(headers[ownerIndex].lease.interval.table,expected.interval.table);
                EXPECT_EQ(headers[ownerIndex].lease.interval.admission,expected.interval.admission);
                EXPECT_EQ(headers[ownerIndex].lease.interval.sequence,expected.interval.sequence);
                EXPECT_EQ(headers[ownerIndex].decision_bytes,0u);EXPECT_EQ(slices[ownerIndex].live_bytes,0u);
                EXPECT_EQ(slices[ownerIndex].byte_offset,plan.frame.decisions);
            }
            const auto delivered=read<n::FreePlanHeader>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.header,1).front();
            EXPECT_EQ(delivered.status,q::adaptive_lookup::Status::Invariant);EXPECT_EQ(delivered.decision_bytes,0u);
            EXPECT_EQ(delivered.lease.interval.sequence,1u);EXPECT_EQ(delivered.version,n::Version);
            EXPECT_EQ(read<std::uint64_t>(static_cast<const std::byte*>(io->gpu_data_ptr()),ioOffsets.record_bytes,1).front(),0u);
            const auto ownerFrame=read<p::OwnerFrame>(static_cast<const std::byte*>(oroot.frame->gpu_data_ptr()),ownerPlan.frame.control,1).front();
            EXPECT_EQ(ownerFrame.status,q::adaptive_lookup::Status::Invariant);
            EXPECT_EQ(ownerFrame.held_pins,0u);EXPECT_EQ(ownerFrame.reservations,0u);
            const auto allocation=read<p::AllocationFrame>(static_cast<const std::byte*>(oroot.frame->gpu_data_ptr()),ownerPlan.frame.allocation_control,1).front();
            EXPECT_EQ(allocation.phase,p::AllocationPhase::Leased);EXPECT_EQ(allocation.count,0u);
            const auto physical=read<p::PhysicalOwnerControl>(static_cast<const std::byte*>(oroot.cache->gpu_data_ptr()),ownerPlan.cache.control,1).front();
            EXPECT_EQ(physical.filled,0u);EXPECT_EQ(physical.next_slot_generation,0u);
            EXPECT_EQ(identity,coordinator->captureIdentity());
        }
        EXPECT_THROW(code->initialize(*coordinator,stream),std::logic_error);
        std::cout<<"POOLED_NATIVE_FAILED_EPOCH actual_algorithm="<<((code->codeIdentity()>>32)&0xffU)
            <<" original_scalar_late_clock=2 fresh_compact_error_headers=1 failed_replay_rejected=1 rollback_required=0\n";
    }
    /**
     * @brief Replay the exact zero-capacity cancellation transaction through original scalar operations.
     * @param original Same recorded input order submitted to the native transaction.
     * @note This is a no-source protocol control, never a cold RAM/SSD or cache-hit timing claim.
     */
    void advanceZeroCapacityScalarPolicy(std::span<const std::uint64_t> original){
        requireNative(slots==0 && mode==RecordedTrace::PlanCancellation,
            "Zero-capacity scalar timing control received a foreign source/owner mode");
        auto& oracle=*scalarPolicy;
        oracle.begin(original,1);
        oracle.all<n::OwnerAction::Availability>();
        const auto available=oracle.headers<n::Availability>();
        oracle.args.input.frontier=available.data();
        oracle.run<n::CoordinatorAction::PlanFree>();
        oracle.args.input={};
        oracle.all<n::OwnerAction::CancelFree>();
        const auto cancelled=oracle.headers<p::FreeAllocationReceipt>();
        oracle.args.input.frontier=cancelled.data();
        oracle.run<n::CoordinatorAction::CompleteFree>();
        oracle.args.input={};
        oracle.all<n::OwnerAction::Release>();
        oracle.all<n::OwnerAction::Retire>();
        const n::SourceFrontier source{oracle.view().frame->interval,{},n::SourceDisposition::Unpublished,
            q::adaptive_lookup::Status::Success,n::Version};
        oracle.args.input.source=&source;
        oracle.run<n::CoordinatorAction::Cancel>();
        oracle.args.input={};
        const auto retired=oracle.headers<p::RetiredHeader>();
        oracle.args.input.frontier=retired.data();
        oracle.run<n::CoordinatorAction::Finish>();
        oracle.args.input={};
    }
    /**
     * @brief Compare complete terminal policy cells and real accepted free/victim plans with independent scalar replay.
     *
     * All reads occur after the retained transaction terminal, outside its timed
     * interval. They never select a branch, repair device state or upload the
     * oracle. Directory/radix/queue fields, including unoccupied canonical
     * metadata, are test outputs needed for exact later-victim policy proof.
     */
    void compareScalarPolicy(){
        auto expected=scalarPolicy->view();
        const auto& plan=coordinator->plan().parameters();
        const auto* frame=static_cast<const std::byte*>(coordinator->roots().frame->gpu_data_ptr());
        if(!plan.policy.entries()){
            EXPECT_EQ(expected.policy,nullptr);
            const auto actual=read<p::LookupFrame>(frame,plan.frame.control,1).front();
            // Compare the complete typed state, rather than depending on
            // host/device object padding being initialized identically.
            const auto& reference=*expected.frame;
            EXPECT_EQ(actual.interval.table,reference.interval.table);
            EXPECT_EQ(actual.interval.admission,reference.interval.admission);
            EXPECT_EQ(actual.interval.sequence,reference.interval.sequence);
            EXPECT_EQ(actual.source.table,reference.source.table);
            EXPECT_EQ(actual.source.generation,reference.source.generation);
            EXPECT_EQ(actual.source.sequence,reference.source.sequence);
            EXPECT_EQ(actual.ordinal,reference.ordinal);
            EXPECT_EQ(actual.command_ordinal,reference.command_ordinal);
            EXPECT_EQ(actual.source_storage_generation,reference.source_storage_generation);
            EXPECT_EQ(actual.initial_clock,reference.initial_clock);
            EXPECT_EQ(actual.phase,reference.phase);
            EXPECT_EQ(actual.status,reference.status);
            EXPECT_EQ(actual.original_ids,reference.original_ids);
            EXPECT_EQ(actual.unique_rows,reference.unique_rows);
            EXPECT_EQ(actual.outstanding,reference.outstanding);
            EXPECT_EQ(actual.free_planned,reference.free_planned);
            EXPECT_EQ(actual.victim_planned,reference.victim_planned);
            for(unsigned list=0;list<3;++list)EXPECT_EQ(actual.eligible_cursor[list],reference.eligible_cursor[list]);
            EXPECT_EQ(actual.purpose,reference.purpose);
            EXPECT_EQ(actual.observed,reference.observed);
            EXPECT_EQ(actual.queues_applied,reference.queues_applied);
            EXPECT_EQ(actual.free_exhausted,reference.free_exhausted);
            EXPECT_EQ(actual.cancelled_completion,reference.cancelled_completion);
            return;
        }
        const auto* cache=static_cast<const std::byte*>(coordinator->roots().cache->gpu_data_ptr());
        const auto actual=read<p::CoordinatorPolicyControl>(cache,plan.cache.control,1).front();
        EXPECT_EQ(actual.logical_clock,expected.policy->logical_clock);
        EXPECT_EQ(actual.directory_root,expected.policy->directory_root);
        EXPECT_EQ(actual.node_free,expected.policy->node_free);
        EXPECT_EQ(actual.protected_count,expected.policy->protected_count);
        EXPECT_EQ(actual.flags,expected.policy->flags);
        EXPECT_EQ(actual.aging_accesses,expected.policy->aging_accesses);
        EXPECT_EQ(actual.lane_width,expected.policy->lane_width);
        EXPECT_EQ(actual.protected_limit,expected.policy->protected_limit);
        for(unsigned list=0;list<3;++list){
            EXPECT_EQ(actual.head[list],expected.policy->head[list]);
            EXPECT_EQ(actual.tail[list],expected.policy->tail[list]);
        }
        const auto cells=std::size_t{4}*plan.policy.laneWidth();
        const auto epochs=read<std::uint32_t>(cache,plan.cache.epochs,cells);
        const auto counts=read<std::uint16_t>(cache,plan.cache.counts,cells);
        EXPECT_TRUE(std::equal(epochs.begin(),epochs.end(),expected.frequency_epochs));
        EXPECT_TRUE(std::equal(counts.begin(),counts.end(),expected.frequency_counts));
        if(slots==4)EXPECT_LT(plan.policy.laneWidth(),7u); // Original seven-row pressure trace forces collisions.
        if(actual.aging_accesses==17 && actual.logical_clock>17){
            EXPECT_TRUE(std::any_of(epochs.begin(),epochs.end(),[](auto epoch){return epoch>0;}));
        }
        if(actual.aging_accesses==(1ull<<32) && actual.logical_clock>65535){
            EXPECT_TRUE(std::any_of(counts.begin(),counts.end(),[](auto count){return count==65535;}));
        }
        const auto entries=read<p::DirectoryEntry>(cache,plan.cache.entries,plan.policy.entries());
        for(std::size_t i=0;i<entries.size();++i){
            const auto& a=entries[i];const auto& b=expected.entries[i];
            EXPECT_EQ(a.locator.row,b.locator.row);EXPECT_EQ(a.locator.generation,b.locator.generation);
            EXPECT_EQ(a.locator.owner,b.locator.owner);EXPECT_EQ(a.locator.slot,b.locator.slot);
            EXPECT_EQ(a.touched,b.touched);EXPECT_EQ(a.observed_frame,b.observed_frame);
            EXPECT_EQ(a.previous,b.previous);EXPECT_EQ(a.next,b.next);
            EXPECT_EQ(a.stream,b.stream);EXPECT_EQ(a.present,b.present);
        }
        const auto nodes=read<q::adaptive_lookup::RadixNode>(cache,plan.cache.nodes,plan.policy.entries()-1);
        for(std::size_t i=0;i<nodes.size();++i){
            EXPECT_EQ(nodes[i].child[0],expected.nodes[i].child[0]);
            EXPECT_EQ(nodes[i].child[1],expected.nodes[i].child[1]);
            EXPECT_EQ(nodes[i].split_bit,expected.nodes[i].split_bit);
        }
        const auto state=read<p::LookupFrame>(frame,plan.frame.control,1).front();
        EXPECT_EQ(state.initial_clock,expected.frame->initial_clock);
        EXPECT_EQ(state.original_ids,expected.frame->original_ids);EXPECT_EQ(state.unique_rows,expected.frame->unique_rows);
        EXPECT_EQ(state.outstanding,expected.frame->outstanding);EXPECT_EQ(state.phase,expected.frame->phase);
        EXPECT_EQ(state.status,expected.frame->status);EXPECT_EQ(state.queues_applied,expected.frame->queues_applied);
        for(unsigned list=0;list<3;++list)EXPECT_EQ(state.eligible_cursor[list],expected.frame->eligible_cursor[list]);
        EXPECT_EQ(state.free_planned,expected.frame->free_planned);
        EXPECT_EQ(state.victim_planned,expected.frame->victim_planned);
        const auto decisions=std::size_t{state.free_planned}+state.victim_planned;
        const auto actualDecisions=read<p::AllocationDecision>(frame,plan.frame.decisions,decisions);
        const auto& ownerPlan=owner->plan().parameters();
        const auto* ownerFrame=static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr());
        const auto accepted=read<p::AllocationDecision>(ownerFrame,ownerPlan.frame.accepted_plan,decisions);
        for(std::size_t i=0;i<decisions;++i){
            const auto& b=expected.decisions[i];
            for(const auto* a:{&actualDecisions[i],&accepted[i]}){
                EXPECT_EQ(a->row,b.row);EXPECT_EQ(a->unique,b.unique);EXPECT_EQ(a->kind,b.kind);
                EXPECT_EQ(a->victim.row,b.victim.row);EXPECT_EQ(a->victim.generation,b.victim.generation);
                EXPECT_EQ(a->victim.owner,b.victim.owner);EXPECT_EQ(a->victim.slot,b.victim.slot);
                EXPECT_EQ(a->ordinal.value,b.ordinal.value);
            }
        }
        // Repeated/mixed requests make this frame's frequency/queue state
        // determine the next real native victim, beyond payload correctness.
        const auto ownerPlans=read<p::CoordinatorOwnerPlan>(frame,plan.frame.owner_plans,plan.policy.participants());
        for(std::size_t i=0;i<ownerPlans.size();++i){
            const auto& a=ownerPlans[i];const auto& b=expected.owner_plans[i];
            EXPECT_EQ(a.free_count,b.free_count);EXPECT_EQ(a.victim_count,b.victim_count);
            EXPECT_EQ(a.free_received,b.free_received);EXPECT_EQ(a.victim_received,b.victim_received);
            EXPECT_EQ(a.free_offset,b.free_offset);EXPECT_EQ(a.victim_offset,b.victim_offset);
        }
    }
    std::unique_ptr<pooled_scalar_replay::Pool> scalarPolicy; ///< Independent host oracle; never a device state mirror.
    DeviceId device;IWorkerGPUContext& worker;IBackend* backend;void* stream;std::uint32_t maximum,slots;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryAllocationLease gpuInputs,hostInputs,sourceGPU,sourceHost;
    PhysicalMemoryPlan declaredBOM;
    bool materialized=false;
    std::shared_ptr<RetainedParticipantGraphExecutable> graphOwner;
    std::uint64_t submissionOrdinal=0;
    std::uint32_t frequencyFaultCell=q::adaptive_lookup::None; ///< Exact independent scalar-authored last-lane fault cell.
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<TensorBase> ids,count,trace,source,io,output;
    std::vector<std::byte> expectedOutput;
    f::IO ioOffsets;
    RecordedTrace mode;
    n::OriginalIdWidth originalWidth;
    NativeReplayIntent intent;
    NativeReplayHistory history;
    std::uint32_t payloadWrittenRows=0; ///< Diagnostic scalar-derived dirty-IO frontier, outside event clocks.
    BufferArena arena;
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code;
    std::unique_ptr<IDeviceContext> context;
    std::vector<std::unique_ptr<IComputeStage>> stages;
    ComputeGraph compute;
    DeviceGraphExecutor executor;
    IGPUGraphCapture* graph=nullptr;
    std::uint64_t identity=0,hotGeneration=0;
    GPUGraphExecutableClass graphClass=GPUGraphExecutableClass::General;
};
/** @brief Exercise both cooperative Begin paths and original N>>U under actual native graph replay. */
void nativeReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    PhysicalMemoryPlanBuilder plan;
    for(const auto maximum:{32u,32768u})for(const auto slots:{0u,2u}){
        NativeReplay declaration(device,*session,maximum,slots,RecordedTrace::PlanCancellation,n::OriginalIdWidth::UInt64,true);
        for(const auto& row:declaration.declaration().resources())plan.add(row);
    }
    session->admit(plan.build());
    for(const auto maximum:{32u,32768u})for(const auto slots:{0u,2u}){
        NativeReplay replay(device,*session,maximum,slots);
        worker.submitAndWait([&]{std::vector<std::uint64_t> large(maximum);
        for(std::size_t i=0;i<large.size();++i)large[i]=(i%3==0)?0x100000007ULL:((i%3==1)?9:3);
        replay.replay(1,large);replay.replay(2,std::array<std::uint64_t,1>{0x100000007ULL});replay.replay(3,{});replay.replay(4,large);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
        const bool inspected=replay.graph->inspectKernelNodes(nodes,&detail);
        requireNative(inspected,detail.c_str());
        EXPECT_EQ(nodes.size(),(maximum<=512?16u:63u)+(slots?1u:0u));
        const bool hasFrequency=replay.coordinator->plan().parameters().policy.entries()!=0;
        const auto action=std::to_string(static_cast<std::uint32_t>(n::CoordinatorAction::PlanFree));
        const auto mangledAction="CoordinatorActionE"+action+"E";
        const auto demangledAction="CoordinatorAction)"+action+">";
        unsigned planFreeNodes=0;
        for(const auto& node:nodes){
            const bool fourLane=node.name.find("coordinatorPlanFree")!=std::string::npos;
            const bool scalar=node.name.find("coordinatorPhase")!=std::string::npos &&
                (node.name.find(mangledAction)!=std::string::npos ||
                 node.name.find(demangledAction)!=std::string::npos ||
                 node.name.find("CoordinatorAction::PlanFree")!=std::string::npos);
            if(!fourLane && !scalar)continue;
            ++planFreeNodes;
            EXPECT_EQ(fourLane,hasFrequency);
            EXPECT_EQ(node.grid_x,1u);EXPECT_EQ(node.grid_y,1u);EXPECT_EQ(node.grid_z,1u);
            EXPECT_EQ(node.block_x,hasFrequency?4u:1u);EXPECT_EQ(node.block_y,1u);EXPECT_EQ(node.block_z,1u);
        }
        EXPECT_EQ(planFreeNodes,1u);
        for(const auto& node:nodes){requireNative(node.valid(),"Native diagnostic kernel resource inventory incomplete");
            std::cout<<"POOLED_NATIVE_NODE name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
        std::cout<<"POOLED_NATIVE_REPLAY maximum_original="<<maximum<<" unique=3 owner_slots="<<slots<<" graph_nodes="<<nodes.size()<<" serving=unqualified source_transport=decoded_fixture\n";
        });
    }
    session->requirePayloadRetired();session->retire();
}
/** @brief Actual unsigned UInt32 originals exercise both sides of the 512/513 dispatch boundary. */
void narrowBoundaryReplay(DeviceId device){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled UInt32 native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    PhysicalMemoryPlanBuilder plan;
    for(const auto maximum:{512u,513u})for(const auto slots:{0u,2u}){
        NativeReplay declaration(device,*session,maximum,slots,RecordedTrace::PlanCancellation,n::OriginalIdWidth::UInt32,true);
        for(const auto& row:declaration.declaration().resources())plan.add(row);
    }
    session->admit(plan.build());
    for(const auto maximum:{512u,513u})for(const auto slots:{0u,2u}){
        NativeReplay replay(device,*session,maximum,slots,RecordedTrace::PlanCancellation,n::OriginalIdWidth::UInt32);
        worker.submitAndWait([&]{
        std::vector<std::uint64_t> original(maximum);for(std::size_t i=0;i<original.size();++i)original[i]=i%3==0?0xffff0007u:i%3==1?9u:3u;
        replay.replay(1,original);replay.replay(2,std::array<std::uint64_t,1>{0xffff0007u});replay.replay(3,{});replay.replay(4,original);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
        const bool inspected=replay.graph->inspectKernelNodes(nodes,&detail);
        requireNative(inspected,detail.c_str());
        EXPECT_EQ(nodes.size(),(maximum<=512?16u:63u)+(slots?1u:0u));
        for(const auto& node:nodes){requireNative(node.valid(),"Native UInt32 boundary resource inventory incomplete");
            std::cout<<"POOLED_NATIVE_UINT32_NODE maximum_original="<<maximum<<" original_bits=32 owner_slots="<<slots<<" name="<<node.name<<" registers="<<node.registers_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" local_bytes="<<node.local_memory_bytes_per_thread<<'\n';}
        });
    }
    session->requirePayloadRetired();session->retire();
}
/**
 * @brief Actual admitted original rows and ACKs drive successful free+victim, Ready, reader and retained-hit frames.
 * @param device Original explicitly requested CUDA or ROCm participant.
 * @param policyProof Add the complete scalar policy witness and pressure/aging/saturation trace.
 */
void successfulNativeReplay(DeviceId device,bool policyProof=false){
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled successful native backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    PhysicalMemoryPlanBuilder plan;
    const std::vector<std::uint64_t> periods=policyProof?std::vector<std::uint64_t>{17,1ull<<32}:std::vector<std::uint64_t>{65536};
    const std::vector<n::OriginalIdWidth> widths=policyProof?std::vector{n::OriginalIdWidth::UInt64}:
        std::vector{n::OriginalIdWidth::UInt32,n::OriginalIdWidth::UInt64};
    for(const auto maximum:{32u,32768u})for(const auto width:widths)for(const auto aging:periods){
        NativeReplay declaration(device,*session,maximum,4,RecordedTrace::SuccessfulSource,width,true,policyProof,aging);
        for(const auto& row:declaration.declaration().resources())plan.add(row);
    }
    session->admit(plan.build());
    for(const auto maximum:{32u,32768u})for(const auto width:widths)for(const auto aging:periods){
        NativeReplay replay(device,*session,maximum,4,RecordedTrace::SuccessfulSource,width,false,policyProof,aging);
        worker.submitAndWait([&]{
        std::vector<std::uint64_t> original(maximum,900);original[0]=700;original[1]=701;
        replay.replay(1,std::array<std::uint64_t,2>{101,102});replay.replay(2,original);replay.replay(3,std::array<std::uint64_t,1>{900});replay.replay(4,{});replay.replay(5,original);
        if(policyProof){
            replay.replay(6,original);
            std::vector<std::uint64_t> pressure(maximum);
            for(std::size_t i=0;i<pressure.size();++i)pressure[i]=901+i%2;
            pressure[0]=900;
            replay.replay(7,pressure);
            replay.replay(8,pressure);
        }
        // Resource inspection follows all unprofiled timing records; source/diagnostic kernels have their own class.
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
        const bool inspected=replay.graph->inspectKernelNodes(nodes,&detail);
        requireNative(inspected,detail.c_str());
        EXPECT_EQ(nodes.size(),maximum<=512?47u:76u);
        if(policyProof){
            const auto writers=std::count_if(nodes.begin(),nodes.end(),[](const auto& node){
                return node.name.find("coordinatorPlanFree")!=std::string::npos &&
                    node.grid_x==1 && node.grid_y==1 && node.grid_z==1 &&
                    node.block_x==4 && node.block_y==1 && node.block_z==1;
            });
            EXPECT_EQ(writers,1);
            EXPECT_EQ((replay.code->codeIdentity()>>25)&0x7fU,4u);
            EXPECT_EQ((replay.code->codeIdentity()>>32)&0xffU,7u);
        }
        for(const auto& node:nodes){requireNative(node.valid(),"Native successful kernel resource inventory incomplete");
            const std::array coordinatorNames{"coordinator","Original","radix","coalesceSmall","seedKeys","markGroups","scanGroups","materializeGroups","materializeUnique"};
            const auto scope=node.name.find("pooled_source_fixture")!=std::string::npos?"diagnostic_source":
                std::any_of(coordinatorNames.begin(),coordinatorNames.end(),[&](const auto* name){return node.name.find(name)!=std::string::npos;})?"coordinator":"owner";
            std::cout<<"POOLED_NATIVE_SOURCE_NODE maximum_original="<<maximum<<" original_bits="<<(width==n::OriginalIdWidth::UInt32?32:64)<<" scope="<<scope<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
        std::cout<<"POOLED_NATIVE_SUCCESS maximum_original="<<maximum<<" original_bits="<<(width==n::OriginalIdWidth::UInt32?32:64)<<" owner_slots=4 graph_nodes="<<nodes.size()<<" source=admitted_diagnostic_original_IQ4_NL production_RAM_SSD_transport=unqualified serving=unqualified\n";
        });
    }
    session->requirePayloadRetired();session->retire();
}
/** @brief Capture the production failure frontier with one original prepared owner/session.
 * @param device Explicit CUDA/ROCm participant; absent hardware is a hard native test failure.
 * @note The test accepts actual original-scalar scheduling for its control; v5 must fail stale-header assertions.
 */
void failedEpochNativeReplay(DeviceId device){
    auto* backend=getBackendFor(device);
    requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit fatal-epoch replay backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    PhysicalMemoryPlanBuilder plan;
    NativeReplay declaration(device,*session,32,4,RecordedTrace::SuccessfulSource,n::OriginalIdWidth::UInt64,true,true,17,NativeReplayIntent::FailedEpochReplay);
    for(const auto& row:declaration.declaration().resources())plan.add(row);
    session->admit(plan.build());
    {
        NativeReplay replay(device,*session,32,4,RecordedTrace::SuccessfulSource,n::OriginalIdWidth::UInt64,false,true,17,NativeReplayIntent::FailedEpochReplay);
        worker.submitAndWait([&]{replay.replayFailedEpoch();});
    }
    session->requirePayloadRetired();session->retire();
}
/** @brief Distinct request histories retain their authored arithmetic policy and source scope. */
enum class TransactionTimingHistory {WarmRepeat,WarmInterleaved,WarmProtected16,ChangingPressure,ZeroCapacity};
/** @brief One exact opt-in timing cell; no code identity or production dispatch is selected here. */
struct TransactionTimingCase {
    std::uint32_t live_ids;
    TransactionTimingHistory history;
    /** @return Exact physical query bound; empty work still retains its declared control graph. */
    std::uint32_t maximum()const{return live_ids?live_ids:16;}
    /** @return Original physical owner capacity; C0 is an explicit no-source cancellation control. */
    std::uint32_t slots()const{return history==TransactionTimingHistory::ZeroCapacity?0:history==TransactionTimingHistory::WarmProtected16?32:4;}
    /** @return Authored policy: pressure controls age every original observation to admit actual changing cold rows. */
    std::uint64_t aging()const{return history==TransactionTimingHistory::ChangingPressure?1:65536;}
    /** @return Existing complete source/owner transaction used by this cell. */
    RecordedTrace trace()const{return slots()?RecordedTrace::SuccessfulSource:RecordedTrace::PlanCancellation;}
    /** @return Stable literal metadata name; it never changes actual native code identity. */
    const char* name()const{
        switch(history){
        case TransactionTimingHistory::WarmRepeat:return live_ids?"warm_repeat":"empty_positive_capacity";
        case TransactionTimingHistory::WarmInterleaved:return "warm_interleaved";
        case TransactionTimingHistory::WarmProtected16:return "warm_protected16";
        case TransactionTimingHistory::ChangingPressure:return "changing_hot_cold_pressure_aging1";
        case TransactionTimingHistory::ZeroCapacity:return live_ids?"zero_capacity_no_source_plan_cancel":"empty_zero_capacity_plan_cancel";
        }
        throw std::logic_error("Unknown transaction timing history");
    }
};
/**
 * @brief Author the exact first successful-source conditioning input already certified by NativeReplay.
 * @param count Actual positive original-ID count.
 * @return Original [700,701,900,...] ordering, including its duplicate observations.
 */
std::vector<std::uint64_t> timingWarmRows(std::uint32_t count){
    std::vector<std::uint64_t> rows(count,900);
    if(count>0)rows[0]=700;
    if(count>1)rows[1]=701;
    return rows;
}
/**
 * @brief Author consumed current inputs; these are diagnostic histories rather than inferred model ngrams.
 * @param cell Exact declared capacity/policy/history cell.
 * @param ordinal Original zero-based history position, continuing across warmups and measurements.
 * @return Exactly cell.live_ids rows; no padding is a live work item.
 */
std::vector<std::uint64_t> timingRows(const TransactionTimingCase& cell,std::uint32_t ordinal){
    if(cell.history==TransactionTimingHistory::WarmRepeat)return timingWarmRows(cell.live_ids);
    if(cell.history==TransactionTimingHistory::WarmProtected16){
        std::vector<std::uint64_t> rows(cell.live_ids);
        for(std::size_t i=0;i<rows.size();++i)rows[i]=101+i%16;
        return rows;
    }
    if(cell.history==TransactionTimingHistory::WarmInterleaved){
        constexpr std::array<std::uint64_t,3> hot{700,701,900};
        std::vector<std::uint64_t> rows(cell.live_ids);
        for(std::size_t i=0;i<rows.size();++i)rows[i]=hot[i%hot.size()];
        return rows;
    }
    std::vector<std::uint64_t> rows(cell.live_ids,901+ordinal%4);
    if(!rows.empty())rows[0]=900; // This genuine retained hot row stays pinned while cold/pressure rows rotate.
    return rows;
}
/**
 * @brief Preserve every original input through compact typed run records, without hashes or omitted observations.
 * @param rows Exact ordered consumed request inputs.
 * @return [row,positive-repeat-count] records covering every original position.
 */
nlohmann::json timingInputRuns(std::span<const std::uint64_t> rows){
    auto result=nlohmann::json::array();
    for(std::size_t first=0;first<rows.size();){
        auto end=first+1;
        while(end<rows.size() && rows[end]==rows[first])++end;
        result.push_back({rows[first],end-first});
        first=end;
    }
    return result;
}
/**
 * @brief Collect opt-in complete transaction intervals through the actual production native owner.
 * @param device Explicit CUDA or ROCm participant; unavailable hardware fails rather than skips.
 * @param selected Optional isolated interleaved/protected16 N7168 cell; absent preserves the original 26-cell trajectory.
 * @note Generic timing reports its actual bundle identity and accepts coherent original-scheduling/current-common-arithmetic
 *       and candidate builds. The separate four-lane geometry regression remains identity-specific.
 */
void transactionTiming31(DeviceId device,std::optional<TransactionTimingCase> selected=std::nullopt){
    constexpr unsigned warmups=5,samples=31,conditioning=2;
    auto* backend=getBackendFor(device);
    requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit native transaction timing backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    std::vector<TransactionTimingCase> cells;
    if(selected){
        requireNative(selected->live_ids==7168 &&
            (selected->history==TransactionTimingHistory::WarmInterleaved || selected->history==TransactionTimingHistory::WarmProtected16),
            "Isolated native timing requires an authored interleaved/protected16 N7168 cell");
        cells.push_back(*selected);
    }else{
        for(const auto n:{16u,32u,48u,64u,1024u,7168u}){
            cells.push_back({n,TransactionTimingHistory::WarmRepeat});
            cells.push_back({n,TransactionTimingHistory::WarmInterleaved});
            cells.push_back({n,TransactionTimingHistory::ChangingPressure});
            cells.push_back({n,TransactionTimingHistory::ZeroCapacity});
        }
        cells.push_back({0,TransactionTimingHistory::WarmRepeat});
        cells.push_back({0,TransactionTimingHistory::ZeroCapacity});
    }
    // The process root admits one original source/worker authority. Each
    // independently initialized cell owns a sequential payload/graph lifetime
    // under that same session, as in successfulNativeReplay above.
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    PhysicalMemoryPlanBuilder plan;
    for(const auto cell:cells){
        NativeReplay declaration(device,*session,cell.maximum(),cell.slots(),cell.trace(),n::OriginalIdWidth::UInt64,
            true,true,cell.aging(),NativeReplayIntent::TransactionTiming,NativeReplayHistory::AuthoredPolicy);
        for(const auto& resource:declaration.declaration().resources())plan.add(resource);
    }
    session->admit(plan.build());
    for(const auto cell:cells){
        {
            NativeReplay replay(device,*session,cell.maximum(),cell.slots(),cell.trace(),n::OriginalIdWidth::UInt64,
                false,true,cell.aging(),NativeReplayIntent::TransactionTiming,NativeReplayHistory::AuthoredPolicy);
            worker.submitAndWait([&]{
                std::vector<GPUGraphKernelNodeInfo> kernels;
                std::string detail;
                const bool inspected=replay.graph->inspectKernelNodes(kernels,&detail);
                requireNative(inspected,detail.c_str());
                auto manifest=nlohmann::json::array();
                for(const auto& kernel:kernels){
                    requireNative(kernel.valid(),"Transaction timing has incomplete actual native kernel metadata");
                    manifest.push_back({{"name",kernel.name},{"function_identity",kernel.function_identity},
                        {"grid",{kernel.grid_x,kernel.grid_y,kernel.grid_z}},
                        {"block",{kernel.block_x,kernel.block_y,kernel.block_z}},
                        {"registers_per_thread",kernel.registers_per_thread},
                        {"local_memory_bytes_per_thread",kernel.local_memory_bytes_per_thread},
                        {"static_shared_memory_bytes",kernel.static_shared_memory_bytes},
                        {"dynamic_shared_memory_bytes",kernel.dynamic_shared_memory_bytes},
                        {"max_active_blocks_per_sm",kernel.max_active_blocks_per_sm}});
                }
                const nlohmann::json metadata{{"schema","qfn.native_transaction_timing.v1"},{"case",cell.name()},
                    {"device",device.toString()},{"target_live_ids",cell.live_ids},{"maximum_original",cell.maximum()},
                    {"owner_slots",cell.slots()},
                    {"owner_reply_capacity",replay.owner->plan().parameters().native.reply_capacity},
                    {"coordinator_reply_capacity",replay.coordinator->plan().parameters().native.reply_capacity},
                    {"aging_accesses",cell.aging()},{"warmups",warmups},{"measurements",samples},
                    {"conditioning_requests",conditioning},{"matrix_cells",cells.size()},
                    {"initial_state","fresh original owner/coordinator metadata; no policy upload or reset"},
                    {"source_lifetime","one original prepared session admitted for the complete timing matrix"},
                    {"actual_code_identity",replay.code->codeIdentity()},{"actual_native_algorithm",(replay.code->codeIdentity()>>32)&0xffU},
                    {"encoded_planfree_lane_field",(replay.code->codeIdentity()>>25)&0x7fU},
                    {"actual_capture_identity",replay.coordinator->captureIdentity()},{"kernel_manifest",manifest},
                    {"source_scope",cell.slots()?"admitted diagnostic IQ4_NL rows, not RAM/SSD transport":"no-source C0 planned-cancellation protocol"},
                    {"timer_scope","complete retained native transaction event interval; not isolated PlanFree"},
                    {"scalar_reference","original serial-position frequency and original queue body; checks outside event interval"},
                    {"full_path_launch_economy_qualified",false}};
                std::cout<<"POOLED_NATIVE_TRANSACTION_METADATA "<<metadata.dump()<<'\n';
                std::uint64_t sequence=0;
                auto observe=[&](const char* phase,unsigned index,std::span<const std::uint64_t> rows){
                    NativeTransactionEventSample event;
                    replay.replay(++sequence,rows,&event);
                    requireNative(event.valid,"Transaction timing scalar/terminal checks failed; no valid sample was published");
                    if(cell.history==TransactionTimingHistory::WarmInterleaved && sequence>conditioning){
                        requireNative(event.unique_rows==3 && event.initial_hit_rows==3 && event.source_rows==0,
                            "Warm-interleaved timing requires all three actual retained hot rows and no source fetch");
                    }
                    if(cell.history==TransactionTimingHistory::WarmProtected16 && sequence>conditioning){
                        requireNative(event.unique_rows==16 && event.initial_hit_rows==16 && event.source_rows==0 &&
                            replay.scalarPolicy->original_queue_batch_eligible.value_or(false),
                            "Warm-protected16 timing requires actual retained hits and original scalar queue guard eligibility");
                    }
                    const auto admission=cell.slots() && sequence>=3?2u:1u;
                    const nlohmann::json record{{"schema","qfn.native_transaction_timing.v1"},{"case",cell.name()},
                        {"device",device.toString()},{"target_live_ids",cell.live_ids},{"owner_slots",cell.slots()},
                        {"aging_accesses",cell.aging()},{"actual_code_identity",replay.code->codeIdentity()},
                        {"phase",phase},{"index",index},{"sequence",sequence},{"admission",admission},
                        {"actual_live_ids",rows.size()},{"input_runs",timingInputRuns(rows)},
                        {"event_ms",event.elapsed_ms},{"event_us",double(event.elapsed_ms)*1000.0},
                        {"initial_hit_unique_rows",event.initial_hit_rows},{"completed_unique_rows",event.unique_rows},
                        {"free_planned_rows",event.free_rows},{"victim_planned_rows",event.victim_rows},
                        {"source_rows",event.source_rows},{"source_native_bytes",event.source_native_bytes},
                        {"scalar_and_terminal_checks_passed",true},
                        {"scalar_queue_batch_eligible",replay.scalarPolicy->original_queue_batch_eligible.value_or(false)},
                        {"timer_scope","complete retained native transaction"}};
                    std::cout<<"POOLED_NATIVE_TRANSACTION_SAMPLE "<<record.dump()<<'\n';
                    return event.elapsed_ms;
                };
                if(cell.history==TransactionTimingHistory::WarmProtected16){
                    std::array<std::uint64_t,16> seed{};for(std::size_t i=0;i<seed.size();++i)seed[i]=101+i;
                    observe("conditioning",0,seed);observe("conditioning",1,seed);
                }else{
                    observe("conditioning",0,std::array<std::uint64_t,2>{101,102});
                    const auto seed=timingWarmRows(cell.maximum());observe("conditioning",1,seed);
                }
                for(unsigned index=0;index<warmups;++index){
                    const auto rows=timingRows(cell,index);
                    observe("warmup",index,rows);
                }
                std::vector<float> raw;raw.reserve(samples);
                for(unsigned index=0;index<samples;++index){
                    const auto rows=timingRows(cell,warmups+index);
                    raw.push_back(observe("measurement",index,rows));
                }
                requireNative(raw.size()==samples,"Transaction timing raw sample count is incomplete");
                const nlohmann::json complete{{"schema","qfn.native_transaction_timing.v1"},{"case",cell.name()},
                    {"device",device.toString()},{"target_live_ids",cell.live_ids},{"owner_slots",cell.slots()},
                    {"actual_code_identity",replay.code->codeIdentity()},{"actual_capture_identity",replay.coordinator->captureIdentity()},
                    {"completed_measurements",raw.size()},{"raw_event_ms",raw},{"no_recapture",true},
                    {"lookup_leaf_latency_measured",false},{"profile_qualification",false}};
                std::cout<<"POOLED_NATIVE_TRANSACTION_COMPLETE "<<complete.dump()<<'\n';
            });
        }
    }
    session->requirePayloadRetired();
    session->retire();
}

/**
 * @brief Replay protected-hit batching against the actual original scalar queue body with full policy/bytes.
 * @param device Exact original prepared native owner; no host policy is uploaded to select a GPU branch.
 * @note The captured production transaction retains the same owner/source lifecycle through large/small/empty reuse.
 */
void protectedHitQueueBatchNativeReplay(DeviceId device)
{
    auto* backend=getBackendFor(device);requireNative(backend && backend->deviceCount()>device.ordinal,"Explicit pooled queue backend unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(std::array{device});
    auto& worker=*session->permit(device).worker();
    NativeReplay declaration(device,*session,7168,32,RecordedTrace::SuccessfulSource,
        n::OriginalIdWidth::UInt64,true,true,65536,NativeReplayIntent::ContractReplay,NativeReplayHistory::AuthoredPolicy);
    PhysicalMemoryPlanBuilder bom;for(const auto& resource:declaration.declaration().resources())bom.add(resource);
    session->admit(bom.build());
    {
        NativeReplay replay(device,*session,7168,32,RecordedTrace::SuccessfulSource,
            n::OriginalIdWidth::UInt64,false,true,65536,NativeReplayIntent::ContractReplay,NativeReplayHistory::AuthoredPolicy);
        EXPECT_EQ(replay.owner->plan().parameters().owner.capacity(),32u);
        EXPECT_EQ(replay.owner->plan().parameters().native.reply_capacity,16u);
        EXPECT_EQ(replay.coordinator->plan().parameters().native.reply_capacity,16u);
        worker.submitAndWait([&]{
            std::array<std::uint64_t,16> seed{};for(std::size_t i=0;i<seed.size();++i)seed[i]=101+i;
            replay.replay(1,seed);replay.replay(2,seed); // Initial probation then original protected promotion.
            std::vector<std::uint64_t> original(7168);
            for(std::size_t i=0;i<original.size();++i)original[i]=seed[i%seed.size()];
            replay.replay(3,original);
            ASSERT_TRUE(replay.scalarPolicy->original_queue_batch_eligible.value_or(false));
            std::reverse(original.begin(),original.end());
            replay.replay(4,std::span<const std::uint64_t>(original).first(1024));
            ASSERT_TRUE(replay.scalarPolicy->original_queue_batch_eligible.value_or(false));
            replay.replay(5,std::array<std::uint64_t,1>{101});
            ASSERT_TRUE(replay.scalarPolicy->original_queue_batch_eligible.value_or(false));
            replay.replay(6,{});EXPECT_FALSE(replay.scalarPolicy->original_queue_batch_eligible.value_or(true));
            for(auto& row:original)if(row==seed.back())row=900; // Replace a key: keep the original 16-unique bound.
            // Mixed missing/probation frame must keep original ordered mutation.
            replay.replay(7,original);EXPECT_FALSE(replay.scalarPolicy->original_queue_batch_eligible.value_or(true));
            replay.replay(8,seed); // Protected pressure from the new row is validated by the original oracle too.
        });
    }
    session->requirePayloadRetired();session->retire();
}
}
#ifdef HAVE_CUDA
TEST(PooledImmutableNativeDevice,CUDAProtectedHitQueueBatchCapturedReplay){protectedHitQueueBatchNativeReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,DISABLED_CUDAActualIdentityTransactionTiming31){transactionTiming31(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,DISABLED_CUDAWarmProtected16_7168TransactionTiming31){
    transactionTiming31(DeviceId::cuda(0),TransactionTimingCase{7168,TransactionTimingHistory::WarmProtected16});
}
TEST(PooledImmutableNativeDevice,CUDAPlanFreeFailedEpochCapturedReplay){failedEpochNativeReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,CUDAFourFrequencyLanePolicyCapturedReplay){successfulNativeReplay(DeviceId::cuda(0),true);}
TEST(PooledImmutableNativeDevice,CUDAUInt32DispatchBoundaryReplay){narrowBoundaryReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,CUDACapturedSuccessfulSourceReplay){successfulNativeReplay(DeviceId::cuda(0));}
TEST(PooledImmutableNativeDevice,CUDACapturedPlanCancelLargeSmallEmptyReplay){nativeReplay(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(PooledImmutableNativeDevice,ROCmProtectedHitQueueBatchCapturedReplay){protectedHitQueueBatchNativeReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,DISABLED_ROCmActualIdentityTransactionTiming31){transactionTiming31(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,DISABLED_ROCmWarmProtected16_7168TransactionTiming31){
    transactionTiming31(DeviceId::rocm(0),TransactionTimingCase{7168,TransactionTimingHistory::WarmProtected16});
}
TEST(PooledImmutableNativeDevice,DISABLED_ROCmWarmInterleaved7168TransactionTiming31){
    transactionTiming31(DeviceId::rocm(0),TransactionTimingCase{7168,TransactionTimingHistory::WarmInterleaved});
}
TEST(PooledImmutableNativeDevice,ROCmPlanFreeFailedEpochCapturedReplay){failedEpochNativeReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,ROCmFourFrequencyLanePolicyCapturedReplay){successfulNativeReplay(DeviceId::rocm(0),true);}
TEST(PooledImmutableNativeDevice,ROCmUInt32DispatchBoundaryReplay){narrowBoundaryReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,ROCmCapturedSuccessfulSourceReplay){successfulNativeReplay(DeviceId::rocm(0));}
TEST(PooledImmutableNativeDevice,ROCmCapturedPlanCancelLargeSmallEmptyReplay){nativeReplay(DeviceId::rocm(0));}
#endif
}
