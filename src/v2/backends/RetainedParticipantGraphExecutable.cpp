/**
 * @file RetainedParticipantGraphExecutable.cpp
 * @brief Original General graph device tokens retained through process reclamation.
 *
 * This owner adds no independent resource allowance, stream or live byte ledger.
 * Closed startup calibration owns one timing-event pair within the same General
 * reservation; ordinary retained serving graphs do not create timing events.
 * The original PMA reservation moves once into its canonical retirement slot.
 * Partial native creation retains possible backing even when no handle escaped.
 */
#include "execution/local_execution/graph/ComputeGraph.h"
#include "RetainedParticipantGraphExecutable.h"
#include "GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <stdexcept>
#include <atomic>
#include <limits>
#include <algorithm>
#include <new>
namespace llaminar2 {
namespace {
/** @brief Sole device reservation; original process owner retains it without retaining a worker. */
class GeneralGraphObligation final:public IContextNativeRetirementObligation {
public:
    /** @brief Move the existing admitted token into one original process-lifetime obligation. */
    GeneralGraphObligation(std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,
        std::uint64_t generation,std::string policy,PhysicalMemoryOwnerReservation reservation,
        std::optional<PhysicalMemoryOwnerReservation> host={})
        :memory_(std::move(memory)),device_(device),generation_(generation),policy_(std::move(policy)),reservation_(std::move(reservation)),host_(std::move(host)){}
    /** @copydoc IContextNativeRetirementObligation::originalAuthority */
    const std::shared_ptr<PhysicalMemoryAuthority>& originalAuthority()const noexcept override{return memory_;}
    /** @copydoc IContextNativeRetirementObligation::device */
    DeviceId device()const noexcept override{return device_;}
    /** @copydoc IContextNativeRetirementObligation::poolGeneration */
    std::uint64_t poolGeneration()const noexcept override{return generation_;}
    /** @copydoc IContextNativeRetirementObligation::certificateIdentity */
    const std::string& certificateIdentity()const noexcept override{return policy_;}
    /** @copydoc IContextNativeRetirementObligation::physicalLifetime */
    NativePhysicalLifetime physicalLifetime()const noexcept override{return NativePhysicalLifetime::ProcessRuntime;}
private:
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    DeviceId device_;
    std::uint64_t generation_;
    std::string policy_;
    PhysicalMemoryOwnerReservation reservation_;
    std::optional<PhysicalMemoryOwnerReservation> host_; ///< Original CPU0 opaque token, not device/graph-payload credit.
};
}
/** @brief Recording-only native facade; no retained base reference can submit or recapture. */
class RetainedParticipantGraphExecutable::RecordingCapture final:public IGPUGraphCapture {
public:
    /** @brief Bind the same original definition; this facade creates no native resource. */
    explicit RecordingCapture(RetainedParticipantGraphExecutable& owner):owner_(owner){}
    /** @copydoc IGPUGraphCapture::beginCapture */
    bool beginCapture()override{
        if(owner_.phase_!=Phase::Fresh)throw std::logic_error("Original participant recording may begin exactly once");
        owner_.phase_=Phase::Recording;
        try{owner_.requireRecordingAuthority();if(!native().beginCapture()){owner_.phase_=Phase::Failed;return false;}return true;}catch(...){owner_.phase_=Phase::Failed;throw;}
    }
    /** @copydoc IGPUGraphCapture::endCapture */
    bool endCapture()override{
        if(owner_.phase_!=Phase::Recording && owner_.phase_!=Phase::RecordingFailed)
            throw std::logic_error("Original participant has no active recording");
        const bool failed = owner_.phase_==Phase::RecordingFailed ||
            (owner_.sharedChild_ && !owner_.sharedChildRecorded_);
        try {
            owner_.requireRecordingAuthority();
            if(!native().endCapture()) {owner_.phase_=Phase::Failed;return false;}
            owner_.phase_=failed ? Phase::Failed : Phase::Recorded;
            return true;
        } catch(...) {owner_.phase_=Phase::Failed;throw;}
    }
    /** @copydoc IGPUGraphCapture::instantiate */
    bool instantiate()override{
        if(owner_.phase_!=Phase::Recorded)throw std::logic_error("Original participant definition must seal once before submission");
        try{owner_.requireRecordingAuthority();
            owner_.validateRemoteRecording();
            if(!native().instantiate()){owner_.phase_=Phase::Failed;return false;}
            if(!native().hasExecutable())throw std::logic_error("Original participant instantiation produced no executable");
            owner_.phase_=Phase::Ready;return true;
        }catch(...){owner_.phase_=Phase::Failed;throw;}
    }
    /** @copydoc IGPUGraphCapture::prepareRuntimeContextStorage */
    bool prepareRuntimeContextStorage(const std::shared_ptr<PhysicalMemoryAuthority>& memory)override{
        if(owner_.phase_!=Phase::Recorded)throw std::logic_error("Runtime context storage belongs to original sealed recording");
        try{owner_.requireRecordingAuthority();
            if(!memory || memory.get()!=owner_.memory_.get() || memory.owner_before(owner_.memory_) || owner_.memory_.owner_before(memory))
                throw std::invalid_argument("Runtime graph storage requires the same original prepared PMA/control block");
            // A flat program still crosses the native pre-instantiation boundary.
            // CUDA inspects its actual captured kernels and accepts existing
            // context stack capacity; the original prepared-source pool rejects
            // any growth before a probe/limit mutation. HIP authenticates the
            // sealed original owner without instance-time stack expansion.
            // This delegates one backend authority, not a flat-graph no-op or
            // permission to add conditional/source context storage.
            if(!native().prepareRuntimeContextStorage(memory)){owner_.phase_=Phase::Failed;return false;}
            owner_.recordedRuntimePrepared_=true;return true;
        }catch(...){owner_.phase_=Phase::Failed;throw;}
    }
    /** @brief Reject an untracked native enqueue through any retained base reference. */
    bool launch()override{throw std::logic_error("Participant launches require the owner-controlled original invocation");}
    /** @brief Reject alternate-stream/untracked launch rather than guessing an owner. */
    bool launchOnStream(void*)const override{throw std::logic_error("Participant launch stream is sealed by its original owner");}
    /** @copydoc IGPUGraphCapture::executionStream */
    void* executionStream()const noexcept override{return owner_.stream_;}
    /** @brief Graph update/recapture is never a supported request reset operation. */
    GraphUpdateResult tryUpdate()override{throw std::logic_error("Participant reset changes data, never captured topology");}
    /** @copydoc IGPUGraphCapture::supportsExecutableUpdate */
    bool supportsExecutableUpdate()const noexcept override{return false;}
    /** @copydoc IGPUGraphCapture::hasExecutable */
    bool hasExecutable()const override{return native().hasExecutable();}
    /** @copydoc IGPUGraphCapture::residentMemoryBytes */
    std::size_t residentMemoryBytes()const noexcept override{return owner_.capture_?owner_.capture_->residentMemoryBytes():0;}
    /** @copydoc IGPUGraphCapture::nodeCount */
    std::size_t nodeCount()const override{return native().nodeCount();}
    /** @copydoc IGPUGraphCapture::inspectKernelNodes */
    bool inspectKernelNodes(std::vector<GPUGraphKernelNodeInfo>& nodes,std::string* error)const override{return native().inspectKernelNodes(nodes,error);}
    /** @brief Allow construction-abort cleanup only; it irrevocably fails the original definition. */
    void reset()override{
        owner_.requireConstruction();owner_.phase_=Phase::Failed;owner_.requireRecordingAuthority();native().reset();
    }
    /** @copydoc IGPUGraphCapture::backendName */
    const char* backendName()const override{return native().backendName();}
protected:
    /** @brief Inspect the same native definition under the original bounded helper contract.
     * @param error Receives native kind/count rejection without replacing its diagnostic.
     * @return True only when the actual backend certifies its flat helper node kinds.
     */
    bool validateFlatHelperNodeKinds(std::string* error)const override{
        return native().validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper,error);
    }
private:
    /** @return Actual original native owner, never a replacement definition. */
    IGPUGraphCapture& native()const{
        if(!owner_.capture_)throw std::logic_error("Original participant definition no longer exists");return *owner_.capture_;
    }
    RetainedParticipantGraphExecutable& owner_;
};
const char* RetainedParticipantGraphExecutable::policyIdentity()noexcept{
    return "existing-general-graph-device-bound/v1/conservative-process-retention";
}
RetainedParticipantGraphExecutable::RetainedParticipantGraphExecutable(
    std::shared_ptr<IWorkerGPUContext> worker,std::shared_ptr<PhysicalMemoryAuthority> memory,std::uint64_t generation,DeviceId device,void* stream,NativeContextRetirementSlot retirement,
    std::unique_ptr<IContextNativeRetirementObligation> pending)
    :worker_(std::move(worker)),memory_(std::move(memory)),generation_(generation),device_(device),stream_(stream),retirement_(std::move(retirement)),pendingObligation_(std::move(pending)){}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::generalBOM(PhysicalMemoryResource host,PhysicalMemoryResource gpu,std::size_t slots){
    return definitionBOM(host,gpu,slots,DefinitionPurpose::RetainedParticipant);
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::nativeStorageBOM(const PooledNativeOwnerGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    if(program.geometry().owner!=PhysicalMemoryAllocatorIdentity{gpu.world_rank,gpu.device} || host.world_rank!=gpu.world_rank)
        throw std::invalid_argument("Native storage General BOM changed original owner/rank resources");
    return definitionBOM(host,gpu,1,DefinitionPurpose::NativeStorageGeneral);
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::calibrationBOM(PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    PhysicalMemoryPlanBuilder plan(definitionBOM(host,gpu,1,DefinitionPurpose::PreparedServiceCalibration));
    plan.add(host,PhysicalMemoryOwner::NativePreparationGraphExecutable,PreparedServiceCalibrationSlot::physicalBytes());
    return plan.build();
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::definitionBOM(PhysicalMemoryResource host,PhysicalMemoryResource gpu,
    std::size_t slots,DefinitionPurpose purpose){
    const auto owner=purpose==DefinitionPurpose::PreparedServiceCalibration
        ? PhysicalMemoryOwner::NativePreparationGraphExecutable
        : purpose==DefinitionPurpose::NativeStorageGeneral ? PhysicalMemoryOwner::NativeStorageGraphExecutable
        : PhysicalMemoryOwner::NativeGraphExecutable;
    if(!host.valid() || host.device!=DeviceId::cpu() || !gpu.valid() || !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
        throw std::invalid_argument("General graph BOM requires original CPU0 and same-rank GPU observations");
    const auto hostUnit=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(gpu.device);
    const auto deviceUnit=GPUGraphMemoryContract::reservationBytesPerExecutable(gpu.device,GPUGraphExecutableClass::General);
    const auto ownerBytes=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),owner,DeviceId::cpu()).physicalBytes();
    const auto nativeBytes=nativeGraphCaptureObjectBytes(gpu.device);
    if(!nativeBytes || ownerBytes>std::numeric_limits<std::size_t>::max()-nativeBytes ||
       ownerBytes+nativeBytes>std::numeric_limits<std::size_t>::max()-sizeof(RecordingCapture) ||
       hostUnit>std::numeric_limits<std::size_t>::max()-(ownerBytes+nativeBytes+sizeof(RecordingCapture)))
        throw std::overflow_error("General graph known host obligation overflows");
    const auto completeHost=hostUnit+ownerBytes+nativeBytes+sizeof(RecordingCapture);
    if(slots>std::numeric_limits<std::size_t>::max()/completeHost || slots>std::numeric_limits<std::size_t>::max()/deviceUnit)
        throw std::overflow_error("General graph declared slot allowance overflows");
    PhysicalMemoryPlanBuilder plan;
    if(slots){plan.add(host,owner,slots*completeHost);
        plan.add(gpu,owner,slots*deviceUnit);}
    return plan.build();
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::hostBOMWithSelection(const PooledRemoteGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu,const std::optional<NativeAuxiliaryResourceSelection>& selected,
    std::size_t nativeObjectBytes){
    std::optional<NativeAuxiliaryResourceAllowance> allowance;
    if(selected){
        if(!selected->program)throw std::invalid_argument("Remote graph selection omitted its original program");
        allowance.emplace(NativeAuxiliaryResourceAllowance{selected->hostBytes,selected->deviceBytes,selected->evidence,
            selected->lifetime,*selected->program,selected->remoteFamily});
    }
    return hostBOMWithAllowance(program,host,gpu,allowance,nativeObjectBytes);
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::hostBOMWithAllowance(const PooledRemoteGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu,const std::optional<NativeAuxiliaryResourceAllowance>& selected,
    std::size_t nativeObjectBytes){
    const auto& geometry=program.geometry();
    if(!host.valid() || host.device!=DeviceId::cpu() || !gpu.valid() ||
        PhysicalMemoryAllocatorIdentity{gpu.world_rank,gpu.device}!=geometry.owner || host.world_rank!=gpu.world_rank || !nativeObjectBytes)
        throw std::invalid_argument("Remote graph host BOM requires original CPU0 and exact addressed GPU observations");
    PhysicalMemoryPlanBuilder plan;
    plan.add(host,PhysicalMemoryOwner::NativeStorageGraphExecutable,
        CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),PhysicalMemoryOwner::NativeStorageGraphExecutable,DeviceId::cpu()).physicalBytes());
    plan.add(host,PhysicalMemoryOwner::NativeStorageGraphExecutable,
        CPUStorageRoot<std::byte>(GPUFlatGraphScratchLayout::fromLimits(program.limits()).bytes,PhysicalMemoryOwner::NativeStorageGraphExecutable,DeviceId::cpu()).physicalBytes());
    plan.add(host,PhysicalMemoryOwner::NativeStorageGraphExecutable,nativeObjectBytes);
    plan.add(host,PhysicalMemoryOwner::NativeStorageGraphExecutable,sizeof(RecordingCapture));
    if(selected){
        if(!selected->hostBytes || selected->deviceBytes || selected->lifetime!=NativePhysicalLifetime::ProcessRuntime ||
            selected->program!=NativeAuxiliaryResourceProgram::pooledRemoteOwner(program) ||
            !selected->remoteFamily || !selected->remoteFamily->accepts(program.limits(),PooledRemoteGraphProgramClass::argumentLayout()))
            throw std::invalid_argument("Remote graph host program cannot borrow device bytes or a shorter untested lifetime");
        plan.add(host,PhysicalMemoryOwner::NativeStorageGraphExecutable,selected->hostBytes);
    }
    return plan.build();
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::hostBOM(const PooledRemoteGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    const auto allowance=NativeAuxiliaryResourceCatalog::allowance(program.geometry().owner.device.type,
        NativeAuxiliaryResourceProgram::pooledRemoteOwner(program));
    return hostBOMWithAllowance(program,host,gpu,allowance,nativeGraphCaptureObjectBytes(program.geometry().owner.device));
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::create(
    const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program){
    if(program.geometry().owner.device!=original.device())throw std::invalid_argument("Remote graph host program has a foreign original GPU");
    auto selected=NativeAuxiliaryResourceCatalog::require(original.device(),NativeAuxiliaryResourceProgram::pooledRemoteOwner(program));
    return createBound(original,stream,HostProgramBinding{program,std::move(selected),
        nativeGraphCaptureObjectBytes(original.device()),GraphHostQualification::InstalledCertified});
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::create(
    const PreparedGPUContextUsePermit& original,void* stream,const PooledNativeOwnerGraphProgram& program){
    const auto memory=original.memory();
    if(!memory || program.geometry().owner!=PhysicalMemoryAllocatorIdentity{memory->worldRank(),original.device()})
        throw std::invalid_argument("Native storage General creator changed original owner/rank");
    auto result=createBound(original,stream,std::nullopt,GeneralHostOwnership::OriginalAllowance,DefinitionPurpose::NativeStorageGeneral);
    result->storageProgram_=PooledStorageOwnerProgram(program);return result;
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::createRecordedRemote(
    const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program,
    NativeAuxiliaryResourceSelection selected,std::size_t nativeObjectBytes){
    return createBound(original,stream,HostProgramBinding{program,std::move(selected),nativeObjectBytes,GraphHostQualification::RecordedReplay});
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::createDiagnosticRemote(
    const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program){
    return createBound(original,stream,HostProgramBinding{program,std::nullopt,
        nativeGraphCaptureObjectBytes(original.device()),GraphHostQualification::UnqualifiedDiagnostic});
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::create(
    const PreparedGPUContextUsePermit& original,void* stream){
    return createBound(original,stream,std::nullopt,GeneralHostOwnership::OriginalAllowance);
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::createDeviceBound(
    const PreparedGPUContextUsePermit& original,void* stream){return createBound(original,stream,std::nullopt);}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::createBound(
    const PreparedGPUContextUsePermit& original,void* stream,std::optional<HostProgramBinding> host,GeneralHostOwnership generalHost,
    DefinitionPurpose purpose){
    const auto worker=original.worker();
    if(!worker)throw std::invalid_argument("Retained participant requires its live original worker capability");
    const auto memory=original.memory();
    if(!memory || !original.device().is_gpu() || !worker || worker->ownsCurrentThread() ||
       isGraphCaptureActive() || reinterpret_cast<std::uintptr_t>(stream)<=2 || stream!=worker->defaultStream())
        throw std::invalid_argument("Retained participant capture requires original off-worker PMA/provenance/exact stream");
    const auto current=GPUDeviceContextPool::instance().retainExistingContextProvenance(original.device());
    current.requireWorker(worker);
    if(current.generation()!=original.generation())throw std::invalid_argument("Retained participant capture has stale original generation");
    const bool originalGeneral=generalHost==GeneralHostOwnership::OriginalAllowance;
    if((purpose==DefinitionPurpose::PreparedServiceCalibration || purpose==DefinitionPurpose::NativeStorageGeneral) && (!originalGeneral || host))
        throw std::invalid_argument("Prepared/native-storage General purpose requires its exact original allowance");
    const auto allocationOwner=purpose==DefinitionPurpose::PreparedServiceCalibration
        ? PhysicalMemoryOwner::NativePreparationGraphExecutable
        : (host || purpose==DefinitionPurpose::NativeStorageGeneral) ? PhysicalMemoryOwner::NativeStorageGraphExecutable : PhysicalMemoryOwner::NativeGraphExecutable;
    if(originalGeneral && host)throw std::invalid_argument("General and authored remote host obligations cannot be double charged");
    std::string policy=policyIdentity();
    if(purpose==DefinitionPurpose::PreparedServiceCalibration)policy+="/prepared-service-calibration";
    if(purpose==DefinitionPurpose::NativeStorageGeneral)policy+="/native-storage-general";
    std::optional<PhysicalMemoryOwnerReservation> opaqueHost;
    AlignedVector<std::byte> ownerStorage,scratch;
    PhysicalMemoryAllocationLease nativeObject,recordingObject;
    if(originalGeneral){
        const auto bytes=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(original.device());
        opaqueHost.emplace(memory->reserveNewAllocations(DeviceId::cpu(),allocationOwner,bytes));
        policy+="/general-host-process-allowance="+std::to_string(bytes);
        ownerStorage=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),allocationOwner,DeviceId::cpu(),
            CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        nativeObject=memory->claimNewAllocation(DeviceId::cpu(),allocationOwner,nativeGraphCaptureObjectBytes(original.device()));
        recordingObject=memory->claimNewAllocation(DeviceId::cpu(),allocationOwner,sizeof(RecordingCapture));
    }
    if(host){
        const auto& geometry=host->program.geometry();
        if(geometry.owner.device!=original.device() || geometry.owner.world_rank!=memory->worldRank() || !host->nativeObjectBytes)
            throw std::invalid_argument("Remote graph declaration has a foreign original device/rank/object class");
        if(host->selected){
            const auto& selected=*host->selected;const auto key=current.originalBirthKey();
            const auto digest=[](const std::string& text){return text.size()==64 && text.find_first_not_of('0')!=std::string::npos &&
                std::all_of(text.begin(),text.end(),[](char c){return (c>='0'&&c<='9') || (c>='a'&&c<='f');});};
            if(!selected.hostBytes || selected.deviceBytes || selected.lifetime!=NativePhysicalLifetime::ProcessRuntime ||
                !selected.program || *selected.program!=NativeAuxiliaryResourceProgram::pooledRemoteOwner(host->program) ||
                !selected.remoteFamily || !selected.remoteFamily->accepts(host->program.limits(),PooledRemoteGraphProgramClass::argumentLayout()) ||
                selected.contextGeneration!=original.generation() || selected.resourceSource!=NativeAuxiliaryResourceCatalog::resourceSourceIdentity() ||
                selected.sourceIdentity!=NativeAuxiliaryResourceCatalog::liveSourceIdentity(*key) || !digest(selected.evidence) ||
                (host->qualification!=GraphHostQualification::InstalledCertified && host->qualification!=GraphHostQualification::RecordedReplay))
                throw std::invalid_argument("Remote graph host claim is not the same original source/generation/process program");
            opaqueHost.emplace(memory->reserveNewAllocations(DeviceId::cpu(),allocationOwner,selected.hostBytes));
            policy+="/remote-host/"+host->program.qualificationClass().identity()+"/"+selected.resourceSource+"/"+selected.evidence+"/"+selected.sourceIdentity+"/host="+std::to_string(selected.hostBytes)+"/"+selected.remoteFamily->identity();
        }else{
            if(host->qualification!=GraphHostQualification::UnqualifiedDiagnostic)
                throw std::invalid_argument("Remote graph requires installed host authority; diagnostics must be explicit");
            policy+="/explicit-unqualified-remote-host/"+host->program.qualificationClass().identity();
        }
        // Known buffers/objects remain exact and distinct from the opaque SDK reservation.
        ownerStorage=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),allocationOwner,
            DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        const auto layout=GPUFlatGraphScratchLayout::fromLimits(host->program.limits());
        scratch=CPUStorageRoot<std::byte>(layout.bytes,allocationOwner,DeviceId::cpu(),
            CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        nativeObject=memory->claimNewAllocation(DeviceId::cpu(),allocationOwner,host->nativeObjectBytes);
        recordingObject=memory->claimNewAllocation(DeviceId::cpu(),allocationOwner,sizeof(RecordingCapture));
    }
    auto reservation=memory->reserveNewAllocations(original.device(),allocationOwner,
        GPUGraphMemoryContract::reservationBytesPerExecutable(original.device(),GPUGraphExecutableClass::General));
    auto retirement=original.provenance().reserveNativeRetirementSlot(memory,policy);
    // Per-instance retention: no implicit process-family borrowing or model reload allowance.
    static std::atomic<std::uint64_t> next{1};
    auto instance=next.load(std::memory_order_relaxed);
    do {
        if(instance==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Retained graph instance identity exhausted");
    }while(!next.compare_exchange_weak(instance,instance+1,std::memory_order_relaxed));
    retirement.prepareProcessRuntime("retained-participant-general-graph-instance-v1:"+std::to_string(instance));
    auto pending=std::make_unique<GeneralGraphObligation>(memory,original.device(),original.generation(),policy,std::move(reservation),std::move(opaqueHost));
    std::shared_ptr<RetainedParticipantGraphExecutable> result;
    if(host || originalGeneral){
        static_assert(alignof(RetainedParticipantGraphExecutable)<=AlignedVector<std::byte>::ALIGNMENT);
        auto* object=new(ownerStorage.data()) RetainedParticipantGraphExecutable(
            worker,memory,original.generation(),original.device(),stream,std::move(retirement),std::move(pending));
        // Existing CPUStorageRoot frees its original backing only AFTER the complete object destructor.
        result=std::shared_ptr<RetainedParticipantGraphExecutable>(object,
            [storage=std::move(ownerStorage)](RetainedParticipantGraphExecutable* object)mutable{object->~RetainedParticipantGraphExecutable();storage={};});
        if(host){result->storageProgram_=PooledStorageOwnerProgram(host->program);result->hostQualification_=host->qualification;}
        else result->hostQualification_=GraphHostQualification::GeneralProcessAllowance;
        result->inspectionScratch_=std::move(scratch);result->nativeObjectLease_=std::move(nativeObject);result->recordingObjectLease_=std::move(recordingObject);
    }else result=std::shared_ptr<RetainedParticipantGraphExecutable>(new RetainedParticipantGraphExecutable(
        worker,memory,original.generation(),original.device(),stream,std::move(retirement),std::move(pending)));
    result->purpose_=purpose;
    result->self_=result;
    try {
        worker->submitAndWait([&]{
            result->retirement_.adopt(std::move(result->pendingObligation_));
            result->nativeAttempted_=true;
            result->retirement_.armNativeAttempt();
            result->capture_=worker->createGraphCapture(stream);
            if(!result->capture_ || result->capture_->executionStream()!=stream ||
               result->capture_->hasExecutable() || result->capture_->nodeCount())
                throw std::runtime_error("Retained participant requires a pristine capture on its exact original stream");
            result->recording_=std::make_unique<RecordingCapture>(*result);
        });
    }catch(...){result.reset();throw;}
    return result;
}
void RetainedParticipantGraphExecutable::requireRecordingAuthority()const{
    if(!worker_->ownsCurrentThread())throw std::logic_error("Graph recording mutation requires its exact original worker");
    requireLiveOriginal();
}
void RetainedParticipantGraphExecutable::requireLiveOriginal()const{
    const auto current=GPUDeviceContextPool::instance().retainExistingContextProvenance(device_);
    current.requireWorker(worker_);
    if(current.generation()!=generation_ || memory_->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
        throw std::invalid_argument("Graph recording mutation has stale original prepared authority");
}
bool RetainedParticipantGraphExecutable::readyForSubmission()const{return phase_==Phase::Ready && capture_ && capture_->hasExecutable();}
void RetainedParticipantGraphExecutable::sealSharedChildDefinition()
{
    requireRecordingAuthority();
    if(phase_!=Phase::Recorded || purpose_!=DefinitionPurpose::RetainedParticipant ||
       storageProgram_ || sharedChild_ || !capture_ || capture_->hasExecutable() || !capture_->nodeCount())
        throw std::logic_error("Shared child requires its original nonempty General graph-only definition");
    phase_=Phase::RecordedSharedChild;
}
void RetainedParticipantGraphExecutable::bindSharedChildDefinition(
    const std::shared_ptr<RetainedParticipantGraphExecutable>& child)
{
    requireRecordingAuthority();
    if(isGraphCaptureActive() || phase_!=Phase::Fresh || sharedChild_ || storageProgram_ ||
       purpose_!=DefinitionPurpose::RetainedParticipant || !child || child.get()==this)
        throw std::invalid_argument("Shared child binding requires a fresh original General parent");
    // An off-worker retirement can hold this lock while awaiting the
    // original worker. Reject that overlap before touching native ownership.
    std::unique_lock definition(child->embeddedDefinitionMutex_,std::try_to_lock);
    if(!definition.owns_lock())
        throw std::logic_error("Shared child is already closing its original definition");
    if(child->phase_!=Phase::RecordedSharedChild || !child->capture_ ||
       child->self_.owner_before(child) || child.owner_before(child->self_) ||
       child->device_!=device_ || child->generation_!=generation_ || child->worker_!=worker_ ||
       child->worker_.owner_before(worker_) || worker_.owner_before(child->worker_) ||
       child->memory_!=memory_ || child->memory_.owner_before(memory_) || memory_.owner_before(child->memory_))
        throw std::invalid_argument("Shared child changed its original definition, worker, device, generation or PMA");
    child->requireLiveOriginal();
    sharedChild_=child;
    child->embeddedBorrowers_.fetch_add(1,std::memory_order_acq_rel);
}
void RetainedParticipantGraphExecutable::recordSharedChildAtFrontier()
{
    requireRecordingAuthority();
    if(!isGraphCaptureActive() || phase_!=Phase::Recording || !sharedChild_ || sharedChildRecorded_)
        throw std::logic_error("Shared child import requires its original active parent frontier exactly once");
    try {
        if(!capture_->appendActiveCaptureChild(*sharedChild_->capture_))
            throw std::runtime_error("Original native shared child frontier import failed");
        sharedChildRecorded_=true;
    } catch(...) {
        // Close the exact native interval normally before discarding the failed
        // parent. Its child stays alive even after a partial SDK mutation.
        phase_=Phase::RecordingFailed;
        throw;
    }
}
void RetainedParticipantGraphExecutable::requireConstruction()const{
    if(phase_!=Phase::Fresh && phase_!=Phase::Recording && phase_!=Phase::Recorded)
        throw std::logic_error("Original participant recording is sealed or failed");
}
IGPUGraphCapture& RetainedParticipantGraphExecutable::capture()const{
    requireConstruction();
    if(!recording_ || !capture_ || capture_->executionStream()!=stream_)
        throw std::logic_error("Retained participant lost its original recording owner/stream");
    return *recording_;
}
bool RetainedParticipantGraphExecutable::submit(std::uint64_t ordinal){
    requireLiveOriginal();
    if(isGraphCaptureActive() || phase_!=Phase::Ready || !capture_ || !capture_->hasExecutable() || !ordinal ||
       lastSubmission_==std::numeric_limits<std::uint64_t>::max() || ordinal!=lastSubmission_+1)
        throw std::logic_error("Original participant submission is unsealed, failed, skipped or duplicated");
    struct Enqueue {RetainedParticipantGraphExecutable* owner;bool invoked=false,result=false;} enqueue{this};
    const bool priorPending=pending_;
    pending_=true;lastSubmission_=ordinal; // Reader ownership precedes its possible first native enqueue.
    const auto launch=[&enqueue]{
        enqueue.invoked=true;
        enqueue.result=enqueue.owner->capture_->launch();
    };
    try{if(worker_->ownsCurrentThread())launch();else worker_->submitAndWait(launch);}
    catch(...){
        if(enqueue.invoked)phase_=Phase::Failed;
        else {pending_=priorPending;--lastSubmission_;} // Original queue rejected BEFORE any native call.
        throw;
    }
    if(!enqueue.result)phase_=Phase::Failed;
    return enqueue.result;
}
bool RetainedParticipantGraphExecutable::observeTerminal(){
    requireLiveOriginal();
    if(phase_==Phase::RecordedStorageBody)throw std::logic_error("Embedded storage requires its actual parent terminal proof");
    if(isGraphCaptureActive() || worker_->ownsCurrentThread())throw std::logic_error("Retained participant terminal aggregation requires its original whole-request boundary");
    GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
    worker_->submitAndWait([&]{state=worker_->queryStreamExecutionState(stream_,"retained participant request terminal");});
    if(state==GPUStreamExecutionState::Complete){pending_=false;return true;}
    return false;
}
void RetainedParticipantGraphExecutable::bindCohortDefinition(){
    bool expected=false;
    if(phase_.load(std::memory_order_acquire)!=Phase::Fresh || !cohortBound_.compare_exchange_strong(expected,true))
        throw std::logic_error("Original native definition already belongs to a retained participant program");
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::createPreparedCalibration(
    const PreparedGPUContextUsePermit& original,void* stream){
    // Choose the closed purpose before any PMA claim or native creation;
    // serving arena reservations cannot consume this admitted startup slot.
    auto owner=createBound(original,stream,std::nullopt,GeneralHostOwnership::OriginalAllowance,
        DefinitionPurpose::PreparedServiceCalibration);
    // The same conservative General obligation covers this closed startup
    // graph plus its one persistent timing pair; ordinary General graphs do
    // not acquire timing instrumentation. Partial creation stays owner-held.
    owner->requireLiveOriginal();
    owner->worker_->submitAndWait([&]{
        for(auto& event:owner->calibrationTimingEvents_){
            event=owner->worker_->createEvent(GPUEventPurpose::Timing);
            if(!event)throw std::runtime_error("Original prepared calibration timing pair creation failed");
        }
    });
    return owner;
}
void RetainedParticipantGraphExecutable::resetPreparedCalibration(){
    requireLiveOriginal();
    if(purpose_!=DefinitionPurpose::PreparedServiceCalibration || storageProgram_ || cohortBound_.load() ||
       worker_->ownsCurrentThread() || pending_ || phase_!=Phase::Ready || !capture_ || lastSubmission_ <= calibrationRetiredSubmission_)
        throw std::logic_error("Only completed original startup calibration can replace its cold definition");
    try{
        worker_->submitAndWait([&]{capture_->reset();
            if(capture_->hasExecutable() || capture_->nodeCount())
                throw std::logic_error("Original calibration reset retained native definition handles");
        });
        recordedRuntimePrepared_=false;
        calibrationRetiredSubmission_=lastSubmission_;
        phase_=Phase::Fresh;
    }catch(...){phase_=Phase::Failed;throw;}
}
void RetainedParticipantGraphExecutable::validateRemoteRecording(){
    if(!storageProgram_)return;
    const auto* mapped=std::get_if<PooledRemoteGraphProgram>(&*storageProgram_);
    if(!mapped){
        if(purpose_!=DefinitionPurpose::NativeStorageGeneral || hostQualification_!=GraphHostQualification::GeneralProcessAllowance)
            throw std::logic_error("Native storage definition lost its original General admission contract");
        return; // Actual native semantics were checked by the original graph builder; no fake flat node scope.
    }
    GPUFlatGraphStructure observed;std::string error;
    if(!capture_->inspectFlatGraphStructure(mapped->limits(),
        {inspectionScratch_.data(),inspectionScratch_.size()},observed,&error))
        throw std::runtime_error("Original remote graph failed its declared flat structural seal: "+error);
    const auto& declared=mapped->limits();
    if(observed.nodes!=declared.maximumNodes || observed.edges!=declared.maximumEdges ||
        !observed.linearChain || observed.kinds[static_cast<unsigned>(GPUFlatGraphNodeKind::Kernel)]!=observed.nodes)
        throw std::runtime_error("Original remote graph differs from its exact source-authored kernel chain");
}
void RetainedParticipantGraphExecutable::sealRecordedStorageBody(){
    requireRecordingAuthority();
    if(phase_!=Phase::Recorded || !storageProgram_ || !recordedRuntimePrepared_ ||
       !cohortBound_.load(std::memory_order_acquire) || !capture_ || capture_->hasExecutable())
        throw std::logic_error("Embedded storage requires its original prepared graph-only source definition");
    try{validateRemoteRecording();phase_=Phase::RecordedStorageBody;}
    catch(...){phase_=Phase::Failed;throw;}
}
RetainedParticipantGraphExecutable::RecordedStorageDefinition::RecordedStorageDefinition(
    std::shared_ptr<RetainedParticipantGraphExecutable> owner,PreparedGPUContextUsePermit parent,
    std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation)
    :owner_(std::move(owner)),originalParent_(std::move(parent)),
     sourcePlan_(preparation->definition().plan),sourcePreparation_(std::move(preparation)),
     sourceIdentity_(sourcePlan_->captureIdentity()),definitionGeneration_(owner_->definitionGeneration_){
    owner_->embeddedBorrowers_.fetch_add(1,std::memory_order_acq_rel);
}
RetainedParticipantGraphExecutable::RecordedStorageDefinition::RecordedStorageDefinition(RecordedStorageDefinition&& other)noexcept
    :owner_(std::move(other.owner_)),originalParent_(std::move(other.originalParent_)),
     sourcePlan_(std::move(other.sourcePlan_)),sourcePreparation_(std::move(other.sourcePreparation_)),
     sourceIdentity_(other.sourceIdentity_),definitionGeneration_(other.definitionGeneration_),attempted_(other.attempted_){}
RetainedParticipantGraphExecutable::RecordedStorageDefinition::~RecordedStorageDefinition(){
    if(owner_)owner_->embeddedBorrowers_.fetch_sub(1,std::memory_order_acq_rel);
}
void RetainedParticipantGraphExecutable::RecordedStorageDefinition::requireCurrent()const{
    if(!owner_ || !sourcePlan_ || !sourcePreparation_ || owner_->phase_!=Phase::RecordedStorageBody ||
       owner_->definitionGeneration_!=definitionGeneration_ ||
       sourcePlan_->captureIdentity()!=sourceIdentity_)
        throw std::logic_error("Recorded storage borrow lost its immutable source definition");
    owner_->requireLiveOriginal();originalParent_.requireWorker(owner_->worker_);
    sourcePreparation_->requireCurrentDefinition();
    if(sourcePreparation_->definition().plan!=sourcePlan_)
        throw std::logic_error("Recorded storage changed its original source plan");
    sourcePlan_->requireCurrentStorageDefinition(*owner_);
}
bool RetainedParticipantGraphExecutable::RecordedStorageDefinition::attachTo(
    IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress){
    requireCurrent();
    if(attempted_ || !owner_->worker_->ownsCurrentThread() || parent.hasExecutable() || !parent.nodeCount() ||
       reinterpret_cast<std::uintptr_t>(parent.executionStream())<=2 ||
       (progress && (progress->device()!=owner_->device_ || !progress->authorityIdentity() || progress->name().empty() ||
       progress->originalSourceDefinition()!=sourcePreparation_)))
        throw std::logic_error("Storage body requires its original matched source definition/worker/parent attachment");
    sourcePlan_->acquirePreparedStorageDefinition(*owner_,parent.executionStream());
    attempted_=true;
    const IGPUGraphCapture* body=owner_->capture_.get();
    return progress ? progress->attachWithFiniteBody(parent,{&body,1}) : parent.appendFiniteBodySiblings({&body,1});
}
RetainedParticipantGraphExecutable::RecordedStorageDefinition RetainedParticipantGraphExecutable::retainRecordedStorageDefinition(
    const PreparedGPUContextUsePermit& originalParent,std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation){
    requireLiveOriginal();
    std::lock_guard definition(embeddedDefinitionMutex_);
    const auto sameMemory=[&](const auto& memory){return memory && memory.get()==memory_.get() &&
        !memory.owner_before(memory_) && !memory_.owner_before(memory);};
    if(phase_!=Phase::RecordedStorageBody || !preparation || originalParent.device()!=device_ ||
       originalParent.generation()!=generation_ || !sameMemory(originalParent.memory()))
        throw std::invalid_argument("Recorded storage requires its same original PMA/parent and cold source definition");
    originalParent.requireWorker(worker_);preparation->requireDefinition();
    if(!sameMemory(preparation->originalSource().memory()))
        throw std::invalid_argument("Recorded storage source belongs to another original PMA");
    const auto& sourcePlan=preparation->definition().plan;
    sourcePlan->requirePreparedStorageDefinition(*this);
    auto self=self_.lock();if(!self)throw std::logic_error("Recorded storage lost its original owner control block");
    RecordedStorageDefinition result(std::move(self),originalParent,std::move(preparation));
    result.requireCurrent();return result;
}
void RetainedParticipantGraphExecutable::releaseSourceDefinition(){
    requireLiveOriginal();
    if(worker_->ownsCurrentThread())throw std::logic_error("Original source definition release requires its terminal owner");
    std::lock_guard definition(embeddedDefinitionMutex_);
    if(!storageProgram_ || purpose_==DefinitionPurpose::PreparedServiceCalibration || pending_ ||
       embeddedBorrowers_.load(std::memory_order_acquire) || !capture_ ||
       (phase_!=Phase::Ready && phase_!=Phase::RecordedStorageBody))
        throw std::logic_error("Original source slot still has a live, failed or incomplete native definition");
    try{
        worker_->submitAndWait([&]{
            if(worker_->queryStreamExecutionState(stream_,"original source serial definition retirement")!=GPUStreamExecutionState::Complete)
                throw std::logic_error("Original source definition stream has not retired");
            capture_->reset();
            if(capture_->nodeCount() || capture_->hasExecutable())
                throw std::logic_error("Original source definition retained native handles after reset");
            recording_.reset();
        });
        recordingObjectLease_={};
        recordedRuntimePrepared_=false;cohortBound_.store(false,std::memory_order_release);
        phase_=Phase::DormantSourceSlot;
    }catch(...){phase_=Phase::Failed;throw;}
}
void RetainedParticipantGraphExecutable::reacquireSourceDefinition(){
    requireLiveOriginal();
    if(worker_->ownsCurrentThread())throw std::logic_error("Original source recording requires its cold owner");
    std::lock_guard definition(embeddedDefinitionMutex_);
    if(!storageProgram_ || phase_!=Phase::DormantSourceSlot || pending_ || !capture_ || recording_ ||
       capture_->nodeCount() || capture_->hasExecutable() || embeddedBorrowers_.load(std::memory_order_acquire) ||
       cohortBound_.load(std::memory_order_acquire) || definitionGeneration_==std::numeric_limits<std::uint64_t>::max())
        throw std::logic_error("Original source slot cannot replace its active or exhausted definition");
    try{
        recordingObjectLease_=memory_->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable,sizeof(RecordingCapture));
        ++definitionGeneration_;
        // The SAME original backend capture wrapper/known-object claim remains
        // source owned, exactly like serial prepared calibration. Only the cold
        // definition facade changes; there is no new opaque native creator.
        worker_->submitAndWait([&]{recording_=std::make_unique<RecordingCapture>(*this);});
        phase_=Phase::Fresh;
    }catch(...){
        recording_.reset();recordingObjectLease_={};
        phase_=Phase::Failed;throw;
    }
}
std::size_t RetainedParticipantGraphSourceSlots::physicalBytes(std::size_t count){
    if(!count || count>(std::numeric_limits<std::size_t>::max()-sizeof(RetainedParticipantGraphSourceSlots))/sizeof(Slot))
        throw std::invalid_argument("Original remote source slot roster is empty or overflows");
    static_assert(alignof(Slot)<=alignof(RetainedParticipantGraphSourceSlots));
    return CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphSourceSlots)+count*sizeof(Slot),
        PhysicalMemoryOwner::NativeStorageGraphExecutable,DeviceId::cpu()).physicalBytes();
}
std::shared_ptr<RetainedParticipantGraphSourceSlots> RetainedParticipantGraphSourceSlots::create(
    std::uint64_t source,std::size_t count,const std::shared_ptr<PhysicalMemoryAuthority>& memory){
    if(!source || !memory)throw std::invalid_argument("Original source slots require their actual source/PMA");
    auto storage=CPUStorageRoot<std::byte>(physicalBytes(count),PhysicalMemoryOwner::NativeStorageGraphExecutable,DeviceId::cpu(),
        CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
    auto* owner=new(storage.data())RetainedParticipantGraphSourceSlots(source,count,memory);
    for(std::size_t i=0;i<count;++i)std::construct_at(owner->slots()+i);
    return {owner,[storage=std::move(storage)](RetainedParticipantGraphSourceSlots* value)mutable{
        value->~RetainedParticipantGraphSourceSlots();storage={};}};
}
RetainedParticipantGraphSourceSlots::~RetainedParticipantGraphSourceSlots(){
    for(std::size_t i=0;i<count_;++i)std::destroy_at(slots()+i);
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphSourceSlots::acquire(
    std::uint64_t source,const PreparedGPUContextUsePermit& original,void* stream,const PooledStorageOwnerProgram& program){
    std::lock_guard lock(mutex_);
    const auto coordinate=pooledStorageOwnerGeometry(program).coordinate;
    if(source!=sourceIdentity_ || coordinate>=count_ || original.memory().get()!=memory_.get() ||
       original.memory().owner_before(memory_) || memory_.owner_before(original.memory()) ||
       pooledStorageOwnerGeometry(program).owner!=PhysicalMemoryAllocatorIdentity{memory_->worldRank(),original.device()})
        throw std::invalid_argument("Original remote slot changed its source/PMA/owner coordinate");
    auto& slot=slots()[coordinate];if(slot.failure)std::rethrow_exception(slot.failure);
    try{
        if(!slot.owner)slot.owner=std::visit([&](const auto& contract){return RetainedParticipantGraphExecutable::create(original,stream,contract);},program);
        else{
            auto& owner=*slot.owner;
            if(slot.owner.use_count()!=1 || !owner.storageProgram_ || *owner.storageProgram_!=program ||
               owner.memory_.get()!=memory_.get() || owner.device_!=original.device() || owner.generation_!=original.generation() ||
               owner.stream_!=stream || owner.worker_!=original.worker() ||
               owner.worker_.owner_before(original.worker()) || original.worker().owner_before(owner.worker_))
                throw std::invalid_argument("Original remote source slot still has borrowers or changed program/worker/stream");
            owner.requireLiveOriginal();
            if(owner.phase_==RetainedParticipantGraphExecutable::Phase::DormantSourceSlot)owner.reacquireSourceDefinition();
            else if(owner.phase_!=RetainedParticipantGraphExecutable::Phase::Fresh)
                throw std::logic_error("Original remote source slot has not retired its previous definition");
        }
        return slot.owner;
    }catch(...){
        // Invalid overlapping/foreign borrows must not poison an otherwise
        // healthy source; an actual failed native construction remains sticky.
        if(!slot.owner || slot.owner->phase_==RetainedParticipantGraphExecutable::Phase::Failed)slot.failure=std::current_exception();
        throw;
    }
}
void RetainedParticipantGraphSourceSlots::release(std::uint64_t source,const PooledStorageOwnerProgram& program,
    const std::shared_ptr<RetainedParticipantGraphExecutable>& owner){
    std::lock_guard lock(mutex_);const auto coordinate=pooledStorageOwnerGeometry(program).coordinate;
    if(source!=sourceIdentity_ || coordinate>=count_ || !owner || slots()[coordinate].owner.get()!=owner.get() ||
       slots()[coordinate].owner.owner_before(owner) || owner.owner_before(slots()[coordinate].owner) ||
       !owner->storageProgram_ || *owner->storageProgram_!=program || slots()[coordinate].owner.use_count()!=2)
        throw std::invalid_argument("Original source definition retirement changed its exact slot/program or retained borrowers");
    owner->releaseSourceDefinition();
}
void RetainedParticipantGraphSourceSlots::abandon(std::uint64_t source,const PooledStorageOwnerProgram& program,
    const std::shared_ptr<RetainedParticipantGraphExecutable>& owner,std::exception_ptr failure){
    std::lock_guard lock(mutex_);const auto coordinate=pooledStorageOwnerGeometry(program).coordinate;
    if(source!=sourceIdentity_ || coordinate>=count_ || !owner || !failure ||
       slots()[coordinate].owner!=owner || !owner->storageProgram_ || *owner->storageProgram_!=program)
        throw std::logic_error("Failed source definition lost its exact original slot/diagnostic");
    owner->retireDefinition();
    slots()[coordinate].failure=std::move(failure);
}
void RetainedParticipantGraphExecutable::retireDefinition(){
    if(worker_->ownsCurrentThread())throw std::logic_error("Retained native definition requires its original off-worker terminal retirement");
    std::lock_guard definition(embeddedDefinitionMutex_);
    if(pending_ || embeddedBorrowers_.load(std::memory_order_acquire))throw std::logic_error("Retained native definition requires its original off-worker terminal retirement");
    if(phase_==Phase::Retired)return;
    if(capture_)worker_->submitAndWait([&]{capture_.reset();});
    if(sharedChild_) {
        if(!sharedChild_->embeddedBorrowers_.fetch_sub(1,std::memory_order_acq_rel))std::terminate();
        sharedChild_.reset();
    }
    if(calibrationTimingEvents_[0] || calibrationTimingEvents_[1])
        worker_->submitAndWait([&]{for(auto& event:calibrationTimingEvents_)if(event){worker_->destroyEvent(event);event=nullptr;}});
    nativeObjectLease_={};inspectionScratch_={};
    // Actual backend teardown is fatal on failed native destroy. Only its
    // successful return can close visible handles; physical backing stays held.
    if(nativeAttempted_){retirement_.completeNativeHandles();nativeAttempted_=false;}
    else pendingObligation_.reset();
    phase_=Phase::Retired;
}
RetainedParticipantGraphExecutable::~RetainedParticipantGraphExecutable()noexcept{
    try{retireDefinition();}catch(...){std::terminate();}
}
std::size_t PreparedServiceCalibrationSlot::physicalBytes(){
    return CPUStorageRoot<std::byte>(sizeof(PreparedServiceCalibrationSlot),
        PhysicalMemoryOwner::NativePreparationGraphExecutable,DeviceId::cpu()).physicalBytes();
}
std::shared_ptr<PreparedServiceCalibrationSlot> PreparedServiceCalibrationSlot::create(
    std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device){
    if(!memory || !device.is_gpu())throw std::invalid_argument("Prepared calibration slot requires its original GPU/source");
    auto storage=CPUStorageRoot<std::byte>(physicalBytes(),PhysicalMemoryOwner::NativePreparationGraphExecutable,
        DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
    auto* slot=new(storage.data())PreparedServiceCalibrationSlot(memory,device);
    return {slot,[storage=std::move(storage)](auto* value)mutable{value->~PreparedServiceCalibrationSlot();storage={};}};
}
std::shared_ptr<RetainedParticipantGraphExecutable> PreparedServiceCalibrationSlot::acquire(
    const PreparedGPUContextUsePermit& original,void* stream){
    if(original.device()!=device_ || original.memory().get()!=memory_.get() ||
       original.memory().owner_before(memory_) || memory_.owner_before(original.memory()) ||
       !original.worker() || original.worker()->ownsCurrentThread() || !stream || stream!=original.worker()->defaultStream())
        throw std::invalid_argument("Prepared calibration slot changed its original source/worker/stream");
    std::lock_guard lock(mutex_);
    if(closed_.load(std::memory_order_acquire))throw std::logic_error("Original prepared calibration source slot is retiring");
    if(failure_)std::rethrow_exception(failure_);
    if(owner_){
        if(owner_.use_count()!=1)throw std::logic_error("Original prepared calibration still has a live borrower");
        auto& owner=*owner_;
        original.requireWorker(owner.worker_);
        if(owner.memory_.get()!=memory_.get() || owner.memory_.owner_before(memory_) || memory_.owner_before(owner.memory_) ||
           owner.device_!=device_ || owner.generation_!=original.generation() || owner.stream_!=stream ||
           owner.purpose_!=RetainedParticipantGraphExecutable::DefinitionPurpose::PreparedServiceCalibration ||
           owner.storageProgram_ || owner.pending_ || owner.embeddedBorrowers_.load(std::memory_order_acquire) ||
           owner.cohortBound_.load(std::memory_order_acquire) || owner.phase_!=RetainedParticipantGraphExecutable::Phase::Fresh ||
           owner.lastSubmission_!=owner.calibrationRetiredSubmission_ || !owner.capture_ ||
           owner.capture_->nodeCount() || owner.capture_->hasExecutable())
            throw std::logic_error("Original prepared calibration has no completed reset definition or changed identity");
        owner.requireLiveOriginal();
    }else{
        try{owner_=RetainedParticipantGraphExecutable::createPreparedCalibration(original,stream);}
        catch(...){failure_=std::current_exception();throw;}
    }
    return owner_;
}
void PreparedServiceCalibrationSlot::retire(){
    std::lock_guard lock(mutex_);
    // The pool first closes new holder lookups. This same slot lock also closes
    // an already-issued off-pool borrow before it can enter a native creator.
    closeAcquisition();
    if(!owner_)return;
    if(owner_.use_count()!=1)throw std::logic_error("Original prepared calibration retirement has a live borrower");
    const auto phase=owner_->phase_.load(std::memory_order_acquire);
    if(phase!=RetainedParticipantGraphExecutable::Phase::Fresh && phase!=RetainedParticipantGraphExecutable::Phase::Retired &&
       phase!=RetainedParticipantGraphExecutable::Phase::Failed)
        throw std::logic_error("Original prepared calibration must reset its completed definition before final retirement");
    owner_->retireDefinition();
    owner_.reset();
}

}
