/**
 * @file RetainedParticipantGraphExecutable.cpp
 * @brief Original General graph device tokens retained through process reclamation.
 *
 * This owner adds no resource allowance, stream, event or live byte counter.
 * The original PMA reservation moves once into its canonical retirement slot.
 * Partial native creation retains possible backing even when no handle escaped.
 */
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
        if(owner_.phase_!=Phase::Recording)throw std::logic_error("Original participant has no active recording");
        try{owner_.requireRecordingAuthority();if(!native().endCapture()){owner_.phase_=Phase::Failed;return false;}owner_.phase_=Phase::Recorded;return true;}catch(...){owner_.phase_=Phase::Failed;throw;}
    }
    /** @copydoc IGPUGraphCapture::instantiate */
    bool instantiate()override{
        if(owner_.phase_!=Phase::Recorded)throw std::logic_error("Original participant definition must seal once before submission");
        try{owner_.requireRecordingAuthority();
            if(owner_.remoteProgram_){
                GPUFlatGraphStructure observed;std::string error;
                if(!native().inspectFlatGraphStructure(owner_.remoteProgram_->limits(),
                    {owner_.inspectionScratch_.data(),owner_.inspectionScratch_.size()},observed,&error))
                    throw std::runtime_error("Original remote graph failed its declared flat structural seal: "+error);
                const auto& declared=owner_.remoteProgram_->limits();
                if(observed.nodes!=declared.maximumNodes || observed.edges!=declared.maximumEdges ||
                    !observed.linearChain || observed.kinds[static_cast<unsigned>(GPUFlatGraphNodeKind::Kernel)]!=observed.nodes)
                    throw std::runtime_error("Original remote graph differs from its exact source-authored kernel chain");
            }
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
            if(!native().prepareRuntimeContextStorage(memory)){owner_.phase_=Phase::Failed;return false;}return true;
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
    if(!host.valid() || host.device!=DeviceId::cpu() || !gpu.valid() || !gpu.device.is_gpu() || host.world_rank!=gpu.world_rank)
        throw std::invalid_argument("General graph BOM requires original CPU0 and same-rank GPU observations");
    const auto hostUnit=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(gpu.device);
    const auto deviceUnit=GPUGraphMemoryContract::reservationBytesPerExecutable(gpu.device,GPUGraphExecutableClass::General);
    const auto ownerBytes=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),PhysicalMemoryOwner::NativeGraphExecutable,DeviceId::cpu()).physicalBytes();
    const auto nativeBytes=nativeGraphCaptureObjectBytes(gpu.device);
    if(!nativeBytes || ownerBytes>std::numeric_limits<std::size_t>::max()-nativeBytes ||
       ownerBytes+nativeBytes>std::numeric_limits<std::size_t>::max()-sizeof(RecordingCapture) ||
       hostUnit>std::numeric_limits<std::size_t>::max()-(ownerBytes+nativeBytes+sizeof(RecordingCapture)))
        throw std::overflow_error("General graph known host obligation overflows");
    const auto completeHost=hostUnit+ownerBytes+nativeBytes+sizeof(RecordingCapture);
    if(slots>std::numeric_limits<std::size_t>::max()/completeHost || slots>std::numeric_limits<std::size_t>::max()/deviceUnit)
        throw std::overflow_error("General graph declared slot allowance overflows");
    PhysicalMemoryPlanBuilder plan;
    if(slots){plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,slots*completeHost);
        plan.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,slots*deviceUnit);}
    return plan.build();
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::hostBOMWithSelection(const PooledRemoteGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu,const std::optional<NativeAuxiliaryResourceSelection>& selected,
    std::size_t nativeObjectBytes){
    const auto& geometry=program.geometry();
    if(!host.valid() || host.device!=DeviceId::cpu() || !gpu.valid() ||
        PhysicalMemoryAllocatorIdentity{gpu.world_rank,gpu.device}!=geometry.owner || host.world_rank!=gpu.world_rank || !nativeObjectBytes)
        throw std::invalid_argument("Remote graph host BOM requires original CPU0 and exact addressed GPU observations");
    PhysicalMemoryPlanBuilder plan;
    plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,
        CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),PhysicalMemoryOwner::NativeGraphExecutable,DeviceId::cpu()).physicalBytes());
    plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,
        CPUStorageRoot<std::byte>(GPUFlatGraphScratchLayout::fromLimits(program.limits()).bytes,PhysicalMemoryOwner::NativeGraphExecutable,DeviceId::cpu()).physicalBytes());
    plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,nativeObjectBytes);
    plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,sizeof(RecordingCapture));
    if(selected){
        if(!selected->hostBytes || selected->deviceBytes || selected->lifetime!=NativePhysicalLifetime::ProcessRuntime ||
            !selected->program || *selected->program!=NativeAuxiliaryResourceProgram::pooledRemoteOwner(program) ||
            !selected->remoteFamily || !selected->remoteFamily->accepts(program.limits(),PooledRemoteGraphProgramClass::argumentLayout()))
            throw std::invalid_argument("Remote graph host program cannot borrow device bytes or a shorter untested lifetime");
        plan.add(host,PhysicalMemoryOwner::NativeGraphExecutable,selected->hostBytes);
    }
    return plan.build();
}
PhysicalMemoryPlan RetainedParticipantGraphExecutable::hostBOM(const PooledRemoteGraphProgram& program,
    PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    const auto selected=NativeAuxiliaryResourceCatalog::require(program.geometry().owner.device,
        NativeAuxiliaryResourceProgram::pooledRemoteOwner(program));
    return hostBOMWithSelection(program,host,gpu,selected,nativeGraphCaptureObjectBytes(program.geometry().owner.device));
}
std::shared_ptr<RetainedParticipantGraphExecutable> RetainedParticipantGraphExecutable::create(
    const PreparedGPUContextUsePermit& original,void* stream,const PooledRemoteGraphProgram& program){
    if(program.geometry().owner.device!=original.device())throw std::invalid_argument("Remote graph host program has a foreign original GPU");
    auto selected=NativeAuxiliaryResourceCatalog::require(original.device(),NativeAuxiliaryResourceProgram::pooledRemoteOwner(program));
    return createBound(original,stream,HostProgramBinding{program,std::move(selected),
        nativeGraphCaptureObjectBytes(original.device()),GraphHostQualification::InstalledCertified});
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
    const PreparedGPUContextUsePermit& original,void* stream,std::optional<HostProgramBinding> host,GeneralHostOwnership generalHost){
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
    if(originalGeneral && host)throw std::invalid_argument("General and authored remote host obligations cannot be double charged");
    std::string policy=policyIdentity();
    std::optional<PhysicalMemoryOwnerReservation> opaqueHost;
    AlignedVector<std::byte> ownerStorage,scratch;
    PhysicalMemoryAllocationLease nativeObject,recordingObject;
    if(originalGeneral){
        const auto bytes=GPUGraphMemoryContract::generalHostReservationBytesPerExecutable(original.device());
        opaqueHost.emplace(memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,bytes));
        policy+="/general-host-process-allowance="+std::to_string(bytes);
        ownerStorage=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),PhysicalMemoryOwner::NativeGraphExecutable,DeviceId::cpu(),
            CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        nativeObject=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,nativeGraphCaptureObjectBytes(original.device()));
        recordingObject=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,sizeof(RecordingCapture));
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
            opaqueHost.emplace(memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,selected.hostBytes));
            policy+="/remote-host/"+host->program.qualificationClass().identity()+"/"+selected.resourceSource+"/"+selected.evidence+"/"+selected.sourceIdentity+"/host="+std::to_string(selected.hostBytes)+"/"+selected.remoteFamily->identity();
        }else{
            if(host->qualification!=GraphHostQualification::UnqualifiedDiagnostic)
                throw std::invalid_argument("Remote graph requires installed host authority; diagnostics must be explicit");
            policy+="/explicit-unqualified-remote-host/"+host->program.qualificationClass().identity();
        }
        // Known buffers/objects remain exact and distinct from the opaque SDK reservation.
        ownerStorage=CPUStorageRoot<std::byte>(sizeof(RetainedParticipantGraphExecutable),PhysicalMemoryOwner::NativeGraphExecutable,
            DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        const auto layout=GPUFlatGraphScratchLayout::fromLimits(host->program.limits());
        scratch=CPUStorageRoot<std::byte>(layout.bytes,PhysicalMemoryOwner::NativeGraphExecutable,DeviceId::cpu(),
            CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        nativeObject=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,host->nativeObjectBytes);
        recordingObject=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,sizeof(RecordingCapture));
    }
    auto reservation=memory->reserveNewAllocations(original.device(),PhysicalMemoryOwner::NativeGraphExecutable,
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
        if(host){result->remoteProgram_=host->program;result->hostQualification_=host->qualification;}
        else result->hostQualification_=GraphHostQualification::GeneralProcessAllowance;
        result->inspectionScratch_=std::move(scratch);result->nativeObjectLease_=std::move(nativeObject);result->recordingObjectLease_=std::move(recordingObject);
    }else result=std::shared_ptr<RetainedParticipantGraphExecutable>(new RetainedParticipantGraphExecutable(
        worker,memory,original.generation(),original.device(),stream,std::move(retirement),std::move(pending)));
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
void RetainedParticipantGraphExecutable::retireDefinition(){
    if(worker_->ownsCurrentThread() || pending_)throw std::logic_error("Retained native definition requires its original off-worker terminal retirement");
    if(phase_==Phase::Retired)return;
    if(capture_)worker_->submitAndWait([&]{capture_.reset();});
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
}
