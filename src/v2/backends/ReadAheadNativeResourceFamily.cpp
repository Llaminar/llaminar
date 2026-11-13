/**
 * @file ReadAheadNativeResourceFamily.cpp
 * @brief Pre-create certified same-authority commitments and exact event-owned retirement.
 *
 * The compiled catalog prices supported native resource families separately
 * from their original physical provenance. Process-retained native backing
 * stays charged after handle destruction and context reset; a successful API
 * call is not evidence of physical pool release.
 * Device-free replay injects a private certificate solely to test lifecycle;
 * production cannot supply a byte allowance or stamp its own certificate.
 */
#include "ReadAheadNativeResourceFamily.h"
#include "NativeAuxiliaryResourceCatalog.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeContextPreparation.h"
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
namespace llaminar2 {
namespace {
/** @brief Reject incomplete source/runtime identity without invoking the native runtime. */
void requireClosure(const ReadAheadNativeResourceClosure& c){
    if(!c.device.is_gpu() || !c.contextGeneration || c.physicalUuid.empty() || c.architecture.empty() ||
       c.driver.empty() || c.runtime.empty() || c.backendSource.empty() || c.resourceSource.empty() || c.sourceIdentity.size()!=64)
        throw std::invalid_argument("Read-ahead native family requires complete original device/context/runtime/source identity");
}
/** @brief Native default-stream sentinels are never valid retained resource identities. */
bool exactHandle(void* value){return reinterpret_cast<std::uintptr_t>(value)>2;}
}
std::string readAheadProcessPopulationIdentity(const ReadAheadNativeResourceClosure& c,const std::string& proof,std::size_t host,std::size_t device){
    std::string result;
    for(const auto& value:{proof,c.device.toString(),c.physicalUuid,c.architecture,c.driver,c.runtime,c.backendSource,c.resourceSource,c.sourceIdentity,std::to_string(host),std::to_string(device)})
        result+=std::to_string(value.size())+":"+value;
    return result;
}
ReadAheadNativeResourceCertificate ReadAheadNativeResourceFamily::requireInstalledCertificate(DeviceId device){
    if(!device.is_gpu())throw std::invalid_argument("Read-ahead resource certification requires an exact GPU");
    const auto selected=NativeAuxiliaryResourceCatalog::require(device,NativeAuxiliaryResourceProgram::readAheadSide());
    const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);const auto key=original.originalBirthKey();
    if(original.generation()!=selected.contextGeneration)throw std::logic_error("Read-ahead selection changed original context generation");
    ReadAheadNativeResourceClosure closure{device,original.generation(),key->physical_uuid,key->architecture,key->driver_version,key->runtime_version,key->backend_sha256,selected.resourceSource,selected.sourceIdentity};
    return {std::move(closure),selected.hostBytes,selected.deviceBytes,selected.evidence,
        selected.lifetime==NativePhysicalLifetime::ProcessRuntime?ReadAheadNativeRetirementPolicy::OriginalProcessRuntime:ReadAheadNativeRetirementPolicy::OriginalContextReset};
}
PhysicalMemoryPlan ReadAheadNativeResourceFamily::bom(DeviceId selected,
    PhysicalMemoryResource host,PhysicalMemoryResource device){
    if(!selected.is_gpu())throw std::invalid_argument("Read-ahead BOM requires an exact GPU endpoint");
    const auto allowance=NativeAuxiliaryResourceCatalog::allowance(selected.type,NativeAuxiliaryResourceProgram::readAheadSide());
    if(!host.valid() || !device.valid() || host.device!=DeviceId::cpu() || device.device!=selected || host.world_rank!=device.world_rank)
        throw std::invalid_argument("Read-ahead opaque-family BOM changed original host/device allocators");
    PhysicalMemoryPlanBuilder builder;builder.add(host,PhysicalMemoryOwner::NativeReadAheadFamily,allowance.hostBytes);
    builder.add(device,PhysicalMemoryOwner::NativeReadAheadFamily,allowance.deviceBytes);return builder.build();
}
std::shared_ptr<ReadAheadNativeResourceFamily> ReadAheadNativeResourceFamily::materialize(
    std::shared_ptr<PhysicalMemoryAuthority> memory,std::shared_ptr<IWorkerGPUContext> worker,
    DeviceId device,void* primaryStream){
    const auto prepared=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
    const auto original=prepared.provenance();
    original.requireWorker(worker);
    auto certificate=requireInstalledCertificate(device);
    ReadAheadNativeResourceRequest request{certificate.closure_,primaryStream};
    return materializeCertified(std::move(memory),std::move(worker),std::move(request),std::move(certificate),original);
}
ReadAheadNativeResourceFamily::ReadAheadNativeResourceFamily(std::shared_ptr<PhysicalMemoryAuthority> memory,
    std::shared_ptr<IWorkerGPUContext> worker,ReadAheadNativeResourceRequest request,ReadAheadNativeResourceCertificate certificate,
    PhysicalMemoryOwnerReservation host,PhysicalMemoryOwnerReservation device,std::optional<NativeContextRetirementSlot> retirement)
    :memory_(std::move(memory)),worker_(std::move(worker)),request_(std::move(request)),certificate_(std::move(certificate)),
     hostCommitment_(std::move(host)),deviceCommitment_(std::move(device)),retirementSlot_(std::move(retirement)){}
std::shared_ptr<ReadAheadNativeResourceFamily> ReadAheadNativeResourceFamily::materializeCertified(
    std::shared_ptr<PhysicalMemoryAuthority> memory,std::shared_ptr<IWorkerGPUContext> worker,
    ReadAheadNativeResourceRequest request,ReadAheadNativeResourceCertificate certificate,const GPUContextProvenance& original){
    original.requireWorker(worker);
    if(original.device()!=request.closure.device || original.generation()!=request.closure.contextGeneration)
        throw std::invalid_argument("Read-ahead source certificate changed its exact original pool device/generation");
    requireClosure(request.closure);
    if(isGraphCaptureActive() || !memory || !worker || !exactHandle(request.primaryStream) ||
       !worker->isInitialized() || worker->deviceOrdinal()!=request.closure.device.gpu_ordinal() ||
       certificate.closure_!=request.closure || !certificate.hostBytes() || !certificate.deviceBytes() || certificate.proofIdentity().empty())
        throw std::invalid_argument("Read-ahead resource birth requires original cold worker, exact primary stream and compatible certified closure");
    if(certificate.retirement_!=ReadAheadNativeRetirementPolicy::HandleDestruction &&
       certificate.retirement_!=ReadAheadNativeRetirementPolicy::OriginalContextReset &&
       certificate.retirement_!=ReadAheadNativeRetirementPolicy::OriginalProcessRuntime)
        throw std::invalid_argument("Unknown source-certified native retirement policy");
    const bool process=certificate.retirement_==ReadAheadNativeRetirementPolicy::OriginalProcessRuntime;
    const auto proof=process?readAheadProcessPopulationIdentity(request.closure,certificate.proofIdentity(),certificate.hostBytes(),certificate.deviceBytes()):certificate.proofIdentity();
    std::optional<NativeContextRetirementSlot> retirement;
    if(certificate.retirement_!=ReadAheadNativeRetirementPolicy::HandleDestruction)
        retirement.emplace(original.reserveNativeRetirementSlot(memory,proof));
    const bool borrowed=process && retirement->tryBorrowProcessRuntime("read-ahead-one-stream-two-ordering-events");
    PhysicalMemoryOwnerReservation host,device;
    if(!borrowed){
        host=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadFamily,certificate.hostBytes());
        device=memory->reserveNewAllocations(request.closure.device,PhysicalMemoryOwner::NativeReadAheadFamily,certificate.deviceBytes());
        if(retirement){
            auto leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(
                memory,request.closure.device,original.generation(),proof,std::move(host),
                std::optional<PhysicalMemoryOwnerReservation>(std::move(device)),
                process?NativePhysicalLifetime::ProcessRuntime:NativePhysicalLifetime::ContextGeneration));
            retirement->adopt(std::move(leaf));
        }
    }
    auto family=std::shared_ptr<ReadAheadNativeResourceFamily>(new ReadAheadNativeResourceFamily(
        std::move(memory),std::move(worker),std::move(request),std::move(certificate),std::move(host),std::move(device),std::move(retirement)));
    family->self_=family;
    try{family->createNativeHandles();}catch(...){family->destroyHandles();throw;}
    return family;
}
void ReadAheadNativeResourceFamily::createNativeHandles(){
    worker_->submitAndWait([&]{
        if(isGraphCaptureActive())throw std::logic_error("Read-ahead resource birth cannot enter native capture");
        if(retirementSlot_)retirementSlot_->armNativeAttempt();
        const auto side=worker_->createStream();
        if(!exactHandle(side) || side==primaryStream())throw std::runtime_error("Read-ahead native stream creation failed or aliased primary");
        side_=side;
        const auto fork=worker_->createEvent(GPUEventPurpose::Ordering);
        if(!exactHandle(fork))throw std::runtime_error("Read-ahead native fork event creation failed");
        fork_=fork;
        const auto join=worker_->createEvent(GPUEventPurpose::Ordering);
        if(!exactHandle(join) || join==fork)throw std::runtime_error("Read-ahead native join event creation failed or aliased fork");
        join_=join;
    });
}
NativeAuxiliaryOwnershipKind ReadAheadNativeResourceFamily::ownershipKind()const noexcept {
    return std::holds_alternative<ReadAheadNativeResourceCertificate>(certificate_)
        ?NativeAuxiliaryOwnershipKind::InstalledCertified:NativeAuxiliaryOwnershipKind::UnqualifiedDiagnostic;
}
void ReadAheadNativeResourceFamily::destroyHandles()noexcept{
    if(!side_ && !fork_ && !join_)return;
    try{worker_->submitAndWait([&]{
        if(isGraphCaptureActive())throw std::logic_error("Read-ahead native destruction cannot enter capture");
        if(join_){worker_->destroyEvent(join_);join_=nullptr;}
        if(fork_){worker_->destroyEvent(fork_);fork_=nullptr;}
        if(side_){worker_->destroyStream(side_);side_=nullptr;}
    });}catch(...){std::terminate();} // Do not release obligations after failed native cleanup.
}
ReadAheadNativeResourceFamily::~ReadAheadNativeResourceFamily()noexcept{
    if(submission_!=Submission::Idle)std::terminate();
    destroyHandles();
    if(retirementSlot_)retirementSlot_->completeNativeHandles();
    // Handle-scoped certificates release their original members here. The
    // context-scoped slot retains the SAME moved tokens through physical reset;
    // native handle success alone never returns its capacity to another owner.
}
std::uint64_t ReadAheadNativeResourceFamily::captureIdentity()const noexcept{
    std::uint64_t value=1469598103934665603ULL;const auto mix=[&](std::uint64_t input){value=(value^input)*1099511628211ULL;};
    mix(request_.closure.contextGeneration);mix(static_cast<std::uint64_t>(device().type));mix(device().gpu_ordinal());
    for(const auto* text:{&request_.closure.physicalUuid,&request_.closure.architecture,&request_.closure.driver,&request_.closure.runtime,&request_.closure.backendSource,&request_.closure.resourceSource,&request_.closure.sourceIdentity}){mix(text->size());for(const unsigned char c:*text)mix(c);}
    for(void* handle:{request_.primaryStream,side_,fork_,join_})mix(reinterpret_cast<std::uintptr_t>(handle));
    mix(static_cast<std::uint64_t>(ownershipKind()));
    if(const auto* installed=std::get_if<ReadAheadNativeResourceCertificate>(&certificate_)) {
        mix(installed->proofIdentity().size());for(const unsigned char c:installed->proofIdentity())mix(c);
        mix(installed->hostBytes());mix(installed->deviceBytes());mix(static_cast<std::uint64_t>(installed->retirement_));
    }else {mix(0xd1a690571cULL);} // Explicit unqualified provenance cannot alias an installed identity.
    return value?value:1;
}
void ReadAheadNativeResourceFamily::bindTerminal(std::shared_ptr<void> event){
    if(isGraphCaptureActive())throw std::logic_error("Read-ahead resource submission lifecycle cannot enter native capture");
    std::lock_guard lock(lifecycle_);
    if(!exactHandle(event.get()) || event.get()==fork_ || event.get()==join_ || submission_!=Submission::Idle)
        throw std::logic_error("Read-ahead terminal event must bind once to its original independent parent lifetime");
    if(terminal_){
        if(terminal_.get()==event.get() && !terminal_.owner_before(event) && !event.owner_before(terminal_))return;
        throw std::logic_error("Read-ahead serialized variants replaced their original terminal event owner");
    }
    terminal_=std::move(event);
}
void ReadAheadNativeResourceFamily::beginSubmission(){
    if(isGraphCaptureActive())throw std::logic_error("Read-ahead resource submission lifecycle cannot enter native capture");
    std::lock_guard lock(lifecycle_);
    if(!terminal_ || submission_!=Submission::Idle)throw std::logic_error("Read-ahead parent submission requires bound terminal and retired previous readers");
    if(submissionSequence_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Read-ahead parent submission identity exhausted");
    ++submissionSequence_;
    submission_=Submission::Launched;
}
ReadAheadNativeResourceFamily::PublishedTerminal::PublishedTerminal(const std::shared_ptr<ReadAheadNativeResourceFamily>& family,
    std::shared_ptr<void> event,void* producer):family_(family),event_(std::move(event)),producer_(producer){
    if(!family)throw std::invalid_argument("Original terminal publication has no resource owner");
    std::lock_guard lock(family->lifecycle_);
    if(family->self_.owner_before(family) || family.owner_before(family->self_))
        throw std::logic_error("Original terminal publisher substituted resource holder control block");
    if(family->submission_!=Submission::Launched)throw std::logic_error("Original terminal publisher has no active read-ahead request");
    submission_=family->submissionSequence_;
}
void ReadAheadNativeResourceFamily::acceptTerminalPublication(const PublishedTerminal& proof){
    if(isGraphCaptureActive())throw std::logic_error("Read-ahead terminal publication cannot enter native capture");
    std::lock_guard lock(lifecycle_);
    const auto original=proof.family_.lock();
    if(submission_!=Submission::Launched || !original || original.get()!=this || proof.submission_!=submissionSequence_ || proof.family_.owner_before(self_) || self_.owner_before(proof.family_) ||
       proof.producer_!=primaryStream() || !terminal_ || proof.event_.get()!=terminal_.get() ||
       proof.event_.owner_before(terminal_) || terminal_.owner_before(proof.event_))
        throw std::logic_error("Read-ahead original terminal publication is stale, duplicated or substituted");
    submission_=Submission::TerminalRecorded;
}
void ReadAheadNativeResourceFamily::recordTerminal(){
    if(isGraphCaptureActive())throw std::logic_error("Read-ahead resource submission lifecycle cannot enter native capture");
    std::lock_guard lock(lifecycle_);
    if(submission_!=Submission::Launched)throw std::logic_error("Read-ahead terminal record requires its actual active parent submission");
    worker_->submitAndWait([&]{if(!worker_->recordEventChecked(terminal_.get(),primaryStream()))throw std::runtime_error("Read-ahead parent terminal event publication failed");});
    submission_=Submission::TerminalRecorded;
}
bool ReadAheadNativeResourceFamily::observeTerminal(){
    if(isGraphCaptureActive())throw std::logic_error("Read-ahead resource submission lifecycle cannot enter native capture");
    std::lock_guard lock(lifecycle_);
    if(submission_==Submission::Idle)return true;
    if(submission_!=Submission::TerminalRecorded)throw std::logic_error("Read-ahead terminal observation precedes actual parent event publication");
    bool ready=false;worker_->submitAndWait([&]{if(!worker_->queryEventChecked(terminal_.get(),ready))throw std::runtime_error("Read-ahead original parent terminal query failed");});
    if(ready)submission_=Submission::Idle;return ready;
}
bool ReadAheadNativeResourceFamily::hasOutstandingSubmission()const{
    std::lock_guard lock(lifecycle_);return submission_!=Submission::Idle;
}
} // namespace llaminar2
