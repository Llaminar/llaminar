/**
 * @file ReadAheadNativeParentTerminal.cpp
 * @brief Pre-create one independently certified baseline event and retire it through its actual completion.
 *
 * No read-ahead-family or graph-executable bytes pay for this population. The
 * event has its own source-bound catalog and named same-PMA reservations. An
 * unsupported closure fails before any creator; no native bound is invented.
 */
#include "ReadAheadNativeParentTerminal.h"
#include "NativeAuxiliaryResourceCatalog.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeContextPreparation.h"
#include <exception>
#include <stdexcept>
namespace llaminar2 {
ReadAheadNativeParentTerminal::Certificate ReadAheadNativeParentTerminal::requireInstalled(DeviceId device){
    if(!device.is_gpu())throw std::invalid_argument("Read-ahead parent terminal requires an exact GPU");
    const auto selected=NativeAuxiliaryResourceCatalog::require(device,NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::readAheadParentTerminal()));
    const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(device);const auto key=original.originalBirthKey();
    if(original.generation()!=selected.contextGeneration)throw std::logic_error("Parent terminal selection changed original generation");
    ReadAheadNativeResourceClosure closure{device,original.generation(),key->physical_uuid,key->architecture,key->driver_version,key->runtime_version,key->backend_sha256,selected.resourceSource,selected.sourceIdentity};
    return {std::move(closure),selected.hostBytes,selected.deviceBytes,selected.evidence,
        selected.lifetime==NativePhysicalLifetime::ProcessRuntime?ReadAheadNativeRetirementPolicy::OriginalProcessRuntime:ReadAheadNativeRetirementPolicy::OriginalContextReset};
}
PhysicalMemoryPlan ReadAheadNativeParentTerminal::bom(DeviceId device,PhysicalMemoryResource host,PhysicalMemoryResource gpu){
    const auto certificate=requireInstalled(device);
    if(host.device!=DeviceId::cpu() || gpu.device!=device || host.world_rank!=gpu.world_rank)
        throw std::invalid_argument("Parent terminal changed its original CPU/GPU allocators");
    PhysicalMemoryPlanBuilder builder;builder.add(host,PhysicalMemoryOwner::NativeReadAheadParentTerminal,certificate.hostBytes);
    builder.add(gpu,PhysicalMemoryOwner::NativeReadAheadParentTerminal,certificate.deviceBytes);return builder.build();
}
std::shared_ptr<ReadAheadNativeParentTerminal> ReadAheadNativeParentTerminal::materialize(std::shared_ptr<PhysicalMemoryAuthority> memory,
    std::shared_ptr<IWorkerGPUContext> worker,DeviceId device){
    const auto prepared=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
    const auto original=prepared.provenance();
    original.requireWorker(worker);
    auto certificate=requireInstalled(device);
    return materializeCertified(std::move(memory),std::move(worker),std::move(certificate),original);
}
ReadAheadNativeParentTerminal::ReadAheadNativeParentTerminal(std::shared_ptr<PhysicalMemoryAuthority> memory,
    std::shared_ptr<IWorkerGPUContext> worker,Certificate certificate,PhysicalMemoryOwnerReservation host,std::optional<PhysicalMemoryOwnerReservation> device,std::optional<NativeContextRetirementSlot> retirement)
    :memory_(std::move(memory)),worker_(std::move(worker)),certificate_(std::move(certificate)),host_(std::move(host)),device_(std::move(device)),retirementSlot_(std::move(retirement)){}
std::shared_ptr<ReadAheadNativeParentTerminal> ReadAheadNativeParentTerminal::materializeCertified(std::shared_ptr<PhysicalMemoryAuthority> memory,
    std::shared_ptr<IWorkerGPUContext> worker,Certificate certificate,const GPUContextProvenance& original){
    original.requireWorker(worker);
    const auto& c=certificate.closure;
    if(original.device()!=c.device || original.generation()!=c.contextGeneration)
        throw std::invalid_argument("Parent terminal source certificate changed its exact original pool device/generation");
    if(isGraphCaptureActive() || !memory || !worker || !worker->isInitialized() || !c.device.is_gpu() ||
       worker->deviceOrdinal()!=c.device.ordinal || !c.contextGeneration || c.physicalUuid.empty() || c.architecture.empty() ||
       c.driver.empty() || c.runtime.empty() || c.backendSource.empty() || c.resourceSource.empty() || c.sourceIdentity.size()!=64 ||
       !certificate.hostBytes || certificate.proof.empty())throw std::invalid_argument("Parent terminal requires complete original certified closure and cold worker");
    if(certificate.retirement!=ReadAheadNativeRetirementPolicy::HandleDestruction &&
       certificate.retirement!=ReadAheadNativeRetirementPolicy::OriginalContextReset &&
       certificate.retirement!=ReadAheadNativeRetirementPolicy::OriginalProcessRuntime)
        throw std::invalid_argument("Unknown parent terminal retirement policy");
    const bool process=certificate.retirement==ReadAheadNativeRetirementPolicy::OriginalProcessRuntime;
    const auto proof=process?readAheadProcessPopulationIdentity(c,certificate.proof,certificate.hostBytes,certificate.deviceBytes):certificate.proof;
    std::optional<NativeContextRetirementSlot> retirement;
    if(certificate.retirement!=ReadAheadNativeRetirementPolicy::HandleDestruction)
        retirement.emplace(original.reserveNativeRetirementSlot(memory,proof));
    const bool borrowed=process && retirement->tryBorrowProcessRuntime("read-ahead-one-fixture-ordering-terminal");
    PhysicalMemoryOwnerReservation host;
    std::optional<PhysicalMemoryOwnerReservation> device;
    if(!borrowed){
        host=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadParentTerminal,certificate.hostBytes);
        if(certificate.deviceBytes)device.emplace(memory->reserveNewAllocations(c.device,PhysicalMemoryOwner::NativeReadAheadParentTerminal,certificate.deviceBytes));
        if(retirement){
            auto leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(
                memory,c.device,original.generation(),proof,std::move(host),std::move(device),
                process?NativePhysicalLifetime::ProcessRuntime:NativePhysicalLifetime::ContextGeneration));
            retirement->adopt(std::move(leaf));
        }
    }
    auto owner=std::shared_ptr<ReadAheadNativeParentTerminal>(new ReadAheadNativeParentTerminal(std::move(memory),std::move(worker),std::move(certificate),std::move(host),std::move(device),std::move(retirement)));
    owner->createNativeHandle();return owner;
}
void ReadAheadNativeParentTerminal::createNativeHandle(){worker_->submitAndWait([&]{if(isGraphCaptureActive())throw std::logic_error("Parent terminal birth cannot enter capture");if(retirementSlot_)retirementSlot_->armNativeAttempt();const auto event=worker_->createEvent(GPUEventPurpose::Ordering);if(reinterpret_cast<std::uintptr_t>(event)<=2)throw std::runtime_error("Parent terminal event creation failed");event_=event;});}
NativeAuxiliaryOwnershipKind ReadAheadNativeParentTerminal::ownershipKind()const noexcept {
    return std::holds_alternative<Certificate>(certificate_)?NativeAuxiliaryOwnershipKind::InstalledCertified:NativeAuxiliaryOwnershipKind::UnqualifiedDiagnostic;
}
std::shared_ptr<void> ReadAheadNativeParentTerminal::eventOwner(){if(!event_)throw std::logic_error("Parent terminal has no original native event");return {shared_from_this(),event_};}
ReadAheadNativeParentTerminal::~ReadAheadNativeParentTerminal()noexcept{
    if(!event_){if(retirementSlot_)retirementSlot_->completeNativeHandles();return;}
    try{worker_->submitAndWait([&]{if(isGraphCaptureActive())throw std::logic_error("Parent terminal destruction cannot enter capture");bool ready=false;if(!worker_->queryEventChecked(event_,ready) || !ready)throw std::runtime_error("Parent terminal still has actual native readers");worker_->destroyEvent(event_);event_=nullptr;});if(retirementSlot_)retirementSlot_->completeNativeHandles();}catch(...){std::terminate();}
}
} // namespace llaminar2
