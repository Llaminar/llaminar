/** @file NativeOrderingEventGroup.cpp
 * @brief Original-PMA group obligations around existing creators and physical retirement.
 *
 * Planning claims capacity before allocating roots, but a root-allocation failure
 * has not created native event backing. Keep those original tokens cancellable
 * until the first native creator can run; only that attempt transfers them to
 * the certified context/process retirement owner. No bytes are claimed twice.
 */
#include "NativeOrderingEventGroup.h"
#include "GPUCaptureDependencyRoster.h"
#include "NativeContextPreparation.h"
#include "BackendManager.h"
#include "IBackend.h"
#include "execution/moe/MoEOverlayInferenceOriginalSlot.h"
#include "NativeOrderingEventBudgetProjection.h"
#include "GPUDeviceContextPool.h"
#include "ReadAheadNativeResourceObligation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/Sha256.h"
#include <atomic>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
namespace llaminar2 {
NativeOrderingEventProgram NativeOrderingEventProgram::retainedCaptureTerminals(const MoEOverlayInferenceOriginalSlotDomain& domain){
    return retainedCaptureTerminals(domain.geometry());
}
NativeOrderingEventProgram NativeOrderingEventProgram::retainedCaptureTerminals(const MoEOverlayInferenceOriginalSlotGeometry& geometry){
    if(geometry.slotCount()<2 || geometry.slotCount()-1>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Retained capture terminals require original bounded protocol slots");
    return {Kind::RetainedCaptureTerminals,static_cast<std::uint32_t>(geometry.slotCount()-1),nullptr};
}

struct NativeOrderingEventGroup::State final {
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::optional<GPUContextProvenance> original;
    NativeOrderingEventProgram program;
    NativeContextRetirementSlot retirement;
    std::unique_ptr<ReadAheadNativeResourceObligation> pending;
    enum class Phase {Claimed,Attempted,ContextOwned,HandlesRetired};Phase phase=Phase::Claimed;
    /** @brief Retain one reserved slot and its original cancellable precreator claims. */
    State(std::shared_ptr<PhysicalMemoryAuthority> m,GPUContextProvenance p,NativeOrderingEventProgram v,
        NativeContextRetirementSlot s,std::unique_ptr<ReadAheadNativeResourceObligation> obligation)
        :memory(std::move(m)),original(std::move(p)),program(v),retirement(std::move(s)),pending(std::move(obligation)){}
};
PhysicalMemoryPlan NativeOrderingEventGroup::bom(NativeOrderingEventProgram p,PhysicalMemoryResource h,PhysicalMemoryResource g){return NativeOrderingEventBudgetProjection::installed().bom(p,h,g);}
PhysicalMemoryPlan NativeOrderingEventGroup::growthBOM(NativeOrderingEventProgram p,NativeOrderingEventProgram a,PhysicalMemoryResource h,PhysicalMemoryResource g){return NativeOrderingEventBudgetProjection::installed().growthBOM(p,a,h,g);}
NativeOrderingEventGroup::NativeOrderingEventGroup(std::unique_ptr<State> state):state_(std::move(state)){}
std::unique_ptr<NativeOrderingEventGroup> NativeOrderingEventGroup::claim(std::shared_ptr<PhysicalMemoryAuthority> memory,const GPUContextProvenance& original,NativeOrderingEventProgram program){
    return claimWithProjection(std::move(memory),original,program,NativeOrderingEventBudgetProjection::installed());
}
std::unique_ptr<NativeOrderingEventGroup> NativeOrderingEventGroup::claimWithProjection(std::shared_ptr<PhysicalMemoryAuthority> memory,const GPUContextProvenance& original,NativeOrderingEventProgram program,const NativeOrderingEventBudgetProjection& projection){
    original.requireWorker(original.worker());if(!memory || isGraphCaptureActive())throw std::invalid_argument("Ordering group requires original cold authority/provenance");
    projection.requireOriginalSource(memory,original);
    const auto selected=projection.select(program,original.device());
    if((selected.contextGeneration && selected.contextGeneration!=original.generation()) || !selected.hostBytes)throw std::invalid_argument("Ordering certificate changed original generation/bounds");
    const auto kind=program.kind()==NativeOrderingEventProgram::Kind::ReadAheadParentTerminal?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent;
    // Preserve the original PMA failure while identifying the exact closed
    // event population. No diagnostic reads live device state or retries a claim.
    const auto reserve=[&](DeviceId allocationDevice,std::size_t bytes){
        try { return memory->reserveNewAllocations(allocationDevice,kind,bytes); }
        catch(const std::logic_error& failure){
            throw std::logic_error("Native ordering reservation failed: program_kind="+
                std::to_string(static_cast<unsigned>(program.kind()))+
                " root_mask="+std::to_string(program.nonemptyRootMask())+
                " event_count="+std::to_string(program.orderingEvents())+
                " device="+original.device().toString()+
                " context_generation="+std::to_string(original.generation())+
                " allocation_device="+allocationDevice.toString()+
                " cause={"+failure.what()+"}");
        }
    };
    auto host=reserve(DeviceId::cpu(),selected.hostBytes);
    std::optional<PhysicalMemoryOwnerReservation> gpu;if(selected.deviceBytes)gpu.emplace(reserve(original.device(),selected.deviceBytes));
    const auto countIdentity=program.kind()==NativeOrderingEventProgram::Kind::RetainedCaptureTerminals
        ? ":events="+std::to_string(program.orderingEvents()):std::string{};
    const auto identity=sha256BytesHex(countIdentity+selected.sourceIdentity+":"+selected.resourceSource+":"+selected.evidence+":"+std::to_string(static_cast<unsigned>(program.kind()))+":"+std::to_string(program.nonemptyRootMask()));
    if(!identity)throw std::runtime_error("Ordering original certificate identity hashing failed");
    auto slot=original.reserveNativeRetirementSlot(memory,*identity);
    if(selected.lifetime==NativePhysicalLifetime::ProcessRuntime){
        static std::atomic<std::uint64_t> next{1};auto value=next.load(std::memory_order_relaxed);
        do{if(value==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Ordering instance identity exhausted");}while(!next.compare_exchange_weak(value,value+1,std::memory_order_relaxed));
        slot.prepareProcessRuntime("ordering-instance-v1:"+std::to_string(value)); // Per-instance; no incompatible population borrow.
    }
    auto leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(memory,original.device(),original.generation(),*identity,std::move(host),std::move(gpu),selected.lifetime));
    auto state=std::make_unique<State>(memory,original,program,std::move(slot),std::move(leaf));
    auto group=std::unique_ptr<NativeOrderingEventGroup>(new NativeOrderingEventGroup(std::move(state)));
    return group;
}
void NativeOrderingEventGroup::requireOriginal(const std::shared_ptr<PhysicalMemoryAuthority>& memory,const GPUContextProvenance& original,NativeOrderingEventProgram program)const{
    if(!state_ || !state_->original || !memory || memory.get()!=state_->memory.get() || memory.owner_before(state_->memory) || state_->memory.owner_before(memory) || state_->program!=program || original.device()!=state_->original->device() || original.generation()!=state_->original->generation())throw std::logic_error("Ordering group substituted original program/authority/generation");
    state_->original->requireWorker(original.worker());
}
void NativeOrderingEventGroup::beginNativeAttempt(){
    if(!state_ || state_->phase!=State::Phase::Claimed || isGraphCaptureActive())
        throw std::logic_error("Ordering program creation is cold and one-shot");
    // Transfer the SAME preclaimed tokens before the first SDK attempt. Even a
    // failing first creator may retain private driver backing until reset/exit.
    state_->retirement.adopt(std::move(state_->pending));
    state_->retirement.armNativeAttempt();state_->phase=State::Phase::Attempted;
}
void NativeOrderingEventGroup::delegateNativeHandlesToContext()noexcept{
    if(!state_ || state_->phase!=State::Phase::Attempted)std::terminate();
    state_->retirement.delegateHandlesToContext();
    state_->phase=State::Phase::ContextOwned;
    // Context owns this group. Dropping the creator borrow avoids a worker ->
    // roster -> worker cycle; the same original slot still guards reset.
    state_->original.reset();
}
void NativeOrderingEventGroup::completeNativeHandles()noexcept{
    if(!state_ || state_->phase==State::Phase::HandlesRetired)std::terminate();
    if(state_->phase==State::Phase::Claimed){
        // No creator was attempted: there is no opaque native allocation to
        // retain. Cancel the original reservation immediately; the empty slot
        // follows this holder's destruction without publishing a process root.
        state_->pending.reset();
    }else state_->retirement.completeNativeHandles();
    state_->phase=State::Phase::HandlesRetired;
}
NativeOrderingEventGroup::~NativeOrderingEventGroup()noexcept{if(state_ && (state_->phase==State::Phase::Attempted || state_->phase==State::Phase::ContextOwned))std::terminate();}
}

namespace llaminar2 {
GPUCaptureDependencyRoster::GPUCaptureDependencyRoster(const PreparedGPUContextUsePermit& p)
    :device_(p.device()),generation_(p.generation()),backend_(getBackendFor(p.device())),
     memory_(p.memory().get()),worker_(p.worker().get()){
    if(!backend_)throw std::runtime_error("Capture dependency roster lost original backend");
}
std::shared_ptr<GPUCaptureDependencyRoster> GPUCaptureDependencyRoster::create(
    const PreparedGPUContextUsePermit& p,std::size_t lanes,std::size_t groups){
    auto result=std::shared_ptr<GPUCaptureDependencyRoster>(new GPUCaptureDependencyRoster(p));
    result->grow(p,lanes,groups);return result;
}
void* GPUCaptureDependencyRoster::append(const PreparedGPUContextUsePermit& p){
    auto group=NativeOrderingEventGroup::claim(p.memory(),p.provenance(),NativeOrderingEventProgram::captureSubmissionRetirement());
    // Preallocate the owning slot before entering a native creator.
    ownership_.push_back(std::move(group));
    auto& owner=*ownership_.back();owner.beginNativeAttempt();
    void* event=nullptr;
    try{
        event=backend_->createEvent(device_.gpu_ordinal());
        if(!event)throw std::runtime_error("Capture dependency event creator returned null");
    }catch(...){owner.completeNativeHandles();ownership_.pop_back();throw;}
    owner.delegateNativeHandlesToContext();return event;
}
void GPUCaptureDependencyRoster::grow(const PreparedGPUContextUsePermit& p,std::size_t lanes,std::size_t groups){
    if(isGraphCaptureActive() || groups>lanes || p.device()!=device_ || p.generation()!=generation_ ||
       p.memory().get()!=memory_ || p.worker().get()!=worker_)
        throw std::invalid_argument("Capture dependency roster changed original cold source/geometry");
    p.requireWorker(p.worker());
    auto recording=acquire();
    // Capacity bookkeeping precedes each creator; a failed append never loses
    // a native handle through vector growth. Old captured addresses stay live.
    admissions_.reserve(lanes);shared_.reserve(groups);
    ownership_.reserve(2+std::max(lanes,admissions_.size())+std::max(groups,shared_.size()));
    if(!cold_publication_)cold_publication_=append(p);
    if(lanes && !producer_)producer_=append(p);
    while(admissions_.size()<lanes)admissions_.push_back(append(p));
    while(shared_.size()<groups)shared_.push_back(append(p));
}
GPUCaptureDependencyRoster::CaptureBorrow GPUCaptureDependencyRoster::acquire(){
    bool expected=false;
    if(!recording_.compare_exchange_strong(expected,true,std::memory_order_acq_rel))
        throw std::logic_error("Internal dependency roster has a concurrent or reentrant recording borrower");
    return CaptureBorrow(recording_);
}
GPUCaptureDependencyRoster::CaptureBorrow GPUCaptureDependencyRoster::borrow(std::size_t lanes,std::size_t groups){
    auto recording=acquire();
    if(!isGraphCaptureActive() || !lanes || lanes>admissions_.size() || groups>shared_.size() || !producer_)
        throw std::logic_error("Internal dependency events require their prepared native capture interval");
    return recording;
}
void GPUCaptureDependencyRoster::requireOriginalMemory(const std::shared_ptr<PhysicalMemoryAuthority>& memory)const{
    if(!memory || memory.get()!=memory_)throw std::logic_error("Batch metadata changed its original memory authority");
    const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device_,memory);
    if(original.generation()!=generation_ || original.worker().get()!=worker_)
        throw std::logic_error("Batch metadata changed its original physical context generation");
}
GPUCaptureDependencyRoster::~GPUCaptureDependencyRoster()noexcept{
    if(recording_.exchange(true,std::memory_order_acq_rel))std::terminate();
    std::unique_lock publication(cold_publication_mutex_,std::try_to_lock);if(!publication.owns_lock())std::terminate();
    if(cold_publication_)backend_->destroyEvent(cold_publication_,device_.gpu_ordinal());
    if(producer_)backend_->destroyEvent(producer_,device_.gpu_ordinal());
    for(auto* event:admissions_)backend_->destroyEvent(event,device_.gpu_ordinal());
    for(auto* event:shared_)backend_->destroyEvent(event,device_.gpu_ordinal());
    for(auto& owner:ownership_)owner->completeNativeHandles();
}
}
