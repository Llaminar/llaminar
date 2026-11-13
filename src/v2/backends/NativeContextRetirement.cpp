/**
 * @file NativeContextRetirement.cpp
 * @brief Event-aware original reservation transfer and physical-reset discharge.
 *
 * Slots are allocated before native creators. Adoption moves the original PMA
 * tokens; lifecycle transitions never derive another budget or claim memory
 * growth after the fact. A live handle or missing reset is fatal. Certified
 * process roots stay in the original process authority after reset, with one
 * serialized same-PMA borrower per bounded population and no copied balance.
 */
#include "NativeContextRetirement.h"
#include "GPUDeviceContextPool.h"
#include "GPUDeviceProcessLifetime.h"
#include "collective/NativeStoragePreparationPmaBinding.h"
#include "NativeContextOpaqueContract.h"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <limits>
#include <utility>
namespace llaminar2 {
/** @brief One actual lifecycle, not a shadow count of bytes. */
struct NativeContextRetirementSlotState {
    enum class Phase {Reserved,NativeAttempted,ContextOwnedHandles,HandlesRetired};
    std::string certificate;
    std::unique_ptr<IContextNativeRetirementObligation> obligation;
    Phase phase=Phase::Reserved;
    std::shared_ptr<NativeContextRetirementSlotState> original_process_root;
    std::string process_population;
    bool borrowed_slot=true;
    bool process_use_active=false; // Guarded by the original process authority mutex.
    std::shared_ptr<PhysicalMemoryAuthority> original_memory;
    DeviceId original_device=DeviceId::invalid();
    // Baseline context-birth roots have no auxiliary slot or fake pool generation.
    std::shared_ptr<NativeStoragePreparationPmaBinding> preparing_source;
    std::shared_ptr<PhysicalMemoryOpaqueRuntimeCommitment> preparing_opaque;
};
NativeContextRetirementOwner::NativeContextRetirementOwner(DeviceId device,std::uint64_t generation)
    :device_(device),generation_(generation){if(!device.is_gpu() || !generation)throw std::invalid_argument("Native retirement owner needs original physical generation");}
NativeContextRetirementOwner::~NativeContextRetirementOwner()noexcept{
    if(!slots_.empty())std::terminate();
}
NativeContextRetirementSlot NativeContextRetirementOwner::reserve(const std::shared_ptr<PhysicalMemoryAuthority>& memory,const std::string& certificate){
    if(!memory || certificate.empty())throw std::invalid_argument("Native retirement slot needs original authority and certificate");
    std::lock_guard lock(mutex_);
    if(retired_ || (authority_ && (authority_.get()!=memory.get() || authority_.owner_before(memory) || memory.owner_before(authority_))))
        throw std::logic_error("Native retirement slot changed original authority or retired generation");
    auto state=std::make_shared<NativeContextRetirementSlotState>();state->certificate=certificate;state->original_memory=memory;state->original_device=device_;
    slots_.push_back(state);authority_=memory;
    return {shared_from_this(),std::move(state)};
}
bool NativeContextRetirementOwner::empty()const noexcept{std::lock_guard lock(mutex_);return slots_.empty();}
bool NativeContextRetirementOwner::discharged()const noexcept{std::lock_guard lock(mutex_);return retired_;}
void NativeContextRetirementOwner::requireHandlesRetired()const{
    std::lock_guard lock(mutex_);
    // Only closed backend stream/roster owners may delegate live native handles.
    // Its record dies inside context cleanup; physical-reset validation below
    // still requires actual destruction and release of this same slot.
    for(const auto& slot:slots_)if((!slot->obligation && !slot->original_process_root) ||
        (slot->phase!=NativeContextRetirementSlotState::Phase::ContextOwnedHandles &&
         (slot->phase!=NativeContextRetirementSlotState::Phase::HandlesRetired || slot->borrowed_slot)))
        throw std::logic_error("Original native retirement slot still owns live or borrowed handles");
}
void NativeContextRetirementOwner::physicallyRetired()noexcept{
    std::lock_guard lock(mutex_);
    if(retired_)std::terminate();
    for(const auto& slot:slots_)if((!slot->obligation && !slot->original_process_root) || slot->phase!=NativeContextRetirementSlotState::Phase::HandlesRetired || slot->borrowed_slot)std::terminate();
    for(const auto& slot:slots_){
        if(slot->original_process_root)GPUDeviceProcessLifetime::finishProcessUse(slot->original_process_root);
        else if(slot->obligation->physicalLifetime()==NativePhysicalLifetime::ProcessRuntime)GPUDeviceProcessLifetime::finishProcessUse(slot);
    }
    retired_=true;slots_.clear();authority_.reset();
}
void NativeContextRetirementOwner::requireOriginalProcessAuthority(const std::shared_ptr<NativeContextRetirementSlotState>& root,
    DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority>& memory){
    if(!device.is_gpu())throw std::invalid_argument("Process source authority requires an exact selected GPU");
    requireOriginalAuthority(root,memory);
}
void NativeContextRetirementOwner::requireOriginalAuthority(const std::shared_ptr<NativeContextRetirementSlotState>& root,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory){
    const auto& original=root->original_memory;
    if(!memory || original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original))
        throw std::logic_error("Original process root prohibits independent source authority adoption");
}
void NativeContextRetirementOwner::requireUniqueProcessRoot(const std::shared_ptr<NativeContextRetirementSlotState>& prior,
    const std::shared_ptr<NativeContextRetirementSlotState>& root){
    requireOriginalProcessAuthority(prior,root->original_device,root->original_memory);
    if(prior->original_device==root->original_device && prior->process_population==root->process_population)
        throw std::logic_error("Bounded original process population already published; borrow its original commitment");
}
std::shared_ptr<NativeContextRetirementSlotState> NativeContextRetirementOwner::preparingProcessRoot(
    const std::shared_ptr<NativeStoragePreparationPmaBinding>& source,
    const std::shared_ptr<PhysicalMemoryOpaqueRuntimeCommitment>& opaque,const NativeContextOpaqueContract& contract){
    auto root=std::make_shared<NativeContextRetirementSlotState>();
    root->preparing_source=source;root->preparing_opaque=opaque;root->original_memory=source->memory();root->original_device=contract.key().device;
    root->process_population="native-context-birth";root->certificate=contract.evidenceSha256();
    return root;
}
bool NativeContextRetirementOwner::matchesProcessRoot(const std::shared_ptr<NativeContextRetirementSlotState>& root,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device,const std::string& certificate,const std::string& population){
    if(root->original_device!=device)return false;
    const auto& original=root->original_memory;
    if(original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original) )
        throw std::logic_error("Process population changed original authority");
    if(root->process_population!=population)return false;
    if(root->certificate!=certificate)throw std::logic_error("Process population changed original source certificate");
    if(root->process_use_active || !root->obligation || root->obligation->physicalLifetime()!=NativePhysicalLifetime::ProcessRuntime)
        throw std::logic_error("Original process population still has an active generation use");
    root->process_use_active=true;return true;
}
void NativeContextRetirementOwner::startProcessUse(const std::shared_ptr<NativeContextRetirementSlotState>& root)noexcept{if(!root || root->process_use_active)std::terminate();root->process_use_active=true;}
void NativeContextRetirementOwner::finishProcessUse(const std::shared_ptr<NativeContextRetirementSlotState>& root)noexcept{
    if(!root || !root->process_use_active)std::terminate();root->process_use_active=false;
}
void NativeContextRetirementSlot::prepareProcessRuntime(const std::string& population){
    if(!owner_ || !state_ || population.empty())throw std::invalid_argument("Process population needs its original reserved slot");
    std::lock_guard lock(owner_->mutex_);
    if(owner_->retired_ || state_->obligation || state_->original_process_root || state_->phase!=NativeContextRetirementSlotState::Phase::Reserved)
        throw std::logic_error("Process population preparation follows original prebirth reservation");
    state_->process_population=population;
}
bool NativeContextRetirementSlot::tryBorrowProcessRuntime(const std::string& population){
    prepareProcessRuntime(population);
    std::lock_guard lock(owner_->mutex_);
    auto original=GPUDeviceProcessLifetime::borrowProcessRoot(owner_->authority_,owner_->device_,state_->certificate,population);
    if(!original)return false;
    state_->original_process_root=std::move(original);return true;
}
NativeContextRetirementSlot::NativeContextRetirementSlot(std::shared_ptr<NativeContextRetirementOwner> owner,std::shared_ptr<NativeContextRetirementSlotState> state)noexcept
    :owner_(std::move(owner)),state_(std::move(state)){}
NativeContextRetirementSlot::NativeContextRetirementSlot(NativeContextRetirementSlot&& other)noexcept:owner_(std::move(other.owner_)),state_(std::move(other.state_)){}
NativeContextRetirementSlot& NativeContextRetirementSlot::operator=(NativeContextRetirementSlot&& other)noexcept{
    if(this!=&other){release();owner_=std::move(other.owner_);state_=std::move(other.state_);}return *this;
}
NativeContextRetirementSlot::~NativeContextRetirementSlot()noexcept{release();}
void NativeContextRetirementSlot::adopt(std::unique_ptr<IContextNativeRetirementObligation> obligation){
    if(!owner_ || !state_ || !obligation)std::terminate();
    std::lock_guard lock(owner_->mutex_);
    const auto& memory=obligation->originalAuthority();
    if(owner_->retired_ || state_->obligation || state_->original_process_root || state_->phase!=NativeContextRetirementSlotState::Phase::Reserved ||
       obligation->device()!=owner_->device_ || obligation->poolGeneration()!=owner_->generation_ || obligation->certificateIdentity()!=state_->certificate ||
       memory.get()!=owner_->authority_.get() || memory.owner_before(owner_->authority_) || owner_->authority_.owner_before(memory))std::terminate();
    if(obligation->physicalLifetime()==NativePhysicalLifetime::ProcessRuntime){
        if(state_->process_population.empty())std::terminate();
        GPUDeviceProcessLifetime::retainProcessRoot(state_);
    }else if(obligation->physicalLifetime()!=NativePhysicalLifetime::ContextGeneration || !state_->process_population.empty())std::terminate();
    state_->obligation=std::move(obligation);
}
void NativeContextRetirementSlot::armNativeAttempt()noexcept{
    if(!owner_ || !state_)std::terminate();std::lock_guard lock(owner_->mutex_);
    if(owner_->retired_ || (!state_->obligation && !state_->original_process_root) || state_->phase!=NativeContextRetirementSlotState::Phase::Reserved)std::terminate();
    state_->phase=NativeContextRetirementSlotState::Phase::NativeAttempted;
}
void NativeContextRetirementSlot::completeNativeHandles()noexcept{
    if(!owner_ || !state_)std::terminate();std::lock_guard lock(owner_->mutex_);
    if(owner_->retired_ || (!state_->obligation && !state_->original_process_root) || state_->phase==NativeContextRetirementSlotState::Phase::HandlesRetired)std::terminate();
    state_->phase=NativeContextRetirementSlotState::Phase::HandlesRetired;
}
void NativeContextRetirementSlot::delegateHandlesToContext()noexcept{
    if(!owner_ || !state_)std::terminate();std::lock_guard lock(owner_->mutex_);
    if(owner_->retired_ || (!state_->obligation && !state_->original_process_root) ||
       state_->phase!=NativeContextRetirementSlotState::Phase::NativeAttempted || !state_->borrowed_slot)std::terminate();
    state_->phase=NativeContextRetirementSlotState::Phase::ContextOwnedHandles;
}
void NativeContextRetirementSlot::release()noexcept{
    if(!owner_ && !state_)return;if(!owner_ || !state_)std::terminate();
    bool cancelled=false;
    {
        std::lock_guard lock(owner_->mutex_);
        if(owner_->retired_ || (state_->phase==NativeContextRetirementSlotState::Phase::NativeAttempted || state_->phase==NativeContextRetirementSlotState::Phase::ContextOwnedHandles))std::terminate();
        state_->borrowed_slot=false;
        if(state_->obligation || state_->original_process_root)state_->phase=NativeContextRetirementSlotState::Phase::HandlesRetired;
        else {std::erase(owner_->slots_,state_);if(owner_->slots_.empty())owner_->authority_.reset();cancelled=true;}
    }
    if(cancelled)GPUDeviceContextPool::instance().discardEmptyNativeRetirementOwner(owner_->device_,owner_->generation_,owner_);
    state_.reset();owner_.reset();
}
}
