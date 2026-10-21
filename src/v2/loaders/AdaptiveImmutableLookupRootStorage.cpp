/**
 * @file AdaptiveImmutableLookupRootStorage.cpp
 * @brief Allocate only source-declared roots with exact canonical claims and no host mirrors.
 *
 * TransferEngine remains the sole allocation/coherence boundary. Device-only
 * tensor construction reserves no native host backing. The owner issues no
 * copies, metadata initialization, admission-generation writes or graph work;
 * those belong to the existing prepared source/boundary after setup returns.
 */
#include "AdaptiveImmutableLookupRootStorage.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include "transfer/TransferEngine.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <limits>
#include <stdexcept>
namespace llaminar2
{
std::shared_ptr<const AdaptiveImmutableLookupRootStorage> AdaptiveImmutableLookupRootStorage::create(
    const qwen4exp::AdaptiveImmutableRowCacheNativePlan &plan,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void *setup_stream)
{
    const auto &p=plan.parameters();
    if(isGraphCaptureActive() || !memory || memory->worldRank()!=p.world_rank ||
        !p.device.is_gpu() || !p.device.is_valid() || !setup_stream ||
        reinterpret_cast<std::uintptr_t>(setup_stream)<=2 || !p.frame.bytes ||
        (p.slots==0)!=(p.cache.bytes==0))
        throw std::invalid_argument("Adaptive native roots require exact admitted rank/GPU geometry and setup stream");
    auto result=std::shared_ptr<AdaptiveImmutableLookupRootStorage>(
        new AdaptiveImmutableLookupRootStorage(plan,std::move(memory)));
    if(p.cache.bytes)
    {
        result->cache_lease_=result->memory_->claimNewAllocation(p.device,PhysicalMemoryOwner::ImmutableTableStorage,p.cache.bytes);
        result->cache_.reset(new INT8Tensor({p.cache.bytes},p.device,INT8Tensor::DeviceOnlyStorageTag{}));
        TransferEngine::prepareDeviceOutput(result->cache_.get(),p.device,setup_stream);
    }
    result->frame_lease_=result->memory_->claimNewAllocation(p.device,PhysicalMemoryOwner::ImmutableLookupStaging,p.frame.bytes);
    result->frame_.reset(new INT8Tensor({p.frame.bytes},p.device,INT8Tensor::DeviceOnlyStorageTag{}));
    TransferEngine::prepareDeviceOutput(result->frame_.get(),p.device,setup_stream);
    constexpr auto protocol_bytes=sizeof(qwen4exp::ImmutableLookupDeviceState);
    static_assert(protocol_bytes==64 && protocol_bytes%sizeof(std::int32_t)==0);
    result->protocol_lease_=result->memory_->claimNewAllocation(p.device,PhysicalMemoryOwner::ImmutableLookupStaging,protocol_bytes);
    result->protocol_.reset(new INT32Tensor({protocol_bytes/sizeof(std::int32_t)},p.device,INT32Tensor::DeviceOnlyStorageTag{}));
    TransferEngine::prepareDeviceOutput(result->protocol_.get(),p.device,setup_stream);
    result->requireFor(plan,result->memory_);
    return result;
}
void AdaptiveImmutableLookupRootStorage::bindPreparedStorage(
    const std::shared_ptr<qwen4exp::AdaptiveLookupPreparedStorage> &storage) const
{
    if(!storage)throw std::invalid_argument("Adaptive root setup requires its actual source-issued prepared owner");
    std::lock_guard lock(setup_mutex_);
    if(ever_bound_)
    {
        if(prepared_.lock()!=storage)
            throw std::logic_error("Adaptive physical roots already belong to another or retired source owner");
        return;
    }
    prepared_=storage;ever_bound_=true;
}
AdaptiveImmutableLookupRoots AdaptiveImmutableLookupRootStorage::roots() const
{
    const auto self=shared_from_this();
    return {cache_?std::shared_ptr<TensorBase>(self,cache_.get()):nullptr,
        std::shared_ptr<TensorBase>(self,frame_.get()),std::shared_ptr<TensorBase>(self,protocol_.get())};
}
void AdaptiveImmutableLookupRootStorage::requireFor(const qwen4exp::AdaptiveImmutableRowCacheNativePlan &plan,
    const std::shared_ptr<const PhysicalMemoryAuthority> &memory) const
{
    const auto &p=plan.parameters();
    const auto require=[&](const std::unique_ptr<TensorBase> &root,const PhysicalMemoryAllocationLease &lease,
        std::size_t bytes,TensorType type,PhysicalMemoryOwner owner) {
        if(!bytes)
        {
            if(root || lease.valid() || lease.bytes())throw std::logic_error("C0 adaptive root acquired optional backing");
            return;
        }
        if(!root || !lease.valid() || lease.owner()!=owner || lease.bytes()!=bytes ||
            root->size_bytes()!=lease.bytes() || root->native_type()!=type || root->home_device()!=device_ ||
            !root->current_device() || *root->current_device()!=device_ || !root->gpu_data_ptr())
            throw std::logic_error("Adaptive original native root/lease authority changed");
    };
    if(plan!=plan_ || !memory || memory.get()!=memory_.get() || p.world_rank!=memory_->worldRank() || p.device!=device_)
        throw std::logic_error("Adaptive native roots cannot change their canonical authority or device");
    require(cache_,cache_lease_,p.cache.bytes,TensorType::INT8,PhysicalMemoryOwner::ImmutableTableStorage);
    require(frame_,frame_lease_,p.frame.bytes,TensorType::INT8,PhysicalMemoryOwner::ImmutableLookupStaging);
    require(protocol_,protocol_lease_,sizeof(qwen4exp::ImmutableLookupDeviceState),TensorType::INT32,
        PhysicalMemoryOwner::ImmutableLookupStaging);
}
AdaptiveImmutableLookupOwnerObservation AdaptiveImmutableLookupRootStorage::retainedOwnerObservation() const
{
    requireFor(plan_,memory_);
    if(!frame_lease_.valid() || !protocol_lease_.valid() ||
        frame_lease_.bytes()>std::numeric_limits<std::size_t>::max()-protocol_lease_.bytes())
        throw std::logic_error("Adaptive native staging has no complete representable live claims");
    return {memory_,{memory_->worldRank(),device_},cache_lease_.bytes(),frame_lease_.bytes()+protocol_lease_.bytes()};
}
} // namespace llaminar2
