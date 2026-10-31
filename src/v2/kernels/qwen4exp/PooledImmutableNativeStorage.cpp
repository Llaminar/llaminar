/**
 * @file PooledImmutableNativeStorage.cpp
 * @brief Exact PMA claim/materialization binding and event-safe setup lifetime.
 *
 * INT8 device-only Tensor byte_size is the exact request sent to the canonical
 * CUDA/HIP allocator. Root padding is explicitly materialized and charged;
 * subrange 64-byte alignment is not used as an approximate physical estimate.
 * Backend allocation borrowers and prepared runtime seals authenticate reuse.
 */
#include "PooledImmutableNativeRootStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
namespace llaminar2::qwen4exp::detail
{
namespace {
/**
 * @brief Retire only failed/last-owner setup writes on their exact original event.
 * @note This infrastructure boundary never runs during successful setup,
 * capture, replay, per-token execution or request-state publication.
 */
void retirePooledNativeSetupOwner(IBackend& backend,DeviceId device,void* event,void* stream,bool recorded)noexcept
{
    if((!recorded && !backend.recordEvent(event,device.ordinal,stream)) || !backend.waitForEvent(event,device.ordinal))
        std::terminate();
}
}
void PooledNativeRootStorage::requireStream(void* stream)
{
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Pooled native storage requires the exact non-default stream");
}
std::shared_ptr<PooledNativeRootStorage> PooledNativeRootStorage::create(
    PhysicalMemoryAllocatorIdentity allocator,std::array<std::size_t,3> bytes,std::uint64_t identity,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)
{
    requireStream(stream);
    if(isGraphCaptureActive() || !identity || !memory || allocator.world_rank!=memory->worldRank() ||
        !allocator.device.is_gpu() || !allocator.device.is_valid() || !bytes[1] || !bytes[2])
        throw std::invalid_argument("Pooled native roots require exact admitted rank/GPU geometry before capture");
    auto result=std::shared_ptr<PooledNativeRootStorage>(new PooledNativeRootStorage(allocator,bytes,identity,std::move(memory)));
    std::vector<TensorBase*> roots;roots.reserve(3);
    for(std::size_t i=0;i<3;++i){
        if(!bytes[i])continue;
        result->claims_[i]=result->memory_->claimNewAllocation(allocator.device,
            i?PhysicalMemoryOwner::ImmutableLookupStaging:PhysicalMemoryOwner::ImmutableTableStorage,bytes[i]);
        result->tensors_[i].reset(new INT8Tensor({bytes[i]},allocator.device,INT8Tensor::DeviceOnlyStorageTag{}));
        if(result->tensors_[i]->size_bytes()!=result->claims_[i].bytes())
            throw std::logic_error("Pooled native tensor allocation extent differs from its exact PMA claim");
        // This canonical operation allocates storage only: no kernel, copy or
        // memset is submitted before initialize() owns its retirement event.
        TransferEngine::prepareDeviceOutput(result->tensors_[i].get(),allocator.device,stream);
        roots.push_back(result->tensors_[i].get());
    }
    result->prepared_.emplace(allocator.device,roots);result->prepared_->seal();
    for(std::size_t i=0;i<3;++i)if(result->tensors_[i])
        result->captured_[i]=TransferEngine::instance().retainCapturedDeviceStorage(*result->tensors_[i],allocator.device,bytes[i]);
    result->backend_=result->tensors_[1]->resolveBackend(allocator.device);
    result->require();return result;
}
PooledNativeRootStorage::~PooledNativeRootStorage()noexcept
{
    // Setup is the only possible submitted work if graph ownership never
    // publishes. Successful graph families retain this owner until retirement.
    if(setup_event_){
        if(setup_stream_)retirePooledNativeSetupOwner(*backend_,allocator_.device,setup_event_,setup_stream_,setup_recorded_);
        backend_->destroyEvent(setup_event_,allocator_.device.ordinal);
    }
}
PooledImmutableNativeRoots PooledNativeRootStorage::roots()const
{
    const auto self=shared_from_this();
    return {tensors_[0]?std::shared_ptr<TensorBase>(self,tensors_[0].get()):nullptr,
        std::shared_ptr<TensorBase>(self,tensors_[1].get()),std::shared_ptr<TensorBase>(self,tensors_[2].get())};
}
void PooledNativeRootStorage::require()const
{
    if(!memory_ || memory_->worldRank()!=allocator_.world_rank || !prepared_)
        throw std::logic_error("Pooled native roots lost their original authority or prepared declaration");
    prepared_->validate();
    for(std::size_t i=0;i<3;++i){
        if(!bytes_[i]){
            if(tensors_[i] || claims_[i].valid() || captured_[i])
                throw std::logic_error("Pooled C0 optional root acquired undeclared backing");
            continue;
        }
        const auto& root=tensors_[i];
        if(!root || !claims_[i].valid() || claims_[i].bytes()!=bytes_[i] ||
            claims_[i].owner()!=(i?PhysicalMemoryOwner::ImmutableLookupStaging:PhysicalMemoryOwner::ImmutableTableStorage) ||
            root->transferStorageOwner()!=root.get() || root->is_view() || root->native_type()!=TensorType::INT8 ||
            root->size_bytes()!=bytes_[i] || root->home_device()!=allocator_.device || root->current_device()!=allocator_.device ||
            !root->gpu_data_ptr() || reinterpret_cast<std::uintptr_t>(root->gpu_data_ptr())%64 || !captured_[i])
            throw std::logic_error("Pooled native physical root/claim/type/extent changed");
        TransferEngine::instance().requireCapturedDeviceStorage(*captured_[i],*root,allocator_.device,bytes_[i]);
    }
}
void PooledNativeRootStorage::requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const
{
    require();if(!memory || memory.get()!=memory_.get())
        throw std::logic_error("Pooled physical roots cannot replace their original PMA");
}
ImmutableLookupNativeOwnerObservation PooledNativeRootStorage::retainedOwnerObservation()const
{
    require();
    return {memory_,allocator_,claims_[0].valid()?claims_[0].bytes():0,
        claims_[1].bytes()+claims_[2].bytes()};
}
std::uint64_t PooledNativeRootStorage::identity()const
{
    require();std::uint64_t value=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t x){value=(value^x)*1099511628211ULL;};
    mix(pooled_native::Version);mix(geometry_);mix(reinterpret_cast<std::uintptr_t>(memory_.get()));
    mix(allocator_.world_rank);mix(static_cast<std::uint64_t>(allocator_.device.type));mix(allocator_.device.ordinal);
    for(const auto bytes:bytes_)mix(bytes);mix(prepared_->preparedIdentity());return value?value:1;
}
void PooledNativeRootStorage::initialize(DeviceType backend,void* stream,const std::function<void()>& submit)
{
    requireStream(stream);
    std::lock_guard lock(mutex_);
    if(isGraphCaptureActive() || setup_!=Setup::Declared || backend!=allocator_.device.type)
        throw std::logic_error("Pooled native metadata setup is once-only, outside capture, on its exact backend");
    require();setup_=Setup::Failed;
    setup_event_=backend_->createEvent(allocator_.device.ordinal);
    if(!setup_event_)throw std::runtime_error("Pooled native setup could not retain its producer event");
    // The exact event and stream exist before the first metadata submission.
    // Any later exception retires that partial producer in the last-owner path.
    setup_stream_=stream;
    submit();
    for(const auto& root:tensors_)if(root)TransferEngine::publishDeviceWrite(root.get(),allocator_.device,stream);
    if(!backend_->recordEvent(setup_event_,allocator_.device.ordinal,stream))
        throw std::runtime_error("Pooled native setup could not publish its exact producer event");
    setup_recorded_=true;setup_=Setup::Submitted;
}
void PooledNativeRootStorage::requireInitialized()const
{
    std::lock_guard lock(mutex_);require();
    if(setup_!=Setup::Submitted || !setup_event_ || !setup_recorded_)
        throw std::logic_error("Pooled native phase has no successfully submitted original setup");
}
void PooledNativeRootStorage::acquire(void* stream)const
{
    requireStream(stream);requireInitialized();
    if(!backend_->streamWaitEvent(stream,setup_event_,allocator_.device.ordinal))
        throw std::runtime_error("Pooled native consumer could not acquire its exact setup producer");
}
std::byte* PooledNativeRootStorage::address(std::size_t i)const
{
    require();if(i>=3)throw std::out_of_range("Pooled native root index is invalid");
    return tensors_[i]?static_cast<std::byte*>(tensors_[i]->gpu_data_ptr()):nullptr;
}
}
namespace llaminar2::qwen4exp {
std::shared_ptr<PooledImmutableOwnerPreparedStorage> PooledImmutableOwnerPreparedStorage::create(
    PooledImmutableOwnerNativePlan plan,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)
{
    const auto& p=plan.parameters();
    auto roots=detail::PooledNativeRootStorage::create(plan.allocator(),{p.cache.bytes,p.frame.bytes,p.native.bytes},p.identity,std::move(memory),stream);
    return std::shared_ptr<PooledImmutableOwnerPreparedStorage>(new PooledImmutableOwnerPreparedStorage(std::move(plan),std::move(roots)));
}
PooledImmutableNativeRoots PooledImmutableOwnerPreparedStorage::roots()const{return roots_->roots();}
void PooledImmutableOwnerPreparedStorage::requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const{roots_->requireAuthority(memory);}
ImmutableLookupNativeOwnerObservation PooledImmutableOwnerPreparedStorage::retainedOwnerObservation()const{return roots_->retainedOwnerObservation();}
std::uint64_t PooledImmutableOwnerPreparedStorage::captureIdentity()const{return roots_->identity();}
void PooledImmutableOwnerPreparedStorage::acquireSetup(void* stream)const{roots_->acquire(stream);}
std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> PooledImmutableCoordinatorPreparedStorage::create(
    PooledImmutableCoordinatorNativePlan plan,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)
{
    const auto& p=plan.parameters();
    auto roots=detail::PooledNativeRootStorage::create(plan.allocator(),{p.cache.bytes,p.frame.bytes,p.native.bytes},p.identity,std::move(memory),stream);
    return std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>(new PooledImmutableCoordinatorPreparedStorage(std::move(plan),std::move(roots)));
}
PooledImmutableNativeRoots PooledImmutableCoordinatorPreparedStorage::roots()const{return roots_->roots();}
void PooledImmutableCoordinatorPreparedStorage::requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const{roots_->requireAuthority(memory);}
ImmutableLookupNativeOwnerObservation PooledImmutableCoordinatorPreparedStorage::retainedOwnerObservation()const{return roots_->retainedOwnerObservation();}
std::uint64_t PooledImmutableCoordinatorPreparedStorage::captureIdentity()const{return roots_->identity();}
void PooledImmutableCoordinatorPreparedStorage::acquireSetup(void* stream)const{roots_->acquire(stream);}
}
