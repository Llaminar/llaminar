/**
 * @file PooledImmutableLocalHotPreparedStorage.cpp
 * @brief Exact original-source/PMA local roots and the original mapped ready lease.
 *
 * Last graph/transport owners retain this object through physical native
 * retirement. Destruction closes only its original handles; no host value is
 * read to infer whether a GPU query is complete or safe to reset.
 */
#include "PooledImmutableLocalHotPreparedStorage.h"
#include "PooledImmutableLocalHotReadyRoot.h"
#include "PooledImmutableNativeRootStorage.h"
#include "PooledImmutableSourcePreparedStorage.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/NativeContextPreparation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>
#include <optional>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject missing/default stream handles before any native operation. */
void exact(void* stream)
{
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Local hot storage requires its exact non-default stream");
}
/** @return Same original shared owner/control block, not an equal raw address. */
template<class A,class B> bool sameOwner(const std::shared_ptr<A>& a,const std::shared_ptr<B>& b)
{return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);}
}
/** @brief Physical ownership only; the cache/frame state exists exclusively in native roots. */
struct PooledImmutableLocalHotPreparedStorage::Impl {
    PreparedGPUContextUsePermit context; ///< Declared first so every original native/mapped handle retires before this borrow.
    std::shared_ptr<PooledImmutableLocalHotReadyRoot> ready;
    std::shared_ptr<detail::PooledNativeRootStorage> roots;
    NativeEmbeddingIQTables tables{};
    local_hot::TerminalObservationAccumulator terminal_observations; ///< Same lifetime as the actual cumulative native counters.
    std::optional<MappedTimelineKernelPublish64Binding> ready_binding;
    /** @brief Retain the exact existing prepared context through the last local-root destructor. */
    explicit Impl(PreparedGPUContextUsePermit original,local_hot::TerminalReceiveRole role,std::size_t rowBytes)
        :context(std::move(original)),terminal_observations(role,rowBytes){}
};
PooledImmutableLocalHotPreparedStorage::PooledImmutableLocalHotPreparedStorage(
    PooledImmutableLocalHotNativePlan plan,std::unique_ptr<Impl> impl)
    :plan_(std::move(plan)),impl_(std::move(impl)){}
std::shared_ptr<PooledImmutableLocalHotPreparedStorage> PooledImmutableLocalHotPreparedStorage::create(
    PooledImmutableLocalHotGeometry geometry,std::shared_ptr<PooledImmutableSourcePreparedStorage> source,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)
{
    exact(stream);
    if(isGraphCaptureActive() || !source || !memory || geometry.allocator().world_rank!=memory->worldRank())
        throw std::invalid_argument("Local hot roots require the original admitted source/rank before capture");
    if(!sameOwner(source->retainedOwnerObservation().authority(),memory))
        throw std::invalid_argument("Local hot roots cannot replace their original source PMA");
    auto plan=PooledImmutableLocalHotNativePlan::fromOriginalSource(std::move(geometry),std::move(source));
    auto context=GPUDeviceContextPool::instance().retainExistingPreparedContext(plan.geometry().allocator().device,memory);
    const auto role=plan.geometry().allocator()==plan.originalSource()->geometry().allocator()
        ?local_hot::TerminalReceiveRole::Coordinator:local_hot::TerminalReceiveRole::Reader;
    auto impl=std::make_unique<Impl>(std::move(context),role,plan.geometry().rowBytes());
    impl->roots=detail::PooledNativeRootStorage::create(plan.geometry().allocator(),plan.geometry().rootBytes(),
        plan.captureIdentity(),memory,stream,NativeOrderingEventProgram::pooledLocalHot(plan.geometry().slots()!=0));
    impl->ready=PooledImmutableLocalHotReadyRoot::create(plan.geometry().allocator(),memory);
    impl->ready_binding.emplace(TransferEngine::instance().bindMappedTimelineKernelPublish64(
        *impl->ready->mappedRegion(),0,1,plan.geometry().allocator().device));
    auto result=std::shared_ptr<PooledImmutableLocalHotPreparedStorage>(
        new PooledImmutableLocalHotPreparedStorage(std::move(plan),std::move(impl)));
    return result;
}
PooledImmutableLocalHotPreparedStorage::~PooledImmutableLocalHotPreparedStorage()=default;
PooledImmutableNativeRoots PooledImmutableLocalHotPreparedStorage::roots()const
{return impl_->roots->roots();}
ImmutableLookupNativeOwnerObservation PooledImmutableLocalHotPreparedStorage::retainedOwnerObservation()const
{return impl_->roots->retainedOwnerObservation();}
ImmutableLookupNativeOwnerObservation PooledImmutableLocalHotPreparedStorage::retainedReadyObservation()const
{return impl_->ready->retainedOwnerObservation();}
void PooledImmutableLocalHotPreparedStorage::requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const
{
    impl_->roots->requireAuthority(memory);
    if(!sameOwner(retainedOwnerObservation().authority(),memory) ||
        !sameOwner(retainedReadyObservation().authority(),memory))
        throw std::logic_error("Local hot physical roots changed their original PMA control block");
}
void PooledImmutableLocalHotPreparedStorage::requireFor(const PooledImmutableLocalHotGeometry& geometry,
    const std::shared_ptr<PooledImmutableSourcePreparedStorage>& source,
    const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const
{
    if(geometry!=plan_.geometry() || !sameOwner(source,plan_.originalSource()) ||
        !source || source->captureIdentity()!=plan_.originalSource()->captureIdentity())
        throw std::logic_error("Local hot reuse changed its actual resolved geometry/original source owner");
    requireAuthority(memory);
}
const std::shared_ptr<MappedHostTransferRegion>& PooledImmutableLocalHotPreparedStorage::readyRegion()const noexcept
{return impl_->ready->mappedRegion();}
void PooledImmutableLocalHotPreparedStorage::acquireSetup(void* stream)const
{exact(stream);impl_->roots->acquire(stream);}
void PooledImmutableLocalHotPreparedStorage::enqueueReadyWait(void* stream)const
{
    exact(stream);requireInitialized();
    TransferEngine::instance().enqueueMappedTimelineWait64(*readyRegion(),0,1,geometry().allocator().device,stream);
}
void PooledImmutableLocalHotPreparedStorage::enqueueLocalReadCompleteWait(void* stream)const
{
    exact(stream);requireInitialized();
    TransferEngine::instance().enqueueMappedTimelineWait64(*readyRegion(),0,2,geometry().allocator().device,stream);
}
std::uint64_t PooledImmutableLocalHotPreparedStorage::consumerStorageIdentity()const
{return impl_->roots->identity();}
void PooledImmutableLocalHotPreparedStorage::requireInitialized()const
{impl_->roots->requireInitialized();}
local_hot::NativeRoots PooledImmutableLocalHotPreparedStorage::nativeRoots()const
{
    const auto& g=geometry();const auto source=plan_.originalSource();
    if(!impl_->ready || !impl_->ready->mappedRegion()->isBound())
        throw std::logic_error("Local hot native roots lost their actual readiness owner");
    if(!impl_->ready_binding || !impl_->ready_binding->valid())
        throw std::logic_error("Local hot readiness lost its cold original mapped binding");
    local_hot::Parameters p{g.cacheOffsets(),g.frameOffsets(),g.scratchOffsets(),g.extent().rows(),
        plan_.table(),plan_.sourceStorageGeneration(),consumerStorageIdentity(),g.rowBytes(),g.slotStride(),
        g.maximumOriginalIds(),g.slots(),g.laneWidth(),g.protectedLimit(),std::uint32_t(g.rowElements()),
        g.policy(),g.rowKeyWidth(),plan_.nativeFormat()};
    return {p,impl_->roots->address(0),impl_->roots->address(1),impl_->roots->address(2),impl_->ready_binding->deviceSignal(),impl_->tables};
}
void PooledImmutableLocalHotPreparedStorage::initializeNative(DeviceType backend,NativeEmbeddingIQTables tables,
    void* stream,const std::function<void(const local_hot::NativeRoots&)>& initialize)
{
    exact(stream);
    if(backend!=geometry().allocator().device.type || !initialize)
        throw std::invalid_argument("Local hot initializer has a foreign backend or missing native program");
    impl_->tables=tables;
    impl_->roots->initialize(backend,stream,[&]{initialize(nativeRoots());});
}
std::uint64_t PooledImmutableLocalHotPreparedStorage::captureIdentity()const
{
    requireInitialized();auto value=plan_.captureIdentity();
    const auto mix=[&](std::uint64_t x){value=(value^x)*1099511628211ULL;};
    mix(consumerStorageIdentity());mix(reinterpret_cast<std::uintptr_t>(readyRegion().get()));
    mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq3s));mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq3xxs));
    mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq2s));mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq2xs));
    mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq2xxs));mix(reinterpret_cast<std::uintptr_t>(impl_->tables.iq1s));
    return value?value:1;
}
local_hot::TerminalPublication PooledImmutableLocalHotPreparedStorage::observeCompletedTerminal(const local_hot::TerminalObservation& current)
{
    requireInitialized();return impl_->terminal_observations.observe(current);
}
} // namespace llaminar2::qwen4exp
