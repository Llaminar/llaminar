/**
 * @file PooledImmutableDirectedTransport.cpp
 * @brief Materialize only original two-channel cross-vendor edges priced by the shared runtime geometry.
 *
 * All candidate/floor/growth BOMs use one source-owned channel shape. No table
 * identity is invented for planning. Cold setup borrows the already prepared
 * original participant workers; no capability, alternative transport or hidden
 * nested graph is introduced. Expert placement/rebalancing is independent.
 */
#include "PooledImmutableDirectedTransport.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Distinguish original mandatory protocol roots, candidate roots and optional growth. */
enum class Contribution {Mandatory,Complete,Growth};
/** @return The exact immutable original allocator observation, rejecting duplicated or missing resources. */
PhysicalMemoryResource resource(std::span<const PhysicalMemoryResource> resources,PhysicalMemoryAllocatorIdentity allocator){
    const PhysicalMemoryResource* found=nullptr;
    for(const auto& r:resources)if(r.world_rank==allocator.world_rank && r.device==allocator.device){if(found)throw std::invalid_argument("Duplicate original directed resource");found=&r;}
    if(!found)throw std::invalid_argument("Missing original directed GPU resource");return *found;
}
/** @brief Reject mapped substitution for same-vendor/rank edges before quoting or creating transport. */
void requireRemote(PhysicalMemoryAllocatorIdentity coordinator,PhysicalMemoryAllocatorIdentity owner){
    if(owner==coordinator)return;
    if(!owner.device.is_gpu() || !coordinator.device.is_gpu() || owner.world_rank!=coordinator.world_rank || owner.device.type==coordinator.device.type)
        throw std::invalid_argument("Directed immutable transport requires original rank-local opposite-vendor endpoints; native/rank factory remains unsupported");
}
/** @return One exact source-owned contribution, never subtracting a capacity balance. */
PhysicalMemoryPlan contribution(const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& policy,
    PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources,Contribution kind){
    PooledImmutableCoordinatorNativePlan::requireOriginalCohort(physical,policy);
    if(!host.device.is_cpu() || host.world_rank!=physical.coordinator().world_rank)throw std::invalid_argument("Directed channel changed original host allocator");
    const auto coordinator=resource(resources,physical.coordinator());PhysicalMemoryPlanBuilder builder;
    for(const auto& owner:physical.owners()){
        requireRemote(physical.coordinator(),owner.allocator);if(owner.allocator==physical.coordinator())continue;
        const auto observed=resource(resources,owner.allocator);
        const auto floor=PooledImmutableDirectedChannelGeometry::mandatory(physical,policy,owner.coordinate);
        const auto actual=PooledImmutableDirectedChannelGeometry::fromGeometry(physical,policy,owner.coordinate);
        const std::array baseline{floor.forwardMemory(),floor.reverseMemory()},complete{actual.forwardMemory(),actual.reverseMemory()};
        for(std::size_t direction=0;direction<2;++direction){
            auto memory=kind==Contribution::Mandatory?baseline[direction]:complete[direction];
            if(kind==Contribution::Growth){
                if(memory.mapped_host_bytes<baseline[direction].mapped_host_bytes || memory.cursor_bytes_per_device!=baseline[direction].cursor_bytes_per_device)
                    throw std::logic_error("Directed transport growth changed original mandatory cursor roots");
                memory.mapped_host_bytes-=baseline[direction].mapped_host_bytes;memory.cursor_bytes_per_device=0;
            }
            if(memory.mapped_host_bytes)builder.add(host,PhysicalMemoryOwner::ActivationTransportStaging,memory.mapped_host_bytes);
            if(memory.cursor_bytes_per_device){builder.add(coordinator,PhysicalMemoryOwner::ActivationTransportStaging,memory.cursor_bytes_per_device);builder.add(observed,PhysicalMemoryOwner::ActivationTransportStaging,memory.cursor_bytes_per_device);}
        }
    }
    return builder.build();
}
}
PhysicalMemoryPlan PooledImmutableDirectedTransport::mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Mandatory);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::completeBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Complete);}
PhysicalMemoryPlan PooledImmutableDirectedTransport::slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry& p,const PooledImmutableCoordinatorGeometry& c,PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> resources){return contribution(p,c,host,resources,Contribution::Growth);}
std::shared_ptr<const PooledImmutableDirectedTransport> PooledImmutableDirectedTransport::create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary){
    if(isGraphCaptureActive() || !boundary || !boundary->isPooled())throw std::invalid_argument("Directed transport requires original cold pooled source");
    const auto storage=boundary->pooledStorage();const auto& retained=storage->retainedPreparedSourceWorkers();
    if(!retained || retained->owners().size()!=storage->originalOwners().size())throw std::logic_error("Directed transport requires actual original prepared source participants");
    const auto memory=retained->source()->memory();storage->coordinator()->requireAuthority(memory);
    const auto& coordinator=retained->coordinator();
    const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(coordinator.device(),memory);
    coordinator.requireWorker(current.worker());
    if(current.generation()!=coordinator.generation())throw std::logic_error("Directed coordinator original generation changed");
    if(coordinator.worker()->ownsCurrentThread())throw std::invalid_argument("Directed cohort setup requires its off-worker cold coordinator");
    std::vector<void*> ownerStreams(retained->owners().size());
    // Authenticate the COMPLETE actual participant population before either
    // channel materializes. An unsupported later owner cannot create a partial
    // apparently usable mapped topology.
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];const auto& permit=retained->owners()[i];owner->requireAuthority(memory);
        const auto actual=GPUDeviceContextPool::instance().retainExistingPreparedContext(permit.device(),memory);
        permit.requireWorker(actual.worker());if(actual.generation()!=permit.generation())throw std::logic_error("Directed owner original generation changed");
        requireRemote(storage->coordinator()->plan().allocator(),owner->plan().allocator());
        if(permit.worker()->ownsCurrentThread())throw std::invalid_argument("Directed setup cannot occupy an original participant worker");
    }
    auto result=std::shared_ptr<PooledImmutableDirectedTransport>(new PooledImmutableDirectedTransport(std::move(boundary)));
    result->requireCurrent();result->remotes_.reserve(retained->owners().size());
    void* coordinatorStream=nullptr;coordinator.worker()->submitAndWait([&]{coordinatorStream=coordinator.worker()->defaultStream();});
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];const auto& permit=retained->owners()[i];
        if(owner->plan().allocator()==storage->coordinator()->plan().allocator())continue;
        permit.worker()->submitAndWait([&]{ownerStreams[i]=permit.worker()->defaultStream();});
    }
    for(std::uint32_t i=0;i<retained->owners().size();++i){
        const auto& owner=storage->originalOwners()[i];
        if(owner->plan().allocator()==storage->coordinator()->plan().allocator())continue;
        void* ownerStream=ownerStreams[i];
        const auto pin=PooledImmutableCapturedPinEdge::bind(storage->coordinator(),owner,memory,coordinatorStream,ownerStream);
        const auto returned=PooledImmutableCapturedPinReturn::bind(pin,memory,ownerStream,coordinatorStream);
        const auto free=PooledImmutableCapturedFreeWave::bind(returned,memory);
        const auto victim=PooledImmutableCapturedVictimWave::bind(free,memory);
        const auto source=PooledImmutableCapturedSourceWave::bind(victim,result->boundary_,memory);
        const auto publication=PooledImmutableCapturedPublicationWave::bind(source,memory);
        const auto read=PooledImmutableCapturedReadWave::bind(publication,memory);
        const auto retirement=PooledImmutableCapturedRetirementWave::bind(read,memory);
        result->remotes_.push_back({i,pin,returned,free,victim,source,publication,read,retirement});
    }
    result->requireCurrent();return result;
}
const PooledImmutableRemoteTransport& PooledImmutableDirectedTransport::remote(std::uint32_t coordinate)const{
    requireCurrent();const auto found=std::find_if(remotes_.begin(),remotes_.end(),[&](const auto& value){return value.coordinate==coordinate;});
    if(found==remotes_.end())throw std::invalid_argument("Directed transport coordinate is local or absent");return *found;
}
void PooledImmutableDirectedTransport::requireCurrent()const{
    const auto storage=boundary_->pooledStorage();const auto& retained=storage->retainedPreparedSourceWorkers();
    if(!retained)throw std::logic_error("Directed source lost original worker cohort");
    const auto& memory=retained->source()->memory();const auto observed=storage->retainedOwnerObservation().authority();
    if(observed.get()!=memory.get() || observed.owner_before(memory) || memory.owner_before(observed) || retained->rootIdentity()!=storage->captureIdentity())
        throw std::logic_error("Directed source changed original worker/PMA/root authority");
    for(const auto& remote:remotes_){remote.pin->requireCurrent();remote.returned->requireCurrent();remote.free->requireCurrent();remote.victim->requireCurrent();remote.source->requireCurrent();remote.publication->requireCurrent();remote.read->requireCurrent();remote.retirement->requireCurrent();}
}
}
