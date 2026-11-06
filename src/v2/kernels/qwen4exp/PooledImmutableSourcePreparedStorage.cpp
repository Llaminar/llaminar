/**
 * @file PooledImmutableSourcePreparedStorage.cpp
 * @brief Exact physical source staging with no duplicate PMA ledger or host state shadow.
 *
 * The same source geometry also declares its closed setup/publication event
 * program before native materialization; no independent event count is authored.
 */
#include "PooledImmutableSourcePreparedStorage.h"
#include "PooledImmutableNativeRootStorage.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include <stdexcept>
#include <limits>
namespace llaminar2::qwen4exp {
std::shared_ptr<RetainedParticipantGraphExecutable> PooledImmutableSourcePreparedStorage::acquireOriginalRemoteDefinition(
    const PooledRemoteGraphProgram& program,void* stream)
{
    const auto& g=program.geometry();const auto coordinate=g.coordinate;
    if(!prepared_workers_ || prepared_workers_->rootIdentity()!=captureIdentity() ||
       coordinate>=geometry_.originalOwners().size() || coordinate>=prepared_workers_->owners().size() ||
       g.coordinator!=geometry_.allocator() || g.owner!=geometry_.originalOwners()[coordinate].allocator() ||
       g.owner==geometry_.allocator() || g.rowBytes!=geometry_.parameters().owner.row_bytes ||
       g.sourceRows!=geometry_.parameters().coordinator.policy.sourceRows())
        throw std::invalid_argument("Remote definition slot changed its original source/coordinator/owner roots");
    const auto& original=prepared_workers_->owners()[coordinate];
    const auto memory=prepared_workers_->source()->memory();
    if(!remote_definition_slots_)
        remote_definition_slots_=RetainedParticipantGraphSourceSlots::create(captureIdentity(),geometry_.originalOwners().size(),memory);
    // The existing boundary cohort claim serializes this cold source factory.
    // The holder separately rejects surviving runner aliases before rearming.
    return remote_definition_slots_->acquire(captureIdentity(),original,stream,program);
}
void PooledImmutableSourcePreparedStorage::releaseOriginalRemoteDefinition(const PooledRemoteGraphProgram& program,
    const std::shared_ptr<RetainedParticipantGraphExecutable>& owner)
{
    if(!prepared_workers_ || prepared_workers_->rootIdentity()!=captureIdentity() || !remote_definition_slots_)
        throw std::logic_error("Remote definition retirement lost its original source slot holder");
    remote_definition_slots_->release(captureIdentity(),program,owner);
}
void PooledImmutableSourcePreparedStorage::abandonOriginalRemoteDefinition(const PooledRemoteGraphProgram& program,
    const std::shared_ptr<RetainedParticipantGraphExecutable>& owner,std::exception_ptr failure)
{
    if(!remote_definition_slots_)throw std::logic_error("Failed source definition has no original slot holder");
    remote_definition_slots_->abandon(captureIdentity(),program,owner,std::move(failure));
}
void PooledImmutableSourcePreparedStorage::bindPreparedSourceWorkers(std::shared_ptr<NativeContextPreparation> source,
    PreparedGPUContextUsePermit coordinator,std::vector<PreparedGPUContextUsePermit> owners){
    if(prepared_workers_ || !source || owners.size()!=owners_.size())
        throw std::logic_error("Prepared source worker materialization is missing, duplicate or incomplete");
    const auto observed=retainedOwnerObservation();const auto& memory=source->memory();
    if(observed.authority().get()!=memory.get() || observed.authority().owner_before(memory) || memory.owner_before(observed.authority()))
        throw std::logic_error("Prepared source roots have a foreign original PMA/control block");
    const auto authenticate=[&](const PreparedGPUContextUsePermit& permit,PhysicalMemoryAllocatorIdentity allocator){
        if(permit.memory().get()!=memory.get() || permit.memory().owner_before(memory) || memory.owner_before(permit.memory()) ||
           allocator!=PhysicalMemoryAllocatorIdentity{memory->worldRank(),permit.device()})
            throw std::logic_error("Materialized native source has a foreign physical worker/authority");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(permit.device(),memory);
        original.requireWorker(permit.worker());
        if(original.generation()!=permit.generation())throw std::logic_error("Materialized native source worker generation changed");
    };
    authenticate(coordinator,coordinator_->plan().allocator());
    for(std::size_t i=0;i<owners.size();++i)authenticate(owners[i],owners_[i]->plan().allocator());
    // The private ModelContext caller reaches here only after all original native
    // initializers and source materialization completed, never at prebinding.
    prepared_workers_=PreparedImmutableSourceWorkers(std::move(source),std::move(coordinator),std::move(owners),captureIdentity());
}
std::shared_ptr<PooledImmutableSourcePreparedStorage> PooledImmutableSourcePreparedStorage::materialize(
    PooledImmutableSourceGeometry geometry,std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,
    void* stream,std::unique_ptr<NativeOrderingEventGroup> group)
{
    const auto& p=geometry.parameters();
    if(!coordinator || owners.size()!=geometry.originalOwners().size() || coordinator->plan().parameters().identity!=p.coordinator.identity)
        throw std::invalid_argument("Pooled source replaced its original complete coordinator/cohort");
    for(const auto& owner:owners)if(owner && owner->plan().allocator()!=coordinator->plan().allocator()){
        const auto a=owner->plan().allocator(),c=coordinator->plan().allocator();
        ImmutableRowTableDirectedTransportPolicy::require(c, a, owner->plan().peerAccess());
    }
    coordinator->requireAuthority(memory);coordinator->captureIdentity();
    for(std::size_t i=0;i<owners.size();++i){
        if(!owners[i] || owners[i]->plan().parameters().identity!=geometry.originalOwners()[i].parameters().identity)
            throw std::invalid_argument("Pooled source replaced original ordered owner roots");
        coordinator->plan().requireOwnerPlan(owners[i]->plan());owners[i]->requireAuthority(memory);owners[i]->captureIdentity();
    }
    auto roots=group ? detail::PooledNativeRootStorage::createWithOrderingGroup(geometry.allocator(),
        {0,p.offsets.frame_bytes,p.offsets.native_bytes},p.identity,std::move(memory),stream,geometry.orderingProgram(),std::move(group))
        : detail::PooledNativeRootStorage::create(geometry.allocator(),{0,p.offsets.frame_bytes,p.offsets.native_bytes},p.identity,std::move(memory),stream,geometry.orderingProgram());
    return std::shared_ptr<PooledImmutableSourcePreparedStorage>(new PooledImmutableSourcePreparedStorage(
        std::move(geometry),std::move(owners),std::move(coordinator),std::move(roots)));
}
PooledImmutableNativeRoots PooledImmutableSourcePreparedStorage::roots()const{return roots_->roots();}
void PooledImmutableSourcePreparedStorage::requireFor(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,const ImmutableRowTableSource& source,
    std::uint32_t expected_compute_readers,const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const{
    coordinator_->requireAuthority(memory);roots_->requireAuthority(memory);
    const auto& original=coordinator_->plan().parameters();std::vector<PooledImmutableOwnerNativePlan> declarations;
    for(std::size_t i=0;i<owners_.size();++i){owners_[i]->requireAuthority(memory);declarations.push_back(PooledImmutableOwnerNativePlan::fromGeometry(physical,coordinator,i,original.table,original.source_storage_generation));
        if(declarations.back().parameters().identity!=owners_[i]->plan().parameters().identity)throw std::logic_error("Pooled source changed original physical owner declaration");}
    const auto policy=PooledImmutableCoordinatorNativePlan::fromGeometry(physical,coordinator,original.table,original.source_storage_generation);
    if(policy.parameters().identity!=original.identity)
        throw std::logic_error("Pooled source changed its original coordinator policy");
    geometry_.requireOriginalPlans(declarations,policy,source,expected_compute_readers);
    captureIdentity();
}
ImmutableLookupNativeOwnerObservation PooledImmutableSourcePreparedStorage::retainedOwnerObservation()const{
    const auto source=roots_->retainedOwnerObservation();std::size_t storage=source.tableStorageBytes(),staging=source.lookupStagingBytes();
    std::vector<ImmutableLookupNativeOwnerObservation> parts{coordinator_->retainedOwnerObservation()};
    if(geometry_.parameters().local_owner!=adaptive_lookup::None)parts.push_back(owner()->retainedOwnerObservation());
    // Remote roots retain/report their own allocator claims; never fold them into this source allocator.
    for(const auto& part:parts){
        if(part.authority()!=source.authority() || part.allocator()!=source.allocator())
            throw std::logic_error("Local pooled lease observation has foreign original authorities or allocators");
        if(part.tableStorageBytes()>std::numeric_limits<std::size_t>::max()-storage ||
           part.lookupStagingBytes()>std::numeric_limits<std::size_t>::max()-staging)
            throw std::overflow_error("Pooled actual physical lease observation overflows");
        storage+=part.tableStorageBytes();staging+=part.lookupStagingBytes();
    }
    return {source.authority(),source.allocator(),storage,staging};
}
std::uint64_t PooledImmutableSourcePreparedStorage::captureIdentity()const{
    auto value=roots_->identity();for(const auto& owner:owners_)value=(value^owner->captureIdentity())*1099511628211ULL;
    value=(value^coordinator_->captureIdentity())*1099511628211ULL;return value?value:1;
}
void PooledImmutableSourcePreparedStorage::acquireSetup(void* stream)const{if(geometry_.parameters().local_owner!=adaptive_lookup::None)owner()->acquireSetup(stream);coordinator_->acquireSetup(stream);roots_->acquire(stream);}
}
