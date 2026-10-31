/**
 * @file PooledImmutableSourcePreparedStorage.cpp
 * @brief Exact physical source staging with no duplicate PMA ledger or host state shadow.
 */
#include "PooledImmutableSourcePreparedStorage.h"
#include "PooledImmutableNativeRootStorage.h"
#include <stdexcept>
#include <limits>
namespace llaminar2::qwen4exp {
std::shared_ptr<PooledImmutableSourcePreparedStorage> PooledImmutableSourcePreparedStorage::materialize(
    PooledImmutableSourceGeometry geometry,std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream)
{
    const auto& p=geometry.parameters();
    if(!owner || !coordinator || owner->plan().parameters().identity!=p.owner.identity || coordinator->plan().parameters().identity!=p.coordinator.identity)
        throw std::invalid_argument("Pooled source cannot replace its original accepted-grant or coordinator roots");
    // This first production source slice has one exact local delivery edge.
    // The declaration itself remains independent; absent routed/native edges
    // are diagnosed before materialization rather than inferred from a cohort.
    if(p.owner.owner.participants()!=1 || p.owner.owner.coordinate()!=0 || p.owner.device!=p.coordinator.device ||
        owner->plan().allocator()!=coordinator->plan().allocator())
        throw std::invalid_argument("Pooled source edge requires the pending explicit directed owner transport; local delivery cannot represent this declared edge");
    // Equal-shaped certificates are not authority. Authenticate the original
    // claim owners before source allocation or service registration can publish.
    owner->requireAuthority(memory);coordinator->requireAuthority(memory);
    owner->captureIdentity();coordinator->captureIdentity();
    auto roots=detail::PooledNativeRootStorage::create(geometry.allocator(),{0,p.offsets.frame_bytes,p.offsets.native_bytes},p.identity,std::move(memory),stream);
    return std::shared_ptr<PooledImmutableSourcePreparedStorage>(new PooledImmutableSourcePreparedStorage(
        std::move(geometry),std::move(owner),std::move(coordinator),std::move(roots)));
}
PooledImmutableNativeRoots PooledImmutableSourcePreparedStorage::roots()const{return roots_->roots();}
void PooledImmutableSourcePreparedStorage::requireFor(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,const ImmutableRowTableSource& source,
    const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const{
    owner_->requireAuthority(memory);coordinator_->requireAuthority(memory);roots_->requireAuthority(memory);
    const auto& original=owner_->plan().parameters();
    const auto owner=PooledImmutableOwnerNativePlan::fromGeometry(physical,coordinator,original.owner.coordinate(),
        original.table,original.source_storage_generation);
    const auto policy=PooledImmutableCoordinatorNativePlan::fromGeometry(physical,coordinator,
        original.table,original.source_storage_generation);
    const auto expected=PooledImmutableSourceGeometry::fromPlans(owner,policy,source);
    if(owner.parameters().identity!=original.identity || policy.parameters().identity!=coordinator_->plan().parameters().identity ||
       expected.parameters().identity!=geometry_.parameters().identity)
        throw std::logic_error("Pooled source reuse changed original source, policy or physical root declarations");
    captureIdentity();
}
ImmutableLookupNativeOwnerObservation PooledImmutableSourcePreparedStorage::retainedOwnerObservation()const{
    const auto source=roots_->retainedOwnerObservation();std::size_t storage=source.tableStorageBytes(),staging=source.lookupStagingBytes();
    for(const auto& part:{owner_->retainedOwnerObservation(),coordinator_->retainedOwnerObservation()}){
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
    auto value=roots_->identity();value=(value^owner_->captureIdentity())*1099511628211ULL;
    value=(value^coordinator_->captureIdentity())*1099511628211ULL;return value?value:1;
}
void PooledImmutableSourcePreparedStorage::acquireSetup(void* stream)const{owner_->acquireSetup(stream);coordinator_->acquireSetup(stream);roots_->acquire(stream);}
}
