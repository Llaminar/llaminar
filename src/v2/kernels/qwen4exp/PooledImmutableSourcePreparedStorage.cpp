/**
 * @file PooledImmutableSourcePreparedStorage.cpp
 * @brief Exact physical source staging with no duplicate PMA ledger or host state shadow.
 */
#include "PooledImmutableSourcePreparedStorage.h"
#include "PooledImmutableNativeRootStorage.h"
#include <stdexcept>
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
    owner->captureIdentity();coordinator->captureIdentity();
    auto roots=detail::PooledNativeRootStorage::create(geometry.allocator(),{0,p.offsets.frame_bytes,p.offsets.native_bytes},p.identity,std::move(memory),stream);
    return std::shared_ptr<PooledImmutableSourcePreparedStorage>(new PooledImmutableSourcePreparedStorage(
        std::move(geometry),std::move(owner),std::move(coordinator),std::move(roots)));
}
PooledImmutableNativeRoots PooledImmutableSourcePreparedStorage::roots()const{return roots_->roots();}
std::uint64_t PooledImmutableSourcePreparedStorage::captureIdentity()const{
    auto value=roots_->identity();value=(value^owner_->captureIdentity())*1099511628211ULL;
    value=(value^coordinator_->captureIdentity())*1099511628211ULL;return value?value:1;
}
void PooledImmutableSourcePreparedStorage::acquireSetup(void* stream)const{owner_->acquireSetup(stream);coordinator_->acquireSetup(stream);roots_->acquire(stream);}
}
