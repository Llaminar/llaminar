/**
 * @file ImmutableLookupNativeOwnership.h
 * @brief Native setup stream and immutable observations authored only by actual pooled PMA lease owners.
 *
 * Observations retain the original authority and exact live claims. They grant
 * neither materialization nor a lookup pin and never copy device cache state.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <utility>
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableSourcePreparedStorage;namespace detail {class PooledNativeRootStorage;}}
/** @brief Original participant's exact ordinary native metadata initialization stream. */
struct AdaptiveImmutableLookupDevicePreparation final {DeviceId device;void* stream;};
/** @brief Immutable lease observation, distinct from planner root declarations or live occupancy. */
class ImmutableLookupNativeOwnerObservation final {
public:
    /** @return Original sole physical-memory authority retained by the actual roots. */
    const std::shared_ptr<const PhysicalMemoryAuthority>& authority()const noexcept{return authority_;}
    /** @return Actual original physical allocator. */
    PhysicalMemoryAllocatorIdentity allocator()const noexcept{return allocator_;}
    /** @return Actual retained native cache/policy leases; zero at C0. */
    std::size_t tableStorageBytes()const noexcept{return storage_;}
    /** @return Actual retained frame/native/source staging leases. */
    std::size_t lookupStagingBytes()const noexcept{return staging_;}
private:
    friend class qwen4exp::detail::PooledNativeRootStorage;
    friend class qwen4exp::PooledImmutableSourcePreparedStorage;
    /** @brief Only actual physical owners can construct exact lease observations.
     * @param authority Original retained PMA. @param allocator Actual physical endpoint.
     * @param storage Retained cache/policy claims. @param staging Retained frame/native claims. */
    ImmutableLookupNativeOwnerObservation(std::shared_ptr<const PhysicalMemoryAuthority> authority,
        PhysicalMemoryAllocatorIdentity allocator,std::size_t storage,std::size_t staging)
        :authority_(std::move(authority)),allocator_(allocator),storage_(storage),staging_(staging){}
    std::shared_ptr<const PhysicalMemoryAuthority> authority_;
    PhysicalMemoryAllocatorIdentity allocator_;
    std::size_t storage_,staging_;
};
}
