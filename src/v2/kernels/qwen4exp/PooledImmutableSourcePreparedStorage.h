/**
 * @file PooledImmutableSourcePreparedStorage.h
 * @brief Existing-service-issued original PMA source/accepted-owner packet roots.
 *
 * Only ImmutableLookupService creates/reuses this registration. It owns no
 * provider or service handle (avoiding a source/storage lifetime cycle), no host
 * payload mirror and no live model counters. Source kernels and the existing
 * retained graph boundary keep the actual service binding alive independently.
 */
#pragma once
#include "PooledImmutableSourceGeometry.h"
#include "PooledImmutableNativeStorage.h"
namespace llaminar2::qwen4exp {
class ImmutableLookupService;
/** @brief One service channel's original native source storage and exact owner/coordinator associations. */
class PooledImmutableSourcePreparedStorage final {
public:
    /** @return Immutable exact original source packet geometry. */
    const PooledImmutableSourceGeometry& geometry()const noexcept{return geometry_;}
    /** @return The same original physical owner, retaining its actual slots and accepted grant banks. */
    const std::shared_ptr<PooledImmutableOwnerPreparedStorage>& owner()const noexcept{return owner_;}
    /** @return The same independent original policy coordinator. */
    const std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>& coordinator()const noexcept{return coordinator_;}
    /** @return Original canonical source frame/native root aliases; optional cache is always absent. */
    PooledImmutableNativeRoots roots()const;
    /** @return Complete actual PMA/native-runtime/address/geometry/owner identity; reject any drift. */
    std::uint64_t captureIdentity()const;
    /** @brief Join only original native setup producers on the exact future consumer stream. */
    void acquireSetup(void* exact_stream)const;
private:
    friend class ImmutableLookupService;
    friend class IPooledImmutableSourceKernels;
    /**
     * @brief Materialize exactly admitted source staging through the existing canonical root materializer.
     * @param geometry Actual source-issued checked geometry.
     * @param owner Same original owner roots retained by the source declaration.
     * @param coordinator Same original global directory/policy roots.
     * @param memory The service's original rank PMA; the caller already authenticated equality.
     * @param stream Exact ordinary setup stream; no metadata kernel or source request runs here.
     * @return Complete strong root owner or an exception with no published service registration.
     */
    static std::shared_ptr<PooledImmutableSourcePreparedStorage> materialize(PooledImmutableSourceGeometry geometry,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream);
    /** @brief Only successful original-service materialization publishes these retained associations. */
    PooledImmutableSourcePreparedStorage(PooledImmutableSourceGeometry geometry,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :geometry_(std::move(geometry)),owner_(std::move(owner)),coordinator_(std::move(coordinator)),roots_(std::move(roots)){}
    PooledImmutableSourceGeometry geometry_;
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner_;
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
};
}
