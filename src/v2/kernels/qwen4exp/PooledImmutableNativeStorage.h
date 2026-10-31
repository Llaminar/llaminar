/**
 * @file PooledImmutableNativeStorage.h
 * @brief Strong exact PMA-backed persistent roots for participant-local pooled phases.
 *
 * Factories claim the exact tensor allocation request before materialization.
 * Root subrange alignment never substitutes for physical allocation extent.
 * Native allocation borrowers, runtime seals and one setup-only retirement event
 * retain actual bytes/claims. No GPU execution value is copied to the host.
 */
#pragma once
#include "PooledImmutableNativePlan.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "loaders/ImmutableLookupNativeOwnership.h"
#include "tensors/TensorClasses.h"
#include <memory>
namespace llaminar2::qwen4exp
{
namespace detail {class PooledNativeRootStorage;}
class IPooledImmutableNativeKernels;
/** @brief Original canonical tensor aliases, each retaining the actual allocation and its PMA claims. */
struct PooledImmutableNativeRoots
{
    std::shared_ptr<TensorBase> cache; ///< Absent exactly for static C0/no-policy geometry.
    std::shared_ptr<TensorBase> frame;
    std::shared_ptr<TensorBase> native;
};
/** @brief One exact owner allocation set; request resets preserve every physical root and capture identity. */
class PooledImmutableOwnerPreparedStorage final
{
public:
    /**
     * @brief Materialize only exact already-admitted owner/cache/frame/native roots outside capture.
     * @param plan Closed original geometry; it chooses no additional capacity.
     * @param memory Original canonical rank PMA, with complete composed source/transport/executable BOM.
     * @param setup_stream Exact non-default ordinary allocation stream.
     * @return One retained storage owner with no host payload mirror and no initialized cache.
     * @throws std::exception Before foreign/unadmitted/default-stream materialization or on allocation failure.
     */
    static std::shared_ptr<PooledImmutableOwnerPreparedStorage> create(PooledImmutableOwnerNativePlan plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* setup_stream);
    /** @brief Last-owner teardown drains only its actual setup producer, before freeing bytes then claims. */
    ~PooledImmutableOwnerPreparedStorage()=default;
    /** @return Complete immutable owner geometry. */
    const PooledImmutableOwnerNativePlan& plan()const noexcept{return plan_;}
    /** @return Actual canonical root aliases, with no replacement or allocation. */
    PooledImmutableNativeRoots roots()const;
    /** @brief Authenticate the actual original PMA, never an equal-sized replacement certificate. */
    void requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const;
    /** @return Actual retained physical root claims without device state observation. */
    ImmutableLookupNativeOwnerObservation retainedOwnerObservation()const;
    /** @return Complete prepared geometry/allocator/runtime/root identity, rejecting stale storage. */
    std::uint64_t captureIdentity()const;
    /** @brief Validate actual roots and join only their original setup event on the exact consumer stream. */
    void acquireSetup(void* exact_stream)const;
    PooledImmutableOwnerPreparedStorage(const PooledImmutableOwnerPreparedStorage&)=delete;
    PooledImmutableOwnerPreparedStorage& operator=(const PooledImmutableOwnerPreparedStorage&)=delete;
private:
    friend class IPooledImmutableNativeKernels;
    /** @brief Construct only around the one exact materialized claim owner. */
    PooledImmutableOwnerPreparedStorage(PooledImmutableOwnerNativePlan plan,std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :plan_(std::move(plan)),roots_(std::move(roots)){}
    PooledImmutableOwnerNativePlan plan_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
};
/** @brief One independent coordinator allocation set, including complete static P geometry. */
class PooledImmutableCoordinatorPreparedStorage final
{
public:
    /**
     * @brief Materialize exact already-admitted independent coordinator roots outside capture.
     * @param plan Closed policy/coalescing/native declaration; no owner/consumer identity is inferred.
     * @param memory Same canonical rank PMA that admitted every component contribution.
     * @param setup_stream Exact non-default ordinary allocation stream.
     * @return One retained original allocation set; static owner metadata is initialized by native setup.
     * @throws std::exception On foreign geometry/authority/stream or failed physical materialization.
     */
    static std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> create(PooledImmutableCoordinatorNativePlan plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* setup_stream);
    /** @brief Retire original setup before actual allocation/claim release. */
    ~PooledImmutableCoordinatorPreparedStorage()=default;
    /** @return Complete original native policy/owner-order declaration. */
    const PooledImmutableCoordinatorNativePlan& plan()const noexcept{return plan_;}
    /** @return Strong actual canonical root aliases. */
    PooledImmutableNativeRoots roots()const;
    /** @brief Authenticate the actual original PMA before model/source reuse. */
    void requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const;
    /** @return Actual original policy/frame/native root claims. */
    ImmutableLookupNativeOwnerObservation retainedOwnerObservation()const;
    /** @return Complete immutable native geometry and prepared physical identity. */
    std::uint64_t captureIdentity()const;
    /** @brief Acquire the exact existing setup producer event before graph capture; no host wait. */
    void acquireSetup(void* exact_stream)const;
    PooledImmutableCoordinatorPreparedStorage(const PooledImmutableCoordinatorPreparedStorage&)=delete;
    PooledImmutableCoordinatorPreparedStorage& operator=(const PooledImmutableCoordinatorPreparedStorage&)=delete;
private:
    friend class IPooledImmutableNativeKernels;
    /** @brief Only create() binds the exact canonical materialized claim owner. */
    PooledImmutableCoordinatorPreparedStorage(PooledImmutableCoordinatorNativePlan plan,std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :plan_(std::move(plan)),roots_(std::move(roots)){}
    PooledImmutableCoordinatorNativePlan plan_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
};
}
