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
#include "collective/NativeLocalTPContextBorrow.h"
#include "PooledImmutableNativePlan.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "loaders/ImmutableLookupNativeOwnership.h"
#include "tensors/TensorClasses.h"
#include "collective/NativeCollectiveCompletion.h"
#include <memory>
namespace llaminar2 {class Qwen4ExpReplicatedPLECohort;class NativeOrderingEventGroup;namespace test {struct PooledImmutableFactoryReplayAccess;}}
namespace llaminar2 { class CapturedTransferBinding; class CapturedTransferReceiveCompletion; }
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
    /** @return Original optional invocation-channel capture identity, separate from immutable physical roots. */
    std::uint64_t invocationChannelIdentity() const;
    /** @return Original bound receive receipt; this never reads device progress on the host. */
    CapturedTransferReceiveCompletion invocationReceiveCompletion(void* exact_stream) const;
    /** @return Complete prepared geometry/allocator/runtime/root identity, rejecting stale storage. */
    std::uint64_t captureIdentity()const;
    /** @brief Validate actual roots and join only their original setup event on the exact consumer stream. */
    void acquireSetup(void* exact_stream)const;
    PooledImmutableOwnerPreparedStorage(const PooledImmutableOwnerPreparedStorage&)=delete;
    PooledImmutableOwnerPreparedStorage& operator=(const PooledImmutableOwnerPreparedStorage&)=delete;
private:
    friend struct ::llaminar2::test::PooledImmutableFactoryReplayAccess;
    /**
     * @brief Shared factory body; only the closed fixture friend may supply an exact recorded group.
     * @param plan Original checked native root geometry and closed initialization program.
     * @param memory Same final admitted authority that owns every root and group.
     * @param stream Exact non-default setup producer, unchanged through initialization.
     * @param group Closed original claim, or empty only for the public installed-selector path.
     * @return Original storage wrapper retaining the common root owner.
     * @throws std::exception Before creators for foreign claims, or on actual allocation failure.
     */
    static std::shared_ptr<PooledImmutableOwnerPreparedStorage> createImpl(PooledImmutableOwnerNativePlan plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,std::unique_ptr<NativeOrderingEventGroup> group);
    friend class IPooledImmutableNativeKernels;
    friend class PooledImmutableCapturedPinEdge;
    friend class PooledImmutableTransferBinding;
    /** @brief Release only this exact cold/captured invocation after its original graphs have retired. */
    void releaseInvocationChannel(const CapturedTransferBinding& original);
    /** @brief Drop one original immutable native cohort borrow after actual parent retirement, never on request reset. */
    void releaseNativeInvocation(const NativeLocalTPContextBorrow& original, int participant);

    /** @brief Bind once to the original owner invocation receiver before any graph is recorded. */
    void bindInvocationChannel(const CapturedTransferBinding& original);
    /** @brief Bind an original native transport member’s abort capability before any parent capture. */
    void bindNativeInvocation(NativeLocalTPContextBorrow cohort, int participant);
    /** @return Original immutable native communicator/abort identity; absent on mapped/local owners. */
    ncclLlaminarCompletionAuthority nativeInvocationAuthority() const;

    /** @brief Construct only around the one exact materialized claim owner. */
    PooledImmutableOwnerPreparedStorage(PooledImmutableOwnerNativePlan plan,std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :plan_(std::move(plan)),roots_(std::move(roots)){}
    PooledImmutableOwnerNativePlan plan_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
    /** Destroy the original allocation borrow before releasing its physical root/claim owner. */
    std::shared_ptr<const CapturedTransferBinding> original_invocation_;
    std::optional<NativeLocalTPContextBorrow> native_invocation_cohort_;
    int native_invocation_participant_ = -1;
    ncclLlaminarCompletionAuthority native_invocation_authority_{}; ///< Immutable original borrow remains valid through abort and physical retirement.
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
    friend struct ::llaminar2::test::PooledImmutableFactoryReplayAccess;
    /**
     * @brief Shared factory body; only the closed fixture friend may supply an exact recorded group.
     * @param plan Original checked native root geometry and closed initialization program.
     * @param memory Same final admitted authority that owns every root and group.
     * @param stream Exact non-default setup producer, unchanged through initialization.
     * @param group Closed original claim, or empty only for the public installed-selector path.
     * @return Original storage wrapper retaining the common root owner.
     * @throws std::exception Before creators for foreign claims, or on actual allocation failure.
     */
    static std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> createImpl(PooledImmutableCoordinatorNativePlan plan,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,std::unique_ptr<NativeOrderingEventGroup> group);
    friend class IPooledImmutableNativeKernels;
    /** @brief Only create() binds the exact canonical materialized claim owner. */
    PooledImmutableCoordinatorPreparedStorage(PooledImmutableCoordinatorNativePlan plan,std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :plan_(std::move(plan)),roots_(std::move(roots)){}
    PooledImmutableCoordinatorNativePlan plan_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
};
}
