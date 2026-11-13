/**
 * @file PooledImmutableLocalHotPreparedStorage.h
 * @brief Original-PMA local native roots, bounded response and independent all-hit readiness.
 *
 * The original generic pooled root materializer retains all three native leases.
 * A canonical mapped word and the original native parent producer frontier
 * order LocalRead without lower-tier service on an all-hit query.
 */
#pragma once
#include "PooledImmutableLocalHotNativePlan.h"
#include "PooledImmutableNativeStorage.h"
#include "transfer/TransferEngine.h"
#include "PooledImmutableLocalHotNativeTypes.h"
#include <functional>
namespace llaminar2 {class DeviceGraphOrchestrator;}
namespace llaminar2::qwen4exp {
class IPooledImmutableLocalHotKernels;
/** @brief One exact participant's retained physical roots; no mutable host residency mirror exists. */
class PooledImmutableLocalHotPreparedStorage final {
public:
    /**
     * @brief Materialize only the separately admitted local roots over the SAME original source and PMA.
     * @param geometry Exact admitted per-consumer local geometry.
     * @param original_source Genuine service-issued existing pool source, retained without duplication.
     * @param memory Same original canonical rank authority that owns source and local BOMs.
     * @param exact_setup_stream Non-null non-default cold initialization stream on this consumer.
     * @return One retained owner; native metadata initialization is submitted by the checked kernel bundle.
     * @throws std::exception Before any foreign source/PMA/stream or unadmitted native creation.
     */
    static std::shared_ptr<PooledImmutableLocalHotPreparedStorage> create(
        PooledImmutableLocalHotGeometry geometry,
        std::shared_ptr<PooledImmutableSourcePreparedStorage> original_source,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* exact_setup_stream);
    /** @brief Retire actual native/event/mapped users before their original claims release. */
    ~PooledImmutableLocalHotPreparedStorage();
    /** @return Exact admitted local declaration. */
    const PooledImmutableLocalHotGeometry& geometry()const noexcept{return plan_.geometry();}
    /** @return SAME original source owner/control block used at factory admission. */
    const std::shared_ptr<PooledImmutableSourcePreparedStorage>& originalSource()const noexcept{return plan_.originalSource();}
    /** @brief Authenticate exact resolved geometry, original shared owner and SAME original authority on reuse. */
    void requireFor(const PooledImmutableLocalHotGeometry&,
        const std::shared_ptr<PooledImmutableSourcePreparedStorage>&,
        const std::shared_ptr<const PhysicalMemoryAuthority>&)const;
    /** @return Actual separately retained CPU readiness-page claim, never duplicated as GPU native staging. */
    ImmutableLookupNativeOwnerObservation retainedReadyObservation()const;
    /** @return Original immutable native table/geometry association. */
    const PooledImmutableLocalHotNativePlan& plan()const noexcept{return plan_;}
    /** @return Exact optional cache/mandatory frame/scratch tensor aliases. */
    PooledImmutableNativeRoots roots()const;
    /** @return Actual LOCAL native lease observation; append beside existing pool observations, never aggregate it twice. */
    ImmutableLookupNativeOwnerObservation retainedOwnerObservation()const;
    /** @brief Reject another PMA even if its certificate has equal capacity. */
    void requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const;
    /** @return Same canonical mapped region retaining its original host allocation claim. */
    const std::shared_ptr<MappedHostTransferRegion>& readyRegion()const noexcept;
    /** @brief Join original cold setup event on the exact consumer stream, without host observation. */
    void acquireSetup(void* exact_stream)const;
    /** @brief Enqueue original native unsigned-GEQ readiness wait on an exact stream. */
    void enqueueReadyWait(void* exact_main_stream)const;
    /** @brief Acquire the actual sole LocalRead completion on aux before policy/fill/retirement. */
    void enqueueLocalReadCompleteWait(void* exact_aux_stream)const;
    /** @return Actual original generic native-root identity, never a host-created query nonce. */
    std::uint64_t consumerStorageIdentity()const;
    /** @return Complete physical/stream/table/generation identity, excluding mutable device values. */
    std::uint64_t captureIdentity()const;
    PooledImmutableLocalHotPreparedStorage(const PooledImmutableLocalHotPreparedStorage&)=delete;
    PooledImmutableLocalHotPreparedStorage& operator=(const PooledImmutableLocalHotPreparedStorage&)=delete;
private:
    friend class ::llaminar2::DeviceGraphOrchestrator;
    /** @brief Only the actual runner terminal consumer may advance this original storage's immutable reporting baseline. */
    local_hot::TerminalPublication observeCompletedTerminal(const local_hot::TerminalObservation&);
    friend class IPooledImmutableLocalHotKernels;
    friend class PooledImmutableLocalHotOperands;
    struct Impl;
    /** @brief Construct after the complete exact cold owner is admitted. */
    PooledImmutableLocalHotPreparedStorage(PooledImmutableLocalHotNativePlan plan,std::unique_ptr<Impl> impl);
    /** @return Immutable actual native root projections, without observing any device value. */
    local_hot::NativeRoots nativeRoots()const;
    /** @brief Exact same backend cold initializer, bracketed by original root setup publication. */
    void initializeNative(DeviceType,NativeEmbeddingIQTables,void*,
        const std::function<void(const local_hot::NativeRoots&)>&);
    /** @brief Reject launch before the actual ordinary metadata initializer submitted. */
    void requireInitialized()const;
    PooledImmutableLocalHotNativePlan plan_;
    std::unique_ptr<Impl> impl_;
};
} // namespace llaminar2::qwen4exp
