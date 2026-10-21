/**
 * @file CpuAdaptiveImmutableRowCache.h
 * @brief Bounded original-native CPU RAM cache with independent or closed subordinate source access.
 *
 * This CPU storage component never owns GPU inference or a GPU residency map.
 * The independent factory rejects GPU-coordinated layouts. The subordinate
 * factory binds exact GPU requester membership, then bars independent reads;
 * only the existing acquired-request processor can authorize source work.
 * Existing whole-tier serving retains its original owner and semantics.
 * Every variable-sized array is bounded and included once in the source-owned
 * PMA BOM. Returned pins retain storage/source/authority through final release.
 */
#pragma once
#include "loaders/ImmutableRowTable.h"
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include "loaders/CpuImmutableRowCachePolicy.h"
#include "loaders/CpuSubordinateImmutableRowRead.h"
#include <memory>
#include <optional>
namespace llaminar2 {
namespace qwen4exp { class ImmutableLookupHostProcessor; }
class HostImmutableRowProvider;
class CpuAdaptiveImmutableRowCache;
/** @brief Immutable source-owned array geometry, without capacity selection or materialization authority. */
class CpuAdaptiveImmutableRowCacheGeometry final {
public:
    /**
     * @brief Authenticate one local host-coordinated adaptive RAM/SSD component.
     * @param layout Checked installed native layout; no GPU coordinator or positive GPU pools.
     * @param host Exact CPU service/allocator identity, independent of NUMA locality.
     * @param maximum_live_rows Mandatory maximum caller response rows, authored before quotas freeze.
     * @param policy Explicit replacement constants; epoch and victim ties follow published logical order.
     * @return Exact cache/SSD/mandatory-scratch geometry; no free-memory observation is made.
     * @throws std::invalid_argument For foreign source/service, unsupported domain, or invalid policy.
     * @throws std::overflow_error For unrepresentable bounded arrays or native rows.
     */
    static CpuAdaptiveImmutableRowCacheGeometry fromLayout(const ImmutableRowTableLayoutGeometry &layout,
        PhysicalMemoryAllocatorIdentity host,std::size_t maximum_live_rows,CpuImmutableRowCachePolicy policy = {});
    /**
     * @brief Project the same CPU owner as a closed subordinate of one declared GPU requester.
     * @param layout Exact adaptive RAM/SSD/GPU endpoint declaration; no replicas or fixed ranges.
     * @param host Exact local CPU service and PMA allocator.
     * @param requester Sole local GPU consumer; positive GPU slots require this existing coordinator.
     * @param policy CPU admission/eviction policy for ordinary authorized source reads.
     * @return Same source-owned CPU geometry, with independent read entrypoints barred.
     * @throws std::exception For foreign membership, zero/overflowing request bounds or unsupported endpoints.
     * @note Zero GPU slots preserve this access discipline without creating a residency coordinator.
     * Topology eligibility beyond exact endpoint membership belongs to the canonical planner.
     */
    static CpuAdaptiveImmutableRowCacheGeometry forSubordinateGPU(const ImmutableRowTableLayoutGeometry &layout,
        PhysicalMemoryAllocatorIdentity host,PhysicalMemoryAllocatorIdentity requester,CpuImmutableRowCachePolicy policy = {});
    /**
     * @brief Contribute exact source-owned storage and mandatory execution to canonical PMA.
     * @param resource Exact observed host resource from the canonical rank planner.
     * @return Native cache+SSD bytes under ImmutableTableStorage and request scratch under ImmutableLookupStaging.
     * @throws std::invalid_argument For a foreign physical resource.
     * @note Execution contribution must be included before expert quotas/graph rows freeze; cache storage is lowest priority.
     */
    PhysicalMemoryPlan bom(PhysicalMemoryResource resource) const;
    /** @return Exact SSD service plus request scratch before quotas freeze; resource identity must match this host. */
    PhysicalMemoryPlan mandatoryBOM(PhysicalMemoryResource resource) const;
    /** @return Only optional slot-sized native cache storage after quotas freeze; resource identity must match this host. */
    PhysicalMemoryPlan cacheBOM(PhysicalMemoryResource resource) const;
    /** @return Actual RAM cache slot count; zero still has a complete explicit SSD response path. */
    std::size_t slots() const noexcept { return slots_; }
    /** @return Exact page-rounded native cache payload/index/policy allocation, excluding original SSD service. */
    std::size_t storageBytes() const noexcept { return storage_bytes_; }
    /** @return Exact original source service bytes from its unchanged bounded-mmap factory. */
    std::size_t sourceBytes() const noexcept { return source_bytes_; }
    /** @return Exact mandatory bounded request dedup array, excluding caller-owned response. */
    std::size_t executionBytes() const noexcept { return execution_bytes_; }
    /** @return Original native source geometry retained without requantization or a payload hash. */
    const ImmutableRowTableExtent &extent() const noexcept { return extent_; }
private:
    friend class CpuAdaptiveImmutableRowCache;
    /** @brief Share exact array projection between checked independent/subordinate factories. */
    static CpuAdaptiveImmutableRowCacheGeometry project(const ImmutableRowTableLayoutGeometry &layout,
        PhysicalMemoryAllocatorIdentity host,std::size_t maximum_live_rows,CpuImmutableRowCachePolicy policy,
        std::optional<PhysicalMemoryAllocatorIdentity> requester);
    /** @brief Factory-only source/endpoint projection; array dimensions are completed by the checked factory. */
    CpuAdaptiveImmutableRowCacheGeometry(ImmutableRowTableExtent extent,PhysicalMemoryAllocatorIdentity host,
        ImmutableRowTableStoragePolicy source_policy,ImmutableRowTableHostPlacement locality,
        CpuImmutableRowCachePolicy policy):extent_(extent),host_(host),source_policy_(source_policy),
        locality_(locality),policy_(policy) {}
    ImmutableRowTableExtent extent_;
    PhysicalMemoryAllocatorIdentity host_;
    ImmutableRowTableStoragePolicy source_policy_;
    ImmutableRowTableHostPlacement locality_;
    CpuImmutableRowCachePolicy policy_;
    std::optional<PhysicalMemoryAllocatorIdentity> requester_;
    std::size_t slots_=0,maximum_live_rows_=0,index_count_=0,sketch_count_=0,request_count_=0,protected_limit_=0;
    std::size_t index_offset_=0,heap_offset_=0,sketch_offset_=0,payload_offset_=0;
    std::size_t plan_offset_=0,scatter_offset_=0,grant_offset_=0,read_rows_offset_=0;
    std::size_t storage_bytes_=0,source_bytes_=0,execution_bytes_=0;
};
/**
 * @brief Actual host cache work; source-window counters cover completed source rows only.
 * A throwing source gather may have additional partial mapping work available
 * from sourceStats(); zero successful-row work never proves zero failed I/O.
 */
struct CpuImmutableRowCacheStats final {
    std::uint64_t completed_batches=0,completed_rows=0,completed_bytes=0,failures=0;
    std::uint64_t ram_rows=0,source_rows=0,duplicate_rows=0,admissions=0,evictions=0,admission_rejections=0;
    std::uint64_t all_pinned_responses=0,source_window_hits=0,source_window_misses=0,source_window_evictions=0;
    std::size_t resident_rows=0,pinned_rows=0;
    std::uint64_t invalidations=0; ///< Successful removal of a RAM row from new subordinate reads.
};
/** @brief Work authored inside one successful gather's mutex interval; no racing lifetime-counter sampling. */
struct CpuImmutableRowGatherObservation final {
    std::uint64_t rows=0,bytes=0,ram_rows=0,source_rows=0,duplicate_rows=0,admissions=0,evictions=0;
    std::uint64_t admission_rejections=0,all_pinned_responses=0;
    ImmutableRowGatherObservation windows;
    std::uint64_t policy_ordinal=0; ///< Owner-minted successful CPU policy order; never a GPU channel sequence.
    std::uint64_t invalidated_rows=0; ///< Exact per-call RAM detachments; never inferred GPU publications.
};
/** @brief One exact native source and bounded CPU residency authority, with no public SSD-service escape. */
class CpuAdaptiveImmutableRowCache final {
    struct Impl;
public:
    /** @brief Move-only immutable resident-row borrow; its last owner retains source, physical storage and PMA. */
    class Pin final {
    public:
        ~Pin();
        Pin(Pin &&) noexcept;
        Pin &operator=(Pin &&) noexcept;
        Pin(const Pin &)=delete;
        Pin &operator=(const Pin &)=delete;
        /** @return Generation-checked original native bytes; moved-from pins reject and never expose freed storage. */
        std::span<const std::byte> bytes() const;
        /** @return Exact source row admitted by this immutable grant. */
        std::uint64_t row() const noexcept { return row_; }
        /** @return Slot reuse generation; it is not a source/arena identity. */
        std::uint64_t generation() const noexcept { return generation_; }
    private:
        friend class CpuAdaptiveImmutableRowCache;
        /** @brief Only a successful resident pin may publish a retained native view. */
        Pin(std::shared_ptr<Impl> owner,std::uint32_t slot,std::uint64_t row,std::uint64_t generation)
            :owner_(std::move(owner)),slot_(slot),row_(row),generation_(generation) {}
        /** @brief Release exactly one generation-matched pin, including after facade retirement; invariant failure is fatal. */
        void release() noexcept;
        std::shared_ptr<Impl> owner_;
        std::uint32_t slot_=0;
        std::uint64_t row_=0,generation_=0;
    };
    /**
     * @brief Create one host cache and its private existing explicit SSD source under the same PMA.
     * @param source Original loader-authenticated native descriptor; source metadata is authenticated before cache allocation.
     * @param geometry Exact checked source/host/array geometry.
     * @param memory Sole rank-local PMA which admitted all three contributions.
     * @return Complete owner; no mutable row or partial cache escapes on failure.
     * @throws std::exception For foreign source/authority, missing admission, NUMA rejection or native I/O failure.
     */
    static std::unique_ptr<CpuAdaptiveImmutableRowCache> create(ImmutableRowTableSource source,
        CpuAdaptiveImmutableRowCacheGeometry geometry,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /**
     * @brief Authenticate an open source owner for setup/reuse without issuing a gather.
     * @throws std::runtime_error For retired/failed owners or changed source metadata.
     * @throws std::exception For underlying source metadata I/O or observation overflow.
     * @details The same owner mutex orders authentication with gathers, pins and
     * retirement. No policy ordinal, recency, cache work, successful-gather counter
     * or physical claim changes. The private source may record its actual metadata
     * check. Source authentication failure poisons new owner work without claiming
     * a failed gather; existing pin lifetime and generation validation are unchanged.
     */
    void authenticateSource() const;
    ~CpuAdaptiveImmutableRowCache();
    CpuAdaptiveImmutableRowCache(const CpuAdaptiveImmutableRowCache &)=delete;
    CpuAdaptiveImmutableRowCache &operator=(const CpuAdaptiveImmutableRowCache &)=delete;
    /**
     * @brief Copy original rows synchronously, deduplicating source work and restoring exact caller order.
     * @param ids Complete valid source row IDs; validate the full frame before any output/cache mutation.
     * @param destination Exact caller-owned response bytes, already admitted by its execution owner.
     * @return This successful call's actual work and owner-minted policy ordinal, linearized under the copy mutex.
     * @throws std::exception For subordinate access, invalid/alias/stale-source/faulted-state inputs; I/O failure publishes no successful batch.
     * @note Zero/all-pinned caches serve through the explicit SSD response path without allocation, wait or mode substitution.
     */
    CpuImmutableRowGatherObservation gather(std::span<const std::int64_t> ids,std::span<std::byte> destination);
    /**
     * @brief Pin an existing native row without reading a missing source or altering logical replacement recency.
     * @param row Valid source row ID.
     * @return Immutable generation-retaining borrow, or absence if the row is not resident.
     * @throws std::exception For subordinate access, retired/faulted/stale source or invalid row/pin overflow.
     */
    std::optional<Pin> tryPinResident(std::int64_t row);
    /** @return Exact admitted request ID capacity, independent of cache slot count. */
    std::size_t maximumLiveRows() const noexcept;
    /** @brief Bar new work without waiting; outstanding pins retain their exact backing until final release. */
    void retire() noexcept;
    /**
     * @brief Authenticate open storage and observe only this owner's actual live leases.
     * @return Original source plus cache storage bytes, request-staging bytes and exact host/locality/PMA.
     * @throws std::exception For retired/failed/stale ownership; source failure poisons this same cache.
     * @note No gather/order/policy work occurs. This snapshot does not pin backing after retirement.
     */
    HostImmutableRowOwnerObservation retainedOwnerObservation() const;
    /** @return Coherent host-only observations; no source scan, GPU probe or live GPU mirror. */
    CpuImmutableRowCacheStats stats() const;
    /** @return Private SSD owner's coherent lifetime observations, including partial work on a failed row; never a per-client delta. */
    ImmutableRowTableStats sourceStats() const;
    /** @return Original authenticated source descriptor; this does not expose its gather API. */
    const ImmutableRowTableSource &source() const noexcept;
    /** @return Exact original open-file metadata identity. */
    const ImmutableRowFileIdentity &identity() const noexcept;
private:
    friend class HostImmutableRowProvider;
    friend class qwen4exp::ImmutableLookupHostProcessor;
    /** @brief Existing processor-owned acquired request; only friends bind actual channel publications. */
    class ReadGrant final {
    public:
        ReadGrant(ReadGrant &&) noexcept = default;
        ReadGrant &operator=(ReadGrant &&) noexcept = default;
        ReadGrant(const ReadGrant &)=delete;
        ReadGrant &operator=(const ReadGrant &)=delete;
        ~ReadGrant()=default;
    private:
        friend class CpuAdaptiveImmutableRowCache;
        friend class HostImmutableRowProvider;
        friend class qwen4exp::ImmutableLookupHostProcessor;
        /** @brief Retain the exact owner and original acquired live frame, not a copied GPU locator. */
        ReadGrant(std::shared_ptr<Impl> owner,const qwen4exp::ImmutableLookupHostProcessor *channel,
            std::uint64_t generation,std::uint64_t sequence,std::span<const std::int64_t> ids,
            std::span<const ImmutableRowSubordinateIntent> intents,
            std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements)
            :owner_(std::move(owner)),channel_(channel),generation_(generation),sequence_(sequence),
             row_count_(ids.size()),ids_(ids),intents_(intents),acknowledgements_(acknowledgements) {}
        /** @brief Retain only CPU-owned completed rows after response release; mapped input is mutable after consumption. */
        void complete() noexcept {ids_={};intents_={};acknowledgements_={};}
        std::shared_ptr<Impl> owner_;
        const qwen4exp::ImmutableLookupHostProcessor *channel_;
        std::uint64_t generation_,sequence_;
        std::size_t row_count_;
        std::span<const std::int64_t> ids_;
        std::span<const ImmutableRowSubordinateIntent> intents_;
        std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements_;
    };
    /** @return Sole declared requester, absent for independently readable CPU domains. */
    std::optional<PhysicalMemoryAllocatorIdentity> subordinateRequester() const noexcept;
    /** @brief Bind one actual retained processor once; no independent channel may bypass invalidation. */
    void bindSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel);
    /** @brief Retire that binding without waiting; old pins retain physical backing. */
    void retireSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel) noexcept;
    /** @brief Mint a retained read capability only for the actual acquired bound channel frame. */
    ReadGrant authorizeSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel,
        std::uint64_t generation,std::uint64_t sequence,std::span<const std::int64_t> ids,
        std::span<const ImmutableRowSubordinateIntent> intents,
        std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements);
    /** @brief Shared exact gather; null grant is legal only for independent CPU ownership. */
    CpuImmutableRowGatherObservation gatherImpl(std::span<const std::int64_t> ids,
        std::span<std::byte> destination,const ReadGrant *grant);
    /** @brief Borrow only an actual current authorized response row; no public raw-row permit exists. */
    std::optional<Pin> pinSubordinate(const ReadGrant &grant,std::size_t response_row);
    /** @brief Preserve existing pin authentication while enforcing the access discipline. */
    std::optional<Pin> pinImpl(std::int64_t row,const ReadGrant *grant);
    /** @brief Publish only a completely constructed shared physical/source owner. */
    explicit CpuAdaptiveImmutableRowCache(std::shared_ptr<Impl> owner):impl_(std::move(owner)) {}
    std::shared_ptr<Impl> impl_;
};
} // namespace llaminar2
