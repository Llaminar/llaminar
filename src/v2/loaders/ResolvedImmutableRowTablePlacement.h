/**
 * @file ResolvedImmutableRowTablePlacement.h
 * @brief One authoritative native-row placement, including an explicit fixed-source component.
 *
 * Whole, fixed-head and adaptive serving placements are selected by canonical
 * PMA. Each alternative retains its sole source/layout ownership; mixed cache
 * locations remain with actual GPU/CPU owners and are never inferred from a
 * whole-table tier or mirrored in setup metadata.
 * Requested ceilings are retained once for reuse identity, separately from final
 * admitted slots. Exactly one alternative owns the
 * admitted immutable source/topology; no layout mirror or live byte ledger is
 * retained. Adaptive slot locations and cache order remain with the actual owner.
 */
#pragma once
#include "ImmutableRowTableLayoutGeometry.h"
#include "ImmutableRowTableSourceRange.h"
#include "ImmutableRowAllocatorInventory.h"
#include "CpuImmutableRowCachePolicy.h"
#include <span>
#include <variant>
namespace llaminar2 {
class ImmutableRowTablePlacementPlanner;
class Qwen4ExpFixedHeadShards;
/** @brief Exact rank placement, never an unresolved preference or allocation permit. */
class ResolvedImmutableRowTablePlacement final {
public:
    /** @return Whether the sealed alternative is explicit host adaptive RAM/SSD. */
    bool isHostAdaptive() const noexcept;
    /** @return Whether this is explicit single-GPU coordinated VRAM/RAM/SSD caching. */
    bool isGPUAdaptive() const noexcept;
    /** @return Whether this is the checked fixed-head source component; no serving selector exists yet. */
    bool isFixedHeadShards() const noexcept;
    /** @return Whole-tier concrete policy; adaptive and fixed shards reject instead of pretending a full copy. */
    const ImmutableRowTableStoragePolicy &storagePolicy() const;
    /** @return Whole-tier concrete kind; rejects adaptive storage and fixed shards. */
    ImmutableRowStorage storage() const;
    /** @return Immutable host locality, separate from allocator identity; fixed GPU shards reject. */
    const ImmutableRowTableHostPlacement &hostPlacement() const;
    /** @return Exact original unchanged native extent. */
    const ImmutableRowTableExtent &extent() const noexcept;
    /** @return Exact physical owners, including the one host source/cache owner. */
    ImmutableRowAllocatorInventory allocators() const noexcept;
    /** @return Actual PLE consumers; expert-only participants are excluded. */
    std::span<const PhysicalMemoryAllocatorIdentity> consumers() const noexcept;
    /** @return Whole-tier storage for an authenticated consumer; adaptive and fixed shards reject. */
    ImmutableRowStorage storageForConsumer(PhysicalMemoryAllocatorIdentity identity) const;
    /** @return Actual resident/host execution contract for an authenticated consumer. */
    ImmutableRowLookupExecution lookupForConsumer(PhysicalMemoryAllocatorIdentity identity) const;
    /** @return Exact owning rank. */
    int worldRank() const noexcept { return allocators().front().world_rank; }
    /** @return Whether the identity names an actual admitted physical owner. */
    bool containsAllocator(PhysicalMemoryAllocatorIdentity identity) const noexcept;
    /** @return One authoritative checked adaptive layout, including final slots; whole-tier rejects. */
    const ImmutableRowTableLayoutGeometry &hostAdaptiveLayout() const;
    /** @return Frozen experimental replacement policy; whole-tier rejects. */
    const CpuImmutableRowCachePolicy &hostAdaptivePolicy() const;
    /** @return Original requested ceilings, distinct from admitted layout slots; whole-tier rejects. */
    const HostAdaptiveRAMSSDRequest &hostAdaptiveRequest() const;
    /** @brief Authenticate the exact adaptive request or compatible existing whole-tier policy before reuse. */
    void requireRequestedPolicy(const ImmutableRowTablePolicy &requested) const;
    /** @return Maximum original ID response rows shared across all serialized host clients. */
    std::size_t hostAdaptiveMaximumLiveRows() const;
    /** @return Sole checked mixed layout; other modes reject, never project to whole mmap. */
    const ImmutableRowTableLayoutGeometry &gpuAdaptiveLayout() const;
    /** @return Exact original GPU/CPU slot and byte ceilings and replacement policies. */
    const AdaptiveVRAMRAMSSDRequest &gpuAdaptiveRequest() const;
    /** @return Exact original native row element count, not inferred from byte capacity. */
    std::uint64_t gpuAdaptiveRowElements() const;
    /** @return One shared original-ID envelope for GPU frame and subordinate source. */
    std::size_t gpuAdaptiveMaximumLiveRows() const;
    /** @return The sole fixed range/owner/consumer inventory; other alternatives reject. */
    const ImmutableRowTableLayoutGeometry &fixedHeadShardLayout() const;
    /** @return Exact original-native source range for one actual owner; foreign or whole-tier access rejects. */
    ImmutableRowTableSourceRange fixedHeadSourceRangeFor(PhysicalMemoryAllocatorIdentity owner) const;
    /**
     * @brief Author exact optional native payload BOM for every fixed source owner.
     * @param resources Exact observed homogeneous cohort; duplicates, missing and foreign members reject.
     * @return Source-owned ImmutableTableStorage lines for the canonical aggregate PMA plan.
     * @throws std::logic_error For another placement arm.
     * @throws std::invalid_argument For inconsistent resource observations.
     * @note This prices no distributed execution/collectives and does not certify a serving mode.
     */
    PhysicalMemoryPlan fixedHeadShardStorageBOM(std::span<const PhysicalMemoryResource> resources) const;
    bool operator==(const ResolvedImmutableRowTablePlacement &) const = default;
private:
    friend class ImmutableRowTablePlacementPlanner;
    friend class Qwen4ExpFixedHeadShards;
    /** @brief Existing whole-tier owner record, without a duplicate retained layout. */
    struct WholeTier {
        ImmutableRowTableStoragePolicy storage;
        ImmutableRowTableExtent extent;
        std::vector<PhysicalMemoryAllocatorIdentity> allocators, consumers;
        ImmutableRowTableHostPlacement locality;
        bool operator==(const WholeTier &) const = default;
    };
    /** @brief One adaptive geometry authority; exact source/topology exists only in layout. */
    struct HostAdaptive {
        ImmutableRowTableLayoutGeometry layout;
        HostAdaptiveRAMSSDRequest request;
        bool operator==(const HostAdaptive &) const = default;
    };
    /** @brief Mixed residency retains only its sole layout and the distinct original request. */
    struct GPUAdaptive {
        ImmutableRowTableLayoutGeometry layout;
        AdaptiveVRAMRAMSSDRequest request;
        std::uint64_t row_elements; ///< Original source metadata absent from generic byte extent.
        bool operator==(const GPUAdaptive &) const = default;
    };
    /** @brief Exactly one existing checked layout owns all fixed range/storage metadata. */
    struct FixedHeadShards {
        ImmutableRowTableLayoutGeometry layout;
        bool operator==(const FixedHeadShards &) const = default;
    };
    /** @brief Declare checked fixed source ownership; PMA admission remains separately required before materialization. */
    explicit ResolvedImmutableRowTablePlacement(ImmutableRowTableLayoutGeometry fixed_layout);
    /** @brief Seal existing whole-tier ownership after complete PMA admission. */
    ResolvedImmutableRowTablePlacement(ImmutableRowTableStoragePolicy storage, ImmutableRowTableExtent extent,
        std::vector<PhysicalMemoryAllocatorIdentity> allocators,
        std::vector<PhysicalMemoryAllocatorIdentity> consumers, ImmutableRowTableHostPlacement locality);
    /** @brief Seal explicit host-adaptive geometry after complete PMA admission. */
    ResolvedImmutableRowTablePlacement(ImmutableRowTableLayoutGeometry layout, HostAdaptiveRAMSSDRequest requested);
    /** @brief Seal mixed geometry after canonical complete final admission. */
    ResolvedImmutableRowTablePlacement(ImmutableRowTableLayoutGeometry layout,AdaptiveVRAMRAMSSDRequest requested,
        std::uint64_t row_elements);
    /** @brief Require exact membership before projecting either alternative. */
    void requireConsumer(PhysicalMemoryAllocatorIdentity identity) const;
    std::variant<WholeTier,HostAdaptive,FixedHeadShards,GPUAdaptive> value_;
};
} // namespace llaminar2
