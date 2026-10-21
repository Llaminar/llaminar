/**
 * @file ImmutableRowTableLayoutGeometry.h
 * @brief Checked native layout metadata with independent storage owners and consumers.
 *
 * Geometry describes one lookup domain, without reading rows, observing free
 * memory, choosing a capacity, allocating, or issuing a materialization permit.
 * Canonical PMA must price mandatory execution before rows/quotas freeze, then
 * price every declared slot and its metadata from the remaining allowed budget.
 * Original file/tensor identity stays with the existing immutable source owner.
 * Adaptive pools have no source-row ranges; fixed ranges never describe mutable
 * cache locations. Live generations, grants, pins and replacement are absent.
 */
#pragma once
#include "ImmutableRowTableExtent.h"
#include "loaders/ImmutableRowTableHostPlacement.h"
#include "loaders/ImmutableRowTablePolicy.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <utility>
namespace llaminar2 {
/** @brief Allowed runtime-tier permissions, independent of positive capacity and initial GGUF preload. */
class ImmutableRowTableTiers final {
public:
    /**
     * @brief Validate any of the seven nonempty concrete-tier masks.
     * @param tiers Allowed concrete VRAM/RAM/SSD choices, with no Automatic or duplicates; native tiers may receive zero slots.
     * @return Canonical immutable set.
     * @throws std::invalid_argument For empty, unknown or duplicate tiers.
     */
    static ImmutableRowTableTiers of(std::span<const ImmutableRowStorage> tiers);
    /** @return Whether a concrete runtime tier is permitted; this promises no payload capacity. Unknown values return false. */
    bool contains(ImmutableRowStorage tier) const noexcept;
    /** @return Canonical mask for complete declaration/capture identity. */
    std::uint8_t bits() const noexcept { return bits_; }
    bool operator==(const ImmutableRowTableTiers &) const = default;
private:
    /** @brief Only the validated set factory constructs masks. @param bits Canonical nonempty concrete-tier bits. */
    explicit ImmutableRowTableTiers(std::uint8_t bits):bits_(bits) {}
    std::uint8_t bits_;
};
/** @brief Fixed source ranges versus mutable pooled cache placement. */
enum class ImmutableRowTableResidency : std::uint8_t { Fixed, Adaptive };
/** @brief One physical storage endpoint and its concrete tier, independent of consumers. */
struct ImmutableRowTableOwnerKey final {
    PhysicalMemoryAllocatorIdentity allocator;
    ImmutableRowStorage tier;
    bool operator==(const ImmutableRowTableOwnerKey &) const = default;
};
/** @brief Exact native slot geometry, including declared empty endpoints; ceilings are requests, never a free-byte ledger. */
struct ImmutableRowTableLayoutOwner final {
    ImmutableRowTableOwnerKey key;
    std::uint64_t fixed_rows = 0; ///< Exact slots occupied by immutable fixed range copies.
    std::uint64_t cache_rows = 0; ///< Pooled slots eligible for ANY original source row.
    std::optional<std::size_t> physical_byte_limit; ///< Total payload+metadata ceiling; PMA checks final BOM.
    ImmutableRowTableHostPlacement host_placement = ImmutableRowTableHostPlacement::taskPolicy(); ///< RAM locality, never inferred from CPU allocator ordinal.
    bool operator==(const ImmutableRowTableLayoutOwner &) const = default;
};
/** @brief One explicit immutable copy; replica records account for every additional native row. */
enum class ImmutableRowTableCopyRole : std::uint8_t { Primary, Replica };
/** @brief Half-open source interval for fixed residency only, never an adaptive home quota. */
struct ImmutableRowTableFixedRange final {
    std::uint64_t first_row, rows;
    ImmutableRowTableOwnerKey owner;
    ImmutableRowTableCopyRole role;
    bool operator==(const ImmutableRowTableFixedRange &) const = default;
};
/** @brief CPU endpoint permitted to serve the original source at runtime, with concrete window policy. */
struct ImmutableRowTableSourceService final {
    PhysicalMemoryAllocatorIdentity allocator;
    ImmutableRowTableStoragePolicy windows;
    bool operator==(const ImmutableRowTableSourceService &) const = default;
};
/** @brief Explicit adaptive replica eligibility; payload locations and row IDs remain device-owned. */
struct ImmutableRowTableReplicaGroup final {
    std::vector<ImmutableRowTableOwnerKey> owners;
    std::uint32_t maximum_copies_per_row;
    bool operator==(const ImmutableRowTableReplicaGroup &) const = default;
};
/** @brief Source/owner geometry consumed by planner and graph construction, not a resolved PMA seal. */
class ImmutableRowTableLayoutGeometry final {
public:
    /**
     * @brief Check one source-authored fixed or pooled-adaptive lookup domain.
     * @param extent Original unchanged native source geometry.
     * @param tiers Allowed runtime tiers; native bits remain present when their payload capacity is zero. SSD absence prohibits runtime source reads.
     * @param residency Fixed intervals or mutable pooled slots.
     * @param owners Unique native storage endpoints, including zero-capacity members, independent of consumers.
     * @param consumers Exact unique physical consumers; storage-only owners are allowed.
     * @param fixed_ranges Source intervals for Fixed only; each extra native copy is explicitly Replica. An SSD service supplies implicit primary coverage for the original source.
     * @param source_services Exact CPU SSD services, required iff SSD is enabled.
     * @param replicas Explicit adaptive replica groups; absence means unique primary cache residency.
     * @param gpu_coordinator Exact GPU directory owner, required iff adaptive GPU slots exist.
     * @param maximum_inflight_rows Source-authored request-row bound, not a live count. Host adaptive placements retain it for execution scratch; only GPU pools add it to directory capacity.
     * @return Checked immutable metadata; PMA remains sole admission/materialization authority.
     * @throws std::invalid_argument For invalid identities, coverage, modes, capacities or policies.
     * @throws std::overflow_error For unrepresentable source, slot or record arithmetic.
     * @note Adaptive no-SSD primary migration is not representable by these fixed ranges and is rejected.
     * @note Allowed native tiers may have no positive payload capacity. Canonical domain/transport
     * membership remains its existing authority and must not be inferred from nonzero owners.
     * Empty endpoints require no native allocation; mandatory empty-packet/service execution
     * remains the separate source-owned pre-quota provision. This value never drops policy bits.
     */
    static ImmutableRowTableLayoutGeometry checked(ImmutableRowTableExtent extent,
        ImmutableRowTableTiers tiers, ImmutableRowTableResidency residency,
        std::vector<ImmutableRowTableLayoutOwner> owners,
        std::vector<PhysicalMemoryAllocatorIdentity> consumers,
        std::vector<ImmutableRowTableFixedRange> fixed_ranges,
        std::vector<ImmutableRowTableSourceService> source_services,
        std::vector<ImmutableRowTableReplicaGroup> replicas = {},
        std::optional<PhysicalMemoryAllocatorIdentity> gpu_coordinator = std::nullopt,
        std::uint64_t maximum_inflight_rows = 0);
    /**
     * @brief Project the EXISTING admitted whole-tier modes through the same invariant implementation.
     * @param storage Existing concrete mode and original mmap geometry.
     * @param extent Existing loader-authenticated native extent.
     * @param owners Existing exact whole-copy owners.
     * @param consumers Existing actual PLE consumers.
     * @param host_placement Existing explicit RAM locality, independent of allocator ordinal.
     * @return Equivalent fixed layout; no new runtime mode, allocation or accounting is introduced.
     * @throws std::invalid_argument For malformed legacy metadata.
     */
    static ImmutableRowTableLayoutGeometry wholeTier(const ImmutableRowTableStoragePolicy &storage,
        ImmutableRowTableExtent extent, std::span<const PhysicalMemoryAllocatorIdentity> owners,
        std::span<const PhysicalMemoryAllocatorIdentity> consumers,
        ImmutableRowTableHostPlacement host_placement = ImmutableRowTableHostPlacement::taskPolicy());
    /** @return Complete immutable declaration equality, without live row locations or pointer identity. */
    bool operator==(const ImmutableRowTableLayoutGeometry &) const = default;
    /** @return Exact original native source extent. */
    const ImmutableRowTableExtent &extent() const noexcept { return extent_; }
    /** @return Allowed runtime tiers, including any that received zero native slots; no Automatic value exists here. */
    ImmutableRowTableTiers tiers() const noexcept { return tiers_; }
    /** @return Fixed or adaptive policy, never inferred from shape or row count. */
    ImmutableRowTableResidency residency() const noexcept { return residency_; }
    /** @return Unique native endpoints and capacity ceilings, including explicitly empty endpoints. */
    std::span<const ImmutableRowTableLayoutOwner> owners() const noexcept { return owners_; }
    /** @return Exact actual consumers; they need not own any payload. */
    std::span<const PhysicalMemoryAllocatorIdentity> consumers() const noexcept { return consumers_; }
    /** @return Only fixed immutable source copies; empty for every adaptive layout. */
    std::span<const ImmutableRowTableFixedRange> fixedRanges() const noexcept { return fixed_ranges_; }
    /** @return All explicitly enabled runtime SSD services. */
    std::span<const ImmutableRowTableSourceService> sourceServices() const noexcept { return source_services_; }
    /** @return Explicit bounded adaptive replication eligibility; no current locations. */
    std::span<const ImmutableRowTableReplicaGroup> replicaGroups() const noexcept { return replicas_; }
    /** @return Exact GPU coordinator when adaptive GPU pools require one. */
    std::optional<PhysicalMemoryAllocatorIdentity> gpuCoordinator() const noexcept { return gpu_coordinator_; }
    /** @return Source-authored maximum in-flight rows, not execution state. */
    std::uint64_t maximumInflightRows() const noexcept { return maximum_inflight_rows_; }
    /** @return GPU slots plus admitted in-flight rows when a GPU coordinator exists; otherwise zero, including bounded host requests. */
    std::uint64_t gpuDirectoryRecordBound() const noexcept { return gpu_directory_records_; }
    /**
     * @brief Compute only original native payload bytes for one declared owner.
     * @param owner Exact tier and physical identity.
     * @return Checked native payload floor; alignment/metadata must be added by their source-owned BOM factory.
     * @throws std::invalid_argument If the owner is not in this domain.
     */
    std::size_t nativePayloadBytes(ImmutableRowTableOwnerKey owner) const;
private:
    /**
     * @brief Factory-only immutable construction after all range/endpoint/overflow checks.
     * @param extent Original source geometry.
     * @param tiers Validated concrete runtime tier set.
     * @param residency Fixed ranges or pooled adaptation.
     * @param owners Unique physical native payload endpoints.
     * @param consumers Unique actual lookup endpoints.
     * @param ranges Checked immutable fixed copies; empty for adaptation.
     * @param services Explicit original-source runtime services.
     * @param replicas Explicit bounded adaptive replica eligibility.
     * @param coordinator Exact GPU directory owner, when required.
     * @param inflight Source-owned maximum pending row records.
     * @param directory Checked GPU slot-plus-in-flight record bound.
     */
    ImmutableRowTableLayoutGeometry(ImmutableRowTableExtent extent, ImmutableRowTableTiers tiers,
        ImmutableRowTableResidency residency, std::vector<ImmutableRowTableLayoutOwner> owners,
        std::vector<PhysicalMemoryAllocatorIdentity> consumers,
        std::vector<ImmutableRowTableFixedRange> ranges,
        std::vector<ImmutableRowTableSourceService> services,
        std::vector<ImmutableRowTableReplicaGroup> replicas,
        std::optional<PhysicalMemoryAllocatorIdentity> coordinator,
        std::uint64_t inflight, std::uint64_t directory)
        : extent_(extent),tiers_(tiers),residency_(residency),owners_(std::move(owners)),
          consumers_(std::move(consumers)),fixed_ranges_(std::move(ranges)),source_services_(std::move(services)),
          replicas_(std::move(replicas)),gpu_coordinator_(coordinator),maximum_inflight_rows_(inflight),
          gpu_directory_records_(directory) {}
    ImmutableRowTableExtent extent_;
    ImmutableRowTableTiers tiers_;
    ImmutableRowTableResidency residency_;
    std::vector<ImmutableRowTableLayoutOwner> owners_;
    std::vector<PhysicalMemoryAllocatorIdentity> consumers_;
    std::vector<ImmutableRowTableFixedRange> fixed_ranges_;
    std::vector<ImmutableRowTableSourceService> source_services_;
    std::vector<ImmutableRowTableReplicaGroup> replicas_;
    std::optional<PhysicalMemoryAllocatorIdentity> gpu_coordinator_;
    std::uint64_t maximum_inflight_rows_,gpu_directory_records_;
};
} // namespace llaminar2
