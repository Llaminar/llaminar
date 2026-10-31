/**
 * @file PooledImmutableCoordinatorState.h
 * @brief Sole pooled policy authority and bounded immutable two-wave protocol records.
 *
 * Static owner coordinates describe declared allocation geometry. Per-frame
 * leases, plans, grants and completion records are protocol evidence, never
 * an occupancy/pin mirror or an alternate physical-byte ledger. Actual owners
 * alone mutate slots, free chains, pin bits and full slot generations.
 */
#pragma once
#include "PooledImmutableOwnerPlanOps.h"
namespace llaminar2::qwen4exp { class PooledImmutableCoordinatorGeometry; }
namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/** @brief Immutable exact coordinate geometry, retained separately from native execution state. */
struct CoordinatorOwnerGeometry
{
    std::uint32_t first_entry, slots;
};
/** @brief Closed checked metadata projection; arbitrary explicit owner cohorts remain representable. */
class CoordinatorParameters final
{
public:
    /** @return Complete original source row extent. */
    QFN_ADAPTIVE_HD std::uint64_t sourceRows() const { return rows_; }
    /** @return Total static policy-entry geometry, never current occupancy. */
    QFN_ADAPTIVE_HD std::uint32_t entries() const { return entries_; }
    /** @return Maximum admitted distinct rows. */
    QFN_ADAPTIVE_HD std::uint32_t maximumUnique() const { return maximum_unique_; }
    /** @return Maximum admitted original IDs, including interleaved duplicates. */
    QFN_ADAPTIVE_HD std::uint32_t maximumOriginal() const { return maximum_original_; }
    /** @return Complete explicit owner cohort, including C0 and independent of consumers. */
    QFN_ADAPTIVE_HD std::uint32_t participants() const { return participants_; }
    /** @return Checked four-lane frequency width. */
    QFN_ADAPTIVE_HD std::uint32_t laneWidth() const { return lane_width_; }
    /** @return Checked protected policy-queue capacity. */
    QFN_ADAPTIVE_HD std::uint32_t protectedLimit() const { return protected_; }
    /** @return Policy aging interval in actual original observations. */
    QFN_ADAPTIVE_HD std::uint64_t agingAccesses() const { return aging_; }
private:
    friend class llaminar2::qwen4exp::PooledImmutableCoordinatorGeometry;
    /** @brief Only the checked geometry factory constructs complete native policy geometry. */
    CoordinatorParameters() = default;
    std::uint64_t rows_ = 0, aging_ = 0;
    std::uint32_t entries_ = 0, maximum_unique_ = 0, maximum_original_ = 0, participants_ = 0;
    std::uint32_t lane_width_ = 0, protected_ = 0;
};
/** @brief Only directory/recency metadata; the frame marker records an actual acquired owner grant. */
struct DirectoryEntry
{
    Locator locator;
    std::uint64_t touched, observed_frame;
    std::uint32_t previous, next;
    Stream stream;
    std::uint32_t present;
};
static_assert(sizeof(DirectoryEntry) == 56);
/** @brief Exactly one frequency/directory/queue policy authority; no free/occupied/pin state. */
struct alignas(64) CoordinatorPolicyControl
{
    std::uint64_t logical_clock;
    std::uint32_t directory_root, node_free, head[3], tail[3], protected_count, flags;
    std::uint64_t aging_accesses; ///< Immutable arithmetic policy identity established at setup.
    std::uint32_t lane_width, protected_limit;
};
static_assert(sizeof(CoordinatorPolicyControl) == 64);
/** @brief Typed frame phases require all acquired owner completions before source publication. */
enum class LookupPhase : std::uint32_t
{
    Uninitialized, Idle, Coalescing, Pinning, Availability, FreePlanned, FreeCompleted,
    VictimPlanned, VictimCompleted, Source, Ready, Draining
};
/** @brief One original unique coordinate and its actual acquired lifetime or planned directed operation. */
enum class UniquePhase : std::uint32_t
{
    Missing, NeedsPin, PinPending, Read, FreePlanned, FreeGranted,
    VictimPlanned, VictimGranted, SourceOnly
};
/** @brief One bounded invocation, retaining original lookup ordinal across request reset. */
struct alignas(64) LookupFrame
{
    Interval interval{};
    SourceWindow source{};
    std::uint64_t ordinal = 0, command_ordinal = 0, source_storage_generation = 0, initial_clock = 0;
    LookupPhase phase = LookupPhase::Uninitialized;
    Status status = Status::Success;
    std::uint32_t original_ids = 0, unique_rows = 0, outstanding = 0;
    std::uint32_t free_planned = 0, victim_planned = 0, eligible_cursor[3]{};
    bool observed = false, queues_applied = false, free_exhausted = false, cancelled_completion = false;
};
/** @brief Append-only immutable command/grant observations; neither grants nor unique coordinates are reordered. */
struct CoordinatorUnique
{
    Grant grant{};
    OwnerCommand pin{};
    AllocationDecision decision{};
    std::uint64_t row = 0;
    std::uint32_t entry = None, first_position = 0, decision_index = None, allocation_owner = None;
    UniquePhase phase = UniquePhase::Missing;
    bool queue_started = false;
};
/** @brief Actual original row-position sort item; used only at live original/unique extents. */
struct CoordinatorOccurrence
{
    std::uint64_t row;
    std::uint32_t position, unique;
};
static_assert(sizeof(CoordinatorOccurrence) == 16);
/**
 * @brief Internal stable observation runs in the dead original coalescing banks.
 *
 * Runs never reorder original observations. Each 256-position partition stores
 * compact {actual row, original first position, positive repeat count} records
 * in sort_a, and one authored run count in sort_b[partition].unique. Native
 * parallel producers alone supply this view to the private policy entrypoint.
 */
struct OriginalObservationRuns
{
    const CoordinatorOccurrence* records;
    const CoordinatorOccurrence* partitions;
    static constexpr std::uint32_t PartitionRows=256;
};
/** @brief Transient immutable lease and directed plan extents; counts describe messages, never physical occupancy. */
struct CoordinatorOwnerPlan
{
    AvailabilityLease lease{};
    std::uint32_t free_offset = 0, free_count = 0, free_received = 0;
    std::uint32_t victim_offset = 0, victim_count = 0, victim_received = 0;
    std::uint32_t plan_write = 0; ///< Private plan-builder cursor; actual receipt counters are never scratch.
};
/** @brief Exact borrowed roots and capacities; the production binder retains their native allocation/events. */
struct CoordinatorView
{
    CoordinatorPolicyControl *policy;
    DirectoryEntry *entries;
    RadixNode *nodes;
    std::uint32_t *frequency_epochs;
    std::uint16_t *frequency_counts;
    LookupFrame *frame;
    CoordinatorUnique *unique;
    CoordinatorOccurrence *sort_a, *sort_b;
    std::uint32_t *radix_counts, *original_to_unique;
    CoordinatorOwnerPlan *owner_plans;
    AllocationDecision *decisions;
    const CoordinatorOwnerGeometry *owners;
    CoordinatorParameters parameters;
    std::uint32_t unique_capacity, original_capacity, owner_capacity;
    std::uint64_t table, source_storage_generation;
};
}
