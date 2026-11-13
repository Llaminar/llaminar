/**
 * @file AdaptiveImmutableRowCacheState.h
 * @brief Fixed native metadata for one GPU-coordinated adaptive row cache.
 *
 * Records describe logical occupancy in already-PMA-owned allocations. They
 * never subtract physical capacity. One mandatory frame owns operation phase;
 * optional cache state preserves warmed rows and policy across request reset.
 * Row32/row64 are checked source geometries, not truncated input types.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_ADAPTIVE_HD __host__ __device__
#else
#define QFN_ADAPTIVE_HD
#endif
namespace llaminar2::qwen4exp::adaptive_lookup
{
inline constexpr std::uint32_t None = 0xffffffffU, LeafTag = 0x80000000U, IndexMask = 0x7fffffffU;
inline constexpr std::uint32_t MaximumSlots = 0x7fffffffU;
/** @brief Complete source row key width chosen before physical materialization. */
enum class RowKeyWidth : std::uint32_t
{
    UInt32,
    UInt64
};
/** @brief Native payload readability; Reserved bytes are never a lookup hit. */
enum class SlotState : std::uint32_t
{
    Empty,
    Reserved,
    Ready
};
/** @brief Three full logical-recency streams; demotions never merge by scanning. */
enum class Stream : std::uint32_t
{
    NewProbation,
    DemotedProbation,
    Protected,
    None
};
/** @brief Sole active-frame lifecycle, present even when optional cache capacity is0. */
enum class Phase : std::uint32_t
{
    Idle,
    Building,
    Prepared,
    Published,
    ResponseCopied,
    Validated
};
/** @brief Bounded common-operation failure; native adapters preserve canonical first error. */
enum class Status : std::uint32_t
{
    Success,
    InvalidGeometry,
    InvalidPhase,
    Invariant,
    Exhausted,
    SourceFailure,
    Cancelled
};
inline constexpr std::uint32_t StateMask = 3U, StreamMask = 12U, FramePin = 16U, KnownSlotFlags = 31U;
inline constexpr std::uint32_t OwnerInitialized = 1U, OwnerFailed = 2U;
/** @brief32B original source-key geometry with full32-bit queue indices and64-bit lifetime counters. */
struct Slot32
{
    std::uint64_t generation, touched;
    std::uint32_t row, previous, next, flags;
};
/** @brief40B source64 geometry; input/source identities are never silently narrowed. */
struct Slot64
{
    std::uint64_t generation, touched, row;
    std::uint32_t previous, next, flags;
};
static_assert(sizeof(Slot32) == 32 && sizeof(Slot64) == 40);
static_assert(offsetof(Slot32, row) == 16 && offsetof(Slot32, flags) == 28 && offsetof(Slot64, flags) == 32);
/** @brief12B parentless compressed-radix node; free nodes use child0 as their pool link. */
struct RadixNode
{
    std::uint32_t child[2], split_bit;
};
static_assert(sizeof(RadixNode) == 12);
/** @brief64B optional model-lifetime policy state; contains NO active-frame phase. */
struct alignas(64) OwnerControl
{
    std::uint64_t logical_clock, next_slot_generation;
    std::uint32_t directory_root, node_free, slot_free, filled;
    std::uint32_t head[3], tail[3];
    std::uint32_t protected_count, flags;
};
/** @brief64B mandatory work state, including the sole active phase and eligible cursors. */
struct alignas(64) FrameControl
{
    std::uint64_t original_ids, unique_rows, source_rows;
    Phase phase;
    Status status;
    std::uint32_t eligible_cursor[3], held_pins;
    std::uint32_t observed_ids, policy_started;
    std::uint32_t promotions,
        evictions; ///< Actual completed owner transitions in this frame, not inferred occupancy.
};
static_assert(sizeof(OwnerControl) == 64 && sizeof(FrameControl) == 64);
/** @brief Stable sort metadata; original source payloads never enter the sort. */
struct KeyIndex
{
    std::uint64_t row;
    std::uint32_t original_coordinate, reserved;
};
/** @brief One unique row's actual hit/reservation and eventual compact source coordinate. */
struct UniqueRecord
{
    std::uint64_t row, expected_slot_generation;
    std::uint32_t first_original_coordinate, slot, response_coordinate, flags;
};
static_assert(sizeof(KeyIndex) == 16 && sizeof(UniqueRecord) == 32);
inline constexpr std::uint32_t InitialHit = 1U, ReservedFill = 2U, Probed = 8U;
/** @brief Explicit immutable frequency/recency policy, independent of model arithmetic. */
struct Policy
{
    std::uint64_t aging_accesses;
    std::uint32_t protected_numerator, protected_denominator;
    bool operator==(const Policy &) const = default;
};
/** @brief Nonowning metadata view; the ordinary prepared roots retain all physical bytes. */
template <class Slot> struct Metadata
{
    Slot *slots;
    RadixNode *nodes;
    OwnerControl *owner;
    FrameControl *frame;
    std::uint32_t *frequency_epochs;
    std::uint16_t *frequency_counts;
    std::uint32_t capacity, protected_limit, lane_width;
    std::uint64_t source_rows, aging_accesses;
};
/** @brief Source-native slot selection; absent slot means mandatory response completion, never a mode switch.
 */
struct Decision
{
    std::uint32_t slot = None;
    std::uint64_t generation = 0;
};
} // namespace llaminar2::qwen4exp::adaptive_lookup
