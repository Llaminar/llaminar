/**
 * @file PooledImmutableNativeTypes.h
 * @brief Exact native operands and live wire extents for participant-local pooled phases.
 *
 * All records are decoded inputs/outputs, never source or transport capabilities.
 * Owners consume only compact immutable invocation/source frontiers and their
 * directed payload. Every persistent byte belongs to existing declared roots;
 * actual slots, pins, free chains and generations remain OwnerOperations-owned.
 */
#pragma once
#include "PooledImmutableCoordinatorGeometry.h"
#include "kernels/common/DeviceRowRange.h"
namespace llaminar2::qwen4exp
{
namespace pooled_native
{
namespace p=adaptive_lookup::pooled;
/** @brief Version of immutable launch/schema/arithmetic policy embedded in capture identity. */
inline constexpr std::uint64_t Version=4;
/**
 * @brief Smallest even four-bit radix pass count for one checked immutable exclusive key bound.
 * @param exclusive_limit Positive declared source/position bound, never a mutable device count.
 * @return Two through sixteen passes, preserving each producer's fixed input/output bank parity.
 * @note Zero is conservatively covered; checked launch geometry separately rejects a zero physical bound.
 */
QFN_ADAPTIVE_HD constexpr unsigned radixPasses(std::uint64_t exclusive_limit)
{
    auto maximum=exclusive_limit?exclusive_limit-1:0;unsigned digits=1;
    while(maximum>>=4)++digits;
    return (digits+1u)&~1u;
}

/** @brief Compact immutable original invocation projection consumed by all physical owners. */
struct Invocation
{
    p::Interval interval;
    std::uint64_t source_storage_generation;
    std::uint32_t original_ids, unique_rows;
    adaptive_lookup::Status status;
    std::uint32_t version;
};
static_assert(sizeof(Invocation)==48);
/** @brief Actual owner availability, including failure; only the coordinator consumes this record. */
struct Availability
{
    p::AvailabilityLease lease;
    adaptive_lookup::Status status;
    std::uint32_t version;
};
static_assert(sizeof(Availability)==40);
/** @brief Immutable first-wave directed plan metadata; allocation payload is a separate exact live prefix. */
struct FreePlanHeader
{
    p::AvailabilityLease lease;
    std::uint64_t decision_bytes;
    adaptive_lookup::Status status;
    std::uint32_t version;
};
/** @brief Nominal second-wave metadata prevents treating first-wave completion as victim acceptance. */
struct VictimPlanHeader
{
    p::AvailabilityLease lease;
    std::uint64_t decision_bytes;
    adaptive_lookup::Status status;
    std::uint32_t version;
};
static_assert(sizeof(FreePlanHeader)==48 && sizeof(VictimPlanHeader)==48);
/** @brief Distinguish actual source publication from its later consumed producer. */
enum class SourceDisposition : std::uint32_t { Unpublished, Published, Consumed, Aborted };
/** @brief Fixed captured source branch; no host decision selects an eager transaction. */
enum class SourceCondition : std::uint32_t {Always,AfterSuccess,AfterFailure};
/** @brief Exact source branch validation result; invalid never silently skips work. */
enum class SourceGate : std::uint32_t {Run,Skip,Invalid};
/** @brief Required all-owner original source frontier; C0 receives this metadata and no row/ACK payload. */
struct SourceFrontier
{
    p::Interval interval;
    p::SourceWindow source;
    SourceDisposition disposition;
    adaptive_lookup::Status status;
    std::uint64_t version;
};
static_assert(sizeof(SourceFrontier)==64);
/** @brief Actual private source response associated with one original immutable owner grant. */
struct Fill
{
    p::Grant original;
    p::FillAcknowledgement acknowledgement;
};
static_assert(sizeof(Fill)==96);
/** @brief Dynamic offset and exact live byte count within one persistent wire bank. */
struct DirectedSlice { std::uint64_t byte_offset, live_bytes; };
static_assert(sizeof(DirectedSlice)==16);
/** @brief Native payload delivery lifecycle; Ready cannot precede the actual complete copy producer. */
enum class Delivery : std::uint32_t { None, Validated, Copied };
/** @brief Protocol-output validity only; this contains no physical occupancy/pin/free/generation shadow. */
struct alignas(64) OwnerProgress
{
    std::uint64_t setup_identity=0, reply_bytes=0, release_bytes=0, header_bytes=0;
    std::uint32_t initialized=0, released=0;
    Delivery delivery=Delivery::None;
    p::SourceRetirement retirement=p::SourceRetirement::Unpublished; ///< Actual acquired source completion, never inferred from request health.
};
static_assert(sizeof(OwnerProgress)==64);
/** @brief Native-only mandatory owner subranges, separate from the already priced canonical frame. */
struct OwnerOffsets
{
    std::size_t progress=0, header=0, replies=0, bytes=0;
    std::uint32_t reply_capacity=0;
    bool operator==(const OwnerOffsets&) const=default;
};
/** @brief Native-only mandatory coordinator subranges; sorted releases preserve the acquired original bank. */
struct CoordinatorOffsets
{
    std::size_t invocation=0, pin_commands=0, pin_slices=0, pin_cursors=0, radix_offsets=0, availability=0, free_headers=0, free_slices=0;
    std::size_t victim_headers=0, victim_slices=0, sorted_releases=0, bytes=0;
    std::uint32_t radix_blocks=0, reply_capacity=0;
    bool operator==(const CoordinatorOffsets&) const=default;
};
/** @brief Checked immutable physical owner projection; no mutable native scalar is read by host code. */
struct OwnerParameters
{
    p::OwnerParameters owner;
    PooledPhysicalOwnerOffsets cache;
    PooledPhysicalFrameOffsets frame;
    OwnerOffsets native;
    DeviceId device;
    std::uint32_t maximum_original;
    std::size_t row_bytes, slot_stride;
    std::uint64_t table, source_storage_generation, identity;
};
/** @brief Checked immutable coordinator projection independent of storage/query owner membership. */
struct CoordinatorParameters
{
    p::CoordinatorParameters policy;
    PooledCoordinatorOffsets cache;
    PooledCoordinatorFrameOffsets frame;
    CoordinatorOffsets native;
    DeviceId device;
    std::uint64_t table, source_storage_generation, identity;
};
/** @brief Every action is one participant-local compute operation; transport is always an explicit graph node. */
enum class OwnerAction : std::uint32_t
{
    Begin, Pin, SealPins, AcquirePins, Availability,
    AcceptFree, ExecuteFree, FreeReceipt, AcquireFree,
    AcceptVictim, ExecuteVictim, VictimReceipt,
    SealSource, ApplySourceDelivery, ValidateFills, AcquirePublication,
    MakeReady, ReadyReceipt, Release, CancelFree, CancelVictim, CancelSource, Retire
};
/** @brief One actual coordinator phase with device-owned live extents and no host branch selection. */
enum class CoordinatorAction : std::uint32_t
{
    Begin, PinCommands, AcquirePins, SealPins, PlanFree, AcquireFree, CompleteFree,
    PlanVictims, AcquireVictims, CompleteVictims, SealSource, Publish, Cancel, Finish
};
/** @brief Compact acquired records or a bounded read-only view of one actual directed allocation bank. */
enum class RecordLayout:std::uint32_t {Compact,DirectedAllocationBank};
/** @brief Immutable original/acquired owner operands; null is legal only for a schema-declared absent bank. */
struct OwnerInputs
{
    const Invocation *invocation;
    const void *records;
    const std::uint64_t *record_bytes;
    const void *frontier;
    const std::byte *payload;
    const std::uint64_t *payload_bytes;
    SourceCondition condition=SourceCondition::Always;
    const SourceFrontier *outcome=nullptr;
    RecordLayout record_layout=RecordLayout::Compact;
    const DirectedSlice *allocation_slice=nullptr;
    std::uint64_t record_origin=0,record_capacity=0;
};
/** @brief Device-only checked launch arguments; public enqueue supplies these after root/operand sealing. */
struct OwnerArgs
{
    OwnerParameters plan;
    std::byte *cache, *frame, *native;
    OwnerInputs input;
};
/** @brief Coordinator inputs retain the original counted prefix, actual admission producer and acquired peers. */
struct CoordinatorInputs
{
    const void *records;
    const std::uint64_t *record_bytes;
    const void *frontier;
    const SourceFrontier *source;
    SourceCondition condition=SourceCondition::Always;
    const SourceFrontier *outcome=nullptr;
};
/** @brief Setup operands contain no invented query/count/admission descriptor. */
struct CoordinatorSetupArgs
{
    CoordinatorParameters plan;
    std::byte *cache, *frame, *native;
};
/** @brief Device-only coordinator view over its sole original roots and immutable acquired operands. */
struct CoordinatorArgs
{
    CoordinatorParameters plan;
    std::byte *cache, *frame, *native;
    CoordinatorInputs input;
};
/** @brief Actual original producer width, independent of physical source/slot row width. */
enum class OriginalIdWidth:std::uint32_t {UInt32,UInt64};
/** @brief Complete immutable query declaration; metadata-only phases have no dummy query or width. */
struct OriginalRows {DeviceRowRange rows;OriginalIdWidth id_width;};
/** @brief Original counted query belongs only to Begin; its typed producer width enters capture identity. */
struct CoordinatorQuery
{
    OriginalRows original;
    const void* ids;
    const std::uint64_t* admission;
};
/** @brief Complete original Begin operands; private native launch never fabricates a placeholder query. */
struct CoordinatorBeginArgs:CoordinatorArgs
{
    CoordinatorQuery query;
};
}
}
