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
#include "PooledImmutableOwnerPlanOps.h"
#include "kernels/common/DeviceRowRange.h"
#include "transfer/CapturedTransferChannelProtocol.h"
#include <cstddef>
namespace llaminar2::qwen4exp
{
namespace pooled_native
{
namespace p=adaptive_lookup::pooled;
/** @brief Version of immutable launch/schema/arithmetic policy embedded in capture identity. */
inline constexpr std::uint64_t Version=12;
static_assert(Version<=std::uint64_t{0xffffffff},"Nominal native wire version must fit original Invocation.version32");
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
/**
 * @brief Original received invocation and distinct transport-published phase count.
 *
 * Invocation is immutable for one lookup frame. The byte scalar is a mutable
 * per-phase receive publication, never a second policy/count shadow. Storage
 * alignment is not a wire extent: Invocation48 and count8 have separate producers.
 */
struct OwnerIngress
{
    Invocation invocation;
    std::uint64_t record_bytes;
    std::uint64_t payload_bytes; ///< Independent source-delivery receive count, never an outgoing reply count.
};
static_assert(sizeof(OwnerIngress)==64 && offsetof(OwnerIngress,record_bytes)==48 && offsetof(OwnerIngress,payload_bytes)==56);
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
/** @brief Actual owner source-seal outcome; transport ordering alone cannot certify a successful seal.
 * This record projects original owner/allocation state after the attempted operation.
 * It is not a source-consumption or worker-retirement capability.
 */
struct OwnerSourceSealReceipt {
    p::Interval interval;
    p::SourceWindow source;
    std::uint32_t owner;
    adaptive_lookup::Status status;
    std::uint64_t version;
};
static_assert(sizeof(p::Interval)==24 && sizeof(p::SourceWindow)==24);
static_assert(sizeof(adaptive_lookup::Status)==4 && sizeof(OwnerSourceSealReceipt)==64);
static_assert(offsetof(OwnerSourceSealReceipt,owner)==48 && offsetof(OwnerSourceSealReceipt,status)==52 && offsetof(OwnerSourceSealReceipt,version)==56);

/**
 * @brief Original source last-reader gate, published only after output and every issued read edge retire.
 *
 * The captured channel fixes the original addressed owner. Exact64 bytes carry
 * the original interval/window/status/disposition plus a closed version/kind;
 * no extra root, count or duplicated physical pin state is introduced.
 */
struct ReadersRetiredGate {
    p::Interval interval;
    p::SourceWindow source;
    adaptive_lookup::Status status;
    SourceDisposition disposition;
    std::uint32_t version;
    std::uint32_t kind;
};
static_assert(sizeof(ReadersRetiredGate)==64 && offsetof(ReadersRetiredGate,status)==48 &&
    offsetof(ReadersRetiredGate,disposition)==52 && offsetof(ReadersRetiredGate,version)==56 && offsetof(ReadersRetiredGate,kind)==60);
static_assert(Version<=0xffffffffULL);
/** @brief Actual private source response associated with one original immutable owner grant. */
struct Fill
{
    p::Grant original;
    p::FillAcknowledgement acknowledgement;
};
static_assert(sizeof(Fill)==96);
/** @brief Dynamic offset and exact live byte count within one persistent wire bank. */
using DirectedSlice=CapturedTransferSourceSlice;
static_assert(sizeof(DirectedSlice)==16);
/** @brief Native payload delivery lifecycle; Ready cannot precede the actual complete copy producer. */
enum class Delivery : std::uint32_t { None, Validated, Copied };
/** @brief Actual nominal metadata publication, authored only with the typed header bytes. */
enum class OwnerHeaderKind:std::uint32_t {None,Pin,Availability,FreeReceipt,VictimReceipt,SourceSeal,Publish,Ready,Retired,PublicationAcquired,HitRead,ReadersRetired};
/** @brief Actual completed publication-frontier acquisition, distinct from fill validation and physical Ready. */
struct PublicationAcquiredHeader {p::Interval interval;p::SourceWindow source;std::uint32_t owner;adaptive_lookup::Status status;};
static_assert(sizeof(PublicationAcquiredHeader)==56);
/** @brief Header plus the original nominal witness; alignment bytes are never communicated. */
template<class Header>struct NominalHeaderWire {Header record;OwnerHeaderKind kind;};
inline constexpr std::size_t NominalHeaderWireBytes=60;
static_assert(offsetof(NominalHeaderWire<PublicationAcquiredHeader>,kind)==56);
static_assert(sizeof(NominalHeaderWire<PublicationAcquiredHeader>)==64);
static_assert(offsetof(NominalHeaderWire<PublicationAcquiredHeader>,kind)+sizeof(OwnerHeaderKind)==NominalHeaderWireBytes);
/** @brief Protocol-output validity only; this contains no physical occupancy/pin/free/generation shadow. */
struct alignas(64) OwnerProgress
{
    std::uint64_t setup_identity=0, reply_bytes=0, release_bytes=0, header_bytes=0;
    std::uint32_t initialized=0, released=0;
    Delivery delivery=Delivery::None;
    p::SourceRetirement retirement=p::SourceRetirement::Unpublished; ///< Actual acquired source completion, never inferred from request health.
    OwnerHeaderKind header_kind=OwnerHeaderKind::None; ///< Actual typed producer witness in original metadata padding.
};
static_assert(sizeof(OwnerProgress)==64 && offsetof(OwnerProgress,header_bytes)==24 && offsetof(OwnerProgress,header_kind)==48);
/** @brief Native-only mandatory owner subranges, separate from the already priced canonical frame. */
struct OwnerOffsets
{
    std::size_t progress=0, header=0, replies=0, ingress=0, source_outcome=0, source_fills=0, source_payload=0, bytes=0;
    std::uint32_t reply_capacity=0;
    bool operator==(const OwnerOffsets&) const=default;
};
/** @brief Native-only mandatory coordinator subranges; sorted releases preserve the acquired original bank. */
struct CoordinatorOffsets
{
    std::size_t invocation=0, pin_commands=0, pin_slices=0, pin_cursors=0, radix_offsets=0, availability=0, free_headers=0, free_slices=0;
    std::size_t victim_headers=0, victim_slices=0, sorted_releases=0, ingress_count=0, ingress_frontier=0, bytes=0;
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
    MakeReady, ReadyReceipt, Release, CancelFree, CancelVictim, CancelSource, Retire, DrainSourceFailure, PublishWire=24, ReadyWire=25, PublicationAcquiredWire=26, RetiredWire=27, AcquireReaderRetirement=28, ReadRows=32
};
/** @brief One actual coordinator phase with device-owned live extents and no host branch selection. */
enum class CoordinatorAction : std::uint32_t
{
    Begin, PinCommands, AcquirePins, SealPins, PlanFree, AcquireFree, CompleteFree,
    PlanVictims, AcquireVictims, CompleteVictims, SealSource, Publish, Cancel, Finish,
    ImportLocalPinHeader, ImportLocalAvailability, ImportLocalFreeReceipt, ImportLocalVictimReceipt
};
/** @return Whether this exact coordinator operation imports one original local metadata publication. */
QFN_ADAPTIVE_HD constexpr bool localFrontierImport(CoordinatorAction action)
{
    return action==CoordinatorAction::ImportLocalPinHeader || action==CoordinatorAction::ImportLocalAvailability ||
        action==CoordinatorAction::ImportLocalFreeReceipt || action==CoordinatorAction::ImportLocalVictimReceipt;
}
/** @return Exact typed metadata bytes, never slot capacity or allocation alignment. */
QFN_ADAPTIVE_HD constexpr std::size_t localFrontierBytes(CoordinatorAction action)
{
    return action==CoordinatorAction::ImportLocalPinHeader?sizeof(p::PinHeader):
        action==CoordinatorAction::ImportLocalAvailability?sizeof(Availability):
        action==CoordinatorAction::ImportLocalFreeReceipt?sizeof(p::FreeAllocationReceipt):
        action==CoordinatorAction::ImportLocalVictimReceipt?sizeof(p::VictimAllocationReceipt):0;
}
/** @brief Explicit local compact/directed plans or an exact counted transport-acquired allocation prefix.
 * TransportedAllocationPrefix binds the original UInt64 incoming count and requires
 * count/header equality before plan acceptance; local modes preserve their existing producers. */
enum class RecordLayout:std::uint32_t {Compact,DirectedAllocationBank,TransportedAllocationPrefix,DirectedCommandBank};
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
    std::uint32_t local_owner=adaptive_lookup::None; ///< Authenticated original coordinate; local imports/grant acquisition only.
    std::uint32_t local_reply_capacity=0; ///< Exact original addressed reply-bank bound, never a live count.
    const p::OwnerFrame *local_control=nullptr; ///< Actual original metadata producer's interval/status/phase.
    const p::AllocationFrame *local_allocation=nullptr; ///< Original wave identity; never a copied receipt ledger.
    const OwnerHeaderKind *local_header_kind=nullptr; ///< Original actual nominal producer witness.
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
