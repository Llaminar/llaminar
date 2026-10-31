/**
 * @file PooledImmutablePhysicalOwnerState.h
 * @brief Bounded owner-local wire and physical state for pooled adaptive rows.
 *
 * These are decoded protocol values, not source/Rank capabilities. An eventual
 * explicit graph transport must retain/authenticate their original producers,
 * controller, source window and ordered declared cohort. No factory here mints
 * such a serving proof. Only the physical owner mutates slots/free/generations;
 * one future coordinator owns directory, frequency and replacement policy.
 */
#pragma once
#include "AdaptiveImmutablePhysicalSlot.h"

namespace llaminar2::qwen4exp
{
class PooledImmutablePhysicalOwnerGeometry;
}

namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/**
 * @brief Closed immutable native projection of the existing checked declared layout.
 *
 * Only the geometry factory supplies P/coordinate/capacity/key width. Copying
 * this declaration does not mint actual Rank membership, source authentication
 * or a physical-materialization permit. The future stage must retain/bind its
 * full actual layout, PMA owner, original controller and transport producers.
 */
class OwnerParameters final
{
public:
    /** @return Original source extent, not a source-home quota. */
    QFN_ADAPTIVE_HD std::uint64_t sourceRows() const { return source_rows_; }
    /** @return This exact declared owner's optional capacity, including zero. */
    QFN_ADAPTIVE_HD std::uint32_t capacity() const { return capacity_; }
    /** @return Source-declared maximum unique request rows. */
    QFN_ADAPTIVE_HD std::uint32_t maximumUnique() const { return maximum_unique_; }
    /** @return Explicit original cohort coordinate, independent of physical ordinal sorting. */
    QFN_ADAPTIVE_HD std::uint32_t coordinate() const { return coordinate_; }
    /** @return Complete declared cohort size, including all empty participants. */
    QFN_ADAPTIVE_HD std::uint32_t participants() const { return participants_; }
    /** @return Actual checked row32/row64 physical record geometry. */
    QFN_ADAPTIVE_HD RowKeyWidth rowWidth() const { return width_; }

private:
    friend class llaminar2::qwen4exp::PooledImmutablePhysicalOwnerGeometry;
    /** @brief Only the checked declared-layout factory constructs native geometry values. */
    OwnerParameters(std::uint64_t rows, std::uint32_t capacity, std::uint32_t maximum,
                    std::uint32_t coordinate, std::uint32_t participants, RowKeyWidth width)
        : source_rows_(rows), capacity_(capacity), maximum_unique_(maximum), coordinate_(coordinate),
          participants_(participants), width_(width) {}
    std::uint64_t source_rows_;
    std::uint32_t capacity_, maximum_unique_, coordinate_, participants_;
    RowKeyWidth width_;
};
static_assert(std::is_trivially_copyable_v<OwnerParameters>);

/** @brief Actual canonical admission and transport-frame identities, never counters minted here. */
struct Interval
{
    std::uint64_t table, admission, sequence;
};

/** @brief Source transaction whose private response/ACK must retire before owner frame reuse. */
struct SourceWindow
{
    std::uint64_t table, generation, sequence;
};

/** @brief The sole mandatory owner-frame phase, including C0. */
enum class OwnerPhase : std::uint32_t
{
    Uninitialized,
    Idle,
    Pinning,
    PinsSealed,
    Reserving,
    AvailabilityLeased,
    PlanReady,
    Filling,
    FillValidated,
    Publishable,
    Draining
};

/** @brief Response absence is normal optional admission, not a source fallback or a live free count. */
enum class GrantKind : std::uint32_t
{
    Absent,
    Read,
    Reservation
};

/**
 * @brief Exact immutable frame reply; no coordinator copy is authoritative for physical pins.
 *
 * Opaque CPU destination token is global unique ordinal+1, independent of local
 * slot generations. Equal numeric generations on different owners are legal.
 * Source-only responses use Absent, slot=None, generation/token=0.
 */
struct Grant
{
    Interval interval;
    std::uint64_t row, generation, destination_token;
    std::uint32_t unique, owner, slot;
    GrantKind kind;
};
static_assert(sizeof(Grant) == 64);

/** @brief Existing coordinator locator identity, without a copied pin/free state. */
struct Locator
{
    std::uint64_t row, generation;
    std::uint32_t owner, slot;
};
static_assert(sizeof(Locator) == 24);

/** @brief One actually acquired owner pin-frontier header, emitted only after local pins seal. */
struct PinHeader
{
    Interval interval;
    std::uint32_t owner;
    Status status;
};
static_assert(sizeof(PinHeader) == 32);

/** @brief One acquired owner consensus header for the same actual source window. */
struct PublishHeader
{
    Interval interval;
    SourceWindow source;
    std::uint32_t owner;
    Status status;
};
static_assert(sizeof(PublishHeader) == 56);

/**
 * @brief Privately copied fill acknowledgement after original source completion validation.
 *
 * The future graph binding must prove source completion/intent coordinate and
 * payload copy happen before this value is consumed. A numeric echo alone is
 * not a source capability. This parser checks exact active interval/window and
 * owner-issued reservation again at publication; it performs no mapped reads.
 */
struct FillAcknowledgement
{
    SourceWindow source;
    std::uint64_t destination_token;
};
static_assert(sizeof(FillAcknowledgement) == 32);

/**
 * @brief Bounded immutable increasing-unique view without changing original grant/ACK order.
 *
 * Each order record stores (uint64(unique)<<32)|original_grant_index. The
 * physical parser checks the bound, original unique and strictly increasing
 * order before any phase mutation. The owner plan retains this permutation
 * after both allocation waves until source and all readers have retired.
 */
struct OrderedReservationFills
{
    const Grant *grants;
    const FillAcknowledgement *acknowledgements;
    const std::uint64_t *order;
    std::uint32_t count;
};

/** @brief Actual owner release result; never guessed from a coordinator phase. */
enum class ReleaseDisposition : std::uint32_t { Failed, RetainedReady, RecycledEmpty };
/** @brief Immutable result authored after actual owner release at the last-reader edge. */
struct ReleaseReceipt
{
    Grant original;
    ReleaseDisposition disposition;
    std::uint32_t reserved;
};
static_assert(sizeof(ReleaseReceipt) == 72);

/** @brief Explicit local command discriminator; transport remains a separate graph operation. */
enum class CommandKind : std::uint32_t
{
    PinInitial,
    ReserveFree,
    ReserveVictim,
    PublishReady,
    Release
};

/** @brief Device-authored dispatch order, independent of source unique coordinate and slot generation. */
struct CommandOrdinal { std::uint64_t value; };

/** @brief Bounded command-bank record; only live command count is eligible for transport. */
struct OwnerCommand
{
    Grant target;
    std::uint64_t requested_row; ///< New original row for reserve commands; target retains expected old locator.
    CommandKind kind;
    std::uint32_t reserved;
    CommandOrdinal ordinal;
};
static_assert(sizeof(OwnerCommand) == 88);

/** @brief Exact source retirement observation; Unknown never permits clean cancellation/reuse. */
enum class SourceRetirement : std::uint32_t
{
    Unpublished,
    Consumed,
    Aborted, ///< Exact source-worker retirement without an acquired completion; cancellation only.
    Unknown
};

/** @brief Bounded current transaction; retains last accepted identity while Idle to reject late replay. */
struct alignas(64) OwnerFrame
{
    Interval interval;
    SourceWindow source;
    OwnerPhase phase;
    Status status;
    std::uint32_t held_pins, reservations, initial_pin_count;
    std::uint64_t last_command;
};
static_assert(sizeof(OwnerFrame) == 128);

/** @brief Emitted only after actual owner publication completed with no remaining reservations. */
struct ReadyHeader { Interval interval; SourceWindow source; std::uint32_t owner; Status status; };
/** @brief Emitted only after actual owner reader release and successful physical retirement. */
struct RetiredHeader { Interval interval; SourceWindow source; std::uint32_t owner; Status status; };
static_assert(sizeof(ReadyHeader) == 56 && sizeof(RetiredHeader) == 56);

/**
 * @brief Optional physical lifetime authority, with no directory/sketch/queue/policy state.
 *
 * The prefix fields are the ONLY owner free/generation state for this mode.
 * Frame pin count is diagnostic retirement accounting; slot FramePin is the
 * actual physical reader authority. Persistent occupancy is never replicated
 * on a coordinator, and these logical records are not PMA byte accounting.
 */
struct alignas(64) PhysicalOwnerControl
{
    std::uint64_t next_slot_generation;
    std::uint32_t filled, slot_free, recycled_free, flags;
};
static_assert(sizeof(PhysicalOwnerControl) == 64);

/** @brief Original owner-local arrays borrowed for one exclusive integer operation. */
template <class Slot> struct OwnerView
{
    Slot *slots;
    PhysicalOwnerControl *owner;
    OwnerFrame *frame;
    std::uint64_t table;
    OwnerParameters declaration;
    Grant *initial_pins; ///< Mandatory K-bank authored by actual pinInitial, never coordinator pin state.
};
} // namespace llaminar2::qwen4exp::adaptive_lookup::pooled
