/**
 * @file EffectiveUseOrdering.h
 * @brief Per-operation effective-use LEASES with retained replica records.
 *
 * Design §8.2 and R11: effective use is one successful compatible attachment.
 * Each successful touch mints a MOVE-ONLY LEASE that owns its acknowledgement
 * authority (exact full identity: content/table version, replica incarnation,
 * fresh ordinal). A snapshot is NOT a collection of leases: borrowing a
 * snapshot obtains no capability (each consumer must touch and own its lease).
 *
 * Retained replica lifecycle/recency state is SEPARATE from temporary lease
 * cells: replica records carry the monotonic owner authentication and the
 * acknowledged/unacknowledged counts, so lease-cell reuse can never erase the
 * identity needed to distinguish old/duplicate touches, and eviction
 * eligibility of one replica never depends on another's temporary scratch.
 * Refusals are ROLLBACK-FREE (no state changes before admission).
 *
 * Storage is NON-OWNING and admitted by the owner (PMA); this module never
 * allocates and never keeps a lifetime-unbounded history. Both arrays are
 * sized from admitted membership (lease cells = concurrency, replica records
 * = distinct live incarnations).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace llaminar2
{
/** @brief One accepted effective-use touch identity; fresh ordinal, never zero. */
struct EffectiveUseReceipt final
{
    std::uint64_t content_version = 0;
    std::uint64_t table_version = 0;
    std::uint64_t replica_incarnation = 0;
    std::uint64_t sequence = 0;
    bool operator==(const EffectiveUseReceipt &) const = default;
};

/** @brief Typed touch outcome (never a silent drop, never partial mutation). */
enum class EffectiveUseStatus
{
    Accepted,    ///< A lease was minted (admission succeeded).
    Duplicate,   ///< Exact same ordinal already live: refused, no mutation.
    Obsolete,    ///< Older ordinal/superseded identity: refused, no mutation.
    JournalFull, ///< All lease cells (concurrency) occupied: refused, no mutation.
    Sealed,      ///< Ordinal space exhausted: terminal, no mutation.
};

/** @brief Move-only authoritative lease for ONE accepted touch. */
class EffectiveUseLease final
{
public:
    EffectiveUseLease() = default;
    ~EffectiveUseLease();
    EffectiveUseLease(EffectiveUseLease &&other) noexcept;
    EffectiveUseLease &operator=(EffectiveUseLease &&other) noexcept;
    EffectiveUseLease(const EffectiveUseLease &) = delete;
    EffectiveUseLease &operator=(const EffectiveUseLease &) = delete;

    /** @return Whether this handle owns a live lease. */
    explicit operator bool() const noexcept { return owner_ != nullptr; }
    /** @return Exact identity this lease acknowledges. */
    const EffectiveUseReceipt &identity() const noexcept { return identity_; }
    /**
     * @brief Ordering point: acknowledge THIS lease exactly once.
     * @return True when the lease was live and is now acknowledged; false for
     * a moved-from/empty/already-acknowledged handle (idempotent refusal).
     */
    bool acknowledge() noexcept;
    /** @brief Whether this lease has reached its ordering point. */
    bool acknowledged() const noexcept { return acknowledged_; }
private:
    friend class EffectiveUseJournal;
    EffectiveUseLease(class EffectiveUseJournal *owner, EffectiveUseReceipt identity);
    void release() noexcept;
    class EffectiveUseJournal *owner_ = nullptr;
    EffectiveUseReceipt identity_;
    bool acknowledged_ = false;
};

/**
 * @brief Lease cells + retained replica records (one ownership authority).
 *
 * Lease cells are TEMPORARY (one per live touch); replica records are RETAINED
 * (one per distinct incarnation ever accepted until explicit retirement) and
 * track the per-incarnation touch/ack counts plus the monotonic owner
 * authentication. Reuse of lease cells cannot erase replica identity.
 */
class EffectiveUseJournal final
{
public:
    /** @brief One temporary lease cell. */
    struct LeaseCell
    {
        EffectiveUseReceipt receipt;
        bool live = false;
    };
    /** @brief One retained replica record. */
    struct ReplicaRecord
    {
        EffectiveUseReceipt auth;      ///< Monotonic owner authentication.
        std::size_t touches = 0;       ///< Accepted touches on this replica.
        std::size_t acknowledged = 0;  ///< Acknowledged touches on this replica.
        bool live = false;
        bool retired = false;          ///< Explicitly retired; slot reclaimable.
    };
    /** @brief Unbound journal: every operation refuses typed. */
    EffectiveUseJournal() noexcept = default;
    /** @brief Bind admitted lease-cell and replica-record storage. */
    EffectiveUseJournal(std::span<LeaseCell> cells, std::span<ReplicaRecord> records) noexcept
        : cells_(cells.data()), cell_capacity_(cells.size()),
          records_(records.data()), record_capacity_(records.size())
    {
        for (std::size_t i = 0; i < cells.size(); ++i) cells[i] = LeaseCell{};
        for (std::size_t i = 0; i < records.size(); ++i) records[i] = ReplicaRecord{};
    }
    /** @brief Bind storage once (two-phase owner construction); refuses if live. */
    bool bind(std::span<LeaseCell> cells, std::span<ReplicaRecord> records) noexcept
    {
        if (bound() || hasPending())
            return false; // Already bound or live leases: refuse.
        cells_ = cells.data(); cell_capacity_ = cells.size();
        records_ = records.data(); record_capacity_ = records.size();
        for (std::size_t i = 0; i < cells.size(); ++i) cells[i] = LeaseCell{};
        for (std::size_t i = 0; i < records.size(); ++i) records[i] = ReplicaRecord{};
        return true;
    }
    EffectiveUseJournal(const EffectiveUseJournal &) = delete;
    EffectiveUseJournal &operator=(const EffectiveUseJournal &) = delete;

    /**
     * @brief Mint one lease for a fresh, authenticated touch (rollback-free).
     * @return The owned lease on acceptance; empty with NO state change on
     * duplicate/obsolete/full/sealed refusal.
     */
    std::optional<EffectiveUseLease> touch(EffectiveUseReceipt receipt);
    /**
     * @brief Ordering point for one lease (passes through to the lease).
     * @return True when THIS lease was live and is now acknowledged.
     */
    bool acknowledge(const EffectiveUseLease &lease) noexcept;
    /** @return Whether @p incarnation may be evicted (fully acknowledged, not retired-lived). */
    bool mayEvict(std::uint64_t incarnation) const noexcept;
    /**
     * @brief Explicit retirement of one retained replica record.
     * @return True when the record was retired (all touches acknowledged);
     * false when any touch is still unacknowledged or the incarnation is
     * untracked (no mutation on refusal).
     */
    bool retireReplica(std::uint64_t incarnation) noexcept;
    /** @return Whether any lease cell is live. */
    bool hasPending() const noexcept;
    /** @return Highest accepted ordinal over the retained records. */
    std::uint64_t watermark() const noexcept;
    /** @return Admitted lease-cell capacity (0 when unbound). */
    std::size_t leaseCapacity() const noexcept { return cell_capacity_; }
    /** @return Admitted replica-record capacity (0 when unbound). */
    std::size_t replicaCapacity() const noexcept { return record_capacity_; }
    /** @return Whether storage is bound. */
    bool bound() const noexcept { return cells_ != nullptr && records_ != nullptr; }
    /** @brief Test/storage access: retained record at @p index (bounds checked). */
    ReplicaRecord *replicaRecordAt(std::size_t index) noexcept
    {
        return index < record_capacity_ ? records_ + index : nullptr;
    }
    ReplicaRecord *replicaRecordAt(std::size_t index) const noexcept
    {
        return index < record_capacity_ ? records_ + index : nullptr;
    }
    /** @brief Sealed ordinal sentinel (terminal exhaustion marker). */
    static constexpr std::uint64_t kSealedSequence() noexcept { return 0; }
private:
    friend class EffectiveUseLease;
    void releaseLease(const EffectiveUseLease &lease) noexcept;
    LeaseCell *cells_ = nullptr;
    std::size_t cell_capacity_ = 0;
    ReplicaRecord *records_ = nullptr;
    std::size_t record_capacity_ = 0;
};
} // namespace llaminar2
