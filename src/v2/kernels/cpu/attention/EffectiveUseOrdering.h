/**
 * @file EffectiveUseOrdering.h
 * @brief Per-operation effective-use LEASES with retained replica records.
 *
 * R12/R13 disposition (combined review): every successful touch mints a
 * MOVE-ONLY lease that owns its acknowledgement authority, and the lease
 * retains an OWNING CONTROL BLOCK (journal + lifetime token) so journal
 * storage can never be freed while a lease is outstanding. Admission is
 * ROLLBACK-FREE: the free lease cell is located BEFORE any replica-record
 * mutation, so a full journal refuses without consuming a record. Losing the
 * only capability for an unacknowledged lease is a HARD ERROR unless the
 * consumer explicitly ABANDONS its obligation (bounded, never a silent leak).
 * Retirement is retried automatically when the last obligation of a
 * retire-requested replica discharges.
 *
 * Storage is NON-OWNING and admitted by the owner (PMA): cells and records are
 * owner arrays; the lease's lifetime token keeps that owner alive.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
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
    JournalFull, ///< All lease cells occupied: refused, NO record mutation.
    Sealed,      ///< Ordinal space exhausted: terminal, no mutation.
};

class EffectiveUseJournal;
/** @brief Lease control block: journal + lifetime token owning its storage. */
struct EffectiveUseLeaseControl
{
    EffectiveUseJournal *journal = nullptr;
    std::shared_ptr<void> lifetime; ///< Original owner (table) retained by the lease.
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
    explicit operator bool() const noexcept { return static_cast<bool>(control_); }
    /** @return Exact identity this lease acknowledges. */
    const EffectiveUseReceipt &identity() const noexcept;
    /** @brief Ordering point: acknowledge THIS lease exactly once. */
    bool acknowledge() noexcept;
    /**
     * @brief Explicit abandonment: the consumer gives up its obligation.
     *
     * The obligation is recorded as abandoned (never fabricated as
     * acknowledged); eviction eligibility counts abandoned obligations as
     * discharged, and the lease's capability is released exactly once.
     * This is the BOUNDED alternative to a permanent pin: the consumer exists
     * and expressly relinquishes, so no ordinal is silently lost either way.
     */
    bool abandon() noexcept;
    /** @return Whether this lease has reached its ordering point. */
    bool acknowledged() const noexcept { return acknowledged_; }
private:
    friend class EffectiveUseJournal;
    EffectiveUseLease(std::shared_ptr<EffectiveUseLeaseControl> control,
        EffectiveUseReceipt identity);
    void release(bool abandoned) noexcept;
    std::shared_ptr<EffectiveUseLeaseControl> control_;
    EffectiveUseReceipt identity_;
    bool acknowledged_ = false;
    bool abandoned_ = false;
};

/**
 * @brief Lease cells + retained replica records (one ownership authority).
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
        std::size_t abandoned = 0;     ///< Explicitly abandoned obligations.
        bool live = false;
        bool retire_requested = false; ///< Retirement pending final discharge.
        bool retired = false;          ///< Retired; slot reclaimed.
        /** @return Whether every obligation (acknowledged or abandoned) discharged. */
        bool discharged() const noexcept { return acknowledged + abandoned == touches; }
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
            return false;
        cells_ = cells.data(); cell_capacity_ = cells.size();
        records_ = records.data(); record_capacity_ = records.size();
        for (std::size_t i = 0; i < cells.size(); ++i) cells[i] = LeaseCell{};
        for (std::size_t i = 0; i < records.size(); ++i) records[i] = ReplicaRecord{};
        return true;
    }
    EffectiveUseJournal(const EffectiveUseJournal &) = delete;
    EffectiveUseJournal &operator=(const EffectiveUseJournal &) = delete;

    /**
     * @brief Mint one lease for a fresh, authenticated touch (ROLLBACK-FREE).
     * @param lifetime The owner's shared lifetime token so the journal storage
     * cannot be freed while the lease is outstanding.
     * @return The owned lease on acceptance; empty with NO state change on
     * duplicate/obsolete/full/sealed refusal (a full journal does NOT consume
     * a replica record).
     */
    std::optional<EffectiveUseLease> touch(EffectiveUseReceipt receipt,
        std::shared_ptr<void> lifetime = {});
    /** @brief Ordering point for one lease. */
    bool acknowledge(const EffectiveUseLease &lease) noexcept;
    /** @brief Explicit abandonment for one lease. */
    bool abandon(const EffectiveUseLease &lease) noexcept;
    /** @return Whether @p incarnation may be evicted (all obligations discharged). */
    bool mayEvict(std::uint64_t incarnation) const noexcept;
    /**
     * @brief Explicit retirement of one retained replica record.
     * @return True when the record was retired (all obligations discharged);
     * false when any touch is still outstanding, in which case retire_requested
     * is recorded and the retire completes automatically when the last
     * obligation discharges (late-ack retry).
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
    bool releaseLeaseFor(const EffectiveUseLease &lease, bool abandoned) noexcept;
    bool findFreeCell(std::size_t &out) const noexcept;
    LeaseCell *cells_ = nullptr;
    std::size_t cell_capacity_ = 0;
    ReplicaRecord *records_ = nullptr;
    std::size_t record_capacity_ = 0;
};
} // namespace llaminar2
