/**
 * @file EffectiveUseOrdering.cpp
 * @brief Per-operation lease + retained replica-record implementation.
 *
 * R12/R13 disposition invariants:
 * - touch() is ROLLBACK-FREE: the free lease cell is located BEFORE any
 *   replica-record mutation, so a full journal refuses without consuming a
 *   record.
 * - A lease retains an OWNING CONTROL BLOCK (journal + lifetime token), so
 *   journal storage cannot be freed while a lease is outstanding.
 * - Destroying an unacknowledged lease without abandon() or acknowledge() is a
 *   HARD ERROR (std::terminate): the only capability for that obligation is
 *   never silently lost.
 * - retireReplica records retire_requested when obligations are outstanding;
 *   the retirement completes automatically when the LAST obligation discharges
 *   (late-ack retry).
 */
#include "EffectiveUseOrdering.h"
#include <cassert>
#include <utility>

namespace llaminar2
{
namespace
{
/** @brief Locate (or allocate) the retained record for one incarnation. */
EffectiveUseJournal::ReplicaRecord *locateRecord(EffectiveUseJournal &journal,
    std::uint64_t incarnation, bool allocate)
{
    for (std::size_t i = 0; i < journal.replicaCapacity(); ++i)
    {
        auto *record = journal.replicaRecordAt(i);
        if (record->live && record->auth.replica_incarnation == incarnation)
            return record;
    }
    if (!allocate) return nullptr;
    for (std::size_t i = 0; i < journal.replicaCapacity(); ++i)
    {
        auto *record = journal.replicaRecordAt(i);
        if (!record->live)
        {
            record->auth = {};
            record->auth.replica_incarnation = incarnation;
            record->touches = 0;
            record->acknowledged = 0;
            record->abandoned = 0;
            record->live = true;
            record->retire_requested = false;
            record->retired = false;
            return record;
        }
    }
    return nullptr;
}
} // namespace

bool EffectiveUseJournal::findFreeCell(std::size_t &out) const noexcept
{
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (!cells_[i].live) { out = i; return true; }
    return false;
}

std::optional<EffectiveUseLease> EffectiveUseJournal::touch(EffectiveUseReceipt receipt,
    std::shared_ptr<void> lifetime)
{
    if (!bound())
        return std::nullopt;
    if (receipt.sequence == 0 || receipt.sequence == kSealedSequence())
        return std::nullopt; // Zero or sealed ordinal: refuse.
    // ROLLBACK-FREE: locate the free cell FIRST — a full journal must not
    // consume any replica record (no partial mutation on refusal).
    std::size_t cell = 0;
    if (!findFreeCell(cell))
        return std::nullopt; // All lease cells live: JournalFull, NO mutation.
    auto *record = locateRecord(*this, receipt.replica_incarnation, true);
    if (!record)
        return std::nullopt; // No retained record capacity: refuse (cell untouched).
    // Ordering/ownership checks: a first touch authenticates by content/table;
    // later touches need the same identity and a STRICTLY newer ordinal.
    if (record->touches != 0)
    {
        if (record->auth.content_version != receipt.content_version ||
            record->auth.table_version != receipt.table_version)
            return std::nullopt; // Foreign content/table (no mutation).
        if (receipt.sequence <= record->auth.sequence)
            return std::nullopt; // Duplicate or obsolete ordinal (no mutation).
    }
    // ADMIT: only now mutate (all refusal checks precede every change).
    if (record->touches == 0)
        record->auth = receipt;
    else
        record->auth.sequence = receipt.sequence;
    cells_[cell] = LeaseCell{receipt, true};
    ++record->touches;
    auto control = std::make_shared<EffectiveUseLeaseControl>();
    control->journal = this;
    control->lifetime = std::move(lifetime);
    return EffectiveUseLease(std::move(control), receipt);
}

bool EffectiveUseJournal::acknowledge(const EffectiveUseLease &lease) noexcept
{
    if (!lease.control_ || lease.control_->journal != this || lease.acknowledged_ ||
        lease.abandoned_ || !lease.control_)
        return false; // Foreign/moved-from/already-resolved: idempotent.
    auto *record = locateRecord(*this, lease.identity_.replica_incarnation, false);
    if (!record)
        return false;
    if (record->auth.content_version != lease.identity_.content_version ||
        record->auth.table_version != lease.identity_.table_version)
        return false; // Acknowledging a foreign identity: refuse.
    if (!releaseLeaseFor(lease, false))
        return false; // The exact lease cell vanished: refuse (no double ack).
    ++record->acknowledged;
    // Late-ack retirement retry: if this replica had retirement requested and
    // every obligation has now discharged, complete the retirement.
    if (record->retire_requested && record->discharged())
    {
        record->live = false;
        record->retired = true;
    }
    return true;
}

bool EffectiveUseJournal::abandon(const EffectiveUseLease &lease) noexcept
{
    if (!lease.control_ || lease.control_->journal != this || lease.acknowledged_ ||
        lease.abandoned_ || !lease.control_)
        return false;
    auto *record = locateRecord(*this, lease.identity_.replica_incarnation, false);
    if (!record)
        return false;
    if (record->auth.content_version != lease.identity_.content_version ||
        record->auth.table_version != lease.identity_.table_version)
        return false;
    if (!releaseLeaseFor(lease, true))
        return false;
    ++record->abandoned;
    if (record->retire_requested && record->discharged())
    {
        record->live = false;
        record->retired = true;
    }
    return true;
}

bool EffectiveUseJournal::releaseLeaseFor(const EffectiveUseLease &lease, bool abandoned) noexcept
{
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (cells_[i].live && cells_[i].receipt == lease.identity_)
        {
            cells_[i].live = false;
            cells_[i] = LeaseCell{};
            const_cast<EffectiveUseLease &>(lease).control_.reset();
            const_cast<EffectiveUseLease &>(lease).acknowledged_ = !abandoned;
            const_cast<EffectiveUseLease &>(lease).abandoned_ = abandoned;
            return true;
        }
    return false;
}

bool EffectiveUseJournal::mayEvict(std::uint64_t incarnation) const noexcept
{
    // A RETIRED record (all obligations discharged and reclamation completed)
    // is evictable even though it is no longer live; an untracked incarnation
    // never is. Discharged-live records are evictable too.
    for (std::size_t i = 0; i < record_capacity_; ++i)
    {
        const auto &record = records_[i];
        if (record.auth.replica_incarnation != incarnation)
            continue;
        if (record.live)
            return record.discharged();
        return record.retired; // Fully retired: evictable.
    }
    return false; // Untracked replica: never evictable through here.
}

bool EffectiveUseJournal::retireReplica(std::uint64_t incarnation) noexcept
{
    auto *record = locateRecord(*this, incarnation, false);
    if (!record)
        return false; // Untracked: refusal (idempotent).
    if (record->discharged())
    {
        record->live = false;
        record->retired = true;
        return true;
    }
    // Obligations outstanding: record the request; the retirement completes
    // automatically when the LAST obligation discharges (late-ack retry).
    record->retire_requested = true;
    return false;
}

bool EffectiveUseJournal::hasPending() const noexcept
{
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (cells_[i].live) return true;
    return false;
}

std::uint64_t EffectiveUseJournal::watermark() const noexcept
{
    std::uint64_t max = 0;
    for (std::size_t i = 0; i < record_capacity_; ++i)
        if (records_[i].live && records_[i].auth.sequence > max)
            max = records_[i].auth.sequence;
    return max;
}

EffectiveUseLease::EffectiveUseLease(std::shared_ptr<EffectiveUseLeaseControl> control,
    EffectiveUseReceipt identity)
    : control_(std::move(control)), identity_(identity) {}
EffectiveUseLease::~EffectiveUseLease()
{
    if (!control_ || acknowledged_ || abandoned_)
        return;
    // Losing the only capability for an unacknowledged obligation is a HARD
    // ERROR: the consumer must acknowledge or explicitly abandon first.
    std::terminate();
}
EffectiveUseLease::EffectiveUseLease(EffectiveUseLease &&other) noexcept
    : control_(std::move(other.control_)), identity_(other.identity_),
      acknowledged_(other.acknowledged_), abandoned_(other.abandoned_)
{
    other.identity_ = EffectiveUseReceipt{};
    other.acknowledged_ = false;
    other.abandoned_ = false;
}
EffectiveUseLease &EffectiveUseLease::operator=(EffectiveUseLease &&other) noexcept
{
    if (this != &other)
    {
        if (control_ && !acknowledged_ && !abandoned_)
            std::terminate(); // Never overwrite a live obligation.
        control_ = std::move(other.control_);
        identity_ = other.identity_;
        acknowledged_ = other.acknowledged_;
        abandoned_ = other.abandoned_;
        other.identity_ = EffectiveUseReceipt{};
        other.acknowledged_ = false;
        other.abandoned_ = false;
    }
    return *this;
}
const EffectiveUseReceipt &EffectiveUseLease::identity() const noexcept
{
    return identity_;
}
bool EffectiveUseLease::acknowledge() noexcept
{
    if (!control_ || acknowledged_ || abandoned_)
        return false;
    const bool ok = control_->journal->acknowledge(*this);
    return ok && acknowledged_;
}
bool EffectiveUseLease::abandon() noexcept
{
    if (!control_ || acknowledged_ || abandoned_)
        return false;
    const bool ok = control_->journal->abandon(*this);
    return ok && abandoned_;
}
void EffectiveUseLease::release(bool abandoned) noexcept
{
    if (control_ && !acknowledged_ && !abandoned_)
    {
        control_->journal->releaseLeaseFor(*this, abandoned);
        acknowledged_ = !abandoned;
        abandoned_ = abandoned;
    }
    control_.reset();
}
} // namespace llaminar2
