/**
 * @file EffectiveUseOrdering.cpp
 * @brief Per-operation lease + retained replica-record implementation.
 *
 * R11 invariants: touch() is ROLLBACK-FREE (refusals never mutate cells or
 * records before admission of a live lease); lease cells are temporary while
 * replica records are retained and track per-incarnation touch/ack counts with
 * monotonic owner authentication; a lease owns its acknowledgement (acks
 * exactly once); ordinal exhaustion is terminal and rejects BEFORE increment.
 */
#include "EffectiveUseOrdering.h"

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
            record->live = true;
            record->retired = false;
            return record;
        }
    }
    return nullptr;
}
} // namespace

std::optional<EffectiveUseLease> EffectiveUseJournal::touch(EffectiveUseReceipt receipt)
{
    if (!bound())
        return std::nullopt; // No storage: refusal-equivalent (no lease).
    if (receipt.sequence == 0 || receipt.sequence == kSealedSequence())
        return std::nullopt; // Zero or sealed ordinal: obsolete/terminal.
    auto *record = locateRecord(*this, receipt.replica_incarnation, true);
    if (!record)
        return std::nullopt; // No retained record capacity: refuse.
    // ORDERING/OWNERSHIP checks (no mutation): a first touch authenticates by
    // content/table; later touches need the same content/table and a STRICTLY
    // newer ordinal (duplicate/obsolete refuse).
    if (record->touches != 0)
    {
        if (record->auth.content_version != receipt.content_version ||
            record->auth.table_version != receipt.table_version)
            return std::nullopt; // Foreign content/table on live replica.
        if (receipt.sequence <= record->auth.sequence)
            return std::nullopt; // Duplicate or obsolete ordinal (no mutation).
    }
    // A temporary lease cell is required for admission (refusal is
    // rollback-free: no cell/record change below happens before admission).
    std::size_t cell = cell_capacity_;
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (!cells_[i].live) { cell = i; break; }
    if (cell == cell_capacity_)
        return std::nullopt; // All lease cells live: full (no mutation).
    // ADMIT: only now mutate.
    if (record->touches == 0)
    {
        record->auth = receipt; // First touch authenticates the replica.
    }
    else
    {
        record->auth.sequence = receipt.sequence; // Monotonic owner auth.
    }
    cells_[cell] = LeaseCell{receipt, true};
    ++record->touches;
    return EffectiveUseLease(this, receipt);
}

bool EffectiveUseJournal::acknowledge(const EffectiveUseLease &lease) noexcept
{
    if (lease.owner_ != this || lease.acknowledged_ || !lease.owner_)
        return false; // Foreign/moved-from/already-acknowledged: idempotent.
    auto *record = locateRecord(const_cast<EffectiveUseJournal &>(*this),
        lease.identity_.replica_incarnation, false);
    if (!record)
        return false; // Tracked identity vanished: refuse.
    // OWNERSHIP check: the content/table identity must match the retained
    // replica record (the ordinal is per-LEASE, so a specific lease acks its
    // own cell regardless of which sibling is the newest auth).
    if (record->auth.content_version != lease.identity_.content_version ||
        record->auth.table_version != lease.identity_.table_version)
        return false; // Acknowledging a foreign identity: refuse.
    // Release exactly THIS lease cell (identified by its exact receipt).
    bool found = false;
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (cells_[i].live && cells_[i].receipt == lease.identity_)
        {
            cells_[i].live = false;
            cells_[i] = LeaseCell{};
            found = true;
            break;
        }
    if (!found)
        return false; // The exact lease cell is gone: refuse (no double ack).
    ++record->acknowledged;
    return true;
}

bool EffectiveUseJournal::mayEvict(std::uint64_t incarnation) const noexcept
{
    const auto *record = locateRecord(const_cast<EffectiveUseJournal &>(*this),
        incarnation, false);
    if (!record)
        return false; // Untracked replica: never evictable through here.
    return record->acknowledged == record->touches;
}

bool EffectiveUseJournal::retireReplica(std::uint64_t incarnation) noexcept
{
    auto *record = locateRecord(*this, incarnation, false);
    if (!record)
        return false; // Untracked: refusal (idempotent).
    if (record->acknowledged != record->touches)
        return false; // Live obligation: refuse (no silent retirement).
    record->live = false;
    record->retired = true;
    return true;
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

void EffectiveUseJournal::releaseLease(const EffectiveUseLease &lease) noexcept
{
    // Destruction without acknowledgement releases the temporary cell; the
    // RETAINED record keeps its unacknowledged count, so mayEvict stays false
    // (the obligation is not silently retired by lease destruction).
    for (std::size_t i = 0; i < cell_capacity_; ++i)
        if (cells_[i].live && cells_[i].receipt == lease.identity_)
        {
            cells_[i].live = false;
            cells_[i] = LeaseCell{};
            break;
        }
}

EffectiveUseLease::EffectiveUseLease(EffectiveUseJournal *owner, EffectiveUseReceipt identity)
    : owner_(owner), identity_(identity) {}
EffectiveUseLease::~EffectiveUseLease()
{
    if (owner_ && !acknowledged_)
        owner_->releaseLease(*this);
}
EffectiveUseLease::EffectiveUseLease(EffectiveUseLease &&other) noexcept
    : owner_(other.owner_), identity_(other.identity_), acknowledged_(other.acknowledged_)
{
    other.owner_ = nullptr;
}
EffectiveUseLease &EffectiveUseLease::operator=(EffectiveUseLease &&other) noexcept
{
    if (this != &other)
    {
        if (owner_ && !acknowledged_) owner_->releaseLease(*this);
        owner_ = other.owner_;
        identity_ = other.identity_;
        acknowledged_ = other.acknowledged_;
        other.owner_ = nullptr;
    }
    return *this;
}
bool EffectiveUseLease::acknowledge() noexcept
{
    if (!owner_ || acknowledged_)
        return false; // Moved-from/already-acked: idempotent refusal.
    if (!owner_->acknowledge(*this))
        return false;
    acknowledged_ = true;
    owner_ = nullptr;
    return true;
}
void EffectiveUseLease::release() noexcept
{
    if (owner_ && !acknowledged_)
        owner_->releaseLease(*this);
    owner_ = nullptr;
}
} // namespace llaminar2
