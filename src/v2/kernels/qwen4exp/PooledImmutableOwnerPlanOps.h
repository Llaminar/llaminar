/**
 * @file PooledImmutableOwnerPlanOps.h
 * @brief Fixed-phase availability leases and accepted physical allocation plans.
 *
 * This owner-local parser reuses the canonical physical grant/slot operations.
 * Its free count is a frozen frame lease over actual slots, never a coordinator
 * occupancy mirror or a physical-byte ledger. No source, transport, allocator or
 * runtime capability is minted by decoded records. One exclusive native writer
 * owns this view until the accepted plan and all source/readers retire.
 */
#pragma once
#include "PooledImmutablePhysicalOwnerOps.h"
namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/** @brief Availability authored only after initial reader pins are sealed. */
struct AvailabilityLease { Interval interval; std::uint32_t owner, free_slots; };
static_assert(sizeof(AvailabilityLease) == 32);
/** @brief Immutable decoded allocation decision; no victim is selected by its owner. */
struct AllocationDecision
{
    std::uint64_t row;
    std::uint32_t unique;
    CommandKind kind;
    Locator victim;
    CommandOrdinal ordinal;
};
static_assert(sizeof(AllocationDecision) == 48);
/** @brief Mandatory owner-local plan state, separate from persistent slot authority. */
enum class AllocationPhase : std::uint32_t { Idle, Leased, Accepted, Executed, FreeFrontier, SourceSealed, Cancelled };
/** @brief Closed first-wave view over directed free decisions from one frozen owner lease. */
struct FreeAllocationPlan { AvailabilityLease lease; const AllocationDecision *decisions; std::uint32_t count; };
/** @brief Closed second-wave view; admitted only after every actual owner's free completion. */
struct VictimAllocationPlan { AvailabilityLease lease; const AllocationDecision *decisions; std::uint32_t count; };
/** @brief Internal wave identity retained across cancellation; public plans have distinct types. */
enum class AllocationWave : std::uint32_t { Free, Victim };
/** @brief Actual owner-authored free completion; unused capacity cannot authorize any eviction. */
struct FreeAllocationReceipt
{
    AvailabilityLease lease;
    std::uint32_t granted;
    Status status;
};
/** @brief Actual second-wave completion; cancellation reports zero when no victim plan was accepted. */
struct VictimAllocationReceipt
{
    AvailabilityLease lease;
    std::uint32_t granted;
    Status status;
};
static_assert(sizeof(FreeAllocationReceipt) == 40 && sizeof(VictimAllocationReceipt) == 40);
/** @brief One bounded frame's immutable plan metadata, not a second cache directory. */
struct AllocationFrame
{
    AvailabilityLease lease{};
    AllocationPhase phase = AllocationPhase::Idle;
    std::uint32_t count = 0, wave_offset = 0, wave_count = 0, free_count = 0;
    AllocationWave wave = AllocationWave::Free;
    bool all_free_consumed = false;
};
/** @brief Preadmitted linear-work validation scratch, written only at live command extent. */
struct AllocationValidationScratch
{
    std::uint64_t *keys_a, *keys_b;
    std::uint32_t *radix_counts; ///< Exactly256 bins, never a source/table-sized clearing operation.
};
/** @brief Typed capacity for disjoint preadmitted owner-private plan banks. */
struct AllocationPlanStorage
{
    AllocationDecision *accepted;
    Grant *grants;
    AllocationValidationScratch scratch;
    std::uint32_t capacity;
};
/** @brief Allocation-free native parser over one exclusive owner's actual backing. */
template<class Slot> class OwnerPlanOperations final
{
public:
    /** @brief Bind existing owner state and preadmitted decision/grant banks of maximumUnique capacity. */
    QFN_ADAPTIVE_HD OwnerPlanOperations(OwnerView<Slot> owner, AllocationFrame &frame,
        AllocationPlanStorage storage)
        : owner_(owner), frame_(frame), accepted_(storage.accepted), grants_(storage.grants),
          scratch_(storage.scratch), capacity_(storage.capacity) {}
    /** @brief Freeze exact actual free slots after the complete initial-pin frontier. */
    QFN_ADAPTIVE_HD bool acquire(Interval current)
    {
        if (!validStorage() ||
            !same(current, owner_.frame->interval) || owner_.frame->phase != OwnerPhase::Reserving ||
            owner_.frame->status != Status::Success || frame_.phase != AllocationPhase::Idle || frame_.count || frame_.wave_count ||
            frame_.wave_offset || frame_.free_count || frame_.all_free_consumed ||
            !accepted_ || !grants_) return fail();
        std::uint32_t free = 0;
        if (owner_.declaration.capacity())
        {
            const auto &physical = *owner_.owner;
            if (physical.filled > owner_.declaration.capacity() || physical.recycled_free > physical.filled ||
                ((physical.slot_free == None) != (physical.recycled_free == 0)))
                return fail();
            free = owner_.declaration.capacity() - physical.filled + physical.recycled_free;
        }
        frame_.lease = {current, owner_.declaration.coordinate(), free};
        owner_.frame->phase = OwnerPhase::AvailabilityLeased;
        frame_.phase = AllocationPhase::Leased;
        return true;
    }
    /** @brief Retain the complete free wave before any physical allocation; no victim command is admissible. */
    QFN_ADAPTIVE_HD bool accept(FreeAllocationPlan plan)
    { return acceptWave(AllocationWave::Free, plan.lease, plan.decisions, plan.count); }
    /** @brief Append the victim wave only after actual complete free exhaustion; original grants remain immutable. */
    QFN_ADAPTIVE_HD bool accept(VictimAllocationPlan plan)
    { return acceptWave(AllocationWave::Victim, plan.lease, plan.decisions, plan.count); }
    /** @brief Execute only the accepted first wave; cancellation also drains this same immutable prefix. */
    QFN_ADAPTIVE_HD bool executeFree() { return executeWave(AllocationWave::Free); }
    /** @brief Execute only the accepted second wave without minting a frame or resetting physical generations. */
    QFN_ADAPTIVE_HD bool executeVictims() { return executeWave(AllocationWave::Victim); }
private:
    /**
     * @brief Validate and retain the entire immutable plan before any physical victim mutation.
     * @param lease Exact owner-issued current availability record, not a new capacity request.
     * @param decisions Live directed owner decisions; count is their actual extent.
     * @param count Exact bounded decision count. Free decisions precede victim decisions.
     * @return False on duplicate/stale/foreign or over-lease plans, before changing slots.
     */
    QFN_ADAPTIVE_HD bool acceptWave(AllocationWave wave, AvailabilityLease lease, const AllocationDecision *decisions, std::uint32_t count)
    {
        if (!validStorage() || owner_.frame->status != Status::Success ||
            (wave != AllocationWave::Free && wave != AllocationWave::Victim) ||
            (wave == AllocationWave::Free
                ? frame_.phase != AllocationPhase::Leased || owner_.frame->phase != OwnerPhase::AvailabilityLeased
                : frame_.phase != AllocationPhase::FreeFrontier || owner_.frame->phase != OwnerPhase::PlanReady ||
                  (count && !frame_.all_free_consumed)) || !same(lease.interval, frame_.lease.interval) ||
            lease.owner != frame_.lease.owner || lease.free_slots != frame_.lease.free_slots ||
            count > owner_.declaration.maximumUnique() ||
            frame_.count > owner_.declaration.maximumUnique() - count ||
            owner_.frame->initial_pin_count > owner_.declaration.maximumUnique() - count - frame_.count || (count && (!decisions || overlapsStorage(decisions, std::size_t{count} * sizeof(AllocationDecision)))) ||
            owner_.frame->held_pins > owner_.declaration.maximumUnique() - count) return fail();
        std::uint32_t free = 0;
        std::uint64_t ordinal = owner_.frame->last_command;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const auto &d = decisions[i];
            if (d.row >= owner_.declaration.sourceRows() || d.unique >= owner_.declaration.maximumUnique() ||
                d.ordinal.value <= ordinal) return fail();
            ordinal = d.ordinal.value;
            if (d.kind == CommandKind::ReserveFree) { if (wave != AllocationWave::Free || ++free > lease.free_slots) return fail(); }
            else if (d.kind == CommandKind::ReserveVictim)
            {
                if (wave != AllocationWave::Victim) return fail();
                if (d.victim.owner != lease.owner || d.victim.slot >= owner_.declaration.capacity()) return fail();
                const auto &slot = owner_.slots[d.victim.slot];
                if (slot.row != d.victim.row || slot.generation != d.victim.generation ||
                    PhysicalSlot<Slot>::state(slot) != SlotState::Ready || PhysicalSlot<Slot>::pinned(slot)) return fail();
            }
            else return fail();

        }
        if (count && (!scratch_.keys_a || !scratch_.keys_b || !scratch_.radix_counts)) return fail();
        // Three closed integer keys use a fixed number of radix passes: O(live commands),
        // independent of physical cache/source capacity and adversarial key collisions.
        if (!uniqueKeys(decisions, count, 0, 8) || !uniqueKeys(decisions, count, 1, 4) ||
            !uniqueKeys(decisions, count, 2, 4)) return fail();
        if (owner_.declaration.capacity() &&
            count > ~std::uint64_t{0} - owner_.owner->next_slot_generation) return fail();
        for (std::uint32_t i = 0; i < count; ++i) accepted_[frame_.count + i] = decisions[i];
        frame_.wave_offset = frame_.count;
        frame_.wave_count = count;
        frame_.count += count;
        frame_.wave = wave;
        frame_.phase = AllocationPhase::Accepted;
        return true;
    }
    /** @brief Drain the accepted immutable plan in bounded command order, even after cancellation intent. */
    QFN_ADAPTIVE_HD bool executeWave(AllocationWave wave)
    {
        if (!validStorage() || frame_.phase != AllocationPhase::Accepted || frame_.wave != wave ||
            frame_.count > owner_.declaration.maximumUnique() || frame_.wave_offset > frame_.count ||
            frame_.wave_count != frame_.count - frame_.wave_offset) return fail();
        if ((owner_.frame->phase != OwnerPhase::AvailabilityLeased && owner_.frame->phase != OwnerPhase::PlanReady) ||
            owner_.frame->status != Status::Success) return fail();
        owner_.frame->phase = OwnerPhase::Reserving;
        OwnerOperations<Slot> physical(owner_);
        for (std::uint32_t i = frame_.wave_offset; i < frame_.count; ++i)
        {
            const auto &d = accepted_[i];
            grants_[i] = d.kind == CommandKind::ReserveFree
                ? physical.reserveFree(frame_.lease.interval, d.row, d.unique, d.ordinal)
                : physical.reserveVictim(frame_.lease.interval, d.victim, d.row, d.unique, d.ordinal);
            if (grants_[i].kind != GrantKind::Reservation) return fail();
        }
        owner_.frame->phase = OwnerPhase::PlanReady;
        if (wave == AllocationWave::Free) frame_.free_count = frame_.wave_count;
        frame_.phase = AllocationPhase::Executed;
        return true;
    }
public:
    /** @brief Publish actual free-wave consumption; clean cancellation remains terminal and forbids every victim. */
    QFN_ADAPTIVE_HD FreeAllocationReceipt freeReceipt()
    {
        if (!validStorage() || frame_.wave != AllocationWave::Free ||
            !((frame_.phase == AllocationPhase::Executed && owner_.frame->phase == OwnerPhase::PlanReady &&
                owner_.frame->status == Status::Success) ||
              (frame_.phase == AllocationPhase::Cancelled && owner_.frame->phase == OwnerPhase::Draining &&
                owner_.frame->status == Status::Cancelled)))
        { fail(); return {frame_.lease, 0, Status::Invariant}; }
        return {frame_.lease, frame_.free_count, owner_.frame->status};
    }
    /**
     * @brief Publish actual second-wave completion, including drained accepted cancellation.
     * @return Exact executed victim count; unaccepted cancelled plans report zero rather than guessed grants.
     * @note The immutable free prefix remains retained; source/reader retirement still controls frame reuse.
     */
    QFN_ADAPTIVE_HD VictimAllocationReceipt victimReceipt()
    {
        if (!validStorage() || frame_.count < frame_.free_count ||
            !((frame_.phase == AllocationPhase::Executed && frame_.wave == AllocationWave::Victim &&
                owner_.frame->phase == OwnerPhase::PlanReady && owner_.frame->status == Status::Success) ||
              (frame_.phase == AllocationPhase::Cancelled && owner_.frame->phase == OwnerPhase::Draining &&
                owner_.frame->status == Status::Cancelled)))
        { fail(); return {frame_.lease, 0, Status::Invariant}; }
        return {frame_.lease, frame_.count - frame_.free_count, owner_.frame->status};
    }
    /**
     * @brief Acquire the complete actual free-wave frontier before any victim wave.
     * @param receipts Exact acquired owner-issued records in original declared order.
     * @param count Actual owner count, including C0; source/transport binding authenticates producers.
     * @note Unused free quota permits only an all-free/no-victim frame, never an eviction.
     */
    QFN_ADAPTIVE_HD bool acceptFreeFrontier(const FreeAllocationReceipt *receipts, std::uint32_t count)
    {
        if (!validStorage() || frame_.phase != AllocationPhase::Executed || frame_.wave != AllocationWave::Free ||
            owner_.frame->status != Status::Success || !receipts || count != owner_.declaration.participants()) return fail();
        bool consumed = true;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const auto &r = receipts[i];
            if (!same(r.lease.interval, frame_.lease.interval) || r.lease.owner != i ||
                (r.status != Status::Success && r.status != Status::Cancelled) || r.granted > r.lease.free_slots || r.granted > owner_.declaration.maximumUnique()) return fail();
            if (i == owner_.declaration.coordinate() &&
                (r.lease.free_slots != frame_.lease.free_slots || r.granted != frame_.free_count ||
                 r.status != owner_.frame->status)) return fail();
            consumed = consumed && r.status == Status::Success && r.granted == r.lease.free_slots;
        }
        frame_.all_free_consumed = consumed;
        frame_.phase = AllocationPhase::FreeFrontier;
        return true;
    }
    /** @brief Seal one source window after completed allocation waves; no second lookup frame is minted. */
    QFN_ADAPTIVE_HD bool sealSource(SourceWindow source)
    {
        if (!validStorage() || owner_.frame->status != Status::Success ||
            frame_.count > owner_.declaration.maximumUnique() || frame_.count != owner_.frame->reservations ||
            (frame_.phase != AllocationPhase::FreeFrontier &&
            !(frame_.phase == AllocationPhase::Executed && frame_.wave == AllocationWave::Victim))) return fail();
        // Acceptance scratch is dead after both waves. Reuse its first bank for
        // an immutable permutation, keeping append-only accepted/grant records
        // in actual command order for cancellation and original observations.
        for (std::uint32_t i = 0; i < frame_.count; ++i)
            scratch_.keys_a[i] = (std::uint64_t{grants_[i].unique} << 32) | i;
        auto *input = scratch_.keys_a;
        auto *output = scratch_.keys_b;
        for (unsigned pass = 4; pass < 8; ++pass)
        {
            for (unsigned bin = 0; bin < 256; ++bin) scratch_.radix_counts[bin] = 0;
            for (std::uint32_t i = 0; i < frame_.count; ++i)
                ++scratch_.radix_counts[(input[i] >> (pass * 8)) & 255];
            std::uint32_t cursor = 0;
            for (unsigned bin = 0; bin < 256; ++bin)
            {
                const auto size = scratch_.radix_counts[bin];
                scratch_.radix_counts[bin] = cursor;
                cursor += size;
            }
            for (std::uint32_t i = 0; i < frame_.count; ++i)
                output[scratch_.radix_counts[(input[i] >> (pass * 8)) & 255]++] = input[i];
            auto *swap = input; input = output; output = swap;
        }
        if (!OwnerOperations<Slot>(owner_).sealReservations(frame_.lease.interval, source)) return false;
        frame_.phase = AllocationPhase::SourceSealed;
        return true;
    }
    /**
     * @brief Validate actual ACKs through the immutable increasing-unique view of retained grants.
     * @param acknowledgements Private acquired ACKs in original append-only grant order.
     * @param count Exact live reservation count, never K or physical capacity.
     * @return Actual owner publication header; malformed order/source/payload tokens poison the owner.
     * @note Sorting neither overwrites original grants nor changes original source coordinates.
     */
    QFN_ADAPTIVE_HD PublishHeader validateFills(const FillAcknowledgement *acknowledgements, std::uint32_t count)
    {
        if (!validStorage() || frame_.phase != AllocationPhase::SourceSealed || count != frame_.count)
        {
            fail();
            return {frame_.lease.interval, {}, owner_.declaration.coordinate(), Status::Invariant};
        }
        return OwnerOperations<Slot>(owner_).validateFills(frame_.lease.interval,
            OrderedReservationFills{grants_, acknowledgements, scratch_.keys_a, count});
    }
    /**
     * @brief Close an unaccepted lease without victim mutation, or drain an accepted plan before cancellation.
     * @note Existing source-window Consumed/Unpublished rules remain the physical owner's authority.
     */
    QFN_ADAPTIVE_HD bool cancel(SourceRetirement retirement, SourceWindow source = {})
    {
        if (!validStorage()) return fail();
        if (frame_.phase == AllocationPhase::Accepted && !executeWave(frame_.wave)) return false;
        if (frame_.phase != AllocationPhase::Leased && frame_.phase != AllocationPhase::Executed &&
            frame_.phase != AllocationPhase::FreeFrontier && frame_.phase != AllocationPhase::SourceSealed) return fail();
        if (!OwnerOperations<Slot>(owner_).cancel(frame_.lease.interval, retirement, source)) return false;
        frame_.phase = AllocationPhase::Cancelled;
        return true;
    }
    /** @brief Retire actual source/readers before allowing another lease; generations never reset. */
    QFN_ADAPTIVE_HD bool retire(SourceRetirement retirement, SourceWindow source = {})
    {
        if (!validStorage() || (frame_.phase != AllocationPhase::Executed && frame_.phase != AllocationPhase::Cancelled &&
            frame_.phase != AllocationPhase::FreeFrontier && frame_.phase != AllocationPhase::SourceSealed)) return fail();
        if (!OwnerOperations<Slot>(owner_).retire(frame_.lease.interval, retirement, source)) return false;
        frame_.phase = AllocationPhase::Idle;
        frame_.count = frame_.wave_count = frame_.wave_offset = frame_.free_count = 0;
        frame_.wave = AllocationWave::Free;
        frame_.all_free_consumed = false;
        return true;
    }

private:
    /** @brief Checked byte-range relation over declared native banks, never payload contents. */
    QFN_ADAPTIVE_HD static bool overlap(const void *a, std::size_t an, const void *b, std::size_t bn)
    {
        if (!an || !bn) return false;
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        if (!a || !b || x > ~std::uintptr_t{0} - an || y > ~std::uintptr_t{0} - bn) return true;
        return x < y + bn && y < x + an;
    }
    /** @brief Reject any input alias of retained mutable plan/slot/control banks before scratch writes. */
    QFN_ADAPTIVE_HD bool overlapsStorage(const void *data, std::size_t bytes) const
    {
        const auto k = std::size_t{owner_.declaration.maximumUnique()};
        return overlap(data, bytes, accepted_, k * sizeof(AllocationDecision)) ||
            overlap(data, bytes, grants_, k * sizeof(Grant)) ||
            overlap(data, bytes, scratch_.keys_a, k * sizeof(std::uint64_t)) ||
            overlap(data, bytes, scratch_.keys_b, k * sizeof(std::uint64_t)) ||
            overlap(data, bytes, scratch_.radix_counts, 256 * sizeof(std::uint32_t)) ||
            overlap(data, bytes, owner_.initial_pins, k * sizeof(Grant)) ||
            overlap(data, bytes, owner_.slots, std::size_t{owner_.declaration.capacity()} * sizeof(Slot)) ||
            overlap(data, bytes, owner_.frame, sizeof(OwnerFrame)) ||
            overlap(data, bytes, &frame_, sizeof(AllocationFrame)) ||
            (owner_.declaration.capacity() && overlap(data, bytes, owner_.owner, sizeof(PhysicalOwnerControl)));
    }
    /** @brief Validate exact typed bank capacity, pointers and pairwise disjoint ownership in constant work. */
    QFN_ADAPTIVE_HD bool validStorage() const
    {
        if (!owner_.frame || !owner_.initial_pins || !owner_.table || !accepted_ || !grants_ ||
            !scratch_.keys_a || !scratch_.keys_b || !scratch_.radix_counts ||
            !capacity_ || capacity_ != owner_.declaration.maximumUnique() ||
            (owner_.declaration.capacity() && (!owner_.slots || !owner_.owner))) return false;
        const void *banks[]{accepted_, grants_, scratch_.keys_a, scratch_.keys_b,
            scratch_.radix_counts, owner_.initial_pins, owner_.slots, owner_.frame, &frame_, owner_.owner};
        const auto k = std::size_t{capacity_};
        const std::size_t bytes[]{k * sizeof(AllocationDecision), k * sizeof(Grant), k * 8, k * 8,
            256 * sizeof(std::uint32_t), k * sizeof(Grant), std::size_t{owner_.declaration.capacity()} * sizeof(Slot),
            sizeof(OwnerFrame), sizeof(AllocationFrame), owner_.declaration.capacity() ? sizeof(PhysicalOwnerControl) : 0};
        for (unsigned i = 0; i < 10; ++i)
            for (unsigned j = 0; j < i; ++j)
                if (overlap(banks[i], bytes[i], banks[j], bytes[j])) return false;
        return true;
    }
    /** @brief Validate one integer identity with admitted radix scratch; no slot/pin writes occur. */
    QFN_ADAPTIVE_HD bool uniqueKeys(const AllocationDecision *decisions, std::uint32_t count,
                                    unsigned key, unsigned bytes)
    {
        const auto pins = key == 2 ? 0u : owner_.frame->initial_pin_count + frame_.count;
        if (pins > owner_.declaration.maximumUnique() - count) return false;
        for (std::uint32_t i = 0; i < pins; ++i)
        {
            const auto &pin = i < owner_.frame->initial_pin_count ? owner_.initial_pins[i] :
                grants_[i - owner_.frame->initial_pin_count];
            if (!same(pin.interval, frame_.lease.interval) || (i < owner_.frame->initial_pin_count ? pin.kind != GrantKind::Read : pin.kind != GrantKind::Reservation) ||
                pin.owner != owner_.declaration.coordinate()) return false;
            scratch_.keys_a[i] = key == 0 ? pin.row : pin.unique;
        }
        if (!count && !pins) return true;
        for (std::uint32_t i = 0; i < count; ++i)
            scratch_.keys_a[pins + i] = key == 0 ? decisions[i].row : key == 1 ? decisions[i].unique :
                decisions[i].kind == CommandKind::ReserveVictim ? decisions[i].victim.slot : None;
        count += pins;
        auto *input = scratch_.keys_a;
        auto *output = scratch_.keys_b;
        for (unsigned pass = 0; pass < bytes; ++pass)
        {
            for (unsigned bin = 0; bin < 256; ++bin) scratch_.radix_counts[bin] = 0;
            const auto shift = pass * 8;
            for (std::uint32_t i = 0; i < count; ++i) ++scratch_.radix_counts[(input[i] >> shift) & 255];
            std::uint32_t cursor = 0;
            for (unsigned bin = 0; bin < 256; ++bin)
            {
                const auto size = scratch_.radix_counts[bin];
                scratch_.radix_counts[bin] = cursor;
                cursor += size;
            }
            for (std::uint32_t i = 0; i < count; ++i)
                output[scratch_.radix_counts[(input[i] >> shift) & 255]++] = input[i];
            auto *swap = input; input = output; output = swap;
        }
        for (std::uint32_t i = 1; i < count; ++i)
            if (input[i] == input[i - 1] && !(key == 2 && input[i] == None)) return false;
        return true;
    }
    /** @return Exact immutable interval equality, without a source/transport capability claim. */
    QFN_ADAPTIVE_HD static bool same(Interval a, Interval b)
    { return a.table == b.table && a.admission == b.admission && a.sequence == b.sequence; }
    /** @brief Poison the existing actual owner on malformed/stale protocol; no alternate path. */
    QFN_ADAPTIVE_HD bool fail()
    {
        if (owner_.frame && owner_.frame->status == Status::Success) owner_.frame->status = Status::Invariant;
        return false;
    }
    OwnerView<Slot> owner_;
    AllocationFrame &frame_;
    AllocationDecision *accepted_;
    Grant *grants_;
    AllocationValidationScratch scratch_;
    std::uint32_t capacity_;
};
}
