/**
 * @file PooledImmutablePhysicalOwnerOps.h
 * @brief Allocation-free physical-owner grant parser shared by CPU, CUDA and HIP.
 *
 * One owner writer executes decoded commands in explicit graph phase order.
 * No command chooses a policy victim, accesses another device, performs source
 * I/O or publishes a coordinator locator. Transport authentication, actual
 * source ACK validation and last-reader event edges are binding obligations,
 * deliberately not replaced by a caller-created capability in this component.
 */
#pragma once
#include "PooledImmutablePhysicalOwnerState.h"

namespace llaminar2::qwen4exp::adaptive_lookup::pooled
{
/** @brief Exact owner-local lifetime operations over canonical physical slots. */
template <class Slot> class OwnerOperations final
{
public:
    /** @brief Borrow existing owner arrays; storage and current device/control belong to the future stage. */
    QFN_ADAPTIVE_HD explicit OwnerOperations(OwnerView<Slot> view) : v_(view) {}

    /**
     * @brief Initialize a virgin slot once at setup, using the single-GPU physical leaf operation.
     * @param slot Exact local slot coordinate; invalid geometry/coordinates and post-setup calls do nothing.
     */
    QFN_ADAPTIVE_HD void initializeSlot(std::uint32_t slot)
    {
        if (validGeometry() && v_.frame->phase == OwnerPhase::Uninitialized &&
            v_.frame->status == Status::Success && (!v_.owner || !v_.owner->flags) &&
            slot < v_.declaration.capacity())
            PhysicalSlot<Slot>::initialize(v_.slots[slot]);
    }

    /**
     * @brief Initialize mandatory frame and optional physical authority once.
     * @return False without touching slots for invalid or repeated setup.
     * @note Caller supplies zero-initialized frame/control roots and separately initialized slots.
     */
    QFN_ADAPTIVE_HD bool initialize()
    {
        if (!v_.frame || v_.frame->phase != OwnerPhase::Uninitialized ||
            v_.frame->status != Status::Success || !validGeometry())
            return false;
        if (v_.declaration.capacity())
        {
            if (v_.owner->flags)
                return false;
            *v_.owner = PhysicalOwnerControl{};
            v_.owner->slot_free = None;
            v_.owner->flags = OwnerInitialized;
        }
        *v_.frame = OwnerFrame{};
        v_.frame->phase = OwnerPhase::Idle;
        return true;
    }

    /**
     * @brief Adopt actual admission/transport identities without minting a second sequence.
     * @param current Original controller/transport projection supplied by the authenticated stage.
     * @return True only for a healthy Idle owner and strictly newer canonical frame identity.
     * @note Reset calls begin; it never initializes slots or rewinds slot generations.
     */
    QFN_ADAPTIVE_HD bool begin(Interval current)
    {
        if (!validGeometry() || v_.frame->phase != OwnerPhase::Idle ||
            (v_.frame->status != Status::Success && v_.frame->status != Status::Cancelled) ||
            v_.frame->held_pins || v_.frame->reservations || !validInterval(current) ||
            (v_.declaration.capacity() && (v_.owner->flags != OwnerInitialized ||
                             v_.owner->filled > v_.declaration.capacity() || v_.owner->recycled_free > v_.owner->filled ||
                             ((v_.owner->slot_free == None) != (v_.owner->recycled_free == 0)))))
            return reject(Status::InvalidPhase);
        const auto old = v_.frame->interval;
        if (old.table && (current.admission < old.admission ||
                          current.sequence <= old.sequence))
            return reject(Status::InvalidPhase);
        *v_.frame = OwnerFrame{};
        v_.frame->interval = current;
        v_.frame->phase = OwnerPhase::Pinning;
        return true;
    }

    /**
     * @brief Pin one unique initial hit after checking the actual owner row and full generation.
     * @param current Actual live admission/frame, reread by the eventual device wrapper at each operation.
     * @param locator Sole coordinator directory's immutable row/owner/slot/generation projection.
     * @param unique Global deduplicated row ordinal within the admitted frame bound.
     * @param ordinal Strictly increasing actual device dispatch order, not the source coordinate.
     * @return Read grant or Absent after recording the first invalid command.
     */
    QFN_ADAPTIVE_HD Grant pinInitial(Interval current, Locator locator, std::uint32_t unique, CommandOrdinal ordinal)
    {
        if (!active(current, OwnerPhase::Pinning) || !validRow(locator.row, unique) ||
            !validLocator(locator) || v_.frame->held_pins >= v_.declaration.maximumUnique() ||
            v_.frame->initial_pin_count >= v_.declaration.maximumUnique() ||
            !newCommand(ordinal))
        {
            reject(Status::Invariant);
            return absent(current, locator.row, unique);
        }
        auto &slot = v_.slots[locator.slot];
        if (PhysicalSlot<Slot>::state(slot) != SlotState::Ready || PhysicalSlot<Slot>::pinned(slot))
        {
            reject(Status::Invariant);
            return absent(current, locator.row, unique);
        }
        if (!v_.initial_pins) { reject(Status::InvalidGeometry); return absent(current, locator.row, unique); }
        PhysicalSlot<Slot>::pin(slot);
        slot.touched = std::uint64_t{unique} + 1;
        ++v_.frame->held_pins;
        const auto result = grant(current, locator.row, unique, locator.slot, GrantKind::Read);
        v_.initial_pins[v_.frame->initial_pin_count++] = result;
        return result;
    }

    /**
     * @brief Seal all local initial pins before emitting the one acquired cohort header.
     * @param current Actual original admission and frame identity.
     * @return Header for this owner, with the first failure retained on rejection.
     */
    QFN_ADAPTIVE_HD PinHeader sealPins(Interval current)
    {
        if (!active(current, OwnerPhase::Pinning))
            reject(Status::InvalidPhase);
        else
            v_.frame->phase = OwnerPhase::PinsSealed;
        return {current, v_.declaration.coordinate(), v_.frame ? v_.frame->status : Status::InvalidGeometry};
    }

    /**
     * @brief Validate the entire actual ordered cohort header bank before enabling any replacement.
     * @param current Actual interval, matching all original producer headers.
     * @param headers Acquired P headers in declared communicator-coordinate order.
     * @param count Exact P; capacity-sized/padded header banks are not accepted.
     * @return True after complete prevalidation; no prefix of an invalid bank changes phase or pins.
     */
    QFN_ADAPTIVE_HD bool acceptPinFrontier(Interval current, const PinHeader *headers, std::uint32_t count)
    {
        if (!active(current, OwnerPhase::PinsSealed) || !headers || count != v_.declaration.participants())
            return reject(Status::InvalidPhase);
        for (std::uint32_t owner = 0; owner < count; ++owner)
            if (!same(headers[owner].interval, current) || headers[owner].owner != owner ||
                headers[owner].status != Status::Success)
                return reject(Status::Invariant);
        v_.frame->phase = OwnerPhase::Reserving;
        return true;
    }

    /**
     * @brief Reserve an actual owner free/virgin slot in O(1), or decline normally when none exists.
     * @param current Actual live admission/frame, unchanged through this command.
     * @param row Complete original source row, eligible for every physical owner.
     * @param unique Original source unique coordinate, independent of dispatch order.
     * @param ordinal Strictly increasing actual device command order, including declined probes.
     * @return Full physical grant or ordinary Absent on no free capacity; invalid commands record failure.
     * @note This is the leaf executor for an already accepted immutable free allocation plan.
     * Serving bindings use OwnerPlanOperations availability leases; they must never probe
     * owners with per-miss collectives. No source row has a hash/head home quota.
     */
    QFN_ADAPTIVE_HD Grant reserveFree(Interval current, std::uint64_t row, std::uint32_t unique, CommandOrdinal ordinal)
    {
        if (!active(current, OwnerPhase::Reserving) || !validRow(row, unique) || !newCommand(ordinal))
        {
            reject(Status::InvalidPhase);
            return absent(current, row, unique);
        }
        if (!v_.declaration.capacity())
            return absent(current, row, unique);
        auto &owner = *v_.owner;
        const bool recycled = owner.slot_free != None;
        const auto index = recycled ? owner.slot_free : owner.filled;
        if (!recycled && index == v_.declaration.capacity())
            return absent(current, row, unique);
        if (index >= v_.declaration.capacity() || !canReserve() ||
            PhysicalSlot<Slot>::state(v_.slots[index]) != SlotState::Empty ||
            PhysicalSlot<Slot>::pinned(v_.slots[index]) ||
            v_.slots[index].flags != (static_cast<std::uint32_t>(Stream::None) << 2) ||
            v_.slots[index].next != None ||
            (!recycled && v_.slots[index].previous != None) ||
            (recycled && (v_.slots[index].previous == index ||
                          (v_.slots[index].previous != None && v_.slots[index].previous >= owner.filled))))
        {
            reject(Status::Invariant);
            return absent(current, row, unique);
        }
        if (recycled)
        {
            if (!owner.recycled_free) { reject(Status::Invariant); return absent(current, row, unique); }
            --owner.recycled_free;
            owner.slot_free = v_.slots[index].previous;
        }
        else
            ++owner.filled;
        return reserve(current, row, unique, index);
    }

    /**
     * @brief Apply the sole coordinator policy's explicit victim decision at its physical owner.
     * @param current Actual live admission/frame.
     * @param victim Expected old Ready locator; its actual row/generation/pin must still match.
     * @param row New complete original row selected by the sole coordinator policy.
     * @param unique Original source unique coordinate; never a dispatch clock.
     * @param ordinal Strictly increasing actual device command order.
     * @return New full-generation reservation; stale/pinned/foreign commands fail without replacing backing.
     * @note No local frequency/queue algorithm exists. All-pinned rejection changes no slot.
     */
    QFN_ADAPTIVE_HD Grant reserveVictim(Interval current, Locator victim,
                                      std::uint64_t row, std::uint32_t unique, CommandOrdinal ordinal)
    {
        if (!active(current, OwnerPhase::Reserving) || !validRow(row, unique) ||
            !validLocator(victim) || !newCommand(ordinal) || !canReserve())
        {
            reject(Status::Invariant);
            return absent(current, row, unique);
        }
        const auto &slot = v_.slots[victim.slot];
        if (PhysicalSlot<Slot>::state(slot) != SlotState::Ready || PhysicalSlot<Slot>::pinned(slot))
        {
            reject(Status::Invariant);
            return absent(current, row, unique);
        }
        return reserve(current, row, unique, victim.slot);
    }

    /**
     * @brief Freeze reservations and adopt the real source window, before payload/fill graph operations.
     * @param current Actual live admission/frame.
     * @param source Exact current source window; all zero means no source transaction was published.
     * @return True after freezing the local reservation frontier; malformed identities fail before phase changes.
     * @note Reservations require a nonzero matching table/window; source-only C0 work may also have one.
     */
    QFN_ADAPTIVE_HD bool sealReservations(Interval current, SourceWindow source)
    {
        if ((!active(current, OwnerPhase::Reserving) && !active(current, OwnerPhase::PlanReady)) ||
            (source.table ? !validSource(source) : source.generation || source.sequence) ||
            (v_.frame->reservations && !source.table))
            return reject(Status::InvalidPhase);
        v_.frame->source = source;
        v_.frame->phase = OwnerPhase::Filling;
        return true;
    }

    /**
     * @brief Validate the complete private local fill/ACK bank before emitting owner success.
     * @param current Actual live admission/frame.
     * @param grants Exactly the active local reservations in increasing original-unique order.
     * @param acknowledgements Private copied ACKs from the authenticated source response producer.
     * @param count Actual local reservation count, zero for empty owners.
     * @return Original owner header; no phase or slot prefix mutates when any record rejects.
     * @note Slot.touched is the physical-only grant coordinate, not a second replacement clock.
     */
    QFN_ADAPTIVE_HD PublishHeader validateFills(Interval current, const Grant *grants,
                                               const FillAcknowledgement *acknowledgements,
                                               std::uint32_t count)
    { return validateFillsImpl(current, grants, acknowledgements, nullptr, count); }

    /**
     * @brief Validate the same physical fills through a retained immutable grant/ACK permutation.
     * @param current Actual live admission/frame.
     * @param fills Exact live bounded view in increasing original global unique order.
     * @return Owner header after complete validation; a malformed permutation changes no slot or phase prefix.
     */
    QFN_ADAPTIVE_HD PublishHeader validateFills(Interval current, OrderedReservationFills fills)
    {
        if (fills.count && !fills.order)
        {
            reject(Status::Invariant);
            return {current, sourceWindow(), v_.declaration.coordinate(), v_.frame ? v_.frame->status : Status::InvalidGeometry};
        }
        return validateFillsImpl(current, fills.grants, fills.acknowledgements, fills.order, fills.count);
    }

    /**
     * @brief Apply acquired all-owner success consensus after fill/ACK validation.
     * @param current Actual live admission/frame.
     * @param headers Exactly P immutable acquired headers, never fabricated from local health alone.
     * @param count Exact complete declared cohort size, including empty owners.
     * @return True only after the whole bank agrees; failure changes no slot or phase prefix.
     * @note This is a wire parser. Its future graph must authenticate header producers, source ACK
     * copies and exact copy-before-publication edges; this method issues no source capability.
     */
    QFN_ADAPTIVE_HD bool acceptPublishFrontier(Interval current, const PublishHeader *headers,
                                              std::uint32_t count)
    {
        if (!active(current, OwnerPhase::FillValidated) || !headers || count != v_.declaration.participants())
            return reject(Status::InvalidPhase);
        for (std::uint32_t owner = 0; owner < count; ++owner)
            if (!same(headers[owner].interval, current) ||
                !same(headers[owner].source, v_.frame->source) ||
                headers[owner].owner != owner || headers[owner].status != Status::Success)
                return reject(Status::Invariant);
        v_.frame->phase = OwnerPhase::Publishable;
        return true;
    }

    /**
     * @brief Publish one filled reservation only after revalidating live admission and full slot identity.
     * @param current Actual original admission/frame reread immediately before Ready publication.
     * @param g Owner-issued reservation with complete row/owner/slot/generation/ordinal identity.
     * @param ack Private source-validated ACK record, not an unconsumed mapped response span.
     * @return False on cancellation, late/foreign ACK, stale slot or failed owner, without Ready mutation.
     */
    QFN_ADAPTIVE_HD bool makeReady(Interval current, const Grant &g, FillAcknowledgement ack)
    {
        if (!active(current, OwnerPhase::Publishable) || !validGrant(g, GrantKind::Reservation) ||
            !same(ack.source, v_.frame->source) || ack.destination_token != g.destination_token ||
            !validSource(ack.source) || !v_.frame->reservations)
            return reject(Status::Invariant);
        auto &slot = v_.slots[g.slot];
        if (PhysicalSlot<Slot>::state(slot) != SlotState::Reserved || !PhysicalSlot<Slot>::pinned(slot))
            return reject(Status::Invariant);
        PhysicalSlot<Slot>::ready(slot);
        --v_.frame->reservations;
        return true;
    }

    /**
     * @brief Authenticate one parallel reader against actual immutable owner bytes and its live pin.
     * @param current Exact original active frame, acquired after all-owner Ready publication.
     * @param grant Original owner-issued read/reservation record, never a shadow occupancy claim.
     * @return True only while the matching original slot is Ready and its exact pin remains live.
     * @note This read-only operation neither changes policy nor invents another state writer.
     */
    QFN_ADAPTIVE_HD bool readable(Interval current,const Grant& grant) const
    {
        return active(current,OwnerPhase::Publishable) &&
            (grant.kind==GrantKind::Read || grant.kind==GrantKind::Reservation) && validGrant(grant,grant.kind) &&
            PhysicalSlot<Slot>::state(v_.slots[grant.slot])==SlotState::Ready && PhysicalSlot<Slot>::pinned(v_.slots[grant.slot]);
    }

    /**
     * @brief Stop new publication; preserve already-Ready bytes only after exact source retirement.
     * @param current Actual active admission/frame; foreign or already-retired intervals reject.
     * @param retirement Unpublished before source issue, or Consumed after the real matching ACK.
     * @param source Actual retired source window; Unknown/mismatch poisons rather than guessing.
     * @return True only when the source window is proven drained; uncertainty is sticky storage failure.
     * @note Every issued unique grant still needs release after its last native reader.
     */
    QFN_ADAPTIVE_HD bool cancel(Interval current, SourceRetirement retirement, SourceWindow source = {})
    {
        if (!validGeometry() || !same(current, v_.frame->interval) || v_.frame->phase == OwnerPhase::Idle ||
            v_.frame->phase == OwnerPhase::Uninitialized)
            return reject(Status::InvalidPhase);
        const bool drained = sourceRetired(retirement, source);
        if (!drained)
            reject(Status::SourceFailure);
        else if (v_.frame->status == Status::Success)
            v_.frame->status = Status::Cancelled;
        v_.frame->phase = OwnerPhase::Draining;
        return drained;
    }

    /**
     * @brief Retire one grant after final native readers, recycling only matching unreadable reservations.
     * @param current Actual current interval; the caller owns the exact last-reader event edge.
     * @param g Original immutable owner reply, not a mutable coordinator pin-state mirror.
     * @return True after one actual pin retires; stale/foreign/repeated releases fail before slot writes.
     * @note Cleanup is allowed after sticky failure; a stale/foreign grant cannot unpin a newer occupant.
     */
    QFN_ADAPTIVE_HD bool release(Interval current, const Grant &g)
    {
        if (!validGeometry() || !same(current, v_.frame->interval) || v_.frame->phase == OwnerPhase::Idle ||
            (v_.frame->phase != OwnerPhase::Publishable && v_.frame->phase != OwnerPhase::Draining) || !v_.frame->held_pins ||
            (g.kind != GrantKind::Read && g.kind != GrantKind::Reservation) || !validGrant(g, g.kind))
            return reject(Status::Invariant);
        auto &slot = v_.slots[g.slot];
        if (!PhysicalSlot<Slot>::pinned(slot))
            return reject(Status::Invariant);
        if (PhysicalSlot<Slot>::state(slot) == SlotState::Reserved)
        {
            if (g.kind != GrantKind::Reservation || !v_.frame->reservations)
                return reject(Status::Invariant);
            PhysicalSlot<Slot>::recycle(slot, v_.owner->slot_free);
            v_.owner->slot_free = g.slot;
            ++v_.owner->recycled_free;
            --v_.frame->reservations;
        }
        else if (PhysicalSlot<Slot>::state(slot) == SlotState::Ready)
            PhysicalSlot<Slot>::unpin(slot);
        else
            return reject(Status::Invariant);
        --v_.frame->held_pins;
        return true;
    }

    /**
     * @brief Release an original grant and report the actual physical retention disposition.
     * @param current Actual original frame whose final-reader dependency was acquired.
     * @param g Exact immutable owner grant, including original source unique/token.
     * @return RetainedReady or RecycledEmpty only after actual checked release succeeded.
     * @note This receipt closes cancellation between owner Ready and directory publication.
     * It is transient protocol evidence, not a coordinator-owned slot-state mirror.
     */
    QFN_ADAPTIVE_HD ReleaseReceipt releaseWithReceipt(Interval current, const Grant &g)
    {
        if (!validGeometry() || (g.kind != GrantKind::Read && g.kind != GrantKind::Reservation) ||
            !validGrant(g, g.kind))
        {
            reject(Status::Invariant);
            return {g, ReleaseDisposition::Failed, 0};
        }
        const auto before = PhysicalSlot<Slot>::state(v_.slots[g.slot]);
        if (!release(current, g))
            return {g, ReleaseDisposition::Failed, 0};
        return {g, before == SlotState::Ready ? ReleaseDisposition::RetainedReady :
                                              ReleaseDisposition::RecycledEmpty, 0};
    }

    /**
     * @brief Retire the actual frame only after readers and any issued source response are drained.
     * @param current Actual current interval, retained in Idle only for late-command rejection.
     * @param retirement Exact source-window retirement kind, never a guess from controller health.
     * @param source Actual consumed source identity, or all-zero for unpublished source work.
     * @return True for successful retirement or proven clean cancellation; sticky owner failure remains.
     * @note Kept Ready slots/generations survive reset. Last interval identity remains to reject replay.
     */
    QFN_ADAPTIVE_HD bool retire(Interval current, SourceRetirement retirement, SourceWindow source = {})
    {
        if (!validGeometry() || !same(current, v_.frame->interval) || v_.frame->held_pins || v_.frame->reservations ||
            (v_.frame->phase != OwnerPhase::Publishable && v_.frame->phase != OwnerPhase::Draining) ||
            !sourceRetired(retirement, source) ||
            (retirement==SourceRetirement::Aborted &&
                (v_.frame->phase!=OwnerPhase::Draining || v_.frame->status!=Status::Cancelled)))
            return reject(Status::InvalidPhase);
        v_.frame->phase = OwnerPhase::Idle;
        return v_.frame->status == Status::Success || v_.frame->status == Status::Cancelled;
    }

    /** @return Actual local publication completion, or a failure header without inventing success. */
    QFN_ADAPTIVE_HD ReadyHeader readyHeader(Interval current)
    {
        if (!v_.frame) return {current, {}, v_.declaration.coordinate(), Status::InvalidGeometry};
        if (!active(current, OwnerPhase::Publishable) || v_.frame->reservations)
            reject(Status::InvalidPhase);
        return {current, v_.frame->source, v_.declaration.coordinate(), v_.frame->status};
    }
    /** @return Actual joined owner-frame retirement; readers must already have released their grants. */
    QFN_ADAPTIVE_HD RetiredHeader retiredHeader(Interval current)
    {
        if (!v_.frame) return {current, {}, v_.declaration.coordinate(), Status::InvalidGeometry};
        if (!validGeometry() || !same(current, v_.frame->interval) || v_.frame->phase != OwnerPhase::Idle ||
            v_.frame->held_pins || v_.frame->reservations)
            reject(Status::InvalidPhase);
        return {current, v_.frame->source, v_.declaration.coordinate(), v_.frame->status};
    }

private:
    /** @brief Validate live physical identities before the one FillValidated phase commit, with optional ordered indirection. */
    QFN_ADAPTIVE_HD PublishHeader validateFillsImpl(Interval current, const Grant *grants,
        const FillAcknowledgement *acknowledgements, const std::uint64_t *order, std::uint32_t count)
    {
        if (!active(current, OwnerPhase::Filling) || count != v_.frame->reservations ||
            count > v_.declaration.maximumUnique() || (count && (!grants || !acknowledgements)))
            reject(Status::InvalidPhase);
        else
        {
            bool valid = true;
            std::uint32_t previous = 0;
            for (std::uint32_t i = 0; i < count; ++i)
            {
                const auto at = order ? static_cast<std::uint32_t>(order[i]) : i;
                if (at >= count) { valid = false; break; }
                const auto &g = grants[at];
                const auto &ack = acknowledgements[at];
                if ((order && order[i] >> 32 != g.unique) || !validGrant(g, GrantKind::Reservation) ||
                    PhysicalSlot<Slot>::state(v_.slots[g.slot]) != SlotState::Reserved ||
                    !PhysicalSlot<Slot>::pinned(v_.slots[g.slot]) ||
                    !same(ack.source, v_.frame->source) || ack.destination_token != g.destination_token ||
                    (i && g.unique <= previous))
                { valid = false; break; }
                previous = g.unique;
            }
            if (valid) v_.frame->phase = OwnerPhase::FillValidated;
            else reject(Status::Invariant);
        }
        return {current, sourceWindow(), v_.declaration.coordinate(), v_.frame ? v_.frame->status : Status::InvalidGeometry};
    }
    OwnerView<Slot> v_;

    /** @return Whole original identity equality without truncation or payload inspection. */
    QFN_ADAPTIVE_HD static bool same(Interval a, Interval b)
    {
        return a.table == b.table && a.admission == b.admission && a.sequence == b.sequence;
    }
    /** @return Exact source transaction identity, including source-free zero. */
    QFN_ADAPTIVE_HD static bool same(SourceWindow a, SourceWindow b)
    {
        return a.table == b.table && a.generation == b.generation && a.sequence == b.sequence;
    }
    /** @return Checked physical declaration, including mandatory C0 frame and complete row width. */
    QFN_ADAPTIVE_HD bool validGeometry() const
    {
        return v_.frame && v_.table && v_.declaration.sourceRows() && v_.declaration.maximumUnique() &&
               v_.declaration.maximumUnique() <= MaximumSlots && v_.declaration.capacity() <= MaximumSlots &&
               (v_.declaration.participants() >= 1 && v_.declaration.participants() <= MaximumSlots) &&
               v_.declaration.coordinate() < v_.declaration.participants() &&
               ((sizeof(v_.slots[0].row) == 4) == (v_.declaration.rowWidth() == RowKeyWidth::UInt32)) &&
               (sizeof(v_.slots[0].row) == 8 || v_.declaration.sourceRows() <= 0x100000000ULL) &&
               (!v_.declaration.capacity() || (v_.slots && v_.owner));
    }
    /** @return Nonzero source-authored table and actual interval coordinates. */
    QFN_ADAPTIVE_HD bool validInterval(Interval value) const
    {
        return value.table == v_.table && value.admission && value.sequence;
    }
    /** @return Complete actual source window, without inventing a sequence or admission owner. */
    QFN_ADAPTIVE_HD bool validSource(SourceWindow value) const
    {
        return value.table == v_.table && value.generation && value.sequence;
    }
    /** @return Healthy matching live phase; wrapper must additionally bind the actual original controller. */
    QFN_ADAPTIVE_HD bool active(Interval current, OwnerPhase phase) const
    {
        return validGeometry() && same(current, v_.frame->interval) && validInterval(current) &&
               v_.frame->phase == phase && v_.frame->status == Status::Success &&
               v_.frame->held_pins <= v_.declaration.capacity() && v_.frame->reservations <= v_.frame->held_pins &&
               (!v_.declaration.capacity() || (v_.owner->flags == OwnerInitialized &&
                                  v_.owner->filled <= v_.declaration.capacity() &&
                                  v_.frame->held_pins <= v_.owner->filled &&
                                  v_.frame->reservations <= v_.frame->held_pins));
    }
    /** @return Whole original row and admitted global unique coordinate bounds. */
    QFN_ADAPTIVE_HD bool validRow(std::uint64_t row, std::uint32_t unique) const
    {
        return row < v_.declaration.sourceRows() && unique < v_.declaration.maximumUnique();
    }
    /** @return Actual owner metadata matches complete expected locator identity. */
    QFN_ADAPTIVE_HD bool validLocator(Locator locator) const
    {
        if (locator.owner != v_.declaration.coordinate() || locator.slot >= v_.declaration.capacity() ||
            !v_.owner || locator.slot >= v_.owner->filled ||
            locator.row >= v_.declaration.sourceRows() || !locator.generation)
            return false;
        const auto &slot = v_.slots[locator.slot];
        return static_cast<std::uint64_t>(slot.row) == locator.row &&
               slot.generation == locator.generation && !(slot.flags & ~KnownSlotFlags) &&
               (slot.flags & StreamMask) == (static_cast<std::uint32_t>(Stream::None) << 2);
    }
    /** @return Matching bounded grant, with globally unique source token independent of local generation. */
    QFN_ADAPTIVE_HD bool validGrant(const Grant &g, GrantKind kind) const
    {
        return g.kind == kind && same(g.interval, v_.frame->interval) &&
               validRow(g.row, g.unique) && validLocator({g.row, g.generation, g.owner, g.slot}) &&
               v_.slots[g.slot].touched == std::uint64_t{g.unique} + 1 &&
               g.destination_token == (kind == GrantKind::Reservation ? std::uint64_t{g.unique} + 1 : 0);
    }
    /** @return Strict device dispatch order; original unique IDs may arrive out of order. */
    QFN_ADAPTIVE_HD bool newCommand(CommandOrdinal ordinal)
    {
        if (!ordinal.value || ordinal.value <= v_.frame->last_command)
            return false;
        v_.frame->last_command = ordinal.value;
        return true;
    }
    /** @return Generation and frame pin budgets before mutating free chain or replacing native bytes. */
    QFN_ADAPTIVE_HD bool canReserve()
    {
        if (v_.owner->next_slot_generation == ~std::uint64_t{0} ||
            v_.frame->held_pins >= v_.declaration.maximumUnique() || v_.frame->reservations >= v_.declaration.maximumUnique())
            return reject(Status::Exhausted);
        return true;
    }
    /** @return Bounded ordinary decline response; it carries no physical pin or generation. */
    QFN_ADAPTIVE_HD Grant absent(Interval current, std::uint64_t row, std::uint32_t unique) const
    {
        return {current, row, 0, 0, unique, v_.declaration.coordinate(), None, GrantKind::Absent};
    }
    /** @return Immutable reply derived from the actual current owner slot. */
    QFN_ADAPTIVE_HD Grant grant(Interval current, std::uint64_t row, std::uint32_t unique,
                                std::uint32_t slot, GrantKind kind) const
    {
        return {current, row, v_.slots[slot].generation,
                kind == GrantKind::Reservation ? std::uint64_t{unique} + 1 : 0,
                unique, v_.declaration.coordinate(), slot, kind};
    }
    /** @brief Issue generation only at its physical owner; shared leaf writes preserve exact native state. */
    QFN_ADAPTIVE_HD Grant reserve(Interval current, std::uint64_t row, std::uint32_t unique,
                                  std::uint32_t index)
    {
        const auto generation = ++v_.owner->next_slot_generation;
        PhysicalSlot<Slot>::reserve(v_.slots[index], row, generation, std::uint64_t{unique} + 1);
        // Physical-only records have no coordinator policy-region membership.
        v_.slots[index].flags |= static_cast<std::uint32_t>(Stream::None) << 2;
        ++v_.frame->held_pins;
        ++v_.frame->reservations;
        return grant(current, row, unique, index, GrantKind::Reservation);
    }
    /** @return Current bounded expected source transaction; not an independent source sequence owner. */
    QFN_ADAPTIVE_HD SourceWindow sourceWindow() const
    {
        return v_.frame ? v_.frame->source : SourceWindow{};
    }
    /** @return Source window is proven absent or exactly consumed; unknown retirement never passes. */
    QFN_ADAPTIVE_HD bool sourceRetired(SourceRetirement retirement, SourceWindow source) const
    {
        return v_.frame->source.table
                   ? (retirement == SourceRetirement::Consumed || retirement == SourceRetirement::Aborted) && same(source, v_.frame->source)
                   : retirement == SourceRetirement::Unpublished && !source.table &&
                         !source.generation && !source.sequence;
    }
    /** @brief Preserve first error and poison only storage/invariant failure, never proven cancellation. */
    QFN_ADAPTIVE_HD bool reject(Status status)
    {
        if (v_.frame && v_.frame->status == Status::Success)
            v_.frame->status = status;
        if (v_.declaration.capacity() && v_.owner && status != Status::Cancelled)
            v_.owner->flags |= OwnerFailed;
        return false;
    }
};
} // namespace llaminar2::qwen4exp::adaptive_lookup::pooled
