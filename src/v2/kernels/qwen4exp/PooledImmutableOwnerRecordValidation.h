/**
 * @file PooledImmutableOwnerRecordValidation.h
 * @brief Immutable per-record predicates for cooperative original owner validation.
 *
 * These predicates publish no status and mutate no owner or physical slot. The
 * original lane-zero phase/source gates and ordered commit remain authoritative.
 * CPU scalar owner operations deliberately retain their independent predicates.
 */
#pragma once
#include "PooledImmutableNativeOps.h"
namespace llaminar2::qwen4exp::pooled_native::record_validation {
/** @brief Validate one original Pin command before any slot mutation.
 * @tparam Slot Original physical slot encoding.
 * @param a Authenticated original invocation and physical owner.
 * @param command One command inside its previously checked live span.
 * @param priorOrdinal Original preceding command ordinal, or frame.last_command for the first.
 * @return Whether all immutable fields and the original slot identity agree.
 * Slot bounds are checked before dereferencing the physical slot array.
 */
template<class Slot> QFN_ADAPTIVE_HD bool pin(const OwnerArgs& a,const p::OwnerCommand& command,std::uint64_t priorOrdinal){
    const auto& grant=command.target;
    if(command.kind!=p::CommandKind::PinInitial || command.reserved ||
       !same(grant.interval,a.input.invocation->interval) || grant.kind!=p::GrantKind::Read ||
       grant.destination_token || grant.unique>=a.input.invocation->unique_rows ||
       grant.row>=a.plan.owner.sourceRows() || grant.owner!=a.plan.owner.coordinate() ||
       grant.slot>=a.plan.owner.capacity() || !grant.generation || command.requested_row!=grant.row ||
       command.ordinal.value<=priorOrdinal)return false;
    const auto view=ownerView<Slot>(a);
    const auto& slot=view.slots[grant.slot];
    return slot.row==grant.row && slot.generation==grant.generation &&
        adaptive_lookup::PhysicalSlot<Slot>::state(slot)==adaptive_lookup::SlotState::Ready &&
        !adaptive_lookup::PhysicalSlot<Slot>::pinned(slot);
}
/** @brief Validate retained initial-pin or accepted-reservation identity in each key domain.
 * @param grant One original retained grant inside its authenticated live prefix.
 * @param interval Original allocation-frame lease interval.
 * @param kind Read for initial pins, Reservation for prior accepted grants.
 * @param owner Exact original owner coordinate.
 * @return Whether all original retained-grant fields checked by the scalar projection agree.
 */
QFN_ADAPTIVE_HD inline bool retained(const p::Grant& grant,p::Interval interval,p::GrantKind kind,std::uint32_t owner){
    return same(grant.interval,interval) && grant.kind==kind && grant.owner==owner;
}
}
