/**
 * @file PageTableAdmission.cpp
 * @brief Atomic pool+table admission implementation.
 *
 * Order: validate same authority, then acquire the pool grant, then the table
 * bank. If the table cannot supply a bank, the pool grant is rolled back. If
 * a Table bank is acquired but its later update fails, its RAII rollback
 * returns the bank. The pool grant rolls back through its own RAII on failure
 * paths; no usable partial state outlives this function.
 */
#include "PageTableAdmission.h"
#include "planning/AllResourceCompletionWitness.h"
#include <utility>
#include <vector>

namespace llaminar2
{
std::optional<PageTableAdmission> admitPoolAndTable(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table, std::size_t pages)
{
    if (!pages || pages > pool.geometry().max_pages_per_request)
        throw std::invalid_argument("Combined page/table admission requires positive pages within the pool bound");
    // Same-PMA authority: the table is attached to the same original instance.
    if (table.issuerForAdmission() != pool.issuerForAdmission())
        throw std::invalid_argument("Combined page/table admission requires one original PMA authority");
    auto grant = pool.tryAdmit(pages);
    if (!grant) return std::nullopt; // No growth; nothing else acquired.
    auto update = table.tryBeginUpdate();
    if (!update) return std::nullopt; // Grant RAII rolls back on scope exit.
    return PageTableAdmission{std::move(*grant), std::move(*update)};
}

std::optional<PageTableAdmission> admitPoolTableReserved(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table, std::size_t pages,
    CPUFP32PagedKVTable::IssuanceReservation &reservation)
{
    if (!pages || pages > pool.geometry().max_pages_per_request)
        throw std::invalid_argument("Reserved page/table admission requires positive pages within the pool bound");
    if (table.issuerForAdmission() != pool.issuerForAdmission())
        throw std::invalid_argument("Reserved page/table admission requires one original PMA authority");
    // Refuse BEFORE acquiring anything when the handle cannot authorize an
    // accepted update (foreign, moved-from or exhausted reservation).
    if (!reservation || !reservation.remaining())
        return std::nullopt;
    auto grant = pool.tryAdmit(pages);
    if (!grant) return std::nullopt; // No growth; the reservation is untouched.
    auto update = table.tryBeginReserved(reservation);
    if (!update) return std::nullopt; // Grant RAII rolls back; no capacity consumed.
    return PageTableAdmission{std::move(*grant), std::move(*update)};
}

namespace
{
/** @brief Prove the ALL-CLASS completion order before any acquisition. */
bool proveAllClassOrder(PhysicalPagePool &pool, CPUFP32PagedKVTable &table,
    PageTableAdmissionWorkspace &workspace,
    std::size_t declared_peak, std::size_t initial_grant)
{
    // B4: ZERO per-call allocation. The caller owns the membership-sized
    // workspace; this admission path never constructs a heap vector. The
    // workspace is REUSED across calls: refill by index, never append, so
    // repeated admissions cannot grow the caller-owned cells.
    auto &frame_workspace = workspace.frame_workspace;
    auto &members = workspace.members;
    members.clear(); // Never append across calls; the cells stay caller-owned.
    AllResourceCapacity capacity;
    capacity.frames = pool.geometry().frames;
    capacity.banks = table.bankCount();
    capacity.lease_cells = table.leaseCellCount();
    capacity.base = pool.physicalBaseOccupancyForWitness();
    AllResourceRequest proposed;
    proposed.generation = 0;
    proposed.peak_frames = declared_peak;
    proposed.occupied_frames = initial_grant;
    // B4: real class demand, never hard-coded. The proposed request declares
    // ONE admitted table bank and ONE effective-use lease at its peak (the
    // typed per-request contract of the owning table); existing live requests
    // are OBSERVED through liveClassDemand() (published banks + live lease
    // cells), so no class assumes a count that the authority does not hold.
    proposed.peak_banks = 1;   // Declared per-request table contract (one bank).
    proposed.peak_leases = 1;  // Declared per-request lease contract (one lease).
    // Existing live requests enter at their declared frame peaks (frames-only
    // pool records); the table's OWN observed live class demand is supplied
    // as the base, never fabricated per-request. C3: NO per-call allocation —
    // the pool fills the caller's witness cells by index (never a new vector),
    // and every write is checked against the membership capacity so the
    // caller-owned workspace can never grow on the heap.
    const std::size_t membership = frame_workspace.size();
    if (membership < pool.geometry().request_records + 1)
        return false; // Caller workspace too small: refuse, nothing acquired.
    pool.fillWitnessRequests(
        std::span<planning::WitnessRequest>(frame_workspace.data(), membership));
    for (std::size_t i = 0; i < membership; ++i)
    {
        const auto &r = frame_workspace[i];
        if (!r.peak_frames && !r.record && !r.occupied_frames) continue; // Unused cell.
        if (members.size() >= membership)
            return false; // Membership short for live+proposed: refuse, never grow.
        AllResourceRequest existing;
        existing.generation = r.generation;
        existing.record = r.record;
        existing.peak_frames = r.peak_frames;
        existing.occupied_frames = r.occupied_frames;
        existing.preserved = r.preserved;
        members.push_back(existing);
    }
    if (members.size() >= membership)
        return false; // No room for the proposed request either.
    members.push_back(proposed);
    capacity.requests = members;
    const auto live_demand = table.liveClassDemand();
    capacity.live_banks = live_demand.occupied_banks;
    capacity.live_leases = live_demand.live_leases;
    return proveAllResourceCompletion(capacity, frame_workspace).valid;
}
}

std::optional<PageTableAdmission> admitPoolTableWitnessed(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table,
    PageTableAdmissionWorkspace &workspace,
    std::size_t declared_peak, std::size_t initial_grant,
    CPUFP32PagedKVTable::IssuanceReservation &reservation)
{
    if (table.issuerForAdmission() != pool.issuerForAdmission())
        throw std::invalid_argument("Witnessed pool/table admission requires one original PMA authority");
    if (!reservation || !reservation.remaining())
        return std::nullopt;
    // ALL-CLASS proof precedes ANY acquisition: frames, table banks and
    // effective-use lease cells are proven together before the pool converts
    // the initial grant, the accepted-work handle authorizes the table bank,
    // or ANY reservation capacity is consumed. On refusal every class is
    // untouched (proof-first, not acquire-then-rollback).
    if (!proveAllClassOrder(pool, table, workspace, declared_peak, initial_grant))
        return std::nullopt;
    auto grant = pool.tryAdmitWithWitness(declared_peak, initial_grant);
    if (!grant) return std::nullopt; // No conversion: untouched.
    auto update = table.tryBeginReserved(reservation);
    if (!update) return std::nullopt; // Grant RAII rolls back; no issuance consumed.
    return PageTableAdmission{std::move(*grant), std::move(*update)};
}
} // namespace llaminar2
