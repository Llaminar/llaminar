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
    std::size_t declared_peak, std::size_t initial_grant)
{
    AllResourceCapacity capacity;
    capacity.frames = pool.geometry().frames;
    capacity.banks = table.bankCount();
    capacity.lease_cells = table.leaseCellCount();
    capacity.base = pool.physicalBaseOccupancyForWitness();
    AllResourceRequest proposed;
    proposed.generation = 0;
    proposed.peak_frames = declared_peak;
    proposed.occupied_frames = initial_grant;
    proposed.peak_banks = 1;   // One table bank per admitted request.
    proposed.peak_leases = 1;  // One effective-use lease per admitted request.
    // Existing live requests enter at their declared peaks (frames only are
    // tracked by the pool; the table's live bank/lease demand is admitted by
    // the table itself, so the composed proof counts the admitted maxima).
    auto live = pool.liveWitnessRequests();
    for (const auto &r : live)
    {
        AllResourceRequest existing;
        existing.generation = r.generation;
        existing.record = r.record;
        existing.peak_frames = r.peak_frames;
        existing.occupied_frames = r.occupied_frames;
        existing.peak_banks = 1;
        existing.peak_leases = 1;
        existing.preserved = r.preserved;
        capacity.requests.push_back(existing);
    }
    capacity.requests.push_back(proposed);
    return proveAllResourceCompletion(capacity).valid;
}
}

std::optional<PageTableAdmission> admitPoolTableWitnessed(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table,
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
    if (!proveAllClassOrder(pool, table, declared_peak, initial_grant))
        return std::nullopt;
    auto grant = pool.tryAdmitWithWitness(declared_peak, initial_grant);
    if (!grant) return std::nullopt; // No conversion: untouched.
    auto update = table.tryBeginReserved(reservation);
    if (!update) return std::nullopt; // Grant RAII rolls back; no issuance consumed.
    return PageTableAdmission{std::move(*grant), std::move(*update)};
}
} // namespace llaminar2
