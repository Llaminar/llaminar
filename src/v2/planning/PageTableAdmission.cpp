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

std::optional<PageTableAdmission> admitPoolTableWitnessed(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table,
    std::size_t declared_peak, std::size_t initial_grant,
    CPUFP32PagedKVTable::IssuanceReservation &reservation)
{
    if (table.issuerForAdmission() != pool.issuerForAdmission())
        throw std::invalid_argument("Witnessed pool/table admission requires one original PMA authority");
    // The completion witness is proved BEFORE ANY ACQUISITION by the pool's
    // own witness admission (proof precedes mutation; a refused invocation
    // leaves every class untouched, including the reservation handle). Only
    // when the proof holds does the pool convert the initial grant, and only
    // then does the accepted-work handle authorize the table bank. If the
    // table cannot supply the bank, the grant's RAII rolls back and the
    // reservation capacity is NOT consumed.
    if (!reservation || !reservation.remaining())
        return std::nullopt;
    auto grant = pool.tryAdmitWithWitness(declared_peak, initial_grant);
    if (!grant) return std::nullopt; // No completion order or no growth: untouched.
    auto update = table.tryBeginReserved(reservation);
    if (!update) return std::nullopt; // Grant RAII rolls back; no issuance consumed.
    return PageTableAdmission{std::move(*grant), std::move(*update)};
}
} // namespace llaminar2
