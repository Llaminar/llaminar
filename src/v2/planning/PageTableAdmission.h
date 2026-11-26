/**
 * @file PageTableAdmission.h
 * @brief All-class page-authority admission: pool growth plus a table bank as one transaction.
 *
 * Plan P1 requires all-resource prepare/refusal: an admitted transaction may
 * need, at the same instant, protected pool growth (G) and a free table bank
 * (metadata). This helper obtains both atomically on one PMA authority and
 * rolls everything back if any class cannot be acquired, so refusal never
 * leaves a usable table reservation or a partial growth grant behind.
 * Issued versions/nonces are never restored (plan P1 identity rule).
 */
#pragma once
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "planning/PhysicalPagePool.h"
#include <optional>

namespace llaminar2
{
/** @brief One atomic all-class admission result; both cells valid together or neither. */
struct PageTableAdmission
{
    PhysicalPagePool::Grant grant;                     ///< Protected pool growth (G).
    CPUFP32PagedKVTable::Update table_update;          ///< Reserved table bank.
    bool operator==(const PageTableAdmission &) const = delete;
};

/**
 * @brief Admit protected pool growth and one table bank atomically.
 * @param pool Original PMA page authority (same authority as the table).
 * @param table Original table owner; its authority must equal the pool's.
 * @param pages Protected pool growth requested.
 * @return Both cells on success; an empty optional refuses before any usable
 * mutation (a burned table version may remain, per the identity rule).
 * @throws std::invalid_argument for foreign authority or invalid geometry.
 */
std::optional<PageTableAdmission> admitPoolAndTable(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table, std::size_t pages);

/**
 * @brief Admit protected growth, one table bank and one accepted-work
 *        reservation issuance as ONE all-resource transaction.
 *
 * plan P1: bounded optimistic admission protects issuance for accepted work;
 * a normal begin can never consume the reserved headroom, so only a minted
 * reservation handle authorizes the accepted update. The reservation is
 * consumed exactly once; if the table bank cannot be acquired (or the
 * reservation is foreign/exhausted), the pool grant rolls back and no
 * reservation capacity is lost.
 * @param reservation Authority-issued accepted-work handle from the SAME
 * table; a foreign or exhausted handle refuses before bank acquisition.
 */
std::optional<PageTableAdmission> admitPoolTableReserved(
    PhysicalPagePool &pool, CPUFP32PagedKVTable &table, std::size_t pages,
    CPUFP32PagedKVTable::IssuanceReservation &reservation);
} // namespace llaminar2
