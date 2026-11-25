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
} // namespace llaminar2
