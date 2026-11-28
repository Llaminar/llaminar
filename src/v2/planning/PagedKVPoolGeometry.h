/**
 * @file PagedKVPoolGeometry.h
 * @brief Costs and resolves a paged KV pool against an admitted resource budget.
 *
 * Design §4.2: after the complete non-pool PMA BOM, the remaining admitted
 * capacity on a participant bounds the resident KV pool; `auto` maximizes it,
 * an explicit cap bounds the same pool. This resolver is PURE: it never reads
 * host free memory, never subtracts from any authority ledger and never
 * infers a reserved percentage. It consumes an EXACT remaining-budget input
 * (the value PMA reports after the non-pool BOM) and emits a candidate
 * geometry that the caller then re-validates through `PhysicalMemoryAuthority`
 * (the only physical accounting authority). Increasing a tier cap can never
 * steal protected execution memory because the remaining budget is supplied
 * by the admitted non-pool BOM, not by this class.
 */
#pragma once
#include "kernels/kvcache/KVStorageLayout.h"
#include "planning/PhysicalPagePool.h"
#include <cstddef>

namespace llaminar2
{
/** @brief Typed resolution inputs; all arithmetic is checked before use. */
struct PagedKVPoolBudget
{
    std::size_t remaining_budget_bytes = 0; ///< PMA-reported bytes after the complete non-pool BOM.
    std::size_t frame_bytes = 0;            ///< Exact one-page frame byte cost (native layout priced).
    std::size_t request_records = 0;        ///< Concurrent record slots (owner policy, not inferred).
    std::size_t max_pages_per_request = 0;  ///< Admitted per-request bound.
    bool capped = false;                    ///< False: `auto` maximizes; true: cap is authoritative.
    std::size_t cap_bytes = 0;              ///< Explicit cap; ignored when capped == false.
};

/** @brief Resolved candidate geometry plus exact admitted byte cost. */
struct PagedKVPoolResolution
{
    PhysicalPagePoolGeometry geometry;
    std::size_t budget_bytes = 0; ///< Bytes the geometry actually consumes.
};

/**
 * @brief Resolve the maximal (or capped) legal pool geometry for one participant.
 * @param budget Exact supplied budget and geometry constraints.
 * @return Geometry whose frame count is the largest whole-frame count inside
 * the effective limit, or empty when the budget cannot hold one frame.
 * @throws std::invalid_argument For zero frame cost, zero records, a zero
 * per-request bound, or arithmetic overflow in the cost calculation.
 * @note The returned geometry must then be admitted by PMA (which remains the
 * sole physical authority); this function performs no accounting.
 */
std::optional<PagedKVPoolResolution> resolvePagedKVPoolGeometry(
    const PagedKVPoolBudget &budget);
} // namespace llaminar2
