/**
 * @file PagedKVInstancePlan.h
 * @brief One checked pooled-KV candidate: pool frames + table metadata + size.
 *
 * Design §4.2/§4.3: the resident pool and its table metadata are one admitted
 * instance whose complete BOM must be checked and validated together before
 * PMA admission. This bundler combines the tier intent (auto/cap, host tier),
 * the exact page geometry and the typed table geometry into a single
 * candidate and prices it overflow-checked. It performs NO accounting: PMA
 * remains the sole physical authority, and the caller re-validates the
 * produced geometry through the authority before construction. Metadata is
 * never inferred from a remaining budget; the caller supplies the exact
 * geometric policy (bank count, reader envelope, page token size) and the
 * bundler refuses an unrepresentable combination.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"
#include "execution/prefix_cache/PagedKVDiskRoute.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "planning/PagedKVPoolGeometry.h"
#include "planning/PhysicalPagePool.h"
#include <cstddef>
#include <optional>

namespace llaminar2
{
/** @brief Exact page/table policy inputs for one pooled-KV instance. */
struct PagedKVInstancePolicy
{
    int page_tokens = 0;              ///< Native logical block size per page.
    std::size_t bank_count = 0;       ///< Concurrent metadata banks (nameable by owner policy).
    std::size_t readers_per_bank = 0; ///< Bounded reader envelope per published bank.
    std::size_t request_records = 0;  ///< Concurrent request record slots.
};

/** @brief Bundled candidate; both geometries are consistent or none exists. */
struct PagedKVInstancePlan
{
    PhysicalPagePoolGeometry pool_geometry;
    CPUFP32PagedKVTableGeometry table_geometry;
    std::size_t total_bytes = 0; ///< Pool + table metadata physical BOM.
    /**
     * @brief The validated composed disk route (quota + membership + exact
     * directory); present exactly when the tier's disk is enabled. The
     * membership is checked against the planned geometry's typed need =
     * frames + one in-flight write + one pending replacement + one retained
     * old generation + one journal/rounding record; the quota is checked to
     * hold at least one rounded extent of the admitted frame geometry.
     * Quota floor note: the floor uses the pool frame bytes WITHOUT the
     * archive header (a lower bound on the true serialized extent the store
     * charges = header + payload, rounded). A per-section put can still
     * refuse when the exact serialized size crosses the block boundary; that
     * per-put refusal is precise and numeric — an impossible composed route
     * still fails closed in planning, never at a silent boundary.
     */
    std::optional<PagedKVDiskRoute> disk_route;
};

/**
 * @brief Bundle one candidate from the typed tier intent and exact policy.
 * @param config Authoring tier intents (pool auto/cap rule applied here).
 * @param remaining PRef-resolved remaining budget (pool bytes after non-pool BOM).
 * @param policy Exact geometric policy for frames/table/banks.
 * @param device CPU device whose storage root prices the tables.
 * @param placement Original placement (typically local).
 * @return The complete candidate, or empty when the remaining budget cannot
 * hold even one frame plus the typed table metadata; no partial plan exists.
 * @throws std::invalid_argument For an invalid tier intent, policy geometry
 * or a table metadata size that exceeds the remaining budget.
 */
std::optional<PagedKVInstancePlan> planPagedKVInstance(
    const PagedKVTierConfig &config, const PagedKVPoolBudget &remaining,
    const PagedKVInstancePolicy &policy, DeviceId device,
    CPUStoragePlacement placement = CPUStoragePlacement::local());
} // namespace llaminar2
