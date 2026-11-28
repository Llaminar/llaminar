/**
 * @file PagedKVDemotionExecutor.h
 * @brief Mechanical demotion executor over a real composed route (P3 M9).
 *
 * The decision planner (PagedKVRetentionDemotion) decides; THIS module moves
 * real bytes: it publishes each planned move's page bytes through the
 * composed instance's disk store (PagedKVArchiveStore::put — durable
 * completion, quota, exact extent identity), maps the plan's page record to
 * its stored extent, and is ALL-OR-NONE at the storage level — a failed put
 * aborts the remaining moves and RETIRES the extents already published
 * (rollback; no partial persistence, no quota residue). The page bytes
 * provider is the prefix-flow's page table (the DGO wiring supplies real
 * pooled page snapshots; tests supply the R23 oracle export).
 *
 * The executor never re-plans, never derives quota and never invents an
 * extent: it consumes the exact plan and the exact route. RAM-side retirement
 * and binding re-anchor remain the DGO/runner wiring (next boundary).
 */
#pragma once
#include "execution/prefix_cache/PagedKVRetentionDemotion.h"
#include "planning/PagedKVInstance.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief One executed move with its REAL store-backed extent. */
struct ExecutedPagedDemotionMove
{
    PagedDemotionMove move;
    PagedKVArchiveExtentKey stored_extent;
};

/** @brief All-or-none execution outcome. */
struct PagedDemotionExecution
{
    std::vector<ExecutedPagedDemotionMove> executed;
    std::string refusal; ///< Nonempty exactly when nothing was persisted.
    bool ok() const noexcept { return refusal.empty(); }
};

/**
 * @brief Execute one demotion plan against a composed instance's store.
 * @param plan Exact plan from planTailFirstDemotion.
 * @param instance The composed instance (its disk store owns quota + files).
 * @param page_bytes Real page bytes for one planned move (the prefix-flow's
 * page table); std::nullopt = the page is unavailable (refusal, rollback).
 * @return Executed moves with real extents (each store.put durable + quota
 * charged, deduplicated storage: a byte-identical page is stored once), or a
 * refusal with NOTHING persisted (all published extents retired on failure).
 */
PagedDemotionExecution executePagedDemotionPlan(
    const PagedDemotionPlan &plan, PagedKVInstance &instance,
    const std::function<std::optional<std::vector<std::byte>>(
        const PagedDemotionMove &)> &page_bytes);
} // namespace llaminar2
