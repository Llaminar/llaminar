/**
 * @file PooledKVPrefixFlow.h
 * @brief One typed prefix flow over a composed instance (P3 M11).
 *
 * The prefix-flow WIRING: request admission (standard retention fields via
 * the §8.4 resolver with the operator ceiling) -> retention-owner record
 * (bounded admitted metadata) -> demotion run (PagedKVRetentionDemotion plan
 * + PagedKVDemotionExecutor over REAL binding exports, RAM-side retirement =
 * the binding truncate at the call boundary) -> restore run (real
 * trust/ancestry expectations from the ROUTE, not the archive's own record:
 * the flow stamps withProvenance at write so the table-level authenticated
 * attach enforces for real).
 *
 * ONE composed instance owns everything: the RAM pool, the table, and the
 * disk store (route-validated, quota-admitted). All prefix records are
 * bounded admitted metadata (refuse beyond; no unbounded key map). The flow
 * performs NO page/byte accounting itself: PMA, the pool and the store remain
 * the physical owners; the flow composes the existing authorities.
 *
 * Boundary: the DGO/runner still owns request lifecycle (create bindings,
 * drive steps); the flow owns prefix retention, demotion and restore around
 * them. Live/protected prefixes are never demoted (the retention owner's
 * rule); storePrefix is meaningful only for fps ended (idle) requests.
 */
#pragma once
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
#include "execution/prefix_cache/PagedKVDemotionExecutor.h"
#include "execution/prefix_cache/RetentionIntentResolver.h"
#include "execution/prefix_cache/RetentionOwner.h"
#include "planning/PagedKVInstance.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief One bounded prefix record: resident head + stamped tail extents. */
struct PooledPrefixRecord
{
    std::uint64_t namespace_id = 0;
    std::uint64_t prefix_id = 0;
    RetentionEntryKey owner_key;
    /** @brief Rows still resident in the RAM tier (the hybrid head). */
    int resident_head_rows = 0;
    /** @brief Stamped head extents in the disk tier (at demotion the
     * complete prefix becomes store-resident: head + tail per layer). */
    std::vector<PagedKVArchiveExtentKey> head_extents;
    /** @brief Stamped tail extents in the disk tier (with trust/ancestry). */
    std::vector<PagedKVArchiveExtentKey> tail_extents;
    /** @brief The authenticated route values stamped at write time; the
     * restore passes THE SAME values as the expectation parameters. */
    std::uint64_t trust_domain = 0;
    std::uint64_t ancestry_id = 0;
    bool live = false;
};

/** @brief The outcome of one demotion run on one prefix. */
struct PooledPrefixDemotionOutcome
{
    std::vector<PagedDemotionMove> moved; ///< Real planned+executed moves.
    int head_rows = 0;                    ///< New RAM-resident row count.
    std::string refusal;                  ///< Nonempty = nothing moved/truncated.
    bool ok() const noexcept { return refusal.empty(); }
};

class PooledKVPrefixFlow final
{
public:
    /**
     * @param instance The composed instance (RAM pool + table + disk route).
     * @param metadata_capacity Bounded prefix-record membership (typed, > 0).
     * @param retention_capacity Bounded retention-owner membership (typed).
     * @param ceiling Optional operator retention-class ceiling.
     * @param trust_domain The authenticated route trust domain (never 0).
     */
    PooledKVPrefixFlow(PagedKVInstance &instance,
        std::size_t metadata_capacity, std::size_t retention_capacity,
        std::optional<RetentionClass> ceiling, std::uint64_t trust_domain);

    /**
     * @brief Admit one request's retention intent (standard fields).
     * @return The typed record identity on success, or a precise refusal
     * (unsupported retention value, above ceiling, full metadata).
     */
    std::optional<PooledPrefixRecord> admit(std::uint64_t namespace_id,
        std::uint64_t prefix_id, std::string_view cache_retention,
        std::string_view cache_key);

    /** @brief Mark a prefix live (protected) or idle (victimizable/demotable). */
    bool setLive(std::uint64_t namespace_id, std::uint64_t prefix_id, bool live);

    /**
     * @brief Demote the TAIL pages of one idle prefix to the disk tier.
     * @param namespace_id/prefix_id Record identity.
     * @param binding The binding whose live publication holds the prefix
     * (the flow exports its pages — the R23 seam — stamps them with the
     * route's trust/ancestry, plans tail-first deficit demotion, executes
     * through the composed store, and on success TRUNCATES the binding to
     * the resident head (RAM-side retirement at the call boundary).
     * @param deficit_pages Page-rounded deficit to demote.
     * @return The executed moves + new head rows, or a refusal with NOTHING
     * moved and NO truncation (all-or-none). On success the COMPLETE prefix
     * is store-resident (head + tail per layer, both stamped with the
     * route's trust/ancestry) and the binding is truncated to the resident
     * head (RAM-side retirement at the call boundary).
     */
    PooledPrefixDemotionOutcome demoteTail(std::uint64_t namespace_id,
        std::uint64_t prefix_id, PooledKVDecodeBinding &binding,
        std::size_t deficit_pages);

    /**
     * @brief Restore a hybrid prefix into a FRESH binding.
     * @param namespace_id/prefix_id Record identity.
     * @param binding Fresh destination binding (same pool/table authority).
     * @return True on success: every stamped head+tail extent reads back
     * through the store, concat-merges into one full per-layer section, and
     * imports through attachStoredImport with the ROUTE's real
     * trust/ancestry expectations (the authenticated attach enforces; a
     * tampered or wrong-domain extent refuses). The record's extents are
     * cleared only after the complete import succeeded.
     */
    bool restore(std::uint64_t namespace_id, std::uint64_t prefix_id,
        PooledKVDecodeBinding &binding, std::string *refusal = nullptr);

    /** @brief Bounded record access (no unbounded map on the hot path). */
    const PooledPrefixRecord *record(std::uint64_t namespace_id,
        std::uint64_t prefix_id) const;

    /** @return Admitted record count. */
    std::size_t count() const noexcept { return count_; }
    /** @return Admitted record membership. */
    std::size_t capacity() const noexcept { return records_.size(); }

private:
    PooledPrefixRecord *findRecord(std::uint64_t namespace_id, std::uint64_t prefix_id);
    PooledKVDecodeBinding &requireBindingPtr(PooledKVDecodeBinding &binding);
    PagedKVInstance &instance_;
    RetentionOwner owner_;
    RetentionIntentResolver resolver_;
    std::uint64_t trust_domain_ = 0;
    std::vector<PooledPrefixRecord> records_;
    std::size_t count_ = 0;
};
} // namespace llaminar2
