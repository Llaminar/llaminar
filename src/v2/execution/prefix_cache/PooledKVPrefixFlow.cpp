/**
 * @file PooledKVPrefixFlow.cpp
 * @brief One typed prefix flow over a composed instance.
 */
#include "execution/prefix_cache/PooledKVPrefixFlow.h"

#include <stdexcept>

namespace llaminar2
{
PooledKVPrefixFlow::PooledKVPrefixFlow(PagedKVInstance &instance,
    std::size_t metadata_capacity, std::size_t retention_capacity,
    std::optional<RetentionClass> ceiling, std::uint64_t trust_domain)
    : instance_(instance), owner_(retention_capacity),
      resolver_(RetentionIntentResolver(ceiling)),
      trust_domain_(trust_domain), records_(metadata_capacity)
{
    if (metadata_capacity == 0)
        throw std::invalid_argument("prefix flow requires a positive metadata membership");
    if (trust_domain == 0)
        throw std::invalid_argument("prefix flow requires a nonzero authenticated route trust domain");
    if (instance.diskStore() == nullptr)
        throw std::invalid_argument("prefix flow requires a composed instance with a disk route");
}

PooledPrefixRecord *PooledKVPrefixFlow::findRecord(std::uint64_t ns,
    std::uint64_t prefix)
{
    for (std::size_t i = 0; i < count_; ++i)
        if (records_[i].namespace_id == ns && records_[i].prefix_id == prefix)
            return &records_[i];
    return nullptr;
}

std::optional<PooledPrefixRecord> PooledKVPrefixFlow::admit(
    std::uint64_t namespace_id, std::uint64_t prefix_id,
    std::string_view cache_retention, std::string_view cache_key)
{
    const auto resolution = resolver_.resolve(cache_retention, cache_key);
    if (!resolution.ok())
        return std::nullopt;
    if (findRecord(namespace_id, prefix_id) != nullptr)
        return std::nullopt; // Duplicate identity: no mutation.
    if (count_ == capacity())
        return std::nullopt; // Bounded metadata full.
    const auto owner_key = RetentionEntryKey{namespace_id, prefix_id};
    RetentionValueScore value;
    value.recency = 0; // Score is owned by the §8.2 observers; fresh = 0.
    if (!owner_.admit(owner_key, *resolution.intent, /*write_ordinal=*/ordinal(),
            value, /*live=*/true))
        return std::nullopt;
    PooledPrefixRecord rec;
    rec.namespace_id = namespace_id;
    rec.prefix_id = prefix_id;
    rec.owner_key = owner_key;
    rec.live = true;
    records_[count_++] = rec;
    return rec;
}

bool PooledKVPrefixFlow::setLive(std::uint64_t ns, std::uint64_t prefix, bool live)
{
    auto *rec = findRecord(ns, prefix);
    if (!rec) return false;
    rec->live = live;
    return owner_.setLive(rec->owner_key, live);
}

PooledPrefixRecord *PooledKVPrefixFlow::mutableRecord(
    std::uint64_t ns, std::uint64_t prefix)
{
    return findRecord(ns, prefix);
}

const PooledPrefixRecord *PooledKVPrefixFlow::record(std::uint64_t ns,
    std::uint64_t prefix) const
{
    for (std::size_t i = 0; i < count_; ++i)
        if (records_[i].namespace_id == ns && records_[i].prefix_id == prefix)
            return &records_[i];
    return nullptr;
}

namespace
{
/** @brief R51 retry: pending retire failures (real remove errors) are
 * re-attempted BEFORE any gate of the operation can refuse — a
 * live/zero-deficit/not-resident/whole-prefix refusal still heals them.
 * Failures stay owned in the record (never anonymous). */
void retryPendingRetire(PooledPrefixRecord &rec, PagedKVArchiveStore &store)
{
    if (rec.pending_retire.empty())
        return;
    std::vector<PagedKVArchiveExtentKey> still;
    for (const auto &key : rec.pending_retire)
        if (!store.retire(key)) still.push_back(key);
    rec.pending_retire = std::move(still);
}
} // namespace

namespace
{
/** @brief §8.4 cascade-down: on a REAL store overflow (quota refusal for a
 * put) discard the lowest-ranked idle victim's extents (retire + owner
 * remove) so the store can admit the new extent. Exact-or-fail: victims are
 * never live/protected (the owner's live rule) and every retire is exact.
 * @return True when at least one victim was discarded. */
bool discardLowestVictim(PooledKVPrefixFlow &flow, RetentionOwner &owner,
    PagedKVArchiveStore &store, std::uint64_t now)
{
    auto victims = owner.selectVictims(now, 1);
    if (victims.empty())
        return false;
    const auto &key = victims.front();
    auto *rec = flow.mutableRecord(key.namespace_id, key.entry_id);
    if (!rec)
        return false;
    bool any = false;
    for (const auto &extent : rec->head_extents)
        if (store.retire(extent)) any = true;
    for (const auto &extent : rec->tail_extents)
        if (store.retire(extent)) any = true;
    for (const auto &extent : rec->pending_retire)
        if (store.retire(extent)) any = true;
    rec->head_extents.clear();
    rec->tail_extents.clear();
    rec->pending_retire.clear();
    rec->stored_rows = 0;
    rec->resident_head_rows = 0;
    (void)owner.remove(key);
    return any;
}
} // namespace

PooledPrefixDemotionOutcome PooledKVPrefixFlow::demoteTail(
    std::uint64_t ns, std::uint64_t prefix, PooledKVDecodeBinding &binding,
    std::size_t deficit_pages)
{
    PooledPrefixDemotionOutcome outcome;
    auto *rec = findRecord(ns, prefix);
    if (!rec || rec->live)
    {
        outcome.refusal = "demotion requires an idle admitted prefix record";
        return outcome;
    }
    auto *store = instance_.diskStore();
    retryPendingRetire(*rec, *store); // Before ANY operation gate.
    if (deficit_pages == 0)
        return outcome; // Zero deficit: nothing to do.
    if (binding.pool() != &instance_.pool())
    {
        outcome.refusal = "demotion requires the composed instance's binding";
        return outcome;
    }
    // Export the CURRENT live publication ONCE per layer (N1: no double
    // export), stamp the ROUTE's trust/ancestry (the same values the restore
    // expects), and slice the HEAD + TAIL pages out as their own per-layer
    // extents. Rows are UNIFORM across layers (the binding's publication
    // invariant); a skew refuses before any byte moves.
    const auto page_tokens = binding.pageTokens();
    const auto layers = binding.layerCount();
    std::vector<PagedKVArchive> heads;
    std::vector<PagedKVArchive> tails;
    int head_rows = 0;
    std::optional<int> rows0;
    for (int layer = 0; layer < layers; ++layer)
    {
        auto section = binding.exportLayer(layer);
        if (!section)
        {
            outcome.refusal = "the binding has no live publication to demote";
            return outcome;
        }
        const auto rows = section->meta().rows;
        if (!rows0.has_value())
            rows0 = static_cast<int>(rows);
        else if (*rows0 != static_cast<int>(rows))
        {
            outcome.refusal = "demotion requires uniform per-layer extents";
            return outcome;
        }
        const auto pages = (rows + static_cast<std::size_t>(page_tokens) - 1) /
            static_cast<std::size_t>(page_tokens);
        if (deficit_pages > pages)
            deficit_pages = pages; // Deficit-only: bounded by the chain
                                   // (hoist: same value for every layer).
        // TAIL-FIRST page arithmetic: the LAST `deficit_pages` pages of the
        // chain are demoted; the final page may be PARTIAL (a mid-page
        // terminal boundary), so the tail row count is the exact sum of the
        // last pages' rows — never deficit*page_tokens blindly.
        std::size_t tail_rows = 0;
        for (std::size_t k = 0; k < deficit_pages; ++k)
        {
            const auto p = pages - 1 - k; // Page ordinal from the start.
            const auto page_first = p * static_cast<std::size_t>(page_tokens);
            const auto rows_of = rows - page_first >=
                    static_cast<std::size_t>(page_tokens)
                ? static_cast<std::size_t>(page_tokens)
                : rows - page_first;
            tail_rows += rows_of;
        }
        if (tail_rows >= rows)
        {
            outcome.refusal = "the tail deficit covers the whole prefix (no resident head remains)";
            return outcome;
        }
        const auto head = rows - tail_rows;
        if (layer == 0)
            head_rows = static_cast<int>(head);
        heads.push_back(section->slice(0, head).withProvenance(trust_domain_, prefix));
        tails.push_back(section->slice(head, rows).withProvenance(trust_domain_, prefix));
    }
    // ALL-or-none against the store: publish the RESIDENT HEAD and the TAIL
    // as separate per-layer extents (the restore path owns the complete
    // prefix in the store — head + tail are both needed because a hijacked
    // restore must be reconstructable without any live binding). A failure
    // retires the already-published ones and truncates NOTHING.

    std::vector<PagedKVArchiveExtentKey> published;
    auto putExtent = [&](const PagedKVArchive &section, std::uint64_t gen)
        -> std::optional<PagedKVArchiveExtentKey>
    {
        std::string refusal;
        auto key = store->put(section, gen, ns, &refusal);
        if (key) return key;
        // §8.4 deficit-only cascade: a REAL store overflow discards the
        // lowest-ranked idle victim's extents (exact retires; live/protected
        // are never selected) and retries ONCE. A second refusal is a real
        // overflow and fails the demotion precisely (never a silent partial).
        if (refusal.find("quota") != std::string::npos)
        {
            if (discardLowestVictim(*this, owner_, *store, ordinal()))
            {
                refusal.clear();
                key = store->put(section, gen, ns, &refusal);
                if (key) return key;
            }
        }
        return std::nullopt;
    };
    for (std::size_t layer = 0; layer < heads.size(); ++layer)
    {
        auto key = putExtent(heads[layer], prefix);
        if (!key)
        {
            for (auto &done : published) (void)store->retire(done);
            outcome.refusal = "head demotion store.put failed at layer " + std::to_string(layer);
            return outcome;
        }
        published.push_back(*key);
    }
    for (std::size_t layer = 0; layer < tails.size(); ++layer)
    {
        auto key = putExtent(tails[layer], prefix);
        if (!key)
        {
            for (auto &done : published) (void)store->retire(done);
            outcome.refusal = "tail demotion store.put failed at layer " + std::to_string(layer);
            return outcome;
        }
        published.push_back(*key);
    }
    // RAM-side retirement at the call boundary: the binding-level ATOMIC
    // truncate (F1) — every layer prechecked and published all-or-none; a
    // refusal means NOTHING changed (the store is rolled back and no layer
    // lost its rows).
    if (!binding.truncateAllLayers(head_rows))
    {
        for (auto &done : published)
            (void)store->retire(done);
        outcome.refusal = "binding atomic truncate refused (no layer changed; store rolled back)";
        return outcome;
    }
    rec->resident_head_rows = head_rows;
    rec->stored_rows = binding.priorFor(0)->snapshot.view().rows();
    // Split the published extents: the first `layers` are HEAD slices.
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(layers); ++layer)
        rec->head_extents.push_back(published[layer]);
    for (std::size_t layer = static_cast<std::size_t>(layers);
         layer < published.size(); ++layer)
        rec->tail_extents.push_back(published[layer]);
    rec->trust_domain = trust_domain_;
    rec->ancestry_id = prefix;
    // Record the executed tail-first moves for the caller's accounting.
    for (std::size_t layer = 0; layer < rec->tail_extents.size(); ++layer)
    {
        PagedDemotionMove move;
        move.namespace_id = ns;
        move.prefix_id = prefix;
        move.page_index = layer;
        move.extent = rec->tail_extents[layer];
        outcome.moved.push_back(move);
    }
    outcome.head_rows = head_rows;
    return outcome;
}

bool PooledKVPrefixFlow::storeFullPrefix(std::uint64_t ns,
    std::uint64_t prefix, PooledKVDecodeBinding &binding,
    std::string *refusal)
{
    if (refusal) refusal->clear();
    auto *rec = findRecord(ns, prefix);
    if (!rec)
    {
        if (refusal) *refusal = "storeFullPrefix requires an admitted prefix record";
        return false;
    }
    if (binding.pool() != &instance_.pool())
    {
        if (refusal) *refusal = "storeFullPrefix requires the composed instance's binding";
        return false;
    }
    auto *store = instance_.diskStore();
    retryPendingRetire(*rec, *store); // Before ANY gate (R51 contract).
    const auto layers = binding.layerCount();
    std::vector<PagedKVArchive> sections;
    int total_rows = 0;
    std::optional<int> rows0;
    for (int layer = 0; layer < layers; ++layer)
    {
        auto section = binding.exportLayer(layer);
        if (!section)
        {
            if (refusal) *refusal = "storeFullPrefix has no live publication to store";
            return false;
        }
        const auto rows = section->meta().rows;
        if (!rows0.has_value()) rows0 = static_cast<int>(rows);
        else if (*rows0 != static_cast<int>(rows))
        {
            if (refusal) *refusal = "storeFullPrefix requires uniform per-layer extents";
            return false;
        }
        total_rows = static_cast<int>(rows);
        sections.push_back(section->withProvenance(trust_domain_, prefix));
    }
    if (total_rows <= 0)
    {
        if (refusal) *refusal = "storeFullPrefix requires a positive live publication";
        return false;
    }
    // ALL-or-none: every layer's complete section becomes ONE head extent
    // (the tail list stays empty — a single-page prefix has no tail).
    std::vector<PagedKVArchiveExtentKey> published;
    for (const auto &section : sections)
    {
        std::string put_refusal;
        auto key = store->put(section, prefix, ns, &put_refusal);
        if (!key && put_refusal.find("quota") != std::string::npos)
        {
            if (discardLowestVictim(*this, owner_, *store, ordinal()))
            {
                put_refusal.clear();
                key = store->put(section, prefix, ns, &put_refusal);
            }
        }
        if (!key)
        {
            for (auto &done : published) (void)store->retire(done);
            if (refusal)
                *refusal = "storeFullPrefix store.put failed: " + put_refusal;
            return false;
        }
        published.push_back(*key);
    }
    rec->head_extents = std::move(published);
    rec->tail_extents.clear();
    rec->resident_head_rows = 0;
    rec->stored_rows = total_rows;
    rec->trust_domain = trust_domain_;
    rec->ancestry_id = prefix;
    return true;
}

bool PooledKVPrefixFlow::restore(std::uint64_t ns, std::uint64_t prefix,
    PooledKVDecodeBinding &binding, std::string *refusal)
{
    if (refusal) refusal->clear();
    auto *rec = findRecord(ns, prefix);
    if (!rec)
    {
        if (refusal) *refusal = "restore requires an admitted prefix record";
        return false;
    }
    retryPendingRetire(*rec, *instance_.diskStore()); // Before ANY gate.
    if (binding.pool() != &instance_.pool())
    {
        if (refusal) *refusal = "restore requires the composed instance's binding";
        return false;
    }
    if (rec->head_extents.empty())
    {
        if (refusal) *refusal = "the prefix is not store-resident (no head extent)";
        return false;
    }
    if (rec->tail_extents.empty() && rec->stored_rows <= 0)
    {
        if (refusal) *refusal = "the prefix is not fully resident in the store (head-only requires the stored row count)";
        return false;
    }
    // Read every stamped head+tail extent back (the store verifies identity
    // and digest; the binding enforces the ROUTE's trust/ancestry
    // expectations BEFORE a single byte is staged).
    std::vector<PagedKVArchive> archives;
    auto *store = instance_.diskStore();
    const auto layers = rec->head_extents.size();
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        std::string head_refusal, tail_refusal;
        auto head = store->get(rec->head_extents[layer], &head_refusal);
        if (!head)
        {
            if (refusal)
                *refusal = "head read-back refused at layer " +
                    std::to_string(layer) + ": " + head_refusal;
            return false;
        }
        if (rec->tail_extents.empty())
        {
            // Head-only store (single-page prefix): the head covers the whole
            // prefix — validate the stored row count before serving it.
            if (static_cast<int>(head->meta().rows) != rec->stored_rows)
            {
                if (refusal)
                    *refusal = "head-only restore row count disagrees with the record ("
                        + std::to_string(head->meta().rows) + " vs " +
                        std::to_string(rec->stored_rows) + ")";
                return false;
            }
            archives.push_back(std::move(*head));
            continue;
        }
        auto tail = store->get(rec->tail_extents[layer], &tail_refusal);
        if (!tail)
        {
            if (refusal)
                *refusal = "tail read-back refused at layer " +
                    std::to_string(layer) + ": " + tail_refusal;
            return false;
        }
        try
        {
            // Head + tail carry the SAME stamped provenance (concat refuses a
            // mismatch before merging).
            archives.push_back(head->concat(*tail));
        }
        catch (const std::exception &e)
        {
            if (refusal) *refusal = std::string("head+tail concat refused at layer ") +
                std::to_string(layer) + ": " + e.what();
            return false;
        }
    }
    if (!binding.attachStoredImport(archives, rec->trust_domain, rec->ancestry_id))
    {
        if (refusal)
            *refusal = "restored tail import refused (trust/ancestry or provenance gate)";
        return false;
    }
    // §8.2/§8.4: the successful restore IS effective use — refresh the
    // lifetime receipt under the same ordered path (an expired entry gains
    // its class back; the value score/sequence advance with it).
    (void)owner_.refresh(rec->owner_key, ordinal());

    // F2: move-between-tiers CREDITS THE SOURCE AFTER PRESERVATION — the
    // demoted extents are retired now (store quota returns to its
    // pre-demotion value), AFTER the import succeeded (failed restores keep
    // the extents and the record). R51 hardening: a REAL retire failure
    // (filesystem remove error — not the missing-file healing, which
    // succeeds) keeps that extent in the record's pending_retire (admitted
    // and OWNED, never anonymous) and is reported in the refusal string; the
    // return value stays true because the restore itself succeeded.
    auto *store_retire = instance_.diskStore();
    std::vector<PagedKVArchiveExtentKey> failed;
    for (const auto &head : rec->head_extents)
        if (!store_retire->retire(head)) failed.push_back(head);
    rec->head_extents.clear();
    for (const auto &tail : rec->tail_extents)
        if (!store_retire->retire(tail)) failed.push_back(tail);
    rec->tail_extents.clear();
    for (auto &f : failed) rec->pending_retire.push_back(std::move(f));
    rec->resident_head_rows = 0; // Fully resident again: no demoted head.
    rec->stored_rows = 0;        // The store no longer holds this prefix.
    if (!failed.empty() && refusal)
        *refusal = "restore succeeded; " + std::to_string(failed.size()) +
            " demoted extent(s) could not be retired and remain owned in the "
            "record (disk quota still charged); retirement is re-attempted on "
            "the next demotion/restore";
    return true;
}
} // namespace llaminar2
