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
    if (!owner_.admit(owner_key, *resolution.intent, /*write_ordinal=*/0,
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

const PooledPrefixRecord *PooledKVPrefixFlow::record(std::uint64_t ns,
    std::uint64_t prefix) const
{
    for (std::size_t i = 0; i < count_; ++i)
        if (records_[i].namespace_id == ns && records_[i].prefix_id == prefix)
            return &records_[i];
    return nullptr;
}

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
    if (deficit_pages == 0)
        return outcome; // Zero deficit: nothing to do.
    if (binding.pool() != &instance_.pool())
    {
        outcome.refusal = "demotion requires the composed instance's binding";
        return outcome;
    }
    // Export the CURRENT live publication per layer (R23 oracle seam), stamp
    // the ROUTE's trust/ancestry (the same values the restore expects), and
    // slice the TAIL pages out as their own per-layer extents.
    const auto page_tokens = binding.pageTokens();
    const auto layers = binding.layerCount();
    std::vector<PagedKVArchive> tails;
    int head_rows = 0;
    for (int layer = 0; layer < layers; ++layer)
    {
        auto section = binding.exportLayer(layer);
        if (!section)
        {
            outcome.refusal = "the binding has no live publication to demote";
            return outcome;
        }
        const auto rows = section->meta().rows;
        const auto pages = (rows + static_cast<std::size_t>(page_tokens) - 1) /
            static_cast<std::size_t>(page_tokens);
        if (deficit_pages > pages)
            deficit_pages = pages; // Deficit-only: bounded by the chain.
        // TAIL-FIRST page arithmetic: the LAST `deficit_pages` pages of the
        // chain are demoted; the final page may be PARTIAL (a mid-page
        // terminal boundary), so the tail row count is the exact sum of the
        // last pages' rows — never deficit*page_tokens blindly.
        // The last `deficit_pages` pages counted FROM THE END (their exact
        // rows): page p (0-based from the start) holds
        // min(page_tokens, rows - p*page_tokens); a mid-page terminal
        // boundary never over/under-demotes.
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
        // Page-aligned head; the tail slice is exactly the last tail_rows.
        auto tail = section->slice(head, rows)
            .withProvenance(trust_domain_, prefix);
        tails.push_back(std::move(tail));
    }
    // ALL-or-none against the store: publish the RESIDENT HEAD and the TAIL
    // as separate per-layer extents (the restore path owns the complete
    // prefix in the store — head + tail are both needed because a hijacked
    // restore must be reconstructable without any live binding). A failure
    // retires the already-published ones and truncates NOTHING.
    auto *store = instance_.diskStore();
    std::vector<PagedKVArchiveExtentKey> published;
    auto putExtent = [&](const PagedKVArchive &section, std::uint64_t gen)
        -> std::optional<PagedKVArchiveExtentKey>
    {
        std::string refusal;
        auto key = store->put(section, gen, ns, &refusal);
        if (!key) return std::nullopt;
        return key;
    };
    // Head slices are the FIRST head_rows of each exported layer (the head
    // version keeps the record's own stamped provenance).
    std::vector<PagedKVArchive> heads;
    for (int layer = 0; layer < layers; ++layer)
    {
        auto section = binding.exportLayer(layer);
        if (!section)
        {
            for (auto &done : published) (void)store->retire(done);
            outcome.refusal = "head export refused during demotion";
            return outcome;
        }
        heads.push_back(section->slice(0, static_cast<std::size_t>(head_rows))
            .withProvenance(trust_domain_, prefix));
    }
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
    // RAM-side retirement at the call boundary: truncate EVERY layer to the
    // resident head (all-or-none again: a truncate refusal rolls the store
    // back — the store retire is exact).
    for (int layer = 0; layer < layers; ++layer)
    {
        if (!binding.truncateLayer(layer, head_rows))
        {
            for (auto &done : published)
                (void)store->retire(done);
            outcome.refusal = "binding truncate refused at layer " + std::to_string(layer);
            return outcome;
        }
    }
    rec->resident_head_rows = head_rows;
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
    if (binding.pool() != &instance_.pool())
    {
        if (refusal) *refusal = "restore requires the composed instance's binding";
        return false;
    }
    if (rec->head_extents.empty() || rec->tail_extents.empty())
    {
        if (refusal) *refusal = "the prefix is not fully resident in the store (head+tail required)";
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
        auto tail = store->get(rec->tail_extents[layer], &tail_refusal);
        if (!head || !tail)
        {
            if (refusal)
                *refusal = "head+tail read-back refused at layer " +
                    std::to_string(layer) + ": " + head_refusal + " | " + tail_refusal;
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
    // The import succeeded: the hybrid prefix is fully resident again; the
    // tail record clears (the extents stay admitted until retirement plays
    // out through the store lifecycle — the flow keeps them until the
    // next demotion or the record's removal, so a subsequent restore is
    // idempotent-safe only after the owner re-orders; here we clear the
    // TAIL list so the record reflects residency).
    rec->tail_extents.clear();
    rec->head_extents.clear();
    rec->resident_head_rows = 0;
    return true;
}
} // namespace llaminar2
