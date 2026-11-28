/**
 * @file PagedKVDemotionExecutor.cpp
 * @brief Mechanical demotion executor implementation.
 */
#include "execution/prefix_cache/PagedKVDemotionExecutor.h"

namespace llaminar2
{
namespace
{
bool isSamePhysical(const PagedKVArchiveExtentKey &a, const PagedKVArchiveExtentKey &b)
{
    return a.namespace_id == b.namespace_id && a.digest == b.digest &&
        a.bytes == b.bytes;
}
} // namespace

PagedDemotionExecution executePagedDemotionPlan(
    const PagedDemotionPlan &plan, PagedKVInstance &instance,
    const std::function<std::optional<std::vector<std::byte>>(
        const PagedDemotionMove &)> &page_bytes)
{
    PagedDemotionExecution result;
    if (!plan.ok())
    {
        result.refusal = "cannot execute a non-ok plan: " + plan.refusal;
        return result;
    }
    auto *store = instance.diskStore();
    if (store == nullptr)
    {
        result.refusal = "the composed instance has no disk route to execute against";
        return result;
    }
    // All-or-none at the storage level: every put must succeed AND the
    // stored extent must BE the planned physical identity; on any failure
    // retire every extent already published (rollback — no partial
    // persistence, no quota residue, no file debris). Rollback retire
    // failures are surfaced in the refusal detail (never silently voided).
    std::vector<ExecutedPagedDemotionMove> undone;
    const auto rollback = [&](std::string &detail) {
        for (auto &done : undone)
            if (!store->retire(done.stored_extent))
                detail += " retire-failed:ns=" +
                    std::to_string(done.stored_extent.namespace_id) +
                    " gen=" + std::to_string(done.stored_extent.generation);
    };
    for (const auto &move : plan.moves)
    {
        auto bytes = page_bytes(move);
        if (!bytes.has_value())
        {
            std::string detail;
            rollback(detail);
            result.refusal = "a planned page has no available bytes (published extents rolled back" +
                (detail.empty() ? std::string() : "; " + detail) + ")";
            return result;
        }
        std::optional<PagedKVArchive> archive;
        try
        {
            archive = PagedKVArchive::parse(*bytes);
        }
        catch (const std::exception &e)
        {
            std::string detail;
            rollback(detail);
            result.refusal = std::string("a planned page does not serialize as a section: ") +
                e.what() + (detail.empty() ? std::string() : "; " + detail);
            return result;
        }
        std::string refusal;
        auto key = store->put(*archive, move.extent.generation,
            move.extent.namespace_id, &refusal);
        if (!key.has_value())
        {
            std::string detail;
            rollback(detail);
            result.refusal = "a planned page store.put failed: " + refusal +
                (detail.empty() ? std::string() : "; " + detail);
            return result;
        }
        // F1: the stored extent must BE the planned physical identity (the
        // plan gated on move.extent's namespace/digest/bytes). A provider
        // that yields different bytes than the planner gated is a stale-plan
        // disagreement: refuse and roll back everything.
        if (!isSamePhysical(*key, move.extent))
        {
            undone.push_back({move, *key}); // The mismatched extent is
            // published: it MUST be retired by the rollback too.
            std::string detail;
            rollback(detail);
            result.refusal = "stored extent identity disagrees with the planned one (ns " +
                std::to_string(key->namespace_id) + " digest " +
                std::to_string(key->digest) + " vs planned digest " +
                std::to_string(move.extent.digest) + ")" +
                (detail.empty() ? std::string() : "; " + detail);
            return result;
        }
        undone.push_back({move, *key});
    }
    result.executed = std::move(undone);
    return result;
}
} // namespace llaminar2
