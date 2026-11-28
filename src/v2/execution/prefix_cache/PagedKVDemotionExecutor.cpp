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
    // All-or-none at the storage level: every put must succeed; on any failure
    // retire every extent already published (rollback — no partial
    // persistence, no quota residue, no file debris).
    std::vector<ExecutedPagedDemotionMove> undone;
    for (const auto &move : plan.moves)
    {
        auto bytes = page_bytes(move);
        if (!bytes.has_value())
        {
            for (auto &done : undone)
                (void)store->retire(done.stored_extent);
            result.refusal = "a planned page has no available bytes (rollback of the published extents)";
            return result;
        }
        std::optional<PagedKVArchive> archive;
        try
        {
            archive = PagedKVArchive::parse(*bytes);
        }
        catch (const std::exception &e)
        {
            for (auto &done : undone)
                (void)store->retire(done.stored_extent);
            result.refusal = std::string("a planned page does not serialize as a section: ") + e.what();
            return result;
        }
        std::string refusal;
        auto key = store->put(*archive, move.extent.generation,
            move.extent.namespace_id, &refusal);
        if (!key.has_value())
        {
            for (auto &done : undone)
                (void)store->retire(done.stored_extent);
            result.refusal = "a planned page store.put failed: " + refusal;
            return result;
        }
        undone.push_back({move, *key});
    }
    result.executed = std::move(undone);
    return result;
}
} // namespace llaminar2
