/**
 * @file PooledServingSlotOwner.cpp
 * @brief R52 serving-side pooled slot lifecycle owner implementation.
 */
#include "execution/serving/PooledServingSlotOwner.h"
#include "execution/prefix_cache/BlockHash.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
namespace
{
std::uint64_t contentHash(const void *data, std::size_t bytes)
{
    return hashPrefixBytes(data, bytes);
}
} // namespace

PooledServingSlotOwner::PooledServingSlotOwner(PagedKVInstance &instance,
    int slots, int kv_heads, int head_dim, int layers, int page_tokens,
    int admitted_issuances, int readers_per_bank,
    std::optional<RetentionClass> ceiling, std::uint64_t trust_domain,
    std::size_t metadata_capacity, std::size_t retention_capacity)
    : instance_(instance),
      kv_heads_(kv_heads), head_dim_(head_dim), layers_(layers),
      page_tokens_(page_tokens), admitted_issuances_(admitted_issuances),
      readers_per_bank_(readers_per_bank), trust_domain_(trust_domain),
      bindings_(static_cast<std::size_t>(std::max(0, slots))),
      identities_(static_cast<std::size_t>(std::max(0, slots)))
{
    flow_ = std::make_unique<PooledKVPrefixFlow>(instance_,
        metadata_capacity, retention_capacity, ceiling, trust_domain_);
}

bool PooledServingSlotOwner::admitSlot(int slot, std::uint64_t request_id,
    std::string_view retention, std::string_view key,
    const std::vector<int32_t> &prompt, int &restored_tokens,
    std::string &error)
{
    restored_tokens = 0;
    error.clear();
    if (slot < 0 || static_cast<std::size_t>(slot) >= bindings_.size())
    {
        error = "pooled slot out of range: " + std::to_string(slot);
        return false;
    }
    if (prompt.empty())
    {
        error = "pooled slot admission requires a nonempty prompt";
        return false;
    }
    if (bindings_[static_cast<std::size_t>(slot)])
    {
        error = "pooled slot is already live (release before re-admit)";
        return false;
    }
    bindings_[static_cast<std::size_t>(slot)] =
        std::make_unique<PooledKVDecodeBinding>(&instance_.pool(),
            &instance_.table(), request_id, slot, kv_heads_, head_dim_,
            page_tokens_, layers_, admitted_issuances_);
    auto *binding = bindings_[static_cast<std::size_t>(slot)].get();

    // Typed identity policy (byte identity only — the outer layer owns trust
    // isolation): namespace = explicit key hash (0 = unkeyed domain);
    // prefix = prompt content hash.
    const std::uint64_t ns = key.empty()
        ? 0
        : contentHash(key.data(), key.size());
    const auto prefix = contentHash(prompt.data(),
        prompt.size() * sizeof(int32_t));

    // HIT: a record with store-resident extents restores into the fresh
    // binding (route trust/ancestry enforced by the authenticated attach).
    const auto *rec = flow_->record(ns, prefix);
    if (rec && !rec->head_extents.empty() && !rec->tail_extents.empty())
    {
        if (!flow_->restore(ns, prefix, *binding, &error))
        {
            bindings_[static_cast<std::size_t>(slot)].reset();
            return false;
        }
        if (!flow_->setLive(ns, prefix, true))
        {
            error = "pooled restored prefix could not be marked live";
            bindings_[static_cast<std::size_t>(slot)].reset();
            return false;
        }
        identities_[static_cast<std::size_t>(slot)] = {ns, prefix};
        restored_tokens = binding->nextPosition();
        return true;
    }
    // MISS: admit the retention intent (resolver + operator ceiling + typed
    // refusals), mark live.
    auto admitted = flow_->admit(ns, prefix, retention, key);
    if (!admitted)
    {
        error = "pooled retention admission refused (unsupported value, "
                "above the operator ceiling, duplicate or full metadata)";
        bindings_[static_cast<std::size_t>(slot)].reset();
        return false;
    }
    if (!flow_->setLive(ns, prefix, true))
    {
        error = "pooled prefix could not be marked live";
        bindings_[static_cast<std::size_t>(slot)].reset();
        return false;
    }
    identities_[static_cast<std::size_t>(slot)] = {ns, prefix};
    return true;
}

bool PooledServingSlotOwner::demoteSlot(int slot, std::string &error)
{
    error.clear();
    if (slot < 0 || static_cast<std::size_t>(slot) >= bindings_.size())
    {
        error = "pooled slot out of range: " + std::to_string(slot);
        return false;
    }
    auto &binding = bindings_[static_cast<std::size_t>(slot)];
    if (!binding)
        return true; // Nothing to demote (already released).
    const int rows = binding->hasLivePublication()
        ? binding->priorFor(0)->snapshot.view().rows()
        : 0;
    const auto [ns, prefix] = identities_[static_cast<std::size_t>(slot)];
    if (rows <= 0)
    {
        // Empty request: release, nothing to preserve.
        releaseSlot(slot);
        return true;
    }
    // Tail-first deficit: demote every page except the final resident one
    // (the R49 tail-first semantics at the serving boundary; a single-page
    // prefix keeps its one page resident).
    const int total_pages = (rows + page_tokens_ - 1) / page_tokens_;
    if (total_pages > 1)
    {
        if (!flow_->setLive(ns, prefix, false))
        {
            error = "pooled prefix could not be marked idle for demotion";
            return false;
        }
        auto outcome = flow_->demoteTail(ns, prefix, *binding,
            static_cast<std::size_t>(total_pages - 1));
        if (!outcome.ok())
        {
            error = "pooled demotion refused: " + outcome.refusal;
            return false;
        }
    }
    else if (flow_->record(ns, prefix))
    {
        // Single-page prefix: keep it resident (no demotion); the record
        // stays with its extents empty until the next demotion opportunity.
        (void)flow_->setLive(ns, prefix, false);
    }
    releaseSlot(slot);
    return true;
}

void PooledServingSlotOwner::releaseSlot(int slot)
{
    if (slot < 0 || static_cast<std::size_t>(slot) >= bindings_.size())
        return;
    bindings_[static_cast<std::size_t>(slot)].reset();
    identities_[static_cast<std::size_t>(slot)] = {0, 0};
}
} // namespace llaminar2
