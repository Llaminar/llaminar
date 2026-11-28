/**
 * @file PooledServingSlotOwner.h
 * @brief R52 serving-side pooled slot lifecycle owner (P5-boundary wiring).
 *
 * ONE composed paged-KV instance + ONE prefix flow serve every resident
 * serving slot: each slot owns ONE PooledKVDecodeBinding for the lifetime of
 * its request (created at slot admission over the instance's pool/table with
 * the pooled request policy), and the flow owns admission/restore/demotion
 * around it.
 *
 * The typed identity policy is the WIRING's single authority (byte identity
 * only — trust isolation is the outer HTTP layer's job, exactly as the
 * RetentionIntentResolver documents):
 *   namespace_id = hash(provider prompt_cache_key) when present, else 0
 *                  (the unkeyed domain);
 *   prefix_id    = hash(prompt token bytes) — the content identity; a later
 *                  request with the same prompt bytes restores the stored
 *                  prefix (a hit) and a different prompt prefills fresh.
 *
 * Lifecycle (all-or-nothing per slot):
 *   admitSlot  — miss: flow.admit (record admitted, live=true, no restore);
 *                hit:  flow.restore into the slot's FRESH binding (live=true,
 *                restored_tokens = the attachment's row count; the remaining
 *                prompt prefills through the pooled single row-batch);
 *                a restored attach is required to be fresh — the previous
 *                request's binding is RELEASED at retirement, so a reused slot
 *                always starts fresh.
 *   demoteSlot — retirement: live -> false, tail-first deficit demotion of
 *                every page except the final resident one (the R49 tail-first
 *                semantics at the serving boundary), then the slot binding is
 *                released (the next request on the slot is fresh).
 *
 * The owner performs NO page/byte accounting: the instance's pool, table and
 * disk store and the flow's retention owner remain the physical/typed owners
 * (exactly the composed-instance authority of P3 M11).
 */
#pragma once
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
#include "execution/prefix_cache/PooledKVPrefixFlow.h"
#include "planning/PagedKVInstance.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
/** @brief R52 serving-side pooled slot owner. */
class PooledServingSlotOwner final
{
public:
    /**
     * @param instance The composed instance (pool + table + disk route).
     * @param slots Resident serving slots (one binding per slot at a time).
     * @param kv_heads/head_dim/layers The model's pooled KV geometry (the
     *        same geometry as the admitted KV cache — the binding requests it
     *        exactly at construction).
     * @param page_tokens Pooled page granularity.
     * @param admitted_issuances Per-request accepted-work envelope (>= the
     *        request's forwards: prefill chunks + decode steps + margins).
     * @param readers_per_bank Table reader headroom per bank.
     * @param ceiling Operator retention-class ceiling (the flow resolver
     *        enforces it at admission).
     * @param trust_domain The local route's authenticated trust domain
     *        (stamped at demotion, enforced at restore; never 0).
     * @param metadata_capacity/retention_capacity Bounded flow membership.
     */
    PooledServingSlotOwner(PagedKVInstance &instance, int slots,
        int kv_heads, int head_dim, int layers, int page_tokens,
        int admitted_issuances, int readers_per_bank,
        std::optional<RetentionClass> ceiling, std::uint64_t trust_domain,
        std::size_t metadata_capacity, std::size_t retention_capacity);

    /** @brief Admit one request on @p slot (miss or restored hit); see the
     * file header for the identity policy. @p restored_tokens is the restored
     * prefix row count on a hit, 0 on a miss. */
    bool admitSlot(int slot, std::uint64_t request_id,
        std::string_view retention, std::string_view key,
        const std::vector<int32_t> &prompt, int &restored_tokens,
        std::string &error);

    /** @brief Retirement: demote the finished request's prefix to the disk
     * tier (tail-first, keep one resident page) and RELEASE the slot binding
     * (the next request on the slot starts fresh). */
    bool demoteSlot(int slot, std::string &error);

    /** @brief Drop the slot binding without demotion (cancellation path). */
    void releaseSlot(int slot);

    /** @return The slot's live binding (nullptr after release). */
    PooledKVDecodeBinding *bindingFor(int slot) const noexcept
    {
        return slot >= 0 && static_cast<std::size_t>(slot) < bindings_.size()
            ? bindings_[static_cast<std::size_t>(slot)].get()
            : nullptr;
    }

    /** @return The owned prefix flow (the one typed retention path). */
    PooledKVPrefixFlow &flow() noexcept { return *flow_; }
    const PooledKVPrefixFlow &flow() const noexcept { return *flow_; }

    /** @return Sampled level count (no-op raw accounting; the flow owns the
     * record semantics). */
    int slots() const noexcept { return static_cast<int>(bindings_.size()); }

    /** @brief Phase classification for one serving forward call shape (the
     * DGO contract): true = pooled GROUP decode batch (every row one token);
     * false = pooled SINGLE prefill row-batch (exactly one row-batch). */
    static bool isDecodePhase(const std::vector<std::vector<int>> &rows) noexcept
    {
        return rows.size() > 1 ||
            (rows.size() == 1 && rows[0].size() == 1);
    }

private:
    PagedKVInstance &instance_;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    int layers_ = 0;
    int page_tokens_ = 0;
    int admitted_issuances_ = 0;
    int readers_per_bank_ = 0;
    std::uint64_t trust_domain_ = 0;
    std::vector<std::unique_ptr<PooledKVDecodeBinding>> bindings_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> identities_;
    std::unique_ptr<PooledKVPrefixFlow> flow_;
};
} // namespace llaminar2
