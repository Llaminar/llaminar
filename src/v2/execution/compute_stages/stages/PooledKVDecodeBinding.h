/**
 * @file PooledKVDecodeBinding.h
 * @brief Runner-owned pooled-KV decode binding: one table+pool, per-layer cells.
 *
 * This is the P2 production caller seam. The orchestrator owns one binding per
 * pooled decode request (CPU single-row); the model graph consumes the table
 * and the per-layer `PooledPublicationCell` ref cells. The append stage writes
 * the cell during graph execution; the attention stage reads the same cell in
 * the next node, so both stages see ONE publication and the Snapshot keeping
 * the bank alive is the cell's own.
 *
 * The binding holds a NON-owning view of the admitted instance: the
 * orchestrator owns the `PagedKVInstance` and guarantees the binding lives no
 * longer than it. The cells themselves own no memory beyond the retained
 * snapshot. Binding is neither a physical ledger nor a second admission.
 */
#pragma once
#include "execution/compute_stages/stages/PooledKVAppendStage.h"
#include "kernels/PagedKVSnapshotRef.h"
#include "planning/PhysicalPagePool.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include <vector>

namespace llaminar2
{
class PooledKVDecodeBinding final
{
public:
    PooledKVDecodeBinding() = default;
    PooledKVDecodeBinding(PooledKVDecodeBinding &&) = default;
    PooledKVDecodeBinding &operator=(PooledKVDecodeBinding &&) = default;
    PooledKVDecodeBinding(const PooledKVDecodeBinding &) = delete;
    PooledKVDecodeBinding &operator=(const PooledKVDecodeBinding &) = delete;

    /** @brief Bind one admitted pooled instance for one decode request. */
    PooledKVDecodeBinding(PhysicalPagePool *pool, CPUFP32PagedKVTable *table,
        std::uint64_t request_id, int slot, int kv_heads, int head_dim,
        int page_tokens, int layers)
        : pool_(pool), table_(table), request_id_(request_id), slot_(slot),
          kv_heads_(kv_heads), head_dim_(head_dim), page_tokens_(page_tokens),
          cells_(static_cast<std::size_t>(layers))
    {
    }

    PhysicalPagePool *pool() const noexcept { return pool_; }
    CPUFP32PagedKVTable *table() const noexcept { return table_; }
    std::uint64_t requestId() const noexcept { return request_id_; }
    int slot() const noexcept { return slot_; }
    int kvHeads() const noexcept { return kv_heads_; }
    int headDim() const noexcept { return head_dim_; }
    int pageTokens() const noexcept { return page_tokens_; }
    /** @return Ref cell of one layer; null when the layer is out of range. */
    PooledPublicationCell *cellFor(int layer) noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return nullptr;
        return &cells_[static_cast<std::size_t>(layer)];
    }
    PooledPublicationCell *cellFor(int layer) const noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return nullptr;
        return const_cast<PooledPublicationCell *>(&cells_[static_cast<std::size_t>(layer)]);
    }
    /** @return Prior publication of one layer (null before the first append). */
    PooledPublicationCell *priorFor(int layer) noexcept
    {
        auto *cell = cellFor(layer);
        return cell && cell->live() ? cell : nullptr;
    }
    PooledPublicationCell *priorFor(int layer) const noexcept
    {
        auto *cell = cellFor(layer);
        return cell && cell->live() ? cell : nullptr;
    }

private:
    PhysicalPagePool *pool_ = nullptr;
    CPUFP32PagedKVTable *table_ = nullptr;
    std::uint64_t request_id_ = 0;
    int slot_ = 0;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    int page_tokens_ = 0;
    std::vector<PooledPublicationCell> cells_;
};
} // namespace llaminar2
