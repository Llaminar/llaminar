/**
 * @file PooledKVDecodeBinding.h
 * @brief Runner-owned pooled-KV decode binding: one table+pool, per-layer twin cells.
 *
 * This is the P2 production caller seam. The orchestrator owns one binding per
 * pooled decode request (CPU single-row); the model graph consumes the table
 * and per-layer publication cells. Ownership rule (M1): each layer keeps TWO
 * cells — slot 0 is the CURRENT publication (the prior), slot 1 is the NEXT
 * cell the current graph writes. The append stage never clears the prior cell;
 * attention reads the next cell inside the same graph. Only after the whole
 * forward succeeds does `advanceStep()` move next→current (releasing the old
 * prior after the new publication exists); on failure `cancelStep()` drops the
 * next cells, so no layer publication becomes visible unless every layer
 * published (M6).
 *
 * Grant rule (M6): the runner admits the step's COMPLETE page demand, for all
 * layers, before the graph runs (`prepareStep`), under one pool grant. The
 * append stages consume frames from that grant; a page shortage refuses before
 * any graph executes. The binding is neither a physical ledger nor a second
 * admission authority: the pool grant is the only ledger.
 *
 * The binding holds a NON-owning view of the admitted instance: the
 * orchestrator owns the `PagedKVInstance` and guarantees the binding lives no
 * longer than it. The cells own no memory beyond the retained snapshots.
 *
 * Decode position authority (review R24): the binding is the single decode
 * position authority. `nextPosition()` is the live publication row count (the
 * absolute position of the next row). The orchestrator publishes it as the
 * graph position_offset and asserts the host cursor agrees; no other code
 * derives pooled decode positions.
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
    int layerCount() const noexcept { return static_cast<int>(cells_.size()); }

    /**
     * @brief Cell the CURRENT graph appends into and attention reads (next).
     * @return Ref cell of one layer, or null when the layer is out of range.
     */
    PooledPublicationCell *nextCell(int layer) noexcept
    {
        auto *pair = pairFor(layer);
        return pair ? &pair->cells[1] : nullptr;
    }
    PooledPublicationCell *nextCell(int layer) const noexcept
    {
        auto *pair = pairFor(layer);
        return pair ? &const_cast<PooledPublicationCellPair *>(pair)->cells[1] : nullptr;
    }
    /**
     * @brief Prior publication of one layer (slot 0, the live current).
     * @return The prior cell, or null before the first append.
     */
    PooledPublicationCell *priorFor(int layer) noexcept
    {
        auto *pair = pairFor(layer);
        if (!pair) return nullptr;
        return pair->cells[0].live() ? &pair->cells[0] : nullptr;
    }
    PooledPublicationCell *priorFor(int layer) const noexcept
    {
        auto *pair = pairFor(layer);
        if (!pair) return nullptr;
        return pair->cells[0].live() ? &const_cast<PooledPublicationCellPair *>(pair)->cells[0]
                                     : nullptr;
    }
    /** @return Whether any layer already holds a live prior publication. */
    bool hasLivePublication() const noexcept
    {
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
            if (cells_[layer].cells[0].live())
                return true;
        return false;
    }

    /**
     * @brief Complete page demand of one layer for appending @p rows.
     * @return Pages needed (COW extension + fresh pages), never zero for rows>0.
     */
    int pagesForStep(int rows) const
    {
        if (rows <= 0) throw std::invalid_argument("Pooled step requires positive rows");
        std::size_t tail = 0;
        bool has_prior = false;
        if (!cells_.empty())
        {
            const auto &prior = cells_.front().cells[0];
            if (prior.live())
            {
                const auto pages = prior.snapshot.view().pages();
                if (!pages.empty())
                {
                    has_prior = true;
                    tail = static_cast<std::size_t>(pages.back().rows());
                }
            }
        }
        if (!has_prior || tail == static_cast<std::size_t>(page_tokens_))
        {
            // No prior (or a full tail): pure fresh pages.
            return (rows + page_tokens_ - 1) / page_tokens_;
        }
        // Partial tail: one COW extension fills it, remaining rows are fresh.
        const int tail_room = page_tokens_ - static_cast<int>(tail);
        const int rows_after_tail = rows - tail_room;
        return 1 + (rows_after_tail > 0 ? (rows_after_tail + page_tokens_ - 1) / page_tokens_ : 0);
    }

    /**
     * @brief Admit the COMPLETE page demand of one step before the graph runs.
     * @param rows Live rows this step appends (prefill width or decode 1).
     * @return False on page pressure; the binding mutates nothing and the
     * orchestrator must refuse the forward before building/executing a graph.
     * @note The grant is one pool request record covering every layer; the
     * append stages consume frames in layer order, so the total matches the
     * admitted demand exactly when the graph is well-formed.
     */
    bool prepareStep(int rows)
    {
        if (!pool_) return false;
        const int per_layer = pagesForStep(rows);
        const int total = per_layer * static_cast<int>(cells_.size());
        step_grant_ = pool_->tryAdmit(static_cast<std::size_t>(total));
        return static_cast<bool>(step_grant_) && step_grant_->remainingCommitment() >=
            static_cast<std::size_t>(total);
    }
    /** @return The pre-admitted step grant (null until prepareStep succeeds). */
    PhysicalPagePool::Grant *stepGrant() noexcept
    {
        return step_grant_ ? &*step_grant_ : nullptr;
    }
    bool stepPrepared() const noexcept { return static_cast<bool>(step_grant_); }

    /**
     * @brief Publish the step: next→current for every layer simultaneously.
     *
     * Called only after the complete forward graph succeeded. The new
     * publications already exist (bound in the table); this releases each
     * layer's prior only now, after the replacement is visible.
     */
    void advanceStep() noexcept
    {
        for (auto &pair : cells_)
            pair.cells[0] = std::move(pair.cells[1]);
        step_grant_.reset();
    }

    /**
     * @brief Drop every not-yet-published next cell (failure rollback).
     *
     * No layer publication becomes visible; each layer's prior (current) cell
     * is untouched and remains the live history. The step grant's unused
     * commitment returns to the pool.
     */
    void cancelStep() noexcept
    {
        for (auto &pair : cells_)
            pair.cells[1] = {};
        step_grant_.reset();
    }

    /**
     * @brief Absolute position of the next appended row (single authority).
     * @return Current live publication rows; 0 before any publication.
     */
    int nextPosition() const noexcept
    {
        if (cells_.empty()) return 0;
        const auto &prior = cells_.front().cells[0];
        if (!prior.live()) return 0;
        return prior.snapshot.view().rows();
    }

private:
    struct PooledPublicationCellPair
    {
        PooledPublicationCell cells[2]{}; ///< [0]=live prior, [1]=this step's next.
    };
    PooledPublicationCellPair *pairFor(int layer) noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return nullptr;
        return &cells_[static_cast<std::size_t>(layer)];
    }
    const PooledPublicationCellPair *pairFor(int layer) const noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return nullptr;
        return &cells_[static_cast<std::size_t>(layer)];
    }

    PhysicalPagePool *pool_ = nullptr;
    CPUFP32PagedKVTable *table_ = nullptr;
    std::uint64_t request_id_ = 0;
    int slot_ = 0;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    int page_tokens_ = 0;
    std::vector<PooledPublicationCellPair> cells_;
    std::optional<PhysicalPagePool::Grant> step_grant_;
};
} // namespace llaminar2
