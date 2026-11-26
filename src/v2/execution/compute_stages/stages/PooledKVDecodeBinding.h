/**
 * @file PooledKVDecodeBinding.h
 * @brief Runner-owned pooled-KV decode binding: one table+pool, per-layer twin cells.
 *
 * This is the P2 production caller seam. The orchestrator owns one binding per
 * pooled decode request (CPU single-row); the model graph consumes the table
 * and per-layer publication cells. Ownership rule (M1): each layer keeps TWO
 * cells — slot 0 is the CURRENT publication (the prior), slot 1 is the NEXT
 * cell the current graph writes. The append stage never clears the prior cell;
 * attention reads the next cell inside the same graph. Only a fully successful
 * forward advances next→current (`StepGuard::completeAndAdvance`), releasing
 * the old prior after the new publication exists; any unwind or failed forward
 * drops the next cells, so no layer publication becomes visible unless every
 * layer published (M6, N1).
 *
 * Exception safety (N1): `prepareStep` opens a move-only `StepGuard` that
 * OWNS the step and `cancelStep()`s on destruction unless completed. Any throw
 * between prepareStep and completeAndAdvance (graph construction, execution,
 * layer failure) releases the step grant and the not-yet-visible cells.
 * `completeAndAdvance` requires EVERY layer's next cell to be live with exactly
 * prior rows + step rows; otherwise it cancels and fails (missing-layer
 * negative).
 *
 * Admission (M6 + N2): `prepareStep` admits the step's COMPLETE page demand
 * for all layers under one pool grant AND checks the table's accepted-work
 * issuance reservation (versions) plus per-layer bank headroom BEFORE the graph
 * runs. The reservation is admitted once when the binding is constructed from
 * the admitted request envelope (`admitted_issuances` = layers × total
 * forwards); versions are consumed per layer per step; exhaustion refuses
 * before any graph runs, so a request drains (or is refused) before the
 * finite budget is exhausted.
 *
 * The binding holds a NON-owning view of the admitted instance: the
 * orchestrator owns the `PagedKVInstance` and guarantees the binding lives no
 * longer than it. The cells own no memory beyond the retained snapshots.
 *
 * Decode position authority (R24): the binding is the single decode position
 * authority. `nextPosition()` is the live publication row count. The
 * orchestrator publishes it as the graph position_offset and asserts the host
 * cursor agrees; no other code derives pooled decode positions.
 */
#pragma once
#include "execution/compute_stages/stages/PooledKVAppendStage.h"
#include "kernels/PagedKVSnapshotRef.h"
#include "planning/PhysicalPagePool.h"
#include "planning/PagedKVInstancePlan.h"
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
        int page_tokens, int layers, std::size_t admitted_issuances)
        : pool_(pool), table_(table), request_id_(request_id), slot_(slot),
          kv_heads_(kv_heads), head_dim_(head_dim), page_tokens_(page_tokens),
          admitted_issuances_(admitted_issuances),
          cells_(static_cast<std::size_t>(layers))
    {
        if (!pool_ || !table_)
            throw std::invalid_argument("Pooled decode binding requires the admitted pool and table");
        // N2: the request envelope prices the table geometry. The table's
        // issuance HEADROOM is the accepted-work budget and must hold
        // layers × total issuances; the physical bank count must hold the
        // two live banks per layer (prior + next). A geometry that cannot
        // admit the accepted work is refused here, before any forward.
        if (table_->bankCount() < kRequiredBanksPerLayer * static_cast<std::size_t>(layers))
            throw std::invalid_argument("Pooled table bank geometry cannot hold two banks per layer");
        if (table_->tableGeometry().issuance_headroom <
            seatedVersionDemand())
            throw std::invalid_argument("Pooled table issuance headroom is smaller than the admitted request envelope");
        // The request's WHOLE version budget is reserved now (P1 accepted
        // work); a table that cannot admit it refuses at binding time.
        reservation_ = table_->reserveAcceptedWork(seatedVersionDemand());
        if (!reservation_)
            throw std::invalid_argument("Pooled table cannot reserve the admitted request version envelope");
    }

    /**
     * @brief Derive the planning policy for one pooled request from its
     * admitted envelope (never test constants).
     * @param page_tokens Native block size.
     * @param layers Attended transformer layers (each holds two live banks).
     * @param admitted_issuances Total table issuances the request may consume
     * (prefill + decode forwards, × layers).
     * @param readers_per_bank Table reader envelope per bank.
     * @return A policy whose bank_count and issuance budget seat the request.
     */
    static PagedKVInstancePolicy requestPolicy(int page_tokens, int layers,
        std::size_t admitted_issuances, int readers_per_bank = 4)
    {
        if (page_tokens <= 0 || layers <= 0 || !admitted_issuances || readers_per_bank <= 0)
            throw std::invalid_argument("Pooled request policy requires positive exact envelope");
        // bank_count seats 2 banks per layer AND the whole accepted-work
        // headroom (headroom == bank_count in the instance plan algebra).
        const std::size_t banks_required = admitted_issuances >
            kRequiredBanksPerLayer * static_cast<std::size_t>(layers)
            ? admitted_issuances
            : kRequiredBanksPerLayer * static_cast<std::size_t>(layers);
        return PagedKVInstancePolicy{page_tokens, banks_required,
            readers_per_bank, /*request_records=*/static_cast<int>(layers + 1)};
    }

    PhysicalPagePool *pool() const noexcept { return pool_; }
    CPUFP32PagedKVTable *table() const noexcept { return table_; }
    std::uint64_t requestId() const noexcept { return request_id_; }
    int slot() const noexcept { return slot_; }
    int kvHeads() const noexcept { return kv_heads_; }
    int headDim() const noexcept { return head_dim_; }
    int pageTokens() const noexcept { return page_tokens_; }
    int layerCount() const noexcept { return static_cast<int>(cells_.size()); }
    std::size_t admittedIssuances() const noexcept { return admitted_issuances_; }
    /** @return Versions this request may consume (layers × total issuances). */
    std::size_t seatedVersionDemand() const noexcept
    {
        return static_cast<std::size_t>(cells_.size()) * admitted_issuances_;
    }
    /** @return Remaining accepted-work issuances of the request envelope. */
    std::size_t remainingIssuances() const noexcept
    {
        return reservation_ ? reservation_->remaining() : 0;
    }

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

    /** @brief RAII step ownership: cancels on unwind unless completed. */
    class StepGuard final
    {
    public:
        StepGuard() = default;
        ~StepGuard() { cancel(); }
        StepGuard(StepGuard &&other) noexcept
            : binding_(std::exchange(other.binding_, nullptr)) {}
        StepGuard &operator=(StepGuard &&other) noexcept
        {
            if (this != &other) { cancel(); binding_ = std::exchange(other.binding_, nullptr); }
            return *this;
        }
        StepGuard(const StepGuard &) = delete;
        StepGuard &operator=(const StepGuard &) = delete;
        explicit operator bool() const noexcept { return binding_ != nullptr; }

        /**
         * @brief Validate EVERY layer published, then advance next→current.
         * @return True when every layer's next cell is live with exactly
         * prior rows + step rows; the step is then committed (ownership
         * released). False (with the step cancelled) on any missing layer,
         * wrong row count or empty publication.
         */
        bool completeAndAdvance() noexcept
        {
            if (!binding_) return false;
            if (!binding_->stepComplete())
            {
                cancel();
                return false;
            }
            binding_->advanceStep();
            binding_ = nullptr;
            return true;
        }
        /** @brief Explicitly cancel the step (releases grant + next cells). */
        void cancel() noexcept
        {
            if (binding_) { binding_->cancelStep(); binding_ = nullptr; }
        }
    private:
        friend class PooledKVDecodeBinding;
        explicit StepGuard(PooledKVDecodeBinding *binding) : binding_(binding) {}
        PooledKVDecodeBinding *binding_ = nullptr;
    };

    /**
     * @brief Admit one step: complete page demand + version + bank headroom.
     * @param rows Live rows this step appends (prefill width or decode 1).
     * @return A StepGuard owning the step, or an empty guard when admission
     * refuses (page pressure, version exhaustion, bank headroom). The
     * orchestrator must refuse the forward before building/executing a graph.
     */
    StepGuard prepareStep(int rows)
    {
        if (!pool_ || !reservation_) return StepGuard{};
        const std::size_t layers = cells_.size();
        // N2: the accepted-work version envelope must cover THIS step.
        if (reservation_->remaining() < layers)
            return StepGuard{};
        // N2: bank headroom — every layer holds its prior bank plus this
        // step's next bank; readers share existing banks, never new ones.
        const auto demand = table_->liveClassDemand();
        if (demand.occupied_banks + kRequiredBanksPerLayer * layers > table_->bankCount())
            return StepGuard{};
        const int per_layer = pagesForStep(rows);
        const int total = per_layer * static_cast<int>(layers);
        step_grant_ = pool_->tryAdmit(static_cast<std::size_t>(total));
        if (!step_grant_ || step_grant_->remainingCommitment() <
            static_cast<std::size_t>(total))
        {
            step_grant_.reset();
            return StepGuard{};
        }
        step_rows_ = rows;
        return StepGuard(this);
    }
    /** @return The pre-admitted step grant (null until prepareStep succeeds). */
    PhysicalPagePool::Grant *stepGrant() noexcept
    {
        return step_grant_ ? &*step_grant_ : nullptr;
    }
    bool stepPrepared() const noexcept { return static_cast<bool>(step_grant_); }
    /** @return The accepted-work issuance reservation consumed per layer per step. */
    CPUFP32PagedKVTable::IssuanceReservation *issuance() noexcept
    {
        return reservation_ ? &*reservation_ : nullptr;
    }

    /**
     * @brief Whether every layer published next = prior rows + step rows.
     */
    bool stepComplete() const noexcept
    {
        if (!step_grant_ || step_rows_ <= 0) return false;
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            const auto &next = cells_[layer].cells[1];
            if (!next.live())
                return false;
            // N1: exact row contract — prior rows + this step's rows. A layer
            // that silently skipped its append cannot pass.
            const int prior_rows = cells_[layer].cells[0].live()
                ? cells_[layer].cells[0].snapshot.view().rows() : 0;
            if (next.snapshot.view().rows() != prior_rows + step_rows_)
                return false;
        }
        return true;
    }

    /**
     * @brief Publish the step: next→current for every layer simultaneously.
     * Only called after `stepComplete()` (StepGuard::completeAndAdvance).
     * The new publications already exist (bound in the table); this releases
     * each layer's prior only now, after the replacement is visible.
     */
    void advanceStep() noexcept
    {
        for (auto &pair : cells_)
            pair.cells[0] = std::move(pair.cells[1]);
        step_grant_.reset();
        step_rows_ = 0;
    }

    /**
     * @brief Drop every not-yet-published next cell (failure rollback).
     * No layer publication becomes visible; each layer's prior (current) cell
     * is untouched. The step grant's unused commitment returns to the pool.
     */
    void cancelStep() noexcept
    {
        for (auto &pair : cells_)
            pair.cells[1] = {};
        step_grant_.reset();
        step_rows_ = 0;
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
    static constexpr std::size_t kRequiredBanksPerLayer = 2;
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
    std::size_t admitted_issuances_ = 0;
    std::vector<PooledPublicationCellPair> cells_;
    std::optional<CPUFP32PagedKVTable::IssuanceReservation> reservation_;
    std::optional<PhysicalPagePool::Grant> step_grant_;
    int step_rows_ = 0;
};
} // namespace llaminar2
