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
        int page_tokens, int layers, std::size_t admitted_issuances,
        // Economic decision (measured): the persistent in-place private tail
        // copies 268 bytes/step vs 1280 for the COW extension (TailCopyBytesMeasurement;
        // the in-place path never re-copies prior tail rows), so it is the
        // production default. Callers needing the COW observation arm pass
        // false explicitly (the measurement test drives both).
        bool persistent_tail = true)
        : pool_(pool), table_(table), request_id_(request_id), slot_(slot),
          kv_heads_(kv_heads), head_dim_(head_dim), page_tokens_(page_tokens),
          admitted_issuances_(admitted_issuances),
          cells_(static_cast<std::size_t>(layers)),
          persistent_tail_(persistent_tail)
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
        // In-place private tail (P2 economy): ONE persistent frame per layer is
        // admitted from the pool ONCE at binding time. It is the request's
        // exclusive tail storage; every step extends it in place while it
        // remains private and exclusively owned (design: "every writer owns an
        // exclusive tail or COW destination"). A pool that cannot seat the
        // request's tail frames refuses at binding time — persistent storage
        // is sized from admitted membership, never allocated in the hot path.
        if (persistent_tail_)
        {
            // The request's peak is its admitted per-request page bound; only
            // its tail frames (one per layer) are protected NOW. Further
            // growth is witnessed (extendWithWitness) inside prepareStep, so
            // a small live request never pre-reserves its whole window.
            const auto peak = pool_->geometry().max_pages_per_request;
            tail_grant_ = pool_->tryAdmitTail(static_cast<std::size_t>(layers), peak);
            if (!tail_grant_)
                throw std::invalid_argument("Pooled request tail frames cannot be admitted from the envelope");
            tail_frames_.assign(static_cast<std::size_t>(layers), std::nullopt);
            fresh_tail_frames_.assign(static_cast<std::size_t>(layers), std::nullopt);
            for (int layer = 0; layer < layers; ++layer)
            {
                try { tail_frames_[static_cast<std::size_t>(layer)] = tail_grant_->assignFrame(); }
                catch (const std::exception &e)
                {
                    throw std::invalid_argument(
                        std::string("Pooled request tail frame assignment refused: ") + e.what());
                }
            }
        }
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
     * @brief Whether this request still holds a live exclusive tail grant.
     *
     * The tail grant is this request's PERSISTENT private storage: one frame
     * per layer, admitted ONCE at binding construction from the admitted
     * envelope (never a hot-path admission), owned for the full request
     * lifetime. The partial tail page is extended in place through it; only
     * full pages become shared history.
     */
    bool hasTailGrant() const noexcept { return static_cast<bool>(tail_grant_); }
    /** @return The persistent exclusive-tail grant (frames owned per layer). */
    PhysicalPagePool::Grant *tailGrant() noexcept
    {
        return tail_grant_ ? &*tail_grant_ : nullptr;
    }
    /** @return The persistent exclusive-tail frame of layer @p layer (or null). */
    PhysicalPagePool::FrameHandle *tailFrame(int layer) noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= tail_frames_.size())
            return nullptr;
        auto &frame = tail_frames_[static_cast<std::size_t>(layer)];
        return frame ? &*frame : nullptr;
    }
    /**
     * @return The FRESH tail frame rolled for this step's final partial page
     * (null when none was rolled; the stage then uses @ref tailFrame).
     */
    PhysicalPagePool::FrameHandle *freshTailFrame(int layer) noexcept
    {
        if (layer < 0 || static_cast<std::size_t>(layer) >= fresh_tail_frames_.size())
            return nullptr;
        auto &frame = fresh_tail_frames_[static_cast<std::size_t>(layer)];
        return frame ? &*frame : nullptr;
    }
    /**
     * @brief Whether layer @p layer's persistent tail frame is an EMPTY
     * private frame (the next partial tail page should be written there).
     */
    bool tailFrameReady(int layer) noexcept
    {
        if (!pool_ || !tail_grant_) return false;
        auto *frame = tailFrame(layer);
        return frame && pool_->tailFrameEmpty(*tail_grant_, *frame);
    }
    /**
     * @brief Whether layer @p layer's persistent tail frame holds the CURRENT
     * private publication of the prior tail page (in-place extension legal).
     */
    bool tailHeldByFrame(int layer) noexcept
    {
        if (!pool_ || !tail_grant_) return false;
        auto *frame = tailFrame(layer);
        return frame && pool_->tailFrameExclusive(*tail_grant_, *frame);
    }
    /**
     * @brief Roll one layer's tail frame over to a FRESH empty frame for this
     * step's final partial page (recorded in @p fresh_tail_frames_).
     *
     * When the pre-assigned tail frame is still EMPTY (fresh start) no roll is
     * needed and @p fresh_tail_frames_ stays empty; the stage uses the
     * pre-assigned @ref tailFrame. When it holds a full/shared publication
     * (in-place tail filled, or a full page rolled into history), a new frame
     * is grown through the all-resource witness and recorded as the fresh
     * destination. The frame holding the CURRENT prior tail is left untouched
     * for the in-place extension.
     * @return True when a fresh empty tail destination exists; false on
     * refusal (the caller then falls back to the COW/fresh-page path).
     */
    bool rollTailFrame(int layer)
    {
        if (!pool_ || !tail_grant_) return false;
        if (layer < 0 || static_cast<std::size_t>(layer) >= tail_frames_.size())
            return false;
        if (tailFrameReady(layer))
            return true; // Pre-assigned empty frame is the destination (no roll).
        auto &fresh = fresh_tail_frames_[static_cast<std::size_t>(layer)];
        if (fresh && pool_->tailFrameEmpty(*tail_grant_, *fresh))
            return true; // Already rolled for this step.
        try
        {
            if (!pool_->extendWithWitness(*tail_grant_, 1))
                return false;
            auto handle = tail_grant_->assignFrame();
            fresh = handle;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
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
     * @brief Cancel the pooled request and retire ALL its state exactly.
     *
     * Cancellation under a running step: the step guard cancels on unwind
     * (N1); an explicit cancellation between steps drops every not-yet-
     * published next cell, releases the step grant and the request's
     * accepted-work envelope, and drops every pre-assigned tail frame
     * (persistent tail) WITHOUT releasing frames another reader still
     * borrows — the pool retires each frame after its last pin. After this
     * call the binding is inert: prepareStep/export/truncate refuse.
     */
    void cancelRequest() noexcept
    {
        step_grant_.reset();
        step_rows_ = 0;
        step_prepared_ = false;
        for (auto &pair : cells_)
        {
            pair.cells[1] = {};
            pair.cells[0] = {};
        }
        for (auto &fresh : fresh_tail_frames_)
            fresh.reset();
        tail_frames_.clear();
        fresh_tail_frames_.clear();
        tail_grant_.reset();
        private_publication_grant_.reset();
        reservation_.reset();
        cancelled_ = true;
    }
    /** @return Whether the request was explicitly cancelled. */
    bool cancelled() const noexcept { return cancelled_; }

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
     * @brief Per-layer tail mode of the step the orchestrator is about to run.
     */
    enum class StepTailMode : std::uint8_t { None, InPlace, FinalFresh };
    /**
     * @return Whether the prior tail page is EXACTLY this layer's tail frame
     * (in-place extension legal) and the step may extend it.
     */
    StepTailMode stepTailMode(int layer, int rows) noexcept
    {
        if (!persistent_tail_ || !tail_grant_ || rows <= 0)
            return StepTailMode::None;
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return StepTailMode::None;
        const auto &prior = cells_[static_cast<std::size_t>(layer)].cells[0];
        if (!prior.live())
        {
            // Fresh start: a partial final page lands on the (empty) tail frame.
            return (rows % page_tokens_ != 0) && tailHeldByFrame(layer)
                ? StepTailMode::FinalFresh : StepTailMode::None;
        }
        const auto pages = prior.snapshot.view().pages();
        if (pages.empty()) return StepTailMode::None;
        const int tail_rows = pages.back().rows();
        if (tail_rows < page_tokens_)
            return tailHeldByFrame(layer) ? StepTailMode::InPlace : StepTailMode::None;
        // The prior tail is FULL: its private page is shared into the new bank
        // (frame.converts to shared) and a fresh tail frame takes the final
        // partial page of this step (rolled at prepareStep).
        return (rows % page_tokens_ != 0) ? StepTailMode::FinalFresh : StepTailMode::None;
    }
    /**
     * @return Whether the step's LAST page (partial) must be written into the
     * persistent tail frame (FinalFresh, or an InPlace step whose remainder
     * ends in a partial page after the in-place fill).
     */
    bool stepFinishesOnTailFrame(int layer, int rows) noexcept
    {
        const auto mode = stepTailMode(layer, rows);
        if (mode == StepTailMode::FinalFresh) return true;
        if (mode == StepTailMode::InPlace)
        {
            const auto &prior = cells_[static_cast<std::size_t>(layer)].cells[0];
            if (!prior.live()) return false;
            const auto pages = prior.snapshot.view().pages();
            if (pages.empty()) return false;
            const int tail_rows = pages.back().rows();
            const int tail_room = page_tokens_ - tail_rows;
            const int remaining = rows - tail_room;
            return remaining > 0 && (remaining % page_tokens_) != 0;
        }
        return false;
    }
    /**
     * @brief Prepare the step's tail plan: roll tail frames for FinalFresh and
     * admit only the fresh pages NOT covered by the exclusive tail extension.
     * @param rows Live rows this step appends.
     * @return Fresh page count for the step grant, or -1 when the tail plan
     * cannot be completed (the caller then uses the COW fallback page count).
     */
    int freshPagesWithTail(int rows)
    {
        if (!persistent_tail_ || !tail_grant_ || rows <= 0)
            return pagesForStep(rows);
        // Decide per layer; all layers advance symmetrically in this caller,
        // so consult layer 0 and apply the plan to every layer.
        const auto mode = stepTailMode(0, rows);
        if (mode == StepTailMode::InPlace)
        {
            const int tail_rows = cells_.front().cells[0].live()
                ? cells_.front().cells[0].snapshot.view().pages().back().rows() : 0;
            const int tail_room = page_tokens_ - tail_rows;
            const int remaining = rows - tail_room;
            if (remaining <= 0)
                return 0; // Extension covers everything; no fresh page admitted.
            // The final partial page must be written into a fresh tail frame.
            const bool final_partial = (remaining % page_tokens_ != 0);
            const int pages = (remaining + page_tokens_ - 1) / page_tokens_;
            if (final_partial)
            {
                for (std::size_t layer = 0; layer < tail_frames_.size(); ++layer)
                    if (!rollTailFrame(static_cast<int>(layer)))
                        return pagesForStep(rows); // Fallback: full COW with admission.
                return pages - 1;
            }
            return pages;
        }
        if (mode == StepTailMode::FinalFresh)
        {
            const bool final_partial = (rows % page_tokens_ != 0);
            const int pages = (rows + page_tokens_ - 1) / page_tokens_;
            if (final_partial)
            {
                for (std::size_t layer = 0; layer < tail_frames_.size(); ++layer)
                    if (!rollTailFrame(static_cast<int>(layer)))
                        return pagesForStep(rows);
                return pages - 1;
            }
            return pages;
        }
        return pagesForStep(rows);
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
        if (cancelled_ || !pool_ || !reservation_) return StepGuard{};
        const std::size_t layers = cells_.size();
        // N2: the accepted-work version envelope must cover THIS step.
        if (reservation_->remaining() < layers)
        {
            LOG_ERROR("[PooledKVDecodeBinding] Step admission refused: version envelope exhausted"
                      << " (remaining=" << reservation_->remaining()
                      << " layers=" << layers << ")");
            return StepGuard{};
        }
        // N2: bank headroom — every layer holds its prior bank plus this
        // step's next bank; readers share existing banks, never new ones.
        const auto demand = table_->liveClassDemand();
        if (demand.occupied_banks + kRequiredBanksPerLayer * layers > table_->bankCount())
        {
            LOG_ERROR("[PooledKVDecodeBinding] Step admission refused: bank headroom"
                      << " (occupied=" << demand.occupied_banks
                      << " required=" << kRequiredBanksPerLayer * layers
                      << " banks=" << table_->bankCount() << ")");
            return StepGuard{};
        }
        // Page demand: fresh pages only. The exclusive tail extends the
        // request's PERSISTENT tail frames (already admitted at binding); a
        // step fully covered by the tail admits NO step grant. A step with a
        // fresh remainder admits exactly the fresh page count (M6: the
        // complete fresh demand before the graph runs).
        const int fresh = freshPagesWithTail(rows);
        if (fresh > 0)
        {
            step_grant_ = pool_->tryAdmit(static_cast<std::size_t>(fresh));
            if (!step_grant_ || step_grant_->remainingCommitment() <
                static_cast<std::size_t>(fresh))
            {
                LOG_ERROR("[PooledKVDecodeBinding] Step admission refused: page pressure"
                          << " (fresh=" << fresh
                          << " frames=" << pool_->geometry().frames
                          << " max_per_request=" << pool_->geometry().max_pages_per_request
                          << " records=" << pool_->geometry().request_records << ")");
                step_grant_.reset();
                return StepGuard{};
            }
        }
        else
            step_grant_.reset(); // Tail-only step: no fresh frames needed.
        step_rows_ = rows;
        step_prepared_ = true;
        return StepGuard(this);
    }
    /** @return The pre-admitted step grant (null until prepareStep succeeds). */
    PhysicalPagePool::Grant *stepGrant() noexcept
    {
        return step_grant_ ? &*step_grant_ : nullptr;
    }
    bool stepPrepared() const noexcept { return step_prepared_; }
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
        if (!step_prepared_ || step_rows_ <= 0) return false;
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
        // Promote each rolled fresh tail frame to the persistent tail frame
        // (the step's final partial page is now the live tail). Frames that
        // were NOT rolled (in-place or empty-start) keep their position.
        for (std::size_t layer = 0; layer < fresh_tail_frames_.size(); ++layer)
        {
            if (fresh_tail_frames_[layer])
            {
                tail_frames_[layer] = fresh_tail_frames_[layer];
                fresh_tail_frames_[layer].reset();
            }
        }
        step_grant_.reset();
        step_rows_ = 0;
        step_prepared_ = false;
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
        // Rolled fresh frames were never written: return them to the tail
        // grant's owned frames as empty spares (they stay eligible for the
        // next roll; nothing was published on them).
        for (auto &fresh : fresh_tail_frames_)
            fresh.reset();
        step_grant_.reset();
        step_rows_ = 0;
        step_prepared_ = false;
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

    /**
     * @brief Typed cursor lifecycle: generation of the publication row counts.
     *
     * Truncate/import change the live row counts OUTSIDE a forward step; the
     * host cursor must be re-seated to the publication's new count. The
     * binding owns that change and exposes it as a monotonic epoch so the
     * orchestrator can ADOPT the authoritative position instead of a
     * test-only direct setter. Epoch 0 = never changed since construction.
     * @return Current cursor generation.
     */
    std::uint64_t cursorEpoch() const noexcept { return cursor_epoch_; }

    /**
     * @brief Export one layer's complete live pooled history as an archive.
     * @param layer Attended layer.
     * @return The exact archive of that layer's publication (provenance is
     * the layer's native geometry), or empty when the layer has no live
     * publication.
     */
    std::optional<PagedKVArchive> exportLayer(int layer) const
    {
        const auto *prior = priorFor(layer);
        if (!prior) return std::nullopt;
        return PagedKVArchive::fromView(prior->snapshot.view());
    }

    /**
     * @brief Truncate ONE layer's history to @p rows and replace its prior.
     *
     * The truncated view is an exact prefix of the retained pages (same
     * referents, fewer visible rows); a NEW bank is published with the
     * clipped extent and becomes the prior cell. The old publication is
     * retired after the substitution (a reader that still holds it keeps its
     * own referents).
     * @param layer Attended layer.
     * @param rows Exact new visible row count (0 < rows <= current rows).
     * @return True on success; false (no mutation) when the layer has no
     * live publication, @p rows is out of range, or a new bank cannot be
     * admitted (never partially truncates).
     */
    bool truncateLayer(int layer, int rows)
    {
        if (!pool_ || !table_ || rows <= 0) return false;
        auto *prior = priorFor(layer);
        if (!prior) return false;
        const int current = prior->snapshot.view().rows();
        if (rows > current) return false;
        if (rows == current) return true; // No-op.
        // The production truncation path (kernel
        // TruncateExportReattachStaysByteExact): exact prefix view -> archive
        // -> re-materialized private pages through attachArchivedPrefix, so a
        // mid-page clip never exposes a partially-visible page to the kernel.
        auto clipped = prior->snapshot.view().prefix(static_cast<std::size_t>(rows));
        auto archive = PagedKVArchive::fromView(clipped);
        if (archive.empty()) return false;
        const auto pages_needed =
            (static_cast<std::size_t>(rows) + static_cast<std::size_t>(page_tokens_) - 1) /
                static_cast<std::size_t>(page_tokens_);
        auto grant = pool_->tryAdmit(pages_needed ? pages_needed : 1);
        if (!grant) { LOG_ERROR("[PooledKVDecodeBinding] truncate grant refused"); return false; }
        // The truncation publishes ONE new bank inside this request's OWN
        // accepted-work version envelope (a maintenance issuance), never the
        // normal-admission budget (which other requests own).
        auto truncated = reservation_
            ? table_->tryBeginReserved(*reservation_)
            : table_->tryBeginUpdate();
        if (!truncated)
        {
            LOG_ERROR("[PooledKVDecodeBinding] truncate bank refused");
            return false;
        }
        try
        {
            truncated->attachArchivedPrefix(archive, *grant, page_tokens_,
                /*trust_domain=*/0, /*ancestry_id=*/0);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[PooledKVDecodeBinding] truncateArchive attach refused: " << e.what());
            return false;
        }
        auto snapshot = truncated->commit();
        if (!table_->bindPublication(snapshot, request_id_, slot_))
        { LOG_ERROR("[PooledKVDecodeBinding] truncate bind refused"); return false; }
        auto ref = table_->refFor(snapshot, request_id_, slot_);
        if (!ref.valid())
        { LOG_ERROR("[PooledKVDecodeBinding] truncate ref refused"); return false; }
        if (layer < 0 || static_cast<std::size_t>(layer) >= cells_.size())
            return false;
        // The new bank's pages are PRIVATE (materialized) publications: their
        // frames stay owned by THIS grant. The previous private-publication
        // grant (if any) may now be released — its bank was already replaced
        // by the new snapshot (readers of the superseded bank keep reading;
        // share() of a superseded retiring frame correctly refuses). The
        // grant must outlive this function or the frames become retiring.
        private_publication_grant_ = std::move(grant);
        auto &cell = cells_[static_cast<std::size_t>(layer)].cells[0];
        cell.ref = ref;
        cell.snapshot = std::move(snapshot);
        // Typed cursor lifecycle: the row counts changed outside a forward;
        // the orchestrator adopts the authoritative position via cursorEpoch.
        ++cursor_epoch_;
        return true;
    }

    /**
     * @brief P3 M1: attach a compatible PREFIX of another live request's
     * publication to THIS request's fresh bank as DEDUPLICATED shared frames.
     *
     * Every full page strictly inside the prefix is adopted as a SHARED
     * immutable reference to the source's actual frame (zero-copy: the pool
     * keeps the frame while either request holds it, and the source may keep
     * decoding — its next step COWs/moves around the shared page). The final
     * page of the prefix, when it ends MID-PAGE, is COW'd privately (the
     * shared page's prefix bytes are copied into a fresh private page clipped
     * to the exact remainder); a prefix that ends exactly on a page boundary
     * shares every page and copies nothing.
     *
     * Ownership/custody: the new bank is published under THIS request's own
     * accepted-work issuance (tryBeginReserved); the COW'd private page is
     * admitted as ONE normal pool grant per layer, retained for the bank's
     * lifetime (the same rule as truncate). The source publication is never
     * mutated; two requests sharing a prefix only ever share IMMUTABLE pages.
     *
     * Identity checks (fail-closed, all verifiable here): the source must be
     * a DISTINCT live binding of the same pool+table authority (the pooled
     * authority IS the authentic isolation domain — a foreign table has a
     * different incarnation and refuses before any byte is shared) with a
     * live publication of identical extent on EVERY layer, identical
     * geometry (kv heads / head dim / page tokens / layer count), and the
     * requested prefix extent must be page-consistent with the source pages.
     * The collision-resolving ancestry proof is the source publication's
     * exact ref identity (authority_id + publication_version + request_id +
     * slot) — recorded verbatim and exposed via prefixAncestryRefs(); two
     * different contents can never carry the same ancestry. This attachment
     * restores NO sampler / RNG / history / placement state — only native
     * K/V bytes enter the bank.
     *
     * @param source Other live binding sharing this binding's pool+table.
     * @param rows Exact prefix row count (0 < rows <= source rows).
     * @return True on success (this binding now has a live publication of
     * exactly @p rows rows per layer); false without mutation on any refuse.
     */
    bool attachSharedPrefix(const PooledKVDecodeBinding &source, int rows)
    {
        if (!pool_ || !table_ || rows <= 0 || cancelled_)
            return false;
        if (this == &source || source.pool() != pool_ || source.table() != table_)
        {
            LOG_ERROR("[PooledKVDecodeBinding] Shared prefix attach requires a DISTINCT binding of the SAME pool/table");
            return false;
        }
        if (source.cancelled())
        {
            LOG_ERROR("[PooledKVDecodeBinding] Shared prefix source is cancelled");
            return false;
        }
        if (hasLivePublication())
        {
            LOG_ERROR("[PooledKVDecodeBinding] Shared prefix attach requires a FRESH destination (live history exists)");
            return false;
        }
        // Same geometry: identical kv_heads/head_dim/page_tokens/layers.
        if (source.kv_heads_ != kv_heads_ || source.head_dim_ != head_dim_ ||
            source.page_tokens_ != page_tokens_ || source.cells_.size() != cells_.size())
        {
            LOG_ERROR("[PooledKVDecodeBinding] Shared prefix geometry mismatch");
            return false;
        }
        // Every layer of the source must hold the same exact row extent
        // (a per-layer skew is refused — no partial-layer port).
        int source_rows = 0;
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            const auto *src_prior = source.priorFor(static_cast<int>(layer));
            if (!src_prior || !src_prior->live())
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix source layer " << layer
                          << " has no live publication");
                return false;
            }
            const int layer_rows = src_prior->snapshot.view().rows();
            if (layer == 0) source_rows = layer_rows;
            else if (layer_rows != source_rows)
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix source layers differ in extent");
                return false;
            }
        }
        if (rows > source_rows)
        {
            LOG_ERROR("[PooledKVDecodeBinding] Shared prefix rows exceed the source extent");
            return false;
        }
        const int full_pages = rows / page_tokens_;
        const int tail_rows = rows % page_tokens_;
        const std::size_t cow_pages = tail_rows > 0 ? cells_.size() : 0;
        std::optional<PhysicalPagePool::Grant> cow_grant;
        if (cow_pages)
        {
            cow_grant = pool_->tryAdmit(cow_pages);
            if (!cow_grant)
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix COW grant refused");
                return false;
            }
        }
        // All-or-none across layers: pre-admit every bank before mutating
        // any cell; any failure leaves every cell untouched.
        std::vector<std::optional<CPUFP32PagedKVTable::Update>> updates;
        updates.reserve(cells_.size());
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            auto update = table_->tryBeginReserved(*reservation_);
            if (!update)
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix bank refused at layer " << layer);
                return false;
            }
            const auto &src_pages = source.priorFor(static_cast<int>(layer))
                                        ->snapshot.view().pages();
            if (full_pages > static_cast<int>(src_pages.size()) ||
                (tail_rows > 0 && full_pages >= static_cast<int>(src_pages.size())))
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix page geometry malformed");
                return false;
            }
            for (int p = 0; p < full_pages; ++p)
            {
                auto shared = src_pages[static_cast<std::size_t>(p)].share();
                update->appendShared(std::move(shared), kv_heads_, head_dim_);
            }
            if (tail_rows > 0)
            {
                // Partial COW: copy ONLY the prefix remainder rows into a
                // fresh private page (never the whole shared page).
                const auto &last = src_pages[static_cast<std::size_t>(full_pages)];
                const auto layout = KVNativeStorageLayout::resolve(
                    KVCacheFamily::AttentionOnly, kv_heads_, head_dim_,
                    "fp32", DeviceId::cpu());
                const auto stride = layout.pagePayload(1).totalBytes();
                const auto &frame = cow_grant->assignFrame();
                auto write = cow_grant->prepareAppend(
                    frame, static_cast<std::size_t>(tail_rows) * stride);
                if (!write)
                {
                    LOG_ERROR("[PooledKVDecodeBinding] Shared prefix COW append refused");
                    return false;
                }
                write->copyFrom(last.storage().first(
                    static_cast<std::size_t>(tail_rows) * stride));
                auto committed = write->commit();
                update->append(CPUFP32KVPage(std::move(committed), kv_heads_, head_dim_));
            }
            updates.push_back(std::move(*update));
        }
        // Publish ALL layers all-or-none: every update is committed first,
        // every bank is bound and reffed, and NO cell becomes live until
        // EVERY layer succeeded. On any mid-loop failure every committed
        // snapshot is dropped (its release drops the bank reader count to
        // zero and the bank returns to Free, fully unbound — the
        // destination was fresh, so no reader or consumer ever saw a
        // partial publication) and the COW grant falls out of scope. The
        // invalid state is therefore UNREPRESENTABLE: a failed attach leaves
        // this binding exactly as it was (no live publication, epoch
        // unchanged, no private grant).
        std::vector<CPUFP32PagedKVTable::Snapshot> snapshots;
        snapshots.reserve(cells_.size());
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
            snapshots.push_back(updates[static_cast<std::size_t>(layer)]->commit());
        std::vector<PagedKVSnapshotRef> refs;
        refs.reserve(cells_.size());
        bool publish_ok = true;
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            if (!table_->bindPublication(snapshots[static_cast<std::size_t>(layer)],
                    request_id_, slot_))
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix bind refused at layer " << layer);
                publish_ok = false;
                break;
            }
            const auto ref = table_->refFor(snapshots[static_cast<std::size_t>(layer)],
                request_id_, slot_);
            if (!ref.valid())
            {
                LOG_ERROR("[PooledKVDecodeBinding] Shared prefix ref refused at layer " << layer);
                publish_ok = false;
                break;
            }
            refs.push_back(ref);
        }
        if (!publish_ok)
        {
            // Rollback: no cell was assigned yet, so dropping every snapshot
            // here clears every published bank (readers 1 -> 0 -> clear ->
            // Free/unbound). The COW grant local releases on return.
            snapshots.clear();
            return false;
        }
        // Every layer published and bound: now (and only now) make the
        // cells live.
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            auto &cell = cells_[static_cast<std::size_t>(layer)].cells[0];
            cell.ref = refs[static_cast<std::size_t>(layer)];
            cell.snapshot = std::move(snapshots[static_cast<std::size_t>(layer)]);
        }
        // Retain the COW grant for the bank's lifetime (my private pages
        // stay owned by this binding, never retiring while read).
        if (cow_grant)
            private_publication_grant_ = std::move(cow_grant);
        // Ancestry = the SOURCE's exact publication refs (the content
        // identity the prefix was exported from), per layer.
        prefix_ancestry_refs_.clear();
        prefix_ancestry_refs_.reserve(cells_.size());
        for (std::size_t layer = 0; layer < cells_.size(); ++layer)
        {
            const auto *src_cell = source.priorFor(static_cast<int>(layer));
            prefix_ancestry_refs_.push_back(src_cell ? src_cell->ref : PagedKVSnapshotRef{});
        }
        ++cursor_epoch_;
        return true;
    }

    /**
     * @return The exact source ref identities this binding's prefix was
     * attached from (per layer), or empty when no shared attach happened.
     * The recorded identity is the collision-resolving ancestry proof of the
     * shared content (never a content hash the caller could collide).
     */
    const std::vector<PagedKVSnapshotRef> &prefixAncestryRefs() const noexcept
    {
        return prefix_ancestry_refs_;
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
    bool cancelled_ = false;
    /** Monotonic cursor-generation counter: bumped on every authority
     * operation that changes the publication row counts (truncate/import).
     * The DGO adopts the binding's authoritative position when it observes
     * a new epoch — a typed cursor lifecycle, never a raw host setter. */
    std::uint64_t cursor_epoch_ = 0;
    std::optional<PhysicalPagePool::Grant> step_grant_;
    /** Grant owning the frames of the LAST truncation/import bank (private
     * materialized pages must stay owned for the bank's lifetime). */
    std::optional<PhysicalPagePool::Grant> private_publication_grant_;
    /** Exact source-ref ancestry recorded by attachSharedPrefix (per layer). */
    std::vector<PagedKVSnapshotRef> prefix_ancestry_refs_;
    int step_rows_ = 0;
    bool step_prepared_ = false;
    /** Persistent exclusive-tail grant (one frame per layer), admitted at binding. */
    std::optional<PhysicalPagePool::Grant> tail_grant_;
    std::vector<std::optional<PhysicalPagePool::FrameHandle>> tail_frames_;
    std::vector<std::optional<PhysicalPagePool::FrameHandle>> fresh_tail_frames_;
    /**
     * Persistent exclusive-tail mode. The production default is true
     * (measured: 268 vs 1280 bytes/step — TailCopyBytesMeasurement); the
     * default-constructed binding (which is only ever moved-from / stored
     * in containers before real construction) keeps the SAME default so
     * no code path silently observes the COW arm.
     */
    bool persistent_tail_ = true;
};
} // namespace llaminar2
