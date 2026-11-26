/**
 * @file ServingSession.h
 * @brief The continuous-batching serving session owned by OrchestrationRunner.
 *
 * OrchestrationRunner is the sole lifecycle authority of serving; it owns one
 * ServingSession and delegates its serving ABI (beginServingBatch,
 * stepServingBatch, ...) to it. The session holds every request's host-owned
 * row state — slot, sampler, seed, prompt progress, grant, forced tokens,
 * cancellation, retained decision logits — and drives slot-addressed shared
 * forwards through IServingRowExecutor, the narrow seam to the inference
 * runner. That seam also lets device-free tests replay recorded executor
 * behaviour through this exact production code.
 *
 * Step order (one call to step()):
 *  1. apply cancellations (queued requests finish at once);
 *  2. admit queued requests into free slots, restoring cached prefixes;
 *  3. decide from retained logits where allowed (forced token, unpaced or
 *     granted), committing tokens and finishing on stop/length;
 *  4. one decode forward (one token per row), then one prefill transaction
 *     with the oldest prefilling request's next chunk; a row whose prompt
 *     completes archives its prefix and decides;
 *  5. retire finished rows, freeing their slots.
 * A failed shared forward fails every resident and queued request and seals
 * the session.
 */
#pragma once

#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include "execution/serving/ServingBatch.h"
#include "execution/serving/ServingContinuationManifest.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/Sampler.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Slot-addressed row execution the session drives (the inference runner).
     *
     * Slots are KV sequences 0..capacity-1. A forward appends each row's
     * tokens to its slot at the slot's current position and exposes, per row,
     * the logits of that row's last token.
     */
    class IServingRowExecutor
    {
    public:
        virtual ~IServingRowExecutor() = default;

        /** @brief One shared forward; row r appends rows[r] to slot slots[r]. */
        virtual bool forwardServingRows(const std::vector<int> &slots,
                                        const std::vector<std::vector<int>> &rows) = 0;

        /** @brief Logits (vocab floats) of row @p row's last token after the last forward. */
        virtual const float *servingRowLogits(int row) const = 0;

        /** @brief Empty slot @p slot (sequence retirement). */
        virtual bool resetServingRow(int slot) = 0;

        /** @brief Restore a cached prompt prefix into a fresh slot; see IInferenceRunner. */
        virtual int restoreServingRowPrefix(int slot, const std::vector<int32_t> &prompt,
                                            PrefixLookupResult &admission) = 0;

        /** @brief Archive a slot's consumed prompt; see IInferenceRunner. */
        virtual bool harvestServingRowPrefix(int slot, const PrefixLookupResult &admission,
                                             const std::vector<int32_t> &prompt) = 0;

        /** @brief Diagnostic of the last failed forward (may be empty). */
        virtual std::string servingRowError() const { return {}; }

        /**
         * @brief Export the complete KV continuation of one resident slot.
         *
         * plan P4: suspension must retain the ACTUAL executor KV/auxiliary
         * referents, not host metadata. An executor that cannot export the
         * exact slot state returns empty; the session then refuses suspension
         * with a precise pre-mutation diagnostic (never resets/erases state
         * it cannot restore).
         */
        virtual std::optional<std::vector<std::byte>> exportSlotKV(int slot) const
        {
            return std::nullopt; // Unsupported: refuse suspension, never lose state.
        }

        /**
         * @brief Import an exported KV continuation into its slot.
         * @return True when the exact exported bytes were reinstalled at the
         * slot's post-reset position; false when the import is unsupported or
         * the export identity does not match (restore then refuses, never
         * fabricates continuity).
         */
        virtual bool importSlotKV(int slot, std::span<const std::byte> bytes)
        {
            (void)slot; (void)bytes;
            return false; // Unsupported: restore refuses pre-mutation.
        }
    };

    /**
     * @brief Diagnostic observer of a session (trace recording for replay).
     *
     * Called synchronously on the session's owning thread; it must not call
     * back into the session.
     */
    class IServingTraceRecorder
    {
    public:
        virtual ~IServingTraceRecorder() = default;

        /** @brief A request entered slot @p slot with its resolved (nonzero) seed. */
        virtual void admitted(const ServingBatchRequest &request, int slot, int restored_prompt_tokens) = 0;

        /**
         * @brief A RESIDENT request was suspended at a transaction boundary.
         * @param id The request whose complete continuation was captured.
         * @param slot The physical slot retired for reuse.
         * @param cached_tokens KV length at the settlement boundary.
         */
        virtual void suspended(std::uint64_t id, int slot, int cached_tokens) {}

        /**
         * @brief A suspended request was restored with its complete continuation.
         * @param id The restored request identity (original S record).
         * @param slot The physical slot the continuation now occupies.
         * @param cached_tokens KV length advertised at restore.
         */
        virtual void restored(std::uint64_t id, int slot, int cached_tokens) {}

        /** @brief A shared forward returned these per-row logits (vocab floats each). */
        virtual void forwarded(const std::vector<int> &slots, const std::vector<std::vector<int>> &rows,
                               const std::vector<const float *> &row_logits, int vocab) = 0;

        /** @brief The events of one completed step. */
        virtual void stepped(const std::vector<ServingBatchEvent> &events) = 0;
    };

    /** @brief Production sampler of a decision (serial/MTP output-position contract). */
    int sampleServingDecision(Sampler &sampler, const float *logits, std::size_t vocab_size,
                              const SamplingParams &params, int logical_position);

    class ServingSession
    {
    public:
        /** @brief Max prompt tokens one entering request adds per shared step. */
        static constexpr int kPrefillChunk = 64;

        /**
         * @param rows Row executor; must outlive the session.
         * @param context Admitted context (tokens per slot).
         * @param vocab Logits per decision.
         */
        ServingSession(IServingRowExecutor &rows, int context, int vocab);

        /**
         * @brief Open with @p capacity resident slots (all reset).
         * @param slots_to_reset Slots owned by the executor (>= capacity).
         * @param queue_capacity Admitted bounded queue depth Q (design §4.1,
         * exact-or-reject: requests beyond it are Refused, never silently
         * accepted). Zero = any queued request is Refused (Q=0).
         */
        bool begin(int capacity, int slots_to_reset, std::string &error,
                   std::shared_ptr<PhysicalMemorySuballocationLease> decision_storage = nullptr,
                   int queue_capacity = 0);
        bool active() const noexcept { return active_; }
        /** @brief Fresh per-session issue nonce (C2: manifest session binding). */
        std::uint64_t sessionNonce() const noexcept { return session_nonce_; }

        /**
         * @brief Submit one immutable request with the TYPED outcome.
         *
         * Direct when a free resident slot exists, Queued when admitted into
         * the bounded queue, Refused otherwise (invalid request, duplicate
         * live id, sealed session, queue full). @p error is populated on
         * Refused with the precise diagnostic; the caller's response
         * incarnation is only minted on Direct/Queued.
         */
        ServingSubmissionOutcome submit(ServingBatchRequest request, std::string &error);
        void cancel(std::uint64_t id);
        void grant(std::uint64_t id, int tokens);
        void force(std::uint64_t id, int32_t token);

        /** @brief One shared execution step (see file comment). */
        std::vector<ServingBatchEvent> step();

        /**
         * @brief Suspend one resident request at a transaction boundary.
         *
         * The row's complete continuation (identity, law, prompt/KV progress,
         * output credit, forced instructions, cancellation and sampler/RNG
         * history) is captured into a manifest, the slot is reset and retired
         * for reuse, and the request keeps its original S record/response
         * lease semantics (no new Q credit is consumed).
         * @return The manifest on success; empty with @p error when the id is
         * not resident, the session is inactive, or the row is mid-forward.
         */
        std::optional<ServingContinuationManifest> suspend(std::uint64_t id, std::string &error);

        /**
         * @brief Restore one suspended request into its original request state.
         * @param manifest Continuation captured by suspend() (rvalue; C2).
         * @return True when the row was restored (its next decision is the
         * exact no-movement continuation); false with @p error when the
         * manifest is incomplete, foreign, the slot is busy, or the session is
         * sealed. On ANY refusal the manifest is NOT consumed (the caller
         * keeps the continuation; no silent loss), and the request stays
         * tracked as an outstanding suspension until a defined terminal.
         */
        bool restore(ServingContinuationManifest &&manifest, std::string &error);

        bool idle() const noexcept { return rows_.empty() && queue_.empty(); }
        bool hasRunnableWork() const;
        ServingBatchOccupancy occupancy() const noexcept
        {
            return {static_cast<int>(rows_.size()), static_cast<int>(queue_.size())};
        }
        /** @brief Why the session sealed; empty while healthy. */
        const std::string &failure() const noexcept { return failure_; }

        /** @brief Close: reset resident slots and release all state. */
        void end();

        void setTraceRecorder(std::shared_ptr<IServingTraceRecorder> recorder) { recorder_ = std::move(recorder); }

        /**
         * @brief Pressure hook consulted before a queued request is promoted.
         *
         * @param request The queued request about to occupy a free slot.
         * @return True to promote, false to KEEP it queued (bounded
         * backpressure) so the step continues with the current cohort. The
         * default admits (no instance); production wiring supplies the
         * all-resource witness check (design §5.1). No request is dropped.
         */
        using PressureProbe =
            std::function<bool(const ServingBatchRequest &)>;
        void setPressureProbe(PressureProbe probe) { pressure_probe_ = std::move(probe); }

    private:
        /**
         * @brief One resident request; owns slot `slot` for its lifetime, its
         *        sampler/history and position-keyed RNG, so its tokens do not
         *        depend on which requests share a step. Logits after a forward
         *        are retained until the request may decide (unpaced, granted or
         *        forced): deferring a position-keyed decision does not change it.
         */
        struct Row
        {
            ServingBatchRequest request;
            int slot = -1;
            Sampler sampler;
            std::size_t prompt_consumed = 0; ///< Prompt tokens already in KV.
            int cached_tokens = 0;           ///< KV length of this slot.
            int completion_tokens = 0;
            int32_t last_token = 0;          ///< Last committed token.
            bool needs_feed = false;         ///< last_token is not yet in KV.
            bool has_logits = false;         ///< The slot's decision row holds the next decision.
            int grant = 0;                   ///< Paced tokens still allowed.
            std::deque<int32_t> forced;      ///< Tokens to commit instead of sampling.
            bool cancelled = false;
            int cached_prompt_tokens = 0;    ///< Prompt tokens restored from the prefix cache.
            PrefixLookupResult prefix_admission; ///< Immutable identity for the prompt harvest.
            std::vector<int32_t> sampler_decisions; ///< Every committed decision (RNG history).
            Sampler::State sampler_state;           ///< Exact RNG/penalty/DRY state (P4).
        };

        /**
         * @brief Fail all resident and queued requests, seal, and return @p events.
         *
         * The returned vector is the NORMAL terminal channel: every resident
         * and queued request receives exactly one Error event. Callers that
         * seal from outside a step (suspension reset failure) must PUBLISH
         * this vector through pending_events_ (drained by the next step),
         * never discard it — a dropped vector leaves response owners without
         * their terminal event.
         */
        std::vector<ServingBatchEvent> sealAll(const std::string &reason, std::vector<ServingBatchEvent> events);
        /** @brief Terminal events produced outside a step (suspension seal), drained once by step(). */
        std::vector<ServingBatchEvent> pending_events_;

        float *decisionLogits(int slot)
        {
            return decision_logits_.data() + static_cast<std::size_t>(slot) * static_cast<std::size_t>(vocab_);
        }

        IServingRowExecutor &executor_;
        const int context_;
        const int vocab_;
        PressureProbe pressure_probe_; ///< Optional §5.1 all-resource witness probe.
        int capacity_ = 0;
        int queue_capacity_ = 0; ///< Admitted bounded queue depth Q (design §4.1).
        bool active_ = false;
        std::uint64_t session_nonce_ = 0;       ///< Fresh per-session issue nonce (C2).
        /** @brief Terminal metadata of one outstanding suspension (C2). */
        struct OutstandingSuspension
        {
            int prompt_tokens = 0;
            int completion_tokens = 0;
        };
        /** @brief Outstanding suspended requests NOT yet restored (C2). */
        std::map<std::uint64_t, OutstandingSuspension> outstanding_suspensions_;
        std::vector<Row> rows_;                 ///< Resident requests (any order).
        std::vector<bool> slot_busy_;
        std::deque<ServingBatchRequest> queue_; ///< Waiting for a free slot (FIFO).
        std::vector<std::uint64_t> cancellations_;
        std::map<std::uint64_t, int> pending_grants_;                 ///< Received while queued.
        std::map<std::uint64_t, std::vector<int32_t>> pending_forced_; ///< Received while queued.
        /**
         * @brief One retained decision-logits row per resident slot (capacity x
         *        vocab FP32), allocated when the session opens and priced by
         *        the serving pool's ServingSlotDecisionGeometry.
         */
        std::vector<float> decision_logits_;
        /// PMA claim backing decision_logits_ (released after the store is freed).
        std::shared_ptr<PhysicalMemorySuballocationLease> decision_storage_;
        std::string failure_;
        std::shared_ptr<IServingTraceRecorder> recorder_;
    };
} // namespace llaminar2
