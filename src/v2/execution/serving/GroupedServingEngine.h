/**
 * @file GroupedServingEngine.h
 * @brief Deterministic grouped shared-execution engine prototype (roadmap-3 acceptance shape).
 *
 * Lifecycle-protocol engine: one worker advances admitted rows; each row
 * advances exactly its intrinsic window under a single-use, BACKED window
 * grant; a finishing row retires and a waiting NEW request enters its slot in
 * the same step while other rows continue (no cohort drain). Explicit restore
 * states (NotRequired/Pending/Published) gate admission; harvest holds the
 * source until an authenticating SnapshotBorrowReceipt releases it; grants are
 * validated against the request's OutputGrantBudget. The decode callback is a
 * pure per-row mock — the production row-invariant grouped kernel is the
 * caller's integration point.
 */

#pragma once

#include "execution/serving/OutputGrantBudget.h"
#include "execution/serving/ServingRowState.h"
#include "execution/serving/WindowGrant.h"

#include <cstdint>
#include <atomic>
#include <cstddef>
#include <functional>
#include <cstring>
#include <set>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Shared, mutex-protected borrow state: no raw engine pointer in the
     *         receipt; the engine (via the row) and every receipt copy share this
     *         state, and release flips it exactly once (nonce-authenticated). */
    class SnapshotBorrowState
    {
    public:
        std::mutex mutex;

    private:
        friend class GroupedServingEngine;
        friend class SnapshotBorrowReceipt;
        bool released = false;
        std::uint64_t nonce = 0;
    };


    /** @brief Engine-owned destination backing (REAL ownership): the engine alone
     *         creates it (private ctor), the producer marks it materialized exactly
     *         once, and release is exactly once. No numeric bank alone can
     *         substitute for this authority. */
    class DestinationBacking
    {
    public:
        [[nodiscard]] bool materialized() const noexcept
        {
            return buffer_ != nullptr && !released_;
        }
        [[nodiscard]] const std::byte *data() const noexcept { return buffer_.get(); }
        [[nodiscard]] std::uint64_t size() const noexcept { return capacity_; }
        [[nodiscard]] bool released() const noexcept { return released_; }
        [[nodiscard]] std::uint64_t capacity() const noexcept { return capacity_; }
        /** @brief Engine-only construction key (only the engine can make a Key). */
        struct Key
        {
        private:
            friend class GroupedServingEngine;
            Key() = default;
        };
        explicit DestinationBacking(Key, std::uint64_t capacity) : capacity_(capacity) {}
        DestinationBacking(const DestinationBacking &) = delete;
        DestinationBacking &operator=(const DestinationBacking &) = delete;
        DestinationBacking(DestinationBacking &&) = delete;
        DestinationBacking &operator=(DestinationBacking &&) = delete;
        ~DestinationBacking() = default; // buffer freed by the unique_ptr

    private:
        friend class GroupedServingEngine;
        void markMaterialized()
        {
            if (buffer_ != nullptr || released_)
                throw std::logic_error("Materialization exactly once (before release)");
            // REAL payload materialization: capacity bytes are actually allocated.
            buffer_ = std::make_unique<std::byte[]>(capacity_);
        }
        void markReleased()
        {
            if (released_)
                throw std::logic_error("Backing released exactly once");
            // REAL release: materialized payload bytes are actually freed.
            buffer_.reset();
            released_ = true;
        }
        /** @brief REAL output publication: produced bytes are physically written
         *         into the materialized destination buffer. */
        void writeAt(std::uint64_t offset, const std::byte *src, std::uint64_t bytes)
        {
            if (buffer_ == nullptr || released_)
                throw std::logic_error("Destination backing not materialized");
            if (offset + bytes > capacity_ || src == nullptr)
                throw std::logic_error("Destination write beyond the materialized backing");
            std::memcpy(buffer_.get() + offset, src, static_cast<std::size_t>(bytes));
            if (offset + bytes > live_extent_)
                live_extent_ = offset + bytes;
            // RESPONSE-OWNED publication metadata: the committed OUTPUT window is
            // (output_offset_, output_length_) — never prompt/restore state and
            // never unwritten holes. Lives on the backing, so it survives slot
            // reuse exactly as long as the response's backing does.
            if (output_length_ == 0)
            {
                output_offset_ = offset;
                output_length_ = bytes;
            }
            else
            {
                const auto end = output_offset_ + output_length_;
                if (offset < end)
                {
                    const auto overlap = end - offset;
                    if (overlap > bytes)
                        throw std::logic_error("Publication windows overlap illegally");
                    output_length_ += (bytes - overlap);
                }
                else
                {
                    output_length_ += bytes;
                }
            }
        }
        /** @brief TYPED RESTORE write: prompt/restore state occupies [offset,offset+bytes)
         *         but NEVER contributes to the committed-output publication
         *         metadata (design §5.1/§10.1: output only). */
        void writeRestored(std::uint64_t offset, const std::byte *src, std::uint64_t bytes)
        {
            if (buffer_ == nullptr || released_)
                throw std::logic_error("Destination backing not materialized");
            if (offset + bytes > capacity_ || src == nullptr)
                throw std::logic_error("Restore write beyond the materialized backing");
            std::memcpy(buffer_.get() + offset, src, static_cast<std::size_t>(bytes));
            if (offset + bytes > live_extent_)
                live_extent_ = offset + bytes;
            if (offset + bytes > restored_extent_)
                restored_extent_ = offset + bytes;
        }
        [[nodiscard]] std::uint64_t restoredExtent() const noexcept { return restored_extent_; }

        [[nodiscard]] std::uint64_t liveExtent() const noexcept { return live_extent_; }
        [[nodiscard]] std::uint64_t outputOffset() const noexcept { return output_offset_; }
        [[nodiscard]] std::uint64_t outputLength() const noexcept { return output_length_; }
        std::uint64_t capacity_ = 0;
        std::unique_ptr<std::byte[]> buffer_;
        std::uint64_t live_extent_ = 0; ///< REAL written live extent (never capacity).
        std::uint64_t output_offset_ = 0; ///< Committed output start (response-owned).
        std::uint64_t output_length_ = 0; ///< Committed output length (no holes).
        std::uint64_t restored_extent_ = 0; ///< Typed restore region (never output).
        bool released_ = false;
    };

    /** @brief Engine-owned restore COPY LANE: the producer completes it exactly
     *         once (no public flag setter can fabricate completion). */
    class RestoreCopyLane
    {
    public:
        struct Key
        {
        private:
            friend class GroupedServingEngine;
            Key() = default;
        };
        explicit RestoreCopyLane(Key, std::uint64_t expected_bytes)
            : expected_bytes_(expected_bytes)
        {
            if (expected_bytes_ == 0)
                throw std::logic_error("Restore lane requires a positive extent");
            buffer_ = std::make_unique<std::byte[]>(expected_bytes_);
        }
        RestoreCopyLane(const RestoreCopyLane &) = delete;
        RestoreCopyLane &operator=(const RestoreCopyLane &) = delete;
        RestoreCopyLane(RestoreCopyLane &&) = delete;
        RestoreCopyLane &operator=(RestoreCopyLane &&) = delete;
        [[nodiscard]] bool completed() const noexcept
        {
            return copied_bytes_ == expected_bytes_;
        }
        [[nodiscard]] std::uint64_t copiedBytes() const noexcept { return copied_bytes_; }
        [[nodiscard]] const std::byte *data() const noexcept { return buffer_.get(); }
        [[nodiscard]] std::uint64_t size() const noexcept { return expected_bytes_; }

    private:
        friend class GroupedServingEngine;
        /** @brief REAL byte transfer: copies n bytes from the producer source into
         *         the lane's owned destination buffer. */
        void advanceFrom(const std::byte *src, std::uint64_t bytes)
        {
            if (src == nullptr)
                throw std::logic_error("Restore copy requires a real producer source");
            if (bytes == 0 || copied_bytes_ + bytes > expected_bytes_)
                throw std::logic_error("Copy lane advance beyond the admitted extent");
            std::memcpy(buffer_.get() + copied_bytes_, src, static_cast<std::size_t>(bytes));
            copied_bytes_ += bytes;
        }
        std::uint64_t expected_bytes_ = 0;
        std::uint64_t copied_bytes_ = 0;
        std::unique_ptr<std::byte[]> buffer_;
    };

    /** @brief Explicit restore state (NoRestore vs Pending vs Published — never implicit). */
    enum class RestoreState : std::uint8_t
    {
        NotRequired,
        Pending,
        Published,
    };

    /** @brief One admitted live row inside the engine. */
    struct EngineRow
    {
        ServingRowState state{};           ///< slot/generation/kind/window/rng.
        std::uint64_t destination_reservation_ = 0; ///< Reserved identity (bank key).
        std::uint64_t destination_bytes_ = 0;        ///< Backing capacity bytes.
        std::uint64_t prompt_bytes_ = 0;              ///< Admitted prompt extent (payload derivation).
        std::uint64_t emitted_bytes_ = 0;             ///< REAL produced output bytes (written to backing).
        std::string restored_prefix_;               ///< Restored prompt bytes (decoder READS these).
        std::uint64_t reservation_generation_ = 0;   ///< Reservation ownership generation.
        std::shared_ptr<DestinationBacking> backing_;  ///< REAL owned backing authority.
        std::shared_ptr<RestoreCopyLane> copy_lane_;   ///< REAL copy lane (producer completion).
        bool delivered = false;      ///< Delivery separate from execution (§10.3).
        RestoreState restore_state = RestoreState::NotRequired; ///< Queued restore state.
        std::uint64_t restore_bytes_required = 0;  ///< Exact restore geometry (bytes).
        std::uint64_t restore_session = 0;         ///< Session issued with the restore request.
        bool restore_copy_completed = false;       ///< Producer copy finished (gate).
        bool restore_copy_in_progress = false;     ///< Copy lane actively copying.
        bool restore_receipt_issued = false;       ///< Receipt minted exactly once.
        std::uint64_t restore_copy_bytes = 0;      ///< Completed copy byte count.
        WindowGrantAuthority authority{};  ///< Sole one-window authority for this row.
        OutputGrantBudget budget{};        ///< Sealed per-request output budget.
        bool complete = false;
        bool throttled = false;            ///< Slow consumer: defers itself only.
        std::uint8_t mtp_max_depth = 0;    ///< Admitted MTP depth (0 = no MTP).
        bool mtp_depth_set = false;          ///< Depth admitted exactly once (incl. zero).
        bool mtp_verified_this_step = false; ///< Verifier accepted EXACTLY ONCE per step.
        std::uint8_t mtp_verify_calls_this_step = 0; ///< Step counter for the guard.
        bool harvest_requested = false;    ///< Canonical harvest requested for this row.
        bool source_frozen = false;        ///< Harvest freeze happened.
        bool snapshot_copy_in_flight = false;
        bool snapshot_receipt_issued = false;
        std::shared_ptr<SnapshotBorrowState> snapshot_lease; ///< Shared borrow lease (auth).
    };

    /** @brief Pure per-row decode callback (production: row-invariant grouped kernel). */
    using RowDecodeFn = std::string (*)(std::uint64_t seed, std::uint64_t position,
                                        std::uint64_t window,
                                        const std::string &restored_prefix);

    /** @brief Grouped engine: one worker, N admitted rows, no cohort drain. */
    class GroupedServingEngine
    {
    public:
        /** @brief Default serving pool width (a default placeholder, NEVER an
         *         HTTP-derived capacity unit; the admitted plan binds the real width). */
        static constexpr std::size_t kDefaultPoolWidth = 32;

        /** @brief Typed serving pool plan (bound at engine construction; the
         *         production binding is the ResolvedServingCapacity admission). */
        struct ServingPoolPlan
        {
            std::size_t destination_slots;
            constexpr ServingPoolPlan() : destination_slots(kDefaultPoolWidth) {}
            constexpr ServingPoolPlan(std::size_t slots) : destination_slots(slots) {}
        };
        explicit GroupedServingEngine(RowDecodeFn decode, std::uint64_t serving_instance = 1,
                                      ServingPoolPlan plan = ServingPoolPlan{})
            : decode_(decode), serving_instance_(serving_instance),
              max_destinations_(plan.destination_slots),
              engine_identity_(next_engine_identity_.fetch_add(1) + 1)
        {
            if (plan.destination_slots == 0)
                throw std::logic_error("Serving destination pool width must be positive");
            claim_slots_.reserve(plan.destination_slots); // NON-ALLOCATING claim commit
        }
        GroupedServingEngine(const GroupedServingEngine &) = delete;
        GroupedServingEngine &operator=(const GroupedServingEngine &) = delete;
        GroupedServingEngine(GroupedServingEngine &&) = delete;
        GroupedServingEngine &operator=(GroupedServingEngine &&) = delete;
        /// @brief Unique per-engine identity (atomic; engines are non-copyable/
        ///         non-movable so identity can NEVER be duplicated).

        /** @brief Admit a new request into a free slot (NewRequest only). */
        void admitNew(std::uint32_t slot, std::uint64_t generation,
                      std::uint64_t response_limit, std::uint64_t quantum, std::uint32_t seed,
                      std::uint64_t destination_reservation = 0,
                      std::uint64_t prompt_bytes = 0)
        {
            if (failed_)
                throw std::logic_error("Serving engine is failed: " + failed_reason_);
            if (rows_.count(slot) != 0 && !rows_[slot].complete)
                throw std::logic_error("Slot is occupied");
            if (rows_.count(slot) != 0 && rows_[slot].complete &&
                ((rows_[slot].harvest_requested || rows_[slot].source_frozen) &&
                 !snapshotLeaseReleased(rows_[slot].snapshot_lease)))
                throw std::logic_error(
                    "Slot harvest/source not retired (requested or frozen and not released); "
                    "cannot admit");
            if (response_limit == std::numeric_limits<std::uint64_t>::max())
                throw std::logic_error(
                    "Response limit UINT64_MAX is unsupported window geometry (rejected at admission)");
            if (destination_reservation == 0)
                throw std::logic_error("Admission requires an owned destination reservation");
            if (quantum == 0)
                throw std::logic_error("Window quantum must be positive");
            requireDestinationUnclaimed(destination_reservation);
            EngineRow row;
            row.state.slot = slot;
            row.state.generation = generation;
            row.state.kind = AdmissionKind::NewRequest;
            row.state.response_limit = response_limit;
            row.state.window = IntrinsicWindowPlan{quantum, response_limit};
            row.state.rng.seed = resolveRequestSeed(0, seed);
            row.budget.response_limit = response_limit;
            row.destination_reservation_ = destination_reservation;
            row.prompt_bytes_ = prompt_bytes;
            {
                auto backing = destination_backing(destination_reservation);
                row.backing_ = backing;
                row.destination_bytes_ = backing->capacity();
                if (!backing->materialized())
                    throw std::logic_error(
                        "Destination backing must be materialized by the producer "
                        "before admission");
                const auto derived_payload = derivedPayload(prompt_bytes, response_limit);
                if (backing->capacity() < derived_payload)
                    throw std::logic_error(
                        "Destination backing capacity must cover the derived request payload");
            }
            row.reservation_generation_ = generation;
            row.authority = std::move(WindowGrantAuthority(
                serving_instance_,
                ((response_limit / quantum) > (std::numeric_limits<std::size_t>::max() - 2))
                    ? std::numeric_limits<std::size_t>::max()
                    : ((response_limit / quantum) + 2))); // request-bounded ledger
            // Issue BEFORE placing: a refused grant must not mutate admission state.
            WindowGrant g;
            g.grant_id = grant_serial_++;
            g.slot = slot;
            g.generation = generation;
            g.window = row.state.window;
            g.destination_reservation = destination_reservation;
            g.serving_instance = serving_instance_;
            row.authority.issue(g); // authenticates instance + backing + bounded ledger
            // TRANSACTIONAL admission: storage commits AFTER every validation; the
            // claim commits AFTER the emplace (an allocation failure leaves no
            // claim, and the claim cannot throw after validation).
            rows_.erase(slot);                   // replace ANY existing (completed) row
            rows_.emplace(slot, std::move(row)); // never silently keep an old generation
            claimDestination(destination_reservation);
        }

        /** @brief Queue a waiting NEW request for a lane (admitted when the slot frees). */
        void queueRequest(std::uint32_t slot, std::uint64_t generation,
                          std::uint64_t response_limit, std::uint64_t quantum,
                          std::uint32_t seed, std::uint64_t destination_reservation = 0,
                          std::uint64_t prompt_bytes = 0)
        {
            if (failed_)
                throw std::logic_error("Serving engine is failed: " + failed_reason_);
            if (queued_.count(slot) != 0)
                throw std::logic_error("A request is already queued for this slot");
            if (response_limit == std::numeric_limits<std::uint64_t>::max())
                throw std::logic_error(
                    "Response limit UINT64_MAX is unsupported window geometry (rejected at admission)");
            if (quantum == 0)
                throw std::logic_error("Window quantum must be positive");
            requireDestinationUnclaimed(destination_reservation);
            EngineRow row;
            row.state.slot = slot;
            row.state.generation = generation;
            row.state.kind = AdmissionKind::NewRequest; // §14: C is a NewRequest
            row.state.response_limit = response_limit;
            row.state.window = IntrinsicWindowPlan{quantum, response_limit};
            row.state.rng.seed = resolveRequestSeed(0, seed);
            row.budget.response_limit = response_limit;
            if (destination_reservation == 0)
                throw std::logic_error("Queue requires an owned destination reservation");
            row.destination_reservation_ = destination_reservation;
            row.prompt_bytes_ = prompt_bytes;
            {
                auto backing = destination_backing(destination_reservation);
                row.backing_ = backing;
                row.destination_bytes_ = backing->capacity();
                if (!backing->materialized())
                    throw std::logic_error(
                        "Destination backing must be materialized by the producer "
                        "before queue");
                const auto derived_payload = derivedPayload(prompt_bytes, response_limit);
                if (backing->capacity() < derived_payload)
                    throw std::logic_error(
                        "Destination backing capacity must cover the derived request payload");
            }
            row.reservation_generation_ = generation;
            row.authority = std::move(WindowGrantAuthority(
                serving_instance_,
                ((response_limit / quantum) > (std::numeric_limits<std::size_t>::max() - 2))
                    ? std::numeric_limits<std::size_t>::max()
                    : ((response_limit / quantum) + 2))); // request-bounded ledger
            WindowGrant g;
            g.grant_id = grant_serial_++;
            g.slot = slot;
            g.generation = generation;
            g.window = row.state.window;
            g.destination_reservation = destination_reservation;
            g.serving_instance = serving_instance_;
            row.authority.issue(g);
            // TRANSACTIONAL queue admission: storage commits first (allocation
            // failure leaves no claim), then the claim commits (cannot throw).
            queued_.emplace(slot, std::move(row));
            claimDestination(destination_reservation);
        }

        /** @brief Begin the restore lane for a queued row (explicit geometry). */
        void beginQueuedRestore(std::uint32_t slot, std::uint64_t bytes_to_restore)
        {
            if (queued_.count(slot) == 0)
                throw std::logic_error("No queued row to restore");
            auto &q = queued_.at(slot);
            if (q.restore_state != RestoreState::NotRequired)
                throw std::logic_error("Restore already started (exactly once)");
            if (bytes_to_restore == 0)
                throw std::logic_error("Restore geometry must be positive");
            if (bytes_to_restore != q.prompt_bytes_)
                throw std::logic_error(
                    "Restore geometry must equal the request's prompt extent "
                    "(typed layout: [0,prompt) restored, output after it — no overlap)");
            q.restore_state = RestoreState::Pending;
            q.restore_bytes_required = bytes_to_restore;
            q.restore_session = restore_session_serial_++;
            q.copy_lane_ = std::make_shared<RestoreCopyLane>(RestoreCopyLane::Key{}, bytes_to_restore);
            q.restore_copy_in_progress = true;
        }

        /** @brief Engine-authenticated producer-frontier receipt: only the engine
         *         may create it; it carries slot, generation and an engine-issued
         *         reception nonce so a stale/cross-instance receipt is refused. */
        /** @brief Engine-bound producer-frontier receipt: private ctor, session-bound. */
        class RestoreFrontierReceipt
        {
        public:
            RestoreFrontierReceipt(const RestoreFrontierReceipt &) = delete;
            RestoreFrontierReceipt &operator=(const RestoreFrontierReceipt &) = delete;
            RestoreFrontierReceipt(RestoreFrontierReceipt &&) = default;
            RestoreFrontierReceipt &operator=(RestoreFrontierReceipt &&) = default;

        private:
            friend class GroupedServingEngine;
            RestoreFrontierReceipt(std::uint32_t slot, std::uint64_t generation,
                                   std::uint64_t bytes, std::uint64_t session,
                                   std::uint64_t engine_identity)
                : slot_(slot), generation_(generation), bytes_(bytes), session_(session),
                  engine_identity_(engine_identity) {}
            std::uint32_t slot_ = 0;
            std::uint64_t generation_ = 0;
            std::uint64_t bytes_ = 0;
            std::uint64_t session_ = 0; // bound to the queued row's restore session
            std::uint64_t engine_identity_ = 0; // THIS engine's unique identity
            bool valid_ = true;
        };

        /** @brief Engine issues the receipt bound to THIS engine identity + the
         *         queued row's own restore session: the caller cannot fabricate
         *         tuples or choose a session (no foreign-engine collision). */
        /** @brief Producer copy completion (exactly once, before any receipt). */
        /** @brief Producer copy completion: REQUIRES the current session (stale
         *         completions and wrong geometry are refused) + exact bytes. */
        void completeRestoreCopy(std::uint32_t slot, std::uint64_t session,
                                 std::uint64_t bytes, const std::byte *producer_source)
        {
            auto &q = queued_.at(slot);
            if (q.restore_state != RestoreState::Pending)
                throw std::logic_error("Restore copy completion requires a pending restore");
            if (session != q.restore_session || bytes != q.restore_bytes_required)
                throw std::logic_error(
                    "Restore copy completion requires the current session and exact geometry");
            if (q.restore_copy_completed || q.restore_receipt_issued)
                throw std::logic_error("Restore copy completed exactly once");
            if (q.copy_lane_ == nullptr)
                throw std::logic_error("Restore copy lane missing (beginQueuedRestore first)");
            if (producer_source == nullptr)
                throw std::logic_error("Restore copy requires a real producer source");
            // REAL transfer: the engine copies the admitted extent from the
            // producer's source buffer into the lane's own destination buffer.
            q.copy_lane_->advanceFrom(producer_source, q.restore_bytes_required);
            if (!q.copy_lane_->completed())
                throw std::logic_error("Restore copy lane not completed after the full copy");
            q.restore_copy_completed = true;
            q.restore_copy_in_progress = false;
            q.restore_copy_bytes = bytes;
        }

        RestoreFrontierReceipt issueRestoreReceipt(std::uint32_t slot, std::uint64_t generation,
                                                   std::uint64_t bytes)
        {
            if (queued_.count(slot) == 0)
                throw std::logic_error("No queued row to issue a restore receipt for");
            auto &q = queued_.at(slot);
            // Readiness may ONLY be minted after a COMPLETED producer copy, and
            // only ONCE (no double minting).
            if (!q.restore_copy_completed || q.restore_copy_bytes != bytes)
                throw std::logic_error(
                    "Restore receipt requires a completed producer copy (exact bytes)");
            if (q.restore_receipt_issued)
                throw std::logic_error("Restore receipt minted exactly once");
            if (generation != q.state.generation)
                throw std::logic_error("Restore receipt generation must match the queued row");
            q.restore_receipt_issued = true;
            const auto session = q.restore_session;
            return RestoreFrontierReceipt(slot, generation, bytes, session, engine_identity_);
        }

        /** @brief Publish the exact producer frontier for a queued restore. */
        void publishRestore(std::uint32_t slot, RestoreFrontierReceipt receipt)
        {
            if (queued_.count(slot) == 0)
                throw std::logic_error("No queued row for this slot");
            auto &q = queued_.at(slot);
            if (q.restore_state != RestoreState::Pending)
                throw std::logic_error("Restore must be pending before publication");
            if (receipt.slot_ != slot || receipt.generation_ != q.state.generation ||
                receipt.session_ != q.restore_session || receipt.session_ == 0 ||
                receipt.engine_identity_ != engine_identity_ || !receipt.valid_)
                throw std::logic_error(
                    "Restore readiness requires the exact producer frontier (session-bound)");
            if (receipt.bytes_ != q.restore_bytes_required)
                throw std::logic_error("Restore frontier bytes must match the required geometry");
            // The copied payload is RESTORED INTO the owned destination backing
            // (the response's execution state), not left in an isolated lane.
            auto backing = destination_backing(q.destination_reservation_);
            if (!backing->materialized())
                throw std::logic_error("Restore requires a materialized destination");
            if (q.copy_lane_ == nullptr || !q.copy_lane_->completed())
                throw std::logic_error("Restore publication requires the completed copy");
            backing->writeRestored(0, q.copy_lane_->data(), q.copy_lane_->size());
            // The restored payload becomes the row's readable prefix: execution
            // state actually carries the restored bytes (decoder receives them).
            q.restored_prefix_ =
                std::string(reinterpret_cast<const char *>(q.copy_lane_->data()),
                            static_cast<std::size_t>(q.copy_lane_->size()));
            q.restore_state = RestoreState::Published;
        }

        /** @brief Explicit eligibility: NotRequired or Published (Pending is not). */
        [[nodiscard]] bool restoreEligible(std::uint32_t slot) const
        {
            if (queued_.count(slot) == 0)
                return false;
            const auto &q = queued_.at(slot);
            return q.restore_state != RestoreState::Pending;
        }


        /** @brief Owned copy-retirement receipt (shared lease state; move-only). */
        class SnapshotBorrowReceipt
        {
        public:
            SnapshotBorrowReceipt(const SnapshotBorrowReceipt &) = delete;
            SnapshotBorrowReceipt &operator=(const SnapshotBorrowReceipt &) = delete;
            SnapshotBorrowReceipt(SnapshotBorrowReceipt &&) = default;
            SnapshotBorrowReceipt &operator=(SnapshotBorrowReceipt &&other) noexcept
            {
                if (this != &other)
                {
                    release(); // an existing receipt must release ITS row first
                    state_ = std::move(other.state_);
                }
                return *this;
            }
            ~SnapshotBorrowReceipt() { release(); }

            /** @brief Exactly-once release (idempotent; nonce-authenticated). */
            void release()
            {
                if (state_ == nullptr)
                    return;
                // Move the shared_ptr out under the lock; the lock guard is
                // destroyed BEFORE the moved-out state drops, so a destroyed
                // mutex is never unlocked (safe even if this was the last ref).
                std::shared_ptr<SnapshotBorrowState> state;
                {
                    std::lock_guard l(state_->mutex);
                    if (!state_->released)
                        state_->released = true;
                    state = std::move(state_);
                }
            }

        private:
            friend class GroupedServingEngine;
            explicit SnapshotBorrowReceipt(std::shared_ptr<SnapshotBorrowState> state)
                : state_(std::move(state)) {}
            std::shared_ptr<SnapshotBorrowState> state_;
        };

        /** @brief Copy finished: shared lease authenticates; source held until release. */
        SnapshotBorrowReceipt borrowSnapshot(std::uint32_t slot)
        {
            auto &row = rows_.at(slot);
            if (!row.snapshot_copy_in_flight)
                throw std::logic_error("Snapshot borrow requires an in-flight copy");
            if (row.snapshot_receipt_issued)
                throw std::logic_error("Snapshot borrow already issued (no overlapping borrows)");
            row.snapshot_copy_in_flight = false;
            row.snapshot_lease = std::make_shared<SnapshotBorrowState>();
            std::lock_guard l(row.snapshot_lease->mutex);
            row.snapshot_lease->nonce = snapshot_nonce_serial_++;
            row.snapshot_receipt_issued = true;
            return SnapshotBorrowReceipt(row.snapshot_lease);
        }

        /** @brief Bank an owned destination reservation (capacity bytes). */

        [[nodiscard]] std::size_t maxDestinations() const noexcept { return max_destinations_; }

        /** @brief Bounded destination identity: the pool is a fixed-size wheel, so
         *         banked IDs NEVER grow without bound. A retired ID may be re-banked
         *         (slot reuse, canonical bounded identity), but its GENERATION is
         *         bumped, so a stale response can never act on the new occupant. */
        void bankDestination(std::uint64_t reservation, std::uint64_t bytes)
        {
            if (reservation == 0 || reservation > max_destinations_)
                throw std::logic_error(
                    "Destination reservation must be a wheel slot in [1..pool width]");
            if (bytes == 0)
                throw std::logic_error("Destination reservation must carry capacity");
            if (destinations_.size() >= max_destinations_)
                throw std::logic_error("Destination pool is full (bounded identity wheel)");
            if (destinations_.count(reservation) != 0)
                throw std::logic_error("Destination reservation is already banked (unique live ID)");
            // The generation is a PER-SLOT monotonic counter: the map holds at most
            // kMaxDestinations entries forever (bounded history, never a fresh
            // historical ID) and every re-bank bumps it (ABA-closed delivery keys).
            destination_generation_[reservation] += 1;
            destinations_[reservation] = std::make_shared<DestinationBacking>(DestinationBacking::Key{}, bytes);
        }

        /** @brief Producer materialization event: once, before any issue backs it. */
        void materializeDestination(std::uint64_t reservation)
        {
            auto it = destinations_.find(reservation);
            if (it == destinations_.end())
                throw std::logic_error("Destination reservation is not banked");
            it->second->markMaterialized();
        }


        /** @brief Release a destination reservation at retirement (exactly once). */
        void releaseDestination(std::uint64_t reservation)
        {
            // A LIVE row still referencing this reservation blocks release.
            for (const auto &[slot, row] : rows_)
                if (row.destination_reservation_ == reservation && !row.complete)
                    throw std::logic_error(
                        "Cannot release a reservation a live row still owns");
            for (const auto &[slot, q] : queued_)
                if (q.destination_reservation_ == reservation)
                    throw std::logic_error(
                        "Cannot release a reservation a queued row still owns");
            auto dit = delivered_ledger_.find(reservation);
            if (dit == delivered_ledger_.end())
                throw std::logic_error(
                    "Cannot release before delivery (delivery ledger has no record)");
            if (dit->second != DeliveryState::Delivered)
                throw std::logic_error(
                    "Cannot retire during an in-progress publication "
                    "(consumer still borrowing the buffer)");
            auto it = destinations_.find(reservation);
            if (it == destinations_.end())
                throw std::logic_error("Destination reservation not owned (double release refused)");
            it->second->markReleased(); // REAL backing release (buffer freed)
            destinations_.erase(it);
            removeClaim(reservation); // response retired: claimable again
            completed_ledger_.erase(reservation); // no stale records after retirement
            delivered_ledger_.erase(reservation);
        }

        /** @brief Delivery completion (§10.3 separate from execution): exactly once,
         *         keyed by the RESPONSE reservation (a stale slot can never be
         *         misattributed to its new occupant) and gated on the response
         *         actually completing. */
        /** @brief Delivery requires the CURRENT wheel generation: a stale response
         *         from a previous occupancy of the same ID is refused (ABA-closed;
         *         the generation map is bounded by the wheel, never a fresh ID). */
        void markDelivered(std::uint64_t reservation, std::uint64_t generation)
        {
            if (destinations_.count(reservation) == 0)
                throw std::logic_error("Delivery of an unknown reservation refused");
            if (destination_generation_[reservation] != generation)
                throw std::logic_error(
                    "Delivery generation does not match the CURRENT occupant (stale refused)");
            if (completed_ledger_.count(reservation) == 0 ||
                completed_ledger_[reservation] != generation)
                throw std::logic_error(
                    "Delivery requires the response to have completed (no premature delivery)");
            if (delivered_ledger_.count(reservation) != 0)
                throw std::logic_error("Delivery recorded exactly once");
            // PUBLICATION CONSUMER FIRST: the delivery record commits ONLY after the
            // consumer accepts the delivered bytes. A throwing consumer leaves the
            // response UNDELIVERED (retryable; backing not freed, exactly-once not
            // claimed) — the ledger is never set ahead of actual completion.
            // PUBLICATION STATE MACHINE: the ledger entry becomes Publishing BEFORE
            // the consumer runs (so no two publications overlap and an allocation
            // failure never leaves external consumption unrecorded), then
            // transitions to Delivered AFTER acceptance (non-allocating map write).
            // A THROWING consumer rolls back to no-record (retryable), and
            // retirement is REFUSED while Publishing (a consumer may never free its
            // own in-progress borrowing pointer).
            auto ledger_it = delivered_ledger_.emplace(reservation,
                                                       DeliveryState::Publishing);
            if (!ledger_it.second)
                throw std::logic_error("Delivery already started or recorded (exactly once)");
            try
            {
                if (consumer_)
                {
                    auto it = destinations_.find(reservation);
                    // COMMITTED OUTPUT ONLY (§5.1/§10.1): output window + length;
                    // never prompt/restore state, never unwritten holes.
                    consumer_(reservation,
                              it->second->data() + it->second->outputOffset(),
                              static_cast<std::size_t>(it->second->outputLength()));
                }
            }
            catch (...)
            {
                delivered_ledger_.erase(reservation);
                throw; // response remains UNDELIVERED (retryable)
            }
            ledger_it.first->second = DeliveryState::Delivered; // after acceptance
        }

        void setConsumer(
            std::function<void(std::uint64_t, const std::byte *, std::size_t)> consumer)
        {
            consumer_ = std::move(consumer);
        }

        /** @brief Mark a row slow-consumer: it defers itself, others progress. */
        void throttle(std::uint32_t slot) { rows_.at(slot).throttled = true; }

        /** @brief MTP verifier accepts exactly once per step (the step's ONLY entry). */
        void verifyMTP(std::uint32_t slot)
        {
            auto &row = rows_.at(slot);
            if (row.mtp_max_depth == 0)
                throw std::logic_error("MTP verifier requires an admitted positive depth");
            if (row.state.committed == 0)
                throw std::logic_error("MTP verification requires a produced window");
            if (row.mtp_verify_calls_this_step != 0)
                throw std::logic_error("MTP verifier accepted twice in one step (exactly once)");
            ++row.mtp_verify_calls_this_step;
            row.mtp_verified_this_step = true;
        }

        /** @brief Admit an MTP depth for a row (once, before any production). */
        void setMTPDepth(std::uint32_t slot, std::uint8_t depth)
        {
            auto &row = rows_.at(slot);
            if (row.mtp_depth_set)
                throw std::logic_error("MTP depth already admitted (exactly once, incl. zero)");
            if (depth > 4)
                throw std::logic_error("MTP depth exceeds the supported bound");
            if (row.state.committed > 0)
                throw std::logic_error("MTP depth is immutable after execution starts");
            row.mtp_max_depth = depth;
            row.mtp_depth_set = true;
        }

        /** @brief Cancel a queued row (exactly once); its slot stays free for later. */
        void cancelQueued(std::uint32_t slot)
        {
            if (queued_.count(slot) == 0)
                throw std::logic_error("No queued row to cancel");
            auto &q = queued_.at(slot);
            if (q.restore_state == RestoreState::Pending && q.restore_copy_in_progress)
                throw std::logic_error(
                    "Cannot cancel while a restore copy is in progress (retire first)");
            const auto reservation = q.destination_reservation_;
            queued_.erase(slot); // pending copies retired with the row
            releaseForCancellation(reservation); // no delivery obligation (never ran)
        }

        /** @brief Request the canonical harvest for a row BEFORE it completes. */
        void requestHarvest(std::uint32_t slot)
        {
            auto &row = rows_.at(slot);
            if (row.complete)
                throw std::logic_error("Harvest request must precede completion");
            if (row.harvest_requested)
                throw std::logic_error("Harvest already requested (exactly once)");
            row.harvest_requested = true;
        }

        /** @brief Canonical harvest: freeze a completed row's source (exactly once). */
        void freezeSnapshot(std::uint32_t slot)
        {
            auto &row = rows_.at(slot);
            if (!row.complete)
                throw std::logic_error("Harvest requires a completed row");
            if (row.source_frozen)
                throw std::logic_error("Source already frozen (exactly once)");
            row.source_frozen = true;
        }

        void beginSnapshotCopy(std::uint32_t slot)
        {
            auto &row = rows_.at(slot);
            if (!row.source_frozen || row.snapshot_copy_in_flight ||
                snapshotLeaseReleased(row.snapshot_lease) || row.snapshot_receipt_issued)
                throw std::logic_error(
                    "Snapshot copy requires a frozen source and NO issued receipt "
                    "(no overlapping borrows)");
            row.snapshot_copy_in_flight = true;
        }

        /** @brief The host issues the next single-use, budget-backed grant. */
        void issueNextGrant(std::uint32_t slot)
        {
            if (failed_)
                throw std::logic_error("Serving engine is failed: " + failed_reason_);
            auto &row = rows_.at(slot);
            if (row.complete)
                throw std::logic_error("Cannot grant a completed row");
            WindowGrant g;
            g.grant_id = grant_serial_++;
            g.slot = row.state.slot;
            g.generation = row.state.generation;
            g.window = row.state.window;
            g.destination_reservation = row.destination_reservation_;
            g.serving_instance = serving_instance_;
            if (g.destination_reservation == 0)
                throw std::logic_error("Next grant requires the owned destination reservation");
            // Budget validates the ACTUAL admitted grant (extent/limit/backing).
            row.budget.requireFits(g, row.state.committed);
            row.authority.issue(g); // instance-authenticated + bounded ledger
        }

        /** @brief One grouped step (see header doc). */
        [[nodiscard]] bool failed() const noexcept { return failed_; }
        [[nodiscard]] const std::string &failedReason() const noexcept { return failed_reason_; }

        std::map<std::uint32_t, std::string> step()
        {
            if (failed_)
                throw std::logic_error("Serving engine is failed: " + failed_reason_);
            std::map<std::uint32_t, std::string> produced;
            std::vector<std::uint32_t> completions;
            std::vector<std::uint64_t> retired_destinations;
            // PHASE A (read-only): collect the attach destinations (data only —
            // NOTHING is mutated before validation).
            std::vector<std::uint32_t> free_slots;
            for (const auto &[slot, row] : rows_)
            {
                if (!row.complete)
                    continue;
                const bool harvest_pending =
                    (row.harvest_requested || row.source_frozen) && !snapshotLeaseReleased(row.snapshot_lease);
                if (harvest_pending)
                    continue;
                if (queuedMayAttach(slot))
                    free_slots.push_back(slot);
            }
            for (const auto &[slot, q] : queued_)
            {
                if (rows_.count(slot) == 0 && queuedMayAttach(slot))
                    free_slots.push_back(slot); // fresh slot
            }
            // PHASE B (validate): decode EVERY executable row AND every queued row
            // that will attach, verifying the real-bytes publication bound. NO
            // state is mutated, so a violation cannot leave any row half-completed,
            // any attach applied, or any grant stranded.
            std::vector<std::pair<std::uint32_t, std::string>> pending_outputs;
            auto validate = [&](std::uint32_t slot, const EngineRow &r)
            {
                const auto end = nextWindowEnd(r.state.committed, r.state.window);
                std::string out;
                for (std::uint64_t pos = r.state.committed + 1; pos <= end; ++pos)
                    out += decode_(r.state.rng.seed, pos, r.state.window.quantum,
                                   r.restored_prefix_);
                const auto prospective = r.prompt_bytes_ + r.emitted_bytes_ + out.size();
                if (prospective > r.backing_->capacity())
                {
                    failed_ = true;
                    failed_reason_ = "Live output exceeded the materialized destination backing";
                    throw std::logic_error(failed_reason_);
                }
                pending_outputs.emplace_back(slot, std::move(out));
            };
            for (const auto &[slot, row] : rows_)
            {
                if (row.complete || row.throttled)
                    continue;
                if (!row.authority.mayExecute())
                    continue; // no usable grant: deferred
                validate(slot, row);
            }
            for (const auto slot : free_slots)
                validate(slot, queued_.at(slot));
            // PHASE C (attach): all validations passed — now (and only now) the
            // queue-to-row attachments are applied.
            for (const auto slot : free_slots)
            {
                auto cont = std::move(queued_[slot]);
                queued_.erase(slot);
                rows_.erase(slot);                  // retire old row first
                rows_.emplace(slot, std::move(cont)); // fresh placement (instance adopted)
                produced[slot] +=
                    "|entry:" + std::to_string(rows_[slot].state.generation);
            }
            // PHASE 2 (execute): all validations passed, so every mutation below is
            // committed only for the wholly validated set.
            for (auto &[slot, row] : rows_)
            {
                if (row.complete || row.throttled)
                    continue;
                if (!row.authority.mayExecute())
                    continue;
                if (pending_outputs.empty())
                    continue;
                auto it = std::find_if(pending_outputs.begin(), pending_outputs.end(),
                                       [&](const auto &p) { return p.first == slot; });
                if (it == pending_outputs.end())
                    continue;
                std::string out = std::move(it->second);
                row.mtp_verify_calls_this_step = 0; // step-scoped counter
                const auto end = nextWindowEnd(row.state.committed, row.state.window);
                row.state.committed = end;
                produced[slot] = out;
                // REAL publication: output bytes are physically written into the
                // materialized destination buffer AFTER the prompt/restore region.
                row.emitted_bytes_ += out.size();
                row.backing_->writeAt(row.prompt_bytes_ + row.emitted_bytes_ - out.size(),
                                      reinterpret_cast<const std::byte *>(out.data()),
                                      out.size());
                row.authority.resolve(); // one window per grant (consumed)
                if (row.mtp_max_depth > 0)
                    verifyMTP(slot); // exactly once per step (guarded inside)
                if (row.state.committed >= row.state.response_limit)
                {
                    row.complete = true;
                    completed_ledger_[row.destination_reservation_] =
                        destination_generation_[row.destination_reservation_];
                    const auto dstate = delivered_ledger_.find(row.destination_reservation_);
                    if (dstate != delivered_ledger_.end() &&
                        dstate->second == DeliveryState::Delivered)
                        retired_destinations.push_back(row.destination_reservation_);
                    const bool harvest_pending =
                        (row.harvest_requested || row.source_frozen) && !snapshotLeaseReleased(row.snapshot_lease);
                    if (queuedMayAttach(slot) && !harvest_pending)
                    {
                        // NEVER mutate the map inside the range-for: collect the
                        // replacement and apply it in a post-pass below (the
                        // iterator would be invalidated by erase/emplace).
                        completions.emplace_back(slot);
                    }
                }
            }
            // Post-pass replacements (completed slots with a ready queued row):
            // applied AFTER the iteration so no iterator is invalidated.
            for (const auto slot : completions)
            {
                auto cont = std::move(queued_[slot]);
                queued_.erase(slot);
                // MOVE-ASSIGN into the existing (completed) row: the EngineRow
                // move-assign reaches WindowGrantAuthority's same-instance
                // operator= (the credited bound-transfer fix is DIRECTLY used).
                rows_.at(slot) = std::move(cont);
                produced[slot] += "|entry:" + std::to_string(rows_[slot].state.generation);
            }
            // Retirement release: every COMPLETED + DELIVERED request releases its
            // OWN backing (delivery is separate from execution; live/queued refused).
            for (const auto reservation : retired_destinations)
                releaseDestination(reservation);
            // Delivery-driven release, keyed by the RESPONSE owner so a delivery
            // recorded after the execution slot was reused still releases the
            // ORIGINAL backing (B) without stalling the new occupant (C).
            std::vector<std::uint64_t> released_reservations;
            for (const auto &[reservation, state] : delivered_ledger_)
            {
                // SELECT ONLY ACCEPTED (Delivered) publications: a Publishing entry
                // (consumer still borrowing) must never be auto-retired (the guard
                // would throw after other rows already advanced; the consumer owns
                // the buffer until its callback returns).
                if (state != DeliveryState::Delivered)
                    continue;
                auto it = destinations_.find(reservation);
                if (it != destinations_.end() && !it->second->released())
                    released_reservations.push_back(reservation);
            }
            // Release AFTER the iteration (release erases ledger entries).
            for (const auto reservation : released_reservations)
                releaseDestination(reservation); // exactly-once via ownership
            return produced;
        }

        [[nodiscard]] std::uint64_t restoredExtentFor(std::uint64_t reservation) const
        {
            auto it = destinations_.find(reservation);
            if (it == destinations_.end() || !it->second->materialized())
                throw std::logic_error("No materialized backing for this reservation");
            return it->second->restoredExtent();
        }

        [[nodiscard]] std::uint64_t destinationGeneration(std::uint64_t reservation) const
        {
            auto it = destination_generation_.find(reservation);
            return it == destination_generation_.end() ? 0 : it->second;
        }
        [[nodiscard]] std::uint64_t completedGeneration(std::uint64_t reservation) const
        {
            auto it = completed_ledger_.find(reservation);
            return it == completed_ledger_.end() ? 0 : it->second;
        }

        [[nodiscard]] bool rowComplete(std::uint32_t slot) const
        {
            auto it = rows_.find(slot);
            return it != rows_.end() && it->second.complete;
        }

        [[nodiscard]] std::string destinationContents(std::uint64_t reservation) const
        {
            auto it = destinations_.find(reservation);
            if (it == destinations_.end() || !it->second->materialized())
                throw std::logic_error("No materialized backing for this reservation");
            return std::string(reinterpret_cast<const char *>(it->second->data()),
                               static_cast<std::size_t>(it->second->size()));
        }

        [[nodiscard]] std::size_t grantBound(std::uint32_t slot) const
        {
            return rows_.at(slot).authority.maxGrants();
        }

        [[nodiscard]] bool destinationReleased(std::uint64_t reservation) const
        {
            auto it = destinations_.find(reservation);
            return it == destinations_.end() || it->second->released();
        }

        [[nodiscard]] bool rowSnapshotReleased(std::uint32_t slot) const
        {
            return snapshotLeaseReleased(rows_.at(slot).snapshot_lease);
        }

        /** @brief Test-visible payload of a completed restore lane (bytes copied),
         *         valid for a queued row AND for the attached execution row. */
        [[nodiscard]] std::string restorePayload(std::uint32_t slot) const
        {
            auto it = queued_.find(slot);
            const auto *q = (it != queued_.end()) ? &it->second : &rows_.at(slot);
            if (q->copy_lane_ == nullptr)
                throw std::logic_error("No restore lane on this slot");
            return std::string(reinterpret_cast<const char *>(q->copy_lane_->data()),
                               static_cast<std::size_t>(q->copy_lane_->size()));
        }

        [[nodiscard]] std::uint64_t queuedRestoreSession(std::uint32_t slot) const
        {
            if (queued_.count(slot) == 0)
                throw std::logic_error("No queued row for this slot");
            return queued_.at(slot).restore_session;
        }

        [[nodiscard]] bool rowMayExecute(std::uint32_t slot) const
        {
            return rows_.at(slot).authority.mayExecute();
        }

        [[nodiscard]] const std::map<std::uint32_t, EngineRow> &rows() const noexcept { return rows_; }
        [[nodiscard]] const std::map<std::uint32_t, EngineRow> &queued() const noexcept { return queued_; }

    private:
        /** @brief Cancellation release (engine-internal ONLY): a queued row that
         *         never ran has NO delivery obligation; a LIVE row owning the
         *         backing blocks the release. */
        void releaseForCancellation(std::uint64_t reservation)
        {
            for (const auto &[slot, row] : rows_)
            {
                (void)slot;
                if (row.destination_reservation_ == reservation)
                    throw std::logic_error(
                        "Cannot release a reservation a live row still owns");
            }
            auto it = destinations_.find(reservation);
            if (it == destinations_.end())
                throw std::logic_error("Destination reservation not owned (double release refused)");
            it->second->markReleased();
            destinations_.erase(it);
            removeClaim(reservation);
            completed_ledger_.erase(reservation);
            delivered_ledger_.erase(reservation);
        }

        /** @brief Only banked destination reservations carry capacity authority. */
        void removeClaim(std::uint64_t reservation)
        {
            auto it = std::find(claim_slots_.begin(), claim_slots_.end(), reservation);
            if (it != claim_slots_.end())
                claim_slots_.erase(it);
        }

        void requireDestinationUnclaimed(std::uint64_t reservation)
        {
            // Response ownership lives in the REGISTRY and OUTLIVES the execution
            // row: once B completed and was replaced by C, B's still-undelivered
            // reservation must remain unclaimable by D.
            if (std::find(claim_slots_.begin(), claim_slots_.end(), reservation) != claim_slots_.end())
                throw std::logic_error(
                    "Destination reservation is owned by an unfinished response");
        }
        void claimDestination(std::uint64_t reservation)
        {
            // NON-ALLOCATING commit: claim_slots_ is reserved to the pool width at
            // construction, so this cannot throw (an allocation failure can never
            // leave a row without its claim or a claim without its row).
            for (const auto r : claim_slots_)
                if (r == reservation)
                    throw std::logic_error("Destination reservation already claimed");
            claim_slots_.push_back(reservation);
        }

        [[nodiscard]] static std::uint64_t derivedPayload(std::uint64_t prompt_bytes,
                                                          std::uint64_t response_limit)
        {
            constexpr std::uint64_t kTokenBytes = 8;
            if (response_limit > (std::numeric_limits<std::uint64_t>::max() - prompt_bytes) / kTokenBytes)
                throw std::logic_error("Derived payload overflows (request geometry)");
            return prompt_bytes + response_limit * kTokenBytes;
        }

        [[nodiscard]] std::shared_ptr<DestinationBacking> destination_backing(
            std::uint64_t reservation) const
        {
            auto it = destinations_.find(reservation);
            if (it != destinations_.end())
                return it->second;
            throw std::logic_error("Destination reservation is not banked");
        }

        [[nodiscard]] static bool snapshotLeaseReleased(
            const std::shared_ptr<SnapshotBorrowState> &lease) noexcept
        {
            if (lease == nullptr)
                return false;
            std::lock_guard l(lease->mutex);
            return lease->released;
        }

        [[nodiscard]] bool queuedMayAttach(std::uint32_t slot) const
        {
            if (queued_.count(slot) == 0)
                return false;
            const auto &q = queued_.at(slot);
            return q.restore_state != RestoreState::Pending;
        }

        RowDecodeFn decode_;
        std::uint64_t serving_instance_ = 0;
        std::uint64_t engine_identity_ = 0; // unique per engine instance
        static inline std::atomic<std::uint64_t> next_engine_identity_{0};
        std::uint64_t grant_serial_ = 1;
        std::uint64_t snapshot_nonce_serial_ = 1;
        std::uint64_t restore_session_serial_ = 1;
        std::map<std::uint32_t, EngineRow> rows_;
        std::map<std::uint32_t, EngineRow> queued_;
        std::map<std::uint64_t, std::shared_ptr<DestinationBacking>> destinations_; ///< Banked backing.
        /** @brief Publication lifecycle: Publishing (consumer running) -> Delivered
         *         (accepted, retireable). */
        enum class DeliveryState : std::uint8_t { Publishing, Delivered };
        std::map<std::uint64_t, DeliveryState> delivered_ledger_; ///< Per-RESPONSE state (survives slot reuse).
        std::map<std::uint64_t, std::uint64_t> completed_ledger_; ///< Completion GENERATION per RESPONSE.
        std::vector<std::uint64_t> claim_slots_; ///< Claimed slots (reserved, non-allocating commit).
        std::size_t max_destinations_ = kDefaultPoolWidth; ///< Admitted plan-derived pool width.
        std::map<std::uint64_t, std::uint64_t> destination_generation_; ///< Per-ID generation (bounded reuse).
        bool failed_ = false;
        std::string failed_reason_;
        std::function<void(std::uint64_t, const std::byte *, std::size_t)> consumer_;
    };
} // namespace llaminar2
