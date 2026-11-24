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
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/serving/ServingWindowDecode.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "execution/serving/ServingWindowDecodeBridge.h"

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
    class IOrchestrationRunner; // forward (ABI decode runner binding)
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


    /** @brief TYPED destination-backing allocation seam (fault-observable,
     *         device-free): the production factory is BackingAllocator::system;
     *         tests may substitute a recording/throwing allocator. The deleter
     *         runs at the exact free boundary, so a recording allocator can
     *         observe whether the PMA claim is still held when the buffer is
     *         freed (member-order proof, not totals-only). */
    class BackingAllocator
    {
    public:
        using Bytes = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>;
        std::function<Bytes(std::size_t)> allocate;
        /** @brief Canonical production allocator: real host bytes, delete[] on free. */
        [[nodiscard]] static BackingAllocator system()
        {
            return BackingAllocator{[](std::size_t n) -> Bytes {
                return Bytes(new std::byte[n], [](std::byte *p) { delete[] p; });
            }};
        }
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
        /** @brief TYPED output metadata (read-only; the committed OUTPUT window —
         *         never prompt/restore state and never unwritten holes). */
        [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> outputMetadata() const noexcept
        {
            return {output_offset_, output_length_};
        }
        [[nodiscard]] std::uint64_t reservationTag() const noexcept { return reservation_; }
        [[nodiscard]] std::uint64_t wheelGenerationTag() const noexcept
        {
            return wheel_generation_;
        }
        [[nodiscard]] std::uint64_t rowGenerationTag() const noexcept
        {
            return row_generation_;
        }
    private:
        friend class GroupedServingEngine;
        void sealRow(const std::uint64_t row_generation) noexcept
        {
            row_generation_ = row_generation;
        }
        void deseal() noexcept
        {
            released_seal_ = true; // a released backing is unsealed (dead seal)
        }
    public:
        [[nodiscard]] bool sealed() const noexcept { return !released_seal_; }
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
        ~DestinationBacking() = default; // buffer + lease released by markReleased/destruction

    private:
        friend class GroupedServingEngine;
        friend class EngineOutputSink; // typed destination publication
        void markMaterialized(PhysicalMemorySuballocationLease claim,
                              const BackingAllocator &allocator)
        {
            if (buffer_ != nullptr || released_)
                throw std::logic_error("Materialization exactly once (before release)");
            // Pairwise rule: the (already claimed) lease is a LOCAL parameter; a
            // throwing OR null allocation result destroys/abandons it and returns
            // the preclaim to the reserve immediately (no retained claim beside a
            // failed allocation). Only after a NON-NULL result does the lease
            // move into the backing.
            auto bytes = allocator.allocate(capacity_);
            if (!bytes)
                throw std::logic_error(
                    "Destination backing allocator returned null (no retained claim)");
            buffer_ = std::move(bytes);
            lease_ = std::move(claim);
        }
        void markReleased()
        {
            if (released_)
                throw std::logic_error("Backing released exactly once");
            // REAL release: materialized payload bytes are actually freed, then the
            // child claim is returned to its reserve (lease destruction releases).
            buffer_.reset();
            lease_ = PhysicalMemorySuballocationLease{};
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
        std::uint64_t reservation_ = 0;  ///< Sealed destination identity (bank-time).
        std::uint64_t wheel_generation_ = 0; ///< Per-destination bank counter (wheel).
        std::uint64_t row_generation_ = 0;   ///< ADMITTED row generation (seal).
        // RETAINED-STATE FIELDS: the retained delivery payload lives INSIDE this
        // already-admitted backing object — there is NO separate RetainedPayload
        // allocation, NO metadata claim, NO retention-time heap allocation at
        // all. The state is TERNARY (#159): Pending (retained, never acquired),
        // Leased (the writer holds it), Released (the writer finished; pending
        // retirement) — an unacquired pending delivery can never be confused
        // with a released one.
        enum class RetainedState : std::uint8_t { None, Pending, Leased, Released };
        RetainedState retained_state_ = RetainedState::None;
        std::uint64_t retained_offset_ = 0;
        std::uint64_t retained_length_ = 0;
        std::uint64_t retained_wheel_generation_ = 0;
        bool released_seal_ = false;         ///< Released: the seal is dead.
        /** @brief PMA child claim retained beside the materialized bytes (the
         *         single capacity ledger; released with markReleased). Declared
         *         BEFORE the buffer so default destruction frees the buffer
         *         FIRST and then returns the claim (pairwise order). */
        PhysicalMemorySuballocationLease lease_;
        BackingAllocator::Bytes buffer_;
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
        /** @brief ADMITTED per-row law (ABI path): the exact law node + its
         *         owner are retained (the engine is the row owner and mints the
         *         ABI decode input from these); absent in the pure protocol
         *         mock path. */
        // RAII OWNER of the retained law node: any row destruction path
        // (admit/queue unwind, replacement, attach, engine dtor) releases the
        // pin exactly once — no manual release discipline, no leak.
        ServingLawNodePin law_pin;
        std::uint64_t transaction_epoch = 0;
        bool abi_state_failed = false; ///< State-violated ABI row: never retried.
        EngineRow() = default;
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

    /** @brief TYPED OUTPUT SINK over the engine's destination backing (design
     *         §10 publication): appends real window bytes into the backing (the
     *         backing owns the output metadata) and reports the committed output
     *         extent — the row owner uses it to publish the window. */
    class EngineOutputSink final : public ServingWindowOutputSink
    {
    public:
        /** @brief SEALED + STAGED: the sink is bound to the exact destination
         *         reservation + row generation (immutable; a stale sink cannot
         *         write), stages the window bytes, and PUBLISHES them only via
         *         commit() — the engine calls commit AFTER the authenticated
         *         completion; a refusal discards (no backing write, no stranded
         *         bytes, no write-before-valid-completion). */
        EngineOutputSink(DestinationBacking &backing, std::uint64_t prompt_bytes,
                         std::uint64_t reservation, std::uint64_t generation)
            : backing_(backing), prompt_bytes_(prompt_bytes),
              reservation_(reservation), generation_(generation)
        {
        }
        EngineOutputSink(const EngineOutputSink &) = delete;
        EngineOutputSink &operator=(const EngineOutputSink &) = delete;
        EngineOutputSink(EngineOutputSink &&) = delete;
        EngineOutputSink &operator=(EngineOutputSink &&) = delete;
        void writeOutput(const std::byte *src, std::uint64_t bytes) override
        {
            // AUTHENTICATED SEAL: the sink's tags must match the backing's sealed
            // destination identity (a downcast/stale sink never writes).
            if (!backing_.sealed() || backing_.reservationTag() != reservation_ ||
                backing_.rowGenerationTag() != generation_)
                throw std::logic_error(
                    "Serving output sink destination identity mismatch (unauthenticated)");
            // STAGE (APPEND: multi-chunk production writes accumulate), BOUNDED by
            // the admitted output region INCLUDING the already-committed output.
            if (bytes != 0)
            {
                if (src == nullptr)
                    throw std::logic_error("Serving output sink: null produce source");
                const auto existing = backing_.outputLength();
                const auto used = prompt_bytes_ + existing + staged_.size();
                if (used > backing_.capacity() || bytes > backing_.capacity() - used)
                    throw std::logic_error(
                        "Serving output sink exceeded the admitted destination region");
                staged_.insert(staged_.end(),
                               reinterpret_cast<const std::byte *>(src),
                               reinterpret_cast<const std::byte *>(src) + bytes);
            }
            staged_empty_ = true; // an explicit attempt (empty-vs-absent)
        }

    private:
        friend class GroupedServingEngine; // ENGINE-ONLY commit/discard (no downcast)
        void commit()
        {
            if (committed_)
                throw std::logic_error("Serving output sink committed twice");
            committed_ = true;
            if (!staged_.empty())
                backing_.writeAt(prompt_bytes_ + backing_.outputLength(),
                                 staged_.data(), staged_.size());
        }
        void discard() { staged_.clear(); staged_empty_ = false; }

    public:
        [[nodiscard]] std::uint64_t outputBytes() const override
        {
            return backing_.outputLength();
        }
        [[nodiscard]] bool published() const noexcept
        {
            return staged_empty_ || !staged_.empty();
        }
        [[nodiscard]] std::uint64_t promptBytes() const noexcept { return prompt_bytes_; }

    private:
        DestinationBacking &backing_;
        const std::uint64_t prompt_bytes_;   ///< SEALED: output base.
        const std::uint64_t reservation_;    ///< SEALED: destination identity.
        const std::uint64_t generation_;     ///< SEALED: row generation.
        std::vector<std::byte> staged_;      ///< Staged window bytes.
        bool staged_empty_ = false;          ///< Explicit empty publication.
        bool committed_ = false;             ///< Exactly-once commit (engine-only).
    };

    /** @brief TYPED per-row ABI deferral state. */
    enum class AbiDeferralState : std::uint8_t
    {
        Retryable,           ///< Grant untouched + nothing published: may retry.
        StateViolated,       ///< Original issuance consumed/replaced or bytes staged: NEVER retried.
        Cancelled,           ///< Authenticated cancellation (window deferred).
        MissingPublication,  ///< Successful completion without an explicit publish.
    };

    /** @brief TYPED ABI deferral record (state + precise diagnostic). */
    struct AbiDeferral
    {
        AbiDeferralState state = AbiDeferralState::Retryable;
        std::string reason;
    };

    /** @brief Pure per-row decode callback (production: row-invariant grouped kernel). */    /** @brief Pure per-row decode callback (production: row-invariant grouped kernel). */
    using RowDecodeFn = std::string (*)(std::uint64_t seed, std::uint64_t position,
                                        std::uint64_t window,
                                        const std::string &restored_prefix);

    /** @brief Grouped engine: one worker, N admitted rows, no cohort drain. */
    class GroupedServingEngine
    {
    public:
        /** @brief TYPED RESPONSE FAILURE OUTCOME (§10.3 separate outcomes):
         *         Execution (the invocation failed) vs Delivery (a post-success
         *         transport failure) — a delivery failure never overwrites the
         *         execution history. */
        enum class ResponseFailureKind : std::uint8_t { Execution, Delivery };
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
        explicit GroupedServingEngine(RowDecodeFn decode, std::uint64_t serving_instance,
                                      ServingPoolPlan plan,
                                      PhysicalMemoryOwnerReservation pool_capacity,
                                      BackingAllocator allocator)
            : decode_(decode), serving_instance_(serving_instance),
              max_destinations_(plan.destination_slots),
              engine_identity_(next_engine_identity_.fetch_add(1) + 1)
        {
            // #149 PREALLOCATED OWNERSHIP: the retained-metadata control
            // allowance for ALL slots is claimed ONCE here (the engine's own
            if (plan.destination_slots == 0)
                throw std::logic_error("Serving destination pool width must be positive");
            if (!pool_capacity.valid())
                throw std::logic_error(
                    "PMA-bound serving engine requires a valid serving-pool reservation "
                    "(no unbound engine path)");
            if (pool_capacity.owner() != PhysicalMemoryOwner::ServingResponsePool)
                throw std::logic_error(
                    "Serving engine pool reservation must own ServingResponsePool "
                    "(wrong-owner reservation cannot fund destination backing)");
            if (!pool_capacity.allocatorIdentity().device.is_cpu())
                throw std::logic_error(
                    "Serving engine destination backing is host (CPU) allocation; "
                    "the pool reservation must name the host allocator resource "
                    "(wrong-resource reservation cannot fund host backing)");
            if (!allocator.allocate)
                throw std::logic_error(
                    "BackingAllocator must be non-empty (typed allocator required)");
            // OWNED BY VALUE: the engine alone holds the admitted pool reservation
            // (single authority; no external shared_ptr pointee, so the admitted
            // owner/allocator identity CANNOT be rebound before a later claim).
            pool_capacity_ = std::move(pool_capacity);
            // #149 AGGREGATE CONTROL RESERVE: ONE claim covering ALL destination
            // control storage (the per-slot object/control/map-node geometry) for
            // the whole pool, held strongly by the ENGINE and ALSO by EVERY
            // delivery lease (copied at acquire). The credit returns when the
            // LAST of {engine, leases} drops — which is after EVERY backing
            // object's last use (a lease holds its backing until release and the
            // reserve until its own destruction) — a complete bundle with an
            // owner after ALL frees. CHECKED arithmetic (#157).
            if (controlBytesPerSlot() != 0 &&
                max_destinations_ > std::numeric_limits<std::uint64_t>::max() /
                                        controlBytesPerSlot())
                throw std::logic_error(
                    "Destination control reserve overflows the uint64 bound (#163 checked)");
            // TOKEN-FREE AGGREGATE ADMISSION: the reserve's credit is admitted
            // through the reservation's reserveAggregate (canonical owner-line
            // accounting, no destructor-reclaim claim token). The UNIQUE handle
            // is shared through a shared_ptr in the reserve block's allocator
            // state; the allocator's deallocate resets its shared ref AFTER
            // `delete p` (the reserve storage freed) -> the handle's release
            // returns the credit AFTER the storage free. The handle owns the
            // reservation STATE (not the reservation object) — it survives the
            // engine's own destruction (the W34 property holds).
            const auto aggregate_bytes = controlBytesPerSlot() * max_destinations_;
            auto admission = pool_capacity_.reserveAggregate(aggregate_bytes);
            // HANDOFF: the allocator's TYPED STATE CAPABILITY is constructed FROM
            // the token's record key FIRST (a throwing allocate_shared leaves
            // the local token owning the record -> the dtor ROLLS BACK); only
            // after the block exists is the transfer performed. The copies share
            // the state record (no token object, no heap H/F).
            control_reserve_ = std::allocate_shared<ControlReserve>(
                ReserveAllocator(admission.state(), admission.bytes(),
                                 admission.serial(), admission.nonce()));
            // #175/#176: THE TRANSFER HAPPENS AFTER the block allocation SUCCEEDED
            // (a throwing allocation leaves the record's receiver 0 -> the local
            // token's record-authoritative dtor ROLLS BACK; a former owner with
            // copied credentials CANNOT call transferRecord — it is private +
            // engine-friend-only). The secret is bound into the record; the
            // deallocate reads the CURRENT record receiver (authenticated).
            PhysicalMemoryAggregateHandle::transferRecord(
                admission.state(), admission.serial(), admission.nonce());
            allocator_ = std::move(allocator);
            // #149 AUTHORITY-RECORD IDENTITY: the engine retains the aggregate
            // admission key (the typed state capability + the serial + the
            // admitted amount) so the authenticated CONTROL-RESERVE RECORD
            // (the actual allocated size + the recorded pointer) is observable
            // through controlReserveRecord() — the authority's bound record,
            // not an estimate.
            aggregate_state_ = admission.state();
            aggregate_serial_ = admission.serial();
            aggregate_bytes_ = admission.bytes();
            claim_slots_.reserve(plan.destination_slots); // NON-ALLOCATING claim commit
            retained_.reserve(plan.destination_slots);     // PRECOMPUTED retained slots (no retention alloc)
        }

        /** @brief The engine's admitted pool reservation (single authority; the
         *         pinned identity for the engine lifetime). */
        /** @brief The destination's per-slot control admission (object + control
         *         block + slot bookkeeping; the AGGREGATE reserve admits it). */
        [[nodiscard]] std::uint64_t destinationControlBytes() const noexcept
        {
            return controlBytesPerSlot();
        }

        /** @brief The AGGREGATE control reserve (per-slot × the pool width). */
        /** @brief #149 AUTHENTICATED CONTROL-RESERVE RECORD (read-only
         *         diagnostic; requires a LIVE engine): the authority-bound
         *         record of the engine's OWN aggregate ControlReserve admission
         *         — the ADMITTED bytes, the ACTUAL allocated size (the
         *         allocator's n*sizeof), the recorded pointer and the retiring
         *         flag — an IMMUTABLE read-time snapshot. The record's erasure
         *         happens when the FINAL holder of the control reserve retires
         *         (the engine + the delivery leases share it — the W42 lease
         *         contract), NOT necessarily at the engine's own destructor, so
         *         a query after construction is a snapshot of that moment; the
         *         method is NOT valid on a destroyed engine. */
        [[nodiscard]] PhysicalMemoryAggregateHandle::RecordView
        controlReserveRecord() const noexcept
        {
            return PhysicalMemoryAggregateHandle::recordView(
                aggregate_state_, aggregate_serial_);
        }
        [[nodiscard]] std::uint64_t destinationReserveBytes() const noexcept
        {
            return controlBytesPerSlot() * max_destinations_;
        }

        /** @brief The aggregate admission's TYPED record-node coverage (the map
         *         node + the tuple the authority ledger admits per admission). */
        [[nodiscard]] std::uint64_t aggregateNodeBytes() const noexcept
        {
            // ONE PMA-owned typed charge (no duplicated production literal).
            return PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        }



        [[nodiscard]] const PhysicalMemoryOwnerReservation &servingPool() const noexcept
        {
            return pool_capacity_;
        }

        /** @brief Bind the ABI decode runner (the engine mints the typed
         *         serving-window command per row and advances by the
         *         AUTHENTICATED result). The mode is IMMUTABLE after the first
         *         admission (a mid-stream mode change is refused). */
        void setAbiRunner(IOrchestrationRunner *runner)
        {
            if (admission_started_)
                throw std::logic_error(
                    "ABI mode is immutable after admission (no mid-stream mode change)");
            abi_runner_ = runner;
        }
        [[nodiscard]] bool abiBridged() const noexcept { return abi_runner_ != nullptr; }
        /** @brief The LAST step's ABI deferrals (slot -> precise diagnostic; the
         *         per-row failure-atomic deferral is observable, never dropped). */
        [[nodiscard]] std::uint64_t windowEpoch() const noexcept { return window_epoch_; }
        [[nodiscard]] const std::map<std::uint32_t, AbiDeferral> &abiDeferrals() const noexcept
        {
            return last_abi_deferrals_;
        }
        /** @brief Read-only per-destination output metadata (the committed OUTPUT
         *         window (offset, length) of a banked destination). */
        [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> abiDestinationMetadata(
            std::uint64_t reservation) const
        {
            return destination_backing(reservation)->outputMetadata();
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
                      std::uint64_t prompt_bytes = 0,
                      const ServingLawBinding *law = nullptr,
                      std::uint64_t transaction_epoch = 0)
        {
            admission_started_ = true;
            if (abi_runner_ != nullptr && transaction_epoch == 0)
                throw std::logic_error(
                    "ABI-bridged engine requires a positive transaction epoch "
                    "(admission time; epoch-0 rows are not admitted)");
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
            if (abi_runner_ != nullptr &&
                (law == nullptr || !law->valid() ||
                 law->admissionIdentityOfNode() == 0))
                throw std::logic_error(
                    "ABI-bridged engine requires a LIVE admitted per-row law "
                    "(no silent scalar mock; released/default law refused)");
            if (abi_runner_ != nullptr && transaction_epoch == 0)
                throw std::logic_error(
                    "ABI-bridged engine requires a positive transaction epoch "
                    "(admission time; epoch-0 rows are not admitted)");
            if (law != nullptr)
            {
                row.law_pin = ServingLawNodePin(*law); // RAII retain (holder pin)
                row.transaction_epoch = transaction_epoch;
            }
            {
                auto backing = destination_backing(destination_reservation);
                backing->sealRow(row.state.generation); // SEAL row->destination
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
                          std::uint64_t prompt_bytes = 0,
                          const ServingLawBinding *law = nullptr,
                          std::uint64_t transaction_epoch = 0)
        {
            admission_started_ = true;
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
            if (abi_runner_ != nullptr &&
                (law == nullptr || !law->valid() ||
                 law->admissionIdentityOfNode() == 0))
                throw std::logic_error(
                    "ABI-bridged engine requires a LIVE admitted per-row law "
                    "(no silent scalar mock; released/default law refused)");
            if (abi_runner_ != nullptr && transaction_epoch == 0)
                throw std::logic_error(
                    "ABI-bridged engine requires a positive transaction epoch "
                    "(admission time; epoch-0 rows are not admitted)");
            if (law != nullptr)
            {
                row.law_pin = ServingLawNodePin(*law); // RAII retain (holder pin)
                row.transaction_epoch = transaction_epoch;
            }
            {
                auto backing = destination_backing(destination_reservation);
                backing->sealRow(row.state.generation); // SEAL row->destination
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
            if (bytes > std::numeric_limits<std::size_t>::max() - 1)
                throw std::logic_error(
                    "Destination capacity overflows the allocation size domain (#156)");
            // A re-bank REFUSES while a RETAINED delivery payload is LEASED
            // (the writer still holds it) or while a terminal-error reason is
            // undelivered (#151). A retained entry that is NOT leased is a
            // RETIRED delivery (the lease released): it is unlinked here
            // (non-allocating; the backing dies) and the re-bank proceeds.
            // Everything runs BEFORE any mutation.
            {
                // ORDER (#159): ALL refusals run BEFORE any unlink — a rejected
                // bank never destroys bytes. A PENDING/LEASED retained delivery
                // refuses (undelivered); only a RELEASED entry is retired.
                if (retainedPending(reservation))
                    throw std::logic_error(
                        "Cannot re-bank while a retained delivery is pending/leased (undelivered)");
                {
                    auto rit = response_failures_.find(reservation);
                    if (rit != response_failures_.end() && !rit->second.reason_consumed)
                        throw std::logic_error(
                            "Cannot re-bank while the terminal error reason is undelivered");
                }
                if (auto retained = retainedFind(reservation))
                    retainedUnlink(reservation, retained.get()); // RELEASED -> retire
            }
            // The generation is a PER-SLOT monotonic counter: the map holds at most
            // kMaxDestinations entries forever (bounded history, never a fresh
            // historical ID) and every re-bank bumps it (ABA-closed delivery keys).
            destination_generation_[reservation] += 1;
            // #149-NARROW: the backing OBJECT + its shared control block + the
            // destinations-map node are a REAL separate allocation chain that the
            // capacity-byte claim does NOT cover. They are admitted by a per-slot
            // CONTROL claim bound DIRECTLY to the backing object via a custom
            // deleter on the backing's own shared_ptr: the deleter runs AFTER
            // `delete p` (the object storage freed) and only then returns the
            // control credit — the credit's owner is the object's allocation
            // itself (never a member position or a wrapper).
            // The backing object is a PLAIN allocate (its storage is covered by
            // the engine's aggregate ControlReserve — see the constructor; no
            // per-bank claim, no custom deleter, no holder allocation).
            destinations_[reservation] = std::make_shared<DestinationBacking>(
                DestinationBacking::Key{}, bytes);
            // A re-bank is a NEW occupant: the prior response's completion,
            // delivery ledger and failure record are ALL cleared (a stale
            // failure cannot authorize a release of the NEW backing; a stale
            // completion cannot gate the NEW occupant).
            response_failures_.erase(reservation);
            completed_ledger_.erase(reservation);
            delivered_ledger_.erase(reservation);
            // (the retained list entry is unlinked above when not leased)
            // SEALED identity: the backing knows its destination + generation at
            // bank time (the output sink authenticates against these).
            destinations_[reservation]->reservation_ = reservation;
            destinations_[reservation]->wheel_generation_ = destination_generation_[reservation];
        }

        /** @brief Producer materialization event: once, before any issue backs it.
         *
         *         The backed claim is the EXACT capacity (S3.1 buffer-before-
         *         credit order). The retained-metadata control allowance is a
         *         PREALLOCATED pool-level claim owned by the engine (see the
         *         constructor) — never part of the destination claim. */
        void materializeDestination(std::uint64_t reservation)
        {
            auto it = destinations_.find(reservation);
            if (it == destinations_.end())
                throw std::logic_error("Destination reservation is not banked");
            auto claim = claimPoolBytes(it->second->capacity()); // EXACT backing claim (S3.1)
            it->second->markMaterialized(std::move(claim), allocator_);
        }

        /** @brief Claim the exact bytes from the PINNED admitted pool reservation
         *         (the reservation is owned by value; the identity cannot be
         *         rebound — a rebind is impossible and never silently claimed). */
        [[nodiscard]] PhysicalMemorySuballocationLease claimPoolBytes(std::uint64_t bytes) const
        {
            return pool_capacity_.claimAllocation(bytes);
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
            it->second->deseal();    // the seal dies with the release
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
            if (response_failures_.count(reservation) != 0)
                throw std::logic_error(
                    "A failed response cannot be delivered (failure is final)");
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
            queued_.erase(slot); // pending copies retired with the row (pin RAII)
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

            if (window_epoch_ == std::numeric_limits<std::uint64_t>::max())
                throw std::logic_error("Serving window epoch exhausted (unguarded wrap)");
            ++window_epoch_; // sealed per-step window epoch (monotonic)
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
            std::vector<std::uint64_t> failed_reservations; // violated-row releases (post-attach)
            last_abi_deferrals_.clear();
            auto validate = [&](std::uint32_t slot, EngineRow &r)
            {
                const auto end = nextWindowEnd(r.state.committed, r.state.window);
                std::string out;
                if (abi_runner_ != nullptr && r.law_pin.retainedNode() != nullptr)
                {
                    // ABI PATH (READ-ONLY validation): the prospective is bounded
                    // by the intrinsic-grid bound (no decode, no mutation — the
                    // AUTHENTIC decode + commit run in PHASE 2 per row, where a
                    // failure defers ONLY this row and leaves its grant untouched).
                    out.assign(static_cast<std::size_t>(end - r.state.committed), '\x01');
                }
                else
                {
                    for (std::uint64_t pos = r.state.committed + 1; pos <= end; ++pos)
                        out += decode_(r.state.rng.seed, pos, r.state.window.quantum,
                                       r.restored_prefix_);
                }
                const auto prospective = r.prompt_bytes_ + r.emitted_bytes_ + out.size();
                if (prospective > r.backing_->capacity())
                {
                    failed_ = true;
                    failed_reason_ = "Live output exceeded the materialized destination backing";
                    throw std::logic_error(failed_reason_);
                }
                pending_outputs.emplace_back(slot, std::move(out));
            };
            for (auto &[slot, row] : rows_)
            {
                if (row.complete || row.throttled || row.abi_state_failed)
                    continue; // a state-violated ABI row is NEVER retried
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
                rows_.erase(slot); // retire the old row FIRST (its pin RAII-releases)
                // EMPLACE the successor DIRECTLY from the queued entry — an
                // allocation failure during the emplace leaves the QUEUED entry
                // INTACT (never a pre-moved, pre-erased successor). Only after
                // the emplace succeeded is the queue row erased (its pin releases).
                rows_.emplace(slot, std::move(queued_[slot]));
                queued_.erase(slot);
                produced[slot] +=
                    "|entry:" + std::to_string(rows_[slot].state.generation);
            }
            // PHASE 2 (execute): all validations passed, so every mutation below is
            // committed only for the wholly validated set.
            for (auto &[slot, row] : rows_)
            {
                if (row.complete || row.throttled)
                    continue;
                // ABI rows: the decode CONSUMED the row's grant exactly once (the
                // authenticated completion) — the window applies; the next grant
                // is re-issued by issueNextGrant on the protocol path. The mock
                // path keeps the mayExecute gate (no consumption).
                const bool abi_row = abi_runner_ != nullptr && row.law_pin.retainedNode() != nullptr;
                if (!row.authority.mayExecute() && !abi_row)
                    continue;
                if (pending_outputs.empty())
                    continue;
                row.mtp_verify_calls_this_step = 0; // SHARED step-scoped reset (both paths)
                std::string out; // window output (ABI publication or mock fill)
                bool abi_completed = false;
                auto it = std::find_if(pending_outputs.begin(), pending_outputs.end(),
                                       [&](const auto &p) { return p.first == slot; });
                if (it == pending_outputs.end())
                    continue;
                if (abi_row)
                {
                    // AUTHENTIC ABI COMMIT (per-row transaction): the engine (row
                    // owner) mints the typed command with the row's OWN progress +
                    // full issuance pins and runs the ACCEPTED checked decode
                    // (ingress + correspondence + state contract). The grant is
                    // consumed exactly once BY THE AUTHENTIC completion; a refusal
                    // defers ONLY this row (its grant untouched) — never the whole
                    // step, never a rollback.
                    auto backing = destination_backing(row.destination_reservation_);
                    EngineOutputSink sink(*backing, row.prompt_bytes_,
                                          row.destination_reservation_,
                                          row.state.generation);
                    const auto before_output = sink.outputBytes();
                    const bool terminal = servingWindowIsTerminal(
                        row.state.committed, row.state.window.quantum,
                        row.state.window.response_limit);
                    ServingWindowDecodeInput input(
                        row.authority, row.law_pin.retainedNode(), row.law_pin.owner(), row.state.rng.seed,
                        row.transaction_epoch, row.authority.current().slot,
                        row.authority.current().serving_instance,
                        row.authority.authorityIdentity(),
                        row.authority.current().generation,
                        row.authority.current().grant_id, row.authority.issuanceNonce(),
                        window_epoch_, row.state.committed, terminal,
                        row.state.committed == 0, &sink);
                    // A THROWING production decode (a staging/geometry violation
                    // inside the runner) is CONTAINED: the row is state-violated
                    // (never retried), the diagnostic is retained.
                    bool decode_threw = false;
                    auto res = [&]() -> ServingWindowDecodeResult {
                        try
                        {
                            return abiDecodeChecked(*abi_runner_, input);
                        }
                        catch (const std::exception &e)
                        {
                            decode_threw = true; // STRUCTURED signal (no prose search)
                            return ServingWindowDecodeResult::failure(
                                std::string("decode threw: ") + e.what());
                        }
                    }();
                    if (!res.ok() && !res.cancelled())
                    {
                        // CLASSIFIED deferral: a retryable refusal leaves the grant
                        // untouched + NOTHING staged (deferrable); a STATE
                        // VIOLATION (the original issuance consumed/replaced OR
                        // bytes already STAGED — the dirty-only branch) is NEVER
                        // retried (replaying would double-write/strand).
                        const bool original_consumed =
                            row.authority.authorityIdentity() != input.expected_authority ||
                            !row.authority.mayExecute() ||
                            row.authority.current().grant_id != input.expected_grant_id;
                        const bool dirty = sink.published(); // BEFORE discard
                        const bool threw = decode_threw; // structured (never prose)
                        sink.discard(); // no staged bytes survive a refusal
                        if (original_consumed || dirty || threw)
                        {
                            row.abi_state_failed = true;
                            last_abi_deferrals_.emplace(
                                slot, AbiDeferral{AbiDeferralState::StateViolated,
                                                  res.error()});
                        }
                        else
                        {
                            last_abi_deferrals_.emplace(
                                slot, AbiDeferral{AbiDeferralState::Retryable, res.error()});
                        }
                        continue;
                    }
                    if (res.cancelled())
                    {
                        sink.discard();
                        last_abi_deferrals_.emplace(
                            slot, AbiDeferral{AbiDeferralState::Cancelled,
                                              "authenticated cancellation (window deferred)"});
                        // CANCELLATION IS A NON-TERMINAL DEFER: the row stays
                        // LIVE + reissuable; the destination stays OWNED by the
                        // row. There is NO retention and NO ownership transfer on
                        // a cancelled window (a later continuation executes
                        // against the SAME destination — never a lost backing).
continue;
                    }
                    if (!sink.published())
                    {
                        // ABSENCE of a publish is a violation (never a heuristic
                        // fallback); an EXPLICIT EMPTY publish (published=true,
                        // delta 0) is a legitimate zero-extent window.
                        row.abi_state_failed = true;
                        last_abi_deferrals_.emplace(
                            slot, AbiDeferral{AbiDeferralState::MissingPublication,
                                              "production decode published no output (violation)"});
                        sink.discard(); // no bytes survive a violation
                        continue;
                    }
                    // FULL-WINDOW EXCEPTION CONTAINMENT: the ENTIRE window
                    // application (authenticated advance, MTP verification before
                    // the commit, the commit, out.assign, the emitted advance)
                    // is inside the try — ANY throw classifies the row
                    // state-violated (never retried, never half-applied).
                    try
                    {
                        row.state.committed += res.produced(); // authenticated advance
                        if (row.mtp_max_depth > 0)
                            verifyMTP(slot);                  // exactly once, pre-commit
                        sink.commit();
                        const auto after_output = sink.outputBytes();
                        const auto delta = after_output - before_output;
                        out.assign(reinterpret_cast<const char *>(
                                       backing->data() +
                                       (sink.promptBytes() + before_output)),
                                   static_cast<std::size_t>(delta));
                        row.emitted_bytes_ += out.size();
                    }
                    catch (const std::exception &e)
                    {
                        row.abi_state_failed = true;
                        sink.discard();
                        last_abi_deferrals_.emplace(
                            slot, AbiDeferral{AbiDeferralState::StateViolated,
                                              std::string("window application failed: ") + e.what()});
                        continue;
                    }
                    const bool abi_terminal =
                        res.termination() == WindowTermination::Terminal ||
                        res.termination() == WindowTermination::EndOfSequence ||
                        res.termination() == WindowTermination::Forced ||
                        res.termination() == WindowTermination::StopMatched ||
                        row.state.committed >= row.state.response_limit;
                    if (abi_terminal)
                        row.complete = true;
                    // FALL THROUGH to the shared output/completion block (no
                    // continue): the ABI row's completion registration + the
                    // queued-row attach run through the SAME path.
                    produced[slot] = std::move(out);
                    abi_completed = true;
                    // shared completion registration runs below (no continue).
                }
                else
                {
                    out = std::move(it->second); // pending mock output (phase-B validated)
                    const auto end = nextWindowEnd(row.state.committed, row.state.window);
                    row.state.committed = end;
                    produced[slot] = out;
                    // REAL publication (mock path): output bytes physically written.
                    row.emitted_bytes_ += out.size();
                    row.backing_->writeAt(row.prompt_bytes_ + row.emitted_bytes_ - out.size(),
                                          reinterpret_cast<const std::byte *>(out.data()),
                                          out.size());
                }
                // SHARED completion registration (both paths):
                // One window per grant (consumed) — MOCK path only: the ABI
                // production decode already consumed the grant exactly once.
                if (!abi_row)
                    row.authority.resolve();
                if (row.mtp_max_depth > 0 && !abi_row)
                    verifyMTP(slot); // mock path: exactly once per step (ABI path verified pre-commit)
                if (row.state.committed >= row.state.response_limit)
                    row.complete = true; // mock path completion
                if (row.complete) // SHARED trigger: ABI EOS/forced/stop complete too
                {
                    // #164: a completed row can never write again — its backing
                    // ref is dropped at completion (the old object releases;
                    // a re-bank's new object never accumulates beyond the live-row
                    // set; the aggregate reserve stays M*N).
                    row.backing_.reset();
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
            // FAILED-ROW RETIREMENT: a state-violated ABI row is retired NOW (the
            // slot + destination released, the queued request may enter) — it is
            // never retried AND never strands the lifecycle.
            // VIOLATED ROWS ERASED FIRST (no partial mutation): a violated row is
            // removed BEFORE the post-pass (pin RAII releases; the destination has
            // NO delivery obligation); the waiting successor is EMPLACED afresh
            // (never a member-wise assignment into a violated row) and the old
            // destination releases AFTER the replacement.
            for (auto it = rows_.begin(); it != rows_.end();)
            {
                auto &row = it->second;
                if (row.abi_state_failed)
                {
                    const auto reservation = row.destination_reservation_;
                    const auto slot = row.state.slot;
                    failed_reservations.push_back(reservation);
                    // ALWAYS RECORD the failed invocation (Execution-outcome
                    // failure keyed by the RESPONSE WHEEL generation + diagnostic)
                    // — a violated row is retired WITH a failure record, never a
                    // silent cancellation-style discard (fresh-execution failures
                    // included; no completion gate).
                    response_failures_.erase(reservation);
                    response_failures_.emplace(
                        reservation, ResponseFailure{
                            wheelGeneration(reservation),
                            ResponseFailureKind::Execution,
                            last_abi_deferrals_.count(slot) != 0
                                ? last_abi_deferrals_.at(slot).reason
                                : "state-violated row retired"});
                    if (queuedMayAttach(slot))
                        completions.emplace_back(slot);
                    it = rows_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            // Post-pass replacements (completed slots with a ready queued row):
            // applied AFTER the iteration so no iterator is invalidated.
            for (const auto slot : completions)
            {
                if (rows_.count(slot) == 0)
                {
                    // A violated predecessor was ERASED: emplace DIRECTLY from the
                    // queued entry (the map node allocation happens BEFORE the
                    // EngineRow move — an allocation failure leaves the queued
                    // entry INTACT, never moved-out, never lost); only then erase.
                    rows_.emplace(slot, std::move(queued_[slot]));
                    queued_.erase(slot);
                }
                else
                {
                    // MOVE-ASSIGN into the existing (completed) row: the EngineRow
                    // move-assign reaches WindowGrantAuthority's same-instance
                    // operator= (the credited bound-transfer fix is DIRECTLY used).
                    rows_.at(slot) = std::move(queued_[slot]);
                    queued_.erase(slot);
                }
                produced[slot] += "|entry:" + std::to_string(rows_[slot].state.generation);
            }
            for (const auto reservation : failed_reservations)
            {
                // EXECUTION-FAILURE RETENTION: the committed staged output is
                // snapshotted BEFORE the release (a violated execution still
                // delivers the partial + the terminal error — never dropped).
                // FAIL-FAST RETENTION: the retention MUST succeed BEFORE the
                // original backing is released. A snapshot failure (PMA
                // exhaustion, allocation, guard) PROPAGATES — the sole original
                // output is NEVER freed after a swallowed retention failure
                // (#148); the caller observes the fatal state. A row with NO
                // staged bytes retains NOTHING (no metadata at all).
                if (destinations_.count(reservation) != 0 &&
                    destinations_.at(reservation)->outputLength() != 0)
                {
                    // ZERO-ALLOCATION RETAIN: the backing re-hosts onto the
                    // retained list (no allocation can fail after the transfer
                    // starts; the only refusal is the outstanding-lease guard).
                    if (auto existing = retainedFind(reservation);
                        existing &&
                        existing->retained_state_ != DestinationBacking::RetainedState::None &&
                        existing->retained_state_ != DestinationBacking::RetainedState::Released)
                        throw std::logic_error(
                            "Cannot re-retain while a delivery lease is outstanding");
                    retainDestination(reservation);
                }
                else
                {
                    // A zero-output violated row has NOTHING to retain: the
                    // backing is released below (no delivered bytes).
                }
                // The violated row was erased (no live owner); release has no
                // delivery obligation and happens exactly once.
                if (!destinationReleased(reservation))
                    releaseForCancellation(reservation);
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
    public:
        /** @brief RESPONSE-BOUND FAILURE (§10.3 exact-once): a FAILED model
         *         response is recorded with its EXACT generation + reason, its
         *         destination releases exactly once (no delivery obligation), a
         *         later delivery is refused, and a second failure is refused. */
        void failResponse(std::uint64_t reservation, std::uint64_t generation,
                          const std::string &reason,
                          ResponseFailureKind kind = ResponseFailureKind::Delivery)
        {
            // A PUBLIC failure response is BY DEFINITION a POST-SUCCESS event
            // (the execution completed): reporting it as an Execution failure
            // would overwrite the successful execution outcome (§10.3 separate
            // outcomes). Pre-completion execution failures are the interior
            // violated-row path and never call this API.
            if (kind == ResponseFailureKind::Execution)
                throw std::logic_error(
                    "Public failResponse is post-success only (use Delivery);"
                    " a pre-completion execution failure is the interior path");
            auto dit = completed_ledger_.find(reservation);
            if (dit == completed_ledger_.end())
                throw std::logic_error(
                    "Cannot fail a response before its completion (admission gate)");
            if (dit->second != generation)
                throw std::logic_error("Failed response generation mismatch (stale/ABA)");
            if (response_failures_.count(reservation) != 0)
                throw std::logic_error("Response already failed (exactly once)");
            // A response with a BORROWED buffer (Publishing) cannot be failed:
            // the consumer holds the byte buffer and may still be reading. A
            // DELIVERED response is a final delivery outcome and cannot be
            // overwritten by a failure (§10.3 separate outcomes).
            auto dstate = delivered_ledger_.find(reservation);
            if (dstate != delivered_ledger_.end() &&
                dstate->second == DeliveryState::Publishing)
                throw std::logic_error(
                    "Cannot fail a Publishing response (buffer borrowed by the consumer)");
            if (dstate != delivered_ledger_.end() &&
                dstate->second == DeliveryState::Delivered)
                throw std::logic_error(
                    "Cannot fail a Delivered response (delivery outcome is final)");
            response_failures_.emplace(
                reservation, ResponseFailure{generation, kind, reason});
            auto it = destinations_.find(reservation);
            if (it != destinations_.end())
            {
                it->second->markReleased();
                it->second->deseal(); // the seal dies with the failure release too
                destinations_.erase(it);
                removeClaim(reservation);
            }
            delivered_ledger_.erase(reservation);
        }
        [[nodiscard]] bool responseFailed(std::uint64_t reservation) const noexcept
        {
            return response_failures_.count(reservation) != 0;
        }



        /** @brief EXACTLY-ONCE terminal-error REASON consumption (#151): the writer
         *         consumes the failed response's reason before a re-bank can
         *         proceed (the reason is owed; a re-bank refuses while it is
         *         undelivered). A second consume is refused. */
        /** @brief GENERATION-BOUND exactly-once terminal-reason consumption
         *         (#151): the writer consumes the reason for its EXACT failure
         *         generation — a delayed OLD take (previous occupant) can never
         *         consume a NEW occupant's reason (mismatch refused). The
         *         record is MARKED consumed, NOT erased: the terminal history
         *         stays and failResponse remains refused (a post-take
         *         re-failure cannot succeed against the surviving ledger). */
        std::string takeFailureReason(std::uint64_t reservation,
                                      std::uint64_t generation)
        {
            auto it = response_failures_.find(reservation);
            if (it == response_failures_.end())
                throw std::logic_error("No response failure reason to consume");
            if (it->second.generation != generation)
                throw std::logic_error(
                    "Failure reason generation mismatch (delayed stale take refused)");
            if (it->second.reason_consumed)
                throw std::logic_error("Failure reason already consumed (exactly once)");
            auto reason = it->second.reason; // COPY FIRST (may throw before the mark)
            it->second.reason_consumed = true; // ONLY after the copy succeeded
            return reason;
        }

        /** @brief The AGGREGATE CONTROL RESERVE: one claim admitted for the whole
         *         engine's destination-control storage (object + control block +
         *         slot bookkeeping per destination). Strongly owned by the engine
         *         AND copied into every delivery lease: the credit returns only
         *         when the LAST holder drops — after every backing's last use. */
        struct ControlReserve
        {
        };

    private:
        /** @brief The RESERVE ALLOCATOR (PRIVATE, engine-internal): the state is
         *         the TYPED AggregateStateRef capability + the admission's record
         *         key (bytes, serial, random nonce) — no token object inside the
         *         allocator, no heap H/F. The deallocate begins the exclusive
         *         retirement, frees the reserve storage, then consumes the
         *         authenticated record (the canonical credit + the erase are the
         *         last events of the block). */
    private:
        /** @brief The RESERVE ALLOCATOR (PRIVATE — engine-internal only). The
         *         allocator state is the copyable admission key; every copy
         *         SHARES the single-allocation binding (the exact pointer this
         *         instance allocated). The deallocate requires the binding to
         *         match (a different allocation/copy under the same record is a
         *         fatal contract violation — never a cross-claim release) and
         *         consumes the record ONLY with the current transferred secret. */
        template <class T>
        class ReserveAllocatorT
        {
        public:
            using value_type = T;
            explicit ReserveAllocatorT(
                const PhysicalMemoryAggregateHandle::AggregateStateRef &state,
                std::size_t bytes, std::uint64_t serial,
                std::uint64_t nonce)
                : state_(state), bytes_(bytes), serial_(serial), nonce_(nonce)
            {
            }
            ReserveAllocatorT(const ReserveAllocatorT &other) = default;
            template <class U>
            ReserveAllocatorT(const ReserveAllocatorT<U> &other)
                : state_(other.state_), bytes_(other.bytes_), serial_(other.serial_),
                  nonce_(other.nonce_)
            {
            }
            ReserveAllocatorT(ReserveAllocatorT &&other) noexcept = default;
            friend class ReserveAllocatorT<T>;
            template <class U>
            friend class ReserveAllocatorT;
            template <class U>
            struct rebind
            {
                using other = ReserveAllocatorT<U>;
            };
            [[nodiscard]] T *allocate(std::size_t n)
            {
                // SINGLE-ALLOCATION ADMISSION: the PENDING admission happens
                // BEFORE the fallible operator new; a throwing new cancels it
                // (releaseAdmission with the null pointer) — failure atomicity.
                // #149 ENFORCED GEOMETRY (BEFORE the allocation): the pending
                // admission admits the EXACT requested byte count (the standard's
                // allocate_shared n*sizeof(T) — the real physical allocation the
                // system allocator will perform) and the aggregate reserve REFUSES
                // a fit violation BEFORE the fallible operator new (a throwing new
                // still cancels the pending admission).
                PhysicalMemoryAggregateHandle::allocateAdmission(
                    state_, serial_, nonce_, n * sizeof(T));
                T *ptr = nullptr;
                try
                {
                    ptr = static_cast<T *>(::operator new(
                        n * sizeof(T), std::align_val_t{alignof(T)}));
                }
                catch (...)
                {
                    PhysicalMemoryAggregateHandle::releaseAdmission(
                        state_, serial_, nonce_, nullptr); // cancel the pending admission
                    throw;
                }
                PhysicalMemoryAggregateHandle::admitPointer(
                    state_, serial_, nonce_, static_cast<void *>(ptr));
                return ptr;
            }
            void deallocate(T *p, std::size_t) noexcept
            {
                // EXCLUSIVE RETIREMENT TRANSITION BEFORE ANY STORAGE TOUCH: the
                // record atomically transitions to retiring (the SECOND release
                // call REFUSES — only one deallocate may proceed); the record
                // is NOT cleared here (no credit/reuse while p lives).
                if (!PhysicalMemoryAggregateHandle::beginRetirement(
                        state_, serial_, nonce_, static_cast<void *>(p)))
                    std::terminate(); // wrong/duplicate physical release -> fatal
                ::operator delete(p, std::align_val_t{alignof(T)}); // block freed
                // SEALED LIFETIME: the consume (the canonical credit + the node
                // erase) runs WHILE the record is still retiring + live (no
                // re-admission window before the credit — a releaseAdmission
                // clearing first would open exactly that window). ONLY after the
                // credit is the admission cleared; the record is ERASED by the
                // consume, so a stale copy's old pointer can never match a NEW
                // re-admitted record: addresses may recur, but the seal is the
                // ERASED record identity (a new record carries a new
                // serial/nonce) — NOT address inequality.
                PhysicalMemoryAggregateHandle::consumeRecord(
                    state_, serial_, nonce_,
                    PhysicalMemoryAggregateHandle::recordReceiver(state_, serial_));
                // The consume ERASES the record (the sealed lifetime completes;
                // the admission count/pointer/retiring die with it). The
                // releaseAdmission remains ONLY for the allocate-failure cancel
                // (a pending record that still exists).
                // (the AggregateStateRef drops with this allocator — no reset() needed)
            }
            template <class U>
            [[nodiscard]] bool operator==(const ReserveAllocatorT<U> &other) const noexcept
            {
                return state_.stateIdentity() == other.state_.stateIdentity() &&
                       serial_ == other.serial_;
            }
            template <class U>
            [[nodiscard]] bool operator!=(const ReserveAllocatorT<U> &other) const noexcept
            {
                return !(*this == other);
            }

        private:
            template <class U>
            friend class ReserveAllocatorT;
            PhysicalMemoryAggregateHandle::AggregateStateRef state_;
            std::size_t bytes_ = 0;
            std::uint64_t serial_ = 0;
            std::uint64_t nonce_ = 0;
        };
        using ReserveAllocator = ReserveAllocatorT<ControlReserve>;

        /** @brief RETAINED DELIVERY PAYLOAD — ZERO NEW ALLOCATION (#149 closed):
         *         the retained payload IS the destination backing object itself
         *         (already admitted + allocated). The retention re-hosts the
         *         backing on a BOUNDED linked list (the pool width) — there is
         *         NO RetainedPayload allocation, NO ledger map node, NO metadata
         *         claim, NO retention-time heap allocation whatsoever: the
         *         retained metadata lives in the backing's own (already-admitted)
         *         storage, so there is no separate metadata free-before-credit
         *         boundary and no allocator geometry to estimate. The backing's
         *         PMA credit returns with the backing's own destruction (the
         *         accepted S3.1 buffer-before-credit order), AFTER every reader
         *         (the writer's lease) — the lease shares the SAME object. */
        class RetainedPayloadLease
        {
        public:
            RetainedPayloadLease(const RetainedPayloadLease &) = delete;
            RetainedPayloadLease &operator=(const RetainedPayloadLease &) = delete;
            RetainedPayloadLease(RetainedPayloadLease &&other) noexcept
                : payload_(std::move(other.payload_)), reserve_(std::move(other.reserve_)),
                  live_(other.live_)
            {
                other.live_ = false;
            }
            ~RetainedPayloadLease() { release(); }

            [[nodiscard]] bool live() const noexcept { return live_; }

            /** @brief Whether THIS lease still pins the aggregate control
             *         reserve (the #149 last-holder counter — the canonical
             *         reserve-credit holder state, independent of the payload
             *         accessors). */
            [[nodiscard]] bool reserveHeld() const noexcept
            {
                return reserve_ != nullptr;
            }
            [[nodiscard]] std::uint64_t generation() const noexcept
            {
                return payload_ ? payload_->retained_wheel_generation_ : 0;
            }
            [[nodiscard]] std::string_view bytes() const noexcept
            {
                if (!payload_)
                    return {};
                return std::string_view(
                    reinterpret_cast<const char *>(payload_->data() +
                                                   payload_->retained_offset_),
                    payload_->retained_length_);
            }
            void release() noexcept
            {
                if (!live_)
                    return;
                // The lease's release clears the exactly-once flag ON THE SHARED
                // backing object (the retained list still owns a ref — the
                // object is alive) and drops the lease's own ref. No engine
                // access, no allocation, no pointer-identity map: the retained
                // entry is UNLEASED after the release and a later re-bank
                // retires it (non-allocating unlink).
                if (payload_)
                    payload_->retained_state_ = DestinationBacking::RetainedState::Released;
                payload_.reset();   // the backing (payload-before-reserve edge preserved)
                reserve_.reset();   // #165: an inactive handle does NOT pin the aggregate
                live_ = false;
            }

        private:
            explicit RetainedPayloadLease(std::shared_ptr<DestinationBacking> payload,
                                           std::shared_ptr<ControlReserve> reserve)
                : payload_(std::move(payload)), reserve_(std::move(reserve)), live_(true)
            {
            }
            friend class GroupedServingEngine;
            std::shared_ptr<DestinationBacking> payload_;
            std::shared_ptr<ControlReserve> reserve_; ///< Strong: outlives the backing's last use.
            bool live_ = true;
        };

    private:
        [[nodiscard]] std::uint64_t controlBytesPerSlot() const noexcept
        {
            return kDestinationControlBytes;
        }

        /** @brief BOUNDED retained VECTOR (PRIVATE; storage preallocated at the
         *         engine construction — capacity == the pool width, so the
         *         retention NEVER allocates and entries are INDEPENDENT (no
         *         cross-entry next-link pinning: a lease of one retained entry
         *         can never pin an unrelated entry — #160 closed). */
        std::shared_ptr<DestinationBacking> retainedFind(std::uint64_t reservation) const
        {
            for (const auto &entry : retained_)
                if (entry && entry->reservation_ == reservation)
                    return entry;
            return nullptr;
        }

        /** @brief RETAIN (zero-allocation re-host): the destination's backing is
         *         moved from the destinations map into the preallocated retained
         *         vector slot; the retained window metadata is written IN PLACE.
         *         The transfer cannot lose the last owning ref (the slot takes
         *         ownership before the dest entry is erased). */
        std::shared_ptr<DestinationBacking> retainDestination(std::uint64_t reservation)
        {
            auto dit = destinations_.find(reservation);
            if (dit == destinations_.end())
                throw std::logic_error("Destination reservation not owned (retain refused)");
            const auto len = dit->second->outputLength();
            if (len == 0)
                throw std::logic_error("Empty retained payload (nothing to retain)");
            auto backing = dit->second;
            backing->retained_offset_ = backing->outputOffset();
            backing->retained_length_ = len;
            backing->retained_wheel_generation_ = backing->wheel_generation_;
            backing->retained_state_ = DestinationBacking::RetainedState::Pending;
            // Install into the preallocated slot BEFORE the ownership move (no
            // allocation; a slot always exists for a banked destination).
            retained_.push_back(backing);
            destinations_.erase(dit);
            removeClaim(reservation);
            return backing;
        }

        /** @brief Non-allocating unlink (pointer identity; no cross-links). */
        void retainedUnlink(std::uint64_t /*reservation*/, DestinationBacking *ptr) noexcept
        {
            for (auto it = retained_.begin(); it != retained_.end(); ++it)
                if (it->get() == ptr)
                {
                    retained_.erase(it); // slot reuse; the entry's own ref dropped
                    return;
                }
        }

    public:
        /** @brief The writer's delivery-lease API (generation-bound, exactly-once). */
        RetainedPayloadLease acquireRetainedPayload(std::uint64_t reservation,
                                                    std::uint64_t wheel_generation)
        {
            auto backing = retainedFind(reservation);
            if (!backing)
                throw std::logic_error("No retained payload for this reservation");
            if (backing->retained_wheel_generation_ != wheel_generation)
                throw std::logic_error(
                    "Retained payload wheel generation mismatch (stale lease refused)");
            if (backing->retained_state_ != DestinationBacking::RetainedState::Pending)
                throw std::logic_error(
                    "Retained payload not acquirable (leased or released — the delivery "
                    "is exactly-once; a released delivery is final)");
            backing->retained_state_ = DestinationBacking::RetainedState::Leased;
            return RetainedPayloadLease(std::move(backing), control_reserve_);
        }

        [[nodiscard]] bool hasRetainedPayload(std::uint64_t reservation,
                                               std::uint64_t wheel_generation) const
        {
            auto backing = retainedFind(reservation);
            return backing && backing->retained_wheel_generation_ == wheel_generation;
        }

    private:
        [[nodiscard]] bool retainedPending(std::uint64_t reservation) const
        {
            auto backing = retainedFind(reservation);
            return backing &&
                   backing->retained_state_ != DestinationBacking::RetainedState::None &&
                   backing->retained_state_ != DestinationBacking::RetainedState::Released;
        }

        ResponseFailureKind responseFailureKind(std::uint64_t reservation) const
        {
            auto it = response_failures_.find(reservation);
            if (it == response_failures_.end())
                throw std::logic_error("No response failure record for this reservation");
            return it->second.kind;
        }
    public:
        [[nodiscard]] std::string responseFailureReason(std::uint64_t reservation) const
        {
            auto it = response_failures_.find(reservation);
            if (it == response_failures_.end())
                throw std::logic_error("No response failure record for this reservation");
            return it->second.reason;
        }

        /** @brief Retained failure/cancellation payload (the writer's delivery
         *         lease): wheel generation bound + the ADMITTED-extent-bounded
         *         byte copy (never an unadmitted duplicate — the snapshot is
         *         rejected if it exceeds the destination's admitted output
         *         extent). */
    private:
        /// Host allocation alignment quantum (64): the real per-allocation
        /// boundary the system allocator uses.
        static constexpr std::uint64_t kControlAlignment = 64;

        /// TYPED per-slot control admission for the destination backing OBJECT +
        /// its shared control block + the destinations-map node (each a real
        /// separate allocation, no duplication), rounded to the allocator
        /// quantum. Claimed at bank; returned by the object's own deleter AFTER
        /// the object storage free.
        static constexpr std::uint64_t kDestinationControlBytes =
            (sizeof(DestinationBacking) +
                 sizeof(PhysicalMemorySuballocationLease) + // the claim-holder state (deleter)
                 sizeof(std::pair<const std::uint64_t,
                                  std::shared_ptr<DestinationBacking>>) +
                 4 * sizeof(void *) + kControlAlignment - 1) /
                kControlAlignment * kControlAlignment;

        /** @brief The per-destination WHEEL generation (bank counter). */
        std::uint64_t wheelGeneration(std::uint64_t reservation) const
        {
            auto it = destinations_.find(reservation);
            return it != destinations_.end() ? it->second->wheel_generation_ : 0;
        }

        /** @brief RETAIN-BY-TRANSFER: the destination backing (ALREADY admitted
         *         + allocated) is MOVED into the retained entry — there is NO
         *         second copy and NO second PMA suballocation (a retained copy
         *         is byte-identical to the live backing by construction, so the
         *         admitted extent is never exceeded and no allocator BOM needs
         *         estimating). The backing's PMA credit + memory are returned by
         *         the backing's OWN destruction, AFTER the last reader (the
         *         writer's lease) — never a separate copy freed before its
         *         credit. A row with NO committed output retains NOTHING. */


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
            it->second->deseal();    // the seal dies with the cancellation release too
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

        /** @brief PMA serving-pool reservation: the single capacity ledger every
         *         destination backing claims from. OWNED BY VALUE (the engine
         *         pins the admitted identity for its lifetime; no external
         *         rebinding). Declared BEFORE the rows/maps so it is destroyed
         *         AFTER every destination backing (the free-boundary deleter
         *         may safely observe materializedBytes during destruction). */
        PhysicalMemoryOwnerReservation pool_capacity_;
        /// The aggregate control reserve: declared AFTER the reservation (later
        /// declared -> destroyed FIRST) so its deallocate returns the aggregate
        /// credit BEFORE the reservation's destructor runs — the reservation
        /// cannot terminate with aggregate bytes outstanding (#171).
        std::shared_ptr<ControlReserve> control_reserve_;
        /** @brief TYPED destination-backing allocator (production: BackingAllocator::system). */
        BackingAllocator allocator_;
        /// #149 the aggregate control reserve's authority-record identity.
        PhysicalMemoryAggregateHandle::AggregateStateRef aggregate_state_;
        std::uint64_t aggregate_serial_ = 0u;
        std::size_t aggregate_bytes_ = 0u;

        /** @brief TYPED last-step ABI deferral: the state classifies the
         *         per-row outcome (retryable / state-violated / cancelled /
         *         missing-publication) — never a bare string. */
        std::map<std::uint32_t, AbiDeferral> last_abi_deferrals_;
        bool admission_started_ = false;     ///< Mode immutable after admission.
        std::uint64_t window_epoch_ = 0;     ///< Monotonic per-step window epoch.
        RowDecodeFn decode_;                 ///< Pure per-row decode (non-ABI path).
        IOrchestrationRunner *abi_runner_ = nullptr; ///< ABI decode runner (optional).
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
        struct ResponseFailure
        {
            std::uint64_t generation = 0;
            ResponseFailureKind kind = ResponseFailureKind::Execution;
            std::string reason;
            bool reason_consumed = false; ///< The writer consumed the terminal reason (history stays).
        };
        std::map<std::uint64_t, ResponseFailure> response_failures_;
        std::vector<std::shared_ptr<DestinationBacking>> retained_; ///< Preallocated-capacity retained slots.

        /// PMA-representative retained ledger: every retained delivery copy is
        /// CHARGED against the admitted pool reservation (claimPoolBytes) — a
        /// retained copy is a REAL suballocation in the engine's one PMA ledger,
        /// never an unaccounted duplicate. The charge lives for the lease; the
        /// re-bank/lease release returns it.
        std::map<std::uint64_t, std::uint64_t> completed_ledger_; ///< Completion GENERATION per RESPONSE.
        std::vector<std::uint64_t> claim_slots_; ///< Claimed slots (reserved, non-allocating commit).
        std::size_t max_destinations_ = kDefaultPoolWidth; ///< Admitted plan-derived pool width.

        /// The admitted pool capacity also bounds the retained delivery copies
        /// (the engine's PMA-representative ledger: retained copies are NOT an
        /// unaccounted duplicate — they charge this admitted budget and are
        /// refused when it is exhausted).
        std::map<std::uint64_t, std::uint64_t> destination_generation_; ///< Per-ID generation (bounded reuse).
        bool failed_ = false;
        std::string failed_reason_;
        std::function<void(std::uint64_t, const std::byte *, std::size_t)> consumer_;
    };
} // namespace llaminar2
