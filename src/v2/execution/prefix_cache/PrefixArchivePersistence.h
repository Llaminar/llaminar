/**
 * @file PrefixArchivePersistence.h
 * @brief Ordered background persistence of immutable, physically leased prefixes.
 *
 * One service belongs to each process-shared disk archive. It borrows existing
 * payload owners rather than allocating another block or maintaining a memory
 * ledger. Request threads enqueue mutations and consume immutable receipts;
 * only the writer performs native writes and durability I/O; metadata identity
 * determines reuse without inspecting payload contents.
 * Revalidating an unchanged durable record publishes a metadata-only receipt;
 * a stale participant observation never forces a duplicate payload append.
 */
#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace llaminar2
{
    class DiskPrefixStorageBackend;

    /** @brief Distinguish new payload I/O from authenticated existing backing. */
    enum class PrefixArchiveWriteDisposition { Stored, Reused };

    /** @brief Durable put publication, including archive-owned capacity victims. */
    struct PrefixArchiveWritePublication
    {
        PrefixBlockHandle disk_handle;
        std::vector<PrefixCacheKey> evicted_keys;
        std::thread::id executor;
        PrefixArchiveWriteDisposition disposition = PrefixArchiveWriteDisposition::Stored;
    };

    /** @brief A tombstone is durable; earlier queued puts cannot resurrect it. */
    struct PrefixArchiveRetirementPublication { PrefixCacheKey key; };

    /** @brief A queued put was superseded before any native I/O began. */
    struct PrefixArchiveCancelledPublication {};

    /** @brief Fatal durable-tier failure; never convert this into a cache miss. */
    struct PrefixArchivePersistenceFailure { std::string diagnostic; };

    /** @brief Total terminal mutation outcome, with no partial publication state. */
    using PrefixArchivePublication = std::variant<
        PrefixArchiveWritePublication,
        PrefixArchiveRetirementPublication,
        PrefixArchiveCancelledPublication,
        PrefixArchivePersistenceFailure>;

    /** @brief Observable ownership phase, never an allocation/admission ledger. */
    enum class PrefixArchiveMutationPhase { Empty, Queued, Executing, Published };

    /**
     * @brief Read-only completion handle for one archive-authoritative mutation.
     *
     * Absence of a publication means pending work, not released physical bytes.
     * Mutation identity permits a cache to reject an old receipt after same-key
     * replacement without inventing a second archive sequence or allocation ledger.
     */
    class PrefixArchivePersistenceTicket final
    {
    public:
        /** @brief Construct an empty handle, suitable for non-disk cache tiers. */
        PrefixArchivePersistenceTicket() = default;
        /** @return Whether this handle names a real queued mutation. */
        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        /** @return Immutable service-local identity, or zero for an empty handle. */
        [[nodiscard]] uint64_t identity() const noexcept;
        /** @return Whether queue, native writer, or immutable receipt owns work. */
        [[nodiscard]] PrefixArchiveMutationPhase phase() const;
        /** @return Immutable terminal result, or null while the writer owns work. */
        [[nodiscard]] std::shared_ptr<const PrefixArchivePublication> publication() const;
        /**
         * @brief Complete this exact mutation dependency, not the writer queue.
         * @return Immutable outcome after the writer releases its source alias.
         * @throws std::logic_error If this handle names no mutation.
         *
         * Early preparation uses publication() to overlap archive work with
         * inference. Required RAM/SSD publication may join this receipt when
         * its physical capacity depends on the original payload retiring.
         * The caller performs no payload readiness or native I/O.
         */
        [[nodiscard]] std::shared_ptr<const PrefixArchivePublication> waitForPublication() const;

    private:
        friend class PrefixArchivePersistence;
        /** @brief Receipt synchronization is independent of the native I/O lock. */
        struct State
        {
            /** @brief Seal identity before queue publication. */
            explicit State(uint64_t value) : identity(value) {}
            const uint64_t identity;
            mutable std::mutex mutex;
            mutable std::condition_variable completed;
            PrefixArchiveMutationPhase phase = PrefixArchiveMutationPhase::Queued;
            std::shared_ptr<const PrefixArchivePublication> publication;
        };
        /** @brief Only the archive authority can construct a live ticket. */
        explicit PrefixArchivePersistenceTicket(std::shared_ptr<State> state)
            : state_(std::move(state)) {}
        std::shared_ptr<State> state_;
    };

    /**
     * @brief Sole ordered payload writer for one process-shared archive.
     *
     * The backend outlives this service and joins it before native index/scratch
     * retirement. Queued puts retain the original PMA-backed payload leases.
     * A retire cancels unstarted same-key puts and is ordered after an already
     * executing put, so a later replacement never receives its stale receipt.
     */
    class PrefixArchivePersistence final
    {
    public:
        /** @brief Start the archive-owned writer during backend construction. */
        explicit PrefixArchivePersistence(DiskPrefixStorageBackend &backend);
        /** @brief Seal admission and finish ordered durable mutations. */
        ~PrefixArchivePersistence();
        PrefixArchivePersistence(const PrefixArchivePersistence &) = delete;
        PrefixArchivePersistence &operator=(const PrefixArchivePersistence &) = delete;
        /** @brief Enqueue one immutable owner; perform no payload scan or file I/O. */
        PrefixArchivePersistenceTicket write(PrefixBlockHandle handle);
        /** @brief Order retirement after existing work, superseding queued puts. */
        PrefixArchivePersistenceTicket retire(const PrefixCacheKey &key);
        /**
         * @brief Join the writer frontier for administration or focused tests.
         * @param error Receives the first native persistence failure.
         * @return Whether all admitted mutations published without failure.
         * Never call this from ordinary lookup, harvest or inference admission.
         */
        bool waitUntilIdle(std::string *error = nullptr);

    private:
        /** @brief Lifecycle of the writer, not a physical-memory ledger. */
        enum class Lifecycle { Accepting, Sealing, Failed };
        /** @brief A queued supersession does not yet release its physical owner. */
        enum class PutDisposition { Persist, Superseded };
        /** @brief Retain the original payload until the writer retires this put. */
        struct Put
        {
            PrefixBlockHandle handle;
            PutDisposition disposition = PutDisposition::Persist;
        };
        /** @brief Metadata-only mutation; no payload lease is created. */
        struct Retire { PrefixCacheKey key; };
        /** @brief FIFO ownership of one mutation and its eventual receipt. */
        struct Mutation
        {
            std::variant<Put, Retire> operation;
            std::shared_ptr<PrefixArchivePersistenceTicket::State> receipt;
        };
        /** @brief Admit identity and queue ownership under the small queue lock. */
        PrefixArchivePersistenceTicket enqueueLocked(std::variant<Put, Retire> operation);
        /** @brief Publish once; readers never lock the archive or its writer. */
        static void publish(
            const std::shared_ptr<PrefixArchivePersistenceTicket::State> &receipt,
            PrefixArchivePublication result);
        /** @brief Perform FIFO native I/O outside the queue/receipt locks. */
        void run() noexcept;

        DiskPrefixStorageBackend &backend_;
        std::mutex mutex_;
        std::condition_variable changed_;
        std::deque<Mutation> pending_;
        std::optional<uint64_t> executing_;
        uint64_t next_identity_ = 1;
        Lifecycle lifecycle_ = Lifecycle::Accepting;
        std::string failure_;
        // Declared last and explicitly joined before any service state retires.
        std::thread writer_;
    };
}
