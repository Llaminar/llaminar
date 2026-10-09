/**
 * @file DiskPrefixStorageBackend.h
 * @brief Durable, loaded-model-artifact-addressed prefix-cache archive.
 *
 * A disk cache has one append-only `<model-artifact-identity>.kvcache` metadata
 * journal and an adjacent `.blocks` directory of immutable payload inodes.
 * The identity covers the stable filesystem identity of every GGUF shard in
 * the already-loaded model context, without rereading model payload bytes.
 * Each journal record authenticates its small metadata and committed extents; payload
 * bytes are opaque and are never checksummed or scanned for equality.
 * Complete records survive process restart; an interrupted tail is ignored
 * and removed before the next append. The archive owns capacity eviction and
 * metadata compaction. Durable eviction unlinks the retired payload immediately,
 * independently of compaction progress. Selected readers retain only the exact
 * payload inodes they selected, never a whole obsolete archive generation.
 * The ordered writer authenticates backing at each RAM demotion. Local cache
 * indexes cannot certify residency after another participant or process writes.
 */

#pragma once

#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "execution/prefix_cache/PrefixArchivePersistence.h"
#include "execution/prefix_cache/PrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixCacheTelemetry.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/prefix_cache/PrefixArchiveMaintenanceWriter.h"

#include <array>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class RamPrefixStorageBackend;
    class PrefixArchiveReadInode;

    /**
     * @brief Persistent prefix blocks owned by a journal and immutable payload files.
     *
     * Instances opened through openShared() are shared by every local
     * orchestrator targeting the same archive. A separate advisory lock file
     * coordinates independent MPI processes that share the directory.
     */
    class DiskPrefixStorageBackend :
        public IPrefixStorageBackend,
        public std::enable_shared_from_this<DiskPrefixStorageBackend>
    {
    public:
        /** @brief Explicit lifecycle of the archive's sole maintenance worker. */
        enum class CompactionState
        {
            Idle,       ///< No requested archive rewrite.
            Scheduled,  ///< The worker has an admitted request.
            Copying,    ///< Metadata snapshot is copied without cache locks.
            Publishing, ///< The complete replacement is atomically renamed.
            Failed,     ///< Native I/O failed; further archive use is rejected.
            Stopping,   ///< Teardown cancels bounded copying and joins the worker.
        };

        /** @brief Actual data-synchronous native writes, never an admission ledger. */
        using WritebackEvidence = PrefixArchiveMaintenanceWriter::Evidence;

        /** @brief Observable maintenance evidence, not an admission ledger. */
        struct CompactionStatus
        {
            CompactionState state = CompactionState::Idle;
            uint64_t publications = 0;
            std::thread::id last_executor;
            WritebackEvidence last_writeback; ///< Sealed only after real publication.
        };

        /**
         * @brief Move-only snapshot of one physically stable archive record.
         *
         * A ticket snapshots committed payload-section offsets without any payload
         * read and shares an open descriptor for that exact inode.
         * Disk capacity eviction may remove the logical record while RAM makes
         * room; hydration can still stream the snapshotted committed bytes
         * into their final owner without retaining a second full block.
         */
        class HydrationTicket final
        {
        public:
            /** @brief Construct an empty ticket with no inode ownership. */
            HydrationTicket() = default;
            /** @brief Retire the open inode and its shared archive lifetime. */
            ~HydrationTicket();
            HydrationTicket(const HydrationTicket &) = delete;
            HydrationTicket &operator=(const HydrationTicket &) = delete;
            /** @brief Transfer the selected record and its inode exactly once. */
            HydrationTicket(HydrationTicket &&other) noexcept;
            /** @brief Release the old record before accepting a moved ticket. */
            HydrationTicket &operator=(HydrationTicket &&other) noexcept;

            /** @return Whether this ticket names an unconsumed captured or selected record. */
            [[nodiscard]] bool valid() const noexcept;
            /** @return Complete archive payload bytes named by this snapshot. */
            [[nodiscard]] size_t totalBytes() const noexcept;
            /** @return Durable handle metadata captured without payload reads. */
            [[nodiscard]] const PrefixBlockHandle &diskHandle() const noexcept;
            /** @return Payload bytes actually read into final RAM sections. */
            size_t hydratedPayloadBytes() const noexcept { return hydrated_payload_bytes_; }

        private:
            friend class DiskPrefixStorageBackend;

            /** @brief Immutable section extent within the retained inode. */
            struct SectionSnapshot
            {
                uint64_t offset = 0;
                uint64_t bytes = 0;
            };

            /** @brief Bind captured metadata to one owned native descriptor. */
            HydrationTicket(
                std::shared_ptr<DiskPrefixStorageBackend> backend,
                PrefixBlockHandle handle,
                std::shared_ptr<PrefixArchiveReadInode> inode,
                std::array<SectionSnapshot, 6> sections);
            /** @brief Close the retained inode before releasing the backend. */
            void release() noexcept;

            std::shared_ptr<DiskPrefixStorageBackend> backend_;
            PrefixBlockHandle handle_;
            std::shared_ptr<PrefixArchiveReadInode> inode_;
            std::array<SectionSnapshot, 6> sections_{};
            enum class Phase { Captured, Selected, Consumed };
            Phase phase_ = Phase::Captured;
            PrefixPayloadReadSet read_set_ = PrefixPayloadReadSet::WholeArchive;
            size_t hydrated_payload_bytes_ = 0u;
        };

        /**
         * @brief Open or create a model-specific archive.
         *
         * Direct construction is retained for focused tests. Production code
         * should use openShared() so LocalTP children share one in-process
         * index, one synchronization domain, and one accounted I/O scratch.
         */
        DiskPrefixStorageBackend(
            std::filesystem::path archive_path,
            size_t budget_bytes,
            std::string model_artifact_identity);

        /** @brief Cancel background copying and join before scratch retirement. */
        ~DiskPrefixStorageBackend() override;

        /**
         * @brief Return a process-shared backend for an archive pathname.
         * @param archive_path Durable model-specific archive path.
         * @param budget_bytes Configured active payload capacity.
         * @param model_artifact_identity Stable identity encoded by the file.
         * @param memory_authority Sole rank-local CPU allocation ledger.
         * @param error Optional deterministic construction diagnostic.
         */
        static std::shared_ptr<DiskPrefixStorageBackend> openShared(
            const std::filesystem::path &archive_path,
            size_t budget_bytes,
            const std::string &model_artifact_identity,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            std::string *error = nullptr);

        /** @return Whether one nonempty payload fits the configured active budget. */
        bool canStore(size_t bytes) const override;
        /** @brief Produce metadata for a disk block; allocate no payload buffer. */
        PrefixBlockHandle allocate(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout) override;
        /** @brief Commit a tombstone, unlink its payload and schedule metadata compaction. */
        bool release(const PrefixBlockHandle &handle) override;

        /**
         * @brief Publish complete backing and enforce the active-byte budget.
         * @param handle Immutable, ready source with its original physical leases.
         * @param disk_handle Receives archive-owned durable record metadata.
         * @param evicted_keys Receives keys retired by this exact publication.
         * @param error Receives a native failure diagnostic.
         * @param disposition Distinguishes appended payload from identical backing reuse.
         * @return Whether the complete durable publication succeeded.
         * Matching payload identity, layout and flags permit a payload-free
         * durable touch; replacement data still follows the ordinary append path.
         *
         * The source may be event-backed pinned RAM. The method waits only at
         * this true host serialization boundary. Any least-recent archive
         * records evicted to make capacity are returned to the caller so its
         * local lookup index can be updated immediately.
         */
        bool writeBlock(
            const PrefixBlockHandle &handle,
            PrefixBlockHandle *disk_handle,
            std::vector<PrefixCacheKey> *evicted_keys = nullptr,
            std::string *error = nullptr,
            PrefixArchiveWriteDisposition *disposition = nullptr);

        /**
         * @brief Persist an immutable resident through the archive-owned writer.
         * @param handle Existing payload and physical leases, borrowed unchanged.
         * @return Mutation identity and read-only completion receipt.
         * Native I/O and payload readiness waits never execute on this caller.
         */
        PrefixArchivePersistenceTicket scheduleWrite(PrefixBlockHandle handle);

        /** @brief Order a durable tombstone after earlier same-key publications. */
        PrefixArchivePersistenceTicket scheduleRetirement(const PrefixCacheKey &key);

        /**
         * @brief Join admitted persistence for administration/tests, not inference.
         * @param error Receives the first background publication failure.
         * @return True only when all admitted writes/tombstones completed.
         */
        bool waitForPersistence(std::string *error = nullptr);

        /** @brief Unit oracle; production hydration uses admitted RAM directly. */
        bool readBlock(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            PrefixBlockHandle *ram_handle,
            std::string *error = nullptr);

        /**
         * @brief Capture a complete restore source using metadata only.
         *
         * Retain the committed inode and select its complete section geometry
         * before RAM victims retire. Actual payload bytes are read once, directly
         * into their final owners, by hydrateSelected().
         * @param key Exact durable record identity.
         * @param layout Required payload geometry.
         * @param error Optional native I/O or metadata diagnostic; absent keys leave it empty.
         * @return One selected immutable source, or no available record.
         */
        [[nodiscard]] std::optional<HydrationTicket> beginHydration(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            std::string *error = nullptr);

        /**
         * @brief Retain committed offsets without reading any payload section.
         * @param key Exact immutable record identity selected by lookup.
         * @param layout Expected metadata shape.
         * @param error Receives native errors; an ordinary absent key leaves it empty.
         * @return Metadata snapshot retaining its inode across eviction/compaction.
         */
        [[nodiscard]] std::optional<HydrationTicket> captureLookupSource(
            const PrefixCacheKey &key, const PrefixPayloadLayout &layout,
            std::string *error = nullptr);

        /**
         * @brief Select and bound the coordinated read set without reading payload bytes.
         *
         * Publish the selected version's recency before RAM admission demotes
         * victims. Retained older versions never touch a replacement's recency.
         * @param ticket Unconsumed metadata snapshot from this backend.
         * @param read_set Actual sections selected by the restore endpoint.
         * @param error Receives native I/O, metadata or lifecycle diagnostics.
         * @return Whether every selected extent fits the retained inode.
         */
        bool selectHydrationSections(HydrationTicket &ticket, PrefixPayloadReadSet read_set,
            std::string *error = nullptr);

        /**
         * @brief Consume a selected ticket directly into admitted RAM.
         *
         * The immutable record snapshot remains readable even when making RAM
         * capacity caused its logical disk entry to be evicted.  The method
         * reads each selected section directly into final storage once and
         * permits exactly one attempt per ticket.
         * @param ticket Selected, unconsumed source belonging to this backend.
         * @param ram_backend Authority-backed destination for the selected sections.
         * @param ram_handle Receives their owning handle after successful hydration.
         * @param error Receives corruption, allocation or lifecycle diagnostics.
         * @return Whether every selected byte reached its final owner with matching extents.
         */
        bool hydrateSelected(
            HydrationTicket &ticket,
            RamPrefixStorageBackend &ram_backend,
            PrefixBlockHandle *ram_handle,
            std::string *error = nullptr);

        /**
         * @brief Hydrate directly into an admitted RAM prefix tier.
         *
         * Runtime-state bytes are attached through the RAM backend so both
         * logical capacity and the canonical physical ledger remain exact.
         */
        bool readBlockIntoRamBackend(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            RamPrefixStorageBackend &ram_backend,
            PrefixBlockHandle *ram_handle,
            std::string *error = nullptr);

        /**
         * @brief Discover restart-persistent records for one runtime layout.
         */
        std::vector<PrefixBlockHandle> compatibleEntries(
            uint64_t fingerprint,
            const PrefixPayloadLayout &layout,
            std::string *error = nullptr);

        /** @return Whether startup and every completed native operation succeeded. */
        bool ready() const;
        /** @return Thread-safe startup or asynchronous maintenance diagnostic. */
        std::string initializationError() const;
        /** @return Immutable archive pathname; the published inode may change. */
        const std::filesystem::path &archivePath() const { return archive_path_; }
        /** @return Configured active-payload capacity, not an I/O traffic extent. */
        size_t budgetBytes() const { return budget_bytes_; }
        /** @return Passive metadata publisher, retaining no payload or storage owner. */
        std::shared_ptr<const PrefixCacheTierTelemetry> telemetry() const { return telemetry_; }
        /** @return Indexed committed payload bytes from the archive authority. */
        size_t usedBytes() const;

        /**
         * @brief Enqueue compaction without copying payloads on the caller.
         * @param error Receives a fatal archive readiness diagnostic.
         * @return Whether maintenance is scheduled or already in flight.
         * Automatic capacity enforcement uses the same worker. This explicit
         * administrative entrypoint also permits small-archive lifecycle tests.
         */
        bool requestCompaction(std::string *error = nullptr);

        /** @return Typed progress evidence from the one archive authority. */
        [[nodiscard]] CompactionStatus compactionStatus() const;

        /**
         * @brief Join the requested maintenance frontier (administration/tests).
         * @param error Receives a stored native maintenance failure.
         * @return Whether every requested rewrite completed successfully.
         * Never call this from inference: foreground reads/appends do not join.
         */
        bool waitForCompaction(std::string *error = nullptr);

    private:
        /** @brief Claim the canonical scratch before starting native maintenance. */
        DiskPrefixStorageBackend(
            std::filesystem::path archive_path,
            size_t budget_bytes,
            std::string model_artifact_identity,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority);

        static constexpr size_t kSectionCount = 6;

        /** @brief Serialized section geometry in the current published inode. */
        struct SectionIndex
        {
            uint64_t offset = 0;
            uint64_t bytes = 0;
        };

        /** @brief Journal put extent, immutable payload inode identity and logical recency. */
        struct RecordIndex
        {
            PrefixBlockHandle handle;
            PrefixPayloadIdentity storage_identity;
            std::weak_ptr<PrefixArchiveReadInode> lookup_inode;
            uint64_t record_offset = 0;
            uint64_t record_bytes = 0;
            uint64_t sequence = 0;
            std::array<SectionIndex, kSectionCount> sections{};
        };

        /** @brief Validate model identity and recover the crash-tolerant archive. */
        bool initialize(std::string *error);
        /**
         * @brief Retain a failed native mutation and reject later archive work.
         * @param message Precise failure at the owned native mutation boundary.
         * @param error Optional output receiving the retained diagnostic.
         * @return Always false after making the failure terminal.
         */
        bool failMutationLocked(std::string message, std::string *error);
        /**
         * @param identity Opaque generation assigned by the archive, never a digest.
         * @return Exact archive-owned filename without accepting external path text.
         */
        std::filesystem::path payloadPath(const PrefixPayloadIdentity &identity) const;
        /**
         * @brief Persist payload directory entries before journal references or after unlink.
         * @param error Optional native directory publication diagnostic.
         * @return Whether the exact payload directory was durably synchronized.
         */
        bool syncPayloadDirectory(std::string *error) const;
        /**
         * @brief Unlink only payloads retired by an already-durable journal frontier.
         * @param identities Storage generations removed by this exact transaction.
         * @param error Receives native retirement or directory durability failures.
         * @return Whether every retirement completed; retained readers own open inodes.
         */
        bool retirePayloadsLocked(std::span<const PrefixPayloadIdentity> identities, std::string *error);
        /**
         * @brief Validate live extents and remove crash-orphaned payloads under the writer lock.
         * @param error Optional metadata, extent or native retirement diagnostic.
         * @return Whether every live extent is valid and unreferenced files retired.
         */
        bool recoverPayloadFilesLocked(std::string *error);
        /**
         * @brief Retain one immutable native descriptor per live record's lookup cohort.
         * @param record Authoritative live record receiving a weak cohort reference.
         * @param error Optional native-open or committed-extent diagnostic.
         * @return Exact retained payload inode, or null after a storage failure.
         */
        std::shared_ptr<PrefixArchiveReadInode> openPayloadLocked(RecordIndex &record, std::string *error);
        /** @brief Scan committed metadata after appends or atomic inode replacement. */
        bool refreshIndexLocked(int archive_fd, std::string *error);
        /** @brief Reconstruct the index; return the exact last complete footer. */
        bool scanRecordsLocked(
            int archive_fd,
            uint64_t start_offset,
            uint64_t file_bytes,
            uint64_t *valid_end,
            std::string *error);
        /** @brief Append one tombstone and remove only its logical payload owner. */
        bool appendDeleteLocked(
            int archive_fd,
            const PrefixCacheKey &key,
            std::string *error);
        /**
         * @brief Append a payload-free access record and refresh persistent LRU.
         *
         * The active put record remains the payload authority. A touch updates
         * only its logical sequence, allowing restart recovery and cooperating
         * processes to agree on which bottom-tier block is oldest without
         * rewriting the potentially large payload.
         */
        bool appendTouchLocked(
            int archive_fd,
            const PrefixCacheKey &key,
            std::string *error);
        /** @brief Schedule stale-history reclamation, never copy under the lock. */
        bool compactIfNeededLocked(int archive_fd, std::string *error);
        /** @brief Wake the sole worker; mutex_ must already be held. */
        bool scheduleCompactionLocked(std::string *error);
        /** @brief Run requested native rewrites and retain asynchronous failures. */
        void maintainArchive(std::stop_token stop) noexcept;
        /** @brief Background outcome distinguishes peer ownership from failure. */
        enum class CompactionOutcome { Published, PeerOwned, Cancelled, Failed };
        /**
         * @brief Compact journal metadata without reading any payload inode.
         * @param stop Teardown cancellation, observed between bounded I/O chunks.
         * @param error Deterministic native I/O or identity failure diagnostic.
         * @return Exact publication/cancellation/failure outcome.
         * Snapshot copying runs outside cache locks. A finite metadata-only
         * tail and durable rename join the writer frontier without chasing new
         * payload appends; payload lifetime is independent of this maintenance.
         */
        CompactionOutcome rewriteArchive(std::stop_token stop, std::string *error);

        /** @brief Replace one key and update active-byte capacity exactly once. */
        void applyPutRecord(RecordIndex record);
        /** @brief Remove the indexed logical owner, not retained reader bytes. */
        void applyDeleteRecord(const PrefixCacheKey &key);
        /** @brief Apply committed recency without rebinding a put-record extent. */
        bool applyTouchRecord(
            const PrefixCacheKey &key,
            uint64_t sequence);

        std::filesystem::path archive_path_;
        std::filesystem::path lock_path_;
        std::filesystem::path payload_directory_;
        size_t budget_bytes_ = 0;
        std::shared_ptr<PrefixCacheTierTelemetry> telemetry_ =
            std::make_shared<PrefixCacheTierTelemetry>(budget_bytes_);
        std::string model_artifact_identity_;

        mutable std::mutex mutex_;
        bool ready_ = false;
        std::string initialization_error_;
        uint64_t scan_offset_ = 0;
        uint64_t next_sequence_ = 1;
        uint64_t active_bytes_ = 0;
        uint64_t active_record_bytes_ = 0; ///< Live journal framing/metadata only, never payload capacity.
        uint64_t archive_device_ = 0;
        uint64_t archive_inode_ = 0;
        std::unordered_map<PrefixCacheKey, RecordIndex, PrefixCacheKeyHasher> records_;
        std::condition_variable_any compaction_changed_;
        CompactionStatus compaction_status_;

        /*
         * Declaration order is intentional: reverse destruction frees the
         * scratch allocation before its ledger lease, and the lease before
         * the authority it retains. The sole maintenance worker owns this copy
         * buffer. Foreground hydration reads directly into final owners and
         * needs no verification or staging allocation.
         */
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority_;
        PhysicalMemoryAllocationLease archive_scratch_memory_lease_;
        std::vector<uint8_t> archive_scratch_;
        // The writer is joined before compaction and the shared native index.
        std::unique_ptr<PrefixArchivePersistence> persistence_;
        // Last member and explicitly joined: no worker can outlive its buffer.
        std::jthread compaction_worker_;
    };
} // namespace llaminar2
