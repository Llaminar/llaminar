/**
 * @file DiskPrefixStorageBackend.h
 * @brief Durable, loaded-model-artifact-addressed prefix-cache archive.
 *
 * A disk cache is one append-only `<model-artifact-identity>.kvcache` file.
 * The identity covers the stable filesystem identity of every GGUF shard in
 * the already-loaded model context, without rereading model payload bytes.
 * Each put or delete operation is an independently checksummed record.
 * Complete records survive process restart; an interrupted tail is ignored
 * and removed before the next append. The archive owns capacity eviction and
 * compaction so the configured disk budget remains a real bounded tier.
 */

#pragma once

#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "execution/prefix_cache/PrefixStorageBackend.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <array>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class RamPrefixStorageBackend;

    /**
     * @brief Persistent prefix blocks stored in one model-specific blob file.
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
        /**
         * @brief Move-only proof of one verified, physically stable archive record.
         *
         * A ticket snapshots immutable put-record offsets after checksum
         * verification and defers in-process compaction until destruction.
         * Disk capacity eviction may remove the logical record while RAM makes
         * room; hydration can still stream the snapshotted committed bytes
         * into their final owner without retaining a second full block.
         */
        class HydrationTicket final
        {
        public:
            HydrationTicket() = default;
            ~HydrationTicket();
            HydrationTicket(const HydrationTicket &) = delete;
            HydrationTicket &operator=(const HydrationTicket &) = delete;
            HydrationTicket(HydrationTicket &&other) noexcept;
            HydrationTicket &operator=(HydrationTicket &&other) noexcept;

            /** @return Whether this ticket names one unconsumed verified record. */
            [[nodiscard]] bool valid() const noexcept;
            /** @return Exact RAM capacity required by the verified record. */
            [[nodiscard]] size_t totalBytes() const noexcept;
            /** @return Durable handle metadata captured during verification. */
            [[nodiscard]] const PrefixBlockHandle &diskHandle() const noexcept;

        private:
            friend class DiskPrefixStorageBackend;

            struct SectionSnapshot
            {
                uint64_t offset = 0;
                uint64_t bytes = 0;
                uint64_t checksum = 0;
            };

            HydrationTicket(
                std::shared_ptr<DiskPrefixStorageBackend> backend,
                PrefixBlockHandle handle,
                uint64_t archive_device,
                uint64_t archive_inode,
                std::array<SectionSnapshot, 6> sections);
            void release() noexcept;

            std::shared_ptr<DiskPrefixStorageBackend> backend_;
            PrefixBlockHandle handle_;
            uint64_t archive_device_ = 0;
            uint64_t archive_inode_ = 0;
            std::array<SectionSnapshot, 6> sections_{};
            bool consumed_ = false;
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

        bool canStore(size_t bytes) const override;
        PrefixBlockHandle allocate(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout) override;
        bool release(const PrefixBlockHandle &handle) override;

        /**
         * @brief Append one complete block and enforce the active-byte budget.
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
            std::string *error = nullptr);

        bool readBlock(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            PrefixBlockHandle *ram_handle,
            std::string *error = nullptr);

        /**
         * @brief Verify every serialized section through the bounded scratch.
         *
         * This is the first half of disk-to-RAM hydration.  It proves the
         * durable record before the cache retires a RAM victim, without
         * materializing a second full prefix block.  The later direct read
         * validates each section again while filling its final RAM owner.
         *
         * @param key Exact durable record to verify.
         * @param layout Runtime payload geometry required by the caller.
         * @param error Optional deterministic corruption/layout diagnostic.
         * @return true only when all section sizes and checksums match.
         */
        [[nodiscard]] std::optional<HydrationTicket> beginVerifiedHydration(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            std::string *error = nullptr);

        /**
         * @brief Consume a verified ticket directly into admitted RAM.
         *
         * The immutable record snapshot remains readable even when making RAM
         * capacity caused its logical disk entry to be evicted.  The method
         * performs a second checksum while filling final storage and permits
         * exactly one attempt per ticket.
         */
        bool hydrateVerified(
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

        bool ready() const;
        const std::string &initializationError() const;
        const std::filesystem::path &archivePath() const { return archive_path_; }
        size_t budgetBytes() const { return budget_bytes_; }
        size_t usedBytes() const;

    private:
        DiskPrefixStorageBackend(
            std::filesystem::path archive_path,
            size_t budget_bytes,
            std::string model_artifact_identity,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority);

        static constexpr size_t kSectionCount = 6;

        struct SectionIndex
        {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            uint64_t checksum = 0;
        };

        struct RecordIndex
        {
            PrefixBlockHandle handle;
            uint64_t record_offset = 0;
            uint64_t record_bytes = 0;
            uint64_t sequence = 0;
            std::array<SectionIndex, kSectionCount> sections{};
        };

        bool initialize(std::string *error);
        bool refreshIndexLocked(int archive_fd, std::string *error);
        bool scanRecordsLocked(
            int archive_fd,
            uint64_t start_offset,
            uint64_t file_bytes,
            uint64_t *valid_end,
            std::string *error);
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
        bool compactIfNeededLocked(int archive_fd, std::string *error);
        bool rewriteArchiveLocked(int source_fd, std::string *error);
        bool verifyRecordPayloadLocked(
            int archive_fd,
            const RecordIndex &record,
            std::string *error);
        void releaseHydrationTicket() noexcept;
        void applyPutRecord(RecordIndex record);
        void applyDeleteRecord(const PrefixCacheKey &key);
        bool applyTouchRecord(
            const PrefixCacheKey &key,
            uint64_t sequence);

        std::filesystem::path archive_path_;
        std::filesystem::path lock_path_;
        size_t budget_bytes_ = 0;
        std::string model_artifact_identity_;

        mutable std::mutex mutex_;
        bool ready_ = false;
        std::string initialization_error_;
        uint64_t scan_offset_ = 0;
        uint64_t next_sequence_ = 1;
        uint64_t active_bytes_ = 0;
        uint64_t archive_device_ = 0;
        uint64_t archive_inode_ = 0;
        std::unordered_map<PrefixCacheKey, RecordIndex, PrefixCacheKeyHasher> records_;
        size_t active_hydration_tickets_ = 0u;

        /*
         * Declaration order is intentional: reverse destruction frees the
         * scratch allocation before its ledger lease, and the lease before
         * the authority it retains.  The process-shared archive serializes all
         * uses of this one buffer under mutex_.
         */
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority_;
        PhysicalMemoryAllocationLease archive_scratch_memory_lease_;
        std::vector<uint8_t> archive_scratch_;
    };
} // namespace llaminar2
