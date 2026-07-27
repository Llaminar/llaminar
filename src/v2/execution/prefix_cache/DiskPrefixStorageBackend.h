/**
 * @file DiskPrefixStorageBackend.h
 * @brief Durable, model-addressed prefix-cache archive.
 *
 * A disk cache is one append-only `<model-sha256>.kvcache` file. Each put or
 * delete operation is an independently checksummed record. Complete records
 * survive process restart; an interrupted tail is ignored and removed before
 * the next append. The archive owns capacity eviction and compaction so the
 * configured disk budget remains a real bounded tier.
 */

#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <array>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Persistent prefix blocks stored in one model-specific blob file.
     *
     * Instances opened through openShared() are shared by every local
     * orchestrator targeting the same archive. A separate advisory lock file
     * coordinates independent MPI processes that share the directory.
     */
    class DiskPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        /**
         * @brief Open or create a model-specific archive.
         *
         * Direct construction is retained for focused tests. Production code
         * should use openShared() so LocalTP children share one in-process
         * index and one synchronization domain.
         */
        DiskPrefixStorageBackend(
            std::filesystem::path archive_path,
            size_t budget_bytes,
            std::string model_sha256);

        /**
         * @brief Return a process-shared backend for an archive pathname.
         */
        static std::shared_ptr<DiskPrefixStorageBackend> openShared(
            const std::filesystem::path &archive_path,
            size_t budget_bytes,
            const std::string &model_sha256,
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
        bool readBlockIntoRamBackend(
            const PrefixCacheKey &key,
            const PrefixPayloadLayout &layout,
            IPrefixStorageBackend &ram_backend,
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
        void applyPutRecord(RecordIndex record);
        void applyDeleteRecord(const PrefixCacheKey &key);
        bool applyTouchRecord(
            const PrefixCacheKey &key,
            uint64_t sequence);

        std::filesystem::path archive_path_;
        std::filesystem::path lock_path_;
        size_t budget_bytes_ = 0;
        std::string model_sha256_;

        mutable std::mutex mutex_;
        bool ready_ = false;
        std::string initialization_error_;
        uint64_t scan_offset_ = 0;
        uint64_t next_sequence_ = 1;
        uint64_t active_bytes_ = 0;
        uint64_t archive_device_ = 0;
        uint64_t archive_inode_ = 0;
        std::unordered_map<PrefixCacheKey, RecordIndex, PrefixCacheKeyHasher> records_;
    };
} // namespace llaminar2
