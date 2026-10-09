/**
 * @file PrefixArchiveMaintenanceWriter.h
 * @brief Bounded, data-synchronous writes on the archive's background worker.
 *
 * A small scratch buffer alone does not bound dirty filesystem pages. Each
 * maintenance write therefore completes with O_DSYNC before the next chunk is
 * produced. Unlike sync_file_range, this contract follows the filesystem's
 * native write implementation, including an OverlayFS backing file. There is
 * no submitted-range queue, join lifecycle, extra thread or payload allocation.
 *
 * Only the archive maintenance worker uses this writer, for journal metadata.
 * Payloads live in separate immutable inodes and are never read by compaction.
 * The final metadata tail, file fsync, rename and directory fsync own publication;
 * payload eviction does not depend on this worker catching a moving frontier.
 */
#pragma once

#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace llaminar2
{
    /**
     * @brief One bounded native-write contract borrowing a worker-owned inode.
     *
     * The descriptor owner must outlive this object and keep its write mode
     * unchanged. Construction verifies the actual kernel flags rather than a
     * caller's advertised capability. A failed write permanently forbids more
     * work; the unpublished archive must never become a successful replacement.
     */
    class PrefixArchiveMaintenanceWriter final
    {
    public:
        /** @brief Completed native I/O evidence, never an admission ledger. */
        struct Evidence final
        {
            std::uint64_t durable_writes = 0; ///< Successful native write calls.
            std::uint64_t durable_bytes = 0; ///< Exact returned byte extents.
            std::uint64_t largest_write_bytes = 0; ///< Never exceeds one chunk.
            int native_open_flags = 0; ///< Actual F_GETFL flags, including O_DSYNC.
        };

        /**
         * @brief Validate a writable, data-synchronous destination before copying.
         * @param fd Borrowed descriptor of the exact unpublished output inode.
         * @throws std::runtime_error If native flags do not certify O_DSYNC.
         */
        explicit PrefixArchiveMaintenanceWriter(int fd) : fd_(fd)
        {
            int flags;
            do { flags = ::fcntl(fd_, F_GETFL); }
            while (flags < 0 && errno == EINTR);
            if (flags < 0 || (flags & O_DSYNC) != O_DSYNC ||
                (flags & O_ACCMODE) == O_RDONLY)
            {
                throw std::runtime_error(
                    "prefix maintenance requires a writable O_DSYNC destination");
            }
            evidence_.native_open_flags = flags;
        }

        PrefixArchiveMaintenanceWriter(const PrefixArchiveMaintenanceWriter &) = delete;
        PrefixArchiveMaintenanceWriter &operator=(const PrefixArchiveMaintenanceWriter &) = delete;

        /**
         * @brief Complete only these live bytes before admitting the next chunk.
         * @param bytes Nonempty extent no larger than the canonical copy scratch.
         * @param error Optional precise geometry or native-I/O failure.
         * @return Whether every byte completed its data-synchronous native write.
         *
         * Short writes retain only their unwritten suffix. EINTR retries that
         * same operation, never an alternative path. Successful O_DSYNC returns
         * are the native completion edge; no range-flush receipt is invented.
         */
        bool write(std::span<const std::byte> bytes, std::string *error = nullptr)
        {
            if (state_ == State::Failed)
                return fail("prefix maintenance writer already failed", error);
            if (bytes.empty() || bytes.data() == nullptr ||
                bytes.size() > PrefixArchiveIOGeometry::compactionBytes())
            {
                return fail("prefix maintenance write has an empty or oversized extent", error);
            }
            while (!bytes.empty())
            {
                const ssize_t written = ::write(fd_, bytes.data(), bytes.size());
                if (written < 0 && errno == EINTR)
                    continue;
                if (written <= 0)
                    return fail("failed durable prefix maintenance write: " +
                        std::string(written < 0 ? std::strerror(errno) : "no native progress"), error);
                const auto completed = static_cast<std::size_t>(written);
                ++evidence_.durable_writes;
                evidence_.durable_bytes += completed;
                evidence_.largest_write_bytes = std::max(
                    evidence_.largest_write_bytes, static_cast<std::uint64_t>(completed));
                bytes = bytes.subspan(completed);
            }
            return true;
        }

        /** @return Actual kernel flags and completed native writes, not promises. */
        [[nodiscard]] Evidence evidence() const noexcept { return evidence_; }

    private:
        /** @brief There is no pending-I/O state: native write completion owns it. */
        enum class State { Ready, Failed };

        /** @brief Make failure terminal without retrying I/O or publishing output. */
        bool fail(const std::string &message, std::string *error)
        {
            state_ = State::Failed;
            if (error)
                *error = message;
            return false;
        }

        int fd_; ///< Borrowed only; FileDescriptor retains sole native ownership.
        State state_ = State::Ready;
        Evidence evidence_;
    };
} // namespace llaminar2
