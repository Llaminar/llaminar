#pragma once

/**
 * @file MmapRegion.h
 * @brief RAII wrapper for mmap'd file regions
 *
 * Used by ModelLoader to memory-map GGUF files for zero-syscall tensor loading.
 * When active, tensor data is read via memcpy from the mmap'd region instead of
 * seekg+read through an ifstream under file_mutex_, eliminating serialization.
 *
 * Usage:
 *   auto region = MmapRegion::create("/path/to/model.gguf");
 *   if (region) {
 *       const uint8_t* tensor_data = region->data() + data_offset + tensor_offset;
 *       std::memcpy(dst, tensor_data, tensor_size);
 *   }
 */

#include "../utils/Logger.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace llaminar2
{

    /**
     * @brief RAII wrapper for a memory-mapped file region
     *
     * Maps an entire file into the process address space using mmap(MAP_PRIVATE | MAP_POPULATE).
     * MAP_POPULATE pre-faults all pages via kernel readahead, avoiding per-page faults during access.
     * MAP_PRIVATE ensures the mapping is read-only and copy-on-write (safe for concurrent access).
     *
     * Lifetime: The mapped region stays valid until this object is destroyed.
     * Thread safety: Read access to the mapped region is inherently thread-safe.
     */
    class MmapRegion
    {
    public:
        ~MmapRegion()
        {
#ifdef __linux__
            if (base_ != MAP_FAILED && base_ != nullptr)
            {
                ::munmap(base_, length_);
            }
            if (fd_ >= 0)
            {
                ::close(fd_);
            }
#endif
        }

        // Non-copyable
        MmapRegion(const MmapRegion &) = delete;
        MmapRegion &operator=(const MmapRegion &) = delete;

        // Movable
        MmapRegion(MmapRegion &&other) noexcept
            : base_(other.base_), length_(other.length_), fd_(other.fd_), path_(std::move(other.path_))
        {
            other.base_ = nullptr;
            other.length_ = 0;
            other.fd_ = -1;
        }

        MmapRegion &operator=(MmapRegion &&other) noexcept
        {
            if (this != &other)
            {
#ifdef __linux__
                if (base_ != MAP_FAILED && base_ != nullptr)
                {
                    ::munmap(base_, length_);
                }
                if (fd_ >= 0)
                {
                    ::close(fd_);
                }
#endif
                base_ = other.base_;
                length_ = other.length_;
                fd_ = other.fd_;
                path_ = std::move(other.path_);
                other.base_ = nullptr;
                other.length_ = 0;
                other.fd_ = -1;
            }
            return *this;
        }

        /** @brief Get base pointer to mapped region */
        const uint8_t *data() const { return static_cast<const uint8_t *>(base_); }

        /** @brief Get total size of mapped region in bytes */
        size_t size() const { return length_; }

        /** @brief Get the file path that was mapped */
        const std::string &path() const { return path_; }

        /**
         * @brief Create an MmapRegion by memory-mapping an entire file
         *
         * Uses MAP_PRIVATE | MAP_POPULATE for read-only access with pre-faulted pages.
         * Also calls posix_fadvise(SEQUENTIAL) and madvise(WILLNEED) for optimal readahead.
         *
         * @param file_path Path to the file to map
         * @return Unique pointer to MmapRegion, or nullptr on failure
         */
        static std::unique_ptr<MmapRegion> create(const std::string &file_path)
        {
#ifdef __linux__
            // Open file read-only
            int fd = ::open(file_path.c_str(), O_RDONLY);
            if (fd < 0)
            {
                LOG_ERROR("[MmapRegion] Failed to open file: " << file_path << " (errno=" << errno << ")");
                return nullptr;
            }

            // Get file size
            struct stat st;
            if (::fstat(fd, &st) != 0)
            {
                LOG_ERROR("[MmapRegion] Failed to stat file: " << file_path << " (errno=" << errno << ")");
                ::close(fd);
                return nullptr;
            }

            size_t file_size = static_cast<size_t>(st.st_size);
            if (file_size == 0)
            {
                LOG_ERROR("[MmapRegion] File is empty: " << file_path);
                ::close(fd);
                return nullptr;
            }

            // Hint for sequential access (improves kernel readahead)
            ::posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

            // Memory-map the file
            // MAP_PRIVATE: Copy-on-write (safe, no file modification)
            // MAP_POPULATE: Pre-fault all pages (avoid per-page faults during tensor loading)
            void *base = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
            if (base == MAP_FAILED)
            {
                LOG_ERROR("[MmapRegion] mmap failed for " << file_path
                                                          << " (" << (file_size / (1024 * 1024)) << " MB)"
                                                          << " (errno=" << errno << ")");
                ::close(fd);
                return nullptr;
            }

            // Additional hint: we'll need all of this data
            ::madvise(base, file_size, MADV_WILLNEED);

            LOG_DEBUG("[MmapRegion] Mapped " << file_path << " (" << (file_size / (1024 * 1024)) << " MB)");

            return std::unique_ptr<MmapRegion>(new MmapRegion(base, file_size, fd, file_path));
#else
            LOG_WARN("[MmapRegion] mmap not supported on this platform, falling back to ifstream");
            return nullptr;
#endif
        }

    private:
        MmapRegion(void *base, size_t length, int fd, const std::string &path)
            : base_(base), length_(length), fd_(fd), path_(path) {}

        void *base_ = nullptr;
        size_t length_ = 0;
        int fd_ = -1;
        std::string path_;
    };

} // namespace llaminar2
