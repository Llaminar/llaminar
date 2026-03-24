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

#ifdef HAVE_NUMA
#include <numaif.h>
#include <numa.h>
#endif

#ifdef _OPENMP
#include <omp.h>
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
         * When numa_node >= 0, uses mbind(MPOL_BIND) to ensure pages are allocated
         * on the specified NUMA node, then madvise(MADV_WILLNEED) to pre-fault.
         * When numa_node < 0, uses MAP_POPULATE for immediate pre-faulting (default
         * NUMA policy applies — pages land on the calling CPU's local node).
         *
         * @param file_path Path to the file to map
         * @param numa_node Target NUMA node for page placement (-1 = default)
         * @return Unique pointer to MmapRegion, or nullptr on failure
         */
        static std::unique_ptr<MmapRegion> create(const std::string &file_path, int numa_node = -1)
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

            const bool numa_bind = (numa_node >= 0);

            if (numa_bind)
            {
                // Evict any stale page-cache pages for this file so that our
                // first-touch loop below allocates fresh pages on the target
                // NUMA node. Without this, cached pages from a prior run (or
                // another process) may reside on the wrong node.
                ::posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
            }

            // Hint for sequential access (improves kernel readahead)
            ::posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

            // Choose mmap strategy based on NUMA binding requirement:
            // - With NUMA binding: mmap without MAP_POPULATE, then mbind() to target node
            // - Without NUMA binding: mmap with MAP_POPULATE for immediate pre-fault
            int mmap_flags = MAP_PRIVATE;
            if (!numa_bind)
            {
                mmap_flags |= MAP_POPULATE;
            }

            void *base = ::mmap(nullptr, file_size, PROT_READ, mmap_flags, fd, 0);
            if (base == MAP_FAILED)
            {
                LOG_ERROR("[MmapRegion] mmap failed for " << file_path
                                                          << " (" << (file_size / (1024 * 1024)) << " MB)"
                                                          << " (errno=" << errno << ")");
                ::close(fd);
                return nullptr;
            }

#ifdef HAVE_NUMA
            // Bind mmap pages to the target NUMA node before pre-faulting.
            // This ensures all page-cache pages allocated for this mapping
            // land on the correct NUMA node, avoiding cross-socket bandwidth
            // penalties for memory-bandwidth-bound GEMV decode.
            if (numa_bind && numa_available() >= 0)
            {
                unsigned long nodemask = 1UL << numa_node;
                long rc = ::mbind(base, file_size, MPOL_BIND, &nodemask,
                                  sizeof(nodemask) * 8, 0);
                if (rc != 0)
                {
                    LOG_WARN("[MmapRegion] mbind to NUMA node " << numa_node
                                                                << " failed (errno=" << errno << "), pages may be on wrong node");
                }
                else
                {
                    LOG_DEBUG("[MmapRegion] Bound " << (file_size / (1024 * 1024))
                                                    << " MB to NUMA node " << numa_node);
                }
            }
#endif

            if (numa_bind)
            {
                // Explicit parallel first-touch: OMP threads inherit the process's
                // cpu-set binding (e.g. cores 28-55 for cpu:1), so page faults from
                // these threads allocate on the target NUMA node. This is more
                // reliable than madvise(MADV_WILLNEED), whose async readahead
                // completes on kernel worker threads that may be on any node.
                const size_t page_size = 4096;
                const size_t num_pages = (file_size + page_size - 1) / page_size;
                const volatile uint8_t *p = static_cast<const volatile uint8_t *>(base);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < num_pages; i++)
                {
                    (void)p[i * page_size];
                }

                // Request transparent huge pages after first-touch. In "madvise"
                // mode, khugepaged will collapse 4KB pages into 2MB pages in the
                // background, reducing TLB misses for the 7+ GB model weights.
                ::madvise(base, file_size, MADV_HUGEPAGE);

                LOG_DEBUG("[MmapRegion] Mapped " << file_path << " (" << (file_size / (1024 * 1024))
                                                 << " MB) with NUMA first-touch on node " << numa_node
                                                 << " (" << num_pages << " pages, THP requested)");
            }
            else
            {
                // Non-NUMA path already pre-faulted via MAP_POPULATE
                ::madvise(base, file_size, MADV_WILLNEED);
                ::madvise(base, file_size, MADV_HUGEPAGE);
                LOG_DEBUG("[MmapRegion] Mapped " << file_path << " (" << (file_size / (1024 * 1024)) << " MB, THP requested)");
            }

            return std::unique_ptr<MmapRegion>(new MmapRegion(base, file_size, fd, file_path));
#else
            (void)numa_node;
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
