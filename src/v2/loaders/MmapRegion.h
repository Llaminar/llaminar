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
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <iomanip>
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
     * Maps an entire file into the process address space. CPU paths may use
     * MAP_POPULATE to pre-fault pages, while GPU upload paths can demand-page
     * the mapping to avoid blocking on a whole-file cold fault before H2D copy.
     * MAP_PRIVATE ensures the mapping is read-only and copy-on-write (safe for concurrent access).
     *
     * Lifetime: The mapped region stays valid until this object is destroyed.
     * Thread safety: Read access to the mapped region is inherently thread-safe.
     */
    class MmapRegion
    {
    public:
        /**
         * @brief Controls whether mmap creation eagerly faults file pages.
         *
         * CPU weight paths benefit from eager prefaulting because decode may read
         * directly from the mapped file. GPU paths usually upload tensors to
         * device memory immediately, so eager whole-file prefaulting can turn a
         * cold 16+ GB model load into a long blocking page-fault stall before the
         * first H2D copy even starts.
         */
        enum class PrefaultPolicy
        {
            Auto,          ///< Preserve historical behavior for this mapping mode.
            EagerPopulate, ///< Use MAP_POPULATE when NUMA binding is not requested.
            DemandPaged,   ///< Map lazily and rely on sequential readahead hints.
        };

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
            : base_(other.base_), length_(other.length_), fd_(other.fd_), path_(std::move(other.path_)),
              eager_prefaulted_(other.eager_prefaulted_)
        {
            other.base_ = nullptr;
            other.length_ = 0;
            other.fd_ = -1;
            other.eager_prefaulted_ = false;
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
                eager_prefaulted_ = other.eager_prefaulted_;
                other.base_ = nullptr;
                other.length_ = 0;
                other.fd_ = -1;
                other.eager_prefaulted_ = false;
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
         * @brief Whether create() used a synchronous eager prefault path.
         *
         * This is intentionally exposed for diagnostics and regression tests:
         * GPU-target model loading must stay demand-paged, while CPU/NUMA model
         * loading can deliberately prefault to stabilize host-side decode.
         */
        bool wasEagerPrefaulted() const { return eager_prefaulted_; }

        /**
         * @brief Release physical pages backing this mmap region.
         *
         * Calls madvise(MADV_DONTNEED) to tell the kernel it can reclaim the
         * physical pages. The virtual address range remains valid — future reads
         * will re-fault pages from the underlying file (via page cache).
         *
         * This is safe to call after all tensor data has been copied to owned
         * buffers (e.g., VNNI interleaved format). Small weights like FP32 norms
         * (~0.7 MB) will transparently re-fault on next access with negligible cost.
         *
         * @return Number of bytes advised, or 0 on failure/unsupported platform
         */
        size_t adviseDontneed()
        {
#ifdef __linux__
            if (base_ && length_ > 0)
            {
                if (::madvise(base_, length_, MADV_DONTNEED) == 0)
                {
                    return length_;
                }
                LOG_WARN("[MmapRegion] madvise(MADV_DONTNEED) failed: errno=" << errno);
            }
#endif
            return 0;
        }

        /**
         * @brief Release physical pages for an arbitrary address range within any mmap'd region.
         *
         * Page-aligns the range and calls madvise(MADV_DONTNEED). Safe to call on
         * sub-ranges of mmap'd files — the VA range stays valid and will re-fault
         * from the page cache on next access.
         *
         * Use this to incrementally release mmap pages after tensor data has been
         * copied to owned buffers (e.g., VNNI interleaved engines), reducing peak RSS
         * during weight preparation.
         *
         * @param addr Start of the range (need not be page-aligned)
         * @param len  Length in bytes
         * @return Number of bytes advised (page-aligned), or 0 on failure
         */
        static size_t adviseDontneedRange(const void *addr, size_t len)
        {
#ifdef __linux__
            if (!addr || len == 0)
                return 0;

            const size_t page_size = 4096;
            auto raw = reinterpret_cast<uintptr_t>(addr);
            uintptr_t aligned_start = raw & ~(page_size - 1);
            uintptr_t aligned_end = (raw + len + page_size - 1) & ~(page_size - 1);
            size_t aligned_len = aligned_end - aligned_start;

            if (::madvise(reinterpret_cast<void *>(aligned_start), aligned_len, MADV_DONTNEED) == 0)
            {
                return aligned_len;
            }
            // Silently ignore failures (addr might not be in an mmap'd region)
#endif
            return 0;
        }

        /**
         * @brief Pre-populate the OS page cache for a file using sequential I/O
         *
         * Reads the file sequentially with large buffers to warm the page cache
         * at full disk bandwidth. Call this on a single rank before multi-rank
         * mmap to avoid each rank independently thrashing the disk.
         *
         * @param file_path Path to the file
         * @return true if successful, false on error
         */
        static bool prepopulatePageCache(const std::string &file_path)
        {
#ifdef __linux__
            int fd = ::open(file_path.c_str(), O_RDONLY);
            if (fd < 0)
            {
                LOG_WARN("[MmapRegion] prepopulatePageCache: failed to open " << file_path);
                return false;
            }

            struct stat st;
            if (::fstat(fd, &st) != 0)
            {
                ::close(fd);
                return false;
            }
            const size_t file_size = static_cast<size_t>(st.st_size);

            // Use large sequential reads for maximum disk throughput.
            // 8MB buffer aligns with typical SSD/NVMe command queue depth.
            ::posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

            constexpr size_t BUF_SIZE = 8 * 1024 * 1024; // 8 MB
            // Use mmap for the read buffer to avoid stack overflow and get page-aligned memory
            void *buf = ::mmap(nullptr, BUF_SIZE, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buf == MAP_FAILED)
            {
                ::close(fd);
                return false;
            }

            size_t total_read = 0;
            auto start = std::chrono::steady_clock::now();
            while (total_read < file_size)
            {
                size_t to_read = std::min(BUF_SIZE, file_size - total_read);
                ssize_t n = ::read(fd, buf, to_read);
                if (n <= 0)
                    break;
                total_read += static_cast<size_t>(n);
            }
            auto elapsed = std::chrono::steady_clock::now() - start;
            double secs = std::chrono::duration<double>(elapsed).count();
            double mbps = (total_read / (1024.0 * 1024.0)) / (secs > 0 ? secs : 1e-9);

            ::munmap(buf, BUF_SIZE);
            ::close(fd);

            LOG_DEBUG("[MmapRegion] Page cache pre-populated: " << file_path
                                                                << " (" << (total_read / (1024 * 1024)) << " MB in "
                                                                << std::fixed << std::setprecision(1) << secs << "s, "
                                                                << std::fixed << std::setprecision(0) << mbps << " MB/s)");
            return total_read == file_size;
#else
            (void)file_path;
            return false;
#endif
        }

        /**
         * @brief Create an MmapRegion by memory-mapping an entire file
         *
         * When numa_node >= 0, uses mbind(MPOL_BIND) to ensure pages are allocated
         * on the specified NUMA node, then parallel first-touch to pre-fault.
         * When numa_node < 0, Auto/EagerPopulate uses MAP_POPULATE for immediate
         * pre-faulting. DemandPaged maps lazily and issues readahead hints instead.
         *
         * @param file_path Path to the file to map
         * @param numa_node Target NUMA node for page placement (-1 = default)
         * @param skip_cache_eviction When true, skip POSIX_FADV_DONTNEED before NUMA
         *        mmap. Set this when the page cache has been pre-populated by
         *        prepopulatePageCache() — the first-touch loop will fault from the
         *        warm page cache instead of disk, giving ~10× faster loading.
         * @param prefault_policy Controls whole-file prefault behavior.
         * @return Unique pointer to MmapRegion, or nullptr on failure
         */
        static std::unique_ptr<MmapRegion> create(const std::string &file_path, int numa_node = -1,
                                                  bool skip_cache_eviction = false,
                                                  PrefaultPolicy prefault_policy = PrefaultPolicy::Auto)
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

            if (numa_bind && !skip_cache_eviction)
            {
                // Evict any stale page-cache pages for this file so that our
                // first-touch loop below allocates fresh pages on the target
                // NUMA node. Without this, cached pages from a prior run (or
                // another process) may reside on the wrong node.
                //
                // SKIP this in multi-rank mode (skip_cache_eviction=true):
                // prepopulatePageCache() has already warmed the cache, and
                // evicting it would force re-reading from disk via the OMP
                // parallel first-touch loop, which creates N concurrent page
                // fault streams that destroy disk readahead throughput.
                ::posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
            }

            // Hint for sequential access (improves kernel readahead)
            ::posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

            // Choose mmap strategy based on NUMA binding requirement:
            // - With NUMA binding: mmap without MAP_POPULATE, then mbind() to target node.
            // - Without NUMA binding: historical Auto eagerly prefaults with MAP_POPULATE.
            // - GPU upload paths select DemandPaged to avoid a whole-file cold-fault stall.
            const bool eager_populate =
                !numa_bind &&
                prefault_policy != PrefaultPolicy::DemandPaged;
            int mmap_flags = MAP_PRIVATE;
            if (eager_populate)
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
                                                 << " (" << num_pages << " pages"
                                                 << (skip_cache_eviction ? ", cache-warm" : ", cold")
                                                 << ", THP requested)");
            }
            else
            {
                if (!eager_populate)
                {
                    // Demand-paged GPU staging should not request whole-file
                    // prefault or prefetch. Sequential access hints are enough
                    // for kernel readahead as the actual tensor uploads walk the
                    // GGUF, and they avoid cold-load stalls before the first H2D.
                    ::madvise(base, file_size, MADV_SEQUENTIAL);
                }
                else
                {
                    ::madvise(base, file_size, MADV_WILLNEED);
                }
                ::madvise(base, file_size, MADV_HUGEPAGE);
                LOG_DEBUG("[MmapRegion] Mapped " << file_path
                                                 << " (" << (file_size / (1024 * 1024)) << " MB"
                                                 << (eager_populate ? ", eager-prefault" : ", demand-paged")
                                                 << ", THP requested)");
            }

            return std::unique_ptr<MmapRegion>(
                new MmapRegion(base, file_size, fd, file_path, numa_bind || eager_populate));
#else
            (void)numa_node;
            (void)skip_cache_eviction;
            (void)prefault_policy;
            LOG_WARN("[MmapRegion] mmap not supported on this platform, falling back to ifstream");
            return nullptr;
#endif
        }

    private:
        MmapRegion(void *base, size_t length, int fd, const std::string &path, bool eager_prefaulted)
            : base_(base), length_(length), fd_(fd), path_(path), eager_prefaulted_(eager_prefaulted) {}

        void *base_ = nullptr;
        size_t length_ = 0;
        int fd_ = -1;
        std::string path_;
        bool eager_prefaulted_ = false;
    };

} // namespace llaminar2
