/**
 * @file PinnedMemory.h
 * @brief Pinned (page-locked) host memory for fast GPU<->CPU transfers
 *
 * Pinned memory enables:
 * - DMA transfers that bypass CPU cache
 * - Async memcpy that can overlap with compute
 * - ~10x faster D2H/H2D transfers vs pageable memory
 *
 * The HIP/CUDA runtime automatically detects if a host pointer is pinned
 * and uses the fast DMA path. No changes needed in memcpy calls.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include "utils/Logging.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    /**
     * @brief RAII wrapper for pinned (page-locked) host memory
     *
     * Allocates memory using hipHostMalloc/cudaHostAlloc for fast GPU transfers,
     * falling back to aligned_alloc for CPU-only builds or allocation failures.
     *
     * Usage:
     *   PinnedBuffer<float> buffer(1024);  // 1024 floats in pinned memory
     *   buffer.data()[0] = 1.0f;
     *   // GPU transfers to/from buffer.data() will use DMA
     *
     * @tparam T Element type
     */
    template <typename T>
    class PinnedBuffer
    {
    public:
        using value_type = T;
        using size_type = size_t;

        static constexpr size_t ALIGNMENT = 64; // Cache line alignment

        /// Default constructor (empty buffer)
        PinnedBuffer() : data_(nullptr), size_(0), is_pinned_(false) {}

        /// Construct with size (zero-initialized)
        explicit PinnedBuffer(size_t n) : data_(nullptr), size_(n), is_pinned_(false)
        {
            if (n > 0)
            {
                allocate(n);
                std::memset(data_, 0, n * sizeof(T));
            }
        }

        /// Non-copyable (pinned memory is expensive to allocate)
        PinnedBuffer(const PinnedBuffer &) = delete;
        PinnedBuffer &operator=(const PinnedBuffer &) = delete;

        /// Move constructor
        PinnedBuffer(PinnedBuffer &&other) noexcept
            : data_(other.data_), size_(other.size_), is_pinned_(other.is_pinned_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.is_pinned_ = false;
        }

        /// Move assignment
        PinnedBuffer &operator=(PinnedBuffer &&other) noexcept
        {
            if (this != &other)
            {
                deallocate();
                data_ = other.data_;
                size_ = other.size_;
                is_pinned_ = other.is_pinned_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.is_pinned_ = false;
            }
            return *this;
        }

        ~PinnedBuffer()
        {
            deallocate();
        }

        // ========== Accessors ==========

        T *data() { return data_; }
        const T *data() const { return data_; }

        size_t size() const { return size_; }
        size_t byte_size() const { return size_ * sizeof(T); }
        bool empty() const { return size_ == 0; }

        T &operator[](size_t i) { return data_[i]; }
        const T &operator[](size_t i) const { return data_[i]; }

        /// Returns true if memory is actually pinned (vs fallback aligned_alloc)
        bool is_pinned() const { return is_pinned_; }

        // ========== Resize ==========

        void resize(size_t new_size)
        {
            if (new_size == size_)
                return;

            PinnedBuffer<T> new_buffer(new_size);
            if (data_ && new_buffer.data_)
            {
                size_t copy_size = std::min(size_, new_size);
                std::memcpy(new_buffer.data_, data_, copy_size * sizeof(T));
            }
            *this = std::move(new_buffer);
        }

        void clear()
        {
            deallocate();
            data_ = nullptr;
            size_ = 0;
            is_pinned_ = false;
        }

    private:
        T *data_;
        size_t size_;
        bool is_pinned_;

        void allocate(size_t n)
        {
            size_t bytes = n * sizeof(T);

#ifdef HAVE_ROCM
            // Try HIP pinned allocation first
            hipError_t err = hipHostMalloc(&data_, bytes, hipHostMallocDefault);
            if (err == hipSuccess)
            {
                is_pinned_ = true;
                LOG_TRACE("[PinnedBuffer] Allocated " << bytes << " bytes of HIP pinned memory");
                return;
            }
            LOG_WARN("[PinnedBuffer] hipHostMalloc failed (" << hipGetErrorString(err)
                                                             << "), falling back to aligned_alloc");
#endif

#ifdef HAVE_CUDA
            // Try CUDA pinned allocation
            cudaError_t err = cudaMallocHost(&data_, bytes);
            if (err == cudaSuccess)
            {
                is_pinned_ = true;
                LOG_TRACE("[PinnedBuffer] Allocated " << bytes << " bytes of CUDA pinned memory");
                return;
            }
            LOG_WARN("[PinnedBuffer] cudaMallocHost failed (" << cudaGetErrorString(err)
                                                              << "), falling back to aligned_alloc");
#endif

            // Fallback to aligned_alloc
            size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
            data_ = static_cast<T *>(std::aligned_alloc(ALIGNMENT, aligned_bytes));
            if (!data_)
            {
                throw std::bad_alloc();
            }
            is_pinned_ = false;
            LOG_TRACE("[PinnedBuffer] Allocated " << bytes << " bytes of aligned (non-pinned) memory");
        }

        void deallocate()
        {
            if (!data_)
                return;

            if (is_pinned_)
            {
#ifdef HAVE_ROCM
                hipHostFree(data_);
#elif defined(HAVE_CUDA)
                cudaFreeHost(data_);
#endif
            }
            else
            {
                std::free(data_);
            }
            data_ = nullptr;
        }
    };

    // ============================================================================
    // Global Pinned Memory Statistics (for debugging)
    // ============================================================================

    struct PinnedMemoryStats
    {
        std::atomic<size_t> total_allocated{0};
        std::atomic<size_t> total_pinned{0};
        std::atomic<size_t> total_fallback{0};
        std::atomic<size_t> allocation_count{0};

        static PinnedMemoryStats &instance()
        {
            static PinnedMemoryStats stats;
            return stats;
        }
    };

} // namespace llaminar2
