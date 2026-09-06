/**
 * @file AlignedVector.h
 * @brief Cache-aligned vector container for high-performance SIMD operations
 *
 * Provides a std::vector-like container with 64-byte alignment guarantees,
 * enabling unconditional use of:
 * - Streaming stores (_mm512_stream_ps) - bypass cache for write-only data
 * - Aligned loads (_mm512_load_ps) - faster than unaligned loads
 * - Cache line optimization - avoid cache line splits
 *
 * Dedicated NUMA mappings retain inaccessible virtual guards on both sides.
 * Guards prevent huge-page sharing across independent first-touch owners while
 * large payloads remain huge-page aligned and eligible. Physical accounting is
 * the page-rounded payload extent; unfaultable guards consume no physical RAM.
 *
 * Performance impact:
 * - BF16/FP16 conversion: 15% faster with streaming stores
 * - Large tensor operations: 5-20% better cache utilization
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstdlib> // aligned_alloc, free
#include <cstdint> // uintptr_t
#include <cstring> // memcpy
#include <stdexcept>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory> // uninitialized_copy, uninitialized_fill
#include <new>    // bad_alloc
#include <utility>
#include <vector>
#ifdef __linux__
#include <sys/mman.h> // madvise, MADV_HUGEPAGE
#include <unistd.h>   // sysconf
#endif

namespace llaminar2
{

    /**
     * @brief Vector with 64-byte aligned memory allocation
     *
     * Drop-in replacement for std::vector<T> with guaranteed 64-byte alignment.
     * Uses aligned_alloc/free instead of new[]/delete[].
     *
     * 64-byte alignment ensures:
     * - Cache line alignment (modern CPUs have 64-byte cache lines)
     * - Streaming store support (_mm512_stream_ps requires 64-byte alignment)
     * - Aligned SIMD load support (faster than unaligned loads)
     *
     * @tparam T Element type (typically float, uint16_t, int8_t)
     */
    template <typename T>
    class AlignedVector
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;

        /// Cache line alignment (64 bytes on x86-64)
        static constexpr size_t ALIGNMENT = 64;

        /**
         * @brief Physical allocation authority retained by this vector.
         *
         * Ordinary tensor storage uses the process heap.  Exact NUMA receive
         * buffers use a private anonymous mapping so destroying the owner
         * revokes the complete virtual range instead of returning potentially
         * accelerator-pinned pages to a reusable libc arena.
         */
        enum class StorageKind : std::uint8_t
        {
            Heap,
            AnonymousPageMapping,
        };

        // ========== Constructors ==========

        /// Default constructor (empty vector)
        AlignedVector() : data_(nullptr), size_(0), capacity_(0) {}

        /// Construct with size (default-initialized elements)
        explicit AlignedVector(size_t n) : data_(nullptr), size_(n), capacity_(n)
        {
            if (n > 0)
            {
                allocate(n);
                // Default-initialize elements
                std::uninitialized_fill_n(data_, n, T{});
            }
        }

        /// Construct with size and fill value
        AlignedVector(size_t n, const T &value) : data_(nullptr), size_(n), capacity_(n)
        {
            if (n > 0)
            {
                allocate(n);
                std::uninitialized_fill_n(data_, n, value);
            }
        }

        /// Construct from initializer list
        AlignedVector(std::initializer_list<T> init)
            : data_(nullptr), size_(init.size()), capacity_(init.size())
        {
            if (size_ > 0)
            {
                allocate(size_);
                std::uninitialized_copy(init.begin(), init.end(), data_);
            }
        }

        /**
         * @brief Copy ordinary vector elements into aligned storage.
         * @param source Fully initialized source elements.
         *
         * The destination is allocated without a preceding fill pass, then
         * constructed directly from @p source. This is the compatibility edge
         * for callers that do not transfer an existing AlignedVector owner.
         */
        explicit AlignedVector(const std::vector<T> &source)
            : data_(nullptr), size_(source.size()), capacity_(source.size())
        {
            if (size_ > 0u)
            {
                allocate(size_);
                std::uninitialized_copy(source.begin(), source.end(), data_);
            }
        }

        /// Copy constructor
        AlignedVector(const AlignedVector &other)
            : data_(nullptr), size_(other.size_), capacity_(other.size_)
        {
            if (size_ > 0)
            {
                allocate(size_);
                std::uninitialized_copy_n(other.data_, size_, data_);
            }
        }

        /// Move constructor
        AlignedVector(AlignedVector &&other) noexcept
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
              allocation_bytes_(other.allocation_bytes_),
              allocation_alignment_(other.allocation_alignment_),
              storage_kind_(other.storage_kind_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
            other.allocation_bytes_ = 0;
            other.allocation_alignment_ = ALIGNMENT;
            other.storage_kind_ = StorageKind::Heap;
        }

        /// Destructor
        ~AlignedVector()
        {
            if (data_)
            {
                // Destroy elements in reverse order
                for (size_t i = size_; i > 0; --i)
                {
                    data_[i - 1].~T();
                }
                release_raw(data_, allocation_bytes_, storage_kind_);
            }
        }

        // ========== Assignment ==========

        /// Copy assignment
        AlignedVector &operator=(const AlignedVector &other)
        {
            if (this != &other)
            {
                AlignedVector tmp(other);
                swap(tmp);
            }
            return *this;
        }

        /// Move assignment
        AlignedVector &operator=(AlignedVector &&other) noexcept
        {
            if (this != &other)
            {
                AlignedVector tmp(std::move(other));
                swap(tmp);
            }
            return *this;
        }

        // ========== Capacity ==========

        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        /// Reserve capacity (does not change size)
        void reserve(size_t new_capacity)
        {
            if (new_capacity <= capacity_)
                return;

            size_t new_allocation_bytes = 0;
            T *new_data = allocate_raw(
                new_capacity,
                std::max(ALIGNMENT, allocation_alignment_),
                storage_kind_,
                new_allocation_bytes);

            if (data_)
            {
                // Move existing elements
                std::uninitialized_copy_n(data_, size_, new_data);

                // Destroy old elements
                for (size_t i = 0; i < size_; ++i)
                {
                    data_[i].~T();
                }
                release_raw(data_, allocation_bytes_, storage_kind_);
            }

            data_ = new_data;
            capacity_ = new_capacity;
            allocation_bytes_ = new_allocation_bytes;
        }

        /**
         * @brief Reserve storage with an explicit minimum address alignment.
         *
         * NUMA receive buffers must begin on a page boundary even when their
         * logical payload is smaller than one page.  Ordinary vectors retain
         * the cache-line default; infrastructure with a stronger ownership
         * contract opts in through this method.  Existing elements are copied
         * if the current allocation does not satisfy @p minimum_alignment.
         *
         * @param new_capacity Minimum element capacity after the call.
         * @param minimum_alignment Power-of-two byte alignment, at least 64.
         * @throws std::invalid_argument For a non-power-of-two alignment.
         */
        void reserve_aligned(
            size_t new_capacity,
            size_t minimum_alignment)
        {
            minimum_alignment = std::max(minimum_alignment, ALIGNMENT);
            if ((minimum_alignment & (minimum_alignment - 1)) != 0)
            {
                throw std::invalid_argument(
                    "AlignedVector minimum alignment must be a power of two");
            }
            const bool current_alignment_satisfies =
                !data_ ||
                (reinterpret_cast<uintptr_t>(data_) % minimum_alignment) == 0;
            if (new_capacity <= capacity_ && current_alignment_satisfies)
                return;

            const size_t target_capacity =
                std::max(new_capacity, capacity_);
            size_t new_allocation_bytes = 0;
            T *new_data = allocate_raw(
                target_capacity,
                minimum_alignment,
                storage_kind_,
                new_allocation_bytes);
            if (data_)
            {
                std::uninitialized_copy_n(data_, size_, new_data);
                for (size_t i = 0; i < size_; ++i)
                    data_[i].~T();
                release_raw(data_, allocation_bytes_, storage_kind_);
            }
            data_ = new_data;
            capacity_ = target_capacity;
            allocation_bytes_ = new_allocation_bytes;
            allocation_alignment_ = minimum_alignment;
        }

        /**
         * @brief Create uninitialized storage in one dedicated page mapping.
         *
         * The returned vector has ordinary value semantics, but its complete
         * capacity is released with `munmap` rather than `free`.  This is the
         * required owner for buffers whose placement will subsequently be
         * established and certified by NUMA first touch.
         * Virtual guards isolate each mapping from neighbouring NUMA owners;
         * they are retained and retired with the vector but are not payload RAM.
         *
         * @param element_count Logical number of elements in the mapping.
         * @return A page-aligned vector whose elements remain uninitialized.
         * @throws std::bad_alloc If the mapping cannot be created.
         * @throws std::runtime_error On platforms without anonymous mappings.
         */
        [[nodiscard]] static AlignedVector pageMappedUninitialized(
            size_t element_count)
        {
            AlignedVector result;
            result.storage_kind_ = StorageKind::AnonymousPageMapping;
            if (element_count != 0)
            {
                result.reserve_aligned(
                    element_count, runtimePageSize());
                result.size_ = element_count;
            }
            return result;
        }

        /// Resize vector (may allocate/deallocate)
        void resize(size_t new_size)
        {
            if (new_size > capacity_)
            {
                // Need to reallocate
                reserve(new_size);
            }

            if (new_size > size_)
            {
                // Default-initialize new elements
                std::uninitialized_fill_n(data_ + size_, new_size - size_, T{});
            }
            else if (new_size < size_)
            {
                // Destroy excess elements
                for (size_t i = new_size; i < size_; ++i)
                {
                    data_[i].~T();
                }
            }

            size_ = new_size;
        }

        /// Resize with fill value
        void resize(size_t new_size, const T &value)
        {
            if (new_size > capacity_)
            {
                reserve(new_size);
            }

            if (new_size > size_)
            {
                std::uninitialized_fill_n(data_ + size_, new_size - size_, value);
            }
            else if (new_size < size_)
            {
                for (size_t i = new_size; i < size_; ++i)
                {
                    data_[i].~T();
                }
            }

            size_ = new_size;
        }

        /// Resize without initializing new elements (NUMA-friendly).
        /// On multi-socket systems, avoids binding all pages to one NUMA node
        /// via single-threaded zero-init. Caller MUST write all elements before
        /// reading (e.g., via an OMP parallel first-touch loop).
        void resize_uninitialized(size_t new_size)
        {
            if (new_size > capacity_)
            {
                reserve(new_size);
            }

            if (new_size < size_)
            {
                for (size_t i = new_size; i < size_; ++i)
                {
                    data_[i].~T();
                }
            }

            size_ = new_size;
        }

        /**
         * @brief Resize without first-touch using an explicit allocation alignment.
         *
         * This combines @ref reserve_aligned with the uninitialized growth
         * semantics required by DMA, MPI, and NUMA-bound destination buffers.
         * The caller must overwrite every new element before reading it.
         */
        void resize_uninitialized_aligned(
            size_t new_size,
            size_t minimum_alignment)
        {
            reserve_aligned(new_size, minimum_alignment);
            if (new_size < size_)
            {
                for (size_t i = new_size; i < size_; ++i)
                    data_[i].~T();
            }
            size_ = new_size;
        }

        /// Clear all elements
        void clear()
        {
            for (size_t i = 0; i < size_; ++i)
            {
                data_[i].~T();
            }
            size_ = 0;
        }

        /**
         * @brief Release capacity when the vector no longer owns live data.
         *
         * Large packed-weight staging arrays call this after their temporary
         * representation has been consumed.  Matching `std::vector` semantics
         * keeps aligned storage usable for those arrays without preserving a
         * second, unexpectedly resident copy of model weights.
         */
        void shrink_to_fit()
        {
            if (size_ == capacity_)
                return;

            if (size_ == 0)
            {
                AlignedVector empty;
                swap(empty);
                return;
            }

            AlignedVector compact;
            compact.resize_uninitialized(size_);
            std::uninitialized_copy_n(data_, size_, compact.data_);
            swap(compact);
        }

        // ========== Element Access ==========

        T &operator[](size_t i) { return data_[i]; }
        const T &operator[](size_t i) const { return data_[i]; }

        T &at(size_t i)
        {
            if (i >= size_)
                throw std::out_of_range("AlignedVector::at: index out of range");
            return data_[i];
        }

        const T &at(size_t i) const
        {
            if (i >= size_)
                throw std::out_of_range("AlignedVector::at: index out of range");
            return data_[i];
        }

        T &front() { return data_[0]; }
        const T &front() const { return data_[0]; }

        T &back() { return data_[size_ - 1]; }
        const T &back() const { return data_[size_ - 1]; }

        T *data() { return data_; }
        const T *data() const { return data_; }

        // ========== Iterators ==========

        T *begin() { return data_; }
        const T *begin() const { return data_; }
        const T *cbegin() const { return data_; }

        T *end() { return data_ + size_; }
        const T *end() const { return data_ + size_; }
        const T *cend() const { return data_ + size_; }

        // ========== Modifiers ==========

        void push_back(const T &value)
        {
            if (size_ >= capacity_)
            {
                reserve(capacity_ == 0 ? 16 : capacity_ * 2);
            }
            new (data_ + size_) T(value);
            ++size_;
        }

        void push_back(T &&value)
        {
            if (size_ >= capacity_)
            {
                reserve(capacity_ == 0 ? 16 : capacity_ * 2);
            }
            new (data_ + size_) T(std::move(value));
            ++size_;
        }

        void pop_back()
        {
            if (size_ > 0)
            {
                --size_;
                data_[size_].~T();
            }
        }

        void swap(AlignedVector &other) noexcept
        {
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
            std::swap(capacity_, other.capacity_);
            std::swap(allocation_bytes_, other.allocation_bytes_);
            std::swap(allocation_alignment_, other.allocation_alignment_);
            std::swap(storage_kind_, other.storage_kind_);
        }

        // ========== Alignment Query ==========

        /// Check if data pointer is properly aligned
        bool is_aligned() const
        {
            return (reinterpret_cast<uintptr_t>(data_) % ALIGNMENT) == 0;
        }

        /// Get alignment of data pointer
        size_t alignment() const { return ALIGNMENT; }

        /** @return The allocation authority that owns the current capacity. */
        [[nodiscard]] StorageKind storageKind() const noexcept
        {
            return storage_kind_;
        }

        /** @return Page-rounded bytes owned by the active allocation. */
        [[nodiscard]] size_t allocationBytes() const noexcept
        {
            return allocation_bytes_;
        }

        /**
         * @brief Resolve the exact physical byte extent for a future allocation.
         *
         * Memory admission calls this same arithmetic before construction and
         * the allocator calls it again while materializing storage. Keeping
         * page promotion, caller alignment, and overflow handling here prevents
         * subsystem planners from maintaining approximate copies of the heap
         * or anonymous-mapping contract.
         *
         * @param element_count Number of `T` elements to own.
         * @param minimum_alignment Minimum power-of-two address alignment.
         * @param storage_kind Heap or dedicated anonymous mapping authority.
         * @return Exact byte extent later passed to `free` or `munmap`.
         * @throws std::invalid_argument for an invalid alignment/storage pair.
         * @throws std::overflow_error when element or rounding arithmetic wraps.
         */
        [[nodiscard]] static size_t requiredAllocationBytes(
            size_t element_count,
            size_t minimum_alignment = ALIGNMENT,
            StorageKind storage_kind = StorageKind::Heap)
        {
            if (element_count == 0u)
                return 0u;
            if (element_count >
                std::numeric_limits<size_t>::max() / sizeof(T))
            {
                throw std::overflow_error(
                    "AlignedVector element byte count overflows size_t");
            }
            const size_t logical_bytes = element_count * sizeof(T);
            const size_t allocation_alignment = effectiveAllocationAlignment(
                logical_bytes, minimum_alignment, storage_kind);
            if (logical_bytes >
                std::numeric_limits<size_t>::max() -
                    (allocation_alignment - 1u))
            {
                throw std::overflow_error(
                    "AlignedVector allocation rounding overflows size_t");
            }
            return (logical_bytes + allocation_alignment - 1u) &
                   ~(allocation_alignment - 1u);
        }

        /** @return Whether the current address satisfies @p byte_alignment. */
        bool is_aligned_to(size_t byte_alignment) const
        {
            return byte_alignment != 0 &&
                   (byte_alignment & (byte_alignment - 1)) == 0 &&
                   (reinterpret_cast<uintptr_t>(data_) % byte_alignment) == 0;
        }

    private:
        T *data_;
        size_t size_;
        size_t capacity_;
        size_t allocation_bytes_ = 0;
        size_t allocation_alignment_ = ALIGNMENT;
        StorageKind storage_kind_ = StorageKind::Heap;

        /** @return The platform page size required by anonymous mappings. */
        static size_t runtimePageSize()
        {
#ifdef __linux__
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0)
                throw std::runtime_error(
                    "AlignedVector could not resolve the runtime page size");
            return static_cast<size_t>(page_size);
#else
            throw std::runtime_error(
                "AlignedVector anonymous page mappings require Linux");
#endif
        }

        /**
         * @brief Resolve address alignment from allocator and payload geometry.
         *
         * Heap allocations at least one page large are page-aligned so NUMA
         * tooling never rounds into a neighboring allocation. Anonymous
         * mappings inherently begin on a runtime page boundary and cannot
         * promise a stronger alignment without an over-allocation owner.
         */
        static size_t effectiveAllocationAlignment(
            size_t logical_bytes,
            size_t minimum_alignment,
            StorageKind storage_kind)
        {
            minimum_alignment = std::max(minimum_alignment, ALIGNMENT);
            if ((minimum_alignment & (minimum_alignment - 1u)) != 0u)
            {
                throw std::invalid_argument(
                    "AlignedVector minimum alignment must be a power of two");
            }

            size_t allocation_alignment = minimum_alignment;
#ifdef __linux__
            const size_t page_size = runtimePageSize();
            if (logical_bytes >= page_size)
            {
                allocation_alignment = std::max(
                    allocation_alignment, page_size);
            }
            if (storage_kind == StorageKind::AnonymousPageMapping)
            {
                if (allocation_alignment > page_size)
                {
                    throw std::invalid_argument(
                        "AlignedVector anonymous mapping alignment exceeds the runtime page size");
                }
                allocation_alignment = page_size;
            }
#else
            if (storage_kind == StorageKind::AnonymousPageMapping)
            {
                throw std::runtime_error(
                    "AlignedVector anonymous page mappings require Linux");
            }
#endif
            return allocation_alignment;
        }

        /**
         * @brief Allocate raw storage using the vector's exact owner kind.
         * @param n Element capacity.
         * @param minimum_alignment Required power-of-two address alignment.
         * @param storage_kind Heap or dedicated anonymous mapping authority.
         * @param allocation_bytes Receives physical payload capacity, excluding virtual guards.
         * @return Uninitialized aligned storage.
         */
        static T *allocate_raw(
            size_t n,
            size_t minimum_alignment,
            StorageKind storage_kind,
            size_t &allocation_bytes)
        {
            if (n == 0)
                return nullptr;

            /* Admission and allocation deliberately share these two helpers. */
            const size_t aligned_bytes = requiredAllocationBytes(
                n, minimum_alignment, storage_kind);
            const size_t logical_bytes = n * sizeof(T);
            const size_t allocation_alignment = effectiveAllocationAlignment(
                logical_bytes, minimum_alignment, storage_kind);

            void *ptr = nullptr;
            if (storage_kind == StorageKind::Heap)
            {
                ptr = std::aligned_alloc(
                    allocation_alignment, aligned_bytes);
                if (!ptr)
                    throw std::bad_alloc();
            }
            else
            {
#ifdef __linux__
                // Guard VMAs prevent adjacent independent NUMA owners from
                // merging into one transparent huge page. Without guards, a
                // neighbour's first touch can place our pages on its node and
                // its MADV_DONTNEED can split a huge PTE during certification.
                // Align large payloads to preserve huge-page coverage, but bill
                // only page-rounded payload bytes: guards have no physical RAM.
                const size_t page = runtimePageSize();
                constexpr size_t huge_page = 2 * 1024 * 1024;
                const size_t mapping_alignment = aligned_bytes >= huge_page ? huge_page : page;
                if (aligned_bytes > std::numeric_limits<size_t>::max() - mapping_alignment - 2 * page)
                    throw std::bad_alloc();
                const size_t reserved_bytes = aligned_bytes + mapping_alignment + 2 * page;
                void *reservation = ::mmap(
                    nullptr,
                    reserved_bytes,
                    PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0);
                if (reservation == MAP_FAILED)
                    throw std::bad_alloc();
                const uintptr_t base = reinterpret_cast<uintptr_t>(reservation);
                const uintptr_t payload = (base + page + mapping_alignment - 1) &
                    ~(static_cast<uintptr_t>(mapping_alignment) - 1);
                ptr = reinterpret_cast<void *>(payload);
                if (::mprotect(ptr, aligned_bytes, PROT_READ | PROT_WRITE) != 0)
                {
                    ::munmap(reservation, reserved_bytes);
                    throw std::bad_alloc();
                }
                // Keep exactly one inaccessible page on either side. Their
                // extent is derived on release, so moves need no extra owner.
                const size_t leading = payload - page - base;
                const uintptr_t end = payload + aligned_bytes + page;
                const size_t trailing = base + reserved_bytes - end;
                if ((leading && ::munmap(reservation, leading) != 0) ||
                    (trailing && ::munmap(reinterpret_cast<void *>(end), trailing) != 0))
                    std::terminate();
#else
                (void)aligned_bytes;
                throw std::runtime_error(
                    "AlignedVector anonymous page mappings require Linux");
#endif
            }
            allocation_bytes = aligned_bytes;

#ifdef __linux__
            // Request transparent huge pages for large allocations (>2MB).
            // Reduces TLB pressure for multi-GB weight buffers.
            if (aligned_bytes >= 2 * 1024 * 1024)
            {
                madvise(ptr, aligned_bytes, MADV_HUGEPAGE);
            }
#endif

            return static_cast<T *>(ptr);
        }

        /**
         * @brief Release storage through the authority that created it.
         * @param ptr Allocation base.
         * @param allocation_bytes Physical payload capacity; anonymous owners also release both guards.
         * @param storage_kind Heap or anonymous-mapping owner.
         */
        static void release_raw(
            T *ptr,
            size_t allocation_bytes,
            StorageKind storage_kind) noexcept
        {
            if (!ptr)
                return;
            if (storage_kind == StorageKind::Heap)
            {
                std::free(ptr);
                return;
            }
#ifdef __linux__
            const size_t page = runtimePageSize();
            // The mapped owner includes two inaccessible virtual guards;
            // allocation_bytes continues to describe physical payload capacity.
            auto *mapping = reinterpret_cast<std::uint8_t *>(ptr) - page;
            if (allocation_bytes == 0 ||
                ::munmap(mapping, allocation_bytes + 2 * page) != 0)
            {
                std::terminate();
            }
#else
            (void)allocation_bytes;
            std::terminate();
#endif
        }

        /** @brief Allocate ordinary default-kind storage for constructors. */
        void allocate(size_t n)
        {
            data_ = allocate_raw(
                n,
                ALIGNMENT,
                storage_kind_,
                allocation_bytes_);
            allocation_alignment_ = ALIGNMENT;
        }
    };

} // namespace llaminar2
