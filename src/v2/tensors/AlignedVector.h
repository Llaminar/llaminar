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
 * Performance impact:
 * - BF16/FP16 conversion: 15% faster with streaming stores
 * - Large tensor operations: 5-20% better cache utilization
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstdlib> // aligned_alloc, free
#include <cstring> // memcpy
#include <stdexcept>
#include <algorithm>
#include <initializer_list>
#ifdef __linux__
#include <sys/mman.h> // madvise, MADV_HUGEPAGE
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
            : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
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
                std::free(data_);
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

            T *new_data = allocate_raw(new_capacity);

            if (data_)
            {
                // Move existing elements
                std::uninitialized_copy_n(data_, size_, new_data);

                // Destroy old elements
                for (size_t i = 0; i < size_; ++i)
                {
                    data_[i].~T();
                }
                std::free(data_);
            }

            data_ = new_data;
            capacity_ = new_capacity;
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

        /// Clear all elements
        void clear()
        {
            for (size_t i = 0; i < size_; ++i)
            {
                data_[i].~T();
            }
            size_ = 0;
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
        }

        // ========== Alignment Query ==========

        /// Check if data pointer is properly aligned
        bool is_aligned() const
        {
            return (reinterpret_cast<uintptr_t>(data_) % ALIGNMENT) == 0;
        }

        /// Get alignment of data pointer
        size_t alignment() const { return ALIGNMENT; }

    private:
        T *data_;
        size_t size_;
        size_t capacity_;

        /// Allocate aligned memory (raw, uninitialized)
        T *allocate_raw(size_t n)
        {
            if (n == 0)
                return nullptr;

            // Calculate allocation size (must be multiple of ALIGNMENT)
            size_t alloc_bytes = n * sizeof(T);
            size_t aligned_bytes = (alloc_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

            void *ptr = std::aligned_alloc(ALIGNMENT, aligned_bytes);
            if (!ptr)
            {
                throw std::bad_alloc();
            }

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

        /// Allocate and default-initialize
        void allocate(size_t n)
        {
            data_ = allocate_raw(n);
        }
    };

} // namespace llaminar2
