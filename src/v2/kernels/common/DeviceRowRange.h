/**
 * @file DeviceRowRange.h
 * @brief Captured physical row geometry with a borrowed device-owned live count.
 *
 * A retained kernel has fixed grid dimensions and scratch strides, while an MTP
 * controller may publish a different number of live rows on every replay.
 * This descriptor separates those facts without allocating memory or copying
 * the count to the host. Slices keep the original count authority and their
 * physical width; only arithmetic admission changes at execution time.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ROW_RANGE_HD __host__ __device__
#else
#define LLAMINAR_ROW_RANGE_HD
#endif

namespace llaminar2
{
    /**
     * @brief Immutable launch descriptor for a contiguous tile of logical rows.
     *
     * The caller owns the published INT32 count and must order its producer
     * before every consumer, including side-stream projections. Captured graph
     * identity must include this pointer and the lifetime of its owner. The
     * pointer is never dereferenced by a host factory or slice operation.
     */
    class DeviceRowRange final
    {
    public:
        /**
         * @brief Describe a fixed-width operation in which every row is live.
         * @param capacity Positive physical matrix row count.
         * @throws std::invalid_argument If capacity is not positive.
         */
        [[nodiscard]] static DeviceRowRange fullyActive(int capacity)
        {
            if (capacity <= 0)
                throw std::invalid_argument("DeviceRowRange requires positive capacity");
            return DeviceRowRange(nullptr, capacity, 0, capacity);
        }

        /**
         * @brief Borrow the controller's count without creating a host shadow.
         * @param capacity Maximum rows admitted by this captured matrix.
         * @param count Stable device INT32 in the inclusive range [0, capacity].
         * @throws std::invalid_argument For a null owner or invalid capacity.
         */
        [[nodiscard]] static DeviceRowRange deviceCounted(
            int capacity, const std::int32_t *count)
        {
            if (capacity <= 0 || !count)
                throw std::invalid_argument("DeviceRowRange requires a count owner and positive capacity");
            return DeviceRowRange(count, capacity, 0, capacity);
        }

        /**
         * @brief Form a nonempty physical tile while retaining the same owner.
         * @param first First row relative to this descriptor's physical tile.
         * @param rows Physical tile width, never the current live width.
         * @throws std::out_of_range If the slice is empty or exceeds the tile.
         */
        [[nodiscard]] DeviceRowRange slice(int first, int rows) const
        {
            if (first < 0 || rows <= 0 || first > rows_ || rows > rows_ - first)
                throw std::out_of_range("DeviceRowRange slice exceeds physical rows");
            return DeviceRowRange(count_, capacity_, first_ + first, rows);
        }

        /** @return Fixed row stride for this tile's split-K scratch layout. */
        [[nodiscard]] LLAMINAR_ROW_RANGE_HD constexpr int physicalRows() const noexcept
        {
            return rows_;
        }

        /** @return Original matrix capacity against which publications are checked. */
        [[nodiscard]] LLAMINAR_ROW_RANGE_HD constexpr int capacity() const noexcept
        {
            return capacity_;
        }

        /** @return Stable borrowed count pointer, or null for fixed-width work. */
        [[nodiscard]] constexpr const std::int32_t *countOwner() const noexcept
        {
            return count_;
        }

        /**
         * @brief Resolve a published count without changing physical geometry.
         * @param published Whole-matrix count, not a tile-relative count.
         * @return Live prefix within this tile, or -1 for a corrupt publication.
         *
         * The pure arithmetic is shared with device-free adversarial tests.
         * Device callers must treat -1 as fatal, not clamp an invalid count.
         */
        [[nodiscard]] LLAMINAR_ROW_RANGE_HD constexpr int activeRowsFor(
            int published) const noexcept
        {
            if (published < 0 || published > capacity_)
                return -1;
            if (published <= first_)
                return 0;
            const int available = published - first_;
            return available < rows_ ? available : rows_;
        }

#if defined(__CUDACC__) || defined(__HIPCC__)
        /**
         * @brief Read the device authority after the producer's ordered edge.
         * @return Live tile rows; invalid publications trap before memory access.
         *
         * This is device-only intentionally: host graph construction must not
         * infer mutable execution geometry by reading the controller's state.
         */
        [[nodiscard]] __device__ __forceinline__ int activeRows() const
        {
            int published = count_ ? *count_ : capacity_;
#if defined(__HIP_DEVICE_COMPILE__)
            // This descriptor and its immutable-per-launch publication are
            // wave-uniform kernel operands. State that fact explicitly: a
            // generic-pointer load otherwise makes LLVM carry divergent EXEC
            // masks through every row/format arm, spilling scalar registers.
            // This is a register broadcast, not another count authority or a
            // communication/synchronization operation.
            published = __builtin_amdgcn_readfirstlane(published);
#endif
            const int active = activeRowsFor(published);
            if (active < 0)
            {
#if defined(__CUDA_ARCH__)
                __trap();
#else
                __builtin_trap();
#endif
            }
            return active;
        }
#endif

    private:
        /** @brief Construct validated metadata; factories own all host checks. */
        constexpr DeviceRowRange(const std::int32_t *count, int capacity,
                                 int first, int rows) noexcept
            : count_(count), capacity_(capacity), first_(first), rows_(rows) {}

        const std::int32_t *count_; ///< Borrowed immutable address, not a host value.
        int capacity_; ///< Original admitted matrix width.
        int first_; ///< Tile offset in the original matrix.
        int rows_; ///< Physical tile width retained across every replay.
    };

    static_assert(std::is_trivially_copyable_v<DeviceRowRange>);
    static_assert(std::is_standard_layout_v<DeviceRowRange>);
}

#undef LLAMINAR_ROW_RANGE_HD
