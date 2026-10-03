/**
 * @file DeviceRequestRowRanges.h
 * @brief Borrowed live prefixes in a captured, request-major row matrix.
 *
 * Each request owns a fixed physical row stride and one device-authored live
 * length. This value describes those immutable coordinates; it never allocates
 * storage, shadows lengths on the host, or changes pointers between replays.
 * It also represents an explicitly full matrix without a mutable count owner.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_REQUEST_ROWS_HD __host__ __device__
#else
#define LLAMINAR_REQUEST_ROWS_HD
#endif

namespace llaminar2
{
    /**
     * @brief Checked request strides and an optional borrowed length authority.
     *
     * The existing producer must publish all lengths before a consumer runs.
     * Graph identity includes the owner pointer and the immutable geometry.
     * Request-local holes are not one flattened prefix: an empty first request
     * does not suppress a live second request or move that request's row base.
     */
    class DeviceRequestRowRanges final
    {
    public:
        /** @brief Admit fixed-width rows without introducing mutable metadata.
         * @param rows Positive complete matrix row count.
         * @return One full request with exactly this physical stride.
         * @throws std::invalid_argument For nonpositive rows. */
        [[nodiscard]] static DeviceRequestRowRanges fullyActive(int rows)
        {
            return bind(1, rows, nullptr);
        }

        /** @brief Bind request-local live prefixes without reading device data.
         * @param requests Positive request count.
         * @param rows_per_request Positive captured request stride.
         * @param lengths Stable device INT32 array with one length per request.
         * @return A trivially copyable, immutable launch descriptor.
         * @throws std::invalid_argument For absent metadata or invalid geometry. */
        [[nodiscard]] static DeviceRequestRowRanges deviceCounted(
            int requests, int rows_per_request, const std::int32_t *lengths)
        {
            if (!lengths)
                throw std::invalid_argument("Device request rows require a length authority");
            return bind(requests, rows_per_request, lengths);
        }

        /** @return Frozen number of disjoint request intervals. */
        [[nodiscard]] LLAMINAR_REQUEST_ROWS_HD constexpr int requests() const noexcept
        { return requests_; }

        /** @return Physical request stride, not its current live length. */
        [[nodiscard]] LLAMINAR_REQUEST_ROWS_HD constexpr int rowsPerRequest() const noexcept
        { return rows_per_request_; }

        /** @return Admitted complete matrix row count; multiplication is checked at construction. */
        [[nodiscard]] LLAMINAR_REQUEST_ROWS_HD constexpr int physicalRows() const noexcept
        { return requests_ * rows_per_request_; }

        /** @return Borrowed length-array identity, or null for explicitly full work. */
        [[nodiscard]] constexpr const std::int32_t *lengthOwner() const noexcept
        { return lengths_; }

        /** @brief Validate one publication without accessing its device owner.
         * @param request Request coordinate in the frozen interval set.
         * @param published Live row count in [0, rowsPerRequest()].
         * @return The exact length, or -1 for corruption; never clamps. */
        [[nodiscard]] LLAMINAR_REQUEST_ROWS_HD constexpr int activeRowsFor(
            int request, int published) const noexcept
        {
            return request >= 0 && request < requests_ && published >= 0 &&
                published <= rows_per_request_ ? published : -1;
        }

#if defined(__CUDACC__) || defined(__HIPCC__)
        /** @brief Read one already-ordered request length on device.
         * @param request Workgroup-uniform request coordinate.
         * @return Exact live length; malformed metadata traps before row access. */
        [[nodiscard]] __device__ inline __attribute__((always_inline)) int activeRows(int request) const
        {
            if (request < 0 || request >= requests_)
            {
#if defined(__CUDA_ARCH__)
                __trap();
#else
                __builtin_trap();
#endif
            }
            int published = lengths_ ? lengths_[request] : rows_per_request_;
#if defined(__HIP_DEVICE_COMPILE__)
            published = __builtin_amdgcn_readfirstlane(published);
#endif
            const int active = activeRowsFor(request, published);
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
        /** @brief Reject malformed immutable coordinates before graph recording. */
        [[nodiscard]] static DeviceRequestRowRanges bind(
            int requests, int rows_per_request, const std::int32_t *lengths)
        {
            if (requests <= 0 || rows_per_request <= 0 ||
                requests > std::numeric_limits<int>::max() / rows_per_request)
                throw std::invalid_argument("Device request rows require positive, representable strides");
            return DeviceRequestRowRanges(requests, rows_per_request, lengths);
        }

        /** @brief Only checked factories construct launch geometry. */
        constexpr DeviceRequestRowRanges(int requests, int rows_per_request,
            const std::int32_t *lengths) noexcept
            : lengths_(lengths), requests_(requests), rows_per_request_(rows_per_request) {}

        const std::int32_t *lengths_; ///< Borrowed producer-owned array; no host read is permitted.
        int requests_; ///< Frozen request count.
        int rows_per_request_; ///< Stable row stride for every input/output plane.
    };
    static_assert(std::is_trivially_copyable_v<DeviceRequestRowRanges>);
    static_assert(std::is_standard_layout_v<DeviceRequestRowRanges>);
}
#undef LLAMINAR_REQUEST_ROWS_HD
