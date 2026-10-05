/**
 * @file NativeAllreduceRequestRows.h
 * @brief Checked request-strided banks with independent device-owned live prefixes.
 *
 * Ragged requests occupy fixed arena banks, not one contiguous live prefix.
 * This immutable declaration borrows each request's published row count and
 * lowers its bank independently to the existing native row collective. Counts
 * stay on device; storage capacity never becomes a communication extent.
 */
#pragma once

#include "NativeCollectiveRows.h"

namespace llaminar2
{
    /**
     * @brief Request-major allreduce geometry without host count shadows.
     *
     * All participants retain the same request order and bank capacities. The
     * graph producer publishes identical counts before the collective stage,
     * and owns their storage through completion of every bank operation.
     */
    class NativeAllreduceRequestRows final
    {
    public:
        /**
         * @brief Freeze complete request banks and their count authorities.
         * @param requests Positive number of request banks.
         * @param rows_per_request Admitted physical row stride of each bank.
         * @param elements_per_row Exact scalar width of a row.
         * @param live_counts Stable device INT32 array with one count per request.
         * @throws std::invalid_argument For missing, misaligned or empty geometry.
         * @throws std::overflow_error For unrepresentable storage or count extents.
         */
        NativeAllreduceRequestRows(int requests, int rows_per_request,
            std::size_t elements_per_row, const std::int32_t *live_counts)
            : requests_(requests), first_bank_(
                DeviceRowRange::deviceCounted(rows_per_request, live_counts), elements_per_row)
        {
            const auto address = reinterpret_cast<std::uintptr_t>(live_counts);
            if (requests <= 0 || address % alignof(std::int32_t))
                throw std::invalid_argument("Request-row allreduce requires positive banks and aligned device counts");
            const auto maximum = std::numeric_limits<std::size_t>::max();
            if (first_bank_.bankElements() > maximum / static_cast<std::size_t>(requests) ||
                static_cast<std::size_t>(requests) >
                    (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(std::int32_t))
                throw std::overflow_error("Request-row allreduce storage or count array overflows");
        }

        /** @return Number of independently counted request banks. */
        [[nodiscard]] int requests() const noexcept { return requests_; }
        /** @return Retained elements in one request bank, never its live payload. */
        [[nodiscard]] std::size_t bankElements() const noexcept { return first_bank_.bankElements(); }
        /** @return Complete request-major storage extent. */
        [[nodiscard]] std::size_t storageElements() const noexcept
        { return static_cast<std::size_t>(requests_) * bankElements(); }
        /** @return Borrowed device count array; no host accessor reads its values. */
        [[nodiscard]] const std::int32_t *countOwners() const noexcept
        { return first_bank_.rows().countOwner(); }

        /**
         * @brief Observe aggregate useful bytes across every request bank.
         * @param counter Aligned device UINT64 retained through graph retirement.
         * @return A new immutable declaration sharing the existing row authorities.
         * @throws std::invalid_argument For a missing or misaligned receipt.
         */
        [[nodiscard]] NativeAllreduceRequestRows withPayloadReceipt(unsigned long long *counter) const
        {
            auto observed = *this;
            observed.first_bank_ = first_bank_.withPayloadReceipt(counter);
            return observed;
        }

        /**
         * @brief Obtain the exact prefix authority for one request.
         * @param request Request-major bank index.
         * @return Native prefix relative to that bank's own storage origin.
         * @throws std::out_of_range For an index outside the declaration.
         */
        [[nodiscard]] NativeCollectiveRows requestRows(int request) const
        {
            requireRequest(request);
            const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(
                first_bank_.rows().capacity(), countOwners() + request), first_bank_.elementsPerRow());
            return first_bank_.payloadReceipt() ? rows.withPayloadReceipt(first_bank_.payloadReceipt()) : rows;
        }

        /**
         * @brief Resolve a checked scalar offset within the owning tensor.
         * @param request Request-major bank index.
         * @return Immutable bank origin, independent of its current live count.
         * @throws std::out_of_range For an index outside the declaration.
         */
        [[nodiscard]] std::size_t requestElementOffset(int request) const
        {
            requireRequest(request);
            return static_cast<std::size_t>(request) * bankElements();
        }

    private:
        /**
         * @brief Reject invalid bank indices before deriving pointers or offsets.
         * @param request Request-major bank index to validate.
         * @throws std::out_of_range If the declaration does not own that bank.
         */
        void requireRequest(int request) const
        {
            if (request < 0 || request >= requests_)
                throw std::out_of_range("Request-row allreduce bank index exceeds its declaration");
        }

        int requests_; ///< Immutable request-major geometry, not mutable execution state.
        NativeCollectiveRows first_bank_; ///< Validated stride and borrowed count-array origin.
    };
}
