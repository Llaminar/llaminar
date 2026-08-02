/**
 * @file MTPRequestTerminalPublicationGraph.h
 * @brief Declarative policy and stable layout for MTP request-terminal state.
 *
 * Request-batched prefill flattens every request into a fixed-width physical
 * row.  The real request lengths and that physical row width form one logical
 * geometry record: consumers must never observe one without the other.  This
 * header gives graph definitions a typed publication policy and gives runtime
 * graph machinery one compact, model-lifetime layout for the corresponding
 * device-owned metadata.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Declares how a model publishes terminal hidden rows after prefill.
     *
     * This is architecture policy, not a runtime capability advertisement.
     * GPU graph construction must reject `Unspecified`; it may not invent an
     * ownership model or rebuild a prompt-specific selector during execution.
     */
    enum class MTPRequestTerminalHiddenPublicationPolicy
    {
        Unspecified,

        /**
         * One graph-captured selector exists for every legal request count.
         * Each selector reads the current real lengths and padded row stride
         * from one event-published device geometry record.  Consequently the
         * graph family is total over every positive prompt width that fits the
         * runner's declared activation capacity.
         */
        GraphCapturedDeviceGeometry,
    };

    /**
     * @brief Declares how shifted-MTP prefill obtains main-model hidden rows.
     *
     * Shifted prefill consumes row @c i of the main model together with token
     * @c i+1.  A host-authored row cursor is not an acceptable owner for that
     * relationship on a GPU: it can become stale relative to the graph which
     * advanced either KV cache.  Architecture builders therefore select an
     * explicit device-owned lowering here, and graph machinery materializes
     * the complete bounded graph family before request execution begins.
     */
    enum class MTPShiftedPrefillHiddenPublicationPolicy
    {
        Unspecified,

        /**
         * Derive the next contiguous source range from the canonical main and
         * shifted-MTP KV counts plus the resident request-batch geometry.  The
         * row count and request index are graph topology; prompt width and
         * current progress remain device data.  This produces one reusable
         * captured graph per (request index, grouped row count), with no H2D
         * row cursor and no runtime graph materialization.
         */
        GraphCapturedDeviceKVProgress,
    };

    /**
     * @brief Describes the persistent INT32 request-batch geometry record.
     *
     * The first `request_capacity` scalars are real sequence lengths.  The
     * final scalar is the padded physical row stride shared by every request.
     * Keeping the stride in the same allocation and H2D publication as the
     * lengths makes a mixed-generation geometry structurally impossible.
     *
     * The class performs pointer arithmetic only.  It owns no storage and is
     * safe to copy into graph/runtime objects.
     */
    class DeviceRequestBatchGeometryLayout final
    {
    public:
        constexpr DeviceRequestBatchGeometryLayout() = default;

        explicit constexpr DeviceRequestBatchGeometryLayout(
            int request_capacity)
            : request_capacity_(request_capacity)
        {
        }

        /** @return true when the layout can describe at least one request. */
        [[nodiscard]] constexpr bool valid() const
        {
            return request_capacity_ > 0;
        }

        /** @return Maximum number of request-length entries in the record. */
        [[nodiscard]] constexpr int requestCapacity() const
        {
            return request_capacity_;
        }

        /** @return Number of INT32 scalars required by the complete record. */
        [[nodiscard]] constexpr size_t scalarCount() const
        {
            return valid()
                       ? static_cast<size_t>(request_capacity_) + size_t{1}
                       : size_t{0};
        }

        /** @return Fixed scalar index containing the padded request row stride. */
        [[nodiscard]] constexpr size_t rowStrideIndex() const
        {
            return valid() ? static_cast<size_t>(request_capacity_) : size_t{0};
        }

        /**
         * @brief Resolve the first real-length scalar in a host or device record.
         */
        template <typename Scalar>
        [[nodiscard]] constexpr Scalar *sequenceLengths(Scalar *base) const
        {
            return valid() ? base : nullptr;
        }

        /**
         * @brief Resolve the immutable-address row-stride scalar.
         */
        template <typename Scalar>
        [[nodiscard]] constexpr Scalar *rowStride(Scalar *base) const
        {
            return valid() && base ? base + rowStrideIndex() : nullptr;
        }

    private:
        int request_capacity_ = 0;
    };
} // namespace llaminar2
