/**
 * @file KVCacheWorkspaceContract.h
 * @brief Canonical physical workspace contract for GPU KV-cache conversion.
 * @author Llaminar Team
 *
 * CUDA, ROCm, TurboQuant, and preflight admission all consume this contract.
 * It is the single arithmetic authority for the two graph-stable K/V
 * conversion buffers; callers provide representation-specific row widths but
 * may not independently reproduce the batch/context sizing policy.
 */

#pragma once

#include "KVCacheWorkspaceBuffers.h"
#include "../../execution/local_execution/device/WorkspaceDescriptor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2::kv_cache_workspace
{
    /**
     * @brief Typed geometry for the graph-stable K/V conversion pair.
     *
     * A positive @p requested_batch_size identifies the modern three-argument
     * workspace call. Otherwise @p requested_graph_rows retains the legacy
     * meaning of a batch hint. The configured cache horizon always remains the
     * lower bound, because a later graph replay may read more resident rows
     * than the graph bucket that first declared the workspace.
     */
    struct ConversionGeometry
    {
        int configured_batch_size = 0;
        int configured_context_rows = 0;
        int requested_graph_rows = 0;
        int requested_batch_size = 0;
        std::size_t conversion_row_bytes = 0u;
        std::size_t native_row_bytes = 0u;
    };

    /**
     * @brief Multiply two physical extents with an overflow diagnostic.
     * @param left First extent.
     * @param right Second extent.
     * @return Exact product.
     * @throws std::overflow_error if the product cannot fit in size_t.
     */
    inline std::size_t checkedProduct(
        std::size_t left,
        std::size_t right)
    {
        if (left != 0u &&
            right > std::numeric_limits<std::size_t>::max() / left)
        {
            throw std::overflow_error(
                "KV-cache conversion workspace size overflow");
        }
        return left * right;
    }

    /**
     * @brief Resolve the bytes required by each stable conversion buffer.
     * @param geometry Complete configured and call-site geometry.
     * @return Bytes required independently by K and by V.
     * @throws std::invalid_argument for incomplete configured geometry.
     *
     * The conversion and native widths are alternatives, not simultaneous
     * allocations: a single stable address is reused by every representation,
     * so the physical requirement is their maximum row width.
     */
    inline std::size_t conversionBufferBytes(
        const ConversionGeometry &geometry)
    {
        if (geometry.configured_batch_size <= 0 ||
            geometry.configured_context_rows <= 0 ||
            geometry.conversion_row_bytes == 0u)
        {
            throw std::invalid_argument(
                "KV-cache workspace requires positive configured geometry and conversion width");
        }

        const bool has_explicit_request = geometry.requested_batch_size > 0;
        const int requested_batch = has_explicit_request
                                        ? geometry.requested_batch_size
                                        : (geometry.requested_graph_rows > 0
                                               ? geometry.requested_graph_rows
                                               : geometry.configured_batch_size);
        const int requested_context = has_explicit_request
                                          ? std::max(
                                                geometry.requested_graph_rows,
                                                geometry.configured_context_rows)
                                          : geometry.configured_context_rows;
        const std::size_t bounded_batch = static_cast<std::size_t>(
            std::max({1, requested_batch, geometry.configured_batch_size}));
        const std::size_t bounded_context = static_cast<std::size_t>(
            std::max(1, requested_context));
        const std::size_t row_bytes = std::max(
            geometry.conversion_row_bytes,
            geometry.native_row_bytes);
        return checkedProduct(
            checkedProduct(bounded_batch, bounded_context), row_bytes);
    }

    /**
     * @brief Build the canonical pair of required K/V workspace descriptors.
     * @param geometry Complete configured and call-site geometry.
     * @return Stable-name requirements consumed by the interval planner.
     */
    inline WorkspaceRequirements conversionRequirements(
        const ConversionGeometry &geometry)
    {
        const std::size_t bytes = conversionBufferBytes(geometry);
        WorkspaceRequirements requirements;
        requirements.buffers.emplace_back(
            KVCacheWorkspaceBuffers::CONV_SCRATCH_K,
            bytes,
            256u,
            true);
        requirements.buffers.emplace_back(
            KVCacheWorkspaceBuffers::CONV_SCRATCH_V,
            bytes,
            256u,
            true);
        return requirements;
    }
} // namespace llaminar2::kv_cache_workspace
