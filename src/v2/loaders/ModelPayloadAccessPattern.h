/**
 * @file ModelPayloadAccessPattern.h
 * @brief Typed GGUF payload access policy for model-context construction.
 *
 * Model metadata is always parsed, but tensor payloads have materially
 * different lifetimes.  A dense CPU graph reads mapped weights throughout
 * inference, a GPU graph consumes the mapping only while staging weights, and
 * an ExpertOverlay endpoint materializes a sparse set of expert slices.  This
 * enum keeps those ownership contracts explicit so mmap residency policy is
 * not inferred from an incidental primary device.
 */

#pragma once

#include "WeightIdentity.h"

namespace llaminar2
{

    /**
     * @brief Describes how a model context will consume GGUF tensor payloads.
     */
    enum class ModelPayloadAccessPattern
    {
        /** Final populated weights remain CPU-resident; their source mappings stay demand-paged. */
        DenseCpuResident,

        /** Payload pages are transient staging input for device preparation. */
        DeviceStaging,

        /** Only explicitly selected tensor or expert slices are materialized. */
        SparseSelection,
    };

    /**
     * @brief Resolve source mapping density from its immutable semantic ownership.
     * @param requested Consumer's existing dense, staging or selected access policy.
     * @param role One source role authenticated by the parsed tensor inventory.
     * @return Selected mapping for a separately owned row table; otherwise requested.
     *
     * A CPU graph may populate ordinary final weights while its GGUF also
     * contains a separately admitted RAM or bounded-mmap row table. Source-wide
     * cache warmup must not read that independent table before its owner exists.
     * The resolver retains this warmup distinction; low-level source mappings
     * are always demand-paged, independent of final consumer placement.
     */
    [[nodiscard]] constexpr ModelPayloadAccessPattern resolveModelPayloadAccessPattern(
        ModelPayloadAccessPattern requested, WeightRole role) noexcept
    {
        return requested == ModelPayloadAccessPattern::DenseCpuResident &&
                       role == WeightRole::ImmutableRowTable
                   ? ModelPayloadAccessPattern::SparseSelection
                   : requested;
    }

    /**
     * @brief Return a stable diagnostic name for a payload access pattern.
     * @param pattern Access pattern to describe.
     * @return Process-lifetime string literal.
     */
    [[nodiscard]] constexpr const char *toString(
        ModelPayloadAccessPattern pattern) noexcept
    {
        switch (pattern)
        {
        case ModelPayloadAccessPattern::DenseCpuResident:
            return "dense_cpu_resident";
        case ModelPayloadAccessPattern::DeviceStaging:
            return "device_staging";
        case ModelPayloadAccessPattern::SparseSelection:
            return "sparse_selection";
        }
        return "unknown";
    }

} // namespace llaminar2
