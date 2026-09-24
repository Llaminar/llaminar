/**
 * @file RoPEWorkspaceContract.h
 * @brief Canonical physical workspace ABI for graph-captured GPU RoPE.
 *
 * CUDA, ROCm, and metadata-only admission consume this header so the stable
 * position, immutable inverse-frequency, and device-parameter publications
 * have one byte-accounting authority.
 */

#pragma once

#include "RoPEDeviceParams.h"
#include "RoPEInvariantPublication.h"
#include "interfaces/IWorkspaceConsumer.h"

#include <cstddef>
#include <stdexcept>

namespace llaminar2::rope_workspace
{
    /** Typed graph-family geometry for one RoPE workspace declaration. */
    struct Geometry
    {
        int graph_rows = 0; ///< Maximum position-ID rows retained by the graph.
    };

    /**
     * @brief Build the exact graph-stable RoPE workspace descriptors.
     * @param geometry Maximum retained graph-row geometry.
     * @return Mergeable requirements shared by both GPU backends.
     * @throws std::invalid_argument when graph_rows is non-positive.
     */
    inline WorkspaceRequirements requirements(const Geometry &geometry)
    {
        if (geometry.graph_rows <= 0)
        {
            throw std::invalid_argument(
                "RoPE workspace requires a positive graph-row capacity");
        }

        WorkspaceRequirements result;
        result.buffers.emplace_back(
            RoPEWorkspaceBuffers::POSITION_IDS,
            static_cast<std::size_t>(geometry.graph_rows) * sizeof(int),
            256u,
            true);
        result.buffers.emplace_back(
            RoPEWorkspaceBuffers::INV_FREQ,
            rope::kInvariantPublicationSlots *
                static_cast<std::size_t>(
                    rope::kMaxInverseFrequencyValues) *
                sizeof(float),
            256u,
            true,
            WorkspaceExecutionRegime::Any,
            WorkspaceContentLifetime::SerialGraphFamily);
        result.buffers.emplace_back(
            RoPEWorkspaceBuffers::DEVICE_PARAMS,
            sizeof(rope::RoPEDeviceParams),
            256u,
            true);
        return result;
    }
} // namespace llaminar2::rope_workspace
