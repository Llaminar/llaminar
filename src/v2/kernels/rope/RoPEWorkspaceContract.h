/**
 * @file RoPEWorkspaceContract.h
 * @brief Canonical physical workspace ABI for GPU and CPU RoPE.
 *
 * CUDA, ROCm, and metadata-only admission consume this header so the stable
 * position, immutable inverse-frequency, and device-parameter publications
 * have one byte-accounting authority. The CPU contract prices the one
 * `cpu_rope_angles` buffer through which CPU RoPE borrows its angle planes;
 * the runtime kernel and the memory estimator call the same function.
 */

#pragma once

#include "RoPEDeviceParams.h"
#include "RoPEInvariantPublication.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/cpu/primitives/RoPEAngleRows.h"

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

namespace llaminar2::cpu_rope_workspace
{
    /** Typed graph geometry for the CPU RoPE angle planes. */
    struct Geometry
    {
        int graph_rows = 0; ///< Maximum rows one graph invocation rotates.
        int head_dim = 0;   ///< Physical head width; rotated pairs never exceed head_dim / 2.
    };

    /**
     * @brief Build the exact CPU RoPE angle workspace descriptor.
     * @param geometry Maximum graph rows and physical head width.
     * @return One named buffer, merged by name across the graph's RoPE stages.
     * @throws std::invalid_argument On nonpositive or odd geometry.
     *
     * The extent depends only on graph rows and head geometry, never on the
     * worker count: one team fills one shared set of planes. Partial RoPE uses
     * a prefix of the head-width capacity, so runtime and admission agree.
     */
    inline WorkspaceRequirements requirements(const Geometry &geometry)
    {
        if (geometry.graph_rows <= 0 || geometry.head_dim <= 0 || geometry.head_dim % 2 != 0)
            throw std::invalid_argument("CPU RoPE workspace requires positive rows and an even head width");
        WorkspaceRequirements result;
        result.buffers.emplace_back(
            RoPEWorkspaceBuffers::CPU_ANGLES,
            primitives::rope_angle_layout(geometry.graph_rows, geometry.head_dim / 2).bytes,
            primitives::kRoPEAnglePlaneAlignment,
            true);
        return result;
    }
} // namespace llaminar2::cpu_rope_workspace
