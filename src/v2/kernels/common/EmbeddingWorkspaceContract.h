/**
 * @file EmbeddingWorkspaceContract.h
 * @brief Canonical physical workspace ABI for graph-captured GPU embedding.
 *
 * CUDA, ROCm, and metadata-only memory admission consume this contract so the
 * graph-stable token-ID publication has one owner for its name, geometry,
 * alignment, and byte calculation.
 */

#pragma once

#include "interfaces/IWorkspaceConsumer.h"

#include <cstddef>
#include <stdexcept>

namespace llaminar2::embedding_workspace
{
    /** Immutable geometry for one captured embedding workspace declaration. */
    struct Geometry
    {
        int graph_rows = 0; ///< Maximum token-ID rows retained by the graph.
    };

    /**
     * @brief Build the exact graph-stable embedding workspace descriptors.
     * @param geometry Maximum retained graph-row geometry.
     * @return Requirements shared by CUDA, ROCm, and memory admission.
     * @throws std::invalid_argument when graph_rows is non-positive.
     */
    inline WorkspaceRequirements requirements(const Geometry &geometry)
    {
        if (geometry.graph_rows <= 0)
        {
            throw std::invalid_argument(
                "Embedding workspace requires a positive graph-row capacity");
        }

        WorkspaceRequirements result;
        result.buffers.emplace_back(
            EmbeddingWorkspaceBuffers::TOKEN_IDS,
            static_cast<std::size_t>(geometry.graph_rows) * sizeof(int),
            256u,
            true);
        return result;
    }
} // namespace llaminar2::embedding_workspace
