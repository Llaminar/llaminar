#pragma once

/**
 * @file CUDAMoERouterPrefillPolicy.h
 * @brief Typed CUDA launch policy for the FP32 MoE prefill router GEMM.
 *
 * Router prefill computes `logits[M, E] = hidden[M, K] * gate[E, K]^T`.
 * The arithmetic order inside one output element is deliberately identical for
 * every geometry in this file; only the number of rows and expert columns
 * assigned to one thread block changes.  Keeping the candidate inventory typed
 * lets the performance harness measure the exact production implementations
 * without environment-variable overrides or test-only kernels.
 *
 * All launch bridges require an explicit non-null CUDA stream.  They allocate
 * no memory, perform no transfers, and introduce no synchronization, making
 * them suitable for inclusion in the captured production graph.
 */

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Compiled FP32 router-prefill tile geometries.
     *
     * The names encode the output tile dimensions and, where more than one K
     * strip is compiled for the same output tile, the explicit K width.
     * Changing M/N/K tiling changes grid-level parallelism, operand reuse, and
     * barrier count while preserving each accumulator's increasing-K order.
     */
    enum class CUDAMoERouterPrefillGeometry : uint8_t
    {
        Tile64x64,
        Tile64x32,
        Tile32x64,
        Tile32x32,
        Tile32x32K16,
        Tile32x24,
        Tile24x32,
        Tile64x16,
        Tile32x16,
        Tile16x64,
        Tile16x32,
        Tile16x16,
        Tile8x64,
        Tile8x32,
    };

    /** @brief Human-readable and launch-time facts for one compiled geometry. */
    struct CUDAMoERouterPrefillGeometrySpec
    {
        const char *name;
        int tile_rows;
        int tile_experts;
        int tile_k;
        int rows_per_thread;
        int experts_per_thread;
        int threads_per_block;
    };

    /**
     * @brief Return the immutable specification for a compiled geometry.
     *
     * @param geometry Candidate whose template dimensions are requested.
     * @return A complete launch specification with a stable diagnostic name.
     */
    [[nodiscard]] constexpr CUDAMoERouterPrefillGeometrySpec
    cudaMoERouterPrefillGeometrySpec(
        CUDAMoERouterPrefillGeometry geometry) noexcept
    {
        using Geometry = CUDAMoERouterPrefillGeometry;
        switch (geometry)
        {
        case Geometry::Tile64x64:
            return {"tile64x64", 64, 64, 16, 4, 4, 256};
        case Geometry::Tile64x32:
            return {"tile64x32", 64, 32, 16, 4, 2, 256};
        case Geometry::Tile32x64:
            return {"tile32x64", 32, 64, 16, 2, 4, 256};
        case Geometry::Tile32x32:
            return {"tile32x32", 32, 32, 32, 2, 2, 256};
        case Geometry::Tile32x32K16:
            return {"tile32x32_k16", 32, 32, 16, 2, 2, 256};
        case Geometry::Tile32x24:
            return {"tile32x24", 32, 24, 16, 1, 3, 256};
        case Geometry::Tile24x32:
            return {"tile24x32", 24, 32, 16, 3, 1, 256};
        case Geometry::Tile64x16:
            return {"tile64x16", 64, 16, 16, 4, 1, 256};
        case Geometry::Tile32x16:
            return {"tile32x16", 32, 16, 16, 2, 1, 256};
        case Geometry::Tile16x64:
            return {"tile16x64", 16, 64, 16, 1, 4, 256};
        case Geometry::Tile16x32:
            return {"tile16x32", 16, 32, 16, 1, 2, 256};
        case Geometry::Tile16x16:
            return {"tile16x16", 16, 16, 16, 1, 1, 256};
        case Geometry::Tile8x64:
            return {"tile8x64", 8, 64, 16, 1, 2, 256};
        case Geometry::Tile8x32:
            return {"tile8x32", 8, 32, 16, 1, 1, 256};
        }
        return {"invalid", 0, 0, 0, 0, 0, 0};
    }

    /**
     * @brief Select the stable production geometry embedded during graph capture.
     *
     * The selector is intentionally pure and host-side: it runs while graph
     * topology is constructed, never during graph replay.  Qwen3.6 35B uses
     * exact bucket overlays certified by the production-shape tournament.
     * Other geometries retain the established generic 64x64 tile until they
     * receive equivalent evidence; every positive M still has a valid launch.
     */
    [[nodiscard]] constexpr CUDAMoERouterPrefillGeometry
    selectCUDAMoERouterPrefillGeometry(
        int seq_len,
        int d_model,
        int num_experts) noexcept
    {
        if (d_model == 2048 && num_experts == 256)
        {
            if (seq_len <= 64)
                return CUDAMoERouterPrefillGeometry::Tile16x16;
            if (seq_len <= 128)
                return CUDAMoERouterPrefillGeometry::Tile32x16;
            if (seq_len <= 256)
                return CUDAMoERouterPrefillGeometry::Tile32x32;
            if (seq_len <= 384)
                return CUDAMoERouterPrefillGeometry::Tile64x32;
            if (seq_len <= 640)
                return CUDAMoERouterPrefillGeometry::Tile32x32K16;
            if (seq_len <= 1536)
                return CUDAMoERouterPrefillGeometry::Tile64x32;
        }
        return CUDAMoERouterPrefillGeometry::Tile64x64;
    }
}

/**
 * @brief Launch one exact compiled FP32 router-prefill candidate.
 *
 * @return `true` when argument validation and CUDA launch submission succeed.
 */
extern "C" bool cudaMoE_route_logits_with_geometry(
    const float *hidden,
    const float *gate_weights,
    float *logits,
    int seq_len,
    int d_model,
    int num_experts,
    int device_idx,
    void *stream,
    llaminar2::CUDAMoERouterPrefillGeometry geometry);

/**
 * @brief Query compiler and occupancy evidence for one exact candidate.
 *
 * This setup-only query performs no launch, allocation, transfer, or
 * synchronization.  `local_memory_bytes_per_thread == 0` is the pre-timing
 * spill gate used by the performance tournament.
 */
extern "C" bool cudaMoE_route_logits_query_geometry_resources(
    llaminar2::CUDAMoERouterPrefillGeometry geometry,
    int *registers_per_thread,
    std::size_t *local_memory_bytes_per_thread,
    std::size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm);
