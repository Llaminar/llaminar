/**
 * @file CPUNativeVNNIPrefillSchedule.h
 * @brief Total, device-free worksharing policy for ordinary CPU prefill.
 *
 * Scheduling assigns independent output tiles to workers; it never partitions
 * their K reductions. A column-only schedule reuses weights well, but its last
 * wave can leave most of a socket idle. Use the existing row-pair grid when its
 * bounded per-worker load is substantially smaller, without changing packed
 * formats, accumulator order, scratch ownership, or the decode/MTP policies.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include "CPUNativeVNNIPreparedFootprint.h"

namespace llaminar2::cpu::native_vnni
{
    /**
     * @brief Scheduling intent, independent of the projection's arithmetic.
     *
     * Both purposes retain serial-row byte equivalence. Decode includes grouped
     * speculative verification and uses its certified learned policy. Prefill
     * may amortize expensive compact-weight decoding across wider physical row
     * tiles; row count alone cannot distinguish these two callers.
     */
    enum class ProjectionRowsPurpose
    {
        Decode,  ///< Serial generation or grouped speculative verification.
        Prefill, ///< Ordinary prompt processing, not MTP verification.
    };

    /**
     * @brief Ordinary-prefill schedule, with explicit candidates for diagnostics.
     *
     * Production uses Auto. Forced values authenticate physical candidates in
     * the tuning harness and byte-exact integration tests; they do not select a
     * different arithmetic contract.
     */
    enum class PrefillSchedulePolicy
    {
        Auto,           ///< Resolve the geometry-aware production policy.
        RowChunkGrid,   ///< One independent row and 64-column chunk per task.
        TwoRowNMajor,   ///< One column block per task, visiting all row pairs.
        TwoRowPairGrid, ///< Cartesian product of row pairs and column blocks.
        FourRowGrid,    ///< AVX-512 four-row weight reuse, with exact row tails.
    };

    /** @brief Physical row kernels admitted by the active, not merely built, ISA. */
    enum class PrefillRowKernelSet
    {
        Pairwise, ///< AVX2/scalar policy cannot select an AVX-512 wide tile.
        WideRows, ///< AVX-512 offers both two-row and four-row kernels.
    };

    /** @brief Immutable dimensions and encoding of the admitted projection. */
    struct PrefillScheduleGeometry
    {
        int rows = 0;           ///< Positive runtime M, not bucket capacity.
        int columns = 0;        ///< Logical N, including a partial final chunk.
        int row_stride = 0;     ///< Output stride in floats, at least N.
        int n_block_chunks = 0; ///< Positive number of 64-column chunks per task.
        int workers = 0;        ///< Actual invocation's positive worker budget.
        CPUNativeVNNIEncoding encoding = CPUNativeVNNIEncoding::ExpandedInt8;
        PrefillRowKernelSet row_kernels = PrefillRowKernelSet::Pairwise;
    };

    /**
     * @brief Resolve independent-tile ownership without changing arithmetic.
     * @param requested Auto or one explicitly measured diagnostic schedule.
     * @param geometry Logical output geometry and active worker budget.
     * @return A concrete schedule, never Auto.
     * @throws std::invalid_argument For invalid geometry or an unknown policy.
     *
     * Keep the existing single-row grid for very narrow projections. Otherwise
     * compare the maximum number of row-pair/column tiles assigned to one
     * worker. Requiring at least a 5:4 critical-path load ratio
     * amortizes the pair grid's weaker cross-row weight locality. At least four
     * row pairs amortize grid setup: smaller measured batches retain column
     * reuse even when their idealized tile count predicts a gain. Full waves
     * and single workers likewise retain column reuse.
     * Long grid launches with native dual-scale/compact multi-scale weights
     * amortize their expensive bitplane/grid decoding over four rows. The
     * all-format comparison does not justify widening nibble/expanded INT8
     * encodings. Require 128 rows and four complete worker waves so smaller
     * batches retain the two-row kernel's greater column-level parallelism.
     * All intermediates are widened before rounding or multiplication.
     */
    [[nodiscard]] constexpr PrefillSchedulePolicy resolvePrefillSchedule(
        PrefillSchedulePolicy requested,
        const PrefillScheduleGeometry &geometry)
    {
        if (geometry.rows <= 0 || geometry.columns <= 0 ||
            geometry.row_stride < geometry.columns ||
            geometry.n_block_chunks <= 0 || geometry.workers <= 0 ||
            !isKnownPreparedEncoding(geometry.encoding) ||
            (geometry.row_kernels != PrefillRowKernelSet::Pairwise &&
             geometry.row_kernels != PrefillRowKernelSet::WideRows))
            throw std::invalid_argument("CPU prefill schedule requires valid output geometry and workers");

        switch (requested)
        {
        case PrefillSchedulePolicy::RowChunkGrid:
        case PrefillSchedulePolicy::TwoRowNMajor:
        case PrefillSchedulePolicy::TwoRowPairGrid:
            return requested;
        case PrefillSchedulePolicy::FourRowGrid:
            if (geometry.rows < 3 || geometry.row_kernels != PrefillRowKernelSet::WideRows)
                throw std::invalid_argument("Four-row prefill needs the wide-row ISA and M >= 3");
            return requested;
        case PrefillSchedulePolicy::Auto:
            break;
        default:
            throw std::invalid_argument("Unknown CPU prefill schedule");
        }

        const std::int64_t chunks = (std::int64_t{geometry.columns} + 63) / 64;
        const std::int64_t blocks =
            (chunks + geometry.n_block_chunks - 1) / geometry.n_block_chunks;
        if (geometry.row_stride < 64 ||
            (geometry.rows >= 2 && blocks <= geometry.workers / 4))
            return PrefillSchedulePolicy::RowChunkGrid;

        const std::int64_t pairs = (std::int64_t{geometry.rows} + 1) / 2;
        const std::int64_t n_major_load =
            pairs * ((blocks + geometry.workers - 1) / geometry.workers);
        const std::int64_t pair_grid_load =
            (pairs * blocks + geometry.workers - 1) / geometry.workers;
        if (pairs < 4 || n_major_load * 4 < pair_grid_load * 5)
            return PrefillSchedulePolicy::TwoRowNMajor;
        const bool costly_decode =
            geometry.encoding == CPUNativeVNNIEncoding::Q6KNativeDualScale ||
            geometry.encoding == CPUNativeVNNIEncoding::CompactMultiScale;
        const std::int64_t four_row_tasks =
            ((std::int64_t{geometry.rows} + 3) / 4) * blocks;
        return costly_decode && geometry.row_kernels == PrefillRowKernelSet::WideRows &&
                       geometry.rows >= 128 && four_row_tasks >= std::int64_t{geometry.workers} * 4
                   ? PrefillSchedulePolicy::FourRowGrid
                   : PrefillSchedulePolicy::TwoRowPairGrid;
    }
}
