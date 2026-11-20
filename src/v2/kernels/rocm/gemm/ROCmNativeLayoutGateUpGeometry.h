/**
 * @file ROCmNativeLayoutGateUpGeometry.h
 * @brief Exact wave tasks and immutable role-record ownership for native gate/up.
 *
 * The existing original-P mapping supplies adjacent output columns and complete
 * independent partition workers. A physical wave owns one selected expert and
 * one output tile; a CTA publishes that expert's existing two role records only
 * when one of its waves consumes them. No partition policy or hardware count is
 * inferred here. Wide counters cover every positive signed32 source axis.
 */
#pragma once
#include "ROCmNativeLayoutDownGeometry.h"
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define LLAMINAR_GATE_UP_HD __host__ __device__
#define LLAMINAR_GATE_UP_INLINE __forceinline__
#else
#define LLAMINAR_GATE_UP_HD
#define LLAMINAR_GATE_UP_INLINE
#endif
namespace llaminar2::rocm {
/** @brief Pure original source/task ownership; no retained scheduler state. */
struct NativeLayoutGateUpGeometry {
    using Index=std::uint64_t; ///< Task products may exceed32bits; tensor axes remain original unsigned values.
    static constexpr unsigned waves=NativeLayoutDownGeometry::waves; ///< Existing1024-thread wave64 CTA.
    /** @brief One original selected route and adjacent output interval. */
    struct Task {
        unsigned route; ///< Original selected slot, independent of physical expert ID.
        unsigned first_column; ///< First column of this wave-owned output tile.
    };
    /**
     * @brief Count wave-owned output tiles without rounding a source axis down.
     * @param width Original role N, or zero for empty analytical geometry.
     * @param columns Adjacent columns from the original-P wave mapping.
     * @return Exact ceil(width/columns), or zero for absent columns.
     */
    LLAMINAR_GATE_UP_HD static constexpr Index tiles(unsigned width,unsigned columns) noexcept {
        return columns?(Index(width)+columns-1u)/columns:0;
    }
    /**
     * @brief Count selected-expert/output items without32-bit product overflow.
     * @param width Original output extent. @param routes Original selected slot count.
     * @param columns Adjacent columns from the original-P mapping.
     * @return Exact task count, including only original output tiles.
     */
    LLAMINAR_GATE_UP_HD static constexpr Index tasks(unsigned width,unsigned routes,unsigned columns) noexcept {
        return tiles(width,columns)*routes;
    }
    /**
     * @brief Balance only the original shared wave items over the admitted grid.
     * @param width Original shared output extent, including zero analytical work.
     * @param columns Original-P adjacent-column extent, always positive in production.
     * @param cta Actual CTA index. @param grid Actual positive admitted resident grid.
     * @return Contiguous original task interval; counts differ by at most one.
     * @details Reuses the existing overflow-safe row distributor. Since columns
     * is positive, ceil(width/columns)<=width for positive unsigned width, so
     * the task count narrows exactly. No padded wave item or K/P work is added.
     * Empty CTAs still execute the caller's original unconditional grid join.
     */
    LLAMINAR_GATE_UP_HD LLAMINAR_GATE_UP_INLINE static constexpr NativeLayoutDownRowRange sharedRange(
        unsigned width,unsigned columns,unsigned cta,unsigned grid) noexcept {
        return NativeLayoutDownGeometry::rows(static_cast<unsigned>(tiles(width,columns)),cta,grid);
    }
    /**
     * @brief First original wave item of a physical CTA.
     * @param cta Actual admitted block index. @param wave Physical wave within that CTA.
     * @return Global wave task index, without32-bit multiplication.
     */
    LLAMINAR_GATE_UP_HD static constexpr Index firstTask(unsigned cta,unsigned wave) noexcept {
        return Index(cta)*waves+wave;
    }
    /**
     * @brief Grid-stride for the original per-wave task sequence.
     * @param grid Actual positive resident grid.
     * @return Exact physical-wave stride, not a device-count guess.
     */
    LLAMINAR_GATE_UP_HD static constexpr Index stride(unsigned grid) noexcept {
        return Index(grid)*waves;
    }
    /**
     * @brief Decode a live expert-major item with the fork's subtract-and-advance strategy.
     * @param task Live original task, below output_tiles*routes.
     * @param output_tiles Positive tiles(width,columns).
     * @param columns Original adjacent-column extent.
     * @return Exactly {task/output_tiles,(task%output_tiles)*columns}.
     * @pre Original routes is at most the admitted topK, so at most topK-1 subtractions occur.
     */
    LLAMINAR_GATE_UP_HD LLAMINAR_GATE_UP_INLINE static constexpr Task coordinate(
        Index task,Index output_tiles,unsigned columns) noexcept {
        unsigned route=0;
        while(task>=output_tiles){task-=output_tiles;++route;}
        return {route,static_cast<unsigned>(task*columns)};
    }
    /**
     * @brief Prove that at least one original output item of a route belongs to this CTA.
     * @param route Original selected route. @param output_tiles Positive route tile count.
     * @param cta Actual physical block index, below grid. @param grid Actual positive grid.
     * @return True exactly when a wave of this CTA will consume that route's two records.
     * @details Whole groups of waves consecutive items belong to one CTA, cycling
     * through the grid. The first such group in the route interval is advanced
     * to the requested CTA residue. No output/P work or source-policy choice is
     * added. Invalid/empty analytical geometry returns false.
     */
    LLAMINAR_GATE_UP_HD LLAMINAR_GATE_UP_INLINE static constexpr bool ctaOwnsRoute(
        unsigned route,Index output_tiles,unsigned cta,unsigned grid) noexcept {
        if(!output_tiles||!grid||cta>=grid)return false;
        const Index first=Index(route)*output_tiles/waves;
        const Index last=(Index(route+1u)*output_tiles-1u)/waves;
        const Index advance=(Index(cta)+grid-first%grid)%grid;
        return first+advance<=last;
    }
    /**
     * @brief Number of useful CTA wave groups before physical residency admission.
     * @param items Original task count.
     * @return Exact ceil(items/waves), zero only for empty analytical work.
     */
    LLAMINAR_GATE_UP_HD static constexpr Index usefulCTAs(Index items) noexcept {
        return items/waves+(items%waves!=0);
    }
};
}
#undef LLAMINAR_GATE_UP_HD
#undef LLAMINAR_GATE_UP_INLINE
