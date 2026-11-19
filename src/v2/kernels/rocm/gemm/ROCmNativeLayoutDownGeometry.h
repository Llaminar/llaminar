/**
 * @file ROCmNativeLayoutDownGeometry.h
 * @brief Shared cold/device scheduler for wave-uniform native-layout down batches.
 *
 * Existing64-column CTA output ownership is independent of the wave-local
 * original-partition mapping. One immutable expert packet belongs to each wave;
 * only original p<maximum tasks execute, so padding adds no numerical work.
 */
#pragma once
#include "kernels/common/ROCmCompiledWaveContract.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define LLAMINAR_DOWN_HD __host__ __device__
#define LLAMINAR_DOWN_DEVICE_INLINE __forceinline__
#else
#define LLAMINAR_DOWN_HD
#define LLAMINAR_DOWN_DEVICE_INLINE
#endif
namespace llaminar2::rocm {
/**
 * @brief Resolve an original interval start, using blocks as the inactive sentinel.
 * @param blocks Original positive K/32, at most INT_MAX/32.
 * @param width Original ceil(blocks/parts), with original parts in1..64.
 * @param part Original partition index, below parts when participating.
 * @param participating Original row/format/partition eligibility.
 * @return Clamped original start or the nonparticipating blocks sentinel.
 * @pre The original source policy owns width; no alternative partitioning is chosen.
 */
LLAMINAR_DOWN_HD constexpr unsigned nativeLayoutIntervalFirst(
    unsigned blocks,unsigned width,unsigned part,bool participating) noexcept
{
    const unsigned first=participating?part*width:blocks;
    return first<blocks?first:blocks;
}
/**
 * @brief Original capped end, used only while constructing the union envelope.
 * @param blocks Original K/32. @param first Start or inactive blocks sentinel.
 * @param width Original immutable partition width.
 * @return min(blocks,first+width), without overflow at the sentinel.
 */
LLAMINAR_DOWN_HD constexpr unsigned nativeLayoutIntervalEnd(
    unsigned blocks,unsigned first,unsigned width) noexcept
{return first+(width<blocks-first?width:blocks-first);}
/**
 * @brief Test one original interval without retaining a lane-dependent end.
 * @param block Current union cursor, strictly below original blocks.
 * @param first Clamped start or inactive blocks sentinel. @param width Original width.
 * @return True exactly for first<=block<min(blocks,first+width).
 * @pre blocks<=INT_MAX/32, width<=blocks. For block<first, unsigned subtraction
 * exceeds width; first==blocks therefore rejects every inactive block. No lane
 * eligibility mask or changed K order is introduced.
 */
LLAMINAR_DOWN_HD constexpr bool nativeLayoutIntervalContains(
    unsigned block,unsigned first,unsigned width) noexcept
{return block-first<width;}
/** @brief Contiguous output ownership; no column belongs to two resident CTAs. */
struct NativeLayoutDownRowRange {
    unsigned begin,end;
};
/** @brief Existing64-column CTA ownership and fixed metadata capacity, without per-packet scheduling. */
struct NativeLayoutDownGeometry {
    static constexpr unsigned columns=64; ///< Adjacent native-pack output columns per partition team.
    static constexpr unsigned waves=1024/rocmOrderedWaveWidth; ///< Original resident CTA wave-task stride.
    static constexpr unsigned groups=1024/columns; ///< Physical partition teams in the admitted CTA.
    static constexpr unsigned maximum_capacity=4*groups; ///< Largest EB4 descriptor/route table.
    /**
     * @brief First shared-local item belonging to the same original physical wave.
     * @param wave Original wave task index, strictly below waves.
     * @param routed_tasks Number of routed items preceding the shared task interval.
     * @return Smallest shared-local index s with (routed_tasks+s)%waves==wave.
     * @details The original mixed loop visits wave+k*waves in increasing order.
     * Separating its two contiguous intervals preserves that exact per-wave order;
     * shared items start at this index and retain the same waves stride. Empty
     * shared intervals simply execute no iteration. This does not resolve a source
     * partition policy or assign different wave ownership.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE static constexpr unsigned firstSharedTask(
        unsigned wave,unsigned routed_tasks) noexcept {
        return (wave+waves-routed_tasks%waves)%waves;
    }
    /**
     * @brief Balance contiguous output columns over the actual admitted resident grid.
     * @param rows Original positive output width (zero is an empty diagnostic).
     * @param cta Actual resident CTA index, strictly below resident_grid.
     * @param resident_grid Actual positive admitted grid; never a fixed GPU size.
     * @return Disjoint contiguous slice, differing from another slice by at most one row.
     * @details Quotient/remainder distribution avoids rows*cta overflow and
     * a64-bit device divide. Column reassignment changes no per-column P/K/route
     * arithmetic. The first remainder CTAs own one extra row.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE static constexpr NativeLayoutDownRowRange rows(
        unsigned rows,unsigned cta,unsigned resident_grid) noexcept {
        if(!resident_grid||cta>=resident_grid)return {0,0};
        const unsigned quotient=rows/resident_grid,remainder=rows%resident_grid;
        const unsigned begin=cta*quotient+(cta<remainder?cta:remainder);
        return {begin,begin+quotient+(cta<remainder?1u:0u)};
    }

};

/**
 * @brief Wave-owned original partitions with the widest legal adjacent columns.
 * @details One ordered wave remains inside one EB4 packet or shared expert.
 * The original policy supplies maximum P; this mapping never changes P itself.
 * At most16 workers mirror the fork K-lane scope. Analytical helpers cover
 * original P up to64; the production source contract admits P<=16 and uses one
 * register bank. No extra partition bank is executed. Tensor offsets retain size_t.
 */
struct NativeLayoutDownWaveGeometry {
    unsigned workers; ///< Original-P workers per wave, at most16.
    unsigned columns; ///< Adjacent native columns per wave, exactly64/workers.
    /**
     * @brief Select the smallest power-of-two worker group covering original P, capped16.
     * @param maximum Original admitted source-policy maximum in1..64.
     * @return Exact worker/column dimensions, or zero fields for invalid P.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE static constexpr NativeLayoutDownWaveGeometry resolve(unsigned maximum) noexcept {
        if(!maximum||maximum>64)return {0,0};
        unsigned workers=1;
        while(workers<maximum&&workers<16)workers*=2;
        return {workers,64/workers};
    }
    /**
     * @brief Bit position of the resolved power-of-two adjacent-column extent.
     * @return Exact log2(columns), in2..6.
     * @pre This geometry was returned by resolve with a valid original maximum.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE constexpr unsigned columnShift() const noexcept {
        return __builtin_ctz(columns);
    }
    /** @brief Lane-local adjacent column; no output/P arithmetic is broadcast.
     * @param lane Original physical lane in0..63. @return lane modulo columns.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE constexpr unsigned laneColumn(unsigned lane) const noexcept {
        return lane&(columns-1u);
    }
    /** @brief Lane-local original partition worker.
     * @param lane Original physical lane in0..63. @return lane divided by columns.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE constexpr unsigned laneWorker(unsigned lane) const noexcept {
        return lane>>columnShift();
    }
    /** @brief One original packet/output-tile coordinate, without additional stored geometry. */
    struct RoutedTask {
        unsigned first_route; ///< First original route in the EB4 packet.
        unsigned first_column; ///< First relative output column in this wave tile.
    };
    /**
     * @brief Decode the original packet-major task using bounded subtraction.
     * @param task Live routed item, below output_tiles*ceil(topK/4).
     * @param output_tiles Positive tiles(outputs), with outputs in1..64.
     * @return Exactly {4*(task/output_tiles),(task%output_tiles)*columns}.
     * @pre Original topK is in1..16, so at most three subtractions execute.
     * @details Task order and wave stride are unchanged. One decode supplies both
     * coordinates, avoiding general quotient/remainder state across the decoder.
     */
    LLAMINAR_DOWN_HD LLAMINAR_DOWN_DEVICE_INLINE constexpr RoutedTask routedTask(
        unsigned task,unsigned output_tiles) const noexcept {
        unsigned packet=0;
        while(task>=output_tiles){task-=output_tiles;++packet;}
        return {packet<<2,task<<columnShift()};
    }
    /** @brief Number of independent output tiles in one existing64-column CTA tile.
     * @param outputs Actual owned columns, at most64. @return Number of live wave tiles.
     */
    LLAMINAR_DOWN_HD constexpr unsigned tiles(unsigned outputs) const noexcept {
        return (outputs+columns-1u)>>columnShift();
    }
    /** @brief Live EB4/output items for this exact original geometry.
     * @param outputs Actual owned columns<=64. @param routes Original topK<=16.
     * @return Number of useful independent routed wave items, without padding.
     */
    LLAMINAR_DOWN_HD constexpr unsigned routedTasks(unsigned outputs,unsigned routes) const noexcept {
        return tiles(outputs)*((routes+3u)/4u);
    }
    /** @brief First original route owned by a routed wave item.
     * @param task Live routed item. @param output_tiles Positive tiles(outputs).
     * @return First slot of this original EB4 packet; its tail retains route<topK.
     */
    LLAMINAR_DOWN_HD constexpr unsigned firstRoute(unsigned task,unsigned output_tiles) const noexcept {
        return routedTask(task,output_tiles).first_route;
    }
    /** @brief First relative output column owned by a routed wave item.
     * @param task Live routed item. @param output_tiles Positive tiles(outputs).
     * @return Column within the existing64-column CTA tile.
     */
    LLAMINAR_DOWN_HD constexpr unsigned routedFirstColumn(unsigned task,unsigned output_tiles) const noexcept {
        return routedTask(task,output_tiles).first_column;
    }
    /** @brief First relative output column owned by a shared wave item.
     * @param task Live shared item. @param routed_tasks Exact preceding routed item count.
     * @return Column within the existing64-column CTA tile.
     */
    LLAMINAR_DOWN_HD constexpr unsigned sharedFirstColumn(unsigned task,unsigned routed_tasks) const noexcept {
        return (task-routed_tasks)<<columnShift();
    }
    /** @brief Source lane of one original p for this output column.
     * @param part Original zero-based partition in0..63.
     * @param column Column within this wave's adjacent output span.
     * @return Physical lane in0..63; its bank is part/workers.
     */
    LLAMINAR_DOWN_HD constexpr unsigned sourceLane(unsigned part,unsigned column) const noexcept {
        return ((part&(workers-1u))<<columnShift())|column;
    }
};
}
#undef LLAMINAR_DOWN_HD
#undef LLAMINAR_DOWN_DEVICE_INLINE
