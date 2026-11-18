/**
 * @file NativeTileCompletion.h
 * @brief Device-owned nonblocking tile publication with fixed original contributors.
 *
 * This is protocol metadata, never a floating reduction or a residency barrier.
 * Every original split-K CTA arrives once after its stores. Nonfinal CTAs retire;
 * only the unique final CTA publishes the unchanged ordered arithmetic, then
 * resets its counter before the enclosing kernel retires. Arena concurrency,
 * exact stream ordering and an immutable retained lease own reuse across calls.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define LLAMINAR_TILE_COMPLETION_HD __host__ __device__
#else
#define LLAMINAR_TILE_COMPLETION_HD
#endif
namespace llaminar2
{
/** @brief Explicit original publication policy; unsupported backend/mode selections are fatal. */
enum class MoEGateUpPublicationPolicy { Staged, LastProducer };
/** @brief Ordered weighted-route publication on the unchanged original split-K grid. */
enum class MoEDownPublicationPolicy { Staged, LastProducerRouteFold };
/** @brief Pure interpretation of one acquired integer completion ticket. */
enum class NativeTileCompletionDecision { Retire, Publish, Invalid };
/** @brief Borrowed exact device counter interval, with no host mirror of live arrivals. */
struct NativeTileCompletionView
{
    std::uint32_t* counters=nullptr;
    std::uint32_t tiles=0,routes=0,contributors=0;
};
/**
 * @brief Decode one acquire/release RMW ticket independently of scheduling order.
 * @param ticket Number of earlier completed producers observed by the device RMW.
 * @param contributors Exact original split-K policy's physical producer count.
 * @return Unique last-producer decision or fatal invalid protocol state.
 */
LLAMINAR_TILE_COMPLETION_HD constexpr NativeTileCompletionDecision nativeTileCompletionDecision(
    std::uint32_t ticket,std::uint32_t contributors) noexcept
{
    return !contributors||ticket>=contributors?NativeTileCompletionDecision::Invalid:
        ticket==contributors-1?NativeTileCompletionDecision::Publish:NativeTileCompletionDecision::Retire;
}
/** @brief Immutable overflow-checked physical tile/route/partition protocol shape. */
class NativeTileCompletionPlan final
{
public:
    /**
     * @param columns Positive original output width, including a possible32-column final tile.
     * @param routes Positive original route capacity, independent of device-owned valid IDs.
     * @param contributors Positive original source-policy maximum partition count.
     * @return Exact metadata geometry, without an arbitrary model/device cap.
     * @throws std::invalid_argument For absent geometry or an axis outside the native ABI.
     * @throws std::overflow_error If the exact counter byte interval exceeds size_t.
     */
    static NativeTileCompletionPlan make(std::uint64_t columns,std::uint64_t routes,std::uint64_t contributors)
    {
        if(!columns)throw std::invalid_argument("Native tile completion requires positive columns");
        return fromTileCount(columns/64+(columns%64!=0),routes,contributors);
    }
    /**
     * @brief Admit protocol counters for an already-derived work-sized physical tile extent.
     * @param tiles Positive physical tile count from the caller's immutable column geometry.
     * @param routes Original selected slot extent.
     * @param contributors Original source-policy physical split-K count, never rescaled by tile width.
     * @return Canonical checked counter shape, shared by64-column and paired128-column producers.
     * @throws std::invalid_argument For zero/native-ABI-incompatible axes.
     * @throws std::overflow_error For an unrepresentable named byte interval.
     */
    static NativeTileCompletionPlan fromTileCount(std::uint64_t tiles,std::uint64_t routes,std::uint64_t contributors)
    {
        if(!tiles||!routes||!contributors||tiles>std::numeric_limits<std::uint32_t>::max()||
            routes>std::numeric_limits<std::uint32_t>::max()||contributors>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Native tile completion requires positive native task geometry");
        if(routes>std::numeric_limits<std::size_t>::max()/tiles/sizeof(std::uint32_t))
            throw std::overflow_error("Native tile completion counter extent exceeds size_t");
        return NativeTileCompletionPlan(static_cast<std::uint32_t>(tiles),static_cast<std::uint32_t>(routes),
            static_cast<std::uint32_t>(contributors));
    }
    /** @return Number of physical publication tiles per route; caller geometry owns their width. */
    std::uint32_t tiles() const noexcept {return tiles_;}
    /** @return Fixed original route capacity, never an observed live count. */
    std::uint32_t routes() const noexcept {return routes_;}
    /** @return Exact split-K physical producers discharging each tile. */
    std::uint32_t contributors() const noexcept {return contributors_;}
    /** @return Exact counter count, with no launch padding. */
    std::size_t counters() const noexcept {return std::size_t(tiles_)*routes_;}
    /** @return Exact PMA-owned counter storage; placement/alignment belong to the existing arena. */
    std::size_t bytes() const noexcept {return counters()*sizeof(std::uint32_t);}
    /** @param counters Stable admitted interval. @return Borrowed device view; a null binding is rejected. */
    NativeTileCompletionView bind(std::uint32_t* counters) const
    {
        if(!counters)throw std::invalid_argument("Native tile completion requires admitted device counter storage");
        return {counters,tiles_,routes_,contributors_};
    }
private:
    /** @brief Seal an already validated native shape. */
    NativeTileCompletionPlan(std::uint32_t tiles,std::uint32_t routes,std::uint32_t contributors)
        :tiles_(tiles),routes_(routes),contributors_(contributors){}
    std::uint32_t tiles_,routes_,contributors_;
};
/** @brief Two independent epochs: partition completion per route, then route completion per output tile. */
struct NativeDownCompletionView
{
    NativeTileCompletionView partitions;
    std::uint32_t* routes=nullptr;
};
/** @brief Exact named counter geometry for the original down producer and canonical slot fold. */
class NativeDownCompletionPlan final
{
public:
    static constexpr std::uint32_t columns_per_tile=128; ///< Two original independent64-column tasks per physical CTA.
    /**
     * @param columns Original full output H, including the final partial128-column physical tile.
     * @param routes Original selected slot extent, including invalid/empty slot obligations.
     * @param contributors Original source-policy maximum physical split-K count.
     * @return Overflow-checked metadata for both nonspinning publication levels.
     * @throws std::invalid_argument For absent/native-ABI-incompatible geometry.
     * @throws std::overflow_error For an unrepresentable combined byte interval.
     */
    static NativeDownCompletionPlan make(std::uint64_t columns,std::uint64_t routes,std::uint64_t contributors)
    {
        if(!columns)throw std::invalid_argument("Native down completion requires positive columns");
        const auto tiles=columns/columns_per_tile+(columns%columns_per_tile!=0);
        auto partitions=NativeTileCompletionPlan::fromTileCount(tiles,routes,contributors);
        auto folds=NativeTileCompletionPlan::fromTileCount(tiles,1,routes);
        if(partitions.bytes()>std::numeric_limits<std::size_t>::max()-folds.bytes())
            throw std::overflow_error("Native down completion counter extent exceeds size_t");
        return NativeDownCompletionPlan(partitions,folds);
    }
    /** @return Original partition-level task/counter authority. */
    const NativeTileCompletionPlan& partitions() const noexcept {return partitions_;}
    /** @return Exact uint32 counter count, with one route-fold counter per output tile. */
    std::size_t counters() const noexcept {return partitions_.counters()+folds_.counters();}
    /** @return Exact PMA bytes; no independent allocation or reserve. */
    std::size_t bytes() const noexcept {return partitions_.bytes()+folds_.bytes();}
    /** @param counters Named retained arena interval. @return Both disjoint device-owned epochs. */
    NativeDownCompletionView bind(std::uint32_t* counters) const
    {return {partitions_.bind(counters),counters+partitions_.counters()};}
private:
    /** @brief Retain already checked immutable shapes. */
    NativeDownCompletionPlan(NativeTileCompletionPlan partitions,NativeTileCompletionPlan folds)
        :partitions_(partitions),folds_(folds){}
    NativeTileCompletionPlan partitions_,folds_;
};

}
#undef LLAMINAR_TILE_COMPLETION_HD
