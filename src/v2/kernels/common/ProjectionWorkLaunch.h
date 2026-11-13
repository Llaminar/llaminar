/**
 * @file ProjectionWorkLaunch.h
 * @brief Capture-time projection-work binding and native original-row cursors.
 *
 * Host helpers copy immutable descriptor geometry only. Device cursors resolve
 * borrowed request publications after their original producer edge; no count
 * is downloaded, compacted, republished or used to change physical matrix M.
 */
#pragma once
#include "ProjectionRowWork.h"
namespace llaminar2 {
/**
 * @brief Bind independent work admission without selecting verifier arithmetic.
 * @param physical Original matrix row capacity used by all tensor strides.
 * @param prefix Optional existing prefix authority, retained for consistency.
 * @return A by-value kernel descriptor borrowing the exact original publications.
 */
inline ProjectionRowWork projectionWorkForLaunch(int physical,const DeviceRowRange* prefix=nullptr)
{
    if (physical <= 0 || (prefix && prefix->physicalRows()!=physical))
        throw std::invalid_argument("Projection work launch disagrees with physical matrix rows");
    const auto base=prefix?*prefix:DeviceRowRange::fullyActive(physical);
    if(const auto* work=currentProjectionRowWork(physical))
        return prefix?work->withInheritedPrefix(*prefix):*work;
    return ProjectionRowWork::fromPrefix(base);
}
/**
 * @brief Select the compact legacy prefix ABI unless explicit work is bound.
 * @tparam Launch Typed host callback whose argument discriminates captured kernels.
 * @param physical Original physical matrix row capacity.
 * @param prefix Actual optional prefix authority; absent is not an inherited origin.
 * @param launch Callback receiving DeviceRowRange or ProjectionRowWork by value.
 * @return The exact selected callback result, without reading a live publication.
 */
template<class Launch>
decltype(auto) dispatchProjectionRows(int physical,const DeviceRowRange* prefix,Launch&& launch)
{
    if(physical<=0 || (prefix && prefix->physicalRows()!=physical))
        throw std::invalid_argument("Projection dispatch disagrees with physical matrix rows");
    if(const auto* work=currentProjectionRowWork(physical))
        return launch(prefix?work->withInheritedPrefix(*prefix):*work);
    return launch(prefix?*prefix:DeviceRowRange::fullyActive(physical));
}
/**
 * @brief Select a compile-time native work mode from immutable scope presence.
 * @tparam Launch Host callback exposing operator()<bool HasWork>().
 * @param physical Original physical row extent authenticated by the scope.
 * @param launch One shared numeric launch body with mode-specific native symbols.
 * @return Exact callback result; no device publication is read by this dispatch.
 */
template<class Launch>
decltype(auto) dispatchProjectionWorkMode(int physical,Launch&& launch)
{
    if(currentProjectionRowWork(physical))return launch.template operator()<true>();
    return launch.template operator()<false>();
}
#if defined(__CUDACC__) || defined(__HIPCC__)
/**
 * @brief Resolve a lane-local original row without promoting differing requests.
 * @param rows Compact legacy prefix or explicit original-work authority.
 * @param row Original physical row coordinate for this lane.
 * @return Whether the original row may be read or written.
 */
__device__ inline bool projectionRowIsActive(DeviceRowRange rows,int row)
{return row>=0 && row<rows.activeRows();}
/** @copydoc projectionRowIsActive(DeviceRowRange,int) */
__device__ inline bool projectionRowIsActive(ProjectionRowWork rows,int row)
{return rows.rowIsActive(row);}
/**
 * @brief Resolve one CTA-uniform original row before decoder register lifetimes.
 * @param work Borrowed native interval authority.
 * @param row Identical original row in every participating lane.
 * @return Uniform admission; invalid publications still trap in the resolver.
 * @pre The caller proves identical row/request coordinates across its wave.
 */
__device__ inline bool projectionWorkRowIsActiveUniform(ProjectionRowWork work,int row)
{
    const int admitted=work.rowIsActive(row);
#if defined(__HIPCC__)
    return __builtin_amdgcn_readfirstlane(admitted)!=0;
#else
    return admitted!=0;
#endif
}
/**
 * @brief Iterate bounded row workers over original request-local intervals.
 * @details Each worker visits its own tiles in every immutable request. Empty
 * requests perform no arithmetic or tensor reads; physical M remains unchanged.
 */
template<class Rows>
class ProjectionWorkCursor final {
public:
    /**
     * @brief Borrow one descriptor and seal independent worker/tile geometry.
     * @param work By-value original request/interval authority.
     * @param worker Native independent row-worker index.
     * @param workers Native positive row-worker grid extent.
     * @param width Original compiled rows per work tile.
     */
    __device__ ProjectionWorkCursor(Rows work,int worker,int workers,int width)
        :work_(work),worker_(worker),workers_(workers),width_(width){}
    /**
     * @brief Resolve the next nonempty original physical tile before decoding.
     * @param first First original physical row owned by this tile.
     * @param end Exclusive original request work bound, used by partial lanes.
     * @return True only when this tile owns at least one authored row.
     */
    __device__ bool next(int& first,int& end){
        while(request_<work_.requests()){
            if(!opened_){span_=work_.resolveDevice(request_);
#if defined(__HIPCC__)
                // Cursor request/worker geometry is wave-uniform. Per-element
                // rowIsActive users deliberately do not use this promotion.
                span_.first=__builtin_amdgcn_readfirstlane(span_.first);
                span_.end=__builtin_amdgcn_readfirstlane(span_.end);
#endif
                next_=std::int64_t(span_.first)+std::int64_t(worker_)*width_;opened_=true;}
            if(next_<span_.end){first=int(next_);end=span_.end;next_+=std::int64_t(workers_)*width_;return true;}
            ++request_;opened_=false;
        }
        return false;
    }
private:
    Rows work_; ///< Sole borrowed publication authority.
    ProjectionRowSpan span_{}; ///< Current validated original request interval.
    int worker_,workers_,width_; ///< Immutable native independent-worker geometry.
    int request_=0;std::int64_t next_=0;bool opened_=false; ///< Local traversal state, never GPU publication.
};
/**
 * @brief Compact legacy prefix traversal without generic request descriptors.
 * @details The native prefix publication is resolved before decoder lifetimes.
 * The integer traversal is the original strided row-worker schedule; no work
 * pointer, request table, threshold or causal arithmetic enters this kernel ABI.
 */
template<>
class ProjectionWorkCursor<DeviceRowRange> final {
public:
    /**
     * @brief Seal original row-worker geometry and borrow the legacy live prefix.
     * @param rows Original compact prefix descriptor.
     * @param worker Native independent row-worker coordinate.
     * @param workers Positive native row-worker grid extent.
     * @param width Compiled rows per arithmetic tile.
     */
    __device__ ProjectionWorkCursor(DeviceRowRange rows,int worker,int workers,int width)
        :next_(worker*width),end_(rows.activeRows()),step_(workers*width){}
    /**
     * @brief Visit the next original physical prefix tile.
     * @param first First original row owned by this tile.
     * @param end Exclusive published original live-prefix bound.
     * @return Whether the tile contains an authored row.
     */
    __device__ bool next(int& first,int& end){
        if(next_>=end_)return false;
        first=next_;end=end_;
#if defined(__CUDACC__) && !defined(__HIPCC__)
        // Original CUDA grouped grids assign exactly one prefix tile per CTA.
        // Sealing the cursor after that tile lets the legacy while reduce to if.
        next_=end_;
#else
        // Original HIP row workers traverse their fixed residue class.
        next_+=step_;
#endif
        return true;
    }
private:
    int next_,end_,step_; ///< Compact original worker bounds only.
};
/**
 * @brief Compile out generic work resolution for original prefix kernels.
 * @tparam HasWork Captured immutable launch-mode discriminant.
 * @param work Original work descriptor, unused when HasWork is false.
 * @param prefix Original compact prefix authority used by legacy arithmetic.
 * @param worker Native row-worker coordinate, uniform for the participating CTA.
 * @param workers Positive row-worker grid extent.
 * @param width Original compiled arithmetic tile width.
 * @return Typed cursor containing only the selected mode's admission state.
 */
template<bool HasWork>
__device__ inline auto projectionCursorFor(ProjectionRowWork work,DeviceRowRange prefix,
    int worker,int workers,int width)
{
    if constexpr(HasWork)return ProjectionWorkCursor(work,worker,workers,width);
    else return ProjectionWorkCursor(prefix,worker,workers,width);
}
#endif
}
