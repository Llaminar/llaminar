/**
 * @file FloatingProjectionRows.h
 * @brief Sealed floating-projection admission for prefixes or request-major holes.
 *
 * DeviceRowRange retains its contiguous-prefix meaning. Only updated floating
 * projection kernels accept this separate by-value descriptor. Request lengths
 * remain borrowed from DeviceRequestRowRanges; factories never read them.
 */
#pragma once
#include "DeviceRowRange.h"
#include "DeviceRequestRowRanges.h"
#include <algorithm>
#include <stdexcept>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FLOAT_ROWS_HD __host__ __device__
#else
#define LLAMINAR_FLOAT_ROWS_HD
#endif
namespace llaminar2 {
/** @brief Immutable floating-kernel admission; never converts holes into a prefix count. */
class FloatingProjectionRows final {
public:
    /** @brief Bind exactly one selection source and validate physical geometry without reading counts. */
    static FloatingProjectionRows bind(int physical,const DeviceRowRange *prefix,
        const DeviceRequestRowRanges *requests) {
        if(physical<=0 || (prefix && requests) ||
            (prefix && prefix->physicalRows()!=physical) ||
            (requests && requests->physicalRows()!=physical))
            throw std::invalid_argument("Floating projection requires one exact prefix/request selection");
        return FloatingProjectionRows(prefix ? *prefix : DeviceRowRange::fullyActive(physical),
            requests ? *requests : DeviceRequestRowRanges::fullyActive(physical),requests!=nullptr);
    }
    /** @brief Hash only immutable request selection metadata, never the published values. */
    static std::uint64_t requestRecordingIdentity(DeviceRequestRowRanges rows) noexcept {
        std::uint64_t result=1469598103934665603ULL;
        const auto mix=[&](std::uint64_t word){result^=word;result*=1099511628211ULL;};
        mix(rows.requests());mix(rows.rowsPerRequest());
        mix(reinterpret_cast<std::uintptr_t>(rows.lengthOwner()));
        return result ? result : 1;
    }
    /** @return Immutable complete matrix row capacity. */
    LLAMINAR_FLOAT_ROWS_HD int physicalRows() const noexcept { return prefix_.physicalRows(); }
    /** @brief Pure row admission against one explicitly supplied native publication. */
    LLAMINAR_FLOAT_ROWS_HD int rowAdmissionFor(int row,int published) const noexcept {
        if(row<0 || row>=physicalRows())return 0;
        if(!request_major_) {
            const int active=prefix_.activeRowsFor(published);
            return active<0 ? -1 : int(row<active);
        }
        const int request=row/requests_.rowsPerRequest();
        const int active=requests_.activeRowsFor(request,published);
        return active<0 ? -1 : int(row%requests_.rowsPerRequest()<active);
    }
    /** @brief Validate all CPU-owned publications before any projection write. */
    void validateHostPublications() const {
        if(request_major_) {
            for(int r=0;r<requests_.requests();++r)
                if(requests_.activeRowsFor(r,requests_.lengthOwner() ? requests_.lengthOwner()[r] : requests_.rowsPerRequest())<0)
                    throw std::invalid_argument("Floating projection request length exceeds admitted stride");
        } else if(prefix_.activeRowsFor(prefix_.countOwner() ? *prefix_.countOwner() : prefix_.capacity())<0)
            throw std::invalid_argument("Floating projection prefix exceeds admitted capacity");
    }
    /** @brief Resolve a CPU-owned row without reading any other request's activation. */
    bool hostRowIsActive(int row) const {
        if(row<0 || row>=physicalRows())return false;
        const int published=request_major_ ? (requests_.lengthOwner() ?
            requests_.lengthOwner()[row/requests_.rowsPerRequest()] : requests_.rowsPerRequest()) :
            (prefix_.countOwner() ? *prefix_.countOwner() : prefix_.capacity());
        const int admitted=rowAdmissionFor(row,published);
        if(admitted<0)throw std::invalid_argument("Floating projection live publication is invalid");
        return admitted!=0;
    }
    /** @return Request-aligned tile count; no tile crosses an independent request boundary. */
    int hostTiles(int width) const {
        if(width<=0)throw std::invalid_argument("Floating projection tile width must be positive");
        return request_major_ ? requests_.requests()*((requests_.rowsPerRequest()-1)/width+1) :
            (physicalRows()-1)/width+1;
    }
    /** @return First physical row of one fixed request-aligned CPU tile. */
    int hostTileFirst(int tile,int width) const {
        if(tile<0 || tile>=hostTiles(width))throw std::out_of_range("Floating projection tile exceeds request geometry");
        if(!request_major_)return tile*width;
        const int per=(requests_.rowsPerRequest()-1)/width+1;
        return (tile/per)*requests_.rowsPerRequest()+(tile%per)*width;
    }
    /** @return Exact active prefix inside one request-aligned CPU tile. */
    int hostTileRows(int tile,int width) const {
        const int first=hostTileFirst(tile,width);int live=0;
        while(live<width && first+live<physicalRows() &&
            (!request_major_ || (first+live)/requests_.rowsPerRequest()==first/requests_.rowsPerRequest()) &&
            hostRowIsActive(first+live))++live;
        return live;
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    /** @brief Resolve native row admission before loading activations/weights or writing outputs.
     * @details Request coordinates may differ between lanes: no wave-uniform broadcast is valid here. */
    __device__ inline __attribute__((always_inline)) bool rowIsActive(int row) const {
        if(row<0 || row>=physicalRows())return false;
        if(!request_major_)return row<prefix_.activeRows();
        const int r=row/requests_.rowsPerRequest();
        const int published=requests_.lengthOwner() ? requests_.lengthOwner()[r] : requests_.rowsPerRequest();
        const int admitted=rowAdmissionFor(row,published);
        if(admitted<0) {
#if defined(__CUDA_ARCH__)
            __trap();
#else
            __builtin_trap();
#endif
        }
        return admitted!=0;
    }
#endif
private:
    /** @brief Only checked bind constructs a kernel argument. */
    FloatingProjectionRows(DeviceRowRange prefix,DeviceRequestRowRanges requests,bool request_major)
        :prefix_(prefix),requests_(requests),request_major_(request_major) {}
    DeviceRowRange prefix_; ///< Original prefix geometry, unchanged and never reinterpreted.
    DeviceRequestRowRanges requests_; ///< Sole request-length authority when request_major_ is true.
    bool request_major_; ///< Immutable selection alternative in the captured launch identity.
};
static_assert(std::is_trivially_copyable_v<FloatingProjectionRows>);
}
#undef LLAMINAR_FLOAT_ROWS_HD
