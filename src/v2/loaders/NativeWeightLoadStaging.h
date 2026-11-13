/**
 * @file NativeWeightLoadStaging.h
 * @brief Shared bounded CPU native-input and simultaneous consumer scratch geometry.
 *
 * This immutable declaration prices one serialized source transaction. The
 * original reader and consumer use one PMA-owned byte arena. GPU upload rings
 * may coexist and retain their separate physical allocations. No live balance
 * or final prepared/resident storage is owned here.
 */
#pragma once
#include "loaders/GPUHostLoadPreflight.h"
#include "memory/CPUStorageRoot.h"
#include "utils/DebugEnv.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
namespace llaminar2 {
/** @return Existing startup cap, shared by CPU sources without changing per-GPU semantics. */
inline std::optional<size_t> configuredNativeWeightLoadStagingBudgetBytes() {
    return gpuPerDeviceLoadStagingBudgetBytes(debugEnv().rocm.repack_budget_mb);
}
/** @brief Original native granularity and simultaneous consumer workspace per useful unit. */
struct NativeWeightLoadStagingRequest {
    size_t native_bytes_per_unit=0; ///< One complete encoded block/row, never an entire-tensor requirement.
    size_t scratch_bytes_per_unit=0; ///< Scratch coexisting with original native bytes.
    size_t fixed_scratch_bytes=0; ///< Consumer's count-independent workspace.
    size_t maximum_units=0; ///< Useful remaining population in original source.
    bool operator==(const NativeWeightLoadStagingRequest&)const=default;
};
/** @brief Exact subregions of one allocator-owned staging root. */
struct NativeWeightLoadStagingGeometry {
    size_t maximum_units_per_window=0;
    size_t native_bytes=0;
    size_t scratch_offset=0;
    size_t scratch_bytes=0;
    size_t physical_bytes=0;
    bool operator==(const NativeWeightLoadStagingGeometry&)const=default;
};
/**
 * @brief Price the largest useful native window and simultaneous scratch under the original cap.
 * @param request Consumer-authored indivisible unit and workspace geometry.
 * @param placement Original source owner's CPU physical placement.
 * @param budget Existing configured cap; absent preserves its explicit unlimited policy.
 * @return Original useful source/scratch subranges and exact physical root bytes.
 * @throws std::invalid_argument For invalid geometry or one indivisible unit exceeding the cap.
 * @throws std::overflow_error For unrepresentable source/workspace/physical geometry.
 */
inline NativeWeightLoadStagingGeometry resolveNativeWeightLoadStaging(
    NativeWeightLoadStagingRequest request,CPUStoragePlacement placement,std::optional<size_t> budget) {
    if(!request.native_bytes_per_unit || !request.maximum_units || (budget && !*budget))
        throw std::invalid_argument("Native source staging requires positive original units and cap");
    const auto add=[](size_t a,size_t b) {
        if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("Native source staging extent overflows");
        return a+b;
    };
    const auto mul=[](size_t a,size_t b) {
        if(a && b>std::numeric_limits<size_t>::max()/a)throw std::overflow_error("Native source staging extent overflows");
        return a*b;
    };
    const auto quote=[&](size_t units) {
        const auto native=mul(units,request.native_bytes_per_unit);
        const auto scratch=add(mul(units,request.scratch_bytes_per_unit),request.fixed_scratch_bytes);
        constexpr size_t scratch_alignment=64;
        const auto offset=scratch ? add(native,scratch_alignment-1)&~(scratch_alignment-1) : native;
        return NativeWeightLoadStagingGeometry{units,native,offset,scratch,
            placement.physicalBytes<uint8_t>(add(offset,scratch))};
    };
    // Overflow of an impossible full population must not reject a bounded
    // window that fits. Only candidate windows that can fit the cap are priced.
    if(!budget)return quote(request.maximum_units);
    size_t high=std::min(request.maximum_units,*budget/request.native_bytes_per_unit);
    size_t low=0;
    while(low<high) {
        const auto mid=low+(high-low)/2+1;
        bool fits=false;
        try { fits=quote(mid).physical_bytes<=*budget; }
        catch(const std::overflow_error&) { /* An overflowing candidate cannot fit a finite cap. */ }
        if(fits)low=mid;else high=mid-1;
    }
    if(!low)throw std::invalid_argument("Native source cap cannot hold one encoded unit and simultaneous consumer scratch");
    return quote(low);
}
} // namespace llaminar2
