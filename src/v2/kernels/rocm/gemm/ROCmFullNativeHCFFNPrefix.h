/**
 * @file ROCmFullNativeHCFFNPrefix.h
 * @brief Caller-owned storage and bounded geometry for the fused production HC prefix.
 *
 * Physical ownership changes neither source-selected arithmetic partitions nor
 * public spans. One cold extent function owns admission and capture validation;
 * its immutable bound is captured directly for the device body. Dynamic storage
 * is phase-disjoint with the following FFN;
 * the small local activation bank survives until the final HC publication edge.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include "ROCmFullNativeHCFFN.h"
#if defined(__HIPCC__)
#define LLAMINAR_HC_GEOMETRY_INLINE __host__ __device__ inline
#else
#define LLAMINAR_HC_GEOMETRY_INLINE inline
#endif
namespace llaminar2::rocm {
/** @brief Local original TwoTerm banks, retained until all raw readers retire. */
struct alignas(16) FullNativeHCFFNPrefixScratch {
 float activated[512]; ///< Rounded public activation, pending alias-safe publication.
 signed char primary[512],residual[512]; ///< Exact original local Q8 operands.
 float primary_scales[16],residual_scales[16]; ///< Original per-K32 stored scales.
};
static_assert(sizeof(FullNativeHCFFNPrefixScratch)==3200);
/** @brief Integer minimum without a backend/runtime dependency.
 * @param a First extent. @param b Second extent. @return Smaller extent.
 */
LLAMINAR_HC_GEOMETRY_INLINE constexpr std::size_t fullHCMin(std::size_t a,std::size_t b) {return a<b?a:b;}
/** @brief Integer maximum without a backend/runtime dependency.
 * @param a First extent. @param b Second extent. @return Larger extent.
 */
LLAMINAR_HC_GEOMETRY_INLINE constexpr std::size_t fullHCMax(std::size_t a,std::size_t b) {return a>b?a:b;}
/** @brief Exact cardinal down-column ownership, including empty norm producers. */
struct FullHCDownOwnership {unsigned first,columns;};
/**
 * @brief Assign all columns to the idle cohort when one exists.
 * @param bid Physical CTA. @param grid Exact admitted physical grid.
 * @param columns Original down output extent. @return Unique complete-column interval.
 * @details At grids of one through four CTAs every CTA also has a norm task;
 * those geometries use the same complete-column program after norm publication.
 * Empty owners execute all phase joins and never divide by their column count.
 */
LLAMINAR_HC_GEOMETRY_INLINE FullHCDownOwnership fullHCDownOwnership(
 unsigned bid,unsigned grid,unsigned columns) {
 const unsigned norm=grid<4?grid:4;
 if(grid>norm&&bid<norm)return {0,0};
 const unsigned owners=grid>norm?grid-norm:grid;
 const unsigned index=grid>norm?bid-norm:bid;
 const unsigned first=static_cast<unsigned>(std::uint64_t(index)*columns/owners);
 const unsigned end=static_cast<unsigned>(std::uint64_t(index+1)*columns/owners);
 return {first,end-first};
}
/**
 * @brief Actual phase maximum, capped at32KiB, with no device-state readback.
 * @param prefix Original cold geometry and source-selected arithmetic policy.
 * @param grid Exact admitted grid, greater than zero.
 * @return Required aligned dynamic bytes, or zero for an invalid cold geometry.
 * @details Norm retains up to one bounded POST/gain cache; excess channels use
 * their original published bytes. Down bounds column, partition and K32 windows;
 * up bounds channel panels. Continuations preserve each original accumulator.
 * Grid1 bounds every positive grid: its complete down interval is R, its up
 * stripe is H, and norm storage does not depend on the physical grid. Cold
 * actual-function admission may therefore use grid1 without an occupancy cycle.
 * Thus increasing H or reducing the physical grid never truncates real work or
 * imposes a two-packet limit. The original R64..512 envelope is unchanged.
 */
inline std::size_t fullNativeHCFFNPrefixDynamicBytes(
 const FullNativeHCFFNPrefixArgs& prefix,unsigned grid) {
 const auto& a=prefix.hc;
 if(!grid||a.norm.hidden<=0||a.down_weights.n<=0||a.down_weights.k<=0||a.partitions<=0)return 0;
 constexpr std::size_t cap=8192;
 const std::size_t h=static_cast<unsigned>(a.norm.hidden),r=static_cast<unsigned>(a.down_weights.n);
 const std::size_t p=static_cast<unsigned>(a.partitions),blocks=static_cast<unsigned>(a.down_weights.k)/32;
 const std::size_t per=(blocks+p-1)/p,part_batch=fullHCMin(p,32),window=fullHCMin(per,32);
 const unsigned owners=grid>4?grid-4:grid;
 const std::size_t owned=(r+owners-1)/owners,channels=(h+grid-1)/grid;
 const auto norm=fullHCMin(cap,256+2*h);
 const auto down=fullHCMin(cap,part_batch*(window+1)*owned);
 const auto up=fullHCMin(cap,2*(r/32)*4*channels);
 const auto words=fullHCMax(norm,fullHCMax(down,fullHCMax(up,256)));
 return ((words*4+15)/16)*16;
}
} // namespace llaminar2::rocm
#undef LLAMINAR_HC_GEOMETRY_INLINE
