/**
 * @file GDNPlanarRecurrence.h
 * @brief Typed planar recurrence geometry, physical state format and borrowed ownership.
 *
 * Device replay owns live lengths. The immutable layout describes every initial,
 * terminal and snapshot state bank for its entire lifetime; it is never inferred
 * from dimensions. This tuple owns no storage, conversion or execution policy.
 */
#pragma once
#include "kernels/common/DeviceRequestRowRanges.h"
#include "kernels/common/GDNRecurrenceStateLayout.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::gdn
{
/** @brief Borrow original expanded planar operands and one canonical state-format authority. */
struct PlanarRecurrenceInput final
{
    DeviceRequestRowRanges rows; ///< Original request-major physical rows and optional live lengths.
    int heads,key_width,value_width; ///< Expanded planar heads and supported64/128 key width.
    bool normalize_qk; ///< Original raw-input normalization flag.
    const float *query,*key,*value; ///< Original [row,head,key/value] source planes.
    const float *alpha,*beta_raw,*A_log,*dt_bias; ///< Original projected and learned gates.
    float *output; ///< Exclusive [row,head,value] activation owner.
    const float *initial_state; ///< Canonical layout input; may equal updated_state exactly.
    float *updated_state,*snapshots; ///< Same-layout terminal and optional per-row state owners.
    int snapshot_stride_floats,max_snapshot_rows; ///< Full admitted snapshot envelope, never live work.
    GDNRecurrenceStateLayout recurrence_layout=GDNRecurrenceStateLayout::KeyMajorValueContiguous;
};
namespace planar_recurrence_detail
{
/** @brief One checked half-open native range; not an allocation ledger. */
struct Range final {std::uintptr_t first=0,last=0;};
/**
 * @brief Check original native float/int storage without reading live values.
 * @param pointer Original borrowed address.
 * @param bytes Complete physical protected extent.
 * @param result Checked half-open range.
 * @return False for absent/misaligned/wrapping nonempty storage.
 */
inline bool range(const void* pointer,std::size_t bytes,Range& result) noexcept
{
    const auto address=reinterpret_cast<std::uintptr_t>(pointer);
    if(!pointer||address%alignof(float)||bytes>std::numeric_limits<std::uintptr_t>::max()-address)return false;
    result={address,address+bytes};return true;
}
/** @brief Pure checked-range intersection.
 * @param a First admitted range.
 * @param b Second admitted range.
 * @return Whether physical bytes overlap.
 */
inline bool overlaps(Range a,Range b) noexcept{return a.first<b.last&&b.first<a.last;}
}
/**
 * @brief Validate typed planar geometry and every original read/write extent.
 * @param p Complete borrowed immutable native tuple.
 * @return Whether the tuple can enter either explicit state-layout program.
 * @details Per-head and flattened input coordinates retain their original INT32
 * ABI bounds; request-state and snapshot byte offsets use size_t. On supported
 * 64-bit targets, rows*heads*DV<=INT_MAX and DK<=128 bound the full request-state
 * span below 2^40 bytes without imposing an invalid INT32 total-state limit.
 * Only exact initial==updated aliasing is admitted. Count values remain device-owned.
 */
inline bool validPlanarRecurrenceStorage(const PlanarRecurrenceInput& p) noexcept
{
    using namespace planar_recurrence_detail;
    if(!validGDNRecurrenceStateLayout(p.recurrence_layout)||p.heads<=0||
        (p.key_width!=64&&p.key_width!=128)||p.value_width<=0)return false;
    const auto max=std::size_t(std::numeric_limits<int>::max());
    const auto heads=std::size_t(p.heads),dk=std::size_t(p.key_width),dv=std::size_t(p.value_width);
    const auto rows=std::size_t(p.rows.physicalRows()),requests=std::size_t(p.rows.requests());
    if(heads>max/dk||heads>max/dv||heads*dk>max/dv||rows>max/(heads*dk)||rows>max/(heads*dv)||requests>max/heads)return false;
    const auto state=heads*dk*dv*sizeof(float),all_state=requests*state;
    const auto qk=rows*heads*dk*sizeof(float),values=rows*heads*dv*sizeof(float),gates=rows*heads*sizeof(float),learned=heads*sizeof(float);
    std::size_t shots=0;
    if(p.snapshots)
    {
        if(p.snapshot_stride_floats<=0||std::size_t(p.snapshot_stride_floats)<state/sizeof(float)||p.max_snapshot_rows<=0)return false;
        const auto stride=std::size_t(p.snapshot_stride_floats)*sizeof(float);
        if(std::size_t(p.max_snapshot_rows)>std::numeric_limits<std::size_t>::max()/stride)return false;
        shots=stride*p.max_snapshot_rows;
    }
    Range reads[9]{},writes[3]{};
    const void* pointers[]{p.query,p.key,p.value,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.initial_state,p.rows.lengthOwner()};
    const std::size_t sizes[]{qk,qk,values,gates,gates,learned,learned,all_state,sizeof(std::int32_t)*requests};
    for(int i=0;i<9;++i)
    {
        if(i==8&&!pointers[i])continue;
        if(!range(pointers[i],sizes[i],reads[i]))return false;
    }
    if(!range(p.output,values,writes[0])||!range(p.updated_state,all_state,writes[1])||
        (p.snapshots&&!range(p.snapshots,shots,writes[2])))return false;
    for(int i=0;i<3;++i)
    {
        for(int j=0;j<i;++j)if(overlaps(writes[i],writes[j]))return false;
        for(int j=0;j<9;++j)if(overlaps(writes[i],reads[j])&&!(i==1&&j==7&&p.initial_state==p.updated_state))return false;
    }
    return true;
}
/** @brief Enqueue the explicit typed planar ROCm program.
 * @param inputs Borrowed original operands, row authority and canonical physical state layout.
 * @param stream Exact non-default native stream retained through captured retirement.
 * @return Whether cold admission and native enqueue succeeded; never converts layout.
 */
bool launchROCmPlanarRecurrence(const PlanarRecurrenceInput& inputs,void* stream);
/** @brief Enqueue the explicit typed planar CUDA program.
 * @param inputs Borrowed original operands, row authority and canonical physical state layout.
 * @param stream Exact non-default native stream retained through captured retirement.
 * @return Whether cold admission and native enqueue succeeded; never converts layout.
 */
bool launchCUDAPlanarRecurrence(const PlanarRecurrenceInput& inputs,void* stream);
}
