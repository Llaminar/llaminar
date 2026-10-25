/**
 * @file GDNPreparedQK.h
 * @brief Scalar prepared-Q/K activation geometry and borrowed native operands.
 *
 * The graph arena/PMA owns the transient activation. This header contributes
 * one checked capacity formula, never a live allocation ledger. Raw convolution
 * values and every existing cache/snapshot bank retain their original owners.
 */
#pragma once
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "config/GDNHeadAssignment.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2::gdn
{
/** @brief Exact scalar Q-then-K plane capacity shared by schema and PMA BOM. */
struct ScalarPreparedQKLayout final
{
private:
    /**
     * @brief Derive every public capacity through one checked plane formula.
     * @param plane Positive FP32 elements in one original Q or K plane.
     * @return The exact Q-then-K activation extent and bytes.
     * @throws std::invalid_argument For zero or overflowing capacity.
     */
    [[nodiscard]] static ScalarPreparedQKLayout fromPlaneElements(std::size_t plane)
    {
        if (!plane || plane > std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)))
            throw std::invalid_argument("Scalar prepared GDN Q/K capacity overflows");
        return {plane, 2 * plane, 2 * plane * sizeof(float)};
    }
public:
    /**
     * @brief Identify the existing scalar prepared native geometry before admission.
     * @param key_width Original recurrent key width.
     * @param convolution_width Original depthwise convolution width.
     * @return Whether the native producer/consumer pair implements this geometry.
     */
    [[nodiscard]] static constexpr bool supportsNativeGeometry(
        int key_width, int convolution_width) noexcept
    {
        return (key_width == 64 || key_width == 128) && convolution_width == 4;
    }
    /**
     * @brief Price a valid participant's key count without using its data origin.
     * @param kh Original global key-head count.
     * @param vh Original global value-head count.
     * @param dk Native key width.
     * @param local_query_heads Already-admitted query interval width.
     * @param global_query_heads Original query partition width.
     * @return The same compact capacity as the actual whole-head assignment.
     * @throws std::invalid_argument For nonintegral or unsupported head geometry.
     * @details Integral GDN boundaries make counts proportional to interval width.
     * The normalized origin is capacity-only; it never selects source/state data.
     * Actual offset validation remains with the canonical TP declaration.
     */
    [[nodiscard]] static ScalarPreparedQKLayout fromHeadPartition(
        int kh, int vh, int dk, int local_query_heads, int global_query_heads)
    {
        const auto counts = GDNHeadAssignment::fromPartition(
            kh, vh, 0, local_query_heads, global_query_heads);
        const auto limit = std::size_t(std::numeric_limits<int>::max());
        if ((dk != 64 && dk != 128) ||
            std::size_t(counts.localKeyHeads()) > limit / (2 * std::size_t(dk)))
            throw std::invalid_argument("Scalar prepared GDN key capacity is invalid");
        return fromPlaneElements(std::size_t(counts.localKeyHeads()) * dk);
    }
    std::size_t plane_elements;
    std::size_t elements;
    std::size_t bytes;
    /**
     * @brief Derive the compact bank from original merged/value projection widths.
     * @param merged_columns Original two-key-plane plus value width.
     * @param value_columns Original participant-local value width.
     * @return One physical row's exact Q/K activation capacity.
     * @throws std::invalid_argument For zero, inconsistent or overflowing geometry.
     */
    [[nodiscard]] static ScalarPreparedQKLayout fromProjectionWidths(
        std::size_t merged_columns,std::size_t value_columns)
    {
        if(!value_columns||merged_columns<=value_columns||
            (merged_columns-value_columns)%2||
            merged_columns-value_columns>std::numeric_limits<std::size_t>::max()/sizeof(float))
            throw std::invalid_argument("Scalar prepared GDN Q/K geometry is invalid");
        const auto elements=merged_columns-value_columns;
        return fromPlaneElements(elements / 2);
    }
    /**
     * @brief Derive the same capacity from checked original key/value heads.
     * @param kh Original participant-local compact key head count.
     * @param vh Original participant-local value head count.
     * @param dk Original supported native key width,64 or128.
     * @param dv Original positive value width.
     * @return Exact single-row Q/K capacity.
     * @throws std::invalid_argument For invalid dimensions or native index overflow.
     */
    [[nodiscard]] static ScalarPreparedQKLayout fromHeads(int kh,int vh,int dk,int dv)
    {
        const auto limit=std::size_t(std::numeric_limits<int>::max());
        if(kh<=0||vh<=0||(dk!=64&&dk!=128)||dv<=0||
            std::size_t(kh)>limit/(2*std::size_t(dk))||std::size_t(vh)>limit/std::size_t(dv))
            throw std::invalid_argument("Scalar prepared GDN head geometry is invalid");
        const auto keys=2*std::size_t(kh)*dk,values=std::size_t(vh)*dv;
        if(keys>limit-values)throw std::invalid_argument("Scalar prepared GDN channel width overflows");
        return fromProjectionWidths(keys+values,values);
    }
};
/** @brief Immutable scalar convolution inputs; state is supplied by its cache owner. */
struct ScalarPreparedQKConvolution final
{
    int key_heads,value_heads,key_width,value_width;
    bool normalize_qk;
    const float *input,*weight,*bias;
    float *raw_output,*prepared;
};
/** @brief Original convolution live/speculative/snapshot owners plus its operands. */
struct ScalarPreparedQKConvolutionInput final
{
    ScalarPreparedQKConvolution operands;
    const float *initial_state;
    float *updated_state,*snapshots;
    int snapshot_stride_floats,max_snapshot_rows;
};
/** @brief Immutable scalar recurrence operands; original state owner resolves banks. */
struct ScalarPreparedQKRecurrence final
{
    int key_heads,value_heads,key_width,value_width,global_value_offset;
    bool normalize_qk;
    const float *merged,*prepared,*alpha,*beta_raw,*A_log,*dt_bias;
    float *output;
};
/** @brief Original merged recurrence authority plus its explicit compact Q/K root. */
struct ScalarPreparedQKRecurrenceInput final
{
    MergedRecurrenceInput original;
    const float *prepared;
};
/**
 * @brief Validate scalar producer spans against the original state/snapshot owners.
 * @param p Original producer operands and complete admitted state publication spans.
 * @return False before enqueue for invalid geometry, overflow, or any write overlap.
 * @details Only initial_state==updated_state is legal. All read-only source bytes
 * stay immutable, including the projection row used later to update convolution
 * history; the compact activation never aliases the raw source or raw output.
 */
inline bool validScalarPreparedQKConvolutionStorage(const ScalarPreparedQKConvolutionInput& p) noexcept
{
    using namespace planar_recurrence_detail;
    const auto& a=p.operands;
    ScalarPreparedQKLayout layout{};
    try{layout=ScalarPreparedQKLayout::fromHeads(a.key_heads,a.value_heads,a.key_width,a.value_width);}
    catch(const std::invalid_argument&){return false;}
    const auto channels=layout.elements+std::size_t(a.value_heads)*a.value_width;
    // The cache owner's original state-size ABI is int, independent of size_t ranges.
    if(channels>std::size_t(std::numeric_limits<int>::max())/3)return false;
    const auto raw=channels*sizeof(float),state=3*raw;
    std::size_t shots=0;
    if(p.snapshots)
    {
        if(p.snapshot_stride_floats<=0||std::size_t(p.snapshot_stride_floats)<3*channels||p.max_snapshot_rows<=0)return false;
        const auto stride=std::size_t(p.snapshot_stride_floats)*sizeof(float);
        if(std::size_t(p.max_snapshot_rows)>std::numeric_limits<std::size_t>::max()/stride)return false;
        shots=stride*p.max_snapshot_rows;
    }
    Range reads[4]{},writes[4]{};
    const void* inputs[]{a.input,a.weight,a.bias,p.initial_state};
    const std::size_t input_bytes[]{raw,4*raw,raw,state};
    for(int i=0;i<4;++i)
    {
        if(i==2&&!inputs[i])continue;
        if(!range(inputs[i],input_bytes[i],reads[i]))return false;
    }
    if(!range(a.raw_output,raw,writes[0])||!range(a.prepared,layout.bytes,writes[1])||
        !range(p.updated_state,state,writes[2])||(p.snapshots&&!range(p.snapshots,shots,writes[3])))return false;
    for(int i=0;i<4;++i)
    {
        for(int j=0;j<i;++j)if(overlaps(writes[i],writes[j]))return false;
        for(int j=0;j<4;++j)
            if(overlaps(writes[i],reads[j])&&!(i==2&&j==3&&p.initial_state==p.updated_state))return false;
    }
    return true;
}
/**
 * @brief Validate the compact consumer and every original owner before enqueue.
 * @param p Original merged tuple and its explicit producer-prepared activation root.
 * @return Whether this exact scalar ValueMajor program has a complete disjoint binding.
 * @details The original merged validator remains authoritative for its operands.
 * The additional compact plane is exclusive activation storage, disjoint from all
 * original read and write owners, including full snapshot and count capacities.
 */
inline bool validScalarPreparedQKRecurrenceStorage(const ScalarPreparedQKRecurrenceInput& p) noexcept
{
    using namespace planar_recurrence_detail;
    const auto& a=p.original;
    if(a.rows.requests()!=1||a.rows.rowsPerRequest()!=1||
        a.recurrence_layout!=GDNRecurrenceStateLayout::ValueMajorKeyContiguous||
        !validMergedRecurrenceStorage(a))return false;
    ScalarPreparedQKLayout layout{};
    try{layout=ScalarPreparedQKLayout::fromHeads(a.key_heads,a.value_heads,a.key_width,a.value_width);}
    catch(const std::invalid_argument&){return false;}
    Range compact{};
    if(!range(p.prepared,layout.bytes,compact))return false;
    const auto values=std::size_t(a.value_heads)*a.value_width;
    const auto state=values*a.key_width*sizeof(float);
    const auto raw=(layout.elements+values)*sizeof(float),gates=std::size_t(a.value_heads)*sizeof(float);
    const auto shots=a.snapshots?std::size_t(a.snapshot_stride_floats)*sizeof(float)*a.max_snapshot_rows:0;
    const void* originals[]{a.merged,a.alpha,a.beta_raw,a.A_log,a.dt_bias,a.initial_state,
        a.updated_state,a.output,a.snapshots,a.rows.lengthOwner()};
    const std::size_t bytes[]{raw,gates,gates,gates,gates,state,state,values*sizeof(float),shots,sizeof(std::int32_t)};
    for(int i=0;i<10;++i)
    {
        if((i==8||i==9)&&!originals[i])continue;
        Range original{};
        if(!range(originals[i],bytes[i],original)||overlaps(compact,original))return false;
    }
    return true;
}
/**
 * @brief Enqueue exact scalar convolution/QK publication on a non-default stream.
 * @param input Borrowed original scalar operands, state owners and compact arena root.
 * @param stream Exact non-default stream retained through native retirement.
 * @return Whether immutable admission and native submission succeeded.
 */
bool launchROCmScalarPreparedQKConvolution(const ScalarPreparedQKConvolutionInput& input,void* stream);
/**
 * @brief Consume producer-prepared Q/K without normalizing or scaling again.
 * @param input Borrowed original merged operands, state owners and compact arena root.
 * @param stream Exact non-default stream retained through native retirement.
 * @return Whether immutable admission and native submission succeeded.
 */
bool launchROCmScalarPreparedQKRecurrence(const ScalarPreparedQKRecurrenceInput& input,void* stream);
}
