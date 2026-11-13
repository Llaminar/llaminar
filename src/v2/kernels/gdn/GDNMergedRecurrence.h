/**
 * @file GDNMergedRecurrence.h
 * @brief Borrowed native recurrence inputs in the original post-convolution QKV layout.
 *
 * Reuses DeviceRequestRowRanges and GDN deinterleave coordinate admission.
 * This borrowed native tuple owns no allocation, stage policy or state ledger.
 * All sources/state/output are caller-admitted original physical regions and
 * survive capture/replay plus terminal retirement on the exact caller stream.
 */
#pragma once
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include "kernels/gdn/GDNPlanarRecurrence.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2::gdn
{
/** @brief Borrowed original raw operands and exact physical coordinate authority. */
struct MergedRecurrenceInput final
{
    DeviceRequestRowRanges rows; ///< Original physical requests and device live-length owner.
    int key_heads; ///< Query/key heads encoded in each merged source row.
    int value_heads; ///< Participant-local value/output heads.
    int key_width; ///< Native query/key width.
    int value_width; ///< Positive value/output width.
    int global_value_offset; ///< Signed modular source-head offset.
    bool use_qk_l2norm; ///< Original normalization policy, never pre-normalized operands.
    const float *merged; ///< Original read-only merged source rows.
    const float *alpha; ///< Per-row/head alpha projection.
    const float *beta_raw; ///< Per-row/head beta projection.
    const float *A_log; ///< Per-head learned gate scale.
    const float *dt_bias; ///< Per-head learned time-step bias.
    float *output; ///< Exclusive physical output, disjoint from original read-only sources.
    const float *initial_state; ///< Existing cache-owned input bank; may equal updated_state for ordinary decode.
    float *updated_state; ///< Existing live or speculative state destination.
    float *snapshots; ///< Optional physical post-row state snapshots.
    int snapshot_stride_floats; ///< Admitted snapshot row stride.
    int max_snapshot_rows; ///< Admitted physical snapshot row capacity.
    GDNRecurrenceStateLayout recurrence_layout=GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< One immutable physical state-bank authority.

    /**
     * @brief Prove original merged reads and output writes have disjoint physical bytes.
     * @return False for invalid geometry, null/overflowing byte ranges or any overlap.
     * @details This inspects immutable pointer/extent metadata only, never device
     * payloads. It intentionally permits initial_state==updated_state, required
     * by ordinary in-place decode. PMA and the cache remain allocation owners.
     */
    [[nodiscard]] bool sourceOutputDisjoint() const noexcept
    {
        if (!validGDNRecurrenceStateLayout(recurrence_layout) || !merged || !output ||
            gdnDeinterleaveElementsPerRequest(
                rows, key_heads, value_heads, key_width, value_width) <= 0)
            return false;

        const std::size_t source_width =
            2 * std::size_t(key_heads) * key_width +
            std::size_t(value_heads) * value_width;
        const std::size_t output_width =
            std::size_t(value_heads) * value_width;
        const auto count = std::size_t(rows.physicalRows());
        const auto maximum = std::numeric_limits<std::size_t>::max();
        if (source_width > maximum / count || output_width > maximum / count)
            return false;
        const auto source_elements = count * source_width;
        const auto output_elements = count * output_width;
        if (source_elements > maximum / sizeof(float) ||
            output_elements > maximum / sizeof(float))
            return false;

        const auto source_bytes = source_elements * sizeof(float);
        const auto output_bytes = output_elements * sizeof(float);
        const auto source_begin = reinterpret_cast<std::uintptr_t>(merged);
        const auto output_begin = reinterpret_cast<std::uintptr_t>(output);
        const auto pointer_max = std::numeric_limits<std::uintptr_t>::max();
        if (source_bytes > pointer_max - source_begin ||
            output_bytes > pointer_max - output_begin)
            return false;
        return source_begin + source_bytes <= output_begin ||
            output_begin + output_bytes <= source_begin;
    }
};
/**
 * @brief Validate all original merged read/write spans without reading live lengths.
 * @param p Exact borrowed physical sources, state format and publication envelope.
 * @return Whether every native extent/alias is representable and disjoint.
 * @details The existing compact-row helper validates original coordinate products.
 * Initial and updated state may alias exactly; all other writable overlaps fail.
 */
inline bool validMergedRecurrenceStorage(const MergedRecurrenceInput& p) noexcept
{
    using namespace planar_recurrence_detail;
    if(!p.sourceOutputDisjoint()||p.value_heads<=0||(p.key_width!=64&&p.key_width!=128)||p.value_width<=0)return false;
    const auto max=std::size_t(std::numeric_limits<int>::max());
    const auto heads=std::size_t(p.value_heads),dk=std::size_t(p.key_width),dv=std::size_t(p.value_width);
    const auto rows=std::size_t(p.rows.physicalRows()),requests=std::size_t(p.rows.requests());
    if(heads>max/dk||heads*dk>max/dv||heads>max/dv||requests>max/heads)return false;
    const auto source=(2*std::size_t(p.key_heads)*dk+heads*dv)*rows*sizeof(float);
    const auto state=heads*dk*dv*sizeof(float),all_state=requests*state;
    const auto output=rows*heads*dv*sizeof(float),gates=rows*heads*sizeof(float),learned=heads*sizeof(float);
    std::size_t shots=0;
    if(p.snapshots)
    {
        if(p.snapshot_stride_floats<=0||std::size_t(p.snapshot_stride_floats)<state/sizeof(float)||p.max_snapshot_rows<=0)return false;
        const auto stride=std::size_t(p.snapshot_stride_floats)*sizeof(float);
        if(std::size_t(p.max_snapshot_rows)>std::numeric_limits<std::size_t>::max()/stride)return false;
        shots=stride*p.max_snapshot_rows;
    }
    Range reads[7]{},writes[3]{};
    const void* ptrs[]{p.merged,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.initial_state,p.rows.lengthOwner()};
    const std::size_t bytes[]{source,gates,gates,learned,learned,all_state,sizeof(std::int32_t)*requests};
    for(int i=0;i<7;++i){if(i==6&&!ptrs[i])continue;if(!range(ptrs[i],bytes[i],reads[i]))return false;}
    if(!range(p.output,output,writes[0])||!range(p.updated_state,all_state,writes[1])||
        (p.snapshots&&!range(p.snapshots,shots,writes[2])))return false;
    for(int i=0;i<3;++i)
    {
        for(int j=0;j<i;++j)if(overlaps(writes[i],writes[j]))return false;
        for(int j=0;j<7;++j)if(overlaps(writes[i],reads[j])&&!(i==1&&j==5&&p.initial_state==p.updated_state))return false;
    }
    return true;
}

/**
 * @brief Enqueue one explicitly merged-layout recurrence, without deinterleave.
 * @param input Exact borrowed native operands/shape; no buffers are allocated.
 * @param stream Exact non-default caller stream, alive through graph retirement.
 * @return False for invalid immutable admission or enqueue failure; no alternate path.
 */
bool launchROCmMergedRecurrence(const MergedRecurrenceInput& input, void* stream);
/**
 * @brief Enqueue CUDA's original scalar recurrence over explicit merged input.
 * @param input Borrowed immutable source/state/output geometry and publication owners.
 * @param stream Exact non-default CUDA stream retained through native retirement.
 * @return False on invalid bindings or enqueue failure; never changes input layout.
 */
bool launchCUDAMergedRecurrence(const MergedRecurrenceInput& input, void* stream);
}
