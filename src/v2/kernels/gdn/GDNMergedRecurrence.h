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

    /**
     * @brief Prove original merged reads and output writes have disjoint physical bytes.
     * @return False for invalid geometry, null/overflowing byte ranges or any overlap.
     * @details This inspects immutable pointer/extent metadata only, never device
     * payloads. It intentionally permits initial_state==updated_state, required
     * by ordinary in-place decode. PMA and the cache remain allocation owners.
     */
    [[nodiscard]] bool sourceOutputDisjoint() const noexcept
    {
        if (!merged || !output ||
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
