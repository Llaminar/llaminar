/**
 * @file GDNDeinterleaveRows.h
 * @brief Shared CUDA/HIP live-row mapping for merged GDN Q/K/V inputs.
 *
 * Copy admission comes from request-local device lengths. Output plane bases
 * and request strides remain physical, including after empty or shrinking
 * replay. This is a lossless layout transform; it changes neither recurrence
 * arithmetic nor modular query/key head ownership.
 */
#pragma once
#include "kernels/common/DeviceRequestRowRanges.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2
{
    /**
     * @brief Validate representable merged and split matrix indexing.
     * @param rows Immutable physical request geometry.
     * @param key_heads Merged key/query head count.
     * @param value_heads Local split-output head count.
     * @param key_width Key/query head width.
     * @param value_width Value head width.
     * @return Split elements per physical request, or zero for invalid geometry.
     *
     * Factories already reject row-stride overflow. Check products before sums
     * so malformed head dimensions cannot wrap a launch grid or workspace index.
     * This describes coordinates only; allocation admission remains PMA-owned.
     */
    [[nodiscard]] inline int gdnDeinterleaveElementsPerRequest(
        DeviceRequestRowRanges rows, int key_heads, int value_heads,
        int key_width, int value_width) noexcept
    {
        constexpr auto maximum = std::numeric_limits<int>::max();
        if (key_heads <= 0 || value_heads <= 0 || key_width <= 0 || value_width <= 0 ||
            key_heads > maximum / key_width || value_heads > maximum / key_width ||
            value_heads > maximum / value_width)
            return 0;
        const auto query = std::int64_t{key_heads} * key_width;
        const auto split_query = std::int64_t{value_heads} * key_width;
        const auto value = std::int64_t{value_heads} * value_width;
        const auto source_width = 2 * query + value;
        const auto output_width = 2 * split_query + value;
        if (source_width > maximum || output_width > maximum ||
            source_width * rows.physicalRows() > maximum ||
            output_width * rows.physicalRows() > maximum)
            return 0;
        return static_cast<int>(output_width * rows.rowsPerRequest());
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    /**
     * @brief Copy one worker's live request coordinates without touching padding.
     * @param merged Ordered row-major Q/K/V input.
     * @param out_q Physical Q plane.
     * @param out_k Physical K plane.
     * @param out_v Physical V plane.
     * @param rows Frozen request coordinates and borrowed device length owner.
     * @param key_heads Merged query/key heads.
     * @param value_heads Local value/output heads.
     * @param key_width Query/key dimensions.
     * @param value_width Value dimensions.
     * @param global_value_offset Global modular repeat coordinate.
     *
     * Grid Y owns a request, so its count is uniform within a workgroup. Grid X
     * visits only that request's compact live prefix; destinations retain the
     * original physical row coordinate. No inactive merged row is fetched and
     * no inactive output is cleared. Negative offsets retain positive modulo.
     */
    __device__ inline __attribute__((always_inline)) void gdnDeinterleaveLiveRows(
        const float *merged, float *out_q, float *out_k, float *out_v,
        DeviceRequestRowRanges rows, int key_heads, int value_heads,
        int key_width, int value_width, int global_value_offset)
    {
        const int request = static_cast<int>(blockIdx.y);
        const int live_rows = rows.activeRows(request);
        const int query_source_width = key_heads * key_width;
        const int query_output_width = value_heads * key_width;
        const int value_output_width = value_heads * value_width;
        const int source_width = 2 * query_source_width + value_output_width;
        const int output_width = 2 * query_output_width + value_output_width;
        const auto live_elements = static_cast<std::size_t>(live_rows) * output_width;
        const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < live_elements; index += step)
        {
            // Host admission proves the complete physical extent fits INT32.
            // Narrow before division: copying floats must not introduce costly
            // 64-bit division into every thread's address calculation.
            const int element_index = static_cast<int>(index);
            const int local_row = element_index / output_width;
            const int row = request * rows.rowsPerRequest() + local_row;
            const int element = element_index % output_width;
            const auto source = static_cast<std::size_t>(row) * source_width;
            if (element < 2 * query_output_width)
            {
                const bool key = element >= query_output_width;
                const int column = key ? element - query_output_width : element;
                const int head = column / key_width;
                const int within_head = column % key_width;
                int source_head = static_cast<int>(
                    (std::int64_t{head} + global_value_offset) % key_heads);
                if (source_head < 0) source_head += key_heads;
                const auto destination = static_cast<std::size_t>(row) * query_output_width + column;
                const auto source_column = source_head * key_width + within_head +
                    (key ? query_source_width : 0);
                (key ? out_k : out_q)[destination] = merged[source + source_column];
            }
            else
            {
                const int column = element - 2 * query_output_width;
                out_v[static_cast<std::size_t>(row) * value_output_width + column] =
                    merged[source + 2 * query_source_width + column];
            }
        }
    }
#endif
}
