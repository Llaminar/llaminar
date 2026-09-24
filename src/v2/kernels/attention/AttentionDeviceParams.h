/**
 * @file AttentionDeviceParams.h
 * @brief Shared device-owned attention replay parameters and grouped-row policy.
 *
 * CUDA and ROCm attention graph replay consume this compact parameter block
 * directly from device memory.  The header also owns the maximum grouped MTP
 * verifier span so graph preparation, backend validation, device writers, and
 * workspace planning cannot silently drift to different row capacities.
 */

#pragma once

#include <algorithm>

namespace llaminar2
{
    namespace attention
    {
        /**
         * @brief Maximum number of target-model rows in one grouped verifier pass.
         *
         * An MTP draft depth of up to fifteen produces sixteen target-model
         * rows: the current token plus each drafted continuation. Production
         * GPU attention therefore reserves sixteen row-local parameter records
         * and sixteen rows of split-decode partials. The canonical kernel sweep
         * additionally probes M=31 as a stress geometry, but that is not a
         * production speculative-depth promise.
         */
        inline constexpr int kMaxGroupedVerifierAttentionRows = 16;

        /**
         * @brief Independent cardinalities used by persistent attention scratch.
         *
         * `graph_query_rows` is intentionally absent from this result.
         * Prompt-prefill M describes queries within each request; it must never
         * multiply request-major K/V conversion storage. Split-decode scratch
         * reserves the complete grouped-verifier capacity for every graph
         * family member so decode capture does not depend on which prompt
         * shape happened to materialize first.
         */
        struct AttentionWorkspaceCardinality
        {
            int compact_query_rows = 1; ///< Rows of split-decode partials.
            int request_count = 1;      ///< Independent request-major K/V banks.
        };

        /**
         * @brief Plan attention scratch without conflating prompt M and requests.
         *
         * @param graph_query_rows Declared graph M. It is accepted to make the
         *        discarded axis explicit and regression-testable.
         * @param stage_request_count Number of independent request sequences.
         * @return Fixed compact-row capacity plus positive request cardinality.
         */
        [[nodiscard]] inline constexpr AttentionWorkspaceCardinality
        planAttentionWorkspaceCardinality(
            int graph_query_rows,
            int stage_request_count) noexcept
        {
            (void)graph_query_rows;
            const int request_count =
                std::max(1, stage_request_count);
            return AttentionWorkspaceCardinality{
                .compact_query_rows = std::max(
                    kMaxGroupedVerifierAttentionRows,
                    request_count),
                .request_count = request_count,
            };
        }

        /**
         * @brief Row-local dynamic geometry consumed by graph-captured attention.
         *
         * Device kernels publish these fields immediately before attention
         * replay.  Keeping the block device-owned avoids a host mirror whose
         * lifetime or ordering could disagree with the captured GPU graph.
         */
        struct AttentionDeviceParams
        {
            int kv_len = 0;          ///< Visible KV positions for this query row.
            int kv_stride = 0;       ///< Physical request-major K/V row capacity.
            int position_offset = 0; ///< Absolute model position of this query row.
            int mask_stride = 0;     ///< Row stride of the optional attention mask.
            int ring_row_origin = 0; ///< Physical row containing logical KV row zero.
            int ring_row_capacity = 0; ///< Ring modulus; zero names contiguous K/V.
        };
    } // namespace attention
} // namespace llaminar2
