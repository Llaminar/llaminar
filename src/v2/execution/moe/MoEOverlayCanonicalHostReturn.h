/**
 * @file MoEOverlayCanonicalHostReturn.h
 * @brief Gather sparse host returns into the existing canonical route format.
 *
 * This boundary performs no expert arithmetic and owns no allocation or
 * placement state. Graph-owned return stages append immutable raw expert rows
 * to an arena-owned packed bank. The existing MoECanonicalRouteReduceStage
 * validates complete original-slot coverage and performs the single ordered
 * fold after the final return. Thus transport order never defines FP32 order.
 */

#pragma once

#include "CanonicalMoERouteRecord.h"
#include "MoEOverlaySparseCollective.h"

#include <algorithm>
#include <span>

namespace llaminar2
{
    /** @brief Whether this graph edge opens or extends one packed publication. */
    enum class MoEOverlayCanonicalGatherBoundary : uint8_t
    {
        Begin,
        Append,
    };

    /**
     * @brief Append only live canonical rows to a stable graph-owned bank.
     * @param rows Authenticated transport payload; row IDs are original slots.
     * @param bank Packed FP32 storage including its count trailer.
     * @param boundary Begin resets the count; Append retains preceding returns.
     * @return False on an invalid contract, without changing the destination.
     *
     * Duplicate/missing route identities are validated together by the final
     * canonical reducer, which already owns that invariant for rooted LocalTP.
     * No participant may perform a partial weighted fold before that stage.
     */
    inline bool gatherMoEOverlayCanonicalHostReturn(
        const MoEOverlayReturnRows &rows,
        std::span<float> bank,
        MoEOverlayCanonicalGatherBoundary boundary)
    {
        namespace record = canonical_moe_route_record;
        if (rows.layout != MoEOverlayReturnLayout::CanonicalExpertRoutes ||
            rows.d_model <= 0 || !rows.row_ids_host || !rows.output_rows_fp32 ||
            rows.live_row_count > rows.row_capacity || bank.empty() ||
            (rows.live_row_count != 0 && rows.residency_epoch == 0) ||
            (boundary != MoEOverlayCanonicalGatherBoundary::Begin &&
             boundary != MoEOverlayCanonicalGatherBoundary::Append))
            return false;
        const size_t capacity = record::recordCapacity(bank.size(), rows.d_model);
        const size_t first = boundary == MoEOverlayCanonicalGatherBoundary::Begin
                                 ? 0u : record::readRecordCount(bank.data(), bank.size());
        if (first > capacity || rows.live_row_count > capacity - first ||
            first + rows.live_row_count > std::numeric_limits<uint32_t>::max())
            return false;
        for (size_t row = 0; row < rows.live_row_count; ++row)
            if (rows.row_ids_host[row] < 0)
                return false;

        // Validate the complete append before exposing a new count. The graph
        // dependency on the final reducer is the publication edge on CPU.
        for (size_t row = 0; row < rows.live_row_count; ++row)
        {
            float *destination = record::record(bank.data(), first + row, rows.d_model);
            std::copy_n(rows.output_rows_fp32 + row * static_cast<size_t>(rows.d_model),
                        rows.d_model, destination);
            record::writeFlatRouteSlot(destination, rows.d_model,
                                      static_cast<size_t>(rows.row_ids_host[row]));
        }
        return record::writeRecordCount(bank.data(), bank.size(), first + rows.live_row_count);
    }
} // namespace llaminar2
