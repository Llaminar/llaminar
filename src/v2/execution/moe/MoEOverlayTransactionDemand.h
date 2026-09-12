/**
 * @file MoEOverlayTransactionDemand.h
 * @brief Fixed-capacity transaction demand storage borrowed from an existing bank.
 *
 * The histogram generation owns publication, writer exclusion, reset and physical
 * memory. This view adds no lifecycle or allocation authority. It retains a whole
 * transaction crossing the requested sample size, then closes demand admission
 * until the parent rotates or increases its existing adaptive window budget.
 * Inference itself never waits for space.
 * Marginal counters must be updated only for a Recorded result so they describe
 * precisely the same sample, not later traffic omitted from transaction storage.
 */
#pragma once

#include "MoEOverlayTransactionCost.h"
#include "ExpertHistogramSource.h"
#include <cstdint>

#if defined(__CUDACC__)
#define LLAMINAR_OVERLAY_DEMAND_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_OVERLAY_DEMAND_HD __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_OVERLAY_DEMAND_HD inline
#endif

namespace llaminar2::moe_overlay_economy
{
    /** @brief One exact transaction in a layer's compact selected-expert array. */
    struct TransactionDemandRecord
    {
        std::uint64_t first_route_slot = 0;
        std::uint32_t logical_rows = 0;
        std::uint32_t top_k = 0;
        ExpertHistogramSource phase = ExpertHistogramSource::DecodeToken;
    };

    /** @brief Parent-generation-owned publication frontier, reset only when retired. */
    struct TransactionDemandFrontier
    {
        std::uint64_t route_slots = 0;
        std::uint32_t logical_rows = 0;
        std::uint32_t transactions = 0;
    };

    /** @brief Typed distinction between a complete sample and malformed evidence. */
    enum class TransactionDemandAppendStatus : std::uint32_t
    {
        Recorded,
        SampleComplete,
        InvalidGeometry,
        InvalidExpert,
        InvalidPhase,
        InvalidFrontier,
    };

    /**
     * @brief Exact storage geometry; callers contribute these bytes to the PMA BOM.
     *
     * target_rows is the admitted maximum, not a second live window policy.
     * At most target_rows transactions fit before closure (one row each). The
     * last transaction may cross the target but may never be split into serial
     * rows. Thus route storage covers target_rows + max_transaction_rows - 1.
     */
    struct TransactionDemandCapacity
    {
        std::uint32_t target_rows = 0;
        std::uint32_t max_transaction_rows = 0;
        std::uint32_t top_k = 0;

        /** @return Whether every declared row, slot and byte count is representable. */
        [[nodiscard]] LLAMINAR_OVERLAY_DEMAND_HD bool valid() const noexcept
        {
            if (target_rows == 0 || max_transaction_rows == 0 || top_k == 0) return false;
            const std::uint64_t rows = static_cast<std::uint64_t>(target_rows) + max_transaction_rows - 1u;
            if (rows > ~std::uint32_t{0}) return false;
            const std::uint64_t fixed = sizeof(TransactionDemandFrontier) +
                static_cast<std::uint64_t>(target_rows) * sizeof(TransactionDemandRecord);
            return routeSlots() <= (~std::uint64_t{0} - fixed) / sizeof(std::int32_t);
        }

        /** @return Exact compact route slots; call valid() before physical admission. */
        [[nodiscard]] LLAMINAR_OVERLAY_DEMAND_HD std::uint64_t routeSlots() const noexcept
        {
            if (target_rows == 0 || max_transaction_rows == 0 || top_k == 0) return 0;
            const std::uint64_t rows = static_cast<std::uint64_t>(target_rows) + max_transaction_rows - 1u;
            if (rows > ~std::uint32_t{0}) return 0;
            return rows * top_k;
        }

        /** @return Payload bytes for one layer/bank, or zero for invalid geometry. */
        [[nodiscard]] LLAMINAR_OVERLAY_DEMAND_HD std::uint64_t allocationBytes() const noexcept
        {
            if (!valid()) return 0;
            return sizeof(TransactionDemandFrontier) +
                static_cast<std::uint64_t>(target_rows) * sizeof(TransactionDemandRecord) +
                routeSlots() * sizeof(std::int32_t);
        }
    };

    /**
     * @brief Borrowed layer storage under the parent bank's exclusive writer lease.
     *
     * Host callers use the histogram's writer ownership; device callers use its
     * exact producer stream. No thread can reset or inspect a mutable bank here.
     * Pointers and capacity are graph identity and must remain stable throughout
     * capture/replay. The physical owner, not this view, allocates and releases.
     */
    struct TransactionDemandBank
    {
        TransactionDemandCapacity capacity;
        TransactionDemandFrontier *frontier = nullptr;
        TransactionDemandRecord *transactions = nullptr;
        std::int32_t *expert_ids = nullptr;

        /**
         * @brief Retain all logical rows or leave the bank completely unchanged.
         * @param phase Actual production transaction phase, not inferred from M.
         * @param routes Borrowed padded input; only logical selected IDs are copied.
         * @param expert_count Model expert-ID upper bound, independent of topology.
         * @param sample_target_rows Existing parent's live window policy, bounded
         *        by the admitted capacity. It is not copied into another controller.
         * @return Recorded (also admit marginal counters), SampleComplete (no demand
         *         mutation), or a fatal geometry/frontier/route diagnostic.
         */
        [[nodiscard]] LLAMINAR_OVERLAY_DEMAND_HD TransactionDemandAppendStatus append(
            ExpertHistogramSource phase, TransactionRoutes routes,
            std::uint32_t expert_count, std::uint32_t sample_target_rows) const noexcept
        {
            if (phase != ExpertHistogramSource::DecodeToken &&
                phase != ExpertHistogramSource::PrefillChunk &&
                phase != ExpertHistogramSource::GroupedVerifier)
                return TransactionDemandAppendStatus::InvalidPhase;
            if (!capacity.valid() || !frontier || !transactions || !expert_ids ||
                !routes.expert_ids || expert_count == 0 || capacity.top_k > expert_count ||
                sample_target_rows == 0 || sample_target_rows > capacity.target_rows ||
                routes.logical_rows == 0 || routes.logical_rows > capacity.max_transaction_rows ||
                (phase == ExpertHistogramSource::DecodeToken && routes.logical_rows != 1) ||
                routes.top_k != capacity.top_k || routes.row_stride < routes.top_k)
                return TransactionDemandAppendStatus::InvalidGeometry;
            const std::uint64_t required_slots =
                static_cast<std::uint64_t>(routes.logical_rows - 1u) * routes.row_stride + routes.top_k;
            if (required_slots > routes.capacity_slots)
                return TransactionDemandAppendStatus::InvalidGeometry;

            // A corrupt publication frontier must not look like normal sample
            // closure. These O(1) relationships are maintained by the sole writer.
            if (frontier->route_slots > capacity.routeSlots() ||
                frontier->route_slots != static_cast<std::uint64_t>(frontier->logical_rows) * capacity.top_k ||
                frontier->transactions > capacity.target_rows ||
                frontier->transactions > frontier->logical_rows ||
                ((frontier->transactions == 0) != (frontier->logical_rows == 0)))
                return TransactionDemandAppendStatus::InvalidFrontier;
            if (frontier->transactions != 0)
            {
                const auto &last = transactions[frontier->transactions - 1u];
                const auto last_slots = static_cast<std::uint64_t>(last.logical_rows) * last.top_k;
                if (last.logical_rows == 0 || last.top_k != capacity.top_k ||
                    last_slots > frontier->route_slots ||
                    last.first_route_slot != frontier->route_slots - last_slots)
                    return TransactionDemandAppendStatus::InvalidFrontier;
            }

            // Authenticate all logical IDs before mutating a single destination
            // byte. Poisoned bucket/stride padding is intentionally not inspected.
            for (std::uint32_t row = 0; row < routes.logical_rows; ++row)
                for (std::uint32_t slot = 0; slot < routes.top_k; ++slot)
                {
                    const auto expert = routes.expert_ids[static_cast<std::uint64_t>(row) * routes.row_stride + slot];
                    if (expert < 0 || static_cast<std::uint32_t>(expert) >= expert_count)
                        return TransactionDemandAppendStatus::InvalidExpert;
                }
            if (frontier->logical_rows >= sample_target_rows)
                return TransactionDemandAppendStatus::SampleComplete;

            const std::uint64_t copied_slots = static_cast<std::uint64_t>(routes.logical_rows) * routes.top_k;
            if (frontier->transactions >= capacity.target_rows ||
                copied_slots > capacity.routeSlots() - frontier->route_slots)
                return TransactionDemandAppendStatus::InvalidFrontier;
            const auto first_slot = frontier->route_slots;
            for (std::uint32_t row = 0; row < routes.logical_rows; ++row)
                for (std::uint32_t slot = 0; slot < routes.top_k; ++slot)
                    expert_ids[first_slot + static_cast<std::uint64_t>(row) * routes.top_k + slot] =
                        routes.expert_ids[static_cast<std::uint64_t>(row) * routes.row_stride + slot];

            // Payload precedes its descriptor and frontier. Visibility to readers
            // comes from the parent's retired-bank/event edge, not from polling
            // these non-atomic fields or adding an independent publication flag.
            transactions[frontier->transactions] = {first_slot, routes.logical_rows, routes.top_k, phase};
            frontier->route_slots += copied_slots;
            frontier->logical_rows += routes.logical_rows;
            ++frontier->transactions;
            return TransactionDemandAppendStatus::Recorded;
        }
    };
}

#undef LLAMINAR_OVERLAY_DEMAND_HD
