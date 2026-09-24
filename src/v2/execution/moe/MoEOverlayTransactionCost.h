/**
 * @file MoEOverlayTransactionCost.h
 * @brief Allocation-free host/device arithmetic for transaction-shaped MoE cost.
 *
 * One transaction runs expert participants concurrently; successive transactions
 * cannot overlap their dependent model state. Its modeled service is therefore
 * the maximum participant work, and a window is the sum of those maxima. The
 * measured prices are inputs, not timings collected by this helper. No model
 * tensor, placement, lifecycle state, or memory ledger is owned here.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LLAMINAR_OVERLAY_COST_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_OVERLAY_COST_HD __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_OVERLAY_COST_HD inline
#endif

namespace llaminar2::moe_overlay_economy
{
    /** @brief Fatal input/arithmetic status, shared by host and device callers. */
    enum class TransactionCostStatus : std::uint32_t
    {
        Complete,
        InvalidGeometry,
        InvalidExpert,
        InvalidOwner,
        UnpricedParticipant,
        Overflow,
    };

    /** @brief Borrowed exact logical rows; padding is never routed demand. */
    struct TransactionRoutes
    {
        const std::int32_t *expert_ids = nullptr;
        std::uint64_t capacity_slots = 0;
        std::uint32_t logical_rows = 0;
        std::uint32_t top_k = 0;
        std::uint32_t row_stride = 0;
    };

    /** @brief One phase/layer's immutable owner maps and measured endpoint prices. */
    struct TransactionPlacementCosts
    {
        const std::int32_t *before_owners = nullptr;
        const std::int32_t *after_owners = nullptr;
        const std::uint64_t *participant_ns_per_activation = nullptr;
        std::uint32_t expert_count = 0;
        std::uint32_t participant_count = 0;
    };

    /** @brief Before/after work; scratch uses the same layout as a complete cost. */
    struct ServiceCostPair
    {
        std::uint64_t before_ns = 0;
        std::uint64_t after_ns = 0;
    };

    /** @brief A failed score never carries a publishable partial cost. */
    struct TransactionCostResult
    {
        TransactionCostStatus status = TransactionCostStatus::InvalidGeometry;
        ServiceCostPair cost;

        /** @return Whether both costs were completely and exactly accumulated. */
        [[nodiscard]] LLAMINAR_OVERLAY_COST_HD bool complete() const noexcept
        {
            return status == TransactionCostStatus::Complete;
        }
    };

    /**
     * @brief Price one real transaction without allocation or synchronization.
     * @param routes Exact unpadded row prefix and its physical storage contract.
     * @param placement Complete current/candidate maps and positive service prices.
     * @param scratch Invocation-exclusive storage for all physical participants;
     *        its contents are unspecified after failure and overwritten on entry.
     * @param scratch_count Number of available entries, not an inferred topology cap.
     * @return Complete parallel-participant maxima, or a fatal typed failure.
     *
     * Owner maps and prices remain borrowed and immutable. Callers must turn any
     * non-complete status into their normal fatal host/device diagnostic; neither
     * saturation nor a marginal-histogram estimate is a replacement result.
     */
    [[nodiscard]] LLAMINAR_OVERLAY_COST_HD TransactionCostResult scoreTransaction(
        TransactionRoutes routes, TransactionPlacementCosts placement,
        ServiceCostPair *scratch, std::uint32_t scratch_count) noexcept
    {
        if (!routes.expert_ids || !placement.before_owners || !placement.after_owners ||
            !placement.participant_ns_per_activation || !scratch ||
            routes.logical_rows == 0 || routes.top_k == 0 ||
            routes.top_k > placement.expert_count || routes.row_stride < routes.top_k ||
            placement.expert_count == 0 || placement.participant_count == 0 ||
            scratch_count < placement.participant_count)
            return {TransactionCostStatus::InvalidGeometry, {}};

        // All factors are uint32_t. Widen before multiplying: even the largest
        // legal last-row endpoint is representable in uint64_t.
        const std::uint64_t required_slots =
            static_cast<std::uint64_t>(routes.logical_rows - 1u) * routes.row_stride +
            routes.top_k;
        if (required_slots > routes.capacity_slots)
            return {TransactionCostStatus::InvalidGeometry, {}};
        for (std::uint32_t participant = 0; participant < placement.participant_count; ++participant)
        {
            if (placement.participant_ns_per_activation[participant] == 0)
                return {TransactionCostStatus::UnpricedParticipant, {}};
            scratch[participant] = {};
        }

        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        for (std::uint32_t row = 0; row < routes.logical_rows; ++row)
        {
            const auto *ids = routes.expert_ids + static_cast<std::uint64_t>(row) * routes.row_stride;
            for (std::uint32_t slot = 0; slot < routes.top_k; ++slot)
            {
                const std::int32_t expert = ids[slot];
                if (expert < 0 || static_cast<std::uint32_t>(expert) >= placement.expert_count)
                    return {TransactionCostStatus::InvalidExpert, {}};
                const std::int32_t before = placement.before_owners[expert];
                const std::int32_t after = placement.after_owners[expert];
                if (before < 0 || after < 0 ||
                    static_cast<std::uint32_t>(before) >= placement.participant_count ||
                    static_cast<std::uint32_t>(after) >= placement.participant_count)
                    return {TransactionCostStatus::InvalidOwner, {}};
                const auto before_price = placement.participant_ns_per_activation[before];
                const auto after_price = placement.participant_ns_per_activation[after];
                // Check both additions before publishing either scratch update.
                // Overflow cannot become a saturated but apparently cheap move.
                if (before_price > maximum - scratch[before].before_ns ||
                    after_price > maximum - scratch[after].after_ns)
                    return {TransactionCostStatus::Overflow, {}};
                scratch[before].before_ns += before_price;
                scratch[after].after_ns += after_price;
            }
        }

        ServiceCostPair critical;
        for (std::uint32_t participant = 0; participant < placement.participant_count; ++participant)
        {
            if (scratch[participant].before_ns > critical.before_ns)
                critical.before_ns = scratch[participant].before_ns;
            if (scratch[participant].after_ns > critical.after_ns)
                critical.after_ns = scratch[participant].after_ns;
        }
        return {TransactionCostStatus::Complete, critical};
    }

    /**
     * @brief Append one dependent transaction to a phase-pure service window.
     * @param window Caller-owned sum, changed only if the entire append succeeds.
     * @param transaction Complete score from the same phase, layer and prices.
     * @return Complete or the fatal input/overflow status; no partial sum escapes.
     */
    [[nodiscard]] LLAMINAR_OVERLAY_COST_HD TransactionCostStatus appendTransaction(
        ServiceCostPair &window, TransactionCostResult transaction) noexcept
    {
        if (!transaction.complete()) return transaction.status;
        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        if (transaction.cost.before_ns > maximum - window.before_ns ||
            transaction.cost.after_ns > maximum - window.after_ns)
            return TransactionCostStatus::Overflow;
        // A phase/window is sequential even when each participant set is parallel.
        // Commit both sums together so a failed append leaves the caller intact.
        window.before_ns += transaction.cost.before_ns;
        window.after_ns += transaction.cost.after_ns;
        return TransactionCostStatus::Complete;
    }
}

#undef LLAMINAR_OVERLAY_COST_HD
