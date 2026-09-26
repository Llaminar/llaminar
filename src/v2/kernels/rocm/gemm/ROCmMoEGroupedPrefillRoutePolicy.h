#ifndef LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
#define LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H

/**
 * @file ROCmMoEGroupedPrefillRoutePolicy.h
 * @brief Typed capture capacity and live route strategy for grouped MoE.
 *
 * Route-owned projections minimize directory and publication overhead for
 * sparse verifier buckets. Expert-tiled projections amortize weight decode
 * when many rows reuse an expert. The crossover depends on the complete MoE
 * topology, not M alone, so measured production exceptions are represented as
 * exact immutable keys while the conservative generic policy remains total.
 * Capture retains all reachable families; the ordered device group plan
 * admits exactly one on replay. The arithmetic is also host-callable so unit
 * tests can prove policy totality without occupying a GPU.
 */

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#if defined(__HIPCC__)
#define LLAMINAR_MOE_ROUTE_HD __host__ __device__
#else
#define LLAMINAR_MOE_ROUTE_HD
#endif

namespace llaminar2::rocm
{
    /** Projection/publication topology captured by one grouped MoE graph. */
    enum class ROCmMoEGroupedPrefillRouteStrategy : std::uint8_t
    {
        Invalid = 0,   ///< Malformed dimensions; graph construction must fail.
        RouteOwned,    ///< One compact route owns each projection/publication.
        ExpertTiled,   ///< Device directories group rows to reuse expert weights.
    };

    /** Complete immutable identity for a measured production route decision. */
    struct ROCmMoEGroupedPrefillRouteKey
    {
        std::uint8_t gateup_codebook = 0;
        std::uint8_t down_codebook = 0;
        std::int32_t hidden_size = 0;
        std::int32_t expert_width = 0;
        std::int32_t expert_count = 0;
        std::int32_t top_k = 0;
        std::int32_t rows = 0;

        /** @return true only when every capture-identity field is equal. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr bool operator==(
            const ROCmMoEGroupedPrefillRouteKey& other) const noexcept
        {
            return gateup_codebook == other.gateup_codebook &&
                   down_codebook == other.down_codebook &&
                   hidden_size == other.hidden_size &&
                   expert_width == other.expert_width &&
                   expert_count == other.expert_count &&
                   top_k == other.top_k && rows == other.rows;
        }
    };

    /** Strategy plus provenance needed by diagnostics and policy tests. */
    struct ROCmMoEGroupedPrefillRouteDecision
    {
        ROCmMoEGroupedPrefillRouteStrategy strategy =
            ROCmMoEGroupedPrefillRouteStrategy::Invalid;
        bool exact = false; ///< Whether a measured full-key entry selected it.

        /** @return true when the decision names an executable graph topology. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr bool valid() const noexcept
        {
            return strategy != ROCmMoEGroupedPrefillRouteStrategy::Invalid;
        }
    };

    /** One measured full-key exception to the total generic crossover. */
    struct ROCmMoEGroupedPrefillExactRoutePolicy
    {
        ROCmMoEGroupedPrefillRouteKey key{};
        ROCmMoEGroupedPrefillRouteStrategy strategy =
            ROCmMoEGroupedPrefillRouteStrategy::Invalid;
    };

    /**
     * Exact Qwen3.5-122B sparse endpoint policies measured on gfx906.
     *
     * The retained endpoint graph family uses powers-of-two capacities. At
     * top-k=1, route-owned Q8_0 was 2.35x faster at M=16 and 1.88x faster at
     * M=32 than the expert-tiled path while remaining bit-identical. M<=8 is
     * already covered by the generic low-latency policy; larger or different
     * topologies remain tiled until independently trained.
     */
    inline constexpr std::array<ROCmMoEGroupedPrefillExactRoutePolicy, 2>
        kROCmMoEGroupedPrefillExactRoutePolicies{{
            {
                .key = {
                    .gateup_codebook = 19, // NativeVNNI Q8_0 execution id.
                    .down_codebook = 19,
                    .hidden_size = 3072,
                    .expert_width = 1024,
                    .expert_count = 256,
                    .top_k = 1,
                    .rows = 16,
                },
                .strategy = ROCmMoEGroupedPrefillRouteStrategy::RouteOwned,
            },
            {
                .key = {
                    .gateup_codebook = 19, // NativeVNNI Q8_0 execution id.
                    .down_codebook = 19,
                    .hidden_size = 3072,
                    .expert_width = 1024,
                    .expert_count = 256,
                    .top_k = 1,
                    .rows = 32,
                },
                .strategy = ROCmMoEGroupedPrefillRouteStrategy::RouteOwned,
            },
        }};

    /**
     * @brief Match immutable exact keys without a host-table device pointer.
     * @tparam Index Compile-time table position; the small table is unrolled.
     * @param key Complete execution geometry, with live rather than padded rows.
     * @return Exact strategy, or Invalid when generic selection owns this key.
     */
    template <std::size_t Index = 0>
    [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr
        ROCmMoEGroupedPrefillRouteStrategy
    matchROCmMoEGroupedPrefillExactRoute(
        const ROCmMoEGroupedPrefillRouteKey& key) noexcept
    {
        if constexpr (Index < kROCmMoEGroupedPrefillExactRoutePolicies.size())
        {
            constexpr auto entry = kROCmMoEGroupedPrefillExactRoutePolicies[Index];
            if (entry.key == key)
                return entry.strategy;
            return matchROCmMoEGroupedPrefillExactRoute<Index + 1>(key);
        }
        return ROCmMoEGroupedPrefillRouteStrategy::Invalid;
    }

    /**
     * @brief Select the graph topology for one complete grouped-MoE identity.
     *
     * Exact measured entries take precedence. The total generic policy keeps
     * M<=8 route-owned and uses expert tiling above that crossover, matching
     * the pre-existing all-format behavior for every untrained topology.
     *
     * @param key Complete immutable graph identity.
     * @return Executable strategy and whether an exact measured entry won.
     */
    [[nodiscard]] LLAMINAR_MOE_ROUTE_HD inline constexpr ROCmMoEGroupedPrefillRouteDecision
    selectROCmMoEGroupedPrefillRouteStrategy(
        const ROCmMoEGroupedPrefillRouteKey& key) noexcept
    {
        if (key.hidden_size <= 0 || key.expert_width <= 0 ||
            key.expert_count <= 0 || key.top_k <= 0 ||
            key.top_k > key.expert_count || key.rows <= 0)
        {
            return {};
        }

        const auto exact = matchROCmMoEGroupedPrefillExactRoute(key);
        if (exact != ROCmMoEGroupedPrefillRouteStrategy::Invalid)
            return {.strategy = exact, .exact = true};

        return {
            .strategy = key.rows <= 8
                            ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                            : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled,
            .exact = false,
        };
    }

    /**
     * @brief Replay-time route admission from the already published group plan.
     *
     * A captured matrix's row count is capacity, not the amount of expert work.
     * Padding and participant-local masks can leave only a few valid routes in
     * that matrix. The final exclusive offset plus its expert count is the
     * canonical active-route count; borrowing those two words creates no new
     * ledger, readback, allocation, or synchronization edge.
     *
     * Both projection families use this same immutable predicate. Exactly one
     * may write scratch/output on a replay. The current measured selector is
     * evaluated on the ceiling of active routes / top-k, preserving its full
     * key and every existing exact exception. Physical grids and strides never
     * change. Fully populated graphs make exactly the previous decision.
     */
    class ROCmMoEGroupedRouteAdmission final
    {
    public:
        /**
         * @brief Borrow the ordered group publication before graph capture.
         * @param key Physical capacity and immutable weight/topology geometry.
         * @param counts Device expert counts, in ascending expert-id order.
         * @param offsets Device exclusive prefix offsets in that same order.
         * @throws std::invalid_argument For missing publication or geometry.
         */
        ROCmMoEGroupedRouteAdmission(ROCmMoEGroupedPrefillRouteKey key,
                                    const int* counts, const int* offsets)
            : key_(key), counts_(counts), offsets_(offsets)
        {
            if (!counts || !offsets ||
                !selectROCmMoEGroupedPrefillRouteStrategy(key).valid())
                throw std::invalid_argument("MoE route admission requires a complete group publication");
        }

        /**
         * @brief Resolve the pure policy from a validated active-slot count.
         * @param slots Sum of all expert counts, never the allocated slot count.
         * @return Chosen strategy, or Invalid for a corrupt publication.
         */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr
            ROCmMoEGroupedPrefillRouteStrategy strategyForSlots(int slots) const noexcept
        {
            if (slots < 0 || static_cast<std::int64_t>(slots) >
                    static_cast<std::int64_t>(key_.rows) * key_.top_k)
                return ROCmMoEGroupedPrefillRouteStrategy::Invalid;
            // An empty participant still uses the compact publisher to zero
            // its output; it performs no expert dot products.
            auto live = key_;
            live.rows = slots == 0 ? 1 : 1 + (slots - 1) / key_.top_k;
            return selectROCmMoEGroupedPrefillRouteStrategy(live).strategy;
        }

        /**
         * @brief Decide which families must exist in the retained graph.
         * @param strategy Candidate family to include.
         * @return Whether any admitted live route count can select this family.
         *
         * This walks immutable geometry only. Capture must not inspect counts.
         */
        [[nodiscard]] bool maySelect(ROCmMoEGroupedPrefillRouteStrategy strategy) const noexcept
        {
            for (int rows = 1; rows <= key_.rows; ++rows)
            {
                auto live = key_;
                live.rows = rows;
                if (selectROCmMoEGroupedPrefillRouteStrategy(live).strategy == strategy)
                    return true;
            }
            return false;
        }

#if defined(__HIPCC__)
        /**
         * @brief Admit one uniform workgroup before any barrier or weight load.
         * @param strategy This kernel's physical projection/publication family.
         * @return True only for the unique selected family on this replay.
         */
        [[nodiscard]] __device__ __forceinline__ bool allows(
            ROCmMoEGroupedPrefillRouteStrategy strategy) const
        {
            const int last = key_.expert_count - 1;
            const int offset = __builtin_amdgcn_readfirstlane(offsets_[last]);
            const int count = __builtin_amdgcn_readfirstlane(counts_[last]);
            if (offset < 0 || count < 0 ||
                static_cast<std::int64_t>(offset) + count > INT32_MAX)
                __builtin_trap();
            const auto selected = strategyForSlots(offset + count);
            if (selected == ROCmMoEGroupedPrefillRouteStrategy::Invalid)
                __builtin_trap();
            return selected == strategy;
        }
#endif

    private:
        ROCmMoEGroupedPrefillRouteKey key_; ///< Complete immutable capture key.
        const int* counts_; ///< Borrowed ordered group-plan publication.
        const int* offsets_; ///< Exclusive prefixes owned by the same producer.
    };
} // namespace llaminar2::rocm

#undef LLAMINAR_MOE_ROUTE_HD

#endif // LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
