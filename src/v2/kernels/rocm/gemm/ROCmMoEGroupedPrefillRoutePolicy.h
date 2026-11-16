#ifndef LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
#define LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H

/**
 * @file ROCmMoEGroupedPrefillRoutePolicy.h
 * @brief Typed capture capacity and live route strategy for grouped MoE.
 *
 * Route-owned projections minimize directory and publication overhead for
 * sparse verifier buckets. Expert-tiled projections amortize weight decode
 * when many rows reuse an expert. The crossover depends on the complete MoE
 * operation and topology, not M alone, so measured production exceptions are represented as
 * exact immutable keys while the conservative generic policy remains total.
 * Capture retains all reachable families; the ordered device group plan
 * admits exactly one on replay. The arithmetic is also host-callable so unit
 * tests can prove policy totality without occupying a GPU.
 */

#pragma once

#include "../../common/MoEPrefillProjectionExecution.h"

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
        MoEPrefillMeasuredOperation operation = MoEPrefillMeasuredOperation::Unmeasured;
            ///< Exact exceptions require the measured complete/publication/scope identity.

        /** @return true only when every capture-identity field is equal. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr bool operator==(
            const ROCmMoEGroupedPrefillRouteKey& other) const noexcept
        {
            return gateup_codebook == other.gateup_codebook &&
                   down_codebook == other.down_codebook &&
                   hidden_size == other.hidden_size &&
                   expert_width == other.expert_width &&
                   expert_count == other.expert_count &&
                   top_k == other.top_k && rows == other.rows && operation == other.operation;
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
                    .gateup_codebook = 19, // NativeQuantised Q8_0 execution id.
                    .down_codebook = 19,
                    .hidden_size = 3072,
                    .expert_width = 1024,
                    .expert_count = 256,
                    .top_k = 1,
                    .rows = 16,
                    .operation = MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
                },
                .strategy = ROCmMoEGroupedPrefillRouteStrategy::RouteOwned,
            },
            {
                .key = {
                    .gateup_codebook = 19, // NativeQuantised Q8_0 execution id.
                    .down_codebook = 19,
                    .hidden_size = 3072,
                    .expert_width = 1024,
                    .expert_count = 256,
                    .top_k = 1,
                    .rows = 32,
                    .operation = MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
                },
                .strategy = ROCmMoEGroupedPrefillRouteStrategy::RouteOwned,
            },
        }};

    /** @brief Canonical generic crossover, shared by the scalar oracle and capture lowering. */
    inline constexpr int kROCmMoEGenericRouteOwnedRows = 8;

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
            .strategy = key.rows <= kROCmMoEGenericRouteOwnedRows
                            ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                            : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled,
            .exact = false,
        };
    }

    /** @brief Capture-time lowering of the canonical selector to device slot intervals.
     *
     * Only immutable geometry is resolved here. The device still owns the live
     * count, but each CTA avoids division and repeated full-key matching. Exact
     * entries retain table order, their original strategy and their complete
     * non-row identity. No alternative policy or device-state mirror exists. */
    class ROCmMoELiveSlotPolicy final
    {
    public:
        /** @brief Lower the original selector without reading a route publication.
         * @param key Complete original capacity/format/operation identity.
         * @return Sealed interval policy covering every valid live slot count.
         * @throws std::invalid_argument For invalid or unrepresentable geometry. */
        [[nodiscard]] static ROCmMoELiveSlotPolicy lower(ROCmMoEGroupedPrefillRouteKey key)
        {
            const auto capacity = static_cast<int64_t>(key.rows) * key.top_k;
            if (!selectROCmMoEGroupedPrefillRouteStrategy(key).valid() || capacity > INT32_MAX)
                throw std::invalid_argument("MoE live-slot policy requires representable original geometry");
            ROCmMoELiveSlotPolicy result;
            result.capacity_ = static_cast<int>(capacity);
            const auto generic_limit = static_cast<int64_t>(kROCmMoEGenericRouteOwnedRows) * key.top_k;
            result.ordered_limit_ = static_cast<int>(capacity < generic_limit ? capacity : generic_limit);
            auto empty = key;
            empty.rows = 1;
            result.empty_ = selectROCmMoEGroupedPrefillRouteStrategy(empty).strategy;
            for (const auto &entry : kROCmMoEGroupedPrefillExactRoutePolicies)
            {
                auto candidate = key;
                candidate.rows = entry.key.rows;
                if (candidate != entry.key || entry.key.rows <= 0 || entry.key.rows > key.rows) continue;
                if (entry.strategy != ROCmMoEGroupedPrefillRouteStrategy::RouteOwned &&
                    entry.strategy != ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)
                    throw std::invalid_argument("MoE exact live-slot policy names an invalid strategy");
                const auto first = static_cast<int64_t>(entry.key.rows - 1) * key.top_k;
                const auto last = static_cast<int64_t>(entry.key.rows) * key.top_k;
                result.exact_[result.exact_count_++] = {static_cast<int>(first),static_cast<int>(last),entry.strategy};
            }
            return result;
        }

        /** @brief Apply the lowered policy to the device's current slot count.
         * @param slots Original active count, including nonmultiples of top-k.
         * @return The original selector's decision, or Invalid outside admitted capacity. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr ROCmMoEGroupedPrefillRouteStrategy choose(int slots) const noexcept
        {
            if (slots < 0 || slots > capacity_) return ROCmMoEGroupedPrefillRouteStrategy::Invalid;
            if (slots == 0) return empty_;
            const auto exact = matchExact(slots);
            if (exact != ROCmMoEGroupedPrefillRouteStrategy::Invalid) return exact;
            return slots <= ordered_limit_ ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                                          : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled;
        }

        /** @brief Validate a device interval before adding its signed endpoints.
         * @param offset Original nonnegative exclusive offset. @param count Original nonnegative count.
         * @return Whether the safe integer sum lies within the original admitted capacity. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr bool validSpan(int offset,int count) const noexcept
        { return offset >= 0 && count >= 0 && offset <= capacity_ && count <= capacity_-offset; }

    private:
        /** @brief Only lower() may create a complete immutable policy. */
        ROCmMoELiveSlotPolicy() = default;
        /** @brief One ordered exact override on (first,last], preserving canonical table priority. */
        struct ExactInterval
        {
            int first = 0, last = 0;
            ROCmMoEGroupedPrefillRouteStrategy strategy = ROCmMoEGroupedPrefillRouteStrategy::Invalid;
        };
        /** @brief Constant-index traversal avoids addressable private arrays in a native packet.
         * @tparam Index Canonical lowered table position. @param slots Current device count.
         * @return First matching exact strategy, or Invalid when generic selection owns it. */
        template<size_t Index = 0>
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr ROCmMoEGroupedPrefillRouteStrategy matchExact(int slots) const noexcept
        {
            if constexpr (Index < kROCmMoEGroupedPrefillExactRoutePolicies.size())
            {
                if (Index >= exact_count_) return ROCmMoEGroupedPrefillRouteStrategy::Invalid;
                const auto &entry = exact_[Index];
                if (slots > entry.first && slots <= entry.last) return entry.strategy;
                return matchExact<Index+1>(slots);
            }
            return ROCmMoEGroupedPrefillRouteStrategy::Invalid;
        }
        int capacity_ = 0, ordered_limit_ = 0;
        std::array<ExactInterval,kROCmMoEGroupedPrefillExactRoutePolicies.size()> exact_{};
        static_assert(kROCmMoEGroupedPrefillExactRoutePolicies.size() <= UINT8_MAX);
        uint8_t exact_count_ = 0;
        ROCmMoEGroupedPrefillRouteStrategy empty_ = ROCmMoEGroupedPrefillRouteStrategy::Invalid;
    };

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
            : key_(key), counts_(counts), offsets_(offsets), slot_policy_(ROCmMoELiveSlotPolicy::lower(key))
        {
            if (!counts || !offsets)
                throw std::invalid_argument("MoE route admission requires a complete group publication");
        }

        /** @param key Complete expected immutable route policy.
         * @param counts Expected authoritative device counts. @param offsets Their exclusive prefixes.
         * @return Whether this packet borrows exactly that policy/publication, without reading device state. */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr bool binds(
            const ROCmMoEGroupedPrefillRouteKey &key,const int *counts,const int *offsets) const noexcept
        { return key_ == key && counts_ == counts && offsets_ == offsets; }

        /**
         * @brief Resolve the pure policy from a validated active-slot count.
         * @param slots Sum of all expert counts, never the allocated slot count.
         * @return Chosen strategy, or Invalid for a corrupt publication.
         */
        [[nodiscard]] LLAMINAR_MOE_ROUTE_HD constexpr
            ROCmMoEGroupedPrefillRouteStrategy strategyForSlots(int slots) const noexcept
        {
            return slot_policy_.choose(slots);
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
            return maximumSlotsFor(strategy) > 0;
        }

        /**
         * @brief Bound a retained family's grid by its reachable compact work.
         * @param strategy Projection family whose immutable launch is planned.
         * @return Largest admitted compact prefix, or zero for an absent family.
         *
         * The selector, including every exact-key exception, owns this bound.
         * A sparse family must not launch the full physical matrix merely
         * because another family can consume that capacity. This examines no
         * device state; replay still selects the family through allows().
         */
        [[nodiscard]] int maximumSlotsFor(
            ROCmMoEGroupedPrefillRouteStrategy strategy) const noexcept
        {
            for (int rows = key_.rows; rows > 0; --rows)
            {
                auto live = key_;
                live.rows = rows;
                if (selectROCmMoEGroupedPrefillRouteStrategy(live).strategy == strategy)
                    return rows * key_.top_k;
            }
            return 0;
        }

#if defined(__HIPCC__)
        /**
         * @brief Read the live compact prefix owned by the device group plan.
         * @return Number of initialized grouped rows on this replay.
         *
         * Projections and elementwise consumers borrow the same publication;
         * captured capacity is never evidence that a scratch row is valid.
         * A malformed prefix traps before any consumer can access scratch.
         */
        [[nodiscard]] __device__ __forceinline__ int activeSlots() const
        {
            const int last = key_.expert_count - 1;
            const int offset = __builtin_amdgcn_readfirstlane(offsets_[last]);
            const int count = __builtin_amdgcn_readfirstlane(counts_[last]);
            if (!slot_policy_.validSpan(offset,count)) __builtin_trap();
            return offset + count;
        }

        /**
         * @brief Resolve a valid compact row to its canonical expert owner.
         * @param slot Compact row strictly below activeSlots().
         * @return Expert containing the row, skipping empty expert intervals.
         *
         * All lanes in the caller's workgroup resolve the same immutable row.
         * Binary search reads the existing ordered group plan; no inverse map,
         * extra allocation, or host mirror is introduced. Invalid publications
         * trap before a weight descriptor can be dereferenced.
         */
        [[nodiscard]] __device__ __forceinline__ int expertForSlot(int slot) const
        {
            int first = 0;
            int end = key_.expert_count;
            while (first < end)
            {
                const int middle = first + (end - first) / 2;
                const int offset = __builtin_amdgcn_readfirstlane(offsets_[middle]);
                const int count = __builtin_amdgcn_readfirstlane(counts_[middle]);
                const auto past = static_cast<std::int64_t>(offset) + count;
                if (offset < 0 || count < 0 || past > INT32_MAX)
                    __builtin_trap();
                if (past <= slot)
                    first = middle + 1;
                else
                    end = middle;
            }
            if (slot < 0 || first >= key_.expert_count ||
                __builtin_amdgcn_readfirstlane(offsets_[first]) > slot)
                __builtin_trap();
            return first;
        }

        /**
         * @brief Admit one uniform workgroup before any barrier or weight load.
         * @param strategy This kernel's physical projection/publication family.
         * @return True only for the unique selected family on this replay.
         */
        [[nodiscard]] __device__ __forceinline__ bool allows(
            ROCmMoEGroupedPrefillRouteStrategy strategy) const
        {
            const auto selected = strategyForSlots(activeSlots());
            if (selected == ROCmMoEGroupedPrefillRouteStrategy::Invalid)
                __builtin_trap();
            return selected == strategy;
        }
#endif

    private:
        ROCmMoEGroupedPrefillRouteKey key_; ///< Complete immutable capture key.
        const int* counts_; ///< Borrowed ordered group-plan publication.
        const int* offsets_; ///< Exclusive prefixes owned by the same producer.
        ROCmMoELiveSlotPolicy slot_policy_; ///< Sealed lowering of key_, never live device state.
    };
} // namespace llaminar2::rocm

#undef LLAMINAR_MOE_ROUTE_HD

#endif // LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
