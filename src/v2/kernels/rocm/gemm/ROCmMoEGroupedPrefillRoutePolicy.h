#ifndef LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
#define LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H

/**
 * @file ROCmMoEGroupedPrefillRoutePolicy.h
 * @brief Typed capture-time route strategy for ROCm grouped MoE prefill.
 *
 * Route-owned projections minimize directory and publication overhead for
 * sparse verifier buckets. Expert-tiled projections amortize weight decode
 * when many rows reuse an expert. The crossover depends on the complete MoE
 * topology, not M alone, so measured production exceptions are represented as
 * exact immutable keys while the conservative generic policy remains total.
 * This header is device-free so unit tests can lock the capture decision
 * without occupying a GPU.
 */

#pragma once

#include <array>
#include <cstdint>

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
        [[nodiscard]] constexpr bool operator==(
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
        [[nodiscard]] constexpr bool valid() const noexcept
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
     * @brief Select the graph topology for one complete grouped-MoE identity.
     *
     * Exact measured entries take precedence. The total generic policy keeps
     * M<=8 route-owned and uses expert tiling above that crossover, matching
     * the pre-existing all-format behavior for every untrained topology.
     *
     * @param key Complete immutable graph identity.
     * @return Executable strategy and whether an exact measured entry won.
     */
    [[nodiscard]] inline constexpr ROCmMoEGroupedPrefillRouteDecision
    selectROCmMoEGroupedPrefillRouteStrategy(
        const ROCmMoEGroupedPrefillRouteKey& key) noexcept
    {
        if (key.hidden_size <= 0 || key.expert_width <= 0 ||
            key.expert_count <= 0 || key.top_k <= 0 ||
            key.top_k > key.expert_count || key.rows <= 0)
        {
            return {};
        }

        for (const auto& policy : kROCmMoEGroupedPrefillExactRoutePolicies)
        {
            if (policy.key == key)
                return {.strategy = policy.strategy, .exact = true};
        }

        return {
            .strategy = key.rows <= 8
                            ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                            : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled,
            .exact = false,
        };
    }
} // namespace llaminar2::rocm

#endif // LLAMINAR2_KERNELS_ROCM_GEMM_ROCMMOEGROUPEDPREFILLROUTEPOLICY_H
