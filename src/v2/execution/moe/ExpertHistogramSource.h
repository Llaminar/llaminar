/**
 * @file ExpertHistogramSource.h
 * @brief Canonical inference-phase identity for host and device routed demand.
 *
 * Marginal histograms, retained transactions and economy prices use this single
 * phase definition. It deliberately has no host histogram, allocator or runtime
 * dependency so native CUDA/HIP producers can use the same wire identity.
 */
#pragma once

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Inference phase that produced a routed-expert selection.
     *
     * Production phases have different batching and measured service costs.
     * SyntheticTest is an ingestion-only decode alias for device-free fixtures;
     * it is not a fourth production phase or a legal captured transaction tag.
     */
    enum class ExpertHistogramSource
    {
        DecodeToken,
        PrefillChunk,
        GroupedVerifier,
        SyntheticTest,
    };

    /** Number of independently retained production phases, preserving wire order. */
    inline constexpr std::size_t kExpertHistogramProductionSourceCount = 3;

    /** @return Dense production-phase index, or the source-count sentinel. */
    [[nodiscard]] constexpr std::size_t expertHistogramProductionSourceIndex(
        ExpertHistogramSource source) noexcept
    {
        switch (source)
        {
        case ExpertHistogramSource::DecodeToken: return 0;
        case ExpertHistogramSource::PrefillChunk: return 1;
        case ExpertHistogramSource::GroupedVerifier: return 2;
        case ExpertHistogramSource::SyntheticTest: break;
        }
        return kExpertHistogramProductionSourceCount;
    }
}
