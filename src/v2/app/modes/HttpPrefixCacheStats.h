/**
 * @file HttpPrefixCacheStats.h
 * @brief HTTP projection of passive RAM/disk occupancy and tier traffic.
 *
 * The HTTP statistics mutex owns reset/observation ordering. Sources are frozen
 * at server construction; only their small metadata publishers remain live.
 * Shared archives contribute once even when several TP or PP children use them.
 */
#pragma once
#include "execution/prefix_cache/PrefixCacheTelemetry.h"
#include <nlohmann/json.hpp>

namespace llaminar2
{
    /** @brief Epoch accounting over cache-owned metadata, with no inference API access. */
    class HttpPrefixCacheStats
    {
    public:
        /** @brief Freeze sources; initial occupancy survives the new measurement epoch. */
        explicit HttpPrefixCacheStats(PrefixCacheTelemetrySources sources = {});
        /** @brief Reset traffic baselines while preserving cache contents and occupancy. */
        void reset();
        /** @return Current occupancy and completed traffic since reset, under the caller's stats lock. */
        [[nodiscard]] nlohmann::json snapshot() const;

    private:
        /** @brief One coherent observation per distinct authority in this HTTP sample. */
        struct Reading
        {
            nlohmann::json tiers;
            PrefixTierActivity ram, disk;
        };
        /** @brief Copy bounded host metadata, without storage I/O or runner traversal. */
        [[nodiscard]] Reading read() const;
        const PrefixCacheTelemetrySources sources_;
        PrefixTierActivity ram_baseline_, disk_baseline_;
    };
}
