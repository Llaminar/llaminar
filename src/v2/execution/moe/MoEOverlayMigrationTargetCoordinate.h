/**
 * @file MoEOverlayMigrationTargetCoordinate.h
 * @brief One directed complete-target/layer coordinate for existing economy evidence.
 *
 * Whole participants and fixed CPU cohorts use the same typed route alternatives
 * as the acquired placement bank. A coordinate carries no geometry, source,
 * readiness or publication permit: its caller must resolve both targets against
 * the retained original live snapshot and checked complete placement metadata.
 */
#pragma once
#include "MoEOverlayRouteTarget.h"
#include <compare>
#include <tuple>
namespace llaminar2
{
    /** @brief Logical movement coordinate independent of physical shard count. */
    struct MoEOverlayMigrationTargetCoordinate
    {
        MoEOverlayRouteTarget source_target;
        MoEOverlayRouteTarget destination_target;
        int layer = -1;
        /** @return Whether both explicit targets name a non-self model-layer edge. */
        bool valid() const noexcept
        {
            return layer >= 0 && !source_target.isAbsent() && !destination_target.isAbsent() &&
                   source_target.word() != destination_target.word();
        }
        /** @param other Original complete coordinate.
         * @return Canonical source/destination/layer ordering; words are never participant aliases. */
        auto operator<=>(const MoEOverlayMigrationTargetCoordinate &other) const noexcept
        {
            return std::tuple{source_target.word(), destination_target.word(), layer} <=>
                   std::tuple{other.source_target.word(), other.destination_target.word(), other.layer};
        }
        /** @param other Original complete coordinate.
         * @return Equality of both typed alternatives and original model-global layer. */
        bool operator==(const MoEOverlayMigrationTargetCoordinate &other) const noexcept
        {
            return (*this <=> other) == 0;
        }
    };
}
