/**
 * @file MoEOverlayPhysicalProjectionGeometry.h
 * @brief Passive original-coordinate movement geometry, without source or allocation permits.
 *
 * Runtime normalization, setup demand and measurement completeness retain these
 * same member rectangles. No operation, source lease, readiness or owner ledger
 * belongs to this geometry.
 */
#pragma once
#include "ExpertTierWeightDeviceLayout.h"
#include "backends/GlobalDeviceAddress.h"
#include <array>
#include <vector>
namespace llaminar2 {
enum class ExpertTierWeightProjection : std::uint8_t;
    /** @brief One actual source/destination overlap, without allocation padding. */
    struct MoEOverlayPhysicalProjectionPiece
    {
        ExpertTierWeightProjection projection;
        std::size_t destination_member;
        std::size_t source_member;
        int destination_participant;
        int source_participant;
        ExpertTierWeightProjectionIntersection intersection;
    };

    /** @brief Passive original member geometry; this carries no source or execution permission. */
    struct MoEOverlayPhysicalProjectionMember
    {
        int participant_id = -1;
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();
        GlobalDeviceAddress address;
        std::array<MoEExpertProjectionOwnership::Projection, 3> projections{};
        /** @param other Original member geometry. @return Complete passive identity equality. */
        bool operator==(const MoEOverlayPhysicalProjectionMember &other) const = default;
    };

    /** @brief One declared whole target or complete fixed-cut target, never a scalar cohort alias. */
    struct MoEOverlayPhysicalProjectionTarget
    {
        int tier_idx = -1;
        std::vector<MoEOverlayPhysicalProjectionMember> members;
        /** @param other Original target geometry. @return Ordered member and tier equality. */
        bool operator==(const MoEOverlayPhysicalProjectionTarget &other) const = default;
    };

}
