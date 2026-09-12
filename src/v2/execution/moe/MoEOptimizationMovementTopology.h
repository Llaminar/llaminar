/**
 * @file MoEOptimizationMovementTopology.h
 * @brief Immutable movement opportunities projected from a model-frozen plan.
 *
 * This is geometry, not a planner, profitability decision, or live placement
 * mirror. The existing capacity authority has already assigned real experts
 * to tiers. Consumers can require evidence for those expressible objectives
 * without deriving them from device names, CLI defaults, or observed moves.
 */
#pragma once

#include "execution/moe/MoEOptimizationStatus.h"

namespace llaminar2
{
    struct MoERoutedExpertPlacementPlan;

    /** Independent placement degrees of freedom in the admitted topology. */
    enum class MoEOptimizationMovementAxes : std::uint8_t
    {
        None,
        TierResidency,
        ParticipantPlacement,
        Both,
    };

    /** @return Whether the topology permits exchanges between priorities. */
    [[nodiscard]] constexpr bool hasTierResidencyAxis(MoEOptimizationMovementAxes axes) noexcept
    {
        return axes == MoEOptimizationMovementAxes::TierResidency ||
               axes == MoEOptimizationMovementAxes::Both;
    }

    /** @return Whether independently owned experts can exchange participants. */
    [[nodiscard]] constexpr bool hasParticipantPlacementAxis(MoEOptimizationMovementAxes axes) noexcept
    {
        return axes == MoEOptimizationMovementAxes::ParticipantPlacement ||
               axes == MoEOptimizationMovementAxes::Both;
    }

    /** Value-only projection; it neither selects a policy nor guarantees payoff. */
    struct MoEOptimizationMovementTopology
    {
        MoEOptimizationAuthority authority = MoEOptimizationAuthority::None;
        MoEOptimizationMovementAxes axes = MoEOptimizationMovementAxes::None;

        /** @return Whether the projection has a coherent, recognized identity. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            const bool recognized_axes = axes == MoEOptimizationMovementAxes::None ||
                axes == MoEOptimizationMovementAxes::TierResidency ||
                axes == MoEOptimizationMovementAxes::ParticipantPlacement ||
                axes == MoEOptimizationMovementAxes::Both;
            return recognized_axes &&
                ((authority == MoEOptimizationAuthority::None && axes == MoEOptimizationMovementAxes::None) ||
                 authority == MoEOptimizationAuthority::Host || authority == MoEOptimizationAuthority::Device);
        }

        /** @brief Compare immutable geometry across a model's requests. */
        bool operator==(const MoEOptimizationMovementTopology &) const = default;
    };

    /**
     * @brief Describe exchanges possible in concrete per-layer tier membership.
     * @param plan Model-resolved plan; declarative capacity requests are invalid.
     * @return Independent tier-priority and within-domain participant axes.
     * @throws std::invalid_argument for missing or malformed physical geometry.
     *
     * A one-expert-per-device exchange can improve unequal-rate service. Two
     * experts and two apportioned participants therefore suffice; tensor-
     * sharded/replicated participants do not independently own those experts.
     */
    [[nodiscard]] MoEOptimizationMovementAxes availableMoEOptimizationMovementAxes(
        const MoERoutedExpertPlacementPlan &plan);

    /**
     * @brief Project the canonical frozen authority and geometry, without I/O.
     * @param plan Null/disabled means no overlay; enabled plans must be frozen.
     * @return Immutable topology, independent of Static/Dynamic policy activity.
     * @throws std::invalid_argument for unresolved or contradictory authority.
     */
    [[nodiscard]] MoEOptimizationMovementTopology describeMoEOptimizationMovementTopology(
        const MoERoutedExpertPlacementPlan *plan);
}
