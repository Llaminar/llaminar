/**
 * @file MoEExpertProjectionOwnership.h
 * @brief Immutable projection geometry shared by residency, preparation and sizing.
 *
 * Complete-expert placement and owner-local gate/up with distributed down output
 * columns are different physical layouts. This value describes that difference
 * without becoming a second placement controller: the existing expert authority
 * supplies the owner or resident count. No route, epoch, allocation or capacity
 * decision is retained here. Constructing geometry does not admit an execution
 * topology; the graph and movement implementation must support the same layout.
 */
#pragma once

#include "loaders/WeightIdentity.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief One projection's physical residency and movement obligation. */
    enum class MoEProjectionResidence
    {
        ExpertOwner, ///< Complete projection follows the authoritative expert owner.
        ParticipantOutputSlice, ///< Fixed full-K output slice of every domain expert.
    };

    /**
     * @brief A validated homogeneous-domain partition, not a live ownership map.
     *
     * The domain covers the declared expert inventory. A tier with only a subset
     * of that inventory cannot apply this contract to the whole model implicitly.
     * Output-column partitions are equal because the retained native allgather
     * publishes equally sized slices. Unequal extents are rejected, not padded
     * behind the memory authority's back.
     */
    class MoEExpertProjectionOwnership final
    {
    public:
        /** @brief Model-authenticated geometry of one routed layer. */
        struct Geometry
        {
            int experts;
            int model_columns;
            int intermediate_columns;
            /** @brief Compare every source dimension, including the expert axis. */
            bool operator==(const Geometry &) const = default;
        };

        /** @brief Exact source/output interval; K is always complete and unchanged. */
        struct Projection
        {
            MoEProjectionResidence residence;
            int source_rows;
            int source_columns;
            int first_row;
            int rows;
            /** @brief Compare ownership and complete arithmetic/slice identity. */
            bool operator==(const Projection &) const = default;
        };

        /**
         * @brief Describe ordinary whole-expert ownership at one participant.
         * @param geometry Complete source layer, before any expert apportionment.
         * @param participant Participant coordinate used by the expert authority.
         * @param participants Complete domain width, not the rank-local subset.
         * @return Validated geometry for gate, up and down resident at one owner.
         */
        static MoEExpertProjectionOwnership completeExperts(
            Geometry geometry, int participant = 0, int participants = 1)
        {
            return {Layout::CompleteExperts, geometry, participant, participants};
        }

        /**
         * @brief Keep gate/up owner-local and partition down output rows.
         * @param geometry Complete source layer with the domain's full expert set.
         * @param participant Output-slice coordinate, independent of device ordinal.
         * @param participants Equal output-column partitions in the domain.
         * @return A layout that never changes down K or the serial reduction tree.
         * @throws std::invalid_argument for empty/nonintegral partitions.
         */
        static MoEExpertProjectionOwnership gateUpOwnedDownColumns(
            Geometry geometry, int participant, int participants)
        {
            return {Layout::GateUpOwnedDownColumns, geometry, participant, participants};
        }

        /** @return Complete immutable source dimensions. */
        Geometry geometry() const noexcept { return geometry_; }
        /** @return This layout's coordinate in the authoritative domain. */
        int participant() const noexcept { return participant_; }
        /** @return Full domain width, including participants on other ranks. */
        int participants() const noexcept { return participants_; }

        /** @return The sole projection set priced and copied by expert movement. */
        DeviceMoEProjectionSet movableProjections() const noexcept
        {
            return layout_ == Layout::CompleteExperts
                ? DeviceMoEProjectionSet::CompleteExpert : DeviceMoEProjectionSet::GateUp;
        }

        /**
         * @brief Resolve a semantic projection without inferring it from its shape.
         * @param role One of the three routed-expert weight roles.
         * @return Complete original matrix and exact participant output interval.
         * @throws std::invalid_argument for dense/shared/router/unknown roles.
         */
        Projection projection(WeightRole role) const
        {
            switch (role)
            {
            case WeightRole::MoEExpertGate:
            case WeightRole::MoEExpertUp:
                return {MoEProjectionResidence::ExpertOwner,
                    geometry_.intermediate_columns, geometry_.model_columns,
                    0, geometry_.intermediate_columns};
            case WeightRole::MoEExpertDown:
                if (layout_ == Layout::CompleteExperts)
                    return {MoEProjectionResidence::ExpertOwner,
                        geometry_.model_columns, geometry_.intermediate_columns,
                        0, geometry_.model_columns};
                // Division was proved exact at construction. Each output dot
                // retains every input column; assembly needs no FP reduction.
                return {MoEProjectionResidence::ParticipantOutputSlice,
                    geometry_.model_columns, geometry_.intermediate_columns,
                    participant_ * (geometry_.model_columns / participants_),
                    geometry_.model_columns / participants_};
            default:
                throw std::invalid_argument("Projection ownership requires a routed-expert role");
            }
        }

        /**
         * @brief Project an existing expert owner's identity into physical residency.
         * @param role Routed projection being prepared.
         * @param expert_owner Owner coordinate supplied by the sole placement authority.
         * @return Whether this participant prepares that expert's named projection.
         * @throws std::invalid_argument for an owner outside the bound domain.
         */
        bool prepares(WeightRole role, int expert_owner) const
        {
            if (expert_owner < 0 || expert_owner >= participants_)
                throw std::invalid_argument("Expert owner is outside its projection domain");
            return projection(role).residence == MoEProjectionResidence::ParticipantOutputSlice ||
                expert_owner == participant_;
        }

        /**
         * @brief Translate owner cardinality into the projection's physical matrix count.
         * @param role Routed projection being priced or prepared.
         * @param owned_experts Exact count supplied by the existing residency authority.
         * @return Owner count for movable projections, full domain count for fixed down.
         * @throws std::invalid_argument when the count exceeds the source inventory.
         */
        std::size_t residentExperts(WeightRole role, std::size_t owned_experts) const
        {
            if (owned_experts > static_cast<std::size_t>(geometry_.experts))
                throw std::invalid_argument("Projection ownership exceeds the expert inventory");
            return projection(role).residence == MoEProjectionResidence::ParticipantOutputSlice
                ? static_cast<std::size_t>(geometry_.experts) : owned_experts;
        }

        /** @brief Complete value identity for prepared views and setup-plan comparisons. */
        bool operator==(const MoEExpertProjectionOwnership &) const = default;

    private:
        /** @brief Closed physical layouts; there is no independent mutable phase flag. */
        enum class Layout { CompleteExperts, GateUpOwnedDownColumns };

        /** @brief Prove source extents and collective geometry before publishing a value. */
        MoEExpertProjectionOwnership(Layout layout, Geometry geometry, int participant, int participants)
            : layout_(layout), geometry_(geometry), participant_(participant), participants_(participants)
        {
            if (geometry.experts <= 0 || geometry.model_columns <= 0 ||
                geometry.intermediate_columns <= 0 || participants <= 0 ||
                participant < 0 || participant >= participants)
                throw std::invalid_argument("Projection ownership has invalid source/domain geometry");
            if (layout == Layout::GateUpOwnedDownColumns && geometry.model_columns % participants != 0)
                throw std::invalid_argument("Down-output partition requires equal nonempty output slices");
            // Validate the original parent as well as the smaller local view:
            // otherwise overflow could make an impossible inventory look small.
            const auto n = static_cast<std::size_t>(geometry.model_columns);
            const auto k = static_cast<std::size_t>(geometry.intermediate_columns);
            const auto e = static_cast<std::size_t>(geometry.experts);
            if (n > std::numeric_limits<std::size_t>::max() / k ||
                n * k > std::numeric_limits<std::size_t>::max() / e)
                throw std::overflow_error("Projection source inventory overflows size_t");
        }

        Layout layout_;
        Geometry geometry_;
        int participant_;
        int participants_;
    };
} // namespace llaminar2
