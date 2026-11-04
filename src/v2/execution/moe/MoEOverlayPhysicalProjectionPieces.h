/**
 * @file MoEOverlayPhysicalProjectionPieces.h
 * @brief Original-coordinate normalization for the existing physical residency fabric.
 *
 * One logical expert edge becomes only its nonempty physical intersections,
 * ordered by role, original destination member and original source member.
 * This passive geometry carries no source readiness, allocation, epoch or
 * capacity authority. Fabric reservations and operations consume the same
 * records; neither a cohort nor a transport relay becomes a scalar owner.
 */
#pragma once

#include "MoEOverlayPhysicalProjectionGeometry.h"
#include "MoEExpertShardCohort.h"
#include "MoEOverlayResidencyAuthority.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Enumerate only nonempty overlaps of two original target layouts.
     * @param source Complete ordered source target declaration.
     * @param destination Complete ordered destination target declaration.
     * @return Role/destination/source ordered live pieces, without padded storage.
     * @throws std::invalid_argument For absent or duplicate original member identities.
     * @note Setup demand and authenticated runtime movement share this geometry
     *       operation. These passive layouts cannot mint a prepared source lease.
     */
    inline std::vector<MoEOverlayPhysicalProjectionPiece>
    physicalMoEOverlayProjectionPieces(const MoEOverlayPhysicalProjectionTarget& source,
                                      const MoEOverlayPhysicalProjectionTarget& destination)
    {
        const auto require=[](const auto& target) {
            if(target.tier_idx<0 || target.members.empty())
                throw std::invalid_argument("Physical projection target requires its declared nonempty membership");
            std::vector<int> ids;
            for(const auto& member:target.members) {
                if(member.participant_id<0 || !member.device.is_valid() ||
                   std::find(ids.begin(),ids.end(),member.participant_id)!=ids.end())
                    throw std::invalid_argument("Physical projection target changed original member identities");
                ids.push_back(member.participant_id);
            }
        };
        require(source);require(destination);
        std::vector<MoEOverlayPhysicalProjectionPiece> result;
        for(size_t role=0;role<3;++role)
            for(size_t dst=0;dst<destination.members.size();++dst)
                for(size_t src=0;src<source.members.size();++src) {
                    auto overlap=ExpertTierWeightProjectionIntersection::between(
                        source.members[src].projections[role],destination.members[dst].projections[role]);
                    if(!overlap.empty())result.push_back({static_cast<ExpertTierWeightProjection>(role),dst,src,
                        destination.members[dst].participant_id,source.members[src].participant_id,std::move(overlap)});
                }
        return result;
    }

    /**
     * @brief Authenticate original runtime targets and expose their passive projection layouts.
     * @param migration Original logical expert edge retained by the authority.
     * @param source Original complete source geometry/format, never a local slice.
     * @param experts Original full expert population, not a tier quota.
     * @param topology Original participant inventory resolving actual endpoints.
     * @return Checked ordered source/destination layouts, without a prepared-source permit.
     * @throws std::invalid_argument For foreign source, endpoints, model axes or cuts.
     * @note No source-bank lease or execution readiness follows from this result.
     */
    inline std::pair<MoEOverlayPhysicalProjectionTarget,MoEOverlayPhysicalProjectionTarget>
    physicalMoEOverlayProjectionLayouts(
        const MoEOverlayTierMigration &migration,
        const MoEOverlayLayerWeightManifest &source,
        int experts,
        const MoEExpertOwnerMap &topology)
    {
        const auto from = MoEExpertPlacementTarget::fromPlacement(migration.source);
        const auto to = MoEExpertPlacementTarget::fromPlacement(migration.destination);
        if (!from || !to || !source.valid() || experts <= 0 ||
            migration.layer_idx != source.layer_idx ||
            migration.source.layer_idx != source.layer_idx ||
            migration.destination.layer_idx != source.layer_idx ||
            migration.expert_id < 0 || migration.expert_id >= experts ||
            migration.source.expert_id != migration.expert_id ||
            migration.destination.expert_id != migration.expert_id)
            throw std::invalid_argument("Physical expert pieces require the original complete logical source edge");

        // Resolve each original endpoint through the canonical inventory. In
        // particular a CPU NUMA coordinate never creates a device ordinal.
        const auto validate = [&](const MoEExpertOwner &owner,
                                  const MoEExpertPlacementTarget &target)
        {
            if (owner.shard_cohort)
            {
                const auto &cohort = *owner.shard_cohort;
                if (cohort.source().layer_idx != source.layer_idx || cohort.participants().empty() ||
                    cohort.ownership(0).geometry().experts != experts)
                    throw std::invalid_argument("Physical expert cohort changed original source geometry or format");
                // A valid manifest's array order is not source identity. Every
                // original role still matches its exact dimensions and format.
                for (const auto &projection : source.projections)
                    if (cohort.source().projectionForRole(projection.projection) != projection)
                        throw std::invalid_argument("Physical expert cohort changed an original role geometry or format");
            }
            for (std::size_t index = 0; index < target.participantCount(); ++index)
            {
                const auto *member = topology.participantForId(target.participantId(index));
                if (!member || member->tier_idx != owner.tier_idx ||
                    member->domain_name != owner.domain_name)
                    throw std::invalid_argument("Physical expert member is absent from its original tier inventory");
                if (owner.shard_cohort)
                {
                    const auto &original = owner.shard_cohort->participants()[index];
                    if (!member->device.is_cpu() || !member->world_rank_known ||
                        member->world_rank != original.world_rank || member->address != original.address)
                        throw std::invalid_argument("Physical CPU cohort member changed original rank or endpoint");
                }
                else if (member->device != owner.device || member->address != owner.address ||
                         member->world_rank_known != owner.owner_world_rank_known ||
                         (member->world_rank_known && member->world_rank != owner.owner_world_rank))
                    throw std::invalid_argument("Physical whole owner changed its original endpoint identity");
            }
        };
        validate(migration.source, *from);
        validate(migration.destination, *to);

        const auto weight_role = [](ExpertTierWeightProjection role)
        {
            switch (role)
            {
            case ExpertTierWeightProjection::Gate: return WeightRole::MoEExpertGate;
            case ExpertTierWeightProjection::Up: return WeightRole::MoEExpertUp;
            case ExpertTierWeightProjection::Down: return WeightRole::MoEExpertDown;
            }
            throw std::invalid_argument("Unknown physical expert projection role");
        };
        const auto rectangle = [&](const MoEExpertOwner &owner,
                                   std::size_t member,
                                   const MoEOverlayProjectionWeightManifest &projection)
        {
            if (owner.shard_cohort)
                return owner.shard_cohort->ownership(member).projection(weight_role(projection.projection));
            return MoEExpertProjectionOwnership::Projection{
                MoEProjectionResidence::ExpertOwner,
                projection.N, projection.K, 0, projection.N, 0, projection.K};
        };
        const auto layout=[&](const MoEExpertOwner& owner,const MoEExpertPlacementTarget& target) {
            MoEOverlayPhysicalProjectionTarget result{.tier_idx=owner.tier_idx};
            for(size_t member=0;member<target.participantCount();++member) {
                const auto* original=topology.participantForId(target.participantId(member));
                MoEOverlayPhysicalProjectionMember value{.participant_id=original->participant_id,
                    .world_rank=original->world_rank_known?original->world_rank:-1,
                    .device=original->device,.address=original->address};
                for(const auto& projection:source.projections)
                    value.projections[static_cast<size_t>(projection.projection)]=rectangle(owner,member,projection);
                result.members.push_back(std::move(value));
            }
            return result;
        };
        return {layout(migration.source,*from),layout(migration.destination,*to)};
    }

    /**
     * @brief Enumerate authenticated original-coordinate runtime pieces.
     * @param migration Exact logical source/destination edge.
     * @param source Original full source matrix contract.
     * @param experts Original full E axis.
     * @param topology Canonical original endpoint inventory.
     * @return Only nonempty pieces, using the same enumerator as setup demand.
     */
    inline std::vector<MoEOverlayPhysicalProjectionPiece> physicalMoEOverlayProjectionPieces(
        const MoEOverlayTierMigration& migration,const MoEOverlayLayerWeightManifest& source,
        int experts,const MoEExpertOwnerMap& topology)
    {
        const auto [from,to]=physicalMoEOverlayProjectionLayouts(migration,source,experts,topology);
        return physicalMoEOverlayProjectionPieces(from,to);
    }
}
