/**
 * @file MoEOverlayPhysicalRemotePieceOrdinals.h
 * @brief Stable global MPI slot ordinals for the existing normalized physical wave.
 *
 * An expert/role can fan out to several remote members. Its old logical
 * migration-times-three index is not a unique physical lane. Number every
 * actual cross-rank piece in the complete original wave before local endpoint
 * filtering, so source, receiver and uninvolved ranks retain identical slots.
 * This value owns no allocation, transport, publication or live byte ledger.
 */
#pragma once
#include "MoEOverlayPhysicalProjectionPieces.h"
#include <limits>
#include <optional>
#include <vector>

namespace llaminar2 {
/** @brief One optional remote slot per original piece, in the same complete wave order. */
struct MoEOverlayPhysicalRemotePieceOrdinals {
    std::vector<std::vector<std::optional<size_t>>> by_migration;
    size_t remote_piece_count=0;
};
/**
 * @brief Assign complete-wave remote slots without accepting a rank-local filter.
 * @param pieces Exact normalized logical-edge/role/destination/source sequence.
 * @param topology Original endpoint inventory from the same authenticated snapshot.
 * @param world_size Exact original communicator size; one accepts original local unknown rank aliases.
 * @return Optional slot for every original piece; same-rank pieces remain absent.
 * @throws std::invalid_argument For missing or out-of-range original endpoint ranks.
 * @throws std::overflow_error For an unrepresentable remote piece population.
 */
inline MoEOverlayPhysicalRemotePieceOrdinals physicalMoEOverlayRemotePieceOrdinals(
    const std::vector<std::vector<MoEOverlayPhysicalProjectionPiece>>& pieces,
    const MoEExpertOwnerMap& topology,int world_size)
{
    if(world_size<=0)throw std::invalid_argument("Physical remote ordinals require the original communicator");
    const auto rank=[&](int participant) {
        const auto* member=topology.participantForId(participant);
        if(!member)throw std::invalid_argument("Physical remote ordinal lost an original endpoint");
        if(!member->world_rank_known && world_size==1)return 0;
        if(!member->world_rank_known || member->world_rank<0 || member->world_rank>=world_size)
            throw std::invalid_argument("Physical remote ordinal has no exact original endpoint rank");
        return member->world_rank;
    };
    MoEOverlayPhysicalRemotePieceOrdinals result;
    result.by_migration.resize(pieces.size());
    for(size_t edge=0;edge<pieces.size();++edge) {
        auto& ordinals=result.by_migration[edge];ordinals.resize(pieces[edge].size());
        for(size_t piece=0;piece<pieces[edge].size();++piece) {
            const auto& current=pieces[edge][piece];
            if(rank(current.source_participant)==rank(current.destination_participant))continue;
            if(result.remote_piece_count==std::numeric_limits<size_t>::max())
                throw std::overflow_error("Physical remote ordinal population overflows size_t");
            ordinals[piece]=result.remote_piece_count++;
        }
    }
    return result;
}
} // namespace llaminar2
