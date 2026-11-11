/**
 * @file OriginalControllerFabricObservation.h
 * @brief Bounded diagnostic borrowing from an already-bound controller topology.
 *
 * The controller session is the sole cohort authority. A smaller telemetry
 * observation cannot query another owner, change participant order or acquire
 * remote aliases. This pure fixture helper validates original inventory facts;
 * it never stages memory, launches a backend or declares another topology.
 */
#pragma once
#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "planning/ObservedPeerGrouping.h"
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2::test {
/**
 * @brief Borrow a bounded ordered observation from an existing native cohort.
 * @param inventory Original immutable world that authenticated the session.
 * @param topology Actual already-bound complete controller participant list.
 * @param family Requested native backend, not an owner-selection instruction.
 * @param count Positive prefix extent within that declared backend cohort.
 * @return Same original owner and addresses, without new physical selection.
 * @throws std::invalid_argument For incomplete/foreign ownership or invalid extent.
 */
inline std::pair<int,std::vector<GlobalDeviceAddress>> originalControllerFabricObservation(
    const ClusterInventory& inventory,const MoEOverlayDeviceControllerTopology& topology,
    DeviceType family,int count)
{
    if((family!=DeviceType::CUDA && family!=DeviceType::ROCm) || count<=0)
        throw std::invalid_argument("Controller observation requires native family and positive extent");
    std::pair<int,std::vector<GlobalDeviceAddress>> result{-1,{}};
    std::set<std::string> identities;
    for(const auto& participant:topology.participants) {
        if(participant.device.type!=family)continue;
        if(!participant.world_rank_known || participant.world_rank<0 ||
            participant.world_rank>=inventory.world_size ||
            participant.address.toLocalDeviceId()!=participant.device)
            throw std::invalid_argument("Controller observation lost its declared original participant");
        const auto& owner=inventory.getRank(participant.world_rank);
        const auto& gpu=ObservedPeerGrouping(owner).device(participant.device);
        if(!rankOwnsGPUDevice(owner,gpu) || participant.address.hostname!=owner.hostname ||
            participant.address.numa_node!=gpu.numa_node ||
            !identities.insert(gpu.uuid).second ||
            (result.first!=-1 && result.first!=owner.rank))
            throw std::invalid_argument("Controller observation changed original native owner identity");
        result.first=owner.rank;
        result.second.push_back(participant.address);
    }
    if(result.second.size()<static_cast<std::size_t>(count))
        throw std::invalid_argument("Controller observation exceeds declared native cohort");
    result.second.resize(static_cast<std::size_t>(count));
    return result;
}
} // namespace llaminar2::test
