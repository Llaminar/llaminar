/**
 * @file ObservedNativeReplayOwner.h
 * @brief Pure selection of original replay ownership from immutable rank observations.
 *
 * Capability predicates participate in admission before any native owner is
 * constructed. Counts, physical ownership and directed edges are independent
 * obligations; a later eligible owner is selected without executing a failed
 * candidate or changing the requested native transport.
 */
#pragma once
#include "planning/RankHardwareOwnership.h"
#include <array>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <utility>
namespace llaminar2::test {
    /** @brief Exact positive participant count for one native family. */
    struct NativeReplayFamily final
    {
        DeviceType family; ///< Original CUDA or ROCm device family.
        size_t degree; ///< Positive number of distinct devices owned by one original rank.
        std::string_view native_isa = {}; ///< Optional exact original ISA before target suffixes.
    };

/** @brief An original discovery owner and its authored positive family population. */
struct ObservedNativeReplayOwner final {
    int discovery_owner; ///< Original rank identity, never inferred from NUMA arithmetic.
    std::vector<DeviceId> devices; ///< Requested rank-owned coordinates in observation order.
};
/**
 * @brief Select the first physically complete owner satisfying the declared capability.
 * @param inventory Original immutable discovery world.
 * @param families Positive same-owner family cardinalities and optional exact ISA.
 * @param accepts Optional pure capability query over one original rank.
 * @return Qualified original owner, or empty when no actual deployment qualifies.
 * @throws std::invalid_argument For an empty or invalid family request.
 */
inline std::optional<ObservedNativeReplayOwner> selectObservedNativeReplayOwner(
    const ClusterInventory &inventory, std::span<const NativeReplayFamily> families,
    const std::function<bool(const RankInventory&)> &accepts = {}) {
    if(families.empty() || inventory.world_size<=0 ||
        inventory.ranks.size()!=static_cast<std::size_t>(inventory.world_size))
        throw std::invalid_argument("Incomplete original replay world or family request");
    for(const auto &entry:families)
        if(!entry.degree || (entry.family!=DeviceType::CUDA && entry.family!=DeviceType::ROCm))
            throw std::invalid_argument("Invalid original replay family request");
    for(const auto &rank:inventory.ranks) {
        std::vector<DeviceId> devices;
        bool complete=true;
        for(const auto &entry:families) {
            std::vector<DeviceId> owned;
            for(const auto &gpu:rank.gpus)
                if(gpu.type==entry.family && rankOwnsGPUDevice(rank,gpu) &&
                    (entry.native_isa.empty() || std::string_view(gpu.native_isa).substr(0,gpu.native_isa.find(':'))==entry.native_isa))
                    owned.emplace_back(entry.family,gpu.local_device_id);
            if(owned.size()<entry.degree) {complete=false;break;}
            devices.insert(devices.end(),owned.begin(),owned.begin()+entry.degree);
        }
        if(complete && (!accepts || accepts(rank)))
            return ObservedNativeReplayOwner{rank.rank,std::move(devices)};
    }
    return std::nullopt;
}
/**
 * @brief Find one directed destination-accesses-source edge on an original owner.
 * @param rank Original immutable CPU/GPU ownership publication.
 * @param family Requested native backend.
 * @param access Original observed directed capability query, never a new probe.
 * @return Source/destination pair or empty for a fully observed edge-free owner.
 * @throws std::invalid_argument When any owned directed observation is missing.
 */
template<class Access>
std::optional<std::array<DeviceId,2>> observedNativeDirectedPeerPair(
    const RankInventory &rank,DeviceType family,Access access) {
    std::vector<DeviceId> owned;
    for(const auto &gpu:rank.gpus)
        if(gpu.type==family && rankOwnsGPUDevice(rank,gpu)) owned.emplace_back(family,gpu.local_device_id);
    std::optional<std::array<DeviceId,2>> result;
    for(const auto source:owned) for(const auto destination:owned) if(source!=destination) {
        const auto available=access(destination,source);
        if(!available.has_value()) throw std::invalid_argument("Missing original directed peer observation");
        if(*available && !result) result=std::array{source,destination};
    }
    return result;
}
} // namespace llaminar2::test
